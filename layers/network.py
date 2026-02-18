 import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleDepthwiseBlock(nn.Module):
    """
    Multiscale depthwise convolution along the patch sequence + GELU.
    
    Combines GLCN's multi-kernel idea with non-linear activation.
    Convolutions are depthwise (groups=d_model) for efficiency.
    Operates along the N_p dimension with d_model as channels.
    """
    def __init__(self, d_model, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                d_model, d_model,
                kernel_size=k,
                padding=k // 2,  # 'same' padding
                groups=d_model   # depthwise
            )
            for k in kernel_sizes
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (B*D, N_p, d_model)
        Returns:
            (B*D, N_p, d_model) — with residual connection
        """
        residual = x
        h = x.transpose(1, 2)  # (B*D, d_model, N_p)
        f_sum = sum(conv(h) for conv in self.convs)
        f_sum = F.gelu(f_sum)
        f_sum = f_sum.transpose(1, 2)  # (B*D, N_p, d_model)
        return self.norm(residual + f_sum)


class DilatedCausalConvBlock(nn.Module):
    """
    Dilated causal convolution along the patch sequence dimension.
    
    Provides structured inter-patch communication with locality bias
    and logarithmic receptive field growth (WaveNet/TCN-style).
    """
    def __init__(self, d_model, kernel_size=3, dilation=1):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0  # manual causal padding
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (B*D, N_p, d_model)
        Returns:
            (B*D, N_p, d_model) — with residual
        """
        residual = x
        h = x.transpose(1, 2)  # (B*D, d_model, N_p)
        
        # Causal padding: pad left only
        h = F.pad(h, (self.causal_pad, 0))
        h = F.gelu(self.conv(h))
        h = F.gelu(self.pointwise(h))
        
        h = h.transpose(1, 2)  # (B*D, N_p, d_model)
        return self.norm(residual + h)


class SeasonalStream(nn.Module):
    """
    Non-linear stream for seasonal (high-frequency) component.
    
    Pipeline:
      1. Overlapping patching
      2. Patch embedding with skip connection
      3. N_blocks × (MultiscaleDepthwiseConv + DilatedCausalConv)
      4. Pool + project to prediction length
    """
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8,
                 d_model=128, n_blocks=2, kernel_sizes=(3, 5, 7),
                 padding_patch='end'):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.pred_len = pred_len
        self.padding_patch = padding_patch
        
        # Compute number of patches
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        
        # Patch embedding with skip (preserves linear features)
        self.embed_nonlinear = nn.Linear(patch_len, d_model)
        self.embed_skip = nn.Linear(patch_len, d_model)
        self.embed_norm = nn.BatchNorm1d(self.patch_num)
        
        # Processing blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                'intra': MultiscaleDepthwiseBlock(d_model, kernel_sizes),
                'inter': DilatedCausalConvBlock(d_model, kernel_size=3, dilation=2**i),
            }))
        
        # Prediction head
        self.head = nn.Linear(self.patch_num, pred_len)
    
    def forward(self, s):
        """
        Args:
            s: (B*D, L) — seasonal component
        Returns:
            (B*D, pred_len)
        """
        if self.padding_patch == 'end':
            s = self.padding_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: (B*D, patch_num, patch_len)
        
        h = F.gelu(self.embed_nonlinear(s)) + self.embed_skip(s)
        h = self.embed_norm(h)
        
        for block in self.blocks:
            h = block['intra'](h)
            h = block['inter'](h)
        
        h = h.mean(dim=2)      # (B*D, N_p)
        return self.head(h)     # (B*D, pred_len)


class TrendStream(nn.Module):
    """
    Linear stream for trend (low-frequency) component.
    
    Pipeline:
      1. Aggregate Conv1D (GLCN boundary smoothing)
      2. Non-overlapping patching
      3. FC → AvgPool → LayerNorm (×2, no activations — deliberately linear)
      4. Flatten → project to prediction length
    """
    def __init__(self, seq_len, pred_len, patch_len_trend=32, agg_kernel=5):
        super().__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len_trend
        
        # Pad sequence to be divisible by patch_len
        self.pad_len = (patch_len_trend - seq_len % patch_len_trend) % patch_len_trend
        effective_len = seq_len + self.pad_len
        self.n_patches = effective_len // patch_len_trend
        
        # Aggregate Conv1D (GLCN innovation)
        self.agg_conv = nn.Conv1d(1, 1, kernel_size=agg_kernel, padding=agg_kernel // 2)
        
        # Stage 1
        half_patch = patch_len_trend // 2
        self.fc1 = nn.Linear(patch_len_trend, half_patch)
        self.ln1 = nn.LayerNorm(half_patch)
        
        # Stage 2
        quarter_patch = half_patch // 2
        self.fc2 = nn.Linear(half_patch, quarter_patch)
        self.ln2 = nn.LayerNorm(quarter_patch)
        
        # Compute final patch count after two AvgPool(2)
        n_patches_final = self.n_patches
        for _ in range(2):
            n_patches_final = (n_patches_final + 1) // 2  # ceil division
        self.n_patches_final = max(n_patches_final, 1)
        
        self.fc_expand = nn.Linear(self.n_patches_final * quarter_patch, pred_len)
    
    def forward(self, t):
        """
        Args:
            t: (B*D, L) — trend component
        Returns:
            (B*D, pred_len)
        """
        # Aggregate conv + residual
        t_conv = self.agg_conv(t.unsqueeze(1)).squeeze(1)
        t = t + t_conv
        
        # Pad for clean non-overlapping patching
        if self.pad_len > 0:
            t = F.pad(t, (0, self.pad_len))
        
        BD, L = t.shape
        t = t.reshape(BD, self.n_patches, self.patch_len)
        
        # Stage 1: FC + AvgPool + LN (no activation)
        t = self.fc1(t)
        t = t.transpose(1, 2)
        t = F.avg_pool1d(t, kernel_size=2, ceil_mode=True)
        t = t.transpose(1, 2)
        t = self.ln1(t)
        
        # Stage 2
        t = self.fc2(t)
        t = t.transpose(1, 2)
        t = F.avg_pool1d(t, kernel_size=2, ceil_mode=True)
        t = t.transpose(1, 2)
        t = self.ln2(t)
        
        # Flatten + project
        t = t.reshape(BD, -1)
        expected = self.fc_expand.in_features
        actual = t.shape[1]
        if actual > expected:
            t = t[:, :expected]
        elif actual < expected:
            t = F.pad(t, (0, expected - actual))
        
        return self.fc_expand(t)


class GatedFusion(nn.Module):
    """
    Gated adaptive fusion of trend and seasonal predictions.
    Per-horizon blending via sigmoid gate, initialized at 0.5.
    """
    def __init__(self, pred_len):
        super().__init__()
        self.gate_linear = nn.Linear(pred_len * 2, pred_len)
        nn.init.zeros_(self.gate_linear.bias)
        nn.init.xavier_uniform_(self.gate_linear.weight)
    
    def forward(self, y_trend, y_seasonal):
        gate_input = torch.cat([y_trend, y_seasonal], dim=-1)
        g = torch.sigmoid(self.gate_linear(gate_input))
        return g * y_trend + (1.0 - g) * y_seasonal


class CrossVariableGate(nn.Module):
    """
    Optional lightweight cross-variable refinement.
    beta initialized to 0 (starts channel-independent).
    """
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, y, D):
        """
        Args:
            y: (B, D, pred_len)
        Returns:
            (B, D, pred_len)
        """
        v = y.mean(dim=-1)  # (B, D)
        scale = D ** 0.5
        attn = torch.softmax(
            torch.bmm(v.unsqueeze(2), v.unsqueeze(1)) / scale, dim=-1
        )
        y_cross = torch.bmm(attn, y)
        return y + self.beta * y_cross


class AdaPatchNetwork(nn.Module):
    """
    Full AdaPatch network. Drop-in replacement for xPatch's Network class.
    Same I/O format: receives (seasonal, trend) in [Batch, Input, Channel],
    returns [Batch, Output, Channel].
    """
    def __init__(self, seq_len, pred_len,
                 patch_len=16, stride=8, padding_patch='end',
                 d_model=128, n_blocks=2, kernel_sizes=(3, 5, 7),
                 patch_len_trend=32, agg_kernel=5,
                 use_cross_variable=False):
        super().__init__()
        self.pred_len = pred_len
        self.use_cross_variable = use_cross_variable
        
        self.seasonal_stream = SeasonalStream(
            seq_len, pred_len, patch_len, stride,
            d_model, n_blocks, kernel_sizes, padding_patch
        )
        self.trend_stream = TrendStream(
            seq_len, pred_len, patch_len_trend, agg_kernel
        )
        self.fusion = GatedFusion(pred_len)
        
        if use_cross_variable:
            self.cross_var = CrossVariableGate()
    
    def forward(self, seasonal, trend):
        """
        Args:
            seasonal: (Batch, Input, Channel)
            trend:    (Batch, Input, Channel)
        Returns:
            (Batch, Output, Channel)
        """
        seasonal = seasonal.permute(0, 2, 1)  # (B, C, L)
        trend = trend.permute(0, 2, 1)
        
        B, C, L = seasonal.shape
        s = seasonal.reshape(B * C, L)
        t = trend.reshape(B * C, L)
        
        y_seas = self.seasonal_stream(s)
        y_trend = self.trend_stream(t)
        y = self.fusion(y_trend, y_seas)
        
        y = y.reshape(B, C, self.pred_len)
        
        if self.use_cross_variable:
            y = self.cross_var(y, C)
        
        return y.permute(0, 2, 1)  # (B, pred_len, C)
