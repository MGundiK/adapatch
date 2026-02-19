import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedCausalConvBlock(nn.Module):
    """
    Dilated causal convolution along patch sequence dimension.
    
    CRITICAL: Uses groups=d_model (DEPTHWISE) so each intra-patch
    position communicates across patches independently. This preserves
    the positional structure from the depthwise→pointwise pipeline.
    
    With groups=1 (the old version), all P intra-patch positions get
    scrambled together at every inter-patch step — destroying structure
    and wasting params (P×P×k instead of P×k).
    """
    def __init__(self, d_model, kernel_size=3, dilation=1):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        # DEPTHWISE: each of P positions slides independently along N_p
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            dilation=dilation, padding=0,
            groups=d_model  # ← FIX: was groups=1
        )
        self.norm = nn.BatchNorm1d(d_model)
    
    def forward(self, x):
        """x: (B*D, N_p, P) → (B*D, N_p, P)"""
        residual = x
        h = x.transpose(1, 2)                   # (B*D, P, N_p)
        h = F.pad(h, (self.causal_pad, 0))       # causal: pad left only
        h = F.gelu(self.conv(h))                  # (B*D, P, N_p)
        h = self.norm(h)
        return residual + h.transpose(1, 2)


class SeasonalStream(nn.Module):
    """
    Non-linear stream for seasonal (high-frequency) component.
    
    Architecture:
      1. Overlapping patching (P=16, S=8)
      2. Embed P→P² + GELU + BN (same as xPatch)
      3. Depthwise conv P²→P with k=stride=P (same as xPatch: full coverage)
      4. Residual from embed (P²→P via linear)
      5. Multiscale depthwise conv along N_p (GLCN multi-kernel, inter-patch)
      6. Pointwise 1×1 conv across patches (same as xPatch)
      7. N × DilatedCausalConv along N_p (depthwise, inter-patch)
      8. Flatten + MLP head → pred_len
    
    Key differences from xPatch:
      - Step 5: multiple kernel sizes instead of single depthwise
      - Step 7: dilated causal conv for structured long-range inter-patch comm
    
    Key differences from v1 (buggy):
      - Step 3: k=stride=P (was k=3,5,7 with stride=P → sparse subsampling)
      - Step 7: groups=P depthwise (was groups=1 → position scrambling)
    """
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8,
                 n_blocks=1, kernel_sizes=(3, 5, 7), padding_patch='end'):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pred_len = pred_len
        self.padding_patch = padding_patch
        
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        
        self.dim = patch_len * patch_len  # P² = 256

        # ── Patch Embedding: P → P² (same as xPatch) ────────────
        self.embed = nn.Linear(patch_len, self.dim)
        self.embed_gelu = nn.GELU()
        self.embed_bn = nn.BatchNorm1d(self.patch_num)

        # ── Depthwise Conv: P² → P (same as xPatch) ─────────────
        # Single kernel with k=stride=P for FULL coverage of each
        # P-element window within the P² embedding. Each patch 
        # processed independently (groups=patch_num).
        self.dw_conv = nn.Conv1d(
            self.patch_num, self.patch_num,
            kernel_size=patch_len, stride=patch_len,
            groups=self.patch_num
        )
        self.dw_gelu = nn.GELU()
        self.dw_bn = nn.BatchNorm1d(self.patch_num)
        
        # ── Residual: P² → P ────────────────────────────────────
        self.res_proj = nn.Linear(self.dim, patch_len)
        
        # ── Multiscale Depthwise Conv along N_p (INTER-patch) ───
        # GLCN's multi-kernel idea applied to the PATCH SEQUENCE.
        # groups=patch_len → each intra-patch position sees its own
        # multi-scale view of the patch sequence independently.
        # This is TRUE multiscale inter-patch communication.
        self.ms_convs = nn.ModuleList([
            nn.Conv1d(
                patch_len, patch_len,
                kernel_size=k, padding=k // 2,
                groups=patch_len  # depthwise: preserves position structure
            )
            for k in kernel_sizes
        ])
        self.ms_gelu = nn.GELU()
        self.ms_bn = nn.BatchNorm1d(self.patch_num)
        
        # ── Pointwise 1×1 Conv (INTER-patch) ────────────────────
        # Same as xPatch: mix all patches at each position
        self.pointwise = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.pw_gelu = nn.GELU()
        self.pw_bn = nn.BatchNorm1d(self.patch_num)
        
        # ── Dilated Causal Conv (INTER-patch, depthwise) ────────
        # Each position independently communicates across patches
        # with exponentially growing receptive field.
        self.causal_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.causal_blocks.append(
                DilatedCausalConvBlock(patch_len, kernel_size=3, dilation=2**i)
            )
        
        # ── Flatten + Head (same as xPatch) ─────────────────────
        flat_dim = self.patch_num * patch_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.head_fc1 = nn.Linear(flat_dim, pred_len * 2)
        self.head_gelu = nn.GELU()
        self.head_fc2 = nn.Linear(pred_len * 2, pred_len)
    
    def forward(self, s):
        """s: (B*D, L) → (B*D, pred_len)"""
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: (B*D, N_p, P)
        
        # Embed: P → P²
        s = self.embed_gelu(self.embed(s))      # (B*D, N_p, P²)
        s = self.embed_bn(s)
        
        # Save residual
        res = self.res_proj(s)                   # (B*D, N_p, P)
        
        # Depthwise conv: P² → P (xPatch-style, k=stride=P, full coverage)
        s = self.dw_gelu(self.dw_conv(s))        # (B*D, N_p, P)
        s = self.dw_bn(s)
        
        # Add residual
        s = s + res                              # (B*D, N_p, P)
        
        # Multiscale depthwise conv (INTER-patch, position-preserving)
        # Transpose so channels=P, spatial=N_p
        h = s.transpose(1, 2)                    # (B*D, P, N_p)
        ms_sum = sum(conv(h) for conv in self.ms_convs)
        ms_sum = self.ms_gelu(ms_sum)
        s = self.ms_bn(s + ms_sum.transpose(1, 2))  # (B*D, N_p, P) residual
        
        # Pointwise 1×1 across patches
        s = self.pw_gelu(self.pointwise(s))      # (B*D, N_p, P)
        s = self.pw_bn(s)
        
        # Dilated causal conv blocks (inter-patch, depthwise)
        for block in self.causal_blocks:
            s = block(s)
        
        # Flatten + head
        s = self.flatten(s)                      # (B*D, N_p × P)
        s = self.head_gelu(self.head_fc1(s))     # (B*D, 2T)
        s = self.head_fc2(s)                     # (B*D, T)
        return s


class TrendStream(nn.Module):
    """
    Linear stream for trend (low-frequency) component.
    
    Matches xPatch's proven FC design + GLCN aggregate conv.
    FC → AvgPool → LN → FC → AvgPool → LN → FC (no activations).
    """
    def __init__(self, seq_len, pred_len, agg_kernel=5):
        super().__init__()
        self.pred_len = pred_len
        
        # Aggregate Conv1D (GLCN boundary smoothing)
        self.agg_conv = nn.Conv1d(1, 1, kernel_size=agg_kernel, padding=agg_kernel // 2)
        
        # xPatch linear stream
        self.fc1 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        
        self.fc2 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)
        
        self.fc3 = nn.Linear(pred_len // 2, pred_len)
    
    def forward(self, t):
        """t: (B*D, L) → (B*D, pred_len)"""
        t_conv = self.agg_conv(t.unsqueeze(1)).squeeze(1)
        t = t + t_conv
        
        t = self.fc1(t)
        t = self.avgpool1(t.unsqueeze(1)).squeeze(1)
        t = self.ln1(t)
        
        t = self.fc2(t)
        t = self.avgpool2(t.unsqueeze(1)).squeeze(1)
        t = self.ln2(t)
        
        t = self.fc3(t)
        return t


class GatedFusion(nn.Module):
    """Per-horizon gated blending of trend and seasonal predictions."""
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
    """Optional cross-variable refinement. beta=0 at init."""
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, y, D):
        v = y.mean(dim=-1)
        scale = D ** 0.5
        attn = torch.softmax(
            torch.bmm(v.unsqueeze(2), v.unsqueeze(1)) / scale, dim=-1
        )
        return y + self.beta * torch.bmm(attn, y)


class AdaPatchNetwork(nn.Module):
    """
    Full AdaPatch network. Drop-in replacement for xPatch's Network class.
    """
    def __init__(self, seq_len, pred_len,
                 patch_len=16, stride=8, padding_patch='end',
                 n_blocks=1, kernel_sizes=(3, 5, 7),
                 agg_kernel=5, use_cross_variable=False,
                 d_model=None):  # unused, kept for CLI compat
        super().__init__()
        self.pred_len = pred_len
        self.use_cross_variable = use_cross_variable
        
        self.seasonal_stream = SeasonalStream(
            seq_len, pred_len, patch_len, stride,
            n_blocks, kernel_sizes, padding_patch
        )
        self.trend_stream = TrendStream(seq_len, pred_len, agg_kernel)
        self.fusion = GatedFusion(pred_len)
        
        if use_cross_variable:
            self.cross_var = CrossVariableGate()
    
    def forward(self, seasonal, trend):
        """
        seasonal, trend: (Batch, Input, Channel) → (Batch, Output, Channel)
        """
        seasonal = seasonal.permute(0, 2, 1)
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
        
        return y.permute(0, 2, 1)
