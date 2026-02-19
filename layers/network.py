import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedCausalConvBlock(nn.Module):
    """Depthwise dilated causal conv along patch sequence."""
    def __init__(self, d_model, kernel_size=3, dilation=1):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            dilation=dilation, padding=0,
            groups=d_model
        )
        self.norm = nn.BatchNorm1d(d_model)
    
    def forward(self, x):
        residual = x
        h = x.transpose(1, 2)
        h = F.pad(h, (self.causal_pad, 0))
        h = F.gelu(self.conv(h))
        h = self.norm(h)
        return residual + h.transpose(1, 2)


class SeasonalStream(nn.Module):
    """
    Non-linear stream for seasonal component.
    
    Ablation flags:
      use_multiscale: True = multi-kernel depthwise (innovation), False = skip
      use_causal:     True = dilated causal conv (innovation), False = skip
    """
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8,
                 n_blocks=1, kernel_sizes=(3, 5, 7), padding_patch='end',
                 use_multiscale=True, use_causal=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pred_len = pred_len
        self.padding_patch = padding_patch
        self.use_multiscale = use_multiscale
        self.use_causal = use_causal
        
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        
        self.dim = patch_len * patch_len  # P²

        # Patch Embedding: P → P² (xPatch baseline)
        self.embed = nn.Linear(patch_len, self.dim)
        self.embed_gelu = nn.GELU()
        self.embed_bn = nn.BatchNorm1d(self.patch_num)

        # Depthwise Conv: P² → P (xPatch baseline)
        self.dw_conv = nn.Conv1d(
            self.patch_num, self.patch_num,
            kernel_size=patch_len, stride=patch_len,
            groups=self.patch_num
        )
        self.dw_gelu = nn.GELU()
        self.dw_bn = nn.BatchNorm1d(self.patch_num)
        
        # Residual: P² → P
        self.res_proj = nn.Linear(self.dim, patch_len)
        
        # [ABLATABLE] Multiscale depthwise conv along N_p
        if use_multiscale:
            self.ms_convs = nn.ModuleList([
                nn.Conv1d(
                    patch_len, patch_len,
                    kernel_size=k, padding=k // 2,
                    groups=patch_len
                )
                for k in kernel_sizes
            ])
            self.ms_gelu = nn.GELU()
            self.ms_bn = nn.BatchNorm1d(self.patch_num)
        
        # Pointwise 1×1 Conv (xPatch baseline)
        self.pointwise = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.pw_gelu = nn.GELU()
        self.pw_bn = nn.BatchNorm1d(self.patch_num)
        
        # [ABLATABLE] Dilated causal conv
        if use_causal and n_blocks > 0:
            self.causal_blocks = nn.ModuleList()
            for i in range(n_blocks):
                self.causal_blocks.append(
                    DilatedCausalConvBlock(patch_len, kernel_size=3, dilation=2**i)
                )
        else:
            self.causal_blocks = nn.ModuleList()  # empty
        
        # Flatten + Head (xPatch baseline)
        flat_dim = self.patch_num * patch_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.head_fc1 = nn.Linear(flat_dim, pred_len * 2)
        self.head_gelu = nn.GELU()
        self.head_fc2 = nn.Linear(pred_len * 2, pred_len)
    
    def forward(self, s):
        if self.padding_patch == 'end':
            s = self.padding_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Embed + depthwise (xPatch baseline)
        s = self.embed_gelu(self.embed(s))
        s = self.embed_bn(s)
        res = self.res_proj(s)
        s = self.dw_gelu(self.dw_conv(s))
        s = self.dw_bn(s)
        s = s + res
        
        # [ABLATABLE] Multiscale depthwise
        if self.use_multiscale:
            h = s.transpose(1, 2)
            ms_sum = sum(conv(h) for conv in self.ms_convs)
            ms_sum = self.ms_gelu(ms_sum)
            s = self.ms_bn(s + ms_sum.transpose(1, 2))
        
        # Pointwise (xPatch baseline)
        s = self.pw_gelu(self.pointwise(s))
        s = self.pw_bn(s)
        
        # [ABLATABLE] Dilated causal
        for block in self.causal_blocks:
            s = block(s)
        
        # Flatten + head (xPatch baseline)
        s = self.flatten(s)
        s = self.head_gelu(self.head_fc1(s))
        s = self.head_fc2(s)
        return s


class TrendStream(nn.Module):
    """
    Linear trend stream.
    
    Ablation flag:
      use_agg_conv: True = GLCN aggregate conv (innovation), False = skip
    """
    def __init__(self, seq_len, pred_len, agg_kernel=5, use_agg_conv=True):
        super().__init__()
        self.pred_len = pred_len
        self.use_agg_conv = use_agg_conv
        
        if use_agg_conv:
            self.agg_conv = nn.Conv1d(1, 1, kernel_size=agg_kernel, padding=agg_kernel // 2)
        
        self.fc1 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        
        self.fc2 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)
        
        self.fc3 = nn.Linear(pred_len // 2, pred_len)
    
    def forward(self, t):
        if self.use_agg_conv:
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
    """Per-horizon gated blending of trend and seasonal."""
    def __init__(self, pred_len):
        super().__init__()
        self.gate_linear = nn.Linear(pred_len * 2, pred_len)
        nn.init.zeros_(self.gate_linear.bias)
        nn.init.xavier_uniform_(self.gate_linear.weight)
    
    def forward(self, y_trend, y_seasonal):
        gate_input = torch.cat([y_trend, y_seasonal], dim=-1)
        g = torch.sigmoid(self.gate_linear(gate_input))
        return g * y_trend + (1.0 - g) * y_seasonal


class ConcatFusion(nn.Module):
    """xPatch-style: concatenate + linear (ablation baseline)."""
    def __init__(self, pred_len):
        super().__init__()
        self.linear = nn.Linear(pred_len * 2, pred_len)
    
    def forward(self, y_trend, y_seasonal):
        return self.linear(torch.cat([y_trend, y_seasonal], dim=-1))


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
    Full AdaPatch network with ablation flags.
    
    Ablation flags:
      use_multiscale:    multiscale depthwise conv in seasonal stream
      use_causal:        dilated causal conv in seasonal stream
      use_gated_fusion:  gated fusion (True) vs concat+linear (False)
      use_agg_conv:      aggregate conv in trend stream
    
    Learnable alpha is controlled externally via alpha_lr_mult=0.
    """
    def __init__(self, seq_len, pred_len,
                 patch_len=16, stride=8, padding_patch='end',
                 n_blocks=1, kernel_sizes=(3, 5, 7),
                 agg_kernel=5, use_cross_variable=False,
                 use_multiscale=True, use_causal=True,
                 use_gated_fusion=True, use_agg_conv=True,
                 d_model=None):
        super().__init__()
        self.pred_len = pred_len
        self.use_cross_variable = use_cross_variable
        
        self.seasonal_stream = SeasonalStream(
            seq_len, pred_len, patch_len, stride,
            n_blocks, kernel_sizes, padding_patch,
            use_multiscale=use_multiscale,
            use_causal=use_causal,
        )
        self.trend_stream = TrendStream(
            seq_len, pred_len, agg_kernel,
            use_agg_conv=use_agg_conv,
        )
        
        if use_gated_fusion:
            self.fusion = GatedFusion(pred_len)
        else:
            self.fusion = ConcatFusion(pred_len)
        
        if use_cross_variable:
            self.cross_var = CrossVariableGate()
    
    def forward(self, seasonal, trend):
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
