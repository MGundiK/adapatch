import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.cross_variable import CrossVariableMixing


class DilatedCausalConvBlock(nn.Module):
    """Depthwise dilated causal conv along patch sequence."""
    def __init__(self, d_model, kernel_size=3, dilation=1):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            dilation=dilation, padding=0, groups=d_model
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
    
    Cross-variable mixing (when enabled) is applied AFTER depthwise conv,
    on learned patch representations (B, C, N_p*P) — NOT on raw values.
    This is the key insight: variables are mixed on rich features that
    encode each variable's temporal patterns, not on raw normalized inputs.
    """
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8,
                 n_blocks=1, kernel_sizes=(3, 5, 7), padding_patch='end',
                 use_multiscale=True, use_causal=True,
                 n_vars=1, cv_mode='none', cv_rank=32, cv_kernel=7,
                 cv_depth=1, cv_post_pw=False):
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

        # Patch Embedding: P → P²
        self.embed = nn.Linear(patch_len, self.dim)
        self.embed_gelu = nn.GELU()
        self.embed_bn = nn.BatchNorm1d(self.patch_num)

        # Depthwise Conv: P² → P
        self.dw_conv = nn.Conv1d(
            self.patch_num, self.patch_num,
            kernel_size=patch_len, stride=patch_len,
            groups=self.patch_num
        )
        self.dw_gelu = nn.GELU()
        self.dw_bn = nn.BatchNorm1d(self.patch_num)
        
        # Residual: P² → P
        self.res_proj = nn.Linear(self.dim, patch_len)
        
        # ── CROSS-VARIABLE MIXING on learned representations ──────
        # Applied here: after depthwise conv (rich features), before
        # pointwise conv (cross-patch mixing). Each variable's patch
        # features now carry temporal pattern info that can meaningfully
        # inform other variables.
        self.cv_mode = cv_mode
        if cv_mode != 'none':
            # Mixing operates on flattened (N_p * P) features per variable
            self.cv_mixer = CrossVariableMixing(
                n_vars=n_vars, seq_len=self.patch_num * patch_len,
                mode=cv_mode, rank=cv_rank, conv_kernel=cv_kernel,
                depth=cv_depth
            )
        else:
            self.cv_mixer = None
        
        # ── SECOND MIXER after pointwise conv (optional) ─────────
        # After pointwise mixes across patches, the features capture
        # both local (depthwise) and cross-patch (pointwise) info.
        # A second mixing layer here sees even richer representations.
        self.cv_post_pw = cv_post_pw
        if cv_mode != 'none' and cv_post_pw:
            self.cv_mixer_post = CrossVariableMixing(
                n_vars=n_vars, seq_len=self.patch_num * patch_len,
                mode=cv_mode, rank=cv_rank, conv_kernel=cv_kernel,
                depth=cv_depth
            )
        else:
            self.cv_mixer_post = None
        
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
        
        # Pointwise 1×1 Conv
        self.pointwise = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.pw_gelu = nn.GELU()
        self.pw_bn = nn.BatchNorm1d(self.patch_num)
        
        # [ABLATABLE] Dilated causal conv
        if use_causal and n_blocks > 0:
            self.causal_blocks = nn.ModuleList([
                DilatedCausalConvBlock(patch_len, kernel_size=3, dilation=2**i)
                for i in range(n_blocks)
            ])
        else:
            self.causal_blocks = nn.ModuleList()
        
        # Flatten + Head
        flat_dim = self.patch_num * patch_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.head_fc1 = nn.Linear(flat_dim, pred_len * 2)
        self.head_gelu = nn.GELU()
        self.head_fc2 = nn.Linear(pred_len * 2, pred_len)
    
    def forward(self, s, B=None, C=None):
        """
        s: (B*C, L) → (B*C, pred_len)
        B, C: batch and channel dims for cross-variable mixing reshape
        """
        if self.padding_patch == 'end':
            s = self.padding_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: (B*C, N_p, P)
        
        # Embed + depthwise
        s = self.embed_gelu(self.embed(s))
        s = self.embed_bn(s)
        res = self.res_proj(s)
        s = self.dw_gelu(self.dw_conv(s))
        s = self.dw_bn(s)
        s = s + res
        # s: (B*C, N_p, P) — learned patch features
        
        # ── CROSS-VARIABLE MIXING ─────────────────────────────────
        # Reshape to expose variable dimension, mix, reshape back.
        # Each variable's N_p*P features encode its temporal patterns.
        # The mixer learns which variables' patterns inform each other.
        if self.cv_mixer is not None and B is not None:
            s_flat = s.reshape(B, C, -1)        # (B, C, N_p*P)
            s_flat = s_flat.transpose(1, 2)      # (B, N_p*P, C)
            s_flat = self.cv_mixer(s_flat)        # mix across C
            s_flat = s_flat.transpose(1, 2)      # (B, C, N_p*P)
            s = s_flat.reshape(B * C, self.patch_num, self.patch_len)
        # ──────────────────────────────────────────────────────────
        
        # Multiscale depthwise
        if self.use_multiscale:
            h = s.transpose(1, 2)
            ms_sum = sum(conv(h) for conv in self.ms_convs)
            ms_sum = self.ms_gelu(ms_sum)
            s = self.ms_bn(s + ms_sum.transpose(1, 2))
        
        # Pointwise
        s = self.pw_gelu(self.pointwise(s))
        s = self.pw_bn(s)
        
        # ── POST-POINTWISE CROSS-VARIABLE MIXING ─────────────────
        # After pointwise has mixed across patches, features are even
        # richer — each variable now has cross-patch context too.
        if self.cv_mixer_post is not None and B is not None:
            s_flat = s.reshape(B, C, -1)        # (B, C, N_p*P)
            s_flat = s_flat.transpose(1, 2)      # (B, N_p*P, C)
            s_flat = self.cv_mixer_post(s_flat)   # mix across C
            s_flat = s_flat.transpose(1, 2)      # (B, C, N_p*P)
            s = s_flat.reshape(B * C, self.patch_num, self.patch_len)
        # ──────────────────────────────────────────────────────────
        
        # Dilated causal
        for block in self.causal_blocks:
            s = block(s)
        
        # Flatten + head
        s = self.flatten(s)
        s = self.head_gelu(self.head_fc1(s))
        s = self.head_fc2(s)
        return s


class TrendStream(nn.Module):
    """
    Linear trend stream with optional cross-variable mixing.
    
    Mixing applied after first FC+pool+LN — on compressed trend
    representations, not raw values.
    """
    def __init__(self, seq_len, pred_len, agg_kernel=5, use_agg_conv=True,
                 n_vars=1, cv_mode='none', cv_rank=32, cv_kernel=7,
                 cv_depth=1):
        super().__init__()
        self.pred_len = pred_len
        self.use_agg_conv = use_agg_conv
        
        if use_agg_conv:
            self.agg_conv = nn.Conv1d(1, 1, kernel_size=agg_kernel, padding=agg_kernel // 2)
        
        self.fc1 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        
        # ── CROSS-VARIABLE MIXING on learned trend features ──────
        self.cv_mode = cv_mode
        if cv_mode != 'none':
            self.cv_mixer = CrossVariableMixing(
                n_vars=n_vars, seq_len=pred_len * 2,
                mode=cv_mode, rank=cv_rank, conv_kernel=cv_kernel,
                depth=cv_depth
            )
        else:
            self.cv_mixer = None
        
        self.fc2 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)
        
        self.fc3 = nn.Linear(pred_len // 2, pred_len)
    
    def forward(self, t, B=None, C=None):
        """
        t: (B*C, L) → (B*C, pred_len)
        B, C: batch and channel dims for cross-variable mixing reshape
        """
        if self.use_agg_conv:
            t_conv = self.agg_conv(t.unsqueeze(1)).squeeze(1)
            t = t + t_conv
        
        t = self.fc1(t)
        t = self.avgpool1(t.unsqueeze(1)).squeeze(1)
        t = self.ln1(t)
        # t: (B*C, pred_len*2) — compressed trend features
        
        # ── CROSS-VARIABLE MIXING ─────────────────────────────────
        if self.cv_mixer is not None and B is not None:
            t_r = t.reshape(B, C, -1)            # (B, C, pred_len*2)
            t_r = t_r.transpose(1, 2)            # (B, pred_len*2, C)
            t_r = self.cv_mixer(t_r)              # mix across C
            t_r = t_r.transpose(1, 2)            # (B, C, pred_len*2)
            t = t_r.reshape(B * C, self.pred_len * 2)
        # ──────────────────────────────────────────────────────────
        
        t = self.fc2(t)
        t = self.avgpool2(t.unsqueeze(1)).squeeze(1)
        t = self.ln2(t)
        
        t = self.fc3(t)
        return t


class ConcatFusion(nn.Module):
    """xPatch-style: concatenate + linear."""
    def __init__(self, pred_len):
        super().__init__()
        self.linear = nn.Linear(pred_len * 2, pred_len)
    
    def forward(self, y_trend, y_seasonal):
        return self.linear(torch.cat([y_trend, y_seasonal], dim=-1))


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


class AdaPatchNetwork(nn.Module):
    """
    Full AdaPatch network.
    
    Cross-variable mixing is applied INSIDE both streams on learned
    representations, not at the input or output level.
    """
    def __init__(self, seq_len, pred_len,
                 patch_len=16, stride=8, padding_patch='end',
                 n_blocks=1, kernel_sizes=(3, 5, 7),
                 agg_kernel=5, use_cross_variable=False,
                 use_multiscale=True, use_causal=True,
                 use_gated_fusion=True, use_agg_conv=True,
                 n_vars=1, cv_mode='none', cv_rank=32, cv_kernel=7,
                 cv_depth=1, cv_post_pw=False,
                 d_model=None):
        super().__init__()
        self.pred_len = pred_len
        
        self.seasonal_stream = SeasonalStream(
            seq_len, pred_len, patch_len, stride,
            n_blocks, kernel_sizes, padding_patch,
            use_multiscale=use_multiscale,
            use_causal=use_causal,
            n_vars=n_vars, cv_mode=cv_mode,
            cv_rank=cv_rank, cv_kernel=cv_kernel,
            cv_depth=cv_depth, cv_post_pw=cv_post_pw,
        )
        self.trend_stream = TrendStream(
            seq_len, pred_len, agg_kernel,
            use_agg_conv=use_agg_conv,
            n_vars=n_vars, cv_mode=cv_mode,
            cv_rank=cv_rank, cv_kernel=cv_kernel,
            cv_depth=cv_depth,
        )
        
        if use_gated_fusion:
            self.fusion = GatedFusion(pred_len)
        else:
            self.fusion = ConcatFusion(pred_len)
    
    def forward(self, seasonal, trend):
        """
        seasonal, trend: (Batch, Input, Channel) → (Batch, Output, Channel)
        """
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        B, C, L = seasonal.shape
        
        s = seasonal.reshape(B * C, L)
        t = trend.reshape(B * C, L)
        
        # Pass B, C so streams can reshape for cross-variable mixing
        y_seas = self.seasonal_stream(s, B=B, C=C)
        y_trend = self.trend_stream(t, B=B, C=C)
        y = self.fusion(y_trend, y_seas)
        
        y = y.reshape(B, C, self.pred_len)
        return y.permute(0, 2, 1)
