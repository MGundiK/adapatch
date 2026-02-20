import torch
import torch.nn as nn

from layers.learnable_ema import LearnableEMA
from layers.network import AdaPatchNetwork
from layers.revin import RevIN


class Model(nn.Module):
    """
    AdaPatch: Adaptive Decomposition with Structured Patch Communication
    for Long-Term Time Series Forecasting.
    
    Architecture:
        RevIN → LearnableEMA → SeasonalStream(+CV mixing) + TrendStream(+CV mixing) → Fusion → RevIN⁻¹
    
    Cross-variable mixing (cv_mode='mlp' or 'conv') operates on LEARNED
    REPRESENTATIONS inside both streams — after patch embedding (seasonal)
    and after first FC layer (trend). This lets variables share information
    based on their temporal patterns, not raw values.
    """
    def __init__(self, configs):
        super().__init__()
        
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in
        
        # RevIN
        self.revin = getattr(configs, 'revin', 1)
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        
        # Learnable EMA decomposition
        alpha_init = getattr(configs, 'alpha', 0.3)
        ema_reg = getattr(configs, 'ema_reg_lambda', 0.0)
        ema_backend = getattr(configs, 'ema_backend', 'matrix')
        self.decomp = LearnableEMA(
            num_features=c_in,
            alpha_init=alpha_init,
            backend=ema_backend,
            reg_lambda=ema_reg
        )
        
        # Network (cross-variable mixing happens inside streams)
        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', 8)
        padding_patch = getattr(configs, 'padding_patch', 'end')
        n_blocks = getattr(configs, 'n_blocks', 1)
        kernel_sizes = getattr(configs, 'kernel_sizes', (3, 5, 7))
        agg_kernel = getattr(configs, 'agg_kernel', 5)
        
        self.net = AdaPatchNetwork(
            seq_len=seq_len, pred_len=pred_len,
            patch_len=patch_len, stride=stride,
            padding_patch=padding_patch,
            n_blocks=n_blocks,
            kernel_sizes=kernel_sizes,
            agg_kernel=agg_kernel,
            use_multiscale=getattr(configs, 'use_multiscale', True),
            use_causal=getattr(configs, 'use_causal', True),
            use_gated_fusion=getattr(configs, 'use_gated_fusion', False),
            use_agg_conv=getattr(configs, 'use_agg_conv', True),
            # Cross-variable mixing (inside streams, on representations)
            n_vars=c_in,
            cv_mode=getattr(configs, 'cv_mixing', 'none'),
            cv_rank=getattr(configs, 'cv_rank', 32),
            cv_kernel=getattr(configs, 'cv_kernel', 7),
        )
    
    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        seasonal, trend = self.decomp(x)
        out = self.net(seasonal, trend)
        
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        
        return out
    
    def get_ema_reg_loss(self):
        return self.decomp.regularization_loss()
    
    def get_alpha_values(self):
        return self.decomp.alpha.detach().cpu().numpy()
