import torch
import torch.nn as nn

from layers.learnable_ema import LearnableEMA
from layers.network import AdaPatchNetwork
from layers.revin import RevIN


class Model(nn.Module):
    """
    AdaPatch: Adaptive Decomposition with Structured Patch Communication
    for Long-Term Time Series Forecasting.
    
    Drop-in replacement for xPatch — same input/output format:
        Input:  (Batch, Input_len, Channels)
        Output: (Batch, Pred_len, Channels)
    
    Key innovations over GLCN and xPatch:
      1. Learnable per-variable EMA decomposition (adaptive alpha)
      2. Multiscale depthwise conv with non-linearity (GLCN multi-kernel + GELU)
      3. Dilated causal conv for structured inter-patch communication
      4. Gated fusion replacing concatenation + linear
      5. Optional cross-variable gate for high-dimensional datasets
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
        ema_reg = getattr(configs, 'ema_reg_lambda', 0.01)
        ema_backend = getattr(configs, 'ema_backend', 'matrix')
        self.decomp = LearnableEMA(
            num_features=c_in,
            alpha_init=alpha_init,
            backend=ema_backend,
            reg_lambda=ema_reg
        )
        
        # Network
        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', 8)
        padding_patch = getattr(configs, 'padding_patch', 'end')
        d_model = getattr(configs, 'd_model', 128)
        n_blocks = getattr(configs, 'n_blocks', 2)
        kernel_sizes = getattr(configs, 'kernel_sizes', (3, 5, 7))
        patch_len_trend = getattr(configs, 'patch_len_trend', 32)
        agg_kernel = getattr(configs, 'agg_kernel', 5)
        use_cross_variable = getattr(configs, 'use_cross_variable', False)
        
        self.net = AdaPatchNetwork(
            seq_len=seq_len, pred_len=pred_len,
            patch_len=patch_len, stride=stride,
            padding_patch=padding_patch,
            d_model=d_model, n_blocks=n_blocks,
            kernel_sizes=kernel_sizes,
            patch_len_trend=patch_len_trend,
            agg_kernel=agg_kernel,
            use_cross_variable=use_cross_variable,
        )
    
    def forward(self, x):
        """
        Args:
            x: (Batch, Input_len, Channels)
        Returns:
            (Batch, Pred_len, Channels)
        """
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        seasonal, trend = self.decomp(x)
        out = self.net(seasonal, trend)
        
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        
        return out
    
    def get_ema_reg_loss(self):
        """Return EMA alpha regularization loss to add to training loss."""
        return self.decomp.regularization_loss()
    
    def get_alpha_values(self):
        """Return current learned alpha values for analysis."""
        return self.decomp.alpha.detach().cpu().numpy()
