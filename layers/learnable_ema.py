import torch
import torch.nn as nn
import math


class LearnableEMA(nn.Module):
    """
    Learnable Exponential Moving Average decomposition.
    
    Each variable gets its own alpha, parameterized through sigmoid for
    constrained optimization in (0, 1). Initialized at alpha=0.3 (xPatch default).
    
    Supports two backends:
      - 'matrix': O(L) time via cumsum — fast for L <= 720
      - 'scan':   O(L) time, O(L) memory — sequential fallback
    """
    def __init__(self, num_features, alpha_init=0.3, backend='matrix', reg_lambda=0.01):
        super().__init__()
        self.backend = backend
        self.reg_lambda = reg_lambda
        self.alpha_init = alpha_init
        
        # Parameterize through inverse-sigmoid so sigmoid(alpha_raw) = alpha_init
        logit_init = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_raw = nn.Parameter(torch.full((num_features,), logit_init))
    
    @property
    def alpha(self):
        """Current alpha values per variable, constrained to (0, 1)."""
        return torch.sigmoid(self.alpha_raw)
    
    def regularization_loss(self):
        """L2 penalty pulling alpha toward the xPatch-validated default."""
        return self.reg_lambda * ((self.alpha - self.alpha_init) ** 2).sum()
    
    def forward(self, x):
        """
        Args:
            x: [Batch, Input, Channel] — xPatch convention
        Returns:
            seasonal: [Batch, Input, Channel]
            trend:    [Batch, Input, Channel]
        """
        if self.backend == 'matrix':
            trend = self._forward_matrix(x)
        else:
            trend = self._forward_scan(x)
        seasonal = x - trend
        return seasonal, trend
    
    def _forward_matrix(self, x):
        """
        Vectorized EMA using cumulative sum (matches xPatch's optimized impl).
        s_t = alpha * x_t + (1-alpha) * s_{t-1}, s_0 = x_0
        
        CRITICAL: Uses float64 for weight computation to avoid underflow.
        (1-alpha)^T underflows in float32 for T>150 with typical alpha values.
        xPatch uses the same float64 trick in their EMA implementation.
        """
        B, T, C = x.shape
        alpha = self.alpha.double()              # (C,) in float64
        one_minus_alpha = 1.0 - alpha            # (C,)
        
        # powers_rev: [T-1, T-2, ..., 0] in float64
        powers_rev = torch.arange(T - 1, -1, -1, dtype=torch.float64, device=x.device)
        
        # (1-alpha)^power for each channel: (T, C) in float64
        decay = one_minus_alpha.unsqueeze(0).pow(powers_rev.unsqueeze(1))
        
        # Weights: first element keeps full decay, rest get alpha multiplier
        weights = decay.clone()
        weights[1:] = weights[1:] * alpha.unsqueeze(0)
        
        # Apply: cumsum(x_float64 * weights) / decay, then back to float32
        x_d = x.double()
        weighted = x_d * weights.unsqueeze(0)       # (B, T, C) float64
        cumulative = torch.cumsum(weighted, dim=1)   # (B, T, C) float64
        trend = cumulative / decay.unsqueeze(0)      # (B, T, C) float64
        
        return trend.float()  # back to float32
    
    def _forward_scan(self, x):
        """Sequential scan fallback for very long sequences."""
        B, T, C = x.shape
        alpha = self.alpha
        one_minus_alpha = 1.0 - alpha
        
        s = x[:, 0, :]  # (B, C)
        result = [s.unsqueeze(1)]
        for t in range(1, T):
            s = alpha * x[:, t, :] + one_minus_alpha * s
            result.append(s.unsqueeze(1))
        return torch.cat(result, dim=1)
