import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossVariableMixingMLP(nn.Module):
    """
    Bottleneck MLP mixing across variables (TSMixer-style).
    
    At each timestep, mixes information across all C variables
    through a low-rank bottleneck: C → r → C.
    
    Params: 2 * C * r + r + C ≈ 2*C*r
    Traffic (C=862, r=32): ~55K params
    Weather (C=21, r=16):  ~700 params
    """
    def __init__(self, n_vars, rank=32):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(n_vars, rank),
            nn.GELU(),
            nn.Linear(rank, n_vars),
        )
        self.norm = nn.LayerNorm(n_vars)
    
    def forward(self, x):
        """x: (B, L, C) → (B, L, C)"""
        return self.norm(x + self.mix(x))


class CrossVariableMixingConv(nn.Module):
    """
    Depthwise conv mixing across adjacent variables.
    
    Applies a shared 1D convolution along the variable dimension
    at each timestep. Assumes variables with nearby indices have
    meaningful spatial relationships (true for traffic sensors,
    electricity meters with geographic ordering).
    
    Uses bottleneck channels: Conv1d(1→r, k) → GELU → Conv1d(r→1, 1)
    
    Params: r*k + r + r + 1 ≈ r*(k+2)
    With r=8, k=7: ~72 params (independent of C!)
    """
    def __init__(self, n_vars, rank=8, conv_kernel=7):
        super().__init__()
        self.conv_up = nn.Conv1d(
            1, rank, kernel_size=conv_kernel, padding=conv_kernel // 2
        )
        self.conv_down = nn.Conv1d(rank, 1, kernel_size=1)
        self.norm = nn.LayerNorm(n_vars)
    
    def forward(self, x):
        """x: (B, L, C) → (B, L, C)"""
        B, L, C = x.shape
        # Reshape: each (batch, timestep) independently mixes across C
        h = x.reshape(B * L, 1, C)      # (B*L, 1, C)
        h = F.gelu(self.conv_up(h))       # (B*L, r, C)
        h = self.conv_down(h)             # (B*L, 1, C)
        h = h.reshape(B, L, C)
        return self.norm(x + h)


class CrossVariableMixing(nn.Module):
    """
    Wrapper that selects mixing strategy based on mode.
    
    Modes:
      'none' - no cross-variable mixing (channel-independent baseline)
      'mlp'  - bottleneck MLP mixing (C→r→C at each timestep)
      'conv' - local conv mixing (shared kernel across C dimension)
    """
    def __init__(self, n_vars, seq_len, mode='none', rank=32, conv_kernel=7):
        super().__init__()
        self.mode = mode
        
        if mode == 'mlp':
            self.mixer = CrossVariableMixingMLP(n_vars, rank=rank)
        elif mode == 'conv':
            self.mixer = CrossVariableMixingConv(
                n_vars, rank=min(rank, 8), conv_kernel=conv_kernel
            )
        else:
            self.mixer = None
    
    def forward(self, x):
        """x: (B, L, C) → (B, L, C)"""
        if self.mixer is None:
            return x
        return self.mixer(x)
