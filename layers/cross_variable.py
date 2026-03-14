"""
Cross-Variable Mixing modules for AdaPatch.

Modes:
  'none'         - no mixing (channel-independent baseline)
  'mlp'          - bottleneck MLP (C→r→C at each timestep)
  'conv'         - local conv mixing (shared kernel across C)
  'hydra'        - Hydra Attention: O(Cd) cosine-similarity mixing
  'hydra_gated'  - Hydra + sigmoid gate (filters noisy channels)

All operate on (B, L, C) tensors — mix across the C dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Existing modules (unchanged)
# =====================================================================

class CrossVariableMixingMLP(nn.Module):
    """Bottleneck MLP mixing: C → r → C at each timestep."""
    def __init__(self, n_vars, rank=32, depth=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(n_vars, rank),
                nn.GELU(),
                nn.Linear(rank, n_vars),
            ))
            self.norms.append(nn.LayerNorm(n_vars))

    def forward(self, x):
        """x: (B, L, C) → (B, L, C)"""
        for mix, norm in zip(self.layers, self.norms):
            x = norm(x + mix(x))
        return x


class CrossVariableMixingConv(nn.Module):
    """Depthwise conv mixing across adjacent variables."""
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
        h = x.reshape(B * L, 1, C)
        h = F.gelu(self.conv_up(h))
        h = self.conv_down(h)
        h = h.reshape(B, L, C)
        return self.norm(x + h)


# =====================================================================
# New: Hydra Attention mixing
# =====================================================================

class HydraAttentionCore(nn.Module):
    """
    Core Hydra Attention: O(Nd) linear attention.
    
    output = Q ⊙ Σ_n(K_n ⊙ V_n) with L2-normalized Q, K.
    """
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, N, D) → (B, N, D)"""
        Q = F.normalize(self.W_q(x), p=2, dim=-1)
        K = F.normalize(self.W_k(x), p=2, dim=-1)
        V = self.W_v(x)
        global_feat = (K * V).sum(dim=1, keepdim=True)
        return self.dropout(Q * global_feat)


class CrossVariableMixingHydra(nn.Module):
    """
    Hydra Attention for cross-variable mixing.
    
    Input (B, L, C): at each of L positions, mixes across C variables
    using content-aware cosine similarity attention.
    
    Optionally uses a bottleneck: project L→rank, Hydra, project rank→L.
    This controls parameter count when L is large.
    
    Args:
        n_vars:  Number of variables (C) — used for LayerNorm
        seq_len: Feature dimension per variable (L)
        rank:    Bottleneck rank. If rank >= seq_len, no bottleneck.
        gated:   If True, apply sigmoid gate to filter noisy channels
    """
    def __init__(self, n_vars, seq_len, rank=32, gated=False):
        super().__init__()
        self.gated = gated
        self.use_bottleneck = rank < seq_len

        if self.use_bottleneck:
            self.proj_down = nn.Linear(seq_len, rank)
            self.attn = HydraAttentionCore(rank)
            self.proj_up = nn.Linear(rank, seq_len)
            attn_dim = rank
        else:
            self.attn = HydraAttentionCore(seq_len)
            attn_dim = seq_len

        if gated:
            self.gate = nn.Sequential(
                nn.Linear(attn_dim, attn_dim),
                nn.Sigmoid()
            )

        self.norm = nn.LayerNorm(n_vars)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: (B, L, C) → (B, L, C)
        Transposes internally to treat C as tokens, L as features.
        """
        # (B, L, C) → (B, C, L): each variable is a token with L features
        h = x.transpose(1, 2)

        if self.use_bottleneck:
            h_low = self.proj_down(h)           # (B, C, rank)
            h_attn = self.attn(h_low)           # (B, C, rank)
            if self.gated:
                gate = self.gate(h_low)
                h_attn = h_attn * gate
            h_mixed = self.proj_up(h_attn)      # (B, C, L)
        else:
            h_attn = self.attn(h)               # (B, C, L)
            if self.gated:
                gate = self.gate(h)
                h_attn = h_attn * gate
            h_mixed = h_attn

        # Residual + transpose back
        out = h + self.alpha * h_mixed          # (B, C, L)
        out = out.transpose(1, 2)               # (B, L, C)

        # LayerNorm on C dimension (matching MLP mixer)
        return self.norm(out)


# =====================================================================
# Unified wrapper
# =====================================================================

class CrossVariableMixing(nn.Module):
    """
    Wrapper that selects mixing strategy based on mode.

    Modes:
      'none'        - no mixing (channel-independent)
      'mlp'         - bottleneck MLP (C→r→C)
      'conv'        - local conv (shared kernel across C)
      'hydra'       - Hydra Attention (cosine similarity)
      'hydra_gated' - Hydra + sigmoid gate
    """
    def __init__(self, n_vars, seq_len, mode='none', rank=32,
                 conv_kernel=7, depth=1):
        super().__init__()
        self.mode = mode

        if mode == 'mlp':
            self.mixer = CrossVariableMixingMLP(n_vars, rank=rank, depth=depth)
        elif mode == 'conv':
            self.mixer = CrossVariableMixingConv(
                n_vars, rank=min(rank, 8), conv_kernel=conv_kernel
            )
        elif mode == 'hydra':
            self.mixer = CrossVariableMixingHydra(
                n_vars, seq_len, rank=rank, gated=False
            )
        elif mode == 'hydra_gated':
            self.mixer = CrossVariableMixingHydra(
                n_vars, seq_len, rank=rank, gated=True
            )
        else:
            self.mixer = None

    def forward(self, x):
        """x: (B, L, C) → (B, L, C)"""
        if self.mixer is None:
            return x
        return self.mixer(x)
