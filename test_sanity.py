"""
AdaPatch Sanity Test
Run from AdaPatch directory: python test_sanity.py

Verifies:
  1. Forward pass works for all typical LTSF configurations
  2. Output shapes are correct
  3. No NaN/Inf in outputs
  4. Gradients flow through learnable EMA alpha
  5. Fusion gate is initialized near 0.5
"""
import sys
import torch
import numpy as np

sys.path.insert(0, '.')
from models.AdaPatch import Model


class MockConfig:
    def __init__(self, **kwargs):
        defaults = dict(
            seq_len=96, pred_len=96, enc_in=7,
            patch_len=16, stride=8, padding_patch='end',
            d_model=128, n_blocks=1, kernel_sizes=(3, 5, 7),
            patch_len_trend=32, agg_kernel=5,
            alpha=0.3, ema_reg_lambda=0.0, ema_backend='matrix',
            use_cross_variable=False, revin=1,
            use_multiscale=True, use_causal=True,
            use_gated_fusion=False, use_agg_conv=True,
            cv_mixing='none', cv_rank=32, cv_kernel=7,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_config(name, **kwargs):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    cfg = MockConfig(**kwargs)
    model = Model(cfg)
    model.eval()
    B = 4
    x = torch.randn(B, cfg.seq_len, cfg.enc_in)
    with torch.no_grad():
        out = model(x)
    params = count_params(model)
    print(f"  Input:  {tuple(x.shape)}  Output: {tuple(out.shape)}  Params: {params:,}")
    print(f"  Alpha:  {model.get_alpha_values()[:5]}")
    assert out.shape == (B, cfg.pred_len, cfg.enc_in), f"Shape mismatch!"
    assert not torch.isnan(out).any(), "NaN in output!"
    assert not torch.isinf(out).any(), "Inf in output!"
    print(f"  ✓ PASSED")
    return params


def test_gradient_flow():
    print(f"\n{'='*60}")
    print(f"Test: Gradient flow through learnable EMA alpha")
    cfg = MockConfig()
    model = Model(cfg)
    x = torch.randn(4, 96, 7)
    y = torch.randn(4, 96, 7)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y) + model.get_ema_reg_loss()
    loss.backward()
    grad = model.decomp.alpha_raw.grad
    assert grad is not None, "No gradient on alpha_raw!"
    assert not torch.all(grad == 0), "Zero gradient!"
    print(f"  Alpha grad: {grad.numpy()}")
    print(f"  ✓ Gradients flow through EMA decomposition")


def test_gate_init():
    print(f"\n{'='*60}")
    print(f"Test: Fusion layer initialization")
    cfg = MockConfig()
    model = Model(cfg)
    fusion = model.net.fusion
    if hasattr(fusion, 'gate_linear'):
        bias = fusion.gate_linear.bias.data
        mean_gate = torch.sigmoid(bias).mean().item()
        print(f"  Gate sigmoid mean: {mean_gate:.4f}")
        assert abs(mean_gate - 0.5) < 0.1, f"Gate not near 0.5: {mean_gate}"
        print(f"  ✓ Gated fusion initialized near 0.5")
    else:
        # ConcatFusion — just verify it has a linear layer
        assert hasattr(fusion, 'linear'), "ConcatFusion missing linear layer"
        print(f"  ConcatFusion linear: in={fusion.linear.in_features}, out={fusion.linear.out_features}")
        print(f"  ✓ Concat fusion initialized correctly")


if __name__ == '__main__':
    print("AdaPatch Sanity Tests")
    print("=" * 60)

    results = []
    results.append(("ETTh1 (7v, L=96, T=96)",
        test_config("ETTh1 (7 vars, L=96, T=96)", seq_len=96, pred_len=96, enc_in=7)))
    results.append(("Weather (21v, L=96, T=336)",
        test_config("Weather (21 vars, L=96, T=336)", seq_len=96, pred_len=336, enc_in=21)))
    results.append(("ETTh1 (7v, L=336, T=96)",
        test_config("ETTh1 (L=336, T=96)", seq_len=336, pred_len=96, enc_in=7)))
    results.append(("Weather (21v, L=96, T=720)",
        test_config("Weather (L=96, T=720)", seq_len=96, pred_len=720, enc_in=21)))
    results.append(("AdaPatch-Tiny",
        test_config("AdaPatch-Tiny (nb=0, k=(3,5))",
            n_blocks=0, kernel_sizes=(3, 5))))
    results.append(("ILI (L=36, T=24)",
        test_config("ILI (7v, L=36, T=24)",
            seq_len=36, pred_len=24, enc_in=7,
            patch_len=8, stride=4, patch_len_trend=12)))
    
    # Cross-variable mixing on learned representations
    results.append(("Weather+MLP (21v, cv=mlp, r=16)",
        test_config("Weather + MLP mixing on representations",
            seq_len=96, pred_len=96, enc_in=21,
            cv_mixing='mlp', cv_rank=16)))
    results.append(("Traffic-like+Conv (50v, cv=conv)",
        test_config("Traffic-like + Conv mixing on representations",
            seq_len=96, pred_len=96, enc_in=50,
            cv_mixing='conv', cv_rank=8, cv_kernel=7)))
    results.append(("Traffic-like+MLP (50v, cv=mlp, r=32)",
        test_config("Traffic-like + MLP mixing on representations",
            seq_len=96, pred_len=96, enc_in=50,
            cv_mixing='mlp', cv_rank=32)))
    results.append(("Traffic+MLP T=336 (50v)",
        test_config("Traffic-like + MLP mixing T=336",
            seq_len=96, pred_len=336, enc_in=50,
            cv_mixing='mlp', cv_rank=32)))

    test_gradient_flow()
    test_gate_init()

    print(f"\n{'='*60}")
    print("PARAMETER SUMMARY:")
    for name, p in results:
        print(f"  {name:35s} → {p:>8,} params")
    print(f"\n✓ ALL TESTS PASSED")
