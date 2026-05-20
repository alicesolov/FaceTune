"""Debug script: validates the HybridDetector forward pass.

Checks:
  - embedding shapes from FFT and CLIP branches
  - output logit shape and range
  - no NaN / Inf values
  - parameter counts (frozen vs trainable)

Usage (CPU, no data needed):
    python scripts/debug_hybrid_forward.py

    # After phase-2 unfreeze:
    python scripts/debug_hybrid_forward.py --unfreeze-fft

    # With a real checkpoint:
    python scripts/debug_hybrid_forward.py --weights model/model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_debug(
    unfreeze_fft: bool = False,
    weights_path: Path | None = None,
    device_str: str = "cpu",
) -> bool:
    """Run forward pass debug. Returns True if all checks pass."""
    print("=" * 60)
    print("HybridDetector forward pass debug")
    print("=" * 60)

    device = torch.device(device_str if device_str != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}\n")

    # ── Build model ────────────────────────────────────────────────────────
    print("Building HybridDetector …")
    try:
        from model.hybrid import HybridDetector, load_hybrid_weights
    except ImportError as exc:
        print(f"[FAIL] Cannot import HybridDetector: {exc}")
        return False

    try:
        model = HybridDetector(freeze_fft=True, freeze_clip=True)
    except Exception as exc:
        print(f"[FAIL] HybridDetector() raised: {exc}")
        return False

    if unfreeze_fft:
        model.unfreeze_fft_last_block()
        print("  → FFT layer4 unfrozen (phase-2 mode)")

    if weights_path is not None:
        if not weights_path.exists():
            print(f"[WARN] Checkpoint not found at {weights_path} — skipping weight load")
        else:
            load_hybrid_weights(model, str(weights_path))
            print(f"  → Weights loaded from {weights_path}")

    model = model.to(device).eval()

    # ── Parameter counts ───────────────────────────────────────────────────
    total, trainable = _count_params(model)
    print(f"\nParameters")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}  ({100*trainable/total:.1f}%)")

    # ── Forward pass ───────────────────────────────────────────────────────
    print("\nRunning forward pass (batch_size=2) …")
    fft_dummy = torch.randn(2, 3, 256, 256, device=device)
    clip_dummy = torch.randn(2, 3, 224, 224, device=device)

    checks_passed = True

    with torch.no_grad():
        try:
            logits = model(fft_dummy, clip_dummy)
        except Exception as exc:
            print(f"[FAIL] Forward pass raised: {exc}")
            return False

    # ── Shape check ────────────────────────────────────────────────────────
    shapes = model.embedding_shapes(device=device)
    print("\nEmbedding shapes:")
    for name, shape in shapes.items():
        print(f"  {name}: {shape}")

    expected_logit = (2, 2)
    if tuple(logits.shape) != expected_logit:
        print(f"[FAIL] logits shape {tuple(logits.shape)} ≠ expected {expected_logit}")
        checks_passed = False
    else:
        print(f"\nLogits shape: {tuple(logits.shape)}  ✓")

    # ── NaN / Inf check ────────────────────────────────────────────────────
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    if has_nan:
        print("[FAIL] NaN detected in logits!")
        checks_passed = False
    if has_inf:
        print("[FAIL] Inf detected in logits!")
        checks_passed = False
    if not has_nan and not has_inf:
        print(f"Logit values (first batch): {logits.tolist()}  ✓")

    # ── Softmax check ─────────────────────────────────────────────────────
    probs = torch.softmax(logits, dim=1)
    prob_sums = probs.sum(dim=1)
    if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
        print("[FAIL] Softmax does not sum to 1!")
        checks_passed = False
    else:
        print(f"Softmax probabilities: {probs.tolist()}  ✓")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if checks_passed:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗")
    print("=" * 60)

    return checks_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug HybridDetector forward pass.")
    parser.add_argument("--unfreeze-fft", action="store_true",
                        help="Simulate phase-2: unfreeze FFT layer4")
    parser.add_argument("--weights", default=None, metavar="PATH",
                        help="Optional checkpoint to load weights from")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu | cuda | auto (default: cpu)")
    args = parser.parse_args()

    ok = run_debug(
        unfreeze_fft=args.unfreeze_fft,
        weights_path=Path(args.weights) if args.weights else None,
        device_str=args.device,
    )
    sys.exit(0 if ok else 1)
