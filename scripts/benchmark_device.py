"""Measure CPU-versus-MPS ResNet-50 training-step throughput at a fixed resolution.

This is a hardware readiness gate, not a model-evaluation script. It measures the same core
operations as one iteration of :func:`ai_image_detector.training.fit`: host batch placement,
forward pass, cross-entropy loss, backward pass, optimizer update, and detached loss accounting.
Fixed synthetic CPU tensors deliberately isolate device compute from image decoding and DataLoader
variation. MPS is synchronised around every timed region because Metal execution is asynchronous.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import nn

from ai_image_detector.models import build_resnet50, trainable_parameter_count
from ai_image_detector.reproducibility import environment_snapshot, save_json

BENCHMARK_SCHEMA = "ai_image_detector_device_benchmark_v1"
BENCHMARK_SCOPE = "synthetic_resnet50_fit_training_step_including_host_to_device_transfer"
MPS_RELEVANT_ENVIRONMENT = (
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_MPS_FAST_MATH",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "mps"),
        default=("cpu", "mps"),
        help="Devices to measure sequentially. The default produces the CPU-to-MPS comparison.",
    )
    parser.add_argument(
        "--conditions",
        default="not recorded",
        help="Optional human-readable hardware conditions, for example power and thermal state.",
    )
    return parser.parse_args()


def require_fresh_output(path: Path) -> None:
    """Keep each hardware measurement an archival record rather than silently replacing it."""
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing benchmark artifact: {path}. "
            "Choose a new --output path for a deliberate rerun."
        )


def require_mps_readiness(devices: tuple[str, ...] | list[str]) -> None:
    """Fail closed instead of allowing unsupported MPS operations to fall back to CPU."""
    if "mps" not in devices:
        return
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK=1 would make this benchmark ambiguous because unsupported "
            "MPS operations may execute on CPU. Unset it before measuring MPS throughput."
        )
    if not torch.backends.mps.is_built():
        raise RuntimeError("This PyTorch build does not include the Apple MPS backend.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available on this host at benchmark launch.")


def synchronize(device: torch.device) -> None:
    """Wait for queued MPS work so elapsed time is a real device measurement."""
    if device.type == "mps":
        torch.mps.synchronize()


def mps_memory_snapshot(device: torch.device) -> dict[str, int] | None:
    """Return optional MPS allocator facts without claiming they are peak memory."""
    if device.type != "mps":
        return None
    values: dict[str, int] = {}
    for name in (
        "current_allocated_memory",
        "driver_allocated_memory",
        "recommended_max_memory",
    ):
        getter = getattr(torch.mps, name, None)
        if getter is None:
            continue
        try:
            values[name] = int(getter())
        except RuntimeError:
            # A particular allocator counter may be unavailable on an otherwise usable MPS runtime.
            continue
    return values


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    images_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    device: torch.device,
) -> float:
    """Execute one ``fit``-equivalent core training iteration and return a finite detached loss."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(images_cpu.to(device))
    loss = criterion(logits, labels_cpu.to(device))
    loss.backward()
    optimizer.step()
    # `fit` performs `float(loss.detach())` to accumulate history, so retain this synchronization
    # point instead of advertising an unrealistically asynchronous MPS throughput figure.
    return float(loss.detach())


def benchmark_one_device(
    *,
    trial: int,
    device_name: str,
    initial_state: dict[str, torch.Tensor],
    images_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, Any]:
    """Measure one device once and return JSON-safe throughput and placement evidence."""
    device = torch.device(device_name)
    model = build_resnet50(pretrained=False)
    model.load_state_dict(initial_state)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    memory_before = mps_memory_snapshot(device)
    model_device = next(model.parameters()).device.type
    if model_device != device.type:
        raise RuntimeError(
            f"Model placement mismatch: expected {device.type!r}, observed {model_device!r}"
        )

    try:
        synchronize(device)
        for _ in range(warmup_steps):
            training_step(model, optimizer, criterion, images_cpu, labels_cpu, device)
        synchronize(device)

        start = time.perf_counter()
        final_loss = 0.0
        for _ in range(timed_steps):
            final_loss = training_step(model, optimizer, criterion, images_cpu, labels_cpu, device)
        synchronize(device)
        elapsed_seconds = time.perf_counter() - start

        if not torch.isfinite(torch.tensor(final_loss)):
            raise RuntimeError(f"Non-finite final loss: {final_loss}")
        return {
            "trial": trial,
            "device": device_name,
            "status": "ok",
            "model_parameter_device": model_device,
            "input_tensor_source_device": images_cpu.device.type,
            "label_tensor_source_device": labels_cpu.device.type,
            "elapsed_seconds": elapsed_seconds,
            "mean_step_milliseconds": elapsed_seconds / timed_steps * 1000,
            "steps_per_second": timed_steps / elapsed_seconds,
            "images_per_second": len(images_cpu) * timed_steps / elapsed_seconds,
            "final_loss": final_loss,
            "mps_memory_before_bytes": memory_before,
            "mps_memory_after_bytes": mps_memory_snapshot(device),
        }
    except RuntimeError as exc:
        return {
            "trial": trial,
            "device": device_name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "model_parameter_device": model_device,
            "mps_memory_before_bytes": memory_before,
            "mps_memory_after_bytes": mps_memory_snapshot(device),
        }
    finally:
        del optimizer
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


def trial_device_order(devices: tuple[str, ...] | list[str], trial: int) -> list[str]:
    """Alternate device order to reduce a one-sided thermal or cache-order bias."""
    ordered = list(devices)
    return ordered if trial % 2 == 1 else list(reversed(ordered))


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise per-device trial distributions and compute a median CPU-to-MPS ratio."""
    successful: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") == "ok":
            successful.setdefault(str(result["device"]), []).append(result)

    per_device: dict[str, dict[str, float] | None] = {}
    for device, device_results in successful.items():
        per_device[device] = _distribution(
            [float(result["mean_step_milliseconds"]) for result in device_results]
        )

    cpu = per_device.get("cpu")
    mps = per_device.get("mps")
    if cpu is None or mps is None:
        speedup: float | None = None
        throughput_ratio: float | None = None
    else:
        speedup = float(cpu["median"]) / float(mps["median"])
        throughput_ratio = speedup
    return {
        "per_device_mean_step_milliseconds": per_device,
        "mps_speedup_vs_cpu": speedup,
        "mps_images_per_second_ratio": throughput_ratio,
    }


def benchmark_environment() -> dict[str, Any]:
    """Record device capability and relevant MPS settings alongside every throughput number."""
    snapshot = environment_snapshot()
    snapshot.update(
        {
            "mps_built": torch.backends.mps.is_built(),
            "torch_num_threads": torch.get_num_threads(),
            "mps_environment": {name: os.environ.get(name) for name in MPS_RELEVANT_ENVIRONMENT},
        }
    )
    return snapshot


def main() -> int:
    args = parse_args()
    if args.image_size <= 0 or args.batch_size <= 0:
        raise ValueError("--image-size and --batch-size must be positive")
    if args.warmup_steps < 0 or args.timed_steps <= 0 or args.trials <= 0:
        raise ValueError(
            "--warmup-steps must be non-negative; --timed-steps and --trials must be positive"
        )
    require_fresh_output(args.output)
    require_mps_readiness(args.devices)

    torch.manual_seed(args.seed)
    initial_model = build_resnet50(pretrained=False)
    initial_state = copy.deepcopy(initial_model.state_dict())
    images_cpu = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    labels_cpu = torch.randint(0, 2, (args.batch_size,), dtype=torch.long)

    results: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        for device_name in trial_device_order(args.devices, trial):
            results.append(
                benchmark_one_device(
                    trial=trial,
                    device_name=device_name,
                    initial_state=initial_state,
                    images_cpu=images_cpu,
                    labels_cpu=labels_cpu,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                )
            )
    failures = [result for result in results if result["status"] != "ok"]
    payload = {
        "schema": BENCHMARK_SCHEMA,
        "status": "completed" if not failures else "completed_with_failures",
        "scope": BENCHMARK_SCOPE,
        "limitations": [
            (
                "Synthetic tensors isolate model compute and host-to-device transfer; this benchmark "
                "does not include image decoding, common-raster preprocessing, or DataLoader "
                "throughput."
            ),
            (
                "MPS execution is synchronised before and after each timed region; it is not "
                "synchronised between steps so the measured result represents sustained loop "
                "throughput."
            ),
        ],
        "conditions": args.conditions,
        "environment": benchmark_environment(),
        "model": {
            "architecture": "resnet50",
            "pretrained": False,
            "trainable_parameters": trainable_parameter_count(initial_model),
        },
        "config": {
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "trials": args.trials,
            "seed": args.seed,
            "devices": list(args.devices),
            "input_shape": list(images_cpu.shape),
            "input_dtype": str(images_cpu.dtype),
        },
        "results": results,
        "summary": build_summary(results),
    }
    save_json(payload, args.output)
    print(args.output)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
