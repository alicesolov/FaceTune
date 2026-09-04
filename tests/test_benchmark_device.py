from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_device.py"
SPEC = importlib.util.spec_from_file_location("benchmark_device", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark_device = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_device)


def test_require_fresh_output_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    benchmark_device.require_fresh_output(output)
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        benchmark_device.require_fresh_output(output)


def test_mps_preflight_rejects_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    with pytest.raises(RuntimeError, match="ambiguous"):
        benchmark_device.require_mps_readiness(("mps",))


def test_cpu_synchronization_is_a_noop() -> None:
    benchmark_device.synchronize(torch.device("cpu"))


def test_summary_uses_median_cpu_to_mps_speedup() -> None:
    results = [
        {"device": "cpu", "status": "ok", "mean_step_milliseconds": 9.0},
        {"device": "mps", "status": "ok", "mean_step_milliseconds": 3.0},
        {"device": "cpu", "status": "ok", "mean_step_milliseconds": 12.0},
        {"device": "mps", "status": "ok", "mean_step_milliseconds": 4.0},
        {"device": "cpu", "status": "ok", "mean_step_milliseconds": 15.0},
        {"device": "mps", "status": "ok", "mean_step_milliseconds": 5.0},
    ]

    summary = benchmark_device.build_summary(results)

    assert summary["per_device_mean_step_milliseconds"]["cpu"] == {
        "median": 12.0,
        "minimum": 9.0,
        "maximum": 15.0,
    }
    assert summary["per_device_mean_step_milliseconds"]["mps"] == {
        "median": 4.0,
        "minimum": 3.0,
        "maximum": 5.0,
    }
    assert summary["mps_speedup_vs_cpu"] == 3.0
    assert summary["mps_images_per_second_ratio"] == 3.0


def test_trial_device_order_alternates() -> None:
    assert benchmark_device.trial_device_order(("cpu", "mps"), 1) == ["cpu", "mps"]
    assert benchmark_device.trial_device_order(("cpu", "mps"), 2) == ["mps", "cpu"]
