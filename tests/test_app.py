from __future__ import annotations

import importlib.util
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_SPEC = importlib.util.spec_from_file_location("research_app", APP_PATH)
assert APP_SPEC is not None and APP_SPEC.loader is not None
app_module = importlib.util.module_from_spec(APP_SPEC)
APP_SPEC.loader.exec_module(app_module)


def test_app_ignores_legacy_experiment_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("AI_IMAGE_DETECTOR_EXPERIMENT_DIR", "artifacts/smoke_fft_resnet50_seed7")
    monkeypatch.delenv("AI_IMAGE_DETECTOR_SELECTION_RECORD", raising=False)
    monkeypatch.setattr(app_module, "bundle", None)
    monkeypatch.setattr(app_module, "startup_error", None)

    app_module.load_selected_experiment()

    assert app_module.bundle is None
    assert app_module.startup_error is not None
    assert "AI_IMAGE_DETECTOR_SELECTION_RECORD" in app_module.startup_error


def test_app_passes_only_selection_record_to_loader(monkeypatch, tmp_path) -> None:
    selection_record = tmp_path / "frozen_model.json"
    selected_bundle = object()
    seen: list[str] = []
    monkeypatch.setenv("AI_IMAGE_DETECTOR_SELECTION_RECORD", str(selection_record))
    monkeypatch.setattr(app_module, "bundle", None)
    monkeypatch.setattr(app_module, "startup_error", None)
    monkeypatch.setattr(
        app_module.ModelBundle,
        "load_selected",
        classmethod(lambda cls, record: seen.append(str(record)) or selected_bundle),
    )

    app_module.load_selected_experiment()

    assert app_module.bundle is selected_bundle
    assert app_module.startup_error is None
    assert seen == [str(selection_record)]
