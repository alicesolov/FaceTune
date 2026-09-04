"""Aggregate seeded experiments and make validation-only prototype decisions."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from ai_image_detector.reproducibility import save_json, sha256_file

METRICS = ("roc_auc", "balanced_accuracy", "macro_f1", "fpr_at_tpr_95")
REPRESENTATIVE_SEED = 17
OFFICIAL_REPRESENTATIONS = ("fft", "rgb")
OFFICIAL_SEEDS = (7, 17, 42)
PROTOTYPE_SELECTION_SCHEMA = "ai_image_detector_validation_selection_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CODE_PATHS = (
    "scripts/run_experiment.py",
    "src/ai_image_detector",
    "pyproject.toml",
    "uv.lock",
)


def load_json_object(path: Path, artifact_name: str) -> dict[str, object]:
    """Load one required JSON object with an actionable error message."""
    if not path.is_file():
        raise SystemExit(f"Expected {artifact_name} at {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read valid JSON {artifact_name} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected {artifact_name} at {path} to be a JSON object.")
    return payload


def load_metric_values(
    payload: dict[str, object], path: Path, artifact_name: str
) -> dict[str, float]:
    """Require finite metrics so an incomplete artifact cannot influence a mean."""
    values: dict[str, float] = {}
    for metric in METRICS:
        value = payload.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(
                f"Malformed {artifact_name} at {path}: {metric!r} must be a finite number."
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise SystemExit(
                f"Malformed {artifact_name} at {path}: {metric!r} must be finite."
            )
        values[metric] = numeric
    return values


def load_selection_input(
    experiment_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, float]]:
    """Load only the inputs allowed to influence prototype selection."""
    run_path = experiment_dir / "run.json"
    model_path = experiment_dir / "model.json"
    validation_path = experiment_dir / "validation_metrics.json"
    run = load_json_object(run_path, "run metadata")
    model = load_json_object(model_path, "model launch metadata")
    validation = load_metric_values(
        load_json_object(validation_path, "validation metrics"),
        validation_path,
        "validation metrics",
    )
    return run, model, validation


def load_internal_metrics(experiment_dir: Path) -> dict[str, float] | None:
    """Load optional descriptive internal-test metrics after selection is already decided."""
    summary_path = experiment_dir / "analysis" / "analysis_summary.json"
    if not summary_path.is_file():
        return None
    summary = load_json_object(summary_path, "completed internal-test analysis")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise SystemExit(f"Malformed completed internal-test analysis at {summary_path}: missing metrics.")
    return load_metric_values(metrics, summary_path, "completed internal-test analysis")


def run_identity(run: dict[str, object], experiment_dir: Path) -> tuple[str, int]:
    """Read the two run-specific fields that may vary across controlled repeats."""
    config = run.get("config")
    if not isinstance(config, dict):
        raise SystemExit(f"Malformed run metadata in {experiment_dir}: config must be an object.")
    representation = config.get("representation")
    seed = config.get("seed")
    if not isinstance(representation, str) or not representation.strip():
        raise SystemExit(
            f"Malformed run metadata in {experiment_dir}: config.representation must be non-empty."
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SystemExit(f"Malformed run metadata in {experiment_dir}: config.seed must be an integer.")
    return representation, seed


def validation_threshold(run: dict[str, object], experiment_dir: Path) -> float:
    """Read the frozen, validation-selected operating threshold from the run artifact."""
    threshold = run.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise SystemExit(
            f"Malformed run metadata in {experiment_dir}: validation-selected threshold is missing."
        )
    numeric = float(threshold)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise SystemExit(
            f"Malformed run metadata in {experiment_dir}: validation-selected threshold must be "
            "finite and in [0, 1]."
        )
    return numeric


def checkpoint_sha256(experiment_dir: Path) -> str:
    """Pin a completed candidate checkpoint before it can become the prototype record."""
    checkpoint = experiment_dir / "best_model.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"Expected completed checkpoint at {checkpoint}.")
    return sha256_file(checkpoint)


def assert_model_identity_matches_run(
    run: dict[str, object], model: dict[str, object], experiment_dir: Path
) -> None:
    """Reject an artifact whose launch metadata identifies a different variant than ``run.json``."""
    representation, seed = run_identity(run, experiment_dir)
    launch_options = model.get("launch_options")
    if not isinstance(launch_options, dict) or not isinstance(launch_options.get("requested"), dict):
        raise SystemExit(f"Malformed model launch metadata in {experiment_dir}: requested options missing.")
    requested = launch_options["requested"]
    if requested.get("representation") != representation or requested.get("seed") != seed:
        raise SystemExit(
            f"Model launch metadata and run metadata disagree on representation/seed in {experiment_dir}."
        )


def run_signature(run: dict[str, object], model: dict[str, object]) -> dict[str, object]:
    """Return every controlled-protocol fact that must agree before alternatives compare.

    ``representation``, ``seed`` and ``experiment_name`` identify the variants/repeats and are
    intentionally excluded. The completed model-launch artifact supplies facts that do not belong
    in ``run.json`` (manifest identity, initialization, augmentation, environment, and architecture),
    so aggregation fails closed rather than comparing runs with only partial provenance.
    """
    config = run.get("config")
    preprocessing = run.get("preprocessing")
    sampler = run.get("train_sampler")
    if (
        not isinstance(config, dict)
        or not isinstance(preprocessing, dict)
        or not isinstance(sampler, dict)
    ):
        raise SystemExit("Every experiment must have current config/preprocessing/sampler metadata")
    if not isinstance(preprocessing.get("protocol"), str) or not isinstance(
        preprocessing.get("version"), str
    ) or not isinstance(preprocessing.get("image_size"), int):
        raise SystemExit("Every experiment must have complete preprocessing protocol metadata")
    sampler_choice = sampler.get("choice")
    if not isinstance(sampler_choice, str) or not sampler_choice.strip():
        raise SystemExit("Every experiment must have a non-empty train sampler choice")

    manifest = model.get("manifest")
    model_preprocessing = model.get("preprocessing")
    model_sampler = model.get("train_sampler")
    launch_options = model.get("launch_options")
    model_specification = model.get("model")
    environment = model.get("environment_at_launch")
    if (
        not isinstance(manifest, dict)
        or not isinstance(model_preprocessing, dict)
        or not isinstance(model_sampler, dict)
        or not isinstance(launch_options, dict)
        or not isinstance(model_specification, dict)
        or not isinstance(environment, dict)
    ):
        raise SystemExit(
            "Every experiment must have complete model launch metadata for controlled aggregation"
        )
    manifest_sha256 = manifest.get("manifest_sha256")
    row_counts = manifest.get("row_counts")
    if not isinstance(manifest_sha256, str) or not manifest_sha256.strip() or not isinstance(
        row_counts, dict
    ):
        raise SystemExit("Every experiment must record manifest SHA-256 and split row counts")
    requested = launch_options.get("requested")
    resolved = launch_options.get("resolved")
    if not isinstance(requested, dict) or not isinstance(resolved, dict):
        raise SystemExit("Every experiment must record requested and resolved launch options")
    for field in ("from_scratch", "robust_augmentation"):
        if not isinstance(requested.get(field), bool):
            raise SystemExit(f"Every experiment must record boolean launch option {field!r}")
    for field in ("architecture",):
        value = model_specification.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"Every experiment must record non-empty model field {field!r}")
    if not isinstance(model_specification.get("pretrained"), bool):
        raise SystemExit("Every experiment must record boolean model field 'pretrained'")
    if model_specification["pretrained"] == requested["from_scratch"]:
        raise SystemExit(
            "Every experiment must have consistent from_scratch and model.pretrained metadata"
        )
    trainable_parameters = model_specification.get("trainable_parameters")
    if (
        isinstance(trainable_parameters, bool)
        or not isinstance(trainable_parameters, int)
        or trainable_parameters <= 0
    ):
        raise SystemExit("Every experiment must record a positive model trainable_parameters count")
    git_revision = environment.get("git_revision")
    if not isinstance(git_revision, str) or not git_revision.strip():
        raise SystemExit("Every experiment must record its launch git revision")

    excluded_config_keys = {"experiment_name", "representation", "seed"}
    excluded_launch_keys = {"representation", "seed"}
    return {
        "run": {
            "config": {
                key: value for key, value in config.items() if key not in excluded_config_keys
            },
            "preprocessing": preprocessing,
            "train_sampler": sampler,
        },
        "model_launch": {
            # The resolved checkout path may differ after a legitimate relocation; its immutable
            # manifest bytes and row counts must still agree.
            "manifest": {"manifest_sha256": manifest_sha256, "row_counts": row_counts},
            "preprocessing": model_preprocessing,
            "train_sampler": model_sampler,
            "launch_options": {
                "requested": {
                    key: value for key, value in requested.items() if key not in excluded_launch_keys
                },
                "resolved": resolved,
            },
            "model": model_specification,
            # Git revision is verified separately against training-code paths. Documentation-only
            # commits may legitimately occur while a long multi-seed series is running.
            "environment_at_launch": {
                key: value for key, value in environment.items() if key != "git_revision"
            },
        },
    }


def verify_training_code_revision_equivalence(
    experiment_inputs: list[
        tuple[Path, dict[str, object], dict[str, object], dict[str, float]]
    ],
) -> dict[str, object]:
    """Prove that differing launch commits changed no training-code path.

    Long seed series can overlap a documentation/notebook commit. Treating any HEAD change as a
    model-protocol change would discard valid compute, while silently ignoring revisions would be
    too weak. This gate records every revision and asks Git whether the frozen training paths differ.
    """
    revisions: list[str] = []
    for experiment_dir, _, model, _ in experiment_inputs:
        environment = model.get("environment_at_launch")
        revision = environment.get("git_revision") if isinstance(environment, dict) else None
        if not isinstance(revision, str) or not revision.strip():
            raise SystemExit(f"Missing launch git revision in {experiment_dir}")
        revisions.append(revision)
    unique_revisions = tuple(dict.fromkeys(revisions))
    base = unique_revisions[0]
    for revision in unique_revisions[1:]:
        result = subprocess.run(
            ["git", "diff", "--quiet", base, revision, "--", *TRAINING_CODE_PATHS],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode == 1:
            raise SystemExit(
                "Experiment launch revisions differ in frozen training-code paths: "
                f"{base} versus {revision}"
            )
        if result.returncode != 0:
            raise SystemExit(
                "Could not verify training-code equivalence between launch revisions "
                f"{base} and {revision}"
            )
    return {
        "launch_git_revisions": list(unique_revisions),
        "checked_paths": list(TRAINING_CODE_PATHS),
        "training_code_equal_across_revisions": True,
    }


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Summarise fixed metric columns for an already validated set of seed records."""
    n_seeds = len(frame)
    return {
        metric: {
            "mean": float(pd.to_numeric(frame[metric], errors="raise").mean()),
            "std": (
                float(pd.to_numeric(frame[metric], errors="raise").std(ddof=1))
                if n_seeds > 1
                else None
            ),
            "n_seeds": n_seeds,
        }
        for metric in METRICS
    }


def aggregate_by_representation(
    frame: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Preserve representation boundaries in every aggregate used for comparison."""
    return {
        str(representation): aggregate_metrics(group)
        for representation, group in frame.groupby("representation", sort=True)
    }


def seed_sets_and_duplicate_check(per_seed_validation: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """Return seed sets, rejecting duplicate representation/seed rows outright."""
    result: dict[str, tuple[int, ...]] = {}
    for representation, group in per_seed_validation.groupby("representation", sort=True):
        values = [int(seed) for seed in group["seed"].tolist()]
        if len(set(values)) != len(values):
            raise SystemExit(
                "Duplicate seed for representation "
                f"{representation!r}; each representation/seed pair must be unique."
            )
        result[str(representation)] = tuple(sorted(values))
    return result


def validation_aggregate_rows(
    aggregate: dict[str, dict[str, dict[str, float | int | None]]],
    seed_sets: dict[str, tuple[int, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for representation, metrics in aggregate.items():
        row: dict[str, object] = {
            "representation": representation,
            "seeds": ",".join(str(seed) for seed in seed_sets[representation]),
            "n_seeds": metrics["balanced_accuracy"]["n_seeds"],
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = metrics[metric]["mean"]
            row[f"{metric}_std"] = metrics[metric]["std"]
        rows.append(row)
    return rows


def representative_run(per_seed_validation: pd.DataFrame, representation: str) -> dict[str, object]:
    """Return fixed seed 17, never whichever repeat looks best after measurement."""
    rows = per_seed_validation[
        (per_seed_validation["representation"] == representation)
        & (per_seed_validation["seed"] == REPRESENTATIVE_SEED)
    ]
    if len(rows) != 1:
        raise AssertionError("Representative run must be checked before it is requested")
    return rows.iloc[0].to_dict()


def protocol_issues(seed_sets: dict[str, tuple[int, ...]]) -> list[str]:
    """Describe why validation aggregates are not sufficient for a prototype decision."""
    issues: list[str] = []
    observed_representations = tuple(sorted(seed_sets))
    if observed_representations != OFFICIAL_REPRESENTATIONS:
        issues.append(
            "official prototype selection requires exactly representations "
            f"{list(OFFICIAL_REPRESENTATIONS)}; got {list(observed_representations)}"
        )
    for representation in OFFICIAL_REPRESENTATIONS:
        observed_seeds = seed_sets.get(representation, ())
        if observed_seeds != OFFICIAL_SEEDS:
            issues.append(
                f"representation {representation!r} requires exactly predeclared seeds "
                f"{list(OFFICIAL_SEEDS)}; got {list(observed_seeds)}"
            )
    return issues


def build_prototype_selection(
    per_seed_validation: pd.DataFrame,
    validation_aggregate: dict[str, dict[str, dict[str, float | int | None]]],
    signature: dict[str, object],
) -> dict[str, object]:
    """Create a distinct validation-only record for a later external-evaluation gate."""
    seed_sets = seed_sets_and_duplicate_check(per_seed_validation)
    candidates = validation_aggregate_rows(validation_aggregate, seed_sets)
    issues = protocol_issues(seed_sets)
    selection_rule: dict[str, object] = {
        "kind": "prototype_selection_rule_not_external_validation",
        "metric_source": "validation_metrics.json only",
        "primary_metric": "mean validation balanced_accuracy",
        "tie_breakers": ["mean validation roc_auc", "lexicographic representation name"],
        "expected_representations": list(OFFICIAL_REPRESENTATIONS),
        "expected_seeds": list(OFFICIAL_SEEDS),
        "representative_seed": REPRESENTATIVE_SEED,
        "internal_test_metrics_used_for_selection": False,
        "external_validation_completed": False,
    }
    record: dict[str, object] = {
        "schema_version": PROTOTYPE_SELECTION_SCHEMA,
        "selection_rule": selection_rule,
        "common_run_signature": signature,
        "candidates": candidates,
    }
    if issues:
        record.update(
            {
                "selection_status": "incomplete_protocol",
                "decision": {
                    "made": False,
                    "reason": "; ".join(issues),
                    "selected_representation": None,
                    "representative_seed": REPRESENTATIVE_SEED,
                    "experiment_dir": None,
                },
            }
        )
        return record

    representations = sorted(validation_aggregate)
    selected_representation = min(
        representations,
        key=lambda representation: (
            -float(validation_aggregate[representation]["balanced_accuracy"]["mean"]),
            -float(validation_aggregate[representation]["roc_auc"]["mean"]),
            representation,
        ),
    )
    selected_run = representative_run(per_seed_validation, selected_representation)
    record.update(
        {
            "selection_status": "validation_selected_pending_external_validation",
            "decision": {
                "made": True,
                "reason": (
                    "highest mean validation balanced_accuracy; exact ties resolved by mean "
                    "validation roc_auc then lexicographic representation name"
                ),
                "selected_representation": selected_representation,
                "representative_seed": REPRESENTATIVE_SEED,
                "experiment_dir": selected_run["experiment_dir"],
                "experiment": selected_run["experiment"],
                "checkpoint_sha256": selected_run["checkpoint_sha256"],
                "validation_threshold": selected_run["validation_threshold"],
                "validation_metrics": {
                    metric: selected_run[metric] for metric in METRICS
                },
            },
        }
    )
    return record


def internal_test_payload(
    experiment_inputs: list[
        tuple[Path, dict[str, object], dict[str, object], dict[str, float]]
    ],
) -> tuple[list[dict[str, object]], list[str]]:
    """Optionally collect test reporting, without ever feeding it into selection."""
    records: list[dict[str, object]] = []
    missing: list[str] = []
    for experiment_dir, run, _, _ in experiment_inputs:
        metrics = load_internal_metrics(experiment_dir)
        if metrics is None:
            missing.append(str(experiment_dir))
            continue
        representation, seed = run_identity(run, experiment_dir)
        records.append(
            {
                "experiment": experiment_dir.name,
                "experiment_dir": str(experiment_dir.resolve()),
                "representation": representation,
                "seed": seed,
                **metrics,
            }
        )
    return records, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.experiment_dir) < 2:
        raise SystemExit("Aggregate at least two independently seeded experiments; never choose one best seed.")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing aggregate artifact: {args.output_dir}. "
            "Choose a new --output-dir for a deliberate re-aggregation."
        )

    experiment_inputs: list[
        tuple[Path, dict[str, object], dict[str, object], dict[str, float]]
    ] = []
    signature: dict[str, object] | None = None
    validation_records: list[dict[str, object]] = []
    for experiment_dir in args.experiment_dir:
        run, model, validation_metrics = load_selection_input(experiment_dir)
        assert_model_identity_matches_run(run, model, experiment_dir)
        current_signature = run_signature(run, model)
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise SystemExit(
                "Experiments have incompatible controlled-protocol configuration: "
                f"expected {signature}, got {current_signature} for {experiment_dir}"
            )
        representation, seed = run_identity(run, experiment_dir)
        threshold = validation_threshold(run, experiment_dir)
        checkpoint_hash = checkpoint_sha256(experiment_dir)
        validation_records.append(
            {
                "experiment": experiment_dir.name,
                "experiment_dir": str(experiment_dir.resolve()),
                "representation": representation,
                "seed": seed,
                "checkpoint_sha256": checkpoint_hash,
                "validation_threshold": threshold,
                **validation_metrics,
            }
        )
        experiment_inputs.append((experiment_dir, run, model, validation_metrics))

    per_seed_validation = pd.DataFrame(validation_records).sort_values(["representation", "seed"])
    validation_aggregate = aggregate_by_representation(per_seed_validation)
    assert signature is not None
    signature["revision_equivalence"] = verify_training_code_revision_equivalence(
        experiment_inputs
    )
    prototype_selection = build_prototype_selection(
        per_seed_validation, validation_aggregate, signature
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_validation.to_csv(args.output_dir / "per_seed_validation_metrics.csv", index=False)
    seed_sets = seed_sets_and_duplicate_check(per_seed_validation)
    pd.DataFrame(validation_aggregate_rows(validation_aggregate, seed_sets)).to_csv(
        args.output_dir / "validation_metric_mean_std.csv", index=False
    )
    save_json(prototype_selection, args.output_dir / "prototype_selection_record.json")

    payload: dict[str, object] = {
        "status": "validation_aggregated",
        "selection_rule": (
            "Prototype selection uses validation_metrics.json only and is not external validation. "
            "No best seed is selected from internal-test metrics."
        ),
        "signature": signature,
        "validation_metrics_by_representation": validation_aggregate,
        "prototype_selection_file": "prototype_selection_record.json",
        "prototype_selection": prototype_selection,
    }

    internal_records, missing_internal_analysis = internal_test_payload(experiment_inputs)
    if missing_internal_analysis:
        payload["internal_test_reporting"] = {
            "status": "not_available_for_all_runs",
            "missing_analysis": missing_internal_analysis,
        }
    else:
        per_seed_internal = pd.DataFrame(internal_records).sort_values(["representation", "seed"])
        per_seed_internal.to_csv(args.output_dir / "per_seed_internal_metrics.csv", index=False)
        payload["experiments"] = per_seed_internal.to_dict(orient="records")
        internal_by_representation = aggregate_by_representation(per_seed_internal)
        # ``metrics`` is retained only for the prior single-representation aggregate output.
        # Pooling RGB and FFT test metrics would hide the representation boundary and could be
        # mistaken for a model-selection score.
        if len(internal_by_representation) == 1:
            payload["metrics"] = aggregate_metrics(per_seed_internal)
        payload["internal_test_metrics_by_representation"] = internal_by_representation
        generator_frames: list[pd.DataFrame] = []
        for experiment_dir, run, _, _ in experiment_inputs:
            per_generator_path = experiment_dir / "analysis" / "per_generator_metrics.csv"
            if not per_generator_path.is_file():
                continue
            representation, seed = run_identity(run, experiment_dir)
            per_generator = pd.read_csv(per_generator_path)
            per_generator.insert(0, "seed", seed)
            per_generator.insert(0, "representation", representation)
            per_generator.insert(0, "experiment", experiment_dir.name)
            generator_frames.append(per_generator)
        if generator_frames:
            combined_generators = pd.concat(generator_frames, ignore_index=True)
            combined_generators.to_csv(args.output_dir / "per_seed_generator_metrics.csv", index=False)
            numeric = [metric for metric in METRICS if metric in combined_generators]
            generator_aggregate = (
                combined_generators.groupby("generator", as_index=False)[numeric]
                .agg(["mean", "std"])
                .reset_index()
            )
            generator_aggregate.columns = [
                "_".join(part for part in column if part) if isinstance(column, tuple) else column
                for column in generator_aggregate.columns
            ]
            generator_aggregate.to_csv(args.output_dir / "generator_metric_mean_std.csv", index=False)
            payload["generator_aggregate_file"] = "generator_metric_mean_std.csv"
    save_json(payload, args.output_dir / "aggregate_summary.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
