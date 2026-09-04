from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

EXPECTED_CELLS = (
    "real_coco",
    "fake_dalle3_t2i",
    "fake_sdxl_t2i",
    "fake_sdxl_i2i",
    "fake_sdxl_ti2i",
)
CELL_LABELS = {
    "real_coco": "Real COCO photograph",
    "fake_dalle3_t2i": "AI: DALL-E 3 (T2I)",
    "fake_sdxl_t2i": "AI: SDXL (T2I)",
    "fake_sdxl_i2i": "AI: SDXL (I2I)",
    "fake_sdxl_ti2i": "AI: SDXL (TI2I)",
}
REPRESENTATION_COLORS = {"RGB": "#4472C4", "FFT": "#E07A5F"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_image_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def select_validation_parent(
    manifest: pd.DataFrame,
    project_root: Path,
    *,
    require_files: bool = True,
) -> pd.DataFrame:
    """Select the lowest-ID complete validation parent without using model scores."""
    required = {"split", "parent_group", "parent_coco_image_id", "cell", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    candidates: list[tuple[int, str, pd.DataFrame]] = []
    validation = manifest.loc[manifest["split"].eq("val")]
    for parent_group, rows in validation.groupby("parent_group", sort=False):
        if len(rows) != len(EXPECTED_CELLS) or set(rows["cell"]) != set(EXPECTED_CELLS):
            continue
        if require_files and not all(
            _resolved_image_path(str(path), project_root).is_file() for path in rows["path"]
        ):
            continue
        parent_id = int(rows["parent_coco_image_id"].iloc[0])
        candidates.append((parent_id, str(parent_group), rows.copy()))

    if not candidates:
        raise ValueError("No complete validation parent group with readable images was found")
    _, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    order = {cell: index for index, cell in enumerate(EXPECTED_CELLS)}
    return selected.sort_values("cell", key=lambda values: values.map(order)).reset_index(drop=True)


def fft_log_magnitude(image: Image.Image, image_size: int = 384) -> np.ndarray:
    raster = image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    gray = np.asarray(raster.convert("L"), dtype=np.float32) / 255.0
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    low, high = np.percentile(spectrum, (1.0, 99.5))
    if high <= low:
        return np.zeros_like(spectrum)
    return np.clip((spectrum - low) / (high - low), 0.0, 1.0)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_parent_examples(rows: pd.DataFrame, project_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14.2, 3.15))
    for axis, (_, row) in zip(axes, rows.iterrows(), strict=True):
        image = Image.open(_resolved_image_path(str(row["path"]), project_root)).convert("RGB")
        axis.imshow(image)
        axis.set_title(CELL_LABELS[str(row["cell"])], pad=8, fontweight="bold", fontsize=9.5)
        axis.axis("off")
    fig.subplots_adjust(wspace=0.035)
    _save(fig, output)


def plot_rgb_fft(rows: pd.DataFrame, project_root: Path, output: Path) -> None:
    selected = rows.loc[rows["cell"].isin(["real_coco", "fake_sdxl_i2i"])]
    selected = selected.sort_values(
        "cell", key=lambda values: values.map({"real_coco": 0, "fake_sdxl_i2i": 1})
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.1))
    for row_index, (_, row) in enumerate(selected.iterrows()):
        image = Image.open(_resolved_image_path(str(row["path"]), project_root)).convert("RGB")
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_ylabel(
            "Real" if int(row["label"]) == 0 else "AI-generated", fontweight="bold"
        )
        axes[row_index, 1].imshow(fft_log_magnitude(image), cmap="magma")
        for column in range(2):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
    axes[0, 0].set_title("Common 384 x 384 RGB raster", fontweight="bold")
    axes[0, 1].set_title("Shifted log FFT magnitude", fontweight="bold")
    fig.subplots_adjust(hspace=0.06, wspace=0.04)
    _save(fig, output)


def plot_split_composition(manifest: pd.DataFrame, output: Path) -> None:
    counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(["train", "val", "test"])
    labels = ["Train", "Validation", "Internal test"]
    real = counts[0].to_numpy()
    synthetic = counts[1].to_numpy()
    fig, axis = plt.subplots(figsize=(9.2, 3.7))
    y = np.arange(len(labels))
    axis.barh(y, real, color="#5B8E7D", label="Real")
    axis.barh(y, synthetic, left=real, color="#D97A5D", label="AI-generated")
    for index, (real_count, fake_count) in enumerate(zip(real, synthetic, strict=True)):
        axis.text(
            real_count / 2,
            index,
            f"{real_count:,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        axis.text(
            real_count + fake_count / 2,
            index,
            f"{fake_count:,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        axis.text(
            real_count + fake_count + 85,
            index,
            f"total {real_count + fake_count:,}",
            va="center",
            fontsize=9,
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Images")
    axis.set_xlim(0, (real + synthetic).max() * 1.12)
    axis.legend(loc="lower right", frameon=False, ncols=2)
    axis.grid(axis="x", alpha=0.2)
    _save(fig, output)


def load_seed_metrics(project_root: Path) -> pd.DataFrame:
    records = []
    for representation in ("rgb", "fft"):
        for seed in (7, 17, 42):
            path = (
                project_root
                / "artifacts"
                / "experiments"
                / f"dani_{representation}_scratch_seed{seed}_validation_v1"
                / "validation_metrics.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "representation": representation.upper(),
                    "seed": seed,
                    "roc_auc": float(payload["roc_auc"]),
                    "balanced_accuracy": float(payload["balanced_accuracy"]),
                }
            )
    return pd.DataFrame(records)


def plot_seed_comparison(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("roc_auc", "balanced_accuracy"),
        ("Validation ROC-AUC", "Validation balanced accuracy"),
        strict=True,
    ):
        for representation in ("RGB", "FFT"):
            rows = metrics.loc[metrics["representation"].eq(representation)].sort_values("seed")
            axis.plot(
                rows["seed"],
                rows[metric],
                marker="o",
                linewidth=2.2,
                markersize=7,
                color=REPRESENTATION_COLORS[representation],
                label=representation,
            )
        axis.axhline(
            0.5,
            color="#777777",
            linestyle="--",
            linewidth=1,
            label="chance" if metric == "roc_auc" else None,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Seed")
        axis.set_xticks([7, 17, 42])
        axis.set_ylim(0.45, 1.0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, loc="lower left")
    _save(fig, output)


def clean_metric_row(metrics: pd.DataFrame) -> pd.Series:
    rows = metrics.loc[metrics["condition"].eq("clean") & metrics["scope"].eq("all")]
    if len(rows) != 1:
        raise ValueError(f"Expected one clean aggregate row, found {len(rows)}")
    return rows.iloc[0]


def plot_confusion_matrix(metrics: pd.DataFrame, output: Path) -> None:
    row = clean_metric_row(metrics)
    matrix = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
    row_fraction = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axis = plt.subplots(figsize=(5.4, 4.7))
    image = axis.imshow(row_fraction, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if row_fraction[i, j] > 0.55 else "#1f2937"
            axis.text(
                j,
                i,
                f"{matrix[i, j]}\n{row_fraction[i, j]:.1%}",
                ha="center",
                va="center",
                color=color,
                fontsize=13,
                fontweight="bold",
            )
    axis.set_xticks([0, 1], ["Predicted real", "Predicted AI"])
    axis.set_yticks([0, 1], ["Actual real", "Actual AI"])
    axis.set_title("Frozen clean internal test", fontweight="bold", pad=10)
    axis.set_xlabel("Prediction at validation-selected threshold 0.822677")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalised share")
    _save(fig, output)


def plot_generator_performance(metrics: pd.DataFrame, output: Path) -> None:
    rows = metrics.loc[
        metrics["condition"].eq("clean") & metrics["scope"].eq("real_vs_generator")
    ].copy()
    order = ["Dalle3:T2I", "SD_XL:T2I", "SD_XL:I2I", "SD_XL:TI2I"]
    rows["generator"] = pd.Categorical(rows["generator"], order, ordered=True)
    rows = rows.sort_values("generator")
    labels = ["DALL-E 3\nT2I", "SDXL\nT2I", "SDXL\nI2I", "SDXL\nTI2I"]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.4, 4.1))
    axis.bar(x - width / 2, rows["roc_auc"], width, label="ROC-AUC", color="#4C78A8")
    axis.bar(
        x + width / 2, rows["balanced_accuracy"], width, label="Balanced accuracy", color="#F58518"
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.65, 1.0)
    axis.set_ylabel("Metric value")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=2, loc="upper right")
    _save(fig, output)


def plot_robustness(metrics: pd.DataFrame, output: Path) -> None:
    order = [
        "clean",
        "jpeg_q95",
        "jpeg_q75",
        "jpeg_q50",
        "resize_075",
        "resize_050",
        "gaussian_blur_r1",
    ]
    labels = [
        "Clean",
        "JPEG\nQ95",
        "JPEG\nQ75",
        "JPEG\nQ50",
        "Resize\n0.75",
        "Resize\n0.50",
        "Blur\nr=1",
    ]
    rows = metrics.loc[metrics["scope"].eq("all")].set_index("condition").loc[order]
    x = np.arange(len(order))
    fig, axis = plt.subplots(figsize=(10.2, 4.3))
    axis.plot(x, rows["roc_auc"], marker="o", linewidth=2.2, label="ROC-AUC", color="#4C78A8")
    axis.plot(
        x,
        rows["balanced_accuracy"],
        marker="s",
        linewidth=2.2,
        label="Balanced accuracy",
        color="#F58518",
    )
    axis.axhline(0.5, color="#777777", linestyle="--", linewidth=1, label="chance reference")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.3, 1.0)
    axis.set_ylabel("Metric value")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=3, loc="upper right")
    _save(fig, output)


def plot_pipeline(output: Path) -> None:
    labels = [
        "Pinned DANI\nmetadata + licence",
        "Audited 1024 RGB\ncanonical corpus",
        "Parent-disjoint\ntrain / val / test",
        "Common 384 raster\nRGB or FFT",
        "Matched seeds\n7, 17, 42",
        "Validation-only\nselection",
        "Frozen test +\nrobustness",
    ]
    colors = ["#DCEAF7", "#DCEAF7", "#DCEAF7", "#E8E1F5", "#E8E1F5", "#FCE8D5", "#DDF1E5"]
    positions = [
        (0.35, 1.55),
        (2.85, 1.55),
        (5.35, 1.55),
        (7.85, 1.55),
        (1.60, 0.35),
        (4.10, 0.35),
        (6.60, 0.35),
    ]
    fig, axis = plt.subplots(figsize=(10.4, 4.8))
    axis.set_xlim(0, 10.1)
    axis.set_ylim(0, 2.75)
    axis.axis("off")
    for index, (label, color, (x, y)) in enumerate(
        zip(labels, colors, positions, strict=True), start=1
    ):
        box = FancyBboxPatch(
            (x, y),
            1.90,
            0.72,
            boxstyle="round,pad=0.025,rounding_size=0.05",
            facecolor=color,
            edgecolor="#46515C",
            linewidth=1.1,
        )
        axis.add_patch(box)
        axis.text(
            x + 0.95, y + 0.36, label, ha="center", va="center", fontsize=9.2, fontweight="bold"
        )
        axis.text(
            x + 0.10, y + 0.59, str(index), ha="center", va="center", fontsize=8, color="#46515C"
        )
    for start, end in ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6)):
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        axis.annotate(
            "",
            xy=(x1 - 0.08, y1 + 0.36),
            xytext=(x0 + 1.98, y0 + 0.36),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#46515C"},
        )
    x0, y0 = positions[3]
    x1, y1 = positions[4]
    axis.annotate(
        "",
        xy=(x1 + 0.95, y1 + 0.80),
        xytext=(x0 + 0.95, y0 - 0.08),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#46515C"},
    )
    axis.text(
        5.05,
        0.08,
        "Internal test remains locked until the validation decision is frozen",
        ha="center",
        fontsize=9.3,
        color="#8A3A2B",
    )
    _save(fig, output)


def build_figures(
    project_root: Path, manifest_path: Path, metrics_path: Path, output_dir: Path
) -> dict[str, object]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    evaluation = pd.read_csv(metrics_path)
    selected = select_validation_parent(manifest, project_root)
    outputs = {
        "parent_examples": output_dir / "dani_parent_group_examples.png",
        "rgb_fft": output_dir / "rgb_fft_examples.png",
        "split": output_dir / "dataset_split_composition.png",
        "seeds": output_dir / "validation_seed_comparison.png",
        "confusion": output_dir / "clean_confusion_matrix.png",
        "generators": output_dir / "generator_performance.png",
        "robustness": output_dir / "robustness_metrics.png",
        "pipeline": output_dir / "experimental_pipeline.png",
    }
    plot_parent_examples(selected, project_root, outputs["parent_examples"])
    plot_rgb_fft(selected, project_root, outputs["rgb_fft"])
    plot_split_composition(manifest, outputs["split"])
    plot_seed_comparison(load_seed_metrics(project_root), outputs["seeds"])
    plot_confusion_matrix(evaluation, outputs["confusion"])
    plot_generator_performance(evaluation, outputs["generators"])
    plot_robustness(evaluation, outputs["robustness"])
    plot_pipeline(outputs["pipeline"])

    first = selected.iloc[0]
    record: dict[str, object] = {
        "schema_version": "ai_image_detector_report_figures_v1",
        "selection_rule": "lowest numeric COCO parent ID among complete validation groups with readable files",
        "selected_parent_group": str(first["parent_group"]),
        "selected_parent_coco_image_id": int(first["parent_coco_image_id"]),
        "selected_split": "val",
        "selected_category": str(first.get("category", "")),
        "coco_license_name": str(first.get("official_coco_license_name", "")),
        "coco_license_url": str(first.get("official_coco_license_url", "")),
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "evaluation_metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in outputs.items()
        },
    }
    (output_dir / "report_figures_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible figures for the coursework PDF"
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/audits/dani_rgb1024_integrity_v1/training_manifest.csv"),
    )
    parser.add_argument(
        "--evaluation-metrics",
        type=Path,
        default=Path(
            "artifacts/evaluations/dani_fft_seed17_internal_test_v1/internal_test_metrics.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/report_figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else project_root / args.manifest
    metrics_path = (
        args.evaluation_metrics
        if args.evaluation_metrics.is_absolute()
        else project_root / args.evaluation_metrics
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    )
    record = build_figures(project_root, manifest_path, metrics_path, output_dir)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
