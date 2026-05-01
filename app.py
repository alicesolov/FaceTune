"""Streamlit app for editorial image authenticity screening.

Supports image upload, FFT preprocessing, ResNet-50 inference (when a trained
checkpoint is available), and diagnostic tabs for the newsroom workflow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from model.predict import DEFAULT_WEIGHTS_PATH, MODES, Predictor, PredictionResult
from preprocessing.fft_transform import fft_preprocess
from preprocessing.visualizations import compute_power_spectrum_stats, render_fft_colormap


SUPPORTED_FORMATS = ("png", "jpg", "jpeg", "webp")
MAX_UPLOAD_MB = 10

MODE_LABELS = {
    "screening": "Screening  (high recall — first-pass filter)",
    "confirm":   "Confirm    (high precision — final decision)",
}
MODE_HELP = {
    "screening": (
        "Threshold 0.50 — catches ~92 % of AI images but also flags ~60 % of "
        "real photos. Use as a first-pass filter; all flagged images need review."
    ),
    "confirm": (
        "Threshold 0.95 — when the tool says AI, it is almost certainly right (~95 % "
        "precision). Recall drops to ~50 %. Use for a final publication gate."
    ),
}

VERDICT_LABEL = {
    "real": "Likely a real photograph",
    "ai_generated": "Likely AI-generated",
    "uncertain": "Uncertain — manual review required",
    "unsupported": "Unsupported content type",
}

EDITORIAL_GUIDANCE = {
    "real": (
        "The frequency pattern is consistent with natural photographic noise. "
        "Standard source verification still applies — confirm origin and context "
        "before publication."
    ),
    "ai_generated": (
        "The frequency pattern shows artefacts typical of AI image generators. "
        "Do not publish without independent source confirmation. "
        "Request the original file and verify with the supplier."
    ),
    "uncertain": (
        "The model cannot confidently classify this image. "
        "Treat it as unverified. Request the original high-resolution file from "
        "the source and apply additional editorial scrutiny."
    ),
    "unsupported": (
        "This tool is designed for natural photographs only. "
        "Screenshots, infographics, diagrams, and UI captures contain structural "
        "patterns that differ fundamentally from photographic content, causing "
        "the model to produce unreliable results."
    ),
}

NEXT_STEPS = {
    "real": [
        "Confirm the photographer and original file metadata",
        "Check for cropping or compositing artefacts separately",
        "Proceed with standard editorial review",
    ],
    "ai_generated": [
        "Do not publish until source is independently confirmed",
        "Request the original RAW or high-resolution file",
        "Escalate to senior editor if source cannot be verified",
    ],
    "uncertain": [
        "Request the original file from the source",
        "Run a reverse image search to check for prior appearances",
        "Consider a second opinion from another verification tool",
    ],
    "unsupported": [
        "Upload the original photograph directly (not a screenshot of it)",
        "If the photo is embedded in a page, save or download the image file",
        "Crop to the photo area only, removing any UI or surrounding content",
    ],
}


# ── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_checkpoint_meta(weights_path: Path) -> dict | None:
    """Load epoch and validation metrics from a saved checkpoint."""
    if not weights_path.exists():
        return None
    try:
        ckpt = torch.load(weights_path, map_location="cpu")
        return {k: ckpt.get(k) for k in ("epoch", "val_accuracy", "val_f1",
                                          "val_precision", "val_recall")}
    except Exception:
        return None


# ── Upload helpers ────────────────────────────────────────────────────────────

def load_uploaded_image(uploaded_file) -> Image.Image:
    if uploaded_file is None:
        raise ValueError("No image was uploaded.")
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds the {MAX_UPLOAD_MB} MB size limit.")
    try:
        return Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The file could not be read as an image.") from exc


# ── Visualizations ────────────────────────────────────────────────────────────

def render_probability_gauge(ai_prob: float, lower: float, upper: float) -> plt.Figure:
    """Horizontal gauge showing AI-generated probability with zone coloring."""
    fig, ax = plt.subplots(figsize=(6, 0.9))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.barh(0, lower, color="#2ecc71", height=0.55, alpha=0.85)
    ax.barh(0, upper - lower, left=lower, color="#f39c12", height=0.55, alpha=0.85)
    ax.barh(0, 1.0 - upper, left=upper, color="#e74c3c", height=0.55, alpha=0.85)

    for xv in (lower, upper):
        ax.axvline(xv, color="#222", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.plot([ai_prob], [0], marker="v", markersize=13,
            color="white", markeredgecolor="#111", markeredgewidth=1.8, zorder=10)

    ax.text(lower / 2, 0.38, "REAL",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="white")
    ax.text((lower + upper) / 2, 0.38, "?",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="white")
    ax.text((upper + 1.0) / 2, 0.38, "AI",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="white")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.55, 0.75)
    ax.set_yticks([])
    ax.set_xticks([0.0, lower, upper, 1.0])
    ax.set_xticklabels(["0%", f"{lower:.0%}", f"{upper:.0%}", "100%"], fontsize=8.5)
    ax.set_xlabel("AI-generated probability", fontsize=8.5, labelpad=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig


# ── Session history ───────────────────────────────────────────────────────────

def _init_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def _record_result(filename: str, result: PredictionResult) -> None:
    entry = {
        "filename": filename,
        "verdict": result.verdict,
        "confidence": result.confidence,
        "ai_prob": result.probabilities.get("ai_generated", 0.0),
    }
    history: list = st.session_state.history
    if not history or history[-1]["filename"] != filename:
        history.append(entry)
    if len(history) > 10:
        st.session_state.history = history[-10:]


def render_session_history() -> None:
    history: list = st.session_state.get("history", [])
    if not history:
        return

    st.divider()
    st.caption(f"Session — {len(history)} image(s) analyzed")

    icons = {"real": "✅", "ai_generated": "🚫", "uncertain": "⚠️", "unsupported": "🚫"}
    for entry in reversed(history):
        icon = icons.get(entry["verdict"], "•")
        name = entry["filename"][:22] + "…" if len(entry["filename"]) > 25 else entry["filename"]
        st.caption(f"{icon} {name} — {entry['confidence']:.0%}")


# ── Tab renderers ─────────────────────────────────────────────────────────────

def render_prediction_tab(image: Image.Image, result: PredictionResult,
                          mode: str = "screening") -> None:
    col_img, col_verdict = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col_verdict:
        label = VERDICT_LABEL.get(result.verdict, result.verdict)

        if result.verdict == "real":
            st.success(f"### ✅ {label}")
        elif result.verdict == "ai_generated":
            st.error(f"### 🚫 {label}")
        elif result.verdict == "unsupported":
            st.warning(f"### 🚫 {label}")
            st.caption(result.note)
        else:
            st.warning(f"### ⚠️ {label}")

        if result.verdict == "unsupported":
            pass  # guidance rendered below, no model metrics to show
        elif result.used_placeholder:
            st.info(
                "**No trained model available.**\n\n"
                "Run `python -m training.train` to train the model. "
                "Until then all verdicts are placeholders."
            )
        else:
            ai_prob = result.probabilities.get("ai_generated", 0.0)
            real_prob = result.probabilities.get("real", 0.0)

            st.metric("Confidence", f"{result.confidence:.0%}")

            c1, c2 = st.columns(2)
            c1.metric("Real", f"{real_prob:.1%}")
            c2.metric("AI-generated", f"{ai_prob:.1%}")

            lower, upper = MODES[mode]
            st.pyplot(render_probability_gauge(ai_prob, lower, upper),
                      use_container_width=True)

    st.divider()
    guidance = EDITORIAL_GUIDANCE.get(result.verdict, "")
    steps = NEXT_STEPS.get(result.verdict, [])

    col_g, col_s = st.columns([3, 2], gap="large")
    with col_g:
        st.subheader("Editorial assessment")
        st.write(guidance)
    with col_s:
        st.subheader("Recommended actions")
        for step in steps:
            st.markdown(f"- {step}")


def render_fft_tab(image: Image.Image) -> None:
    col_orig, col_fft = st.columns(2, gap="medium")

    with col_orig:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    with col_fft:
        st.subheader("FFT log-magnitude spectrum")
        colorized = render_fft_colormap(image)
        st.image(colorized, use_container_width=True)

    st.caption(
        "The spectrum shows the distribution of spatial frequencies in the image. "
        "Natural photographs have smooth, irregular spectra. "
        "AI-generated images may show concentric rings, grid patterns, or unnatural "
        "symmetry caused by the upsampling steps in diffusion and GAN models."
    )

    stats = compute_power_spectrum_stats(image)
    with st.expander("Spectrum statistics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{stats['mean']:.3f}")
        c2.metric("Std dev", f"{stats['std']:.3f}")
        c3.metric("Centre energy", f"{stats['center_energy']:.3f}",
                  help="Mean energy in the central 32×32 region (low-frequency / DC content).")
        c4.metric("Edge energy", f"{stats['edge_energy']:.3f}",
                  help="Mean energy in the outermost 16-pixel rows (high-frequency detail).")

        ratio = (stats["center_energy"] / stats["edge_energy"]
                 if stats["edge_energy"] > 0 else 0.0)
        st.metric("Centre / edge ratio", f"{ratio:.2f}",
                  help="Higher ratio = more energy in low frequencies. "
                       "AI images sometimes show unusually high ratios.")


def render_diagnostics_tab(result: PredictionResult, predictor: Predictor,
                           mode: str = "screening") -> None:
    # ── Model status ──────────────────────────────────────────────────────
    st.subheader("Model status")

    if predictor.using_placeholder:
        st.warning("No trained checkpoint — predictions are placeholders only.")
        st.caption(predictor.model_status_note)
        st.code(
            "# Run these commands to train the model:\n"
            "python -m training.audit --all-splits\n"
            "python -m training.train --epochs 10 --batch-size 32\n"
            "python -m training.evaluate --split test",
            language="bash",
        )
    else:
        st.success("Trained checkpoint loaded.")
        st.caption(predictor.model_status_note)

        meta = load_checkpoint_meta(DEFAULT_WEIGHTS_PATH)
        if meta:
            st.divider()
            st.subheader("Checkpoint quality")
            cols = st.columns(5)
            labels = [
                ("Epoch", "epoch", ""),
                ("Val accuracy", "val_accuracy", ".1%"),
                ("Val F1", "val_f1", ".3f"),
                ("Val precision", "val_precision", ".3f"),
                ("Val recall", "val_recall", ".3f"),
            ]
            for col, (label, key, fmt) in zip(cols, labels):
                val = meta.get(key)
                if val is None:
                    col.metric(label, "n/a")
                elif fmt == "":
                    col.metric(label, str(val))
                elif fmt == ".1%":
                    col.metric(label, f"{val:.1%}")
                else:
                    col.metric(label, f"{val:{fmt}}")

    # ── Content type check result ─────────────────────────────────────────
    if result.content_check is not None:
        st.divider()
        st.subheader("Content screening")
        cc = result.content_check
        if cc.is_supported:
            st.success(f"✅ Passed — content type: **{cc.content_type}**  (confidence {cc.confidence:.0%})")
        else:
            st.warning(f"🚫 Rejected — content type: **{cc.content_type}**")
            st.caption(f"Reason: `{cc.reason}`")
            st.caption(cc.note)

    # ── Probability breakdown ─────────────────────────────────────────────
    if not result.used_placeholder and result.verdict != "unsupported":
        st.divider()
        st.subheader("Probability breakdown")

        probs = result.probabilities
        ai_prob = probs.get("ai_generated", 0.0)

        col1, col2 = st.columns(2)
        col1.metric("Real", f"{probs.get('real', 0.0):.1%}")
        col2.metric("AI-generated", f"{ai_prob:.1%}")

        lower, upper = MODES[mode]
        st.pyplot(render_probability_gauge(ai_prob, lower, upper), use_container_width=True)

        st.divider()
        st.subheader("Threshold policy")
        st.markdown(
            f"| Zone | AI-generated probability | Verdict |\n"
            f"|---|---|---|\n"
            f"| 🟢 Real | < {lower:.0%} | `real` |\n"
            f"| 🟡 Uncertain | {lower:.0%} – {upper:.0%} | `uncertain` |\n"
            f"| 🔴 AI-generated | > {upper:.0%} | `ai_generated` |\n\n"
            f"Current AI probability: **{ai_prob:.1%}** → verdict: **`{result.verdict}`**\n\n"
            f"Mode: **{mode}** — {MODE_HELP[mode]}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="AI Image Detector",
        page_icon="🔍",
        layout="wide",
    )

    _init_session()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("AI Image Detector")
        st.caption("Newsroom image authenticity screening")
        st.divider()

        predictor = get_predictor()

        if predictor.using_placeholder:
            st.warning("⚠️ Model not trained")
            st.caption("Results are placeholders until a checkpoint is available.")
        else:
            st.success("✅ Model ready")
            meta = load_checkpoint_meta(DEFAULT_WEIGHTS_PATH)
            if meta and meta.get("val_f1") is not None:
                st.caption(
                    f"Checkpoint: epoch {meta['epoch']}  "
                    f"val F1 {meta['val_f1']:.3f}"
                )

        st.divider()

        mode = st.radio(
            "Operating mode",
            options=list(MODE_LABELS.keys()),
            format_func=lambda m: MODE_LABELS[m],
            help=(
                "**Screening** — first-pass filter, high recall. "
                "**Confirm** — final decision, high precision."
            ),
        )
        st.caption(MODE_HELP[mode])

        st.divider()

        uploaded_file = st.file_uploader(
            "Upload a photograph",
            type=list(SUPPORTED_FORMATS),
            help=(
                f"Natural photographs only — PNG, JPG, JPEG, WEBP, max {MAX_UPLOAD_MB} MB. "
                "Screenshots, infographics, and diagrams are not supported."
            ),
        )

        render_session_history()

        st.divider()
        st.caption(
            "This tool is a decision aid, not a forensic verdict. "
            "Always apply editorial judgment and source verification."
        )

    # ── Landing page ──────────────────────────────────────────────────────
    if uploaded_file is None:
        st.markdown("## Upload a photograph to begin")
        st.markdown(
            "Use the sidebar to upload a natural photograph (press photo, reportage). "
            "The tool screens for AI-generation artefacts using "
            "frequency analysis (FFT) and a ResNet-50 classifier."
        )

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Upload**")
            st.caption("Upload a clean photograph — not a screenshot or infographic.")
        with col2:
            st.markdown("**2. Screen**")
            st.caption("The model analyses the frequency spectrum for AI artefacts.")
        with col3:
            st.markdown("**3. Decide**")
            st.caption(
                "Review the verdict, confidence score, and editorial guidance "
                "before publication."
            )

        st.divider()
        col_v, col_s = st.columns([3, 2])
        with col_v:
            st.info(
                "**Verdict guide**\n\n"
                "✅ **Real** — frequency pattern consistent with a natural photograph\n\n"
                "🚫 **AI-generated** — artefacts typical of diffusion or GAN models detected\n\n"
                "⚠️ **Uncertain** — model confidence below threshold; manual review required"
            )
        with col_s:
            st.warning(
                "**Supported content**\n\n"
                "✅ Press photos\n\n"
                "✅ Reportage photographs\n\n"
                "🚫 Screenshots\n\n"
                "🚫 Infographics / diagrams\n\n"
                "🚫 Collages or composites"
            )
        return

    # ── Image loading ─────────────────────────────────────────────────────
    try:
        image = load_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return

    # ── Inference ─────────────────────────────────────────────────────────
    try:
        result = predictor.predict(image, mode=mode)
    except Exception as error:  # pragma: no cover
        st.error(f"Inference failed: {error}")
        return

    _record_result(uploaded_file.name, result)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Prediction", "FFT Spectrum", "Diagnostics"])

    with tab1:
        render_prediction_tab(image, result, mode=mode)

    with tab2:
        render_fft_tab(image)

    with tab3:
        render_diagnostics_tab(result, predictor, mode=mode)


if __name__ == "__main__":
    main()
