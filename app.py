"""Minimal Streamlit app for the baseline inference flow.

The app supports image upload, validates the image, runs FFT preprocessing, and
uses a local ResNet-50 checkpoint when available. If weights are missing or
unloadable, it returns a safe placeholder result instead.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from model.predict import Predictor
from preprocessing.fft_transform import fft_preprocess


SUPPORTED_FORMATS = ("png", "jpg", "jpeg", "webp")


@st.cache_resource
def get_predictor() -> Predictor:
    """Create and cache the predictor instance.

    Purpose:
    Avoid rebuilding predictor state on every Streamlit rerun.

    Inputs:
    - None.

    Outputs:
    - Cached `Predictor` instance.

    Key assumptions:
    - The current app runs on CPU with placeholder-safe fallback behavior.

    Failure modes:
    - Propagates constructor errors from `Predictor`.
    """

    return Predictor()


def load_uploaded_image(uploaded_file) -> Image.Image:
    """Read the uploaded file into a PIL image.

    Purpose:
    Centralize upload decoding and ensure the app fails with actionable errors.

    Inputs:
    - uploaded_file: Streamlit uploaded file object.

    Outputs:
    - PIL image converted to RGB mode.

    Key assumptions:
    - The uploaded file points to an image in a supported format.

    Failure modes:
    - Raises `ValueError` if the upload is empty or not a valid image.
    """

    if uploaded_file is None:
        raise ValueError("No image was uploaded.")

    data = uploaded_file.getvalue()
    if not data:
        raise ValueError("The uploaded file is empty.")

    try:
        image = Image.open(uploaded_file)
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file could not be read as an image.") from exc


def render_fft_preview(image: Image.Image) -> None:
    """Show the FFT preprocessing output as an image preview.

    Purpose:
    Help users verify that the upload was processed through the frequency-based
    preprocessing pipeline.

    Inputs:
    - image: Uploaded PIL image.

    Outputs:
    - None. Renders a preview into the Streamlit app.

    Key assumptions:
    - The FFT tensor uses identical values across its repeated channels.

    Failure modes:
    - Propagates preprocessing errors if the image is malformed.
    """

    tensor = fft_preprocess(image)
    preview = tensor[0].detach().cpu().numpy()
    preview_uint8 = np.clip(preview * 255.0, 0, 255).astype(np.uint8)
    st.image(preview_uint8, caption="FFT log-magnitude preview", clamp=True)


def main() -> None:
    """Render the minimal Streamlit upload and inference flow.

    Purpose:
    Provide a small but runnable UI for image upload and placeholder-safe
    prediction while the training stack is still under construction.

    Inputs:
    - None.

    Outputs:
    - None. Renders the app in Streamlit.

    Key assumptions:
    - Model weights may not exist yet.

    Failure modes:
    - Displays short actionable error messages for upload and inference failures.
    """

    st.set_page_config(page_title="AI Image Detector", layout="centered")
    st.title("AI Image Detector")
    st.write(
        "Upload an editorial image to run the current baseline inference flow. "
        "This build supports image validation, FFT preprocessing, and ResNet-50 "
        "checkpoint inference when `model/model.pth` is available."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=list(SUPPORTED_FORMATS),
        help="Supported formats: PNG, JPG, JPEG, WEBP",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        image = load_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return

    st.image(image, caption="Uploaded image")

    predictor = get_predictor()

    try:
        result = predictor.predict(image)
    except Exception as error:  # pragma: no cover - defensive UI handling.
        st.error(f"Inference failed: {error}")
        return

    st.subheader("Result")
    st.write(f"Verdict: `{result.verdict}`")
    st.write(f"Confidence: `{result.confidence:.2f}`")
    if result.used_placeholder:
        st.warning(result.note)
    else:
        st.write(result.note)
        st.write(
            "Class probabilities: "
            f"`real={result.probabilities['real']:.2f}`, "
            f"`ai_generated={result.probabilities['ai_generated']:.2f}`"
        )

    st.subheader("Diagnostics")
    render_fft_preview(image)


if __name__ == "__main__":
    main()
