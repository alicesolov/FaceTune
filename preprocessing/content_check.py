"""Content type pre-screening for editorial image authenticity screening.

Classifies an uploaded image as a clean photograph or an unsupported content
type (screenshot, infographic, UI capture, collage) before the FFT pipeline
runs. Feeding non-photo content into the model causes domain shift and
unreliable verdicts.

Only natural photographs should reach the FFT + model stage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image


# ── Thresholds ────────────────────────────────────────────────────────────────
# Conservative: false-positive on real photos is worse than letting some
# borderline screenshots through.

ASPECT_RATIO_MIN = 0.15   # below → extreme landscape banner / strip
ASPECT_RATIO_MAX = 7.0    # above → extreme portrait / phone screenshot strip
FLAT_BLOCK_THRESHOLD = 8.0   # gray-level std below which a 16×16 block is "flat"
FLAT_BLOCK_RATIO_MAX = 0.60  # fraction of flat blocks above which we reject
DOMINANT_COLOR_MAX = 0.45    # single color fraction above which we reject


@dataclass(frozen=True)
class ContentCheckResult:
    """Outcome of the pre-inference content type screen.

    Purpose:
    Carry enough structured information for both the predictor (gate decision)
    and the app (diagnostic display).

    Fields:
    - is_supported: True → proceed to FFT + model. False → return early.
    - content_type: "photo" | "screenshot" | "unknown".
    - confidence: Heuristic confidence in the classification [0, 1].
    - reason: Internal diagnostic code for logging.
    - note: Editor-facing explanation of the check outcome.
    """

    is_supported: bool
    content_type: str
    confidence: float
    reason: str
    note: str


def check_content_type(image: Image.Image) -> ContentCheckResult:
    """Classify an image as a clean photograph or unsupported content.

    Purpose:
    Gate the FFT + model pipeline so that only content matching the training
    distribution (natural photographs) reaches the classifier. Three heuristics
    run in order of cost; the first rejection wins.

    Inputs:
    - image: PIL image, any mode — converted internally as needed.

    Outputs:
    - ContentCheckResult with is_supported=True for photographs and False for
      screenshots, infographics, and UI-heavy images.

    Heuristics (in order):
    1. Aspect ratio extremes — catches banners, strips, ultra-wide screenshots.
    2. Flat block ratio — primary discriminator; natural photos have continuous
       tonal gradients while UIs have large solid-colour regions.
    3. Dominant colour fraction — catches solid-background images and diagrams.

    Key assumptions:
    - Thresholds are tuned conservatively: a photo that passes all three
      checks always proceeds to the model even if borderline.
    - A false positive (rejecting a real photo) is worse than a false negative
      (letting a borderline screenshot through).

    Failure modes:
    - Propagates errors for images that cannot be decoded as RGB.
    """

    w, h = image.size

    # ── 1. Aspect ratio ───────────────────────────────────────────────────────
    ratio = w / h
    if ratio < ASPECT_RATIO_MIN or ratio > ASPECT_RATIO_MAX:
        return ContentCheckResult(
            is_supported=False,
            content_type="screenshot",
            confidence=0.90,
            reason=f"extreme_aspect_ratio({ratio:.2f})",
            note=(
                f"This image has an extreme aspect ratio ({w}×{h} px). "
                "The model is optimised for standard photographic formats "
                "(landscape, portrait, or square). "
                "Please crop to the photo area before uploading."
            ),
        )

    # Work on a 128×128 thumbnail for all subsequent checks.
    thumb_rgb = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    gray = np.asarray(thumb_rgb.convert("L"), dtype=np.float32)

    # ── 2. Flat block ratio ───────────────────────────────────────────────────
    # Divide the 128×128 thumbnail into 64 non-overlapping 16×16 blocks and
    # count those with very low standard deviation (= visually flat / solid).
    block_size = 16
    n = 128 // block_size  # 8 blocks per axis → 64 total
    blocks = gray.reshape(n, block_size, n, block_size).transpose(0, 2, 1, 3)
    block_stds = blocks.std(axis=(-2, -1))  # shape (n, n)
    flat_ratio = float((block_stds < FLAT_BLOCK_THRESHOLD).mean())

    if flat_ratio > FLAT_BLOCK_RATIO_MAX:
        return ContentCheckResult(
            is_supported=False,
            content_type="screenshot",
            confidence=min(0.95, 0.65 + flat_ratio * 0.35),
            reason=f"flat_block_ratio({flat_ratio:.2f})",
            note=(
                "This image contains large uniform areas typical of screenshots, "
                "UI captures, diagrams, or infographics. "
                "The model is trained on natural photographs and will not produce "
                "reliable results on this type of content."
            ),
        )

    # ── 3. Dominant colour fraction ───────────────────────────────────────────
    # Quantise to 64 colours; if one bucket holds >45 % of all pixels the image
    # is likely a solid-background graphic or a near-monochrome capture.
    quantized = thumb_rgb.quantize(colors=64, method=Image.Quantize.FASTOCTREE)
    counts = Counter(quantized.getdata())
    total_px = 128 * 128
    top1_fraction = counts.most_common(1)[0][1] / total_px

    if top1_fraction > DOMINANT_COLOR_MAX:
        return ContentCheckResult(
            is_supported=False,
            content_type="screenshot",
            confidence=0.78,
            reason=f"dominant_colour({top1_fraction:.2f})",
            note=(
                "A single colour dominates this image, which is unusual for photographs. "
                "Infographics, poster-style images, and solid-background screenshots "
                "are not supported by this tool."
            ),
        )

    # ── Passed all checks ─────────────────────────────────────────────────────
    photo_confidence = max(0.5, 1.0 - flat_ratio)
    return ContentCheckResult(
        is_supported=True,
        content_type="photo",
        confidence=round(photo_confidence, 3),
        reason="passed_all_checks",
        note="Image passed content screening — proceeding with analysis.",
    )
