"""Safe, in-memory decoding for the local research interface."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def load_image(contents: bytes) -> Image.Image:
    """Decode one bounded raster image without retaining the original upload."""
    if not contents:
        raise ValueError("The uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file exceeds the 12 MB limit")
    try:
        with Image.open(BytesIO(contents)) as probe:
            probe.verify()
        with Image.open(BytesIO(contents)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("The image has too many pixels")
            return image
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ValueError("The upload is not a supported, safe raster image") from error
