from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from ai_image_detector.uploads import MAX_UPLOAD_BYTES, load_image


def test_upload_decoder_normalises_to_rgb_and_rejects_empty_input() -> None:
    source = Image.new("L", (17, 11), color=128)
    encoded = BytesIO()
    source.save(encoded, format="PNG")

    decoded = load_image(encoded.getvalue())

    assert decoded.mode == "RGB"
    assert decoded.size == (17, 11)
    with pytest.raises(ValueError, match="empty"):
        load_image(b"")
    with pytest.raises(ValueError, match="12 MB"):
        load_image(b"0" * (MAX_UPLOAD_BYTES + 1))
