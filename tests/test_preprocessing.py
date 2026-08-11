"""Unit tests for image decoding and inference preprocessing."""

from io import BytesIO

import pytest
import torch
from PIL import Image

from cats_dogs_mlops.preprocessing import InvalidImageError, preprocess_image_bytes


def _encode_image(image: Image.Image, image_format: str = "PNG") -> bytes:
    """Encode an in-memory PIL image for the byte-oriented production API."""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_preprocess_image_bytes_converts_to_rgb_shape_and_unit_range() -> None:
    """A grayscale upload should become a normalized RGB inference batch."""
    grayscale_image = Image.new("L", (19, 11), color=128)
    input_batch = preprocess_image_bytes(_encode_image(grayscale_image), image_size=32)

    assert input_batch.shape == (1, 3, 32, 32)
    assert input_batch.dtype == torch.float32
    assert torch.isfinite(input_batch).all()
    assert float(input_batch.min()) >= 0.0
    assert float(input_batch.max()) <= 1.0
    assert torch.allclose(input_batch[:, 0], input_batch[:, 1])
    assert torch.allclose(input_batch[:, 1], input_batch[:, 2])


@pytest.mark.parametrize("invalid_payload", [b"", b"this is not an image"])
def test_preprocess_image_bytes_rejects_invalid_bytes(invalid_payload: bytes) -> None:
    """Malformed uploads must fail with the package's stable domain exception."""
    with pytest.raises(InvalidImageError):
        preprocess_image_bytes(invalid_payload)
