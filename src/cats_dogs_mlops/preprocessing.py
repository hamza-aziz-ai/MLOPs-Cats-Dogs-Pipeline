"""Image validation and preprocessing used by training and serving."""

from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from cats_dogs_mlops.config import (
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a supported image."""


def build_training_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Create stochastic augmentation for baseline-CNN training.

    Augmentation teaches the model invariance to small geometric and colour
    changes. Normalization maps RGB values from approximately ``[0, 1]`` to
    ``[-1, 1]``, which usually improves gradient-based optimization.

    Args:
        image_size: Square output height and width in pixels.

    Returns:
        transforms.Compose: Callable that converts a PIL image to a normalized
        tensor with the shape ``(3, image_size, image_size)``.
    """

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )


def build_evaluation_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Create deterministic preprocessing for validation, test, and inference.

    Args:
        image_size: Square output height and width in pixels.

    Returns:
        transforms.Compose: Callable returning a normalized RGB tensor with
        shape ``(3, image_size, image_size)``.
    """

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )


def canonicalize_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> Image.Image:
    """Convert an image to RGB and crop it to the required square dimensions.

    ``ImageOps.fit`` preserves the aspect ratio while applying a centred crop, so
    the image is not stretched. EXIF orientation is applied before conversion.

    Args:
        image: Decoded PIL image in any supported colour mode.
        image_size: Required square height and width in pixels.

    Returns:
        Image.Image: RGB image with size ``(image_size, image_size)``.

    Raises:
        ValueError: If ``image_size`` is not positive.
    """

    if image_size <= 0:
        raise ValueError("image_size must be a positive integer")

    oriented_image = ImageOps.exif_transpose(image)
    rgb_image = oriented_image.convert("RGB")
    return ImageOps.fit(
        rgb_image,
        (image_size, image_size),
        method=Image.Resampling.LANCZOS,
    )


def preprocess_image_bytes(
    image_bytes: bytes,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Decode uploaded bytes and create one inference batch.

    Args:
        image_bytes: Encoded JPEG, PNG, or another Pillow-supported image.
        image_size: Required square height and width in pixels.

    Returns:
        torch.Tensor: Float tensor with shape ``(1, 3, image_size, image_size)``.

    Raises:
        InvalidImageError: If the bytes are empty, corrupt, or unsupported.
    """

    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty")

    try:
        with Image.open(BytesIO(image_bytes)) as decoded_image:
            canonical_image = canonicalize_image(decoded_image, image_size)
            tensor = build_evaluation_transform(image_size)(canonical_image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image") from exc

    # Inference models expect a batch dimension before channels, height, width.
    return tensor.unsqueeze(0)

