"""Baseline convolutional neural network and checkpoint utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cats_dogs_mlops.config import CLASS_NAMES, IMAGE_SIZE


class CatDogCNN(nn.Module):
    """Small CNN baseline for two-class RGB image classification.

    The convolutional blocks learn increasingly abstract spatial features.
    Adaptive average pooling makes the classifier independent of intermediate
    feature-map dimensions and reduces the parameter count compared with a
    large flattened fully connected layer.

    Attributes:
        features: Three convolution, normalisation, activation, and pooling
            blocks operating on tensors shaped ``(N, 3, H, W)``.
        classifier: Dropout and linear projection to two class logits.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        """Initialise the CNN layers.

        Args:
            num_classes: Number of mutually exclusive output classes.

        Raises:
            ValueError: If fewer than two classes are requested.
        """

        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute unnormalised class scores for an image batch.

        Args:
            inputs: RGB float tensor shaped ``(batch, 3, height, width)``.

        Returns:
            torch.Tensor: Logits shaped ``(batch, num_classes)``. Softmax is
            intentionally excluded because cross-entropy loss applies it in a
            numerically stable form.
        """

        feature_maps = self.features(inputs)
        return self.classifier(feature_maps)


def create_model(num_classes: int = len(CLASS_NAMES)) -> CatDogCNN:
    """Construct a fresh baseline model.

    Args:
        num_classes: Number of output classes.

    Returns:
        CatDogCNN: Randomly initialised network ready for training.
    """

    return CatDogCNN(num_classes=num_classes)


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    class_names: list[str] | tuple[str, ...] = CLASS_NAMES,
    image_size: int = IMAGE_SIZE,
    model_version: str,
    metrics: dict[str, float],
) -> None:
    """Serialize model weights and the minimum inference contract.

    Args:
        model: Trained PyTorch module.
        checkpoint_path: Destination ``.pt`` path.
        class_names: Index-to-label mapping used during training.
        image_size: Training and inference image dimension.
        model_version: Human-readable release identifier.
        metrics: Final evaluation metrics stored with the model.

    Returns:
        None: The checkpoint is written to disk.
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
        "image_size": int(image_size),
        "model_version": str(model_version),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[CatDogCNN, dict[str, Any]]:
    """Load a trusted project checkpoint for inference.

    ``weights_only=True`` limits deserialisation to tensor and primitive data
    types, which is safer than loading arbitrary pickled Python objects.

    Args:
        checkpoint_path: Existing ``.pt`` checkpoint.
        device: PyTorch device used for weights and inference.

    Returns:
        tuple[CatDogCNN, dict[str, Any]]: Evaluation-mode model and metadata.

    Raises:
        FileNotFoundError: If the model artifact is missing.
        ValueError: If the checkpoint does not contain required fields.
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    required_fields = {"model_state_dict", "class_names", "image_size", "model_version"}
    missing_fields = required_fields.difference(checkpoint)
    if missing_fields:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing_fields)}")

    class_names = list(checkpoint["class_names"])
    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "class_names": class_names,
        "image_size": int(checkpoint["image_size"]),
        "model_version": str(checkpoint["model_version"]),
        "metrics": checkpoint.get("metrics", {}),
        "sha256": checkpoint_sha256(checkpoint_path),
    }
    return model, metadata


def checkpoint_sha256(checkpoint_path: Path) -> str:
    """Calculate a stable SHA-256 identifier for a model artifact.

    Args:
        checkpoint_path: Path to the serialized model.

    Returns:
        str: Lower-case hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

