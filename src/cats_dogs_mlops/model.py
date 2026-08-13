"""ResNet-50 (from scratch) baseline and checkpoint utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cats_dogs_mlops.config import CLASS_NAMES, IMAGE_SIZE

# Flattened feature size after the backbone's average pool, for the default
# 224x224 input: 224 -> 112 -> 55 -> 28 -> 14 -> 7 -> 4 spatially, 2048 channels.
_FLATTENED_FEATURES = 4 * 4 * 2048


class _IdentityBlock(nn.Module):
    """ResNet identity block: 3-layer skip connection with no dimension change.

    Args:
        in_channels: Number of channels entering the block. Always equal to
            F3 for every call site in this network (identity blocks never
            change channel count), but kept explicit rather than inferred.
        f: Kernel size of the middle (F2) convolution.
        filters: (F1, F2, F3) output-channel counts for the three convolutions.
    """

    def __init__(self, in_channels: int, f: int, filters: tuple[int, int, int]) -> None:
        super().__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=f, stride=1, padding=f // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(f3)

        self.relu = nn.ReLU(inplace=True)
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.xavier_uniform_(conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + shortcut
        return self.relu(out)


class _ConvolutionalBlock(nn.Module):
    """ResNet convolutional block: skip connection with a projection shortcut.

    Used where the input and output dimensions differ, so a 1x1 strided
    convolution on the shortcut path resizes the input to match the main
    path before the final addition.

    Args:
        in_channels: Number of channels entering the block.
        f: Kernel size of the middle (F2) convolution.
        filters: (F1, F2, F3) output-channel counts for the three convolutions.
        s: Stride applied by the first main-path convolution and the shortcut
            projection, so both paths shrink the spatial dimensions identically.
    """

    def __init__(
        self, in_channels: int, f: int, filters: tuple[int, int, int], s: int = 2
    ) -> None:
        super().__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=s, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=f, stride=1, padding=f // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(f3)

        self.shortcut_conv = nn.Conv2d(in_channels, f3, kernel_size=1, stride=s, padding=0, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(f3)

        self.relu = nn.ReLU(inplace=True)
        for conv in (self.conv1, self.conv2, self.conv3, self.shortcut_conv):
            nn.init.xavier_uniform_(conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_bn(self.shortcut_conv(x))

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + shortcut
        return self.relu(out)


class CatDogResNet50(nn.Module):
    """ResNet-50, trained from scratch, for two-class RGB image classification.

    CONV -> BN -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK ->
    IDBLOCK*3 -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL ->
    FLATTEN -> DENSE*3

    Randomly initialised (no pretrained ImageNet weights) -- the standard
    ResNet-50 bottleneck stage layout (He et al., 2015), not a smaller
    "baseline" variant. See ``notebooks/01_model_development.ipynb`` for the
    architecture theory and a layer-by-layer derivation.

    Attributes:
        backbone: Stem plus 5 bottleneck stages, producing a (4, 4, 2048)
            feature map for the default 224x224 input.
        classifier: Flatten and a 3-layer fully-connected head to class logits.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES), in_channels: int = 3) -> None:
        """Initialise the backbone and classification head.

        Args:
            num_classes: Number of mutually exclusive output classes.
            in_channels: Number of input image channels.

        Raises:
            ValueError: If fewer than two classes are requested.
        """

        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.zero_pad = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.stage2 = nn.Sequential(
            _ConvolutionalBlock(64, f=3, filters=(64, 64, 256), s=1),
            _IdentityBlock(256, f=3, filters=(64, 64, 256)),
            _IdentityBlock(256, f=3, filters=(64, 64, 256)),
        )
        self.stage3 = nn.Sequential(
            _ConvolutionalBlock(256, f=3, filters=(128, 128, 512), s=2),
            _IdentityBlock(512, f=3, filters=(128, 128, 512)),
            _IdentityBlock(512, f=3, filters=(128, 128, 512)),
            _IdentityBlock(512, f=3, filters=(128, 128, 512)),
        )
        self.stage4 = nn.Sequential(
            _ConvolutionalBlock(512, f=3, filters=(256, 256, 1024), s=2),
            _IdentityBlock(1024, f=3, filters=(256, 256, 1024)),
            _IdentityBlock(1024, f=3, filters=(256, 256, 1024)),
            _IdentityBlock(1024, f=3, filters=(256, 256, 1024)),
            _IdentityBlock(1024, f=3, filters=(256, 256, 1024)),
            _IdentityBlock(1024, f=3, filters=(256, 256, 1024)),
        )
        self.stage5 = nn.Sequential(
            _ConvolutionalBlock(1024, f=3, filters=(512, 512, 2048), s=2),
            _IdentityBlock(2048, f=3, filters=(512, 512, 2048)),
            _IdentityBlock(2048, f=3, filters=(512, 512, 2048)),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False)

        fc1 = nn.Linear(_FLATTENED_FEATURES, 256)
        fc2 = nn.Linear(256, 128)
        fc3 = nn.Linear(128, num_classes)
        for layer in (fc1, fc2, fc3):
            nn.init.xavier_uniform_(layer.weight)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            fc1,
            nn.ReLU(inplace=True),
            fc2,
            nn.ReLU(inplace=True),
            fc3,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute unnormalised class scores for an image batch.

        Args:
            inputs: RGB float tensor shaped ``(batch, 3, height, width)``,
                224x224 by default (see ``_FLATTENED_FEATURES``).

        Returns:
            torch.Tensor: Logits shaped ``(batch, num_classes)``. Softmax is
            intentionally excluded because cross-entropy loss applies it in a
            numerically stable form.
        """

        x = self.zero_pad(inputs)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.avg_pool(x)
        return self.classifier(x)


def create_model(num_classes: int = len(CLASS_NAMES)) -> CatDogResNet50:
    """Construct a fresh baseline model.

    Args:
        num_classes: Number of output classes.

    Returns:
        CatDogResNet50: Randomly initialised network ready for training.
    """

    return CatDogResNet50(num_classes=num_classes)


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
) -> tuple[CatDogResNet50, dict[str, Any]]:
    """Load a trusted project checkpoint for inference.

    ``weights_only=True`` limits deserialisation to tensor and primitive data
    types, which is safer than loading arbitrary pickled Python objects.

    Args:
        checkpoint_path: Existing ``.pt`` checkpoint.
        device: PyTorch device used for weights and inference.

    Returns:
        tuple[CatDogResNet50, dict[str, Any]]: Evaluation-mode model and metadata.

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
