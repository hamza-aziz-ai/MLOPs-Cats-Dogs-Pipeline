"""Central configuration shared by model training and online inference."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_baseline.pt"
DEFAULT_FEEDBACK_PATH = PROJECT_ROOT / "runtime" / "feedback.csv"

CLASS_NAMES: tuple[str, str] = ("cats", "dogs")
IMAGE_SIZE = 224
NORMALIZATION_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
NORMALIZATION_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)


def model_path_from_environment() -> Path:
    """Return the configured checkpoint path.

    The environment variable allows the same container image to use a mounted
    or registry-delivered model without changing application code.

    Returns:
        Path: Absolute or relative path supplied through ``MODEL_PATH``. When
        absent, the versioned model bundled in this repository is used.
    """

    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def feedback_path_from_environment() -> Path:
    """Return the CSV path used for optional prediction feedback.

    Returns:
        Path: Location where non-sensitive prediction labels are appended.
    """

    return Path(os.getenv("FEEDBACK_PATH", str(DEFAULT_FEEDBACK_PATH)))

