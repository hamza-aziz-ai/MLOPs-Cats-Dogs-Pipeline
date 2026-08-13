"""Framework-level prediction utilities shared by API and unit tests."""

from __future__ import annotations

from typing import Sequence, TypedDict

import torch
from torch import nn


class PredictionResult(TypedDict):
    """Typed result returned by ``predict_tensor``."""

    label: str
    confidence: float
    probabilities: dict[str, float]


@torch.inference_mode()
def predict_tensor(
    model: nn.Module,
    input_batch: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device | str = "cpu",
) -> PredictionResult:
    """Run inference and convert logits to a labelled probability response.

    For class logits ``z_i``, softmax computes
    ``p_i = exp(z_i) / sum_j(exp(z_j))``. The returned label is the class with
    maximum probability.

    Args:
        model: Evaluation-ready PyTorch classifier.
        input_batch: Float tensor shaped ``(N, 3, H, W)``.
        class_names: Ordered label names matching the model's output indices.
        device: Device on which inference is executed.

    Returns:
        PredictionResult: Predicted label, confidence, and all class
        probabilities for the first item in the batch.

    Raises:
        ValueError: If tensor or output dimensions violate the inference
        contract.
    """

    if input_batch.ndim != 4:
        raise ValueError("input_batch must have shape (batch, channels, height, width)")
    if input_batch.shape[0] < 1:
        raise ValueError("input_batch must contain at least one image")
    if len(class_names) < 2:
        raise ValueError("At least two class names are required")

    model.eval()
    logits = model(input_batch.to(device))
    if logits.ndim != 2 or logits.shape[1] != len(class_names):
        raise ValueError(
            "Model output must have shape (batch, number_of_class_names)"
        )

    probabilities_tensor = torch.softmax(logits[0], dim=0).cpu()
    predicted_index = int(torch.argmax(probabilities_tensor).item())
    probabilities = {
        label: float(probabilities_tensor[index].item())
        for index, label in enumerate(class_names)
    }
    return {
        "label": class_names[predicted_index],
        "confidence": probabilities[class_names[predicted_index]],
        "probabilities": probabilities,
    }
