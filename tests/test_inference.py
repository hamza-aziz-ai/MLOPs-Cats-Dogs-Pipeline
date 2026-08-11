"""Unit tests for deterministic model-output post-processing."""

import pytest
import torch
from torch import nn

from cats_dogs_mlops.inference import predict_tensor


class FixedLogitModel(nn.Module):
    """Return fixed logits so inference assertions do not depend on training."""

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Return logits favouring the second class for every batch item."""
        logits = torch.tensor([[-1.0, 2.0]], dtype=input_batch.dtype, device=input_batch.device)
        return logits.expand(input_batch.shape[0], -1)


def test_predict_tensor_returns_probabilities_and_winning_label() -> None:
    """Softmax probabilities should sum to one and select the largest logit."""
    prediction = predict_tensor(
        FixedLogitModel(),
        torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        class_names=("cats", "dogs"),
        device="cpu",
    )

    probabilities = prediction["probabilities"]
    assert set(probabilities) == {"cats", "dogs"}
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-7)
    assert prediction["label"] == "dogs"
    assert prediction["confidence"] == pytest.approx(probabilities["dogs"])
    assert probabilities["dogs"] > probabilities["cats"]
