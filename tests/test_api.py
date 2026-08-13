"""Integration tests for FastAPI startup, health, and image prediction."""

from collections.abc import Iterator
from io import BytesIO

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from cats_dogs_mlops.api import app
from cats_dogs_mlops.model import create_model, save_checkpoint


def _jpeg_upload() -> bytes:
    """Create a small valid RGB JPEG without relying on repository test data."""
    image = Image.new("RGB", (32, 32), color=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def api_client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    """Start the API lifespan with an isolated, valid model checkpoint."""
    torch.manual_seed(7)
    checkpoint_path = tmp_path / "resnet50_baseline.pt"
    save_checkpoint(
        create_model(),
        checkpoint_path,
        class_names=("cats", "dogs"),
        image_size=224,
        model_version="test-model-v1",
        metrics={"accuracy": 1.0},
    )
    monkeypatch.setenv("MODEL_PATH", str(checkpoint_path))
    with TestClient(app) as client:
        yield client


def test_health_reports_ready_model(api_client: TestClient) -> None:
    """The health endpoint should only succeed after checkpoint startup."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_accepts_multipart_image(api_client: TestClient) -> None:
    """The public endpoint should accept field `image` and return probabilities."""
    response = api_client.post(
        "/predict",
        files={"image": ("sample.jpg", _jpeg_upload(), "image/jpeg")},
    )

    assert response.status_code == 200, response.text
    prediction = response.json()
    assert prediction["label"] in {"cats", "dogs"}
    assert set(prediction["probabilities"]) == {"cats", "dogs"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["confidence"] == pytest.approx(
        prediction["probabilities"][prediction["label"]]
    )
