"""Fail-fast post-deployment smoke test for health and prediction paths."""

from __future__ import annotations

import argparse
import json
from io import BytesIO

import httpx
from PIL import Image, ImageDraw


def create_smoke_image() -> bytes:
    """Create a valid synthetic JPEG without depending on project data.

    A smoke test verifies service wiring and response contracts, not model
    accuracy. Post-deployment predictive quality is measured separately by
    ``post_deployment_evaluation.py`` using labelled held-out images.

    Returns:
        bytes: Encoded 224 x 224 RGB JPEG suitable for multipart upload.
    """

    image = Image.new("RGB", (224, 224), color=(205, 180, 145))
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((45, 50, 105, 110), fill=(70, 55, 45))
    drawing.ellipse((120, 50, 180, 110), fill=(70, 55, 45))
    drawing.rectangle((70, 100, 155, 185), fill=(125, 95, 70))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def run_smoke_test(base_url: str, timeout_seconds: float) -> dict[str, object]:
    """Call readiness, prediction, model-info, and metrics endpoints.

    Args:
        base_url: Root URL of the deployed API.
        timeout_seconds: Per-request network timeout.

    Returns:
        dict[str, object]: Validated health, prediction, and model metadata.

    Raises:
        httpx.HTTPError: If an endpoint is unreachable or returns an error.
        AssertionError: If a successful response violates the API contract.
    """

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds) as client:
        health_response = client.get("/health")
        health_response.raise_for_status()
        health_payload = health_response.json()
        assert health_payload["status"] == "ready"
        assert health_payload["model_loaded"] is True

        prediction_response = client.post(
            "/predict",
            files={"image": ("smoke.jpg", create_smoke_image(), "image/jpeg")},
        )
        prediction_response.raise_for_status()
        prediction_payload = prediction_response.json()
        assert prediction_payload["label"] in {"cats", "dogs"}
        assert 0.0 <= prediction_payload["confidence"] <= 1.0
        assert set(prediction_payload["probabilities"]) == {"cats", "dogs"}
        assert abs(sum(prediction_payload["probabilities"].values()) - 1.0) < 1e-5
        assert prediction_payload["model_version"] == health_payload["model_version"]

        model_response = client.get("/model/info")
        model_response.raise_for_status()
        model_payload = model_response.json()
        assert len(model_payload["sha256"]) == 64

        metrics_response = client.get("/metrics")
        metrics_response.raise_for_status()
        assert "cats_dogs_http_requests_total" in metrics_response.text
        assert "cats_dogs_predictions_total" in metrics_response.text

    return {
        "health": health_payload,
        "prediction": prediction_payload,
        "model": model_payload,
        "metrics_endpoint": "ok",
    }


def parse_arguments() -> argparse.Namespace:
    """Parse service URL and timeout."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    """Execute checks and print JSON evidence for CI logs."""

    arguments = parse_arguments()
    result = run_smoke_test(arguments.base_url, arguments.timeout_seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

