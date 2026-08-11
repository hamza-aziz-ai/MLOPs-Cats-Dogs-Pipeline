"""Evaluate the deployed HTTP service on a labelled held-out image batch."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import httpx
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def select_labelled_images(
    test_dir: Path,
    *,
    samples_per_class: int,
    seed: int,
) -> list[tuple[Path, str]]:
    """Select a balanced, reproducible batch from test/cats and test/dogs.

    Args:
        test_dir: Held-out test split root.
        samples_per_class: Maximum requests sent for each class.
        seed: Deterministic sampling seed.

    Returns:
        list[tuple[Path, str]]: Shuffled image-path and ground-truth pairs.

    Raises:
        FileNotFoundError: If a class directory is missing.
        ValueError: If a class contains no supported images.
    """

    selected: list[tuple[Path, str]] = []
    for class_index, class_name in enumerate(("cats", "dogs")):
        class_dir = test_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing held-out class directory: {class_dir}")
        candidates = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not candidates:
            raise ValueError(f"No supported test images found in {class_dir}")
        random.Random(seed + class_index).shuffle(candidates)
        selected.extend(
            (path, class_name) for path in candidates[:samples_per_class]
        )

    random.Random(seed).shuffle(selected)
    return selected


def evaluate_deployment(
    base_url: str,
    labelled_images: list[tuple[Path, str]],
    *,
    timeout_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Call the live API and calculate classification and latency metrics.

    Args:
        base_url: Root URL of the deployed service.
        labelled_images: Image paths paired with their observed true classes.
        timeout_seconds: Per-request timeout.

    Returns:
        tuple[dict[str, object], list[dict[str, object]]]: Summary metrics and
        request-level, non-sensitive prediction records.
    """

    records: list[dict[str, object]] = []
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    latencies_ms: list[float] = []

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds) as client:
        health_response = client.get("/health")
        health_response.raise_for_status()
        if health_response.json().get("status") != "ready":
            raise RuntimeError("Deployment is not ready")

        for image_path, true_label in labelled_images:
            started_at = time.perf_counter()
            with image_path.open("rb") as image_file:
                response = client.post(
                    "/predict",
                    data={"true_label": true_label},
                    files={
                        "image": (
                            image_path.name,
                            image_file,
                            "image/jpeg",
                        )
                    },
                )
            network_latency_ms = (time.perf_counter() - started_at) * 1000
            response.raise_for_status()
            payload = response.json()

            predicted_label = str(payload["label"])
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            latencies_ms.append(network_latency_ms)
            records.append(
                {
                    "image_name": image_path.name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == true_label,
                    "confidence": float(payload["confidence"]),
                    "api_latency_ms": float(payload["latency_ms"]),
                    "network_latency_ms": network_latency_ms,
                    "model_version": str(payload["model_version"]),
                    "request_id": str(payload["request_id"]),
                }
            )

    sorted_latencies = sorted(latencies_ms)
    percentile_index = min(
        len(sorted_latencies) - 1,
        max(0, int(round(0.95 * len(sorted_latencies) + 0.5)) - 1),
    )
    summary: dict[str, object] = {
        "request_count": len(records),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision_weighted": precision_score(
            true_labels, predicted_labels, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            true_labels, predicted_labels, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            true_labels, predicted_labels, average="weighted", zero_division=0
        ),
        "mean_network_latency_ms": statistics.fmean(latencies_ms),
        "p95_network_latency_ms": sorted_latencies[percentile_index],
        "model_version": records[0]["model_version"],
        "evaluation_source": "deployed_http_api",
    }
    return summary, records


def write_evaluation_artifacts(
    summary: dict[str, object],
    records: list[dict[str, object]],
    *,
    summary_path: Path,
    predictions_path: Path,
) -> None:
    """Persist summary JSON and request-level CSV evidence."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with predictions_path.open("w", newline="", encoding="utf-8") as predictions_file:
        writer = csv.DictWriter(predictions_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_arguments() -> argparse.Namespace:
    """Parse deployed-service evaluation parameters."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--test-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("metrics/post_deployment_metrics.json"),
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("artifacts/post_deployment_predictions.csv"),
    )
    return parser.parse_args()


def main() -> None:
    """Execute evaluation, write evidence, and print metrics for CI logs."""

    arguments = parse_arguments()
    if arguments.samples_per_class <= 0:
        raise ValueError("samples-per-class must be positive")
    labelled_images = select_labelled_images(
        arguments.test_dir,
        samples_per_class=arguments.samples_per_class,
        seed=arguments.seed,
    )
    summary, records = evaluate_deployment(
        arguments.base_url,
        labelled_images,
        timeout_seconds=arguments.timeout_seconds,
    )
    write_evaluation_artifacts(
        summary,
        records,
        summary_path=arguments.summary_path,
        predictions_path=arguments.predictions_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

