"""Train, evaluate, serialize, and MLflow-track the baseline CNN."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from cats_dogs_mlops.model import checkpoint_sha256, create_model, save_checkpoint
from cats_dogs_mlops.preprocessing import (
    build_evaluation_transform,
    build_training_transform,
)

# MLflow >=3.x refuses the filesystem tracking backend ("file:./mlruns") by
# default now that it's in maintenance mode upstream. This project's existing
# run history and submitted evidence (SUBMISSION_REPORT.md, VIDEO_DEMO_SCRIPT.md)
# are tied to that file-based mlruns/ directory, so opt back in rather than
# migrating backends.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments.

    Args:
        seed: Integer used by every pseudorandom generator.

    Returns:
        None: Global random states and deterministic backend flags are updated.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_loaders(
    data_dir: Path,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, DataLoader], list[str]]:
    """Build augmented training and deterministic evaluation data loaders.

    Args:
        data_dir: Processed root containing train/validation/test folders.
        image_size: Input dimension expected by the model.
        batch_size: Images processed per optimisation step.
        num_workers: Background loader processes; zero is safest on Windows.
        seed: Seed used for deterministic training shuffling.

    Returns:
        tuple[dict[str, DataLoader], list[str]]: Loaders keyed by split and the
        alphabetical class-to-index ordering used by ImageFolder.

    Raises:
        FileNotFoundError: If a required split is absent.
        ValueError: If class mappings differ between splits.
    """

    datasets: dict[str, ImageFolder] = {}
    for split_name in ("train", "validation", "test"):
        split_path = data_dir / split_name
        if not split_path.is_dir():
            raise FileNotFoundError(f"Required split directory not found: {split_path}")
        transform = (
            build_training_transform(image_size)
            if split_name == "train"
            else build_evaluation_transform(image_size)
        )
        datasets[split_name] = ImageFolder(split_path, transform=transform)

    reference_classes = datasets["train"].classes
    for split_name, image_dataset in datasets.items():
        if image_dataset.classes != reference_classes:
            raise ValueError(
                f"Class mapping mismatch for {split_name}: {image_dataset.classes}"
            )

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split_name: DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if split_name == "train" else None,
        )
        for split_name, image_dataset in datasets.items()
    }
    return loaders, list(reference_classes)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation pass through a dataset.

    When an optimizer is supplied, gradients are computed and weights updated.
    Without one, the same function performs evaluation under disabled gradients.

    Args:
        model: CNN classifier.
        data_loader: Batches of image tensors and integer class indices.
        loss_function: Cross-entropy objective.
        device: CPU or CUDA execution device.
        optimizer: Optional optimizer indicating training mode.

    Returns:
        tuple[float, float]: Mean sample-weighted loss and accuracy.
    """

    is_training = optimizer is not None
    model.train(mode=is_training)
    accumulated_loss = 0.0
    correct_predictions = 0
    sample_count = 0

    for image_batch, target_batch in data_loader:
        image_batch = image_batch.to(device)
        target_batch = target_batch.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(image_batch)
            loss = loss_function(logits, target_batch)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = target_batch.shape[0]
        accumulated_loss += float(loss.item()) * batch_size
        correct_predictions += int((logits.argmax(dim=1) == target_batch).sum().item())
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("A data loader contained no samples")
    return accumulated_loss / sample_count, correct_predictions / sample_count


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Collect true and predicted indices for test-set metrics.

    Args:
        model: Trained classifier in evaluation mode.
        data_loader: Held-out test batches.
        device: CPU or CUDA execution device.

    Returns:
        tuple[list[int], list[int]]: Ground-truth and predicted class indices.
    """

    model.eval()
    true_indices: list[int] = []
    predicted_indices: list[int] = []
    for image_batch, target_batch in data_loader:
        logits = model(image_batch.to(device))
        true_indices.extend(target_batch.tolist())
        predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())
    return true_indices, predicted_indices


def plot_learning_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Save loss and accuracy curves for MLflow and the submission package."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["validation_loss"], marker="o", label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], marker="o", label="Validation")
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    true_indices: list[int],
    predicted_indices: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save a labelled test-set confusion matrix."""

    matrix = confusion_matrix(
        true_indices,
        predicted_indices,
        labels=list(range(len(class_names))),
    )
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title("Held-out test confusion matrix")
    display.figure_.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.figure_.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(display.figure_)


def git_commit_or_unknown() -> str:
    """Return the current Git revision without making training depend on Git."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"


def train(arguments: argparse.Namespace) -> dict[str, Any]:
    """Execute one tracked training run and create all required artifacts.

    Args:
        arguments: Validated command-line options controlling data, optimiser,
        output paths, and MLflow configuration.

    Returns:
        dict[str, Any]: Final test metrics and provenance metadata.
    """

    seed_everything(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_names = create_data_loaders(
        arguments.data_dir,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
    )

    model = create_model(num_classes=len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    arguments.artifacts_dir.mkdir(parents=True, exist_ok=True)
    arguments.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    mlflow.set_tracking_uri(arguments.tracking_uri)
    mlflow.set_experiment(arguments.experiment_name)
    training_started = time.perf_counter()

    with mlflow.start_run(run_name=f"baseline-cnn-{arguments.model_version}") as active_run:
        mlflow.log_params(
            {
                "architecture": "CatDogResNet50",
                "image_size": arguments.image_size,
                "batch_size": arguments.batch_size,
                "epochs": arguments.epochs,
                "learning_rate": arguments.learning_rate,
                "weight_decay": arguments.weight_decay,
                "seed": arguments.seed,
                "device": str(device),
                "class_names": ",".join(class_names),
                "train_samples": len(cast(Sized, loaders["train"].dataset)),
                "validation_samples": len(cast(Sized, loaders["validation"].dataset)),
                "test_samples": len(cast(Sized, loaders["test"].dataset)),
                "git_commit": git_commit_or_unknown(),
            }
        )
        mlflow.set_tags(
            {
                "course": "AIMLCZG523",
                "assignment": "2",
                "use_case": "pet-adoption-cats-vs-dogs",
                "model_version": arguments.model_version,
            }
        )

        for epoch_index in range(arguments.epochs):
            train_loss, train_accuracy = run_epoch(
                model,
                loaders["train"],
                loss_function,
                device,
                optimizer,
            )
            validation_loss, validation_accuracy = run_epoch(
                model,
                loaders["validation"],
                loss_function,
                device,
            )

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
                step=epoch_index + 1,
            )
            print(
                f"Epoch {epoch_index + 1:02d}/{arguments.epochs:02d} "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
            )

        true_indices, predicted_indices = collect_predictions(
            model,
            loaders["test"],
            device,
        )
        test_metrics = {
            "test_accuracy": accuracy_score(true_indices, predicted_indices),
            "test_precision_weighted": precision_score(
                true_indices,
                predicted_indices,
                average="weighted",
                zero_division=0,
            ),
            "test_recall_weighted": recall_score(
                true_indices,
                predicted_indices,
                average="weighted",
                zero_division=0,
            ),
            "test_f1_weighted": f1_score(
                true_indices,
                predicted_indices,
                average="weighted",
                zero_division=0,
            ),
        }

        loss_curves_path = arguments.artifacts_dir / "loss_curves.png"
        confusion_matrix_path = arguments.artifacts_dir / "confusion_matrix.png"
        plot_learning_curves(history, loss_curves_path)
        plot_confusion_matrix(
            true_indices,
            predicted_indices,
            class_names,
            confusion_matrix_path,
        )

        save_checkpoint(
            model,
            arguments.model_path,
            class_names=class_names,
            image_size=arguments.image_size,
            model_version=arguments.model_version,
            metrics={key: float(value) for key, value in test_metrics.items()},
        )
        model_digest = checkpoint_sha256(arguments.model_path)
        elapsed_seconds = time.perf_counter() - training_started
        result: dict[str, Any] = {
            **{key: float(value) for key, value in test_metrics.items()},
            "training_duration_seconds": elapsed_seconds,
            "model_version": arguments.model_version,
            "model_sha256": model_digest,
            "mlflow_run_id": active_run.info.run_id,
            "class_names": class_names,
            "image_size": arguments.image_size,
            "device": str(device),
            "git_commit": git_commit_or_unknown(),
            "history": history,
        }
        arguments.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        metadata_path = arguments.artifacts_dir / "model_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key != "history"
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        mlflow.log_metrics({key: float(value) for key, value in test_metrics.items()})
        mlflow.log_metric("training_duration_seconds", elapsed_seconds)
        mlflow.log_artifact(str(arguments.model_path), artifact_path="model")
        mlflow.log_artifact(str(arguments.metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(loss_curves_path), artifact_path="plots")
        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="plots")
        mlflow.log_artifact(str(metadata_path), artifact_path="model")

    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse training, output, and MLflow configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-path", type=Path, default=Path("models/resnet50_baseline.pt"))
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("metrics/training_metrics.json"),
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-version", default="1.0.0")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="cats-vs-dogs-baseline")
    arguments = parser.parse_args()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    return arguments


def main() -> None:
    """Run a complete tracked experiment from the command line."""

    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    train(parse_arguments())


if __name__ == "__main__":
    main()

