"""Validate, resize, split, and fingerprint Cats vs Dogs images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from cats_dogs_mlops.preprocessing import canonicalize_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ALIASES = {
    "cats": {"cat", "cats"},
    "dogs": {"dog", "dogs"},
}


def discover_class_images(raw_dir: Path, class_name: str) -> list[Path]:
    """Discover images whose nearest class directory identifies their label.

    The Kaggle dataset has changed nesting across mirrors (for example,
    ``training_set/training_set/cats``). Searching recursively by parent folder
    supports those layouts while retaining an explicit, auditable label rule.

    Args:
        raw_dir: Root of the downloaded dataset.
        class_name: Canonical class name: ``cats`` or ``dogs``.

    Returns:
        list[Path]: Sorted candidate image paths for that class.

    Raises:
        ValueError: If an unsupported class name is requested.
    """

    if class_name not in CLASS_ALIASES:
        raise ValueError(f"Unsupported class name: {class_name}")

    aliases = CLASS_ALIASES[class_name]
    candidates = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.parent.name.lower() in aliases
    ]
    return sorted(set(candidates), key=lambda path: str(path).lower())


def _deduplicate_by_content(paths: list[Path]) -> list[Path]:
    """Drop paths whose raw bytes duplicate an earlier path's content.

    The Kaggle Cats vs Dogs dataset ships the same photo under different
    filenames across its mirrored folder layouts. ``discover_class_images``
    only dedupes by path, so a duplicate can be shuffled into a different
    split than its twin — leaking test/validation answers into training.
    Hashing raw bytes here, before splitting, keeps every surviving image
    content-unique so no split can share an image with another.

    Args:
        paths: Candidate image paths, already deduped by path.

    Returns:
        list[Path]: ``paths`` with byte-identical duplicates removed,
        keeping the first occurrence of each content hash.
    """

    seen_hashes: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        digest = _sha256_file(path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique_paths.append(path)
    return unique_paths


def split_paths(
    paths: list[Path],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Create deterministic train, validation, and test partitions.

    Args:
        paths: Validated candidates for one class.
        train_ratio: Fraction assigned to training.
        validation_ratio: Fraction assigned to validation.
        test_ratio: Fraction assigned to testing.
        seed: Random seed controlling path shuffling.

    Returns:
        dict[str, list[Path]]: Non-overlapping paths keyed by split name.

    Raises:
        ValueError: If ratios do not sum to one or fewer than three images are
        available for a class.
    """

    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train, validation, and test ratios must sum to 1.0")
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("all split ratios must be positive")
    if len(paths) < 3:
        raise ValueError("at least three images per class are required")

    shuffled_paths = list(paths)
    random.Random(seed).shuffle(shuffled_paths)

    training_count = max(1, int(len(shuffled_paths) * train_ratio))
    validation_count = max(1, int(len(shuffled_paths) * validation_ratio))
    if training_count + validation_count >= len(shuffled_paths):
        training_count = len(shuffled_paths) - 2
        validation_count = 1

    return {
        "train": shuffled_paths[:training_count],
        "validation": shuffled_paths[
            training_count : training_count + validation_count
        ],
        "test": shuffled_paths[training_count + validation_count :],
    }


def _sha256_file(file_path: Path) -> str:
    """Return the SHA-256 digest of one processed image."""

    digest = hashlib.sha256()
    with file_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Create an empty, guarded processed-data destination."""

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
            )
        if output_dir.resolve().name.lower() != "processed":
            raise ValueError("Refusing to overwrite a directory not named 'processed'")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    image_size: int = 224,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    max_images_per_class: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build a clean RGB dataset and a content-fingerprint manifest.

    Corrupt files are skipped and counted. Each retained image is converted to
    RGB, centre-cropped without distortion, resized to ``224 x 224`` by
    default, and written as JPEG. Splitting occurs independently per class, so
    every partition remains balanced when equal class caps are used.

    Args:
        raw_dir: Downloaded dataset root.
        output_dir: Destination root for processed splits.
        image_size: Required square output size.
        train_ratio: Training fraction.
        validation_ratio: Validation fraction.
        test_ratio: Testing fraction.
        max_images_per_class: Optional deterministic cap for low-resource runs.
        seed: Reproducible shuffle seed.
        overwrite: Whether an existing ``processed`` directory may be reset.

    Returns:
        dict[str, object]: Counts, skipped files, and preprocessing parameters.

    Raises:
        FileNotFoundError: If the raw directory does not exist.
        ValueError: If no supported images are discovered or parameters are
        invalid.
    """

    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if max_images_per_class is not None and max_images_per_class < 3:
        raise ValueError("max_images_per_class must be at least 3")

    _prepare_output_directory(output_dir, overwrite)
    manifest_rows: list[dict[str, str | int]] = []
    skipped_files: list[str] = []
    duplicate_content_images_dropped = 0
    split_counts: Counter[str] = Counter()

    for class_index, class_name in enumerate(("cats", "dogs")):
        candidates = discover_class_images(raw_dir, class_name)
        if not candidates:
            raise ValueError(f"No '{class_name}' images discovered below {raw_dir}")

        deduplicated_candidates = _deduplicate_by_content(candidates)
        duplicate_content_images_dropped += len(candidates) - len(deduplicated_candidates)
        candidates = deduplicated_candidates

        # Shuffle before capping so the subset is representative and reproducible.
        random.Random(seed + class_index).shuffle(candidates)
        if max_images_per_class is not None:
            candidates = candidates[:max_images_per_class]

        valid_paths: list[Path] = []
        for source_path in candidates:
            try:
                with Image.open(source_path) as source_image:
                    source_image.verify()
                valid_paths.append(source_path)
            except (UnidentifiedImageError, OSError, ValueError):
                skipped_files.append(str(source_path.relative_to(raw_dir)))

        partitions = split_paths(
            valid_paths,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed + class_index,
        )

        for split_name, split_paths_for_class in partitions.items():
            destination_dir = output_dir / split_name / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            for image_index, source_path in enumerate(split_paths_for_class):
                destination_path = destination_dir / f"{class_name}_{image_index:05d}.jpg"
                try:
                    with Image.open(source_path) as source_image:
                        processed_image = canonicalize_image(source_image, image_size)
                        processed_image.save(
                            destination_path,
                            format="JPEG",
                            quality=90,
                            optimize=True,
                        )
                except (UnidentifiedImageError, OSError, ValueError):
                    skipped_files.append(str(source_path.relative_to(raw_dir)))
                    continue

                relative_destination = destination_path.relative_to(output_dir)
                manifest_rows.append(
                    {
                        "split": split_name,
                        "label": class_name,
                        "source_relative_path": str(source_path.relative_to(raw_dir)),
                        "processed_relative_path": str(relative_destination),
                        "sha256": _sha256_file(destination_path),
                        "width": image_size,
                        "height": image_size,
                    }
                )
                split_counts[f"{split_name}/{class_name}"] += 1

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        fieldnames = list(manifest_rows[0])
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary: dict[str, object] = {
        "image_size": image_size,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "max_images_per_class": max_images_per_class,
        "processed_images": len(manifest_rows),
        "skipped_images": len(skipped_files),
        "duplicate_content_images_dropped": duplicate_content_images_dropped,
        "split_counts": dict(sorted(split_counts.items())),
        "skipped_relative_paths": skipped_files,
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line parameters for preprocessing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--max-images-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute preprocessing and print a concise audit summary."""

    arguments = parse_arguments()
    summary = prepare_dataset(
        arguments.raw_dir,
        arguments.output_dir,
        image_size=arguments.image_size,
        train_ratio=arguments.train_ratio,
        validation_ratio=arguments.validation_ratio,
        test_ratio=arguments.test_ratio,
        max_images_per_class=arguments.max_images_per_class,
        seed=arguments.seed,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

