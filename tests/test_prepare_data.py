"""Unit tests for raw-dataset discovery, dedup, and split partitioning."""

from pathlib import Path

import pytest

from prepare_data import _deduplicate_by_content, split_paths


def test_deduplicate_by_content_drops_byte_identical_duplicates(tmp_path: Path) -> None:
    """Kaggle mirrors ship the same photo under different filenames; splitting on
    raw paths without deduping lets one copy land in train and its duplicate in
    test, leaking test-set answers into training. Dedup must catch that before
    any split happens."""
    original = tmp_path / "cat.1.jpg"
    original.write_bytes(b"identical-bytes")
    duplicate = tmp_path / "cat.1 (copy).jpg"
    duplicate.write_bytes(b"identical-bytes")
    distinct = tmp_path / "cat.2.jpg"
    distinct.write_bytes(b"different-bytes")

    unique = _deduplicate_by_content([original, duplicate, distinct])

    assert unique == [original, distinct]


def test_split_paths_never_splits_deduplicated_input(tmp_path: Path) -> None:
    """End-to-end guard: once dedup removes content duplicates, no sha256 can
    appear in more than one split, because each surviving path is unique content."""
    paths = [tmp_path / f"cat.{i}.jpg" for i in range(6)]
    for path in paths:
        path.write_bytes(b"unique-per-file" + path.name.encode())

    partitions = split_paths(
        paths, train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25, seed=42
    )

    assigned = [p for group in partitions.values() for p in group]
    assert sorted(assigned) == sorted(paths)
    assert len(assigned) == len(set(assigned))
