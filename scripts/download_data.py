"""Download the configured public Kaggle Cats vs Dogs dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import kagglehub


def download_dataset(
    dataset_handle: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download a public Kaggle dataset and copy it into the DVC workspace.

    KaggleHub maintains a content-aware cache outside the repository. Copying
    the selected version into ``data/raw`` gives DVC a stable directory to hash
    and lets preprocessing run without depending on the cache layout.

    Args:
        dataset_handle: Kaggle owner-and-dataset slug, for example
            ``tongpython/cat-and-dog``.
        output_dir: Repository directory recorded as the DVC stage output.
        overwrite: Whether an existing ``raw`` output may be replaced.

    Returns:
        Path: Populated output directory.

    Raises:
        FileExistsError: If output exists and overwrite is not enabled.
        ValueError: If an unsafe overwrite target or malformed handle is used.
    """

    if dataset_handle.count("/") != 1:
        raise ValueError("dataset_handle must use the form 'owner/dataset'")

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
            )
        if output_dir.name.lower() != "raw":
            raise ValueError("Refusing to overwrite a directory not named 'raw'")
        shutil.rmtree(output_dir)

    cached_dataset_path = Path(kagglehub.dataset_download(dataset_handle)).resolve()
    if not cached_dataset_path.is_dir():
        raise FileNotFoundError(f"KaggleHub returned no dataset directory: {cached_dataset_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cached_dataset_path, output_dir, dirs_exist_ok=False)

    provenance = {
        "dataset_handle": dataset_handle,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "kagglehub_cache_path": str(cached_dataset_path),
    }
    (output_dir / "dataset_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(f"Dataset '{dataset_handle}' copied to {output_dir}")
    return output_dir


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the DVC download stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-handle", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the dataset download stage."""

    arguments = parse_arguments()
    download_dataset(
        arguments.dataset_handle,
        arguments.output_dir,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()

