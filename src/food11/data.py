"""Prepare full and mini Food-11 datasets for image classification."""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}
SPLITS = ("training", "evaluation", "validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPOSITORY_ROOT / "data" / "food11_raw"
PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "food11_processed"
MINI_ROOT = REPOSITORY_ROOT / "data" / "food11_processed_mini"


def class_id_from_filename(path: Path) -> int:
    """Return and validate the integer before the first filename underscore."""
    prefix, separator, _ = path.stem.partition("_")
    if not separator or not prefix.isdigit():
        raise ValueError(
            f"Invalid Food-11 filename {path.name!r}: expected an integer class "
            "prefix followed by an underscore."
        )

    class_id = int(prefix)
    if class_id not in CLASS_NAMES:
        raise ValueError(
            f"Invalid class prefix {class_id} in {path.name!r}: expected 0 through 10."
        )
    return class_id


def collect_raw_images() -> dict[str, dict[int, list[Path]]]:
    """Validate the raw layout and return deterministically ordered image paths."""
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f"Required raw dataset directory is missing: {RAW_ROOT}")

    images: dict[str, dict[int, list[Path]]] = {}
    for split in SPLITS:
        split_root = RAW_ROOT / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Required raw split directory is missing: {split_root}")

        by_class: dict[int, list[Path]] = defaultdict(list)
        entries = sorted(split_root.iterdir(), key=lambda path: path.name)
        for path in entries:
            if not path.is_file():
                raise ValueError(f"Unexpected non-file entry in raw split: {path}")
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension for raw file: {path}")
            by_class[class_id_from_filename(path)].append(path)

        if not entries:
            raise ValueError(f"Raw split contains no images: {split_root}")
        images[split] = {class_id: by_class[class_id] for class_id in CLASS_NAMES}

    return images


def create_category_directories(root: Path) -> None:
    """Create every required split/category directory."""
    for split in SPLITS:
        for class_name in CLASS_NAMES.values():
            (root / split / class_name).mkdir(parents=True, exist_ok=True)


def resize_image(source: Path, destination: Path) -> None:
    """Read one image, convert it to RGB, resize it, and save it."""
    try:
        with Image.open(source) as image:
            resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            try:
                resized.save(destination)
            finally:
                resized.close()
    except (OSError, UnidentifiedImageError) as error:
        raise RuntimeError(f"Could not process image {source}: {error}") from error


def prepare_datasets(images: dict[str, dict[int, list[Path]]]) -> dict[str, tuple[int, int]]:
    """Rebuild the processed datasets and return full/mini counts by split."""
    for output_root in (PROCESSED_ROOT, MINI_ROOT):
        if output_root.exists():
            shutil.rmtree(output_root)

    try:
        create_category_directories(PROCESSED_ROOT)
        create_category_directories(MINI_ROOT)

        counts: dict[str, tuple[int, int]] = {}
        for split in SPLITS:
            full_count = 0
            mini_count = 0
            for class_id, class_name in CLASS_NAMES.items():
                for index, source in enumerate(images[split][class_id]):
                    processed_path = PROCESSED_ROOT / split / class_name / source.name
                    resize_image(source, processed_path)
                    full_count += 1

                    if index < MINI_LIMIT:
                        mini_path = MINI_ROOT / split / class_name / source.name
                        shutil.copyfile(processed_path, mini_path)
                        mini_count += 1
            counts[split] = (full_count, mini_count)
        return counts
    except Exception:
        for output_root in (PROCESSED_ROOT, MINI_ROOT):
            if output_root.exists():
                shutil.rmtree(output_root)
        raise


def print_summary(images: dict[str, dict[int, list[Path]]]) -> None:
    """Print deterministic full and mini counts for every split/category."""
    print(f"{'Split':<12} {'Category':<20} {'Processed':>9} {'Mini':>6}")
    print("-" * 51)
    for split in SPLITS:
        for class_id, class_name in CLASS_NAMES.items():
            full_count = len(images[split][class_id])
            mini_count = min(full_count, MINI_LIMIT)
            print(f"{split:<12} {class_name:<20} {full_count:>9} {mini_count:>6}")
    print("-" * 51)
    for split in SPLITS:
        full_count = sum(len(paths) for paths in images[split].values())
        mini_count = sum(min(len(paths), MINI_LIMIT) for paths in images[split].values())
        print(f"{split:<12} {'TOTAL':<20} {full_count:>9} {mini_count:>6}")


def main() -> None:
    """Validate raw data and recreate both processed Food-11 datasets."""
    images = collect_raw_images()
    prepare_datasets(images)
    print_summary(images)


if __name__ == "__main__":
    main()
