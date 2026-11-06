"""Validate the processed Food-11 data contract without modifying any files."""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

if __package__:
    from .data import CLASSES, SPLITS
else:  # Also support the handout's direct script invocation.
    from data import CLASSES, SPLITS


@dataclass
class Check:
    name: str
    passed: bool
    details: list[str]


def validate_dataset(root: Path, min_per_class: int = 50, tau: float = 0.05) -> list[Check]:
    """Return independent contract checks; decode pixels and hash complete file bytes."""
    if min_per_class < 1 or not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("min-per-class must be positive and tau must be finite in [0, 1]")
    errors = {name: [] for name in ["class directories", "image decoding", "RGB 128x128",
                                   "minimum count", "cross-split duplicates", "class distribution"]}
    counts = {split: dict.fromkeys(CLASSES, 0) for split in SPLITS}
    seen: dict[str, tuple[str, Path]] = {}
    for split in SPLITS:
        folder = Path(root) / split
        if not folder.is_dir():
            errors["class directories"].append(f"Missing split: {split}")
            continue
        entries = list(folder.iterdir())
        actual = {p.name for p in entries if p.is_dir()}
        if actual != set(CLASSES) or any(not p.is_dir() for p in entries):
            errors["class directories"].append(
                f"{split}: missing={sorted(set(CLASSES) - actual)}, extra={sorted(actual - set(CLASSES))}; "
                "split must contain only the 11 class directories"
            )
        for category in CLASSES:
            category_dir = folder / category
            if not category_dir.is_dir():
                continue
            for path in sorted(category_dir.iterdir()):
                if not path.is_file():
                    errors["class directories"].append(f"Unexpected nested directory: {path}")
                    continue
                counts[split][category] += 1
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest in seen and seen[digest][0] != split:
                        errors["cross-split duplicates"].append(f"{path} duplicates {seen[digest][1]}")
                    else:
                        seen.setdefault(digest, (split, path))
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != (128, 128):
                            errors["RGB 128x128"].append(f"{path}: {image.mode} {image.size}")
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    errors["image decoding"].append(f"{path}: {type(error).__name__}")
    for split, categories in counts.items():
        for category, count in categories.items():
            if count < min_per_class:
                errors["minimum count"].append(f"{split}/{category}: {count} < {min_per_class}")
    totals = {split: sum(categories.values()) for split, categories in counts.items()}
    if not totals["training"] or not totals["validation"]:
        errors["class distribution"].append("Training and validation must be nonempty")
    else:
        for category in CLASSES:
            delta = abs(counts["training"][category] / totals["training"]
                        - counts["validation"][category] / totals["validation"])
            if delta > tau:
                errors["class distribution"].append(f"{category}: share difference {delta:.6f} > {tau}")
    return [Check(name, not details, details) for name, details in errors.items()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    try:
        checks = validate_dataset(args.directory, args.min_per_class, args.tau)
    except (ValueError, OSError) as error:
        print(f"FAIL configuration/access: {error}")
        return 1
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}")
        for detail in check.details[:5]:
            print(f"  {detail}")
        if len(check.details) > 5:
            print(f"  ... {len(check.details) - 5} more")
    return int(any(not check.passed for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
