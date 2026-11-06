import shutil
import subprocess
import sys

import pytest
from PIL import Image

from src.food11.data import CLASSES, SPLITS
from src.food11.validate_data import validate_dataset


@pytest.fixture
def dataset(tmp_path):
    for split_id, split in enumerate(SPLITS):
        for index, category in enumerate(CLASSES):
            folder = tmp_path / split / category
            folder.mkdir(parents=True)
            Image.new("RGB", (128, 128), (split_id * 60, index * 20, 42)).save(folder / "image.png")
    return tmp_path


def outcomes(root, **kwargs):
    return {check.name: check.passed for check in validate_dataset(root, min_per_class=1, **kwargs)}


def test_valid_contract(dataset):
    assert all(outcomes(dataset).values())


def test_cross_split_duplicate_fails(dataset):
    shutil.copyfile(dataset / "training/Bread/image.png", dataset / "evaluation/Bread/duplicate.png")
    assert not outcomes(dataset)["cross-split duplicates"]


def test_duplicate_within_same_split_is_allowed(dataset):
    shutil.copyfile(dataset / "training/Bread/image.png", dataset / "training/Bread/duplicate.png")
    assert outcomes(dataset, tau=1)["cross-split duplicates"]


@pytest.mark.parametrize("mode,size", [("L", (128, 128)), ("RGB", (64, 128))])
def test_bad_image_contract(dataset, mode, size):
    Image.new(mode, size).save(dataset / "validation/Bread/image.png")
    assert not outcomes(dataset)["RGB 128x128"]


def test_corrupt_image(dataset):
    (dataset / "validation/Bread/image.png").write_bytes(b"not an image")
    assert not outcomes(dataset)["image decoding"]


def test_missing_and_extra_class(dataset):
    (dataset / "validation/Bread").rename(dataset / "validation/Unexpected")
    checks = outcomes(dataset)
    assert not checks["class directories"] and not checks["minimum count"]


def test_minimum_count(dataset):
    checks = validate_dataset(dataset, min_per_class=2)
    assert not next(c.passed for c in checks if c.name == "minimum count")


def test_distribution_shift(dataset):
    shutil.copyfile(dataset / "training/Bread/image.png", dataset / "training/Bread/extra.png")
    assert not outcomes(dataset, tau=0.05)["class distribution"]


def test_missing_dataset(tmp_path):
    assert not all(outcomes(tmp_path / "missing").values())


@pytest.mark.parametrize("minimum,tau", [(0, 0.05), (1, -0.1), (1, float("nan"))])
def test_invalid_options(dataset, minimum, tau):
    with pytest.raises(ValueError):
        validate_dataset(dataset, minimum, tau)


def test_cli_exit_codes(dataset):
    command = [sys.executable, "src/food11/validate_data.py", str(dataset), "--min-per-class", "1"]
    assert subprocess.run(command, capture_output=True).returncode == 0
    shutil.copyfile(dataset / "training/Bread/image.png", dataset / "evaluation/Bread/duplicate.png")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and "FAIL cross-split duplicates" in result.stdout
