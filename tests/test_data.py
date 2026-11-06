import pytest
from PIL import Image

from src.food11.data import CLASSES, category_from_filename, resize_image


@pytest.mark.parametrize("index,category", list(enumerate(CLASSES)))
def test_category_from_filename(index, category):
    assert category_from_filename(f"{index}_12.jpg") == category
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"


@pytest.mark.parametrize("filename", ["11_1.jpg", "-1_2.jpg", "bread.jpg", "x_1.jpg"])
def test_invalid_filename(filename):
    with pytest.raises(ValueError):
        category_from_filename(filename)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [128, 64])
def test_resize_image_shape_and_mode(mode, size):
    with Image.new(mode, (512, 384)) as source:
        result = resize_image(source, size=size)
        assert result.size == (size, size)
        assert result.mode == "RGB"
        assert source.size == (512, 384) and source.mode == mode


def test_eleven_classes():
    assert len(CLASSES) == len(set(CLASSES)) == 11


def test_invalid_size():
    with pytest.raises(ValueError):
        resize_image(Image.new("RGB", (1, 1)), 0)
