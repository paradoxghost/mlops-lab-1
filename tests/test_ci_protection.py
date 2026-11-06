"""Intentional failure confined to the Lab 5 branch-protection exercise PR."""

from src.food11.data import category_from_filename


def test_deliberate_category_failure():
    assert category_from_filename("0_12.jpg") == "Egg"
