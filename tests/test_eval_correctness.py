import polars as pl
from polars.testing import assert_frame_equal

from robin.eval.correctness import incorrectness, structural_zeros


def test_structural_zeros_no_empty():
    target = pl.DataFrame({"age": [10, 30, 50], "license": [0, 1, 1]})
    synthetic = pl.DataFrame({"age": [10, 10, 50], "license": [0, 0, 1]})
    zeros = structural_zeros(target, synthetic)
    assert zeros["age_license"].is_empty()


def test_structural_zeros_license():
    target = pl.DataFrame({"age": [10, 30, 50], "license": [0, 1, 1]})
    synthetic = pl.DataFrame({"age": [10, 10, 50], "license": [0, 1, 1]})
    zeros = structural_zeros(target, synthetic)
    expected = pl.DataFrame({"zero": ["10_1"], "p": [1 / 3]})
    assert_frame_equal(zeros["age_license"], expected)


def test_structural_zeros_license_x2():
    target = pl.DataFrame({"age": [10, 30, 50], "license": [0, 1, 1]})
    synthetic = pl.DataFrame({"age": [10, 10, 50], "license": [2, 1, 1]})
    zeros = structural_zeros(target, synthetic)
    expected = pl.DataFrame({"zero": ["10_1", "10_2"], "p": [1 / 3, 1 / 3]})
    assert_frame_equal(zeros["age_license"].sort(by="zero"), expected)


def test_zero_incorrectness():
    target = pl.DataFrame({"age": [10, 30, 50], "license": [0, 1, 1]})
    synthetic = pl.DataFrame({"age": [10, 10, 50], "license": [0, 0, 1]})
    inc = incorrectness(target, synthetic)
    assert inc == 0


def test_nonzero_incorrectness_single_combination():
    target = pl.DataFrame({"age": [10, 30, 50], "license": [0, 1, 1]})
    synthetic = pl.DataFrame({"age": [10, 10, 50], "license": [0, 1, 1]})
    inc = incorrectness(target, synthetic)
    assert inc == 1 / 3


def test_nonzero_incorrectness_multi_combination():
    target = pl.DataFrame(
        {
            "age": [10, 10, 30, 30, 30],
            "license": [0, 0, 1, 1, 1],
            "income": [0, 1000, 1000, 0, 1000],
        }
    )
    synthetic = pl.DataFrame(
        {
            "age": [10, 30, 30, 30],
            "license": [0, 0, 1, 1],
            "income": [0, 1000, 0, 1000],
        }
    )
    print(structural_zeros(target, synthetic))
    inc = incorrectness(target, synthetic)
    assert inc == 1 / 12
