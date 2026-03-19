import polars as pl

from robin.eval.density import mean_mean_absolute_error


def test_mmae_is_zero_for_equal_inputs():
    data = pl.DataFrame(
        {
            "A": ["1", "2", "1", "2"],
            "B": [3, 4, 3, 4],
            "C": [5.0, 6.0, 5.5, 6.0],
        }
    )
    assert mean_mean_absolute_error(data, data, order=1) == 0


def test_mmae_is_zero_for_equal_inputs_all_orders():
    data = pl.DataFrame(
        {
            "A": ["1", "2", "1", "2"],
            "B": [3, 4, 3, 4],
            "C": [5.0, 6.0, 5.5, 6.0],
        }
    )
    for i in range(1, 4):
        assert mean_mean_absolute_error(data, data, order=i) == 0


def test_mmae_order_one():
    target = pl.DataFrame({"A": ["1", "2", "1", "2"]})
    synthetic = pl.DataFrame({"A": ["1", "1", "1", "2"]})
    assert mean_mean_absolute_error(target, synthetic, order=1) == 1 / 4


def test_mmae_order_one_with_mismatch():
    target = pl.DataFrame({"A": ["1", "2"]})
    synthetic = pl.DataFrame({"A": ["1", "3"]})
    assert mean_mean_absolute_error(target, synthetic, order=1) == 1 / 3


def test_mmae_order_two():
    target = pl.DataFrame(
        {"A": ["1", "2", "1", "2"], "B": ["X", "X", "Y", "Y"]}
    )
    synthetic = pl.DataFrame(
        {"A": ["1", "1", "1", "2"], "B": ["X", "X", "Y", "Y"]}
    )
    assert mean_mean_absolute_error(target, synthetic, order=2) == 1 / 8
