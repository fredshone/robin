import polars as pl
from polars.testing import assert_frame_equal

from robin.eval import binning


def test_cut_and_bin_continuous():
    data = pl.DataFrame(
        {"A": ["1", "2", "3", "4"], "B": [-1.5, -1.0, 1.0, 1.5]}
    )

    cutter, binned = binning.cut_and_bin_continuous(data, bins=2)

    assert cutter == {"B": [0]}

    expected_binned = pl.DataFrame(
        {"A": ["1", "2", "3", "4"], "B": [0, 0, 1, 1]},
        schema=pl.Schema({"A": pl.String, "B": pl.UInt16}),
    )

    assert_frame_equal(binned, expected_binned)


def test_apply_cutter():
    data = pl.DataFrame({"A": ["1", "2", "3", "4"], "B": [-1.0, 1.0, 2.0, 2.5]})

    cutter = {"B": [0]}

    binned = binning.apply_cutter(data, cutter)

    expected_binned = pl.DataFrame(
        {"A": ["1", "2", "3", "4"], "B": [0, 1, 1, 1]},
        schema=pl.Schema({"A": pl.String, "B": pl.UInt16}),
    )

    assert_frame_equal(binned, expected_binned)


def test_bin_continuous():
    target = pl.DataFrame(
        {"A": ["1", "2", "3", "4"], "B": [-1.5, -1.0, 1.0, 1.5]}
    )
    other = pl.DataFrame(
        {"A": ["5", "6", "7", "8"], "B": [-2.0, 1.0, 2.0, 3.0]}
    )

    target_binned, other_binned = binning.bin_continuous(target, other, bins=2)

    expected_target_binned = pl.DataFrame(
        {"A": ["1", "2", "3", "4"], "B": [0, 0, 1, 1]},
        schema=pl.Schema({"A": pl.String, "B": pl.UInt16}),
    )

    expected_other_binned = pl.DataFrame(
        {"A": ["5", "6", "7", "8"], "B": [0, 1, 1, 1]},
        schema=pl.Schema({"A": pl.String, "B": pl.UInt16}),
    )

    assert_frame_equal(target_binned, expected_target_binned)
    assert_frame_equal(other_binned, expected_other_binned)
