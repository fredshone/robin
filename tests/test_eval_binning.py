import polars as pl
from polars.testing import assert_frame_equal

from robin.eval.binning import apply_cutter, bin_continuous, cut_and_bin_continuous


def _wide_float_df():
    return pl.DataFrame({"x": [float(i) for i in range(100)]})


def _narrow_float_df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 20})


# --- cut_and_bin_continuous ---


def test_cut_and_bin_cutter_populated_for_wide_column():
    cutter, _ = cut_and_bin_continuous(_wide_float_df(), bins=10)
    assert "x" in cutter
    assert len(cutter["x"]) > 0


def test_cut_and_bin_output_dtype_is_uint16():
    _, binned = cut_and_bin_continuous(_wide_float_df(), bins=10)
    assert binned["x"].dtype == pl.UInt16


def test_cut_and_bin_narrow_column_not_binned():
    cutter, binned = cut_and_bin_continuous(_narrow_float_df(), bins=10)
    assert "x" not in cutter
    assert binned["x"].dtype == pl.Float64


def test_cut_and_bin_no_float_columns_returns_empty_cutter():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    cutter, binned = cut_and_bin_continuous(df, bins=10)
    assert cutter == {}
    assert binned.schema == df.schema


def test_cut_and_bin_non_float_columns_unchanged():
    df = pl.DataFrame({"x": [float(i) for i in range(100)], "cat": ["a", "b"] * 50})
    _, binned = cut_and_bin_continuous(df, bins=10)
    assert binned["cat"].to_list() == df["cat"].to_list()


# --- apply_cutter ---


def test_apply_cutter_empty_cutter_returns_unchanged():
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = apply_cutter(df, {})
    assert_frame_equal(result, df)


def test_apply_cutter_output_dtype_is_uint16():
    cutter, _ = cut_and_bin_continuous(_wide_float_df(), bins=10)
    result = apply_cutter(_wide_float_df(), cutter)
    assert result["x"].dtype == pl.UInt16


def test_apply_cutter_values_within_bin_range():
    cutter, _ = cut_and_bin_continuous(_wide_float_df(), bins=10)
    result = apply_cutter(_wide_float_df(), cutter)
    assert result["x"].min() >= 0
    assert result["x"].max() <= 9


# --- bin_continuous ---


def test_bin_continuous_no_float_columns_does_not_crash():
    df = pl.DataFrame({"a": ["x", "y"] * 5, "b": [1, 2] * 5})
    target_binned, synth_binned = bin_continuous(df, df, bins=10)
    assert target_binned.schema == df.schema
    assert synth_binned.schema == df.schema


def test_bin_continuous_both_outputs_are_uint16():
    target = pl.DataFrame({"x": [float(i) for i in range(100)]})
    synthetic = pl.DataFrame({"x": [float(i) * 0.9 for i in range(100)]})
    target_binned, synth_binned = bin_continuous(target, synthetic, bins=10)
    assert target_binned["x"].dtype == pl.UInt16
    assert synth_binned["x"].dtype == pl.UInt16


def test_bin_continuous_identical_inputs_produce_equal_outputs():
    df = _wide_float_df()
    target_binned, synth_binned = bin_continuous(df, df, bins=10)
    assert_frame_equal(target_binned, synth_binned)
