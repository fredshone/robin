import polars as pl
from polars.testing import assert_series_equal
from torch import Tensor

from robin.encoders.base import MinMaxEncoder


def test_encode_integer_column():
    data = pl.Series(name="integer", values=[10, 20, 30], dtype=pl.Int64)
    encoder = MinMaxEncoder(data=data)

    assert encoder.dtype == pl.Int64
    assert encoder.encoding == "continuous"
    assert encoder.size == 1

    encoded = encoder.encode(data=data)
    assert isinstance(encoded, Tensor)
    assert [encoded[i] for i in range(len(encoded))] == [-1.0, 0.0, 1.0]


def test_encode_float_column():
    data = pl.Series(name="float", values=[1.5, 2.5, 3.5], dtype=pl.Float64)
    encoder = MinMaxEncoder(data=data)
    assert encoder.dtype == pl.Float64
    assert encoder.encoding == "continuous"
    assert encoder.size == 1

    encoded = encoder.encode(data=data)
    assert isinstance(encoded, Tensor)
    assert [encoded[i] for i in range(len(encoded))] == [-1.0, 0.0, 1.0]


def test_encode_integer_column_with_precision():
    data = pl.Series(name="integer", values=[10, 20, 30], dtype=pl.Int64)
    encoder = MinMaxEncoder(data=data, learn_rounding=True)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    assert_series_equal(data, decoded, check_names=False)


def test_encode_float_column_with_precision():
    data = pl.Series(name="float", values=[1.5, 2.5, 3.5], dtype=pl.Float64)
    encoder = MinMaxEncoder(data=data, learn_rounding=True)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    assert [str(decoded[i]) for i in range(len(decoded))] == [
        "1.5",
        "2.5",
        "3.5",
    ]
