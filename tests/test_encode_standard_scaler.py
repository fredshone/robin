import polars as pl
from polars.testing import assert_series_equal
from torch import Tensor

from robin.encoders.column_encoders.numeric import StandardScalerEncoder


def test_encode_integer_column():
    data = pl.Series(name="integer", values=[10, 20, 30], dtype=pl.Int64)
    encoder = StandardScalerEncoder()
    encoder.fit_and_encode(data=data)

    assert encoder.dtype == pl.Int64
    assert encoder.encoding == "continuous"
    assert encoder.size == 1

    encoded, weights = encoder.encode(data=data)
    assert encoded.shape[0] == weights.shape[0] == 3
    assert weights.shape[1] == 1
    assert isinstance(encoded, Tensor)
    assert isinstance(weights, Tensor)
    assert [weights[i].item() for i in range(len(weights))] == [1.0, 1.0, 1.0]


def test_encode_float_column():
    data = pl.Series(name="float", values=[1.5, 2.5, 3.5], dtype=pl.Float64)
    encoder = StandardScalerEncoder()
    encoder.fit_and_encode(data=data)

    assert encoder.dtype == pl.Float64
    assert encoder.encoding == "continuous"

    encoded, weights = encoder.encode(data=data)
    assert isinstance(encoded, Tensor)
    assert isinstance(weights, Tensor)


def test_encode_decode_round_trip():
    data = pl.Series(name="integer", values=[10, 20, 30], dtype=pl.Int64)
    encoder = StandardScalerEncoder(learn_rounding=True)
    encoded, _ = encoder.fit_and_encode(data)
    decoded = encoder.decode(encoded)
    assert_series_equal(data, decoded, check_names=False)


def test_encode_decode_float_round_trip():
    data = pl.Series(name="float", values=[1.5, 2.5, 3.5], dtype=pl.Float64)
    encoder = StandardScalerEncoder(learn_rounding=True)
    encoded, _ = encoder.fit_and_encode(data)
    decoded = encoder.decode(encoded)
    assert [str(decoded[i]) for i in range(len(decoded))] == ["1.5", "2.5", "3.5"]


def test_zero_variance_raises():
    import pytest

    data = pl.Series(name="constant", values=[5, 5, 5], dtype=pl.Int64)
    encoder = StandardScalerEncoder()
    with pytest.raises(ValueError, match="no variance"):
        encoder.fit_and_encode(data)
