import polars as pl
from polars.testing import assert_series_equal

from robin.encoders.column_encoders.decompose import GMMEncoder


def test_gmm_encoder_encode_decode_consistently():
    x = pl.Series("", [0.1, 0.2, 0.3, 0.4, 0.5])
    encoder = GMMEncoder(name="test", verbose=True, max_components=2)
    h, weights = encoder.fit_and_encode(x)
    assert h.shape[0] == weights.shape[0] == 5
    assert weights.shape[1] == 1
    x_hat = encoder.decode(h)
    assert_series_equal(x, x_hat)


def test_gmm_encoder_single_component():
    x = pl.Series("", [0.1] * 10)
    encoder = GMMEncoder(name="test", verbose=True, max_components=2)
    _, _ = encoder.fit_and_encode(x)
    assert encoder.size == 2


def test_gmm_encoder_two_component():
    x = pl.Series("", [0.1] * 10 + [0.9] * 10)
    encoder = GMMEncoder(name="test", verbose=True, max_components=2)
    _, _ = encoder.fit_and_encode(x)
    assert encoder.size == 2
