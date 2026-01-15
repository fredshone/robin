import polars as pl
from polars.testing import assert_series_equal

from robin.encoders.decompose import GMMEncoder


def test_gmm_encoder_encode_decode_consistently():
    x = pl.Series("", [0.1, 0.2, 0.3, 0.4, 0.5])
    encoder = GMMEncoder(x, name="", verbose=True, max_components=2)
    h = encoder.fit_and_encode(x)
    x_hat = encoder.decode(h)
    assert_series_equal(x, x_hat)


def test_gmm_encoder_single_component():
    x = pl.Series("", [0.1] * 10)
    encoder = GMMEncoder(x, name="", verbose=True, max_components=2)
    _ = encoder.fit_and_encode(x)
    assert encoder.size == 2


def test_gmm_encoder_two_component():
    x = pl.Series("", [0.1] * 10 + [0.9] * 10)
    encoder = GMMEncoder(x, name="", verbose=True, max_components=2)
    _ = encoder.fit_and_encode(x)
    assert encoder.size == 2
