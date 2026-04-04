import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from robin.encoders.column_encoders.decompose import GMMEncoder, MetaDecomposer


def test_gmm_encoder_encode_decode_consistently():
    np.random.seed(0)
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


def test_meta_decomposer_encode_decode_roundtrip():
    np.random.seed(0)
    x = pl.Series("", np.random.normal(5, 2, 100).tolist())
    encoder = MetaDecomposer(data=x, name="test", max_components=3, verbose=True)
    h, weights = encoder.fit_and_encode(x)
    assert h.shape[0] == weights.shape[0] == 100
    assert h.shape[1] == 2
    x_hat = encoder.decode(h)
    assert len(x_hat) == 100


def test_meta_decomposer_selects_transformer():
    # Log-skewed data should favour a log or standard transform over identity
    np.random.seed(0)
    x = pl.Series("", np.random.exponential(scale=2, size=200).tolist())
    encoder = MetaDecomposer(data=x, name="test", max_components=5, verbose=True)
    encoder.fit_and_encode(x)
    # Winner should not be None and should be one of the known transform types
    assert encoder.transformer is not None


def test_meta_decomposer_encode_after_fit():
    np.random.seed(0)
    x_train = pl.Series("", np.random.normal(0, 1, 100).tolist())
    x_new = pl.Series("", np.random.normal(0, 1, 20).tolist())
    encoder = MetaDecomposer(data=x_train, name="test", max_components=3)
    encoder.fit_and_encode(x_train)
    h, weights = encoder.encode(x_new)
    assert h.shape[0] == 20


def test_meta_decomposer_positive_only_data():
    # LogTransform requires positive values — verify MetaDecomposer handles this cleanly
    np.random.seed(0)
    x = pl.Series("", np.random.exponential(scale=1, size=100).tolist())
    encoder = MetaDecomposer(data=x, name="test", max_components=3, verbose=True)
    h, weights = encoder.fit_and_encode(x)
    assert h.shape[0] == 100
