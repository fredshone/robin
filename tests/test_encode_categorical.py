import polars as pl
import torch
from torch import Tensor

from robin.encoders.column_encoders.categorical import CategoricalTokeniser
from robin.encoders.utils import inverse_token_frequency


def test_encode_categorical_column():
    data = pl.Series(name="cats", values=["A", "B", "C", "A"], dtype=pl.String)
    encoder = CategoricalTokeniser()
    encoder.fit_and_encode(data=data)

    assert encoder.dtype == pl.String
    assert encoder.mapping == {"A": 0, "B": 1, "C": 2}
    assert encoder.encoding == "categorical"
    assert encoder.size == 3

    encoded, weights = encoder.encode(data=data)
    assert encoded.shape[0] == weights.shape[0] == 4
    assert weights.shape[1] == 1
    assert isinstance(encoded, Tensor)
    assert isinstance(weights, Tensor)
    assert [encoded[i] for i in range(len(encoded))] == [0, 1, 2, 0]
    # A appears 2x, B and C appear 1x each.
    # inv_freq = [0.5, 1.0, 1.0], mean = 5/6
    # normalised: A=0.6, B=1.2, C=1.2
    expected = Tensor([0.6, 1.2, 1.2, 0.6]).unsqueeze(-1)
    assert torch.allclose(weights, expected, atol=1e-5)


def test_token_weights():
    tokens = Tensor([0, 1, 1]).long()
    weights = inverse_token_frequency(tokens)
    # inv_freq = [1, 0.5], mean = 0.75; normalised = [4/3, 2/3]
    assert torch.allclose(weights, Tensor([4 / 3, 2 / 3]), atol=1e-5)
