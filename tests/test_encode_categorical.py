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
    assert [weights[i] for i in range(len(weights))] == [0.5, 1, 1, 0.5]


def test_token_weights():
    tokens = Tensor([0, 1, 1]).long()
    weights = inverse_token_frequency(tokens)
    assert torch.allclose(weights, Tensor([1, 0.5]))
