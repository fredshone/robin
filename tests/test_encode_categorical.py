import polars as pl
import torch
from torch import Tensor

from robin.encoders.base import CategoricalTokeniser
from robin.encoders.table_encoder import TableEncoder, XDataset
from robin.encoders.utils import inverse_token_frequency


def test_encode_categorical_column():
    original_table = pl.DataFrame(
        {"category": ["A", "B", "C", "A"]},
        schema=pl.Schema({"category": pl.String}),
    )
    encoder = TableEncoder(data=original_table)
    encoder.fit_and_encode(data=original_table)
    col_encoder = encoder.encoders["category"]
    assert isinstance(col_encoder, CategoricalTokeniser)
    assert col_encoder.dtype == pl.String
    assert col_encoder.mapping == {"A": 0, "B": 1, "C": 2}
    assert col_encoder.encoding == "categorical"
    assert col_encoder.size == 3

    encoded = encoder.encode(data=original_table)
    assert isinstance(encoded, XDataset)
    assert [encoded[i] for i in range(len(encoded))] == [0, 1, 2, 0]


def test_token_weights():
    tokens = Tensor([0, 1, 1]).long()
    weights = inverse_token_frequency(tokens)
    print(weights)
    assert torch.allclose(weights, Tensor([1.5, 0.75]))
