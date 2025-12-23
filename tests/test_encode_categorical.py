import polars as pl

from robin.encoders.base import CategoricalTokeniser
from robin.encoders.table_encoder import TableDataset, TableEncoder


def test_encode_categorical_column():
    original_table = pl.DataFrame(
        {"category": ["A", "B", "C", "A"]},
        schema=pl.Schema({"category": pl.String}),
    )
    encoder = TableEncoder(data=original_table)
    col_encoder = encoder.encoders["category"]
    assert isinstance(col_encoder, CategoricalTokeniser)
    assert col_encoder.dtype == pl.String
    assert col_encoder.mapping == {"A": 0, "B": 1, "C": 2}
    assert col_encoder.encoding == "categorical"
    assert col_encoder.size == 3

    encoded = encoder.encode(data=original_table)
    assert isinstance(encoded, TableDataset)
    assert [encoded[i] for i in range(len(encoded))] == [0, 1, 2, 0]
