import polars as pl
from polars.testing import assert_frame_equal

from robin.encoders.base import CategoricalTokeniser, ContinuousEncoder
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


def test_encode_integer_column():
    original_table = pl.DataFrame(
        {"integer": [10, 20, 30]}, schema=pl.Schema({"integer": pl.Int64})
    )
    encoder = TableEncoder(data=original_table)
    col_encoder = encoder.encoders["integer"]
    assert isinstance(col_encoder, ContinuousEncoder)
    assert col_encoder.dtype == pl.Int64
    assert col_encoder.encoding == "continuous"
    assert col_encoder.size == 1

    encoded = encoder.encode(data=original_table)
    assert isinstance(encoded, TableDataset)
    assert [encoded[i] for i in range(len(encoded))] == [0, 0.5, 1.0]


def test_encode_float_column():
    original_table = pl.DataFrame(
        {"float": [1.5, 2.5, 3.5]}, schema=pl.Schema({"float": pl.Float64})
    )
    encoder = TableEncoder(data=original_table)
    col_encoder = encoder.encoders["float"]
    assert isinstance(col_encoder, ContinuousEncoder)
    assert col_encoder.dtype == pl.Float64
    assert col_encoder.encoding == "continuous"
    assert col_encoder.size == 1

    encoded = encoder.encode(data=original_table)
    assert isinstance(encoded, TableDataset)
    assert [encoded[i] for i in range(len(encoded))] == [0.0, 0.5, 1.0]


def test_encode_decode_table():
    original_table = pl.DataFrame(
        {
            "category": ["A", "B", "C", "A"],
            "integer": [10, 20, 30, 40],
            "float": [1.5, 2.5, 3.5, 4.5],
        },
        schema=pl.Schema(
            {"category": pl.Utf8, "integer": pl.Int64, "float": pl.Float64}
        ),
    )
    encoder = TableEncoder(data=original_table)
    encoded = encoder.encode(data=original_table)
    decoded = encoder.decode(encoded)

    assert_frame_equal(decoded, original_table)
