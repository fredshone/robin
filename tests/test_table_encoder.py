import polars as pl
from polars.testing import assert_frame_equal

from robin.encoders.table_encoder import TableEncoder


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
