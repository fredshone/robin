import polars as pl
import pytest

from robin.eval import creativity


@pytest.mark.parametrize(
    "data, expected_uniqueness",
    [
        (pl.DataFrame({"a": [1, 2, 3, 4]}), 1.0),
        (pl.DataFrame({"a": [1, 1, 1, 1]}), 0.25),
        (pl.DataFrame({"a": [1, 2, 2, 3]}), 0.75),
    ],
)
def test_uniqueness_for_single_column(data, expected_uniqueness):
    assert creativity.uniqueness(data) == expected_uniqueness


@pytest.mark.parametrize(
    "data, expected_counts",
    [
        (pl.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1]}), [3]),
        (pl.DataFrame({"a": [1, 2, 1], "b": [1, 2, 1]}), [1, 2]),
        (pl.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}), [1, 1, 1]),
    ],
)
def test_row_counts(data, expected_counts):
    assert (
        creativity.row_counts(data)["counts"].sort().to_list()
        == expected_counts
    )


@pytest.mark.parametrize(
    "data, expected_index",
    [
        (pl.DataFrame({"a": [1, 1, 1, 1], "b": [1, 1, 1, 1]}), 1),
        (
            pl.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 1, 1]}),
            0.75**2 + 0.25**2,
        ),
    ],
)
def test_simpsons_index(data, expected_index):
    assert creativity.simpsons_index(data) == expected_index
