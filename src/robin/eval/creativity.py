import polars as pl


def uniqueness(data: pl.DataFrame) -> float:
    """
    1,2,3,4 => 0.25
    1,1,1,1 => 1.0
    1,1,2,2 => 0.5
    1,2,2,2 => 0.5

    """
    uniques = data.unique()
    return len(uniques) / len(data)


def row_counts(data: pl.DataFrame) -> pl.DataFrame:
    """Compute the counts of each unique row in the DataFrame.

    Args:
        data (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: DataFrame with unique rows and their counts.
    """
    return data.group_by(data.columns).len(name="counts")


def simpsons_index(data):
    counts = row_counts(data)
    probs = counts["counts"] / len(data)
    return (probs**2).sum()


def simpsons_index_of_diversity(data):
    return 1 - simpsons_index(data)
