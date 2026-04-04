import polars as pl

from robin.eval.density import iter_joint_probs


def structural_zeros(
    target: pl.DataFrame, synthetic: pl.DataFrame, ignore_numerical: bool = False
) -> dict[str, pl.DataFrame]:
    """
    Structural zeros are identified as second order marginals with zero probability in the target.
    For example a 5 year old with a driving license should be a structural zero.

    Args:
        target (pl.DataFrame): target distribution
        synthetic (pl.DataFrame): synthetic distribution

    Returns:
        dict[str, pl.DataFrame]: dict keyed by "col1_col2" with DataFrames of zeros and probs
    """
    if ignore_numerical:
        target = _filter_categorical(target)
        synthetic = _filter_categorical(synthetic)

    return _raw_structural_zeros(target, synthetic)


def _filter_categorical(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        [
            col
            for col in df.columns
            if df[col].dtype in [pl.String, pl.Categorical]
        ]
    )


def _raw_structural_zeros(
    target: pl.DataFrame, synthetic: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    all_zeros = {}

    for name, index, target_probs, synthetic_probs in iter_joint_probs(
        target, synthetic, order=2
    ):
        key = name.replace(" & ", "_")
        mask = target_probs == 0
        zeros = index.filter(mask).str.replace_all(" & ", "_")
        probs = synthetic_probs.filter(mask)
        all_zeros[key] = pl.DataFrame().with_columns(
            [zeros.alias("zero"), probs.alias("p")]
        )
    return all_zeros


def incorrectness(
    target: pl.DataFrame, synthetic: pl.DataFrame, ignore_numerical: bool = False
) -> float:
    """
    Calculate the incorrectness metric between target and synthetic DataFrames.
    Calculates the total probability mass assigned to structural zeros in the synthetic data.
    Normalises by number of second order combinations.

    Args:
        target (pl.DataFrame): target distribution
        synthetic (pl.DataFrame): synthetic distribution

    Returns:
        float: correctness score between 0 and 1
    """
    if ignore_numerical:
        target = _filter_categorical(target)
        synthetic = _filter_categorical(synthetic)
    all_zeros = _raw_structural_zeros(target, synthetic)
    if not all_zeros:
        # No 2nd-order combinations to evaluate (e.g. fewer than 2 categorical
        # columns remain after filtering). Return 0.0 rather than dividing by zero.
        return 0.0
    p_incorrectness = 0.0
    for cols, zeros in all_zeros.items():
        p_incorrectness += zeros["p"].sum()
    return p_incorrectness / len(all_zeros)
