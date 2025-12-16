import polars as pl

from robin.eval.density import iter_joint_probs


def structural_zeros(target: pl.DataFrame, synthetic: pl.DataFrame) -> float:
    """
    Structural zeros are identified as second order marginals with zero probability in the target.
    For example a 5 year old with a driving license should be a structural zero.

    Args:
        target (pl.DataFrame): target distribution
        synthetic (pl.DataFrame): synthetic distribution

    Returns:
        float: correctness score between 0 and 1
    """
    all_zeros = {}
    for name, index, target_probs, synthetic_probs in iter_joint_probs(
        target, synthetic, order=2
    ):
        structural_zeros = target_probs == 0
        zeros = index.filter(structural_zeros)
        probs = synthetic_probs.filter(structural_zeros)
        all_zeros[name] = pl.DataFrame().with_columns(
            [zeros.alias("zero"), probs.alias("p")]
        )
    return all_zeros


def incorrectness(target: pl.DataFrame, synthetic: pl.DataFrame) -> float:
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
    all_zeros = structural_zeros(target, synthetic)
    p_incorrectness = 0.0
    for cols, zeros in all_zeros.items():
        p_incorrectness += zeros["p"].sum()
    return p_incorrectness / len(all_zeros)
