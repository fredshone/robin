from typing import Tuple

import polars as pl


def cut_and_bin_continuous(data: pl.DataFrame, bins: int = 10) -> pl.DataFrame:
    binned = data.clone()
    labels = [str(i) for i in range(bins)]
    cutter = {}

    for col in binned.select(pl.col(pl.Float64)).columns:
        if binned[col].n_unique() > bins:
            d = (
                binned[col]
                .qcut(bins, include_breaks=True, labels=labels)
                .alias("qcut")
                .to_frame()
                .unnest("qcut")
            )
            cutter[col] = d["breakpoint"].unique().sort().to_list()[:-1]
            d = d["category"].cast(pl.UInt16)
            binned = binned.with_columns(d.alias(col))
    return cutter, binned


def apply_cutter(data: pl.DataFrame, cutter: dict) -> pl.DataFrame:
    binned = data.clone()
    labels = [str(i) for i in range(len(next(iter(cutter.values()))) + 1)]

    for col, breaks in cutter.items():
        binned = binned.with_columns(
            pl.col(col)
            .cut(breaks=breaks, labels=labels)
            .cast(pl.UInt16)
            .alias(col)
        )
    return binned


def bin_continuous(
    target: pl.DataFrame, other: pl.DataFrame, bins: int = 10
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    cutter, binned = cut_and_bin_continuous(target, bins)
    return binned, apply_cutter(other, cutter)
