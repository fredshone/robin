import math

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def add_joint(
    df: pl.DataFrame,
    columns: tuple[str, str] = ("walls_material", "floor_material"),
    alias: str = "joint",
    separator: str = " & ",
) -> pl.DataFrame:
    return df.with_columns(
        pl.concat_str([columns[0], columns[1]], separator=separator).alias(
            alias
        )
    )


def shorten(labels: list[str] | np.ndarray) -> list[str]:
    return [
        "-".join([word[:4] for word in str(label).split(" ")[:3]])
        .replace("-and-", "&")
        .replace("-or-", "/")
        for label in labels
    ]


def plot_histogram(
    ax, synthetic: pl.DataFrame, target: pl.DataFrame, column: str
):
    combined = pl.concat([target[column], synthetic[column]]).to_numpy()
    bins = np.histogram_bin_edges(combined.astype(float), bins=7)

    ax.hist(
        synthetic[column].to_numpy().astype(float),
        bins=bins,
        label="synthetic",
        density=True,
        alpha=1,
        color="lightcoral",
    )
    ax.hist(
        target[column].to_numpy().astype(float),
        bins=bins,
        label="target",
        density=True,
        alpha=1,
        linewidth=3,
        histtype="step",
        color="royalblue",
    )


def plot_bars(ax, synthetic: pl.DataFrame, target: pl.DataFrame, column: str):
    combined = pl.concat([target[column], synthetic[column]]).to_numpy()
    categories = np.unique(combined.astype(str))

    target_counts = np.array(
        [
            (target[column].to_numpy().astype(str) == category).sum()
            for category in categories
        ]
    ) / len(target)

    synthetic_counts = np.array(
        [
            (synthetic[column].to_numpy().astype(str) == category).sum()
            for category in categories
        ]
    ) / len(synthetic)

    x = np.arange(len(categories))

    ax.bar(
        x,
        synthetic_counts,
        width=0.4,
        label="synthetic",
        facecolor="lightcoral",
    )
    ax.scatter(
        x,
        target_counts,
        label="target",
        color="royalblue",
        lw=3,
        marker="_",
        s=100,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(shorten(categories), rotation=45, ha="right", fontsize=8)


def marginals(
    synthetic: pl.DataFrame,
    target: pl.DataFrame,
    columns: list[str],
    ncols: int = 2,
):
    nrows = math.ceil(len(columns) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(8, 2.5 * nrows))
    axs = np.array(axs).reshape(-1)

    for index, column in enumerate(columns):
        ax = axs[index]

        if target[column].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
            plot_histogram(ax, synthetic, target, column)
        else:
            plot_bars(ax, synthetic, target, column)

        ax.set_title(column)
        if index == 0:
            ax.legend()

    for index in range(len(columns), len(axs)):
        axs[index].axis("off")

    plt.tight_layout()
    return fig


def _joint_distribution(
    synthetic: pl.DataFrame,
    target: pl.DataFrame,
    columns: tuple[str, str],
    head: int,
    tail: int,
):
    target_joint = add_joint(target, columns=columns)
    synthetic_joint = add_joint(synthetic, columns=columns)

    joint = pl.concat(
        [target_joint["joint"], synthetic_joint["joint"]]
    ).to_numpy()
    joint_categories = np.unique(joint.astype(str))

    target_counts = np.array(
        [
            (target_joint["joint"] == category).sum()
            for category in joint_categories
        ]
    ) / len(target)

    synthetic_counts = np.array(
        [
            (synthetic_joint["joint"] == category).sum()
            for category in joint_categories
        ]
    ) / len(synthetic)

    sorted_indices = np.argsort(target_counts)

    least_indices = (
        sorted_indices[:tail] if tail > 0 else np.array([], dtype=int)
    )
    top_indices = (
        sorted_indices[-head:] if head > 0 else np.array([], dtype=int)
    )
    selected_indices = np.concatenate([least_indices, top_indices])

    return {
        "categories": [joint_categories[i] for i in selected_indices],
        "categories_short": shorten(
            [joint_categories[i] for i in selected_indices]
        ),
        "target_counts": target_counts[selected_indices],
        "synthetic_counts": synthetic_counts[selected_indices],
    }


def joint_distributions(
    synthetic: pl.DataFrame,
    target: pl.DataFrame,
    distributions: list[tuple[str, str]],
    head: int = 10,
    tail: int = 10,
):

    nrows = len(distributions)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for index, columns in enumerate(distributions):
        title = f"Joint distribution of {columns[0]} / {columns[1]}"

        distribution = _joint_distribution(
            synthetic=synthetic,
            target=target,
            columns=columns,
            head=head,
            tail=tail,
        )

        x = np.arange(len(distribution["categories"]))
        ax = axes[index]

        ax.bar(
            x,
            distribution["synthetic_counts"],
            width=0.4,
            alpha=1,
            label="synthetic",
            facecolor="lightcoral",
        )
        ax.scatter(
            x,
            distribution["target_counts"],
            label="target",
            color="royalblue",
            lw=3,
            marker="_",
            s=100,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            distribution["categories_short"],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_title(title)
        if index == 0:
            ax.legend()

    plt.tight_layout()
    return fig
