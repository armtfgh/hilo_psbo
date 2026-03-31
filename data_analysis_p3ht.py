"""
Notebook-friendly data analysis utilities for the P3HT dataset.
==============================================================

Plots correspondences between each parameter and the objective.
"""
#%%
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_IMPORT_ERROR = exc


P3HT_DATA_PATH = Path(__file__).with_name("P3HT_dataset.csv")
P3HT_FEATURE_COLUMNS = [
    "P3HT content (%)",
    "D1 content (%)",
    "D2 content (%)",
    "D6 content (%)",
    "D8 content (%)",
]
P3HT_TARGET_COLUMN = "Conductivity"


def _require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError(
            "Matplotlib is required for plotting but is not available. "
            f"Original import error: {MATPLOTLIB_IMPORT_ERROR}"
        )


def load_p3ht_dataframe(path: Path | str = P3HT_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def select_feature_columns(
    df: pd.DataFrame,
    *,
    target: str = P3HT_TARGET_COLUMN,
    feature_columns: Optional[List[str]] = None,
) -> List[str]:
    if feature_columns is not None:
        return list(feature_columns)
    cols = df.select_dtypes(include="number").columns.tolist()
    if target in cols:
        cols.remove(target)
    return cols


def plot_target_distribution(
    df: pd.DataFrame,
    *,
    target: str = P3HT_TARGET_COLUMN,
    bins: int = 40,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(df[target].dropna().to_numpy(), bins=bins, color="#4c72b0", alpha=0.85)
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    ax.set_title("Target distribution")
    ax.grid(True, alpha=0.2)
    return ax


def plot_feature_distributions(
    df: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]] = None,
    bins: int = 30,
) -> "plt.Figure":
    _require_matplotlib()
    cols = select_feature_columns(df, feature_columns=feature_columns)
    n = len(cols)
    if n == 0:
        raise ValueError("No feature columns provided.")
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna().to_numpy(), bins=bins, color="#55a868", alpha=0.85)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Feature distributions", y=0.98)
    fig.tight_layout()
    return fig


def plot_feature_vs_target(
    df: pd.DataFrame,
    feature: str,
    *,
    target: str = P3HT_TARGET_COLUMN,
    sample: Optional[int] = None,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    _require_matplotlib()
    data = df[[feature, target]].dropna()
    if sample is not None and len(data) > sample:
        data = data.sample(sample, random_state=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(data[feature], data[target], s=14, alpha=0.6, color="#4c72b0", edgecolors="none")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"{feature} vs {target}")
    ax.grid(True, alpha=0.2)
    return ax


def plot_feature_bin_trend(
    df: pd.DataFrame,
    feature: str,
    *,
    target: str = P3HT_TARGET_COLUMN,
    bins: int = 12,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    _require_matplotlib()
    data = df[[feature, target]].dropna()
    edges = np.linspace(data[feature].min(), data[feature].max(), bins + 1)
    idx = np.digitize(data[feature], edges, right=True)
    rows = []
    for b in range(1, bins + 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "low": float(edges[b - 1]),
                "high": float(edges[b]),
                "mean_target": float(data.loc[mask, target].mean()),
            }
        )
    stats = pd.DataFrame(rows)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.0))
    centers = 0.5 * (stats["low"].to_numpy() + stats["high"].to_numpy())
    ax.plot(centers, stats["mean_target"].to_numpy(), marker="o", color="#c44e52")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"Mean {target}")
    ax.set_title(f"{feature} binned trend")
    ax.grid(True, alpha=0.2)
    return ax


def plot_pairwise_heatmap(
    df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    *,
    target: str = P3HT_TARGET_COLUMN,
    bins: int = 25,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    _require_matplotlib()
    data = df[[feature_x, feature_y, target]].dropna()
    x = data[feature_x].to_numpy()
    y = data[feature_y].to_numpy()
    z = data[target].to_numpy()
    x_edges = np.linspace(x.min(), x.max(), bins + 1)
    y_edges = np.linspace(y.min(), y.max(), bins + 1)
    sum_grid = np.zeros((bins, bins), dtype=np.float64)
    count_grid = np.zeros((bins, bins), dtype=np.float64)
    x_idx = np.digitize(x, x_edges, right=True) - 1
    y_idx = np.digitize(y, y_edges, right=True) - 1
    valid = (x_idx >= 0) & (x_idx < bins) & (y_idx >= 0) & (y_idx < bins)
    for xi, yi, zi in zip(x_idx[valid], y_idx[valid], z[valid]):
        sum_grid[yi, xi] += zi
        count_grid[yi, xi] += 1
    mean_grid = np.divide(sum_grid, count_grid, out=np.full_like(sum_grid, np.nan), where=count_grid > 0)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(
        mean_grid,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label=f"Mean {target}")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f"{feature_x} vs {feature_y} heatmap")
    return ax


def plot_correlation_matrix(
    df: pd.DataFrame,
    *,
    target: str = P3HT_TARGET_COLUMN,
    feature_columns: Optional[List[str]] = None,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    _require_matplotlib()
    cols = select_feature_columns(df, target=target, feature_columns=feature_columns)
    cols = [target, *cols]
    corr = df[cols].corr()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation matrix")
    return ax


def plot_all_parameter_effects(
    df: pd.DataFrame,
    *,
    target: str = P3HT_TARGET_COLUMN,
    feature_columns: Optional[List[str]] = None,
    sample: Optional[int] = None,
    bins: int = 12,
    heatmap_bins: int = 25,
    show: bool = True,
) -> List["plt.Figure"]:
    _require_matplotlib()
    cols = select_feature_columns(df, target=target, feature_columns=feature_columns)
    figures: List["plt.Figure"] = []

    ax = plot_target_distribution(df, target=target)
    figures.append(ax.figure)

    fig = plot_feature_distributions(df, feature_columns=cols)
    figures.append(fig)

    ax = plot_correlation_matrix(df, target=target, feature_columns=cols)
    figures.append(ax.figure)

    for col in cols:
        ax = plot_feature_vs_target(df, col, target=target, sample=sample)
        figures.append(ax.figure)
        ax = plot_feature_bin_trend(df, col, target=target, bins=bins)
        figures.append(ax.figure)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            ax = plot_pairwise_heatmap(df, cols[i], cols[j], target=target, bins=heatmap_bins)
            figures.append(ax.figure)

    if show:
        plt.show()
    return figures
