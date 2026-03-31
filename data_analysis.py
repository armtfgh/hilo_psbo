"""
Interactive data analysis utilities for the UGI dataset.
=======================================================

This module is notebook-friendly (no CLI entrypoints). It provides:
- Dataset loading + cleaning helpers
- Global optima/minima discovery
- Feature effect summaries
- Range summaries for top-yield quantiles
- Visualization helpers

It also keeps the RandomForest oracle used by the BO code.
"""
###
#%%
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_IMPORT_ERROR = exc

import torch

warnings.filterwarnings("ignore")

UGI_PATH_PATTERN = "ugi_raw/ugi_hyvu_{:04d}.csv"
UGI_FILE_COUNT = 100
UGI_COLUMN_RENAMES = {
    "[amine](mM)": "amine_mM",
    "[aldehyde](mM)": "aldehyde_mM",
    "[isocyanide](mM)": "isocyanide_mM",
}
UGI_OPTIONAL_DROPS = ["spectrum_dir"]
UGI_FEATURE_ORDER = [
    "amine_mM",
    "aldehyde_mM",
    "isocyanide_mM",
    "ptsa",
]


def _require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError(
            "Matplotlib is required for plotting but is not available. "
            f"Original import error: {MATPLOTLIB_IMPORT_ERROR}"
        )


def load_ugi_series(path_pattern: str = UGI_PATH_PATTERN, count: int = UGI_FILE_COUNT) -> pd.DataFrame:
    data_frames = []
    for i in range(count):
        file_path = path_pattern.format(i)
        data_frames.append(pd.read_csv(file_path))
    if not data_frames:
        raise ValueError("No data files were loaded; check path pattern or file count.")
    return pd.concat(data_frames, ignore_index=True)


def merge_ugi_raw_datasets(
    path_pattern: str = UGI_PATH_PATTERN,
    *,
    count: int = UGI_FILE_COUNT,
    drop_columns: Optional[Iterable[str]] = UGI_OPTIONAL_DROPS,
    column_renames: Optional[Dict[str, str]] = UGI_COLUMN_RENAMES,
) -> pd.DataFrame:
    df = load_ugi_series(path_pattern=path_pattern, count=count)
    if drop_columns:
        df = df.drop(columns=list(drop_columns), errors="ignore")
    if column_renames:
        df = df.rename(columns=column_renames)

    missing = [col for col in UGI_FEATURE_ORDER if col not in df.columns]
    if missing:
        raise ValueError("Missing expected UGI parameters: " + ", ".join(missing))

    desired_order = ["yield", *UGI_FEATURE_ORDER]
    remaining_cols = [c for c in df.columns.tolist() if c not in desired_order]
    ordered = desired_order + remaining_cols
    df = df[[c for c in ordered if c in df.columns]].copy()
    return df


def load_ugi_dataframe(
    *,
    path_pattern: str = UGI_PATH_PATTERN,
    count: int = UGI_FILE_COUNT,
    drop_columns: Optional[Iterable[str]] = UGI_OPTIONAL_DROPS,
    column_renames: Optional[Dict[str, str]] = UGI_COLUMN_RENAMES,
) -> pd.DataFrame:
    return merge_ugi_raw_datasets(
        path_pattern=path_pattern,
        count=count,
        drop_columns=drop_columns,
        column_renames=column_renames,
    )


def _ordered_feature_columns(feature_columns: List[str]) -> List[str]:
    preferred = [col for col in UGI_FEATURE_ORDER if col in feature_columns]
    remaining = [col for col in feature_columns if col not in preferred]
    return preferred + remaining


def select_feature_columns(
    df: pd.DataFrame,
    *,
    target: str = "yield",
    feature_columns: Optional[List[str]] = None,
) -> List[str]:
    if feature_columns is not None:
        return list(feature_columns)
    cols = df.select_dtypes(include="number").columns.tolist()
    if target in cols:
        cols.remove(target)
    return _ordered_feature_columns(cols)


def prepare_features(df: pd.DataFrame, target: str = "yield") -> Tuple[pd.DataFrame, List[str]]:
    df_clean = df.dropna(subset=[target])
    numeric_columns = df_clean.select_dtypes(include="number").columns.tolist()
    if target not in numeric_columns:
        raise ValueError(f"Target column '{target}' not found among numeric columns.")
    numeric_columns.remove(target)
    if not numeric_columns:
        raise ValueError("No numeric features available for training.")
    return df_clean, numeric_columns


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[RandomForestRegressor, Dict[str, float], List[Tuple[float, str]]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "oob_score": float(getattr(model, "oob_score_", float("nan"))),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
    }
    importances = sorted(
        zip(model.feature_importances_.tolist(), X.columns.tolist()), reverse=True
    )
    return model, metrics, importances


class RandomForestOracle:
    """Evaluate the trained RandomForest on torch tensors."""

    def __init__(self, model: RandomForestRegressor, feature_columns: List[str]) -> None:
        self.model = model
        self.feature_columns = feature_columns

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0)
        preds = self.model.predict(candidates.detach().cpu().numpy())
        return torch.tensor(preds, dtype=candidates.dtype, device=candidates.device)


def build_ugi_ml_oracle(
    *,
    path_pattern: str = UGI_PATH_PATTERN,
    file_count: int = UGI_FILE_COUNT,
    drop_columns: Optional[Iterable[str]] = UGI_OPTIONAL_DROPS,
    column_renames: Optional[Dict[str, str]] = UGI_COLUMN_RENAMES,
    target: str = "yield",
    save_merged_path: Optional[Path | str] = None,
) -> Dict[str, object]:
    df_merged = merge_ugi_raw_datasets(
        path_pattern=path_pattern,
        count=file_count,
        drop_columns=drop_columns,
        column_renames=column_renames,
    )

    if save_merged_path is not None:
        path = Path(save_merged_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(path, index=False)

    df_ml, feature_columns = prepare_features(df_merged, target=target)
    feature_columns = _ordered_feature_columns(feature_columns)
    X = df_ml[feature_columns]
    y = df_ml[target]

    model, metrics, feature_importances = train_random_forest(X, y)
    oracle = RandomForestOracle(model, feature_columns)
    candidate_pool_df = df_ml[feature_columns].drop_duplicates().reset_index(drop=True)

    return {
        "dataframe": df_ml,
        "feature_columns": feature_columns,
        "model": model,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "oracle": oracle,
        "candidate_pool_df": candidate_pool_df,
    }


def find_global_optima(
    df: pd.DataFrame,
    *,
    target: str = "yield",
    top_k: int = 10,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    cols = select_feature_columns(df, target=target, feature_columns=feature_columns)
    return df.sort_values(target, ascending=False)[[target, *cols]].head(top_k).reset_index(drop=True)


def find_global_minima(
    df: pd.DataFrame,
    *,
    target: str = "yield",
    top_k: int = 10,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    cols = select_feature_columns(df, target=target, feature_columns=feature_columns)
    return df.sort_values(target, ascending=True)[[target, *cols]].head(top_k).reset_index(drop=True)


def summarize_feature_ranges(
    df: pd.DataFrame,
    *,
    target: str = "yield",
    quantile: float = 0.9,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    cols = select_feature_columns(df, target=target, feature_columns=feature_columns)
    if target not in df.columns:
        raise ValueError(f"target={target!r} not found in DataFrame.")
    thresh = float(df[target].quantile(quantile))
    top = df[df[target] >= thresh]
    out = []
    for col in cols:
        series = top[col].dropna()
        out.append(
            {
                "feature": col,
                "min": float(series.min()),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(out)


def summarize_feature_effects(
    df: pd.DataFrame,
    *,
    target: str = "yield",
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    cols = select_feature_columns(df, target=target, feature_columns=feature_columns)
    out = []
    for col in cols:
        sub = df[[col, target]].dropna()
        if sub.empty:
            continue
        pearson = float(sub[col].corr(sub[target], method="pearson"))
        spearman = float(sub[col].corr(sub[target], method="spearman"))
        slope = float(np.polyfit(sub[col].to_numpy(), sub[target].to_numpy(), 1)[0])
        out.append({"feature": col, "pearson": pearson, "spearman": spearman, "slope": slope})
    return pd.DataFrame(out)


def bin_feature_stats(
    df: pd.DataFrame,
    feature: str,
    *,
    target: str = "yield",
    bins: int = 10,
) -> pd.DataFrame:
    if feature not in df.columns or target not in df.columns:
        raise ValueError("feature or target not found in DataFrame.")
    series = df[[feature, target]].dropna()
    edges = np.linspace(series[feature].min(), series[feature].max(), bins + 1)
    idx = np.digitize(series[feature], edges, right=True)
    rows = []
    for b in range(1, bins + 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "low": float(edges[b - 1]),
                "high": float(edges[b]),
                "count": int(mask.sum()),
                "mean_yield": float(series.loc[mask, target].mean()),
                "median_yield": float(series.loc[mask, target].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_yield_distribution(
    df: pd.DataFrame,
    *,
    target: str = "yield",
    bins: int = 40,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(df[target].dropna().to_numpy(), bins=bins, color="#4c72b0", alpha=0.85)
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    ax.set_title("Yield distribution")
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


def plot_feature_vs_yield(
    df: pd.DataFrame,
    feature: str,
    *,
    target: str = "yield",
    sample: Optional[int] = 4000,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    _require_matplotlib()
    data = df[[feature, target]].dropna()
    if sample is not None and len(data) > sample:
        data = data.sample(sample, random_state=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(data[feature], data[target], s=12, alpha=0.5, color="#4c72b0", edgecolors="none")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"{feature} vs {target}")
    ax.grid(True, alpha=0.2)
    return ax


def plot_feature_bin_trend(
    df: pd.DataFrame,
    feature: str,
    *,
    target: str = "yield",
    bins: int = 12,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    _require_matplotlib()
    stats = bin_feature_stats(df, feature, target=target, bins=bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.0))
    centers = 0.5 * (stats["low"].to_numpy() + stats["high"].to_numpy())
    ax.plot(centers, stats["mean_yield"].to_numpy(), marker="o", color="#c44e52")
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
    target: str = "yield",
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
    target: str = "yield",
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
    target: str = "yield",
    feature_columns: Optional[List[str]] = None,
    sample: Optional[int] = 4000,
    bins: int = 12,
    heatmap_bins: int = 25,
    show: bool = True,
) -> List["plt.Figure"]:
    """Generate a full set of plots for single-parameter and pairwise effects."""
    _require_matplotlib()
    cols = select_feature_columns(df, target=target, feature_columns=feature_columns)
    figures: List["plt.Figure"] = []

    ax = plot_yield_distribution(df, target=target)
    figures.append(ax.figure)

    fig = plot_feature_distributions(df, feature_columns=cols)
    figures.append(fig)

    ax = plot_correlation_matrix(df, target=target, feature_columns=cols)
    figures.append(ax.figure)

    for col in cols:
        ax = plot_feature_vs_yield(df, col, target=target, sample=sample)
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


def analyze_dataset(
    df: pd.DataFrame,
    *,
    target: str = "yield",
    feature_columns: Optional[List[str]] = None,
    top_k: int = 10,
    quantile: float = 0.9,
) -> Dict[str, Any]:
    cols = select_feature_columns(df, target=target, feature_columns=feature_columns)
    summary = df[[target, *cols]].describe().T
    optima = find_global_optima(df, target=target, top_k=top_k, feature_columns=cols)
    minima = find_global_minima(df, target=target, top_k=top_k, feature_columns=cols)
    ranges = summarize_feature_ranges(df, target=target, quantile=quantile, feature_columns=cols)
    effects = summarize_feature_effects(df, target=target, feature_columns=cols)
    return {
        "summary": summary,
        "global_optima": optima,
        "global_minima": minima,
        "top_quantile_ranges": ranges,
        "feature_effects": effects,
    }

#%%

df = load_ugi_dataframe()
info = analyze_dataset(df, top_k=10, quantile=0.9)

info["global_optima"]      # top points
info["global_minima"]      # worst points
info["top_quantile_ranges"]  # best ranges per feature
info["feature_effects"]    # correlations and slope

plot_yield_distribution(df)
plot_feature_vs_yield(df, "amine_mM")
plot_pairwise_heatmap(df, "aldehyde_mM", "isocyanide_mM")

#%%
from data_analysis import load_ugi_dataframe, plot_all_parameter_effects

df = load_ugi_dataframe()
plot_all_parameter_effects(df, target="yield", show=True)

# %%
