
#%%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_prior_usage_files(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("awcd_prior_usage_*_thr_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prior usage files found in {root}")
    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _plot_bar(summary: pd.DataFrame, readout: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(summary))
    y = summary["mean"].to_numpy() * 100.0
    err = summary["sem"].to_numpy() * 100.0

    ax.bar(
        x,
        y,
        yerr=err,
        capsize=4,
        color="#d4af37",
        edgecolor="#3b2f0b",
        linewidth=0.8,
        hatch="///",
    )
    for xi, yi, ei in zip(x, y, err):
        ax.text(xi, yi + ei + 2.0, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in summary["confidence"].to_numpy()])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Prior Used (%)")
    ax.set_title(f"{readout.capitalize()} readout")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path("awcd_sweep_ugi_seed36repeat 10")
    df = _load_prior_usage_files(root)
    df.to_csv("prior_usage_merged.csv", index=False)

    for readout in ["good", "bad", "random"]:
        sub = df[df["readout"].str.lower() == readout].copy()
        if sub.empty:
            summary = pd.DataFrame()
        else:
            summary = (
                sub.groupby("confidence")["prior_used"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("confidence")
            )
            summary["sem"] = summary["std"] / np.maximum(summary["count"], 1) ** 0.5
        _plot_bar(summary, readout, Path(f"prior_usage_{readout}.png"))

    # P3HT summaries (from full sweep files)
    p3ht_good_bad = pd.read_csv("p3ht_sweep_all_but_random.csv")
    p3ht_random = pd.read_csv("p3ht_all_sweep_just_random.csv")
    p3ht = pd.concat([p3ht_good_bad, p3ht_random], ignore_index=True)
    p3ht.to_csv("prior_usage_p3ht_merged.csv", index=False)

    def _p3ht_summary(readout: str) -> pd.DataFrame:
        df_hist = p3ht[p3ht["table"] == "history"].copy()
        df_hist = df_hist[df_hist["readout"].str.lower() == readout]
        df_hist = df_hist[(df_hist["method"] == "ensemble") & df_hist["prior_active"].notna() & (df_hist["iter"] >= 1)]
        if df_hist.empty:
            return pd.DataFrame()
        per_seed = (
            df_hist.groupby(["confidence", "seed"])["prior_active"]
            .apply(lambda s: s.astype(bool).mean())
            .reset_index(name="prior_used")
        )
        summary = (
            per_seed.groupby("confidence")["prior_used"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("confidence")
        )
        summary["sem"] = summary["std"] / np.maximum(summary["count"], 1) ** 0.5
        return summary

    for readout in ["good", "bad", "random"]:
        summary = _p3ht_summary(readout)
        _plot_bar(summary, readout, Path(f"prior_usage_p3ht_{readout}.png"))


if __name__ == "__main__":
    main()

# %%
