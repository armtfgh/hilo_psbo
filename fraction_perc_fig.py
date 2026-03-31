#%%
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_THR_RE = re.compile(r"awcd_history_(?P<readout>[^_]+)_thr_(?P<thr>\\d+p\\d+)\\.csv")


def _parse_confidence(name: str) -> tuple[str, float] | None:
    match = _THR_RE.match(name)
    if not match:
        return None
    readout = match.group("readout").strip().lower()
    thr_text = match.group("thr").replace("p", ".")
    try:
        thr = float(thr_text)
    except ValueError:
        return None
    return readout, thr


def _compute_prior_active_from_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prior_active" not in df.columns or "seed" not in df.columns:
        return pd.DataFrame()
    mask = (df["method"] == "ensemble") & df["prior_active"].notna() & (df["iter"] >= 1)
    sub = df.loc[mask, ["seed", "prior_active"]].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["prior_active"] = sub["prior_active"].astype(bool)
    per_seed = sub.groupby("seed")["prior_active"].mean().reset_index()
    per_seed.rename(columns={"prior_active": "prior_active_mean"}, inplace=True)
    return per_seed


def _plot_bar(summary: pd.DataFrame, readout: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(summary))
    y = summary["mean"].to_numpy() * 100.0
    err = summary["sem"].to_numpy() * 100.0

    ax.bar(x, y, yerr=err, capsize=4, color="#4c72b0", edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in summary["confidence"].to_numpy()])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Prior Active (%)")
    ax.set_title(f"{readout.capitalize()} readout")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 1.5, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_dir = Path("awcd_sweep_ugi_seed36repeat 10")
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing directory: {base_dir}")

    records = []
    for path in base_dir.glob("awcd_history_*_thr_*.csv"):
        parsed = _parse_confidence(path.name)
        if parsed is None:
            continue
        readout, confidence = parsed
        if readout not in {"good", "bad", "random"}:
            continue
        per_seed = _compute_prior_active_from_history(path)
        if per_seed.empty:
            continue
        for _, row in per_seed.iterrows():
            records.append(
                {
                    "readout": readout,
                    "confidence": float(confidence),
                    "seed": int(row["seed"]),
                    "prior_active": float(row["prior_active_mean"]),
                }
            )

    if not records:
        raise RuntimeError("No prior usage data found in awcd_history files.")

    df = pd.DataFrame(records)
    for readout in ["good", "bad", "random"]:
        sub = df[df["readout"] == readout]
        if sub.empty:
            summary = pd.DataFrame()
        else:
            summary = (
                sub.groupby("confidence")["prior_active"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("confidence")
            )
            summary["sem"] = summary["std"] / np.maximum(summary["count"], 1) ** 0.5
        out_path = Path(f"prior_active_{readout}.png")
        _plot_bar(summary, readout, out_path)


if __name__ == "__main__":
    main()

# %%
