# prior_benchmark.py

#%%
"""
Short notebook-friendly benchmark to test prior-quality detectors on UGI.

We compare alignment (prior vs observations) and MLL-based NLL difference
between a skeptic GP and a believer GPWithPriorMean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

import matplotlib.pyplot as plt

from data_analysis import build_ugi_ml_oracle, RandomForestOracle
from prior_gp import alignment_on_obs, fit_residual_gp, GPWithPriorMean
from readout_schema import normalize_readout_to_unit_box, readout_to_prior

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


@dataclass
class ContinuousDomain:
    feature_names: List[str]
    mins: Tensor
    maxs: Tensor
    bounds: Tensor
    unit_bounds: Tensor
    oracle: RandomForestOracle


def build_continuous_domain(*, target: str = "yield") -> ContinuousDomain:
    payload = build_ugi_ml_oracle(target=target)
    candidate_pool: pd.DataFrame = payload["candidate_pool_df"]  # type: ignore[index]
    feature_names: List[str] = payload["feature_columns"]  # type: ignore[index]
    oracle: RandomForestOracle = payload["oracle"]  # type: ignore[index]

    mins_np = candidate_pool.min().to_numpy(dtype="float64")
    maxs_np = candidate_pool.max().to_numpy(dtype="float64")
    mins = torch.tensor(mins_np, dtype=DTYPE, device=DEVICE)
    maxs = torch.tensor(maxs_np, dtype=DTYPE, device=DEVICE)
    bounds = torch.stack([mins, maxs])
    unit_bounds = torch.stack(
        [
            torch.zeros(len(feature_names), dtype=DTYPE, device=DEVICE),
            torch.ones(len(feature_names), dtype=DTYPE, device=DEVICE),
        ],
        dim=0,
    )

    return ContinuousDomain(
        feature_names=feature_names,
        mins=mins,
        maxs=maxs,
        bounds=bounds,
        unit_bounds=unit_bounds,
        oracle=oracle,
    )


def unit_to_raw(domain: ContinuousDomain, X_unit: Tensor) -> Tensor:
    mins = domain.mins.to(device=X_unit.device, dtype=X_unit.dtype)
    rng = (domain.maxs - domain.mins).to(device=X_unit.device, dtype=X_unit.dtype).clamp_min(1e-12)
    while X_unit.ndim < mins.ndim:
        X_unit = X_unit.unsqueeze(0)
    return mins + X_unit * rng


def evaluate_oracle(domain: ContinuousDomain, X_unit: Tensor) -> Tensor:
    raw = unit_to_raw(domain, X_unit)
    return domain.oracle(raw)


def _predictive_nll(model: Any, X: Tensor, Y: Tensor) -> float:
    post = model.posterior(X, observation_noise=True)
    mvn = post.mvn
    y = Y.reshape(-1)
    nll = -mvn.log_prob(y)
    return float(nll.detach().cpu().item()) / max(int(y.numel()), 1)


def estimate_optimum_point(
    domain: ContinuousDomain,
    *,
    n_samples: int = 4096,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, float]:
    X = draw_sobol_samples(bounds=domain.unit_bounds, n=n_samples, q=1, seed=seed).squeeze(1)
    y = evaluate_oracle(domain, X).reshape(-1)
    idx = int(torch.argmax(y))
    x_unit = X[idx]
    x_raw = unit_to_raw(domain, x_unit).squeeze(0)
    return x_unit, x_raw, float(y[idx].item())


def build_bad_readout(
    domain: ContinuousDomain,
    *,
    span_frac: float = 0.05,
    penalty: float = 8.0,
    seed: int = 0,
) -> Dict[str, Any]:
    _, x_raw, _ = estimate_optimum_point(domain, n_samples=4096, seed=seed)
    mins = domain.mins.detach().cpu().numpy()
    maxs = domain.maxs.detach().cpu().numpy()
    span = maxs - mins
    constraints = []
    for j in range(int(x_raw.numel())):
        width = float(span_frac) * float(span[j])
        center = float(x_raw[j].item())
        lo = max(float(mins[j]), center - 0.5 * width)
        hi = min(float(maxs[j]), center + 0.5 * width)
        constraints.append(
            {
                "var": f"x{j+1}",
                "range": [lo, hi],
                "reason": "forbid optimum region",
                "penalty": float(penalty),
            }
        )
    return {"constraints": constraints}


def build_good_readout(
    domain: ContinuousDomain,
    *,
    span_frac: float = 0.05,
    amp: float = 0.12,
    seed: int = 0,
) -> Dict[str, Any]:
    _, x_raw, _ = estimate_optimum_point(domain, n_samples=4096, seed=seed)
    mins = domain.mins.detach().cpu().numpy()
    maxs = domain.maxs.detach().cpu().numpy()
    span = maxs - mins
    sigma = (span * float(span_frac)).tolist()
    return {
        "bumps": [
            {
                "mu": x_raw.detach().cpu().tolist(),
                "sigma": sigma,
                "amp": float(amp),
            }
        ]
    }


def _compute_alignment(prior: Any, X_obs: Tensor, Y_obs: Tensor) -> float:
    return float(alignment_on_obs(X_obs, Y_obs, prior))


def _compute_mll_diff(prior: Any, X_obs: Tensor, Y_obs: Tensor, X_val: Tensor, Y_val: Tensor) -> float:
    gp_s = SingleTaskGP(X_obs, Y_obs)
    mll_s = ExactMarginalLogLikelihood(gp_s.likelihood, gp_s)
    fit_gpytorch_mll(mll_s)
    nll_s = _predictive_nll(gp_s, X_val, Y_val)

    gp_resid, alpha = fit_residual_gp(X_obs, Y_obs, prior)
    model_b = GPWithPriorMean(gp_resid, prior, m0_scale=float(alpha))
    nll_b = _predictive_nll(model_b, X_val, Y_val)
    return float(nll_s - nll_b)


def run_prior_quality_benchmark(
    domain: ContinuousDomain,
    *,
    readouts: Dict[str, Dict[str, Any]],
    labels: Dict[str, bool],
    n_init: int = 3,
    n_iter: int = 20,
    seed: int = 0,
    val_size: int = 128,
    val_seed: int = 0,
    min_points: int = 4,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = int(n_init + n_iter)

    for i, (name, readout) in enumerate(readouts.items()):
        label = bool(labels.get(name, False))
        ro_unit = normalize_readout_to_unit_box(readout, domain.mins, domain.maxs, feature_names=domain.feature_names)
        prior = readout_to_prior(ro_unit, feature_names=domain.feature_names)

        X_seq = draw_sobol_samples(
            bounds=domain.unit_bounds,
            n=total,
            q=1,
            seed=int(seed + i * 97),
        ).squeeze(1)
        Y_seq = evaluate_oracle(domain, X_seq).unsqueeze(-1)
        X_val = draw_sobol_samples(
            bounds=domain.unit_bounds,
            n=int(val_size),
            q=1,
            seed=int(val_seed + i * 101),
        ).squeeze(1)
        Y_val = evaluate_oracle(domain, X_val).unsqueeze(-1)

        for t in range(n_iter):
            X_obs = X_seq[: n_init + t]
            Y_obs = Y_seq[: n_init + t]
            if X_obs.shape[0] < min_points:
                align = float("nan")
                mll_diff = float("nan")
            else:
                align = _compute_alignment(prior, X_obs, Y_obs)
                mll_diff = _compute_mll_diff(prior, X_obs, Y_obs, X_val, Y_val)

            rows.append(
                {
                    "readout": name,
                    "is_good": label,
                    "iter": int(t),
                    "alignment": align,
                    "mll_diff": mll_diff,
                }
            )

    return pd.DataFrame(rows)


def plot_metric_over_time(df: pd.DataFrame, *, metric: str) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, d in df.groupby("readout"):
        ax.plot(d["iter"] + 1, d[metric], label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_detection_accuracy(
    df: pd.DataFrame,
    *,
    metric: str,
    threshold: float,
) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    data = df.dropna(subset=[metric]).copy()
    data["pred_good"] = data[metric] >= float(threshold)
    acc = (
        data.groupby("iter")
        .apply(lambda g: float((g["pred_good"] == g["is_good"]).mean()))
        .reset_index(name="accuracy")
    )
    ax.plot(acc["iter"] + 1, acc["accuracy"], marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Detection accuracy")
    ax.set_ylim(0.0, 1.05)
    plt.tight_layout()
    return ax

bad_readout = {
    "effects": {
        "x4": {"effect": "decreasing", "scale": 0.8, "confidence": 0.9, "range_hint": [0, 0.03]},
    },
    "constraints": [
        {"var": "x4", "range": [0.10, 0.3], "reason": "push away from optimum", "penalty": 2.0},
    ],
}

good_readout = {
        "constraints": [
            {"var": "x4", "range": [0, 0.1], "reason": "ptsa too high", "penalty": 8.0},
            # {"var": "x4", "range": [0.18, 0.3], "reason": "ptsa too high", "penalty": 8.0},
            {"var": "x1", "range": [200, 300.0], "reason": "high amine suppresses yield", "penalty": 2.0},
        ],
    }

#%%
# Demo (toggle RUN_DEMO to True in a notebook).
RUN_DEMO = True
if RUN_DEMO:
    domain = build_continuous_domain()
    readouts = {
        "good": good_readout,
        "bad": bad_readout,
    }
    labels = {"good": True, "bad": False}

    results = run_prior_quality_benchmark(
        domain,
        readouts=readouts,
        labels=labels,
        n_init=3,
        n_iter=20,
        seed=13,
        val_size=128,
        val_seed=31,
    )

    plot_metric_over_time(results, metric="alignment")
    plot_metric_over_time(results, metric="mll_diff")
    plot_detection_accuracy(results, metric="alignment", threshold=0.0)
    plot_detection_accuracy(results, metric="mll_diff", threshold=0.0)
    plt.show()

# %%
