"""
Minimal continuous BO benchmark (oracle-based).
============================================

Goal: given a manual JSON readout (prior knowledge), compare:
  - Random search
  - Baseline BO (GP + EI)
  - Hybrid BO (prior mean + residual GP)

This uses a continuous domain with bounds derived from the UGI dataset
and an oracle (RandomForest) trained on the full data.
"""
#%%
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from gpytorch.mlls import ExactMarginalLogLikelihood

import matplotlib.pyplot as plt

from data_analysis import build_ugi_ml_oracle, RandomForestOracle
from prior_gp import alignment_on_obs, fit_residual_gp, GPWithPriorMean
from readout_schema import readout_to_prior, flat_readout, normalize_readout_to_unit_box

warnings.filterwarnings("ignore")

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
    metadata: Dict[str, Any]


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
        metadata={
            "oracle_metrics": payload.get("metrics"),
            "n_candidates": int(candidate_pool.shape[0]),
        },
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


def _sample_initial_unit(domain: ContinuousDomain, n_init: int, seed: int, method: str) -> Tensor:
    method = (method or "sobol").lower()
    if n_init <= 0:
        return torch.empty((0, domain.unit_bounds.shape[1]), device=DEVICE, dtype=DTYPE)
    if method in {"sobol", "sobo"}:
        return draw_sobol_samples(bounds=domain.unit_bounds, n=n_init, q=1, seed=seed).squeeze(1)
    if method in {"lhs", "latin", "latin_hypercube"}:
        d = int(domain.unit_bounds.shape[1])
        rng = np.random.default_rng(int(seed))
        points = np.zeros((n_init, d), dtype=np.float64)
        for j in range(d):
            perm = rng.permutation(n_init)
            points[:, j] = (perm + rng.random(n_init)) / n_init
        return torch.tensor(points, dtype=DTYPE, device=DEVICE)
    raise ValueError(f"Unknown init_method: {method}")


def _constraint_penalty_values(
    X_unit: Tensor,
    readout_unit: Dict[str, Any],
    feature_names: List[str],
) -> Tensor:
    if X_unit.ndim == 1:
        X_unit = X_unit.unsqueeze(0)
    constraints = (readout_unit or {}).get("constraints") or []
    if not constraints:
        return torch.zeros(X_unit.shape[0], device=X_unit.device, dtype=X_unit.dtype)

    idx_lookup = {name: i for i, name in enumerate(feature_names)}
    idx_lookup_lower = {name.lower(): i for i, name in enumerate(feature_names)}

    def _dim_index(key: str) -> Optional[int]:
        if key.startswith("x"):
            try:
                j = int(key[1:]) - 1
            except ValueError:
                return None
            return j if 0 <= j < len(feature_names) else None
        if key in idx_lookup:
            return idx_lookup[key]
        return idx_lookup_lower.get(key.lower())

    penalty = torch.zeros(X_unit.shape[0], device=X_unit.device, dtype=X_unit.dtype)
    for c in constraints:
        if not isinstance(c, dict):
            continue
        var = c.get("var", None)
        r = c.get("range", None)
        if var is None or not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        idx = _dim_index(str(var))
        if idx is None:
            continue
        lo = float(r[0])
        hi = float(r[1])
        if hi < lo:
            lo, hi = hi, lo
        strength = float(c.get("penalty", c.get("weight", 5.0)))
        k = float(c.get("sharpness", c.get("k", 60.0)))
        z = X_unit[:, idx]
        gate = torch.sigmoid(k * (z - lo)) - torch.sigmoid(k * (z - hi))
        gate = gate.clamp(0.0, 1.0)
        penalty = penalty + strength * gate
    return penalty


def _apply_constraint_hardness(
    scores: Tensor,
    X_unit: Tensor,
    readout_unit: Dict[str, Any],
    feature_names: List[str],
    *,
    hardness: float,
    best_f: float,
    hard_mask_threshold: float = 0.999,
) -> Tensor:
    hardness = float(np.clip(hardness, 0.0, 1.0))
    if hardness <= 0.0:
        return scores
    penalties = _constraint_penalty_values(X_unit, readout_unit, feature_names)
    if hardness >= hard_mask_threshold:
        mask = penalties > 0
        scores = scores.clone()
        scores[mask] = -1e12
        return scores
    scale = max(abs(float(best_f)), 1e-6)
    return scores - hardness * penalties * scale


def _record_continuous_sample(
    domain: ContinuousDomain,
    x_unit: Tensor,
    y: float,
    best: float,
    *,
    method: str,
    iteration: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = unit_to_raw(domain, x_unit.detach()).squeeze(0)
    rec: Dict[str, Any] = {
        "iter": int(iteration),
        "y": float(y),
        "best_so_far": float(best),
        "method": method,
    }
    for j in range(raw.numel()):
        rec[f"x{j+1}"] = float(raw[j].item())
    if extra:
        rec.update(extra)
    return rec


def run_random_continuous(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int = 0,
    repeats: int = 1,
) -> pd.DataFrame:
    if repeats <= 1:
        return _run_random_continuous_single(domain, n_init=n_init, n_iter=n_iter, seed=seed)
    dfs = []
    for r in range(repeats):
        s = seed + r
        dfr = _run_random_continuous_single(domain, n_init=n_init, n_iter=n_iter, seed=s)
        dfr["seed"] = s
        dfs.append(dfr)
    return pd.concat(dfs, ignore_index=True)


def _run_random_continuous_single(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int,
) -> pd.DataFrame:
    X_init = draw_sobol_samples(bounds=domain.unit_bounds, n=n_init, q=1, seed=seed).squeeze(1)
    Y_init = evaluate_oracle(domain, X_init).unsqueeze(-1)

    recs: List[Dict[str, Any]] = []
    best = float(Y_init.max().item()) if Y_init.numel() else float("-inf")

    for i in range(n_init):
        y = float(Y_init[i].item())
        best = max(best, y)
        recs.append(_record_continuous_sample(domain, X_init[i], y, best, method="random", iteration=i - n_init))

    for t in range(n_iter):
        x_next = draw_sobol_samples(bounds=domain.unit_bounds, n=1, q=1, seed=seed + 4242 + t).squeeze(1)[0]
        y_next = evaluate_oracle(domain, x_next).unsqueeze(-1)
        y_val = float(y_next.item())
        best = max(best, y_val)
        recs.append(_record_continuous_sample(domain, x_next, y_val, best, method="random", iteration=t))

    df = pd.DataFrame(recs)
    df["seed"] = seed
    return df


def run_baseline_ei_continuous(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int = 0,
    repeats: int = 1,
    init_method: str = "sobol",
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> pd.DataFrame:
    if repeats <= 1:
        return _run_baseline_ei_continuous_single(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            init_method=init_method,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    dfs = []
    for r in range(repeats):
        s = seed + r
        dfr = _run_baseline_ei_continuous_single(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=s,
            init_method=init_method,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        dfr["seed"] = s
        dfs.append(dfr)
    return pd.concat(dfs, ignore_index=True)


def _run_baseline_ei_continuous_single(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int,
    init_method: str,
    num_restarts: int,
    raw_samples: int,
) -> pd.DataFrame:
    X_init = _sample_initial_unit(domain, n_init, seed, init_method)
    Y_init = evaluate_oracle(domain, X_init).unsqueeze(-1)

    X_obs = X_init.clone()
    Y_obs = Y_init.clone()
    recs: List[Dict[str, Any]] = []
    best = float(Y_obs.max().item()) if Y_obs.numel() else float("-inf")

    for i in range(n_init):
        y = float(Y_init[i].item())
        best = max(best, y)
        recs.append(_record_continuous_sample(domain, X_init[i], y, best, method="baseline_ei", iteration=i - n_init))

    for t in range(n_iter):
        gp = SingleTaskGP(X_obs, Y_obs)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        EI = ExpectedImprovement(model=gp, best_f=float(Y_obs.max().item()), maximize=True)
        x_next, _ = optimize_acqf(
            EI,
            bounds=domain.unit_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        x_next = x_next.squeeze(0)
        y_next = evaluate_oracle(domain, x_next).unsqueeze(-1)

        X_obs = torch.cat([X_obs, x_next.unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, y_next], dim=0)

        y_val = float(y_next.item())
        best = max(best, y_val)
        recs.append(_record_continuous_sample(domain, x_next, y_val, best, method="baseline_ei", iteration=t))

    df = pd.DataFrame(recs)
    df["seed"] = seed
    return df


def run_hybrid_continuous(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int = 0,
    repeats: int = 1,
    manual_readout: Optional[Dict[str, Any]] = None,
    num_restarts: int = 10,
    raw_samples: int = 256,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    use_alignment_guard: bool = False,
    alignment_min: float = 0.0,
    constraint_hardness: float = 0.0,
    constraint_pool_size: int = 4096,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
) -> pd.DataFrame:
    if repeats <= 1:
        return _run_hybrid_continuous_single(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            manual_readout=manual_readout,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            prior_strength=prior_strength,
            rho_floor=rho_floor,
            use_alignment_guard=use_alignment_guard,
            alignment_min=alignment_min,
            constraint_hardness=constraint_hardness,
            constraint_pool_size=constraint_pool_size,
            early_prior_boost=early_prior_boost,
            early_prior_steps=early_prior_steps,
        )
    dfs = []
    for r in range(repeats):
        s = seed + r
        dfr = _run_hybrid_continuous_single(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=s,
            manual_readout=manual_readout,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            prior_strength=prior_strength,
            rho_floor=rho_floor,
            use_alignment_guard=use_alignment_guard,
            alignment_min=alignment_min,
            constraint_hardness=constraint_hardness,
            constraint_pool_size=constraint_pool_size,
            early_prior_boost=early_prior_boost,
            early_prior_steps=early_prior_steps,
        )
        dfr["seed"] = s
        dfs.append(dfr)
    return pd.concat(dfs, ignore_index=True)


def _run_hybrid_continuous_single(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int,
    manual_readout: Optional[Dict[str, Any]],
    num_restarts: int,
    raw_samples: int,
    prior_strength: float,
    rho_floor: float,
    use_alignment_guard: bool,
    alignment_min: float,
    constraint_hardness: float,
    constraint_pool_size: int,
    early_prior_boost: bool,
    early_prior_steps: int,
) -> pd.DataFrame:
    X_init = draw_sobol_samples(bounds=domain.unit_bounds, n=n_init, q=1, seed=seed).squeeze(1)
    Y_init = evaluate_oracle(domain, X_init).unsqueeze(-1)

    X_obs = X_init.clone()
    Y_obs = Y_init.clone()
    recs: List[Dict[str, Any]] = []
    best = float(Y_obs.max().item()) if Y_obs.numel() else float("-inf")

    for i in range(n_init):
        y = float(Y_init[i].item())
        best = max(best, y)
        recs.append(_record_continuous_sample(domain, X_init[i], y, best, method="hybrid_manual", iteration=i - n_init))

    ro_raw = manual_readout if manual_readout is not None else flat_readout(feature_names=domain.feature_names)
    ro_unit = normalize_readout_to_unit_box(ro_raw, domain.mins, domain.maxs, feature_names=domain.feature_names)
    prior = readout_to_prior(ro_unit, feature_names=domain.feature_names)

    for t in range(n_iter):
        rho = float("nan")
        m0_scale = float("nan")
        guarded = False
        prior_used = False
        if use_alignment_guard:
            rho = float(alignment_on_obs(X_obs, Y_obs, prior))
            guarded = rho < float(alignment_min)

        if early_prior_boost and t < early_prior_steps and not guarded:
            pool = draw_sobol_samples(bounds=domain.unit_bounds, n=1024, q=1, seed=seed + 505 + t).squeeze(1)
            prior_vals = prior.m0_torch(pool).reshape(-1)
            idx_local = int(torch.argmax(prior_vals))
            x_next = pool[idx_local]
            prior_used = True
        else:
            if not use_alignment_guard:
                rho = float(alignment_on_obs(X_obs, Y_obs, prior))

            if guarded:
                gp_plain = SingleTaskGP(X_obs, Y_obs)
                mll = ExactMarginalLogLikelihood(gp_plain.likelihood, gp_plain)
                fit_gpytorch_mll(mll)
                m0_scale = 0.0
                model_total = gp_plain
            else:
                gp_resid, alpha = fit_residual_gp(X_obs, Y_obs, prior)
                rho_weight = max(abs(float(rho)), rho_floor)
                m0_scale = float(alpha * prior_strength * rho_weight)
                model_total = GPWithPriorMean(gp_resid, prior, m0_scale=m0_scale)
                prior_used = True

            best_f = float(Y_obs.max().item())
            effective_hardness = 0.0 if guarded else float(constraint_hardness)
            if effective_hardness > 0.0:
                pool = draw_sobol_samples(
                    bounds=domain.unit_bounds,
                    n=max(1024, int(constraint_pool_size)),
                    q=1,
                    seed=seed + 202 + t,
                ).squeeze(1)
                EI = ExpectedImprovement(model=model_total, best_f=best_f, maximize=True)
                with torch.no_grad():
                    scores = EI(pool.unsqueeze(1)).reshape(-1)
                    scores = _apply_constraint_hardness(
                        scores,
                        pool,
                        ro_unit,
                        domain.feature_names,
                        hardness=effective_hardness,
                        best_f=best_f,
                    )
                    idx_local = int(torch.argmax(scores))
                    x_next = pool[idx_local]
            else:
                EI = ExpectedImprovement(model=model_total, best_f=best_f, maximize=True)
                x_next, _ = optimize_acqf(
                    EI,
                    bounds=domain.unit_bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
                x_next = x_next.squeeze(0)

        y_next = evaluate_oracle(domain, x_next).unsqueeze(-1)
        X_obs = torch.cat([X_obs, x_next.unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, y_next], dim=0)

        y_val = float(y_next.item())
        best = max(best, y_val)
        recs.append(
            _record_continuous_sample(
                domain,
                x_next,
                y_val,
                best,
                method="hybrid_manual",
                iteration=t,
                extra={"rho": rho, "m0_scale": m0_scale, "guarded": guarded, "prior_used": prior_used},
            )
        )

    df = pd.DataFrame(recs)
    df["seed"] = seed
    return df


def run_manual_prior_benchmark(
    domain: ContinuousDomain,
    manual_readout: Dict[str, Any],
    *,
    n_init: int = 6,
    n_iter: int = 25,
    seed: int = 0,
    repeats: int = 5,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    use_alignment_guard: bool = False,
    alignment_min: float = 0.0,
    constraint_hardness: float = 0.0,
    constraint_pool_size: int = 4096,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
    include_random: bool = True,
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    if include_random:
        rand = run_random_continuous(domain, n_init=n_init, n_iter=n_iter, seed=seed, repeats=repeats)
        dfs.append(rand)
    base = run_baseline_ei_continuous(domain, n_init=n_init, n_iter=n_iter, seed=seed, repeats=repeats)
    dfs.append(base)
    hyb = run_hybrid_continuous(
        domain,
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
        repeats=repeats,
        manual_readout=manual_readout,
        prior_strength=prior_strength,
        rho_floor=rho_floor,
        use_alignment_guard=use_alignment_guard,
        alignment_min=alignment_min,
        constraint_hardness=constraint_hardness,
        constraint_pool_size=constraint_pool_size,
        early_prior_boost=early_prior_boost,
        early_prior_steps=early_prior_steps,
    )
    dfs.append(hyb)
    return pd.concat(dfs, ignore_index=True)


def plot_runs_mean_lookup(
    hist_df: pd.DataFrame,
    *,
    methods: Optional[List[str]] = None,
    ci: str = "sd",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    df = hist_df.copy()
    df = df[df["iter"] >= 0]

    if methods is None:
        methods = list(df["method"].unique())

    for m in methods:
        d = df[df["method"] == m]
        if d.empty:
            continue
        agg = d.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
        if ci == "sem":
            err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        elif ci == "95ci":
            err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        else:
            err = agg["std"]
        x = agg["iter"].to_numpy()
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        ax.plot(x, y, label=m)
        ax.fill_between(x, y - e, y + e, alpha=0.20)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best so far")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_alignment_over_time(
    hist_df: pd.DataFrame,
    *,
    method: str = "hybrid_manual",
    ci: str = "sd",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    df = hist_df.copy()
    df = df[(df["method"] == method) & (df["iter"] >= 0)].copy()
    if "rho" not in df.columns:
        raise ValueError("alignment data not found (missing 'rho' column).")

    agg = df.groupby("iter")["rho"].agg(["mean", "std", "count"]).reset_index()
    if ci == "sem":
        err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
    elif ci == "95ci":
        err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
    else:
        err = agg["std"]
    x = agg["iter"].to_numpy()
    y = agg["mean"].to_numpy()
    e = err.to_numpy()

    ax.plot(x, y, label="alignment (rho)")
    ax.fill_between(x, y - e, y + e, alpha=0.20)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Alignment (rho)")

    if "guarded" in df.columns:
        guard_rate = df.groupby("iter")["guarded"].mean().reset_index()
        ax2 = ax.twinx()
        ax2.plot(guard_rate["iter"], guard_rate["guarded"], color="#c44e52", label="guard rate")
        ax2.set_ylabel("Guard rate")

    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return ax




def estimate_optimum(
    domain: ContinuousDomain,
    *,
    n_samples: int = 4096,
    seed: int = 0,
) -> float:
    X = draw_sobol_samples(bounds=domain.unit_bounds, n=n_samples, q=1, seed=seed).squeeze(1)
    y = evaluate_oracle(domain, X).reshape(-1)
    return float(y.max().item())


def compute_simple_regret_curve(
    hist_df: pd.DataFrame,
    *,
    optimum: float,
    include_init: bool = False,
) -> pd.DataFrame:
    df = hist_df.copy()
    if not include_init:
        df = df[df["iter"] >= 0].copy()
    df["simple_regret"] = float(optimum) - df["best_so_far"].astype(float)
    return df


def plot_simple_regret(
    regret_df: pd.DataFrame,
    *,
    methods: Optional[List[str]] = None,
    ci: str = "sd",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    df = regret_df.copy()
    df = df[df["iter"] >= 0]
    if methods is None:
        methods = list(df["method"].unique())

    for m in methods:
        d = df[df["method"] == m]
        if d.empty:
            continue
        agg = d.groupby("iter")["simple_regret"].agg(["mean", "std", "count"]).reset_index()
        if ci == "sem":
            err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        elif ci == "95ci":
            err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        else:
            err = agg["std"]
        x = agg["iter"].to_numpy()
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        ax.plot(x, y, label=m)
        ax.fill_between(x, y - e, y + e, alpha=0.20)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Simple regret (optimum - best)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_guard_sampling_density(
    hist_df: pd.DataFrame,
    *,
    method: str,
    feature: str = "x4",
    bins: int = 24,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    df = hist_df.copy()
    if "guarded" not in df.columns:
        raise ValueError("guarded data not found (missing 'guarded' column).")
    if feature not in df.columns:
        raise ValueError(f"feature not found: {feature}")

    df = df[(df["method"] == method) & (df["iter"] >= 0)].copy()
    df = df.dropna(subset=[feature, "guarded"])
    if df.empty:
        raise ValueError(f"no rows to plot for method={method}")

    guard_on = df[df["guarded"] == True][feature].to_numpy()
    guard_off = df[df["guarded"] == False][feature].to_numpy()

    if guard_on.size:
        ax.hist(guard_on, bins=bins, density=True, alpha=0.55, label="guard on")
    if guard_off.size:
        ax.hist(guard_off, bins=bins, density=True, alpha=0.55, label="guard off")

    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    return ax


def _unit_from_history(domain: ContinuousDomain, df: pd.DataFrame) -> Tensor:
    cols = [f"x{i+1}" for i in range(len(domain.feature_names))]
    X_raw = torch.tensor(df[cols].to_numpy(dtype=np.float64), device=DEVICE, dtype=DTYPE)
    mins = domain.mins.to(device=DEVICE, dtype=DTYPE)
    rng = (domain.maxs - domain.mins).to(device=DEVICE, dtype=DTYPE).clamp_min(1e-12)
    return ((X_raw - mins) / rng).clamp(0.0, 1.0)


def _exploration_quality(X_unit: Tensor, *, n_bins: int = 10) -> float:
    if X_unit.numel() == 0:
        return 0.0
    var = torch.var(X_unit, dim=0, unbiased=False)
    var_norm = (var / (1.0 / 12.0)).clamp(0.0, 1.0)
    var_score = float(var_norm.mean().item())

    X_np = X_unit.detach().cpu().numpy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    cover = []
    for j in range(X_np.shape[1]):
        idx = np.digitize(X_np[:, j], bins, right=True)
        cover.append(len(np.unique(idx)) / n_bins)
    cover_score = float(np.mean(cover)) if cover else 0.0
    return 0.5 * (var_score + cover_score)


def _auc_best_so_far(df: pd.DataFrame, *, include_init: bool) -> float:
    d = df.copy()
    if not include_init:
        d = d[d["iter"] >= 0].copy()
    d = d.sort_values("iter")
    if d.empty:
        return float("nan")
    x = d["iter"].to_numpy(dtype=np.float64)
    y = d["best_so_far"].to_numpy(dtype=np.float64)
    return float(np.trapz(y, x))


def _iters_to_target(df: pd.DataFrame, *, target: float, include_init: bool) -> int:
    d = df.copy()
    if not include_init:
        d = d[d["iter"] >= 0].copy()
    d = d.sort_values("iter")
    hits = d[d["best_so_far"] >= target]
    if hits.empty:
        return int(d["iter"].max() + 1) if not d.empty else 0
    return int(hits["iter"].iloc[0])


def _inference_rmse(
    domain: ContinuousDomain,
    df: pd.DataFrame,
    *,
    X_test: Tensor,
    y_test: Tensor,
) -> float:
    X_unit = _unit_from_history(domain, df)
    Y = torch.tensor(df["y"].to_numpy(dtype=np.float64), device=DEVICE, dtype=DTYPE).unsqueeze(-1)
    if X_unit.shape[0] < 2:
        return float("nan")
    gp = SingleTaskGP(X_unit, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    with torch.no_grad():
        post = gp.posterior(X_test.unsqueeze(1))
        pred = post.mean.reshape(-1)
    rmse = torch.sqrt(torch.mean((pred - y_test) ** 2)).item()
    return float(rmse)


def score_benchmark(
    hist_df: pd.DataFrame,
    domain: ContinuousDomain,
    *,
    include_init: bool = False,
    optimum: Optional[float] = None,
    optimum_samples: int = 4096,
    optimum_seed: int = 0,
    n_test: int = 256,
    test_seed: int = 123,
    target_frac: float = 0.9,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (summary_table, per_run_table, simple_regret_df)."""
    if optimum is None:
        optimum = estimate_optimum(domain, n_samples=optimum_samples, seed=optimum_seed)

    X_test = draw_sobol_samples(bounds=domain.unit_bounds, n=n_test, q=1, seed=test_seed).squeeze(1)
    y_test = evaluate_oracle(domain, X_test).reshape(-1)

    if "seed" not in hist_df.columns:
        hist_df = hist_df.copy()
        hist_df["seed"] = 0

    per_run: List[Dict[str, Any]] = []
    for (method, seed), df_run in hist_df.groupby(["method", "seed"]):
        df_run = df_run.copy()
        final_best = float(df_run.loc[df_run["iter"] >= 0, "best_so_far"].max())
        auc = _auc_best_so_far(df_run, include_init=include_init)
        target = float(optimum) * float(target_frac)
        iters_to = _iters_to_target(df_run, target=target, include_init=include_init)

        X_unit = _unit_from_history(domain, df_run)
        explore = _exploration_quality(X_unit, n_bins=n_bins)
        rmse = _inference_rmse(domain, df_run, X_test=X_test, y_test=y_test)

        denom = abs(float(optimum)) + 1e-9
        final_score = 1.0 - (float(optimum) - final_best) / denom
        final_score = float(np.clip(final_score, 0.0, 1.0))
        max_iters = int(df_run["iter"].max() + 1) if not df_run.empty else 1
        speed_score = 1.0 - min(iters_to, max_iters) / max_iters
        speed_score = float(np.clip(speed_score, 0.0, 1.0))
        composite = 0.4 * final_score + 0.3 * speed_score + 0.3 * explore

        per_run.append(
            {
                "method": method,
                "seed": int(seed),
                "final_best": final_best,
                "auc": auc,
                "iters_to_target": int(iters_to),
                "exploration_quality": float(explore),
                "inference_rmse": float(rmse),
                "composite_score": float(composite),
            }
        )

    per_run_df = pd.DataFrame(per_run)

    agg_map = {
        "final_best": ["median", "mean", "min", "std"],
        "auc": ["median", "mean", "min", "std"],
        "iters_to_target": ["median", "mean", "max", "std"],
        "exploration_quality": ["median", "mean", "min", "std"],
        "inference_rmse": ["median", "mean", "max", "std"],
        "composite_score": ["median", "mean", "min", "std"],
    }
    summary = per_run_df.groupby("method").agg(agg_map)
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary["n_runs"] = per_run_df.groupby("method").size().values

    regret_df = compute_simple_regret_curve(hist_df, optimum=float(optimum), include_init=include_init)
    return summary, per_run_df, regret_df


def plot_metric_bars(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.2))
    df = summary_df.copy()
    if metric not in df.columns:
        raise ValueError(f"metric={metric!r} not found in summary_df columns.")
    ax.bar(df["method"], df[metric])
    ax.set_ylabel(metric)
    ax.set_xlabel("Method")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    return ax
#%%

if __name__ == "__main__":
    domain = build_continuous_domain()

    manual_readout = {
        "constraints": [
            {"var": "x4", "range": [0, 0.1], "reason": "ptsa too high", "penalty": 8.0},
            {"var": "x4", "range": [0.18, 0.3], "reason": "ptsa too high", "penalty": 8.0},
            {"var": "x1", "range": [200, 300.0], "reason": "high amine suppresses yield", "penalty": 8.0},

        ],
    }

    hist = run_manual_prior_benchmark(
        domain,
        manual_readout,
        seed=12,
        n_init=6,
        n_iter=100,
        repeats=5,
        constraint_hardness=0.2,
        constraint_pool_size=20000,
        use_alignment_guard=False,
        alignment_min=0.0,
        early_prior_boost=True,
        early_prior_steps=5,
        prior_strength=1
    )
    plot_runs_mean_lookup(hist, ci="sem", title="Best-so-far")

    # Simple compliance check for x4 within [a, b] on hybrid (exclude init iters)
    a, b = 0.11, 0.13
    df_hyb = hist[(hist["method"] == "hybrid_manual") & (hist["iter"] >= 0)]
    if not df_hyb.empty and "x4" in df_hyb.columns:
        within = ((df_hyb["x4"] >= a) & (df_hyb["x4"] <= b)).mean()
        print(f"Hybrid x4 within [{a}, {b}]: {within * 100:.1f}%")
    else:
        print("Hybrid compliance check skipped (no rows or x4 missing).")

    summary, per_run, regret = score_benchmark(hist, domain)
    plot_simple_regret(regret, ci="sem", title="Simple Regret")
    plot_metric_bars(summary, metric="composite_score_median", title="Composite Score (Median)")
    plot_alignment_over_time(hist, method="hybrid_manual", ci="sem", title="Alignment over time")

    print(summary.sort_values("composite_score_median", ascending=False).to_string(index=False))
    plt.show()



#%%
# Safety fallback demo with a bad prior (toggle guard on/off)
RUN_BAD_PRIOR_DEMO = True
domain = build_continuous_domain()

if RUN_BAD_PRIOR_DEMO:
    bad_readout = {
        "effects": {
            "x4": {"effect": "decreasing", "scale": 0.8, "confidence": 0.9, "range_hint": [0, 0.03]},
        },
        "constraints": [
            {"var": "x4", "range": [0.10, 0.3], "reason": "push away from optimum", "penalty": 8.0},
        ],
    }

    hist_bad_no = run_manual_prior_benchmark(
        domain,
        bad_readout,
        seed=43,
        n_init=6,
        n_iter=100,
        repeats=3,
        use_alignment_guard=False,
        constraint_hardness=0.3,
        constraint_pool_size=20000,
        prior_strength=1,
        early_prior_boost=True,
        early_prior_steps=5,
    )
    hist_bad_guard = run_manual_prior_benchmark(
        domain,
        bad_readout,
        seed=43,
        n_init=6,
        n_iter=100,
        repeats=3,
        use_alignment_guard=True,
        alignment_min=-0.3,
        constraint_hardness=0.0,
        constraint_pool_size=20000,
        prior_strength=1,
        early_prior_boost=True,
        early_prior_steps=5,
    )

    df_bad_no = hist_bad_no[hist_bad_no["method"] == "hybrid_manual"].copy()
    df_bad_guard = hist_bad_guard[hist_bad_guard["method"] == "hybrid_manual"].copy()
    df_bad_base = hist_bad_guard[hist_bad_guard["method"] == "baseline_ei"].copy()
    df_bad_no["method"] = "hybrid_bad_no_guard"
    df_bad_guard["method"] = "hybrid_bad_guard"
    df_bad_base["method"] = "baseline_ei"
    bad_hist = pd.concat([df_bad_no, df_bad_guard, df_bad_base], ignore_index=True)

    plot_runs_mean_lookup(
        bad_hist,
        methods=["baseline_ei", "hybrid_bad_no_guard", "hybrid_bad_guard"],
        ci="sem",
        title="Bad prior: guard comparison (best-so-far)",
    )
    opt_bad = estimate_optimum(domain, n_samples=4096, seed=99)
    regret_bad = compute_simple_regret_curve(bad_hist, optimum=opt_bad, include_init=False)
    plot_simple_regret(
        regret_bad,
        methods=["baseline_ei", "hybrid_bad_no_guard", "hybrid_bad_guard"],
        ci="sem",
        title="Bad prior: guard comparison (regret)",
    )
    plot_guard_sampling_density(
        bad_hist,
        method="hybrid_bad_guard",
        feature="x4",
        bins=24,
        title="Bad prior: x4 sampling density (guard on vs off)",
    )
    score_no = score_benchmark(hist_bad_no, domain)[0]
    score_guard = score_benchmark(hist_bad_guard, domain)[0]
    print("Bad prior (no guard) composite median:",
            score_no["composite_score_median"].median())
    print("Bad prior (guard on) composite median:",
            score_guard["composite_score_median"].median())




# %%
    # manual_readout = {
    #     "effects": {
    #         "x1": {"effect": "decreasing", "scale": 0.6, "confidence": 0.7, "range_hint": [120.0, 150.0]},
    #         "x2": {"effect": "increasing", "scale": 0.6, "confidence": 0.7, "range_hint": [240.0, 300.0]},
    #         "x3": {"effect": "increasing", "scale": 0.7, "confidence": 0.8, "range_hint": [240.0, 300.0]},
    #         "x4": {"effect": "nonmonotone-peak", "scale": 0.6, "confidence": 0.7, "range_hint": [0.10, 0.18]},
    #     },
    #     "interactions": [{"vars": ["x2", "x3"], "type": "synergy", "scale": 0.5, "confidence": 0.6}],
    #     "bumps": [{"mu": [120.0, 285.0, 285.0, 0.12], "sigma": [15.0, 20.0, 20.0, 0.03], "amp": 0.12}],
    #     "constraints": [
    #         {"var": "x1", "range": [150.0, 300.0], "reason": "high amine suppresses yield", "penalty": 8.0},
    #         {"var": "x2", "range": [120.0, 240.0], "reason": "low aldehyde underperforms", "penalty": 8.0},
    #         {"var": "x3", "range": [120.0, 240.0], "reason": "low isocyanide fails", "penalty": 8.0},
    #         {"var": "x4", "range": [0.02, 0.07], "reason": "ptsa too low", "penalty": 8.0},
    #         {"var": "x4", "range": [0.25, 0.30], "reason": "ptsa too high", "penalty": 8.0},
    #     ],
    # }


    # bad_readout = {
    #     "effects": {
    #         "x4": {"effect": "decreasing", "scale": 0.8, "confidence": 0.9, "range_hint": [0, 0.03]},
    #     },
    #     "constraints": [
    #         {"var": "x4", "range": [0.10, 0.3], "reason": "push away from optimum", "penalty": 8.0},
    #     ],
    # }
