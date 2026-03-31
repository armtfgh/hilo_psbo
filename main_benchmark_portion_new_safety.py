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
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.exceptions import ModelFittingError

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import cholesky_jitter

import matplotlib.pyplot as plt

from data_analysis import build_ugi_ml_oracle, RandomForestOracle
from prior_gp import alignment_on_obs, fit_residual_gp, GPWithPriorMean
from data_to_prior import get_data_derived_prior
from readout_schema import readout_to_prior, flat_readout, normalize_readout_to_unit_box

warnings.filterwarnings("ignore")

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 11
METHOD_COLORS = {
    "random": "#4c72b0",
    "baseline_ei": "#55a868",
    "hybrid_manual": "#dd8452",
    "baseline": "#27ae60",
    "forced_prior": "#e74c3c",
    "ensemble": "#f39c12",
    "baseline_bad": "#27ae60",
    "forced_bad": "#e74c3c",
    "awcd_guard": "#f39c12",
}


def _set_iteration_ticks(ax: plt.Axes, max_iter: int) -> None:
    if max_iter < 0:
        return
    display_max = int(max_iter) + 1
    ticks = [1]
    if display_max >= 10:
        ticks.extend(list(range(10, display_max + 1, 10)))
        if ticks[-1] != display_max:
            ticks.append(display_max)
    elif display_max not in ticks:
        ticks.append(display_max)
    ax.set_xticks(sorted(set(ticks)))


def _method_label(method: str) -> str:
    mapping = {
        "random": "Random",
        "baseline_ei": "Baseline BO",
        "hybrid_manual": "Prior-Shaped BO",
        "baseline": "Baseline",
        "forced_prior": "PSBO-No Guard",
        "ensemble": "PSBO-Guarded",
        "baseline_bad": "Baseline",
        "forced_bad": "PSBO-No Guard",
        "awcd_guard": "PSBO-Guarded",
    }
    return mapping.get(method, method)




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


class WeightedExpectedImprovement(AcquisitionFunction):
    def __init__(
        self,
        model_skeptic: SingleTaskGP,
        model_believer: GPWithPriorMean,
        *,
        weight: float,
        best_f: float,
        readout_unit: Dict[str, Any],
        feature_names: List[str],
        constraint_hardness: float,
    ) -> None:
        super().__init__(model=model_skeptic)
        self.ei_s = ExpectedImprovement(model=model_skeptic, best_f=best_f, maximize=True)
        self.ei_b = ExpectedImprovement(model=model_believer, best_f=best_f, maximize=True)
        self.weight = torch.tensor(float(weight), device=DEVICE, dtype=DTYPE)
        self.readout_unit = readout_unit
        self.feature_names = feature_names
        self.constraint_hardness = float(constraint_hardness)
        self.best_f = float(best_f)
        self.dim = int(len(feature_names))

    def forward(self, X: Tensor) -> Tensor:
        ei_s = self.ei_s(X)
        ei_b = self.ei_b(X)
        if self.constraint_hardness > 0.0:
            scores = ei_b.reshape(-1)
            X_flat = X.reshape(-1, self.dim)
            scores = _apply_constraint_hardness(
                scores,
                X_flat,
                self.readout_unit,
                self.feature_names,
                hardness=self.constraint_hardness,
                best_f=self.best_f,
            )
            ei_b = scores.view_as(ei_b)
        return self.weight * ei_b + (1.0 - self.weight) * ei_s


def _compute_mll(model: SingleTaskGP, X: Tensor, Y: Tensor) -> float:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    val = mll(model(X), Y)
    return float(val.detach().cpu().sum().item())


def _predictive_nll(model: Any, X: Tensor, Y: Tensor) -> float:
    post = model.posterior(X, observation_noise=True)
    mvn = post.mvn
    y = Y.reshape(-1)
    nll = -mvn.log_prob(y)
    return float(nll.detach().cpu().item()) / max(int(y.numel()), 1)

def _awcd_full_score(
    gp_skeptic: SingleTaskGP,
    prior_mean_func: Any,
    *,
    X_obs: Tensor,
    Y_obs: Tensor,
    readout_unit: Dict[str, Any],
    feature_names: List[str],
    pool_n: int,
    top_frac: float,
    seed: int,
    bounds: Tensor,
) -> Dict[str, float]:
    def _hard_constraint_mask(
        X_unit: Tensor,
        readout_unit: Dict[str, Any],
        feature_names: List[str],
    ) -> Tensor:
        constraints = (readout_unit or {}).get("constraints") or []
        if not constraints:
            return torch.zeros(X_unit.shape[0], dtype=torch.bool, device=X_unit.device)

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

        mask = torch.zeros(X_unit.shape[0], dtype=torch.bool, device=X_unit.device)
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
            z = X_unit[:, idx]
            inside = (z >= lo) & (z <= hi)
            mask = mask | inside
        return mask

    pool = draw_sobol_samples(bounds=bounds, n=int(pool_n), q=1, seed=int(seed)).squeeze(1)
    mask = _hard_constraint_mask(pool, readout_unit, feature_names)
    if not bool(mask.any()):
        return {"awcd_constraint": 0.0, "awcd_mean": 0.0, "awcd_total": 0.0}
    if bool((~mask).all()):
        return {"awcd_constraint": 0.0, "awcd_mean": 0.0, "awcd_total": 0.0}

    best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
    EI = ExpectedImprovement(model=gp_skeptic, best_f=best_f, maximize=True)
    with torch.no_grad():
        ei_vals = EI(pool.unsqueeze(1)).reshape(-1)
        try:
            with cholesky_jitter(1e-4):
                post = gp_skeptic.posterior(pool, observation_noise=True)
                std = post.variance.clamp_min(1e-12).sqrt().reshape(-1)
        except RuntimeError:
            try:
                with cholesky_jitter(1e-2):
                    post = gp_skeptic.posterior(pool, observation_noise=True)
                    std = post.variance.clamp_min(1e-12).sqrt().reshape(-1)
            except RuntimeError:
                std = torch.zeros(pool.shape[0], device=pool.device, dtype=pool.dtype)
        pressure = ei_vals * std
        prior_mean = prior_mean_func(pool).reshape(-1)

    k = max(1, int(float(top_frac) * float(pressure.numel())))
    top_idx = torch.topk(pressure, k=k).indices
    top_mask = mask[top_idx]
    if top_mask.numel() == 0:
        return {"awcd_constraint": 0.0, "awcd_mean": 0.0, "awcd_total": 0.0}

    prior_rank = torch.argsort(torch.argsort(prior_mean, descending=True))
    median_rank = int(prior_mean.numel() // 2)
    mask_mean = prior_rank[top_idx] > median_rank

    base_forbidden = float(mask.float().mean().item())
    top_forbidden = float(top_mask.float().mean().item())
    if base_forbidden >= 0.999:
        awcd_constraint = 0.0
    else:
        awcd_constraint = max(0.0, (top_forbidden - base_forbidden) / max(1e-6, 1.0 - base_forbidden))
    awcd_mean = float(mask_mean.float().mean().item())
    return {
        "awcd_constraint": awcd_constraint,
        "awcd_mean": awcd_mean,
        "awcd_total": max(awcd_constraint, awcd_mean),
    }


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
    disable_constraints_when_guarded: bool = False,
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
            disable_constraints_when_guarded=disable_constraints_when_guarded,
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
            disable_constraints_when_guarded=disable_constraints_when_guarded,
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
    disable_constraints_when_guarded: bool,
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
            effective_hardness = float(constraint_hardness)
            if guarded and disable_constraints_when_guarded:
                effective_hardness = 0.0
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


def run_ensemble_continuous(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int = 0,
    repeats: int = 1,
    manual_readout: Optional[Dict[str, Any]] = None,
    init_method: str = "sobol",
    num_restarts: int = 10,
    raw_samples: int = 256,
    prior_strength: float = 1.0,
    constraint_hardness: float = 0.0,
    constraint_pool_size: int = 4096,
    awcd_window: int = 5,
    awcd_threshold: Optional[float] = None,
    awcd_constraint_threshold: float = 0.6,
    awcd_mean_threshold: float = 0.6,
    awcd_warmup: int = 3,
    awcd_pool_size: int = 1024,
    awcd_top_frac: float = 0.1,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
    weight_mode: str = "ensemble",
    method_tag: str = "ensemble",
) -> pd.DataFrame:
    if repeats <= 1:
        if awcd_threshold is not None:
            awcd_constraint_threshold = float(awcd_threshold)
            awcd_mean_threshold = float(awcd_threshold)
        return _run_ensemble_continuous_single(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            manual_readout=manual_readout,
            init_method=init_method,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            prior_strength=prior_strength,
            constraint_hardness=constraint_hardness,
            constraint_pool_size=constraint_pool_size,
            awcd_window=awcd_window,
            awcd_constraint_threshold=awcd_constraint_threshold,
            awcd_mean_threshold=awcd_mean_threshold,
            awcd_warmup=awcd_warmup,
            awcd_pool_size=awcd_pool_size,
            awcd_top_frac=awcd_top_frac,
            early_prior_boost=early_prior_boost,
            early_prior_steps=early_prior_steps,
            weight_mode=weight_mode,
            method_tag=method_tag,
        )
    dfs = []
    for r in range(repeats):
        s = seed + r
        if awcd_threshold is not None:
            awcd_constraint_threshold = float(awcd_threshold)
            awcd_mean_threshold = float(awcd_threshold)
        dfr = _run_ensemble_continuous_single(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=s,
            manual_readout=manual_readout,
            init_method=init_method,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            prior_strength=prior_strength,
            constraint_hardness=constraint_hardness,
            constraint_pool_size=constraint_pool_size,
            awcd_window=awcd_window,
            awcd_constraint_threshold=awcd_constraint_threshold,
            awcd_mean_threshold=awcd_mean_threshold,
            awcd_warmup=awcd_warmup,
            awcd_pool_size=awcd_pool_size,
            awcd_top_frac=awcd_top_frac,
            early_prior_boost=early_prior_boost,
            early_prior_steps=early_prior_steps,
            weight_mode=weight_mode,
            method_tag=method_tag,
        )
        dfr["seed"] = s
        dfs.append(dfr)
    return pd.concat(dfs, ignore_index=True)


def _run_ensemble_continuous_single(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int,
    manual_readout: Optional[Dict[str, Any]],
    init_method: str,
    num_restarts: int,
    raw_samples: int,
    prior_strength: float,
    constraint_hardness: float,
    constraint_pool_size: int,
    awcd_window: int,
    awcd_constraint_threshold: float,
    awcd_mean_threshold: float,
    awcd_warmup: int,
    awcd_pool_size: int,
    awcd_top_frac: float,
    early_prior_boost: bool,
    early_prior_steps: int,
    weight_mode: str,
    method_tag: str,
) -> pd.DataFrame:
    weight_mode = str(weight_mode).lower()
    if weight_mode not in {"ensemble", "skeptic", "believer"}:
        raise ValueError("weight_mode must be one of {'ensemble', 'skeptic', 'believer'}.")
    X_init = _sample_initial_unit(domain, n_init, seed, init_method)
    Y_init = evaluate_oracle(domain, X_init).unsqueeze(-1)

    X_obs = X_init.clone()
    Y_obs = Y_init.clone()
    recs: List[Dict[str, Any]] = []
    best = float(Y_obs.max().item()) if Y_obs.numel() else float("-inf")

    for i in range(n_init):
        y = float(Y_init[i].item())
        best = max(best, y)
        recs.append(
            _record_continuous_sample(
                domain,
                X_init[i],
                y,
                best,
                method=method_tag,
                iteration=i - n_init,
                extra={
                    "weight_believer": float("nan"),
                    "awcd_score": float("nan"),
                    "awcd_mean": float("nan"),
                    "awcd_constraint": float("nan"),
                    "awcd_constraint_mean": float("nan"),
                    "awcd_mean_disagree": float("nan"),
                    "awcd_mean_disagree_mean": float("nan"),
                    "awcd_trend": float("nan"),
                    "prior_active": True,
                },
            )
        )

    ro_raw = manual_readout if manual_readout is not None else flat_readout(feature_names=domain.feature_names)
    ro_unit = normalize_readout_to_unit_box(ro_raw, domain.mins, domain.maxs, feature_names=domain.feature_names)
    prior = readout_to_prior(ro_unit, feature_names=domain.feature_names)
    prior_active = True
    awcd_history: List[float] = []
    awcd_history_constraint: List[float] = []
    awcd_history_mean: List[float] = []

    for t in range(n_iter):
        gp_skeptic = SingleTaskGP(X_obs, Y_obs)
        mll_s = ExactMarginalLogLikelihood(gp_skeptic.likelihood, gp_skeptic)
        try:
            with cholesky_jitter(1e-4):
                fit_gpytorch_mll(mll_s, max_attempts=5)
        except ModelFittingError:
            # Fallback for ill-conditioned fits: add tiny noise + standardize.
            Y_jittered = Y_obs + 1e-6 * torch.randn_like(Y_obs)
            gp_skeptic = SingleTaskGP(
                X_obs,
                Y_jittered,
                input_transform=Normalize(d=X_obs.shape[-1]),
                outcome_transform=Standardize(m=1),
            )
            mll_s = ExactMarginalLogLikelihood(gp_skeptic.likelihood, gp_skeptic)
            with cholesky_jitter(1e-3):
                fit_gpytorch_mll(mll_s, max_attempts=5)

        gp_resid, alpha = fit_residual_gp(X_obs, Y_obs, prior)
        m0 = prior.m0_torch(X_obs).reshape(-1)
        m0_scale = float(alpha * prior_strength)
        model_believer = GPWithPriorMean(gp_resid, prior, m0_scale=m0_scale)
        awcd_metrics = _awcd_full_score(
            gp_skeptic,
            prior.m0_torch,
            X_obs=X_obs,
            Y_obs=Y_obs,
            readout_unit=ro_unit,
            feature_names=domain.feature_names,
            pool_n=awcd_pool_size,
            top_frac=awcd_top_frac,
            seed=seed + 9000 + t,
            bounds=domain.unit_bounds,
        )
        awcd_score = float(awcd_metrics.get("awcd_total", 0.0))
        awcd_constraint = float(awcd_metrics.get("awcd_constraint", 0.0))
        awcd_mean = float(awcd_metrics.get("awcd_mean", 0.0))
        awcd_history.append(awcd_score)
        awcd_history_constraint.append(awcd_constraint)
        awcd_history_mean.append(awcd_mean)
        awcd_score_mean = float(np.mean(awcd_history[-int(awcd_window):])) if awcd_history else 0.0
        awcd_constraint_mean = float(
            np.mean(awcd_history_constraint[-int(awcd_window):])
        ) if awcd_history_constraint else 0.0
        awcd_mean_mean = float(np.mean(awcd_history_mean[-int(awcd_window):])) if awcd_history_mean else 0.0
        awcd_trend = 0.0
        if len(awcd_history) >= 2:
            awcd_trend = float(awcd_history[-1] - awcd_history[-2])

        if weight_mode == "skeptic":
            weight = 0.0
            prior_active = False
        elif weight_mode == "believer":
            weight = 1.0
            prior_active = True
        else:
            if t < int(awcd_warmup) or len(awcd_history) < int(awcd_window):
                prior_good = True
            else:
                prior_good = not (
                    (awcd_constraint_mean > float(awcd_constraint_threshold))
                    or (awcd_mean_mean > float(awcd_mean_threshold))
                )
            weight = 1.0 if prior_good else 0.0
            prior_active = bool(prior_good)
        best_f = float(Y_obs.max().item())
        effective_hardness = float(constraint_hardness) if weight > 0.0 else 0.0
        if bool(early_prior_boost) and t < int(early_prior_steps) and weight > 0.0:
            pool = draw_sobol_samples(
                bounds=domain.unit_bounds,
                n=1024,
                q=1,
                seed=seed + 505 + t,
            ).squeeze(1)
            with torch.no_grad():
                prior_vals = prior.m0_torch(pool).reshape(-1)
                idx_local = int(torch.argmax(prior_vals))
                x_next = pool[idx_local]
        else:
            acq = WeightedExpectedImprovement(
                gp_skeptic,
                model_believer,
                weight=weight,
                best_f=best_f,
                readout_unit=ro_unit,
                feature_names=domain.feature_names,
                constraint_hardness=effective_hardness,
            )
            if effective_hardness > 0.0:
                pool_n = max(1024, int(constraint_pool_size))
                pool = draw_sobol_samples(
                    bounds=domain.unit_bounds,
                    n=pool_n,
                    q=1,
                    seed=seed + 202 + t,
                ).squeeze(1)
                with torch.no_grad():
                    scores = acq(pool.unsqueeze(1)).reshape(-1)
                    idx_local = int(torch.argmax(scores))
                    x_next = pool[idx_local]
            else:
                x_next, _ = optimize_acqf(
                    acq,
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
                method=method_tag,
                iteration=t,
                extra={
                    "weight_believer": float(weight),
                    "awcd_score": float(awcd_score),
                    "awcd_mean": float(awcd_score_mean),
                    "awcd_constraint": float(awcd_constraint),
                    "awcd_constraint_mean": float(awcd_constraint_mean),
                    "awcd_mean_disagree": float(awcd_mean),
                    "awcd_mean_disagree_mean": float(awcd_mean_mean),
                    "awcd_trend": float(awcd_trend),
                    "prior_active": bool(prior_active),
                },
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
    disable_constraints_when_guarded: bool = False,
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
        disable_constraints_when_guarded=disable_constraints_when_guarded,
        early_prior_boost=early_prior_boost,
        early_prior_steps=early_prior_steps,
    )
    dfs.append(hyb)
    return pd.concat(dfs, ignore_index=True)


def run_data_prior_benchmark(
    domain: ContinuousDomain,
    *,
    csv_path: str,
    fraction: float = 0.10,
    data_seed: int = 0,
    feature_names: Optional[List[str]] = None,
    focus_features: Optional[List[str]] = None,
    target_col: str = "yield",
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 800,
    api_key: Optional[str] = None,
    n_init: int = 6,
    n_init_baseline: Optional[int] = None,
    n_init_hybrid: Optional[int] = None,
    n_init_random: Optional[int] = None,
    n_iter: int = 25,
    seed: int = 0,
    repeats: int = 5,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    use_alignment_guard: bool = False,
    alignment_min: float = 0.0,
    constraint_hardness: float = 0.0,
    constraint_pool_size: int = 4096,
    disable_constraints_when_guarded: bool = False,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
    include_random: bool = True,
    include_baseline: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    feat_names = feature_names or domain.feature_names
    readout = get_data_derived_prior(
        csv_path=csv_path,
        fraction=fraction,
        seed=data_seed,
        feature_names=feat_names,
        focus_features=focus_features,
        target_col=target_col,
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        api_key=api_key,
    )
    base_init = int(n_init_baseline) if n_init_baseline is not None else int(n_init)
    hyb_init = int(n_init_hybrid) if n_init_hybrid is not None else int(n_init)
    rand_init = int(n_init_random) if n_init_random is not None else int(base_init)

    dfs: List[pd.DataFrame] = []
    if include_random:
        rand = run_random_continuous(
            domain,
            n_init=rand_init,
            n_iter=n_iter,
            seed=seed,
            repeats=repeats,
        )
        dfs.append(rand)

    if include_baseline:
        base = run_baseline_ei_continuous(
            domain,
            n_init=base_init,
            n_iter=n_iter,
            seed=seed,
            repeats=repeats,
        )
        dfs.append(base)

    hyb = run_hybrid_continuous(
        domain,
        n_init=hyb_init,
        n_iter=n_iter,
        seed=seed,
        repeats=repeats,
        manual_readout=readout,
        prior_strength=prior_strength,
        rho_floor=rho_floor,
        use_alignment_guard=use_alignment_guard,
        alignment_min=alignment_min,
        constraint_hardness=constraint_hardness,
        constraint_pool_size=constraint_pool_size,
        disable_constraints_when_guarded=disable_constraints_when_guarded,
        early_prior_boost=early_prior_boost,
        early_prior_steps=early_prior_steps,
    )
    dfs.append(hyb)

    hist = pd.concat(dfs, ignore_index=True)
    return hist, readout


def portion_benchmark(
    domain: ContinuousDomain,
    *,
    csv_path: str,
    fractions: List[float],
    data_seed: int = 0,
    feature_names: Optional[List[str]] = None,
    focus_features: Optional[List[str]] = None,
    target_col: str = "yield",
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 800,
    api_key: Optional[str] = None,
    n_init: int = 6,
    n_iter: int = 20,
    seed: int = 0,
    repeats: int = 5,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    use_alignment_guard: bool = False,
    alignment_min: float = 0.0,
    constraint_hardness: float = 0.0,
    constraint_pool_size: int = 4096,
    disable_constraints_when_guarded: bool = False,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
    num_restarts: int = 10,
    raw_samples: int = 256,
    auc_iters: Optional[int] = 20,
    include_init: bool = False,
    include_baseline: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not fractions:
        raise ValueError("fractions must be a non-empty list.")
    for frac in fractions:
        if frac <= 0.0 or frac > 1.0:
            raise ValueError("fractions must be in (0, 1].")

    feat_names = feature_names or domain.feature_names
    total_runs = int(len(fractions) * repeats)
    done = 0

    all_runs: List[pd.DataFrame] = []
    auc_rows: List[Dict[str, Any]] = []

    for i, frac in enumerate(fractions):
        frac_seed = int(data_seed + i * 97)
        print(f"[portion_benchmark] Fraction {i + 1}/{len(fractions)}: {frac:.3f}")
        print("  Generating LLM readout...")
        readout = get_data_derived_prior(
            csv_path=csv_path,
            fraction=frac,
            seed=frac_seed,
            feature_names=feat_names,
            focus_features=focus_features,
            target_col=target_col,
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            api_key=api_key,
        )

        for r in range(repeats):
            done += 1
            run_seed = int(seed + i * 1000 + r)
            print(
                f"  Run {done}/{total_runs} "
                f"(fraction={frac:.3f}, repeat={r + 1}/{repeats})"
            )
            df_run = _run_hybrid_continuous_single(
                domain,
                n_init=n_init,
                n_iter=n_iter,
                seed=run_seed,
                manual_readout=readout,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                prior_strength=prior_strength,
                rho_floor=rho_floor,
                use_alignment_guard=use_alignment_guard,
                alignment_min=alignment_min,
                constraint_hardness=constraint_hardness,
                constraint_pool_size=constraint_pool_size,
                disable_constraints_when_guarded=disable_constraints_when_guarded,
                early_prior_boost=early_prior_boost,
                early_prior_steps=early_prior_steps,
            )
            df_run["seed"] = run_seed
            df_run["fraction"] = float(frac)
            df_run["data_seed"] = frac_seed
            all_runs.append(df_run)

            auc_val = _auc_best_so_far_window(df_run, include_init=include_init, n_iters=auc_iters)
            auc_rows.append(
                {
                    "fraction": float(frac),
                    "seed": run_seed,
                    "auc": float(auc_val),
                }
            )

    hist_df = pd.concat(all_runs, ignore_index=True)
    auc_df = pd.DataFrame(auc_rows)
    summary = (
        auc_df.groupby("fraction")["auc"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
        .rename(columns={"median": "auc_median", "mean": "auc_mean", "std": "auc_std", "count": "n_runs"})
    )
    baseline_summary: Optional[pd.DataFrame] = None
    baseline_hist: Optional[pd.DataFrame] = None
    if include_baseline:
        print("[portion_benchmark] Running baseline BO (reference)...")
        baseline_hist = run_baseline_ei_continuous(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            repeats=repeats,
        )
        base_aucs = []
        for s in sorted(baseline_hist["seed"].unique()):
            df_seed = baseline_hist[baseline_hist["seed"] == s]
            auc_val = _auc_best_so_far_window(df_seed, include_init=include_init, n_iters=auc_iters)
            base_aucs.append(float(auc_val))
        if base_aucs:
            baseline_summary = pd.DataFrame(
                [
                    {
                        "method": "baseline_ei",
                        "auc_median": float(np.median(base_aucs)),
                        "auc_mean": float(np.mean(base_aucs)),
                        "auc_std": float(np.std(base_aucs, ddof=0)),
                        "n_runs": int(len(base_aucs)),
                    }
                ]
            )
    return hist_df, summary, baseline_summary, baseline_hist


def _safe_filename(text: str) -> str:
    return text.strip().replace("/", "_").replace(" ", "_")


def portion_benchmark_llms(
    domain: ContinuousDomain,
    *,
    csv_path: str,
    fractions: List[float],
    llm_models: List[str],
    data_seed: int = 0,
    feature_names: Optional[List[str]] = None,
    focus_features: Optional[List[str]] = None,
    target_col: str = "yield",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 800,
    api_key: Optional[str] = None,
    n_init: int = 6,
    n_iter: int = 20,
    seed: int = 0,
    repeats: int = 5,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    use_alignment_guard: bool = False,
    alignment_min: float = 0.0,
    constraint_hardness: float = 0.0,
    constraint_pool_size: int = 4096,
    disable_constraints_when_guarded: bool = False,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
    num_restarts: int = 10,
    raw_samples: int = 256,
    auc_iters: Optional[int] = 20,
    include_init: bool = False,
    include_baseline: bool = True,
    output_dir: str = "llm_study_data",
    save_csv: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not llm_models:
        raise ValueError("llm_models must be a non-empty list.")
    os.makedirs(output_dir, exist_ok=True)

    baseline_summary: Optional[pd.DataFrame] = None
    baseline_hist: Optional[pd.DataFrame] = None
    if include_baseline:
        print("[portion_benchmark_llms] Running baseline BO (reference)...")
        baseline_hist = run_baseline_ei_continuous(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            repeats=repeats,
        )
        base_aucs = []
        for s in sorted(baseline_hist["seed"].unique()):
            df_seed = baseline_hist[baseline_hist["seed"] == s]
            auc_val = _auc_best_so_far_window(df_seed, include_init=include_init, n_iters=auc_iters)
            base_aucs.append(float(auc_val))
        if base_aucs:
            baseline_summary = pd.DataFrame(
                [
                    {
                        "method": "baseline_ei",
                        "auc_median": float(np.median(base_aucs)),
                        "auc_mean": float(np.mean(base_aucs)),
                        "auc_std": float(np.std(base_aucs, ddof=0)),
                        "n_runs": int(len(base_aucs)),
                    }
                ]
            )
        if save_csv:
            baseline_hist.to_csv(os.path.join(output_dir, "baseline_hist.csv"), index=False)
            if baseline_summary is not None:
                baseline_summary.to_csv(os.path.join(output_dir, "baseline_summary.csv"), index=False)

    all_hist: List[pd.DataFrame] = []
    all_auc: List[pd.DataFrame] = []
    all_summary: List[pd.DataFrame] = []

    for model in llm_models:
        print(f"[portion_benchmark_llms] Model: {model}")
        hist_df, _, _, _ = portion_benchmark(
            domain,
            csv_path=csv_path,
            fractions=fractions,
            data_seed=data_seed,
            feature_names=feature_names,
            focus_features=focus_features,
            target_col=target_col,
            llm_model=model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            api_key=api_key,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            repeats=repeats,
            prior_strength=prior_strength,
            rho_floor=rho_floor,
            use_alignment_guard=use_alignment_guard,
            alignment_min=alignment_min,
            constraint_hardness=constraint_hardness,
            constraint_pool_size=constraint_pool_size,
            disable_constraints_when_guarded=disable_constraints_when_guarded,
            early_prior_boost=early_prior_boost,
            early_prior_steps=early_prior_steps,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            auc_iters=auc_iters,
            include_init=include_init,
            include_baseline=False,
        )
        hist_df = hist_df.copy()
        hist_df["llm_model"] = model
        hist_df["llm_temperature"] = float(llm_temperature)
        hist_df["llm_max_tokens"] = int(llm_max_tokens)
        all_hist.append(hist_df)

        auc_rows: List[Dict[str, Any]] = []
        for frac in sorted(hist_df["fraction"].unique()):
            df_frac = hist_df[hist_df["fraction"] == frac]
            for s in sorted(df_frac["seed"].unique()):
                df_seed = df_frac[df_frac["seed"] == s]
                auc_val = _auc_best_so_far_window(df_seed, include_init=include_init, n_iters=auc_iters)
                auc_rows.append(
                    {
                        "llm_model": model,
                        "fraction": float(frac),
                        "seed": int(s),
                        "auc": float(auc_val),
                    }
                )
        auc_df = pd.DataFrame(auc_rows)
        summary_df = (
            auc_df.groupby(["llm_model", "fraction"])["auc"]
            .agg(["median", "mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "median": "auc_median",
                    "mean": "auc_mean",
                    "std": "auc_std",
                    "count": "n_runs",
                }
            )
        )
        all_auc.append(auc_df)
        all_summary.append(summary_df)

        if save_csv:
            safe = _safe_filename(model)
            auc_df.to_csv(os.path.join(output_dir, f"auc_{safe}.csv"), index=False)
            summary_df.to_csv(os.path.join(output_dir, f"summary_{safe}.csv"), index=False)
            hist_df.to_csv(os.path.join(output_dir, f"hist_{safe}.csv"), index=False)

    all_hist_df = pd.concat(all_hist, ignore_index=True)
    all_auc_df = pd.concat(all_auc, ignore_index=True)
    all_summary_df = pd.concat(all_summary, ignore_index=True)

    if save_csv:
        all_hist_df.to_csv(os.path.join(output_dir, "all_hist.csv"), index=False)
        all_auc_df.to_csv(os.path.join(output_dir, "all_auc.csv"), index=False)
        all_summary_df.to_csv(os.path.join(output_dir, "all_summary.csv"), index=False)

    return all_hist_df, all_summary_df, all_auc_df, baseline_summary, baseline_hist


def init_benchmark(
    domain: ContinuousDomain,
    *,
    csv_path: str,
    fraction: float,
    n_inits: List[int],
    data_seed: int = 0,
    feature_names: Optional[List[str]] = None,
    focus_features: Optional[List[str]] = None,
    target_col: str = "yield",
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 800,
    api_key: Optional[str] = None,
    n_iter: int = 20,
    seed: int = 0,
    repeats: int = 5,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    use_alignment_guard: bool = False,
    alignment_min: float = 0.0,
    constraint_hardness: float = 0.0,
    constraint_pool_size: int = 4096,
    disable_constraints_when_guarded: bool = False,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
    num_restarts: int = 10,
    raw_samples: int = 256,
    auc_iters: Optional[int] = 20,
    include_init: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not n_inits:
        raise ValueError("n_inits must be a non-empty list.")
    for n_init in n_inits:
        if int(n_init) <= 0:
            raise ValueError("n_inits must be positive integers.")

    feat_names = feature_names or domain.feature_names
    print("[init_benchmark] Generating single LLM readout...")
    readout = get_data_derived_prior(
        csv_path=csv_path,
        fraction=fraction,
        seed=data_seed,
        feature_names=feat_names,
        focus_features=focus_features,
        target_col=target_col,
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        api_key=api_key,
    )

    total_runs = int(len(n_inits) * repeats * 2)
    done = 0
    records: List[Dict[str, Any]] = []
    all_runs: List[pd.DataFrame] = []

    for i, n_init in enumerate(sorted(n_inits)):
        print(f"[init_benchmark] n_init {i + 1}/{len(n_inits)}: {n_init}")

        for r in range(repeats):
            done += 1
            run_seed = int(seed + i * 1000 + r)
            print(
                f"  Hybrid run {done}/{total_runs} "
                f"(n_init={n_init}, repeat={r + 1}/{repeats})"
            )
            df_run = _run_hybrid_continuous_single(
                domain,
                n_init=int(n_init),
                n_iter=n_iter,
                seed=run_seed,
                manual_readout=readout,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                prior_strength=prior_strength,
                rho_floor=rho_floor,
                use_alignment_guard=use_alignment_guard,
                alignment_min=alignment_min,
                constraint_hardness=constraint_hardness,
                constraint_pool_size=constraint_pool_size,
                disable_constraints_when_guarded=disable_constraints_when_guarded,
                early_prior_boost=early_prior_boost,
                early_prior_steps=early_prior_steps,
            )
            df_run["seed"] = run_seed
            df_run["n_init"] = int(n_init)
            df_run["method"] = "hybrid_manual"
            all_runs.append(df_run)
            auc_val = _auc_best_so_far_window(df_run, include_init=include_init, n_iters=auc_iters)
            records.append(
                {"n_init": int(n_init), "method": "hybrid_manual", "seed": run_seed, "auc": float(auc_val)}
            )

        print(f"  Baseline BO for n_init={n_init}")
        base_hist = run_baseline_ei_continuous(
            domain,
            n_init=int(n_init),
            n_iter=n_iter,
            seed=int(seed + i * 1000),
            repeats=repeats,
        )
        base_hist["n_init"] = int(n_init)
        all_runs.append(base_hist)
        for s in sorted(base_hist["seed"].unique()):
            df_seed = base_hist[base_hist["seed"] == s]
            auc_val = _auc_best_so_far_window(df_seed, include_init=include_init, n_iters=auc_iters)
            records.append(
                {"n_init": int(n_init), "method": "baseline_ei", "seed": int(s), "auc": float(auc_val)}
            )
        done += repeats

    hist_df = pd.concat(all_runs, ignore_index=True)
    auc_df = pd.DataFrame(records)
    summary = (
        auc_df.groupby(["n_init", "method"])["auc"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
        .rename(columns={"median": "auc_median", "mean": "auc_mean", "std": "auc_std", "count": "n_runs"})
    )
    return hist_df, summary


def iter_success_bench(
    domain: ContinuousDomain,
    *,
    iter_list: List[int],
    manual_readout: Dict[str, Any],
    n_init: int = 6,
    seed: int = 0,
    repeats: int = 5,
    target_frac: float = 0.9,
    optimum: Optional[float] = None,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    use_alignment_guard: bool = False,
    alignment_min: float = 0.0,
    constraint_hardness: float = 0.0,
    constraint_pool_size: int = 4096,
    disable_constraints_when_guarded: bool = False,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
    num_restarts: int = 10,
    raw_samples: int = 256,
    include_init: bool = False,
) -> pd.DataFrame:
    if not iter_list:
        raise ValueError("iter_list must be a non-empty list.")
    for n_iter in iter_list:
        if int(n_iter) <= 0:
            raise ValueError("iter_list must contain positive integers.")
    if optimum is None:
        optimum = estimate_optimum(domain, n_samples=4096, seed=0)
    target = float(optimum) * float(target_frac)

    rows: List[Dict[str, Any]] = []
    total_runs = int(len(iter_list) * repeats * 2)
    done = 0

    for i, n_iter in enumerate(sorted(iter_list)):
        print(f"[iter_success_bench] Iter cap {i + 1}/{len(iter_list)}: {n_iter}")

        hyb = run_hybrid_continuous(
            domain,
            n_init=n_init,
            n_iter=int(n_iter),
            seed=int(seed + i * 1000),
            repeats=repeats,
            manual_readout=manual_readout,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            prior_strength=prior_strength,
            rho_floor=rho_floor,
            use_alignment_guard=use_alignment_guard,
            alignment_min=alignment_min,
            constraint_hardness=constraint_hardness,
            constraint_pool_size=constraint_pool_size,
            disable_constraints_when_guarded=disable_constraints_when_guarded,
            early_prior_boost=early_prior_boost,
            early_prior_steps=early_prior_steps,
        )
        done += repeats
        hyb_success = []
        for s in sorted(hyb["seed"].unique()):
            df_seed = hyb[hyb["seed"] == s]
            max_best = float(df_seed.loc[df_seed["iter"] >= 0, "best_so_far"].max())
            hyb_success.append(max_best >= target)
        rows.append(
            {
                "n_iter": int(n_iter),
                "method": "hybrid_manual",
                "success_rate": float(np.mean(hyb_success)) if hyb_success else 0.0,
                "successes": int(np.sum(hyb_success)),
                "n_runs": int(len(hyb_success)),
            }
        )

        base = run_baseline_ei_continuous(
            domain,
            n_init=n_init,
            n_iter=int(n_iter),
            seed=int(seed + i * 1000),
            repeats=repeats,
        )
        done += repeats
        base_success = []
        for s in sorted(base["seed"].unique()):
            df_seed = base[base["seed"] == s]
            max_best = float(df_seed.loc[df_seed["iter"] >= 0, "best_so_far"].max())
            base_success.append(max_best >= target)
        rows.append(
            {
                "n_iter": int(n_iter),
                "method": "baseline_ei",
                "success_rate": float(np.mean(base_success)) if base_success else 0.0,
                "successes": int(np.sum(base_success)),
                "n_runs": int(len(base_success)),
            }
        )
        print(f"  Progress: {done}/{total_runs} runs")

    return pd.DataFrame(rows)


def plot_runs_mean_lookup(
    hist_df: pd.DataFrame,
    *,
    methods: Optional[List[str]] = None,
    ci: str = "sd",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_auc_text: bool = False,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
    else:
        fig = ax.figure

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
        x = agg["iter"].to_numpy() + 1
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        color = METHOD_COLORS.get(m)
        ax.plot(x, y, label=_method_label(m), color=color)
        ax.fill_between(x, y - e, y + e, alpha=0.20, color=color)

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Yield Best-So-Far", fontsize=AXIS_LABEL_SIZE)

    ax.grid(False)
    
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right")
    auc_text = None
    if show_auc_text:
        auc_lines = []
        prior_used = None
        for m in methods:
            d = df[df["method"] == m]
            if d.empty:
                continue
            per_seed = []
            for seed_val, df_seed in d.groupby("seed"):
                per_seed.append(_auc_best_so_far_window(df_seed, include_init=False, n_iters=None))
            if per_seed:
                auc_mean = float(np.mean(per_seed))
                line = f"{_method_label(m)} AUC: {auc_mean:.3f}"
                if m == "ensemble" and "prior_active" in d.columns:
                    prior_used = float(d["prior_active"].astype(bool).mean())
                    line = f"{line} (prior used {prior_used * 100:.1f}%)"
                auc_lines.append((m, line))
        if auc_lines:
            auc_text = " | ".join(line for _, line in auc_lines)
    max_iter = int(df["iter"].max()) if not df.empty else -1
    _set_iteration_ticks(ax, max_iter)
    if auc_text:
        fig.text(
            0.5,
            -0.02,
            auc_text,
            ha="center",
            va="top",
            fontsize=10,
        )
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    else:
        plt.tight_layout()
    return ax


def plot_raw_yield_scatter(
    hist_df: pd.DataFrame,
    *,
    methods: Optional[List[str]] = None,
    ci: str = "sem",
    include_init: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    df = hist_df.copy()
    if not include_init:
        df = df[df["iter"] >= 0]
    if methods is None:
        methods = list(df["method"].unique())

    for m in methods:
        d = df[df["method"] == m]
        if d.empty:
            continue
        agg = d.groupby("iter")["y"].agg(["mean", "std", "count"]).reset_index()
        if ci == "sem":
            err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        elif ci == "95ci":
            err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        else:
            err = agg["std"]
        color = METHOD_COLORS.get(m)
        ax.errorbar(
            agg["iter"].to_numpy() + 1,
            agg["mean"].to_numpy(),
            yerr=err.to_numpy(),
            fmt="o",
            markersize=4,
            capsize=3,
            label=_method_label(m),
            color=color,
        )

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Yield", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    max_iter = int(df["iter"].max()) if not df.empty else -1
    _set_iteration_ticks(ax, max_iter)
    plt.tight_layout()
    return ax


def acc_plot(
    hist_df: pd.DataFrame,
    domain: ContinuousDomain,
    *,
    target_frac: float = 0.9,
    include_init: bool = False,
    optimum: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 4.8))

    if optimum is None:
        optimum = estimate_optimum(domain, n_samples=4096, seed=0)
    target = float(optimum) * float(target_frac)

    rows: List[Dict[str, Any]] = []
    for (method, seed), df_run in hist_df.groupby(["method", "seed"]):
        iters = _iters_to_target(df_run, target=target, include_init=include_init)
        rows.append({"method": method, "seed": int(seed), "iters": int(iters)})

    per_run = pd.DataFrame(rows)
    if per_run.empty:
        raise ValueError("No runs found to compute acceleration plot.")

    agg = per_run.groupby("method")["iters"].agg(["mean", "std", "count"]).reset_index()
    order = ["random", "baseline_ei", "hybrid_manual"]
    agg = agg.set_index("method").reindex(order).reset_index()
    label_map = {
        "random": "Rand",
        "baseline_ei": "BO",
        "hybrid_manual": "PSBO",
    }
    x_labels = [label_map.get(m, _method_label(m)) for m in agg["method"].tolist()]
    errs = agg["std"].to_numpy()
    colors = [METHOD_COLORS["random"], METHOD_COLORS["baseline_ei"], METHOD_COLORS["hybrid_manual"]]
    hatches = ["///", "///", "///"]
    positions = np.arange(len(x_labels)) * 1.5
    bars = ax.bar(
        positions,
        agg["mean"].to_numpy(),
        yerr=errs,
        capsize=4,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(f"Iterations to {int(target_frac * 100)}% optimum", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Method", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)
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
    x = agg["iter"].to_numpy() + 1
    y = agg["mean"].to_numpy()
    e = err.to_numpy()

    ax.plot(x, y, label="alignment (rho)")
    ax.fill_between(x, y - e, y + e, alpha=0.20)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Alignment (rho)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)

    if "guarded" in df.columns:
        guard_rate = df.groupby("iter")["guarded"].mean().reset_index()
        ax2 = ax.twinx()
        ax2.plot(guard_rate["iter"].to_numpy() + 1, guard_rate["guarded"], color="#c44e52", label="guard rate")
        ax2.set_ylabel("Guard rate")

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)
    max_iter = int(df["iter"].max()) if not df.empty else -1
    _set_iteration_ticks(ax, max_iter)
    plt.tight_layout()
    return ax


def plot_alignment_over_time_multi(
    hist_df: pd.DataFrame,
    *,
    methods: List[str],
    ci: str = "sd",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    df = hist_df.copy()
    if "rho" not in df.columns:
        raise ValueError("alignment data not found (missing 'rho' column).")

    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(len(methods), 1)))
    max_iter = -1

    for color, method in zip(colors, methods):
        d = df[(df["method"] == method) & (df["iter"] >= 0)].copy()
        if d.empty:
            continue
        agg = d.groupby("iter")["rho"].agg(["mean", "std", "count"]).reset_index()
        if ci == "sem":
            err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        elif ci == "95ci":
            err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        else:
            err = agg["std"]
        x = agg["iter"].to_numpy() + 1
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        ax.plot(x, y, label=_method_label(method), color=color)
        ax.fill_between(x, y - e, y + e, alpha=0.18, color=color)
        max_iter = max(max_iter, int(agg["iter"].max()))

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Alignment (rho)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)
    _set_iteration_ticks(ax, max_iter)
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
        x = agg["iter"].to_numpy() + 1
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        color = METHOD_COLORS.get(m)
        ax.plot(x, y, label=_method_label(m), color=color)
        ax.fill_between(x, y - e, y + e, alpha=0.20, color=color)

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Simple regret (optimum - best)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    max_iter = int(df["iter"].max()) if not df.empty else -1
    _set_iteration_ticks(ax, max_iter)
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

    ax.set_xlabel(feature, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    return ax


def plot_parameter_trajectories(
    hist_df: pd.DataFrame,
    *,
    params: List[str],
    readout: Optional[Dict[str, Any]] = None,
    methods: Optional[List[str]] = None,
    ci: str = "sem",
    include_init: bool = False,
    feature_names: Optional[List[str]] = None,
    tick_step: Optional[int] = None,
) -> plt.Figure:
    df = hist_df.copy()
    if not include_init:
        df = df[df["iter"] >= 0]
    if methods is None:
        methods = ["baseline_ei", "hybrid_manual"]

    fig, axes = plt.subplots(len(params), 1, figsize=(7.5, 3.0 * len(params)), sharex=True)
    if len(params) == 1:
        axes = [axes]

    effects = (readout or {}).get("effects") or {}
    constraints = (readout or {}).get("constraints") or []
    color_map = METHOD_COLORS

    def _label_for_param(param: str) -> str:
        if feature_names and param.lower().startswith("x"):
            try:
                idx = int(param[1:]) - 1
            except ValueError:
                return param
            if 0 <= idx < len(feature_names):
                return str(feature_names[idx])
        return param

    for ax, param in zip(axes, params):
        spec = effects.get(param, {})
        rh = spec.get("range_hint")
        if isinstance(rh, (list, tuple)) and len(rh) == 2:
            lo, hi = float(rh[0]), float(rh[1])
            if hi < lo:
                lo, hi = hi, lo
            ax.axhspan(lo, hi, color="#4c72b0", alpha=0.12, label="range_hint")

        for c in constraints:
            if not isinstance(c, dict):
                continue
            if str(c.get("var")) != str(param):
                continue
            r = c.get("range")
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                continue
            lo, hi = float(r[0]), float(r[1])
            if hi < lo:
                lo, hi = hi, lo
            ax.axhspan(lo, hi, color="#c44e52", alpha=0.18, label="constraint")

        for method in methods:
            d = df[df["method"] == method]
            if d.empty or param not in d.columns:
                continue
            x = d["iter"].to_numpy() + 1
            y = d[param].to_numpy()
            color = color_map.get(method, "#4c72b0")
            marker = "o" if method == "baseline_ei" else "s"
            ax.scatter(
                x,
                y,
                label=_method_label(method),
                color=color,
                s=22,
                alpha=0.7,
                marker=marker,
                edgecolors="none",
            )

        ax.set_ylabel(_label_for_param(param), fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    max_iter = int(df["iter"].max()) if not df.empty else -1
    _set_iteration_ticks(axes[-1], max_iter)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=LEGEND_FONT_SIZE, frameon=False)
    plt.tight_layout()
    return fig


def plot_prior_surface_heatmaps_ugi(
    *,
    csv_path: str = "ugi_merged_dataset.csv",
    target_col: str = "yield",
    novice_fraction: float = 0.01,
    mid_fraction: float = 0.20,
    expert_fraction: float = 0.90,
    novice_seed: int = 0,
    mid_seed: Optional[int] = None,
    expert_seed: Optional[int] = None,
    focus_features: Optional[List[str]] = None,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    api_key: Optional[str] = None,
    n_grid: int = 80,
    n_opt_samples: int = 2048,
    seed: int = 0,
    print_readouts: bool = True,
) -> plt.Figure:
    """
    Figure 6 Panel B: Prior surface heatmaps (novice vs expert) vs ground truth.

    Uses the data-to-prior prompt pipeline to mirror the data-portion workflow.
    Readouts are generated from the sampled data fractions.
    """
    domain = build_continuous_domain(target=target_col)

    if mid_seed is None:
        if abs(novice_fraction - mid_fraction) < 1e-9:
            mid_seed = novice_seed
        else:
            mid_seed = novice_seed + 1
    if expert_seed is None:
        if abs(expert_fraction - novice_fraction) < 1e-9:
            expert_seed = novice_seed
        elif abs(expert_fraction - mid_fraction) < 1e-9:
            expert_seed = mid_seed
        else:
            expert_seed = novice_seed + 2

    d = int(len(domain.feature_names))
    if d < 4:
        raise ValueError("UGI prior surface expects at least 4 dimensions (x1..x4).")

    pool = draw_sobol_samples(bounds=domain.unit_bounds, n=n_opt_samples, q=1, seed=seed).squeeze(1)
    y_pool = evaluate_oracle(domain, pool).reshape(-1)
    best_idx = int(torch.argmax(y_pool))
    best_unit = pool[best_idx]

    x2_fixed_unit = float(best_unit[1].item()) if d > 1 else 0.5
    x3_fixed_unit = float(best_unit[2].item()) if d > 2 else 0.5

    mins = domain.mins.detach().cpu().numpy()
    maxs = domain.maxs.detach().cpu().numpy()
    span = maxs - mins

    x1_vals = np.linspace(float(mins[0]), float(maxs[0]), n_grid)
    x4_vals = np.linspace(float(mins[3]), float(maxs[3]), n_grid)
    X1, X4 = np.meshgrid(x1_vals, x4_vals)

    x1_unit = (X1 - mins[0]) / max(span[0], 1e-12)
    x4_unit = (X4 - mins[3]) / max(span[3], 1e-12)

    grid_unit = np.zeros((n_grid * n_grid, d), dtype=np.float64)
    grid_unit[:, 0] = x1_unit.ravel()
    grid_unit[:, 3] = x4_unit.ravel()
    if d > 1:
        grid_unit[:, 1] = x2_fixed_unit
    if d > 2:
        grid_unit[:, 2] = x3_fixed_unit

    novice_readout = get_data_derived_prior(
        csv_path=csv_path,
        fraction=novice_fraction,
        seed=novice_seed,
        feature_names=domain.feature_names,
        focus_features=focus_features,
        target_col=target_col,
        model=llm_model,
        temperature=llm_temperature,
        api_key=api_key,
    )
    expert_readout = get_data_derived_prior(
        csv_path=csv_path,
        fraction=expert_fraction,
        seed=expert_seed,
        feature_names=domain.feature_names,
        focus_features=focus_features,
        target_col=target_col,
        model=llm_model,
        temperature=llm_temperature,
        api_key=api_key,
    )
    mid_readout = get_data_derived_prior(
        csv_path=csv_path,
        fraction=mid_fraction,
        seed=mid_seed,
        feature_names=domain.feature_names,
        focus_features=focus_features,
        target_col=target_col,
        model=llm_model,
        temperature=llm_temperature,
        api_key=api_key,
    )
    if print_readouts:
        print("Novice readout:")
        print(json.dumps(novice_readout, indent=2, sort_keys=True))
        print("Mid readout:")
        print(json.dumps(mid_readout, indent=2, sort_keys=True))
        print("Expert readout:")
        print(json.dumps(expert_readout, indent=2, sort_keys=True))

    ro_novice = normalize_readout_to_unit_box(
        novice_readout, domain.mins, domain.maxs, feature_names=domain.feature_names
    )
    ro_mid = normalize_readout_to_unit_box(
        mid_readout, domain.mins, domain.maxs, feature_names=domain.feature_names
    )
    ro_expert = normalize_readout_to_unit_box(
        expert_readout, domain.mins, domain.maxs, feature_names=domain.feature_names
    )
    prior_novice = readout_to_prior(ro_novice, feature_names=domain.feature_names)
    prior_mid = readout_to_prior(ro_mid, feature_names=domain.feature_names)
    prior_expert = readout_to_prior(ro_expert, feature_names=domain.feature_names)

    def _calibrate_prior(prior: Any, X_unit: Tensor, y_true: Tensor) -> Tuple[float, float]:
        m0 = prior.m0_torch(X_unit).reshape(-1)
        m0_np = m0.detach().cpu().numpy()
        y_np = y_true.detach().cpu().numpy()
        var_m0 = float(np.var(m0_np))
        if var_m0 < 1e-12:
            return 0.0, float(np.mean(y_np))
        cov = float(np.cov(m0_np, y_np, bias=True)[0, 1])
        a = cov / (var_m0 + 1e-12)
        b = float(np.mean(y_np) - a * np.mean(m0_np))
        return a, b

    calib_pool = draw_sobol_samples(bounds=domain.unit_bounds, n=1024, q=1, seed=seed + 11).squeeze(1)
    y_calib = evaluate_oracle(domain, calib_pool).reshape(-1)
    a_nov, b_nov = _calibrate_prior(prior_novice, calib_pool, y_calib)
    a_mid, b_mid = _calibrate_prior(prior_mid, calib_pool, y_calib)
    a_exp, b_exp = _calibrate_prior(prior_expert, calib_pool, y_calib)

    grid_t = torch.tensor(grid_unit, dtype=DTYPE, device=DEVICE)
    m0_nov = prior_novice.m0_torch(grid_t).reshape(n_grid, n_grid).detach().cpu().numpy()
    m0_mid = prior_mid.m0_torch(grid_t).reshape(n_grid, n_grid).detach().cpu().numpy()
    m0_exp = prior_expert.m0_torch(grid_t).reshape(n_grid, n_grid).detach().cpu().numpy()
    m0_nov = a_nov * m0_nov + b_nov
    m0_mid = a_mid * m0_mid + b_mid
    m0_exp = a_exp * m0_exp + b_exp
    truth = evaluate_oracle(domain, grid_t).reshape(n_grid, n_grid).detach().cpu().numpy()

    fig = plt.figure(figsize=(10.5, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.25, hspace=0.25)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
    ax10 = fig.add_subplot(gs[1, 0], sharex=ax00, sharey=ax00)
    ax11 = fig.add_subplot(gs[1, 1], sharex=ax00, sharey=ax00)
    cax_prior = fig.add_subplot(gs[0, 2])
    cax_truth = fig.add_subplot(gs[1, 2])
    axes = [ax00, ax01, ax10, ax11]

    titles = [
        f"Prior ({novice_fraction * 100:.1f}% Data)",
        f"Prior ({mid_fraction * 100:.1f}% Data)",
        f"Prior ({expert_fraction * 100:.1f}% Data)",
        "Ground Truth",
    ]
    fields = [m0_nov, m0_mid, m0_exp, truth]

    prior_fields = np.concatenate([m0_nov.ravel(), m0_mid.ravel(), m0_exp.ravel()])
    prior_vmin = float(np.percentile(prior_fields, 2.0))
    prior_vmax = float(np.percentile(prior_fields, 98.0))
    if abs(prior_vmax - prior_vmin) < 1e-12:
        prior_vmin -= 1.0
        prior_vmax += 1.0

    truth_vmin = float(np.percentile(truth, 2.0))
    truth_vmax = float(np.percentile(truth, 98.0))
    if abs(truth_vmax - truth_vmin) < 1e-12:
        truth_vmin -= 1.0
        truth_vmax += 1.0

    prior_ref = None
    truth_ref = None
    for idx, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        if idx < 3:
            prior_ref = ax.contourf(
                X1, X4, field, levels=50, vmin=prior_vmin, vmax=prior_vmax, cmap="magma"
            )
        else:
            truth_ref = ax.contourf(
                X1, X4, field, levels=50, vmin=truth_vmin, vmax=truth_vmax, cmap="magma"
            )

        ax.set_title(title)
        if idx in (2, 3):
            ax.set_xlabel("x1 (Amine)")
        else:
            ax.tick_params(labelbottom=False)
        if idx in (0, 2):
            ax.set_ylabel("x4 (pTSA)")
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    if prior_ref is not None:
        fig.colorbar(prior_ref, cax=cax_prior, label="Prior scale")
    if truth_ref is not None:
        fig.colorbar(truth_ref, cax=cax_truth, label=target_col)
    return fig


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
    if include_init:
        x = x - x.min()
    y = d["best_so_far"].to_numpy(dtype=np.float64)
    return float(np.trapz(y, x))


def _auc_best_so_far_window(
    df: pd.DataFrame,
    *,
    include_init: bool,
    n_iters: Optional[int] = None,
) -> float:
    d = df.copy()
    if not include_init:
        d = d[d["iter"] >= 0].copy()
    if n_iters is not None:
        d = d[d["iter"] < int(n_iters)].copy()
    return _auc_best_so_far(d, include_init=include_init)


def _iters_to_target(df: pd.DataFrame, *, target: float, include_init: bool) -> int:
    d = df.copy()
    if not include_init:
        d = d[d["iter"] >= 0].copy()
    d = d.sort_values("iter")
    hits = d[d["best_so_far"] >= target]
    if hits.empty:
        if d.empty:
            return 0
        max_iter = int(d["iter"].max())
        if include_init:
            min_iter = int(d["iter"].min())
            return max_iter - min_iter + 1
        return max_iter + 1
    if include_init:
        min_iter = int(d["iter"].min())
        return int(hits["iter"].iloc[0]) - min_iter
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
    warm_start: bool = True,
    optimum: Optional[float] = None,
    optimum_samples: int = 4096,
    optimum_seed: int = 0,
    n_test: int = 256,
    test_seed: int = 123,
    target_fracs: Tuple[float, float] = (0.7, 0.8),
    n_bins: int = 10,
    weight_best: float = 0.4,
    weight_auc: float = 0.3,
    weight_speed_70: float = 0.15,
    weight_speed_80: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (summary_table, per_run_table, simple_regret_df)."""
    if warm_start:
        include_init = True
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
        t70 = float(optimum) * float(target_fracs[0])
        t80 = float(optimum) * float(target_fracs[1])
        iters_70 = _iters_to_target(df_run, target=t70, include_init=include_init)
        iters_80 = _iters_to_target(df_run, target=t80, include_init=include_init)

        X_unit = _unit_from_history(domain, df_run)
        explore = _exploration_quality(X_unit, n_bins=n_bins)
        rmse = _inference_rmse(domain, df_run, X_test=X_test, y_test=y_test)

        denom = abs(float(optimum)) + 1e-9
        final_score = 1.0 - (float(optimum) - final_best) / denom
        final_score = float(np.clip(final_score, 0.0, 1.0))
        max_iters = int(df_run["iter"].max() + 1) if not df_run.empty else 1
        if include_init and not df_run.empty:
            max_iters = int(df_run["iter"].max() - df_run["iter"].min() + 1)

        denom_auc = max(float(optimum) * max_iters, 1e-9)
        auc_score = float(np.clip(auc / denom_auc, 0.0, 1.0))

        speed_70 = 1.0 - min(iters_70, max_iters) / max_iters
        speed_80 = 1.0 - min(iters_80, max_iters) / max_iters
        speed_70 = float(np.clip(speed_70, 0.0, 1.0))
        speed_80 = float(np.clip(speed_80, 0.0, 1.0))

        composite = (
            float(weight_best) * final_score
            + float(weight_auc) * auc_score
            + float(weight_speed_70) * speed_70
            + float(weight_speed_80) * speed_80
        )

        per_run.append(
            {
                "method": method,
                "seed": int(seed),
                "final_best": final_best,
                "auc": auc,
                "iters_to_70": int(iters_70),
                "iters_to_80": int(iters_80),
                "auc_score": float(auc_score),
                "speed_70": float(speed_70),
                "speed_80": float(speed_80),
                "exploration_quality": float(explore),
                "inference_rmse": float(rmse),
                "composite_score": float(composite),
            }
        )

    per_run_df = pd.DataFrame(per_run)

    agg_map = {
        "final_best": ["median", "mean", "min", "std"],
        "auc": ["median", "mean", "min", "std"],
        "iters_to_70": ["median", "mean", "max", "std"],
        "iters_to_80": ["median", "mean", "max", "std"],
        "auc_score": ["median", "mean", "min", "std"],
        "speed_70": ["median", "mean", "min", "std"],
        "speed_80": ["median", "mean", "min", "std"],
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
    labels = [_method_label(m) for m in df["method"].tolist()]
    ax.bar(labels, df[metric])
    ax.set_ylabel(metric, fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Method", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_fraction_auc_bars(
    summary_df: pd.DataFrame,
    *,
    baseline_summary: Optional[pd.DataFrame] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
    else:
        fig = ax.figure
    df = summary_df.sort_values("fraction").copy()
    labels = [f"{frac * 100:.1f}%" for frac in df["fraction"].to_numpy()]
    values = df["auc_median"].to_numpy(dtype=np.float64)
    yerr = df["auc_std"].to_numpy(dtype=np.float64)

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(labels)))
    label_list = labels
    value_list = values
    yerr_list = yerr
    color_list = colors

    if baseline_summary is not None and not baseline_summary.empty:
        base = baseline_summary.iloc[0]
        label_list = ["Baseline BO"] + labels
        value_list = np.concatenate([[float(base["auc_median"])], values])
        yerr_list = np.concatenate([[float(base["auc_std"])], yerr])
        color_list = ["#6e6e6e"] + list(colors)

    ax.bar(label_list, value_list, yerr=yerr_list, color=color_list, edgecolor="black", linewidth=0.6, capsize=3)
    if baseline_summary is not None and not baseline_summary.empty:
        ax.patches[0].set_hatch("///")
    ax.set_xlabel("Data Portion", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("AUC (Best-so-far)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.grid(False)
    ax.margins(x=0.06)
    plt.tight_layout()
    return ax


def plot_fraction_best_so_far(
    hist_df: pd.DataFrame,
    *,
    baseline_df: Optional[pd.DataFrame] = None,
    ci: str = "sem",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    df = hist_df.copy()
    df = df[df["iter"] >= 0]
    if df.empty:
        raise ValueError("No hybrid history rows to plot.")

    fractions = sorted(df["fraction"].unique().tolist())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(fractions)))

    for frac, color in zip(fractions, colors):
        d = df[df["fraction"] == frac]
        if d.empty:
            continue
        agg = d.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
        if ci == "sem":
            err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        elif ci == "95ci":
            err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        else:
            err = agg["std"]
        x = agg["iter"].to_numpy() + 1
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        label = f"{frac * 100:.1f}%"
        ax.plot(x, y, label=label, color=color, linewidth=2.0)
        ax.fill_between(x, y - e, y + e, alpha=0.15, color=color)

    if baseline_df is not None and not baseline_df.empty:
        base = baseline_df[baseline_df["iter"] >= 0].copy()
        agg = base.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
        if not agg.empty:
            if ci == "sem":
                err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
            elif ci == "95ci":
                err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
            else:
                err = agg["std"]
            x = agg["iter"].to_numpy() + 1
            y = agg["mean"].to_numpy()
            e = err.to_numpy()
            ax.plot(x, y, label="Baseline BO", color="#6e6e6e", linewidth=2.0, linestyle="--")
            ax.fill_between(x, y - e, y + e, alpha=0.15, color="#6e6e6e")

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Yield Best-So-Far", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE, ncol=2)
    max_iter = int(df["iter"].max()) if not df.empty else -1
    _set_iteration_ticks(ax, max_iter)
    plt.tight_layout()
    return ax


def _best_so_far_curve_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    d = d[d["iter"] >= 0]
    if d.empty:
        return pd.DataFrame(columns=group_cols + ["iter", "mean", "std", "count", "sem", "ci95"])
    agg = d.groupby(group_cols + ["iter"])["best_so_far"].agg(["mean", "std", "count"]).reset_index()
    sem = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
    agg["sem"] = sem
    agg["ci95"] = 1.96 * sem
    return agg


def plot_fraction_best_so_far_panels(
    hist_df: pd.DataFrame,
    *,
    baseline_df: Optional[pd.DataFrame] = None,
    ci: str = "sem",
    ncols: int = 2,
    title: Optional[str] = None,
) -> List[plt.Axes]:
    df = hist_df.copy()
    df = df[df["iter"] >= 0]
    if df.empty:
        raise ValueError("No hybrid history rows to plot.")

    fractions = sorted(df["fraction"].unique().tolist())
    ncols = max(int(ncols), 1)
    nrows = int(np.ceil(len(fractions) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2), sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)

    base_agg = None
    if baseline_df is not None and not baseline_df.empty:
        base = baseline_df[baseline_df["iter"] >= 0].copy()
        base_agg = base.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()

    for idx, frac in enumerate(fractions):
        ax = axes[idx]
        d = df[df["fraction"] == frac]
        agg = d.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
        if ci == "sem":
            err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        elif ci == "95ci":
            err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        else:
            err = agg["std"]
        x = agg["iter"].to_numpy() + 1
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        ax.plot(x, y, color="#4c72b0", linewidth=2.0, label=f"Hybrid ({frac * 100:.1f}%)")
        ax.fill_between(x, y - e, y + e, alpha=0.18, color="#4c72b0")

        if base_agg is not None and not base_agg.empty:
            if ci == "sem":
                berr = base_agg["std"] / np.maximum(base_agg["count"], 1).pow(0.5)
            elif ci == "95ci":
                berr = 1.96 * base_agg["std"] / np.maximum(base_agg["count"], 1).pow(0.5)
            else:
                berr = base_agg["std"]
            bx = base_agg["iter"].to_numpy() + 1
            by = base_agg["mean"].to_numpy()
            be = berr.to_numpy()
            ax.plot(bx, by, color="#6e6e6e", linewidth=2.0, linestyle="--", label="Baseline BO")
            ax.fill_between(bx, by - be, by + be, alpha=0.18, color="#6e6e6e")

        ax.set_title(f"{frac * 100:.1f}% data", fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        max_iter = int(agg["iter"].max()) if not agg.empty else -1
        _set_iteration_ticks(ax, max_iter)
        if idx % ncols == 0:
            ax.set_ylabel("Yield Best-So-Far", fontsize=AXIS_LABEL_SIZE)
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONT_SIZE, ncol=1)

    for j in range(len(fractions), len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=AXIS_LABEL_SIZE)
    plt.tight_layout()
    return list(axes)


def plot_init_best_so_far_panels(
    hist_df: pd.DataFrame,
    *,
    ncols: int = 3,
    ci: str = "sem",
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
) -> List[plt.Axes]:
    df = hist_df.copy()
    df = df[df["iter"] >= 0]
    if df.empty:
        raise ValueError("No history rows to plot.")
    if methods is None:
        methods = ["baseline_ei", "hybrid_manual"]

    n_inits = sorted(df["n_init"].unique().tolist())
    ncols = max(int(ncols), 1)
    nrows = int(np.ceil(len(n_inits) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.2), sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)

    color_map = {"baseline_ei": "#6e6e6e", "hybrid_manual": "#4c72b0"}
    style_map = {"baseline_ei": "--", "hybrid_manual": "-"}

    for idx, n_init in enumerate(n_inits):
        ax = axes[idx]
        d = df[df["n_init"] == n_init]
        for method in methods:
            dm = d[d["method"] == method]
            if dm.empty:
                continue
            agg = dm.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
            if ci == "sem":
                err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
            elif ci == "95ci":
                err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
            else:
                err = agg["std"]
            x = agg["iter"].to_numpy() + 1
            y = agg["mean"].to_numpy()
            e = err.to_numpy()
            color = color_map.get(method, "#4c72b0")
            linestyle = style_map.get(method, "-")
            ax.plot(x, y, color=color, linewidth=2.0, linestyle=linestyle, label=_method_label(method))
            ax.fill_between(x, y - e, y + e, alpha=0.18, color=color)

        ax.set_title(f"n_init={int(n_init)}", fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        max_iter = int(d["iter"].max()) if not d.empty else -1
        _set_iteration_ticks(ax, max_iter)
        if idx % ncols == 0:
            ax.set_ylabel("Yield Best-So-Far", fontsize=AXIS_LABEL_SIZE)
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONT_SIZE, ncol=1)

    for j in range(len(n_inits), len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=AXIS_LABEL_SIZE)
    plt.tight_layout()
    return list(axes)


def plot_init_benchmark_bars(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 4.4))
    df = summary_df.copy()
    n_inits = sorted(df["n_init"].unique().tolist())
    x = np.arange(len(n_inits), dtype=np.float64)
    width = 0.36

    def _get_vals(method: str) -> Tuple[np.ndarray, np.ndarray]:
        vals = []
        errs = []
        for n in n_inits:
            row = df[(df["n_init"] == n) & (df["method"] == method)]
            if row.empty:
                vals.append(float("nan"))
                errs.append(0.0)
            else:
                vals.append(float(row["auc_median"].iloc[0]))
                errs.append(float(row["auc_std"].iloc[0]))
        return np.array(vals, dtype=np.float64), np.array(errs, dtype=np.float64)

    hyb_vals, hyb_err = _get_vals("hybrid_manual")
    base_vals, base_err = _get_vals("baseline_ei")

    ax.bar(
        x - width / 2,
        hyb_vals,
        width,
        yerr=hyb_err,
        color="#dd8452",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        label="Prior-Shaped BO",
    )
    ax.bar(
        x + width / 2,
        base_vals,
        width,
        yerr=base_err,
        color="#55a868",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        label="Baseline BO",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_inits])
    ax.set_xlabel("Initial Points (n_init)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("AUC (Best-so-far)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    if title:
        ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()
    return ax


def plot_iter_success_rates(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.4))
    df = summary_df.copy()
    iters = sorted(df["n_iter"].unique().tolist())
    x = np.arange(len(iters), dtype=np.float64)
    width = 0.36

    def _rates(method: str) -> np.ndarray:
        vals = []
        for n in iters:
            row = df[(df["method"] == method) & (df["n_iter"] == n)]
            vals.append(float(row["success_rate"].iloc[0]) if not row.empty else 0.0)
        return np.array(vals, dtype=np.float64)

    base_rates = _rates("baseline_ei")
    hyb_rates = _rates("hybrid_manual")

    ax.bar(
        x - width / 2,
        hyb_rates,
        width,
        color=METHOD_COLORS.get("hybrid_manual", "#dd8452"),
        edgecolor="black",
        linewidth=0.6,
        label="Prior-Shaped BO",
    )
    ax.bar(
        x + width / 2,
        base_rates,
        width,
        color=METHOD_COLORS.get("baseline_ei", "#55a868"),
        edgecolor="black",
        linewidth=0.6,
        label="Baseline BO",
    )

    ax.set_xlabel("Iterations", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Success rate (>= 90% optimum)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in iters])
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(False)
    plt.tight_layout()
    return ax


def plot_awcd_detection_timeline(
    hist_df: pd.DataFrame,
    *,
    threshold: float = 0.6,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))

    df = hist_df.copy()
    if "awcd_score" not in df.columns:
        raise ValueError("awcd_score column not found in hist_df.")
    df = df[df["awcd_score"].notna() & (df["iter"] >= 0)]
    if df.empty:
        raise ValueError("No AWCD rows to plot (check hist_df).")

    agg = df.groupby("iter")["awcd_score"].agg(["mean", "std", "count"]).reset_index()
    err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
    x = agg["iter"].to_numpy() + 1
    y = agg["mean"].to_numpy()
    e = err.to_numpy()

    ax.plot(x, y, color="#e74c3c", linewidth=2.5, label="AWCD Score", zorder=3)
    ax.fill_between(x, y - e, y + e, alpha=0.25, color="#e74c3c", zorder=2)

    ax.axhline(
        float(threshold),
        color="#c0392b",
        linestyle="--",
        linewidth=2.0,
        label=f"Confidence Level ({threshold:.2f})",
        zorder=4,
    )
    ax.axhspan(0.0, float(threshold), alpha=0.10, color="green", label="Safe (Prior Active)", zorder=1)
    ax.axhspan(float(threshold), 1.0, alpha=0.10, color="red", label="Danger (Prior Disabled)", zorder=1)

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("AWCD Score\n(Constraint Pressure)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right", bbox_to_anchor=(0.98, 0.02))
    ax.grid(False)
    max_iter = int(agg["iter"].max()) if not agg.empty else -1
    _set_iteration_ticks(ax, max_iter)
    plt.tight_layout()
    return ax


def _select_awcd_run(
    hist_df: pd.DataFrame,
    *,
    method: str,
    seed: Optional[int],
) -> pd.DataFrame:
    df = hist_df.copy()
    df = df[df["method"] == method]
    if df.empty:
        raise ValueError(f"No rows for method={method!r}.")
    if seed is None:
        seed = int(sorted(df["seed"].unique())[0]) if "seed" in df.columns else None
    if seed is not None and "seed" in df.columns:
        df = df[df["seed"] == seed]
    df = df[df["iter"] >= 0].sort_values("iter")
    if df.empty:
        raise ValueError("No rows available after filtering.")
    return df


def _raw_constraint_ranges(readout: Dict[str, Any], param: str) -> List[Tuple[float, float]]:
    ranges = []
    for c in (readout or {}).get("constraints") or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("var")) != str(param):
            continue
        r = c.get("range")
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        lo, hi = float(r[0]), float(r[1])
        if hi < lo:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    return ranges


def plot_awcd_topk_pressure_map(
    hist_df: pd.DataFrame,
    domain: ContinuousDomain,
    readout: Dict[str, Any],
    *,
    method: str = "ensemble",
    seed: Optional[int] = None,
    iter_idx: Optional[int] = None,
    pool_n: int = 1500,
    top_frac: float = 0.1,
    feature_x: str = "x1",
    feature_y: str = "x4",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 5.2))

    df = _select_awcd_run(hist_df, method=method, seed=seed)
    if iter_idx is None:
        iter_idx = int(df["iter"].max())
    df_obs = df[df["iter"] <= int(iter_idx)].copy()
    if df_obs.empty:
        raise ValueError("No observations for selected iteration.")

    X_obs = _unit_from_history(domain, df_obs)
    Y_obs = torch.tensor(df_obs["y"].to_numpy(dtype=np.float64), device=DEVICE, dtype=DTYPE).unsqueeze(-1)

    gp = SingleTaskGP(X_obs, Y_obs)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    pool = draw_sobol_samples(bounds=domain.unit_bounds, n=int(pool_n), q=1, seed=123).squeeze(1)
    penalties = _constraint_penalty_values(pool, normalize_readout_to_unit_box(readout, domain.mins, domain.maxs,
                                                                              feature_names=domain.feature_names),
                                          domain.feature_names)
    mask_forbidden = penalties > 0

    best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
    EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
    with torch.no_grad():
        pressure = (EI(pool.unsqueeze(1)).reshape(-1) * gp.posterior(pool).variance.sqrt().reshape(-1))

    k = max(1, int(float(top_frac) * float(pressure.numel())))
    top_idx = torch.topk(pressure, k=k).indices.detach().cpu().numpy()

    raw = unit_to_raw(domain, pool).detach().cpu().numpy()
    x_col = int(feature_x[1:]) - 1
    y_col = int(feature_y[1:]) - 1
    x_vals = raw[:, x_col]
    y_vals = raw[:, y_col]

    ax.scatter(
        x_vals,
        y_vals,
        c=pressure.detach().cpu().numpy(),
        cmap="viridis",
        s=16,
        alpha=0.7,
        label="Pool (pressure)",
    )
    ax.scatter(
        x_vals[top_idx],
        y_vals[top_idx],
        facecolors="none",
        edgecolors="#e74c3c",
        s=60,
        linewidths=1.5,
        label=f"Top {int(top_frac * 100)}% pressure",
    )

    for lo, hi in _raw_constraint_ranges(readout, feature_x):
        ax.axvspan(lo, hi, color="#e74c3c", alpha=0.12)
    for lo, hi in _raw_constraint_ranges(readout, feature_y):
        ax.axhspan(lo, hi, color="#e74c3c", alpha=0.12)

    ax.set_xlabel(feature_x, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(feature_y, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig = ax.figure
        fig.legend(
            handles,
            labels,
            fontsize=LEGEND_FONT_SIZE,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    if ax is None:
        plt.tight_layout(rect=[0, 0, 0.78, 1])
    else:
        ax.figure.tight_layout(rect=[0, 0, 0.78, 1])
    return ax


def plot_awcd_pressure_histogram(
    hist_df: pd.DataFrame,
    domain: ContinuousDomain,
    readout: Dict[str, Any],
    *,
    method: str = "ensemble",
    seed: Optional[int] = None,
    iter_idx: Optional[int] = None,
    pool_n: int = 2000,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 4.4))

    df = _select_awcd_run(hist_df, method=method, seed=seed)
    if iter_idx is None:
        iter_idx = int(df["iter"].max())
    df_obs = df[df["iter"] <= int(iter_idx)].copy()
    X_obs = _unit_from_history(domain, df_obs)
    Y_obs = torch.tensor(df_obs["y"].to_numpy(dtype=np.float64), device=DEVICE, dtype=DTYPE).unsqueeze(-1)

    gp = SingleTaskGP(X_obs, Y_obs)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    pool = draw_sobol_samples(bounds=domain.unit_bounds, n=int(pool_n), q=1, seed=321).squeeze(1)
    penalties = _constraint_penalty_values(pool, normalize_readout_to_unit_box(readout, domain.mins, domain.maxs,
                                                                              feature_names=domain.feature_names),
                                          domain.feature_names)
    mask_forbidden = penalties > 0

    best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
    EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
    with torch.no_grad():
        pressure = (EI(pool.unsqueeze(1)).reshape(-1) * gp.posterior(pool).variance.sqrt().reshape(-1))
    p_forbidden = pressure[mask_forbidden].detach().cpu().numpy()
    p_allowed = pressure[~mask_forbidden].detach().cpu().numpy()

    ax.hist(p_allowed, bins=30, alpha=0.6, color="#3498db", label="Allowed")
    ax.hist(p_forbidden, bins=30, alpha=0.6, color="#e74c3c", label="Forbidden")
    ax.set_xlabel("Pressure (EI × std)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    return ax


def plot_awcd_components_timeline(
    hist_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))

    df = hist_df.copy()
    if "awcd_constraint_mean" not in df.columns or "awcd_mean_disagree_mean" not in df.columns:
        raise ValueError("AWCD component columns not found in hist_df.")
    df = df[df["iter"] >= 0]
    if df.empty:
        raise ValueError("No rows to plot.")

    agg = df.groupby("iter")[["awcd_constraint_mean", "awcd_mean_disagree_mean"]].mean().reset_index()
    x = agg["iter"].to_numpy() + 1
    ax.plot(x, agg["awcd_constraint_mean"], color="#e74c3c", linewidth=2.0, label="Constraint Disagreement")
    ax.plot(x, agg["awcd_mean_disagree_mean"], color="#f39c12", linewidth=2.0, label="Mean Disagreement")
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Disagreement Rate", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    _set_iteration_ticks(ax, int(agg["iter"].max()))
    plt.tight_layout()
    return ax


def _acquisition_influence_rows(
    hist_df: pd.DataFrame,
    domain: ContinuousDomain,
    readout: Dict[str, Any],
    *,
    method: str = "ensemble",
    prior_strength: float = 1.0,
    constraint_hardness: float = 0.0,
    pool_n: int = 1024,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    df = hist_df[(hist_df["method"] == method) & (hist_df["iter"].notna())].copy()
    if df.empty:
        return []

    ro_unit = normalize_readout_to_unit_box(readout, domain.mins, domain.maxs, feature_names=domain.feature_names)
    prior = readout_to_prior(ro_unit, feature_names=domain.feature_names)

    rows: List[Dict[str, Any]] = []
    for seed_val, df_seed in df.groupby("seed"):
        df_seed = df_seed.sort_values("iter")
        iters = sorted(int(v) for v in df_seed["iter"].unique() if v >= 0)
        for t in iters:
            df_obs = df_seed[df_seed["iter"] < t]
            if df_obs.empty:
                continue
            X_obs = _unit_from_history(domain, df_obs)
            Y_obs = torch.tensor(df_obs["y"].to_numpy(dtype=np.float64), device=DEVICE, dtype=DTYPE).unsqueeze(-1)
            if X_obs.shape[0] < 2:
                continue

            gp_skeptic = SingleTaskGP(X_obs, Y_obs)
            mll_s = ExactMarginalLogLikelihood(gp_skeptic.likelihood, gp_skeptic)
            try:
                with cholesky_jitter(1e-4):
                    fit_gpytorch_mll(mll_s, max_attempts=5)
            except ModelFittingError:
                Y_jittered = Y_obs + 1e-6 * torch.randn_like(Y_obs)
                gp_skeptic = SingleTaskGP(
                    X_obs,
                    Y_jittered,
                    input_transform=Normalize(d=X_obs.shape[-1]),
                    outcome_transform=Standardize(m=1),
                )
                mll_s = ExactMarginalLogLikelihood(gp_skeptic.likelihood, gp_skeptic)
                with cholesky_jitter(1e-3):
                    fit_gpytorch_mll(mll_s, max_attempts=5)

            gp_resid, alpha = fit_residual_gp(X_obs, Y_obs, prior)
            m0_scale = float(alpha * prior_strength)
            model_believer = GPWithPriorMean(gp_resid, prior, m0_scale=m0_scale)

            pool = draw_sobol_samples(
                bounds=domain.unit_bounds,
                n=int(pool_n),
                q=1,
                seed=int(seed) + int(seed_val) * 1000 + int(t),
            ).squeeze(1)
            best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
            EI_s = ExpectedImprovement(model=gp_skeptic, best_f=best_f, maximize=True)
            EI_b = ExpectedImprovement(model=model_believer, best_f=best_f, maximize=True)
            with torch.no_grad():
                ei_s = EI_s(pool.unsqueeze(1)).reshape(-1)
                ei_b = EI_b(pool.unsqueeze(1)).reshape(-1)
                if constraint_hardness > 0.0:
                    ei_b = _apply_constraint_hardness(
                        ei_b,
                        pool,
                        ro_unit,
                        domain.feature_names,
                        hardness=constraint_hardness,
                        best_f=best_f,
                    )
                active = True
                if "prior_active" in df_seed.columns:
                    active = bool(df_seed[df_seed["iter"] == t]["prior_active"].iloc[0])
                influence = (ei_b - ei_s).abs().mean()
                denom = ei_s.abs().mean().clamp_min(1e-9)
                score = float((influence / denom).item())
                if not active:
                    score = 0.0
            rows.append({"iter": int(t), "seed": int(seed_val), "influence": score})
    return rows


def plot_awcd_acquisition_influence_timeline(
    hist_df: pd.DataFrame,
    domain: ContinuousDomain,
    readout: Dict[str, Any],
    *,
    method: str = "ensemble",
    prior_strength: float = 1.0,
    constraint_hardness: float = 0.0,
    pool_n: int = 1024,
    seed: int = 0,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.4))

    rows = _acquisition_influence_rows(
        hist_df,
        domain,
        readout,
        method=method,
        prior_strength=prior_strength,
        constraint_hardness=constraint_hardness,
        pool_n=pool_n,
        seed=seed,
    )
    if not rows:
        raise ValueError("No acquisition influence values computed.")

    df_inf = pd.DataFrame(rows)
    agg = df_inf.groupby("iter")["influence"].agg(["mean", "std", "count"]).reset_index()
    x = agg["iter"].to_numpy() + 1
    mean = agg["mean"].to_numpy()
    std = agg["std"].to_numpy()
    count = agg["count"].to_numpy()
    sem = std / np.maximum(count, 1) ** 0.5

    ax.plot(x, mean, color="#2c3e50", linewidth=2.0, label="Acquisition influence")
    ax.fill_between(x, mean - sem, mean + sem, alpha=0.2, color="#2c3e50")
    auc = float(np.sum(mean))
    ax.text(
        0.02,
        0.90,
        f"Influence AUC: {auc:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Normalized Influence", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0.0, max(1.05, float(np.nanmax(mean) + 0.1)))
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right")
    _set_iteration_ticks(ax, int(agg["iter"].max()))
    plt.tight_layout()
    return ax


def plot_awcd_prior_rank_scatter(
    hist_df: pd.DataFrame,
    domain: ContinuousDomain,
    readout: Dict[str, Any],
    *,
    method: str = "ensemble",
    seed: Optional[int] = None,
    iter_idx: Optional[int] = None,
    pool_n: int = 1500,
    top_frac: float = 0.1,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 4.8))

    df = _select_awcd_run(hist_df, method=method, seed=seed)
    if iter_idx is None:
        iter_idx = int(df["iter"].max())
    df_obs = df[df["iter"] <= int(iter_idx)].copy()
    X_obs = _unit_from_history(domain, df_obs)
    Y_obs = torch.tensor(df_obs["y"].to_numpy(dtype=np.float64), device=DEVICE, dtype=DTYPE).unsqueeze(-1)

    gp = SingleTaskGP(X_obs, Y_obs)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    pool = draw_sobol_samples(bounds=domain.unit_bounds, n=int(pool_n), q=1, seed=777).squeeze(1)
    best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
    EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
    with torch.no_grad():
        pressure = (EI(pool.unsqueeze(1)).reshape(-1) * gp.posterior(pool).variance.sqrt().reshape(-1))
        gp_mean = gp.posterior(pool).mean.reshape(-1)
        prior = readout_to_prior(
            normalize_readout_to_unit_box(readout, domain.mins, domain.maxs, feature_names=domain.feature_names),
            feature_names=domain.feature_names,
        )
        prior_mean = prior.m0_torch(pool).reshape(-1)

    k = max(1, int(float(top_frac) * float(pressure.numel())))
    top_idx = torch.topk(pressure, k=k).indices
    prior_rank = torch.argsort(torch.argsort(prior_mean, descending=True))
    median_rank = int(prior_mean.numel() // 2)

    x = prior_rank[top_idx].detach().cpu().numpy()
    y = gp_mean[top_idx].detach().cpu().numpy()
    bad_mask = (prior_rank[top_idx] > median_rank).detach().cpu().numpy()
    colors = np.where(bad_mask, "#e74c3c", "#27ae60")

    ax.scatter(x, y, c=colors, s=40, alpha=0.8, label="Top pressure points")
    ax.axvline(median_rank, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Prior median rank")
    ax.set_xlabel("Prior rank (lower = better)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("GP mean (skeptic)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    return ax


def plot_awcd_prior_active_ribbon(
    hist_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 2.6))

    df = hist_df.copy()
    if "prior_active" not in df.columns:
        raise ValueError("prior_active column not found in hist_df.")
    df = df[df["iter"] >= 0]
    agg = df.groupby("iter")["prior_active"].mean().reset_index()
    x = agg["iter"].to_numpy() + 1
    y = agg["prior_active"].astype(float).to_numpy()

    ax.fill_between(x, 0.0, y, step="mid", color="#f39c12", alpha=0.35, label="Prior Active Fraction")
    ax.plot(x, y, drawstyle="steps-mid", color="#d35400", linewidth=2.0)
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Active Fraction", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper right")
    _set_iteration_ticks(ax, int(agg["iter"].max()))
    plt.tight_layout()
    return ax


def run_awcd_threshold_sweep(
    domain: ContinuousDomain,
    readouts: Dict[str, Dict[str, Any]],
    thresholds: List[float],
    *,
    n_init: int,
    n_iter: int,
    repeats: int,
    seed: int,
    constraint_hardness: float,
    constraint_pool_size: int,
    awcd_top_frac: float,
    awcd_warmup: int,
    awcd_window: int,
    baseline_cache: str = "baseline_cache.csv",
    out_dir: str = "awcd_sweep",
) -> Dict[Tuple[str, float], pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(baseline_cache):
        hist_baseline = pd.read_csv(baseline_cache)
        hist_baseline["method"] = "baseline"
        print(f"Loaded baseline cache from {baseline_cache}")
    else:
        hist_baseline = run_baseline_ei_continuous(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            repeats=repeats,
            init_method="sobol",
        )
        hist_baseline = hist_baseline.copy()
        hist_baseline["method"] = "baseline"
        hist_baseline.to_csv(baseline_cache, index=False)
        print(f"Saved baseline cache to {baseline_cache}")

    results: Dict[Tuple[str, float], pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []
    all_tables: List[pd.DataFrame] = []

    for readout_name, readout in readouts.items():
        for thr in thresholds:
            print(f"[AWCD sweep] readout={readout_name} threshold={thr:.3f}")
            hist_forced = run_ensemble_continuous(
                domain,
                n_init=n_init,
                n_iter=n_iter,
                seed=seed,
                repeats=repeats,
                manual_readout=readout,
                init_method="sobol",
                prior_strength=1.0,
                constraint_hardness=constraint_hardness,
                constraint_pool_size=constraint_pool_size,
                awcd_top_frac=awcd_top_frac,
                awcd_constraint_threshold=thr,
                awcd_mean_threshold=thr,
                awcd_warmup=awcd_warmup,
                awcd_window=awcd_window,
                weight_mode="believer",
                method_tag="forced_prior",
            )
            hist_ensemble = run_ensemble_continuous(
                domain,
                n_init=n_init,
                n_iter=n_iter,
                seed=seed,
                repeats=repeats,
                manual_readout=readout,
                init_method="sobol",
                prior_strength=1.0,
                constraint_hardness=constraint_hardness,
                constraint_pool_size=constraint_pool_size,
                awcd_top_frac=awcd_top_frac,
                awcd_constraint_threshold=thr,
                awcd_mean_threshold=thr,
                awcd_warmup=awcd_warmup,
                awcd_window=awcd_window,
                early_prior_boost=True,
                early_prior_steps=5,
                weight_mode="ensemble",
                method_tag="ensemble",
            )

            hist_all = pd.concat([hist_baseline, hist_forced, hist_ensemble], ignore_index=True)
            results[(readout_name, float(thr))] = hist_all

            tag = f"{readout_name}_thr_{thr:.2f}".replace(".", "p")
            csv_path = os.path.join(out_dir, f"awcd_history_{tag}.csv")
            hist_all.to_csv(csv_path, index=False)
            hist_table = hist_all.copy()
            hist_table["readout"] = readout_name
            hist_table["confidence"] = float(thr)
            hist_table["tag"] = tag
            hist_table["table"] = "history"
            all_tables.append(hist_table)

            per_seed_rows: List[Dict[str, Any]] = []
            for (method, seed_val), df_run in hist_all.groupby(["method", "seed"]):
                auc_val = _auc_best_so_far_window(df_run, include_init=False, n_iters=n_iter)
                max_best = float(df_run["best_so_far"].max()) if not df_run.empty else float("nan")
                prior_used = float("nan")
                if method == "ensemble" and "prior_active" in df_run.columns:
                    prior_used = float(df_run["prior_active"].astype(bool).mean())
                per_seed_rows.append(
                    {
                        "readout": readout_name,
                        "confidence": float(thr),
                        "method": method,
                        "seed": int(seed_val),
                        "auc": float(auc_val),
                        "max_best": max_best,
                        "prior_used": prior_used,
                    }
                )
            per_seed_df = pd.DataFrame(per_seed_rows)
            per_seed_df.to_csv(os.path.join(out_dir, f"awcd_auc_runs_{tag}.csv"), index=False)
            per_seed_table = per_seed_df.copy()
            per_seed_table["tag"] = tag
            per_seed_table["table"] = "auc_runs"
            all_tables.append(per_seed_table)

            auc_summary = (
                per_seed_df.groupby("method")["auc"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .rename(columns={"mean": "auc_mean", "std": "auc_std", "count": "n_runs"})
            )
            auc_summary["auc_sem"] = auc_summary["auc_std"] / np.maximum(auc_summary["n_runs"], 1) ** 0.5
            auc_summary["readout"] = readout_name
            auc_summary["confidence"] = float(thr)
            auc_summary.to_csv(os.path.join(out_dir, f"awcd_auc_summary_{tag}.csv"), index=False)
            auc_table = auc_summary.copy()
            auc_table["tag"] = tag
            auc_table["table"] = "auc_summary"
            all_tables.append(auc_table)

            if "prior_active" in hist_all.columns:
                prior_rows = []
                for seed_val, df_seed in hist_all[hist_all["method"] == "ensemble"].groupby("seed"):
                    prior_rows.append(
                        {
                            "readout": readout_name,
                            "confidence": float(thr),
                            "seed": int(seed_val),
                            "prior_used": float(df_seed["prior_active"].astype(bool).mean()),
                        }
                    )
                if prior_rows:
                    prior_df = pd.DataFrame(prior_rows)
                    prior_df.to_csv(os.path.join(out_dir, f"awcd_prior_usage_{tag}.csv"), index=False)
                    prior_table = prior_df.copy()
                    prior_table["tag"] = tag
                    prior_table["table"] = "prior_usage"
                    all_tables.append(prior_table)

            traj_rows = (
                hist_all.groupby(["method", "iter"])["best_so_far"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .rename(columns={"mean": "best_mean", "std": "best_std", "count": "n"})
            )
            traj_rows["best_sem"] = traj_rows["best_std"] / np.maximum(traj_rows["n"], 1) ** 0.5
            traj_rows["readout"] = readout_name
            traj_rows["confidence"] = float(thr)
            traj_rows.to_csv(os.path.join(out_dir, f"awcd_trajectory_{tag}.csv"), index=False)
            traj_table = traj_rows.copy()
            traj_table["tag"] = tag
            traj_table["table"] = "trajectory"
            all_tables.append(traj_table)

            if "awcd_score" in hist_all.columns:
                awcd_rows = (
                    hist_all[hist_all["method"] == "ensemble"]
                    .groupby("iter")["awcd_score"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                    .rename(columns={"mean": "awcd_mean", "std": "awcd_std", "count": "n"})
                )
                awcd_rows["awcd_sem"] = awcd_rows["awcd_std"] / np.maximum(awcd_rows["n"], 1) ** 0.5
                awcd_rows["readout"] = readout_name
                awcd_rows["confidence"] = float(thr)
                awcd_rows.to_csv(os.path.join(out_dir, f"awcd_score_{tag}.csv"), index=False)
                awcd_table = awcd_rows.copy()
                awcd_table["tag"] = tag
                awcd_table["table"] = "awcd_score"
                all_tables.append(awcd_table)

            for _, row in auc_summary.iterrows():
                summary_rows.append(
                    {
                        "readout": readout_name,
                        "confidence": float(thr),
                        "method": row["method"],
                        "auc_mean": float(row["auc_mean"]),
                        "auc_std": float(row["auc_std"]),
                        "auc_sem": float(row["auc_sem"]),
                        "n_runs": int(row["n_runs"]),
                    }
                )

            def _save(ax: plt.Axes, name: str) -> None:
                fig = ax.figure
                out_path = os.path.join(out_dir, f"{name}_{tag}.png")
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

            _save(
                plot_runs_mean_lookup(
                    hist_all,
                    methods=["baseline", "forced_prior", "ensemble"],
                    ci="sem",
                    title=f"Best-so-far ({readout_name}, conf={thr:.2f})",
                    show_auc_text=True,
                ),
                "best_so_far",
            )
            _save(
                plot_awcd_detection_timeline(
                    hist_all,
                    threshold=thr,
                    title=f"AWCD Detection Timeline ({readout_name}, conf={thr:.2f})",
                ),
                "awcd_timeline",
            )
            _save(
                plot_prior_switching_behavior(
                    hist_all,
                    threshold=thr,
                    title=f"Prior Used ({readout_name}, conf={thr:.2f})",
                ),
                "prior_used",
            )
            _save(
                plot_awcd_performance_comparison(
                    hist_all,
                    title=f"AWCD Guard vs Baseline vs Forced ({readout_name}, conf={thr:.2f})",
                ),
                "awcd_performance",
            )
            _save(
                plot_awcd_auc_comparison(
                    hist_all,
                    methods=["baseline", "forced_prior", "ensemble"],
                    auc_iters=n_iter,
                    title=f"AUC Comparison ({readout_name}, conf={thr:.2f})",
                ),
                "awcd_auc",
            )
            _save(
                plot_awcd_topk_pressure_map(
                    hist_all,
                    domain,
                    readout,
                    method="ensemble",
                    top_frac=awcd_top_frac,
                    title=f"Top-K Pressure Map ({readout_name}, conf={thr:.2f})",
                ),
                "awcd_pressure_map",
            )
            _save(
                plot_awcd_pressure_histogram(
                    hist_all,
                    domain,
                    readout,
                    method="ensemble",
                    title=f"Pressure Histogram ({readout_name}, conf={thr:.2f})",
                ),
                "awcd_pressure_hist",
            )
            _save(
                plot_awcd_components_timeline(
                    hist_all,
                    title=f"AWCD Components ({readout_name}, conf={thr:.2f})",
                ),
                "awcd_components",
            )
            _save(
                plot_awcd_prior_rank_scatter(
                    hist_all,
                    domain,
                    readout,
                    method="ensemble",
                    top_frac=awcd_top_frac,
                    title=f"Prior Rank vs GP Mean ({readout_name}, conf={thr:.2f})",
                ),
                "awcd_rank_scatter",
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(out_dir, "awcd_sweep_summary.csv"), index=False)
        summary_table = summary_df.copy()
        summary_table["tag"] = "all"
        summary_table["table"] = "sweep_summary"
        all_tables.append(summary_table)
    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "awcd_sweep_all.csv"), index=False)
    return results


def plot_prior_switching_behavior(
    hist_df: pd.DataFrame,
    *,
    threshold: float = 0.6,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))

    df = hist_df.copy()
    if "prior_active" not in df.columns or "awcd_score" not in df.columns:
        raise ValueError("prior_active or awcd_score column not found in hist_df.")
    df = df[(df["iter"] >= 0) & (df["prior_active"].notna())]
    if df.empty:
        raise ValueError("No rows to plot (check hist_df).")

    agg_awcd = df.groupby("iter")["awcd_score"].mean().reset_index()

    x = df["iter"].to_numpy() + 1
    y_used = df["prior_active"].astype(float).astype(int).to_numpy()

    ax2 = ax.twinx()
    ax.scatter(x, y_used, color="#3498db", s=28, alpha=0.65, label="Prior Used (0/1)")

    ax2.plot(
        agg_awcd["iter"].to_numpy() + 1,
        agg_awcd["awcd_score"].to_numpy(),
        color="#e74c3c",
        linewidth=2.0,
        alpha=0.7,
        linestyle="--",
        label="AWCD Score",
    )

    ax.axhline(1.0, color="green", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.axhline(0.0, color="red", linestyle=":", alpha=0.5, linewidth=1.0)
    ax2.axhline(float(threshold), color="#c0392b", linestyle="--", alpha=0.5, linewidth=1.0)

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Prior Used (0/1)", fontsize=AXIS_LABEL_SIZE, color="#2c3e50")
    ax2.set_ylabel("AWCD Score", fontsize=AXIS_LABEL_SIZE, color="#e74c3c")
    ax.set_ylim(-0.05, 1.15)
    ax2.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="y", labelcolor="#2c3e50", labelsize=TICK_LABEL_SIZE)
    ax2.tick_params(axis="y", labelcolor="#e74c3c", labelsize=TICK_LABEL_SIZE)
    if title:
        ax.set_title(title)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", bbox_to_anchor=(0.98, 0.02))
    ax.grid(False)
    max_iter = int(df["iter"].max()) if not df.empty else -1
    _set_iteration_ticks(ax, max_iter)
    plt.tight_layout()
    return ax


def plot_awcd_performance_comparison(
    hist_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))

    df = hist_df.copy()
    df = df[df["iter"] >= 0]

    method_config = {
        "baseline_bad": {"label": "Baseline", "color": "#27ae60", "linestyle": "-", "linewidth": 2.0},
        "forced_bad": {"label": "PSBO-No Guard", "color": "#e74c3c", "linestyle": "--", "linewidth": 2.5},
        "awcd_guard": {"label": "PSBO-Guarded", "color": "#f39c12", "linestyle": "-", "linewidth": 2.5},
    }

    for method, style in method_config.items():
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue

        agg = method_df.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
        x = agg["iter"].to_numpy() + 1
        y = agg["mean"].to_numpy()
        err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)

        ax.plot(
            x,
            y,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax.fill_between(x, y - err, y + err, alpha=0.15, color=style["color"])

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Best Yield Found", fontsize=AXIS_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right")
    ax.grid(False)
    if title:
        ax.set_title(title)

    if not df.empty:
        final_iter = int(df["iter"].max())
        forced = df[(df["method"] == "forced_bad") & (df["iter"] == final_iter)]
        guarded = df[(df["method"] == "awcd_guard") & (df["iter"] == final_iter)]
        if not forced.empty and not guarded.empty:
            y_forced = float(forced["best_so_far"].mean())
            y_guarded = float(guarded["best_so_far"].mean())
            if y_forced != 0:
                improvement = ((y_guarded - y_forced) / y_forced) * 100.0
                ax.annotate(
                    "",
                    xy=(final_iter + 1, y_guarded),
                    xytext=(final_iter + 1, y_forced),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=2),
                )
                ax.text(
                    final_iter + 3,
                    (y_forced + y_guarded) / 2.0,
                    f"+{improvement:.1f}%",
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                )

    max_iter = int(df["iter"].max()) if not df.empty else -1
    _set_iteration_ticks(ax, max_iter)
    plt.tight_layout()
    return ax


def plot_awcd_auc_comparison(
    hist_df: pd.DataFrame,
    *,
    methods: Optional[List[str]] = None,
    auc_iters: Optional[int] = None,
    include_init: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 5.6))
    else:
        fig = ax.figure
        fig.set_size_inches(4.6, 5.6, forward=True)

    df = hist_df.copy()
    if not include_init:
        df = df[df["iter"] >= 0]
    if auc_iters is not None:
        df = df[df["iter"] < int(auc_iters)]
    if methods is None:
        methods = ["baseline", "forced_prior", "ensemble"]

    rows: List[Dict[str, Any]] = []
    for (method, seed), df_run in df.groupby(["method", "seed"]):
        if method not in methods:
            continue
        auc_val = _auc_best_so_far_window(df_run, include_init=include_init, n_iters=auc_iters)
        max_best = float(df_run["best_so_far"].max()) if not df_run.empty else float("nan")
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "auc": float(auc_val),
                "max_best": max_best,
            }
        )

    per_run = pd.DataFrame(rows)
    if per_run.empty:
        raise ValueError("No runs found to compute AUC comparison.")

    agg = per_run.groupby("method")["auc"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
    max_best = per_run.groupby("method")["max_best"].max().to_dict()

    order = [m for m in methods if m in agg["method"].tolist()]
    agg = agg.set_index("method").reindex(order).reset_index()
    labels = [_method_label(m) for m in agg["method"].tolist()]
    colors = [METHOD_COLORS.get(m, "#4c72b0") for m in agg["method"].tolist()]

    bars = ax.bar(
        np.arange(len(labels)),
        agg["mean"].to_numpy(),
        yerr=agg["sem"].to_numpy(),
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
    )

    for idx, bar in enumerate(bars):
        method = agg["method"].iloc[idx]
        best_val = max_best.get(method)
        if best_val is None or np.isnan(best_val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + float(agg["sem"].iloc[idx]) + 0.01,
            f"max {best_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC (Best-so-far)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(False)
    if title:
        ax.set_title(title)
    fig.set_size_inches(4.6, 5.6, forward=True)
    plt.tight_layout()
    return ax


def _readout_feature_index(name: Any, feature_names: List[str]) -> Optional[int]:
    if name is None:
        return None
    label = str(name).strip()
    if label.lower().startswith("x") and label[1:].isdigit():
        idx = int(label[1:]) - 1
        return idx if 0 <= idx < len(feature_names) else None
    for idx, fname in enumerate(feature_names):
        if label == fname or label.lower() == fname.lower():
            return idx
    return None


def _collect_readout_features(readout: Dict[str, Any], feature_names: List[str]) -> List[int]:
    idxs: set[int] = set()
    effects = (readout or {}).get("effects") or {}
    for key in effects:
        idx = _readout_feature_index(key, feature_names)
        if idx is not None:
            idxs.add(idx)
    for c in (readout or {}).get("constraints") or []:
        if not isinstance(c, dict):
            continue
        idx = _readout_feature_index(c.get("var"), feature_names)
        if idx is not None:
            idxs.add(idx)
    for b in (readout or {}).get("bumps") or []:
        mu = b.get("mu")
        sigma = b.get("sigma")
        if isinstance(mu, (list, tuple)) and isinstance(sigma, (list, tuple)):
            if len(mu) == len(feature_names) and len(sigma) == len(feature_names):
                idxs.update(range(len(feature_names)))
    if not idxs:
        idxs.update(range(len(feature_names)))
    return sorted(idxs)


def _constraint_ranges_for_idx(
    readout: Dict[str, Any],
    *,
    idx: int,
    feature_names: List[str],
) -> List[Tuple[float, float]]:
    ranges = []
    for c in (readout or {}).get("constraints") or []:
        if not isinstance(c, dict):
            continue
        var = c.get("var")
        target_idx = _readout_feature_index(var, feature_names)
        if target_idx is None or target_idx != idx:
            continue
        r = c.get("range")
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        lo, hi = float(r[0]), float(r[1])
        if hi < lo:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    return ranges


def _effect_spec_for_idx(
    readout: Dict[str, Any],
    *,
    idx: int,
    feature_names: List[str],
) -> Optional[Dict[str, Any]]:
    effects = (readout or {}).get("effects") or {}
    for key, spec in effects.items():
        target_idx = _readout_feature_index(key, feature_names)
        if target_idx is not None and target_idx == idx:
            return spec if isinstance(spec, dict) else None
    return None


def plot_readout_map(
    readout: Dict[str, Any],
    domain: ContinuousDomain,
    *,
    title: Optional[str] = None,
    show_truth: bool = True,
    truth_top_frac: float = 0.1,
    truth_samples: int = 2048,
    truth_seed: int = 0,
) -> plt.Figure:
    feature_names = list(domain.feature_names)
    idxs = _collect_readout_features(readout, feature_names)
    mins = domain.mins.detach().cpu().numpy()
    maxs = domain.maxs.detach().cpu().numpy()

    fig, axes = plt.subplots(len(idxs), 1, figsize=(7.6, 2.2 * len(idxs)), sharex=False)
    axes = np.atleast_1d(axes).tolist()

    constraint_color = "#e74c3c"
    hint_color = "#4c72b0"
    bump_color = "#8172b2"
    base_color = "#95a5a6"
    truth_color = "#2ca02c"

    truth_ranges: Dict[int, Tuple[float, float]] = {}
    if show_truth and truth_top_frac > 0.0:
        pool = draw_sobol_samples(
            bounds=domain.unit_bounds,
            n=int(truth_samples),
            q=1,
            seed=int(truth_seed),
        ).squeeze(1)
        y = evaluate_oracle(domain, pool).reshape(-1)
        k = max(1, int(float(truth_top_frac) * float(y.numel())))
        top_idx = torch.topk(y, k=k).indices
        top_raw = unit_to_raw(domain, pool).detach().cpu().numpy()[top_idx.cpu().numpy()]
        for idx in idxs:
            if top_raw.size == 0:
                continue
            truth_ranges[idx] = (float(top_raw[:, idx].min()), float(top_raw[:, idx].max()))

    for ax, idx in zip(axes, idxs):
        label = feature_names[idx] if idx < len(feature_names) else f"x{idx + 1}"
        lo_dom, hi_dom = float(mins[idx]), float(maxs[idx])

        ax.hlines(0.0, lo_dom, hi_dom, color=base_color, linewidth=3.0, alpha=0.6)
        if idx in truth_ranges:
            lo_t, hi_t = truth_ranges[idx]
            ax.axvspan(
                lo_t,
                hi_t,
                color=truth_color,
                alpha=0.18,
                label=f"top {int(truth_top_frac * 100)}% oracle",
            )

        effect_spec = _effect_spec_for_idx(readout, idx=idx, feature_names=feature_names)
        if effect_spec:
            rh = effect_spec.get("range_hint")
            if isinstance(rh, (list, tuple)) and len(rh) == 2:
                lo, hi = float(rh[0]), float(rh[1])
                if hi < lo:
                    lo, hi = hi, lo
                ax.axvspan(lo, hi, color=hint_color, alpha=0.22, label="range hint")
            eff = str(effect_spec.get("effect", "")).strip()
            if eff:
                ax.text(
                    0.98,
                    0.8,
                    eff,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                    color="#2c3e50",
                )

        for lo, hi in _constraint_ranges_for_idx(readout, idx=idx, feature_names=feature_names):
            ax.axvspan(lo, hi, color=constraint_color, alpha=0.18, label="constraint")

        for b in (readout or {}).get("bumps") or []:
            mu = b.get("mu")
            sigma = b.get("sigma")
            if isinstance(mu, (list, tuple)) and isinstance(sigma, (list, tuple)):
                if idx < len(mu) and idx < len(sigma):
                    mu_val = float(mu[idx])
                    sigma_val = float(sigma[idx])
                    ax.errorbar(
                        mu_val,
                        0.0,
                        xerr=max(sigma_val, 1e-9),
                        fmt="o",
                        color=bump_color,
                        capsize=3,
                        markersize=5,
                        alpha=0.9,
                        label="bump center",
                    )

        ax.set_xlim(lo_dom, hi_dom)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(len(by_label), 4),
            fontsize=LEGEND_FONT_SIZE,
            frameon=False,
        )
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    return fig



def awcd_prior_active_ratio_sweep(
    domain: ContinuousDomain,
    *,
    readout: Dict[str, Any],
    confidences: List[float],
    n_init: int,
    n_iter: int,
    repeats: int,
    seed: int,
    constraint_hardness: float,
    constraint_pool_size: int,
    awcd_top_frac: float,
    awcd_warmup: int,
    awcd_window: int,
    early_prior_boost: bool = True,
    early_prior_steps: int = 5,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for conf in confidences:
        hist = run_ensemble_continuous(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            repeats=repeats,
            manual_readout=readout,
            init_method="sobol",
            prior_strength=1.0,
            constraint_hardness=constraint_hardness,
            constraint_pool_size=constraint_pool_size,
            awcd_top_frac=awcd_top_frac,
            awcd_constraint_threshold=float(conf),
            awcd_mean_threshold=float(conf),
            awcd_warmup=awcd_warmup,
            awcd_window=awcd_window,
            early_prior_boost=early_prior_boost,
            early_prior_steps=early_prior_steps,
            weight_mode="ensemble",
            method_tag="ensemble",
        )
        df = hist[(hist["iter"] >= 0) & (hist["prior_active"].notna())].copy()
        total = int(df.shape[0])
        active = int(df["prior_active"].astype(bool).sum())
        inactive = total - active
        rows.append(
            {
                "confidence": float(conf),
                "active_frac": active / total if total else 0.0,
                "inactive_frac": inactive / total if total else 0.0,
                "active_count": active,
                "inactive_count": inactive,
                "total": total,
            }
        )
    return pd.DataFrame(rows)


def plot_awcd_prior_active_ratio(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    created_ax = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.4))
    df = summary_df.sort_values("confidence").copy()
    x = np.arange(len(df), dtype=np.float64)
    active = df["active_frac"].to_numpy(dtype=np.float64)
    inactive = df["inactive_frac"].to_numpy(dtype=np.float64)

    ax.bar(
        x,
        active,
        color=METHOD_COLORS.get("ensemble", "#f39c12"),
        edgecolor="black",
        linewidth=0.7,
        hatch="///",
        label="Prior Active",
    )
    ax.bar(
        x,
        inactive,
        bottom=active,
        color="#d0d0d0",
        edgecolor="black",
        linewidth=0.7,
        hatch="..",
        label="Prior Disabled",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.2f}" for c in df["confidence"].to_numpy()])
    ax.set_xlabel("Confidence Level", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Fraction of Iterations", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig = ax.figure
        fig.legend(
            handles,
            labels,
            fontsize=LEGEND_FONT_SIZE,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    ax.grid(False)
    if title:
        ax.set_title(title)
    if created_ax:
        plt.tight_layout(rect=[0, 0, 0.78, 1])
    else:
        ax.figure.tight_layout(rect=[0, 0, 0.78, 1])
    return ax










#figure 5 safety check the main top panels 
#%%
good_readout = {
        "constraints": [
            {"var": "x4", "range": [0, 0.1], "reason": "ptsa too high", "penalty": 8.0},
            # {"var": "x4", "range": [0.18, 0.3], "reason": "ptsa too high", "penalty": 8.0},
            {"var": "x1", "range": [200, 300.0], "reason": "high amine suppresses yield", "penalty": 2.0},
        ],
    }

random_readout = {
        "effects": {
            "x1": {"effect": "increasing", "scale": 0.35, "confidence": 0.25, "range_hint": [90.0, 150.0]},
            "x4": {"effect": "decreasing", "scale": 0.3, "confidence": 0.3, "range_hint": [0.02, 0.06]},
        },
        "bumps": [
            {"mu": [150.0, 220.0, 240.0, 0.08], "sigma": [20.0, 25.0, 30.0, 0.02], "amp": 0.08}
        ],
        "constraints": [
            {"var": "x2", "range": [120.0, 160.0], "reason": "random exclusion", "penalty": 5.0},
            {"var": "x3", "range": [260.0, 300.0], "reason": "random exclusion", "penalty": 4.0},
        ],
    }

if __name__ == "__main__":
    domain = build_continuous_domain()

    n_init = 1
    n_iter = 100
    repeats = 3
    seed = 46
    constraint_hardness = 0.2
    constraint_pool_size = 20000
    awcd_top_frac = 0.05
    awcd_constraint_threshold = 0.2
    awcd_mean_threshold = 0.2
    awcd_warmup = 1
    awcd_window = 1
    

    bad_readout = {
        "effects": {
            "x4": {"effect": "decreasing", "scale": 0.8, "confidence": 0.9, "range_hint": [0, 0.03]},
        },
        "constraints": [
            {"var": "x4", "range": [0.10, 0.3], "reason": "push away from optimum", "penalty": 2.0},
        ],
    }

    readout_ = good_readout


    baseline_cache = "baseline_cache.csv"
    if os.path.exists(baseline_cache):
        hist_baseline = pd.read_csv(baseline_cache)
        hist_baseline["method"] = "baseline"
        print(f"Loaded baseline cache from {baseline_cache}")
    else:
        hist_baseline = run_baseline_ei_continuous(
            domain,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            repeats=repeats,
            init_method="sobol",
        )
        hist_baseline = hist_baseline.copy()
        hist_baseline["method"] = "baseline"
        hist_baseline.to_csv(baseline_cache, index=False)
        print(f"Saved baseline cache to {baseline_cache}")
    hist_forced = run_ensemble_continuous(
        domain,
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
        repeats=repeats,
        manual_readout=readout_,
        init_method="sobol",
        prior_strength=1.0,
        constraint_hardness=constraint_hardness,
        constraint_pool_size=constraint_pool_size,
        awcd_top_frac=awcd_top_frac,
        awcd_constraint_threshold=awcd_constraint_threshold,
        awcd_mean_threshold=awcd_mean_threshold,
        awcd_warmup=awcd_warmup,
        awcd_window=awcd_window,
        early_prior_boost=True,
        early_prior_steps=5,
        weight_mode="believer",
        method_tag="forced_prior",
    )
    hist_ensemble = run_ensemble_continuous(
        domain,
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
        repeats=repeats,
        manual_readout=readout_,
        init_method="sobol",
        prior_strength=1.0,
        constraint_hardness=constraint_hardness,
        constraint_pool_size=constraint_pool_size,
        awcd_top_frac=awcd_top_frac,
        awcd_constraint_threshold=awcd_constraint_threshold,
        awcd_mean_threshold=awcd_mean_threshold,
        awcd_warmup=awcd_warmup,
        awcd_window=awcd_window,
        early_prior_boost=True,
        early_prior_steps=5,
        weight_mode="ensemble",
        method_tag="ensemble",
    )

#%%
hist_all = pd.concat([hist_baseline, hist_forced, hist_ensemble], ignore_index=True)
hist_all.to_csv("safety_ensemble_history.csv", index=False)
print("Saved safety run history to safety_ensemble_history.csv")

plot_runs_mean_lookup(
    hist_all,
    methods=["baseline", "forced_prior", "ensemble"],
    ci="sem",
    title="Best-so-far (Baseline vs Forced Prior vs Ensemble)",
    show_auc_text=True,
)
plot_awcd_detection_timeline(
    hist_all,
    threshold=awcd_constraint_threshold,
    title="AWCD Detection Timeline",
)
plot_prior_switching_behavior(
    hist_all,
    threshold=awcd_constraint_threshold,
    title="Prior Weight Switching Behavior",
)
plot_awcd_performance_comparison(
    hist_all,
    title="AWCD Guard vs Baseline vs Forced Prior",
)
plot_awcd_auc_comparison(
    hist_all,
    methods=["baseline", "forced_prior", "ensemble"],
    auc_iters=n_iter,
    title="AUC Comparison (Baseline vs PSBO-No Guard vs PSBO-Guarded)",
)
plot_awcd_topk_pressure_map(
    hist_all,
    domain,
    readout_,
    method="ensemble",
    top_frac=awcd_top_frac,
    title="Top-K Pressure Map (AWCD)",
)
plot_awcd_pressure_histogram(
    hist_all,
    domain,
    readout_,
    method="ensemble",
    title="Pressure Histogram (Allowed vs Forbidden)",
)
plot_awcd_components_timeline(
    hist_all,
    title="AWCD Components Over Time",
)
plot_awcd_acquisition_influence_timeline(
    hist_all,
    domain,
    readout_,
    method="ensemble",
    prior_strength=1.0,
    constraint_hardness=constraint_hardness,
    pool_n=1500,
    seed=seed,
    title="Acquisition Influence Over Time",
)
plot_awcd_prior_rank_scatter(
    hist_all,
    domain,
    readout_,
    method="ensemble",
    top_frac=awcd_top_frac,
    title="Prior Rank vs GP Mean (Top-K Pressure)",
)
plt.show()



# figure 5 top side : readout visualization section here******************8 put the code here.
#%%

RUN_READOUT_VIZ = False
if RUN_READOUT_VIZ:
    if "domain" not in globals():
        domain = build_continuous_domain()
    readout_sets = {}
    if "good_readout" in globals():
        readout_sets["Good Readout"] = good_readout
    if "bad_readout" in globals():
        readout_sets["Bad Readout"] = bad_readout
    if "random_readout" in globals():
        readout_sets["Random Readout"] = random_readout

    for name, ro in readout_sets.items():
        fig = plot_readout_map(
            ro,
            domain,
            title=f"{name}: Readout Map vs Oracle",
            show_truth=True,
            truth_top_frac=0.1,
            truth_samples=2048,
            truth_seed=42,
        )
        fig.savefig(f"readout_map_{name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)



# to visualize the confidence level Vs. fractions
#%% 
n_init = 100
seeds = [13,46,59,21,55,64]
readouts = [
    ("good", good_readout),
    ("random", random_readout),
    ("bad", bad_readout),
]



for seed in seeds:
    for readout_name, readout1_ in readouts:
        RUN_AWCD_ACTIVE_SWEEP = True
        if RUN_AWCD_ACTIVE_SWEEP:
            if "domain" not in globals():
                domain = build_continuous_domain()
            confidence_list = [0.2, 0.4, 0.6, 0.7, 0.8, 0.95]
            
            active_summary = awcd_prior_active_ratio_sweep(
                domain,
                readout=readout1_,
                confidences=confidence_list,
                n_init=n_init,
                n_iter=n_iter,
                repeats=repeats,
                seed=seed,
                constraint_hardness=constraint_hardness,
                constraint_pool_size=constraint_pool_size,
                awcd_top_frac=awcd_top_frac,
                awcd_warmup=awcd_warmup,
                awcd_window=awcd_window,
                early_prior_boost=True,
                early_prior_steps=5,
            )
            ax = plot_awcd_prior_active_ratio(
                active_summary,
                title=f"Prior Active Fraction vs Confidence ({readout_name}, seed={seed})",
            )
            ax.figure.savefig(
                f"awcd_active_ratio_{readout_name}_seed_{seed}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(ax.figure)





# influence chart sweep (AUC of acquisition influence vs confidence)
#%%

seeds = [54]



n_init = 1
n_iter = 100
repeats = 3
seed = 46
constraint_hardness = 0.2
constraint_pool_size = 20000
awcd_top_frac = 0.05
awcd_warmup = 1
awcd_window = 1

good_readout = {
        "constraints": [
            {"var": "x4", "range": [0, 0.1], "reason": "ptsa too high", "penalty": 8.0},
            # {"var": "x4", "range": [0.18, 0.3], "reason": "ptsa too high", "penalty": 8.0},
            {"var": "x1", "range": [200, 300.0], "reason": "high amine suppresses yield", "penalty": 2.0},
        ],
    }

random_readout = {
        "effects": {
            "x1": {"effect": "increasing", "scale": 0.35, "confidence": 0.25, "range_hint": [90.0, 150.0]},
            "x4": {"effect": "decreasing", "scale": 0.3, "confidence": 0.3, "range_hint": [0.02, 0.06]},
        },
        "bumps": [
            {"mu": [150.0, 220.0, 240.0, 0.08], "sigma": [20.0, 25.0, 30.0, 0.02], "amp": 0.08}
        ],
        "constraints": [
            {"var": "x2", "range": [120.0, 160.0], "reason": "random exclusion", "penalty": 5.0},
            {"var": "x3", "range": [260.0, 300.0], "reason": "random exclusion", "penalty": 4.0},
        ],
    }

readouts = [
    ("good", good_readout),
    ("random", random_readout),
    ("bad", bad_readout),
]
RUN_INFLUENCE_CHART = True
if RUN_INFLUENCE_CHART:
    if "domain" not in globals():
        domain = build_continuous_domain()
    confidence_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
    pool_n = 1024
    for readout_name, readout1_ in readouts:
        per_seed_rows: List[Dict[str, Any]] = []
        for conf in confidence_list:
            for seed_val in seeds:
                hist = run_ensemble_continuous(
                    domain,
                    n_init=n_init,
                    n_iter=n_iter,
                    seed=seed_val,
                    repeats=3,
                    manual_readout=readout1_,
                    init_method="sobol",
                    prior_strength=1.0,
                    constraint_hardness=constraint_hardness,
                    constraint_pool_size=constraint_pool_size,
                    awcd_top_frac=awcd_top_frac,
                    awcd_constraint_threshold=float(conf),
                    awcd_mean_threshold=float(conf),
                    awcd_warmup=awcd_warmup,
                    awcd_window=awcd_window,
                    early_prior_boost=True,
                    early_prior_steps=5,
                    weight_mode="ensemble",
                    method_tag="ensemble",
                )
                rows = _acquisition_influence_rows(
                    hist,
                    domain,
                    readout1_,
                    method="ensemble",
                    prior_strength=1.0,
                    constraint_hardness=constraint_hardness,
                    pool_n=pool_n,
                    seed=int(seed_val),
                )
                if not rows:
                    continue
                df_inf = pd.DataFrame(rows)
                per_seed_auc = df_inf.groupby("seed")["influence"].sum().reset_index()
                for _, row in per_seed_auc.iterrows():
                    per_seed_rows.append(
                        {
                            "readout": readout_name,
                            "confidence": float(conf),
                            "seed": int(row["seed"]),
                            "influence_auc": float(row["influence"]),
                        }
                    )

        if not per_seed_rows:
            continue
        per_seed_df = pd.DataFrame(per_seed_rows)
        per_seed_df.to_csv(f"influence_chart_{readout_name}_runs.csv", index=False)

        summary = (
            per_seed_df.groupby("confidence")["influence_auc"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("confidence")
        )
        summary["sem"] = summary["std"] / np.maximum(summary["count"], 1) ** 0.5
        summary.to_csv(f"influence_chart_{readout_name}_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        x = np.arange(len(summary))
        y = summary["mean"].to_numpy()
        err = summary["sem"].to_numpy()
        ax.bar(x, y, yerr=err, capsize=4, color="#2c3e50", edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.2f}" for c in summary["confidence"].to_numpy()])
        ax.set_xlabel("Confidence Level", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel("Influence AUC", fontsize=AXIS_LABEL_SIZE)
        ax.set_title(f"Influence Chart ({readout_name})")
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"influence_chart_{readout_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


#figure ESI for the confidence study
#%%
RUN_THRESHOLD_SWEEP = True

good_readout = {
        "constraints": [
            {"var": "x4", "range": [0, 0.1], "reason": "ptsa too high", "penalty": 8.0},
            # {"var": "x4", "range": [0.18, 0.3], "reason": "ptsa too high", "penalty": 8.0},
            {"var": "x1", "range": [200, 300.0], "reason": "high amine suppresses yield", "penalty": 2.0},
        ],
    }

random_readout = {
        "effects": {
            "x1": {"effect": "increasing", "scale": 0.35, "confidence": 0.25, "range_hint": [90.0, 150.0]},
            "x4": {"effect": "decreasing", "scale": 0.3, "confidence": 0.3, "range_hint": [0.02, 0.06]},
        },
        "bumps": [
            {"mu": [150.0, 220.0, 240.0, 0.08], "sigma": [20.0, 25.0, 30.0, 0.02], "amp": 0.08}
        ],
        "constraints": [
            {"var": "x2", "range": [120.0, 160.0], "reason": "random exclusion", "penalty": 5.0},
            {"var": "x3", "range": [260.0, 300.0], "reason": "random exclusion", "penalty": 4.0},
        ],
    }

bad_readout = {
    "effects": {
        "x4": {"effect": "decreasing", "scale": 0.8, "confidence": 0.9, "range_hint": [0, 0.03]},
    },
    "constraints": [
        {"var": "x4", "range": [0.10, 0.3], "reason": "push away from optimum", "penalty": 2.0},
    ],
}



domain = build_continuous_domain()

n_init = 1
n_iter = 100
repeats = 10
seed = 452
constraint_hardness = 0.2
constraint_pool_size = 20000
awcd_top_frac = 0.05
awcd_warmup = 1
awcd_window = 1





if RUN_THRESHOLD_SWEEP:
    threshold_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]


    
    run_awcd_threshold_sweep(
        # domain,
        # {
        #     "good": good_readout,
        #     "bad": bad_readout,
        #     "random": random_readout,
        # },

        domain,
        {
            "bad": bad_readout,
        },
        threshold_list,
        n_init=n_init,
        n_iter=n_iter,
        repeats=repeats,
        seed=seed,
        constraint_hardness=constraint_hardness,
        constraint_pool_size=constraint_pool_size,
        awcd_top_frac=awcd_top_frac,
        awcd_warmup=awcd_warmup,
        awcd_window=awcd_window,
        baseline_cache=baseline_cache,
        out_dir="awcd_sweep_ugi_seed"+str(seed)+"repeat 10",
    )





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
            {"var": "x4", "range": [0.10, 0.3], "reason": "push away from optimum", "penalty": 2.0},
        ],
    }

    hist_bad_base = run_ensemble_continuous(
        domain,
        n_init=6,
        n_iter=100,
        seed=43,
        repeats=3,
        manual_readout=bad_readout,
        init_method="sobol",
        prior_strength=1.0,
        constraint_hardness=0.3,
        constraint_pool_size=20000,
        early_prior_boost=True,
        early_prior_steps=5,
        weight_mode="skeptic",
        method_tag="baseline_bad",
    )
    hist_bad_forced = run_ensemble_continuous(
        domain,
        n_init=6,
        n_iter=100,
        seed=43,
        repeats=3,
        manual_readout=bad_readout,
        init_method="sobol",
        prior_strength=1.0,
        constraint_hardness=0.3,
        constraint_pool_size=20000,
        early_prior_boost=True,
        early_prior_steps=5,
        weight_mode="believer",
        method_tag="forced_bad",
    )
    hist_bad_awcd = run_ensemble_continuous(
        domain,
        n_init=6,
        n_iter=100,
        seed=43,
        repeats=3,
        manual_readout=bad_readout,
        init_method="sobol",
        prior_strength=1.0,
        constraint_hardness=0.3,
        constraint_pool_size=20000,
        early_prior_boost=True,
        early_prior_steps=5,
        weight_mode="ensemble",
        method_tag="awcd_guard",
    )

    bad_hist = pd.concat([hist_bad_base, hist_bad_forced, hist_bad_awcd], ignore_index=True)
    plot_runs_mean_lookup(
        bad_hist,
        methods=["baseline_bad", "forced_bad", "awcd_guard"],
        ci="sem",
        title="Bad prior: AWCD guard comparison (best-so-far)",
    )
    plt.show()




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

    ##############



#Figure 6 panel B
#%%
plot_prior_surface_heatmaps_ugi(
        csv_path="ugi_merged_dataset.csv",
        target_col="yield",
        novice_fraction=0.1,
        mid_fraction=0.50,
        expert_fraction=0.90,
        # focus_features=["amine_mM", "ptsa"],
        n_grid=80,
        n_opt_samples=1024,
        seed=50,
    )
# %%



#Figure 6 panel C

#%%

PORTION_BENCH = True
if PORTION_BENCH:
    fractions = [0.10, 0.60, 0.90, 0.99]
    portion_hist, portion_summary, baseline_summary, baseline_hist = portion_benchmark(
        domain,
        csv_path="ugi_merged_dataset.csv",
        fractions=fractions,
        data_seed=14,
        target_col="yield",
        llm_model="gpt-4o-mini",
        llm_temperature=0.2,
        n_init=1,
        n_iter=100,
        repeats=5,
        seed=1,
        constraint_hardness=0.2,
        constraint_pool_size=20000,
        use_alignment_guard=False,
        alignment_min=0.1,
        early_prior_boost=True,
        early_prior_steps=5,
        prior_strength=1,
        auc_iters=20,
    )
    plot_fraction_auc_bars(
        portion_summary,
        baseline_summary=baseline_summary,
        title="Prior Fraction vs AUC (Best-so-far)",
    )
    plot_fraction_best_so_far(
        portion_hist,
        baseline_df=baseline_hist,
        ci="sem",
        title="Best-so-far by Fraction (Hybrid vs Baseline)",
    )
    plot_fraction_best_so_far_panels(
        portion_hist,
        baseline_df=baseline_hist,
        ci="sem",
        ncols=2,
        title="Best-so-far per Fraction (Hybrid vs Baseline)",
    )
    frac_group_cols = ["fraction"]
    if "method" in portion_hist.columns:
        frac_group_cols.append("method")
    frac_stats = _best_so_far_curve_stats(portion_hist, frac_group_cols)
    if "method" not in frac_stats.columns:
        frac_stats["method"] = "hybrid_manual"
    if baseline_hist is not None and not baseline_hist.empty:
        base_group_cols = ["method"] if "method" in baseline_hist.columns else []
        base_stats = _best_so_far_curve_stats(baseline_hist, base_group_cols)
        if "method" not in base_stats.columns:
            base_stats["method"] = "baseline_ei"
        base_stats["fraction"] = np.nan
        frac_stats = pd.concat([frac_stats, base_stats], ignore_index=True)
    frac_stats.to_csv("fig6_panel_c_best_so_far_curves.csv", index=False)
    plt.show()


# %%

# LLM portion benchmark (multi-model)
#%%
PORTION_BENCH_LLMS = True
if PORTION_BENCH_LLMS:
    fractions = [0.10, 0.60, 0.90, 0.99]
    llm_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.1-mini",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5-20251101",
        "llama-model-data",
    ]
    llm_hist, llm_summary, llm_auc, llm_base_summary, llm_base_hist = portion_benchmark_llms(
        domain,
        csv_path="ugi_merged_dataset.csv",
        fractions=fractions,
        llm_models=llm_models,
        data_seed=124,
        target_col="yield",
        llm_temperature=0.2,
        n_init=1,
        n_iter=100,
        repeats=5,
        seed=667,
        constraint_hardness=0.2,
        constraint_pool_size=20000,
        use_alignment_guard=False,
        alignment_min=0.1,
        early_prior_boost=True,
        early_prior_steps=5,
        prior_strength=1,
        auc_iters=20,
        output_dir="llm_study_data",
    )
    for model in llm_models:
        safe = _safe_filename(model)
        model_summary = llm_summary[llm_summary["llm_model"] == model]
        if model_summary.empty:
            continue
        plot_fraction_auc_bars(
            model_summary,
            baseline_summary=llm_base_summary,
            title=f"Prior Fraction vs AUC ({model})",
        )
        plt.savefig(os.path.join("llm_study_data", f"auc_bars_{safe}.png"), dpi=150, bbox_inches="tight")
        plt.show()

# %%

# Figure 6 panel D
#%%
INIT_BENCH = True
if INIT_BENCH:
    n_inits = [1, 5, 10, 20, 40, 80]
    init_hist, init_summary = init_benchmark(
        domain,
        csv_path="ugi_merged_dataset.csv",
        fraction=0.1,
        n_inits=n_inits,
        data_seed=14,
        target_col="yield",
        llm_model="gpt-4o-mini",
        llm_temperature=0.2,
        n_iter=100,
        repeats=5,
        seed=1,
        constraint_hardness=0.2,
        constraint_pool_size=20000,
        use_alignment_guard=True,
        alignment_min=0.1,
        early_prior_boost=True,
        early_prior_steps=5,
        prior_strength=1,
        auc_iters=20,
    )

    plot_init_benchmark_bars(
        init_summary,
        title="Init Points vs AUC (Prior-Shaped vs Baseline)",
    )
    plot_init_best_so_far_panels(
        init_hist,
        ncols=3,
        ci="sem",
        title="Best-so-far by n_init (Hybrid vs Baseline)",
    )
    init_group_cols = ["n_init"]
    if "method" in init_hist.columns:
        init_group_cols.append("method")
    init_stats = _best_so_far_curve_stats(init_hist, init_group_cols)
    if "method" not in init_stats.columns:
        init_stats["method"] = "hybrid_manual"
    init_stats.to_csv("fig6_panel_d_best_so_far_curves_frac=90%.csv", index=False)
    
    plt.show()



readout_for_plot = {
 "constraints": [
                {"var": "x4", "range": [0, 0.1], "reason": "ptsa too high", "penalty": 8.0},
                # {"var": "x4", "range": [0.18, 0.3], "reason": "ptsa too high", "penalty": 8.0},
                {"var": "x1", "range": [200, 300.0], "reason": "high amine suppresses yield", "penalty": 2.0},
            ],
        }

readout_for_plot = readout_for_plot
ITER_SUCCESS_BENCH = False
if ITER_SUCCESS_BENCH:
    if readout_for_plot is None:
        raise ValueError("iter_success_bench needs a manual readout.")
    iter_list = [5, 10, 15, 20, 50, 100]
    success_summary = iter_success_bench(
        domain,
        iter_list=iter_list,
        manual_readout=readout_for_plot,
        n_init=6,
        seed=31,
        repeats=5,
        target_frac=0.9,
        constraint_hardness=0.2,
        constraint_pool_size=20000,
        use_alignment_guard=False,
        alignment_min=0.1,
        early_prior_boost=True,
        early_prior_steps=5,
        prior_strength=1,
    )
    plot_iter_success_rates(
        success_summary,
        title="Success rate vs Iteration budget",
    )
    plt.show()
# %%

# %%
