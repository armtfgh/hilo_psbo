"""
Adaptive-AWCD Bayesian-optimization runs.

These drivers replicate the inner-loop logic of the existing
``_run_ensemble_continuous_single`` (UGI) and ``_run_ensemble_lookup_single``
(P3HT) but expose the AWCD thresholds as live state controlled by a
:class:`llm_controller.CuserController` instance. The controller is queried
every ``update_every`` iterations with the most recent telemetry, and its
``new_cuser`` value is used as the AWCD constraint+mean threshold from that
checkpoint until the next.

We deliberately re-use every helper function from the original benchmark
modules (loaded via the same truncated-import trick as
``revision_figures.py``) so this file does not duplicate any BO logic. The
*only* duplicated code is the iteration driver itself, which has to be
re-implemented to thread the controller hook through it.

Both ``run_adaptive_ugi`` and ``run_adaptive_p3ht`` return a pandas
DataFrame in the same shape as the original ``run_ensemble_*`` outputs,
with three extra columns:

    controller_name        - name of the controller used
    cuser_in_use           - C_user value active at this iteration
    controller_rationale   - rationale string from the most recent decision

The drivers run a single seed each; callers (``revision_figures.py``) loop
over seeds to build a multi-run history.
"""
from __future__ import annotations

import os
import re
import sys
import types
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from llm_controller import ControllerDecision, CuserController
from safe_gp import safe_fit_gp


# ---------------------------------------------------------------------------
# import the two benchmark modules without executing their post-__main__
# bare top-level code (same trick revision_figures.py uses)
# ---------------------------------------------------------------------------
def _load_module_no_main(path: str, name: str) -> types.ModuleType:
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'^if __name__ == ["\']__main__["\']\s*:', src, flags=re.MULTILINE)
    if m:
        src = src[:m.start()]
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__file__ = path
    code = compile(src, path, "exec")
    sys.modules[name] = mod
    exec(code, mod.__dict__)
    return mod


_HERE = os.path.dirname(os.path.abspath(__file__))
_ugi = _load_module_no_main(
    os.path.join(_HERE, "main_benchmark_portion_new_safety.py"),
    "main_benchmark_portion_new_safety",
)
_p3ht = _load_module_no_main(
    os.path.join(_HERE, "main_benchmark_portion_new_safety_p3ht.py"),
    "main_benchmark_portion_new_safety_p3ht",
)

# helpers (resolved at import time so we fail fast if any are renamed)
_evaluate_oracle = _ugi.evaluate_oracle
_record_continuous_sample = _ugi._record_continuous_sample
_awcd_full_score = _ugi._awcd_full_score
_sample_initial_unit = _ugi._sample_initial_unit
_WeightedExpectedImprovement_ugi = _ugi.WeightedExpectedImprovement

_select_initial_indices_lookup = _p3ht._select_initial_indices_lookup
_build_initial_from_indices = _p3ht._build_initial_from_indices
_record_lookup_sample = _p3ht._record_lookup_sample
_awcd_full_score_lookup = _p3ht._awcd_full_score_lookup
_WeightedExpectedImprovement_p3ht = _p3ht.WeightedExpectedImprovement

# pull modules that both files import (they re-export these as module-level
# bindings after their own ``from ... import`` lines)
_flat_readout_ugi = _ugi.flat_readout
_normalize_readout_to_unit_box_ugi = _ugi.normalize_readout_to_unit_box
_readout_to_prior_ugi = _ugi.readout_to_prior
_fit_residual_gp_ugi = _ugi.fit_residual_gp
_GPWithPriorMean_ugi = _ugi.GPWithPriorMean

_flat_readout_p3ht = _p3ht.flat_readout
_normalize_readout_to_unit_box_p3ht = _p3ht.normalize_readout_to_unit_box
_readout_to_prior_p3ht = _p3ht.readout_to_prior
_fit_residual_gp_p3ht = _p3ht.fit_residual_gp
_GPWithPriorMean_p3ht = _p3ht.GPWithPriorMean

DEVICE = _ugi.DEVICE
DTYPE = _ugi.DTYPE


# ---------------------------------------------------------------------------
# helpers shared by both adaptive runs
# ---------------------------------------------------------------------------
def _linear_slope(values: List[float]) -> float:
    """Slope of an ordinary least-squares fit of ``values`` vs index.
    ``values`` is assumed to be in chronological (oldest -> newest) order."""
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 0.0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _build_telemetry(
    *,
    dataset: str,
    iteration: int,
    total_iterations: int,
    current_cuser: float,
    prior_active: bool,
    expert_prompt: str,
    readout: Dict[str, Any],
    awcd_history: List[float],
    awcd_history_constraint: List[float],
    awcd_history_mean: List[float],
    best_so_far_history: List[float],
    prior_active_history: List[bool],
    initial_best: Optional[float],
    stagnation_iters: int,
    previous_decisions: List[Dict[str, Any]],
    n_history: int = 12,
    window: int = 5,
) -> Dict[str, Any]:
    """Assemble the payload sent to the controller. Lists in the returned
    dict are most-recent-first; trend signals are precomputed scalars so
    the LLM does not have to read trends out of raw lists.
    """
    # --- best-so-far improvement signals --------------------------------
    bsf_relative_gain = 0.0
    bsf_improved_in_window = False
    if best_so_far_history:
        latest = float(best_so_far_history[-1])
        if initial_best is not None:
            base = float(initial_best)
            denom = max(abs(base), 1e-6)
            bsf_relative_gain = (latest - base) / denom
        window_slice = best_so_far_history[-int(max(window, 2)):]
        if len(window_slice) >= 2:
            bsf_improved_in_window = window_slice[-1] > window_slice[0] + 1e-9

    # --- AWCD pressure recent mean + slope -------------------------------
    recent_pressure = awcd_history_constraint[-int(max(window, 2)):]
    awcd_pressure_recent_mean = (float(np.mean(recent_pressure))
                                  if recent_pressure else 0.0)
    awcd_pressure_trend_slope = _linear_slope(recent_pressure)

    # --- prior-disabled streak ------------------------------------------
    streak = 0
    for active in reversed(prior_active_history or []):
        if bool(active):
            break
        streak += 1

    return {
        "dataset": dataset,
        "iteration": int(iteration),
        "total_iterations": int(total_iterations),
        "current_cuser": float(current_cuser),
        "prior_active": bool(prior_active),
        "expert_prompt": str(expert_prompt or ""),
        "readout": dict(readout or {}),
        "awcd_score_history": list(reversed(awcd_history[-n_history:])),
        "awcd_constraint_history": list(reversed(awcd_history_constraint[-n_history:])),
        "awcd_mean_history": list(reversed(awcd_history_mean[-n_history:])),
        "best_so_far_history": list(reversed(best_so_far_history[-n_history:])),
        "stagnation_iters": int(stagnation_iters),
        "previous_decisions": list(reversed(previous_decisions[-6:])),
        "n_history": int(n_history),
        # precomputed trend signals
        "bsf_relative_gain": float(bsf_relative_gain),
        "bsf_improved_in_window": bool(bsf_improved_in_window),
        "awcd_pressure_recent_mean": float(awcd_pressure_recent_mean),
        "awcd_pressure_trend_slope": float(awcd_pressure_trend_slope),
        "prior_disabled_streak": int(streak),
    }


def _update_stagnation(stag_counter: int, best_history: List[float]) -> int:
    if len(best_history) < 2:
        return 0
    if best_history[-1] > best_history[-2] + 1e-9:
        return 0
    return stag_counter + 1


# ---------------------------------------------------------------------------
# UGI continuous adaptive run
# ---------------------------------------------------------------------------
def run_adaptive_ugi(
    domain: Any,
    *,
    controller: CuserController,
    manual_readout: Dict[str, Any],
    expert_prompt: str,
    n_init: int = 1,
    n_iter: int = 100,
    seed: int = 0,
    init_method: str = "sobol",
    prior_strength: float = 1.0,
    constraint_hardness: float = 0.2,
    constraint_pool_size: int = 20000,
    awcd_top_frac: float = 0.05,
    awcd_warmup: int = 1,
    awcd_window: int = 1,
    awcd_pool_size: int = 1024,
    early_prior_boost: bool = True,
    early_prior_steps: int = 5,
    initial_cuser: float = 0.95,
    update_every: int = 10,
    method_tag: Optional[str] = None,
) -> pd.DataFrame:
    method_tag = method_tag or f"adaptive_{controller.name}"

    X_init = _sample_initial_unit(domain, n_init, seed, init_method)
    Y_init = _evaluate_oracle(domain, X_init).unsqueeze(-1)
    X_obs = X_init.clone()
    Y_obs = Y_init.clone()
    recs: List[Dict[str, Any]] = []
    best = float(Y_obs.max().item()) if Y_obs.numel() else float("-inf")

    for i in range(n_init):
        y = float(Y_init[i].item())
        best = max(best, y)
        recs.append(_record_continuous_sample(
            domain, X_init[i], y, best, method=method_tag, iteration=i - n_init,
            extra={"cuser_in_use": float(initial_cuser),
                   "controller_name": controller.name,
                   "controller_rationale": "init"},
        ))

    ro_raw = manual_readout if manual_readout is not None else _flat_readout_ugi(
        feature_names=domain.feature_names)
    ro_unit = _normalize_readout_to_unit_box_ugi(
        ro_raw, domain.mins, domain.maxs, feature_names=domain.feature_names)
    prior = _readout_to_prior_ugi(ro_unit, feature_names=domain.feature_names)

    awcd_history: List[float] = []
    awcd_history_constraint: List[float] = []
    awcd_history_mean: List[float] = []
    best_so_far_history: List[float] = []
    prior_active_history: List[bool] = []
    decisions: List[Dict[str, Any]] = []
    cuser_in_use = float(initial_cuser)
    cur_rationale = "initial"
    stag_counter = 0
    initial_best = float(best) if best != float("-inf") else None

    for t in range(n_iter):
        gp_skeptic, _mll_s = safe_fit_gp(X_obs, Y_obs)

        gp_resid, alpha = _fit_residual_gp_ugi(X_obs, Y_obs, prior)
        m0_scale = float(alpha * prior_strength)
        model_believer = _GPWithPriorMean_ugi(gp_resid, prior, m0_scale=m0_scale)

        awcd_metrics = _awcd_full_score(
            gp_skeptic, prior.m0_torch, X_obs=X_obs, Y_obs=Y_obs,
            readout_unit=ro_unit, feature_names=domain.feature_names,
            pool_n=awcd_pool_size, top_frac=awcd_top_frac,
            seed=seed + 9000 + t, bounds=domain.unit_bounds,
        )
        awcd_score = float(awcd_metrics.get("awcd_total", 0.0))
        awcd_constraint = float(awcd_metrics.get("awcd_constraint", 0.0))
        awcd_mean = float(awcd_metrics.get("awcd_mean", 0.0))
        awcd_history.append(awcd_score)
        awcd_history_constraint.append(awcd_constraint)
        awcd_history_mean.append(awcd_mean)
        awcd_score_mean = float(np.mean(awcd_history[-int(awcd_window):]))
        awcd_constraint_mean = float(np.mean(awcd_history_constraint[-int(awcd_window):]))
        awcd_mean_mean = float(np.mean(awcd_history_mean[-int(awcd_window):]))

        # --- controller hook every `update_every` iters ------------------
        if t >= int(awcd_warmup) and t > 0 and (t % int(update_every) == 0):
            telemetry = _build_telemetry(
                dataset="UGI",
                iteration=t,
                total_iterations=n_iter,
                current_cuser=cuser_in_use,
                prior_active=(prior_active_history[-1] if prior_active_history else True),
                expert_prompt=expert_prompt,
                readout=manual_readout or {},
                awcd_history=awcd_history,
                awcd_history_constraint=awcd_history_constraint,
                awcd_history_mean=awcd_history_mean,
                best_so_far_history=best_so_far_history,
                prior_active_history=prior_active_history,
                initial_best=initial_best,
                stagnation_iters=stag_counter,
                previous_decisions=decisions,
            )
            dec: ControllerDecision = controller.decide(telemetry)
            cuser_in_use = float(dec.new_cuser)
            cur_rationale = dec.rationale
            decisions.append({
                "iteration": int(t),
                "new_cuser": float(cuser_in_use),
                "rationale": str(cur_rationale),
            })

        # --- AWCD gate using current cuser_in_use ------------------------
        if t < int(awcd_warmup) or len(awcd_history) < int(awcd_window):
            prior_good = True
        else:
            prior_good = not (
                (awcd_constraint_mean > float(cuser_in_use))
                or (awcd_mean_mean > float(cuser_in_use))
            )
        weight = 1.0 if prior_good else 0.0
        prior_active = bool(prior_good)
        prior_active_history.append(prior_active)

        best_f = float(Y_obs.max().item())
        effective_hardness = float(constraint_hardness) if weight > 0.0 else 0.0
        if bool(early_prior_boost) and t < int(early_prior_steps) and weight > 0.0:
            pool = draw_sobol_samples(
                bounds=domain.unit_bounds, n=1024, q=1, seed=seed + 505 + t,
            ).squeeze(1)
            with torch.no_grad():
                prior_vals = prior.m0_torch(pool).reshape(-1)
                idx_local = int(torch.argmax(prior_vals))
                x_next = pool[idx_local]
        else:
            acq = _WeightedExpectedImprovement_ugi(
                gp_skeptic, model_believer, weight=weight, best_f=best_f,
                readout_unit=ro_unit, feature_names=domain.feature_names,
                constraint_hardness=effective_hardness,
            )
            pool_n = max(1024, int(constraint_pool_size))
            pool = draw_sobol_samples(
                bounds=domain.unit_bounds, n=pool_n, q=1, seed=seed + 202 + t,
            ).squeeze(1)
            with torch.no_grad():
                scores = acq(pool.unsqueeze(1)).reshape(-1)
                idx_local = int(torch.argmax(scores))
                x_next = pool[idx_local]

        y_next = _evaluate_oracle(domain, x_next).unsqueeze(-1)
        X_obs = torch.cat([X_obs, x_next.unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, y_next], dim=0)
        y_val = float(y_next.item())
        best = max(best, y_val)
        best_so_far_history.append(best)
        if initial_best is None:
            initial_best = float(best)
        stag_counter = _update_stagnation(stag_counter, best_so_far_history)

        recs.append(_record_continuous_sample(
            domain, x_next, y_val, best, method=method_tag, iteration=t,
            extra={
                "weight_believer": float(weight),
                "awcd_score": float(awcd_score),
                "awcd_mean": float(awcd_score_mean),
                "awcd_constraint": float(awcd_constraint),
                "awcd_constraint_mean": float(awcd_constraint_mean),
                "awcd_mean_disagree": float(awcd_mean),
                "awcd_mean_disagree_mean": float(awcd_mean_mean),
                "prior_active": bool(prior_active),
                "cuser_in_use": float(cuser_in_use),
                "controller_name": controller.name,
                "controller_rationale": str(cur_rationale)[:200],
            },
        ))

    df = pd.DataFrame(recs)
    df["seed"] = seed
    return df


# ---------------------------------------------------------------------------
# P3HT discrete adaptive run
# ---------------------------------------------------------------------------
def run_adaptive_p3ht(
    lookup: Any,
    *,
    controller: CuserController,
    manual_readout: Dict[str, Any],
    expert_prompt: str,
    n_init: int = 1,
    n_iter: int = 20,
    seed: int = 0,
    init_method: str = "sobol",
    prior_strength: float = 1.0,
    constraint_hardness: float = 0.2,
    constraint_pool_size: int = 20000,  # unused for discrete, kept for parity
    awcd_top_frac: float = 0.05,
    awcd_warmup: int = 1,
    awcd_window: int = 1,
    awcd_pool_size: int = 2048,
    early_prior_boost: bool = True,
    early_prior_steps: int = 5,
    initial_cuser: float = 0.95,
    update_every: int = 2,
    method_tag: Optional[str] = None,
) -> pd.DataFrame:
    method_tag = method_tag or f"adaptive_{controller.name}"

    idxs = _select_initial_indices_lookup(lookup, n_init, seed, init_method)
    seen, X_obs, Y_obs, recs = _build_initial_from_indices(
        lookup, idxs, method_tag=method_tag)
    # tag init records with controller metadata
    for r in recs:
        r["cuser_in_use"] = float(initial_cuser)
        r["controller_name"] = controller.name
        r["controller_rationale"] = "init"
    best = max((row["best_so_far"] for row in recs), default=float("-inf"))

    ro_raw = manual_readout if manual_readout is not None else _flat_readout_p3ht(
        feature_names=lookup.feature_names)
    ro_unit = _normalize_readout_to_unit_box_p3ht(
        ro_raw, lookup.mins, lookup.maxs, feature_names=lookup.feature_names)
    prior = _readout_to_prior_p3ht(ro_unit, feature_names=lookup.feature_names)

    awcd_history: List[float] = []
    awcd_history_constraint: List[float] = []
    awcd_history_mean: List[float] = []
    best_so_far_history: List[float] = []
    prior_active_history: List[bool] = []
    decisions: List[Dict[str, Any]] = []
    cuser_in_use = float(initial_cuser)
    cur_rationale = "initial"
    stag_counter = 0
    initial_best = float(best) if best != float("-inf") else None

    rng = np.random.default_rng(int(seed))

    for t in range(n_iter):
        remaining = list(set(range(lookup.n)) - seen)
        if not remaining:
            break

        gp_skeptic, _mll_s = safe_fit_gp(X_obs, Y_obs)

        gp_resid, alpha = _fit_residual_gp_p3ht(X_obs, Y_obs, prior)
        m0_scale = float(alpha * prior_strength)
        model_believer = _GPWithPriorMean_p3ht(gp_resid, prior, m0_scale=m0_scale)

        pool_idx = rng.choice(lookup.n, size=min(int(awcd_pool_size), lookup.n),
                              replace=False)
        pool = lookup.X[pool_idx].to(device=DEVICE, dtype=DTYPE)
        awcd_metrics = _awcd_full_score_lookup(
            gp_skeptic, prior.m0_torch, X_obs=X_obs, Y_obs=Y_obs,
            readout_unit=ro_unit, feature_names=lookup.feature_names,
            pool=pool, top_frac=awcd_top_frac,
        )
        awcd_score = float(awcd_metrics.get("awcd_total", 0.0))
        awcd_constraint = float(awcd_metrics.get("awcd_constraint", 0.0))
        awcd_mean = float(awcd_metrics.get("awcd_mean", 0.0))
        awcd_history.append(awcd_score)
        awcd_history_constraint.append(awcd_constraint)
        awcd_history_mean.append(awcd_mean)
        awcd_score_mean = float(np.mean(awcd_history[-int(awcd_window):]))
        awcd_constraint_mean = float(np.mean(awcd_history_constraint[-int(awcd_window):]))
        awcd_mean_mean = float(np.mean(awcd_history_mean[-int(awcd_window):]))

        if t >= int(awcd_warmup) and t > 0 and (t % int(update_every) == 0):
            telemetry = _build_telemetry(
                dataset="P3HT",
                iteration=t,
                total_iterations=n_iter,
                current_cuser=cuser_in_use,
                prior_active=(prior_active_history[-1] if prior_active_history else True),
                expert_prompt=expert_prompt,
                readout=manual_readout or {},
                awcd_history=awcd_history,
                awcd_history_constraint=awcd_history_constraint,
                awcd_history_mean=awcd_history_mean,
                best_so_far_history=best_so_far_history,
                prior_active_history=prior_active_history,
                initial_best=initial_best,
                stagnation_iters=stag_counter,
                previous_decisions=decisions,
            )
            dec = controller.decide(telemetry)
            cuser_in_use = float(dec.new_cuser)
            cur_rationale = dec.rationale
            decisions.append({
                "iteration": int(t),
                "new_cuser": float(cuser_in_use),
                "rationale": str(cur_rationale),
            })

        if t < int(awcd_warmup) or len(awcd_history) < int(awcd_window):
            prior_good = True
        else:
            prior_good = not (
                (awcd_constraint_mean > float(cuser_in_use))
                or (awcd_mean_mean > float(cuser_in_use))
            )
        weight = 1.0 if prior_good else 0.0
        prior_active = bool(prior_good)
        prior_active_history.append(prior_active)

        if bool(early_prior_boost) and t < int(early_prior_steps) and weight > 0.0:
            pool_remain = lookup.X[remaining].to(device=DEVICE, dtype=DTYPE)
            with torch.no_grad():
                prior_vals = prior.m0_torch(pool_remain).reshape(-1)
                pick_local = int(torch.argmax(prior_vals))
            idx = int(remaining[pick_local])
        else:
            X_pool = lookup.X[remaining].to(device=DEVICE, dtype=DTYPE)
            best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
            acq = _WeightedExpectedImprovement_p3ht(
                gp_skeptic, model_believer, weight=weight, best_f=best_f,
                readout_unit=ro_unit, feature_names=lookup.feature_names,
                constraint_hardness=constraint_hardness,
            )
            with torch.no_grad():
                scores = acq(X_pool.unsqueeze(1)).reshape(-1)
            pick_local = int(torch.argmax(scores))
            idx = int(remaining[pick_local])

        seen.add(idx)
        x_new = lookup.X[idx].unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
        y_new = lookup.y[idx].to(device=DEVICE, dtype=DTYPE).unsqueeze(0).unsqueeze(-1)
        X_obs = torch.cat([X_obs, x_new], dim=0)
        Y_obs = torch.cat([Y_obs, y_new], dim=0)

        y_val = float(y_new.item())
        best = max(best, y_val)
        best_so_far_history.append(best)
        if initial_best is None:
            initial_best = float(best)
        stag_counter = _update_stagnation(stag_counter, best_so_far_history)

        recs.append(_record_lookup_sample(
            lookup, idx, y_val, best, method=method_tag, iteration=t,
            extra={
                "weight_believer": float(weight),
                "awcd_score": float(awcd_score),
                "awcd_mean": float(awcd_score_mean),
                "awcd_constraint": float(awcd_constraint),
                "awcd_constraint_mean": float(awcd_constraint_mean),
                "awcd_mean_disagree": float(awcd_mean),
                "awcd_mean_disagree_mean": float(awcd_mean_mean),
                "prior_active": bool(prior_active),
                "cuser_in_use": float(cuser_in_use),
                "controller_name": controller.name,
                "controller_rationale": str(cur_rationale)[:200],
            },
        ))

    df = pd.DataFrame(recs)
    df["seed"] = seed
    return df


__all__ = [
    "run_adaptive_ugi",
    "run_adaptive_p3ht",
]
