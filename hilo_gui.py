"""
HILO Dashboard (UGI) - Streamlit GUI
===================================

Run:
  streamlit run hilo_gui.py
"""
#%%
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import html as html_lib

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import AcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from gpytorch.mlls import ExactMarginalLogLikelihood

from data_analysis import build_ugi_ml_oracle, RandomForestOracle
from readout_schema import normalize_readout_to_unit_box, readout_to_prior, flat_readout
from prior_gp import fit_residual_gp, GPWithPriorMean

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

READOUT_CURRENT_PATH = "hilo_readout_current.json"
READOUT_HISTORY_PATH = "hilo_readout_history.jsonl"
HISTORY_CSV_DIR = "hilo_runs"


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _append_readout_history(readout: Dict[str, Any], *, source: str) -> None:
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "readout": readout,
    }
    try:
        with open(READOUT_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _save_current_readout(readout: Dict[str, Any], *, source: str) -> None:
    try:
        with open(READOUT_CURRENT_PATH, "w", encoding="utf-8") as f:
            json.dump(readout, f, indent=2)
        _append_readout_history(readout, source=source)
    except Exception:
        pass


def _read_readout_history(last_k: int = 5) -> List[Dict[str, Any]]:
    if not os.path.exists(READOUT_HISTORY_PATH):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(READOUT_HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    if last_k <= 0:
        return rows
    return rows[-int(last_k) :]


def _save_history_csv(*, tag: str) -> Optional[str]:
    hist = st.session_state.get("history", [])
    if not hist:
        return None
    df = pd.DataFrame(hist)
    os.makedirs(HISTORY_CSV_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hilo_history_{tag}_{ts}.csv"
    path = os.path.join(HISTORY_CSV_DIR, filename)
    df.to_csv(path, index=False)
    return path


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

    best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
    EI = ExpectedImprovement(model=gp_skeptic, best_f=best_f, maximize=True)
    with torch.no_grad():
        post = gp_skeptic.posterior(pool)
        gp_std = post.variance.sqrt().reshape(-1)
        prior_mean = prior_mean_func(pool).reshape(-1)
        ei_vals = EI(pool.unsqueeze(1)).reshape(-1)
        pressure = ei_vals * gp_std
        k = max(1, int(float(top_frac) * float(pressure.numel())))
        top_idx = torch.topk(pressure, k=k).indices
        prior_rank = torch.argsort(torch.argsort(prior_mean, descending=True))
        median_rank = int(prior_mean.numel() // 2)
        mask_mean = prior_rank[top_idx] > median_rank

    awcd_constraint = float(mask[top_idx].float().mean().item())
    awcd_mean = float(mask_mean.float().mean().item())
    return {
        "awcd_constraint": awcd_constraint,
        "awcd_mean": awcd_mean,
        "awcd_total": max(awcd_constraint, awcd_mean),
    }


def _call_openai_chat(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float,
    api_key: Optional[str],
) -> str:
    try:
        import httpx  # type: ignore
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("LLM mode requires `openai` and `httpx` to be installed.") from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key not provided.")

    client = OpenAI(api_key=key, http_client=httpx.Client(verify=False))
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned empty response.")
    return content


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("Could not parse JSON from LLM response.")


def _coerce_number(value: Any, default: float, errors: List[str], ctx: str) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            errors.append(f"{ctx} not numeric: {value!r}")
            return float(default)
    errors.append(f"{ctx} not numeric: {value!r}")
    return float(default)


def _sanitize_readout(readout: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    ro: Dict[str, Any] = dict(readout or {})
    effects = ro.get("effects") or {}
    if isinstance(effects, dict):
        allowed_effects = {
            "increasing",
            "decreasing",
            "flat",
            "nonmonotone-peak",
            "nonmonotone-valley",
        }
        for var, spec in effects.items():
            if not isinstance(spec, dict):
                continue
            effect_val = spec.get("effect")
            if not isinstance(effect_val, str) or effect_val not in allowed_effects:
                errors.append(f"effects.{var}.effect invalid: {effect_val!r}")
                continue
            spec["scale"] = _coerce_number(spec.get("scale", 0.4), 0.4, errors, f"effects.{var}.scale")
            spec["confidence"] = _coerce_number(
                spec.get("confidence", 0.5), 0.5, errors, f"effects.{var}.confidence"
            )
            rh = spec.get("range_hint")
            if isinstance(rh, (list, tuple)) and len(rh) == 2:
                lo = _coerce_number(rh[0], 0.0, errors, f"effects.{var}.range_hint[0]")
                hi = _coerce_number(rh[1], 0.0, errors, f"effects.{var}.range_hint[1]")
                spec["range_hint"] = [lo, hi]
        cleaned = {}
        for k, v in effects.items():
            if not isinstance(v, dict) or not isinstance(v.get("effect"), str):
                continue
            if float(v.get("confidence", 0.0)) <= 0.0:
                continue
            if v.get("effect") == "flat" and float(v.get("scale", 0.0)) <= 0.0:
                continue
            cleaned[k] = v
        effects = cleaned
    ro["effects"] = effects

    bumps = ro.get("bumps") or []
    if isinstance(bumps, list):
        for b in bumps:
            if not isinstance(b, dict):
                continue
            mu = b.get("mu")
            if isinstance(mu, (list, tuple)):
                b["mu"] = [_coerce_number(v, 0.0, errors, "bumps.mu") for v in list(mu)]
            else:
                if mu is None:
                    errors.append("bumps.mu missing; defaulted to 0.0")
                    b["mu"] = [0.0]
                else:
                    b["mu"] = [_coerce_number(mu, 0.0, errors, "bumps.mu")]
            sigma = b.get("sigma")
            if isinstance(sigma, (list, tuple)):
                b["sigma"] = [
                    max(_coerce_number(v, 0.01, errors, "bumps.sigma"), 1e-6)
                    for v in list(sigma)
                ]
            b["amp"] = _coerce_number(b.get("amp", 0.1), 0.1, errors, "bumps.amp")
    ro["bumps"] = bumps

    constraints = ro.get("constraints") or []
    if isinstance(constraints, list):
        for c in constraints:
            if not isinstance(c, dict):
                continue
            r = c.get("range")
            if isinstance(r, (list, tuple)) and len(r) == 2:
                lo = _coerce_number(r[0], 0.0, errors, "constraints.range[0]")
                hi = _coerce_number(r[1], 0.0, errors, "constraints.range[1]")
                c["range"] = [lo, hi]
            c["penalty"] = _coerce_number(c.get("penalty", 8.0), 8.0, errors, "constraints.penalty")
    ro["constraints"] = constraints
    return ro, errors


def _canonical_var_name(name: Any, feature_names: List[str]) -> Optional[str]:
    if name is None:
        return None
    raw = str(name).strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower.startswith("x"):
        try:
            idx = int(lower[1:]) - 1
        except ValueError:
            idx = -1
        if 0 <= idx < len(feature_names):
            return f"x{idx + 1}"
    for idx, fname in enumerate(feature_names):
        if lower == str(fname).lower():
            return f"x{idx + 1}"
    synonyms = {
        "amine": "x1",
        "aldehyde": "x2",
        "isocyanide": "x3",
        "ptsa": "x4",
        "p-tsa": "x4",
        "ptsoh": "x4",
        "p-tsoh": "x4",
    }
    for key, mapped in synonyms.items():
        if key in lower:
            return mapped
    return None


def _normalize_readout_vars(readout: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
    ro = dict(readout or {})
    effects = ro.get("effects") or {}
    new_effects: Dict[str, Any] = {}
    if isinstance(effects, dict):
        for var, spec in effects.items():
            mapped = _canonical_var_name(var, feature_names) or str(var)
            new_effects[mapped] = spec
    ro["effects"] = new_effects

    constraints = ro.get("constraints") or []
    if isinstance(constraints, list):
        for c in constraints:
            if not isinstance(c, dict):
                continue
            mapped = _canonical_var_name(c.get("var"), feature_names)
            if mapped:
                c["var"] = mapped
            reason = str(c.get("reason", "")).lower()
            if "ptsa" in reason or "p-tsa" in reason or "ptsoh" in reason or "p-tsoh" in reason:
                c["var"] = "x4"
            if "amine" in reason:
                c["var"] = "x1"
    ro["constraints"] = constraints
    return ro


def _repair_readout_json(
    raw_json: Dict[str, Any],
    errors: List[str],
    *,
    model: str,
    temperature: float,
    api_key: Optional[str],
) -> Dict[str, Any]:
    system = (
        "Fix the JSON readout so all numeric fields are numbers (floats), "
        "not words. Keep the same structure and keys. Return JSON only."
    )
    user = (
        f"Invalid fields:\n{errors}\n\nJSON:\n{json.dumps(raw_json, indent=2)}"
    )
    content = _call_openai_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
    return _extract_json(content)


def generate_readout_from_guidance(
    guidance: str,
    summary: str,
    *,
    data_context: str,
    current_readout: Optional[Dict[str, Any]] = None,
    readout_history: Optional[List[Dict[str, Any]]] = None,
    model: str,
    temperature: float,
    api_key: Optional[str],
) -> Dict[str, Any]:
    system = (
        "You are a scientific co-pilot translating expert intent into a JSON prior. "
        "You must UPDATE the current JSON readout, not replace it, unless the user explicitly asks to reset. "
        "Return JSON ONLY with keys: effects, bumps, constraints. "
        "Schema:\n"
        "- effects: {var: {effect, scale, confidence, range_hint}}\n"
        "  effect ∈ {increasing, decreasing, flat, nonmonotone-peak, nonmonotone-valley}.\n"
        "  scale in [0.2, 1.5], confidence in [0.3, 0.95].\n"
        "  range_hint is [low, high] in RAW units.\n"
        "- bumps: [{mu, sigma, amp}] where mu/sigma are lists (length 4) in RAW units.\n"
        "- constraints: [{var, range, penalty, reason}] where range is [low, high] RAW units.\n"
        "Rules:\n"
        "- Use variable names x1, x2, x3, x4 (not chemical names).\n"
        "- If user says 'avoid below a' -> constraint [min, a]; 'avoid above b' -> [b, max].\n"
        "- If user says 'keep between a and b' -> two constraints outside the band.\n"
        "- Only create constraints when language is prohibitive (avoid, do not, must not, forbidden).\n"
        "- Preference language (prefer, start with, focus on) should be encoded as effects, not constraints.\n"
        "- Default penalty is 8.0 unless explicitly stated otherwise.\n"
        "- Only include effects/bumps/constraints explicitly supported by guidance; do not invent.\n"
        "- If guidance only requests constraints, return constraints only (effects/bumps empty).\n"
        "- Use the provided ranges and tertiles for low/mid/high wording.\n"
        "- Chat history is chronological with timestamps; latest user instruction overrides earlier ones.\n"
        "- If summary conflicts with the chat, trust the chat.\n"
        "- If asked to remove or revise a prior item, delete or update it in the JSON.\n"
        "- If asked to show the current or previous JSON, include it in your reasoning but still return JSON only.\n"
        "Use only the provided info. Return JSON only."
    )
    history_text = ""
    if readout_history:
        lines = []
        for item in readout_history:
            ts = item.get("ts", "")
            src = item.get("source", "")
            ro = item.get("readout", {})
            lines.append(f"[{ts}] {src}: {json.dumps(ro)}")
        history_text = "\n".join(lines)
    user = (
        f"Domain + data context:\n{data_context}\n\n"
        f"Experiment summary:\n{summary}\n\n"
        f"Expert guidance (chronological):\n{guidance}\n"
        f"\nCurrent JSON readout:\n{json.dumps(current_readout or {}, indent=2)}\n"
        f"\nPrevious readout history (most recent last):\n{history_text or 'None'}\n"
    )
    content = _call_openai_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
    raw = _extract_json(content)
    sanitized, errors = _sanitize_readout(raw)
    if errors:
        try:
            repaired = _repair_readout_json(
                raw,
                errors,
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
            sanitized, errors = _sanitize_readout(repaired)
        except Exception:
            pass
    return sanitized


def generate_llm_summary(
    history: List[Dict[str, Any]],
    *,
    model: str,
    temperature: float,
    api_key: Optional[str],
    last_k: int = 5,
) -> str:
    if not history:
        return "No observations yet."
    df = pd.DataFrame(history)
    best = float(df["best_so_far"].max())
    recent = df.tail(max(1, int(last_k)))
    rows = recent[[col for col in recent.columns if col.startswith("x") or col == "y"]].to_dict("records")
    summary = (
        f"Best so far: {best:.3f}\n"
        f"Recent {len(rows)} observations: {rows}"
    )

    system = (
        "You are an expert chemist summarizing BO results for a collaborator. "
        "Be concise, mention trends or constraints hinted by data."
    )
    content = _call_openai_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": summary},
        ],
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
    return content.strip()


def summarize_history_local(history: List[Dict[str, Any]], last_k: int = 5) -> str:
    if not history:
        return "No observations yet."
    df = pd.DataFrame(history)
    best = float(df["best_so_far"].max())
    recent = df.tail(max(1, int(last_k)))
    cols = [c for c in recent.columns if c.startswith("x")] + ["y"]
    rows = recent[cols].to_dict("records")
    return f"Best so far: {best:.3f}\nRecent {len(rows)} observations: {rows}"


def _append_chat(role: str, content: str) -> None:
    st.session_state.chat_history.append(
        {"role": role, "content": content, "ts": datetime.now().strftime("%H:%M:%S")}
    )


def _build_domain_context(domain: ContinuousDomain) -> str:
    mins = domain.mins.detach().cpu().numpy()
    maxs = domain.maxs.detach().cpu().numpy()
    lines = ["Parameter ranges (raw units):"]
    for i, name in enumerate(domain.feature_names):
        lo = float(mins[i])
        hi = float(maxs[i])
        span = hi - lo
        t1 = lo + span / 3.0
        t2 = lo + 2.0 * span / 3.0
        lines.append(
            f"x{i+1} ({name}): min={lo:.4g}, max={hi:.4g} | "
            f"low=[{lo:.4g}, {t1:.4g}], mid=[{t1:.4g}, {t2:.4g}], high=[{t2:.4g}, {hi:.4g}]"
        )
    return "\n".join(lines)


def _build_history_context(history: List[Dict[str, Any]], last_k: int = 10) -> str:
    if not history:
        return "No observations yet."
    df = pd.DataFrame(history)
    df = df[df["iter"] >= 0]
    if df.empty:
        return "No post-init observations yet."
    best = float(df["best_so_far"].max())
    recent = df.tail(max(1, int(last_k)))
    cols = [c for c in recent.columns if c.startswith("x")] + ["y", "best_so_far"]
    rows = recent[cols].to_dict("records")
    return f"Best-so-far: {best:.4g}\nRecent {len(rows)} observations: {rows}"


def _extract_last_user_text(chat_history: List[Dict[str, Any]]) -> str:
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _is_constraint_intent(text: str) -> bool:
    t = text.lower()
    phrases = [
        "avoid",
        "do not",
        "don't",
        "must not",
        "never",
        "forbid",
        "forbidden",
        "prohibit",
        "ban",
        "no more than",
        "not go",
        "stay out",
        "exclude",
        "keep below",
        "keep above",
    ]
    return any(p in t for p in phrases)


def _apply_preference_overrides(
    readout: Dict[str, Any],
    text: str,
    domain: ContinuousDomain,
) -> Dict[str, Any]:
    t = text.lower()
    if not t.strip():
        return readout
    if _is_constraint_intent(t):
        return readout

    var_map = {
        "x1": "x1",
        "amine": "x1",
        "x2": "x2",
        "aldehyde": "x2",
        "x3": "x3",
        "isocyanide": "x3",
        "x4": "x4",
        "ptsa": "x4",
        "p-tsa": "x4",
        "ptsoh": "x4",
        "p-tsoh": "x4",
    }
    target_var = None
    for key, var in var_map.items():
        if key in t:
            target_var = var
            break
    if target_var is None:
        return readout

    mins = domain.mins.detach().cpu().numpy()
    maxs = domain.maxs.detach().cpu().numpy()
    idx = int(target_var[1:]) - 1
    lo = float(mins[idx])
    hi = float(maxs[idx])
    span = hi - lo
    t1 = lo + span / 3.0
    t2 = lo + 2.0 * span / 3.0

    effects = readout.get("effects") or {}
    if target_var in effects:
        return readout

    effect = None
    range_hint = None
    if any(k in t for k in ["low", "lower", "start low", "lower values"]):
        effect = "decreasing"
        range_hint = [lo, t1]
    elif any(k in t for k in ["high", "higher", "upper"]):
        effect = "increasing"
        range_hint = [t2, hi]
    elif any(k in t for k in ["mid", "middle", "moderate"]):
        effect = "nonmonotone-peak"
        range_hint = [t1, t2]

    if effect and range_hint:
        effects[target_var] = {
            "effect": effect,
            "scale": 0.6,
            "confidence": 0.7,
            "range_hint": [float(range_hint[0]), float(range_hint[1])],
        }
        readout["effects"] = effects
    readout["constraints"] = []
    return readout


def _extract_target_vars(text: str) -> List[str]:
    t = text.lower()
    var_map = {
        "x1": "x1",
        "amine": "x1",
        "x2": "x2",
        "aldehyde": "x2",
        "x3": "x3",
        "isocyanide": "x3",
        "x4": "x4",
        "ptsa": "x4",
        "p-tsa": "x4",
        "ptsoh": "x4",
        "p-tsoh": "x4",
    }
    targets: List[str] = []
    for key, var in var_map.items():
        if key in t and var not in targets:
            targets.append(var)
    return targets


def _merge_readout_partial(
    current: Dict[str, Any],
    update: Dict[str, Any],
    *,
    target_vars: List[str],
    allow_bumps: bool,
) -> Dict[str, Any]:
    if not target_vars:
        return update

    merged = json.loads(json.dumps(current or {}))

    # Effects: only update mentioned vars.
    cur_eff = merged.get("effects") or {}
    upd_eff = update.get("effects") or {}
    if isinstance(cur_eff, dict) and isinstance(upd_eff, dict):
        for var in target_vars:
            if var in upd_eff:
                cur_eff[var] = upd_eff[var]
    merged["effects"] = cur_eff

    # Constraints: replace constraints for mentioned vars only.
    cur_cons = merged.get("constraints") or []
    if not isinstance(cur_cons, list):
        cur_cons = []
    remaining = []
    for c in cur_cons:
        if not isinstance(c, dict):
            continue
        var = _canonical_var_name(c.get("var"), st.session_state.domain.feature_names)
        if var and var in target_vars:
            continue
        remaining.append(c)
    upd_cons = update.get("constraints") or []
    if isinstance(upd_cons, list):
        for c in upd_cons:
            if not isinstance(c, dict):
                continue
            var = _canonical_var_name(c.get("var"), st.session_state.domain.feature_names)
            if var in target_vars:
                remaining.append(c)
    merged["constraints"] = remaining

    # Bumps: keep unless explicitly requested.
    if allow_bumps:
        merged["bumps"] = update.get("bumps") or []

    return merged


def _resolve_symbolic_range_hints(
    readout: Dict[str, Any],
    domain: ContinuousDomain,
) -> Dict[str, Any]:
    effects = readout.get("effects") or {}
    if not isinstance(effects, dict):
        return readout
    mins = domain.mins.detach().cpu().numpy()
    maxs = domain.maxs.detach().cpu().numpy()

    def _tert_range(idx: int, tag: str) -> Optional[Tuple[float, float]]:
        lo = float(mins[idx])
        hi = float(maxs[idx])
        span = hi - lo
        t1 = lo + span / 3.0
        t2 = lo + 2.0 * span / 3.0
        tag = tag.lower().strip()
        if tag == "low":
            return (lo, t1)
        if tag in {"mid", "middle"}:
            return (t1, t2)
        if tag == "high":
            return (t2, hi)
        return None

    for var, spec in effects.items():
        if not isinstance(spec, dict):
            continue
        rh = spec.get("range_hint")
        if rh is None:
            continue
        mapped_var = _canonical_var_name(var, domain.feature_names)
        if not mapped_var:
            continue
        idx = int(mapped_var[1:]) - 1

        if isinstance(rh, str):
            rng = _tert_range(idx, rh)
            if rng:
                spec["range_hint"] = [rng[0], rng[1]]
            continue
        if isinstance(rh, (list, tuple)) and len(rh) == 2:
            a, b = rh[0], rh[1]
            if isinstance(a, str) or isinstance(b, str):
                r1 = _tert_range(idx, str(a)) if isinstance(a, str) else None
                r2 = _tert_range(idx, str(b)) if isinstance(b, str) else None
                if r1 and r2:
                    spec["range_hint"] = [min(r1[0], r2[0]), max(r1[1], r2[1])]
                elif r1:
                    spec["range_hint"] = [r1[0], r1[1]]
                elif r2:
                    spec["range_hint"] = [r2[0], r2[1]]
    return readout


def _apply_readout_text_if_changed() -> bool:
    text = st.session_state.get("readout_text", "")
    if not isinstance(text, str):
        return True
    try:
        ro = json.loads(text)
    except Exception:
        st.error("Readout JSON is invalid. Fix it or click Apply JSON.")
        return False
    ro, errors = _sanitize_readout(ro)
    if errors:
        st.warning("Readout had non-numeric fields; coerced to defaults.")
    ro = _resolve_symbolic_range_hints(ro, st.session_state.domain)
    ro = _normalize_readout_vars(ro, st.session_state.domain.feature_names)
    if ro != st.session_state.current_readout:
        st.session_state.current_readout = ro
        st.session_state.readout_text_pending = json.dumps(ro, indent=2)
        _save_current_readout(ro, source="auto_apply_before_run")
    return True


def _init_state() -> None:
    if "domain" not in st.session_state:
        st.session_state.domain = build_continuous_domain()
    if "X_obs" not in st.session_state:
        st.session_state.X_obs = torch.empty((0, len(st.session_state.domain.feature_names)), device=DEVICE, dtype=DTYPE)
    if "Y_obs" not in st.session_state:
        st.session_state.Y_obs = torch.empty((0, 1), device=DEVICE, dtype=DTYPE)
    if "history" not in st.session_state:
        st.session_state.history = []
    if "awcd_history" not in st.session_state:
        st.session_state.awcd_history = []
        st.session_state.awcd_history_constraint = []
        st.session_state.awcd_history_mean = []
    if "iteration_count" not in st.session_state:
        st.session_state.iteration_count = 0
    if "current_readout" not in st.session_state:
        st.session_state.current_readout = flat_readout(feature_names=st.session_state.domain.feature_names)
        _save_current_readout(st.session_state.current_readout, source="init")
    if "summary_text" not in st.session_state:
        st.session_state.summary_text = ""
    if "readout_text" not in st.session_state:
        st.session_state.readout_text = json.dumps(st.session_state.current_readout, indent=2)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_pending" not in st.session_state:
        st.session_state.chat_pending = None
    if "auto_summary" not in st.session_state:
        st.session_state.auto_summary = True
    if "summary_every" not in st.session_state:
        st.session_state.summary_every = 10
    if "summary_mode" not in st.session_state:
        st.session_state.summary_mode = "LLM"
    if "auto_awcd_alert" not in st.session_state:
        st.session_state.auto_awcd_alert = True
    if "last_awcd_alert_iter" not in st.session_state:
        st.session_state.last_awcd_alert_iter = None


def _initialize_with_random(n_init: int, seed: int, init_method: str) -> None:
    domain = st.session_state.domain
    X_init = _sample_initial_unit(domain, n_init, seed, init_method)
    Y_init = evaluate_oracle(domain, X_init).unsqueeze(-1)
    st.session_state.X_obs = X_init.clone()
    st.session_state.Y_obs = Y_init.clone()
    st.session_state.history = []
    st.session_state.awcd_history = []
    st.session_state.awcd_history_constraint = []
    st.session_state.awcd_history_mean = []
    st.session_state.iteration_count = 0

    best = float(Y_init.max().item()) if Y_init.numel() else float("-inf")
    for i in range(int(X_init.shape[0])):
        raw = unit_to_raw(domain, X_init[i]).squeeze(0)
        rec = {
            "iter": int(i - X_init.shape[0]),
            "y": float(Y_init[i].item()),
            "best_so_far": float(best),
            "awcd_score": float("nan"),
            "prior_active": True,
        }
        for j in range(raw.numel()):
            rec[f"x{j+1}"] = float(raw[j].item())
        st.session_state.history.append(rec)


def _step_once(
    *,
    constraint_hardness: float,
    constraint_pool_size: int,
    awcd_top_frac: float,
    awcd_constraint_threshold: float,
    awcd_mean_threshold: float,
    awcd_warmup: int,
    awcd_window: int,
) -> Dict[str, Any]:
    domain = st.session_state.domain
    X_obs = st.session_state.X_obs
    Y_obs = st.session_state.Y_obs
    readout = _normalize_readout_vars(st.session_state.current_readout, domain.feature_names)
    ro_unit = normalize_readout_to_unit_box(readout, domain.mins, domain.maxs, feature_names=domain.feature_names)
    prior = readout_to_prior(ro_unit, feature_names=domain.feature_names)

    gp_skeptic = SingleTaskGP(X_obs, Y_obs)
    mll_s = ExactMarginalLogLikelihood(gp_skeptic.likelihood, gp_skeptic)
    fit_gpytorch_mll(mll_s)

    gp_resid, alpha = fit_residual_gp(X_obs, Y_obs, prior)
    model_believer = GPWithPriorMean(gp_resid, prior, m0_scale=float(alpha))

    awcd_metrics = _awcd_full_score(
        gp_skeptic,
        prior.m0_torch,
        X_obs=X_obs,
        Y_obs=Y_obs,
        readout_unit=ro_unit,
        feature_names=domain.feature_names,
        pool_n=constraint_pool_size,
        top_frac=awcd_top_frac,
        seed=int(st.session_state.seed) + 9000 + st.session_state.iteration_count,
        bounds=domain.unit_bounds,
    )
    awcd_score = float(awcd_metrics.get("awcd_total", 0.0))
    awcd_constraint = float(awcd_metrics.get("awcd_constraint", 0.0))
    awcd_mean = float(awcd_metrics.get("awcd_mean", 0.0))

    st.session_state.awcd_history.append(awcd_score)
    st.session_state.awcd_history_constraint.append(awcd_constraint)
    st.session_state.awcd_history_mean.append(awcd_mean)
    awcd_constraint_mean = float(np.mean(st.session_state.awcd_history_constraint[-int(awcd_window):]))
    awcd_mean_mean = float(np.mean(st.session_state.awcd_history_mean[-int(awcd_window):]))

    if st.session_state.iteration_count < int(awcd_warmup) or len(st.session_state.awcd_history) < int(awcd_window):
        prior_good = True
    else:
        prior_good = not (
            awcd_constraint_mean > float(awcd_constraint_threshold)
            or awcd_mean_mean > float(awcd_mean_threshold)
        )
    weight = 1.0 if prior_good else 0.0
    prior_active = bool(prior_good)

    best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
    effective_hardness = float(constraint_hardness) if weight > 0.0 else 0.0

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
            seed=int(st.session_state.seed) + 202 + st.session_state.iteration_count,
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
            num_restarts=10,
            raw_samples=256,
        )
        x_next = x_next.squeeze(0)

    y_next = evaluate_oracle(domain, x_next).unsqueeze(-1)
    st.session_state.X_obs = torch.cat([X_obs, x_next.unsqueeze(0)], dim=0)
    st.session_state.Y_obs = torch.cat([Y_obs, y_next], dim=0)

    best = float(st.session_state.Y_obs.max().item()) if st.session_state.Y_obs.numel() else float("-inf")
    raw = unit_to_raw(domain, x_next).squeeze(0)
    rec = {
        "iter": int(st.session_state.iteration_count),
        "y": float(y_next.item()),
        "best_so_far": best,
        "awcd_score": float(awcd_score),
        "prior_active": bool(prior_active),
    }
    for j in range(raw.numel()):
        rec[f"x{j+1}"] = float(raw[j].item())
    st.session_state.history.append(rec)
    st.session_state.iteration_count += 1
    completed = st.session_state.iteration_count

    if st.session_state.auto_summary:
        interval = int(st.session_state.summary_every)
        if interval > 0 and completed % interval == 0:
            if st.session_state.summary_mode == "LLM":
                try:
                    summary = generate_llm_summary(
                        st.session_state.history,
                        model=st.session_state.model,
                        temperature=float(st.session_state.temperature),
                        api_key=st.session_state.api_key.strip() or None,
                        last_k=interval,
                    )
                except Exception:
                    summary = summarize_history_local(st.session_state.history, last_k=interval)
            else:
                summary = summarize_history_local(st.session_state.history, last_k=interval)
            _append_chat("assistant", f"Summary at Iteration {completed}: {summary}")

    if st.session_state.auto_awcd_alert:
        threshold = float(st.session_state.awcd_constraint_threshold)
        last_alert = st.session_state.last_awcd_alert_iter
        if awcd_score > threshold and last_alert != completed:
            _append_chat(
                "assistant",
                f"ALERT: AWCD Score is {awcd_score:.2f}. High-pressure candidates appear in forbidden regions. "
                "Prior influence is disabled.",
            )
            st.session_state.last_awcd_alert_iter = completed
    return rec


def _prior_surface_plot(
    domain: ContinuousDomain, readout: Dict[str, Any], n_grid: int = 80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    readout = _normalize_readout_vars(readout, domain.feature_names)
    ro_unit = normalize_readout_to_unit_box(readout, domain.mins, domain.maxs, feature_names=domain.feature_names)
    prior = readout_to_prior(ro_unit, feature_names=domain.feature_names)

    mins = domain.mins.detach().cpu().numpy()
    maxs = domain.maxs.detach().cpu().numpy()
    span = maxs - mins

    x1_vals = np.linspace(float(mins[0]), float(maxs[0]), n_grid)
    x4_vals = np.linspace(float(mins[3]), float(maxs[3]), n_grid)
    X1, X4 = np.meshgrid(x1_vals, x4_vals)

    x1_unit = (X1 - mins[0]) / max(span[0], 1e-12)
    x4_unit = (X4 - mins[3]) / max(span[3], 1e-12)

    grid_unit = np.zeros((n_grid * n_grid, len(domain.feature_names)), dtype=np.float64)
    grid_unit[:, 0] = x1_unit.ravel()
    grid_unit[:, 3] = x4_unit.ravel()
    if len(domain.feature_names) > 1:
        grid_unit[:, 1] = 0.5
    if len(domain.feature_names) > 2:
        grid_unit[:, 2] = 0.5

    grid_t = torch.tensor(grid_unit, dtype=DTYPE, device=DEVICE)
    if readout.get("effects") or readout.get("bumps"):
        m0 = prior.m0_torch(grid_t).reshape(n_grid, n_grid).detach().cpu().numpy()
        label = "Prior mean"
        return X1, X4, m0, label

    mask = np.zeros_like(X1, dtype=np.float64)
    constraints = readout.get("constraints") or []
    for c in constraints:
        if not isinstance(c, dict):
            continue
        var = _canonical_var_name(c.get("var"), domain.feature_names)
        r = c.get("range")
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        lo = float(r[0])
        hi = float(r[1])
        if hi < lo:
            lo, hi = hi, lo
        if var == "x1":
            mask = np.logical_or(mask, (X1 >= lo) & (X1 <= hi))
        if var == "x4":
            mask = np.logical_or(mask, (X4 >= lo) & (X4 <= hi))
    m0 = mask.astype(np.float64)
    label = "Constraint mask (raw ranges)"
    return X1, X4, m0, label


def _prior_scatter_3d(
    domain: ContinuousDomain,
    readout: Dict[str, Any],
    *,
    n_points: int = 1500,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    readout = _normalize_readout_vars(readout, domain.feature_names)
    ro_unit = normalize_readout_to_unit_box(readout, domain.mins, domain.maxs, feature_names=domain.feature_names)
    prior = readout_to_prior(ro_unit, feature_names=domain.feature_names)

    d = int(len(domain.feature_names))
    pool = draw_sobol_samples(bounds=domain.unit_bounds, n=int(n_points), q=1, seed=int(seed)).squeeze(1)
    if d > 2:
        pool[:, 2] = 0.5  # fix x3 to mid
    raw = unit_to_raw(domain, pool)

    if readout.get("effects") or readout.get("bumps"):
        vals = prior.m0_torch(pool).reshape(-1)
        label = "Prior mean"
    else:
        penalties = _constraint_penalty_values(pool, ro_unit, domain.feature_names)
        vals = (penalties > 0).to(dtype=DTYPE)
        label = "Constraint mask (raw ranges)"

    x1 = raw[:, 0].detach().cpu().numpy()
    x2 = raw[:, 1].detach().cpu().numpy()
    x4 = raw[:, 3].detach().cpu().numpy() if d > 3 else raw[:, -1].detach().cpu().numpy()
    v = vals.detach().cpu().numpy()
    return x1, x2, x4, v, label


st.set_page_config(page_title="HILO Dashboard", layout="wide")
_init_state()

if "readout_text_pending" in st.session_state:
    st.session_state.readout_text = st.session_state.pop("readout_text_pending")
if "summary_pending" in st.session_state:
    st.session_state.summary_text = st.session_state.pop("summary_pending")

st.markdown("# HILO Dashboard: Ugi Reaction Optimization")

st.sidebar.header("Expert Channel")
st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), key="api_key")
st.sidebar.text_input("Model", value="gpt-4o-mini", key="model")
st.sidebar.text_input("Readout model", value="gpt-4.1", key="readout_model")
st.sidebar.number_input("Temperature", value=0.2, step=0.05, key="temperature")

st.sidebar.subheader("Optimization Settings")
st.sidebar.number_input("n_init", value=3, step=1, key="n_init")
st.sidebar.number_input("n_iter", value=40, step=5, key="n_iter")
st.sidebar.number_input("seed", value=0, step=1, key="seed")
st.sidebar.text_input("init method", value="sobol", key="init_method")
st.sidebar.number_input("constraint hardness", value=0.2, step=0.05, key="constraint_hardness")
st.sidebar.number_input("constraint pool size", value=4096, step=512, key="constraint_pool_size")
st.sidebar.number_input("awcd top frac", value=0.05, step=0.01, key="awcd_top_frac")
st.sidebar.number_input("awcd constraint threshold", value=0.95, step=0.05, key="awcd_constraint_threshold")
st.sidebar.number_input("awcd mean threshold", value=0.95, step=0.05, key="awcd_mean_threshold")
st.sidebar.number_input("awcd warmup", value=1, step=1, key="awcd_warmup")
st.sidebar.number_input("awcd window", value=1, step=1, key="awcd_window")
st.sidebar.checkbox("auto summary", value=st.session_state.auto_summary, key="auto_summary")
st.sidebar.number_input("summary every N iters", value=int(st.session_state.summary_every), step=1, key="summary_every")
st.sidebar.selectbox("summary mode", ["LLM", "Local"], index=0 if st.session_state.summary_mode == "LLM" else 1, key="summary_mode")
st.sidebar.checkbox("auto safety alert", value=st.session_state.auto_awcd_alert, key="auto_awcd_alert")

if st.sidebar.button("Summarize Results"):
    try:
        summary = generate_llm_summary(
            st.session_state.history,
            model=st.session_state.model,
            temperature=float(st.session_state.temperature),
            api_key=st.session_state.api_key.strip() or None,
        )
        st.session_state.summary_pending = summary
        st.rerun()
    except Exception as exc:
        st.sidebar.error(str(exc))

# Readout editor
st.subheader("Current Prior (manual_readout)")
readout_text = st.text_area("", height=240, key="readout_text")
if st.button("Apply JSON"):
    try:
        ro = json.loads(readout_text)
        ro, errors = _sanitize_readout(ro)
        if errors:
            st.warning("Readout had non-numeric fields; coerced to defaults.")
        ro = _resolve_symbolic_range_hints(ro, st.session_state.domain)
        st.session_state.current_readout = ro
        st.success("Readout updated.")
        _save_current_readout(ro, source="manual_apply")
    except Exception as exc:
        st.error(str(exc))

st.markdown("---")

col_chat, col_main = st.columns([1.1, 1.9])

with col_chat:
    st.subheader("Expert Chat")
    if st.button("Start Chat"):
        system = (
            "You are an expert scientific assistant. Ask ONE focused question at a time "
            "to elicit prior knowledge about the UGI reaction (x1 amine, x4 pTsOH). "
            "Confirm any constraints you will apply, then ask a question."
        )
        try:
            question = _call_openai_chat(
                [{"role": "system", "content": system}, {"role": "user", "content": "Start the interview."}],
                model=st.session_state.model,
                temperature=float(st.session_state.temperature),
                api_key=st.session_state.api_key.strip() or None,
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": question, "ts": datetime.now().strftime("%H:%M:%S")}
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.markdown(
        """
        <style>
        .chat-row {display: flex; margin-bottom: 8px;}
        .chat-row.user {justify-content: flex-end;}
        .chat-row.assistant {justify-content: flex-start;}
        .chat-bubble {max-width: 88%; padding: 8px 12px; border-radius: 12px; border: 1px solid #d6d6d6; color: #111;}
        .chat-bubble.user {background: #e6f2ff; border-color: #bcd7ff;}
        .chat-bubble.assistant {background: #f7f7f7; border-color: #d6d6d6;}
        .chat-meta {font-size: 11px; color: #333; margin-bottom: 4px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    for msg in st.session_state.chat_history:
        role = msg.get("role", "assistant")
        role_class = "user" if role == "user" else "assistant"
        label = "🧑 You" if role == "user" else "🤖 AI"
        ts = msg.get("ts", "")
        content = html_lib.escape(str(msg.get("content", ""))).replace("\n", "<br>")
        ts_html = f"{label} · {ts}" if ts else label
        st.markdown(
            f"""
            <div class="chat-row {role_class}">
              <div class="chat-bubble {role_class}">
                <div class="chat-meta">{ts_html}</div>
                <div>{content}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    user_msg = st.chat_input("Type your response...")
    if user_msg:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg, "ts": datetime.now().strftime("%H:%M:%S")}
        )
        system = (
            "You are an expert scientific assistant. Briefly confirm that you applied the user's "
            "latest request to the readout, then ask: 'Any other request?'. Keep it brief. "
            "If the user asks to show the current or previous JSON readout, you may display it."
        )
        try:
            history_lines = []
            for item in _read_readout_history(last_k=5):
                ts = item.get("ts", "")
                src = item.get("source", "")
                ro = item.get("readout", {})
                history_lines.append(f"[{ts}] {src}: {json.dumps(ro)}")
            history_text = "\n".join(history_lines) if history_lines else "None"
            readout_context = (
                "Current JSON readout:\n"
                f"{json.dumps(st.session_state.current_readout, indent=2)}\n\n"
                "Previous readout history (most recent last):\n"
                f"{history_text}"
            )
            followup = _call_openai_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "system", "content": readout_context},
                ]
                + st.session_state.chat_history,
                model=st.session_state.model,
                temperature=float(st.session_state.temperature),
                api_key=st.session_state.api_key.strip() or None,
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": followup, "ts": datetime.now().strftime("%H:%M:%S")}
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if st.button("Generate Readout from Chat"):
        transcript_lines = []
        for idx, m in enumerate(st.session_state.chat_history):
            ts = m.get("ts", f"{idx:02d}")
            transcript_lines.append(f"[{ts}] {m['role']}: {m['content']}")
        transcript = "\n".join(transcript_lines)
        last_user = _extract_last_user_text(st.session_state.chat_history)
        target_vars = _extract_target_vars(last_user)
        allow_bumps = any(k in last_user.lower() for k in ["bump", "peak", "sweet spot"])
        data_context = _build_domain_context(st.session_state.domain) + "\n" + _build_history_context(
            st.session_state.history, last_k=10
        )
        try:
            ro = generate_readout_from_guidance(
                transcript,
                st.session_state.summary_text,
                data_context=data_context,
                current_readout=st.session_state.current_readout,
                readout_history=_read_readout_history(last_k=5),
                model=st.session_state.readout_model,
                temperature=float(st.session_state.temperature),
                api_key=st.session_state.api_key.strip() or None,
            )
            ro = _apply_preference_overrides(ro, last_user, st.session_state.domain)
            ro = _resolve_symbolic_range_hints(ro, st.session_state.domain)
            ro = _normalize_readout_vars(ro, st.session_state.domain.feature_names)
            current_norm = _normalize_readout_vars(
                st.session_state.current_readout, st.session_state.domain.feature_names
            )
            ro = _merge_readout_partial(
                current_norm,
                ro,
                target_vars=target_vars,
                allow_bumps=allow_bumps,
            )
            st.session_state.current_readout = ro
            st.session_state.readout_text_pending = json.dumps(ro, indent=2)
            _save_current_readout(ro, source="llm_generate")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

with col_main:
    st.subheader("Real-time feedback")
    hist = st.session_state.history
    perf_col, prior_col = st.columns([1.05, 1.25])
    with perf_col:
        st.markdown("**Best-so-far**")
        if hist:
            df = pd.DataFrame(hist)
            df_plot = df[df["iter"] >= 0]
            if not df_plot.empty:
                df_plot = df_plot.copy()
                df_plot["best_so_far_bo"] = df_plot["y"].cummax()
                chart_df = df_plot.groupby("iter")["best_so_far_bo"].mean().reset_index()
                fig = plt.figure(figsize=(4.8, 3.4))
                ax = fig.add_subplot(111)
                ax.plot(chart_df["iter"] + 1, chart_df["best_so_far_bo"], color="#2c3e50", linewidth=2.0)
                ax.scatter(
                    df_plot["iter"] + 1,
                    df_plot["y"],
                    s=18,
                    color="#4c72b0",
                    alpha=0.35,
                    label="Yield",
                )
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Best-so-far")
                ax.grid(False)
                ax.margins(x=0.03)
                ax.legend(loc="lower right", fontsize=8, frameon=False)
                st.pyplot(fig)
            st.dataframe(df.tail(8), use_container_width=True)
        else:
            st.info("No observations yet.")

    with prior_col:
        st.markdown("**Prior / Constraint Surface (x1 vs x4)**")
        X1, X4, m0, surface_label = _prior_surface_plot(
            st.session_state.domain, st.session_state.current_readout
        )
        if surface_label.startswith("Constraint"):
            st.caption("Showing forbidden-region mask from raw constraint ranges (1 = forbidden, 0 = allowed).")
        fig = plt.figure(figsize=(5.2, 3.6))
        ax = fig.add_subplot(111)
        cmap = "magma" if surface_label == "Prior mean" else "Reds"
        cs = ax.contourf(X1, X4, m0, levels=40, cmap=cmap)
        ax.set_xlabel("x1 (Amine)")
        ax.set_ylabel("x4 (pTsOH)")
        fig.colorbar(cs, ax=ax, shrink=0.85, label=surface_label)
        st.pyplot(fig)

        st.markdown("**3D Prior Map (x1, x2, x4)**")
        x1s, x2s, x4s, vals3d, label3d = _prior_scatter_3d(
            st.session_state.domain,
            st.session_state.current_readout,
            n_points=1200,
            seed=int(st.session_state.seed),
        )
        fig3d = plt.figure(figsize=(5.6, 4.0))
        fig3d.subplots_adjust(right=0.78, bottom=0.08, left=0.02)
        ax3d = fig3d.add_subplot(111, projection="3d")
        sc = ax3d.scatter(
            x1s,
            x2s,
            x4s,
            c=vals3d,
            cmap="magma" if label3d == "Prior mean" else "Reds",
            s=8,
            alpha=0.8,
        )
        feature_names = st.session_state.domain.feature_names
        ax3d.set_xlabel(f"x1 ({feature_names[0]})")
        ax3d.set_ylabel(f"x2 ({feature_names[1]})")
        ax3d.set_zlabel(f"x4 ({feature_names[3]})")
        ax3d.grid(False)
        fig3d.colorbar(sc, ax=ax3d, shrink=0.65, pad=0.18, label=label3d)
        st.pyplot(fig3d)
        refresh_col, clear_col = st.columns(2)
        with refresh_col:
            if st.button("Refresh readout view"):
                st.rerun()
        with clear_col:
            if st.button("Clear readout"):
                empty = flat_readout(feature_names=st.session_state.domain.feature_names)
                st.session_state.current_readout = empty
                st.session_state.readout_text_pending = json.dumps(empty, indent=2)
                _save_current_readout(empty, source="clear")
                st.rerun()

    if hist:
        df = pd.DataFrame(hist)
        df_plot = df[df["iter"] >= 0]
        if not df_plot.empty:
            st.markdown("**Parameter ↔ Yield relationships**")
            rel_col1, rel_col2 = st.columns([1.1, 0.9])
            with rel_col1:
                fig = plt.figure(figsize=(6.0, 4.4))
                axes = fig.subplots(2, 2, sharey=True)
                for idx, ax in enumerate(axes.flat):
                    col = f"x{idx+1}"
                    if col in df_plot.columns:
                        ax.scatter(df_plot[col], df_plot["y"], s=18, alpha=0.7, color="#4c72b0")
                        corr = df_plot[[col, "y"]].corr().iloc[0, 1]
                        ax.set_title(f"{col} (r={corr:.2f})", fontsize=10)
                        ax.set_xlabel(col)
                    ax.grid(False)
                axes[0, 0].set_ylabel("Yield")
                axes[1, 0].set_ylabel("Yield")
                fig.tight_layout()
                st.pyplot(fig)
            with rel_col2:
                corrs = []
                labels = []
                for j in range(4):
                    col = f"x{j+1}"
                    if col in df_plot.columns:
                        corr = df_plot[[col, "y"]].corr().iloc[0, 1]
                        corrs.append(0.0 if np.isnan(corr) else float(corr))
                        labels.append(col)
                if corrs:
                    fig = plt.figure(figsize=(4.0, 3.2))
                    ax = fig.add_subplot(111)
                    ax.bar(labels, corrs, color="#dd8452")
                    ax.set_ylim(-1.0, 1.0)
                    ax.set_ylabel("Pearson r")
                    ax.set_title("Correlation summary", fontsize=11)
                    ax.grid(False)
                    st.pyplot(fig)

            st.markdown("**Parameter vs Iteration (colored by Yield)**")
            fig = plt.figure(figsize=(8.4, 4.6))
            axes = fig.subplots(2, 2, sharex=True)
            sc = None
            for idx, ax in enumerate(axes.flat):
                col = f"x{idx+1}"
                if col in df_plot.columns:
                    ymin = float(st.session_state.domain.mins[idx].item())
                    ymax = float(st.session_state.domain.maxs[idx].item())
                    sc = ax.scatter(
                        df_plot["iter"] + 1,
                        df_plot[col],
                        c=df_plot["y"],
                        s=18,
                        cmap="viridis",
                        alpha=0.75,
                    )
                    ax.set_ylabel(col)
                    ax.set_ylim(ymin, ymax)
                ax.grid(False)
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 1].set_xlabel("Iteration")
            if sc is not None:
                fig.subplots_adjust(right=0.72, wspace=0.55, hspace=0.34)
                cax = fig.add_axes([0.84, 0.15, 0.03, 0.7])
                cbar = fig.colorbar(sc, cax=cax)
                cbar.set_label("Yield")
            st.pyplot(fig)

            st.markdown("**Safety + Prior Usage over Iteration**")
            safety_col, prior_col = st.columns([1.15, 0.85])
            with safety_col:
                awcd_scores = np.array(st.session_state.awcd_history, dtype=float)
                if awcd_scores.size:
                    fig = plt.figure(figsize=(6.4, 3.4))
                    ax = fig.add_subplot(111)
                    x = np.arange(1, awcd_scores.size + 1)
                    ax.plot(x, awcd_scores, color="#2c3e50", linewidth=2.0, label="AWCD Score")
                    thresh = float(st.session_state.awcd_constraint_threshold)
                    ax.axhline(thresh, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7, label="threshold")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("AWCD score")
                    ax.grid(False)
                    ax.legend(
                        fontsize=8,
                        frameon=False,
                        ncol=1,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0,
                    )
                    fig.subplots_adjust(right=0.78)
                    st.pyplot(fig)
                else:
                    st.info("No AWCD history yet.")
            with prior_col:
                if "prior_active" in df_plot.columns:
                    fig = plt.figure(figsize=(4.6, 3.4))
                    ax = fig.add_subplot(111)
                    x = df_plot["iter"].to_numpy() + 1
                    y = df_plot["prior_active"].astype(int).to_numpy()
                    ax.step(x, y, where="mid", color="#dd8452", linewidth=2.0)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(["off", "on"])
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Prior used")
                    ax.grid(False)
                    st.pyplot(fig)
                else:
                    st.info("No prior usage data yet.")

st.markdown("---")

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.4])
with c1:
    if st.button("Run 1 Step"):
        if not _apply_readout_text_if_changed():
            st.stop()
        if st.session_state.X_obs.numel() == 0:
            _initialize_with_random(int(st.session_state.n_init), int(st.session_state.seed), st.session_state.init_method)
        _step_once(
            constraint_hardness=float(st.session_state.constraint_hardness),
            constraint_pool_size=int(st.session_state.constraint_pool_size),
            awcd_top_frac=float(st.session_state.awcd_top_frac),
            awcd_constraint_threshold=float(st.session_state.awcd_constraint_threshold),
            awcd_mean_threshold=float(st.session_state.awcd_mean_threshold),
            awcd_warmup=int(st.session_state.awcd_warmup),
            awcd_window=int(st.session_state.awcd_window),
        )
        _save_history_csv(tag="run_1")
        st.rerun()

with c2:
    if st.button("Run 10 Steps"):
        if not _apply_readout_text_if_changed():
            st.stop()
        if st.session_state.X_obs.numel() == 0:
            _initialize_with_random(int(st.session_state.n_init), int(st.session_state.seed), st.session_state.init_method)
        for _ in range(10):
            _step_once(
                constraint_hardness=float(st.session_state.constraint_hardness),
                constraint_pool_size=int(st.session_state.constraint_pool_size),
                awcd_top_frac=float(st.session_state.awcd_top_frac),
                awcd_constraint_threshold=float(st.session_state.awcd_constraint_threshold),
                awcd_mean_threshold=float(st.session_state.awcd_mean_threshold),
                awcd_warmup=int(st.session_state.awcd_warmup),
                awcd_window=int(st.session_state.awcd_window),
            )
        _save_history_csv(tag="run_10")
        st.rerun()

with c3:
    if st.button("Run 20 Steps"):
        if not _apply_readout_text_if_changed():
            st.stop()
        if st.session_state.X_obs.numel() == 0:
            _initialize_with_random(int(st.session_state.n_init), int(st.session_state.seed), st.session_state.init_method)
        for _ in range(20):
            _step_once(
                constraint_hardness=float(st.session_state.constraint_hardness),
                constraint_pool_size=int(st.session_state.constraint_pool_size),
                awcd_top_frac=float(st.session_state.awcd_top_frac),
                awcd_constraint_threshold=float(st.session_state.awcd_constraint_threshold),
                awcd_mean_threshold=float(st.session_state.awcd_mean_threshold),
                awcd_warmup=int(st.session_state.awcd_warmup),
                awcd_window=int(st.session_state.awcd_window),
            )
        _save_history_csv(tag="run_20")
        st.rerun()

with c4:
    if st.button("Run 30 Steps"):
        if not _apply_readout_text_if_changed():
            st.stop()
        if st.session_state.X_obs.numel() == 0:
            _initialize_with_random(int(st.session_state.n_init), int(st.session_state.seed), st.session_state.init_method)
        for _ in range(30):
            _step_once(
                constraint_hardness=float(st.session_state.constraint_hardness),
                constraint_pool_size=int(st.session_state.constraint_pool_size),
                awcd_top_frac=float(st.session_state.awcd_top_frac),
                awcd_constraint_threshold=float(st.session_state.awcd_constraint_threshold),
                awcd_mean_threshold=float(st.session_state.awcd_mean_threshold),
                awcd_warmup=int(st.session_state.awcd_warmup),
                awcd_window=int(st.session_state.awcd_window),
            )
        _save_history_csv(tag="run_30")
        st.rerun()

with c5:
    if st.button("Reset Experiment"):
        _initialize_with_random(int(st.session_state.n_init), int(st.session_state.seed), st.session_state.init_method)
        st.rerun()

awcd_score = st.session_state.awcd_history[-1] if st.session_state.awcd_history else float("nan")
status = "Safe" if awcd_score <= float(st.session_state.awcd_constraint_threshold) else "Circuit Breaker Tripped"
color = "#2ecc71" if status == "Safe" else "#e74c3c"

st.markdown("### Safety Monitor")
metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("AWCD Score", f"{awcd_score:.3f}" if awcd_score == awcd_score else "--")
metric_col2.markdown(
    f"<div style='padding:8px;border-radius:6px;background:{color};color:white;display:inline-block;'>"
    f"{status}"
    "</div>",
    unsafe_allow_html=True,
)

with st.expander("Debug / Raw State"):
    st.write("Iteration:", st.session_state.iteration_count)
    st.write("X_obs shape:", tuple(st.session_state.X_obs.shape))
    st.write("Y_obs shape:", tuple(st.session_state.Y_obs.shape))
    st.write("Latest AWCD:", awcd_score)
    st.json(st.session_state.current_readout)

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_data, file_name="ugi_hilo_history.csv", mime="text/csv")
