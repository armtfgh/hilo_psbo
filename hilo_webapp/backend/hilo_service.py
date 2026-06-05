from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from readout_schema import flat_readout, normalize_readout_to_unit_box, readout_to_prior  # noqa: E402

UGI_FEATURE_ORDER = ["amine_mM", "aldehyde_mM", "isocyanide_mM", "ptsa"]

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
    oracle: Any
    metadata: Dict[str, Any]


class LookupNearestOracle:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.mins = self.X.min(axis=0)
        self.ranges = np.maximum(self.X.max(axis=0) - self.mins, 1e-12)
        self.X_unit = (self.X - self.mins) / self.ranges

    def __call__(self, candidates: Tensor) -> Tensor:
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0)
        arr = candidates.detach().cpu().numpy().astype(np.float64)
        arr_unit = (arr - self.mins) / self.ranges
        preds = []
        for row in arr_unit:
            dist2 = np.sum((self.X_unit - row[None, :]) ** 2, axis=1)
            preds.append(float(self.y[int(np.argmin(dist2))]))
        return torch.tensor(preds, dtype=candidates.dtype, device=candidates.device)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _raw_number_matches(text: str) -> List[float]:
    matches = re.findall(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    out: List[float] = []
    for item in matches:
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def _raw_number_matches_with_positions(text: str) -> List[tuple[float, int, int]]:
    out: List[tuple[float, int, int]] = []
    for match in re.finditer(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        try:
            out.append((float(match.group(0)), int(match.start()), int(match.end())))
        except ValueError:
            continue
    return out


def build_continuous_domain(*, target: str = "yield") -> ContinuousDomain:
    merged_path = ROOT / "ugi_merged_dataset.csv"
    if merged_path.exists():
        df = pd.read_csv(merged_path)
    else:
        raise FileNotFoundError(f"Required merged UGI dataset not found: {merged_path}")
    df = df.dropna(subset=[target]).copy()
    feature_names = [name for name in UGI_FEATURE_ORDER if name in df.columns]
    if not feature_names:
        feature_names = [c for c in df.select_dtypes(include="number").columns if c != target]
    X = df[feature_names]
    y = df[target]
    oracle = LookupNearestOracle(X.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))
    candidate_pool = X.drop_duplicates().reset_index(drop=True)
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
            "oracle_metrics": {
                "mode": "interactive_nearest_lookup",
                "n_rows": int(df.shape[0]),
            },
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
    return domain.oracle(unit_to_raw(domain, X_unit))


def draw_unit_sobol(bounds: Tensor, n: int, q: int, seed: int) -> Tensor:
    d = int(bounds.shape[-1])
    engine = torch.quasirandom.SobolEngine(dimension=d * int(q), scramble=True, seed=int(seed))
    raw = engine.draw(int(n)).to(device=DEVICE, dtype=DTYPE).reshape(int(n), int(q), d)
    lower = bounds[0].to(device=DEVICE, dtype=DTYPE)
    upper = bounds[1].to(device=DEVICE, dtype=DTYPE)
    return lower + raw * (upper - lower)


def sample_initial_unit(domain: ContinuousDomain, n_init: int, seed: int, method: str) -> Tensor:
    method = (method or "sobol").lower()
    if n_init <= 0:
        return torch.empty((0, domain.unit_bounds.shape[1]), device=DEVICE, dtype=DTYPE)
    if method in {"sobol", "sobo"}:
        return draw_unit_sobol(bounds=domain.unit_bounds, n=n_init, q=1, seed=seed).squeeze(1)
    if method in {"lhs", "latin", "latin_hypercube"}:
        d = int(domain.unit_bounds.shape[1])
        rng = np.random.default_rng(int(seed))
        points = np.zeros((n_init, d), dtype=np.float64)
        for j in range(d):
            perm = rng.permutation(n_init)
            points[:, j] = (perm + rng.random(n_init)) / n_init
        return torch.tensor(points, dtype=DTYPE, device=DEVICE)
    raise ValueError(f"Unknown init_method: {method}")


def constraint_penalty_values(X_unit: Tensor, readout_unit: Dict[str, Any], feature_names: List[str]) -> Tensor:
    if X_unit.ndim == 1:
        X_unit = X_unit.unsqueeze(0)
    constraints = (readout_unit or {}).get("constraints") or []
    if not constraints:
        return torch.zeros(X_unit.shape[0], device=X_unit.device, dtype=X_unit.dtype)
    idx_lookup = {name: i for i, name in enumerate(feature_names)}
    idx_lookup_lower = {name.lower(): i for i, name in enumerate(feature_names)}

    def dim_index(key: str) -> Optional[int]:
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
        var = c.get("var")
        r = c.get("range")
        if var is None or not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        idx = dim_index(str(var))
        if idx is None:
            continue
        lo = float(r[0])
        hi = float(r[1])
        if hi < lo:
            lo, hi = hi, lo
        strength = float(c.get("penalty", c.get("weight", 5.0)))
        k = float(c.get("sharpness", c.get("k", 60.0)))
        z = X_unit[:, idx]
        gate = (torch.sigmoid(k * (z - lo)) - torch.sigmoid(k * (z - hi))).clamp(0.0, 1.0)
        penalty = penalty + strength * gate
    return penalty


def apply_constraint_hardness(
    scores: Tensor,
    X_unit: Tensor,
    readout_unit: Dict[str, Any],
    feature_names: List[str],
    *,
    hardness: float,
    best_f: float,
) -> Tensor:
    hardness = float(np.clip(hardness, 0.0, 1.0))
    if hardness <= 0.0:
        return scores
    penalties = constraint_penalty_values(X_unit, readout_unit, feature_names)
    if hardness >= 0.999:
        out = scores.clone()
        out[penalties > 0] = -1e12
        return out
    return scores - hardness * penalties * max(abs(float(best_f)), 1e-6)


def hard_constraint_mask(X_unit: Tensor, readout_unit: Dict[str, Any], feature_names: List[str]) -> Tensor:
    constraints = (readout_unit or {}).get("constraints") or []
    if not constraints:
        return torch.zeros(X_unit.shape[0], dtype=torch.bool, device=X_unit.device)
    idx_lookup = {name: i for i, name in enumerate(feature_names)}
    idx_lookup_lower = {name.lower(): i for i, name in enumerate(feature_names)}

    def dim_index(key: str) -> Optional[int]:
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
        var = c.get("var")
        r = c.get("range")
        if var is None or not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        idx = dim_index(str(var))
        if idx is None:
            continue
        lo = float(r[0])
        hi = float(r[1])
        if hi < lo:
            lo, hi = hi, lo
        z = X_unit[:, idx]
        mask = mask | ((z >= lo) & (z <= hi))
    return mask


def awcd_full_score(
    gp_skeptic: Any,
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
    from botorch.acquisition.analytic import ExpectedImprovement
    from gpytorch.settings import cholesky_jitter

    pool = draw_unit_sobol(bounds=bounds, n=int(pool_n), q=1, seed=int(seed)).squeeze(1)
    mask = hard_constraint_mask(pool, readout_unit, feature_names)
    if not bool(mask.any()):
        return {"awcd_constraint": 0.0, "awcd_mean": 0.0, "awcd_total": 0.0}
    best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
    ei = ExpectedImprovement(model=gp_skeptic, best_f=best_f, maximize=True)
    with torch.no_grad():
        ei_vals = ei(pool.unsqueeze(1)).reshape(-1)
        try:
            with cholesky_jitter(1e-4):
                post = gp_skeptic.posterior(pool, observation_noise=True)
                std = post.variance.clamp_min(1e-12).sqrt().reshape(-1)
        except RuntimeError:
            std = torch.zeros(pool.shape[0], device=pool.device, dtype=pool.dtype)
        pressure = ei_vals * std
        prior_mean = prior_mean_func(pool).reshape(-1)
    k = max(1, int(float(top_frac) * float(pressure.numel())))
    top_idx = torch.topk(pressure, k=k).indices
    prior_rank = torch.argsort(torch.argsort(prior_mean, descending=True))
    mask_mean = prior_rank[top_idx] > int(prior_mean.numel() // 2)
    base_forbidden = float(mask.float().mean().item())
    top_forbidden = float(mask[top_idx].float().mean().item())
    awcd_constraint = 0.0 if base_forbidden >= 0.999 else max(
        0.0,
        (top_forbidden - base_forbidden) / max(1e-6, 1.0 - base_forbidden),
    )
    awcd_mean = float(mask_mean.float().mean().item())
    return {
        "awcd_constraint": awcd_constraint,
        "awcd_mean": awcd_mean,
        "awcd_total": max(awcd_constraint, awcd_mean),
    }


def record_continuous_sample(
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
        rec[f"x{j + 1}"] = float(raw[j].item())
    if extra:
        rec.update(extra)
    return rec


@dataclass
class HiloSettings:
    n_init: int = 3
    seed: int = 0
    init_method: str = "sobol"
    constraint_hardness: float = 0.2
    constraint_pool_size: int = 4096
    awcd_top_frac: float = 0.05
    c_user: float = 0.95
    awcd_warmup: int = 1
    awcd_window: int = 1
    prior_strength: float = 1.0
    num_restarts: int = 5
    raw_samples: int = 128


@dataclass
class HiloCampaign:
    domain: Any
    settings: HiloSettings = field(default_factory=HiloSettings)
    readout: Dict[str, Any] = field(default_factory=dict)
    X_obs: torch.Tensor = field(default_factory=lambda: torch.empty((0, 4), device=DEVICE, dtype=DTYPE))
    Y_obs: torch.Tensor = field(default_factory=lambda: torch.empty((0, 1), device=DEVICE, dtype=DTYPE))
    history: List[Dict[str, Any]] = field(default_factory=list)
    awcd_history: List[float] = field(default_factory=list)
    awcd_constraint_history: List[float] = field(default_factory=list)
    awcd_mean_history: List[float] = field(default_factory=list)
    last_error: Optional[str] = None


class HiloService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        domain = build_continuous_domain()
        self.campaign = HiloCampaign(
            domain=domain,
            readout=flat_readout(feature_names=domain.feature_names),
            X_obs=torch.empty((0, len(domain.feature_names)), device=DEVICE, dtype=DTYPE),
            Y_obs=torch.empty((0, 1), device=DEVICE, dtype=DTYPE),
        )

    def domain_payload(self) -> Dict[str, Any]:
        domain = self.campaign.domain
        mins = domain.mins.detach().cpu().numpy()
        maxs = domain.maxs.detach().cpu().numpy()
        return {
            "feature_names": list(domain.feature_names),
            "ranges": [
                {"name": name, "min": float(mins[i]), "max": float(maxs[i])}
                for i, name in enumerate(domain.feature_names)
            ],
            "metadata": domain.metadata,
        }

    def reset(self, settings_update: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            self._update_settings(settings_update or {})
            c = self.campaign
            s = c.settings
            X_init = sample_initial_unit(c.domain, int(s.n_init), int(s.seed), str(s.init_method))
            Y_init = evaluate_oracle(c.domain, X_init).unsqueeze(-1)
            c.X_obs = X_init.clone()
            c.Y_obs = Y_init.clone()
            c.history = []
            c.awcd_history = []
            c.awcd_constraint_history = []
            c.awcd_mean_history = []
            c.last_error = None
            best = float(Y_init.max().item()) if Y_init.numel() else float("-inf")
            for i in range(X_init.shape[0]):
                y = float(Y_init[i].item())
                best = max(best, y)
                c.history.append(
                    record_continuous_sample(
                        c.domain,
                        X_init[i],
                        y,
                        best,
                        method="init",
                        iteration=-(int(s.n_init) - i),
                        extra={
                            "awcd_score": 0.0,
                            "awcd_constraint": 0.0,
                            "awcd_mean_disagree": 0.0,
                            "weight_believer": 0.0,
                            "prior_active": False,
                        },
                    )
                )
            return self.state()

    def clear_readout(self) -> Dict[str, Any]:
        """Reset the active prior to a neutral flat readout (forgets all expert knowledge)."""
        with self._lock:
            self.campaign.readout = flat_readout(feature_names=self.campaign.domain.feature_names)
            return self.state()

    def update_readout(self, readout: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if not isinstance(readout, dict):
                raise ValueError("Readout must be a JSON object.")
            normalized = dict(readout)
            normalized.setdefault("effects", {})
            normalized.setdefault("bumps", [])
            normalized.setdefault("constraints", [])
            self.campaign.readout = normalized
            return self.state()

    def translate_expert_text(
        self,
        *,
        transcript: str,
        model: str = "claude-opus-4-5-20251101",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        os.environ.setdefault("ANTHROPIC_VERSION", "2023-06-01")
        from llm_study import get_llm_json

        c = self.campaign
        ranges = self.domain_payload()["ranges"]
        system = (
            "You are the HILO language-to-prior translator for a UGI reaction optimization. "
            "Convert human expert natural-language chemistry knowledge into a structured JSON prior. "
            "Return STRICT JSON only with keys: effects, bumps, constraints.\n\n"
            "Allowed schema:\n"
            "- effects: {var: {effect, scale, confidence, range_hint}}\n"
            "  var must be one of amine_mM, aldehyde_mM, isocyanide_mM, ptsa.\n"
            "  effect must be one of increasing, decreasing, flat, nonmonotone-peak, nonmonotone-valley.\n"
            "  scale in [0.0, 1.5], confidence in [0.0, 0.95], range_hint in raw units.\n"
            "- bumps: [{mu, sigma, amp}] with length-4 raw-unit vectors ordered "
            "[amine_mM, aldehyde_mM, isocyanide_mM, ptsa], amp in [0.0, 0.6].\n"
            "- constraints: [{var, range, penalty, reason}] with raw-unit range and penalty in [0.0, 10.0].\n\n"
            "Rules:\n"
            "- Use only the provided expert text, domain ranges, and current readout. Do not infer hidden optima.\n"
            "- Map chemistry terms: amine->amine_mM, aldehyde->aldehyde_mM, isocyanide->isocyanide_mM, "
            "pTSA/p-TsOH/acid->ptsa.\n"
            "- Prohibitive language such as avoid, forbid, harmful, failed, poor becomes a soft constraint.\n"
            "- For constraints, range is the BAD/FORBIDDEN region, not the allowed region.\n"
            "- If the expert says avoid values below/under/less than T, the constraint range must be "
            "[domain_min, T].\n"
            "- If the expert says avoid values above/over/greater than T, the constraint range must be "
            "[T, domain_max].\n"
            "- Preference language such as prefer, beneficial, supports, productive becomes an effect or weak bump.\n"
            "- Ambiguous language must become lower confidence and broader ranges.\n"
            "- Preserve useful current readout components unless the expert text contradicts or replaces them.\n"
            "- Omit unsupported primitives rather than inventing precise numbers."
        )
        payload = {
            "domain_ranges": ranges,
            "current_readout": c.readout,
            "expert_transcript": transcript,
            "output_schema": {"effects": {}, "bumps": [], "constraints": []},
        }
        readout = get_llm_json(
            model,
            json.dumps(payload, indent=2),
            system_prompt=system,
            temperature=float(temperature),
            max_tokens=1800,
            strict_json=True,
            api_key=api_key,
        )
        if not isinstance(readout, dict):
            raise ValueError("LLM translator did not return a JSON object.")
        readout.setdefault("effects", {})
        readout.setdefault("bumps", [])
        readout.setdefault("constraints", [])
        return self._repair_directional_constraints(readout, transcript)

    def _repair_directional_constraints(self, readout: Dict[str, Any], transcript: str) -> Dict[str, Any]:
        """Correct common LLM inversions for phrases such as 'avoid pTSA below 0.15'."""
        text = (transcript or "").lower()
        if not text:
            return readout
        negative_words = ("avoid", "forbid", "exclude", "harmful", "failed", "poor", "bad", "penalty")
        below_words = ("below", "under", "less than", "lower than", "<")
        above_words = ("above", "over", "greater than", "higher than", ">")
        aliases = {
            "amine_mM": ("amine", "x1"),
            "aldehyde_mM": ("aldehyde", "x2"),
            "isocyanide_mM": ("isocyanide", "x3"),
            "ptsa": ("ptsa", "p-tsa", "ptsoh", "p-tsoh", "acid", "x4"),
        }
        domain = self.campaign.domain
        range_lookup = {
            name: (float(domain.mins[i].item()), float(domain.maxs[i].item()))
            for i, name in enumerate(domain.feature_names)
        }
        constraints = [c for c in (readout.get("constraints") or []) if isinstance(c, dict)]
        added: List[Dict[str, Any]] = []
        number_items = _raw_number_matches_with_positions(text)

        for var, words in aliases.items():
            if var not in range_lookup or not any(word in text for word in words):
                continue
            lo_raw, hi_raw = range_lookup[var]
            span = max(hi_raw - lo_raw, 1e-12)
            alias_positions = [
                match.start()
                for word in words
                for match in re.finditer(re.escape(word), text)
            ]
            candidates: List[tuple[float, str]] = []
            for value, start, end in number_items:
                if not (lo_raw - 1e-9 <= value <= hi_raw + 1e-9) or not alias_positions:
                    continue
                nearest = min(alias_positions, key=lambda pos: abs(pos - start))
                if abs(nearest - start) > 160:
                    continue
                w0 = max(0, min(nearest, start) - 80)
                w1 = min(len(text), max(nearest, end) + 80)
                window = text[w0:w1]
                if not any(word in window for word in negative_words):
                    continue
                if any(word in window for word in below_words):
                    candidates.append((value, "below"))
                elif any(word in window for word in above_words):
                    candidates.append((value, "above"))
            if not candidates:
                continue
            threshold, direction = min(candidates, key=lambda item: abs(item[0] - (lo_raw + hi_raw) / 2.0))
            if direction == "below":
                desired = [lo_raw, float(threshold)]
                wrong_starts_at_threshold = True
            else:
                desired = [float(threshold), hi_raw]
                wrong_starts_at_threshold = False

            tol = max(1e-6, 0.02 * span)
            cleaned = []
            for c in constraints:
                if str(c.get("var", "")).lower() != var.lower():
                    cleaned.append(c)
                    continue
                r = c.get("range")
                if not isinstance(r, (list, tuple)) or len(r) != 2:
                    cleaned.append(c)
                    continue
                try:
                    c0, c1 = float(r[0]), float(r[1])
                except Exception:
                    cleaned.append(c)
                    continue
                if c1 < c0:
                    c0, c1 = c1, c0
                is_complement = (
                    abs(c0 - threshold) <= tol and abs(c1 - hi_raw) <= tol
                    if wrong_starts_at_threshold
                    else abs(c0 - lo_raw) <= tol and abs(c1 - threshold) <= tol
                )
                is_duplicate = abs(c0 - desired[0]) <= tol and abs(c1 - desired[1]) <= tol
                if not (is_complement or is_duplicate):
                    cleaned.append(c)
            constraints = cleaned
            added.append(
                {
                    "var": var,
                    "range": desired,
                    "penalty": 8.5,
                    "reason": f"Expert requested avoiding {var} {direction} {threshold:g}; range is the forbidden region.",
                }
            )

        if added:
            readout = dict(readout)
            readout["constraints"] = constraints + added
        return readout

    def run_steps(self, steps: int, settings_update: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            self._update_settings(settings_update or {})
            if self.campaign.X_obs.numel() == 0:
                self.reset()
            for _ in range(max(0, int(steps))):
                self._step_once()
            return self.state()

    def state(self) -> Dict[str, Any]:
        c = self.campaign
        settings = c.settings.__dict__.copy()
        hist = [self._clean_record(row) for row in c.history]
        bo_hist = [row for row in hist if int(row.get("iter", 0)) >= 0]
        latest = bo_hist[-1] if bo_hist else None
        best = max([_safe_float(row.get("best_so_far"), float("-inf")) for row in hist], default=float("nan"))
        awcd = float(c.awcd_history[-1]) if c.awcd_history else float("nan")
        prior_active = bool(latest.get("prior_active")) if latest else False
        status = "not_started"
        if c.awcd_history:
            status = "prior_trusted" if awcd <= float(c.settings.c_user) else "prior_gated"
        return {
            "domain": self.domain_payload(),
            "settings": settings,
            "readout": c.readout,
            "history": hist,
            "summary": {
                "iteration": len(bo_hist),
                "n_observations": len(hist),
                "best_yield": None if not np.isfinite(best) else float(best),
                "latest_yield": None if latest is None else _safe_float(latest.get("y")),
                "awcd_score": None if not np.isfinite(awcd) else float(awcd),
                "c_user": float(c.settings.c_user),
                "prior_active": prior_active,
                "status": status,
                "last_error": c.last_error,
            },
        }

    def prior_surface(self, n_grid: int = 48) -> Dict[str, Any]:
        bundle = self.prior_surfaces(n_grid=n_grid)
        return bundle["surfaces"][0] if bundle["surfaces"] else {}

    def prior_surfaces(self, n_grid: int = 40) -> Dict[str, Any]:
        with self._lock:
            c = self.campaign
            ro_unit = normalize_readout_to_unit_box(
                c.readout,
                c.domain.mins,
                c.domain.maxs,
                feature_names=c.domain.feature_names,
            )
            # For DISPLAY we separate the two kinds of knowledge that m0 fuses:
            #   * the smooth "preference" surface (effects + interactions + bumps), and
            #   * hard "forbidden" regions (constraints).
            # Plotting the raw m0 (= preference - penalty) lets a single penalty of
            # ~8.5 dwarf a preference signal of ~0.25 and makes every panel look flat.
            # So we render the preference surface on its own colour scale and return the
            # forbidden mask separately for an explicit overlay. The optimiser still uses
            # the full m0 (constraints included) elsewhere; this is visualisation only.
            pref_unit = dict(ro_unit)
            pref_unit["constraints"] = []
            pref_prior = readout_to_prior(pref_unit, feature_names=c.domain.feature_names)
            mins = c.domain.mins.detach().cpu().numpy()
            maxs = c.domain.maxs.detach().cpu().numpy()
            span = np.maximum(maxs - mins, 1e-12)
            n = max(12, min(72, int(n_grid)))
            surfaces: List[Dict[str, Any]] = []
            global_min = float("inf")
            global_max = float("-inf")
            d = len(c.domain.feature_names)
            for i in range(d):
                for j in range(i + 1, d):
                    x_vals = np.linspace(float(mins[i]), float(maxs[i]), n)
                    y_vals = np.linspace(float(mins[j]), float(maxs[j]), n)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    grid = np.full((n * n, d), 0.5, dtype=np.float64)
                    grid[:, i] = ((X.ravel() - mins[i]) / span[i]).clip(0.0, 1.0)
                    grid[:, j] = ((Y.ravel() - mins[j]) / span[j]).clip(0.0, 1.0)
                    grid_t = torch.tensor(grid, dtype=DTYPE, device=DEVICE)
                    with torch.no_grad():
                        vals = pref_prior.m0_torch(grid_t).reshape(n, n).detach().cpu().numpy()
                        forbidden = (
                            hard_constraint_mask(grid_t, ro_unit, c.domain.feature_names)
                            .reshape(n, n)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    local_min = float(np.nanmin(vals))
                    local_max = float(np.nanmax(vals))
                    global_min = min(global_min, local_min)
                    global_max = max(global_max, local_max)
                    surfaces.append(
                        {
                            "x_label": c.domain.feature_names[i],
                            "y_label": c.domain.feature_names[j],
                            "x_index": int(i),
                            "y_index": int(j),
                            "x_values": [float(v) for v in x_vals],
                            "y_values": [float(v) for v in y_vals],
                            "values": vals.tolist(),
                            "forbidden": forbidden.astype(int).tolist(),
                            "min": local_min,
                            "max": local_max,
                        }
                    )
            return {
                "surfaces": surfaces,
                "min": 0.0 if not np.isfinite(global_min) else global_min,
                "max": 1.0 if not np.isfinite(global_max) else global_max,
            }

    def _step_once(self) -> None:
        c = self.campaign
        s = c.settings
        domain = c.domain
        X_obs = c.X_obs
        Y_obs = c.Y_obs
        ro_unit = normalize_readout_to_unit_box(c.readout, domain.mins, domain.maxs, feature_names=domain.feature_names)
        prior = readout_to_prior(ro_unit, feature_names=domain.feature_names)
        t = len([row for row in c.history if int(row.get("iter", 0)) >= 0])

        try:
            from botorch.acquisition.analytic import ExpectedImprovement
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms.input import Normalize
            from botorch.models.transforms.outcome import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from gpytorch.settings import cholesky_jitter
            from prior_gp import GPWithPriorMean, fit_residual_gp

            gp_skeptic = SingleTaskGP(X_obs, Y_obs)
            mll_s = ExactMarginalLogLikelihood(gp_skeptic.likelihood, gp_skeptic)
            try:
                with cholesky_jitter(1e-4):
                    fit_gpytorch_mll(mll_s, max_attempts=5)
            except Exception:
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
            model_believer = GPWithPriorMean(gp_resid, prior, m0_scale=float(alpha * s.prior_strength))
            awcd = awcd_full_score(
                gp_skeptic,
                prior.m0_torch,
                X_obs=X_obs,
                Y_obs=Y_obs,
                readout_unit=ro_unit,
                feature_names=domain.feature_names,
                pool_n=int(s.constraint_pool_size),
                top_frac=float(s.awcd_top_frac),
                seed=int(s.seed) + 9000 + t,
                bounds=domain.unit_bounds,
            )
            awcd_score = float(awcd.get("awcd_total", 0.0))
            awcd_constraint = float(awcd.get("awcd_constraint", 0.0))
            awcd_mean = float(awcd.get("awcd_mean", 0.0))
            c.awcd_history.append(awcd_score)
            c.awcd_constraint_history.append(awcd_constraint)
            c.awcd_mean_history.append(awcd_mean)
            constraint_mean = float(np.mean(c.awcd_constraint_history[-int(s.awcd_window) :]))
            mean_mean = float(np.mean(c.awcd_mean_history[-int(s.awcd_window) :]))
            if t < int(s.awcd_warmup) or len(c.awcd_history) < int(s.awcd_window):
                prior_good = True
            else:
                prior_good = not (constraint_mean > float(s.c_user) or mean_mean > float(s.c_user))
            weight = 1.0 if prior_good else 0.0
            best_f = float(Y_obs.max().item())
            effective_hardness = float(s.constraint_hardness) if weight > 0.0 else 0.0
            ei_s = ExpectedImprovement(model=gp_skeptic, best_f=best_f, maximize=True)
            ei_b = ExpectedImprovement(model=model_believer, best_f=best_f, maximize=True)
            pool = draw_unit_sobol(
                bounds=domain.unit_bounds,
                n=max(1024, int(s.constraint_pool_size)),
                q=1,
                seed=int(s.seed) + 202 + t,
            ).squeeze(1)
            with torch.no_grad():
                score_s = ei_s(pool.unsqueeze(1)).reshape(-1)
                score_b = ei_b(pool.unsqueeze(1)).reshape(-1)
                if effective_hardness > 0.0:
                    score_b = apply_constraint_hardness(
                        score_b,
                        pool,
                        ro_unit,
                        domain.feature_names,
                        hardness=effective_hardness,
                        best_f=best_f,
                    )
                scores = float(weight) * score_b + (1.0 - float(weight)) * score_s
                x_next = pool[int(torch.argmax(scores))]
            y_next = evaluate_oracle(domain, x_next).unsqueeze(-1)
            c.X_obs = torch.cat([X_obs, x_next.unsqueeze(0)], dim=0)
            c.Y_obs = torch.cat([Y_obs, y_next], dim=0)
            best = float(c.Y_obs.max().item())
            rec = record_continuous_sample(
                domain,
                x_next,
                float(y_next.item()),
                best,
                method="hilo",
                iteration=t,
                extra={
                    "awcd_score": awcd_score,
                    "awcd_constraint": awcd_constraint,
                    "awcd_mean_disagree": awcd_mean,
                    "awcd_constraint_mean": constraint_mean,
                    "awcd_mean_disagree_mean": mean_mean,
                    "weight_believer": weight,
                    "prior_active": bool(prior_good),
                },
            )
            c.history.append(rec)
            c.last_error = None
        except Exception as exc:
            c.last_error = f"{type(exc).__name__}: {exc}"
            raise

    def _update_settings(self, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        s = self.campaign.settings
        for key, value in updates.items():
            if not hasattr(s, key):
                continue
            old = getattr(s, key)
            if isinstance(old, bool):
                setattr(s, key, bool(value))
            elif isinstance(old, int):
                setattr(s, key, int(value))
            elif isinstance(old, float):
                setattr(s, key, float(value))
            else:
                setattr(s, key, str(value))
        s.n_init = max(1, int(s.n_init))
        s.constraint_pool_size = max(512, int(s.constraint_pool_size))
        s.awcd_top_frac = float(np.clip(s.awcd_top_frac, 0.01, 0.5))
        s.c_user = float(np.clip(s.c_user, 0.2, 0.95))
        s.awcd_window = max(1, int(s.awcd_window))
        s.awcd_warmup = max(0, int(s.awcd_warmup))

    def _clean_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (np.bool_, bool)):
                clean[key] = bool(value)
            elif isinstance(value, (np.integer, int)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = None if not np.isfinite(float(value)) else float(value)
            else:
                clean[key] = value
        return clean


SERVICE = HiloService()


def dumps_state() -> str:
    return json.dumps(SERVICE.state(), indent=2)
