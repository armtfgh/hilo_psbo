"""Domain-agnostic Ask-Tell Bayesian optimizer with human prior shaping.

A practical, manual human-in-the-loop loop that is NOT tied to the UGI case:
the user defines their own continuous parameters and objective, the optimizer
proposes a batch of conditions to run, the user measures them and feeds the
results back, and the optimizer returns the next batch.

Like the UGI study, the human can shape a prior from natural-language knowledge:
free text is translated (by the same LLM translator) into a structured readout
(directional effects, peaks, soft constraints) over the user's own parameters.
That prior is fused with the data through a "believer" GP and gated by the AWCD
safety score, exactly as in the HILO method — only here the experiment is the
user's, supplied one batch at a time.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def _sobol(d: int, n: int, seed: int) -> torch.Tensor:
    eng = torch.quasirandom.SobolEngine(dimension=int(d), scramble=True, seed=int(seed))
    return eng.draw(int(max(1, n))).to(device=DEVICE, dtype=DTYPE)


def _ndigits(p: Dict[str, Any]) -> int:
    span = abs(float(p["max"]) - float(p["min"]))
    if span < 1:
        return 4
    if span < 10:
        return 3
    if span < 100:
        return 2
    return 1


def _flat_readout(names: List[str]) -> Dict[str, Any]:
    return {
        "effects": {n: {"effect": "flat", "scale": 0.0, "confidence": 0.0} for n in names},
        "bumps": [],
        "constraints": [],
    }


@dataclass
class AskTellState:
    configured: bool = False
    objective_name: str = "objective"
    goal: str = "maximize"
    parameters: List[Dict[str, Any]] = field(default_factory=list)  # {name,min,max,unit}
    batch_size: int = 3
    n_init: int = 5
    seed: int = 0
    round: int = 0
    X: List[List[float]] = field(default_factory=list)  # raw param values, parameter order
    Y: List[float] = field(default_factory=list)        # raw objective as entered
    pending: List[Dict[str, Any]] = field(default_factory=list)  # {id, x:{name:val}}
    counter: int = 0
    # ---- human prior shaping ----
    readout: Dict[str, Any] = field(default_factory=dict)
    prior_strength: float = 1.0
    c_user: float = 0.95
    constraint_hardness: float = 0.2
    awcd_top_frac: float = 0.05
    constraint_pool_size: int = 4096
    # telemetry (one entry per GP-driven suggestion)
    awcd_history: List[float] = field(default_factory=list)
    weight_history: List[float] = field(default_factory=list)
    prior_active: bool = False
    last_awcd: Optional[float] = None


class AskTellService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.s = AskTellState()

    # ----- lifecycle -------------------------------------------------------
    def reset_all(self) -> Dict[str, Any]:
        with self._lock:
            self.s = AskTellState()
            return self.state()

    def configure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            params: List[Dict[str, Any]] = []
            for p in payload.get("parameters", []) or []:
                name = str(p.get("name") or "").strip()
                if not name:
                    continue
                try:
                    lo = float(p["min"])
                    hi = float(p["max"])
                except Exception:
                    continue
                if hi < lo:
                    lo, hi = hi, lo
                if hi == lo:
                    hi = lo + 1e-6
                params.append({"name": name, "min": lo, "max": hi, "unit": str(p.get("unit") or "")})
            if not params:
                raise ValueError("Define at least one parameter with a numeric min and max.")
            names = [p["name"] for p in params]
            if len(set(names)) != len(names):
                raise ValueError("Parameter names must be unique.")
            s = AskTellState()
            s.configured = True
            s.objective_name = str(payload.get("objective_name") or "objective").strip() or "objective"
            s.goal = "minimize" if str(payload.get("goal", "maximize")).lower().startswith("min") else "maximize"
            s.parameters = params
            s.batch_size = max(1, min(12, int(payload.get("batch_size", 3) or 3)))
            s.n_init = max(1, min(48, int(payload.get("n_init", 5) or 5)))
            s.seed = int(payload.get("seed", 0) or 0)
            s.c_user = float(payload.get("c_user", 0.95) or 0.95)
            s.prior_strength = float(payload.get("prior_strength", 1.0) or 1.0)
            s.readout = _flat_readout(names)
            self.s = s
            self._suggest(initial=True)
            return self.state()

    # ----- prior shaping ---------------------------------------------------
    def set_readout(self, readout: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if not self.s.configured:
                raise ValueError("Configure a campaign first.")
            if not isinstance(readout, dict):
                raise ValueError("Readout must be a JSON object.")
            ro = dict(readout)
            ro.setdefault("effects", {})
            ro.setdefault("bumps", [])
            ro.setdefault("constraints", [])
            self.s.readout = ro
            return self.state()

    def clear_readout(self) -> Dict[str, Any]:
        with self._lock:
            if not self.s.configured:
                raise ValueError("Configure a campaign first.")
            self.s.readout = _flat_readout([p["name"] for p in self.s.parameters])
            return self.state()

    def translate(self, *, transcript: str, model: str, temperature: float = 0.0,
                  api_key: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if not self.s.configured:
                raise ValueError("Configure a campaign first.")
            s = self.s
        os.environ.setdefault("ANTHROPIC_VERSION", "2023-06-01")
        from llm_study import get_llm_json

        param_lines = "\n".join(
            f"- {p['name']}: [{p['min']:g}, {p['max']:g}]{(' ' + p['unit']) if p['unit'] else ''}"
            for p in s.parameters
        )
        system = (
            "You translate a human expert's natural-language knowledge into a STRUCTURED JSON prior "
            "for Bayesian optimization. Return STRICT JSON only, with keys: effects, bumps, constraints.\n\n"
            f"Objective: {s.objective_name} (the user wants to {s.goal} it).\n"
            "Parameters (name: [min, max] unit):\n"
            f"{param_lines}\n\n"
            "Express every effect in terms of how the parameter changes how GOOD the result is "
            f"(i.e. progress toward the goal of {s.goal}-ing {s.objective_name}):\n"
            "- effects: {param: {effect, scale, confidence, range_hint}}\n"
            "  effect must be one of: increasing (larger is better), decreasing (smaller is better), "
            "flat, nonmonotone-peak (an intermediate value is best), nonmonotone-valley.\n"
            "  scale in [0.0,1.5], confidence in [0.0,0.95], range_hint is [lo,hi] in RAW units.\n"
            "- constraints: [{var, range:[lo,hi], penalty, reason}] where range is the FORBIDDEN region "
            "(raw units), penalty in [0.0,10.0].\n"
            "- bumps: [{mu:[one value per parameter, in parameter order, raw units], sigma, amp}], amp in [0.0,0.6].\n\n"
            "Rules:\n"
            "- Use ONLY the expert text and the parameters above. Map the expert's words to the parameter "
            "names (case-insensitive; obvious synonyms are fine); ignore anything that does not map.\n"
            "- 'avoid/forbid/bad/poor below T' becomes a constraint with range [min, T]; "
            "'above T' becomes [T, max].\n"
            "- 'higher/more is better' -> increasing; 'lower/less is better' -> decreasing; "
            "'best around V' or 'mid-range' -> nonmonotone-peak with range_hint bracketing V.\n"
            "- Ambiguous language -> lower confidence and broader ranges. Omit parameters not mentioned.\n"
            "- Do not invent precise numbers that the expert did not state."
        )
        payload = {
            "parameters": [{"name": p["name"], "min": p["min"], "max": p["max"], "unit": p["unit"]} for p in s.parameters],
            "objective": s.objective_name,
            "goal": s.goal,
            "current_readout": s.readout,
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
        return readout

    def prior_surfaces(self, n_grid: int = 40) -> Dict[str, Any]:
        with self._lock:
            s = self.s
            if not s.configured:
                return {"surfaces": [], "min": 0.0, "max": 1.0}
            from readout_schema import normalize_readout_to_unit_box, readout_to_prior
            from hilo_service import hard_constraint_mask

            names = [p["name"] for p in s.parameters]
            mins = np.array([p["min"] for p in s.parameters], dtype=float)
            maxs = np.array([p["max"] for p in s.parameters], dtype=float)
            mins_t = torch.tensor(mins, dtype=DTYPE, device=DEVICE)
            maxs_t = torch.tensor(maxs, dtype=DTYPE, device=DEVICE)
            span = np.maximum(maxs - mins, 1e-12)
            d = len(names)
            ro_unit = normalize_readout_to_unit_box(s.readout, mins_t, maxs_t, feature_names=names)
            pref_unit = dict(ro_unit)
            pref_unit["constraints"] = []
            pref_prior = readout_to_prior(pref_unit, feature_names=names)
            n = max(12, min(64, int(n_grid)))
            surfaces: List[Dict[str, Any]] = []
            gmin, gmax = float("inf"), float("-inf")
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
                            hard_constraint_mask(grid_t, ro_unit, names).reshape(n, n).detach().cpu().numpy()
                        )
                    lmin, lmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                    gmin, gmax = min(gmin, lmin), max(gmax, lmax)
                    surfaces.append({
                        "x_label": names[i], "y_label": names[j],
                        "x_index": int(i), "y_index": int(j),
                        "x_values": [float(v) for v in x_vals],
                        "y_values": [float(v) for v in y_vals],
                        "values": vals.tolist(),
                        "forbidden": forbidden.astype(int).tolist(),
                        "min": lmin, "max": lmax,
                    })
            return {
                "surfaces": surfaces,
                "min": 0.0 if not np.isfinite(gmin) else gmin,
                "max": 1.0 if not np.isfinite(gmax) else gmax,
            }

    def _has_prior(self) -> bool:
        ro = self.s.readout or {}
        for spec in (ro.get("effects") or {}).values():
            if isinstance(spec, dict):
                eff = str(spec.get("effect", "flat")).lower()
                amp = float(spec.get("scale", 0.0)) * float(spec.get("confidence", 0.0))
                if eff != "flat" and amp > 0.0:
                    return True
        if ro.get("constraints"):
            return True
        if ro.get("bumps"):
            return True
        return False

    # ----- transforms ------------------------------------------------------
    def _bounds(self):
        lo = torch.tensor([p["min"] for p in self.s.parameters], dtype=DTYPE, device=DEVICE)
        hi = torch.tensor([p["max"] for p in self.s.parameters], dtype=DTYPE, device=DEVICE)
        return lo, hi

    def _unit_bounds(self, d: int) -> torch.Tensor:
        return torch.stack([torch.zeros(d, dtype=DTYPE, device=DEVICE), torch.ones(d, dtype=DTYPE, device=DEVICE)], dim=0)

    def _to_unit(self, x_raw: torch.Tensor) -> torch.Tensor:
        lo, hi = self._bounds()
        return ((x_raw - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)

    def _to_raw(self, x_unit: torch.Tensor) -> torch.Tensor:
        lo, hi = self._bounds()
        return lo + x_unit * (hi - lo)

    def _signed_Y(self) -> torch.Tensor:
        y = torch.tensor(self.s.Y, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        return -y if self.s.goal == "minimize" else y

    def _emit(self, chosen_unit: torch.Tensor) -> None:
        s = self.s
        raw = self._to_raw(chosen_unit).detach().cpu().numpy()
        pending: List[Dict[str, Any]] = []
        for row in raw:
            s.counter += 1
            x = {p["name"]: float(round(float(v), _ndigits(p))) for p, v in zip(s.parameters, row)}
            pending.append({"id": int(s.counter), "x": x})
        s.pending = pending

    # ----- core ------------------------------------------------------------
    def _suggest(self, initial: bool = False) -> None:
        s = self.s
        d = len(s.parameters)
        names = [p["name"] for p in s.parameters]
        k = s.n_init if (initial or len(s.Y) == 0) else s.batch_size

        if len(s.Y) < max(2, d + 1):
            self._emit(_sobol(d, k, s.seed + 1000 + s.round))
            s.prior_active = False
            s.last_awcd = None
            return

        from botorch.acquisition.analytic import ExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.settings import cholesky_jitter

        Xu = self._to_unit(torch.tensor(s.X, dtype=DTYPE, device=DEVICE))
        y = self._signed_Y()
        # standardize the objective so the prior (O(1) amplitudes) is meaningful at any scale.
        mu = y.mean()
        sd = y.std().clamp_min(1e-6)
        z = (y - mu) / sd

        gp_s = SingleTaskGP(Xu, z)
        mll = ExactMarginalLogLikelihood(gp_s.likelihood, gp_s)
        try:
            with cholesky_jitter(1e-4):
                fit_gpytorch_mll(mll, max_attempts=5)
        except Exception:
            pass
        best_f = float(z.max().item())
        pool = _sobol(d, max(2048, 256 * k), s.seed + 7 + s.round)
        ei_s = ExpectedImprovement(model=gp_s, best_f=best_f, maximize=True)
        with torch.no_grad():
            score_s = ei_s(pool.unsqueeze(1)).reshape(-1)

        awcd_score: Optional[float] = None
        weight = 0.0
        prior_on = False
        scores = score_s

        if self._has_prior():
            from readout_schema import normalize_readout_to_unit_box, readout_to_prior
            from prior_gp import GPWithPriorMean, fit_residual_gp
            from hilo_service import apply_constraint_hardness, awcd_full_score

            mins_t, maxs_t = self._bounds()
            ro_unit = normalize_readout_to_unit_box(s.readout, mins_t, maxs_t, feature_names=names)
            prior = readout_to_prior(ro_unit, feature_names=names)
            try:
                gp_resid, alpha = fit_residual_gp(Xu, z, prior)
                believer = GPWithPriorMean(gp_resid, prior, m0_scale=float(alpha * s.prior_strength))
                awcd = awcd_full_score(
                    gp_s, prior.m0_torch,
                    X_obs=Xu, Y_obs=z, readout_unit=ro_unit, feature_names=names,
                    pool_n=int(s.constraint_pool_size), top_frac=float(s.awcd_top_frac),
                    seed=int(s.seed) + 9000 + s.round, bounds=self._unit_bounds(d),
                )
                awcd_score = float(awcd.get("awcd_total", 0.0))
                prior_on = awcd_score <= float(s.c_user)
                weight = 1.0 if prior_on else 0.0
                ei_b = ExpectedImprovement(model=believer, best_f=best_f, maximize=True)
                with torch.no_grad():
                    score_b = ei_b(pool.unsqueeze(1)).reshape(-1)
                    if prior_on and float(s.constraint_hardness) > 0.0:
                        score_b = apply_constraint_hardness(
                            score_b, pool, ro_unit, names,
                            hardness=float(s.constraint_hardness), best_f=best_f,
                        )
                    scores = weight * score_b + (1.0 - weight) * score_s
            except Exception:
                awcd_score = None
                scores = score_s

        # greedy top-k with min-distance diversity
        order = torch.argsort(scores, descending=True).tolist()
        chosen_idx: List[int] = []
        min_d = 0.05
        for idx in order:
            cand = pool[idx]
            if all(float(torch.norm(cand - pool[j])) > min_d for j in chosen_idx):
                chosen_idx.append(idx)
            if len(chosen_idx) >= k:
                break
        if not chosen_idx:
            chosen_idx = order[:k]

        s.awcd_history.append(float(awcd_score) if awcd_score is not None else float("nan"))
        s.weight_history.append(float(weight))
        s.prior_active = bool(prior_on)
        s.last_awcd = None if awcd_score is None else float(awcd_score)
        self._emit(pool[chosen_idx])

    def tell(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            s = self.s
            if not s.configured:
                raise ValueError("Configure a campaign first.")
            added = 0
            for r in results or []:
                yv = r.get("y")
                if yv is None or yv == "":
                    continue
                try:
                    yval = float(yv)
                except Exception:
                    continue
                if not np.isfinite(yval):
                    continue
                x = r.get("x") or {}
                row = [float(x.get(p["name"], (p["min"] + p["max"]) / 2.0)) for p in s.parameters]
                s.X.append(row)
                s.Y.append(yval)
                added += 1
            if added == 0:
                raise ValueError("Enter at least one numeric result before submitting.")
            s.round += 1
            self._suggest()
            return self.state()

    def suggest_again(self) -> Dict[str, Any]:
        with self._lock:
            if not self.s.configured:
                raise ValueError("Configure a campaign first.")
            self._suggest()
            return self.state()

    # ----- serialization ---------------------------------------------------
    def state(self) -> Dict[str, Any]:
        s = self.s
        if not s.configured:
            return {"configured": False}
        best = None
        if s.Y:
            arr = np.asarray(s.Y, dtype=float)
            bi = int(np.argmin(arr)) if s.goal == "minimize" else int(np.argmax(arr))
            best = {"y": float(s.Y[bi]), "x": {p["name"]: s.X[bi][i] for i, p in enumerate(s.parameters)}}
        series = []
        cur: Optional[float] = None
        for i, yv in enumerate(s.Y):
            cur = yv if cur is None else (min(cur, yv) if s.goal == "minimize" else max(cur, yv))
            series.append({"n": i + 1, "y": float(yv), "best": float(cur)})
        history = [
            {"n": i + 1, "x": {p["name"]: s.X[i][k] for k, p in enumerate(s.parameters)}, "y": float(s.Y[i])}
            for i in range(len(s.Y))
        ]
        return {
            "configured": True,
            "objective_name": s.objective_name,
            "goal": s.goal,
            "parameters": s.parameters,
            "batch_size": s.batch_size,
            "n_init": s.n_init,
            "round": s.round,
            "n_observations": len(s.Y),
            "best": best,
            "pending": s.pending,
            "history": history,
            "series": series,
            # prior shaping
            "readout": s.readout,
            "c_user": s.c_user,
            "prior_active": bool(s.prior_active),
            "awcd": s.last_awcd,
            "awcd_history": [None if (a != a) else float(a) for a in s.awcd_history],
            "weight_history": [float(w) for w in s.weight_history],
            "has_prior": self._has_prior(),
        }


ASKTELL = AskTellService()
