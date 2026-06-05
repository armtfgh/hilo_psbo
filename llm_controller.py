"""
Online C_user controllers for adaptive AWCD.

Four controller types share a common interface:

    Controller.decide(telemetry) -> ControllerDecision

so the BO driver does not care which one is in use. This file contains:

    - ControllerDecision : tiny dataclass for the return value
    - CuserController    : abstract base
    - StaticController   : C_user fixed to a constant value
    - LinearScheduleController : C_user(t) = max(c_min, c0 - decay * t)
    - RuleBasedController : if-then-else over AWCD-pressure + stagnation
    - LLMController      : delegates the decision to an LLM via
                            llm_study.get_llm_json; the prompts are loaded
                            from prompts/cuser_controller_*.txt

The controllers are deliberately stateless except for the optional
``history`` list of past decisions. That history is passed back to the LLM
controller in every prompt so it can be self-consistent across calls.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Decision object
# ---------------------------------------------------------------------------
@dataclass
class ControllerDecision:
    new_cuser: float
    rationale: str
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bounds & helpers
# ---------------------------------------------------------------------------
CUSER_LOW = 0.05
CUSER_HIGH = 0.99
MAX_DELTA = 0.60  # cap absolute movement per call (relaxed from 0.30 after
                  # observation that LLM intent was being clipped on every
                  # call in adversarial regimes, producing a spurious linear
                  # descent in the mean trajectory)


def _clip_cuser(value: float) -> float:
    return float(max(CUSER_LOW, min(CUSER_HIGH, float(value))))


def _bounded_delta(current: float, proposed: float) -> float:
    delta = proposed - current
    if abs(delta) > MAX_DELTA:
        delta = MAX_DELTA if delta > 0 else -MAX_DELTA
    return _clip_cuser(current + delta)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class CuserController:
    """Abstract controller. Subclasses implement :meth:`decide`."""

    name: str = "abstract"

    def decide(self, telemetry: Dict[str, Any]) -> ControllerDecision:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Static / linear / rule-based controllers
# ---------------------------------------------------------------------------
class StaticController(CuserController):
    name = "static"

    def __init__(self, cuser: float):
        self.cuser = _clip_cuser(cuser)

    def decide(self, telemetry: Dict[str, Any]) -> ControllerDecision:
        return ControllerDecision(new_cuser=self.cuser,
                                  rationale=f"static={self.cuser:.2f}")


class LinearScheduleController(CuserController):
    name = "linear"

    def __init__(self, c0: float = 0.95, c_min: float = 0.4,
                 decay_per_iter: float = 0.005):
        self.c0 = _clip_cuser(c0)
        self.c_min = _clip_cuser(c_min)
        self.decay = float(decay_per_iter)

    def decide(self, telemetry: Dict[str, Any]) -> ControllerDecision:
        t = int(telemetry.get("iteration", 0))
        proposed = max(self.c_min, self.c0 - self.decay * t)
        return ControllerDecision(
            new_cuser=_clip_cuser(proposed),
            rationale=f"linear schedule, t={t}",
        )


class RuleBasedController(CuserController):
    """Threshold-and-stagnation rules over the same telemetry the LLM sees.

    Rules (evaluated in order, first match wins; otherwise no change):

      1. If AWCD pressure mean over the last ``window`` checkpoints exceeds
         ``high_pressure`` AND best-so-far has stagnated for at least
         ``stagnation`` iterations, decrement C_user by ``step_down``.
      2. If AWCD pressure mean stays below ``low_pressure`` for the last
         ``window`` checkpoints AND best-so-far is improving, increment
         C_user by ``step_up`` (but never above the original c0).
    """
    name = "rule_based"

    def __init__(self,
                 c0: float = 0.95,
                 window: int = 3,
                 high_pressure: float = 0.50,
                 low_pressure: float = 0.20,
                 stagnation: int = 5,
                 step_down: float = 0.20,
                 step_up: float = 0.10):
        self.c0 = _clip_cuser(c0)
        self.window = int(window)
        self.high_pressure = float(high_pressure)
        self.low_pressure = float(low_pressure)
        self.stagnation = int(stagnation)
        self.step_down = float(step_down)
        self.step_up = float(step_up)

    def decide(self, telemetry: Dict[str, Any]) -> ControllerDecision:
        current = float(telemetry.get("current_cuser", self.c0))
        awcd_hist = list(telemetry.get("awcd_score_history") or [])
        bsf_hist = list(telemetry.get("best_so_far_history") or [])
        stag = int(telemetry.get("stagnation_iters", 0))

        recent_pressure = awcd_hist[: self.window]
        mean_pressure = (sum(recent_pressure) / len(recent_pressure)
                          if recent_pressure else 0.0)

        improving = False
        if len(bsf_hist) >= 2:
            improving = bsf_hist[0] > bsf_hist[-1]

        if mean_pressure > self.high_pressure and stag >= self.stagnation:
            new_c = _bounded_delta(current, current - self.step_down)
            return ControllerDecision(
                new_cuser=new_c,
                rationale=(f"AWCD pressure mean {mean_pressure:.2f} > "
                            f"{self.high_pressure:.2f} and stagnation "
                            f"{stag} iters -> lower C_user"),
            )
        if mean_pressure < self.low_pressure and improving:
            target = min(self.c0, current + self.step_up)
            new_c = _bounded_delta(current, target)
            return ControllerDecision(
                new_cuser=new_c,
                rationale=(f"AWCD pressure mean {mean_pressure:.2f} < "
                            f"{self.low_pressure:.2f} and improving -> "
                            "restore C_user toward initial"),
            )
        return ControllerDecision(
            new_cuser=current,
            rationale=(f"no rule triggered (pressure={mean_pressure:.2f}, "
                        f"stagnation={stag})"),
        )


# ---------------------------------------------------------------------------
# LLM controller
# ---------------------------------------------------------------------------
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "cuser_controller_system.txt")
_USER_TEMPLATE_PATH = os.path.join(_PROMPTS_DIR, "cuser_controller_user_template.txt")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _fmt_list(values: List[float], *, max_n: int = 12, fmt: str = ".3f") -> str:
    if not values:
        return "[]"
    short = values[:max_n]
    parts = [format(float(v), fmt) for v in short]
    return "[" + ", ".join(parts) + ("" if len(values) <= max_n else ", ...") + "]"


def _fmt_decisions(decisions: List[Dict[str, Any]], *, max_n: int = 6) -> str:
    if not decisions:
        return "[]"
    rows = []
    for d in decisions[:max_n]:
        rows.append(f"  iter={d.get('iteration')}, cuser={float(d.get('new_cuser', float('nan'))):.2f}, "
                    f"why=\"{str(d.get('rationale', ''))[:140]}\"")
    return "[\n" + "\n".join(rows) + "\n]"


def _build_user_prompt(template: str, telemetry: Dict[str, Any]) -> str:
    return template.format(
        dataset=telemetry.get("dataset", "?"),
        iteration=int(telemetry.get("iteration", 0)),
        total_iterations=int(telemetry.get("total_iterations", 0)),
        current_cuser=float(telemetry.get("current_cuser", 0.95)),
        prior_active_str=("active" if bool(telemetry.get("prior_active", True))
                          else "DISABLED"),
        stagnation_iters=int(telemetry.get("stagnation_iters", 0)),
        bsf_relative_gain=float(telemetry.get("bsf_relative_gain", 0.0)),
        bsf_improved_in_window=str(bool(telemetry.get("bsf_improved_in_window", False))),
        awcd_pressure_recent_mean=float(telemetry.get("awcd_pressure_recent_mean", 0.0)),
        awcd_pressure_trend_slope=float(telemetry.get("awcd_pressure_trend_slope", 0.0)),
        prior_disabled_streak=int(telemetry.get("prior_disabled_streak", 0)),
        expert_prompt=str(telemetry.get("expert_prompt", "")),
        readout_json=json.dumps(telemetry.get("readout", {}), indent=2),
        n_history=int(telemetry.get("n_history", 12)),
        awcd_score_history=_fmt_list(telemetry.get("awcd_score_history") or []),
        awcd_constraint_history=_fmt_list(
            telemetry.get("awcd_constraint_history") or []),
        best_so_far_history=_fmt_list(telemetry.get("best_so_far_history") or []),
        previous_decisions=_fmt_decisions(telemetry.get("previous_decisions") or []),
    )


def _parse_decision(payload: Dict[str, Any], *, current: float) -> ControllerDecision:
    raw_new = payload.get("new_cuser", current)
    try:
        new_c = float(raw_new)
    except (TypeError, ValueError):
        new_c = float(current)
    proposed = _bounded_delta(current, new_c)
    rationale = str(payload.get("rationale", "")).strip() or "(no rationale)"
    return ControllerDecision(new_cuser=proposed, rationale=rationale,
                               raw=dict(payload))


class LLMController(CuserController):
    """C_user controller backed by an LLM (via ``llm_study.get_llm_json``).

    Failures (network, malformed JSON, schema issues) fall back to "no
    change" with a logged rationale.

    Parameters
    ----------
    model : str
        Model key in ``llm_study.MODEL_REGISTRY`` (e.g. ``"gpt-4o-mini"``,
        ``"gpt-4o"``, ``"claude-haiku-4-5-20251001"``).
    temperature : float
        Sampling temperature passed to ``get_llm_json``.
    cache_dir : Optional[str]
        If provided, every prompt + raw response + parsed decision is
        cached as JSON for later replay/audit.
    """
    name = "llm"

    def __init__(self,
                 model: str,
                 *,
                 temperature: float = 0.2,
                 cache_dir: Optional[str] = None,
                 max_tokens: int = 400,
                 run_tag: str = "run"):
        from llm_study import get_llm_json  # late import; preserves test
        self._get_llm_json: Callable[..., Dict[str, Any]] = get_llm_json
        self.model = str(model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = _read_text(_SYSTEM_PROMPT_PATH)
        self.user_template = _read_text(_USER_TEMPLATE_PATH)
        self.cache_dir = cache_dir
        self.run_tag = str(run_tag)
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, iteration: int) -> Optional[str]:
        if self.cache_dir is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.run_tag)
        return os.path.join(self.cache_dir,
                             f"{safe}_iter{int(iteration):04d}.json")

    def decide(self, telemetry: Dict[str, Any]) -> ControllerDecision:
        current = float(telemetry.get("current_cuser", 0.95))
        iteration = int(telemetry.get("iteration", 0))
        user_prompt = _build_user_prompt(self.user_template, telemetry)
        record: Dict[str, Any] = {
            "model": self.model,
            "iteration": iteration,
            "temperature": self.temperature,
            "telemetry": telemetry,
            "user_prompt": user_prompt,
            "system_prompt_sha_first120": self.system_prompt[:120],
        }
        try:
            payload = self._get_llm_json(
                self.model,
                user_prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            decision = _parse_decision(payload, current=current)
            record["response"] = payload
            record["decision"] = {"new_cuser": decision.new_cuser,
                                   "rationale": decision.rationale}
        except Exception as exc:  # noqa: BLE001 - explicit soft fallback
            decision = ControllerDecision(
                new_cuser=current,
                rationale=f"LLM call failed ({type(exc).__name__}): {exc}; "
                          "keeping previous C_user",
            )
            record["error"] = repr(exc)
            record["decision"] = {"new_cuser": decision.new_cuser,
                                   "rationale": decision.rationale}

        cache_path = self._cache_path(iteration)
        if cache_path is not None:
            try:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, indent=2, default=str)
            except OSError:
                pass
        return decision


__all__ = [
    "ControllerDecision",
    "CuserController",
    "StaticController",
    "LinearScheduleController",
    "RuleBasedController",
    "LLMController",
]
