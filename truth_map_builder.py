"""
Fit a differentiable prior surface to the oracle and export a JSON readout.

This module provides:
  - TrainablePrior: a torch.nn.Module version of Prior.m0_torch
  - fit_oracle_prior: gradient-based fitting to the oracle surface
  - fit_oracle_to_json: export a manual_readout-style dict after training
"""
#%%
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DTYPE = torch.float32
EPS = 1e-6


def _safe_logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(math.log(p / (1.0 - p)))


def _inv_softplus(x: float) -> float:
    x = float(max(x, 1e-8))
    return float(math.log(math.expm1(x)))


def _resolve_feature_index(name: str, feature_names: List[str]) -> Optional[int]:
    label = str(name).strip()
    if label.lower().startswith("x") and label[1:].isdigit():
        idx = int(label[1:]) - 1
        return idx if 0 <= idx < len(feature_names) else None
    for i, fname in enumerate(feature_names):
        if label == fname or label.lower() == fname.lower():
            return i
    return None


def _unit_to_raw(mins: torch.Tensor, maxs: torch.Tensor, x_unit: torch.Tensor) -> torch.Tensor:
    rng = (maxs - mins).clamp_min(EPS)
    return mins + x_unit * rng


@dataclass
class EffectSpec:
    name: str
    kind: str


class GaussianEffect(nn.Module):
    def __init__(
        self,
        idx: int,
        kind: str,
        *,
        mu_init: float = 0.5,
        sigma_init: float = 0.1,
        amp_init: float = 0.2,
    ) -> None:
        super().__init__()
        self.idx = int(idx)
        self.kind = kind
        self.mu_raw = nn.Parameter(torch.tensor(_safe_logit(mu_init), dtype=DTYPE))
        self.sigma_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_init), dtype=DTYPE))
        self.amp_raw = nn.Parameter(torch.tensor(_inv_softplus(abs(amp_init)), dtype=DTYPE))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mu = torch.sigmoid(self.mu_raw)
        sigma = F.softplus(self.sigma_raw) + EPS
        amp = F.softplus(self.amp_raw)
        z = X[..., self.idx]
        gauss = torch.exp(-0.5 * ((z - mu) / sigma) ** 2)
        sign = 1.0 if self.kind in {"peak", "nonmonotone-peak"} else -1.0
        return sign * amp * gauss

    def export(self) -> Tuple[float, float, float]:
        mu = float(torch.sigmoid(self.mu_raw).detach().cpu().item())
        sigma = float((F.softplus(self.sigma_raw) + EPS).detach().cpu().item())
        amp = float(F.softplus(self.amp_raw).detach().cpu().item())
        return mu, sigma, amp


class SigmoidEffect(nn.Module):
    def __init__(
        self,
        idx: int,
        kind: str,
        *,
        center_init: float = 0.5,
        k_init: float = 6.0,
        scale_init: float = 0.4,
    ) -> None:
        super().__init__()
        self.idx = int(idx)
        self.kind = kind
        self.center_raw = nn.Parameter(torch.tensor(_safe_logit(center_init), dtype=DTYPE))
        self.k_raw = nn.Parameter(torch.tensor(_inv_softplus(k_init), dtype=DTYPE))
        self.scale_raw = nn.Parameter(torch.tensor(_inv_softplus(scale_init), dtype=DTYPE))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        center = torch.sigmoid(self.center_raw)
        k = F.softplus(self.k_raw) + EPS
        scale = F.softplus(self.scale_raw)
        z = X[..., self.idx]
        sig = torch.sigmoid(k * (z - center))
        sign = 1.0 if self.kind in {"increasing", "increase"} else -1.0
        return sign * scale * sig

    def export(self) -> Tuple[float, float, float]:
        center = float(torch.sigmoid(self.center_raw).detach().cpu().item())
        k = float((F.softplus(self.k_raw) + EPS).detach().cpu().item())
        scale = float(F.softplus(self.scale_raw).detach().cpu().item())
        return center, k, scale


class InteractionEffect(nn.Module):
    def __init__(self, idx_a: int, idx_b: int, kind: str, scale_init: float = 0.2) -> None:
        super().__init__()
        self.idx_a = int(idx_a)
        self.idx_b = int(idx_b)
        self.kind = kind
        self.scale_raw = nn.Parameter(torch.tensor(_inv_softplus(scale_init), dtype=DTYPE))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.scale_raw)
        sign = 1.0 if self.kind in {"synergy", "positive"} else -1.0
        return sign * scale * (X[..., self.idx_a] * X[..., self.idx_b])

    def export(self) -> float:
        return float(F.softplus(self.scale_raw).detach().cpu().item())


class TrainablePrior(nn.Module):
    def __init__(
        self,
        *,
        feature_names: List[str],
        structure_template: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.feature_names = [str(n) for n in feature_names]
        self.effects = nn.ModuleList()
        self.effect_specs: List[EffectSpec] = []
        self.interactions = nn.ModuleList()
        self.interaction_specs: List[Tuple[int, int, str]] = []

        template = structure_template
        effects_template = template.get("effects", template)
        for name, spec in (effects_template or {}).items():
            if name == "interactions":
                continue
            kind, params = _parse_effect_spec(spec)
            idx = _resolve_feature_index(str(name), self.feature_names)
            if idx is None:
                continue
            if kind in {"peak", "nonmonotone-peak", "valley", "nonmonotone-valley"}:
                effect = GaussianEffect(
                    idx,
                    kind="peak" if "peak" in kind else "valley",
                    mu_init=params.get("mu", 0.5),
                    sigma_init=params.get("sigma", 0.1),
                    amp_init=params.get("amp", 0.2),
                )
            elif kind in {"increasing", "decreasing", "increase", "decrease"}:
                effect = SigmoidEffect(
                    idx,
                    kind="increasing" if "inc" in kind else "decreasing",
                    center_init=params.get("center", 0.5),
                    k_init=params.get("k", 6.0),
                    scale_init=params.get("scale", 0.4),
                )
            else:
                continue
            self.effects.append(effect)
            self.effect_specs.append(EffectSpec(name=str(name), kind=kind))

        interactions = template.get("interactions", [])
        if isinstance(interactions, dict):
            interactions = [interactions]
        for spec in interactions or []:
            if not isinstance(spec, dict):
                continue
            pair = spec.get("vars") or spec.get("pair") or spec.get("indices")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            idx_a = _resolve_feature_index(str(pair[0]), self.feature_names)
            idx_b = _resolve_feature_index(str(pair[1]), self.feature_names)
            if idx_a is None or idx_b is None:
                continue
            kind = str(spec.get("type", "synergy")).lower()
            scale_init = float(spec.get("scale", 0.2))
            inter = InteractionEffect(idx_a, idx_b, kind=kind, scale_init=scale_init)
            self.interactions.append(inter)
            self.interaction_specs.append((idx_a, idx_b, kind))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        for eff in self.effects:
            out = out + eff(X)
        for inter in self.interactions:
            out = out + inter(X)
        return out

    def export_parameters(self, mins: torch.Tensor, maxs: torch.Tensor) -> Dict[str, Any]:
        effects_out: Dict[str, Any] = {}
        for spec, eff in zip(self.effect_specs, self.effects):
            name = spec.name
            if isinstance(eff, GaussianEffect):
                mu_u, sigma_u, amp = eff.export()
                mu_u = float(np.clip(mu_u, 0.0, 1.0))
                sigma_u = float(np.clip(sigma_u, EPS, 1.0))
                lo_u = max(mu_u - sigma_u, 0.0)
                hi_u = min(mu_u + sigma_u, 1.0)
                mu_raw = float(_unit_to_raw(mins, maxs, torch.tensor(mu_u)).item())
                span = (maxs - mins).clamp_min(EPS)
                sigma_raw = float((span * sigma_u).mean().item())
                lo_raw = float(_unit_to_raw(mins, maxs, torch.tensor(lo_u)).item())
                hi_raw = float(_unit_to_raw(mins, maxs, torch.tensor(hi_u)).item())
                effects_out[name] = {
                    "effect": "nonmonotone-peak" if "peak" in spec.kind else "nonmonotone-valley",
                    "scale": float(amp),
                    "confidence": 1.0,
                    "range_hint": [lo_raw, hi_raw],
                    "mu": mu_raw,
                    "sigma": sigma_raw,
                }
            elif isinstance(eff, SigmoidEffect):
                center_u, k, scale = eff.export()
                center_u = float(np.clip(center_u, 0.0, 1.0))
                width_u = float(np.clip(1.0 / (k + EPS), 0.03, 0.5))
                lo_u = max(center_u - width_u, 0.0)
                hi_u = min(center_u + width_u, 1.0)
                lo_raw = float(_unit_to_raw(mins, maxs, torch.tensor(lo_u)).item())
                hi_raw = float(_unit_to_raw(mins, maxs, torch.tensor(hi_u)).item())
                effects_out[name] = {
                    "effect": "increasing" if "inc" in spec.kind else "decreasing",
                    "scale": float(scale),
                    "confidence": 1.0,
                    "range_hint": [lo_raw, hi_raw],
                    "center": float(_unit_to_raw(mins, maxs, torch.tensor(center_u)).item()),
                    "k": float(k),
                }

        inter_out: List[Dict[str, Any]] = []
        for (idx_a, idx_b, kind), inter in zip(self.interaction_specs, self.interactions):
            inter_out.append(
                {
                    "vars": [self.feature_names[idx_a], self.feature_names[idx_b]],
                    "type": kind,
                    "scale": inter.export(),
                    "confidence": 1.0,
                }
            )

        return {
            "effects": effects_out,
            "interactions": inter_out,
            "bumps": [],
            "constraints": [],
        }


def _parse_effect_spec(spec: Any) -> Tuple[str, Dict[str, float]]:
    params: Dict[str, float] = {}
    if isinstance(spec, str):
        return spec.lower(), params
    if isinstance(spec, dict):
        kind = str(spec.get("type") or spec.get("effect") or spec.get("kind") or "").lower()
        for key in ("mu", "sigma", "amp", "center", "k", "scale"):
            if key in spec:
                params[key] = float(spec[key])
        if not kind:
            kind = str(spec.get("mode", "peak")).lower()
        return kind, params
    return "peak", params


def fit_oracle_prior(
    domain: Any,
    *,
    structure_template: Dict[str, Any],
    steps: int = 1000,
    batch_size: int = 512,
    lr: float = 1e-2,
    seed: int = 0,
    device: Optional[torch.device] = None,
    verbose_every: int = 100,
) -> TrainablePrior:
    torch.manual_seed(int(seed))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_names = list(getattr(domain, "feature_names"))
    mins = getattr(domain, "mins").to(device=device, dtype=DTYPE)
    maxs = getattr(domain, "maxs").to(device=device, dtype=DTYPE)
    oracle = getattr(domain, "oracle")
    d = int(len(feature_names))

    model = TrainablePrior(feature_names=feature_names, structure_template=structure_template).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(int(steps)):
        print(step)
        X_unit = torch.rand(batch_size, d, device=device, dtype=DTYPE)
        X_raw = _unit_to_raw(mins, maxs, X_unit)
        y_target = oracle(X_raw).reshape(-1)
        y_pred = model(X_unit).reshape(-1)
        loss = F.mse_loss(y_pred, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose_every and (step + 1) % int(verbose_every) == 0:
            print(f"[fit_oracle_prior] step {step + 1}/{steps} loss={loss.item():.6f}")

    return model


def fit_oracle_to_json(
    domain: Any,
    *,
    structure_template: Dict[str, Any],
    steps: int = 1000,
    batch_size: int = 512,
    lr: float = 1e-2,
    seed: int = 0,
    device: Optional[torch.device] = None,
    verbose_every: int = 100,
    return_model: bool = False,
) -> Any:
    model = fit_oracle_prior(
        domain,
        structure_template=structure_template,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        verbose_every=verbose_every,
    )
    readout = model.export_parameters(getattr(domain, "mins"), getattr(domain, "maxs"))
    return (readout, model) if return_model else readout

#%%
if __name__ == "__main__":
    # Example usage (UGI):
    from main_benchmark_portion import build_continuous_domain

    domain = build_continuous_domain()
    structure = {
        "x4": "peak",
        "x1": "decreasing",
    }
    optimized = fit_oracle_to_json(domain, structure_template=structure, steps=10, batch_size=512)
    print(optimized)

# %%
