from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import Tensor

from prior_gp_p3ht import Prior


def _feature_keys(
    feature_names: Optional[Iterable[str]] = None,
    *,
    n_fallback: int = 4,
) -> List[str]:
    if feature_names is not None:
        names = [str(name) for name in feature_names]
        if names:
            return names
    return [f"x{i+1}" for i in range(n_fallback)]


def flat_readout(
    feature_names: Optional[Iterable[str]] = None,
    *,
    n_fallback: int = 4,
) -> Dict[str, Any]:
    names = _feature_keys(feature_names, n_fallback=n_fallback)
    effects = {name: {"effect": "flat", "scale": 0.0, "confidence": 0.0} for name in names}
    return {
        "effects": effects,
        "interactions": [],
        "bumps": [],
        "constraints": [],
    }


def _sanitize_readout_minimal(ro: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(ro or {})
    out["effects"] = out.get("effects") or {}
    out["interactions"] = out.get("interactions") or []
    out["bumps"] = out.get("bumps") or []
    out["constraints"] = out.get("constraints") or []

    if not isinstance(out["effects"], dict):
        out["effects"] = {}
    if not isinstance(out["interactions"], list):
        out["interactions"] = []
    if not isinstance(out["bumps"], list):
        out["bumps"] = []
    if not isinstance(out["constraints"], list):
        out["constraints"] = []
    return out


def _normalize_readout_to_unit_box(
    readout: Dict[str, Any],
    mins: Tensor,
    maxs: Tensor,
    *,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert raw-scale readout to unit-box coordinates [0,1]^d."""
    ro = _sanitize_readout_minimal(readout)
    rng = (maxs - mins).clamp_min(1e-12)

    idx_lookup = {name: i for i, name in enumerate(feature_names or [])}
    idx_lookup_lower = {name.lower(): i for i, name in enumerate(feature_names or [])}

    def _dim_index(key: str) -> Optional[int]:
        if key.startswith("x"):
            try:
                j = int(key[1:]) - 1
            except ValueError:
                return None
            return j if 0 <= j < len(mins) else None
        if key in idx_lookup:
            return idx_lookup[key]
        return idx_lookup_lower.get(key.lower())

    # effects: normalize range_hint if present
    effects_out: Dict[str, Any] = {}
    for name, spec in (ro.get("effects", {}) or {}).items():
        spec_out = dict(spec or {})
        rh = spec_out.get("range_hint")
        dim_idx = _dim_index(str(name)) if isinstance(name, str) else None
        if dim_idx is not None and isinstance(rh, (list, tuple)) and len(rh) == 2:
            low = float(rh[0])
            high = float(rh[1])
            low_n = float(((low - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
            high_n = float(((high - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
            spec_out["range_hint"] = [low_n, high_n]
        effects_out[str(name)] = spec_out
    ro["effects"] = effects_out

    # bumps: normalize mu/sigma from raw units
    bumps: List[Dict[str, Any]] = []
    for b in ro.get("bumps", []) or []:
        mu = (b or {}).get("mu", None)
        sigma = (b or {}).get("sigma", 0.15)
        amp = (b or {}).get("amp", 0.1)
        if mu is None:
            continue

        mu_vec = torch.tensor(list(mu), dtype=mins.dtype, device=mins.device)
        mu_norm = ((mu_vec - mins) / rng).clamp(1e-6, 1 - 1e-6)

        if isinstance(sigma, (list, tuple)):
            sigma_vec = torch.tensor(list(sigma), dtype=mins.dtype, device=mins.device)
            sigma_norm = (sigma_vec / rng).clamp_min(1e-6)
            sigma_out: Any = [float(v) for v in sigma_norm.detach().cpu().tolist()]
        else:
            sigma_scalar = float(sigma)
            scale = torch.mean((torch.ones_like(rng) * sigma_scalar) / rng).clamp_min(1e-6)
            sigma_out = float(scale.item())

        bumps.append(
            {
                "mu": [float(v) for v in mu_norm.detach().cpu().tolist()],
                "sigma": sigma_out,
                "amp": float(amp),
            }
        )
    ro["bumps"] = bumps

    # constraints: normalize per-dim interval ranges
    constraints_out: List[Dict[str, Any]] = []
    for c in ro.get("constraints", []) or []:
        if not isinstance(c, dict):
            continue
        var = c.get("var", None)
        r = c.get("range", None)
        if var is None or not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        dim_idx = _dim_index(str(var))
        if dim_idx is None:
            continue

        low = float(r[0])
        high = float(r[1])
        low_n = float(((low - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
        high_n = float(((high - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
        if high_n < low_n:
            low_n, high_n = high_n, low_n

        c_out = dict(c)
        c_out["var"] = str(var)
        c_out["range"] = [float(low_n), float(high_n)]
        constraints_out.append(c_out)
    ro["constraints"] = constraints_out

    return ro


def normalize_readout_to_unit_box(
    readout: Dict[str, Any],
    mins: Tensor,
    maxs: Tensor,
    *,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return _normalize_readout_to_unit_box(readout, mins, maxs, feature_names=feature_names)


def readout_to_prior(
    ro: Dict[str, Any],
    *,
    feature_names: Optional[Iterable[str]] = None,
) -> Prior:
    effects = ro.get("effects", {})
    inter = ro.get("interactions", [])
    bumps = ro.get("bumps", [])
    constraints = ro.get("constraints", [])
    names = _feature_keys(feature_names) if feature_names is not None else None
    return Prior(
        effects=effects,
        interactions=inter,
        bumps=bumps,
        constraints=constraints,
        feature_names=names,
    )
