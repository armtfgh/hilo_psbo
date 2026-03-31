"""
Data distillation pipeline for deriving a prior from historical CSV data.

Notebook-friendly: no CLI entrypoints, just importable helpers.
"""
#%%
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from collections import Counter
import json
import re

import numpy as np
import pandas as pd

from llm_study import get_llm_json, list_models

MIN_SAMPLE_ROWS = 5
_LLM_MODELS = None


def _validate_llm_model(model: str) -> None:
    global _LLM_MODELS
    if _LLM_MODELS is None:
        _LLM_MODELS = list_models()
    if model not in _LLM_MODELS:
        available = ", ".join(sorted(_LLM_MODELS))
        raise ValueError(f"Unknown model '{model}'. Available: {available}")


def list_available_llms() -> List[str]:
    """Return available LLM model names from llm_study registry."""
    global _LLM_MODELS
    if _LLM_MODELS is None:
        _LLM_MODELS = list_models()
    return sorted(_LLM_MODELS)


def sample_legacy_data(
    csv_path: str,
    *,
    fraction: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """Load the CSV and sample a fraction of rows as historical data."""
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in (0, 1].")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty; no rows to sample.")
    n_total = int(df.shape[0])
    sample_n = int(np.ceil(float(fraction) * n_total))
    sample_n = max(sample_n, MIN_SAMPLE_ROWS)
    sample_n = min(sample_n, n_total)
    return df.sample(n=sample_n, random_state=int(seed))


def _resolve_feature_names(
    df: pd.DataFrame,
    feature_names: Optional[Iterable[str]],
    target_col: str,
) -> List[str]:
    if feature_names is not None:
        return [str(c) for c in feature_names]
    cols = df.select_dtypes(include="number").columns.tolist()
    if target_col in cols:
        cols.remove(target_col)
    if not cols:
        raise ValueError("No numeric feature columns found to summarize.")
    return cols


def _filter_feature_names(
    feature_names: Iterable[str],
    focus_features: Optional[Iterable[str]],
) -> List[str]:
    if focus_features is None:
        return [str(name) for name in feature_names]
    names = [str(name) for name in feature_names]
    norm_map = {re.sub(r"[^a-z0-9]+", "", n.lower()): n for n in names}
    resolved: List[str] = []
    seen = set()

    for token in focus_features:
        if token is None:
            continue
        if isinstance(token, int):
            idx = int(token)
            if 0 <= idx < len(names):
                name = names[idx]
                if name not in seen:
                    resolved.append(name)
                    seen.add(name)
                continue

        raw = str(token).strip()
        if not raw:
            continue
        lower = raw.lower()
        if lower.startswith("x") and lower[1:].isdigit():
            idx = int(lower[1:]) - 1
            if 0 <= idx < len(names):
                name = names[idx]
                if name not in seen:
                    resolved.append(name)
                    seen.add(name)
                continue

        exact = next((n for n in names if n.lower() == lower), None)
        if exact:
            if exact not in seen:
                resolved.append(exact)
                seen.add(exact)
            continue

        norm = re.sub(r"[^a-z0-9]+", "", lower)
        if norm in norm_map:
            name = norm_map[norm]
            if name not in seen:
                resolved.append(name)
                seen.add(name)
            continue

        partial = next((n for n in names if norm in re.sub(r"[^a-z0-9]+", "", n.lower())), None)
        if partial:
            if partial not in seen:
                resolved.append(partial)
                seen.add(partial)

    if not resolved:
        raise ValueError("focus_features did not match any available feature names.")
    return resolved


def format_data_for_llm(
    df_sampled: pd.DataFrame,
    *,
    feature_names: Optional[Iterable[str]],
    target_col: str = "yield",
    top_k: int = 20,
    bottom_k: int = 20,
    include_summary_stats: bool = False,
) -> str:
    """Format only top/bottom cases from a sampled portion into a prompt string."""
    top_df, bottom_df, feat_names, df_clean = select_top_bottom_cases(
        df_sampled,
        feature_names=feature_names,
        target_col=target_col,
        top_k=top_k,
        bottom_k=bottom_k,
    )
    required_cols = [target_col, *feat_names]
    n_rows = int(df_clean.shape[0])

    def _fmt_row(row: pd.Series) -> str:
        row_id = row.name
        parts = [f"{target_col}={float(row[target_col]):.4g}"]
        for name in feat_names:
            parts.append(f"{name}={float(row[name]):.4g}")
        return f"row {row_id}: " + ", ".join(parts)

    lines: List[str] = []
    lines.append(f"Sampled portion size: {n_rows} historical experiments.")
    lines.append(f"Features: {', '.join(feat_names)}")
    lines.append(f"Target: {target_col}")
    if include_summary_stats:
        stats = df_clean[required_cols].agg(["mean", "min", "max"]).T
        lines.append("")
        lines.append("Summary statistics (mean/min/max):")
        for col in required_cols:
            mean = float(stats.loc[col, "mean"])
            vmin = float(stats.loc[col, "min"])
            vmax = float(stats.loc[col, "max"])
            lines.append(f"- {col}: mean={mean:.4g}, min={vmin:.4g}, max={vmax:.4g}")

    lines.append("")
    lines.append(f"Best yields were found at (top {len(top_df)}):")
    for _, row in top_df.iterrows():
        lines.append(f"- {_fmt_row(row)}")

    lines.append("")
    lines.append(f"Worst yields (Failures) were found at (bottom {len(bottom_df)}):")
    if bottom_df.empty:
        lines.append("- None (sample too small for distinct failures).")
    else:
        for _, row in bottom_df.iterrows():
            lines.append(f"- {_fmt_row(row)}")

    return "\n".join(lines)


def select_top_bottom_cases(
    df_sampled: pd.DataFrame,
    *,
    feature_names: Optional[Iterable[str]],
    target_col: str = "yield",
    top_k: int = 20,
    bottom_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """Return top/bottom rows from a sampled portion for LLM prompting."""
    if target_col not in df_sampled.columns:
        raise ValueError(f"target_col={target_col!r} not found in sampled data.")

    feat_names = _resolve_feature_names(df_sampled, feature_names, target_col)
    required_cols = [target_col, *feat_names]
    missing = [c for c in required_cols if c not in df_sampled.columns]
    if missing:
        raise ValueError(f"Missing columns in sampled data: {missing}")

    df_clean = df_sampled.dropna(subset=required_cols)
    if df_clean.empty:
        raise ValueError("Sampled data has no valid rows after dropping missing values.")

    n_rows = int(df_clean.shape[0])
    top_n = min(int(top_k), n_rows)
    bottom_n = min(int(bottom_k), n_rows)

    df_sorted = df_clean.sort_values(target_col, ascending=False)
    top_df = df_sorted.head(top_n)
    bottom_df = df_sorted.sort_values(target_col, ascending=True).head(bottom_n)
    bottom_df = bottom_df.loc[~bottom_df.index.isin(top_df.index)]

    return top_df, bottom_df, feat_names, df_clean


def _coerce_constraints_schema(
    readout: Dict[str, Any],
    *,
    default_penalty: float = 5.0,
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(readout or {})
    constraints_in = out.get("constraints") or []
    constraints_out = []
    for c in constraints_in:
        if not isinstance(c, dict):
            continue
        var = c.get("var")
        r = c.get("range")
        if var is None or not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        try:
            lo = float(r[0])
            hi = float(r[1])
        except (TypeError, ValueError):
            continue
        if hi < lo:
            lo, hi = hi, lo
        penalty_raw = c.get("penalty", c.get("weight", default_penalty))
        try:
            penalty = float(penalty_raw)
        except (TypeError, ValueError):
            penalty = float(default_penalty)
        c_out = dict(c)
        c_out["var"] = str(var)
        c_out["range"] = [lo, hi]
        c_out["penalty"] = penalty
        if not c_out.get("reason"):
            c_out["reason"] = "low-yield region"
        constraints_out.append(c_out)
    out["constraints"] = constraints_out
    return out


def _extract_features_from_summary(formatted_text: str) -> List[str]:
    for line in formatted_text.splitlines():
        if line.lower().startswith("features:"):
            raw = line.split(":", 1)[1].strip()
            if not raw:
                return []
            return [token.strip() for token in raw.split(",") if token.strip()]
    return []


def _ensure_effects_for_features(
    readout: Dict[str, Any],
    *,
    feature_names: Iterable[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(readout or {})
    effects = out.get("effects") or {}
    if not isinstance(effects, dict):
        effects = {}
    for name in feature_names:
        if name not in effects:
            effects[name] = {"effect": "flat", "scale": 0.0, "confidence": 0.0}
    out["effects"] = effects
    return out


def _range_span(range_pair: Iterable[float]) -> float:
    vals = list(range_pair)
    if len(vals) != 2:
        return 0.0
    return float(vals[1] - vals[0])


def _compute_ranges(
    df: pd.DataFrame,
    *,
    feature_names: List[str],
    target_col: str,
    top_k: int,
) -> tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    required = [target_col, *feature_names]
    df_clean = df.dropna(subset=required)
    if df_clean.empty:
        return {}, {}

    full_ranges: Dict[str, List[float]] = {}
    for name in feature_names:
        series = pd.to_numeric(df_clean[name], errors="coerce").dropna()
        if series.empty:
            continue
        full_ranges[name] = [float(series.min()), float(series.max())]

    top_k = max(1, min(int(top_k), int(df_clean.shape[0])))
    top_df = df_clean.sort_values(target_col, ascending=False).head(top_k)

    top_ranges: Dict[str, List[float]] = {}
    for name in feature_names:
        series = pd.to_numeric(top_df[name], errors="coerce").dropna()
        if series.empty:
            continue
        top_ranges[name] = [float(series.min()), float(series.max())]
    return top_ranges, full_ranges


def _tighten_range_hints(
    readout: Dict[str, Any],
    *,
    df_sampled: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
    top_k: int = 10,
    max_span_frac: float = 0.9,
    fill_missing: bool = True,
) -> Dict[str, Any]:
    top_ranges, full_ranges = _compute_ranges(
        df_sampled, feature_names=feature_names, target_col=target_col, top_k=top_k
    )
    if not top_ranges or not full_ranges:
        return readout

    out: Dict[str, Any] = dict(readout or {})
    effects = out.get("effects") or {}
    if not isinstance(effects, dict):
        return out

    for name in feature_names:
        spec = effects.get(name)
        if not isinstance(spec, dict):
            continue

        top_range = top_ranges.get(name)
        full_range = full_ranges.get(name)
        if not top_range or not full_range:
            continue

        full_span = _range_span(full_range)
        if full_span <= 0:
            continue

        range_hint = spec.get("range_hint")
        use_top = False

        if range_hint is None and fill_missing:
            use_top = True
        elif isinstance(range_hint, (list, tuple)) and len(range_hint) == 2:
            try:
                lo = float(range_hint[0])
                hi = float(range_hint[1])
            except (TypeError, ValueError):
                use_top = True
            else:
                if hi < lo:
                    lo, hi = hi, lo
                span_frac = (hi - lo) / full_span
                if span_frac >= float(max_span_frac):
                    use_top = True
        else:
            use_top = True

        if use_top:
            lo, hi = float(top_range[0]), float(top_range[1])
            if hi <= lo:
                eps = max(0.01 * full_span, 1e-6)
                center = lo
                lo = max(full_range[0], center - eps)
                hi = min(full_range[1], center + eps)
            spec["range_hint"] = [lo, hi]

    out["effects"] = effects
    return out


def extract_prior_from_statistics(
    formatted_text: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the LLM to extract a readout JSON from the formatted summary."""
    _validate_llm_model(model)
    system_prompt = f"""
You are an Expert Chemometrician and Bayesian Optimization Architect.
Your goal is to extract a "Search Prior" from sparse pilot data to accelerate optimization.
You must balance "Exploitation" (finding peaks) with "Safety" (avoiding dead zones).

DATA CONTEXT:
{formatted_text}

--- ANALYSIS PROTOCOL ---

1. IDENTIFY CONSTRAINTS (Negative Knowledge):
   - Look at the "Failure Cases" (Low Yields).
   - Is there a specific variable range that *consistently* appears in failures but NEVER in successes?
   - If yes, define a "constraint".
   - Set 'penalty' high (e.g., 8.0-10.0) for regions that look chemically invalid or dead.
   - Set 'penalty' moderate (e.g., 2.0-5.0) for regions that are just suboptimal.

2. IDENTIFY BUMPS (Positive Knowledge):
   - Look at the single absolute Best Result in the summary.
   - Create a "bump" centered exactly at those coordinates (`mu`).
   - Use `sigma` to define how wide this peak might be (use ~10% of the variable range if unsure).
   - Set `amp` based on the yield (e.g., if Yield=0.9, amp=0.2; if Yield=0.2, amp=0.05).

3. IDENTIFY TRENDS (Global Effects):
   - Compare "Success Cases" vs. "Failure Cases" generally.
   - If a variable is consistently high in successes and low in failures -> "increasing".
   - If a variable is consistently low in successes -> "decreasing".
   - If successes happen in the middle range -> "nonmonotone-peak".
   - If no clear pattern appears, omit it or set effect to "flat".
   - Set 'confidence' based on consistency: 0.9 if the trend has no exceptions, 0.4 if noisy.

--- JSON OUTPUT SCHEMA ---

Return STRICT JSON. No prose.
{{
  "effects": {{
    "<feature_name>": {{
      "effect": "increasing|decreasing|nonmonotone-peak|flat",
      "scale": <float 0.1 to 2.0, strength of trend slope>,
      "confidence": <float 0.0 to 1.0, reliability of data>,
      "range_hint": [<low_float>, <high_float>] (Optional: focus region)
    }}
  }},
  "bumps": [
    {{
      "mu": [<val_x1>, <val_x2>...],
      "sigma": [<width_x1>, <width_x2>...] (in raw units),
      "amp": <float 0.0 to 0.5>
    }}
  ],
  "constraints": [
    {{
      "var": "<feature_name>",
      "range": [<low_float>, <high_float>],
      "penalty": <float, typically 5.0 to 10.0>,
      "reason": "<short string explaining why>"
    }}
  ]
}}

--- CRITICAL RULES ---
1. RAW UNITS ONLY: Do not normalize. If input is '150 degC', use 150.0.
2. PRECISE BUMPS: The 'bumps' list MUST contain at least one entry centered on the best observed row.
3. CONSERVATIVE CONSTRAINTS: Only constrain a region if the evidence of failure is strong.
4. Feature names must match exactly as provided in the summary
""".strip()
    try:
        readout = get_llm_json(
            model,
            "Return STRICT JSON only (no prose).",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            response_format={"type": "json_object"},
            strict_json=True,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    feature_names = _extract_features_from_summary(formatted_text)
    readout = _ensure_effects_for_features(readout, feature_names=feature_names)
    return _coerce_constraints_schema(readout)


def extract_prior_from_expert_knowledge(
    expert_knowledge: str,
    *,
    feature_names: Optional[Iterable[str]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert expert qualitative/quantitative knowledge into a prior JSON readout.
    """
    if not str(expert_knowledge).strip():
        raise ValueError("expert_knowledge must be a non-empty string.")

    _validate_llm_model(model)
    feat_names: List[str] = [str(name) for name in feature_names] if feature_names is not None else []
    feature_guidance = (
        "Allowed feature names: " + ", ".join(feat_names)
        if feat_names
        else "Infer feature names from the expert text only."
    )

    system_prompt = f"""
You are an Expert Chemometrician and Bayesian Optimization Architect.
Convert the expert knowledge below into a machine-usable search prior JSON.

EXPERT KNOWLEDGE:
{expert_knowledge}

FEATURE GUIDANCE:
{feature_guidance}

Return STRICT JSON. No prose.
{{
  "effects": {{
    "<feature_name>": {{
      "effect": "increasing|decreasing|nonmonotone-peak|flat",
      "scale": <float 0.0 to 2.0>,
      "confidence": <float 0.0 to 1.0>,
      "range_hint": [<low_float>, <high_float>] (optional)
    }}
  }},
  "bumps": [
    {{
      "mu": [<val_x1>, <val_x2>...],
      "sigma": [<width_x1>, <width_x2>...],
      "amp": <float 0.0 to 0.5>
    }}
  ],
  "constraints": [
    {{
      "var": "<feature_name>",
      "range": [<low_float>, <high_float>],
      "penalty": <float, typically 5.0 to 10.0>,
      "reason": "<short string>"
    }}
  ]
}}

Rules:
1. Use raw units as stated in the expert text.
2. Do not invent feature names outside the provided list when feature guidance is given.
3. If a listed feature has no clear effect, use effect="flat", scale=0.0, confidence low.
4. Keep constraints conservative unless expert evidence is explicit.
""".strip()

    try:
        readout = get_llm_json(
            model,
            "Return STRICT JSON only (no prose).",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            response_format={"type": "json_object"},
            strict_json=True,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    if feat_names:
        readout = _ensure_effects_for_features(readout, feature_names=feat_names)
    return _coerce_constraints_schema(readout)


def get_expert_derived_prior(
    *,
    expert_knowledge: str,
    feature_names: Optional[Iterable[str]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """One-shot expert-knowledge prior extraction."""
    return extract_prior_from_expert_knowledge(
        expert_knowledge,
        feature_names=feature_names,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_readout_for_comparison(readout: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize readout shape for semantic comparison:
    - sort effect keys
    - sort bumps/constraints lists by canonical JSON
    - force [low, high] ordering for ranges/range_hint when numeric
    """
    out: Dict[str, Any] = dict(readout or {})

    effects = out.get("effects")
    if isinstance(effects, dict):
        effects_norm: Dict[str, Any] = {}
        for key in sorted(effects.keys(), key=str):
            spec = effects[key]
            if isinstance(spec, dict):
                spec = dict(spec)
                rh = spec.get("range_hint")
                if isinstance(rh, (list, tuple)) and len(rh) == 2:
                    try:
                        lo = float(rh[0])
                        hi = float(rh[1])
                        if hi < lo:
                            lo, hi = hi, lo
                        spec["range_hint"] = [lo, hi]
                    except (TypeError, ValueError):
                        pass
            effects_norm[str(key)] = spec
        out["effects"] = effects_norm

    constraints = out.get("constraints")
    if isinstance(constraints, list):
        constraints_norm: List[Dict[str, Any]] = []
        for c in constraints:
            if not isinstance(c, dict):
                continue
            c2 = dict(c)
            r = c2.get("range")
            if isinstance(r, (list, tuple)) and len(r) == 2:
                try:
                    lo = float(r[0])
                    hi = float(r[1])
                    if hi < lo:
                        lo, hi = hi, lo
                    c2["range"] = [lo, hi]
                except (TypeError, ValueError):
                    pass
            constraints_norm.append(c2)
        constraints_norm.sort(key=_canonical_json)
        out["constraints"] = constraints_norm

    bumps = out.get("bumps")
    if isinstance(bumps, list):
        bumps_norm: List[Dict[str, Any]] = [dict(b) for b in bumps if isinstance(b, dict)]
        bumps_norm.sort(key=_canonical_json)
        out["bumps"] = bumps_norm

    return out


def _compute_reproducibility_report(
    readouts: List[Dict[str, Any]],
    *,
    repeats: int,
    errors: List[Dict[str, Any]],
    include_outputs: bool = True,
) -> Dict[str, Any]:
    n_success = int(len(readouts))
    n_failed = int(len(errors))

    if n_success == 0:
        return {
            "repeats_requested": int(repeats),
            "n_success": n_success,
            "n_failed": n_failed,
            "exact_reproducibility": 0.0,
            "normalized_reproducibility": 0.0,
            "n_unique_exact": 0,
            "n_unique_normalized": 0,
            "errors": errors,
            "run_outputs": [] if include_outputs else None,
            "exact_variants": [],
            "normalized_variants": [],
        }

    exact_serialized = [_canonical_json(ro) for ro in readouts]
    normalized_serialized = [_canonical_json(_normalize_readout_for_comparison(ro)) for ro in readouts]

    exact_counts = Counter(exact_serialized)
    normalized_counts = Counter(normalized_serialized)

    exact_best = int(exact_counts.most_common(1)[0][1])
    normalized_best = int(normalized_counts.most_common(1)[0][1])

    exact_variants = [
        {
            "count": int(count),
            "fraction": float(count / n_success),
            "readout": json.loads(payload) if include_outputs else None,
        }
        for payload, count in exact_counts.most_common()
    ]
    normalized_variants = [
        {
            "count": int(count),
            "fraction": float(count / n_success),
            "readout": json.loads(payload) if include_outputs else None,
        }
        for payload, count in normalized_counts.most_common()
    ]

    run_outputs = (
        [{"run": i + 1, "readout": ro} for i, ro in enumerate(readouts)]
        if include_outputs
        else None
    )

    return {
        "repeats_requested": int(repeats),
        "n_success": n_success,
        "n_failed": n_failed,
        "exact_reproducibility": float(exact_best / n_success),
        "normalized_reproducibility": float(normalized_best / n_success),
        "n_unique_exact": int(len(exact_counts)),
        "n_unique_normalized": int(len(normalized_counts)),
        "errors": errors,
        "run_outputs": run_outputs,
        "exact_variants": exact_variants,
        "normalized_variants": normalized_variants,
    }


def _run_repeated_generation(
    generator: Callable[[], Dict[str, Any]],
    *,
    repeats: int,
    include_outputs: bool = True,
) -> Dict[str, Any]:
    if int(repeats) <= 0:
        raise ValueError("repeats must be >= 1.")

    readouts: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for run_idx in range(int(repeats)):
        try:
            readout = generator()
            if not isinstance(readout, dict):
                raise ValueError("Generator must return a JSON object (dict).")
            readouts.append(readout)
        except Exception as exc:
            errors.append({"run": int(run_idx + 1), "error": str(exc)})

    return _compute_reproducibility_report(
        readouts,
        repeats=int(repeats),
        errors=errors,
        include_outputs=include_outputs,
    )


def test_data_prior_reproducibility(
    *,
    csv_path: str,
    repeats: int = 10,
    fraction: float = 0.10,
    seed: int = 2,
    feature_names: Optional[Iterable[str]] = None,
    focus_features: Optional[Iterable[str]] = None,
    target_col: str = "yield",
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
    top_k: int = 20,
    bottom_k: int = 20,
    include_summary_stats: bool = False,
    tighten_range_hint: bool = True,
    range_hint_top_k: Optional[int] = None,
    range_hint_max_span_frac: float = 0.9,
    fill_missing_range_hint: bool = True,
    include_outputs: bool = True,
    include_formatted_input: bool = True,
) -> Dict[str, Any]:
    """
    Re-run the LLM on the exact same sampled data portion multiple times and
    report JSON reproducibility statistics.
    """
    sampled = sample_legacy_data(csv_path, fraction=fraction, seed=seed)
    feat_names = _resolve_feature_names(sampled, feature_names, target_col)
    feat_names = _filter_feature_names(feat_names, focus_features)
    formatted = format_data_for_llm(
        sampled,
        feature_names=feat_names,
        target_col=target_col,
        top_k=top_k,
        bottom_k=bottom_k,
        include_summary_stats=include_summary_stats,
    )

    def _generate_once() -> Dict[str, Any]:
        readout = extract_prior_from_statistics(
            formatted,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        if tighten_range_hint:
            top_k_for_ranges = range_hint_top_k if range_hint_top_k is not None else top_k
            readout = _tighten_range_hints(
                readout,
                df_sampled=sampled,
                feature_names=feat_names,
                target_col=target_col,
                top_k=top_k_for_ranges,
                max_span_frac=range_hint_max_span_frac,
                fill_missing=fill_missing_range_hint,
            )
        return readout

    report = _run_repeated_generation(
        _generate_once,
        repeats=repeats,
        include_outputs=include_outputs,
    )
    report["mode"] = "data_portion"
    report["input"] = {
        "csv_path": csv_path,
        "fraction": float(fraction),
        "seed": int(seed),
        "target_col": target_col,
        "feature_names": feat_names,
        "top_k": int(top_k),
        "bottom_k": int(bottom_k),
    }
    if include_formatted_input:
        report["formatted_input"] = formatted
    return report


def test_expert_prior_reproducibility(
    *,
    expert_knowledge: str,
    repeats: int = 10,
    feature_names: Optional[Iterable[str]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
    include_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Re-run the LLM on the exact same expert-knowledge text multiple times and
    report JSON reproducibility statistics.
    """
    feat_names = [str(name) for name in feature_names] if feature_names is not None else None

    def _generate_once() -> Dict[str, Any]:
        return extract_prior_from_expert_knowledge(
            expert_knowledge,
            feature_names=feat_names,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    report = _run_repeated_generation(
        _generate_once,
        repeats=repeats,
        include_outputs=include_outputs,
    )
    report["mode"] = "expert_knowledge"
    report["expert_knowledge"] = expert_knowledge
    report["feature_names"] = feat_names
    return report


def get_data_derived_prior(
    *,
    csv_path: str,
    fraction: float = 0.10,
    seed: int = 2,
    feature_names: Optional[Iterable[str]] = None,
    focus_features: Optional[Iterable[str]] = None,
    target_col: str = "yield",
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
    top_k: int = 20,
    bottom_k: int = 20,
    include_summary_stats: bool = False,
    tighten_range_hint: bool = True,
    range_hint_top_k: Optional[int] = None,
    range_hint_max_span_frac: float = 0.9,
    fill_missing_range_hint: bool = True,
) -> Dict[str, Any]:
    """Run sampling + formatting + LLM extraction, returning a readout dict."""
    sampled = sample_legacy_data(csv_path, fraction=fraction, seed=seed)
    feat_names = _resolve_feature_names(sampled, feature_names, target_col)
    feat_names = _filter_feature_names(feat_names, focus_features)
    formatted = format_data_for_llm(
        sampled,
        feature_names=feat_names,
        target_col=target_col,
        top_k=top_k,
        bottom_k=bottom_k,
        include_summary_stats=include_summary_stats,
    )
    readout = extract_prior_from_statistics(
        formatted,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    if tighten_range_hint:
        top_k_for_ranges = range_hint_top_k if range_hint_top_k is not None else top_k
        readout = _tighten_range_hints(
            readout,
            df_sampled=sampled,
            feature_names=feat_names,
            target_col=target_col,
            top_k=top_k_for_ranges,
            max_span_frac=range_hint_max_span_frac,
            fill_missing=fill_missing_range_hint,
        )
    return readout
#%%
rep_data = test_data_prior_reproducibility(
    csv_path="ugi_merged_dataset.csv",
    fraction=0.10,
    seed=2,
    repeats=10,
    model="gpt-4o-mini",
)

#%%
rep_expert = test_expert_prior_reproducibility(
    expert_knowledge="High x1 and moderate x3 usually improve yield; avoid very low x2.",
    feature_names=["x1", "x2", "x3", "x4"],
    repeats=10,
    model="gpt-4o-mini",
)



#%%
if __name__ == "__main__":
    _ = get_data_derived_prior(fraction=0.9, csv_path="ugi_merged_dataset.csv")
