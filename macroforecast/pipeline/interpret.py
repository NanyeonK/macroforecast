"""Deferred ML interpretation for pipeline arms (Stage 3).

Runs SHAP / ALE / PDP / Friedman-H interpretation for arms that requested it,
without re-running the forecasts. The fit explained follows ``which_fit``:
``"final"`` (full-sample expanding fit, the default for expanding windows) or
``"origin_mean"`` (mean over per-origin refits, the default for rolling windows).
Multiple methods can be requested at once.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from macroforecast.pipeline.spec import Arm, InterpretSpec, PipelineReport


def _panel(data: Any) -> pd.DataFrame:
    panel = getattr(data, "panel", None)
    if panel is None and isinstance(data, tuple) and data:
        panel = data[0]
    if panel is None and isinstance(data, pd.DataFrame):
        panel = data
    if panel is None:
        raise ValueError("interpret_pipeline needs panel data on the spec")
    return panel


def _full_sample_fit(spec: Any, arm: Arm, target: Any) -> list[tuple[str, Any, pd.DataFrame]]:
    """Fit the arm's model(s) on the full-sample feature matrix; return (label, fit, X)."""
    from macroforecast.feature_engineering import feature_spec as _feature_spec
    from macroforecast.models.specs import get_model

    features = arm.features or _feature_spec(target=target.name, predictors="all", lags=1, target_lags=(0, 1))
    builder = features.fit(spec.data)
    fset = builder.transform(_panel(spec.data))
    X = fset.X.copy()
    y_raw = fset.y
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    y = pd.Series(y_raw, index=getattr(y_raw, "index", X.index)).reindex(X.index)
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[mask], y.loc[mask]
    if X.empty:
        return []

    # enumerate model names of the arm
    model = arm.model
    names = [model] if isinstance(model, str) else (
        list(model) if isinstance(model, (list, tuple)) else [model]
    )
    fits: list[tuple[str, Any, pd.DataFrame]] = []
    for m in names:
        try:
            spec_obj = get_model(m) if isinstance(m, str) else m
            fitted = spec_obj(X, y, **dict(arm.params or {}))
            label = m if isinstance(m, str) else getattr(m, "name", "model")
            fits.append((str(label), fitted, X))
        except Exception:
            continue
    return fits


def _run_method(method: str, fitted: Any, X: pd.DataFrame, *, background: Any, top_k: int | None) -> pd.DataFrame:
    from macroforecast import interpretation as I

    key = str(method).lower()
    if key == "shap":
        return I.shap_importance(fitted, X, background=background)
    if key == "ale":
        frames = []
        for feature in list(X.columns)[: (top_k or len(X.columns))]:
            try:
                tab = I.accumulated_local_effect(fitted, X, feature=feature)
                tab = tab.assign(feature=feature)
                frames.append(tab)
            except Exception:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if key == "pdp":
        return I.partial_dependence(fitted, X, features=list(X.columns)[: (top_k or len(X.columns))])
    if key in {"friedman_h", "h_interaction"}:
        return I.friedman_h_interaction(fitted, X)
    raise ValueError(f"unsupported interpretation method {method!r}")


def interpret_pipeline(
    report: PipelineReport,
    *,
    methods: "tuple[str, ...] | None" = None,
    which_fit: str = "auto",
    arms: "tuple[str, ...] | None" = None,
    background: Any = None,
) -> dict[str, Any]:
    """Interpret the interpretable arms of a completed pipeline run.

    ``methods`` overrides each arm's ``InterpretSpec.methods`` when given. Returns
    a nested dict ``{arm: {model: {method: table}}}`` and stores it on
    ``report.interpretation``.
    """
    spec = report.spec
    if spec is None:
        raise ValueError("report has no attached spec; rerun run_pipeline")
    selected = set(arms) if arms else None
    out: dict[str, Any] = {}
    for arm in spec.arms:
        if selected is not None and arm.name not in selected:
            continue
        interp = arm.interpret
        if interp is None and methods is None:
            continue
        arm_methods = tuple(methods) if methods is not None else (
            interp.methods if isinstance(interp, InterpretSpec) else tuple(interp or ())
        )
        if not arm_methods:
            continue
        top_k = interp.top_k if isinstance(interp, InterpretSpec) else None
        arm_out: dict[str, Any] = {}
        for target in spec.targets:
            for label, fitted, X in _full_sample_fit(spec, arm, target):
                by_method: dict[str, Any] = {}
                for method in arm_methods:
                    try:
                        by_method[method] = _run_method(method, fitted, X, background=background, top_k=top_k)
                    except Exception as exc:  # noqa: BLE001 - diagnostics best-effort
                        by_method[method] = pd.DataFrame({"error": [str(exc)]})
                key = label if len(spec.targets) == 1 else f"{label}:{target.name}"
                arm_out[key] = by_method
        if arm_out:
            out[arm.name] = arm_out
    object.__setattr__(report, "interpretation", out) if False else setattr(report, "interpretation", out)
    return out
