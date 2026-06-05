"""Deferred ML interpretation for pipeline arms (Stage 3).

Runs SHAP / ALE / PDP / Friedman-H interpretation for arms that requested it,
without re-running the forecasts. ``which_fit`` selects the fit explained:
``"final"`` (full-sample expanding fit), ``"origin_mean"`` (mean over per-origin
expanding refits), or ``"auto"`` (origin_mean for rolling windows, final
otherwise). Multiple methods can be requested at once.
"""
from __future__ import annotations

import dataclasses as _dc
from typing import Any

import pandas as pd

from macroforecast.pipeline.spec import Arm, InterpretSpec, PipelineReport

_WHICH_FIT = {"auto", "final", "origin_mean"}


def _panel(data: Any) -> pd.DataFrame:
    panel = getattr(data, "panel", None)
    if panel is None and isinstance(data, tuple) and data:
        panel = data[0]
    if panel is None and isinstance(data, pd.DataFrame):
        panel = data
    if panel is None:
        raise ValueError("interpret_pipeline needs panel data on the spec")
    return panel


def _align_features(features: Any, target_name: str) -> Any:
    """Re-target an arm's feature spec to ``target_name`` (mirrors run_arms)."""
    if features is None:
        return None
    same = getattr(features, "target", None) == target_name and not getattr(features, "targets", ())
    if same:
        return features
    try:
        kwargs: dict[str, Any] = {"target": target_name}
        if getattr(features, "targets", None):
            kwargs["targets"] = ()
        return _dc.replace(features, **kwargs)
    except Exception:
        return features


def _resolve_which_fit(spec: Any, which_fit: str) -> str:
    if which_fit not in _WHICH_FIT:
        raise ValueError(f"which_fit must be one of {sorted(_WHICH_FIT)}")
    if which_fit != "auto":
        return which_fit
    mode = getattr(getattr(spec.window, "estimation", None), "mode", None)
    return "origin_mean" if mode == "rolling" else "final"


def _arm_models(arm: Arm) -> list[Any]:
    model = arm.model
    if isinstance(model, str):
        return [model]
    if isinstance(model, (list, tuple)):
        return list(model)
    return [model]


def _fit_on_panel(panel: pd.DataFrame, spec: Any, arm: Arm, target: Any) -> list[tuple[str, Any, pd.DataFrame]]:
    """Build features on ``panel`` and fit the arm's model(s); return (label, fit, X)."""
    from macroforecast.feature_engineering import feature_spec as _feature_spec
    from macroforecast.models.specs import get_model

    features = _align_features(
        arm.features or _feature_spec(target=target.name, predictors="all", lags=1, target_lags=(0, 1)),
        target.name,
    )
    builder = features.fit(panel)
    fset = builder.transform(panel)
    X = fset.X.copy()
    y_raw = fset.y
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    y = pd.Series(y_raw, index=getattr(y_raw, "index", X.index)).reindex(X.index)
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[mask], y.loc[mask]
    if X.empty:
        return []
    fits: list[tuple[str, Any, pd.DataFrame]] = []
    for m in _arm_models(arm):
        try:
            spec_obj = get_model(m) if isinstance(m, str) else m
            fitted = spec_obj(X, y, **dict(arm.params or {}))
            label = m if isinstance(m, str) else getattr(m, "name", "model")
            fits.append((str(label), fitted, X))
        except Exception:
            continue
    return fits


def _origin_panels(spec: Any, panel: pd.DataFrame) -> list[pd.DataFrame]:
    """Expanding training panels, one per forecast origin (leak-aware slices)."""
    try:
        plan = spec.window.plan(panel.index)
    except Exception:
        return []
    slices: list[pd.DataFrame] = []
    for _, row in plan.iterrows():
        try:
            start = int(row.get("estimation_start_pos", 0))
            end = int(row["estimation_end_pos"])
        except Exception:
            continue
        if end >= start >= 0:
            slices.append(panel.iloc[max(0, start): end + 1])
    return slices


def _run_method(method: str, fitted: Any, X: pd.DataFrame, *, background: Any, top_k: int | None) -> pd.DataFrame:
    from macroforecast import interpretation as I

    key = str(method).lower()
    if key == "shap":
        return I.shap_importance(fitted, X, background=background)
    if key == "ale":
        frames = []
        for feature in list(X.columns)[: (top_k or len(X.columns))]:
            try:
                tab = I.accumulated_local_effect(fitted, X, feature=feature).assign(feature=feature)
                frames.append(tab)
            except Exception:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if key == "pdp":
        return I.partial_dependence(fitted, X, features=list(X.columns)[: (top_k or len(X.columns))])
    if key in {"friedman_h", "h_interaction"}:
        return I.friedman_h_interaction(fitted, X)
    raise ValueError(f"unsupported interpretation method {method!r}")


def _aggregate(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Average per-origin method tables (mean of numeric columns by non-numeric keys)."""
    tables = [t for t in tables if isinstance(t, pd.DataFrame) and not t.empty]
    if not tables:
        return pd.DataFrame()
    if len(tables) == 1:
        return tables[0]
    combined = pd.concat(tables, ignore_index=True)
    numeric = combined.select_dtypes("number").columns.tolist()
    keys = [c for c in combined.columns if c not in numeric]
    if keys and numeric:
        return combined.groupby(keys, dropna=False, as_index=False)[numeric].mean()
    return combined.groupby(combined.index % len(tables[0]))[numeric].mean() if numeric else tables[0]


def interpret_pipeline(
    report: PipelineReport,
    *,
    methods: "tuple[str, ...] | None" = None,
    which_fit: str = "auto",
    arms: "tuple[str, ...] | None" = None,
    background: Any = None,
) -> dict[str, Any]:
    """Interpret the interpretable arms of a completed pipeline run.

    ``methods`` overrides each arm's ``InterpretSpec.methods`` when given. Returns a
    nested dict ``{arm: {model[:target]: {method: table}}}`` stored on
    ``report.interpretation``. One arm failing to fit degrades to an error frame
    and never aborts the others.
    """
    spec = report.spec
    if spec is None:
        raise ValueError("report has no attached spec; rerun run_pipeline")
    resolved_fit = _resolve_which_fit(spec, which_fit)
    panel = _panel(spec.data)
    origin_panels = _origin_panels(spec, panel) if resolved_fit == "origin_mean" else []
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
            try:
                if resolved_fit == "origin_mean" and origin_panels:
                    # collect per-origin fits keyed by model label
                    per_model: dict[str, list[tuple[Any, pd.DataFrame]]] = {}
                    for sub in origin_panels:
                        for label, fitted, X in _fit_on_panel(sub, spec, arm, target):
                            per_model.setdefault(label, []).append((fitted, X))
                    items = [
                        (label, fits_list) for label, fits_list in per_model.items()
                    ]
                else:
                    items = [
                        (label, [(fitted, X)])
                        for label, fitted, X in _fit_on_panel(panel, spec, arm, target)
                    ]
            except Exception as exc:  # noqa: BLE001 - one arm/target must not abort the rest
                key = arm.name if len(spec.targets) == 1 else f"{arm.name}:{target.name}"
                arm_out[key] = {m: pd.DataFrame({"error": [str(exc)]}) for m in arm_methods}
                continue

            for label, fits_list in items:
                by_method: dict[str, Any] = {}
                for method in arm_methods:
                    try:
                        tables = [_run_method(method, fitted, X, background=background, top_k=top_k)
                                  for fitted, X in fits_list]
                        by_method[method] = _aggregate(tables)
                    except Exception as exc:  # noqa: BLE001
                        by_method[method] = pd.DataFrame({"error": [str(exc)]})
                key = label if len(spec.targets) == 1 else f"{label}:{target.name}"
                arm_out[key] = by_method
        if arm_out:
            out[arm.name] = arm_out
    report.interpretation = out
    return out
