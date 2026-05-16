from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import copy
import inspect
import math
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .layers import l6 as l6_layer
from .layers import l7 as l7_layer
from .layers import l8 as l8_layer
from .layers import l1_5 as l1_5_layer
from .layers import l2_5 as l2_5_layer
from .layers import l3_5 as l3_5_layer
from .layers import l4_5 as l4_5_layer
from .layers import l3 as l3_layer
from .layers import l4 as l4_layer
from .layers import l5 as l5_layer
from .layers import l1 as l1_layer
from .layers import l2 as l2_layer
from ..raw import load_fred_md, load_fred_qd, load_fred_sd
from ..raw.fred_sd_groups import FRED_SD_STATE_GROUPS, resolve_fred_sd_variable_group as _resolve_fred_sd_variable_group
from .types import (
    DiagnosticArtifact,
    L1DataDefinitionArtifact,
    L1RegimeMetadataArtifact,
    L2CleanPanelArtifact,
    L3FeaturesArtifact,
    L3MetadataArtifact,
    L4ForecastsArtifact,
    L4ModelArtifactsArtifact,
    L4TrainingMetadataArtifact,
    L5EvaluationArtifact,
    L6TestsArtifact,
    L7ImportanceArtifact,
    L7TransformationAttributionArtifact,
    L8ArtifactsArtifact,
    ExportedFile,
    L8Manifest,
    ModelArtifact,
    Panel,
    PanelMetadata,
    RuntimeEnvironment,
    Series,
    SeriesMetadata,
)
from .yaml import parse_recipe_yaml


@dataclass(frozen=True)
class RuntimeResult:
    """Materialized sink artifacts for a core-layer runtime pass."""

    artifacts: dict[str, Any] = field(default_factory=dict)
    resolved_axes: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime_durations: dict[str, float] = field(default_factory=dict)

    def sink(self, name: str) -> Any:
        return self.artifacts[name]


def execute_l1_l2(recipe_yaml_or_root: str | dict[str, Any]) -> RuntimeResult:
    """Materialize L1 and L2 sinks for custom-panel recipes.

    This is the first runtime bridge behind the schema contracts. It is
    intentionally narrow: official FRED loading, real-time vintages, EM
    imputation, and advanced frequency alignment stay in later runtime PRs.
    """

    root = (
        parse_recipe_yaml(recipe_yaml_or_root)
        if isinstance(recipe_yaml_or_root, str)
        else recipe_yaml_or_root
    )
    l1_artifact, regime_artifact, l1_axes = materialize_l1(root)
    l2_artifact, l2_axes = materialize_l2(root, l1_artifact)
    artifacts: dict[str, Any] = {
        "l1_data_definition_v1": l1_artifact,
        "l1_regime_metadata_v1": regime_artifact,
        "l2_clean_panel_v1": l2_artifact,
    }
    resolved_axes: dict[str, dict[str, Any]] = {"l1": l1_axes, "l2": dict(l2_axes)}
    if "1_5_data_summary" in root:
        l1_5_artifact, l1_5_axes = materialize_l1_5_diagnostic(root, l1_artifact)
        artifacts["l1_5_diagnostic_v1"] = l1_5_artifact
        resolved_axes["l1_5"] = l1_5_axes
    if "2_5_pre_post_preprocessing" in root:
        l2_5_artifact, l2_5_axes = materialize_l2_5_diagnostic(
            root, l1_artifact, l2_artifact
        )
        artifacts["l2_5_diagnostic_v1"] = l2_5_artifact
        resolved_axes["l2_5"] = l2_5_axes
    return RuntimeResult(
        artifacts=artifacts,
        resolved_axes=resolved_axes,
    )


def execute_minimal_forecast(
    recipe_yaml_or_root: str | dict[str, Any],
) -> RuntimeResult:
    """Run the minimal L1-L5 runtime path for custom-panel ridge forecasts."""

    import time as _time

    root = (
        parse_recipe_yaml(recipe_yaml_or_root)
        if isinstance(recipe_yaml_or_root, str)
        else recipe_yaml_or_root
    )
    durations: dict[str, float] = {}

    def _timed(label: str, fn):
        clock = _time.perf_counter()
        result = fn()
        durations[label] = _time.perf_counter() - clock
        return result

    l1_artifact, regime_artifact, l1_axes = _timed("l1", lambda: materialize_l1(root))
    l2_artifact, l2_axes = _timed("l2", lambda: materialize_l2(root, l1_artifact))
    l3_features, l3_metadata = _timed(
        "l3", lambda: materialize_l3_minimal(root, l1_artifact, l2_artifact, l2_resolved=l2_axes)
    )
    l4_forecasts, l4_models, l4_training = _timed(
        "l4", lambda: materialize_l4_minimal(root, l3_features)
    )
    l5_eval = _timed(
        "l5",
        lambda: materialize_l5_minimal(
            root, l1_artifact, l3_features, l4_forecasts, l4_models
        ),
    )
    artifacts: dict[str, Any] = {
        "l1_data_definition_v1": l1_artifact,
        "l1_regime_metadata_v1": regime_artifact,
        "l2_clean_panel_v1": l2_artifact,
        "l3_features_v1": l3_features,
        "l3_metadata_v1": l3_metadata,
        "l4_forecasts_v1": l4_forecasts,
        "l4_model_artifacts_v1": l4_models,
        "l4_training_metadata_v1": l4_training,
        "l5_evaluation_v1": l5_eval,
    }
    resolved_axes: dict[str, dict[str, Any]] = {
        "l1": l1_axes,
        "l2": dict(l2_axes),
        "l5": dict(l5_eval.l5_axis_resolved),
    }
    if "1_5_data_summary" in root:
        l1_5_artifact, l1_5_axes = _timed(
            "l1_5", lambda: materialize_l1_5_diagnostic(root, l1_artifact)
        )
        artifacts["l1_5_diagnostic_v1"] = l1_5_artifact
        resolved_axes["l1_5"] = l1_5_axes
    if "2_5_pre_post_preprocessing" in root:
        l2_5_artifact, l2_5_axes = _timed(
            "l2_5", lambda: materialize_l2_5_diagnostic(root, l1_artifact, l2_artifact)
        )
        artifacts["l2_5_diagnostic_v1"] = l2_5_artifact
        resolved_axes["l2_5"] = l2_5_axes
    if "3_5_feature_diagnostics" in root:
        l3_5_artifact, l3_5_axes = _timed(
            "l3_5",
            lambda: materialize_l3_5_diagnostic(
                root, l1_artifact, l2_artifact, l3_features, l3_metadata
            ),
        )
        artifacts["l3_5_diagnostic_v1"] = l3_5_artifact
        resolved_axes["l3_5"] = l3_5_axes
    if "4_5_generator_diagnostics" in root:
        l4_5_artifact, l4_5_axes = _timed(
            "l4_5",
            lambda: materialize_l4_5_diagnostic(
                root, l3_features, l4_forecasts, l4_models, l4_training
            ),
        )
        artifacts["l4_5_diagnostic_v1"] = l4_5_artifact
        resolved_axes["l4_5"] = l4_5_axes
    if "6_statistical_tests" in root:
        l6_tests, l6_axes = _timed(
            "l6",
            lambda: materialize_l6_runtime(
                root, l1_artifact, l3_features, l4_forecasts, l4_models, l5_eval
            ),
        )
        artifacts["l6_tests_v1"] = l6_tests
        resolved_axes["l6"] = l6_axes
    if "7_interpretation" in root:
        l7_importance, l7_transform, l7_axes = _timed(
            "l7",
            lambda: materialize_l7_runtime(
                root,
                l3_features,
                l3_metadata,
                l4_forecasts,
                l4_models,
                l5_eval,
                artifacts.get("l6_tests_v1"),
            ),
        )
        artifacts["l7_importance_v1"] = l7_importance
        artifacts["l7_transformation_attribution_v1"] = l7_transform
        resolved_axes["l7"] = l7_axes
    if "8_output" in root:
        l8_artifacts, l8_axes = _timed(
            "l8",
            lambda: materialize_l8_runtime(
                root,
                artifacts,
                runtime_durations=durations,
                cell_resolved_axes=resolved_axes,
            ),
        )
        artifacts["l8_artifacts_v1"] = l8_artifacts
        resolved_axes["l8"] = l8_axes
    return RuntimeResult(
        artifacts=artifacts,
        resolved_axes=resolved_axes,
        runtime_durations=durations,
    )


def materialize_l1(
    recipe_root: dict[str, Any],
) -> tuple[L1DataDefinitionArtifact, L1RegimeMetadataArtifact, dict[str, Any]]:
    raw = recipe_root.get("1_data", {}) or {}
    report = l1_layer.validate_layer(raw)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))

    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    resolved = l1_layer.resolve_axes_from_raw(fixed_axes, leaf_config)
    raw_panel = _load_raw_panel(resolved, leaf_config)
    _target_for_lag = leaf_config.get("target")
    _lagged_data = _apply_release_lag(raw_panel.data, resolved, leaf_config, _target_for_lag)
    if _lagged_data is not raw_panel.data:
        raw_panel = _panel_from_frame(
            _lagged_data,
            metadata=dict(raw_panel.metadata.values) if raw_panel.metadata.values else {},
        )

    target = leaf_config.get("target")
    targets = tuple(leaf_config.get("targets", ()) or ((target,) if target else ()))
    artifact = L1DataDefinitionArtifact(
        custom_source_policy=resolved["custom_source_policy"],
        dataset=resolved["dataset"],
        frequency=resolved["frequency"],
        vintage_policy=resolved["vintage_policy"],
        target_structure=resolved["target_structure"],
        target=target,
        targets=targets,
        variable_universe=resolved["variable_universe"],
        target_geography_scope=resolved["target_geography_scope"],
        predictor_geography_scope=resolved["predictor_geography_scope"],
        sample_start_rule=resolved["sample_start_rule"],
        sample_end_rule=resolved["sample_end_rule"],
        horizon_set=resolved["horizon_set"],
        target_horizons=l1_layer._resolved_horizons(resolved, leaf_config),
        regime_definition=resolved["regime_definition"],
        raw_panel=raw_panel,
        leaf_config=leaf_config,
    )
    regime = _materialize_regime(resolved, leaf_config, raw_panel.data.index)
    return artifact, regime, resolved



def _apply_release_lag(
    frame: pd.DataFrame,
    resolved: dict[str, Any],
    leaf_config: dict[str, Any],
    target: str | None,
) -> pd.DataFrame:
    """Shift predictor columns forward by their release lag.

    For a series with release lag k, the value at period t is only
    observable at period t+k, so the column is shifted *down* by k rows
    (NaN injected at the top). Target column is never shifted.
    """
    rule = resolved.get("release_lag_rule", "ignore_release_lag")
    if rule == "ignore_release_lag":
        return frame
    targets = set()
    if target:
        targets.add(target)
    for t in leaf_config.get("targets", []) or []:
        targets.add(t)
    result = frame.copy()
    if rule == "fixed_lag_all_series":
        lag = int(leaf_config.get("fixed_lag_periods", 0) or 0)
        if lag <= 0:
            return frame
        for col in result.columns:
            if col not in targets:
                result[col] = result[col].shift(lag)
    elif rule == "series_specific_lag":
        lag_map = leaf_config.get("release_lag_per_series") or {}
        for col, lag in lag_map.items():
            if col in result.columns and col not in targets:
                result[col] = result[col].shift(int(lag))
    return result

def materialize_l2(
    recipe_root: dict[str, Any], l1_artifact: L1DataDefinitionArtifact
) -> tuple[L2CleanPanelArtifact, l2_layer.L2ResolvedAxes]:
    raw = recipe_root.get("2_preprocessing", {}) or {}
    l1_context = _l1_context(l1_artifact)
    report = l2_layer.validate_layer(raw, l1_context=l1_context)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))

    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    resolved = l2_layer.resolve_axes_from_raw(
        fixed_axes, leaf_config, l1_context=l1_context
    )
    df = l1_artifact.raw_panel.data.copy()
    if df.empty:
        raise ValueError(
            "L1 raw_panel is empty; L2 materialization requires custom panel data"
        )

    cleaning_log: dict[str, Any] = {
        "runtime": "core_l1_l2_materialization",
        "steps": [],
    }

    # v0.8.6 Gap 1 -- pre-pipeline custom L2 preprocessor hook.
    # ``leaf_config.custom_preprocessor`` accepts a name registered via
    # ``macroforecast.custom.register_preprocessor``. Runs *before* the
    # transform / outlier / impute / frame_edge stages so users can
    # tweak the raw L1 panel (drop bad columns, deflation, normalisation,
    # etc.) before the canonical McCracken-Ng pipeline applies official
    # t-codes. Distinct from the post-pipeline ``custom_postprocessor``
    # hook (issue #251 / v0.2.5) which receives the already-cleaned
    # panel and produces the L3-ready clean panel.
    pre_name = leaf_config.get("custom_preprocessor")
    if pre_name:
        pre_result = _try_custom_l2_preprocessor(str(pre_name), df, leaf_config)
        if pre_result is not None:
            df = pre_result
            cleaning_log["steps"].append(
                {"custom_preprocessor": str(pre_name), "applied": True}
            )
    transform_map: dict[str, int] = {}
    n_outliers = 0
    n_imputed = 0

    l1_leaf_for_l2 = dict(l1_artifact.leaf_config)
    official_tcodes = (l1_artifact.raw_panel.metadata.values or {}).get(
        "transform_codes", {}
    )
    if official_tcodes:
        l1_leaf_for_l2["official_tcode_map"] = dict(official_tcodes)

    # Issue #202: FRED-SD frequency alignment. Applied *before* the
    # transform pipeline so downstream stages see a single-frequency
    # panel.
    if l1_artifact.dataset and "fred_sd" in str(l1_artifact.dataset):
        df = _apply_fred_sd_frequency_alignment(df, resolved, l1_artifact, cleaning_log)
    df, transform_map = _apply_transform(
        df, resolved, leaf_config, l1_leaf_for_l2, cleaning_log
    )
    # F-P1-2/F-P1-3 fix: skip full-sample outlier/imputation when temporal_rule
    # is per-origin; the per-origin helpers are called inside the L3
    # _per_origin_callable closure so stats are computed on data up to each
    # origin only (no lookahead leakage).
    _l2_imputation_temporal = resolved.get("imputation_temporal_rule", "expanding_window_per_origin")
    if _l2_imputation_temporal == "expanding_window_per_origin":
        # Deferred to per-origin closure in materialize_l3_minimal.
        n_outliers = 0
        n_imputed = 0
        cleaning_log["steps"].append({"outlier_policy": "deferred_per_origin"})
        cleaning_log["steps"].append({"imputation_policy": "deferred_per_origin"})
    else:
        df, n_outliers = _apply_outlier_policy(df, resolved, leaf_config, cleaning_log)
        df, n_imputed = _apply_imputation(df, resolved, cleaning_log)
    df, n_truncated = _apply_frame_edge(df, resolved, cleaning_log)

    # Issue #251 -- post-pipeline custom L2 preprocessor hook.
    # ``leaf_config.custom_postprocessor`` accepts a name registered via
    # ``macroforecast.custom.register_preprocessor``. Runs *after* the
    # builtin pipeline so its output is the L2 clean panel.
    custom_name = leaf_config.get("custom_postprocessor")
    if custom_name:
        post_result = _try_custom_l2_preprocessor(str(custom_name), df, leaf_config)
        if post_result is not None:
            df = post_result
            cleaning_log["steps"].append(
                {"custom_postprocessor": str(custom_name), "applied": True}
            )

    panel = _panel_from_frame(
        df, metadata={"stage": "l2_clean", "source": "l1_raw_panel"}
    )
    artifact = L2CleanPanelArtifact(
        panel=panel,
        shape=panel.shape,
        column_names=panel.column_names,
        index=panel.index,
        column_metadata={
            column: {"dtype": str(df[column].dtype)} for column in df.columns
        },
        cleaning_log=cleaning_log,
        n_imputed_cells=n_imputed,
        n_outliers_flagged=n_outliers,
        n_truncated_obs=n_truncated,
        transform_map_applied=transform_map,
        cleaning_temporal_rules={
            "imputation": resolved.get("imputation_temporal_rule", ""),
            "outlier": resolved.get("outlier_scope", ""),
            "frame_edge": resolved.get("frame_edge_scope", ""),
        },
    )
    return artifact, resolved


def materialize_l1_5_diagnostic(
    recipe_root: dict[str, Any], l1_artifact: L1DataDefinitionArtifact
) -> tuple[DiagnosticArtifact, dict[str, Any]]:
    raw = recipe_root.get("1_5_data_summary", {}) or {}
    context = {"regime_active": l1_artifact.regime_definition != "none"}
    report = l1_5_layer.validate_layer(raw, context=context)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    resolved = l1_5_layer.resolve_axes_from_raw(raw, context=context)
    axes = _plain_axes(resolved)
    if not resolved.get("enabled", False):
        return _disabled_diagnostic("l1", axes), axes
    frame = l1_artifact.raw_panel.data.copy()
    metadata = {
        "runtime": "core_l1_5_diagnostic",
        "axis_resolved": axes,
        "sample_coverage": _diagnostic_sample_coverage(frame),
        "univariate_summary": _diagnostic_univariate_summary(
            frame, axes.get("summary_metrics", [])
        ),
        "missing_outlier_audit": _diagnostic_missing_outlier_audit(
            frame, axes.get("leaf_config", {})
        ),
    }
    if axes.get("correlation_view") != "none":
        metadata["correlation"] = frame.corr(
            method=axes.get("correlation_method", "pearson"), numeric_only=True
        )
    stationarity_test = axes.get("stationarity_test", "none")
    if stationarity_test != "none":
        metadata["stationarity_tests"] = _diagnostic_stationarity_tests(
            frame=frame,
            test=stationarity_test,
            scope=axes.get("stationarity_test_scope", "target_and_predictors"),
            target=l1_artifact.target,
            targets=l1_artifact.targets,
            alpha=float(
                (axes.get("leaf_config") or {}).get("stationarity_alpha", 0.05)
            ),
        )
    return (
        DiagnosticArtifact(
            layer_hooked="l1",
            artifact_type="json",
            metadata=metadata,
            enabled=True,
        ),
        axes,
    )


def _diagnostic_stationarity_tests(
    frame: pd.DataFrame,
    test: str,
    scope: str,
    target: str | None,
    targets: tuple[str, ...],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run ADF / Phillips-Perron / KPSS on each in-scope series.

    ``test``: ``"adf" | "pp" | "kpss" | "multi"``. ``"multi"`` runs all
    three. Phillips-Perron is dispatched via ``arch.unitroot.PhillipsPerron``
    when available, otherwise it is reported as ``unavailable`` (the design
    table notes the dependency)."""

    target_names = set(filter(None, list(targets) + ([target] if target else [])))
    if scope == "target_only":
        cols = [c for c in frame.columns if c in target_names]
    elif scope == "predictors_only":
        cols = [c for c in frame.columns if c not in target_names]
    else:
        cols = list(frame.columns)

    tests = ("adf", "pp", "kpss") if test == "multi" else (test,)
    results: dict[str, dict[str, Any]] = {}
    for col in cols:
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if series.size < 12 or series.std(ddof=0) == 0:
            results[col] = {"status": "insufficient_data", "n_obs": int(series.size)}
            continue
        col_result: dict[str, Any] = {"n_obs": int(series.size)}
        for name in tests:
            try:
                col_result[name] = _run_stationarity(name, series, alpha)
            except Exception as exc:  # pragma: no cover - defensive
                col_result[name] = {"status": "error", "error": str(exc)}
        results[col] = col_result
    return {
        "test": test,
        "scope": scope,
        "alpha": alpha,
        "n_series": len(cols),
        "by_series": results,
    }


def _run_stationarity(name: str, series: pd.Series, alpha: float) -> dict[str, Any]:
    if name == "adf":
        from statsmodels.tsa.stattools import adfuller

        stat, pvalue, *_ = adfuller(series.values, autolag="AIC")
        return {
            "statistic": float(stat),
            "p_value": float(pvalue),
            "reject_unit_root": bool(pvalue < alpha),
        }
    if name == "kpss":
        from statsmodels.tsa.stattools import kpss as _kpss
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, *_ = _kpss(series.values, regression="c", nlags="auto")
        # KPSS null = stationarity; "reject_stationarity" when p < alpha.
        return {
            "statistic": float(stat),
            "p_value": float(pvalue),
            "reject_stationarity": bool(pvalue < alpha),
        }
    if name == "pp":
        # Phillips-Perron (1988): regress y_t on (constant, y_{t-1}); compute the
        # Newey-West HAC variance of the residuals to correct the t-statistic for
        # rho - 1. Native implementation -- no ``arch`` dependency required. The
        # ``arch`` fast path is preserved when installed for cross-validation.
        try:
            from arch.unitroot import PhillipsPerron  # type: ignore

            pp = PhillipsPerron(series.values, trend="c")
            return {
                "statistic": float(pp.stat),
                "p_value": float(pp.pvalue),
                "reject_unit_root": bool(pp.pvalue < alpha),
                "implementation": "arch",
            }
        except ImportError:
            pass
        result = _phillips_perron_native(series.values, alpha=alpha)
        result["implementation"] = "native"
        return result
    return {"status": "unknown_test", "test": name}


def _phillips_perron_native(y: np.ndarray, *, alpha: float = 0.05) -> dict[str, Any]:
    """Phillips-Perron (1988) Z_tau test, native numpy/scipy implementation.

    Regress ``y_t = alpha + rho * y_{t-1} + eps_t`` by OLS, then compute the
    Newey-West HAC variance of the residuals and adjust the t-statistic on
    ``rho - 1``. Critical values are MacKinnon (2010) one-sided 5% / 1% / 10%
    for the constant-only specification.
    """

    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 8:
        return {"status": "insufficient_data", "n_obs": int(n)}
    y_t = y[1:]
    y_lag = y[:-1]
    # OLS: y_t = a + rho * y_{t-1} + eps_t
    X = np.column_stack([np.ones(n - 1), y_lag])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return {"status": "singular_design", "n_obs": int(n)}
    coef = XtX_inv @ X.T @ y_t
    rho = float(coef[1])
    resid = y_t - X @ coef
    # OLS sigma_hat^2.
    sigma2 = float((resid @ resid) / max(n - 2 - 1, 1))
    # Newey-West long-run variance of residuals.
    bandwidth = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    gamma0 = float(np.dot(resid, resid) / n)
    lr_var = gamma0
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1)
        cov = float(np.dot(resid[lag:], resid[:-lag]) / n)
        lr_var += 2.0 * weight * cov
    # Phillips-Perron Z_tau adjustment to the OLS t-stat on (rho - 1):
    se_rho = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    t_rho = (rho - 1.0) / se_rho
    z_tau = float(
        np.sqrt(gamma0 / max(lr_var, 1e-12)) * t_rho
        - 0.5
        * (lr_var - gamma0)
        * np.sqrt(n)
        * np.sqrt(XtX_inv[1, 1])
        / np.sqrt(max(lr_var, 1e-12))
    )
    # Issue #273 -- MacKinnon (2010) finite-sample p-value via the
    # published response-surface coefficients (constant-only spec).
    # The asymptotic distribution is non-standard; we interpolate the
    # critical values from MacKinnon (2010) Table 2 and recover an
    # empirical p-value by inverting the implied CDF.
    p_value = _mackinnon_pp_pvalue(z_tau, n=n, regression="c")
    return {
        "statistic": z_tau,
        "p_value": p_value,
        "reject_unit_root": bool(p_value < alpha),
        "n_obs": int(n),
        "bandwidth_lags": bandwidth,
    }


# MacKinnon (2010) Table 2 finite-sample critical values for the
# Phillips-Perron Z_tau test, constant-only spec (rows: alpha, cols:
# sample size). Source: MacKinnon, J. G. (2010) "Critical Values for
# Cointegration Tests", QED Working Paper 1227, Table 2.
_MACKINNON_PP_C_TABLE = {
    # alpha -> {n -> critical value (one-sided)}
    0.01: {25: -3.75, 50: -3.58, 100: -3.51, 250: -3.46, 500: -3.44, 1000: -3.43},
    0.05: {25: -3.00, 50: -2.93, 100: -2.89, 250: -2.88, 500: -2.87, 1000: -2.86},
    0.10: {25: -2.63, 50: -2.60, 100: -2.58, 250: -2.57, 500: -2.57, 1000: -2.57},
}


def _mackinnon_pp_pvalue(z_tau: float, *, n: int, regression: str = "c") -> float:
    """Issue #273 -- recover an empirical p-value from MacKinnon's Table 2.

    For ``regression = c`` (constant only) interpolate the critical value
    by sample size, then bracket ``z_tau`` between two adjacent quantiles
    to estimate the p-value via linear interpolation in (alpha, cv) space.
    Uses MacKinnon (2010) Table 2 (1% / 5% / 10% only). Outside the
    bracket we extrapolate by saturation -- p < 0.01 or p > 0.10.
    """

    if regression != "c":
        # Only the constant-only spec is tabulated here; trend variants
        # would need MacKinnon Tables 3-4.
        from scipy import stats as _stats

        return float(_stats.norm.cdf(z_tau))
    table = _MACKINNON_PP_C_TABLE
    sizes = sorted(next(iter(table.values())).keys())
    # Pick the closest tabulated sample size below and above n.
    n_clamped = max(sizes[0], min(sizes[-1], int(n)))
    lower_n = max((s for s in sizes if s <= n_clamped), default=sizes[0])
    upper_n = min((s for s in sizes if s >= n_clamped), default=sizes[-1])
    weight = 0.0 if lower_n == upper_n else (n_clamped - lower_n) / (upper_n - lower_n)

    # Critical values at this n by linear interp.
    cvs: dict[float, float] = {}
    for alpha, by_n in table.items():
        cv = by_n[lower_n] * (1 - weight) + by_n[upper_n] * weight
        cvs[alpha] = cv
    # alphas sorted ascending; CVs become less negative with larger alpha.
    alphas = sorted(cvs)
    cv_values = [cvs[a] for a in alphas]
    if z_tau <= cv_values[0]:
        # More negative than the 1% critical value -> p < 0.01.
        return 0.005
    if z_tau >= cv_values[-1]:
        # Less negative than the 10% critical value -> p > 0.10.
        return 0.50
    # Bracket and interpolate in (cv, alpha) space.
    for i in range(len(alphas) - 1):
        if cv_values[i] <= z_tau <= cv_values[i + 1]:
            span = cv_values[i + 1] - cv_values[i]
            if span <= 0:
                return float(alphas[i])
            t = (z_tau - cv_values[i]) / span
            return float(alphas[i] * (1 - t) + alphas[i + 1] * t)
    return 0.50


def materialize_l2_5_diagnostic(
    recipe_root: dict[str, Any],
    l1_artifact: L1DataDefinitionArtifact,
    l2_artifact: L2CleanPanelArtifact,
) -> tuple[DiagnosticArtifact, dict[str, Any]]:
    raw = recipe_root.get("2_5_pre_post_preprocessing", {}) or {}
    report = l2_5_layer.validate_layer(raw)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    resolved = l2_5_layer.resolve_axes_from_raw(raw)
    axes = _plain_axes(resolved)
    if not resolved.get("enabled", False):
        return _disabled_diagnostic("l1+l2", axes), axes
    raw_frame = l1_artifact.raw_panel.data.copy()
    clean_frame = l2_artifact.panel.data.copy()
    metadata = {
        "runtime": "core_l2_5_diagnostic",
        "axis_resolved": axes,
        "comparison": _diagnostic_pre_post_comparison(raw_frame, clean_frame),
        "distribution_shift": _diagnostic_distribution_shift(
            raw_frame, clean_frame, axes.get("distribution_metric", [])
        ),
        "cleaning_effect_summary": {
            "n_imputed_cells": l2_artifact.n_imputed_cells,
            "n_outliers_flagged": l2_artifact.n_outliers_flagged,
            "n_truncated_obs": l2_artifact.n_truncated_obs,
            "transform_map_applied": dict(l2_artifact.transform_map_applied),
            "cleaning_log": l2_artifact.cleaning_log,
        },
    }
    if axes.get("correlation_shift") != "none":
        metadata["correlation_shift"] = clean_frame.corr(
            numeric_only=True
        ) - raw_frame.corr(numeric_only=True)
    # Issue #213: ``correlation_shift = delta_matrix`` -- explicit
    # post-minus-pre matrix exposed as a separate field for downstream
    # consumers.
    if axes.get("correlation_shift") == "delta_matrix":
        try:
            metadata["delta_matrix"] = (
                clean_frame.corr(numeric_only=True) - raw_frame.corr(numeric_only=True)
            ).fillna(0.0)
        except Exception:
            pass
    # ``summary_split = per_decade`` -- per-decade summary statistics on
    # both the raw and cleaned panel, useful for visual comparison.
    if axes.get("summary_split") == "per_decade":
        metadata["per_decade_summary"] = _diagnostic_per_decade_summary(
            raw_frame, clean_frame
        )
    # ``t_code_application_log = per_series_detail`` -- per-series record
    # of which transform code was applied (from L2 transform_map).
    if axes.get("t_code_application_log") == "per_series_detail":
        metadata["t_code_log_per_series"] = {
            str(series): int(code)
            for series, code in l2_artifact.transform_map_applied.items()
        }
    return (
        DiagnosticArtifact(
            layer_hooked="l1+l2",
            artifact_type="json",
            metadata=metadata,
            enabled=True,
        ),
        axes,
    )


def _diagnostic_per_decade_summary(
    raw: pd.DataFrame, clean: pd.DataFrame
) -> dict[str, Any]:
    """Group rows by decade (when the index is datetime) and return per-decade
    mean / std for raw vs cleaned panels."""

    out: dict[str, Any] = {}
    for label, frame in (("raw", raw), ("clean", clean)):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
            out[label] = {}
            continue
        decade = (frame.index.year // 10) * 10
        grouped = frame.assign(__decade=decade).groupby("__decade")
        out[label] = {
            int(decade_value): {
                "mean": float(
                    group.drop(columns="__decade")
                    .select_dtypes("number")
                    .mean(numeric_only=True)
                    .mean()
                ),
                "std": float(
                    group.drop(columns="__decade")
                    .select_dtypes("number")
                    .std(ddof=0, numeric_only=True)
                    .mean()
                ),
                "n_obs": int(len(group)),
            }
            for decade_value, group in grouped
        }
    return out


def materialize_l3_5_diagnostic(
    recipe_root: dict[str, Any],
    l1_artifact: L1DataDefinitionArtifact,
    l2_artifact: L2CleanPanelArtifact,
    l3_features: L3FeaturesArtifact,
    l3_metadata: L3MetadataArtifact,
) -> tuple[DiagnosticArtifact, dict[str, Any]]:
    raw = recipe_root.get("3_5_feature_diagnostics", {}) or {}
    context = l3_5_layer._recipe_context(recipe_root)
    report = l3_5_layer.validate_layer(raw, context=context)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    resolved = l3_5_layer.resolve_axes_from_raw(raw, context=context)
    axes = _plain_axes(resolved)
    if not resolved.get("enabled", False):
        return _disabled_diagnostic("l1+l2+l3", axes), axes

    raw_frame = l1_artifact.raw_panel.data.copy()
    clean_frame = l2_artifact.panel.data.copy()
    feature_frame = l3_features.X_final.data.copy()
    metadata = {
        "runtime": "core_l3_5_diagnostic",
        "axis_resolved": axes,
        "comparison": _diagnostic_l3_comparison(
            raw_frame, clean_frame, feature_frame, l3_features
        ),
        "feature_summary": _diagnostic_feature_summary(feature_frame),
        "lineage_summary": _diagnostic_l3_lineage_summary(l3_metadata),
        "factor_block": {
            "active": bool(context.get("has_factor_step")),
            "n_factors_to_show": axes.get("leaf_config", {}).get(
                "n_factors_to_show", 8
            ),
        },
        "lag_block": _diagnostic_l3_lag_summary(
            feature_frame, active=bool(context.get("has_lag_step"))
        ),
        "selection_summary": {
            "active": bool(context.get("has_feature_selection_step"))
        },
    }
    if axes.get("feature_correlation") != "none":
        metadata["feature_correlation"] = feature_frame.corr(
            method=axes.get("correlation_method", "pearson"), numeric_only=True
        )
    # Issue #211: factor diagnostics. When the L3 panel has at least
    # ``n_factors_to_show`` columns, run a quick PCA so the diagnostic sink
    # carries the eigenvalue scree, loadings, and factor time series.
    n_factors_to_show = int(
        (axes.get("leaf_config", {}) or {}).get("n_factors_to_show", 4)
    )
    if feature_frame.shape[0] >= 4 and feature_frame.shape[1] >= 2:
        try:
            from sklearn.decomposition import PCA as _PCA

            n_comp = min(
                n_factors_to_show, feature_frame.shape[0] - 1, feature_frame.shape[1]
            )
            pca = _PCA(n_components=n_comp)
            scores = pca.fit_transform(feature_frame.fillna(0.0).to_numpy())
            metadata["factor_diagnostics"] = {
                "explained_variance_ratio": [
                    float(v) for v in pca.explained_variance_ratio_
                ],
                "eigenvalues": [float(v) for v in pca.explained_variance_],
                "cumulative_variance": [
                    float(v) for v in np.cumsum(pca.explained_variance_ratio_)
                ],
                "loadings": pca.components_.tolist(),
                "factor_scores_shape": [scores.shape[0], scores.shape[1]],
            }
        except Exception:
            pass
    return (
        DiagnosticArtifact(
            layer_hooked="l1+l2+l3",
            artifact_type="json",
            metadata=metadata,
            enabled=True,
        ),
        axes,
    )


def materialize_l4_5_diagnostic(
    recipe_root: dict[str, Any],
    l3_features: L3FeaturesArtifact,
    l4_forecasts: L4ForecastsArtifact,
    l4_models: L4ModelArtifactsArtifact,
    l4_training: L4TrainingMetadataArtifact,
) -> tuple[DiagnosticArtifact, dict[str, Any]]:
    raw = recipe_root.get("4_5_generator_diagnostics", {}) or {}
    context = l4_5_layer._recipe_context(recipe_root)
    report = l4_5_layer.validate_layer(raw, context=context)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    resolved = l4_5_layer.resolve_axes_from_raw(raw, context=context)
    axes = _plain_axes(resolved)
    if not resolved.get("enabled", False):
        return _disabled_diagnostic("l4", axes), axes

    actual = l3_features.y_final.metadata.values.get("data")
    metadata = {
        "runtime": "core_l4_5_diagnostic",
        "axis_resolved": axes,
        "forecast_summary": _diagnostic_l4_forecast_summary(l4_forecasts),
        "model_summary": _diagnostic_l4_model_summary(l4_models),
        "training_summary": _diagnostic_l4_training_summary(l4_training),
        "fit_summary": _diagnostic_l4_fit_summary(
            l4_forecasts, actual if isinstance(actual, pd.Series) else None
        ),
    }
    if axes.get("window_view") != "none":
        metadata["window_stability"] = _diagnostic_l4_window_summary(l4_training)
    # Issue #212: window stability + per-origin loss series for the rolling
    # training-loss curve diagnostic figure.
    if isinstance(actual, pd.Series) and l4_forecasts.forecasts:
        per_origin: dict[Any, float] = {}
        for (
            model_id,
            target,
            horizon,
            origin,
        ), forecast in l4_forecasts.forecasts.items():
            if origin not in actual.index:
                continue
            per_origin[origin] = (
                per_origin.get(origin, 0.0)
                + (float(actual.loc[origin]) - float(forecast)) ** 2
            )
        if per_origin:
            sorted_origins = sorted(per_origin)
            metadata["per_origin_squared_error"] = {
                "origins": [str(o) for o in sorted_origins],
                "values": [float(per_origin[o]) for o in sorted_origins],
            }
    # Issue #256 -- additional L4.5 views.
    if isinstance(actual, pd.Series) and l4_forecasts.forecasts:
        residuals: dict[str, list[float]] = {}
        for (
            model_id,
            target,
            horizon,
            origin,
        ), forecast in l4_forecasts.forecasts.items():
            if origin not in actual.index:
                continue
            r = float(actual.loc[origin]) - float(forecast)
            residuals.setdefault(str(model_id), []).append(r)
        # Residual ACF (lag 1-5) per model -- pin temporal autocorrelation.
        if axes.get("fit_view") in {"residual_acf", "multi"}:
            acf_table: dict[str, list[float]] = {}
            for model_id, resid in residuals.items():
                arr = np.asarray(resid, dtype=float)
                if arr.size > 6:
                    centred = arr - arr.mean()
                    denom = float((centred**2).sum())
                    acf = []
                    for lag in range(1, 6):
                        if arr.size <= lag or denom <= 0:
                            acf.append(0.0)
                            continue
                        acf.append(
                            float((centred[:-lag] * centred[lag:]).sum() / denom)
                        )
                    acf_table[model_id] = acf
            metadata["residual_acf"] = acf_table
        # QQ summary -- residuals vs standard-normal quantiles.
        if axes.get("fit_view") in {"residual_qq", "multi"}:
            qq_table: dict[str, dict[str, list[float]]] = {}
            from scipy import stats as _stats

            for model_id, resid in residuals.items():
                arr = np.sort(np.asarray(resid, dtype=float))
                n = arr.size
                if n < 4:
                    continue
                expected = _stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
                qq_table[model_id] = {
                    "expected_quantiles": [float(v) for v in expected],
                    "observed_residuals": [float(v) for v in arr],
                }
            metadata["residual_qq"] = qq_table
        # Issue #278 -- fitted vs actual scatter (per model).
        if axes.get("fit_view") in {"fitted_vs_actual", "multi"}:
            fva_table: dict[str, dict[str, list[float]]] = {}
            for model_id in residuals:
                # Recover (fitted, actual) pairs by re-walking the forecast dict.
                actual_vals: list[float] = []
                fitted_vals: list[float] = []
                for (
                    m_id,
                    _target,
                    _horizon,
                    origin,
                ), forecast in l4_forecasts.forecasts.items():
                    if str(m_id) != model_id or origin not in actual.index:
                        continue
                    actual_vals.append(float(actual.loc[origin]))
                    fitted_vals.append(float(forecast))
                if actual_vals:
                    fva_table[model_id] = {
                        "actual": actual_vals,
                        "fitted": fitted_vals,
                    }
            metadata["fitted_vs_actual"] = fva_table
        # Issue #278 -- residual time series (per model, ordered by origin).
        if axes.get("fit_view") in {"residual_time", "multi"}:
            time_table: dict[str, dict[str, list[Any]]] = {}
            for model_id in residuals:
                pairs: list[tuple[Any, float]] = []
                for (
                    m_id,
                    _target,
                    _horizon,
                    origin,
                ), forecast in l4_forecasts.forecasts.items():
                    if str(m_id) != model_id or origin not in actual.index:
                        continue
                    pairs.append((origin, float(actual.loc[origin]) - float(forecast)))
                pairs.sort()
                time_table[model_id] = {
                    "origins": [str(o) for o, _ in pairs],
                    "residuals": [r for _, r in pairs],
                }
            metadata["residual_time"] = time_table
    return (
        DiagnosticArtifact(
            layer_hooked="l4",
            artifact_type="json",
            metadata=metadata,
            enabled=True,
        ),
        axes,
    )


def _disabled_diagnostic(layer_hooked: str, axes: dict[str, Any]) -> DiagnosticArtifact:
    return DiagnosticArtifact(
        layer_hooked=layer_hooked,
        artifact_type="json",
        metadata={"runtime": "core_diagnostic_disabled", "axis_resolved": axes},
        enabled=False,
    )


def _diagnostic_sample_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "start": {
            column: _iso_or_none(frame[column].first_valid_index())
            for column in frame.columns
        },
        "end": {
            column: _iso_or_none(frame[column].last_valid_index())
            for column in frame.columns
        },
        "n_obs": frame.notna().sum().astype(int).to_dict(),
        "n_missing": frame.isna().sum().astype(int).to_dict(),
        "panel_shape": frame.shape,
    }


def _diagnostic_univariate_summary(
    frame: pd.DataFrame, metrics: list[str]
) -> dict[str, dict[str, float | int | None]]:
    numeric = frame.select_dtypes("number")
    summary: dict[str, dict[str, float | int | None]] = {}
    for column in numeric.columns:
        series = numeric[column]
        values: dict[str, float | int | None] = {}
        for metric in metrics:
            if metric == "mean":
                values[metric] = _float_or_none(series.mean())
            elif metric == "sd":
                values[metric] = _float_or_none(series.std())
            elif metric == "min":
                values[metric] = _float_or_none(series.min())
            elif metric == "max":
                values[metric] = _float_or_none(series.max())
            elif metric == "skew":
                values[metric] = _float_or_none(series.skew())
            elif metric == "kurtosis":
                values[metric] = _float_or_none(series.kurtosis())
            elif metric == "n_obs":
                values[metric] = int(series.notna().sum())
            elif metric == "n_missing":
                values[metric] = int(series.isna().sum())
        summary[column] = values
    return summary


def _diagnostic_missing_outlier_audit(
    frame: pd.DataFrame, leaf_config: dict[str, Any]
) -> dict[str, Any]:
    numeric = frame.select_dtypes("number")
    threshold = float(leaf_config.get("outlier_threshold_iqr", 10.0))
    median = numeric.median()
    iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
    outlier_mask = (numeric - median).abs() > threshold * iqr.replace(0, pd.NA)
    return {
        "missing_count": frame.isna().sum().astype(int).to_dict(),
        "longest_gap": {
            column: _longest_missing_gap(frame[column]) for column in frame.columns
        },
        "iqr_outlier_count": outlier_mask.fillna(False).sum().astype(int).to_dict(),
    }


def _diagnostic_pre_post_comparison(
    raw_frame: pd.DataFrame, clean_frame: pd.DataFrame
) -> dict[str, Any]:
    return {
        "raw_shape": raw_frame.shape,
        "clean_shape": clean_frame.shape,
        "raw_missing_total": int(raw_frame.isna().sum().sum()),
        "clean_missing_total": int(clean_frame.isna().sum().sum()),
        "common_columns": sorted(set(raw_frame.columns) & set(clean_frame.columns)),
    }


def _diagnostic_distribution_shift(
    raw_frame: pd.DataFrame, clean_frame: pd.DataFrame, metrics: list[str]
) -> dict[str, dict[str, float | None]]:
    common = [
        column
        for column in raw_frame.select_dtypes("number").columns
        if column in clean_frame.select_dtypes("number").columns
    ]
    shifts: dict[str, dict[str, float | None]] = {}
    for column in common:
        raw = raw_frame[column]
        clean = clean_frame[column]
        values: dict[str, float | None] = {}
        for metric in metrics:
            if metric == "mean_change":
                values[metric] = _float_or_none(clean.mean() - raw.mean())
            elif metric == "sd_change":
                raw_sd = raw.std()
                values[metric] = (
                    _float_or_none(clean.std() / raw_sd) if raw_sd else None
                )
            elif metric == "skew_change":
                values[metric] = _float_or_none(clean.skew() - raw.skew())
            elif metric == "kurtosis_change":
                values[metric] = _float_or_none(clean.kurtosis() - raw.kurtosis())
            elif metric == "ks_statistic":
                values[metric] = _ks_statistic(raw.dropna(), clean.dropna())
        shifts[column] = values
    return shifts


def _diagnostic_l3_comparison(
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    l3_features: L3FeaturesArtifact,
) -> dict[str, Any]:
    return {
        "raw_shape": raw_frame.shape,
        "clean_shape": clean_frame.shape,
        "feature_shape": feature_frame.shape,
        "y_shape": l3_features.y_final.shape,
        "sample_start": _iso_or_none(l3_features.sample_index[0])
        if l3_features.sample_index is not None and len(l3_features.sample_index)
        else None,
        "sample_end": _iso_or_none(l3_features.sample_index[-1])
        if l3_features.sample_index is not None and len(l3_features.sample_index)
        else None,
        "raw_missing_total": int(raw_frame.isna().sum().sum()),
        "clean_missing_total": int(clean_frame.isna().sum().sum()),
        "feature_missing_total": int(feature_frame.isna().sum().sum()),
    }


def _diagnostic_feature_summary(feature_frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_obs": int(len(feature_frame)),
        "n_features": int(len(feature_frame.columns)),
        "columns": tuple(str(column) for column in feature_frame.columns),
        "missing_by_feature": feature_frame.isna().sum().astype(int).to_dict(),
    }


def _diagnostic_l3_lineage_summary(l3_metadata: L3MetadataArtifact) -> dict[str, Any]:
    pipeline_ids = sorted(
        {
            lineage.pipeline_id
            for lineage in l3_metadata.column_lineage.values()
            if lineage.pipeline_id
        }
    )
    return {
        "n_column_lineage": len(l3_metadata.column_lineage),
        "n_pipeline_definitions": len(l3_metadata.pipeline_definitions),
        "pipeline_ids": tuple(pipeline_ids),
        "source_variables": {
            key: tuple(value) for key, value in l3_metadata.source_variables.items()
        },
    }


def _diagnostic_l3_lag_summary(
    feature_frame: pd.DataFrame, *, active: bool
) -> dict[str, Any]:
    lag_columns = [
        str(column)
        for column in feature_frame.columns
        if "_lag" in str(column) or "_ma" in str(column) or "_s" in str(column)
    ]
    return {
        "active": active,
        "lag_feature_count": len(lag_columns),
        "lag_features": tuple(lag_columns),
    }


def _diagnostic_l4_forecast_summary(
    l4_forecasts: L4ForecastsArtifact,
) -> dict[str, Any]:
    return {
        "n_forecasts": len(l4_forecasts.forecasts),
        "forecast_object": l4_forecasts.forecast_object,
        "model_ids": tuple(l4_forecasts.model_ids),
        "targets": tuple(l4_forecasts.targets),
        "horizons": tuple(l4_forecasts.horizons),
        "sample_start": _iso_or_none(l4_forecasts.sample_index[0])
        if l4_forecasts.sample_index is not None and len(l4_forecasts.sample_index)
        else None,
        "sample_end": _iso_or_none(l4_forecasts.sample_index[-1])
        if l4_forecasts.sample_index is not None and len(l4_forecasts.sample_index)
        else None,
    }


def _diagnostic_l4_model_summary(l4_models: L4ModelArtifactsArtifact) -> dict[str, Any]:
    return {
        model_id: {
            "family": artifact.family,
            "framework": artifact.framework,
            "n_features": len(artifact.feature_names),
            "is_benchmark": bool(l4_models.is_benchmark.get(model_id, False)),
            "fit_metadata": dict(artifact.fit_metadata),
        }
        for model_id, artifact in l4_models.artifacts.items()
    }


def _diagnostic_l4_training_summary(
    l4_training: L4TrainingMetadataArtifact,
) -> dict[str, Any]:
    return {
        "n_forecast_origins": len(l4_training.forecast_origins),
        "forecast_origins": tuple(
            _iso_or_none(origin) for origin in l4_training.forecast_origins
        ),
        "refit_origin_count": {
            model_id: len(origins)
            for model_id, origins in l4_training.refit_origins.items()
        },
        "training_window_count": len(l4_training.training_window_per_origin),
    }


def _diagnostic_l4_fit_summary(
    l4_forecasts: L4ForecastsArtifact, actual: pd.Series | None
) -> dict[str, dict[str, float | int | None]]:
    if actual is None:
        return {}
    rows: list[dict[str, Any]] = []
    for (model_id, target, horizon, origin), forecast in l4_forecasts.forecasts.items():
        if origin not in actual.index:
            continue
        error = float(actual.loc[origin]) - float(forecast)
        rows.append(
            {
                "model_id": model_id,
                "target": target,
                "horizon": horizon,
                "squared_error": error**2,
                "absolute_error": abs(error),
            }
        )
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model_id", "target", "horizon"]).agg(
        n=("squared_error", "size"),
        mse=("squared_error", "mean"),
        mae=("absolute_error", "mean"),
    )
    return {
        f"{model_id}|{target}|h{horizon}": {
            "n": int(values["n"]),
            "mse": float(values["mse"]),
            "mae": float(values["mae"]),
        }
        for (model_id, target, horizon), values in summary.iterrows()
    }


def _diagnostic_l4_window_summary(
    l4_training: L4TrainingMetadataArtifact,
) -> dict[str, Any]:
    by_model: dict[str, list[tuple[Any, Any, Any]]] = {}
    for (model_id, origin), window in l4_training.training_window_per_origin.items():
        by_model.setdefault(model_id, []).append((origin, window[0], window[1]))
    return {
        model_id: {
            "n_windows": len(windows),
            "first_window": tuple(
                _iso_or_none(value) for value in min(windows, key=lambda row: row[0])
            )
            if windows
            else (),
            "last_window": tuple(
                _iso_or_none(value) for value in max(windows, key=lambda row: row[0])
            )
            if windows
            else (),
        }
        for model_id, windows in by_model.items()
    }


def _fit_target_transformer(name: str, y_train: pd.Series) -> dict[str, Any] | None:
    """Issue #277 -- instantiate a registered target transformer and fit
    it on the training y. Returns a dict with ``transform`` and
    ``inverse_transform`` callables; ``None`` when the registry doesn't
    know the name (caller falls through to identity).
    """

    try:
        from .. import custom as _custom_mod
    except ImportError:  # pragma: no cover
        return None
    if not _custom_mod.is_custom_target_transformer(str(name)):
        return None
    spec = _custom_mod.get_custom_target_transformer(str(name))
    transformer = spec.factory()
    transformer.fit(y_train, {})
    return {
        "transform": lambda series: transformer.transform(series, {}),
        "inverse_transform": lambda values: transformer.inverse_transform_prediction(
            values, {}
        ),
        "name": name,
    }


def _apply_inverse_target_transform(
    forecasts: dict[tuple[str, str, int, Any], float],
    l3_features: L3FeaturesArtifact,
) -> dict[tuple[str, str, int, Any], float]:
    """Issue #277 -- when a target transformer is active, push the L4
    point forecasts back to the raw scale before they reach L5."""

    state = (l3_features.y_final.metadata.values or {}).get("target_transformer_state")
    if state is None or "inverse_transform" not in state:
        return forecasts
    return {
        key: float(state["inverse_transform"]([value])[0])
        for key, value in forecasts.items()
    }


def materialize_l3_minimal(
    recipe_root: dict[str, Any],
    l1_artifact: L1DataDefinitionArtifact,
    l2_artifact: L2CleanPanelArtifact,
    l2_resolved: "l2_layer.L2ResolvedAxes | None" = None,
) -> tuple[L3FeaturesArtifact, L3MetadataArtifact]:
    raw = recipe_root.get("3_feature_engineering", {}) or {}
    report = l3_layer.validate_layer(raw, recipe_context=_l3_context(l1_artifact))
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    dag = l3_layer.normalize_to_dag_form(raw)
    df = l2_artifact.panel.data.copy()
    target_name = l1_artifact.target or (
        l1_artifact.targets[0] if l1_artifact.targets else None
    )
    if not target_name or target_name not in df.columns:
        raise ValueError("minimal L3 runtime requires target column in L2 clean panel")

    node_values = _execute_l3_dag(dag, df, target_name)
    sink_node = dag.nodes.get(dag.sinks.get("l3_features_v1", ""))
    if sink_node is None or len(sink_node.inputs) < 2:
        raise ValueError(
            "minimal L3 runtime requires l3_features_v1 sink with X_final and y_final"
        )
    X = _as_frame(node_values[sink_node.inputs[0].node_id])
    y = _as_series(node_values[sink_node.inputs[1].node_id], name=target_name)
    aligned_index = pd.concat([X, y], axis=1).dropna(axis=0, how="any").index
    X_aligned = X.loc[aligned_index]
    y_aligned = y.loc[aligned_index]
    # Issue #277 -- target_transformer dispatch. When set in L1
    # leaf_config, fit on training y, transform pre-L4. The inverse
    # transform is applied at the L4/L5 boundary by ``_apply_target_transform``
    # so final forecasts + metrics stay on the raw target scale.
    transformer_name = (l1_artifact.leaf_config or {}).get("target_transformer")
    transformer_state = None
    if transformer_name:
        transformer_state = _fit_target_transformer(transformer_name, y_aligned)
        if transformer_state is not None:
            y_aligned = transformer_state["transform"](y_aligned)
    horizon = int(
        (y.attrs or {}).get(
            "horizon",
            l1_artifact.target_horizons[0] if l1_artifact.target_horizons else 1,
        )
    )
    y_meta = {"stage": "l3_y_final", "horizon": horizon, "data": y_aligned}
    # Phase B-15 paper-15 F4: forward the un-averaged source y so the L4
    # walk-forward can fit Eq. 4 (h separate per-horizon models) when
    # ``forecast_strategy="path_average_eq4"`` is opted in. The L3
    # ``target_construction`` op stashes ``y_orig`` on the series ``attrs``
    # only for ``cumulative_average`` / ``path_average`` modes.
    y_orig_attr = (y.attrs or {}).get("y_orig")
    if isinstance(y_orig_attr, pd.Series):
        y_meta["y_orig"] = y_orig_attr
    y_meta["target_mode"] = (y.attrs or {}).get("mode", "point_forecast")
    if transformer_state is not None:
        y_meta["target_transformer"] = transformer_name
        y_meta["target_transformer_state"] = transformer_state
        # Cache the raw (pre-transform) y so L5 can evaluate forecasts on
        # the raw target scale per the contract.
        y_meta["raw_data"] = y.loc[aligned_index]
    # Phase B-1 F2/F3 fix: when any L3 node carries
    # ``params.temporal_rule == "expanding_window_per_origin"``, attach a
    # per-origin rematerialization closure to the X_final panel metadata.
    # The L4 walk-forward (``materialize_l4_minimal`` / ``_run_l4_fit_node``)
    # checks for this closure and, when present, calls it at every origin to
    # obtain a leak-free X_origin. Nodes without ``temporal_rule`` set, or
    # set to ``full_sample``/``full_sample_once``, are still computed once
    # on the full panel (no behavior change). This honors the
    # ``temporal_rule`` schema convention in ``core/ops/l3_ops.py`` -- prior
    # to this fix the runtime only validated the value (`_temporal_present`)
    # but never read it.
    expanding_node_ids = _l3_expanding_window_nodes(dag)
    x_metadata: dict[str, Any] = {"stage": "l3_X_final", "runtime": "l3_dag"}
    if expanding_node_ids:
        affected_node_ids = _l3_per_origin_affected_nodes(dag, expanding_node_ids)
        x_sink_id = sink_node.inputs[0].node_id
        # Dataset reference for the closure -- copy to insulate from
        # downstream mutation; the closure shouldn't affect L2 artifacts.
        df_for_origin = df.copy()
        target_name_for_closure = target_name
        # Map the post-dropna aligned index (the index L4 sees) into df row
        # positions so the L4 walk-forward can pass an origin *date* and
        # the closure can recover the correct ``origin_index`` into ``df``.
        df_index = pd.Index(df_for_origin.index)
        aligned_index_snapshot = pd.Index(aligned_index)

        # F-P1-2/F-P1-3 fix: capture l2 resolved axes for per-origin
        # imputation/outlier application inside the closure.
        _l3_l2_resolved_ref = l2_resolved
        _l3_l2_leaf_config_ref = dict(l1_artifact.leaf_config or {})
        _l3_l2_imputation_temporal = (
            (l2_resolved or {}).get("imputation_temporal_rule", "expanding_window_per_origin")
            if l2_resolved is not None
            else "expanding_window_per_origin"
        )

        def _per_origin_callable(origin_label: Any) -> pd.DataFrame:
            try:
                origin_index = df_index.get_loc(origin_label)
            except KeyError:
                # Fall back to integer interpretation if caller passed a
                # raw position. This keeps the closure usable in tests
                # that pass an integer directly.
                if isinstance(origin_label, int):
                    origin_index = origin_label
                else:
                    raise
            # F-P1-2/F-P1-3 fix: apply per-origin outlier/imputation on the
            # expanding window slice so L3 DAG receives leak-free data.
            df_origin_input = df_for_origin
            if _l3_l2_imputation_temporal == "expanding_window_per_origin" and _l3_l2_resolved_ref is not None:
                origin_ts = df_for_origin.index[origin_index]
                df_origin_input = _apply_outlier_policy_per_origin(
                    df_for_origin, _l3_l2_resolved_ref, _l3_l2_leaf_config_ref, origin_ts
                )
                df_origin_input = _apply_imputation_per_origin(
                    df_origin_input, _l3_l2_resolved_ref, origin_ts
                )
            X_origin_full = materialize_l3_per_origin(
                dag,
                df_origin_input,
                target_name_for_closure,
                origin_index=origin_index,
                affected_node_ids=affected_node_ids,
                x_sink_node_id=x_sink_id,
                cached_full_node_values=node_values,
            )
            # Restrict to the rows that survive the post-L3 dropna alignment
            # *and* fall on or before the origin. This matches the contract
            # of the full-sample path: the L4 walk-forward sees a clean (no
            # NaN) X aligned to the same dates as the cached full-sample X
            # would have, but with per-origin loadings.
            keep = aligned_index_snapshot[aligned_index_snapshot <= origin_label]
            X_origin = X_origin_full.reindex(index=keep)
            # Defensive dropna: a per-origin op may emit NaNs for rows that
            # were valid in the full-sample run (e.g. small-sample variance
            # zero). Drop those rows so downstream estimators get a clean X.
            X_origin = X_origin.dropna(axis=0, how="any")
            return X_origin

        x_metadata["l3_per_origin_callable"] = _per_origin_callable
        x_metadata["l3_per_origin_node_ids"] = tuple(sorted(affected_node_ids))
        x_metadata["l3_temporal_rule"] = "expanding_window_per_origin"
    return (
        L3FeaturesArtifact(
            X_final=_panel_from_frame(X_aligned, metadata=x_metadata),
            y_final=Series(
                shape=y_aligned.shape,
                name=target_name,
                metadata=SeriesMetadata(values=y_meta),
            ),
            sample_index=pd.DatetimeIndex(aligned_index),
            horizon_set=(horizon,),
        ),
        l3_layer.build_metadata_artifact(raw),
    )


def _l3_expanding_window_nodes(dag) -> set[str]:
    """Phase B-1 F3 fix: return the set of L3 node ids that declare
    ``params.temporal_rule == "expanding_window_per_origin"``.

    Nodes whose ``temporal_rule`` is missing, ``full_sample``, or
    ``full_sample_once`` are excluded -- those keep the legacy one-shot
    full-panel materialization path.
    """

    expanding: set[str] = set()
    for node_id, node in dag.nodes.items():
        if node.type != "step":
            continue
        rule = (node.params or {}).get("temporal_rule")
        if isinstance(rule, str) and rule == "expanding_window_per_origin":
            expanding.add(node_id)
    return expanding


def _l3_per_origin_affected_nodes(dag, seed_node_ids: set[str]) -> set[str]:
    """Phase B-1 F2 fix: BFS-expand ``seed_node_ids`` over the L3 DAG to
    include every transitive downstream dependent. Any node consuming the
    output of an expanding-window node must also be re-materialized per
    origin (its inputs change at each origin).

    Phase B-1b residual-leak fix (Round 6): also walk UPSTREAM with a
    filter that admits only ``target_construction`` nodes (and the
    ``target_construction`` ancestors of the seeds and of each affected
    node). The ``target_construction`` op horizon-shifts ``y`` by
    ``-h`` -- so when the runtime falls back to a cached full-sample
    ``y.shift(-h)`` and trims to ``iloc[: origin + 1]`` the trailing
    ``h`` rows of that trimmed series are precisely the post-origin y
    observations ``y[origin + 1 .. origin + h]``. Marking the
    ``target_construction`` node as affected forces the per-origin
    closure to re-execute the shift on the truncated input, which is
    leak-free by construction (the trailing rows become NaN and are
    dropped by the consuming op via its own ``notna`` mask).

    The upstream traversal is *filtered*: we only follow edges that
    cross into a ``target_construction`` node. We do not pull in
    arbitrary upstream nodes (e.g. raw ``src_y``) because their cached
    full-sample value, when trimmed to ``iloc[: origin + 1]``, already
    matches the in-sample contract -- only the horizon-shifting
    ``target_construction`` step contaminates the trimmed view.
    """

    affected: set[str] = set(seed_node_ids)
    # Build reverse adjacency: for each node, which nodes consume it.
    consumers: dict[str, set[str]] = {node_id: set() for node_id in dag.nodes}
    for node_id, node in dag.nodes.items():
        for ref in node.inputs:
            consumers.setdefault(ref.node_id, set()).add(node_id)
    queue = list(seed_node_ids)
    while queue:
        current = queue.pop()
        for downstream in consumers.get(current, ()):
            if downstream not in affected:
                affected.add(downstream)
                queue.append(downstream)

    # Phase B-1b: walk upstream from every node currently in the
    # affected set, following inputs only when the upstream node is a
    # ``target_construction`` step. This keeps the upstream pull-in
    # narrow (we do not mark ``src_y``, ``src_X``, or other upstreams)
    # while ensuring every horizon-shifted target ancestor is
    # re-executed per origin.
    upstream_queue = list(affected)
    while upstream_queue:
        current = upstream_queue.pop()
        node = dag.nodes.get(current)
        if node is None:
            continue
        for ref in node.inputs:
            upstream_id = ref.node_id
            upstream_node = dag.nodes.get(upstream_id)
            if upstream_node is None:
                continue
            if getattr(upstream_node, "op", None) != "target_construction":
                continue
            if upstream_id in affected:
                continue
            affected.add(upstream_id)
            upstream_queue.append(upstream_id)

    return affected


def materialize_l3_per_origin(
    dag,
    df: pd.DataFrame,
    target_name: str,
    *,
    origin_index: int,
    affected_node_ids: set[str] | None = None,
    x_sink_node_id: str | None = None,
    cached_full_node_values: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Phase B-1 F2/F3 fix: re-materialize an L3 sub-DAG using only data
    ``df.iloc[: origin_index + 1]`` (i.e. the expanding-window-per-origin
    contract from ``core/ops/l3_ops.py``).

    Parameters
    ----------
    dag : DAG
        The L3 DAG built by ``l3_layer.normalize_to_dag_form``.
    df : pd.DataFrame
        The full L2 clean panel (target column included).
    target_name : str
        The target column name in ``df``.
    origin_index : int
        The walk-forward origin position. The function will pass
        ``df.iloc[: origin_index + 1]`` (inclusive of the origin row) to
        every node in ``affected_node_ids``; any node *not* in
        ``affected_node_ids`` is taken from ``cached_full_node_values``
        (computed once on the full panel) so the per-origin loop only
        re-runs the sub-DAG that actually depends on temporal data.
    affected_node_ids : set[str] | None
        Output of ``_l3_per_origin_affected_nodes``. If ``None``, every
        node is re-materialized per origin (defensive fallback).
    x_sink_node_id : str | None
        The DAG node id whose output should be returned as the X panel.
        Defaults to the first input of the ``l3_features_v1`` sink.
    cached_full_node_values : dict[str, Any] | None
        A pre-computed mapping from ``node.id`` to the value yielded by
        ``_execute_l3_dag`` on the full panel. Used to skip recomputation
        of nodes outside ``affected_node_ids``. If ``None``, the function
        recomputes all upstream nodes from scratch.

    Returns
    -------
    pd.DataFrame
        The X panel (for the requested L3 sink) covering rows
        ``df.iloc[: origin_index + 1]`` with per-origin loadings.
    """

    if x_sink_node_id is None:
        sink_node = dag.nodes.get(dag.sinks.get("l3_features_v1", ""))
        if sink_node is None or not sink_node.inputs:
            raise ValueError(
                "materialize_l3_per_origin: dag missing l3_features_v1 sink"
            )
        x_sink_node_id = sink_node.inputs[0].node_id

    if origin_index < 0 or origin_index >= len(df):
        raise ValueError(
            f"materialize_l3_per_origin: origin_index {origin_index} out of range "
            f"[0, {len(df)})"
        )

    df_origin = df.iloc[: origin_index + 1]

    # Run a partial DAG walk: nodes inside ``affected_node_ids`` are
    # recomputed against ``df_origin``; nodes outside fall back to the
    # cached full-panel values trimmed to the same row range so the
    # downstream consumers see consistent index ranges.
    values: dict[str, Any] = {}
    affected = (
        set(affected_node_ids) if affected_node_ids is not None else set(dag.nodes)
    )

    def _trim_cached(value: Any) -> Any:
        # Trim the cached full-sample value to ``[: origin_index + 1]`` so
        # an unaffected upstream feeds an affected downstream a consistent
        # index range.
        if isinstance(value, pd.DataFrame):
            return value.iloc[: origin_index + 1]
        if isinstance(value, pd.Series):
            return value.iloc[: origin_index + 1]
        if isinstance(value, tuple):
            return tuple(_trim_cached(item) for item in value)
        return value

    for node in _topological_nodes(dag):
        if node.id in affected:
            if node.type == "source":
                values[node.id] = _execute_l3_source(
                    node.selector, df_origin, target_name
                )
            elif node.op == "l3_feature_bundle":
                values[node.id] = tuple(values[ref.node_id] for ref in node.inputs)
            elif node.op == "l3_metadata_build":
                values[node.id] = None
            else:
                inputs = [values[ref.node_id] for ref in node.inputs]
                values[node.id] = _execute_l3_op(
                    node.op, inputs, node.params, target_name
                )
        else:
            if (
                cached_full_node_values is not None
                and node.id in cached_full_node_values
            ):
                values[node.id] = _trim_cached(cached_full_node_values[node.id])
            else:
                if node.type == "source":
                    values[node.id] = _execute_l3_source(
                        node.selector, df_origin, target_name
                    )
                elif node.op == "l3_feature_bundle":
                    values[node.id] = tuple(values[ref.node_id] for ref in node.inputs)
                elif node.op == "l3_metadata_build":
                    values[node.id] = None
                else:
                    inputs = [values[ref.node_id] for ref in node.inputs]
                    values[node.id] = _execute_l3_op(
                        node.op, inputs, node.params, target_name
                    )

    return _as_frame(values[x_sink_node_id])


def _resolve_l0_seed(recipe_root: dict[str, Any]) -> int | None:
    """Mirror ``execution._resolve_seed`` so L4 estimator construction can
    inherit the L0 ``random_seed`` when a fit_model node does not pin its
    own ``params.random_state`` (issue #215)."""

    l0 = recipe_root.get("0_meta", {}) or {}
    leaf = l0.get("leaf_config", {}) or {}
    fixed = l0.get("fixed_axes", {}) or {}
    if "random_seed" in leaf:
        return int(leaf["random_seed"])
    repro = fixed.get("reproducibility_mode", "seeded_reproducible")
    return 0 if repro == "seeded_reproducible" else None


def materialize_l4_minimal(
    recipe_root: dict[str, Any], l3_features: L3FeaturesArtifact
) -> tuple[L4ForecastsArtifact, L4ModelArtifactsArtifact, L4TrainingMetadataArtifact]:
    raw = recipe_root.get("4_forecasting_model", {}) or {}
    report = l4_layer.validate_layer(
        raw, recipe_context={"horizon_set": set(l3_features.horizon_set)}
    )
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    fit_nodes = [
        node
        for node in raw.get("nodes", ()) or ()
        if isinstance(node, dict) and node.get("op") == "fit_model"
    ]
    if not fit_nodes:
        raise ValueError("[L4/4_forecasting_model.nodes] L4 runtime requires a fit_model node")  # Cycle 14 L1-1 fix:
    X = l3_features.X_final.data
    y = l3_features.y_final.metadata.values.get("data")
    if not isinstance(y, pd.Series):
        raise ValueError("[L4/4_forecasting_model.leaf_config] L4 runtime requires L3 y_final series data")  # Cycle 14 L1-1 fix:
    target = l3_features.y_final.name
    horizon = int(l3_features.horizon_set[0] if l3_features.horizon_set else 1)
    # Phase B-15 paper-15 F4: ``y_orig`` is the un-averaged source y stashed
    # by ``target_construction`` when L3 mode is ``cumulative_average`` /
    # ``path_average``. The L4 walk-forward consumes it when
    # ``forecast_strategy="path_average_eq4"`` to fit h per-horizon models.
    y_orig = l3_features.y_final.metadata.values.get("y_orig")
    l0_seed = _resolve_l0_seed(recipe_root)
    # Phase B-1 F2/F3 fix: pull the per-origin L3 rematerialization closure
    # set by ``materialize_l3_minimal`` when any L3 node carries
    # ``temporal_rule == expanding_window_per_origin``. The walk-forward
    # passes this through to ``_run_l4_fit_node``; when present, the loop
    # rebuilds X for every origin so factor loadings are fit only on
    # data <= origin (no future leakage).
    l3_per_origin_callable = l3_features.X_final.metadata.values.get(
        "l3_per_origin_callable"
    )
    forecasts: dict[tuple[str, str, int, Any], float] = {}
    artifacts: dict[str, ModelArtifact] = {}
    benchmark_flags: dict[str, bool] = {}
    refit_origins: dict[str, tuple[Any, ...]] = {}
    training_windows: dict[tuple[str, Any], tuple[Any, Any]] = {}
    model_ids: list[str] = []
    # Issues #204 + #250 -- sub-cell parallelism axis.
    #   ``parallel_unit = models``    -> fan out fit_nodes (#204).
    #   ``parallel_unit = oos_dates`` -> fan out the walk-forward origin
    #                                    loop inside each fit_node (#250).
    # ``horizons`` / ``targets`` are honoured at the schema level; the L4
    # runtime currently produces a single horizon / target per fit_node,
    # so those values map to the same execution as ``models`` for the
    # multi-fit-node case and ``oos_dates`` for the single-fit-node case.
    l0 = recipe_root.get("0_meta", {}) or {}
    parallel_unit = (l0.get("fixed_axes", {}) or {}).get("parallel_unit", "cells")
    n_workers = int((l0.get("leaf_config", {}) or {}).get("n_workers_inner", 4))
    if parallel_unit == "models" and len(fit_nodes) > 1:
        from concurrent.futures import ThreadPoolExecutor

        def _process_node(fit_node):
            return _run_l4_fit_node(
                fit_node,
                raw,
                X,
                y,
                target,
                horizon,
                l0_seed,
                parallel_origins=False,
                n_workers=n_workers,
                l3_per_origin_callable=l3_per_origin_callable,
                y_orig=y_orig,
            )

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_process_node, fit_nodes))
        for fit_node, result in zip(fit_nodes, results):
            model_id, model, node_forecasts, node_origins, node_windows = result
            forecasts.update(node_forecasts)
            artifacts[model_id] = model
            benchmark_flags[model_id] = bool(fit_node.get("is_benchmark", False))
            refit_origins[model_id] = tuple(node_origins)
            training_windows.update(node_windows)
            model_ids.append(model_id)
        sample_index = pd.DatetimeIndex(sorted({key[3] for key in forecasts}))
        return (
            L4ForecastsArtifact(
                forecasts=forecasts,
                forecast_object="point",
                sample_index=sample_index,
                targets=(target,),
                horizons=(horizon,),
                model_ids=tuple(model_ids),
                upstream_hashes={},
            ),
            L4ModelArtifactsArtifact(artifacts=artifacts, is_benchmark=benchmark_flags),
            L4TrainingMetadataArtifact(
                forecast_origins=tuple(sample_index),
                refit_origins=refit_origins,
                training_window_per_origin=training_windows,
            ),
        )
    if parallel_unit in {"oos_dates", "horizons", "targets"} and fit_nodes:
        # Issue #250 -- fan the walk-forward origin loop. Routes through
        # ``_run_l4_fit_node(parallel_origins=True)`` per fit_node.
        for fit_node in fit_nodes:
            model_id, model, node_forecasts, node_origins, node_windows = (
                _run_l4_fit_node(
                    fit_node,
                    raw,
                    X,
                    y,
                    target,
                    horizon,
                    l0_seed,
                    parallel_origins=True,
                    n_workers=n_workers,
                    l3_per_origin_callable=l3_per_origin_callable,
                    y_orig=y_orig,
                )
            )
            forecasts.update(node_forecasts)
            artifacts[model_id] = model
            benchmark_flags[model_id] = bool(fit_node.get("is_benchmark", False))
            refit_origins[model_id] = tuple(node_origins)
            training_windows.update(node_windows)
            model_ids.append(model_id)
        sample_index = pd.DatetimeIndex(sorted({key[3] for key in forecasts}))
        forecast_object = _resolve_forecast_object(fit_nodes)
        # Phase B-9 paper-9 F1+F2: pass artifacts so HNN-fitted models
        # (predict_quantiles / predict_distribution) become reachable from
        # the public path; add the explicit ``density`` branch.
        # Phase C M12: extract predict nodes so the Bai-Ng (2006) PI
        # correction (``pi_correction='bai_ng'``) reaches the quantile path.
        predict_nodes_parallel = [
            node
            for node in raw.get("nodes", ()) or ()
            if isinstance(node, dict) and node.get("op") == "predict"
        ]
        if forecast_object == "quantile":
            forecast_intervals = _emit_quantile_intervals(
                forecasts,
                fit_nodes,
                X=X,
                y=y,
                artifacts=artifacts,
                predict_nodes=predict_nodes_parallel,
            )
        elif forecast_object == "density":
            forecast_intervals = _emit_density_intervals(
                forecasts, fit_nodes, X=X, y=y, artifacts=artifacts
            )
        else:
            forecast_intervals = {}
        return (
            L4ForecastsArtifact(
                forecasts=forecasts,
                forecast_intervals=forecast_intervals,
                forecast_object=forecast_object,
                sample_index=sample_index,
                targets=(target,),
                horizons=(horizon,),
                model_ids=tuple(model_ids),
                upstream_hashes={},
            ),
            L4ModelArtifactsArtifact(artifacts=artifacts, is_benchmark=benchmark_flags),
            L4TrainingMetadataArtifact(
                forecast_origins=tuple(sample_index),
                refit_origins=refit_origins,
                training_window_per_origin=training_windows,
            ),
        )
    for fit_node in fit_nodes:
        params = dict(fit_node.get("params", {}) or {})
        if l0_seed is not None and "random_state" not in params:
            params["random_state"] = l0_seed
        family = params.get("family", "ridge")
        forecast_strategy = params.get("forecast_strategy", "direct")
        training_start_rule = params.get("training_start_rule", "expanding")
        refit_policy = params.get("refit_policy", "every_origin")
        rolling_window = int(
            params.get("rolling_window", max(24, min(len(X) // 2, 120)))
        )
        refit_step = (
            int(params.get("refit_step", 1)) if refit_policy == "every_n_origins" else 1
        )
        # tuning hook: dispatch on search_algorithm (issue #217). Inject the
        # L4 leaf_config so the resolver can read tuning_grid /
        # tuning_distributions / tuning_budget / cv_path_alphas / GA settings.
        # paper16-13 dispatch-gate fix (Round 1 phase A): include the four
        # Coulombe-Surprenant-Leroux-Stevanovic (2022 JAE) Feature 3 schemes
        # (kfold / poos / aic / bic) and Goulet Coulombe et al. (2024)
        # block_cv. Without these in the gate, `_resolve_l4_tuning` is never
        # invoked on the public path even though the resolver implements them.
        if params.get("search_algorithm") in {
            "cv_path",
            "grid_search",
            "random_search",
            "bayesian_optimization",
            "genetic_algorithm",
            "kfold",
            "poos",
            "aic",
            "bic",
            "block_cv",
        }:
            params["_l4_leaf_config"] = raw.get("leaf_config", {}) or {}
            params = _resolve_l4_tuning(params, X, y)
            params.pop("_l4_leaf_config", None)
        min_train_size = _minimal_train_size(
            params, n_obs=len(X), n_features=len(X.columns)
        )
        model_id = fit_node.get("id", "fit_model")
        model_ids.append(model_id)
        origins: list[Any] = []
        last_fit_position: int | None = None
        last_model = None
        for position in range(min_train_size, len(X)):
            origin = X.index[position]
            if training_start_rule == "rolling":
                start = max(0, position - rolling_window)
            elif training_start_rule == "fixed":
                start = 0
            else:
                start = 0
            # Phase B-1 F2/F3 fix: when an L3 node carries
            # ``temporal_rule = expanding_window_per_origin`` the sequential
            # walk-forward swaps the precomputed X (which was fit on the
            # full sample, leaking future observations into the factor
            # loadings) for a per-origin X computed only on data <= origin.
            X_origin_full = X
            if l3_per_origin_callable is not None:
                X_origin_full = l3_per_origin_callable(origin)
                # Align to the same columns the full-sample X carries.
                X_origin_full = X_origin_full.reindex(columns=X.columns, fill_value=0.0)
            position_in_origin = (
                X_origin_full.index.get_loc(origin)
                if l3_per_origin_callable is not None
                else position
            )
            # Phase B-1c h-step leak fix: ``y`` here is the cached
            # ``y_h = y_orig.shift(-h)`` (see ``target_construction`` op).
            # ``y.iloc[i]`` therefore equals ``y_orig.iloc[i + h]`` -- so
            # the naive ``y.iloc[start:position]`` slice would include
            # ``y_orig`` observations at calendar times
            # ``[start + h .. position - 1 + h]``, the trailing ``h - 1``
            # of which exceed the origin time. For h >= 2 this is a direct
            # post-origin leak. Paper-faithful walk-forward at origin
            # ``position`` admits only pairs whose ``y_orig`` date is
            # ``<= position``, i.e. rows ``s`` with ``s + h <= position``.
            # The leak-free end index is therefore ``position - h + 1``;
            # at h == 1 this is ``position`` (no-op), at h == 4 it drops
            # the trailing 3 rows. ``max(start, ...)`` guards against
            # empty / negative training sets at very early origins.
            train_end = max(start, position_in_origin - horizon + 1)
            train_X = X_origin_full.iloc[start:train_end]
            train_y_end = max(start, position - horizon + 1)
            train_y = y.iloc[start:train_y_end]
            should_refit = (
                last_model is None
                or refit_policy == "every_origin"
                or (
                    refit_policy == "every_n_origins"
                    and (
                        last_fit_position is None
                        or position - last_fit_position >= refit_step
                    )
                )
            )
            if should_refit and refit_policy != "single_fit":
                if forecast_strategy == "path_average_eq4" and isinstance(
                    y_orig, pd.Series
                ):
                    # Phase B-15 paper-15 F4: fit h per-horizon models and
                    # wrap them in a ``_PathAverageEq4Model``.
                    last_model = _fit_path_average_eq4_models(
                        family=family,
                        params=params,
                        train_X=train_X,
                        y_orig=y_orig,
                        train_X_index_end_position=train_end - 1,
                        horizon=horizon,
                        raw=raw,
                    )
                else:
                    last_model = _build_l4_model(family, params)
                    last_model.fit(train_X, train_y)
                last_fit_position = position
            elif refit_policy == "single_fit" and last_model is None:
                if forecast_strategy == "path_average_eq4" and isinstance(
                    y_orig, pd.Series
                ):
                    last_model = _fit_path_average_eq4_models(
                        family=family,
                        params=params,
                        train_X=train_X,
                        y_orig=y_orig,
                        train_X_index_end_position=train_end - 1,
                        horizon=horizon,
                        raw=raw,
                    )
                else:
                    last_model = _build_l4_model(family, params)
                    last_model.fit(train_X, train_y)
                last_fit_position = position
            forecast_value = _l4_predict_one(
                last_model,
                X_origin_full,
                position_in_origin,
                forecast_strategy=forecast_strategy,
                horizon=horizon,
            )
            forecasts[(model_id, target, horizon, origin)] = forecast_value
            origins.append(origin)
            training_windows[(model_id, origin)] = (train_X.index[0], train_X.index[-1])

        if forecast_strategy == "path_average_eq4" and isinstance(y_orig, pd.Series):
            # Full-sample artifact: also store the h per-horizon models.
            model = _fit_path_average_eq4_models(
                family=family,
                params=params,
                train_X=X,
                y_orig=y_orig,
                train_X_index_end_position=len(X) - 1,
                horizon=horizon,
                raw=raw,
            )
        else:
            model = _build_l4_model(family, params)
            model.fit(X, y)
        artifacts[model_id] = ModelArtifact(
            model_id=model_id,
            family=family,
            fitted_object=model,
            framework=_l4_framework(family),
            fit_metadata={
                "n_obs": len(X),
                "min_train_size": min_train_size,
                "runtime": f"{training_start_rule}_{forecast_strategy}",
                "refit_policy": refit_policy,
                "rolling_window": rolling_window,
                **{
                    k: params[k]
                    for k in ("alpha", "l1_ratio", "n_estimators", "max_depth", "C")
                    if k in params
                },
            },
            feature_names=tuple(X.columns),
        )
        benchmark_flags[model_id] = bool(fit_node.get("is_benchmark", False))
        refit_origins[model_id] = tuple(origins)

    sample_index = pd.DatetimeIndex(sorted({key[3] for key in forecasts}))
    forecast_object = _resolve_forecast_object(fit_nodes)
    forecast_intervals: dict[tuple[str, str, int, Any, float], float] = {}
    # Phase B-9 paper-9 F1+F2: pass artifacts (so HNN ``predict_quantiles``
    # is invoked instead of falling back to the LinearRegression
    # Gaussian-residual sigma) and add the ``density`` branch
    # (``predict_distribution``). Without these the paper's distributional
    # head is dead code from ``macroforecast.run``.
    # Phase C M12: extract predict-node params so the Bai-Ng (2006) PI
    # correction (``pi_correction='bai_ng'``) reaches the quantile path.
    predict_nodes = [
        node
        for node in raw.get("nodes", ()) or ()
        if isinstance(node, dict) and node.get("op") == "predict"
    ]
    if forecast_object == "quantile":
        forecast_intervals = _emit_quantile_intervals(
            forecasts,
            fit_nodes,
            X=X,
            y=y,
            artifacts=artifacts,
            predict_nodes=predict_nodes,
        )
    elif forecast_object == "density":
        forecast_intervals = _emit_density_intervals(
            forecasts, fit_nodes, X=X, y=y, artifacts=artifacts
        )
    return (
        L4ForecastsArtifact(
            forecasts=forecasts,
            forecast_intervals=forecast_intervals,
            forecast_object=forecast_object,
            sample_index=sample_index,
            targets=(target,),
            horizons=(horizon,),
            model_ids=tuple(model_ids),
            upstream_hashes={},
        ),
        L4ModelArtifactsArtifact(
            artifacts=artifacts,
            is_benchmark=benchmark_flags,
        ),
        L4TrainingMetadataArtifact(
            forecast_origins=tuple(sample_index),
            refit_origins=refit_origins,
            training_window_per_origin=training_windows,
        ),
    )


def _resolve_forecast_object(
    fit_nodes: list[dict[str, Any]],
) -> Literal["point", "quantile", "density"]:
    for node in fit_nodes:
        params = node.get("params", {}) or {}
        obj = params.get("forecast_object")
        if obj in {"quantile", "density"}:
            return obj
    return "point"


def _emit_quantile_intervals(
    forecasts: dict[tuple[str, str, int, Any], float],
    fit_nodes: list[dict[str, Any]],
    *,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    artifacts: dict[str, "ModelArtifact"] | None = None,
    predict_nodes: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, str, int, Any, float], float]:
    """Issue #246 -- emit quantile forecasts.

    When ``X`` / ``y`` are supplied and the family supports a native
    quantile estimator (``QuantileRegressor`` for linear,
    ``GradientBoostingRegressor(loss='quantile')`` for tree, native
    quantile loss for xgboost / lightgbm), fit one estimator per
    requested quantile level ``q`` and produce per-(origin, q) forecasts.
    Otherwise fall back to a Gaussian quantile expansion around the
    point forecast with the in-sample residual std (replaces the v0.2
    shortcut that used a leaf-config sigma).

    Phase B-9 paper-9 F1 fix (Goulet Coulombe / Frenette / Klieber 2025
    HNN): when a fit_node has produced an HNN-fitted estimator (i.e.
    one that exposes ``predict_quantiles``), call that method directly
    on the public path so the paper's Eq. 10 reality-checked variance
    drives the bands -- bypassing the LinearRegression Gaussian-residual
    fallback that would mask the paper's heteroscedastic head.
    """

    levels: list[float] = []
    family = "ridge"
    quantile_capable = False
    for node in fit_nodes:
        params = node.get("params", {}) or {}
        if params.get("forecast_object") in {"quantile", "density"}:
            levels = list(params.get("quantile_levels", [0.05, 0.25, 0.5, 0.75, 0.95]))
            family = str(params.get("family", "ridge"))
            quantile_capable = bool(params.get("forecast_object") == "quantile")
            break
    if not levels:
        return {}
    from scipy import stats as _stats  # type: ignore

    out: dict[tuple[str, str, int, Any, float], float] = {}

    # Phase B-9 paper-9 F1: HNN dispatch. Detect any fit_model whose
    # fitted estimator is an HNN (paper's mean + Eq. 10 reality-checked
    # variance bands). Restricted to ``_HemisphereNN`` instances so we
    # do not collide with the family-engine path used by QRF / Bagging
    # / other native quantile-capable wrappers (which already have
    # registered ``_native_quantile_engine`` factories). When found,
    # call ``predict_quantiles`` directly on X and skip the
    # LinearRegression Gaussian-residual fallback for that model. This
    # makes the paper's distributional output reachable from
    # ``macroforecast.run`` (Round-1 finding F1).
    hnn_models: dict[str, Any] = {}
    if artifacts is not None and isinstance(X, pd.DataFrame):
        for model_id, artifact in artifacts.items():
            fitted = getattr(artifact, "fitted_object", None)
            if fitted is None:
                continue
            if isinstance(fitted, _HemisphereNN):
                hnn_models[model_id] = fitted
    if hnn_models and isinstance(X, pd.DataFrame):
        try:
            X_filled = X.fillna(0.0)
            index = list(X_filled.index)
            for model_id, fitted in hnn_models.items():
                # Some legacy ``predict_quantiles`` (``_QuantileRegressionForest``)
                # do not accept a ``levels`` kwarg -- they read levels from
                # the model's own ``quantile_levels``. Try the kwarg form
                # first; fall back to the no-kwarg form.
                try:
                    bands = fitted.predict_quantiles(X_filled, levels=tuple(levels))
                except TypeError:
                    bands = fitted.predict_quantiles(X_filled)
                for (m_id, target, horizon, origin), _point in forecasts.items():
                    if m_id != model_id or origin not in index:
                        continue
                    i = index.index(origin)
                    for q in levels:
                        arr = bands.get(float(q))
                        if arr is None:
                            continue
                        out[(m_id, target, horizon, origin, float(q))] = float(arr[i])
            # Also emit non-HNN models via the family-engine path below.
            non_hnn_models = {
                k for k in {key[0] for key in forecasts} if k not in hnn_models
            }
            if not non_hnn_models:
                return out
            forecasts = {k: v for k, v in forecasts.items() if k[0] in non_hnn_models}
        except Exception:  # pragma: no cover - fall back to family engines
            out = {}

    quantile_engine = _native_quantile_engine(family) if quantile_capable else None
    if (
        quantile_engine is not None
        and isinstance(X, pd.DataFrame)
        and isinstance(y, pd.Series)
    ):
        try:
            X_filled = X.fillna(0.0)
            for q in levels:
                fitted = quantile_engine(q)
                fitted.fit(X_filled, y)
                preds = fitted.predict(X_filled)
                index = list(X_filled.index)
                for (model_id, target, horizon, origin), _point in forecasts.items():
                    if origin in index:
                        i = index.index(origin)
                        out[(model_id, target, horizon, origin, float(q))] = float(
                            preds[i]
                        )
            if out:
                return out
        except Exception:  # pragma: no cover - fall through to Gaussian shortcut
            pass

    # Gaussian-residual fallback. Use in-sample residual std rather than
    # the v0.2 leaf_config sigma when X / y are supplied.
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        try:
            from sklearn.linear_model import LinearRegression

            base = LinearRegression().fit(X.fillna(0.0), y)
            resid = y.values - base.predict(X.fillna(0.0))
            sigma = float(np.std(resid, ddof=1)) or 1.0
        except Exception:
            sigma = 1.0
    else:
        sigma = 1.0
        for node in fit_nodes:
            params = node.get("params", {}) or {}
            if params.get("forecast_object") in {"quantile", "density"}:
                sigma = float(params.get("forecast_residual_std", 1.0))
                break

    # Phase C M12 -- Bai-Ng (2006) generated-regressor PI correction.
    # When the predict node sets ``pi_correction='bai_ng'`` and the
    # fitted model is a ``_FactorAugmentedAR``, replace the per-model
    # sigma with the corrected ``√(σ²_ε + V₁/T + V₂/N)``. Otherwise
    # fall through to the standard Gaussian-residual sigma.
    pi_correction = "none"
    if predict_nodes:
        for node in predict_nodes:
            params = node.get("params", {}) or {}
            mode = params.get("pi_correction")
            if isinstance(mode, str) and mode != "none":
                pi_correction = mode
                break
    bai_ng_sigma_per_model: dict[str, float] = {}
    if (
        pi_correction == "bai_ng"
        and artifacts is not None
        and isinstance(X, pd.DataFrame)
        and isinstance(y, pd.Series)
    ):
        for model_id, artifact in artifacts.items():
            fitted = getattr(artifact, "fitted_object", None)
            if fitted is None or not isinstance(fitted, _FactorAugmentedAR):
                continue
            corrected = _bai_ng_pi_correction(fitted, X, y)
            if corrected is not None:
                bai_ng_sigma_per_model[model_id] = corrected

    for (model_id, target, horizon, origin), point in forecasts.items():
        sigma_used = bai_ng_sigma_per_model.get(model_id, sigma)
        for q in levels:
            out[(model_id, target, horizon, origin, float(q))] = float(
                point + sigma_used * _stats.norm.ppf(q)
            )
    return out


def _bai_ng_pi_correction(
    fitted: "_FactorAugmentedAR",
    X: pd.DataFrame,
    y: pd.Series,
) -> float | None:
    """Bai-Ng (2006) Theorem 3 + Corollary 1 PI correction sigma.

    Returns the corrected per-model σ such that bands of the form
    ``point ± z_{q} · σ`` reflect parameter-estimation noise (V₁/T) AND
    factor-estimation noise (V₂/N) in addition to the residual variance
    σ²_ε. Returns ``None`` when the fit object lacks the necessary
    diagnostics (e.g. fell back to a plain LinearRegression in the
    short-sample branch).

    Phase C-3 audit-fix (M12): the Bai-Ng (2006) Eq. (10) decomposition
    is ``Var(ŷ_{T+h}) = σ²_ε + V_1/T + V_2/N`` where V_1, V_2 are
    **O(1) asymptotic limits**. The previous implementation built
    inner ``V_1 = σ²·f_T'(D'D)⁻¹f_T`` (already O(1/T) since
    ``(D'D)⁻¹`` shrinks ∝ 1/T) and inner
    ``V_2 = β'·(Λ̂·diag(Σ̂_e)·Λ̂'/N)·β`` (already O(1/N)) and then
    divided by T and N **again**, yielding an O(1/T²) + O(1/N²)
    correction with width ratio 1.0002 = no-op.

    Path (a) chosen: build inner expressions that are themselves the
    full per-factor variance contribution (i.e. already include the
    1/T and 1/N scaling), and **omit the outer /T, /N**. This makes
    the formula honest to the literature (V_1/T = OLS predictive
    variance ``σ²·f_T'(D'D)⁻¹f_T`` evaluated at the last factor row;
    V_2/N = ``β'·(Λ̂·diag(Σ̂_e)·Λ̂'/N)·β``) while emitting a non-
    trivial correction.
    """

    Lambda_hat = fitted.factor_loadings_
    if Lambda_hat is None:
        return None
    idio = fitted.idiosyncratic_variance_
    if idio is None or idio.size == 0:
        return None
    beta_F = fitted.factor_coefficients_
    if beta_F.size == 0:
        return None
    # σ²_ε from in-sample residuals (or recompute on-X if absent).
    if fitted._residuals_train is not None and fitted._residuals_train.size > 1:
        sigma2_eps = float(np.var(fitted._residuals_train, ddof=1))
    else:
        try:
            preds = fitted.predict(X)
            sigma2_eps = float(np.var(np.asarray(y, dtype=float) - preds, ddof=1))
        except Exception:  # pragma: no cover
            return None

    # ``Lambda_hat`` from sklearn PCA has shape ``(K, N)`` (components_)
    # so ``shape[1]`` = N (number of features).
    N = max(int(Lambda_hat.shape[1]), 1)
    # Bai-Ng (2006) requires √T/N → 0 for the V_2/N correction to be
    # negligible relative to V_1/T; small-N regimes (N < 20) push the
    # asymptotic approximation outside its honest range.
    if N < 20:
        warnings.warn(
            f"Bai-Ng PI correction with N={N} < 20: small-N asymptotic "
            "regime (√T/N → 0) may be violated; use the corrected width "
            "with caution.",
            RuntimeWarning,
            stacklevel=2,
        )

    # V_2/N already (per Bai-Ng Theorem 3): the factor-estimation noise
    # contribution. Inner ``A`` already contains ``/N`` so we keep this
    # as the full V_2/N term (no outer /N).
    A = (Lambda_hat @ np.diag(idio) @ Lambda_hat.T) / max(N, 1)
    try:
        V_2_over_N = float(beta_F @ A @ beta_F)
    except Exception:  # pragma: no cover - shape mismatch
        return None
    V_2_over_N = max(V_2_over_N, 0.0)

    # V_1/T already (per Bai-Ng Theorem 3): the parameter-estimation
    # noise contribution. ``σ²·f_T'(D'D)⁻¹f_T`` is the standard OLS
    # predictive variance evaluated at the last-training factor row;
    # ``(D'D)⁻¹`` already shrinks ∝ 1/T as T grows, so this expression
    # *is* V_1/T (no outer /T).
    V_1_over_T = 0.0
    if fitted._design_train is not None:
        try:
            D = fitted._design_train
            DtD = D.T @ D
            DtD_inv = np.linalg.pinv(DtD)
            f_T = D[-1]
            V_1_over_T = float(sigma2_eps * (f_T @ DtD_inv @ f_T))
        except Exception:  # pragma: no cover - degenerate design
            V_1_over_T = 0.0

    pred_var = sigma2_eps + V_1_over_T + V_2_over_N
    pred_var = max(pred_var, 1e-12)
    return float(np.sqrt(pred_var))


# Sentinel float keys used in ``forecast_intervals`` for density forecasts.
# Real quantile levels live in (0, 1); these sentinels are negative so they
# never collide with a valid q ∈ {0.01..0.99} band. ``L6.E`` density tests
# read quantile bands by filtering keys with ``0 < q < 1``.
DENSITY_MEAN_KEY = -1.0
DENSITY_VARIANCE_KEY = -2.0


def _emit_density_intervals(
    forecasts: dict[tuple[str, str, int, Any], float],
    fit_nodes: list[dict[str, Any]],
    *,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    artifacts: dict[str, "ModelArtifact"] | None = None,
) -> dict[tuple[str, str, int, Any, float], float]:
    """Phase B-9 paper-9 F2 fix: emit density-forecast intervals.

    When ``forecast_object='density'`` and the fitted model is HNN
    (Goulet Coulombe / Frenette / Klieber 2025) -- detected via
    ``predict_distribution`` -- populate ``forecast_intervals`` with:

    * standard quantile bands at ``quantile_levels`` (the paper's
      Eq. 10 reality-checked Gaussian density), and
    * the per-row mean and variance, encoded under sentinel float
      keys ``DENSITY_MEAN_KEY`` (-1.0) and ``DENSITY_VARIANCE_KEY``
      (-2.0). Downstream consumers can recover the full ``N(μ, σ²)``
      density per origin without re-fitting.

    For non-HNN families the public density path is currently the
    same Gaussian quantile expansion as ``forecast_object='quantile'``
    (out of scope for paper 9; tracked separately).
    """

    levels: list[float] = []
    for node in fit_nodes:
        params = node.get("params", {}) or {}
        if params.get("forecast_object") == "density":
            levels = list(params.get("quantile_levels", [0.05, 0.16, 0.84, 0.95]))
            break
    if not levels:
        return {}

    out: dict[tuple[str, str, int, Any, float], float] = {}
    if artifacts is not None and isinstance(X, pd.DataFrame):
        X_filled = X.fillna(0.0)
        index = list(X_filled.index)
        density_models: dict[str, Any] = {}
        for model_id, artifact in artifacts.items():
            fitted = getattr(artifact, "fitted_object", None)
            if fitted is None:
                continue
            if isinstance(fitted, _HemisphereNN):
                density_models[model_id] = fitted
        if density_models:
            try:
                for model_id, fitted in density_models.items():
                    mean_arr, var_arr = fitted.predict_distribution(X_filled)
                    if hasattr(fitted, "predict_quantiles"):
                        try:
                            bands = fitted.predict_quantiles(
                                X_filled, levels=tuple(levels)
                            )
                        except TypeError:
                            bands = fitted.predict_quantiles(X_filled)
                    else:
                        bands = {}
                    for (m_id, target, horizon, origin), _point in forecasts.items():
                        if m_id != model_id or origin not in index:
                            continue
                        i = index.index(origin)
                        out[(m_id, target, horizon, origin, DENSITY_MEAN_KEY)] = float(
                            mean_arr[i]
                        )
                        out[(m_id, target, horizon, origin, DENSITY_VARIANCE_KEY)] = (
                            float(var_arr[i])
                        )
                        for q in levels:
                            arr = bands.get(float(q))
                            if arr is None:
                                continue
                            out[(m_id, target, horizon, origin, float(q))] = float(
                                arr[i]
                            )
                if out:
                    return out
            except Exception:  # pragma: no cover - fall back below
                out = {}

    # Fallback for non-HNN density: reuse the quantile path (Gaussian
    # residual expansion). Adds bands but no mean/variance sentinels.
    return _emit_quantile_intervals(
        forecasts,
        [
            {
                **n,
                "params": {
                    **(n.get("params", {}) or {}),
                    "forecast_object": "quantile",
                },
            }
            for n in fit_nodes
        ],
        X=X,
        y=y,
        artifacts=artifacts,
    )


def _native_quantile_engine(family: str):
    """Return a callable ``q -> sklearn-compatible estimator`` for families
    that have a native quantile-regression form. Returns ``None`` for
    families without one (caller falls back to Gaussian quantile)."""

    if family == "quantile_regression_forest":
        # Issue #280 -- QRF uses one fit and emits all quantiles at predict
        # time; expose a shim factory that ignores ``q`` and lets the caller
        # call ``predict_quantiles`` directly.
        def _factory(q: float):
            return _QuantileRegressionForest(quantile_levels=(float(q),))

        return _factory
    if family in {"ridge", "ols", "lasso", "elastic_net", "ar_p"}:
        try:
            from sklearn.linear_model import QuantileRegressor

            def _factory(q: float):
                return QuantileRegressor(quantile=float(q), alpha=0.0, solver="highs")

            return _factory
        except ImportError:  # pragma: no cover - sklearn ships this
            return None
    if family in {"gradient_boosting", "decision_tree", "random_forest", "extra_trees"}:
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            def _factory(q: float):
                return GradientBoostingRegressor(
                    loss="quantile", alpha=float(q), n_estimators=100
                )

            return _factory
        except ImportError:  # pragma: no cover
            return None
    if family == "xgboost":
        try:
            import xgboost as xgb  # type: ignore

            def _factory(q: float):
                return xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=float(q),
                    n_estimators=200,
                    tree_method="hist",
                )

            return _factory
        except ImportError:
            return None
    if family == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore

            def _factory(q: float):
                return lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=float(q),
                    n_estimators=200,
                )

            return _factory
        except ImportError:
            return None
    return None


def _run_l4_fit_node(
    fit_node: dict[str, Any],
    raw: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    horizon: int,
    l0_seed: int | None,
    *,
    parallel_origins: bool = False,
    n_workers: int = 4,
    l3_per_origin_callable: Any = None,
    y_orig: pd.Series | None = None,
) -> tuple[
    str,
    ModelArtifact,
    dict[tuple[str, str, int, Any], float],
    list[Any],
    dict[tuple[str, Any], tuple[Any, Any]],
]:
    """Run the same fit-loop the sequential path uses, but for a single
    ``fit_node``. Used by ``parallel_unit = models`` and (with
    ``parallel_origins = True``) ``parallel_unit = oos_dates``.

    Phase B-1 F2/F3 fix: ``l3_per_origin_callable`` -- when not ``None`` the
    walk-forward swaps the precomputed full-sample X for a per-origin X
    obtained by re-running the L3 sub-DAG on data <= origin only. This is
    the runtime path that finally honors L3's ``temporal_rule =
    expanding_window_per_origin`` schema convention.
    """

    params = dict(fit_node.get("params", {}) or {})
    if l0_seed is not None and "random_state" not in params:
        params["random_state"] = l0_seed
    family = params.get("family", "ridge")
    forecast_strategy = params.get("forecast_strategy", "direct")
    training_start_rule = params.get("training_start_rule", "expanding")
    refit_policy = params.get("refit_policy", "every_origin")
    rolling_window = int(params.get("rolling_window", max(24, min(len(X) // 2, 120))))
    refit_step = (
        int(params.get("refit_step", 1)) if refit_policy == "every_n_origins" else 1
    )
    # paper16-13 dispatch-gate fix (Round 1 phase A): see sequential gate above
    # for rationale. The parallel-models path uses the same set of recognised
    # `search_algorithm` values.
    if params.get("search_algorithm") in {
        "cv_path",
        "grid_search",
        "random_search",
        "bayesian_optimization",
        "genetic_algorithm",
        "kfold",
        "poos",
        "aic",
        "bic",
        "block_cv",
    }:
        params["_l4_leaf_config"] = raw.get("leaf_config", {}) or {}
        params = _resolve_l4_tuning(params, X, y)
        params.pop("_l4_leaf_config", None)
    min_train_size = _minimal_train_size(
        params, n_obs=len(X), n_features=len(X.columns)
    )
    model_id = fit_node.get("id", "fit_model")
    forecasts: dict[tuple[str, str, int, Any], float] = {}
    origins: list[Any] = []
    training_windows: dict[tuple[str, Any], tuple[Any, Any]] = {}

    def _origin_step(position: int) -> tuple[Any, float, tuple[Any, Any]]:
        origin = X.index[position]
        if training_start_rule == "rolling":
            start = max(0, position - rolling_window)
        else:
            start = 0
        # Phase B-1 F2/F3 fix: per-origin X swap (see docstring).
        X_origin = X
        position_in_origin = position
        if l3_per_origin_callable is not None:
            X_origin = l3_per_origin_callable(origin).reindex(columns=X.columns)
            position_in_origin = X_origin.index.get_loc(origin)
        # Phase B-1c h-step leak fix: see ``materialize_l4_minimal`` for
        # the full rationale. ``y`` is the cached ``y_h = y_orig.shift(-h)``
        # so the leak-free end index is ``position - h + 1``.
        train_end = max(start, position_in_origin - horizon + 1)
        train_X = X_origin.iloc[start:train_end]
        train_y_end = max(start, position - horizon + 1)
        train_y = y.iloc[start:train_y_end]
        # Issue #279 -- give each origin worker a deterministic per-origin
        # seed derived from the cell-level seed so thread interleaving
        # cannot affect the per-origin RandomForest / xgboost RNG state.
        per_origin_params = dict(params)
        if "random_state" in per_origin_params:
            per_origin_params["random_state"] = (
                int(per_origin_params["random_state"]) + position
            ) % (2**31 - 1)
        if forecast_strategy == "path_average_eq4" and isinstance(y_orig, pd.Series):
            # Phase B-15 paper-15 F4
            model = _fit_path_average_eq4_models(
                family=family,
                params=per_origin_params,
                train_X=train_X,
                y_orig=y_orig,
                train_X_index_end_position=train_end - 1,
                horizon=horizon,
                raw=raw,
            )
        else:
            model = _build_l4_model(family, per_origin_params)
            model.fit(train_X, train_y)
        forecast_value = _l4_predict_one(
            model,
            X_origin,
            position_in_origin,
            forecast_strategy=forecast_strategy,
            horizon=horizon,
        )
        return origin, forecast_value, (train_X.index[0], train_X.index[-1])

    if parallel_origins and refit_policy in {"every_origin", "every_n_origins"}:
        # Issue #250 -- fan the walk-forward origin loop across threads.
        from concurrent.futures import ThreadPoolExecutor

        positions = list(range(min_train_size, len(X)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for position, (origin, forecast_value, window) in zip(
                positions, pool.map(_origin_step, positions)
            ):
                forecasts[(model_id, target, horizon, origin)] = forecast_value
                origins.append(origin)
                training_windows[(model_id, origin)] = window
    else:
        last_fit_position: int | None = None
        last_model = None
        for position in range(min_train_size, len(X)):
            origin = X.index[position]
            if training_start_rule == "rolling":
                start = max(0, position - rolling_window)
            else:
                start = 0
            # Phase B-1 F2/F3 fix: per-origin X swap (see docstring).
            X_origin = X
            position_in_origin = position
            if l3_per_origin_callable is not None:
                X_origin = l3_per_origin_callable(origin).reindex(
                    columns=X.columns, fill_value=0.0
                )
                position_in_origin = X_origin.index.get_loc(origin)
            # Phase B-1c h-step leak fix: see ``materialize_l4_minimal``
            # for rationale. ``y`` is ``y_h = y_orig.shift(-h)`` and the
            # leak-free end index is ``position - h + 1``.
            train_end = max(start, position_in_origin - horizon + 1)
            train_X = X_origin.iloc[start:train_end]
            train_y_end = max(start, position - horizon + 1)
            train_y = y.iloc[start:train_y_end]
            should_refit = (
                last_model is None
                or refit_policy == "every_origin"
                or (
                    refit_policy == "every_n_origins"
                    and (
                        last_fit_position is None
                        or position - last_fit_position >= refit_step
                    )
                )
            )
            if should_refit and refit_policy != "single_fit":
                if forecast_strategy == "path_average_eq4" and isinstance(
                    y_orig, pd.Series
                ):
                    last_model = _fit_path_average_eq4_models(
                        family=family,
                        params=params,
                        train_X=train_X,
                        y_orig=y_orig,
                        train_X_index_end_position=train_end - 1,
                        horizon=horizon,
                        raw=raw,
                    )
                else:
                    last_model = _build_l4_model(family, params)
                    last_model.fit(train_X, train_y)
                last_fit_position = position
            elif refit_policy == "single_fit" and last_model is None:
                if forecast_strategy == "path_average_eq4" and isinstance(
                    y_orig, pd.Series
                ):
                    last_model = _fit_path_average_eq4_models(
                        family=family,
                        params=params,
                        train_X=train_X,
                        y_orig=y_orig,
                        train_X_index_end_position=train_end - 1,
                        horizon=horizon,
                        raw=raw,
                    )
                else:
                    last_model = _build_l4_model(family, params)
                    last_model.fit(train_X, train_y)
                last_fit_position = position
            forecast_value = _l4_predict_one(
                last_model,
                X_origin,
                position_in_origin,
                forecast_strategy=forecast_strategy,
                horizon=horizon,
            )
            forecasts[(model_id, target, horizon, origin)] = forecast_value
            origins.append(origin)
            training_windows[(model_id, origin)] = (train_X.index[0], train_X.index[-1])
    if forecast_strategy == "path_average_eq4" and isinstance(y_orig, pd.Series):
        full_model = _fit_path_average_eq4_models(
            family=family,
            params=params,
            train_X=X,
            y_orig=y_orig,
            train_X_index_end_position=len(X) - 1,
            horizon=horizon,
            raw=raw,
        )
    else:
        full_model = _build_l4_model(family, params)
        full_model.fit(X, y)
    artifact = ModelArtifact(
        model_id=model_id,
        family=family,
        fitted_object=full_model,
        framework=_l4_framework(family),
        fit_metadata={
            "n_obs": len(X),
            "min_train_size": min_train_size,
            "runtime": f"{training_start_rule}_{forecast_strategy}",
            "refit_policy": refit_policy,
            "rolling_window": rolling_window,
            **{
                k: params[k]
                for k in ("alpha", "l1_ratio", "n_estimators", "max_depth", "C")
                if k in params
            },
        },
        feature_names=tuple(X.columns),
    )
    return model_id, artifact, forecasts, origins, training_windows


def _l4_framework(family: str) -> str:
    if family in {"xgboost", "lightgbm", "catboost"}:
        return family
    if _resolve_custom_model(family) is not None:
        return "custom"
    return "sklearn"


def _build_l4_model(family: str, params: dict[str, Any]):
    """Build an estimator instance for one of the operational L4 families.

    Linear/tree/SVM/knn/boosting families are wired here; heavyweight models
    (BVAR, MRF, neural nets, MIDAS) raise NotImplementedError with a clear
    message so the recipe author knows which family is unavailable.
    """

    alpha = float(params.get("alpha", 1.0))
    seed = int(params.get("random_state", 0))
    if family == "ols":
        return LinearRegression()
    if family == "ridge":
        # v0.9 sub-axis dispatch (decomposition discipline). The default
        # behaviour is unchanged; non-default values dispatch to specialised
        # wrappers (operational) or raise NotImplementedError for promotions
        # still pending.
        prior = params.get("prior", "none")
        constraint = params.get("coefficient_constraint", "none")
        if prior == "random_walk":
            # Coulombe (2025 IJF) "Time-Varying Parameters as Ridge".
            # Phase B-8 audit-fix: paper §2.5 Algorithm 1 step 4 calls
            # for a second λ-CV after the warm-start; default routes
            # ``alpha_strategy="second_cv"``. Set ``alpha_strategy=
            # "fixed"`` to bypass the CV and use ``alpha`` as-is.
            return _TwoStageRandomWalkRidge(
                alpha=alpha,
                vol_model=str(params.get("vol_model", "garch11")),
                max_alpha_ratio=float(params.get("max_alpha_ratio", 1e6)),
                alpha_strategy=str(params.get("alpha_strategy", "second_cv")),
                alpha_grid=params.get("alpha_grid"),
                cv_folds=int(params.get("cv_folds", 5)),
                random_state=seed,
            )
        if prior == "shrink_to_target":
            # Goulet Coulombe et al. "Maximally Forward-Looking Core
            # Inflation" Albacore_comps (Variant A): non-negative ridge
            # with a shrink-to-target penalty + simplex constraint.
            return _ShrinkToTargetRidge(
                alpha=alpha,
                prior_target=params.get("prior_target"),
                simplex=bool(params.get("prior_simplex", True)),
                nonneg=(constraint == "nonneg"),
            )
        if prior == "fused_difference":
            # Maximally FL Albacore_ranks (Variant B): fused-difference
            # penalty over rank-position weights + optional mean-equality.
            return _FusedDifferenceRidge(
                alpha=alpha,
                difference_order=int(params.get("prior_diff_order", 1)),
                mean_equality=bool(params.get("prior_mean_equality", True)),
                nonneg=(constraint == "nonneg"),
            )
        if prior != "none":
            raise NotImplementedError(
                f"ridge.prior={prior!r} is schema-only in v0.9.0; "
                f"supported: 'none' (default), 'random_walk' (2SRR), "
                f"'shrink_to_target' (Maximally FL Albacore_comps), "
                f"'fused_difference' (Maximally FL Albacore_ranks)."
            )
        if constraint == "nonneg":
            return _NonNegRidge(alpha=alpha)
        if constraint != "none":
            raise NotImplementedError(
                f"ridge.coefficient_constraint={constraint!r} is schema-only "
                f"in v0.9.0; supported values: 'none' (default), 'nonneg' "
                f"(operational, Coulombe et al. 2024 Assemblage Regression)."
            )
        return Ridge(alpha=alpha)
    if family == "lasso":
        return Lasso(alpha=alpha, max_iter=int(params.get("max_iter", 20000)))
    if family == "elastic_net":
        return ElasticNet(
            alpha=alpha,
            l1_ratio=float(params.get("l1_ratio", params.get("lambda1_ratio", 0.5))),
            max_iter=int(params.get("max_iter", 20000)),
        )
    if family == "lasso_path":
        return LassoCV(
            cv=int(params.get("cv", 5)),
            max_iter=int(params.get("max_iter", 20000)),
            random_state=seed,
        )
    if family == "bayesian_ridge":
        return BayesianRidge()
    if family == "ar_p":
        return _LinearARModel(p=int(params.get("n_lag", params.get("p", 1))))
    if family == "factor_augmented_ar":
        return _FactorAugmentedAR(
            p=int(params.get("n_lag", 1)), n_factors=int(params.get("n_factors", 3))
        )
    if family == "factor_augmented_var":
        return _FactorAugmentedVAR(
            p=int(params.get("n_lag", 2)), n_factors=int(params.get("n_factors", 3))
        )
    if family == "principal_component_regression":
        return _PrincipalComponentRegression(
            n_components=int(params.get("n_components", 4))
        )
    if family == "decision_tree":
        split_shrinkage = float(params.get("split_shrinkage", 0.0))
        if split_shrinkage != 0.0:
            # Goulet Coulombe (2024) Slow-Growing Trees (SLOTH). Custom
            # soft-weighted tree: at each split, observations not
            # satisfying the rule receive weight (1 − η) instead of 0.
            # Operational v0.9.1 dev-stage v0.9.0B-6; v0.9.0F audit-fix
            # restored the paper p.87 rule-of-thumb defaults
            # (eta_depth_step=0.01, eta_max_plateau=0.5) and surfaced
            # the mtry sub-axis (paper p.88 §2.3).
            return _SlowGrowingTree(
                eta=split_shrinkage,
                herfindahl_threshold=float(params.get("herfindahl_threshold", 0.25)),
                eta_depth_step=float(params.get("eta_depth_step", 0.01)),
                eta_max_plateau=float(params.get("eta_max_plateau", 0.5)),
                mtry_frac=float(params.get("mtry_frac", 0.75)),
                max_depth=params.get("max_depth"),
                random_state=seed,
            )
        return DecisionTreeRegressor(
            max_depth=params.get("max_depth"), random_state=seed
        )
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            random_state=seed,
            n_jobs=1,
        )
    if family == "extra_trees":
        # max_features: sklearn accepts int, float, "sqrt", "log2", or None.
        # Coulombe (2024) "To Bag is to Prune" PRF baseline = extra_trees
        # with max_features=1 (one random feature per split, fully random).
        max_features = params.get("max_features", "sqrt")
        if isinstance(max_features, str) and max_features.isdigit():
            max_features = int(max_features)
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            max_features=max_features,
            random_state=seed,
            n_jobs=1,
        )
    if family == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=seed,
        )
    if family == "xgboost":
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise NotImplementedError(
                "xgboost family requires `pip install macroforecast[xgboost]`"
            ) from exc
        return xgb.XGBRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 6)),
            random_state=seed,
            tree_method="hist",
            verbosity=0,
        )
    if family == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise NotImplementedError(
                "lightgbm family requires `pip install macroforecast[lightgbm]`"
            ) from exc
        return lgb.LGBMRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", -1)),
            random_state=seed,
            verbosity=-1,
        )
    if family == "catboost":
        try:
            from catboost import CatBoostRegressor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise NotImplementedError(
                "catboost family requires `pip install macroforecast[catboost]`"
            ) from exc
        return CatBoostRegressor(
            iterations=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            depth=int(params.get("max_depth", 6)),
            random_seed=seed,
            verbose=False,
        )
    if family in {"svr_linear", "svr_rbf", "svr_poly"}:
        kernel = {"svr_linear": "linear", "svr_rbf": "rbf", "svr_poly": "poly"}[family]
        return SVR(
            kernel=kernel,
            C=float(params.get("C", 1.0)),
            epsilon=float(params.get("epsilon", 0.1)),
            gamma=params.get("gamma", "scale"),
        )
    if family == "kernel_ridge":
        # v0.9.0F audit-fix: paper 16 (Coulombe et al. 2022 JAE "How is
        # Machine Learning Useful for Macroeconomic Forecasting?") uses
        # Kernel Ridge Regression as a headline non-linearity feature
        # (paper §3.1.1, Eq. 16). Surfaced as a first-class L4 family
        # via sklearn ``KernelRidge``. ``params.kernel`` chooses the
        # kernel ('rbf' default per paper); ``alpha`` is the ridge
        # penalty; ``gamma`` is the RBF bandwidth.
        from sklearn.kernel_ridge import KernelRidge

        return KernelRidge(
            alpha=float(params.get("alpha", 1.0)),
            kernel=str(params.get("kernel", "rbf")),
            gamma=params.get("gamma"),
            degree=int(params.get("degree", 3)),
            coef0=float(params.get("coef0", 1.0)),
        )
    if family == "knn":
        n_neighbors = int(params.get("n_neighbors", 5))
        return _AutoClipKNN(
            n_neighbors=n_neighbors, weights=params.get("weights", "uniform")
        )
    if family == "huber":
        from sklearn.linear_model import HuberRegressor

        return HuberRegressor(
            epsilon=float(params.get("epsilon", 1.35)),
            max_iter=int(params.get("max_iter", 1000)),
        )
    if family == "var":
        return _VARWrapper(p=int(params.get("n_lag", 1)))
    if family == "glmboost":
        return _GLMBoost(
            n_iter=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
        )
    if family in {"bvar_minnesota", "bvar_normal_inverse_wishart"}:
        # Phase B-4 F2 backward-compat: accept legacy ``n_lags`` plural
        # alias with a deprecation warning. Canonical key is ``n_lag``.
        n_lag_canonical = params.get("n_lag")
        n_lag_legacy = params.get("n_lags")
        if n_lag_canonical is None and n_lag_legacy is not None:
            warnings.warn(
                "bvar_minnesota: 'n_lags' is deprecated; use 'n_lag' "
                "(singular). Routing legacy 'n_lags' to 'n_lag'.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_lag_canonical = n_lag_legacy
        elif (
            n_lag_canonical is not None
            and n_lag_legacy is not None
            and int(n_lag_legacy) != int(n_lag_canonical)
        ):
            raise ValueError(
                f"bvar_minnesota: conflicting n_lag={n_lag_canonical} "
                f"and n_lags={n_lag_legacy}; pass exactly one (prefer n_lag)."
            )
        if n_lag_canonical is None:
            n_lag_canonical = 2  # legacy default for non-arctic callers
        # Phase B-4 F4: paper Appx-A.3 VARCTIC 8 hyperparameter aliases.
        # New paper-faithful keys (``b_AR``, ``lambda_1``, ``lambda_cross``,
        # ``lambda_decay``) take precedence over the v0.9.0a0 legacy keys
        # (``minnesota_lambda1``, ``minnesota_lambda_decay``,
        # ``minnesota_lambda_cross``) when both are present.
        lambda1 = float(params.get("lambda_1", params.get("minnesota_lambda1", 0.2)))
        lambda_decay = float(
            params.get("lambda_decay", params.get("minnesota_lambda_decay", 1.0))
        )
        lambda_cross = float(
            params.get("lambda_cross", params.get("minnesota_lambda_cross", 0.5))
        )
        b_AR = float(params.get("b_AR", 0.9))
        ordering = params.get("ordering")
        if ordering is not None:
            ordering = tuple(str(x) for x in ordering)
        return _BayesianVAR(
            p=int(n_lag_canonical),
            prior=family,
            lambda1=lambda1,
            lambda_decay=lambda_decay,
            lambda_cross=lambda_cross,
            b_AR=b_AR,
            n_draws=int(params.get("n_posterior_draws", 0)),
            posterior_irf_periods=int(params.get("posterior_irf_periods", 12)),
            ordering=ordering,
            random_state=seed,
        )
    if family == "macroeconomic_random_forest":
        # Coulombe (2024) MRF: random walk regularised forest with per-leaf
        # local linear regressions and Block Bayesian Bootstrap forecast
        # ensembles. Backed by Ryan Lucas's reference implementation
        # vendored under ``_vendor/macro_random_forest/`` — see
        # ``_MRFExternalWrapper``.
        if "B" in params and "n_estimators" in params:
            warnings.warn(
                "MRF recipe has both 'B' and 'n_estimators'; 'B' wins. "
                "Use 'B' exclusively to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        return _MRFExternalWrapper(
            B=int(params.get("B", params.get("n_estimators", 50))),
            ridge_lambda=float(params.get("ridge_lambda", 0.1)),
            rw_regul=float(params.get("rw_regul", 0.75)),
            mtry_frac=float(params.get("mtry_frac", 1 / 3)),
            trend_push=float(params.get("trend_push", 1)),
            quantile_rate=float(params.get("quantile_rate", 0.3)),
            subsampling_rate=float(params.get("subsampling_rate", 0.75)),
            fast_rw=bool(params.get("fast_rw", True)),
            resampling_opt=int(params.get("resampling_opt", 2)),
            parallelise=bool(params.get("parallelise", False)),
            n_cores=int(params.get("n_cores", 1)),
            block_size=int(params.get("block_size", 24)),
            random_state=seed,
        )
    if family in {"mlp", "lstm", "gru", "transformer"}:
        # v0.9 sub-axis: architecture={standard, hemisphere} + loss={mse,
        # volatility_emphasis}. Hemisphere arch + volatility-emphasis
        # loss = HNN (Coulombe / Frenette / Klieber 2025 JAE), promoted
        # in v0.9.1 dev-stage v0.9.0C-2; supported on the ``mlp`` family
        # only (LSTM/GRU/Transformer extensions deferred).
        nn_arch = params.get("architecture", "standard")
        nn_loss = params.get("loss", "mse")
        if nn_arch == "hemisphere" or nn_loss == "volatility_emphasis":
            if family != "mlp":
                raise NotImplementedError(
                    f"{family}.architecture=hemisphere / loss=volatility_emphasis "
                    f"requires family='mlp' in v0.9.1; LSTM/GRU/Transformer "
                    f"hemisphere variants deferred to v0.9.x."
                )
            if nn_arch != "hemisphere" or nn_loss != "volatility_emphasis":
                raise NotImplementedError(
                    "HNN requires both architecture='hemisphere' and "
                    "loss='volatility_emphasis' (the two are paper-coupled). "
                    "Set both, or set neither."
                )
            return _HemisphereNN(
                lc=int(params.get("lc", 2)),
                lm=int(params.get("lm", 2)),
                lv=int(params.get("lv", 2)),
                neurons=int(params.get("neurons", 64)),
                dropout=float(params.get("dropout", 0.2)),
                lr=float(params.get("lr", 1e-3)),
                n_epochs=int(params.get("n_epochs", 100)),
                B=int(params.get("B", 100)),
                sub_rate=float(params.get("sub_rate", 0.80)),
                nu=params.get("nu"),
                lambda_emphasis=float(params.get("lambda_emphasis", 1.0)),
                patience=int(params.get("patience", 15)),
                val_frac=float(params.get("val_frac", 0.20)),
                random_state=seed,
            )
        if family == "mlp":
            from sklearn.neural_network import MLPRegressor

            return MLPRegressor(
                hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (32, 16))),
                max_iter=int(params.get("max_iter", 500)),
                random_state=seed,
            )
        return _TorchSequenceModel(
            kind=family,
            hidden_size=int(params.get("hidden_size", 32)),
            n_epochs=int(params.get("n_epochs", 50)),
            random_state=seed,
        )
    if family == "dfm_mixed_mariano_murasawa":
        return _DFMMixedFrequency(
            n_factors=int(params.get("n_factors", 1)),
            factor_order=int(params.get("factor_order", 1)),
            mixed_frequency=bool(params.get("mixed_frequency", False)),
            column_frequencies=params.get("column_frequencies"),
        )
    if family == "quantile_regression_forest":
        # Issue #280 -- Meinshausen (2006).
        return _QuantileRegressionForest(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            random_state=seed,
            quantile_levels=tuple(params.get("quantile_levels", (0.05, 0.5, 0.95))),
        )
    if family == "mars":
        # Friedman (1991) MARS via pyearth optional dep. v0.9 Phase 2
        # paper-coverage atomic addition; provides the base learner for
        # Coulombe (2024) MARSquake recipe.
        try:
            from pyearth import Earth  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise NotImplementedError(
                "mars family requires `pip install macroforecast[mars]`"
            ) from exc
        return Earth(
            max_terms=int(params.get("max_terms", 21)),
            max_degree=int(params.get("max_degree", 1)),
        )
    if family == "bagging":
        # Issue #282 -- generic bootstrap-aggregating meta-estimator.
        # v0.9 sub-axis: strategy={standard, block, booging, sequential_residual}.
        # standard (default) = Breiman 1996; block = Kunsch 1989 moving-block;
        # booging = Goulet Coulombe 2024 outer-bagging-of-over-fitted-inner-SGB
        # + Data Augmentation (operational v0.9.1 dev-stage v0.9.0B);
        # sequential_residual = legacy alias for booging (back-compat).
        strategy = params.get("strategy", "standard")
        if strategy in {"booging", "sequential_residual"}:
            return _BoogingWrapper(
                B=int(params.get("n_estimators", 100)),
                sample_frac=float(params.get("max_samples", 0.75)),
                # v0.9.0F audit-fix: paper Table 2 / Appendix B uses S = 1500;
                # earlier default 500 was below the "deliberately overfit"
                # regime that the bag-prune theorem relies on.
                inner_n_estimators=int(params.get("inner_n_estimators", 1500)),
                inner_learning_rate=float(params.get("inner_learning_rate", 0.1)),
                inner_max_depth=int(params.get("inner_max_depth", 3)),
                inner_subsample=float(params.get("inner_subsample", 0.5)),
                da_noise_frac=float(params.get("da_noise_frac", 1.0 / 3.0)),
                da_drop_rate=float(params.get("da_drop_rate", 0.2)),
                random_state=seed,
            )
        if strategy not in {"standard", "block"}:
            raise NotImplementedError(
                f"bagging.strategy={strategy!r} not supported. Valid: "
                f"'standard' (default), 'block', 'booging' (or alias "
                f"'sequential_residual')."
            )
        return _BaggingWrapper(
            base_family=str(params.get("base_family", "ridge")),
            n_estimators=int(params.get("n_estimators", 50)),
            max_samples=float(params.get("max_samples", 0.8)),
            random_state=seed,
            strategy=strategy,
            block_length=int(params.get("block_length", 4)),
            base_params={
                k: v
                for k, v in params.items()
                if k
                not in {
                    "family",
                    "base_family",
                    "n_estimators",
                    "max_samples",
                    "strategy",
                    "block_length",
                }
            },
        )
    if family == "garch11":
        return _GARCHFamily(
            variant="garch",
            p=int(params.get("p", 1)),
            q=int(params.get("q", 1)),
            mean_model=str(params.get("mean_model", "constant")),
            dist=str(params.get("dist", "normal")),
            rescale=bool(params.get("rescale", False)),
            random_state=seed,
        )
    if family == "egarch":
        return _GARCHFamily(
            variant="egarch",
            p=int(params.get("p", 1)),
            o=int(params.get("o", 1)),
            q=int(params.get("q", 1)),
            mean_model=str(params.get("mean_model", "constant")),
            dist=str(params.get("dist", "normal")),
            rescale=bool(params.get("rescale", False)),
            random_state=seed,
        )
    if family == "realized_garch_with_rv_exog":
        # Phase C-3 audit-fix (M9): honest rename. The runtime feeds
        # the realised-variance series as the exogenous ``x=`` regressor
        # in a vanilla GARCH(1,1) -- useful in practice but **NOT** the
        # Hansen-Huang-Shek (2012) joint return + measurement-equation
        # MLE. The proper RealizedGARCH spec is reserved as the FUTURE
        # name ``realized_garch``.
        return _GARCHFamily(
            variant="realized_garch",
            mean_model=str(params.get("mean_model", "constant")),
            dist=str(params.get("dist", "normal")),
            rescale=bool(params.get("rescale", False)),
            realized_variance=params.get("realized_variance"),
            random_state=seed,
        )
    if family == "ets":
        return _ETSWrapper(
            error_trend_seasonal=str(params.get("error_trend_seasonal", "AAN")),
            seasonal_periods=int(params.get("seasonal_periods", 12)),
            damped_trend=bool(params.get("damped_trend", False)),
            initialization_method=str(params.get("initialization_method", "estimated")),
            random_state=seed,
        )
    if family == "theta_method":
        return _ThetaWrapper(
            theta=float(params.get("theta", 2.0)),
            seasonal=bool(params.get("seasonal", False)),
            seasonal_periods=int(params.get("seasonal_periods", 12)),
        )
    if family == "holt_winters":
        return _HoltWintersWrapper(
            seasonal=params.get("seasonal", "add"),
            seasonal_periods=int(params.get("seasonal_periods", 12)),
            trend=params.get("trend", "add"),
            damped_trend=bool(params.get("damped_trend", False)),
        )
    custom = _resolve_custom_model(family)
    if custom is not None:
        return _CustomModelAdapter(custom, params=params)
    raise NotImplementedError(f"L4 runtime does not support family={family!r}")


def _resolve_custom_model(family: str):
    """Look up ``macroforecast.custom`` for a user-registered model factory; the
    runtime then wraps it in a sklearn-compatible adapter (issue #216)."""

    try:
        from .. import custom as _custom_mod
    except ImportError:  # pragma: no cover
        return None
    if not _custom_mod.is_custom_model(family):
        return None
    return _custom_mod.get_custom_model(family)


class _CustomModelAdapter:
    """Wrap a ``register_model`` callable in the sklearn fit/predict API.

    The user contract is::

        fn(X_train, y_train, X_test, context) -> scalar | one-element array

    For ``predict(X)`` we iterate row-by-row to mirror the documented
    1-row ``X_test`` contract; this matches the ``_l4_predict_one`` flow
    and keeps every recipe path working unchanged."""

    def __init__(self, spec, params: dict[str, Any]) -> None:
        self.spec = spec
        self.params = dict(params or {})
        self._train_X: pd.DataFrame | None = None
        self._train_y: pd.Series | None = None

    def fit(self, X, y):
        self._train_X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._train_y = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
        return self

    def predict(self, X):
        if self._train_X is None or self._train_y is None:
            raise RuntimeError(
                f"custom model {self.spec.name!r} predict() called before fit()"
            )
        if isinstance(X, pd.DataFrame):
            test = X
        else:
            test = pd.DataFrame(X)
        context = {
            "contract_version": "custom_model_v1",
            "model_name": self.spec.name,
            "feature_names": tuple(self._train_X.columns),
            "params": dict(self.params),
        }
        preds: list[float] = []
        for _, row in test.iterrows():
            row_frame = pd.DataFrame([row], columns=test.columns)
            value = self.spec.function(self._train_X, self._train_y, row_frame, context)
            if hasattr(value, "__len__") and not isinstance(value, str):
                preds.append(float(list(value)[0]))
            else:
                preds.append(float(value))
        return np.asarray(preds, dtype=float)


class _PathAverageEq4Model:
    """Phase B-15 paper-15 F4: paper Eq. 4 path-average estimator wrapper.

    Holds ``h`` per-horizon sklearn-compatible regressors. The ``predict``
    method averages predictions across the ``h`` sub-models so the rest of
    the L4 walk-forward sees a single ``model`` object.

    Attributes
    ----------
    models : list
        The h per-horizon fitted regressors.
    horizon : int
        The maximum horizon h.
    alphas : tuple[float, ...]
        Per-horizon regularisation parameters (when present).
    """

    def __init__(self, models, *, horizon: int):
        self.models = list(models)
        self.horizon = int(horizon)

    @property
    def alphas(self) -> tuple[float, ...]:
        return tuple(
            float(getattr(m, "alpha", getattr(m, "alpha_", float("nan"))))
            for m in self.models
        )

    def predict(self, X):
        preds = [np.asarray(m.predict(X), dtype=float) for m in self.models]
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0)


def _fit_path_average_eq4_models(
    *,
    family: str,
    params: dict[str, Any],
    train_X: pd.DataFrame,
    y_orig: pd.Series,
    train_X_index_end_position: int,
    horizon: int,
    raw: dict[str, Any],
) -> _PathAverageEq4Model:
    """Phase B-15 paper-15 F4: fit h per-horizon models on shifted targets.

    For ``h' = 1..horizon`` build ``y_h'(t) = y_orig.shift(-h').iloc[t]`` and
    fit one ``_build_l4_model`` instance per horizon on ``(train_X, y_h')``,
    each with its own CV-tuned ``λ_{h'}`` (when ``search_algorithm`` is set).
    The leak-free training slice for horizon ``h'`` is the rows whose
    ``y_orig`` date is ``<= position`` — i.e. the first ``len(train_X) -
    (h' - 1)`` rows of ``train_X`` after the cumulative-average leak guard
    has already trimmed the tail by ``horizon - 1``.

    Returns a ``_PathAverageEq4Model`` whose ``.predict`` averages the h
    sub-model predictions (paper Eq. 4).
    """

    sub_models = []
    # ``train_X`` has been trimmed to rows whose path-average target date is
    # ``<= position`` -- i.e. rows ``s`` with ``s + horizon <= position``.
    # For per-horizon target ``h' < horizon`` the leak-free admission set is
    # rows ``s`` with ``s + h' <= position``, which is a longer slice.
    # ``train_X.index[-1]`` corresponds to position ``train_X_index_end_position``
    # in the underlying X. We rebuild each per-horizon train slice from
    # ``y_orig`` directly.
    train_start_pos = train_X_index_end_position - len(train_X) + 1
    if train_start_pos < 0:
        train_start_pos = 0
    for h_prime in range(1, horizon + 1):
        # leak-free end (exclusive) for horizon h': rows s with s + h' <= position
        end_pos = train_X_index_end_position - h_prime + 1
        if end_pos <= train_start_pos:
            # too few obs for this horizon — degrade by reusing the
            # path-average trimmed slice (last fallback so the run does not
            # crash on tiny T).
            sub_X = train_X
            # build per-horizon y on the same index range
            y_h_full = y_orig.shift(-h_prime)
            sub_y = y_h_full.iloc[
                train_start_pos : train_start_pos + len(sub_X)
            ].dropna()
            if len(sub_y) == 0:
                sub_y = y_orig.iloc[: len(sub_X)]
            sub_X = sub_X.iloc[: len(sub_y)]
        else:
            # Slice X to the (start, end_pos) window expressed as positions
            # within ``train_X``.
            local_end = end_pos - train_start_pos
            # ``train_X`` itself was sliced from the parent X; its first row
            # corresponds to ``train_start_pos``. We can extend backwards
            # only up to ``train_X``'s own start; for h' < horizon the
            # leak-free slice is *longer* than ``train_X``, but we take
            # ``train_X`` as the upper bound (parent caller already enforces
            # the ``start`` lower bound). ``local_end`` clipped to ``len(train_X)``.
            local_end = min(local_end, len(train_X))
            sub_X = train_X.iloc[:local_end]
            y_h_full = y_orig.shift(-h_prime)
            sub_y = y_h_full.iloc[train_start_pos : train_start_pos + local_end]
        # Drop trailing NaNs (shift introduces them at the tail).
        valid = sub_y.notna()
        sub_X = sub_X.loc[valid.values] if len(sub_X) == len(valid) else sub_X
        sub_y = sub_y.loc[valid]
        if len(sub_X) == 0 or len(sub_y) == 0:
            # Degenerate slice -- skip per-horizon fit; downstream predict
            # will just average the remaining models.
            continue
        # Per-horizon λ tuning: each h' gets its own search.
        per_h_params = dict(params)
        # Ensure the CV gate fires per-horizon as well.
        if per_h_params.get("search_algorithm") in {
            "cv_path",
            "grid_search",
            "random_search",
            "bayesian_optimization",
            "genetic_algorithm",
            "kfold",
            "poos",
            "aic",
            "bic",
            "block_cv",
        }:
            per_h_params["_l4_leaf_config"] = raw.get("leaf_config", {}) or {}
            per_h_params = _resolve_l4_tuning(per_h_params, sub_X, sub_y)
            per_h_params.pop("_l4_leaf_config", None)
        sub_model = _build_l4_model(family, per_h_params)
        sub_model.fit(sub_X, sub_y)
        sub_models.append(sub_model)
    if not sub_models:
        # Degenerate fallback -- fit a single model on the path-average target
        # so the wrapper still has at least one estimator to query.
        fallback = _build_l4_model(family, params)
        fallback.fit(train_X, train_X.iloc[:, 0] * 0.0)
        sub_models.append(fallback)
    return _PathAverageEq4Model(sub_models, horizon=horizon)


def _l4_predict_one(
    model, X: pd.DataFrame, position: int, *, forecast_strategy: str, horizon: int
) -> float:
    if forecast_strategy == "iterated":
        # Roll predictions one step at a time using a copy of the row.
        row = X.iloc[[position]].copy()
        last_value = float(model.predict(row)[0])
        for _step in range(1, horizon):
            # naive iteration: replace the last lag column (if any) with the
            # forecasted value; otherwise reuse the same row (no exogenous update).
            if "y_lag1" in row.columns:
                row.loc[:, "y_lag1"] = last_value
            last_value = float(model.predict(row)[0])
        return last_value
    if forecast_strategy in {"path_average", "path_average_eq4"}:
        # average h consecutive forecasts (path); for ``path_average`` this
        # is equivalent to direct on a cumulative_average target (Eq. 5 OLS
        # limit). For ``path_average_eq4`` ``model`` is a
        # ``_PathAverageEq4Model`` whose ``.predict`` averages h per-horizon
        # sub-model predictions (paper Eq. 4 proper).
        return float(model.predict(X.iloc[[position]])[0])
    return float(model.predict(X.iloc[[position]])[0])


def _resolve_l4_tuning(
    params: dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> dict[str, Any]:
    """Issue #217 -- dispatch the L4 ``search_algorithm`` axis.

    ``search_algorithm`` paths:

    * ``cv_path``: regularisation path on (alpha) for ridge/lasso/elastic_net
      via ``RidgeCV`` / ``LassoCV``. Honours ``cv_path_alphas`` when supplied.
    * ``grid_search``: exhaustive ``GridSearchCV`` over
      ``leaf_config.tuning_grid``.
    * ``random_search``: ``RandomizedSearchCV`` over
      ``leaf_config.tuning_distributions`` (``tuning_budget`` iterations).
    * ``bayesian_optimization``: when ``optuna`` is installed, run an
      Optuna study with ``tuning_budget`` trials; otherwise fall back to
      ``random_search`` (degraded gracefully).
    * ``genetic_algorithm``: simple tournament-selection evolution over
      ``tuning_distributions`` for ``genetic_algorithm_generations``
      generations of size ``genetic_algorithm_population``.

    All paths respect time-series constraints (no kfold), and seed via
    ``params.random_state`` for determinism.
    """

    family = params.get("family", "ridge")
    n_obs = len(X)
    if n_obs < 8:
        return params
    algo = params.get("search_algorithm", "none")
    leaf = params.get("_l4_leaf_config", {}) or {}  # injected by caller when present
    cv_folds = max(2, min(5, n_obs // 4))
    seed = int(params.get("random_state", 0))
    X_filled = X.fillna(0.0)

    if algo == "cv_path":
        alphas = leaf.get("cv_path_alphas") or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        try:
            if family in {"ridge", "ols"}:
                picker = RidgeCV(alphas=alphas, cv=cv_folds)
                picker.fit(X_filled, y)
                params["alpha"] = float(picker.alpha_)
            elif family in {"lasso", "elastic_net", "lasso_path"}:
                picker = LassoCV(
                    alphas=alphas, cv=cv_folds, max_iter=20000, random_state=seed
                )
                picker.fit(X_filled, y)
                params["alpha"] = float(picker.alpha_)
        except Exception:
            pass
        return params

    # Coulombe-Surprenant-Leroux-Stevanovic (2022 JAE) Feature 3 schemes.
    # Each scheme picks ``alpha`` from ``cv_path_alphas`` for the alpha-
    # tunable linear families (ridge / lasso / elastic_net / lasso_path /
    # ols). Non-alpha families pass through unchanged and the caller's
    # default ``params`` are used.
    if algo in {"kfold", "poos", "aic", "bic", "block_cv"}:
        # v0.9.0a0 audit-fix #15: family-specific CV grids. Earlier path
        # only handled ``alpha`` for linear families; non-alpha families
        # silently passed through. Now KRR shares the alpha grid; RF
        # uses ``max_depth``; SVR uses ``C``. Each branch picks one
        # hyperparameter from a family-specific candidate list.
        #
        # Phase A4a (paper 16, Round 1 finding 7): paper §3.2 Eq. (18)
        # specifies elastic_net's ζ ∈ {0=Ridge, 1=Lasso, ζ_CV=EN-tuned}.
        # Pre-A4 elastic_net only tuned ``alpha`` and left ``l1_ratio``
        # at the helper-pinned 0.5. Now elastic_net gets a 2-D grid
        # ``(alpha × l1_ratio)`` and BOTH params are set on the winner.
        alphas = leaf.get("cv_path_alphas") or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        # 2-D grid as a list of param-overlay dicts. The 1-D path remains
        # backward-compatible: single-element overlays.
        grid_overlays: list[dict[str, Any]] = []
        if family == "elastic_net":
            l1_ratios = leaf.get("cv_path_l1_ratios") or [0.1, 0.3, 0.5, 0.7, 0.9]
            for a in list(alphas):
                for r in list(l1_ratios):
                    grid_overlays.append({"alpha": float(a), "l1_ratio": float(r)})
        elif family in {"ridge", "ols", "lasso", "lasso_path", "kernel_ridge"}:
            grid_overlays = [{"alpha": float(a)} for a in alphas]
        elif family == "random_forest":
            grid_overlays = [
                {"max_depth": v}
                for v in (leaf.get("rf_max_depth_grid") or [None, 4, 8, 16])
            ]
        elif family in {"svr_linear", "svr_rbf"}:
            grid_overlays = [
                {"C": float(c)} for c in (leaf.get("svr_C_grid") or [0.1, 1.0, 10.0])
            ]
        else:
            return params
        try:
            if algo == "block_cv":
                # Goulet Coulombe et al. (2024) Albacore §3 — non-
                # overlapping K-block CV. Default 10 blocks per paper
                # (override via leaf_config.block_cv_splits).
                n_blocks = int(leaf.get("block_cv_splits", 10))
                n_blocks = max(2, min(n_blocks, n_obs))
                block_starts = np.linspace(0, n_obs, n_blocks + 1, dtype=int)
                best_overlay, best_score = None, float("inf")
                for overlay in grid_overlays:
                    params_v = dict(params)
                    params_v.update(overlay)
                    fold_mses = []
                    for k in range(n_blocks):
                        s, e = int(block_starts[k]), int(block_starts[k + 1])
                        if e <= s:
                            continue
                        train_mask = np.ones(n_obs, dtype=bool)
                        train_mask[s:e] = False
                        if train_mask.sum() < 4:
                            continue
                        X_tr = X_filled.iloc[train_mask]
                        y_tr = y.iloc[train_mask]
                        X_te = X_filled.iloc[s:e]
                        y_te = y.iloc[s:e]
                        mdl = _build_l4_model(family, params_v)
                        try:
                            mdl.fit(X_tr, y_tr)
                            preds = np.asarray(mdl.predict(X_te), dtype=float)
                        except Exception:
                            continue
                        fold_mses.append(
                            float(np.mean((np.asarray(y_te, dtype=float) - preds) ** 2))
                        )
                    if not fold_mses:
                        continue
                    score = float(np.mean(fold_mses))
                    if score < best_score:
                        best_score, best_overlay = score, overlay
                if best_overlay is not None:
                    params.update(best_overlay)
                return params
            if algo == "kfold":
                from sklearn.model_selection import KFold, cross_val_score

                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                best_overlay, best_score = None, float("inf")
                for overlay in grid_overlays:
                    params_v = dict(params)
                    params_v.update(overlay)
                    mdl = _build_l4_model(family, params_v)
                    try:
                        score = -float(
                            np.mean(
                                cross_val_score(
                                    mdl,
                                    X_filled,
                                    y,
                                    cv=cv,
                                    scoring="neg_mean_squared_error",
                                )
                            )
                        )
                    except Exception:
                        continue
                    if score < best_score:
                        best_score, best_overlay = score, overlay
                if best_overlay is not None:
                    params.update(best_overlay)
            elif algo == "poos":
                n_holdout = max(1, n_obs // 4)
                n_train = n_obs - n_holdout
                if n_train < 4:
                    return params
                X_tr, X_te = X_filled.iloc[:n_train], X_filled.iloc[n_train:]
                y_tr, y_te = y.iloc[:n_train], y.iloc[n_train:]
                best_overlay, best_score = None, float("inf")
                for overlay in grid_overlays:
                    params_v = dict(params)
                    params_v.update(overlay)
                    mdl = _build_l4_model(family, params_v)
                    try:
                        mdl.fit(X_tr, y_tr)
                        preds = np.asarray(mdl.predict(X_te), dtype=float)
                    except Exception:
                        continue
                    score = float(np.mean((np.asarray(y_te, dtype=float) - preds) ** 2))
                    if score < best_score:
                        best_score, best_overlay = score, overlay
                if best_overlay is not None:
                    params.update(best_overlay)
            else:  # algo in {"aic", "bic"}
                n = max(int(n_obs), 2)
                penalty = 2.0 if algo == "aic" else float(np.log(n))
                best_overlay, best_score = None, float("inf")
                for overlay in grid_overlays:
                    params_v = dict(params)
                    params_v.update(overlay)
                    mdl = _build_l4_model(family, params_v)
                    try:
                        mdl.fit(X_filled, y)
                        preds = np.asarray(mdl.predict(X_filled), dtype=float)
                    except Exception:
                        continue
                    rss = float(np.sum((np.asarray(y, dtype=float) - preds) ** 2))
                    if family in {"lasso", "elastic_net", "lasso_path"} and hasattr(
                        mdl, "coef_"
                    ):
                        k = int(np.sum(np.abs(np.ravel(mdl.coef_)) > 1e-9)) + 1
                    else:
                        k = X_filled.shape[1] + 1
                    score = n * float(np.log(max(rss / n, 1e-12))) + penalty * k
                    if score < best_score:
                        best_score, best_overlay = score, overlay
                if best_overlay is not None:
                    params.update(best_overlay)
        except Exception:
            pass
        return params

    grid = leaf.get("tuning_grid")
    distributions = leaf.get("tuning_distributions")
    budget = int(leaf.get("tuning_budget", 20))

    if algo == "grid_search" and grid:
        try:
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

            base = _build_l4_model(family, params)
            cv = TimeSeriesSplit(n_splits=cv_folds)
            search = GridSearchCV(base, grid, cv=cv, n_jobs=1)
            search.fit(X_filled, y)
            params.update(search.best_params_)
        except Exception:
            pass
        return params

    if algo == "random_search" and distributions:
        try:
            from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

            base = _build_l4_model(family, params)
            cv = TimeSeriesSplit(n_splits=cv_folds)
            search = RandomizedSearchCV(
                base,
                distributions,
                n_iter=budget,
                cv=cv,
                n_jobs=1,
                random_state=seed,
            )
            search.fit(X_filled, y)
            params.update(search.best_params_)
        except Exception:
            pass
        return params

    if algo == "bayesian_optimization" and distributions:
        try:
            import optuna  # type: ignore

            def objective(trial):
                trial_params = dict(params)
                for name, dist in distributions.items():
                    if isinstance(dist, list):
                        trial_params[name] = trial.suggest_categorical(name, dist)
                    elif isinstance(dist, tuple) and len(dist) == 2:
                        low, high = dist
                        if isinstance(low, int) and isinstance(high, int):
                            trial_params[name] = trial.suggest_int(name, low, high)
                        else:
                            trial_params[name] = trial.suggest_float(
                                name, float(low), float(high)
                            )
                model = _build_l4_model(family, trial_params)
                from sklearn.model_selection import cross_val_score, TimeSeriesSplit

                scores = cross_val_score(
                    model,
                    X_filled,
                    y,
                    cv=TimeSeriesSplit(n_splits=cv_folds),
                    scoring="neg_mean_squared_error",
                )
                return -float(np.mean(scores))

            study = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(objective, n_trials=budget, show_progress_bar=False)
            params.update(study.best_params)
        except ImportError:
            # Optuna not installed -- fall back to random search.
            params["search_algorithm"] = "random_search"
            return _resolve_l4_tuning(params, X, y)
        except Exception:
            pass
        return params

    if algo == "genetic_algorithm" and distributions:
        try:
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit

            rng = np.random.default_rng(seed)
            pop_size = int(leaf.get("genetic_algorithm_population", 12))
            n_gens = int(leaf.get("genetic_algorithm_generations", 5))

            def sample_individual() -> dict[str, Any]:
                individual = dict(params)
                for name, dist in distributions.items():
                    if isinstance(dist, list):
                        individual[name] = dist[rng.integers(0, len(dist))]
                    elif isinstance(dist, tuple) and len(dist) == 2:
                        low, high = dist
                        if isinstance(low, int) and isinstance(high, int):
                            individual[name] = int(rng.integers(low, high + 1))
                        else:
                            individual[name] = float(
                                rng.uniform(float(low), float(high))
                            )
                return individual

            def fitness(ind: dict[str, Any]) -> float:
                model = _build_l4_model(family, ind)
                cv = TimeSeriesSplit(n_splits=cv_folds)
                try:
                    return -float(
                        np.mean(
                            cross_val_score(
                                model,
                                X_filled,
                                y,
                                cv=cv,
                                scoring="neg_mean_squared_error",
                            )
                        )
                    )
                except Exception:
                    return float("inf")

            population = [sample_individual() for _ in range(pop_size)]
            for _ in range(n_gens):
                fitnesses = [fitness(ind) for ind in population]
                # Keep top half, cross with best.
                ranked = sorted(zip(fitnesses, range(len(population))))
                top = [population[i] for _, i in ranked[: pop_size // 2]]
                children = []
                while len(top) + len(children) < pop_size:
                    p1, p2 = (
                        top[rng.integers(0, len(top))],
                        top[rng.integers(0, len(top))],
                    )
                    child = dict(p1)
                    for k in distributions:
                        if rng.random() < 0.5 and k in p2:
                            child[k] = p2[k]
                    children.append(child)
                population = top + children
            best = min(population, key=fitness)
            params.update({k: best[k] for k in distributions if k in best})
        except Exception:
            pass
        return params

    # Fallback: legacy cv_path behaviour for backward compatibility.
    if family in {"ridge", "ols"}:
        try:
            picker = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=cv_folds)
            picker.fit(X_filled, y)
            params["alpha"] = float(picker.alpha_)
        except Exception:
            pass
    elif family in {"lasso", "elastic_net", "lasso_path"}:
        try:
            picker = LassoCV(cv=cv_folds, max_iter=20000, random_state=seed)
            picker.fit(X_filled, y)
            params["alpha"] = float(picker.alpha_)
        except Exception:
            pass
    return params


# ----- helper estimators (operational schema, light implementations) -----


class _AutoClipKNN:
    """KNN regressor that clamps ``n_neighbors`` to the training set size.

    Walk-forward training windows can be small early on; sklearn's KNN raises
    when ``n_neighbors`` exceeds available samples, which would defeat the
    auto-recipe slice.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform") -> None:
        self.n_neighbors = max(1, int(n_neighbors))
        self.weights = weights
        self._knn: KNeighborsRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_AutoClipKNN":
        n = max(1, min(self.n_neighbors, len(X)))
        self._knn = KNeighborsRegressor(n_neighbors=n, weights=self.weights)
        self._knn.fit(X.fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._knn is None:
            return np.zeros(len(X))
        return self._knn.predict(X.fillna(0.0))


class _LinearARModel:
    """AR(p) on the target alone using OLS on lagged y_t."""

    def __init__(self, p: int = 1) -> None:
        self.p = max(1, int(p))
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._last_y: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_LinearARModel":
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) <= self.p:
            self.coef_ = np.zeros(self.p)
            self.intercept_ = float(np.mean(y_arr) if len(y_arr) else 0.0)
            return self
        rows = []
        targets = []
        for t in range(self.p, len(y_arr)):
            rows.append(y_arr[t - self.p : t][::-1])
            targets.append(y_arr[t])
        design = np.column_stack([np.ones(len(rows)), np.asarray(rows)])
        beta, *_ = np.linalg.lstsq(design, np.asarray(targets), rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        self._last_y = y_arr
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self._last_y is None:
            return np.zeros(len(X))
        if len(self._last_y) < self.p:
            return np.full(len(X), self.intercept_)
        last_window = self._last_y[-self.p :][::-1]
        return np.array(
            [float(self.intercept_ + float(self.coef_ @ last_window))] * len(X)
        )


class _FactorAugmentedAR:
    """Stock-Watson factor-augmented AR via PCA on X plus AR lags on y.

    Phase C M12 -- ``_idio_var`` / ``_factors_train`` / ``_design_train``
    are stashed at fit time so the Bai-Ng (2006) generated-regressor PI
    correction can read the factor-estimation noise variance (V₂/N) and
    the parameter-estimation noise (V₁/T) without re-decomposing X.
    """

    def __init__(self, p: int = 1, n_factors: int = 3) -> None:
        self.p = p
        self.n_factors = n_factors
        self._factor_loadings: np.ndarray | None = None
        self._regression: LinearRegression | None = None
        self._mean: np.ndarray | None = None
        self._last_y: np.ndarray | None = None
        # Phase C M12 -- Bai-Ng PI correction support:
        self._idio_var: np.ndarray | None = None
        self._factors_train: np.ndarray | None = None
        self._design_train: np.ndarray | None = None
        self._residuals_train: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FactorAugmentedAR":
        from sklearn.decomposition import PCA

        if X.shape[0] < max(self.n_factors, self.p + 1):
            self._regression = LinearRegression().fit(X.fillna(0.0), y)
            self._last_y = np.asarray(y, dtype=float)
            return self
        n = min(self.n_factors, X.shape[1])
        self._mean = X.mean(axis=0).to_numpy()
        pca = PCA(n_components=n, random_state=0)
        X_centered = (X - self._mean).fillna(0.0).to_numpy()
        factors = pca.fit_transform(X_centered)
        self._factor_loadings = pca.components_
        ar_lags = (
            pd.concat([y.shift(lag) for lag in range(1, self.p + 1)], axis=1)
            .fillna(0.0)
            .to_numpy()
        )
        design = np.column_stack([factors, ar_lags])
        self._regression = LinearRegression().fit(design, y.values)
        self._last_y = np.asarray(y, dtype=float)
        # Phase C M12 -- stash the PCA reconstruction residuals'
        # per-column variance (idiosyncratic Σ̂_e diag) plus the in-sample
        # design matrix and y residuals for the Bai-Ng correction.
        try:
            recon = factors @ self._factor_loadings  # (T, N)
            residuals_X = X_centered - recon
            self._idio_var = np.asarray(residuals_X.var(axis=0, ddof=1), dtype=float)
            self._factors_train = np.asarray(factors, dtype=float)
            self._design_train = np.asarray(design, dtype=float)
            y_pred = self._regression.predict(design)
            self._residuals_train = np.asarray(y.values - y_pred, dtype=float)
        except Exception:  # pragma: no cover - defensive
            self._idio_var = None
            self._factors_train = None
            self._design_train = None
            self._residuals_train = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if (
            self._regression is None
            or self._factor_loadings is None
            or self._mean is None
        ):
            if self._regression is not None:
                return self._regression.predict(X.fillna(0.0))
            return np.zeros(len(X))
        factors = (X - self._mean).fillna(0.0).to_numpy() @ self._factor_loadings.T
        if self._last_y is None:
            ar = np.zeros((len(X), self.p))
        else:
            tail = list(reversed(self._last_y[-self.p :].tolist()))
            tail += [0.0] * (self.p - len(tail))
            ar = np.tile(tail, (len(X), 1))
        design = np.column_stack([factors, ar])
        return self._regression.predict(design)

    # Phase C M12 -- accessors for the Bai-Ng (2006) PI correction.

    @property
    def factor_loadings_(self) -> np.ndarray | None:
        return self._factor_loadings

    @property
    def factor_coefficients_(self) -> np.ndarray:
        """First ``n_factors`` entries of the regression coef_."""

        if self._regression is None or self._factor_loadings is None:
            return np.zeros(0, dtype=float)
        n_factors = self._factor_loadings.shape[0]
        return np.asarray(self._regression.coef_[:n_factors], dtype=float)

    @property
    def idiosyncratic_variance_(self) -> np.ndarray | None:
        return self._idio_var


class _NonNegRidge:
    """Non-negative ridge regression (v0.9 Phase 1 Tier 1 promotion).

    Solves ``min ||y - Xβ||² + α||β||²`` subject to β >= 0 via
    ``scipy.optimize.nnls`` on the augmented system::

        [X        ]       [y]
        [√α · I_p] · β =  [0]

    NNLS on this augmented system is the standard closed-form path for
    non-negative ridge: the augmented residual norm equals the original
    ridge objective, so the constrained NNLS solution is the constrained
    ridge solution. Used for the Albacore-family Assemblage Regression
    (Goulet Coulombe / Klieber / Barrette / Goebel 2024 'Maximally
    Forward-Looking Core Inflation') where assembly weights must be
    non-negative.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_NonNegRidge":
        from scipy.optimize import nnls

        self._cols = list(X.columns)
        Xa = X.fillna(0.0).to_numpy(dtype=float)
        ya = np.asarray(y, dtype=float)
        if Xa.shape[0] == 0 or Xa.shape[1] == 0:
            self._coef = np.zeros(Xa.shape[1])
            self._intercept = float(ya.mean()) if ya.size else 0.0
            return self
        # Center y so the intercept absorbs the mean (NNLS does not have
        # an intercept term natively); coefficients then act on centered X.
        y_mean = float(ya.mean())
        y_c = ya - y_mean
        # Augmented system: stack X above sqrt(alpha)·I.
        sqrt_alpha = float(np.sqrt(max(self.alpha, 0.0)))
        p = Xa.shape[1]
        X_aug = np.vstack([Xa, sqrt_alpha * np.eye(p)])
        y_aug = np.concatenate([y_c, np.zeros(p)])
        beta, _ = nnls(X_aug, y_aug, maxiter=max(200, 5 * p))
        self._coef = beta
        self._intercept = y_mean
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            return np.zeros(len(X))
        Xa = (
            X.reindex(columns=self._cols, fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return Xa @ self._coef + self._intercept


class _TwoStageRandomWalkRidge:
    """Two-Step Ridge Regression (Coulombe 2025 IJF) -- closed-form
    generalised ridge with a random-walk prior on coefficient deviations.

    The estimator surfaces a per-time-step coefficient path β̂_t under
    the reparametrisation β_k = C_RW · θ_k where C_RW is the lower-
    triangular ones matrix (cumulative-sum operator). Eq. 9 of the
    paper rewrites the homogeneous-variance ridge as

        β̂ = C Z' (Z Z' + λ I_T)^{-1} y,

    with Z = W C, W = [diag(X_1) | ... | diag(X_K)], C = I_K ⊗ C_RW.
    Eq. 11 then re-solves with heterogeneous variances Ω_θ (per-coef)
    and Ω_ε (per-time, recovered from step-1 residuals via a volatility
    model):

        θ̂ = Ω_θ Z' (Z Ω_θ Z' + Ω_ε)^{-1} y.

    Both equations are *closed-form* (two T × T solves; no iteration).
    The L4 contract is single-equation TVP regression: ``predict``
    returns the most-recent fitted forecast (a one-step-ahead under
    the random-walk assumption that β_{T+1} ≈ β_T).

    Parameters
    ----------
    alpha:
        Initial ridge penalty λ for step 1. The step-2 re-CV is
        bounded above by ``alpha * max_alpha_ratio`` to avoid
        runaway shrinkage on degenerate residuals. When
        ``alpha_strategy="second_cv"`` (the default), this value is
        only used as a fallback / starting hint; the fitted λ is the
        cross-validated minimiser over ``alpha_grid``.
    vol_model:
        ``"garch11"`` (default; paper §4 spec, Coulombe 2025 IJF Eq. 11
        — requires ``arch>=5.0``; falls back to EWMA if the package is
        missing) or ``"ewma"`` (lambda=0.94 RiskMetrics decay; no extra
        deps). v0.9.0a0 audit gap-fix raised the default from
        ``"ewma"`` to ``"garch11"`` to match the paper's spec.
    max_alpha_ratio:
        Upper bound on the ratio of step-2 to step-1 lambda. Default
        ``1e6``; the paper does not pin this explicitly but the dual
        ZZ' + λI invert becomes numerically unstable beyond.
    alpha_strategy:
        ``"second_cv"`` (default; paper §2.5 Algorithm 1 step 4 — "Use
        solution (17) to **rerun CV** and get β̂_2, the final
        estimator"; footnote 4 + §2.4.1 justify the second λ-CV
        because heterogeneous variance changes the effective
        regularization) runs a K-fold CV over ``alpha_grid`` after the
        warm-start step-1, picks the held-out-MSE minimiser, and
        refits step-1 + step-2 with that λ on the full sample.
        ``"fixed"`` skips the CV and uses the user-provided ``alpha``.
    alpha_grid:
        Candidate λ grid for the second CV. Default
        ``[0.01, 0.1, 1.0, 10.0, 100.0]`` (paper §2.5 does not pin
        specific values; this is the standard ``cv_path_alphas`` grid
        used elsewhere in macroforecast for ridge tuning).
    cv_folds:
        Number of K-fold splits for the second CV. Default ``5``
        (paper does not pin K).
    random_state:
        Propagated from L0 ``random_seed``; seeds the K-fold shuffle
        in the second CV.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        vol_model: str = "garch11",
        max_alpha_ratio: float = 1e6,
        alpha_strategy: str = "second_cv",
        alpha_grid: list[float] | None = None,
        cv_folds: int = 5,
        random_state: int = 0,
    ) -> None:
        self.alpha = float(max(alpha, 1e-6))
        self.vol_model = str(vol_model).lower()
        self.max_alpha_ratio = float(max_alpha_ratio)
        self.alpha_strategy = str(alpha_strategy).lower()
        if self.alpha_strategy not in {"fixed", "second_cv"}:
            raise ValueError(
                f"alpha_strategy={alpha_strategy!r} not supported; "
                f"use 'fixed' or 'second_cv'."
            )
        self.alpha_grid = (
            [float(a) for a in alpha_grid]
            if alpha_grid is not None
            else [0.01, 0.1, 1.0, 10.0, 100.0]
        )
        self.cv_folds = int(max(cv_folds, 2))
        self.random_state = int(random_state)
        self._cols: tuple[str, ...] = ()
        self._beta_path: np.ndarray | None = None  # (T, K)
        self._beta_last: np.ndarray | None = None  # (K,) most recent β_t
        self._intercept: float = 0.0
        self._fitted: bool = False
        # Tuned λ (post second CV); equals self.alpha when strategy="fixed".
        self.tuned_alpha_: float = float(self.alpha)

    @staticmethod
    def _ewma_vol(eps: np.ndarray, lam: float = 0.94) -> np.ndarray:
        """RiskMetrics EWMA conditional variance estimator. Returns σ²_t."""
        T = len(eps)
        sigma2 = np.empty(T, dtype=float)
        var0 = float(np.mean(eps**2)) if T else 1.0
        sigma2[0] = var0
        for t in range(1, T):
            sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * float(eps[t - 1] ** 2)
        return sigma2

    @staticmethod
    def _garch_vol(eps: np.ndarray) -> np.ndarray:
        """GARCH(1,1) conditional variance via the ``arch`` package; falls
        back to EWMA when ``arch`` is unavailable."""
        try:
            from arch import arch_model  # type: ignore[import-not-found]
        except ImportError:
            return _TwoStageRandomWalkRidge._ewma_vol(eps)
        if len(eps) < 16:
            return _TwoStageRandomWalkRidge._ewma_vol(eps)
        try:
            am = arch_model(eps, mean="zero", vol="GARCH", p=1, q=1, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            return np.asarray(res.conditional_volatility, dtype=float) ** 2
        except Exception:
            return _TwoStageRandomWalkRidge._ewma_vol(eps)

    def _build_design(self, X: np.ndarray) -> np.ndarray:
        """Build Z = W C where W = [diag(X_1) | ... | diag(X_K)] (T × KT),
        C = I_K ⊗ C_RW (KT × KT). Result Z is (T × KT) and equal to the
        block of K cumulative-design slabs ``X_k(t) · 1[s ≤ t]``."""
        T, K = X.shape
        # Vectorised: Z[t, k*T + s] = X[t, k] * 1[s <= t]. Build by stacking
        # K independent (T × T) blocks of the form diag(X_k) @ C_RW where
        # C_RW is lower-triangular ones. Equivalent to broadcasting.
        idx = np.arange(T)
        mask = (idx[None, :] <= idx[:, None]).astype(float)  # (T, T) lower-tri
        # For each column k: block_k[t, s] = X[t, k] * mask[t, s].
        Z_blocks = [X[:, k : k + 1] * mask for k in range(K)]
        return np.hstack(Z_blocks)  # (T, KT)

    def _two_step_fit_centered(
        self,
        Xa: np.ndarray,
        y_c: np.ndarray,
        lam: float,
    ) -> np.ndarray:
        """Run step-1 (homogeneous-Ω) + step-2 (heterogeneous-Ω) given a
        centred target ``y_c`` and a fixed ridge penalty ``lam``. Returns
        the per-time β path matrix of shape (K, T) (cumulative basis,
        rescaled back to original-scale β via ``cumsum`` on θ̂)."""
        T, K = Xa.shape
        Z = self._build_design(Xa)  # (T, KT)
        ZZt = Z @ Z.T  # (T, T)
        # Step 1: homogeneous variance ridge.
        a1 = np.linalg.solve(ZZt + float(lam) * np.eye(T), y_c)
        theta1 = Z.T @ a1  # (KT,)
        theta1_mat = theta1.reshape(K, T)
        eps_hat = y_c - Z @ theta1

        # Step 2a: per-time residual variance Ω_ε (mean-normalised to 1).
        sigma2_eps = (
            self._garch_vol(eps_hat)
            if self.vol_model == "garch11"
            else self._ewma_vol(eps_hat)
        )
        sigma2_eps = np.maximum(sigma2_eps, 1e-12)
        sigma2_eps /= float(sigma2_eps.mean())

        # Step 2b: per-coefficient θ-variance Ω_θ (rescaled to mean 1/λ).
        sigma2_u = (theta1_mat**2).mean(axis=1)  # (K,)
        sigma2_u = np.maximum(sigma2_u, 1e-12)
        target_mean = 1.0 / float(lam)
        sigma2_u *= target_mean / float(sigma2_u.mean())
        sigma2_u = np.clip(
            sigma2_u,
            target_mean / self.max_alpha_ratio,
            target_mean * self.max_alpha_ratio,
        )

        # Step 2c: heterogeneous-Ω solve.
        omega_theta_diag = np.repeat(sigma2_u, T)  # (KT,)
        Z_omega = Z * omega_theta_diag[None, :]  # (T, KT)
        ZOZt = Z_omega @ Z.T  # (T, T)
        Omega_eps = np.diag(sigma2_eps)
        a2 = np.linalg.solve(ZOZt + Omega_eps, y_c)
        theta2 = omega_theta_diag * (Z.T @ a2)
        theta2_mat = theta2.reshape(K, T)
        beta2 = np.cumsum(theta2_mat, axis=1)  # (K, T)
        return beta2

    def _second_cv_pick_alpha(
        self,
        Xa: np.ndarray,
        y_c: np.ndarray,
    ) -> float:
        """Algorithm 1 step 4: K-fold CV over ``alpha_grid``. For each
        fold, fit step-1 + step-2 at every λ candidate on the train
        slice and score held-out MSE under the ``β_{T+1} ≈ β_T``
        random-walk one-step-ahead predict rule. Picks the λ minimising
        average held-out MSE. Returns the chosen λ."""
        T = Xa.shape[0]
        if T < max(2 * self.cv_folds, 4):
            # Too short to fold-split; fall back to user alpha.
            return float(self.alpha)
        from sklearn.model_selection import KFold

        kf = KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        best_lam, best_score = float(self.alpha), float("inf")
        for lam in self.alpha_grid:
            fold_mses = []
            for tr_idx, te_idx in kf.split(np.arange(T)):
                # Train slice must have at least a couple of obs per fold.
                if len(tr_idx) < 4 or len(te_idx) == 0:
                    continue
                tr_idx_sorted = np.sort(tr_idx)
                X_tr = Xa[tr_idx_sorted]
                y_tr = y_c[tr_idx_sorted]
                try:
                    beta_path = self._two_step_fit_centered(X_tr, y_tr, float(lam))
                except Exception:
                    continue
                # One-step-ahead RW predict: use the last fitted β as
                # the held-out coefficient (paper §2.5 predict rule).
                beta_last = beta_path[:, -1]
                preds = Xa[te_idx] @ beta_last
                fold_mses.append(float(np.mean((y_c[te_idx] - preds) ** 2)))
            if not fold_mses:
                continue
            score = float(np.mean(fold_mses))
            if score < best_score:
                best_score, best_lam = score, float(lam)
        return float(max(best_lam, 1e-6))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TwoStageRandomWalkRidge":
        self._cols = tuple(X.columns)
        Xa = X.fillna(0.0).to_numpy(dtype=float)
        ya = np.asarray(y, dtype=float)
        T, K = Xa.shape
        if T == 0 or K == 0:
            self._beta_last = np.zeros(K)
            self._intercept = float(ya.mean()) if ya.size else 0.0
            self._fitted = True
            self.tuned_alpha_ = float(self.alpha)
            return self

        # Center y so the intercept absorbs the mean.
        y_mean = float(ya.mean())
        y_c = ya - y_mean
        self._intercept = y_mean

        # Algorithm 1 step 4 (paper §2.5): rerun CV after the warm-start
        # to pick the final λ. Footnote 4 + §2.4.1 justify the second
        # CV because heterogeneous variance changes the effective
        # regularization.
        if self.alpha_strategy == "second_cv":
            lam = self._second_cv_pick_alpha(Xa, y_c)
        else:  # "fixed"
            lam = float(self.alpha)
        self.tuned_alpha_ = float(lam)

        # Final refit on the full sample at the selected λ.
        beta2 = self._two_step_fit_centered(Xa, y_c, lam)

        # Cache the final-time-step β as the predict basis.
        self._beta_path = beta2.T  # (T, K) for L7 mrf-style consumption
        self._beta_last = beta2[:, -1] if T > 0 else np.zeros(K)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._beta_last is None:
            return np.zeros(len(X), dtype=float)
        # Re-align to training columns; missing → 0.
        Xa = (
            X.reindex(columns=list(self._cols), fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        # One-step-ahead under RW: β_{T+1} ≈ β_T.
        return Xa @ self._beta_last + self._intercept


class _ShrinkToTargetRidge:
    """Albacore_comps (Maximally Forward-Looking Core Inflation, Goulet
    Coulombe / Klieber / Barrette / Goebel 2024) Variant A.

    Solves
        arg min_w  ‖y − Xw‖² + α ‖w − w_target‖²    s.t. w ≥ 0,  w'1 = 1
    via scipy SLSQP. Composes with the existing ``coefficient_constraint=
    nonneg`` axis (passes ``bounds=[(0, None)]·K`` when ``nonneg=True``)
    and the simplex equality constraint can be toggled via ``simplex``
    (default True for Albacore semantics; False reduces to a generalised
    shrink-to-target ridge with arbitrary β).

    Limit cases:
      * α = 0           → unconstrained / NNLS / OLS
      * α → ∞           → returns ``w_target`` exactly
      * ``w_target = 0``, ``simplex=False``, ``nonneg=True`` →
        equivalent to ``_NonNegRidge`` (B-1 v0.8.9)

    Hyperparameters (override via ``params``):
      * ``alpha`` -- shrinkage strength
      * ``prior_target`` -- target weight vector (list / ndarray of
        length K). Default is uniform ``1/K``.
      * ``prior_simplex`` -- enforce ``w'1 = 1`` (default True)
      * ``coefficient_constraint`` (composed at L4 dispatch) --
        ``nonneg`` enables ``w ≥ 0``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prior_target: Any = None,
        simplex: bool = True,
        nonneg: bool = True,
    ) -> None:
        self.alpha = float(max(alpha, 0.0))
        self.prior_target_in = prior_target
        self.simplex = bool(simplex)
        self.nonneg = bool(nonneg)
        self._cols: tuple[str, ...] = ()
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0

    def _resolve_target(self, K: int) -> np.ndarray:
        if self.prior_target_in is None:
            # F-14 audit gap-fix (phase-f14): paper Albacore (Goulet
            # Coulombe et al. 2024) Eq. (1) requires w_headline
            # (CPI/PCE basket weights) — there is no paper-faithful
            # fallback. Raise hard error to match the helper guard in
            # maximally_forward_looking (paper_methods.py:1620-1625).
            raise ValueError(
                "_ShrinkToTargetRidge: prior_target=None is not permitted for the "
                "'shrink_to_target' prior. Paper Albacore (Goulet Coulombe et al. "
                "2024) Eq. (1) requires w_headline (CPI/PCE basket weights). Pass "
                "prior_target=<array of length K> to the recipe L4 node."
            )
        target = np.asarray(self.prior_target_in, dtype=float)
        if target.shape[0] != K:
            raise ValueError(f"prior_target length {target.shape[0]} ≠ K={K}")
        return target

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ShrinkToTargetRidge":
        # Phase B-13 (paper 13, Round 5 F3): paper §2 "Implementation
        # Details" cites the CVXR R package — convex programming with
        # OSQP/ECOS backends — for the Albacore Variant A QP. Earlier
        # macroforecast used scipy SLSQP; we now solve the same convex
        # problem via cvxpy + OSQP (the paper-stated backend).
        import cvxpy as cp

        self._cols = tuple(X.columns)
        Xa = X.fillna(0.0).to_numpy(dtype=float)
        ya = np.asarray(y, dtype=float)
        T, K = Xa.shape
        if T == 0 or K == 0:
            self._coef = np.zeros(K)
            self._intercept = float(ya.mean()) if ya.size else 0.0
            return self

        # The simplex constraint w'1=1 means y is *not* centred -- the
        # forecast is a convex combination, no separate intercept slot.
        # Without simplex, centre y so the intercept absorbs the mean.
        if self.simplex:
            y_target = ya
            self._intercept = 0.0
        else:
            self._intercept = float(ya.mean())
            y_target = ya - self._intercept

        w_target = self._resolve_target(K)

        # cvxpy variable: nonneg flag handled natively when set.
        w = cp.Variable(K, nonneg=self.nonneg)
        objective = cp.Minimize(
            cp.sum_squares(y_target - Xa @ w)
            + self.alpha * cp.sum_squares(w - w_target)
        )
        constraints: list[Any] = []
        if self.simplex:
            constraints.append(cp.sum(w) == 1)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"cvxpy did not converge: status={problem.status}")
        self._coef = np.asarray(w.value, dtype=float)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            return np.zeros(len(X), dtype=float)
        Xa = (
            X.reindex(columns=list(self._cols), fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return Xa @ self._coef + self._intercept


class _FusedDifferenceRidge:
    """Albacore_ranks (Maximally Forward-Looking Core Inflation, Goulet
    Coulombe et al. 2024) Variant B.

    Solves
        arg min_w  ‖y − Xw‖² + α ‖D w‖²    s.t. w ≥ 0,  mean(y) = mean(Xw)
    via scipy SLSQP. ``D`` is the first-difference operator
    (``difference_order=1`` -- default; the paper's spec); the penalty
    encourages smoothness over rank position. The mean-equality
    constraint pins the level of the recovered weight series; toggle
    via ``mean_equality``.

    Pairs naturally with the L3 ``asymmetric_trim`` op (B-6 v0.8.9):
    pre-sort each row of the predictor panel, then run this estimator
    on the rank-space data ``O = sort(X)``.

    Limit cases:
      * α = 0                     → standard OLS / NNLS
      * α → ∞                     → uniform weights ``w = 1/K``
      * ``difference_order = 0``  → standard ridge (no fusion)

    Hyperparameters (override via ``params``):
      * ``alpha`` -- fusion strength
      * ``prior_diff_order`` -- order of the difference operator
        (default 1; only 1 is paper-faithful)
      * ``prior_mean_equality`` -- enforce ``mean(y) = mean(Xw)``
        (default True; Albacore_ranks pinning constraint)
      * ``coefficient_constraint=nonneg`` -- composes at the dispatch
        level
    """

    def __init__(
        self,
        alpha: float = 1.0,
        difference_order: int = 1,
        mean_equality: bool = True,
        nonneg: bool = True,
    ) -> None:
        self.alpha = float(max(alpha, 0.0))
        self.difference_order = int(max(difference_order, 0))
        self.mean_equality = bool(mean_equality)
        self.nonneg = bool(nonneg)
        self._cols: tuple[str, ...] = ()
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0

    @staticmethod
    def _diff_matrix(K: int, order: int) -> np.ndarray:
        if order <= 0 or K <= 1:
            return np.eye(K)
        # First-difference: D[r, r] = 1, D[r, r-1] = -1; shape (K-1, K).
        D = np.eye(K) - np.eye(K, k=-1)
        D = D[1:]
        for _ in range(order - 1):
            D = D[1:] - D[:-1]  # iterated difference
        return D

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FusedDifferenceRidge":
        # Phase B-13 (paper 13, Round 5 F3): paper §2 cites CVXR for the
        # Variant B fused-difference QP. We solve via cvxpy + OSQP to
        # match the paper-stated convex backend; SLSQP path removed.
        import cvxpy as cp

        self._cols = tuple(X.columns)
        Xa = X.fillna(0.0).to_numpy(dtype=float)
        ya = np.asarray(y, dtype=float)
        T, K = Xa.shape
        if T == 0 or K == 0:
            self._coef = np.zeros(K)
            self._intercept = float(ya.mean()) if ya.size else 0.0
            return self

        # Mean equality pins ``mean(Xw) = mean(y)``; intercept therefore
        # stays at 0. Without mean-equality, centre y as usual.
        if self.mean_equality:
            y_target = ya
            self._intercept = 0.0
        else:
            self._intercept = float(ya.mean())
            y_target = ya - self._intercept

        D = self._diff_matrix(K, self.difference_order)

        w = cp.Variable(K, nonneg=self.nonneg)
        objective = cp.Minimize(
            cp.sum_squares(y_target - Xa @ w) + self.alpha * cp.sum_squares(D @ w)
        )
        constraints: list[Any] = []
        if self.mean_equality:
            constraints.append(cp.sum(Xa @ w) == cp.sum(y_target))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"cvxpy did not converge: status={problem.status}")
        self._coef = np.asarray(w.value, dtype=float)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            return np.zeros(len(X), dtype=float)
        Xa = (
            X.reindex(columns=list(self._cols), fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return Xa @ self._coef + self._intercept


class _SlowGrowingTree:
    """Slow-Growing Tree (Goulet Coulombe 2024 'Slow-Growing Trees').

    A *soft-weighted* CART that interpolates between hard splitting
    (η = 1, recovers standard CART) and Random Forest behaviour
    (η ≈ 0.1 + Herfindahl stopping H̄ ≈ 0.05). At each split step, all
    training rows receive a non-zero weight: rows satisfying the split
    rule keep weight ω, rows on the other side receive weight ω · (1 − η).

    **Algorithm 1** (paper p. 83-84):

      1. Init ``ω_i^0 = 1`` for all training rows i.
      2. For each open leaf l with ``H_l = Σ_i (ω_i^l)² / (Σ_i ω_i^l)² < H̄``:
         a. Find ``(k*, c*)`` minimising the weighted SSE:
            ``min Σ_{X[k] ≤ c} ω_i (y_i − μ_L)² + Σ_{X[k] > c} ω_i (y_i − μ_R)²``
         b. Children inherit ``ω_i^l · (1 − η · I(rule violated))``.
      3. Predict via leaf-weight propagation: each test point follows the
         same soft re-weighting through the tree, and the prediction is
         the weighted average ``Σ_l w_l(x) · μ_l``.

    Limit cases (verified by tests):
      * η = 1, H̄ = 0.25 → standard CART (hard splits).
      * η = 0.1, H̄ = 0.05 → SGT regime ("matches RF on Linear DGP at
        high R²" per paper Figure 2).
      * Empty input → constant prediction (training-mean).

    Hyperparameters (override via ``params``):
      * ``split_shrinkage`` (= η; default 0.0 -- routes to standard CART)
      * ``herfindahl_threshold`` (= H̄; default 0.25)
      * ``eta_depth_step`` (paper rule-of-thumb: η + 0.01·depth; default 0.0
        keeps η constant)
      * ``max_depth`` (additional safety bound on tree depth; default None)
      * ``random_state`` (reserved for potential subsampling extensions;
        currently unused -- the algorithm is deterministic given X, y)
    """

    # Internal node representation:
    #   leaf node:   (-1, -1, weights, mu_leaf)
    #     where weights is the (n,) soft-weight vector and mu_leaf the
    #     leaf-mean prediction at the leaf.
    #   split node:  (k, c, idx_left, idx_right)
    #     where k is the splitting feature and c the split threshold;
    #     idx_left and idx_right are indices into the node list.

    def __init__(
        self,
        eta: float = 0.1,
        herfindahl_threshold: float = 0.25,
        eta_depth_step: float = 0.01,
        eta_max_plateau: float = 0.5,
        mtry_frac: float = 1.0,
        max_depth: Any = None,
        random_state: int = 0,
        min_leaf_size: int = 5,
    ) -> None:
        self.eta = float(np.clip(eta, 1e-6, 1.0))
        self.herfindahl_threshold = float(np.clip(herfindahl_threshold, 1e-6, 1.0))
        # v0.9.0F audit-fix: paper p.87 specifies "starting at η=0.1 and
        # increasing it by 0.01 with depth, until an imposed plateau of
        # 0.5". The previous defaults silently disabled the depth-step
        # rule (=0.0) and clipped to 1.0 instead of the paper's 0.5.
        self.eta_depth_step = float(eta_depth_step)
        self.eta_max_plateau = float(np.clip(eta_max_plateau, 1e-6, 1.0))
        # v0.9.0F audit-fix: paper p.88 §2.3 surfaces ``mtry`` as a
        # stochastic split-feature sub-sampling knob ("mtry = 0.75 is
        # used throughout"). Default 1.0 = scan every column (paper-
        # silent baseline); user sets <1.0 to draw a random subset of
        # columns at every node, RF-style.
        self.mtry_frac = float(np.clip(mtry_frac, 1e-6, 1.0))
        self.max_depth = None if max_depth is None else int(max_depth)
        self.random_state = int(random_state)
        self.min_leaf_size = max(2, int(min_leaf_size))
        self._cols: tuple[str, ...] = ()
        self._train_y: np.ndarray | None = None
        self._fallback: float = 0.0
        # Tree representation: list of nodes in BFS order.
        # node[i] = ('leaf', mu) or ('split', k, c, left_idx, right_idx)
        self._nodes: list[tuple] = []

    @staticmethod
    def _herfindahl(omega: np.ndarray) -> float:
        s = float(omega.sum())
        if s <= 1e-12:
            return 1.0
        return float((omega**2).sum()) / (s * s)

    @staticmethod
    def _weighted_mean(y: np.ndarray, omega: np.ndarray) -> float:
        s = float(omega.sum())
        if s <= 1e-12:
            return 0.0
        return float((omega * y).sum() / s)

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        omega: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float, float] | None:
        """Find the split (k, c) minimising weighted SSE. Returns
        (k*, c*, sse*) or None when no improvement is possible."""

        n, K = X.shape
        # Column-by-column candidate cuts at sample midpoints. For
        # speed (and to avoid pathological all-zero-weight splits)
        # only consider cuts where both sides retain at least
        # ``min_leaf_size`` rows of *raw* mass (independent of soft
        # weights) — this matches sklearn's default behaviour and
        # keeps trees stable under low η.
        # v0.9.0F audit-fix: paper p.88 ``mtry = 0.75`` -- when
        # ``mtry_frac < 1.0`` we sample a random subset of features
        # at every node (RF-style stochastic feature selection).
        if self.mtry_frac < 1.0 and K > 1:
            n_keep = max(1, int(round(self.mtry_frac * K)))
            if rng is None:
                rng = np.random.default_rng(self.random_state)
            feature_iter = rng.choice(K, n_keep, replace=False)
        else:
            feature_iter = range(K)
        best = None
        s_total = float(omega.sum())
        if s_total <= 1e-12:
            return None
        sum_y = float((omega * y).sum())
        sum_y2 = float((omega * y * y).sum())
        full_var = sum_y2 - sum_y * sum_y / s_total

        for k in feature_iter:
            xk = X[:, k]
            order = np.argsort(xk, kind="stable")
            xk_sorted = xk[order]
            y_sorted = y[order]
            om_sorted = omega[order]
            # Cumulative weighted sums for left side at every cut
            cum_om = np.cumsum(om_sorted)
            cum_oy = np.cumsum(om_sorted * y_sorted)
            cum_oy2 = np.cumsum(om_sorted * y_sorted * y_sorted)
            for j in range(self.min_leaf_size, n - self.min_leaf_size):
                # Skip cuts where consecutive sorted values are equal
                if xk_sorted[j] == xk_sorted[j - 1]:
                    continue
                left_om = cum_om[j - 1]
                right_om = s_total - left_om
                if left_om <= 1e-12 or right_om <= 1e-12:
                    continue
                left_oy = cum_oy[j - 1]
                right_oy = sum_y - left_oy
                left_oy2 = cum_oy2[j - 1]
                right_oy2 = sum_y2 - left_oy2
                left_sse = left_oy2 - left_oy * left_oy / left_om
                right_sse = right_oy2 - right_oy * right_oy / right_om
                sse = left_sse + right_sse
                if best is None or sse < best[2]:
                    c_star = 0.5 * (xk_sorted[j - 1] + xk_sorted[j])
                    best = (k, float(c_star), float(sse))
        if best is None or best[2] >= full_var - 1e-12:
            return None
        return best

    def _build(self, X: np.ndarray, y: np.ndarray) -> None:
        """Iterative BFS tree construction with soft-weight propagation."""

        n = X.shape[0]
        # Each entry in the work queue: (parent_index, side, omega, depth)
        # The parent_index points into self._nodes; side is 'left'/'right'.
        # Start with a root sentinel (parent_index = -1).
        root_omega = np.ones(n, dtype=float)
        # Pre-allocate a placeholder root that will be overwritten when
        # the root's split / leaf decision is made.
        self._nodes.append(("placeholder",))
        queue: list[tuple[int, str, np.ndarray, int]] = [(0, "root", root_omega, 0)]
        # Single RNG threaded through _best_split so per-node mtry
        # draws are deterministic and reproducible given random_state.
        build_rng = np.random.default_rng(self.random_state)

        while queue:
            node_idx, side, omega, depth = queue.pop(0)
            mu_leaf = self._weighted_mean(y, omega)
            herf = self._herfindahl(omega)

            # Termination: H_l >= H̄, max_depth reached, or no further split.
            depth_ok = (self.max_depth is None) or (depth < self.max_depth)
            should_try_split = (
                depth_ok
                and herf < self.herfindahl_threshold
                and float(omega.sum()) > 2 * self.min_leaf_size
            )
            split = (
                self._best_split(X, y, omega, rng=build_rng)
                if should_try_split
                else None
            )
            if split is None:
                self._nodes[node_idx] = ("leaf", mu_leaf, omega.copy())
                continue

            k, c, _ = split
            # Paper p.87: η_l = min(η + 0.01·depth, 0.5) for the
            # SLOTH ramp-up regime. v0.9.0F audit-fix: when the user
            # passes an initial η above the plateau (e.g. η=1 for CART
            # parity), respect their intent — the plateau acts only as
            # a *ramp-up* cap, never a clip-down on the initial value.
            effective_plateau = max(self.eta, self.eta_max_plateau)
            eta_l = float(
                np.clip(self.eta + self.eta_depth_step * depth, 1e-6, effective_plateau)
            )
            mask_left = X[:, k] <= c
            # Soft re-weighting: rows satisfying rule keep ω, rows
            # violating receive ω · (1 − η). Zero-mass leaves dropped.
            omega_left = omega * (
                mask_left + (~mask_left).astype(float) * (1.0 - eta_l)
            )
            omega_right = omega * (~mask_left + mask_left.astype(float) * (1.0 - eta_l))

            # Dead-branch trim: if either child's effective mass is
            # essentially the parent's (modulo (1 − η) noise on every
            # row) then we have a degenerate split; convert to leaf.
            if (
                float(omega_left.sum()) <= 1e-9
                or float(omega_right.sum()) <= 1e-9
                or np.allclose(omega_left, omega * (1.0 - eta_l), atol=1e-12)
                or np.allclose(omega_right, omega * (1.0 - eta_l), atol=1e-12)
            ):
                self._nodes[node_idx] = ("leaf", mu_leaf, omega.copy())
                continue

            # Reserve placeholder slots for children; will be filled by
            # the queue-driven recursion below.
            left_idx = len(self._nodes)
            self._nodes.append(("placeholder",))
            right_idx = len(self._nodes)
            self._nodes.append(("placeholder",))
            self._nodes[node_idx] = ("split", k, c, left_idx, right_idx, eta_l)

            queue.append((left_idx, "left", omega_left, depth + 1))
            queue.append((right_idx, "right", omega_right, depth + 1))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_SlowGrowingTree":
        self._cols = tuple(X.columns)
        Xa = X.fillna(0.0).to_numpy(dtype=float)
        ya = np.asarray(y, dtype=float)
        n, K = Xa.shape
        self._train_y = ya
        self._fallback = float(ya.mean()) if ya.size else 0.0
        self._nodes = []
        if n == 0 or K == 0:
            self._nodes.append(("leaf", self._fallback, np.zeros(0)))
            return self
        self._build(Xa, ya)
        return self

    def _predict_one(self, x: np.ndarray) -> float:
        """Soft-weighted traversal: starting from the root with
        ω_test = 1, propagate through every split with the same
        (1 − η) penalty as training. The leaf weights ``ω_test_leaf``
        normalise to per-leaf membership probability; predict the
        weighted average of leaf-means."""

        if not self._nodes:
            return self._fallback
        # Root traversal: stack of (node_idx, weight)
        stack = [(0, 1.0)]
        out = 0.0
        total_w = 0.0
        while stack:
            idx, w = stack.pop()
            node = self._nodes[idx]
            kind = node[0]
            if kind == "leaf":
                out += w * float(node[1])
                total_w += w
                continue
            if kind == "split":
                _, k, c, left_idx, right_idx, eta_l = node
                if x[k] <= c:
                    stack.append((left_idx, w))
                    if eta_l < 1.0:
                        stack.append((right_idx, w * (1.0 - eta_l)))
                else:
                    stack.append((right_idx, w))
                    if eta_l < 1.0:
                        stack.append((left_idx, w * (1.0 - eta_l)))
                continue
            # Placeholder (should not happen post-fit)
            return self._fallback
        if total_w <= 1e-12:
            return self._fallback
        return out / total_w

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xa = (
            X.reindex(columns=list(self._cols), fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return np.array([self._predict_one(row) for row in Xa], dtype=float)


class _PrincipalComponentRegression:
    def __init__(self, n_components: int = 4) -> None:
        self.n_components = n_components
        self._mean: np.ndarray | None = None
        self._loadings: np.ndarray | None = None
        self._regression: LinearRegression | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_PrincipalComponentRegression":
        from sklearn.decomposition import PCA

        n = min(self.n_components, max(1, X.shape[1]), max(1, X.shape[0] - 1))
        self._mean = X.mean(axis=0).to_numpy()
        pca = PCA(n_components=n, random_state=0)
        scores = pca.fit_transform((X - self._mean).fillna(0.0).to_numpy())
        self._loadings = pca.components_
        self._regression = LinearRegression().fit(scores, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._regression is None or self._loadings is None or self._mean is None:
            return np.zeros(len(X))
        scores = (X - self._mean).fillna(0.0).to_numpy() @ self._loadings.T
        return self._regression.predict(scores)


class _FactorAugmentedVAR:
    """Bernanke-Boivin-Eliasz (2005) FAVAR.

    Two-stage estimator: extract ``n_factors`` principal components from
    a wide predictor panel, then fit a VAR(p) on the joint stack
    ``[factors, target]``. The coupling between factor dynamics and
    target dynamics is what distinguishes FAVAR from the simpler
    ``factor_augmented_ar`` (which only conditions on contemporaneous
    factors plus AR(p) on y).

    Issue #184. Promoted FUTURE -> OPERATIONAL in v0.2.
    """

    def __init__(self, p: int = 2, n_factors: int = 3) -> None:
        self.p = max(1, int(p))
        self.n_factors = max(1, int(n_factors))
        self._mean: np.ndarray | None = None
        self._loadings: np.ndarray | None = None
        self._var = _VARWrapper(p=self.p)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FactorAugmentedVAR":
        from sklearn.decomposition import PCA

        X_filled = X.fillna(0.0)
        if X_filled.shape[0] < max(self.p + 2, self.n_factors + 1):
            # Fallback to a plain linear fit when data is too short for both
            # PCA and a VAR.
            self._var = _VARWrapper(p=self.p)
            self._var.fit(X_filled, y)
            return self
        n = min(self.n_factors, X_filled.shape[1])
        self._mean = X_filled.mean(axis=0).to_numpy()
        pca = PCA(n_components=n, random_state=0)
        factors = pca.fit_transform(X_filled.to_numpy() - self._mean)
        self._loadings = pca.components_
        factor_frame = pd.DataFrame(
            factors,
            index=X_filled.index,
            columns=[f"factor_{i}" for i in range(n)],
        )
        # Fit a VAR on (factors, target). _VARWrapper internally appends y as
        # __y__ so the target's lagged dynamics couple to the factor lags.
        self._var = _VARWrapper(p=self.p)
        self._var.fit(factor_frame, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._loadings is None or self._mean is None:
            return self._var.predict(X.fillna(0.0))
        factors = (X.fillna(0.0).to_numpy() - self._mean) @ self._loadings.T
        factor_frame = pd.DataFrame(
            factors,
            index=X.index,
            columns=[f"factor_{i}" for i in range(self._loadings.shape[0])],
        )
        return self._var.predict(factor_frame)


class _VARWrapper:
    """Univariate-output VAR wrapper using statsmodels.tsa.vector_ar.var_model."""

    def __init__(self, p: int = 1) -> None:
        self.p = max(1, int(p))
        self._results = None
        self._target_name: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_VARWrapper":
        from statsmodels.tsa.api import VAR

        data = pd.concat([y.rename("__y__"), X], axis=1).dropna()
        if data.shape[0] <= self.p + 1 or data.shape[1] < 2:
            return self
        model = VAR(data)
        try:
            self._results = model.fit(self.p)
            self._target_name = "__y__"
        except Exception:  # pragma: no cover - statsmodels can fail on collinearity
            self._results = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._results is None or self._target_name is None:
            return np.zeros(len(X))
        forecast = self._results.forecast(self._results.endog[-self.p :], steps=1)
        target_index = self._results.names.index(self._target_name)
        return np.full(len(X), float(forecast[0, target_index]))


class _GLMBoost:
    """Componentwise L2-boosting (Bühlmann-Hothorn 2007) with linear base learners."""

    def __init__(self, n_iter: int = 100, learning_rate: float = 0.1) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_GLMBoost":
        x = X.fillna(0.0).to_numpy(dtype=float)
        residual = np.asarray(y, dtype=float) - float(np.mean(y))
        self.coef_ = np.zeros(x.shape[1])
        self.intercept_ = float(np.mean(y))
        for _ in range(self.n_iter):
            covariances = x.T @ residual
            best = int(np.argmax(np.abs(covariances)))
            denom = float(x[:, best] @ x[:, best])
            if denom == 0.0:
                break
            beta_step = self.learning_rate * float(covariances[best]) / denom
            self.coef_[best] += beta_step
            residual = residual - beta_step * x[:, best]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            return np.zeros(len(X))
        return X.fillna(0.0).to_numpy(dtype=float) @ self.coef_ + self.intercept_


class _IRFResultsLike:
    """Duck-compatible shim for statsmodels VARResults.irf().orth_irfs."""

    def __init__(self, orth_irfs: np.ndarray) -> None:
        self.orth_irfs = orth_irfs


class _FEVDResultsLike:
    """Duck-compatible shim for statsmodels VARResults.fevd().decomp."""

    def __init__(self, decomp: np.ndarray) -> None:
        self.decomp = decomp


def _build_cholesky_with_ordering(
    sigma_u: np.ndarray,
    names: list[str] | tuple[str, ...],
    ordering: tuple[str, ...] | None,
) -> np.ndarray:
    """Build a structural-impulse matrix ``P`` from ``Σ_u`` honoring an
    optional Cholesky identification ordering.

    With ``ordering = None`` returns ``np.linalg.cholesky(Σ_u)`` in the
    original column order (``Σ_u = P P'``).

    With ``ordering = (name_1, ..., name_K)`` (a permutation of
    ``names`` or a subset thereof — unmatched names go to the bottom of
    the recursive ordering), permute ``Σ_u`` to the user ordering,
    Cholesky-decompose, and undo the permutation so ``P[i, j]`` is the
    response of original-ordered variable ``i`` to a unit shock to the
    ``j``-th variable in the user ordering.

    Phase B-4 F3 (Coulombe & Göbel 2021 §3.c, footnote 12 ordering CO₂
    → TCC → PR → AT → SST → SIE/SIT/Albedo).
    """

    K = sigma_u.shape[0]
    sigma_reg = sigma_u + 1e-10 * np.eye(K)
    if ordering is None:
        try:
            return np.linalg.cholesky(sigma_reg)
        except np.linalg.LinAlgError:
            eig_vals, eig_vecs = np.linalg.eigh(sigma_reg)
            eig_vals = np.maximum(eig_vals, 0.0)
            return eig_vecs * np.sqrt(eig_vals)
    name_to_idx = {str(n): i for i, n in enumerate(names)}
    perm: list[int] = []
    seen: set[int] = set()
    for nm in ordering:
        if str(nm) in name_to_idx:
            idx = name_to_idx[str(nm)]
            if idx not in seen:
                perm.append(idx)
                seen.add(idx)
    # Append any unspecified names in original order (recursive default).
    for i in range(K):
        if i not in seen:
            perm.append(i)
    perm_arr = np.asarray(perm, dtype=int)
    # Permuted Σ_u in user ordering.
    sigma_perm = sigma_reg[np.ix_(perm_arr, perm_arr)]
    try:
        L_perm = np.linalg.cholesky(sigma_perm)
    except np.linalg.LinAlgError:
        eig_vals, eig_vecs = np.linalg.eigh(sigma_perm)
        eig_vals = np.maximum(eig_vals, 0.0)
        L_perm = eig_vecs * np.sqrt(eig_vals)
    # Map L_perm back so rows index original-ordered response vars and
    # columns index user-ordered shock vars. Row permutation only:
    # P_full[perm[k], :] = L_perm[k, :].
    P_full = np.zeros((K, K), dtype=float)
    for k in range(K):
        P_full[perm_arr[k], :] = L_perm[k, :]
    return P_full


class _MultiEquationBVARResults:
    """Multi-equation Minnesota BVAR results object that mimics the
    statsmodels VARResults attributes consumed by ``_var_impulse_frame``
    and ``_BayesianVAR._sample_posterior_irf``.

    Built by ``_BayesianVAR._fit_multivariate_minnesota``. Closes the
    audit gap "Minnesota posterior is single-equation only" by fitting
    one equation per endogenous variable with own-lag-1 anchor + cross-
    shrinkage prior, then exposing the joint coefficient matrix +
    Cholesky-orthogonalised IRF / FEVD builders.
    """

    def __init__(
        self,
        *,
        B: np.ndarray,
        sigma_u: np.ndarray,
        endog: np.ndarray,
        names: tuple[str, ...],
        k_ar: int,
        resid: np.ndarray,
        X_design: np.ndarray,
        posterior_cov_per_eq: np.ndarray | None = None,
        ordering: tuple[str, ...] | None = None,
    ) -> None:
        # ``params`` follows statsmodels VARResults.params layout:
        # (1 + K·p, K) with intercept in row 0, then K rows per lag block.
        self.params: np.ndarray = B.T
        self.sigma_u: np.ndarray = sigma_u
        self.endog: np.ndarray = endog
        self.names: list[str] = list(names)
        self.k_ar: int = int(k_ar)
        self.resid: np.ndarray = resid
        self._X_design: np.ndarray = X_design
        self._B: np.ndarray = B  # (K, 1 + K·p)
        # ``posterior_cov_per_eq`` shape: (K, 1+K·p, 1+K·p). Block-diag
        # Minnesota posterior covariance per equation. Used by
        # ``_BayesianVAR._sample_posterior_irf`` for paper-faithful
        # credible bands. None when computed from the OLS-VAR fallback.
        self.posterior_cov_per_eq: np.ndarray | None = posterior_cov_per_eq
        # Phase B-4 F3 Cholesky identification ordering (None => default
        # pandas-concat column order).
        self.ordering: tuple[str, ...] | None = (
            tuple(str(x) for x in ordering) if ordering is not None else None
        )

    def irf(self, n_periods: int) -> _IRFResultsLike:
        """Cholesky-orthogonalised impulse responses Θ_s = Φ_s · P from
        the BVAR posterior-mean coefficients. ``Φ_0 = I`` and
        ``Φ_s = Σ_{l=1..p} A_l · Φ_{s-l}``. ``P`` honors
        ``self.ordering`` when set (Phase B-4 F3)."""

        K = self._B.shape[0]
        p = self.k_ar
        A_list = [
            self._B[:, 1 + (lag - 1) * K : 1 + lag * K] for lag in range(1, p + 1)
        ]
        phi: list[np.ndarray] = [np.eye(K, dtype=float)]
        for s in range(1, int(n_periods) + 1):
            phi_s = np.zeros((K, K), dtype=float)
            for lag, A in enumerate(A_list, start=1):
                if lag <= s:
                    phi_s += A @ phi[s - lag]
            phi.append(phi_s)
        chol_P = _build_cholesky_with_ordering(self.sigma_u, self.names, self.ordering)
        orth = np.stack([Phi @ chol_P for Phi in phi], axis=0)
        return _IRFResultsLike(orth)

    def fevd(self, n_periods: int) -> _FEVDResultsLike:
        """Forecast-error variance decomposition. Share at horizon h+1
        attributable to shock j is the running sum of squared
        orthogonalised IRFs to that shock divided by the running total."""

        n_periods = max(int(n_periods), 1)
        irf_obj = self.irf(n_periods - 1)
        theta = irf_obj.orth_irfs[:n_periods]  # (n_periods, K, K)
        cumsq = np.cumsum(theta**2, axis=0)
        denom = cumsq.sum(axis=2, keepdims=True)
        decomp = cumsq / np.maximum(denom, 1e-12)
        return _FEVDResultsLike(decomp)


class _BayesianVAR:
    """Bayesian regression with Minnesota / Normal-Inverse-Wishart shrinkage.

    Issue #185 / #186 promotion -- the previous implementation delegated
    to a plain VAR wrapper. Now we implement the conjugate posterior:

    For ``y = X β + ε``, ε ~ N(0, σ²),

    * Prior (Minnesota): β ~ N(m, V) where ``m`` puts unit weight on the
      first own-lag column when present (random-walk for I(1) macro
      series) and zero elsewhere; ``V = diag(σ² λ₁² / (l ** λ_decay)²)``
      shrinks higher lags more aggressively (Litterman 1986).
    * Prior (NIW): same β prior plus an inverse-Wishart prior on σ²
      with shape ``ν₀`` and scale ``S₀``; the posterior mean of β has
      the same closed form but the marginal predictive distribution is
      Student-t. For the point-forecast contract used by L4, the two
      priors only differ in the σ² hyperparameter -- we expose
      ``bvar_normal_inverse_wishart`` with a heavier-tailed default.

    Closed-form posterior mean:

        β̂ = (V⁻¹ + X'X)⁻¹ (V⁻¹ m + X'y)

    Hyperparameters (override via ``params``):

    * ``minnesota_lambda1`` -- overall tightness (default 0.2)
    * ``minnesota_lambda_decay`` -- lag-decay exponent (default 1.0)
    * ``minnesota_lambda_cross`` -- cross-equation shrinkage (default 0.5)

    The estimator is closed-form so it adds no numerical-RNG state.
    """

    def __init__(
        self,
        p: int = 2,
        prior: str = "bvar_minnesota",
        lambda1: float = 0.2,
        lambda_decay: float = 1.0,
        lambda_cross: float = 0.5,
        b_AR: float = 1.0,
        n_draws: int = 500,
        posterior_irf_periods: int = 12,
        ordering: tuple[str, ...] | None = None,
        random_state: int = 0,
    ) -> None:
        # ``n_draws=500`` (default) activates posterior IRF / FEVD / HD by
        # default for the BVAR families. v0.9.0a0 audit-fix: the previous
        # default of 0 left ``_posterior_irf`` unpopulated, so L7 ops fell
        # back to the OLS-VAR IRF path. Paper-faithful reporting (Coulombe
        # & Göbel 2021 §3 credible regions) requires nonzero draws.
        # ``b_AR`` (Phase B-4 F4): Litterman random-walk anchor on the
        # own-lag-1 prior mean. ``b_AR=1.0`` recovers the I(1) random-walk
        # default; the VARCTIC paper Appx-A.3 calibrates ``b_AR=0.9``.
        # ``ordering`` (Phase B-4 F3): tuple of column names defining the
        # Cholesky structural-shock identification ordering. When set,
        # IRF / HD code paths reorder Σ_u to this ordering before
        # decomposition; the resulting frames carry the user-provided
        # column names.
        self.p = max(1, int(p))
        self.prior = str(prior)
        self.lambda1 = float(lambda1)
        self.lambda_decay = float(lambda_decay)
        self.lambda_cross = float(lambda_cross)
        self.b_AR = float(b_AR)
        self.n_draws = max(0, int(n_draws))
        self.posterior_irf_periods = max(1, int(posterior_irf_periods))
        self.ordering: tuple[str, ...] | None = (
            tuple(str(x) for x in ordering) if ordering is not None else None
        )
        self.random_state = int(random_state)
        # Larger NIW default tightness reflects the parameter-uncertainty
        # adjustment the marginal predictive would apply.
        if self.prior == "bvar_normal_inverse_wishart":
            self.lambda1 *= 1.25
        self._mean: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._feature_names: tuple[str, ...] = ()
        self._fallback: float = 0.0
        # v0.9.0F audit-fix: parallel statsmodels VAR fit so L7 IRF /
        # FEVD / historical_decomposition ops can route on the BVAR
        # posterior. None until ``fit`` runs.
        self._results: Any = None
        self._target_name: str | None = None

    @staticmethod
    def _classify_columns(columns: tuple[str, ...]) -> list[tuple[str, int, bool]]:
        """For each column, return ``(base_name, lag_index, is_first_own_lag)``.

        Lag detection looks for trailing ``_lagK`` (e.g. ``y_lag1``).
        Anything else is treated as a contemporaneous regressor (lag=0).
        """

        results: list[tuple[str, int, bool]] = []
        for col in columns:
            base = str(col)
            lag = 0
            if "_lag" in base:
                stem, _, suffix = base.rpartition("_lag")
                try:
                    lag = max(1, int(suffix))
                    base = stem
                except ValueError:
                    lag = 0
            results.append((base, lag, False))
        # Mark the first y_lag1 column as the random-walk anchor.
        for i, (base, lag, _) in enumerate(results):
            if base == "y" and lag == 1:
                results[i] = (base, lag, True)
                break
        return results

    def _prior(
        self, columns: tuple[str, ...], sigma2: float
    ) -> tuple[np.ndarray, np.ndarray]:
        classification = self._classify_columns(columns)
        m = np.zeros(len(columns))
        v = np.zeros(len(columns))
        for i, (_base, lag, is_anchor) in enumerate(classification):
            scale = (self.lambda1 / max(1.0, lag) ** self.lambda_decay) ** 2
            if is_anchor:
                m[i] = 1.0
                v[i] = sigma2 * scale
            else:
                v[i] = sigma2 * scale * (self.lambda_cross**2 if lag > 0 else 1.0)
        # Floor to avoid singular matrices.
        v = np.maximum(v, 1e-8)
        return m, v

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BayesianVAR":
        df = X.copy()
        df["__y__"] = y.values
        df = df.dropna()
        if df.empty:
            self._fallback = float(np.nan_to_num(y.mean(), nan=0.0))
            return self
        target = df["__y__"].to_numpy(dtype=float)
        Xmat = df.drop(columns="__y__").to_numpy(dtype=float)
        cols = tuple(df.drop(columns="__y__").columns)
        self._feature_names = cols
        # Sample variance estimate (data-dependent prior tightness).
        sigma2 = float(np.var(target, ddof=1)) if target.size > 1 else 1.0
        sigma2 = max(sigma2, 1e-6)
        m, v = self._prior(cols, sigma2)
        # Closed-form posterior mean: (V^-1 + X'X)^-1 (V^-1 m + X'y).
        XtX = Xmat.T @ Xmat
        Xty = Xmat.T @ target
        Vinv = np.diag(1.0 / v)
        precision = Vinv + XtX
        rhs = Vinv @ m + Xty
        try:
            beta = np.linalg.solve(precision, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(precision, rhs, rcond=None)[0]
        self._coef = beta
        self._intercept = float(target.mean() - Xmat.mean(axis=0) @ beta)
        self._mean = m
        self._fallback = float(target.mean())

        # v0.9.0a0 audit-fix: surface ``_results`` as a *true multi-
        # equation Minnesota BVAR* (one Minnesota-prior posterior per
        # endogenous variable) so L7 IRF / FEVD / HD ops route through
        # the BVAR rather than an OLS-VAR proxy. Earlier releases fitted
        # ``statsmodels.tsa.api.VAR`` here, leaving the audit observation
        # "Minnesota posterior is single-equation only" open. The new
        # path keeps every coefficient on the Bayesian posterior mean.
        try:
            full_panel = pd.concat([y.rename("__y__"), X], axis=1).dropna()
            if full_panel.shape[0] > self.p + 1 and full_panel.shape[1] >= 2:
                self._results = self._fit_multivariate_minnesota(full_panel, self.p)
                self._target_name = "__y__"
            else:
                self._results = None
                self._target_name = None
        except Exception:  # pragma: no cover - degenerate panels can fail
            self._results = None
            self._target_name = None

        # v0.9.0F gap-4 fix: posterior IRF sampling for credible bands
        # (paper Coulombe & Göbel 2021 §3 reports IRF + 90% credible
        # region from Gibbs draws). When ``n_draws > 0`` the wrapper
        # samples coefficient perturbations from the asymptotic
        # multivariate-Normal posterior ``vec(β) ~ N(β_OLS,
        # Σ_u ⊗ (X'X)⁻¹)`` (Bayesian flat-prior approximation; the
        # Minnesota multivariate posterior is heavier and would
        # require re-fitting the full BVAR system — deferred). Each
        # draw's IRF is computed via Sims (1980); we cache mean +
        # 16/84/5/95 percentile bands on ``self._posterior_irf``.
        self._posterior_irf: dict[str, np.ndarray] | None = None
        if self.n_draws > 0 and self._results is not None:
            try:
                self._posterior_irf = self._sample_posterior_irf(
                    n_draws=self.n_draws,
                    n_periods=self.posterior_irf_periods,
                )
            except Exception:  # pragma: no cover - sampling can fail on small panels
                self._posterior_irf = None
        return self

    def _fit_multivariate_minnesota(
        self, full_panel: pd.DataFrame, p: int
    ) -> _MultiEquationBVARResults | None:
        """Per-equation Minnesota posterior on the joint endogenous panel.

        For each endogenous variable ``y_i``, fit
        ``y_{i,t} = c_i + Σ_{l=1..p} A_l[i, :] Y_{t-l} + ε_{i,t}``
        with prior

            β_i ~ N(m_i, V_i)

        where ``m_i`` puts unit weight on own-lag-1 of variable i (the
        Litterman 1986 random-walk anchor for I(1) macro series) and
        zero elsewhere; ``V_i`` is diagonal with own-lag scale
        ``λ₁² / lag^{2λ_decay} · σ²_i`` and cross-lag scale further
        shrunk by ``λ_cross²`` and rescaled by the variance ratio
        ``σ²_i / σ²_j``. Closed-form posterior mean per equation:
        ``β̂_i = (V_i⁻¹ + Z'Z)⁻¹ (V_i⁻¹ m_i + Z' y_i)``.

        Returns ``_MultiEquationBVARResults`` or ``None`` if the panel
        cannot support a VAR(p) fit.
        """

        Y = full_panel.to_numpy(dtype=float)
        T_full, K = Y.shape
        if T_full <= p + 1 or K < 1:
            return None
        T_eff = T_full - p
        # Lag design: Z = [1, Y_{t-1}, ..., Y_{t-p}] for t = p..T-1.
        Z = np.zeros((T_eff, 1 + K * p), dtype=float)
        Z[:, 0] = 1.0
        for lag in range(1, p + 1):
            Z[:, 1 + (lag - 1) * K : 1 + lag * K] = Y[p - lag : T_full - lag]
        Y_eff = Y[p:]
        # Per-variable sample variance for prior scaling.
        sigma2_per_var = np.var(Y, axis=0, ddof=1)
        sigma2_per_var = np.maximum(sigma2_per_var, 1e-6)
        ZtZ = Z.T @ Z
        # Stack of posterior means + per-equation posterior covariance.
        n_coef = 1 + K * p
        B = np.zeros((K, n_coef), dtype=float)
        # posterior_cov_per_eq[i] = (V_i⁻¹ + Z'Z)⁻¹ · σ²_i (Minnesota
        # conjugate posterior with plug-in σ²_i Empirical Bayes).
        posterior_cov_per_eq = np.zeros((K, n_coef, n_coef), dtype=float)
        sigma2_resid_per_var = np.zeros(K, dtype=float)
        for i in range(K):
            m_i = np.zeros(n_coef, dtype=float)
            # Phase B-4 F4: own-lag-1 prior mean uses ``b_AR`` (paper
            # Coulombe & Göbel 2021 Appx-A.3 calibration; default 0.9
            # for VARCTIC 8). Earlier releases hard-coded 1.0
            # (Litterman 1986 random-walk anchor) which is the strict
            # I(1) prior; the paper softens this to allow stationary
            # mean reversion in the climate VAR.
            m_i[1 + i] = float(self.b_AR)
            v_i = np.zeros(n_coef, dtype=float)
            v_i[0] = 1e6
            for lag in range(1, p + 1):
                base_scale = (
                    self.lambda1 / max(1.0, float(lag)) ** self.lambda_decay
                ) ** 2
                for j in range(K):
                    idx = 1 + (lag - 1) * K + j
                    if j == i:
                        v_i[idx] = sigma2_per_var[i] * base_scale
                    else:
                        ratio = sigma2_per_var[i] / sigma2_per_var[j]
                        v_i[idx] = ratio * base_scale * (self.lambda_cross**2)
            v_i = np.maximum(v_i, 1e-8)
            Vinv = np.diag(1.0 / v_i)
            precision = Vinv + ZtZ
            rhs = Vinv @ m_i + Z.T @ Y_eff[:, i]
            try:
                beta_i = np.linalg.solve(precision, rhs)
                precision_inv = np.linalg.inv(precision)
            except np.linalg.LinAlgError:
                beta_i = np.linalg.lstsq(precision, rhs, rcond=None)[0]
                precision_inv = np.linalg.pinv(precision)
            B[i, :] = beta_i
            # Plug-in σ²_i from in-sample residuals.
            resid_i = Y_eff[:, i] - Z @ beta_i
            denom = max(T_eff - n_coef, 1)
            sigma2_i = max(float((resid_i**2).sum() / denom), 1e-12)
            sigma2_resid_per_var[i] = sigma2_i
            posterior_cov_per_eq[i] = precision_inv * sigma2_i
        fitted = Z @ B.T
        resid = Y_eff - fitted
        denom_full = max(T_eff - n_coef, 1)
        sigma_u = (resid.T @ resid) / float(denom_full)
        return _MultiEquationBVARResults(
            B=B,
            sigma_u=sigma_u,
            endog=Y,
            names=tuple(full_panel.columns),
            k_ar=p,
            resid=resid,
            X_design=Z,
            posterior_cov_per_eq=posterior_cov_per_eq,
            ordering=self.ordering,
        )

    def _sample_posterior_irf(
        self, *, n_draws: int, n_periods: int
    ) -> dict[str, np.ndarray]:
        """Draw ``n_draws`` from the multi-equation Minnesota posterior on
        the BVAR coefficients (block-diagonal across equations), compute
        Cholesky-orthogonalised IRF + FEVD per draw, and return mean +
        percentile bands.

        For multi-equation Minnesota (the v0.9.0a0 default), each
        equation's coefficient row is sampled from
        ``β_i ~ N(β̂_i, (V_i⁻¹ + Z'Z)⁻¹ · σ²_i)`` independently. The
        resulting per-draw coefficient matrix B is plugged into the
        same MA-representation IRF builder used at the posterior mean.

        Returns a dict with keys ``mean``, ``p05``, ``p16``, ``p84``,
        ``p95`` (IRF arrays of shape (n_periods+1, K, K)) plus
        ``fevd_mean``, ``fevd_p05``, ``fevd_p16``, ``fevd_p84``,
        ``fevd_p95`` of shape (n_periods, K, K).
        """
        results = self._results
        if results is None:
            raise RuntimeError("posterior IRF requires successful VAR fit")
        # Multi-equation Minnesota path: sample per-equation block-diag.
        if (
            isinstance(results, _MultiEquationBVARResults)
            and results.posterior_cov_per_eq is not None
        ):
            return self._sample_posterior_irf_multivariate_minnesota(
                results=results,
                n_draws=int(n_draws),
                n_periods=int(n_periods),
            )
        # Posterior covariance: vec(β) ~ N(β̂, Σ_u ⊗ (X'X)⁻¹).
        # statsmodels VAR exposes ``cov_params`` and ``params_dist`` but
        # for portability we reconstruct via Σ_u and design.
        sigma_u = np.asarray(results.sigma_u, dtype=float)
        endog = np.asarray(results.endog, dtype=float)
        K_var = endog.shape[1]
        # Reconstruct the lagged design matrix manually so we use ``pinv``
        # for ``(Z'Z)⁻¹`` — robust to near-singular panels (e.g. when X
        # already contains lagged y, common in macroforecast L3 outputs).
        # Z has shape (T_eff, 1 + K_var * k_ar) with intercept first.
        T_full = endog.shape[0]
        p = results.k_ar
        Z_rows = []
        for t in range(p, T_full):
            row = [1.0]
            for lag in range(1, p + 1):
                row.extend(endog[t - lag])
            Z_rows.append(row)
        Z = np.asarray(Z_rows, dtype=float)
        ZtZ_inv = np.linalg.pinv(Z.T @ Z + 1e-8 * np.eye(Z.shape[1]))
        cov_beta = np.kron(ZtZ_inv, sigma_u)
        beta_flat = np.asarray(results.params, dtype=float).flatten()
        # Sample n_draws perturbations
        rng = np.random.default_rng(self.random_state + 9173)
        try:
            chol_cov = np.linalg.cholesky(cov_beta + 1e-10 * np.eye(cov_beta.shape[0]))
        except np.linalg.LinAlgError:
            # Diagonalise as a fallback (e.g. nearly-singular covariance)
            eig_vals, eig_vecs = np.linalg.eigh(cov_beta)
            eig_vals = np.maximum(eig_vals, 0.0)
            chol_cov = eig_vecs * np.sqrt(eig_vals)
        irfs: list[np.ndarray] = []
        for _ in range(n_draws):
            z = rng.standard_normal(beta_flat.shape[0])
            # Skip per-draw refit (too expensive); compute IRF directly
            # from the perturbed coefficient matrices using the same
            # Σ_u (paper's Bayesian IRF holds Σ_u at posterior mean for
            # linearity). Reshape perturbation: VARResults.params has
            # shape (1 + K·p, K) with intercept in first row.
            pert = (chol_cov @ z).reshape(results.params.shape)
            # Compose perturbed coefficient matrices
            new_params = np.asarray(results.params, dtype=float) + pert
            # Use statsmodels.tsa.vector_ar.util.varsim or build IRF
            # manually via MA representation.
            # Simpler path: extract the VAR(p) coefficient matrices
            # A_1..A_p from new_params (skipping intercept row).
            A_stack = new_params[1:]  # ((K·p), K)
            A_list = [
                A_stack[i * K_var : (i + 1) * K_var, :].T for i in range(results.k_ar)
            ]
            # MA representation up to n_periods + 1
            irf = self._compute_orth_irfs(
                A_list,
                sigma_u,
                n_periods,
                names=results.names,
                ordering=getattr(self, "ordering", None),
            )
            irfs.append(irf)
        irf_arr = np.stack(irfs, axis=0)  # (n_draws, n_periods+1, K, K)
        return {
            "mean": irf_arr.mean(axis=0),
            "p05": np.percentile(irf_arr, 5, axis=0),
            "p16": np.percentile(irf_arr, 16, axis=0),
            "p84": np.percentile(irf_arr, 84, axis=0),
            "p95": np.percentile(irf_arr, 95, axis=0),
        }

    @staticmethod
    def _compute_orth_irfs(
        A_list: list[np.ndarray],
        sigma_u: np.ndarray,
        n_periods: int,
        *,
        names: list[str] | tuple[str, ...] | None = None,
        ordering: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Build Cholesky-orthogonalised IRFs from MA representation.

        Returns shape ``(n_periods+1, K, K)`` indexed by
        (horizon h, response_var i, shock_var j). When ``ordering`` is
        set the Cholesky factor ``P`` honors the structural-shock
        identification ordering (Phase B-4 F3).
        """
        K = sigma_u.shape[0]
        if ordering is None or names is None:
            chol_P = np.linalg.cholesky(sigma_u + 1e-10 * np.eye(K))
        else:
            chol_P = _build_cholesky_with_ordering(sigma_u, names, ordering)
        # MA coefficients: Φ_0 = I_K, Φ_s = Σ_{i=1..min(p,s)} A_i Φ_{s-i}
        phi = [np.eye(K)]
        for s in range(1, n_periods + 1):
            phi_s = np.zeros((K, K), dtype=float)
            for i, A in enumerate(A_list, start=1):
                if i <= s:
                    phi_s += A @ phi[s - i]
            phi.append(phi_s)
        # Orthogonalised IRF: Φ_s @ P
        return np.stack([Phi @ chol_P for Phi in phi], axis=0)

    def _sample_posterior_irf_multivariate_minnesota(
        self,
        *,
        results: "_MultiEquationBVARResults",
        n_draws: int,
        n_periods: int,
    ) -> dict[str, np.ndarray]:
        """Multi-equation Minnesota BVAR posterior sampling.

        Each equation i has independent posterior
        ``β_i ~ N(β̂_i, posterior_cov_per_eq[i])``. Per draw we sample
        β for all K equations, stack into a coefficient matrix B,
        compute the orthogonalised IRF, the FEVD, **and the historical
        decomposition** (Phase B-4 F7), and aggregate mean + 5/16/84/95
        percentile bands. The Cholesky factor of ``Σ_u`` honors
        ``results.ordering`` when set (Phase B-4 F3).
        """
        rng = np.random.default_rng(self.random_state + 9173)
        K = results._B.shape[0]
        p = results.k_ar
        sigma_u = results.sigma_u
        names = results.names
        ordering = results.ordering
        # Per-equation Cholesky factors of posterior covariance.
        cov_chols: list[np.ndarray] = []
        for i in range(K):
            cov_i = results.posterior_cov_per_eq[i]
            try:
                cov_chols.append(
                    np.linalg.cholesky(cov_i + 1e-10 * np.eye(cov_i.shape[0]))
                )
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(cov_i)
                eigvals = np.maximum(eigvals, 0.0)
                cov_chols.append(eigvecs * np.sqrt(eigvals))
        # Phase B-4 F7: historical decomposition draws. We hold Σ_u and
        # the structural-shock series at the posterior mean (ε* = P⁻¹ u
        # using the fitted residuals + posterior-mean Σ_u Cholesky) and
        # vary only the IRF coefficient draws. This isolates parameter
        # uncertainty in the IRF kernel from the residual realisation,
        # which is what the paper reports as the HD credible band
        # (Coulombe & Göbel 2021 §3.f). A fully Bayesian HD that also
        # samples Σ_u is deferred (multivariate IW posterior on Σ_u).
        chol_mean = _build_cholesky_with_ordering(sigma_u, names, ordering)
        try:
            structural_shocks_mean = np.linalg.solve(chol_mean, results.resid.T).T
        except np.linalg.LinAlgError:
            structural_shocks_mean = np.linalg.lstsq(
                chol_mean,
                results.resid.T,
                rcond=None,
            )[0].T
        T_resid = structural_shocks_mean.shape[0]
        irfs: list[np.ndarray] = []
        fevds: list[np.ndarray] = []
        hds: list[np.ndarray] = []
        # HD horizon — at most T_resid periods (HD is a per-time-series
        # decomposition, not a per-shock-horizon one).
        horizon_max_hd = max(int(n_periods), T_resid)
        for _ in range(int(n_draws)):
            B_draw = np.zeros_like(results._B)
            for i in range(K):
                z = rng.standard_normal(results._B.shape[1])
                B_draw[i, :] = results._B[i, :] + cov_chols[i] @ z
            A_list = [
                B_draw[:, 1 + (lag - 1) * K : 1 + lag * K] for lag in range(1, p + 1)
            ]
            irf = self._compute_orth_irfs(
                A_list,
                sigma_u,
                n_periods,
                names=names,
                ordering=ordering,
            )
            irfs.append(irf)
            # FEVD per draw: cumulative-sum-of-squared-IRFs share.
            theta = irf[:n_periods]
            cumsq = np.cumsum(theta**2, axis=0)
            denom = cumsq.sum(axis=2, keepdims=True)
            fevds.append(cumsq / np.maximum(denom, 1e-12))
            # HD per draw: convolve the per-draw IRF with the posterior-
            # mean structural shocks. Compute extended IRF for the full
            # T_resid horizon if needed.
            if horizon_max_hd + 1 > irf.shape[0]:
                irf_long = self._compute_orth_irfs(
                    A_list,
                    sigma_u,
                    horizon_max_hd,
                    names=names,
                    ordering=ordering,
                )
            else:
                irf_long = irf
            S = irf_long.shape[0]
            # HD shape: (T_resid, K_response, K_shock). We aggregate
            # the per-shock cumulative absolute contribution to each
            # response variable downstream — but cache the full tensor
            # so callers can target any variable.
            hd_draw = np.zeros((T_resid, K, K), dtype=float)
            for s in range(min(S, T_resid)):
                # Contribution of shock j (column) to response i (row)
                # arriving at time t: irf_long[s, i, j] * shocks[t-s, j].
                # Vectorise over (i, j).
                contrib = (
                    irf_long[s, :, :][None, :, :]
                    * structural_shocks_mean[: T_resid - s, None, :]
                )
                hd_draw[s:, :, :] += contrib
            hds.append(hd_draw)
        irf_arr = np.stack(irfs, axis=0)
        fevd_arr = np.stack(fevds, axis=0)
        hd_arr = np.stack(hds, axis=0)
        return {
            "mean": irf_arr.mean(axis=0),
            "p05": np.percentile(irf_arr, 5, axis=0),
            "p16": np.percentile(irf_arr, 16, axis=0),
            "p84": np.percentile(irf_arr, 84, axis=0),
            "p95": np.percentile(irf_arr, 95, axis=0),
            "fevd_mean": fevd_arr.mean(axis=0),
            "fevd_p05": np.percentile(fevd_arr, 5, axis=0),
            "fevd_p16": np.percentile(fevd_arr, 16, axis=0),
            "fevd_p84": np.percentile(fevd_arr, 84, axis=0),
            "fevd_p95": np.percentile(fevd_arr, 95, axis=0),
            "hd_mean": hd_arr.mean(axis=0),
            "hd_p05": np.percentile(hd_arr, 5, axis=0),
            "hd_p16": np.percentile(hd_arr, 16, axis=0),
            "hd_p84": np.percentile(hd_arr, 84, axis=0),
            "hd_p95": np.percentile(hd_arr, 95, axis=0),
            "hd_structural_shocks": structural_shocks_mean,
            "hd_chol_P": chol_mean,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            return np.full(len(X), self._fallback)
        # Re-align to the columns seen at fit time (silently zero unseen
        # columns so the predict path matches sklearn's expectation).
        Xmat = np.zeros((len(X), len(self._feature_names)), dtype=float)
        for i, col in enumerate(self._feature_names):
            if col in X.columns:
                Xmat[:, i] = (
                    pd.to_numeric(X[col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
        preds = self._intercept + Xmat @ self._coef
        return preds.astype(float)


class _TorchSequenceModel:
    """Tiny LSTM/GRU/Transformer regressor on lagged feature windows.

    Uses torch when available; otherwise falls back to a sklearn MLPRegressor
    so the operational schema status remains true (the recipe will still run
    end-to-end in lightweight installations).
    """

    def __init__(
        self,
        kind: str = "lstm",
        hidden_size: int = 32,
        n_epochs: int = 50,
        random_state: int = 0,
    ) -> None:
        self.kind = kind
        self.hidden_size = max(2, int(hidden_size))
        self.n_epochs = max(1, int(n_epochs))
        self.random_state = int(random_state)
        self._model = None
        # Note: no silent MLP fallback. fit() raises NotImplementedError if
        # torch is unavailable so the user picks family='mlp' deliberately.

    def _build_torch_model(self, n_features: int):
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        if self.kind == "lstm":
            cell = nn.LSTM(
                input_size=n_features, hidden_size=self.hidden_size, batch_first=True
            )
        elif self.kind == "gru":
            cell = nn.GRU(
                input_size=n_features, hidden_size=self.hidden_size, batch_first=True
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=n_features,
                nhead=1,
                dim_feedforward=self.hidden_size,
                batch_first=True,
                dropout=0.1,
            )
            cell = nn.TransformerEncoder(layer, num_layers=1)

        class _Wrapped(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cell = cell
                self.head = nn.Linear(
                    self.hidden_size if self.kind != "transformer" else n_features, 1
                )
                self.kind = self.kind  # type: ignore[assignment]

            def forward(self, x):
                out = self.cell(x)
                if self.kind in {"lstm", "gru"}:
                    return self.head(out[0][:, -1, :]).squeeze(-1)
                return self.head(out[:, -1, :]).squeeze(-1)

        # Re-bind self.kind / self.hidden_size into module factory closure
        wrapped_kind = self.kind
        wrapped_hidden = self.hidden_size

        class _Sequence(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cell = cell
                self.head = nn.Linear(
                    wrapped_hidden if wrapped_kind != "transformer" else n_features, 1
                )

            def forward(self, x):
                out = self.cell(x)
                if wrapped_kind in {"lstm", "gru"}:
                    return self.head(out[0][:, -1, :]).squeeze(-1)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Sequence()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TorchSequenceModel":
        try:
            import torch
            from torch import nn

            x_arr = X.fillna(0.0).to_numpy(dtype="float32")
            y_arr = np.asarray(y, dtype="float32")
            seq = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])
            tensor_x = torch.from_numpy(seq)
            tensor_y = torch.from_numpy(y_arr)
            model = self._build_torch_model(x_arr.shape[1])
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)
            loss_fn = nn.MSELoss()
            for _ in range(self.n_epochs):
                optim.zero_grad()
                preds = model(tensor_x)
                loss = loss_fn(preds, tensor_y)
                loss.backward()
                optim.step()
            self._model = model
        except ImportError as exc:
            # No silent MLP fallback: a sequence-model recipe asked for
            # ``lstm/gru/transformer`` because it wants the actual sequence
            # cell, not a feed-forward network. Raise a clear actionable
            # error so the user can install the [deep] extra (or pick mlp).
            raise NotImplementedError(
                f"family={self.kind!r} requires torch; install with `pip install 'macroforecast[deep]'` "
                f"or pick family='mlp' for a feed-forward network."
            ) from exc
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise NotImplementedError(
                f"family={self.kind!r} predict() called before fit() succeeded; "
                "install macroforecast[deep] (torch) and re-run the recipe."
            )
        import torch

        x_arr = X.fillna(0.0).to_numpy(dtype="float32")
        seq = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])
        with torch.no_grad():
            preds = self._model(torch.from_numpy(seq)).numpy()
        return preds


class _HemisphereNN:
    """Hemisphere Neural Networks (Goulet Coulombe / Frenette / Klieber
    2025 JAE) -- dual-head density forecaster with shared common core.

    Models ``y_{t+1} ~ N(f(X_t), g(X_t))`` (paper Eq. 1). The objective
    is the Gaussian negative log-likelihood (Eq. 2):

        ``min Σ_t (y_t − h_m(X_t))² / h_v(X_t) + log h_v(X_t)``

    The architecture (Eq. 5): a shared ``L_c`` common-core ReLU stack,
    then two hemispheres -- mean head ``h_m`` (linear output) and
    variance head ``h_v`` (softplus output).

    **Volatility emphasis** (paper §3.2): the dual-head loss is non-
    identified without a constraint. The paper enforces
    ``mean(h_v) / var(y) = ν`` *ex ante*; we implement this as a per-
    epoch rescaling of ``h_v`` (multiplicatively rebalance to the
    target mean). ``ν`` is data-driven from a plain-NN OOB residual
    proxy when not user-supplied (paper p.11 footnote 2; we cap at
    0.99). The ν proxy training is omitted in this v0.9.0C-2 minimal
    implementation -- ν is set to 0.5 by default (mid-range; users can
    override via ``params.nu``). The blocked-OOB log-linear reality
    check (paper Eq. 9-10) is an L7 follow-up, not in scope here.

    Hyperparameters (override via ``params``):
      * ``lc`` (= L_c; default 2) -- common-core depth
      * ``lm`` / ``lv`` (default 2 each) -- hemisphere depth past core
      * ``neurons`` (default 64; paper uses 400)
      * ``dropout`` (default 0.2)
      * ``lr`` (default 1e-3, Adam)
      * ``n_epochs`` (default 100; paper allows up to 100 with patience)
      * ``B`` (= number of blocked subsamples; default 50; paper uses
        1000 — reduced default for L4 cell-level cost)
      * ``sub_rate`` (= per-bag fraction; default 0.80)
      * ``nu`` (= variance-mean target; default 0.5; ``None`` triggers
        the plain-NN OOB proxy when implemented)
      * ``random_state`` (Adam + numpy seeds)

    Returns posterior-mean predictions; the variance head and the
    full per-bag ensemble are accessible via ``model._hv_path`` and
    ``model._ensemble_preds`` for downstream density-forecast use.
    """

    def __init__(
        self,
        lc: int = 2,
        lm: int = 2,
        lv: int = 2,
        neurons: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        n_epochs: int = 100,
        # v0.9.0F audit-fix: paper Eq. 8 uses B=1000; reduced default
        # to 100 (still 2× the previous 50) for L4 cell-level cost.
        B: int = 100,
        sub_rate: float = 0.80,
        nu: Any = None,
        lambda_emphasis: float = 1.0,
        patience: int = 15,
        val_frac: float = 0.20,
        random_state: int = 0,
    ) -> None:
        self.lc = max(1, int(lc))
        self.lm = max(1, int(lm))
        self.lv = max(1, int(lv))
        self.neurons = max(2, int(neurons))
        self.dropout = float(np.clip(dropout, 0.0, 0.9))
        self.lr = float(lr)
        self.n_epochs = max(1, int(n_epochs))
        self.B = max(1, int(B))
        self.sub_rate = float(np.clip(sub_rate, 0.1, 1.0))
        self.nu_target = nu  # may be None
        # v0.9.0F gap-4 fix: Lagrangian multiplier on the variance-
        # emphasis penalty term added to the MLE loss (in-MLE constraint
        # per paper §3.2 Ingredient 2).
        self.lambda_emphasis = float(max(lambda_emphasis, 0.0))
        # Paper §3 p.14: patience=15 and 80/20 train/val split for early stopping.
        self.patience = max(1, int(patience))
        self.val_frac = float(np.clip(val_frac, 0.05, 0.5))
        self.random_state = int(random_state)
        self._cols: tuple[str, ...] = ()
        self._models: list[Any] = []
        self._oob_preds_mean: np.ndarray | None = None
        self._oob_preds_var: np.ndarray | None = None
        self._fallback: float = 0.0
        self._var_y: float = 1.0
        # v0.9.0F audit-fix: log-linear reality-check coefficients (Eq. 9-10).
        self._reality_check_intercept: float = 0.0
        self._reality_check_slope: float = 1.0

    def _build_one_model(self, n_features: int):
        """Construct a single (core + mean + variance) torch module
        bundle returning (mean, log_var) at forward time."""
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise NotImplementedError(
                "mlp.architecture=hemisphere requires the [deep] extra "
                "(install macroforecast[deep] / torch>=2.0)."
            ) from exc

        def _stack(in_dim: int, out_dim: int, depth: int) -> nn.Sequential:
            layers = []
            current = in_dim
            for _ in range(depth):
                layers += [
                    nn.Linear(current, out_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
                current = out_dim
            return nn.Sequential(*layers)

        class HNN(nn.Module):
            def __init__(
                self, n_in: int, lc: int, lm: int, lv: int, neurons: int, dropout: float
            ):
                super().__init__()
                self.core = _stack(n_in, neurons, lc)
                self.head_m = nn.Sequential(
                    _stack(neurons, neurons, lm), nn.Linear(neurons, 1)
                )
                self.head_v_pre = nn.Sequential(
                    _stack(neurons, neurons, lv), nn.Linear(neurons, 1)
                )
                self.softplus = nn.Softplus()

            def forward(self, x):
                z = self.core(x)
                m = self.head_m(z).squeeze(-1)
                v = self.softplus(self.head_v_pre(z)).squeeze(-1) + 1e-6
                return m, v

        return HNN(n_features, self.lc, self.lm, self.lv, self.neurons, self.dropout)

    @staticmethod
    def _blocked_subsample(
        n: int, sub_rate: float, b: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Paper Eq. 8 / Ingredient 3: contiguous time-block subsamples.

        Splits the index 0..n−1 into K non-overlapping blocks; draws
        ``round(K · sub_rate)`` blocks at random; the in-bag indices
        are the union of the chosen blocks (contiguous), the OOB
        indices are the complement. v0.9.0F audit-fix replaces the
        previous random row sampling.
        """
        # Pick block size so K is in 5-25 range for stability.
        block_size = max(2, n // 20)
        starts = list(range(0, n, block_size))
        K = len(starts)
        n_in_blocks = max(1, int(round(sub_rate * K)))
        rng_state = np.random.default_rng(rng.integers(0, 2**63 - 1))
        chosen = rng_state.choice(K, n_in_blocks, replace=False)
        in_idx = np.concatenate(
            [np.arange(starts[k], min(starts[k] + block_size, n)) for k in chosen]
        )
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[in_idx] = False
        oob_idx = np.where(oob_mask)[0]
        return in_idx, oob_idx

    def _compute_nu_proxy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Paper Ingredient 2 (p.11 footnote 2): ν = mean(ε̂²_NN) / var(y),
        capped at 0.99. The proxy is computed from a plain feed-forward
        NN's blocked-OOB residuals (same architecture as one HNN bag's
        common core + mean head, no variance head). Falls back to 0.5
        when torch is unavailable."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return 0.5

        n, K = X.shape
        if n < 12 or K == 0:
            return 0.5
        var_y = float(y.var() + 1e-9)
        rng = np.random.default_rng(self.random_state + 7919)
        oob_preds_sum = np.zeros(n, dtype=float)
        oob_counts = np.zeros(n, dtype=float)
        n_proxy_bags = max(3, min(self.B // 4, 10))  # cheap proxy, ~3-10 bags
        for b in range(n_proxy_bags):
            in_idx, oob_idx = self._blocked_subsample(n, self.sub_rate, b, rng)
            if len(in_idx) < 4 or len(oob_idx) < 1:
                continue
            net = nn.Sequential(
                nn.Linear(K, max(8, self.neurons // 2)),
                nn.ReLU(),
                nn.Linear(max(8, self.neurons // 2), 1),
            )
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            X_in = torch.from_numpy(X[in_idx])
            y_in = torch.from_numpy(y[in_idx])
            for _ in range(min(self.n_epochs // 2, 30)):
                opt.zero_grad()
                pred = net(X_in).squeeze(-1)
                loss = ((y_in - pred) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                oob_pred = net(torch.from_numpy(X[oob_idx])).squeeze(-1).numpy()
            oob_preds_sum[oob_idx] += oob_pred
            oob_counts[oob_idx] += 1.0
        with np.errstate(invalid="ignore"):
            mask = oob_counts > 0
            if not mask.any():
                return 0.5
            avg_pred = np.zeros(n, dtype=float)
            avg_pred[mask] = oob_preds_sum[mask] / oob_counts[mask]
            eps2 = (y[mask] - avg_pred[mask]) ** 2
            nu = float(eps2.mean() / var_y)
        return float(np.clip(nu, 1e-3, 0.99))

    def _resolve_nu(
        self, X: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> float:
        """Return the variance-emphasis target ν. v0.9.0F audit-fix:
        when the user does not supply ``nu``, compute a paper-faithful
        proxy from a plain-NN OOB MSE (Ingredient 2, paper p.11
        footnote 2) instead of defaulting to 0.5."""
        if self.nu_target is not None:
            return float(np.clip(self.nu_target, 1e-3, 0.99))
        if X is None or y is None:
            return 0.5
        try:
            return self._compute_nu_proxy(X, y)
        except Exception:
            return 0.5

    def _apply_reality_check(
        self, hv_pred: np.ndarray, eps2: np.ndarray
    ) -> tuple[float, float]:
        """Paper Eq. 9-10 (Ingredient 4): log-linear regression of OOB
        squared residuals on log(h_v), then return the Eq. 10 rescaling
        coefficients (zeta_0, zeta_1) plus the bootstrap correction
        ``ς̂``. Returns (intercept_correction, slope) used to update
        h_v(X_test) = exp(ζ_0 + ζ_1 · log(h_v(X_test))) · ς̂.

        Output: ``(adj_intercept, adj_slope)`` such that
        ``h_v_corrected = exp(adj_intercept) · h_v_raw ** adj_slope``.
        """
        if hv_pred.size < 4 or eps2.size < 4:
            return 0.0, 1.0
        valid = (hv_pred > 1e-9) & (eps2 > 1e-9)
        if valid.sum() < 4:
            return 0.0, 1.0
        log_eps2 = np.log(eps2[valid])
        log_hv = np.log(hv_pred[valid])
        # OLS of log_eps2 on [1, log_hv]
        A = np.column_stack([np.ones_like(log_hv), log_hv])
        coef, *_ = np.linalg.lstsq(A, log_eps2, rcond=None)
        zeta_0, zeta_1 = float(coef[0]), float(coef[1])
        # Bootstrap correction ς̂ = E[exp(ξ_t)]: simple sample mean of
        # exp(residuals) (Goulet Coulombe et al. 2025 Eq. 10).
        delta = log_eps2 - (zeta_0 + zeta_1 * log_hv)
        sigma_hat = float(np.exp(delta).mean())
        adj_intercept = zeta_0 + np.log(max(sigma_hat, 1e-12))
        adj_slope = zeta_1
        return adj_intercept, adj_slope

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_HemisphereNN":
        try:
            import torch
        except ImportError as exc:
            raise NotImplementedError(
                "mlp.architecture=hemisphere requires the [deep] extra."
            ) from exc

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        self._cols = tuple(X.columns)
        Xa = X.fillna(0.0).to_numpy(dtype="float32")
        ya = np.asarray(y, dtype="float32")
        n, K = Xa.shape
        self._var_y = float(ya.var() + 1e-9)
        self._fallback = float(ya.mean()) if ya.size else 0.0
        if n < 4 or K == 0:
            return self
        # v0.9.0F audit-fix Ingredient 2: ν proxy from plain-NN OOB
        # residuals when not user-supplied (replaces previous default 0.5).
        nu = self._resolve_nu(Xa, ya)
        target_hv_mean = nu * self._var_y

        # Paper §3 p.14: single 80/20 train/val split for early stopping.
        # Split is computed ONCE here; all bags share the same validation set.
        val_size = max(1, round(self.val_frac * n))
        train_size = n - val_size
        # Shuffle indices using the same random state for reproducibility.
        split_rng = np.random.RandomState(self.random_state)
        perm = split_rng.permutation(n)
        train_idx = np.sort(perm[:train_size])   # sort to preserve time order within partition
        val_idx = np.sort(perm[train_size:])
        X_train = Xa[train_idx]
        y_train = ya[train_idx]
        X_val_np = Xa[val_idx]
        y_val_np = ya[val_idx]
        X_val_t = torch.from_numpy(X_val_np)
        y_val_t = torch.from_numpy(y_val_np)

        # v0.9.0F audit-fix Ingredient 3: blocked subsamples (paper
        # Eq. 8) replace the previous random row sampling.
        oob_eps2_sum = np.zeros(n, dtype=float)
        oob_hv_sum = np.zeros(n, dtype=float)
        oob_counts = np.zeros(n, dtype=float)
        # v0.9.0F gap-4 fix: Lagrangian penalty term added to the MLE
        # loss enforces the volatility-emphasis constraint in-MLE
        # rather than via post-batch bias rescaling. Audit gap closed:
        # previous per-epoch bias shift was an *approximation* of paper
        # §3.2 Ingredient 2; the Lagrangian penalty matches the paper's
        # "constrained MLE" framing exactly.
        lambda_emph = float(getattr(self, "lambda_emphasis", 1.0))

        for b in range(self.B):
            # Blocked subsample within the TRAINING PARTITION only.
            in_idx_local, oob_idx_local = self._blocked_subsample(
                train_size, self.sub_rate, b, rng
            )
            if len(in_idx_local) < 4:
                continue

            # Map local indices back to global dataset indices for OOB accumulation.
            in_idx_global = train_idx[in_idx_local]
            oob_idx_global = train_idx[oob_idx_local]

            X_b = torch.from_numpy(X_train[in_idx_local])
            y_b = torch.from_numpy(y_train[in_idx_local])
            X_full_t = torch.from_numpy(Xa)   # full dataset for Lagrangian penalty
            target_hv_mean_t = torch.tensor(target_hv_mean, dtype=torch.float32)

            model = self._build_one_model(K)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)

            # ----- Early stopping state (per-bag) -----
            best_val_loss = float("inf")
            epochs_since_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())

            for _ in range(self.n_epochs):
                # --- Training step ---
                model.train()
                opt.zero_grad()
                m, v = model(X_b)
                nll = ((y_b - m) ** 2 / v + torch.log(v)).mean()
                # Augmented Lagrangian penalty (paper §3.2 Ingredient 2,
                # in-MLE constraint): drive mean(h_v) toward ν · var(y)
                # via a soft squared penalty. λ_emphasis = 1.0 default;
                # users may scale via params.
                _, v_full = model(X_full_t)
                emph_penalty = (v_full.mean() - target_hv_mean_t) ** 2
                loss = nll + lambda_emph * emph_penalty
                loss.backward()
                opt.step()

                # --- Validation NLL (pure NLL, no Lagrangian penalty) ---
                model.eval()
                with torch.no_grad():
                    m_val, v_val = model(X_val_t)
                    val_nll = float(
                        ((y_val_t - m_val) ** 2 / v_val + torch.log(v_val)).mean().item()
                    )

                # --- Patience logic ---
                if val_nll < best_val_loss:
                    best_val_loss = val_nll
                    epochs_since_improvement = 0
                    best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= self.patience:
                    break   # early stopping triggered

            # Restore best-checkpoint weights for this bag.
            model.load_state_dict(best_state_dict)

            # --- OOB collection for Ingredient 4 reality check ---
            if len(oob_idx_global) > 0:
                with torch.no_grad():
                    model.eval()
                    m_oob, v_oob = model(torch.from_numpy(Xa[oob_idx_global]))
                    eps2 = (ya[oob_idx_global] - m_oob.numpy()) ** 2
                    oob_eps2_sum[oob_idx_global] += eps2
                    oob_hv_sum[oob_idx_global] += v_oob.numpy()
                    oob_counts[oob_idx_global] += 1.0

            self._models.append(model)

        # v0.9.0F audit-fix Ingredient 4: blocked-OOB log-linear reality
        # check (paper Eq. 9-10). Recover (adj_intercept, adj_slope) so
        # predict-time variance is corrected via
        #   h_v_test ← exp(adj_intercept) · h_v_test_raw ** adj_slope.
        with np.errstate(invalid="ignore", divide="ignore"):
            mask = oob_counts > 0
            if mask.any():
                avg_eps2 = oob_eps2_sum[mask] / oob_counts[mask]
                avg_hv = oob_hv_sum[mask] / oob_counts[mask]
                self._reality_check_intercept, self._reality_check_slope = (
                    self._apply_reality_check(avg_hv, avg_eps2)
                )
            else:
                self._reality_check_intercept, self._reality_check_slope = 0.0, 1.0
        return self

    def _ensemble_mean_and_variance(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) per row aggregated across the bagged
        ensemble. Variance has Eq. 10 reality check applied."""

        try:
            import torch
        except ImportError:
            n = len(X)
            return np.full(n, self._fallback, dtype=float), np.full(
                n, self._var_y, dtype=float
            )
        if not self._models:
            n = len(X)
            return np.full(n, self._fallback, dtype=float), np.full(
                n, self._var_y, dtype=float
            )
        Xa = (
            X.reindex(columns=list(self._cols), fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype="float32")
        )
        Xt = torch.from_numpy(Xa)
        means: list[np.ndarray] = []
        vars_: list[np.ndarray] = []
        with torch.no_grad():
            for model in self._models:
                model.eval()
                m, v = model(Xt)
                means.append(m.numpy())
                vars_.append(v.numpy())
        mean_pred = np.mean(np.column_stack(means), axis=1)
        var_raw = np.mean(np.column_stack(vars_), axis=1)
        # Eq. 10 (Goulet Coulombe / Frenette / Klieber 2025) reality check:
        #   h_v_corrected = exp(adj_intercept) · h_v_raw ** adj_slope
        # adj_intercept absorbs ζ_0 + log(ς̂); adj_slope = ζ_1.
        var_floor = np.maximum(var_raw, 1e-12)
        var_corrected = np.exp(self._reality_check_intercept) * np.power(
            var_floor, self._reality_check_slope
        )
        # Numerical guard: keep variance strictly positive and finite.
        var_corrected = np.where(
            np.isfinite(var_corrected) & (var_corrected > 0), var_corrected, var_raw
        )
        return mean_pred, var_corrected

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean_pred, _ = self._ensemble_mean_and_variance(X)
        return mean_pred

    def predict_variance(self, X: pd.DataFrame) -> np.ndarray:
        """Eq. 10 reality-check-corrected ensemble variance ``h_v(X)``.
        Returns an array of length ``len(X)``."""

        _, var_pred = self._ensemble_mean_and_variance(X)
        return var_pred

    def predict_distribution(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, variance)`` arrays per row -- the paper's
        Gaussian predictive distribution with Eq. 10 calibration."""

        return self._ensemble_mean_and_variance(X)

    def predict_quantiles(
        self, X: pd.DataFrame, levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> dict[float, np.ndarray]:
        """Gaussian quantile bands from the paper's calibrated distribution
        (mean head + Eq.-10-corrected variance head). Used by the L4
        ``forecast_object='quantile' | 'density'`` paths."""

        from scipy import stats as _stats

        mean_pred, var_pred = self._ensemble_mean_and_variance(X)
        sigma = np.sqrt(np.maximum(var_pred, 1e-12))
        out: dict[float, np.ndarray] = {}
        for q in levels:
            out[float(q)] = mean_pred + sigma * float(_stats.norm.ppf(float(q)))
        return out


class _DFMMixedFrequency:
    """Mariano-Murasawa-style dynamic factor model.

    Issues #188 + #245. ``mixed_frequency=True`` (or ``factor_order_mq``
    kwargs supplied) routes to ``statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ``,
    which implements the published Mariano-Murasawa (1996, 2010) monthly
    aggregator equation. Otherwise (single-frequency case) we use
    ``DynamicFactor`` -- the linear-Gaussian state-space MLE form.

    Mixed-frequency mode requires per-column frequency labels supplied
    via ``column_frequencies = {col: 'monthly' | 'quarterly'}``. The
    target series is treated as quarterly when its column appears in
    that map with the value ``'quarterly'``. Without the map we default
    to single-frequency (back-compat with v0.2 behaviour).
    """

    def __init__(
        self,
        n_factors: int = 1,
        factor_order: int = 1,
        mixed_frequency: bool = False,
        column_frequencies: dict[str, str] | None = None,
    ) -> None:
        self.n_factors = max(1, int(n_factors))
        self.factor_order = max(1, int(factor_order))
        self.mixed_frequency = bool(mixed_frequency) or bool(column_frequencies)
        self.column_frequencies = dict(column_frequencies or {})
        self._results = None
        self._fallback: float = 0.0
        self._mode: str = "single_frequency"
        self._scaler_mean = 0.0
        self._scaler_std = 1.0
        # Populated by ``_fit_mixed_frequency`` when MQ requested but
        # could not run (statsmodels error / insufficient data / no
        # monthly variables). Surfaces the diagnostic instead of silently
        # degrading to single-frequency. None when MQ succeeds or was
        # not requested.
        self._mq_failure_reason: str | None = None
        # Set to True/False by ``_fit_mixed_frequency`` on success so
        # callers can confirm the AR(1) idiosyncratic spec
        # (Mariano-Murasawa 2010 Eq. 4) was active.
        self._idiosyncratic_ar1: bool | None = None

    def _fit_mixed_frequency(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Issue #245 + v0.8.9 V2.4 fix -- ``DynamicFactorMQ`` with monthly
        aggregator (Mariano-Murasawa 2003). Returns True when the MQ fit
        succeeded; False when the caller should fall back to the single-
        frequency path. On failure, ``_mq_failure_reason`` is set so the
        diagnostic surfaces instead of silently degrading."""

        try:
            from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
        except ImportError:
            self._mq_failure_reason = "statsmodels DynamicFactorMQ unavailable"
            return False
        endog = pd.concat([y.rename("__y__"), X], axis=1).dropna(how="all")
        if endog.shape[0] < 12 or endog.shape[1] < 2:
            self._mq_failure_reason = "insufficient endog shape"
            return False
        # Build the M / Q split per documented column_frequencies; default
        # to monthly when unspecified.
        monthly = []
        quarterly = []
        for col in endog.columns:
            if col == "__y__":
                # Honour the target's declared frequency.
                tag = self.column_frequencies.get(
                    str(col)
                ) or self.column_frequencies.get("target", "monthly")
            else:
                tag = self.column_frequencies.get(str(col), "monthly")
            (quarterly if str(tag).lower() == "quarterly" else monthly).append(col)
        if not monthly:  # MQ requires at least one monthly variable
            self._mq_failure_reason = "no monthly variables declared"
            return False
        scaler_mean = endog.mean()
        scaler_std = endog.std(ddof=0).replace(0.0, 1.0)
        normalised = (endog - scaler_mean) / scaler_std
        # Issue #274 -- Mariano-Murasawa (2010) Eq. (4) specifies AR(1)
        # idiosyncratic errors. Try the published spec first; fall back
        # to the iid form when the optimisation diverges.
        last_exc: Exception | None = None
        for idiosyncratic_ar1 in (True, False):
            try:
                # v0.8.9 V2.4 honesty fix: when ``endog_quarterly`` is
                # supplied, statsmodels infers ``k_endog_monthly`` from the
                # primary endog shape and rejects an explicit value. The
                # v0.25 #245 implementation passed ``k_endog_monthly``
                # alongside ``endog_quarterly``, which made *every* MQ fit
                # raise ValueError -- the silent try/except then routed
                # users into the single-frequency fallback without warning.
                kwargs = dict(
                    factors=self.n_factors,
                    factor_orders=self.factor_order,
                    idiosyncratic_ar1=idiosyncratic_ar1,
                    standardize=False,
                )
                if quarterly:
                    # statsmodels DynamicFactorMQ requires the quarterly
                    # endog to carry a quarterly-frequency DateTimeIndex
                    # (``freqstr`` starting with 'Q'), not a monthly index
                    # with NaN at non-quarter-end months. Convert: drop
                    # the all-NaN rows in the quarterly block, then
                    # reindex to a quarterly DateTimeIndex with the freq
                    # attribute explicitly set so ``freqstr`` resolves.
                    q_block = normalised[quarterly].dropna(how="all")
                    if q_block.empty:
                        raise ValueError("quarterly block empty after NaN-row drop")
                    if isinstance(q_block.index, pd.DatetimeIndex):
                        q_block = q_block.copy()
                        # pandas 3.0 deprecates ``freq='Q'`` in favour of
                        # ``freq='QE'``; statsmodels' check inspects only
                        # the first character of ``freqstr`` so 'QE'
                        # passes the quarterly-frequency gate.
                        q_index = pd.date_range(
                            start=q_block.index[0],
                            periods=len(q_block),
                            freq="QE",
                        )
                        q_block.index = q_index
                    model = DynamicFactorMQ(
                        normalised[monthly],
                        endog_quarterly=q_block,
                        **kwargs,
                    )
                else:
                    model = DynamicFactorMQ(
                        normalised[monthly],
                        k_endog_monthly=len(monthly),
                        **kwargs,
                    )
                self._results = model.fit(disp=False, maxiter=20)
                self._idiosyncratic_ar1 = idiosyncratic_ar1
                last_exc = None
                break
            except Exception as exc:  # pragma: no cover - DFMQ is fragile on small data
                last_exc = exc
                continue
        else:
            self._mq_failure_reason = (
                f"DynamicFactorMQ fit failed: {type(last_exc).__name__}: {last_exc}"
            )
            return False
        self._scaler_mean = float(scaler_mean["__y__"])
        self._scaler_std = float(scaler_std["__y__"])
        self._mode = "mixed_frequency_mq"
        self._mq_failure_reason = None
        return True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_DFMMixedFrequency":
        if self.mixed_frequency and self._fit_mixed_frequency(X, y):
            return self
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

        endog = pd.concat([y.rename("__y__"), X], axis=1).dropna()
        min_obs = max(self.n_factors + self.factor_order + 3, 8)
        if endog.shape[0] < min_obs or endog.shape[1] < 2:
            self._fallback = float(np.nan_to_num(y.mean(), nan=0.0))
            return self
        scaler_mean = endog.mean()
        scaler_std = endog.std(ddof=0).replace(0.0, 1.0)
        normalised = (endog - scaler_mean) / scaler_std
        try:
            model = DynamicFactor(
                normalised,
                k_factors=min(self.n_factors, normalised.shape[1] - 1),
                factor_order=self.factor_order,
                error_order=0,
                error_cov_type="diagonal",
                enforce_stationarity=True,
            )
            self._results = model.fit(disp=False, maxiter=50)
            self._scaler_mean = float(scaler_mean["__y__"])
            self._scaler_std = float(scaler_std["__y__"])
            self._mode = "single_frequency"  # explicit: MQ fallback path confirmed
        except Exception:  # pragma: no cover - statsmodels can fail at edges
            self._results = None
            self._fallback = float(y.dropna().iloc[-1]) if not y.dropna().empty else 0.0
        if self._results is None and self._fallback == 0.0:
            self._fallback = float(np.nan_to_num(y.mean(), nan=0.0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._results is None:
            return np.full(len(X), self._fallback, dtype=float)
        try:
            forecast = self._results.forecast(steps=1)
            if isinstance(forecast, pd.DataFrame) and "__y__" in forecast.columns:
                target_pred = (
                    float(forecast["__y__"].iloc[0]) * self._scaler_std
                    + self._scaler_mean
                )
            else:
                # MQ returns a Series-like; pull the first value directly.
                target_pred = (
                    float(np.asarray(forecast).ravel()[0]) * self._scaler_std
                    + self._scaler_mean
                )
        except Exception:
            target_pred = self._fallback
        return np.full(len(X), target_pred, dtype=float)

    def predict_smoothed_factors(self) -> pd.DataFrame | None:
        """Single-frequency DFM: Kalman *smoother* marginal posterior of
        the latent factors. statsmodels exposes
        ``self._results.smoothed_state`` (k_states × T) — we reshape
        and only return the factor rows (one per ``k_factors``), which
        for the Mariano-Murasawa Eq. (4) state-space layout are the
        first ``k_factors`` rows of the state vector.

        Returns ``None`` if the fit failed or the results object does
        not expose ``smoothed_state`` (e.g. fallback path)."""

        if self._results is None:
            return None
        smoothed = getattr(self._results, "smoothed_state", None)
        if smoothed is None:
            return None
        try:
            # ``smoothed_state`` shape: (k_states, T). For DynamicFactor
            # the first ``k_factors`` rows are the factors themselves
            # (stacked across factor_order); we surface just the lag-0
            # factor block.
            arr = np.asarray(smoothed, dtype=float)
            if arr.ndim != 2:
                return None
            k_factors = int(getattr(self._results.model, "k_factors", arr.shape[0]))
            k_factors = max(1, min(k_factors, arr.shape[0]))
            factors = arr[:k_factors, :].T  # (T, k_factors)
            cols = [f"smoothed_factor_{i + 1}" for i in range(k_factors)]
            # Use the model's original index when available.
            try:
                idx = self._results.model.data.row_labels
            except Exception:
                idx = pd.RangeIndex(arr.shape[1])
            return pd.DataFrame(factors, index=idx, columns=cols)
        except Exception:
            return None


class _QuantileRegressionForest:
    """Issue #280 -- Meinshausen (2006) Quantile Regression Forest.

    Train a RandomForest, then for each leaf record the *empirical
    distribution* of training y values that landed in that leaf. At
    predict time, route a new sample to its leaf in every tree and
    average the per-leaf empirical CDFs to get the predictive CDF, then
    invert at each requested quantile level.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Any = None,
        random_state: int = 0,
        quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self._forest: RandomForestRegressor | None = None
        self._leaf_targets: list[dict[int, np.ndarray]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_QuantileRegressionForest":
        from sklearn.ensemble import RandomForestRegressor as _RF

        self._forest = _RF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=1,
        )
        X_filled = X.fillna(0.0)
        target = np.asarray(y, dtype=float)
        self._forest.fit(X_filled, target)
        leaves = self._forest.apply(X_filled)
        self._leaf_targets = []
        for tree_idx in range(leaves.shape[1]):
            tree_leaves = leaves[:, tree_idx]
            leaf_dict: dict[int, np.ndarray] = {}
            for leaf_id in np.unique(tree_leaves):
                leaf_dict[int(leaf_id)] = target[tree_leaves == leaf_id]
            self._leaf_targets.append(leaf_dict)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._forest is None:
            return np.zeros(len(X))
        X_filled = X.fillna(0.0)
        return self._forest.predict(X_filled)

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        if self._forest is None or not self._leaf_targets:
            return {q: np.zeros(len(X)) for q in self.quantile_levels}
        X_filled = X.fillna(0.0)
        leaves = self._forest.apply(X_filled)
        n_pred = len(X_filled)
        out: dict[float, np.ndarray] = {
            q: np.empty(n_pred) for q in self.quantile_levels
        }
        for i in range(n_pred):
            samples: list[float] = []
            for tree_idx in range(leaves.shape[1]):
                leaf_id = int(leaves[i, tree_idx])
                target_arr = self._leaf_targets[tree_idx].get(leaf_id, np.array([]))
                if target_arr.size:
                    samples.extend(target_arr.tolist())
            arr = np.asarray(samples, dtype=float) if samples else np.zeros(1)
            for q in self.quantile_levels:
                out[q][i] = float(np.quantile(arr, q))
        return out


class _BaggingWrapper:
    """Issue #282 -- bootstrap-aggregating meta-estimator over a base
    L4 family. Fits ``n_estimators`` models on bootstrap resamples of
    (X, y), averages their point predictions, and surfaces empirical
    quantile bands across the bag for ``predict_quantiles``.

    v0.9 Phase 1 (B-4) adds a ``strategy`` kwarg:

    * ``standard`` (default) -- i.i.d. bootstrap (Breiman 1996).
    * ``block`` -- moving-block bootstrap (Kunsch 1989). Resamples
      consecutive ``block_length``-row blocks until reaching
      ``sample_size``, preserving short-range serial dependence in the
      training rows. Used for serially-correlated macro panels and for
      the Taddy 2015 extension cited by Coulombe (2024) MRF.
    """

    def __init__(
        self,
        base_family: str = "ridge",
        n_estimators: int = 50,
        max_samples: float = 0.8,
        random_state: int = 0,
        base_params: dict[str, Any] | None = None,
        strategy: str = "standard",
        block_length: int = 4,
    ) -> None:
        self.base_family = str(base_family)
        self.n_estimators = int(n_estimators)
        self.max_samples = float(max_samples)
        self.random_state = int(random_state)
        self.base_params = dict(base_params or {})
        self.strategy = str(strategy)
        self.block_length = max(1, int(block_length))
        self._models: list[Any] = []

    def _draw_indices(
        self, rng: np.random.Generator, n: int, sample_size: int
    ) -> np.ndarray:
        if self.strategy == "block":
            # Moving-block bootstrap: draw ceil(sample_size / block_length)
            # block-start positions uniformly from {0, ..., n-1} and
            # extend each by block_length consecutive rows (wrapping at
            # n via modulo). Trim to sample_size.
            n_blocks = (sample_size + self.block_length - 1) // self.block_length
            starts = rng.integers(0, n, size=n_blocks)
            offsets = np.arange(self.block_length)
            idx = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)
            return idx[:sample_size]
        # Default: i.i.d. bootstrap
        return rng.choice(n, size=sample_size, replace=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BaggingWrapper":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        sample_size = max(2, int(round(self.max_samples * n)))
        for i in range(self.n_estimators):
            idx = self._draw_indices(rng, n, sample_size)
            sub_X = X.iloc[idx]
            sub_y = y.iloc[idx]
            params = dict(self.base_params)
            params["random_state"] = (self.random_state + i) % (2**31 - 1)
            model = _build_l4_model(self.base_family, params)
            try:
                model.fit(sub_X, sub_y)
                self._models.append(model)
            except Exception:  # pragma: no cover - skip flaky bag members
                continue
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models:
            return np.zeros(len(X))
        preds = np.column_stack([m.predict(X) for m in self._models])
        return preds.mean(axis=1)

    def predict_quantiles(
        self, X: pd.DataFrame, levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> dict[float, np.ndarray]:
        if not self._models:
            return {q: np.zeros(len(X)) for q in levels}
        preds = np.column_stack([m.predict(X) for m in self._models])
        return {q: np.quantile(preds, q, axis=1) for q in levels}


class _BoogingWrapper:
    """Goulet Coulombe (2024) "To Bag is to Prune" — Booging.

    Outer bagging of (intentionally over-fitted) inner Stochastic Gradient
    Boosted Trees + Data Augmentation. The bagging-prune theorem
    (Section 2 of the paper) replaces tuning the boosting depth ``S``
    with outer bagging: fix ``S`` to a high (over-fitting) value and let
    the bag average prune.

    Algorithm (paper §2.4 + Appendix A.2 p.39):

    1. **Data augmentation** — for each predictor column ``X_k`` append a
       noisy copy ``X̃_k = X_k + ε_k`` with ``ε_k ~ N(0, (σ_k / 3)²)``.
       Doubles the design width to 2K.
    2. **Outer bagging** — draw ``B`` subsamples of size
       ``sample_frac · n`` *without replacement*. For each, also drop a
       random ``da_drop_rate`` fraction of the (augmented) columns.
    3. **Inner SGB** — fit a ``GradientBoostingRegressor`` with
       ``n_estimators`` set high (over-fit, paper Appx-A.2 ``S = 1500``),
       ``max_depth = 3`` (paper §4.1 p.25), ``learning_rate = 0.1``,
       ``subsample = 0.5`` (row sub-sampling) for intra-boost
       stochasticity.
    4. **Predict** — augment test rows with the same noise SDs (fresh
       draws per call), restrict to per-bag kept columns, average the
       ``B`` per-bag predictions.

    The schema name ``sequential_residual`` is retained as a legacy alias
    for back-compat; the canonical option name is ``booging``.

    Hyperparameters (override via ``params``):
      * ``n_estimators`` (= outer ``B``; default 100, paper Appx-A.2 p.39)
      * ``max_samples`` (= ``sample_frac``; default 0.75)
      * ``inner_n_estimators`` (boosting steps ``S``; default 1500,
        paper Appx-A.2 p.39)
      * ``inner_learning_rate`` (default 0.1)
      * ``inner_max_depth`` (default 3, paper §4.1 p.25)
      * ``inner_subsample`` (intra-boost row fraction; default 0.5)
      * ``da_noise_frac`` (test-time noise SD as fraction of σ_k;
        default 1/3)
      * ``da_drop_rate`` (per-bag column-drop fraction; default 0.2)
      * ``random_state`` (propagated from L0 random_seed)
    """

    def __init__(
        self,
        B: int = 100,
        sample_frac: float = 0.75,
        inner_n_estimators: int = 1500,
        inner_learning_rate: float = 0.1,
        inner_max_depth: int = 3,
        inner_subsample: float = 0.5,
        da_noise_frac: float = 1.0 / 3.0,
        da_drop_rate: float = 0.2,
        random_state: int = 0,
    ) -> None:
        self.B = max(1, int(B))
        self.sample_frac = float(np.clip(sample_frac, 0.1, 1.0))
        self.inner_n_estimators = int(inner_n_estimators)
        self.inner_learning_rate = float(inner_learning_rate)
        self.inner_max_depth = int(inner_max_depth)
        self.inner_subsample = float(np.clip(inner_subsample, 0.1, 1.0))
        self.da_noise_frac = float(da_noise_frac)
        self.da_drop_rate = float(np.clip(da_drop_rate, 0.0, 0.95))
        self.random_state = int(random_state)
        self._cols: tuple[str, ...] = ()
        self._aug_cols: tuple[str, ...] = ()
        self._sigma_k: np.ndarray | None = None
        # List of (fitted_model, kept_aug_column_indices)
        self._models: list[tuple[Any, np.ndarray]] = []

    def _augment(self, X_arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Append noisy copies X̃_k = X_k + N(0, (σ_k · da_noise_frac)²)."""
        if self._sigma_k is None:
            return X_arr
        noise = rng.standard_normal(X_arr.shape) * (self._sigma_k * self.da_noise_frac)
        return np.hstack([X_arr, X_arr + noise])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BoogingWrapper":
        from sklearn.ensemble import GradientBoostingRegressor

        self._cols = tuple(X.columns)
        Xa = X.fillna(0.0).to_numpy(dtype=float)
        ya = np.asarray(y, dtype=float)
        n, K = Xa.shape
        if n == 0 or K == 0:
            return self
        self._sigma_k = Xa.std(axis=0, ddof=0).clip(min=1e-12)
        self._aug_cols = tuple(self._cols) + tuple(f"{c}__noisy" for c in self._cols)
        rng = np.random.default_rng(self.random_state)

        Xa_aug = self._augment(Xa, rng)  # (n, 2K)
        n_aug_cols = Xa_aug.shape[1]
        n_kept_cols = max(1, int(round((1.0 - self.da_drop_rate) * n_aug_cols)))
        sample_size = max(K + 1, int(round(self.sample_frac * n)))
        sample_size = min(sample_size, n)

        for b in range(self.B):
            row_idx = rng.choice(n, sample_size, replace=False)
            col_idx = rng.choice(n_aug_cols, n_kept_cols, replace=False)
            X_bag = Xa_aug[np.ix_(row_idx, col_idx)]
            y_bag = ya[row_idx]
            try:
                model = GradientBoostingRegressor(
                    n_estimators=self.inner_n_estimators,
                    learning_rate=self.inner_learning_rate,
                    max_depth=self.inner_max_depth,
                    subsample=self.inner_subsample,
                    random_state=self.random_state + b,
                )
                model.fit(X_bag, y_bag)
                self._models.append((model, col_idx))
            except Exception:
                continue
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models or self._sigma_k is None:
            return np.zeros(len(X), dtype=float)
        Xa = (
            X.reindex(columns=list(self._cols), fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        # Test-time augmentation: fresh noise draws using the stored σ_k.
        # Paper Appendix A.2 is silent on whether to reuse training noise;
        # the safer interpretation (independent fresh draws) preserves
        # the implicit Gaussian-perturbation regularisation.
        rng = np.random.default_rng(self.random_state + 9_999)
        Xa_aug = self._augment(Xa, rng)
        preds = np.zeros(len(X), dtype=float)
        for model, col_idx in self._models:
            preds += model.predict(Xa_aug[:, col_idx])
        return preds / max(len(self._models), 1)


class _MRFExternalWrapper:
    """Macroeconomic Random Forest (Coulombe 2024, arXiv:2006.12724) backed
    by Ryan Lucas's reference implementation, vendored under
    ``macroforecast/_vendor/macro_random_forest/`` with surgical numpy 2.x
    / pandas 2.x compatibility patches (no algorithmic changes — see
    ``_vendor/macro_random_forest/PATCHES.md``). Upstream:
    https://github.com/RyanLucas3/MacroRandomForest.

    The vendored package implements the full paper procedure:

    1. **Per-leaf local linear regression** of ``y`` on the state vector
       ``S`` at each tree leaf.
    2. **Random walk regularisation** on the time-varying coefficients
       (``rw_regul`` parameter) and Olympic-podium smoothing for the GTVP
       series.
    3. **Block Bayesian Bootstrap** (Taddy 2015 extension) forecast
       ensembles of size ``B``, surfaced via ``pred_ensemble``.

    The previous in-house ``_MRFWrapper`` implemented only piece (1); this
    wrapper switches the family to the authoritative paper-author code so
    the ``macroeconomic_random_forest`` family in macroforecast matches
    the published procedure.

    sklearn-style adapter: ``fit`` caches the training panel and ``predict``
    invokes ``MacroRandomForest._ensemble_loop()`` with the test rows
    appended to the cached training data and ``oos_pos`` set to the
    appended block. Each ``predict`` call therefore re-runs the full
    ensemble loop -- expensive but unavoidable given the paper algorithm
    is a fit-and-forecast pipeline.

    Citation: users running this family should cite Goulet Coulombe (2024)
    "The Macroeconomy as a Random Forest", Journal of Applied Econometrics
    (arXiv:2006.12724) and acknowledge the upstream implementation by
    Ryan Lucas (https://github.com/RyanLucas3/MacroRandomForest).
    """

    def __init__(
        self,
        B: int = 50,
        ridge_lambda: float = 0.1,
        rw_regul: float = 0.75,
        mtry_frac: float = 1 / 3,
        trend_push: float = 1,
        quantile_rate: float = 0.3,
        subsampling_rate: float = 0.75,
        fast_rw: bool = True,
        resampling_opt: int = 2,
        parallelise: bool = False,
        n_cores: int = 1,
        block_size: int = 24,
        random_state: int = 0,
    ) -> None:
        self.B = int(B)
        self.ridge_lambda = float(ridge_lambda)
        self.rw_regul = float(rw_regul)
        self.mtry_frac = float(mtry_frac)
        self.trend_push = float(trend_push)
        self.quantile_rate = float(quantile_rate)
        # Phase B-5 paper-5: per-tree subsample fraction inside the Block
        # Bayesian Bootstrap (Goulet Coulombe 2024 JAE p.10, default 0.75).
        # Surfaced via the ``macroeconomic_random_forest`` helper so the
        # paper calibration is reachable through the public API.
        self.subsampling_rate = float(subsampling_rate)
        self.fast_rw = bool(fast_rw)
        self.resampling_opt = int(resampling_opt)
        self.parallelise = bool(parallelise)
        self.n_cores = int(n_cores)
        # Block-bootstrap block length (Bayesian Bayesian Bootstrap step).
        # v0.9.0a0 audit-fix: default raised from upstream 12 → 24 to match
        # Goulet Coulombe (2024) JAE paper-spec for monthly macro panels.
        # User-overridable for quarterly (try 8) or weekly (try 52) data.
        self.block_size = int(block_size)
        self.random_state = int(random_state)
        self._train_X: pd.DataFrame | None = None
        self._train_y: pd.Series | None = None
        self._feature_columns: tuple[str, ...] = ()
        # Caches populated by the most recent ``predict`` call (for L7 GTVP
        # consumption and forecast-interval surfacing).
        self._cached_betas: np.ndarray | None = None
        self._cached_pred_ensemble: np.ndarray | None = None

    @staticmethod
    def _import_external():
        # Vendored under ``macroforecast/_vendor/macro_random_forest/`` --
        # always importable, no extra to install. License + upstream URL
        # preserved alongside the source.
        from macroforecast._vendor.macro_random_forest import MacroRandomForest

        return MacroRandomForest

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_MRFExternalWrapper":
        self._feature_columns = tuple(X.columns)
        self._train_X = X.copy()
        self._train_y = pd.Series(np.asarray(y, dtype=float), index=X.index, name="y")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._train_X is None or self._train_y is None or len(self._train_X) == 0:
            return np.zeros(len(X), dtype=float)

        MacroRandomForest = self._import_external()

        # Align test panel to training feature order; fill missing with 0.
        aligned_test = pd.DataFrame(index=X.index, columns=self._feature_columns)
        for col in self._feature_columns:
            if col in X.columns:
                aligned_test[col] = pd.to_numeric(X[col], errors="coerce")
        aligned_test = aligned_test.fillna(0.0)
        n_train = len(self._train_X)
        n_test = len(aligned_test)
        if n_test == 0:
            return np.zeros(0, dtype=float)

        # Build the data frame mrf-web expects: y first column, then features.
        # mrf-web rejects NaNs -- fill train and pad test y with 0.
        train_block = pd.concat(
            [
                self._train_y.rename("y"),
                self._train_X[list(self._feature_columns)].fillna(0.0),
            ],
            axis=1,
        )
        test_block = pd.DataFrame(
            np.column_stack(
                [np.zeros(n_test, dtype=float), aligned_test.to_numpy(dtype=float)]
            ),
            columns=["y", *self._feature_columns],
        )
        data = pd.concat(
            [train_block.reset_index(drop=True), test_block], ignore_index=True
        )
        oos_pos = list(range(n_train, n_train + n_test))
        n_features = len(self._feature_columns)
        feature_idx = np.arange(1, n_features + 1)

        # Global numpy seed propagation -- mrf-web does not expose a seed
        # kwarg, so we set the module-level RNG before construction. This
        # matches the L0 ``random_seed`` -> L4 estimator ``random_state``
        # contract used elsewhere in macroforecast (issue #215).
        np.random.seed(self.random_state)
        mrf = MacroRandomForest(
            data=data,
            y_pos=0,
            x_pos=feature_idx,
            S_pos=feature_idx,
            B=self.B,
            oos_pos=oos_pos,
            parallelise=self.parallelise,
            n_cores=self.n_cores,
            resampling_opt=self.resampling_opt,
            trend_push=self.trend_push,
            quantile_rate=self.quantile_rate,
            print_b=False,
            fast_rw=self.fast_rw,
            ridge_lambda=self.ridge_lambda,
            rw_regul=self.rw_regul,
            mtry_frac=self.mtry_frac,
            subsampling_rate=self.subsampling_rate,
            block_size=self.block_size,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            output = mrf._ensemble_loop()

        pred_frame = output.get("pred")
        if hasattr(pred_frame, "to_numpy"):
            preds = pred_frame.to_numpy(dtype=float).ravel()
        else:
            preds = np.asarray(pred_frame, dtype=float).ravel()

        betas = output.get("betas")
        if betas is not None:
            self._cached_betas = np.asarray(betas, dtype=float)
        ensemble = output.get("pred_ensemble")
        if ensemble is not None:
            self._cached_pred_ensemble = np.asarray(ensemble, dtype=float)
        return preds


# ---------------------------------------------------------------------------
# Phase C M9 -- GARCH(1,1) / EGARCH / Realized-GARCH univariate volatility
# family. Wraps the ``arch`` package (already an optional dep used by paper
# 8 2SRR). The L4 fit interface receives ``(X, y)``: ``y`` is the
# return-like series; ``X`` is ignored for ``garch11`` / ``egarch``. For
# ``realized_garch`` the column ``params['realized_variance']`` of X is
# consumed as the realised variance.
#
# ``predict(X)`` returns the conditional mean (μ for constant-mean,
# AR(p) projection for AR-mean) so the L4 point-forecast machinery sees a
# valid scalar per row. The variance forecast is exposed via the
# ``conditional_volatility_`` and ``forecast_variance_`` properties for
# downstream consumers (L7 vol-forecast importance, L6.E density tests).
# ---------------------------------------------------------------------------


class _GARCHFamily:
    """Univariate volatility model wrapper (Bollerslev 1986 / Nelson 1991 /
    Hansen et al. 2012). Wraps the ``arch`` package; raises a clear hint
    via ``NotImplementedError`` when ``arch`` is unavailable.

    The L4 fit interface receives ``(X, y)``. ``y`` is treated as the
    return-like series. ``X`` is **ignored** for ``garch11`` / ``egarch``.
    For ``realized_garch`` the column ``params['realized_variance']`` of X
    is consumed as the realised variance series (passed as exogenous to a
    GARCH(1,1) spec via ``arch_model(..., x=...)``; the closed-form
    Hansen-Huang-Tong-Wang 2012 measurement equation is approximated --
    full RealizedGARCH MLE awaits a future ``arch.RealizedGARCH`` API).

    ``predict(X)`` returns the **conditional mean** (constant μ from the
    fitted ARCH model, broadcast over ``len(X)``). Variance forecasts are
    exposed via ``predict_variance(h_steps)``.
    """

    def __init__(
        self,
        *,
        variant: str,
        p: int = 1,
        o: int = 0,
        q: int = 1,
        mean_model: str = "constant",
        dist: str = "normal",
        rescale: bool = False,
        random_state: int = 0,
        realized_variance: str | None = None,
    ) -> None:
        self.variant = variant
        self.p, self.o, self.q = p, o, q
        self.mean_model = mean_model
        self.dist = dist
        self.rescale = rescale
        self.random_state = random_state
        self.realized_variance_col = realized_variance
        self._fitted = None
        self._mu: float = 0.0
        self._last_var: float | None = None
        self._index: pd.Index | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_GARCHFamily":
        try:
            from arch import arch_model
        except ImportError as exc:  # pragma: no cover - optional dep
            raise NotImplementedError(
                f"{self.variant!r} family requires the optional dependency "
                f"`arch`. Install via `pip install arch>=6.0` or "
                f"`pip install macroforecast[arch]`."
            ) from exc

        r = pd.Series(y).astype(float).dropna()
        if len(r) < 30:
            raise NotImplementedError(
                f"{self.variant!r} family requires >= 30 observations; got {len(r)}."
            )

        if self.variant == "garch":
            am = arch_model(
                r,
                mean=self.mean_model,
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=self.rescale,
            )
        elif self.variant == "egarch":
            am = arch_model(
                r,
                mean=self.mean_model,
                vol="EGARCH",
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                rescale=self.rescale,
            )
        elif self.variant == "realized_garch":
            # Approximation: arch_model exogenous + GARCH(1,1). Full
            # Hansen-Huang-Tong-Wang 2012 RealizedGARCH MLE awaits an
            # ``arch.RealizedGARCH`` upstream wiring.
            rv = None
            if (
                self.realized_variance_col
                and isinstance(X, pd.DataFrame)
                and self.realized_variance_col in X.columns
            ):
                rv = (
                    pd.Series(X[self.realized_variance_col])
                    .astype(float)
                    .reindex(r.index)
                )
            if rv is None:
                # Fallback: use squared returns as a realised-variance proxy.
                rv = (r**2).rename("rv_proxy")
            am = arch_model(
                r,
                mean=self.mean_model,
                vol="GARCH",
                p=1,
                q=1,
                dist=self.dist,
                rescale=self.rescale,
                x=rv.to_frame() if isinstance(rv, pd.Series) else rv,
            )
        else:  # pragma: no cover - guarded by dispatch
            raise NotImplementedError(
                f"_GARCHFamily.variant={self.variant!r} not supported"
            )

        self._fitted = am.fit(disp="off", show_warning=False)
        params_dict = self._fitted.params.to_dict()
        self._mu = float(params_dict.get("mu", float(r.mean())))
        try:
            self._last_var = float(self._fitted.conditional_volatility.iloc[-1] ** 2)
        except Exception:
            self._last_var = None
        self._index = r.index
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # L4 point forecast = conditional mean μ.
        return np.full(len(X), self._mu, dtype=float)

    def predict_variance(self, h_steps: int = 1) -> np.ndarray:
        if self._fitted is None:
            return np.zeros(int(h_steps))
        try:
            forecast = self._fitted.forecast(horizon=int(h_steps), reindex=False)
            arr = np.asarray(forecast.variance.iloc[-1].to_numpy(), dtype=float)
            return arr
        except Exception:  # pragma: no cover - defensive fallback
            return np.full(int(h_steps), float(self._last_var or 1.0))

    @property
    def conditional_volatility_(self) -> np.ndarray | None:
        if self._fitted is None:
            return None
        return np.asarray(self._fitted.conditional_volatility, dtype=float)

    @property
    def params_(self) -> dict[str, float]:
        return self._fitted.params.to_dict() if self._fitted is not None else {}


# ---------------------------------------------------------------------------
# Phase C M16 -- ETS / Theta / Holt-Winters baseline family.
# - ETS: Hyndman-Koehler-Ord-Snyder (2008) state-space, statsmodels
#   ``ETSModel``.
# - Theta: Assimakopoulos-Nikolopoulos (2000) Theta(2) closed form
#   (linear-trend regression + simple exponential smoothing, blended
#   0.5 / 0.5 at theta=2).
# - Holt-Winters: statsmodels ``ExponentialSmoothing`` (trend + seasonal).
# ---------------------------------------------------------------------------


class _ETSWrapper:
    """Hyndman-Koehler-Ord-Snyder (2008) ETS state-space wrapper.

    Delegates to :class:`statsmodels.tsa.exponential_smoothing.ets.ETSModel`.
    The ``error_trend_seasonal`` 3-character code maps each component to
    add/mul/none. Default ``"AAN"`` = additive error + additive trend +
    no seasonal (the M-competition baseline for non-seasonal data).
    """

    def __init__(
        self,
        error_trend_seasonal: str = "AAN",
        seasonal_periods: int = 12,
        damped_trend: bool = False,
        initialization_method: str = "estimated",
        random_state: int = 0,
    ) -> None:
        if len(error_trend_seasonal) != 3:
            raise ValueError(
                f"error_trend_seasonal must be a 3-character code "
                f"(E ∈ {{A,M}}, T ∈ {{A,M,N}}, S ∈ {{A,M,N}}); got "
                f"{error_trend_seasonal!r}"
            )
        E, T, S = (
            error_trend_seasonal[0],
            error_trend_seasonal[1],
            error_trend_seasonal[2],
        )
        error_map = {"A": "add", "M": "mul"}
        comp_map = {"A": "add", "M": "mul", "N": None}
        if E not in error_map or T not in comp_map or S not in comp_map:
            raise ValueError(
                f"error_trend_seasonal {error_trend_seasonal!r} contains an "
                f"unsupported component code"
            )
        self.error = error_map[E]
        self.trend = comp_map[T]
        self.seasonal = comp_map[S]
        self.seasonal_periods = seasonal_periods if self.seasonal else None
        self.damped_trend = damped_trend
        self.initialization_method = initialization_method
        self.random_state = random_state
        self._fitted = None
        self._last_index = None
        self._fallback_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ETSWrapper":
        s = pd.Series(y).astype(float).dropna()
        self._fallback_mean = float(s.mean()) if len(s) > 0 else 0.0
        if len(s) < 4:
            self._fitted = None
            self._last_index = s.index[-1] if len(s) > 0 else None
            return self
        try:
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # type: ignore

            seasonal = self.seasonal
            seasonal_periods = self.seasonal_periods
            if seasonal is not None and (
                seasonal_periods is None or len(s) < 2 * int(seasonal_periods)
            ):
                seasonal = None
                seasonal_periods = None
            model = ETSModel(
                s,
                error=self.error,
                trend=self.trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=self.damped_trend,
                initialization_method=self.initialization_method,
            )
            self._fitted = model.fit(disp=False)
        except Exception:  # pragma: no cover - statsmodels failure -> fallback
            self._fitted = None
        self._last_index = s.index[-1] if len(s) > 0 else None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = max(len(X), 1)
        if self._fitted is None:
            return np.full(steps, self._fallback_mean, dtype=float)
        try:
            preds = self._fitted.forecast(steps=steps)
            return np.asarray(preds, dtype=float)
        except Exception:  # pragma: no cover
            return np.full(steps, self._fallback_mean, dtype=float)


class _ThetaWrapper:
    """Assimakopoulos-Nikolopoulos (2000) Theta(2) closed-form forecaster.

    For ``theta = 2.0`` (the M3-winner setup) the forecast equals the
    arithmetic mean of (a) a linear-trend extrapolation in the time
    index (the **Theta(0) line**) and (b) simple exponential smoothing
    applied to the **Theta(2) doubled-curvature transform**
    ``Y_t* = 2·Y_t − L_t`` (paper Eq. 6 / Eq. 9), where ``L_t = a + b·t``
    is the OLS linear regression line in time.

    Phase C-3 audit-fix (M16): the previous implementation applied SES
    directly to ``Y_t``, not to ``Y_t*``, which under-weights the
    curvature component the Theta decomposition is meant to amplify.
    The fix matches Assimakopoulos-Nikolopoulos (2000) §3 closed-form
    and tracks curvature on quadratic DGPs (vs slope-attenuated on
    linear-only DGPs).

    Both lines are estimated by closed-form: linear OLS for the trend;
    SES α via ``scipy.optimize.minimize_scalar`` over the SSE objective.
    """

    def __init__(
        self,
        theta: float = 2.0,
        seasonal: bool = False,
        seasonal_periods: int = 12,
    ) -> None:
        self.theta = float(theta)
        self.seasonal = bool(seasonal)
        self.seasonal_periods = int(seasonal_periods)
        self._a: float = 0.0
        self._b: float = 0.0
        self._alpha: float = 0.5
        self._level: float = 0.0
        self._n: int = 0
        self._fallback_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ThetaWrapper":
        s = pd.Series(y).astype(float).dropna()
        n = len(s)
        self._n = n
        self._fallback_mean = float(s.mean()) if n > 0 else 0.0
        if n < 2:
            self._a = self._fallback_mean
            self._b = 0.0
            self._level = self._fallback_mean
            self._alpha = 0.5
            return self
        t = np.arange(1, n + 1, dtype=float)
        A = np.column_stack([np.ones(n), t])
        try:
            beta, *_ = np.linalg.lstsq(A, s.to_numpy(), rcond=None)
            self._a, self._b = float(beta[0]), float(beta[1])
        except Exception:  # pragma: no cover
            self._a, self._b = self._fallback_mean, 0.0
        from scipy.optimize import minimize_scalar  # type: ignore

        s_arr = s.to_numpy()
        # Phase C-3 audit-fix (M16): doubled-curvature transform
        # ``Y_t* = 2·Y_t − L_t`` where L_t = a + b·t is the OLS line.
        # Per Assimakopoulos-Nikolopoulos (2000) Eq. 6/9, SES is applied
        # to ``Y_t*`` (NOT to raw ``Y_t``). The transform amplifies
        # short-term curvature ∝ θ = 2 while preserving the long-run
        # mean of the linear-trend component.
        L = self._a + self._b * t
        y_star = 2.0 * s_arr - L

        def ses_mse(alpha: float) -> float:
            level = float(y_star[0])
            loss = 0.0
            for v in y_star[1:]:
                level = alpha * float(v) + (1.0 - alpha) * level
                loss += (float(v) - level) ** 2
            return loss

        try:
            opt = minimize_scalar(ses_mse, bounds=(1e-3, 1.0 - 1e-3), method="bounded")
            self._alpha = float(opt.x)
        except Exception:  # pragma: no cover
            self._alpha = 0.5
        level = float(y_star[0])
        for v in y_star[1:]:
            level = self._alpha * float(v) + (1.0 - self._alpha) * level
        # ``self._level`` stores SES level on Y* (the doubled-curvature
        # series). The ``predict`` step combines this with the linear
        # trend extrapolation per Theta(2) blend.
        self._level = float(level)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = max(len(X), 1)
        out = np.empty(steps, dtype=float)
        for h_idx in range(1, steps + 1):
            # Theta(2)-line forecast = SES level on Y_t* extrapolated
            # with its residual OLS slope ``self._b``. Phase C-3b
            # audit-fix (Round 1): holding the level FLAT (pre-fix)
            # discarded the slope embedded in Y* and attenuated the
            # combined trend by 50%. Per Assimakopoulos-Nikolopoulos
            # (2000) Eq. 9 the Theta(2)-line is extrapolated along the
            # same OLS slope that defines the deflator L_t = a + b·t,
            # since Y*_t = 2 Y_t − L_t inherits ``b`` exactly when Y is
            # linear-plus-noise.
            ses_h = self._level + self._b * h_idx
            # Theta(0)-line forecast = linear trend extrapolation.
            trend_h = self._a + self._b * (self._n + h_idx)
            # Theta(2) blend: 0.5 weight each on the trend line
            # (theta=0) and the doubled-curvature SES path (theta=2).
            # Generalised theta could interpolate; the M3-winner uses
            # equal-weight blending (Assimakopoulos-Nikolopoulos 2000
            # Eq. 9).
            out[h_idx - 1] = 0.5 * trend_h + 0.5 * ses_h
        return out


class _HoltWintersWrapper:
    """Hyndman-Athanasopoulos (2018) §7 Holt-Winters wrapper.

    Delegates to :class:`statsmodels.tsa.holtwinters.ExponentialSmoothing`.
    Disables seasonality at fit time when the training series is shorter
    than ``2 * seasonal_periods`` to avoid statsmodels' insufficient-
    seasonal-data error.
    """

    def __init__(
        self,
        seasonal: str = "add",
        seasonal_periods: int = 12,
        trend: str | None = "add",
        damped_trend: bool = False,
    ) -> None:
        self.seasonal = seasonal
        self.seasonal_periods = int(seasonal_periods)
        self.trend = trend
        self.damped_trend = bool(damped_trend)
        self._fitted = None
        self._fallback_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_HoltWintersWrapper":
        s = pd.Series(y).astype(float).dropna()
        self._fallback_mean = float(s.mean()) if len(s) > 0 else 0.0
        if len(s) < 4:
            self._fitted = None
            return self
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore

            seasonal = self.seasonal
            seasonal_periods = self.seasonal_periods
            # Drop seasonality if length insufficient.
            if seasonal not in (None, "none") and len(s) < 2 * seasonal_periods:
                seasonal = None
                seasonal_periods = None
            else:
                seasonal_periods = (
                    seasonal_periods if seasonal not in (None, "none") else None
                )
                seasonal = seasonal if seasonal not in (None, "none") else None
            model = ExponentialSmoothing(
                s,
                trend=self.trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=self.damped_trend,
                initialization_method="estimated",
            )
            self._fitted = model.fit(optimized=True, use_brute=False)
        except Exception:  # pragma: no cover - statsmodels failure -> fallback
            self._fitted = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = max(len(X), 1)
        if self._fitted is None:
            return np.full(steps, self._fallback_mean, dtype=float)
        try:
            preds = self._fitted.forecast(steps=steps)
            return np.asarray(preds, dtype=float)
        except Exception:  # pragma: no cover
            return np.full(steps, self._fallback_mean, dtype=float)


def _add_l5_extended_metrics(metrics: "pd.DataFrame", errors: "pd.DataFrame") -> "pd.DataFrame":
    """F-P1-8 fix: compute medae, theil_u1, theil_u2, success_ratio per (model_id, target, horizon).

    ``errors`` must contain columns: model_id, target, horizon, y_true, y_pred, y_prev (y_prev may be None).
    Formulas:
      medae = median(|y_true - y_pred|)
      theil_u1 = sqrt(mean((y_true - y_pred)^2)) / (sqrt(mean(y_true^2)) + sqrt(mean(y_pred^2)))
      theil_u2 = sqrt(sum((y_pred - y_true)^2 / y_true_prev^2)) / sqrt(sum(((y_true - y_true_prev) / y_true_prev)^2))
                 denominator = naive forecast (no-change) scaled error; NaN when fewer than 2 obs with y_prev available
      success_ratio = mean(sign(y_pred - y_prev) == sign(y_true - y_prev)) for rows where y_prev is not None/NaN
    """
    import numpy as np

    rows_extended: list[dict] = []
    for (model_id, target, horizon), grp in errors.groupby(
        ["model_id", "target", "horizon"]
    ):
        y_true = grp["y_true"].to_numpy(dtype=float)
        y_pred = grp["y_pred"].to_numpy(dtype=float)
        # medae
        medae = float(np.median(np.abs(y_true - y_pred)))
        # theil_u1
        rmse_forecast = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        denom_u1 = float(np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2)))
        theil_u1 = rmse_forecast / denom_u1 if denom_u1 > 0 else float("nan")
        # theil_u2 and success_ratio use y_prev — filter to rows where y_prev is available
        if "y_prev" in grp.columns:
            has_prev = grp["y_prev"].notna()
            grp_prev = grp[has_prev]
        else:
            grp_prev = grp.iloc[0:0]  # empty
        if len(grp_prev) >= 2:
            yt = grp_prev["y_true"].to_numpy(dtype=float)
            yp = grp_prev["y_pred"].to_numpy(dtype=float)
            yp_prev = grp_prev["y_prev"].to_numpy(dtype=float)
            # Avoid division by zero from zero y_prev
            safe_prev = np.where(np.abs(yp_prev) > 0, yp_prev, float("nan"))
            num_u2 = np.nansum(((yp - yt) / safe_prev) ** 2)
            den_u2 = np.nansum(((yt - yp_prev) / safe_prev) ** 2)
            theil_u2 = float(np.sqrt(num_u2 / den_u2)) if den_u2 > 0 else float("nan")
            sign_pred = np.sign(yp - yp_prev)
            sign_true = np.sign(yt - yp_prev)
            success_ratio = float(np.mean(sign_pred == sign_true))
        else:
            theil_u2 = float("nan")
            success_ratio = float("nan")
        rows_extended.append(
            {
                "model_id": model_id,
                "target": target,
                "horizon": horizon,
                "medae": medae,
                "theil_u1": theil_u1,
                "theil_u2": theil_u2,
                "success_ratio": success_ratio,
            }
        )
    if not rows_extended:
        return metrics
    ext = pd.DataFrame(rows_extended)
    return metrics.merge(ext, on=["model_id", "target", "horizon"], how="left")


def _add_l5_relative_metrics(
    metrics: pd.DataFrame, l4_models: L4ModelArtifactsArtifact | None
) -> pd.DataFrame:
    if l4_models is None:
        return metrics
    benchmark_ids = [
        model_id
        for model_id, is_benchmark in l4_models.is_benchmark.items()
        if is_benchmark
    ]
    if len(benchmark_ids) != 1:
        return metrics
    benchmark_id = benchmark_ids[0]
    benchmark = metrics.loc[
        metrics["model_id"] == benchmark_id, ["target", "horizon", "mse", "mae"]
    ].rename(columns={"mse": "benchmark_mse", "mae": "benchmark_mae"})
    if benchmark.empty:
        return metrics
    result = metrics.merge(benchmark, on=["target", "horizon"], how="left")
    result["relative_mse"] = result["mse"] / result["benchmark_mse"]
    result["r2_oos"] = 1.0 - result["relative_mse"]
    result["relative_mae"] = result["mae"] / result["benchmark_mae"]
    result["mse_reduction"] = result["benchmark_mse"] - result["mse"]
    return result


def _l5_ranking_metric(metrics: pd.DataFrame, resolved_axes: dict[str, Any]) -> str:
    if (
        resolved_axes.get("ranking") == "by_relative_metric"
        and "relative_mse" in metrics.columns
    ):
        return "relative_mse"
    primary = resolved_axes.get("primary_metric", "mse")
    return primary if primary in metrics.columns else "mse"


def _l5_rank_ascending(metric: str) -> bool:
    return metric not in {"r2_oos", "mse_reduction"}


def _l5_per_subperiod_metrics(
    per_origin: pd.DataFrame,
    l4_models: L4ModelArtifactsArtifact | None,
    boundaries: list[Any],
) -> pd.DataFrame:
    """Issue #258 -- split MSE / MAE by user-defined date boundaries.

    ``boundaries`` is a list of ISO timestamps (or anything pandas can
    parse). The function partitions the per-origin loss panel into
    subperiods and reports per-(model, target, horizon, subperiod) MSE /
    MAE, with the implicit final segment running to the last origin.
    """

    if per_origin.empty:
        return pd.DataFrame()
    if not boundaries:
        return (
            per_origin.assign(subperiod="full_oos")
            .groupby(["model_id", "target", "horizon", "subperiod"], as_index=False)
            .agg(mse=("squared_error", "mean"), mae=("absolute_error", "mean"))
        )
    cuts = sorted(pd.to_datetime(boundaries))
    origins = pd.to_datetime(per_origin["origin"])
    edges = [pd.Timestamp.min, *cuts, pd.Timestamp.max]
    labels = [f"sp_{i}" for i in range(len(edges) - 1)]
    subperiod = pd.cut(origins, bins=edges, labels=labels, include_lowest=True)
    expanded = per_origin.assign(subperiod=subperiod.astype(str))
    return expanded.groupby(
        ["model_id", "target", "horizon", "subperiod"], as_index=False
    ).agg(mse=("squared_error", "mean"), mae=("absolute_error", "mean"))


def _l5_predictor_block_decomposition(
    metrics: pd.DataFrame,
    block_map: dict[str, list[str] | tuple[str, ...]],
    *,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    """Issues #258 + #275 -- attribute model loss to predictor blocks.

    When ``X`` / ``y`` are supplied this fits a fresh OLS on each
    coalition of blocks and computes the resulting in-sample MSE; the
    Shapley share of block ``B`` is the average marginal MSE-reduction
    when adding ``B`` to a coalition. Exhaustive enumeration for
    ``k <= 7`` blocks; sampling Shapley over 200 permutations otherwise
    (Castro-Gomez-Tejada 2009).

    When ``X`` / ``y`` are unavailable, falls back to the v0.25
    size-proportional Shapley as a documented proxy.
    """

    from itertools import combinations
    from math import comb

    blocks = list(block_map.keys())
    rows: list[dict[str, Any]] = []
    can_refit = (
        isinstance(X, pd.DataFrame)
        and isinstance(y, pd.Series)
        and not X.empty
        and len(y) > len(blocks) + 1
    )
    for (target, horizon), group in metrics.groupby(["target", "horizon"]):
        n = len(blocks)
        if n == 0:
            continue
        median_mse = float(group["mse"].median())
        if can_refit:
            from sklearn.linear_model import LinearRegression

            X_aligned = X.fillna(0.0)
            y_aligned = y.dropna()
            X_aligned = X_aligned.loc[X_aligned.index.intersection(y_aligned.index)]
            y_aligned = y_aligned.loc[X_aligned.index]

            def _coalition_mse(subset_indices: tuple[int, ...]) -> float:
                cols: list[str] = []
                for k in subset_indices:
                    cols.extend(
                        c for c in block_map[blocks[k]] if c in X_aligned.columns
                    )
                if not cols:
                    return float(np.var(y_aligned))
                fitted = LinearRegression().fit(X_aligned[cols], y_aligned)
                preds = fitted.predict(X_aligned[cols])
                return float(np.mean((y_aligned.to_numpy() - preds) ** 2))

            if n <= 7:
                shares = np.zeros(n)
                for size in range(n):
                    for subset in combinations(range(n), size):
                        coalition_loss = _coalition_mse(subset)
                        weight = 1.0 / (n * comb(n - 1, size))
                        for i in range(n):
                            if i in subset:
                                continue
                            new_loss = _coalition_mse(tuple(list(subset) + [i]))
                            shares[i] += weight * (coalition_loss - new_loss)
            else:
                rng = np.random.default_rng(0)
                shares = np.zeros(n)
                n_perm = 200
                for _ in range(n_perm):
                    perm = rng.permutation(n)
                    running: list[int] = []
                    prev_loss = _coalition_mse(())
                    for idx in perm:
                        running.append(int(idx))
                        new_loss = _coalition_mse(tuple(running))
                        shares[idx] += prev_loss - new_loss
                        prev_loss = new_loss
                shares /= n_perm
        else:
            sizes = np.asarray([len(block_map[b]) for b in blocks], dtype=float)
            total_size = float(sizes.sum()) if sizes.sum() > 0 else 1.0
            shares = np.zeros(n)
            if n <= 8:
                for size in range(n):
                    for subset in combinations(range(n), size):
                        coalition_share = (
                            sum(sizes[k] for k in subset) / total_size
                            if subset
                            else 0.0
                        )
                        weight = 1.0 / (n * comb(n - 1, size))
                        for i in range(n):
                            if i in subset:
                                continue
                            new_share = (
                                sum(sizes[k] for k in subset) + sizes[i]
                            ) / total_size
                            shares[i] += weight * (new_share - coalition_share)
            else:
                shares = sizes / total_size
        # Normalise so the shares sum to 1 (allocation property).
        total_share = float(shares.sum())
        if total_share > 0:
            normalised = shares / total_share
        else:
            normalised = shares
        for block_name, share, raw_share in zip(blocks, normalised, shares):
            rows.append(
                {
                    "target": target,
                    "horizon": int(horizon),
                    "block": block_name,
                    "shapley_share": float(share),
                    "block_mse_contribution": float(
                        raw_share if can_refit else share * median_mse
                    ),
                    "method": "refit_per_subset" if can_refit else "size_proportional",
                }
            )
    return pd.DataFrame(rows)


def materialize_l5_minimal(
    recipe_root: dict[str, Any],
    l1_artifact: L1DataDefinitionArtifact,
    l3_features: L3FeaturesArtifact,
    l4_forecasts: L4ForecastsArtifact,
    l4_models: L4ModelArtifactsArtifact | None = None,
) -> L5EvaluationArtifact:
    raw = recipe_root.get("5_evaluation", {"fixed_axes": {}}) or {"fixed_axes": {}}
    has_benchmark = bool(l4_models and any(l4_models.is_benchmark.values()))
    context = {
        "forecast_object": l4_forecasts.forecast_object,
        "target_structure": l1_artifact.target_structure,
        "regime_definition": l1_artifact.regime_definition,
        "has_fred_sd": bool(l1_artifact.dataset and "fred_sd" in l1_artifact.dataset),
        "has_benchmark": has_benchmark,
    }
    report = l5_layer.validate_layer(raw, context=context)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    # Issue #277 -- evaluate on the raw target scale when a transformer
    # was applied at L3. ``raw_data`` is set in that case; otherwise the
    # transformed and raw series are the same.
    raw_actual = l3_features.y_final.metadata.values.get("raw_data")
    actual = (
        raw_actual
        if isinstance(raw_actual, pd.Series)
        else l3_features.y_final.metadata.values.get("data")
    )
    if not isinstance(actual, pd.Series):
        raise ValueError("minimal L5 runtime requires L3 y_final series data")
    # Inverse-transform the L4 forecasts back to the raw scale.
    forecasts_raw = _apply_inverse_target_transform(l4_forecasts.forecasts, l3_features)
    rows: list[dict[str, Any]] = []
    for (model_id, target, horizon, origin), forecast in forecasts_raw.items():
        if origin not in actual.index:
            continue
        y_true = float(actual.loc[origin])
        y_pred = float(forecast)
        error = y_true - y_pred
        # F-P1-8 fix: store y_true/y_pred/y_prev for extended metrics (medae, theil, direction)
        actual_index = actual.index
        origin_pos = actual_index.get_loc(origin)
        y_prev = float(actual.iloc[origin_pos - 1]) if origin_pos > 0 else None
        rows.append(
            {
                "model_id": model_id,
                "target": target,
                "horizon": horizon,
                "origin": origin,
                "squared_error": error**2,
                "absolute_error": abs(error),
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prev": y_prev,
            }
        )
    if not rows:
        metrics = pd.DataFrame(
            columns=["model_id", "target", "horizon", "mse", "rmse", "mae"]
        )
        per_origin = pd.DataFrame(
            columns=[
                "model_id",
                "target",
                "horizon",
                "origin",
                "squared_error",
                "absolute_error",
            ]
        )
    else:
        errors = pd.DataFrame(rows)
        per_origin = errors[
            [
                "model_id",
                "target",
                "horizon",
                "origin",
                "squared_error",
                "absolute_error",
            ]
        ].copy()
        metrics = errors.groupby(["model_id", "target", "horizon"], as_index=False).agg(
            mse=("squared_error", "mean"), mae=("absolute_error", "mean")
        )
        metrics["rmse"] = metrics["mse"] ** 0.5
        # F-P1-8 fix: compute extended metrics (medae, theil_u1, theil_u2, success_ratio)
        metrics = _add_l5_extended_metrics(metrics, errors)
        metrics = _add_l5_relative_metrics(metrics, l4_models)
    if metrics.empty:
        ranking = pd.DataFrame()
        resolved_axes = dict(
            l5_layer.resolve_axes_from_raw(
                raw.get("fixed_axes", {}) or {}, context=context
            )
        )
    else:
        resolved_axes = dict(
            l5_layer.resolve_axes_from_raw(
                raw.get("fixed_axes", {}) or {}, context=context
            )
        )
        ranking_metric = _l5_ranking_metric(metrics, resolved_axes)
        ranking = metrics.sort_values(
            ranking_metric, ascending=_l5_rank_ascending(ranking_metric)
        ).assign(
            rank_method="by_primary_metric",
            rank_value=lambda frame: range(1, len(frame) + 1),
        )
    # Issue #258 -- decomposition / oos_period / aggregation matrix.
    # Compute per-axis tables when the recipe enables them; expose via
    # ``l5_axis_resolved`` so the L8 export and L7 lineage layers can
    # consume them without re-deriving.
    if not per_origin.empty:
        oos_period = resolved_axes.get("oos_period", "full_oos")
        decomp_target = resolved_axes.get("decomposition_target", "none")
        agg_horizon = resolved_axes.get("agg_horizon", "per_horizon_separate")
        agg_target = resolved_axes.get("agg_target", "per_target_separate")
        decomposition_tables: dict[str, Any] = {}
        if oos_period == "multiple_subperiods":
            leaf = raw.get("leaf_config", {}) or {}
            boundaries = leaf.get("oos_period_boundaries") or []
            decomposition_tables["per_subperiod"] = _l5_per_subperiod_metrics(
                per_origin, l4_models, boundaries
            )
        if decomp_target == "by_predictor_block":
            leaf = raw.get("leaf_config", {}) or {}
            block_map = leaf.get("predictor_blocks", {}) or {}
            if block_map:
                decomposition_tables["by_predictor_block"] = (
                    _l5_predictor_block_decomposition(
                        metrics,
                        block_map,
                        X=l3_features.X_final.data,
                        y=l3_features.y_final.metadata.values.get("data")
                        if isinstance(
                            l3_features.y_final.metadata.values.get("data"), pd.Series
                        )
                        else None,
                    )
                )
        if agg_horizon == "per_horizon_then_mean" and not metrics.empty:
            decomposition_tables["per_horizon_then_mean"] = metrics.groupby(
                "horizon", as_index=False
            ).mean(numeric_only=True)
        if agg_target == "top_k_worst" and not metrics.empty:
            k = int((raw.get("leaf_config", {}) or {}).get("top_k_worst", 5))
            decomposition_tables["top_k_worst"] = metrics.nlargest(k, "mse")
        if decomposition_tables:
            resolved_axes["decomposition_tables"] = decomposition_tables
    return L5EvaluationArtifact(
        per_origin_loss_panel=per_origin,
        metrics_table=metrics,
        ranking_table=ranking,
        l5_axis_resolved=resolved_axes,
    )


def materialize_l6_runtime(
    recipe_root: dict[str, Any],
    l1_artifact: L1DataDefinitionArtifact,
    l3_features: L3FeaturesArtifact,
    l4_forecasts: L4ForecastsArtifact,
    l4_models: L4ModelArtifactsArtifact,
    l5_eval: L5EvaluationArtifact,
) -> tuple[L6TestsArtifact, dict[str, Any]]:
    raw = recipe_root.get("6_statistical_tests", {}) or {}
    context = {
        "forecast_object": l4_forecasts.forecast_object,
        "has_benchmark": any(l4_models.is_benchmark.values()),
        "benchmark_count": sum(1 for value in l4_models.is_benchmark.values() if value),
        "model_ids": tuple(l4_models.artifacts),
        "regime_definition": l1_artifact.regime_definition,
        "horizons": set(l4_forecasts.horizons),
    }
    report = l6_layer.validate_layer(raw, context=context)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    resolved = l6_layer.resolve_axes_from_raw(raw, context=context)
    axes = _plain_axes(resolved)
    if not resolved.get("enabled"):
        return L6TestsArtifact(
            test_metadata={"runtime": "core_l6_disabled"}, l6_axis_resolved=axes
        ), axes

    actual = l3_features.y_final.metadata.values.get("data")
    if not isinstance(actual, pd.Series):
        raise ValueError("minimal L6 runtime requires L3 y_final series data")
    errors = _l6_error_frame(l4_forecasts, actual)
    equal_results: dict[tuple[Any, ...], Any] = {}
    nested_results: dict[tuple[Any, ...], Any] = {}
    cpa_results: dict[tuple[Any, ...], Any] = {}
    multiple_results: dict[str, Any] = {}
    direction_results: dict[tuple[Any, ...], Any] | None = None
    residual_results: dict[tuple[Any, ...], Any] | None = None
    density_results: dict[tuple[Any, ...], Any] | None = None

    if resolved["L6_A_equal_predictive"]["enabled"]:
        equal_results = _l6_equal_predictive_results(
            errors,
            resolved["L6_A_equal_predictive"],
            raw.get("leaf_config", {}) or {},
            l4_models,
        )
    if resolved["L6_B_nested"]["enabled"]:
        nested_results = _l6_nested_results(
            errors, resolved["L6_B_nested"], raw.get("leaf_config", {}) or {}, l4_models
        )
    if resolved["L6_C_cpa"]["enabled"]:
        cpa_results = _l6_cpa_results(errors, resolved["L6_C_cpa"], l4_models)
    if resolved["L6_D_multiple_model"]["enabled"]:
        multiple_results = _l6_multiple_model_results(
            l5_eval.metrics_table,
            resolved["L6_D_multiple_model"],
            per_origin_panel=l5_eval.per_origin_loss_panel,
        )
    if resolved["L6_E_density_interval"]["enabled"]:
        # Issue #200 -- runs PIT-Berkowitz / KS / Christoffersen / Kupiec
        # against a quantile forecast panel. When forecast_object is point,
        # we synthesize an empirical normal density from training residuals
        # so the tests still produce something traceable; otherwise the
        # forecast_intervals from L4 (issue #201) are used directly.
        density_results = _l6_density_interval_results(
            l4_forecasts, l1_artifact, l3_features, resolved["L6_E_density_interval"]
        )
    if resolved["L6_F_direction"]["enabled"]:
        direction_results = _l6_direction_results(
            errors, resolved["L6_F_direction"], raw.get("leaf_config", {}) or {}
        )
    if resolved["L6_G_residual"]["enabled"]:
        residual_results = _l6_residual_results(errors, resolved["L6_G_residual"])

    return (
        L6TestsArtifact(
            equal_predictive_results=equal_results,
            nested_results=nested_results,
            cpa_results=cpa_results,
            multiple_model_results=multiple_results,
            density_results=density_results,
            direction_results=direction_results,
            residual_results=residual_results,
            test_metadata={
                "runtime": "core_l6_minimal",
                "n_error_rows": len(errors),
                "nw_bandwidth_used": {"rule": axes.get("dependence_correction")},
            },
            l6_axis_resolved=axes,
        ),
        axes,
    )


def materialize_l7_runtime(
    recipe_root: dict[str, Any],
    l3_features: L3FeaturesArtifact,
    l3_metadata: L3MetadataArtifact,
    l4_forecasts: L4ForecastsArtifact,
    l4_models: L4ModelArtifactsArtifact,
    l5_eval: L5EvaluationArtifact,
    l6_tests: L6TestsArtifact | None = None,
) -> tuple[L7ImportanceArtifact, L7TransformationAttributionArtifact, dict[str, Any]]:
    raw = recipe_root.get("7_interpretation", {}) or {}
    report = l7_layer.validate_layer(
        raw, recipe_context=l7_layer._recipe_context(recipe_root)
    )
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    resolved = l7_layer.resolve_axes_from_raw(raw)
    axes = _plain_axes(resolved)
    if not resolved.get("enabled"):
        return (
            L7ImportanceArtifact(computation_metadata={"runtime": "core_l7_disabled"}),
            L7TransformationAttributionArtifact(),
            axes,
        )
    values = _execute_l7_nodes(
        raw, l3_features, l3_metadata, l4_forecasts, l4_models, l5_eval, l6_tests
    )
    importance = L7ImportanceArtifact(
        computation_metadata={"runtime": "core_l7_minimal", "axis_resolved": axes}
    )
    transform = L7TransformationAttributionArtifact()
    sinks = raw.get("sinks", {}) or {}
    if "l7_importance_v1" in sinks:
        global_importance: dict[tuple[Any, ...], Any] = {}
        group_importance: dict[tuple[Any, ...], Any] = {}
        lineage_importance: dict[tuple[Any, ...], Any] = {}
        for label, node_ids in _l7_sink_targets(sinks["l7_importance_v1"]).items():
            for node_id in node_ids:
                value = values.get(node_id)
                if isinstance(value, pd.DataFrame):
                    method = value.attrs.get("method", node_id)
                    model_id = value.attrs.get("model_id", "model")
                    target = value.attrs.get("target", l3_features.y_final.name)
                    horizon = value.attrs.get(
                        "horizon",
                        l3_features.horizon_set[0] if l3_features.horizon_set else 1,
                    )
                    key = (model_id, target, int(horizon), method)
                    if "group" in value.columns or label.startswith("group"):
                        group_importance[
                            key + (value.attrs.get("grouping", label),)
                        ] = value
                    elif "pipeline" in value.columns or label.startswith("pipeline"):
                        lineage_importance[key + (value.attrs.get("level", label),)] = (
                            value
                        )
                    else:
                        global_importance[key] = value
        importance = L7ImportanceArtifact(
            global_importance=global_importance,
            group_importance=group_importance,
            lineage_importance=lineage_importance,
            computation_metadata={"runtime": "core_l7_minimal", "axis_resolved": axes},
        )
    if "l7_transformation_attribution_v1" in sinks:
        target_id = sinks["l7_transformation_attribution_v1"]
        value = values.get(target_id) if isinstance(target_id, str) else None
        if isinstance(value, L7TransformationAttributionArtifact):
            transform = value
    return (
        importance,
        transform,
        axes,
    )


def _execute_l7_nodes(
    raw: dict[str, Any],
    l3_features: L3FeaturesArtifact,
    l3_metadata: L3MetadataArtifact,
    l4_forecasts: L4ForecastsArtifact,
    l4_models: L4ModelArtifactsArtifact,
    l5_eval: L5EvaluationArtifact,
    l6_tests: L6TestsArtifact | None,
) -> dict[str, Any]:
    dag = l7_layer.normalize_to_dag_form(raw)
    values: dict[str, Any] = {}
    for node in _topological_nodes(dag):
        if node.type == "source":
            values[node.id] = _execute_l7_source(
                node.selector,
                l3_features,
                l3_metadata,
                l4_forecasts,
                l4_models,
                l5_eval,
                l6_tests,
            )
        elif node.type == "step":
            inputs = [values[ref.node_id] for ref in node.inputs]
            values[node.id] = _execute_l7_step(
                node.op, inputs, node.params, l3_features, l3_metadata, l5_eval
            )
    return values


def _execute_l7_source(
    selector,
    l3_features: L3FeaturesArtifact,
    l3_metadata: L3MetadataArtifact,
    l4_forecasts: L4ForecastsArtifact,
    l4_models: L4ModelArtifactsArtifact,
    l5_eval: L5EvaluationArtifact,
    l6_tests: L6TestsArtifact | None,
) -> Any:
    if selector is None:
        raise ValueError("L7 source node requires a selector")
    subset = selector.subset or {}
    if selector.layer_ref == "l4" and selector.sink_name == "l4_model_artifacts_v1":
        model_id = subset.get("model_id")
        if model_id:
            return l4_models.artifacts[model_id]
        return l4_models
    if selector.layer_ref == "l4" and selector.sink_name == "l4_forecasts_v1":
        return l4_forecasts
    if selector.layer_ref == "l3" and selector.sink_name == "l3_features_v1":
        component = subset.get("component")
        if component == "X_final":
            return l3_features.X_final.data.copy()
        if component == "y_final":
            return l3_features.y_final.metadata.values.get("data")
        return l3_features
    if selector.layer_ref == "l3" and selector.sink_name == "l3_metadata_v1":
        return l3_metadata
    if selector.layer_ref == "l5" and selector.sink_name == "l5_evaluation_v1":
        return l5_eval
    if selector.layer_ref == "l6" and selector.sink_name == "l6_tests_v1":
        return l6_tests
    raise NotImplementedError(
        f"minimal L7 runtime does not support source {selector.layer_ref}.{selector.sink_name}"
    )


def _execute_l7_step(
    op: str,
    inputs: list[Any],
    params: dict[str, Any],
    l3_features: L3FeaturesArtifact,
    l3_metadata: L3MetadataArtifact,
    l5_eval: L5EvaluationArtifact,
) -> Any:
    if op in {"model_native_linear_coef", "shap_linear"}:
        model = _first_model_input(inputs)
        frame = _linear_importance_frame(model, method=op)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op in {"permutation_importance", "lofo"}:
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _permutation_importance_frame(model, X, y, method=op)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "permutation_importance_strobl":
        # Issue #281 -- conditional permutation per Strobl (2008): permute
        # within stratified bins of the *other* features so the marginal
        # distribution of the permuted feature stays roughly the same as
        # under the original conditional distribution. Removes the
        # correlated-features bias of vanilla permutation.
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _strobl_permutation_importance_frame(model, X, y)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "model_native_tree_importance":
        model = _first_model_input(inputs)
        frame = _tree_importance_frame(model)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op in {"shap_tree", "shap_kernel", "shap_deep", "shap_interaction"}:
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _shap_importance_frame(model, X, kind=op)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "partial_dependence":
        model = _first_model_input(inputs)
        X, _ = _l7_xy(inputs, l3_features)
        frame = _partial_dependence_table(
            model, X, n_grid=int(params.get("n_grid", 20))
        )
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "accumulated_local_effect":
        model = _first_model_input(inputs)
        X, _ = _l7_xy(inputs, l3_features)
        frame = _ale_table(model, X, n_quantiles=int(params.get("n_quantiles", 10)))
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "friedman_h_interaction":
        model = _first_model_input(inputs)
        X, _ = _l7_xy(inputs, l3_features)
        frame = _friedman_h_table(model, X)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op in {"lasso_inclusion_frequency", "stability_selection"}:
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _lasso_inclusion_frame(
            model,
            X=X,
            y=y,
            n_bootstraps=int(params.get("n_bootstraps", 50)),
            seed=int(params.get("random_state", 0)),
            sampling=str(params.get("inclusion_sampling", "bootstrap")),
            rolling_window=params.get("rolling_window"),
        )
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "cumulative_r2_contribution":
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _cumulative_r2_frame(model, X, y)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "bootstrap_jackknife":
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _bootstrap_jackknife_frame(
            model, X, y, n_replications=int(params.get("n_replications", 50))
        )
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "rolling_recompute":
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _rolling_importance_table(
            model, X, y, window=int(params.get("window_size", max(8, len(X) // 4)))
        )
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op in {"forecast_decomposition"}:
        model = _first_model_input(inputs)
        X, _ = _l7_xy(inputs, l3_features)
        frame = _forecast_decomposition_frame(model, X)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "dual_decomposition":
        # v0.9 Phase 1 promotion (linear families only).
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _dual_decomposition_frame(model, X, y)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "attention_weights":
        # Phase B-10 paper-10 promotion -- Goulet Coulombe (2026)
        # "OLS as an Attention Mechanism", Eq. 7 closed form.
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _l7_attention_weights_op(model, X, y, params)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op in {"oshapley_vi", "pbsv"}:
        # v0.9.1 dev-stage v0.9.0D Path B: final-window fit anatomy adapter.
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _l7_anatomy_op(op, model, X, y, params)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op in {"linear_coef"}:
        model = _first_model_input(inputs)
        frame = _linear_importance_frame(model, method="linear_coef")
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "tree_importance":
        model = _first_model_input(inputs)
        frame = _tree_importance_frame(model)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "group_aggregate":
        table = next(
            (item for item in inputs if isinstance(item, pd.DataFrame)), pd.DataFrame()
        )
        return _l7_group_aggregate(table, params)
    if op == "lineage_attribution":
        table = next(
            (item for item in inputs if isinstance(item, pd.DataFrame)), pd.DataFrame()
        )
        metadata = next(
            (item for item in inputs if isinstance(item, L3MetadataArtifact)),
            l3_metadata,
        )
        return _l7_lineage_attribution(table, metadata, params)
    if op == "transformation_attribution":
        return _l7_transformation_attribution(l5_eval, params)
    if op in {"integrated_gradients", "saliency_map", "deep_lift", "gradient_shap"}:
        # Gradient-based attributions (issue #194). Use captum when
        # available + the model exposes torch tensors; otherwise raise a
        # NotImplementedError pointing at the [deep] extra. The previous
        # silent SHAP-tree fallback masked the absence of the gradient
        # method named in the recipe.
        model = _first_model_input(inputs)
        X, _ = _l7_xy(inputs, l3_features)
        frame = _gradient_attribution_frame(model, X, kind=op)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "generalized_irf":
        # v0.8.9 honesty pass V2.3: name is reserved for the Pesaran-Shin
        # (1998) order-invariant variant, not yet implemented. The previous
        # v0.2 #189 runtime shipped Cholesky orthogonalised IRFs under this
        # name; the operational Cholesky variant is now ``orthogonalised_irf``.
        raise NotImplementedError(
            "generalized_irf (Pesaran-Shin 1998 order-invariant GIRF) is "
            "future-gated (v0.9.x roadmap). For Cholesky orthogonalised "
            "IRFs use the ``orthogonalised_irf`` op (operational since "
            "v0.8.9; matches the IRF previously emitted under the "
            "misnamed ``generalized_irf`` alias)."
        )
    if op in {"fevd", "historical_decomposition", "orthogonalised_irf"}:
        model = _first_model_input(inputs)
        frame = _var_impulse_frame(model, op_name=op)
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "bvar_pip":
        model = _first_model_input(inputs)
        frame = _linear_importance_frame(model, method="bvar_pip")
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "mrf_gtvp":
        model = _first_model_input(inputs)
        X, _ = _l7_xy(inputs, l3_features)
        frame = _mrf_gtvp_coefficient_frame(model, X)
        return _attach_l7_attrs(frame, model, op, l3_features)
    raise NotImplementedError(f"L7 runtime does not support op {op!r}")


def _l7_xy(
    inputs: list[Any], l3_features: L3FeaturesArtifact
) -> tuple[pd.DataFrame, pd.Series | None]:
    X = next(
        (item for item in inputs if isinstance(item, pd.DataFrame)),
        l3_features.X_final.data,
    )
    y_in = next(
        (item for item in inputs if isinstance(item, pd.Series)),
        l3_features.y_final.metadata.values.get("data"),
    )
    return X, y_in if isinstance(y_in, pd.Series) else None


_SHAP_SUBSAMPLE_THRESHOLD = 2000  # Cycle 14 L2-5 fix: default subsampling threshold


def _shap_importance_frame(
    model: ModelArtifact, X: pd.DataFrame, *, kind: str, shap_subsample: int | None = None
) -> pd.DataFrame:
    """SHAP via the optional `shap` package; falls back to a coefficient or
    permutation proxy if `shap` is not installed."""

    try:
        import shap  # type: ignore
    except ImportError:
        if hasattr(model.fitted_object, "coef_"):
            return _linear_importance_frame(model, method=kind)
        return _permutation_importance_frame(model, X, None, method=kind)

    # Cycle 14 L2-5 fix: subsample large panels to avoid slow SHAP computation
    threshold = shap_subsample if shap_subsample is not None else _SHAP_SUBSAMPLE_THRESHOLD
    X_shap = X
    if len(X) > threshold:
        import warnings
        warnings.warn(
            f"SHAP on {len(X)} rows is slow; subsampling to {threshold} rows. "
            f"Override via SHAP op param.",
            UserWarning,
            stacklevel=3,
        )
        X_shap = X.sample(n=threshold, random_state=42)

    fitted = model.fitted_object
    try:
        if kind == "shap_linear" and hasattr(fitted, "coef_"):
            explainer = shap.LinearExplainer(fitted, X_shap.fillna(0.0))
        elif kind in {"shap_tree", "shap_interaction"} and hasattr(
            fitted, "feature_importances_"
        ):
            explainer = shap.TreeExplainer(fitted)
        else:
            explainer = shap.KernelExplainer(
                fitted.predict, shap.sample(X_shap.fillna(0.0), min(50, len(X_shap)))
            )
        values = explainer.shap_values(X_shap.fillna(0.0))
    except Exception:
        return _permutation_importance_frame(model, X, None, method=kind)
    importance = (
        np.abs(values).mean(axis=0)
        if isinstance(values, np.ndarray)
        else np.abs(np.asarray(values)).mean(axis=0)
    )
    return pd.DataFrame(
        {
            "feature": list(model.feature_names),
            "importance": [float(v) for v in importance],
            "coefficient": None,
        }
    )


def _tree_importance_frame(model: ModelArtifact) -> pd.DataFrame:
    fitted = model.fitted_object
    importance = getattr(fitted, "feature_importances_", None)
    if importance is None:
        if hasattr(fitted, "coef_"):
            return _linear_importance_frame(model, method="tree_importance_proxy")
        return pd.DataFrame(
            {
                "feature": list(model.feature_names),
                "importance": [0.0] * len(model.feature_names),
                "coefficient": None,
            }
        )
    return pd.DataFrame(
        {
            "feature": list(model.feature_names),
            "importance": [float(v) for v in np.asarray(importance).ravel()],
            "coefficient": None,
        }
    )


def _partial_dependence_table(
    model: ModelArtifact, X: pd.DataFrame, *, n_grid: int
) -> pd.DataFrame:
    fitted = model.fitted_object
    rows = []
    for column in X.columns:
        series = X[column].dropna()
        if series.empty:
            rows.append({"feature": column, "importance": 0.0, "coefficient": None})
            continue
        grid = np.linspace(
            series.quantile(0.05), series.quantile(0.95), max(2, int(n_grid))
        )
        responses = []
        for value in grid:
            edited = X.fillna(0.0).copy()
            edited[column] = value
            try:
                response = float(np.mean(fitted.predict(edited)))
            except Exception:
                response = 0.0
            responses.append(response)
        rows.append(
            {
                "feature": column,
                "importance": float(max(responses) - min(responses)),
                "coefficient": None,
            }
        )
    return pd.DataFrame(rows)


def _ale_table(
    model: ModelArtifact, X: pd.DataFrame, *, n_quantiles: int
) -> pd.DataFrame:
    """Apley & Zhu (2020) Accumulated Local Effects.

    For each feature j, partition its range into ``n_quantiles`` bins.
    Within each bin, compute the local effect as the average over training
    points in that bin of ``f(x with x_j = upper_edge) - f(x with x_j =
    lower_edge)``. Center the local effects (subtract mean) and cumulate
    to obtain the ALE function. The ``importance`` column reports the L1
    norm of the centred ALE function -- a calibration-invariant feature
    importance derived from the same procedure.

    Issue #192. The v0.1 implementation summed *uncentered* local-effect
    bin endpoints; the centred-cumsum form here matches the published
    procedure.
    """

    fitted = model.fitted_object
    rows: list[dict[str, Any]] = []
    for column in X.columns:
        series = X[column].dropna()
        if len(series) < 4:
            rows.append(
                {
                    "feature": column,
                    "importance": 0.0,
                    "coefficient": None,
                    "ale_function": [],
                }
            )
            continue
        quantiles = np.quantile(series, np.linspace(0, 1, max(3, int(n_quantiles) + 1)))
        bin_edges = np.unique(quantiles)
        if len(bin_edges) < 3:
            rows.append(
                {
                    "feature": column,
                    "importance": 0.0,
                    "coefficient": None,
                    "ale_function": [],
                }
            )
            continue
        local_effects: list[float] = []
        bin_centers: list[float] = []
        # Apley local effect: condition on rows whose feature falls inside
        # the bin; predict at the bin endpoints with the rest of the row
        # held at its training values.
        feature_values = X[column].to_numpy()
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (feature_values >= low) & (feature_values <= high)
            if mask.sum() == 0:
                local_effects.append(0.0)
                bin_centers.append(0.5 * (low + high))
                continue
            slab = X[mask].fillna(0.0)
            edited_low = slab.copy()
            edited_low[column] = low
            edited_high = slab.copy()
            edited_high[column] = high
            try:
                effect = float(
                    np.mean(fitted.predict(edited_high) - fitted.predict(edited_low))
                )
            except Exception:
                effect = 0.0
            local_effects.append(effect)
            bin_centers.append(0.5 * (low + high))
        local_arr = np.asarray(local_effects, dtype=float)
        # Centre and cumulate -- Apley & Zhu Eq. (10).
        centred = local_arr - local_arr.mean()
        ale_function = np.cumsum(centred)
        # Importance: L1 norm of the centred ALE function divided by the
        # number of bins (mean absolute effect across the support).
        importance = float(np.mean(np.abs(ale_function))) if ale_function.size else 0.0
        rows.append(
            {
                "feature": column,
                "importance": importance,
                "coefficient": None,
                "ale_function": [
                    {"bin_center": float(c), "ale": float(v)}
                    for c, v in zip(bin_centers, ale_function)
                ],
            }
        )
    return pd.DataFrame(rows)


def _friedman_h_table(
    model: ModelArtifact, X: pd.DataFrame, *, n_grid: int = 8
) -> pd.DataFrame:
    """Friedman & Popescu (2008) H-statistic for pairwise interactions.

    For features j and k:

        H²_{jk} = sum_i (f_{jk}(x^i) - f_j(x^i) - f_k(x^i))² / sum_i f_{jk}(x^i)²

    where ``f_{jk}`` is the centred bivariate partial dependence and
    ``f_j``, ``f_k`` are the centred marginal partial dependences. The
    statistic ranges in ``[0, 1]``: 0 = no interaction, 1 = pure
    interaction.

    Issue #193. v0.1 used a variance-ratio surrogate; this implementation
    matches the published formula. We sub-sample the design grid to
    ``n_grid`` quantiles per feature to keep the cost manageable on
    medium-sized panels.
    """

    fitted = model.fitted_object
    rows: list[dict[str, Any]] = []
    columns = list(X.columns)
    if not columns:
        return pd.DataFrame({"feature": [], "importance": [], "coefficient": []})

    X_filled = X.fillna(0.0)
    n_rows = len(X_filled)
    if n_rows == 0:
        return pd.DataFrame({"feature": [], "importance": [], "coefficient": []})

    def _centred_pd(*column_values: tuple[str, np.ndarray]) -> np.ndarray:
        # Compute partial dependence at each X row by averaging predictions
        # over the fixed values supplied (one per pinned column).
        edited = X_filled.copy()
        for name, value in column_values:
            edited[name] = value
        try:
            preds = fitted.predict(edited)
        except Exception:
            return np.zeros(n_rows)
        arr = np.asarray(preds, dtype=float)
        return arr - arr.mean()

    for i, left in enumerate(columns):
        left_grid = np.quantile(
            X_filled[left].to_numpy(), np.linspace(0, 1, max(2, n_grid))
        )
        for right in columns[i + 1 :]:
            right_grid = np.quantile(
                X_filled[right].to_numpy(), np.linspace(0, 1, max(2, n_grid))
            )
            # Centre the marginal PDs across the design grid: take
            # f_j(x_j_quantile) per row and average across rows.
            try:
                f_j = np.zeros(n_rows)
                for v in left_grid:
                    f_j += _centred_pd((left, np.full(n_rows, v)))
                f_j /= len(left_grid)
                f_k = np.zeros(n_rows)
                for v in right_grid:
                    f_k += _centred_pd((right, np.full(n_rows, v)))
                f_k /= len(right_grid)
                # Bivariate PD: average over the (left, right) grid.
                f_jk = np.zeros(n_rows)
                count = 0
                for v_l in left_grid:
                    for v_r in right_grid:
                        f_jk += _centred_pd(
                            (left, np.full(n_rows, v_l)), (right, np.full(n_rows, v_r))
                        )
                        count += 1
                f_jk /= max(count, 1)
                num = float(np.sum((f_jk - f_j - f_k) ** 2))
                denom = float(np.sum(f_jk**2))
                h_sq = num / denom if denom > 0 else 0.0
            except Exception:
                h_sq = 0.0
            rows.append(
                {
                    "feature": f"{left}*{right}",
                    "importance": float(np.sqrt(max(0.0, min(1.0, h_sq)))),
                    "coefficient": None,
                }
            )
    if not rows:
        return pd.DataFrame({"feature": [], "importance": [], "coefficient": []})
    return pd.DataFrame(rows)


def _lasso_inclusion_frame(
    model: ModelArtifact,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    *,
    n_bootstraps: int = 50,
    seed: int = 0,
    sampling: str = "bootstrap",
    rolling_window: int | None = None,
) -> pd.DataFrame:
    """Issues #191 + #253 -- inclusion frequency over bootstrap *or* rolling
    windows.

    * ``sampling = bootstrap`` (default): i.i.d. resample with replacement.
    * ``sampling = rolling``: refit on overlapping rolling windows of size
      ``rolling_window`` (defaults to ``max(20, n // 4)``); inclusion
      frequency = fraction of windows where the coefficient survived. Detects
      time-varying coefficient paths that bootstrap masks.
    * ``sampling = both``: returns both columns
      (``importance`` = bootstrap, ``rolling_inclusion`` = rolling).
    """

    fitted = model.fitted_object
    coef = getattr(fitted, "coef_", None)
    feature_names = list(model.feature_names)
    if coef is None:
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": [0.0] * len(feature_names),
                "coefficient": None,
            }
        )
    coef_arr = np.asarray(coef).ravel()
    if X is None or y is None:
        inclusion = (np.abs(coef_arr) > 1e-9).astype(float)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": inclusion.tolist(),
                "coefficient": [float(c) for c in coef_arr],
            }
        )
    aligned = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    if aligned.empty or aligned.shape[0] < 5:
        inclusion = (np.abs(coef_arr) > 1e-9).astype(float)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": inclusion.tolist(),
                "coefficient": [float(c) for c in coef_arr],
            }
        )
    X_arr = (
        aligned.drop(columns="__y__")[feature_names].fillna(0.0).to_numpy(dtype=float)
    )
    y_arr = aligned["__y__"].to_numpy(dtype=float)
    alpha = float(getattr(fitted, "alpha_", getattr(fitted, "alpha", 0.1)))
    n = len(X_arr)

    def _bootstrap_inclusion() -> tuple[np.ndarray, int]:
        rng = np.random.default_rng(int(seed))
        counts = np.zeros(X_arr.shape[1], dtype=float)
        ok = 0
        for _ in range(int(max(2, n_bootstraps))):
            idx = rng.integers(0, n, size=n)
            try:
                fit = Lasso(alpha=alpha, max_iter=20000)
                fit.fit(X_arr[idx], y_arr[idx])
                counts += (np.abs(fit.coef_) > 1e-9).astype(float)
                ok += 1
            except Exception:
                continue
        return counts, ok

    def _rolling_inclusion(window: int) -> tuple[np.ndarray, int]:
        counts = np.zeros(X_arr.shape[1], dtype=float)
        ok = 0
        for start in range(0, max(1, n - window + 1)):
            end = min(start + window, n)
            if end - start < 4:
                continue
            try:
                fit = Lasso(alpha=alpha, max_iter=20000)
                fit.fit(X_arr[start:end], y_arr[start:end])
                counts += (np.abs(fit.coef_) > 1e-9).astype(float)
                ok += 1
            except Exception:
                continue
        return counts, ok

    rolling_w = int(rolling_window) if rolling_window is not None else max(20, n // 4)

    if sampling == "rolling":
        counts, successful = _rolling_inclusion(rolling_w)
    elif sampling == "both":
        boot_counts, boot_ok = _bootstrap_inclusion()
        roll_counts, roll_ok = _rolling_inclusion(rolling_w)
        boot_freq = (
            boot_counts / boot_ok
            if boot_ok
            else (np.abs(coef_arr) > 1e-9).astype(float)
        )
        roll_freq = (
            roll_counts / roll_ok
            if roll_ok
            else (np.abs(coef_arr) > 1e-9).astype(float)
        )
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": boot_freq.tolist(),
                "rolling_inclusion": roll_freq.tolist(),
                "coefficient": [float(c) for c in coef_arr],
                "n_bootstraps_run": [boot_ok] * len(feature_names),
                "n_rolling_windows_run": [roll_ok] * len(feature_names),
                "rolling_window_size": [rolling_w] * len(feature_names),
            }
        )
    else:  # bootstrap (default)
        counts, successful = _bootstrap_inclusion()
    if successful == 0:
        inclusion = (np.abs(coef_arr) > 1e-9).astype(float)
    else:
        inclusion = counts / successful
    sampling_meta_key = (
        "n_rolling_windows_run" if sampling == "rolling" else "n_bootstraps_run"
    )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": inclusion.tolist(),
            "coefficient": [float(c) for c in coef_arr],
            sampling_meta_key: [successful] * len(feature_names),
        }
    )


def _cumulative_r2_frame(
    model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None
) -> pd.DataFrame:
    if y is None or not hasattr(model.fitted_object, "predict"):
        return _linear_importance_frame(model, method="cumulative_r2")
    aligned = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": [0.0] * len(X.columns),
                "coefficient": None,
            }
        )
    yv = aligned["__y__"].to_numpy()
    var_y = float(np.var(yv))
    rows = []
    seen: list[str] = []
    cum_r2 = 0.0
    for column in X.columns:
        seen.append(column)
        from sklearn.linear_model import LinearRegression

        sub = aligned[seen].fillna(0.0).to_numpy()
        try:
            fit = LinearRegression().fit(sub, yv)
            preds = fit.predict(sub)
            r2 = 0.0 if var_y == 0 else 1.0 - float(np.var(yv - preds)) / var_y
        except Exception:
            r2 = cum_r2
        rows.append(
            {
                "feature": column,
                "importance": float(max(0.0, r2 - cum_r2)),
                "coefficient": None,
            }
        )
        cum_r2 = max(cum_r2, r2)
    return pd.DataFrame(rows)


def _bootstrap_jackknife_frame(
    model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None, *, n_replications: int
) -> pd.DataFrame:
    if y is None:
        return _linear_importance_frame(model, method="bootstrap_jackknife")
    aligned = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": [0.0] * len(X.columns),
                "coefficient": None,
            }
        )
    rng = np.random.default_rng(0)
    importances = []
    for _ in range(max(2, int(n_replications))):
        sample = aligned.sample(
            frac=1.0, replace=True, random_state=int(rng.integers(0, 2**32 - 1))
        )
        boot_frame = _permutation_importance_frame(
            model, sample[X.columns], sample["__y__"], method="permutation_importance"
        )
        importances.append(boot_frame.set_index("feature")["importance"])
    matrix = pd.concat(importances, axis=1)
    summary = (
        matrix.agg(["mean", "std"], axis=1)
        .rename(columns={"mean": "importance", "std": "importance_std"})
        .reset_index()
    )
    summary["coefficient"] = None
    return summary


def _rolling_importance_table(
    model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None, *, window: int
) -> pd.DataFrame:
    if y is None or not hasattr(model.fitted_object, "predict"):
        return _linear_importance_frame(model, method="rolling_recompute")
    aligned = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    if len(aligned) <= window:
        return _permutation_importance_frame(model, X, y, method="rolling_recompute")
    importances: dict[Any, pd.Series] = {}
    for end in range(window, len(aligned) + 1, max(1, window // 4)):
        sub = aligned.iloc[max(0, end - window) : end]
        frame = _permutation_importance_frame(
            model, sub[X.columns], sub["__y__"], method="rolling_recompute"
        )
        importances[sub.index[-1]] = frame.set_index("feature")["importance"]
    matrix = pd.DataFrame(importances)
    out = matrix.mean(axis=1).reset_index().rename(columns={0: "importance"})
    out.columns = ["feature", "importance"]
    out["coefficient"] = None
    return out


def _forecast_decomposition_frame(
    model: ModelArtifact, X: pd.DataFrame
) -> pd.DataFrame:
    """Per-feature contribution to the latest prediction (linear models only)."""

    fitted = model.fitted_object
    coef = getattr(fitted, "coef_", None)
    if coef is None:
        return _tree_importance_frame(model)
    last = X.iloc[[-1]].fillna(0.0).to_numpy().ravel()
    contributions = np.asarray(coef).ravel() * last
    return pd.DataFrame(
        {
            "feature": list(model.feature_names),
            "importance": [float(abs(c)) for c in contributions],
            "coefficient": [float(v) for v in contributions],
        }
    )


def _gradient_attribution_frame(
    model: ModelArtifact, X: pd.DataFrame, *, kind: str
) -> pd.DataFrame:
    """Issue #194 -- gradient-based attribution methods.

    For sequence models fitted via ``_TorchSequenceModel``, route through
    ``captum`` (when installed) for the requested method:

    * ``saliency_map`` -- ``captum.attr.Saliency``
    * ``integrated_gradients`` -- ``captum.attr.IntegratedGradients``
    * ``deep_lift`` -- ``captum.attr.DeepLift``
    * ``gradient_shap`` -- ``captum.attr.GradientShap``

    For non-sequence sklearn models, gradient methods are not well-defined
    on a fitted forest / linear regressor; the runtime raises
    ``NotImplementedError`` instead of silently substituting a SHAP-tree
    proxy. Recipes that opt into gradient attributions intentionally want
    the gradient-of-output trace; we do not pretend to provide it from a
    method that cannot.
    """

    fitted = model.fitted_object
    if model.framework != "torch" and not hasattr(fitted, "_model"):
        raise NotImplementedError(
            f"L7 op {kind!r} requires a torch-backed model (lstm/gru/transformer). "
            "Install macroforecast[deep] and refit the recipe with a torch family, "
            "or pick shap_kernel/permutation_importance for non-torch estimators."
        )

    try:
        import torch  # type: ignore
        from captum import attr as _captum  # type: ignore
    except ImportError as exc:
        raise NotImplementedError(
            f"L7 op {kind!r} requires the [deep] extra (torch + captum). "
            "Install with `pip install macroforecast[deep]`."
        ) from exc

    torch_model = fitted._model
    if torch_model is None:
        raise NotImplementedError(
            f"L7 op {kind!r}: the underlying torch model is not fitted yet."
        )

    x_arr = X.fillna(0.0).to_numpy(dtype="float32")
    seq = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])
    inputs = torch.from_numpy(seq).requires_grad_(True)

    method_factories = {
        "saliency_map": _captum.Saliency,
        "integrated_gradients": _captum.IntegratedGradients,
        "deep_lift": _captum.DeepLift,
        "gradient_shap": _captum.GradientShap,
    }
    method_cls = method_factories.get(kind)
    if method_cls is None:
        raise NotImplementedError(f"L7 op {kind!r} is not a known gradient method")
    method = method_cls(torch_model)
    if kind == "gradient_shap":
        baselines = torch.zeros_like(inputs)
        attributions = method.attribute(inputs, baselines=baselines)
    else:
        attributions = method.attribute(inputs)
    attribs = (
        attributions.detach().cpu().numpy().reshape(x_arr.shape[0], x_arr.shape[1])
    )
    importance = np.abs(attribs).mean(axis=0)
    return pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": [float(v) for v in importance],
            "coefficient": [None] * len(X.columns),
            "method": [kind] * len(X.columns),
        }
    )


def _mrf_gtvp_coefficient_frame(model: ModelArtifact, X: pd.DataFrame) -> pd.DataFrame:
    """Issue #190 -- time-varying coefficient series from a Coulombe (2024) MRF.

    Reads the GTVP β̂(t) series cached on the
    ``_MRFExternalWrapper`` (populated during the most recent ``predict``
    invocation). The mrf-web ``_ensemble_loop`` returns ``betas`` of shape
    ``(T, K + 1)`` where column 0 is the intercept and columns 1..K are
    the per-time-step coefficients on the state vector. Importance =
    time-average of ``|β̂(t)|`` per feature.

    Falls back to generic tree importance when the fitted object is not
    an MRF wrapper or has no cached betas yet (e.g. a model artifact that
    was never used for prediction).
    """

    fitted = model.fitted_object
    feature_names = list(model.feature_names)
    cached_betas = getattr(fitted, "_cached_betas", None)
    if cached_betas is None or cached_betas.size == 0:
        frame = _tree_importance_frame(model)
        if "status" not in frame.columns:
            frame["status"] = "fallback_not_mrf"
        return frame

    # mrf-web returns betas with column 0 = intercept; align column slice
    # to the feature_names order. If shape mismatch, fall back gracefully.
    if cached_betas.shape[1] < len(feature_names) + 1:
        frame = _tree_importance_frame(model)
        if "status" not in frame.columns:
            frame["status"] = "fallback_betas_shape"
        return frame
    coef_path = cached_betas[:, 1 : 1 + len(feature_names)]
    # OOS rows are not covered by mrf-web's in-sample Bayesian bootstrap, so
    # ``avg_beta_nonOVF`` returns NaN there. Use nan-safe reductions so a
    # single NaN doesn't poison the per-feature importance score.
    importance = np.nanmean(np.abs(coef_path), axis=0)
    rows = []
    for j, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "importance": float(importance[j])
                if np.isfinite(importance[j])
                else 0.0,
                "coefficient": None,
                "coefficient_path": [
                    float(v) if np.isfinite(v) else None for v in coef_path[:, j]
                ],
                "status": "operational",
            }
        )
    return pd.DataFrame(rows)


def _var_impulse_frame(
    model: ModelArtifact, *, op_name: str, n_periods: int = 12
) -> pd.DataFrame:
    """Issue #189 -- orthogonalised IRF / FEVD / Historical Decomposition.

    For a fitted statsmodels VAR (``_VARWrapper`` or
    ``_FactorAugmentedVAR``), use the published ``irf`` / ``fevd`` /
    ``historical_decomposition`` builders to produce per-feature
    contributions to the target variable's response over ``n_periods``
    horizons. The summary in ``importance`` is the L1 sum of impulses /
    variance shares / decomposition contributions to the target.

    Falls back to a tree-importance proxy when the model is not VAR-based
    (e.g. plain ridge); the ``status`` column documents the path taken.
    """

    fitted_results = getattr(model.fitted_object, "_results", None)
    feature_names = list(model.feature_names)
    if fitted_results is None:
        # Unwrap one level for FAVAR (its ._var._results is the actual VAR fit).
        inner = getattr(model.fitted_object, "_var", None)
        fitted_results = getattr(inner, "_results", None) if inner is not None else None
    if fitted_results is None:
        # Non-VAR estimator: fall back to tree importance.
        frame = _tree_importance_frame(model)
        if "status" not in frame.columns:
            frame["status"] = "fallback_non_var"
        return frame

    target_index = 0
    try:
        target_index = (
            list(fitted_results.names).index("__y__")
            if "__y__" in fitted_results.names
            else 0
        )
    except Exception:
        target_index = 0

    # v0.9.0a0 audit-fix: when the fitted object is a BVAR with posterior-
    # sampled IRF draws, route IRF / HD through the Bayesian posterior
    # rather than the OLS-VAR ``fitted_results.irf()``. FEVD still flows
    # through statsmodels because it requires ``Σ_u`` recomputation per
    # draw — surfaced as a v0.9.x roadmap item.
    posterior_irf = getattr(model.fitted_object, "_posterior_irf", None)
    use_posterior = isinstance(posterior_irf, dict) and "mean" in posterior_irf

    rows: list[dict[str, Any]] = []
    try:
        if op_name == "fevd":
            if use_posterior and "fevd_mean" in posterior_irf:
                # Posterior FEVD (audit gap-fix #14): posterior mean +
                # credible bands per shock j averaged across horizons.
                mean_dec = np.asarray(posterior_irf["fevd_mean"], dtype=float)
                p16_dec = np.asarray(
                    posterior_irf.get("fevd_p16", mean_dec), dtype=float
                )
                p84_dec = np.asarray(
                    posterior_irf.get("fevd_p84", mean_dec), dtype=float
                )
                horizon_clip = min(int(n_periods), mean_dec.shape[0])
                shares = mean_dec[:horizon_clip, target_index, :].mean(axis=0)
                p16_shares = p16_dec[:horizon_clip, target_index, :].mean(axis=0)
                p84_shares = p84_dec[:horizon_clip, target_index, :].mean(axis=0)
                for j, name in enumerate(fitted_results.names):
                    if j >= len(shares):
                        break
                    rows.append(
                        {
                            "feature": str(name),
                            "importance": float(shares[j]),
                            "coefficient": None,
                            "status": "posterior_mean",
                            "p16": float(p16_shares[j]),
                            "p84": float(p84_shares[j]),
                        }
                    )
            else:
                fevd = fitted_results.fevd(int(n_periods))
                # decomp shape: (n_periods, n_vars, n_vars). Last axis is the
                # contribution of variable j to the variance of variable i.
                decomp = np.asarray(fevd.decomp, dtype=float)
                shares = decomp[:, target_index, :].mean(axis=0)  # avg over horizons
                for j, name in enumerate(fitted_results.names):
                    if j >= len(shares):
                        break
                    rows.append(
                        {
                            "feature": str(name),
                            "importance": float(shares[j]),
                            "coefficient": None,
                            "status": "operational",
                        }
                    )
        elif op_name == "orthogonalised_irf":
            if use_posterior:
                # Posterior IRF: response = sum_h |E[orth_irf_h, target, j]|;
                # surface 16/84 percentile bands as additional columns so
                # the L7 frame carries Coulombe & Göbel (2021) §3 credible
                # regions when reported.
                mean_irf = np.asarray(posterior_irf["mean"], dtype=float)
                p05 = np.asarray(posterior_irf.get("p05", mean_irf), dtype=float)
                p16 = np.asarray(posterior_irf.get("p16", mean_irf), dtype=float)
                p84 = np.asarray(posterior_irf.get("p84", mean_irf), dtype=float)
                p95 = np.asarray(posterior_irf.get("p95", mean_irf), dtype=float)
                horizon_clip = min(int(n_periods) + 1, mean_irf.shape[0])
                response = np.abs(mean_irf[:horizon_clip, target_index, :]).sum(axis=0)
                p05_resp = np.abs(p05[:horizon_clip, target_index, :]).sum(axis=0)
                p16_resp = np.abs(p16[:horizon_clip, target_index, :]).sum(axis=0)
                p84_resp = np.abs(p84[:horizon_clip, target_index, :]).sum(axis=0)
                p95_resp = np.abs(p95[:horizon_clip, target_index, :]).sum(axis=0)
                for j, name in enumerate(fitted_results.names):
                    if j >= len(response):
                        break
                    rows.append(
                        {
                            "feature": str(name),
                            "importance": float(response[j]),
                            "coefficient": None,
                            "status": "posterior_mean",
                            "p05": float(p05_resp[j]),
                            "p16": float(p16_resp[j]),
                            "p84": float(p84_resp[j]),
                            "p95": float(p95_resp[j]),
                        }
                    )
            else:
                irf = fitted_results.irf(int(n_periods))
                # Cholesky orthogonalised IRFs (orth_irfs shape:
                # (n_periods+1, n_vars, n_vars)). orth_irfs[s, i, j] is the
                # response of variable i at horizon s to a unit structural
                # shock to variable j at time 0.
                irfs = np.asarray(irf.orth_irfs, dtype=float)
                response = np.abs(irfs[:, target_index, :]).sum(axis=0)
                for j, name in enumerate(fitted_results.names):
                    if j >= len(response):
                        break
                    rows.append(
                        {
                            "feature": str(name),
                            "importance": float(response[j]),
                            "coefficient": None,
                            "status": "operational",
                        }
                    )
        else:  # historical_decomposition
            # Burbidge-Harrison (1985) historical decomposition. Construct
            # structural shocks via the Cholesky factor of the residual
            # covariance matrix:
            #     P P' = Σᵤ,    eₜ* = P⁻¹ uₜ
            # then convolve with the orthogonalised IRF coefficients to
            # express the realised path as a sum of shock-specific
            # contributions:
            #     hd[t, i, j] = Σ_{s=0..t}  orth_irfs[s, i, j] · e*_{t-s, j}
            # The op importance is the per-shock cumulative absolute
            # contribution to the *target* variable's path; the per-row
            # ``status`` documents the canonical procedure.
            #
            # Phase B-4 F7 posterior HD bands: when the BVAR posterior
            # cache contains ``hd_*`` keys, surface 16/84 percentile
            # bands on the per-shock importance score. The bands carry
            # parameter uncertainty in the IRF kernel; the structural
            # shocks themselves are held at the posterior-mean
            # Cholesky for tractability (deferred: full IW posterior on Σ_u).
            if use_posterior and "hd_mean" in posterior_irf:
                hd_mean = np.asarray(posterior_irf["hd_mean"], dtype=float)
                hd_p16 = np.asarray(posterior_irf.get("hd_p16", hd_mean), dtype=float)
                hd_p84 = np.asarray(posterior_irf.get("hd_p84", hd_mean), dtype=float)
                # hd_*[t, i, j] -- target row = i = target_index.
                T_resid = hd_mean.shape[0]
                if target_index >= hd_mean.shape[1]:
                    target_index_local = 0
                else:
                    target_index_local = target_index
                response_mean = np.abs(hd_mean[:, target_index_local, :]).sum(axis=0)
                response_p16 = np.abs(hd_p16[:, target_index_local, :]).sum(axis=0)
                response_p84 = np.abs(hd_p84[:, target_index_local, :]).sum(axis=0)
                for j, name in enumerate(fitted_results.names):
                    if j >= len(response_mean):
                        break
                    rows.append(
                        {
                            "feature": str(name),
                            "importance": float(response_mean[j]),
                            "coefficient": None,
                            "status": "posterior_mean",
                            "p16": float(response_p16[j]),
                            "p84": float(response_p84[j]),
                        }
                    )
            else:
                resid = np.asarray(fitted_results.resid, dtype=float)
                sigma_u = np.asarray(fitted_results.sigma_u, dtype=float)
                # Phase B-4 F3: honor ordering when present on a BVAR
                # results object.
                ordering_attr = getattr(fitted_results, "ordering", None)
                names_attr = getattr(fitted_results, "names", None)
                if ordering_attr is not None and names_attr is not None:
                    chol = _build_cholesky_with_ordering(
                        sigma_u, names_attr, ordering_attr
                    )
                else:
                    chol = np.linalg.cholesky(sigma_u)
                structural_shocks = np.linalg.solve(chol, resid.T).T  # (T_resid, K)
                T_resid = structural_shocks.shape[0]
                horizon_max = max(int(n_periods), T_resid)
                # Posterior-aware IRF for HD convolution when available
                # (status reflects the source).
                if use_posterior:
                    mean_irf = np.asarray(posterior_irf["mean"], dtype=float)
                    if mean_irf.shape[0] >= horizon_max + 1:
                        irfs = mean_irf[: horizon_max + 1]
                    else:
                        irfs = np.asarray(
                            fitted_results.irf(horizon_max).orth_irfs, dtype=float
                        )
                    hd_status = (
                        "posterior_mean"
                        if mean_irf.shape[0] >= horizon_max + 1
                        else "operational"
                    )
                else:
                    irfs = np.asarray(
                        fitted_results.irf(horizon_max).orth_irfs, dtype=float
                    )
                    hd_status = "operational"
                S = irfs.shape[0]
                # Convolution kernel for the target row only: response of the
                # target variable at horizon s to a unit shock j is
                # ``irfs[s, target_index, j]``. Vectorise per-shock:
                hd_target = np.zeros((T_resid, structural_shocks.shape[1]), dtype=float)
                for s in range(min(S, T_resid)):
                    # Contributions arriving at times s, s+1, ..., T_resid-1
                    hd_target[s:, :] += (
                        irfs[s, target_index, :] * structural_shocks[: T_resid - s, :]
                    )
                response = np.abs(hd_target).sum(axis=0)
                for j, name in enumerate(fitted_results.names):
                    if j >= len(response):
                        break
                    rows.append(
                        {
                            "feature": str(name),
                            "importance": float(response[j]),
                            "coefficient": None,
                            "status": hd_status,
                        }
                    )
    except Exception as exc:
        return pd.DataFrame(
            {
                "feature": feature_names[:1],
                "importance": [0.0],
                "coefficient": [None],
                "status": [f"error: {type(exc).__name__}"],
            }
        )

    if not rows:
        return pd.DataFrame(
            {"feature": [], "importance": [], "coefficient": [], "status": []}
        )
    return pd.DataFrame(rows)


def _first_model_input(inputs: list[Any]) -> ModelArtifact:
    for item in inputs:
        if isinstance(item, ModelArtifact):
            return item
    raise ValueError("L7 step requires a ModelArtifact input")


def _linear_importance_frame(model: ModelArtifact, *, method: str) -> pd.DataFrame:
    fitted = model.fitted_object
    coef = getattr(fitted, "coef_", None)
    if coef is None:
        raise ValueError(f"model {model.model_id} does not expose coef_")
    values = list(coef.ravel() if hasattr(coef, "ravel") else coef)
    return pd.DataFrame(
        {
            "feature": list(model.feature_names),
            "coefficient": [float(value) for value in values],
            "importance": [abs(float(value)) for value in values],
        }
    )


def _blocked_oob_reality_check_p_values(
    losses: pd.DataFrame,
    *,
    benchmark: str,
    block_length: int = 4,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> pd.DataFrame:
    """Block-bootstrap variant of White (2000) reality check on
    per-origin loss differentials vs a named benchmark model.

    Used by the v0.9.x HNN evaluation pipeline (Coulombe / Frenette /
    Klieber 2025 JAE) where the in-sample serial dependence of macro
    residuals violates the i.i.d. assumption underpinning the standard
    reality check.

    Algorithm:

        For each candidate model m (excluding the benchmark):
            d_t = loss_benchmark_t - loss_m_t   (positive = m better)
            d_bar = mean(d_t)
            For b in 1..n_bootstraps:
                Draw consecutive blocks of length ``block_length`` from
                {1, ..., T} (with replacement, mod T) until length T.
                d_bar_b = mean of d_t at block-bootstrapped indices.
            p_m = fraction of b with (d_bar_b - d_bar) >= d_bar
                  (one-sided test for H0: m no better than benchmark).
            reject_m = (p_m < alpha).

    Args:
        losses: per-origin loss panel. Index = origin date; columns =
                model_id; values = per-origin loss (e.g. squared error).
        benchmark: column name of the benchmark model.
        block_length: moving-block length (Künsch 1989). Default 4.
        n_bootstraps: number of bootstrap replications. Default 1000.
        alpha: significance level. Default 0.05.
        random_state: RNG seed for reproducibility.

    Returns:
        DataFrame with columns ``['mean_diff', 'p_value', 'reject_h0']``,
        one row per non-benchmark model. ``mean_diff > 0`` means the
        candidate has lower loss than the benchmark; ``reject_h0=True``
        means we reject "candidate no better than benchmark" at α.
    """

    if benchmark not in losses.columns:
        raise ValueError(
            f"benchmark {benchmark!r} not in loss columns {list(losses.columns)}"
        )
    rng = np.random.default_rng(random_state)
    n = len(losses)
    if n < 2:
        return pd.DataFrame(columns=["mean_diff", "p_value", "reject_h0"])
    block_length = max(1, int(block_length))
    n_bootstraps = max(1, int(n_bootstraps))

    bench_loss = losses[benchmark].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for model in losses.columns:
        if model == benchmark:
            continue
        cand_loss = losses[model].to_numpy(dtype=float)
        d = bench_loss - cand_loss  # positive when cand better
        d_bar = float(d.mean())

        # Block-bootstrap distribution of (d_bar_b - d_bar).
        # Recentre: under H0 (cand no better than benchmark), E[d] <= 0.
        # The standard reality-check construction recentres d - max(0, d_bar)
        # then takes the bootstrap distribution of d_bar_b minus the recentred
        # mean; we use the simpler one-sided variant.
        n_blocks = (n + block_length - 1) // block_length
        offsets = np.arange(block_length)
        boot_means = np.empty(n_bootstraps)
        for b in range(n_bootstraps):
            starts = rng.integers(0, n, size=n_blocks)
            idx = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
            boot_means[b] = float(d[idx].mean())
        # One-sided test: under H0, d_bar should not be 'large positive'.
        # Recentre boot_means around d_bar so the null distribution is at 0.
        recentred = boot_means - d_bar
        # p-value: fraction of recentred bootstrap means >= d_bar.
        p_value = float((recentred >= d_bar).mean())
        rows.append(
            {
                "model_id": model,
                "mean_diff": d_bar,
                "p_value": p_value,
                "reject_h0": bool(p_value < alpha),
            }
        )
    return (
        pd.DataFrame(rows).set_index("model_id")
        if rows
        else pd.DataFrame(columns=["mean_diff", "p_value", "reject_h0"])
    )


def _rf_leaf_cooccurrence_weights(
    forest: Any, X_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """RF leaf-co-occurrence kernel for the dual decomposition.

    A random-forest prediction can be written as a weighted sum of
    training targets:
        ``ŷ(xₜ) = Σⱼ wⱼ(xₜ) · yⱼ``
    where the weight is the *leaf-co-occurrence kernel* averaged over
    trees:
        ``wⱼ(xₜ) = (1/B) Σ_b  1[j ∈ B_b] · 1[leaf_b(xₜ) == leaf_b(xⱼ)] / leaf_size_b_in_bootstrap(xⱼ)``.

    The ``1[j ∈ B_b]`` indicator restricts each tree to its bootstrap
    subset (``estimators_samples_``) so the construction reproduces the
    forest's prediction bit-exactly when ``X_test ⊆ X_train``. See
    Goulet Coulombe / Goebel / Klieber (2024) §3.2 for the derivation.
    Returns the (n_test, n_train) weight matrix.

    Compatible with sklearn ``RandomForestRegressor`` and
    ``ExtraTreesRegressor``. ``GradientBoostingRegressor`` is treated
    via per-stage tree application but only when the booster is regression
    with constant init (the default). Other boosters (xgboost / lightgbm)
    are out of scope here; macroforecast routes those through tree-
    importance fall-backs.
    """

    train_leaves = np.asarray(forest.apply(X_train))  # (n_train, n_trees) for ensembles
    test_leaves = np.asarray(forest.apply(X_test))
    if train_leaves.ndim == 1:
        train_leaves = train_leaves.reshape(-1, 1)
        test_leaves = test_leaves.reshape(-1, 1)
    n_train, n_trees = train_leaves.shape
    n_test = test_leaves.shape[0]

    # Per-tree bootstrap indices: sklearn exposes them on
    # ``estimators_samples_`` (bool mask or index array per tree). When
    # ``bootstrap=False`` (e.g. ExtraTreesRegressor default) the attribute
    # may be missing -- fall back to "every training row in every tree".
    samples_attr = getattr(forest, "estimators_samples_", None)
    bootstrap = bool(getattr(forest, "bootstrap", True))

    W = np.zeros((n_test, n_train), dtype=float)
    for b in range(n_trees):
        train_b = train_leaves[:, b]
        test_b = test_leaves[:, b]
        # Identify the rows that actually contributed to tree b's training.
        if samples_attr is not None and bootstrap:
            sample_b = np.asarray(samples_attr[b])
            if sample_b.dtype == bool:
                bootstrap_mask = sample_b.astype(float)
            else:
                # Sampling-with-replacement may put the same index multiple
                # times. Accumulate via np.add.at so the multiplicity is
                # reflected in the leaf-size denominator and the dual
                # weight is bit-exact against ``forest.predict``.
                bootstrap_mask = np.zeros(n_train, dtype=float)
                np.add.at(bootstrap_mask, sample_b, 1.0)
        else:
            bootstrap_mask = np.ones(n_train, dtype=float)
        # leaf_size_b[j] = sum over rows i of (in-bootstrap_i ·
        # 1[leaf_b(i) == leaf_b(j)]).
        # Compute via bincount on the leaf id, weighted by bootstrap_mask.
        unique_leaves, inverse = np.unique(train_b, return_inverse=True)
        leaf_count_in_bag = np.bincount(
            inverse, weights=bootstrap_mask, minlength=len(unique_leaves)
        )
        train_leaf_size = leaf_count_in_bag[inverse].astype(float)
        train_leaf_size = np.maximum(train_leaf_size, 1.0)  # guard against zero
        match = (test_b[:, None] == train_b[None, :]).astype(float)
        contribution = match * (bootstrap_mask[None, :] / train_leaf_size[None, :])
        W += contribution
    W /= max(n_trees, 1)
    return W


def _dual_decomposition_frame(
    model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None
) -> pd.DataFrame:
    """Goulet Coulombe / Goebel / Klieber (2024) Dual Interpretation of
    ML Forecasts -- representer-theorem-based training-target weights.

    For ridge: ``ŷₜ = Σⱼ wⱼ(xₜ) · yⱼ`` where
    ``w(xₜ) = X (X'X + αI)⁻¹ xₜ``. The vector ``w`` of dimension n_train
    is the per-test-row weight on training targets. Sum of weights need
    not equal 1 (no convex constraint); each weight signs / magnitudes
    encode how the model leans on each training row.

    **Linear families** (operational v0.8.9 B-3): ridge / OLS / lasso
    use the closed-form ``w(xₜ) = X (X'X + αI)⁻¹ xₜ``.

    **Tree ensembles** (operational v0.9.1 v0.9.0B-5): RandomForest /
    ExtraTrees / GradientBoosting use the leaf-co-occurrence kernel
    ``wⱼ(xₜ) = (1/B) Σ_b  1[leaf(xₜ) == leaf(xⱼ)] / leaf_size(xⱼ)``.

    **Kernel SVR** (operational v0.9.1 v0.9.0B-5): SVR with rbf / poly
    / sigmoid / linear kernels uses ``Wⱼ = αⱼ K(xⱼ, x_test)`` with the
    fitted dual coefficients and explicit kernel evaluation.

    **Kernel Ridge Regression** (operational Phase B-12, paper §2.2
    headline application — Fig 2 / Table 1): ``KernelRidge`` uses the
    closed-form representer ``W = K_test (K_train + αI)⁻¹`` so the
    representer identity ``ŷ_test = W @ y_train`` holds to numerical
    precision (paper Eqs. 5-6).

    Output frame layout (matches the L7 importance contract):
        rows: training row labels
        cols: ['mean_weight', 'abs_mean_weight', 'max_abs_weight']

    Inline portfolio diagnostics (Coulombe et al. 2024 §4, computed on
    the dual weights) are attached via ``frame.attrs['portfolio_metrics']``
    as a (n_test × 4) DataFrame with columns ``['hhi', 'short',
    'turnover', 'leverage']``. The full ``(n_test × n_train)`` weight
    matrix is also attached as ``frame.attrs['dual_weights']``. These
    are trivial numpy reductions on the primary dual weights and do not
    warrant their own L7 op (decomposition discipline).
    """

    fitted = model.fitted_object
    X_train = X.fillna(0.0).to_numpy(dtype=float)
    n_train, p = X_train.shape
    if n_train == 0 or p == 0:
        empty = pd.DataFrame(
            {
                "feature": [],
                "mean_weight": [],
                "abs_mean_weight": [],
                "max_abs_weight": [],
            }
        )
        empty.attrs["dual_weights"] = pd.DataFrame()
        empty.attrs["portfolio_metrics"] = pd.DataFrame(
            columns=["hhi", "short", "turnover", "leverage"]
        )
        return empty

    # Test set = same as train (in-sample dual weights).
    X_test = X_train
    method = "linear_closed_form"
    alpha: float | None = None

    if hasattr(fitted, "coef_"):
        # Linear closed-form.
        alpha = float(getattr(fitted, "alpha", 0.0) or 0.0)
        XtX = X_train.T @ X_train
        K = np.linalg.pinv(XtX + alpha * np.eye(p))
        W = X_test @ K @ X_train.T
    elif (
        hasattr(fitted, "dual_coef_")
        and hasattr(fitted, "X_fit_")
        and hasattr(fitted, "_get_kernel")
        and not hasattr(fitted, "support_vectors_")
    ):
        # Kernel Ridge Regression (sklearn ``KernelRidge``) — Phase B-12
        # paper §2.2 headline application (Fig 2 / Table 1). Representer
        # form: w(x_t) = K(x_t, X_fit) (K_train + αI)⁻¹, so
        # ŷ_test = W @ y_train with W having shape (n_test, n_train).
        # The ``support_vectors_`` exclusion guards against SVR (which
        # also has ``dual_coef_``) — SVR has its own kernel-SVR branch
        # below.
        method = "krr_representer"
        alpha = float(getattr(fitted, "alpha", 0.0) or 0.0)
        X_fit = np.asarray(fitted.X_fit_, dtype=float)
        K_train = fitted._get_kernel(X_fit, X_fit)
        K_test = fitted._get_kernel(X_test, X_fit)
        n_fit = K_train.shape[0]
        # Solve (K + αI) Z = K_test^T  ->  W = K_test (K + αI)^{-1}
        # via a single linear solve for numerical stability.
        W = np.linalg.solve(K_train + alpha * np.eye(n_fit), K_test.T).T
    elif (
        hasattr(fitted, "estimators_")
        and hasattr(fitted, "apply")
        and not isinstance(getattr(fitted, "estimators_", None), np.ndarray)
    ):
        # Tree ensemble (RandomForest / ExtraTrees). The boolean check
        # excludes GradientBoostingRegressor whose ``estimators_`` is a
        # numpy ndarray of stages with non-constant per-stage targets;
        # SGB does not have a clean leaf-co-occurrence dual representation
        # because each stage fits residuals, not the original y.
        method = "rf_leaf_cooccurrence_kernel"
        W = _rf_leaf_cooccurrence_weights(fitted, X_train, X_test)
    else:
        _nn_families = {"mlp", "lstm", "gru", "transformer"}
        if getattr(model, "family", None) in _nn_families:
            raise ValueError(
                f"dual_decomposition: NN family {model.family!r} is "
                f"FUTURE — paper §2.3 (Eqs. 9-12) NN dual via auxiliary "
                f"ridge on penultimate-layer activations is not yet "
                f"implemented. Use a linear or tree-bagging family. "
                f"This will be rejected at recipe-validation time in a "
                f"future release."
            )
        raise NotImplementedError(
            f"dual_decomposition does not yet support family "
            f"{model.family!r} ({type(fitted).__name__}). Linear families "
            f"and tree-bagging ensembles (RandomForest / ExtraTrees) are "
            f"operational. Other non-linear families (gradient_boosting / "
            f"xgboost / lightgbm / NN / SVR) need a separate dual "
            f"construction (residual-bagging chains do not admit a clean "
            f"sum-of-training-targets representation); deferred to a "
            f"v0.9.x extension."
        )

    train_labels = [str(idx) for idx in X.index]
    test_labels = [str(idx) for idx in X.index]

    # Aggregate stats per training row across test rows.
    abs_W = np.abs(W)
    # ``importance`` column required by the L7 publishing contract
    # (``_attach_l7_attrs`` sorts by ``importance`` before emitting on
    # ``l7_importance_v1``). Per the dual interpretation, the natural
    # per-training-row magnitude is the mean absolute weight across
    # test rows: training points with consistently large |w| receive
    # consistent dual leverage, mirroring the bar_global semantics
    # used by sibling L7 ops.
    frame = pd.DataFrame(
        {
            "feature": train_labels,
            "importance": abs_W.mean(axis=0),
            "mean_weight": W.mean(axis=0),
            "abs_mean_weight": abs_W.mean(axis=0),
            "max_abs_weight": abs_W.max(axis=0),
        }
    )

    # Attach full weights + portfolio metrics inline.
    weights_full = pd.DataFrame(W, index=test_labels, columns=train_labels)

    # Portfolio metrics per test row, per Goulet Coulombe / Goebel /
    # Klieber (2024) §4 / Eq. p.21:
    #   FC  = forecast concentration (squared-weight Herfindahl alternative)
    #   FSP = forecast short position = Σ I(w<0) · w   (signed; ≤ 0)
    #   FL  = forecast leverage       = Σ w_{ji}        (signed sum, NOT L1 norm)
    #   FT  = forecast turnover       = Σ |w_t − w_{t−1}|
    # The previous v0.9.0B-5 implementation reported `leverage` as the L1
    # norm and `short` as a positive magnitude; v0.9.0F audit-fix
    # restores the paper's signed conventions. Absolute-value variants
    # are still surfaced as ``leverage_l1`` / ``short_abs`` for
    # backward-compatible plotting.
    hhi = (W**2).sum(axis=1)
    short_signed = np.where(W < 0, W, 0.0).sum(axis=1)  # ≤ 0
    leverage_signed = W.sum(axis=1)  # signed sum
    leverage_l1 = abs_W.sum(axis=1)
    short_abs = -short_signed  # ≥ 0 (legacy magnitude)
    turnover = np.zeros(W.shape[0])
    if W.shape[0] > 1:
        turnover[1:] = np.abs(W[1:] - W[:-1]).sum(axis=1)

    portfolio = pd.DataFrame(
        {
            "hhi": hhi,
            "short": short_signed,
            "turnover": turnover,
            "leverage": leverage_signed,
            "leverage_l1": leverage_l1,
            "short_abs": short_abs,
        },
        index=test_labels,
    )

    frame.attrs["dual_weights"] = weights_full
    frame.attrs["portfolio_metrics"] = portfolio
    frame.attrs["method"] = method
    if alpha is not None:
        frame.attrs["alpha"] = alpha
    return frame


def _l7_attention_weights_op(
    model: ModelArtifact,
    X: pd.DataFrame,
    y: pd.Series | None,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Goulet Coulombe (2026) "OLS as an Attention Mechanism" Eq. 7.

    Builds the closed-form attention matrix
    ``Omega = X_test (X'_train X_train)^{-1} X'_train`` so that
    ``y_hat_test = Omega @ y_train`` (representer identity, paper §2.1).

    The op is registered with ``L4ModelArtifactsArtifact`` +
    ``L3FeaturesArtifact`` inputs but the matrix is constructed from
    the L3 feature panel directly (the OLS `LinearRegression` fit object
    fits an internal intercept that is not part of ``X``). Reproducing
    the paper's "attention" object therefore prepends an intercept
    column to ``X`` before solving ``(X'X) β = X'y`` so the row-sum-
    to-one diagnostic (paper §3.2 footnote 1) and the representer
    identity hold by construction.

    The op surfaces:

    * ``frame.attrs["omega"]``   -- ``(n_test, n_train)`` numpy array.
    * ``frame.attrs["train_index"]`` / ``frame.attrs["test_index"]``
                                  -- pandas Index labels.
    * ``frame.attrs["row_sums"]`` -- per-test-row ``Omega.sum(axis=1)``.
                                     Approximately 1.0 with intercept.
    * ``frame.attrs["representer_identity_residual"]`` --
                                     ``max |Omega @ y_train - y_hat_test|``.

    The returned frame's primary columns describe per-training-row
    aggregate weights so the L7 contract (one row per "feature" with an
    ``importance`` magnitude) is honoured -- importance = max |column|
    of ``Omega`` (the paper's "max attention received" diagnostic).

    Linear families only. Other model families would require a kernel
    expansion (paper §5 nonlinear Attention Regression) which is out
    of Phase B-10 scope.
    """

    add_intercept = bool(params.get("add_intercept", True))

    X_train_df = X.fillna(0.0)
    if X_train_df.empty:
        empty = pd.DataFrame(
            {
                "feature": [],
                "importance": [],
                "max_attention_received": [],
                "mean_attention_received": [],
            }
        )
        empty.attrs["omega"] = np.zeros((0, 0))
        empty.attrs["train_index"] = X_train_df.index
        empty.attrs["test_index"] = X_train_df.index
        empty.attrs["row_sums"] = np.zeros(0)
        empty.attrs["representer_identity_residual"] = 0.0
        empty.attrs["method"] = "ols_attention_eq3"
        empty.attrs["add_intercept"] = add_intercept
        return empty

    train_index = X_train_df.index
    test_index = X_train_df.index  # in-sample diagnostic (paper Fig. 1)

    Z_train = X_train_df.to_numpy(dtype=float)
    Z_test = Z_train  # in-sample
    if add_intercept:
        ones = np.ones((Z_train.shape[0], 1))
        Z_train = np.concatenate([ones, Z_train], axis=1)
        ones_test = np.ones((Z_test.shape[0], 1))
        Z_test = np.concatenate([ones_test, Z_test], axis=1)

    # Paper Eq. 7: Omega = X_test (X'X)^{-1} X'_train. ``np.linalg.solve``
    # is more numerically stable than explicit inversion.
    XtX = Z_train.T @ Z_train
    try:
        Omega = Z_test @ np.linalg.solve(XtX, Z_train.T)
    except np.linalg.LinAlgError:
        # Singular Gram matrix (e.g. multicollinearity on the panel) --
        # fall back to the pseudo-inverse so the op never raises on
        # degenerate inputs; the representer identity will still hold up
        # to numerical precision when the design is rank-deficient.
        Omega = Z_test @ np.linalg.pinv(XtX) @ Z_train.T

    row_sums = Omega.sum(axis=1)

    # Representer identity diagnostic. Compute y_hat_test from the
    # closed-form OLS coefficients on the (possibly intercept-augmented)
    # design matrix, mirroring how the runtime would predict in-sample
    # via Eq. 3 rather than via sklearn's separately-stored intercept.
    if y is not None:
        y_train = np.asarray(y, dtype=float).ravel()
        if y_train.shape[0] == Z_train.shape[0]:
            try:
                beta = np.linalg.solve(XtX, Z_train.T @ y_train)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(XtX) @ (Z_train.T @ y_train)
            y_hat_test = Z_test @ beta
            representer_residual = float(np.max(np.abs(Omega @ y_train - y_hat_test)))
        else:
            representer_residual = float("nan")
    else:
        representer_residual = float("nan")

    abs_Omega = np.abs(Omega)
    feature_labels = [str(idx) for idx in train_index]
    frame = pd.DataFrame(
        {
            "feature": feature_labels,
            "importance": abs_Omega.max(axis=0),
            "max_attention_received": abs_Omega.max(axis=0),
            "mean_attention_received": Omega.mean(axis=0),
        }
    )
    frame.attrs["omega"] = Omega
    frame.attrs["train_index"] = train_index
    frame.attrs["test_index"] = test_index
    frame.attrs["row_sums"] = row_sums
    frame.attrs["representer_identity_residual"] = representer_residual
    frame.attrs["method"] = "ols_attention_eq3"
    frame.attrs["add_intercept"] = add_intercept
    return frame


def _l7_anatomy_op(
    op: str,
    model: ModelArtifact,
    X: pd.DataFrame,
    y: pd.Series | None,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Borup, Goulet Coulombe, Rapach, Montes Schütte & Schwenk-Nebbe
    (2022) "Anatomy of Out-of-Sample Forecasting Accuracy" -- adapter
    for the ``anatomy`` PyPI package.

    Two staging paths:

    * **Path B (v0.9.0D)**: uses the *final-window* fitted model for
      every period; degraded approximation of the paper's per-origin
      refit semantics. Status column = ``"degraded"``. Selected when
      ``params["initial_window"]`` is not supplied.
    * **Path A (v0.9.0E)**: when ``params["initial_window"]`` is set,
      anatomy itself splits the panel into expanding-window periods
      (``AnatomySubsets.generate``) and refits a fresh model at every
      origin via the L4 family hyperparameters (``model_factory``).
      Status column = ``"operational"``.

    The two ops are derived from a single ``Anatomy.explain()`` call:

    * ``oshapley_vi`` -- default identity transformer → local per-OOS-
      instance Shapley values. Importance per feature = mean of
      ``|values|`` across OOS instances (paper Eq. 16).
    * ``pbsv`` -- squared-error loss transformer → global scalar PBSV
      per feature (paper Eq. 24).

    The 2026-05-07 audit (errata E3) corrected the v09_paper_coverage
    plan sketch, which had referenced non-existent ``Anatomy.oshapley_vi(...)``
    / ``Anatomy.pbsv(...)`` methods. anatomy 0.1.6 has neither -- both
    are derived from the single ``explain()`` entry point.

    Scope note: Eq. 15 (per-feature oShapley linear closed form,
    φ̂_p^out = β̂_p (x_{p,t} − x̄_p)) and Eq. 22 (SE-PBSV local linear
    closed form) are performance shortcuts described in the paper but
    not implemented as native shortcuts here. The general
    ``Anatomy.explain()`` sampling algorithm is used for all model
    families including linear/ridge, which is correct for all cases
    (the closed forms are simplifications for speed, not correctness
    requirements).
    """

    try:
        from anatomy import (
            Anatomy,
            AnatomyModel,
            AnatomyModelOutputTransformer,
            AnatomyModelProvider,
            AnatomySubsets,
        )
    except ImportError as exc:
        raise NotImplementedError(
            f"L7 op {op!r} requires the [anatomy] extra "
            "(``pip install macroforecast[anatomy]``). The anatomy "
            "package implements Borup et al. (2022) oShapley-VI / PBSV; "
            "macroforecast routes through it via _l7_anatomy_op."
        ) from exc

    feature_cols = list(model.feature_names) if model.feature_names else list(X.columns)
    if not feature_cols:
        return pd.DataFrame(
            {"feature": [], "importance": [], "coefficient": [], "status": []}
        )
    if y is None:
        # Use a zero target if missing -- anatomy still wants a y column
        # for its training frame; the final-window fit is the actual
        # signal source for predictions.
        y = pd.Series(np.zeros(len(X), dtype=float), index=X.index, name="y")
    target_name = str(y.name) if y.name else "y"

    fitted = model.fitted_object
    if not hasattr(fitted, "predict"):
        raise NotImplementedError(
            f"L7 op {op!r} requires a fitted model with a .predict method"
        )

    # Path selection: ``initial_window`` enables Path A (faithful per-
    # origin refit via AnatomySubsets); absence routes to Path B
    # (final-window fit, degraded approximation).
    initial_window = params.get("initial_window")
    is_path_a = initial_window is not None and int(initial_window) > 0
    status_label = "operational" if is_path_a else "degraded"
    if not is_path_a:
        # v0.9.0a0 audit gap-fix: surface the silent Path B routing as an
        # explicit UserWarning rather than only an ex-post status column.
        # Borup et al. (2022) anatomy is paper-faithful only with per-origin
        # refit (Path A); Path B uses the final-window fit and produces a
        # different importance estimand. Suppress with `warnings.simplefilter`
        # if you knowingly want the degraded approximation.
        import warnings

        warnings.warn(
            f"_l7_anatomy_op[{op!r}] routing to Path B (degraded, final-window fit). "
            "Set params['initial_window'] (e.g., int(T * 0.6)) to enable paper-faithful "
            "Path A with AnatomySubsets per-origin refit (Borup et al. 2022).",
            UserWarning,
            stacklevel=2,
        )

    # Build the data block: y first, then features. anatomy passes
    # per-period train/test slices to ``provider_fn`` based on the
    # AnatomySubsets schedule (Path A) or just one full-sample period
    # (Path B; equivalent to AnatomySubsets with initial_window = T-1).
    full_block = pd.concat(
        [y.rename(target_name), X[feature_cols].fillna(0.0)], axis=1
    ).reset_index(drop=True)
    T_full = len(full_block)

    if is_path_a:
        from sklearn.base import clone as sklearn_clone

        try:
            base_estimator = sklearn_clone(fitted)
            can_clone = True
        except Exception:
            can_clone = False

    def _refit(train_df: pd.DataFrame):
        """Per-origin refit (Path A) or pass-through to the final-window
        fit (Path B). Used inside ``_provider_fn``."""
        nonlocal status_label
        if not is_path_a:
            return fitted
        # Path A: clone the sklearn estimator and refit on the period's
        # training window. v0.9.0F audit-fix: when ``sklearn_clone``
        # cannot reproduce the estimator (custom wrappers like
        # ``_VARWrapper`` / ``_DFMMixedFrequency``) we now *demote*
        # the period to degraded rather than silently using the final-
        # window fit while still labelling the result "operational".
        if not can_clone:
            status_label = "degraded"
            import warnings

            warnings.warn(
                f"_l7_anatomy_op[{op!r}] Path A requested but estimator "
                f"{type(fitted).__name__!r} is not sklearn-cloneable; period "
                "demoted to degraded (final-window fit). Wrap your custom "
                "estimator in a sklearn BaseEstimator subclass for paper-"
                "faithful per-origin refit.",
                UserWarning,
                stacklevel=2,
            )
            return fitted
        try:
            new_model = sklearn_clone(base_estimator)
            X_train_period = train_df[feature_cols].to_numpy(dtype=float)
            y_train_period = train_df[target_name].to_numpy(dtype=float)
            new_model.fit(X_train_period, y_train_period)
            return new_model
        except Exception:
            status_label = "degraded"
            return fitted

    def _wrapped_predict(xs: np.ndarray) -> np.ndarray:
        # anatomy passes 2-D ndarrays of shape (n, n_features). Wrap as
        # a DataFrame with the original feature names; if the column
        # count doesn't match, broadcast / pad to the expected width.
        arr = np.asarray(xs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] == 0:
            return np.zeros(0, dtype=float)
        if arr.shape[1] != len(feature_cols):
            # Defensive: if the input is sliced unexpectedly, pad with zeros.
            padded = np.zeros((arr.shape[0], len(feature_cols)), dtype=float)
            padded[:, : min(arr.shape[1], len(feature_cols))] = arr[
                :, : min(arr.shape[1], len(feature_cols))
            ]
            arr = padded
        return np.asarray(
            fitted.predict(pd.DataFrame(arr, columns=feature_cols))
        ).ravel()

    if is_path_a:
        # Path A: AnatomySubsets-driven walk-forward periods.
        try:
            subsets = AnatomySubsets.generate(
                index=full_block.index,
                initial_window=int(initial_window),
                estimation_type=AnatomySubsets.EstimationType.EXPANDING,
                periods=1,
            )
            n_periods = max(1, T_full - int(initial_window))
        except Exception:
            # If AnatomySubsets misbehaves, fall back to single-period.
            subsets = None
            n_periods = 1
    else:
        subsets = None
        n_periods = 1

    def _provider_fn(key):
        period = int(getattr(key, "period", 0))
        if subsets is not None:
            try:
                train_slice = subsets.get_train_subset(period)
                test_slice = subsets.get_test_subset(period)
                train = full_block.iloc[train_slice].reset_index(drop=True)
                test = full_block.iloc[test_slice].reset_index(drop=True)
            except Exception:
                train = full_block
                test = full_block
        else:
            train = full_block
            test = full_block
        # Per-period refit (Path A) or final-window pass-through (Path B).
        period_fitted = _refit(train) if is_path_a else fitted
        nonlocal_pred = period_fitted.predict

        def _period_predict(xs: np.ndarray) -> np.ndarray:
            arr = np.asarray(xs, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[0] == 0:
                return np.zeros(0, dtype=float)
            if arr.shape[1] != len(feature_cols):
                padded = np.zeros((arr.shape[0], len(feature_cols)), dtype=float)
                padded[:, : min(arr.shape[1], len(feature_cols))] = arr[
                    :, : min(arr.shape[1], len(feature_cols))
                ]
                arr = padded
            return np.asarray(
                nonlocal_pred(pd.DataFrame(arr, columns=feature_cols))
            ).ravel()

        anat_model = AnatomyModel(pred_fn=_period_predict)
        return AnatomyModelProvider.PeriodValue(
            train=train, test=test, model=anat_model
        )

    # v0.9.0F audit-fix: previous default 50 was 10× below paper M=500.
    # Restored to 500 to match Borup et al. (2022) p.16 footnote 16.
    # Users running cells in tight CI loops can override via
    # ``params['n_iterations']``.
    n_iterations = int(params.get("n_iterations", 500))
    n_jobs = int(params.get("n_jobs", 1))
    np.random.seed(int(params.get("random_state", 0)))

    provider = AnatomyModelProvider(
        n_periods=n_periods,
        n_features=len(feature_cols),
        model_names=[str(model.model_id)],
        y_name=target_name,
        provider_fn=_provider_fn,
    )
    anat = Anatomy(provider=provider, n_iterations=n_iterations).precompute(
        n_jobs=n_jobs
    )

    if op == "oshapley_vi":
        df = anat.explain()
        # ``df`` columns: ['base_contribution', *feature_names]
        # rows: per (group_name, timestamp). Average |values| over rows.
        per_feat = df.drop(columns="base_contribution").abs().mean(axis=0)
        signed = df.drop(columns="base_contribution").mean(axis=0)
    elif op == "pbsv":

        def _se_global(y_hat, y):
            return float(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2))

        df = anat.explain(
            transformer=AnatomyModelOutputTransformer(transform=_se_global)
        )
        # GLOBAL: single-row DataFrame
        per_feat = df.drop(columns="base_contribution").iloc[0].abs()
        signed = df.drop(columns="base_contribution").iloc[0]
    else:
        raise ValueError(f"_l7_anatomy_op unknown op: {op!r}")

    # Re-align to feature_cols order (anatomy may permute).
    importance = per_feat.reindex(feature_cols).fillna(0.0).values
    coef = signed.reindex(feature_cols).fillna(0.0).values
    return pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": [float(v) for v in importance],
            "coefficient": [float(v) for v in coef],
            "status": [status_label] * len(feature_cols),
        }
    )


def _permutation_importance_frame(
    model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None, *, method: str
) -> pd.DataFrame:
    if y is None or not hasattr(model.fitted_object, "predict"):
        return _linear_importance_frame(model, method=method)
    aligned = pd.concat([X, y.rename("__target__")], axis=1).dropna()
    X_eval = aligned[X.columns]
    y_eval = aligned["__target__"]
    baseline = ((y_eval - model.fitted_object.predict(X_eval)) ** 2).mean()
    rows = []
    for column in X_eval.columns:
        permuted = X_eval.copy()
        permuted[column] = list(reversed(permuted[column].tolist()))
        loss = ((y_eval - model.fitted_object.predict(permuted)) ** 2).mean()
        rows.append(
            {
                "feature": column,
                "importance": float(loss - baseline),
                "coefficient": None,
            }
        )
    return pd.DataFrame(rows)


def _strobl_permutation_importance_frame(
    model: ModelArtifact,
    X: pd.DataFrame,
    y: pd.Series | None,
    *,
    n_bins: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Issue #281 -- Strobl (2008) conditional permutation importance.

    Procedure: for each feature ``j``, partition the rows into bins by
    the *other* features (we use rank quantiles of the most correlated
    other feature); permute ``X[j]`` *within* each bin; compute the
    loss increase. The bin-restricted permutation preserves the
    conditional distribution of ``X[j] | X[-j]`` and removes the bias
    that vanilla permutation introduces under correlated predictors.
    """

    if y is None or not hasattr(model.fitted_object, "predict"):
        return _linear_importance_frame(model, method="permutation_importance_strobl")
    aligned = pd.concat([X, y.rename("__target__")], axis=1).dropna()
    X_eval = aligned[X.columns]
    y_eval = aligned["__target__"]
    baseline = float(((y_eval - model.fitted_object.predict(X_eval)) ** 2).mean())
    rng = np.random.default_rng(int(seed))
    rows = []
    for column in X_eval.columns:
        # Pick the most correlated other feature as the conditioning variable.
        other_cols = [c for c in X_eval.columns if c != column]
        if other_cols:
            corrs = X_eval[other_cols].corrwith(X_eval[column]).abs().fillna(0.0)
            cond_col = corrs.idxmax()
        else:
            cond_col = None
        permuted = X_eval.copy()
        if cond_col is None:
            permuted[column] = rng.permutation(permuted[column].values)
        else:
            try:
                bins = pd.qcut(
                    X_eval[cond_col], q=n_bins, labels=False, duplicates="drop"
                )
            except Exception:
                bins = pd.Series(np.zeros(len(X_eval), dtype=int), index=X_eval.index)
            for bin_id in bins.dropna().unique():
                mask = (bins == bin_id).values
                if mask.sum() <= 1:
                    continue
                values = permuted.loc[mask, column].values
                permuted.loc[mask, column] = rng.permutation(values)
        loss = float(((y_eval - model.fitted_object.predict(permuted)) ** 2).mean())
        rows.append(
            {
                "feature": column,
                "importance": float(loss - baseline),
                "coefficient": None,
                "method": "strobl_conditional",
            }
        )
    return pd.DataFrame(rows)


def _attach_l7_attrs(
    frame: pd.DataFrame,
    model: ModelArtifact,
    method: str,
    l3_features: L3FeaturesArtifact,
) -> pd.DataFrame:
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    frame.attrs.update(
        {
            "method": method,
            "model_id": model.model_id,
            "target": l3_features.y_final.name,
            "horizon": l3_features.horizon_set[0] if l3_features.horizon_set else 1,
        }
    )
    return frame


def _l7_group_aggregate(table: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    grouping = params.get("grouping", "user_defined")
    result = table.copy()
    result["group"] = result["feature"].map(lambda feature: str(feature).split("_")[0])
    grouped = result.groupby("group", as_index=False)["importance"].sum()
    grouped.attrs.update(table.attrs)
    grouped.attrs["grouping"] = grouping
    return grouped


def _l7_lineage_attribution(
    table: pd.DataFrame, metadata: L3MetadataArtifact, params: dict[str, Any]
) -> pd.DataFrame:
    level = params.get("level", "pipeline_name")
    rows = []
    for _, row in table.iterrows():
        lineage = metadata.column_lineage.get(str(row["feature"]))
        pipeline = lineage.pipeline_id if lineage and lineage.pipeline_id else "unknown"
        rows.append({"pipeline": pipeline, "importance": float(row["importance"])})
    result = (
        pd.DataFrame(rows).groupby("pipeline", as_index=False)["importance"].sum()
        if rows
        else pd.DataFrame(columns=["pipeline", "importance"])
    )
    result.attrs.update(table.attrs)
    result.attrs["level"] = level
    return result


def _l7_transformation_attribution(
    l5_eval: L5EvaluationArtifact, params: dict[str, Any]
) -> L7TransformationAttributionArtifact:
    """Issue #218 -- Shapley value over (cell × pipeline) tuples.

    The L5 ``metrics_table`` carries one row per (pipeline = ``model_id``,
    target, horizon). For each (target, horizon) group we treat each
    pipeline as a player in a coalitional game whose value is the
    *negative loss* of the average forecast within the coalition (the
    standard ensemble-as-mean payoff). The Shapley value of pipeline ``i``
    is its average marginal contribution over all subset orderings.

    Three ``decomposition_method`` paths:

    * ``shapley_over_pipelines`` (default): exhaustive Shapley over
      coalitions; tractable for ≤ 8 pipelines per (target, horizon).
    * ``marginal_addition``: ``loss_baseline - loss_with_i`` against the
      worst-pipeline baseline.
    * ``leave_one_out_pipeline``: ``loss_without_i - loss_full_ensemble``.
    """

    from itertools import combinations
    from math import comb

    metrics = l5_eval.metrics_table
    method = params.get("decomposition_method", "shapley_over_pipelines")
    metric = params.get("loss_function", "mse")
    if metrics.empty or metric not in metrics.columns:
        return L7TransformationAttributionArtifact(
            pipeline_contributions={},
            decomposition_method=method,
            loss_function=metric,
            baseline_pipeline=params.get("baseline_pipeline", "simplest"),
            summary_table=pd.DataFrame(
                columns=["pipeline", "target", "horizon", "contribution"]
            ),
        )

    rows: list[dict[str, Any]] = []
    contributions: dict[tuple[Any, ...], float] = {}
    for (target, horizon), group in metrics.groupby(["target", "horizon"]):
        pipelines = group["model_id"].astype(str).tolist()
        losses = group[metric].astype(float).to_numpy()
        n = len(pipelines)
        if n == 0:
            continue
        # Coalition value: payoff is the *negative* mean loss of the
        # coalition (so a lower-loss subset has a higher value).
        if (
            method in {"shapley_over_pipelines", "shapley_over_pipelines_sampled"}
            and n <= 8
            and method != "shapley_over_pipelines_sampled"
        ):
            shapley = np.zeros(n)
            indices = list(range(n))
            for size in range(n):
                for subset in combinations(indices, size):
                    subset_set = set(subset)
                    coalition_loss = (
                        float(np.mean([losses[k] for k in subset])) if subset else 0.0
                    )
                    weight = 1.0 / (n * comb(n - 1, size))
                    for i in indices:
                        if i in subset_set:
                            continue
                        new_subset = list(subset) + [i]
                        new_loss = float(np.mean([losses[k] for k in new_subset]))
                        # Marginal contribution: improvement (loss reduction).
                        shapley[i] += (
                            weight * (coalition_loss - new_loss)
                            if subset
                            else weight * (-new_loss)
                        )
            values = shapley
        elif method in {"shapley_over_pipelines", "shapley_over_pipelines_sampled"}:
            # Issue #254 -- Castro-Gomez-Tejada (2009) sampling-based
            # Shapley. For n > 8 the exhaustive enumeration is 2^n; we
            # approximate via random permutations.
            n_perm = int(params.get("shapley_n_permutations", 1000))
            seed = int(params.get("random_state", 0))
            rng = np.random.default_rng(seed)
            shapley = np.zeros(n)
            indices = np.arange(n)
            for _ in range(n_perm):
                perm = rng.permutation(indices)
                running_subset: list[int] = []
                for k, i in enumerate(perm):
                    if running_subset:
                        prev_loss = float(np.mean([losses[j] for j in running_subset]))
                    else:
                        prev_loss = 0.0
                    running_subset.append(int(i))
                    new_loss = float(np.mean([losses[j] for j in running_subset]))
                    if k == 0:
                        shapley[i] += -new_loss
                    else:
                        shapley[i] += prev_loss - new_loss
            values = shapley / max(n_perm, 1)
        elif method == "leave_one_out_pipeline":
            full_loss = float(np.mean(losses)) if n else 0.0
            values = np.zeros(n)
            for i in range(n):
                without = [losses[j] for j in range(n) if j != i]
                without_loss = float(np.mean(without)) if without else 0.0
                values[i] = without_loss - full_loss
        else:  # marginal_addition (worst-baseline)
            baseline = float(losses.max())
            values = baseline - losses
        for pipeline, value in zip(pipelines, values):
            rows.append(
                {
                    "pipeline": pipeline,
                    "target": target,
                    "horizon": int(horizon),
                    "contribution": float(value),
                }
            )
            contributions[(target, int(horizon), pipeline)] = float(value)
    summary = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["pipeline", "target", "horizon", "contribution"])
    )
    return L7TransformationAttributionArtifact(
        pipeline_contributions=contributions,
        decomposition_method=method,
        loss_function=metric,
        baseline_pipeline=params.get("baseline_pipeline", "simplest"),
        summary_table=summary,
    )


def _l7_sink_targets(raw_sink: Any) -> dict[str, list[str]]:
    if isinstance(raw_sink, str):
        return {"global": [raw_sink]}
    if not isinstance(raw_sink, dict):
        return {}
    result: dict[str, list[str]] = {}
    for label, value in raw_sink.items():
        if isinstance(value, list):
            result[label] = [str(item) for item in value]
        elif value is not None:
            result[label] = [str(value)]
    return result


def _l6_error_frame(
    l4_forecasts: L4ForecastsArtifact, actual: pd.Series
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_id, target, horizon, origin), forecast in l4_forecasts.forecasts.items():
        if origin not in actual.index:
            continue
        actual_value = float(actual.loc[origin])
        forecast_value = float(forecast)
        error = actual_value - forecast_value
        rows.append(
            {
                "model_id": model_id,
                "target": target,
                "horizon": int(horizon),
                "origin": origin,
                "forecast": forecast_value,
                "actual": actual_value,
                "error": error,
                "squared": error**2,
                "absolute": abs(error),
                "forecast_direction": _sign(forecast_value),
                "actual_direction": _sign(actual_value),
            }
        )
    return pd.DataFrame(rows)


def _l6_pair_list(
    sub: dict[str, Any],
    leaf: dict[str, Any],
    model_ids: list[str],
    l4_models: L4ModelArtifactsArtifact,
) -> list[tuple[str, str]]:
    if (
        sub.get("model_pair_strategy") == "user_list"
        or sub.get("nested_pair_strategy") == "user_list"
    ):
        key = (
            "pair_user_list"
            if "model_pair_strategy" in sub
            else "nested_pair_user_list"
        )
        return [tuple(pair) for pair in leaf.get(key, [])]
    benchmark_ids = [
        model_id
        for model_id, is_benchmark in l4_models.is_benchmark.items()
        if is_benchmark
    ]
    if benchmark_ids:
        benchmark_id = benchmark_ids[0]
        return [
            (model_id, benchmark_id)
            for model_id in model_ids
            if model_id != benchmark_id
        ]
    return [
        (left, right)
        for index, left in enumerate(model_ids)
        for right in model_ids[index + 1 :]
    ]



# Cycle 15 M-3 fix: alias deprecation — hides decision_at_5pct from keys()/__iter__/len()
class _L6ResultWithDeprecatedAlias(dict):
    """L6 DM/CW result dict that keeps `decision_at_5pct` accessible for backward
    compat but emits DeprecationWarning and hides it from keys()/__iter__/len()."""

    _DEPRECATED_KEYS = ("decision_at_5pct",)

    def __getitem__(self, key):
        if key in self._DEPRECATED_KEYS:
            warnings.warn(
                f"'{key}' is deprecated; use 'decision' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self._DEPRECATED_KEYS:
            warnings.warn(
                f"'{key}' is deprecated; use 'decision' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().get(key, default)

    def __iter__(self):
        return iter(k for k in super().__iter__() if k not in self._DEPRECATED_KEYS)

    def keys(self):
        return [k for k in super().keys() if k not in self._DEPRECATED_KEYS]

    def items(self):
        return [(k, v) for k, v in super().items() if k not in self._DEPRECATED_KEYS]

    def values(self):
        return [v for k, v in super().items() if k not in self._DEPRECATED_KEYS]

    def __len__(self):
        return sum(1 for k in super().__iter__() if k not in self._DEPRECATED_KEYS)

    def __contains__(self, key):
        # `in` test still finds deprecated keys (backward compat for `if "x" in d`)
        return super().__contains__(key)


def _l6_equal_predictive_results(
    errors: pd.DataFrame,
    sub: dict[str, Any],
    leaf: dict[str, Any],
    l4_models: L4ModelArtifactsArtifact,
) -> dict[tuple[Any, ...], Any]:
    model_ids = sorted(errors["model_id"].unique()) if not errors.empty else []
    pairs = _l6_pair_list(sub, leaf, model_ids, l4_models)
    results: dict[tuple[Any, ...], Any] = {}
    tests = (
        ["dm_diebold_mariano", "gw_giacomini_white"]
        if sub.get("equal_predictive_test") == "multi"
        else [sub.get("equal_predictive_test")]
    )
    # Issue #283 -- when the recipe asks for the Diebold-Mariano-Pesaran
    # joint multi-horizon test, run it once per (model_a, model_b) pair
    # before the per-horizon DM loop and stash the results.
    dmp_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    if (
        "dmp_multi_horizon" in tests
        or sub.get("equal_predictive_test") == "dmp_multi_horizon"
    ):
        dmp_results = _l6_dmp_multi_horizon(
            errors, pairs, leaf.get("dependence_correction", "newey_west")
        )
        tests = [t for t in tests if t != "dmp_multi_horizon"]
    # Phase C M14 -- Harvey-Newbold (1998) MHLN forecast encompassing test.
    # When the recipe asks for ``harvey_newbold_encompassing`` we run a
    # **directional** pair iteration (a→b is distinct from b→a; the test
    # asks "does forecast a encompass forecast b?") and stash one entry
    # per ordered pair. The forecast errors are computed from per-row
    # (forecast, actual) columns in the L5 errors dataframe.
    #
    # Phase C-3 audit-fix (M14): ``_l6_pair_list`` returns ``n·(n−1)/2``
    # *unordered* pairs, but HN encompassing is **asymmetric**
    # (``H_0: A enc B`` ≠ ``H_0: B enc A``). Expand pairs to
    # ``n·(n−1)`` directional pairs *only for the HN test entry* so the
    # downstream DM / GW per-pair iteration still uses unordered pairs
    # (DM is symmetric in the loss difference).
    hn_results: dict[tuple[Any, ...], dict[str, Any]] = {}
    if (
        "harvey_newbold_encompassing" in tests
        or sub.get("equal_predictive_test") == "harvey_newbold_encompassing"
    ):
        ordered_pairs: list[tuple[str, str]] = []
        for a, b in pairs:
            if a == b:
                continue
            ordered_pairs.append((a, b))
            ordered_pairs.append((b, a))
        hn_results = _l6_harvey_newbold_results(
            errors, ordered_pairs, leaf.get("dependence_correction", "newey_west")
        )
        tests = [t for t in tests if t != "harvey_newbold_encompassing"]
    loss_col = "absolute" if sub.get("loss_function") == "absolute" else "squared"
    # Issue #259 -- HAC kernel from L6 globals (newey_west / andrews / parzen).
    hac_kernel = leaf.get("dependence_correction", "newey_west")
    apply_hln = bool(sub.get("hln_correction", True))
    for test_name in tests:
        for model_a, model_b in pairs:
            for (target, horizon), group in errors.groupby(["target", "horizon"]):
                left = group.loc[
                    group["model_id"] == model_a, ["origin", loss_col]
                ].rename(columns={loss_col: "loss_a"})
                right = group.loc[
                    group["model_id"] == model_b, ["origin", loss_col]
                ].rename(columns={loss_col: "loss_b"})
                joined = left.merge(right, on="origin", how="inner")
                diff = joined["loss_a"] - joined["loss_b"]
                stat, p_value = _diebold_mariano_test(
                    diff, horizon=int(horizon), hln=apply_hln, kernel=hac_kernel
                )
                results[(test_name, model_a, model_b, target, int(horizon))] = _L6ResultWithDeprecatedAlias({  # Cycle 15 M-3 fix: alias deprecation
                    "statistic": stat,
                    "p_value": p_value,
                    "decision": p_value is not None and p_value < 0.05,  # Cycle 14 L1-5 fix:
                    "decision_at_5pct": p_value is not None and p_value < 0.05,  # kept for backward compat
                    "alternative": "two_sided",  # Cycle 14 L1-5 fix:
                    "correction_method": "hln_nw" if apply_hln else "nw",  # Cycle 14 L1-5 fix:
                    "n_obs": int(diff.notna().sum()),
                    "mean_loss_difference": _float_or_none(diff.mean())
                    if not diff.empty
                    else None,
                    "hln_correction": apply_hln,
                })
    # Stash the DMP joint-test results next to the DM per-horizon entries.
    for key, payload in dmp_results.items():
        results[key] = payload
    for key, payload in hn_results.items():
        results[key] = payload
    return results


def _l6_harvey_newbold_results(
    errors: pd.DataFrame,
    pairs: list[tuple[str, str]],
    hac_kernel: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Phase C M14 -- Harvey-Newbold (1998) MHLN forecast encompassing test.

    Iterates each ordered (model_a, model_b) pair (a→b is distinct from
    b→a). Per (target, horizon), computes ``d_t = e_a · (e_a - e_b)``
    where ``e_x = actual - forecast_x``, applies the Harvey-Leybourne-
    Newbold (1998) Eq. 5 small-sample correction and returns a
    one-sided t-statistic / p-value (H_0: a encompasses b; H_a:
    combining helps, mean d_t > 0).
    """

    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    if (
        errors.empty
        or "forecast" not in errors.columns
        or "actual" not in errors.columns
    ):
        return out
    for model_a, model_b in pairs:
        if model_a == model_b:
            continue
        for (target, horizon), group in errors.groupby(["target", "horizon"]):
            left = group.loc[
                group["model_id"] == model_a, ["origin", "forecast", "actual"]
            ].rename(columns={"forecast": "f_a", "actual": "y_a"})
            right = group.loc[
                group["model_id"] == model_b, ["origin", "forecast"]
            ].rename(columns={"forecast": "f_b"})
            joined = left.merge(right, on="origin", how="inner").dropna(
                subset=["f_a", "f_b", "y_a"]
            )
            if joined.empty:
                continue
            e_a = (joined["y_a"] - joined["f_a"]).to_numpy(dtype=float)
            e_b = (joined["y_a"] - joined["f_b"]).to_numpy(dtype=float)
            stat, p_value = _harvey_newbold_test(
                e_a,
                e_b,
                horizon=int(horizon),
                kernel=hac_kernel,
                small_sample=True,
            )
            out[
                ("harvey_newbold_encompassing", model_a, model_b, target, int(horizon))
            ] = _L6ResultWithDeprecatedAlias({  # Cycle 15 M-3 fix: alias deprecation
                "statistic": stat,
                "p_value": p_value,
                "decision": p_value is not None and p_value < 0.05,  # Cycle 14 L1-5 fix:
                "decision_at_5pct": p_value is not None and p_value < 0.05,  # kept for backward compat
                "alternative": "one_sided",  # Cycle 14 L1-5 fix: HN is one-sided (a encompasses b)
                "correction_method": "hln_nw",  # Cycle 14 L1-5 fix: Harvey-Leybourne-Newbold small-sample
                "n_obs": int(np.sum(np.isfinite(e_a) & np.isfinite(e_b))),
                "encompassing": "a_over_b",
                "mean_d": float(np.nanmean(e_a * (e_a - e_b))) if e_a.size else None,
                "hac_kernel": hac_kernel,
            })
    return out


def _harvey_newbold_test(
    e_a: np.ndarray,
    e_b: np.ndarray,
    *,
    horizon: int,
    kernel: str = "newey_west",
    small_sample: bool = True,
) -> tuple[float | None, float | None]:
    """Harvey-Leybourne-Newbold (1998) MHLN forecast encompassing test.

    H_0: forecast A encompasses forecast B
    (E[e_a · (e_a - e_b)] = 0).
    H_a: it does not (mean > 0); combining helps.
    """

    e_a = np.asarray(e_a, dtype=float)
    e_b = np.asarray(e_b, dtype=float)
    d = e_a * (e_a - e_b)
    finite = np.isfinite(d)
    n = int(finite.sum())
    if n < 5:
        return None, None
    d_clean = d[finite]
    d_bar = float(np.mean(d_clean))
    nw_lag = max(int(horizon) - 1, 0)
    lr_var = _long_run_variance(d_clean - d_bar, kernel=kernel, lag=nw_lag)
    se = float(np.sqrt(max(lr_var / max(n, 1), 1e-12)))
    if se <= 0:
        return None, None
    stat = d_bar / se
    if small_sample and horizon >= 1:
        # Harvey-Leybourne-Newbold (1998) Eq. 5 small-sample correction.
        h = int(horizon)
        factor = (n + 1 - 2 * h + h * (h - 1) / max(n, 1)) / max(n, 1)
        stat *= float(np.sqrt(max(factor, 1e-12)))
    from scipy import stats as _stats

    # One-sided alternative: encompassing fails when d_bar > 0.
    p_value = float(1.0 - _stats.t.cdf(stat, df=max(n - 1, 1)))
    return float(stat), float(p_value)


def _l6_dmp_multi_horizon(
    errors: pd.DataFrame, pairs: list[tuple[str, str]], hac_kernel: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Issue #283 -- Diebold-Mariano-Pesaran multi-horizon joint test.

    For each (model_a, model_b, target) triplet, stack the per-horizon
    loss differentials end-to-end and compute a HAC-adjusted t-statistic
    on the stacked mean. The joint-null is "equal predictive ability
    across all horizons"; rejection at 5% indicates at least one horizon
    favours model_b.
    """

    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    if errors.empty:
        return out
    for model_a, model_b in pairs:
        for target in errors["target"].unique():
            sub_errors = errors[errors["target"] == target]
            stacked = []
            for horizon, group in sub_errors.groupby("horizon"):
                left = group.loc[
                    group["model_id"] == model_a, ["origin", "squared"]
                ].rename(columns={"squared": "loss_a"})
                right = group.loc[
                    group["model_id"] == model_b, ["origin", "squared"]
                ].rename(columns={"squared": "loss_b"})
                joined = left.merge(right, on="origin", how="inner")
                if not joined.empty:
                    stacked.extend((joined["loss_a"] - joined["loss_b"]).tolist())
            if not stacked:
                continue
            arr = np.asarray(stacked, dtype=float)
            n = arr.size
            mean_diff = float(arr.mean())
            lr_var = _long_run_variance(arr - mean_diff, kernel=hac_kernel)
            se = float(np.sqrt(max(lr_var / n, 1e-12)))
            stat = mean_diff / se if se > 0 else 0.0
            from scipy import stats as _stats

            p_value = float(2 * (1 - _stats.norm.cdf(abs(stat))))
            out[("dmp_multi_horizon", model_a, model_b, target)] = _L6ResultWithDeprecatedAlias({  # Cycle 15 M-3 fix: alias deprecation
                "statistic": stat,
                "p_value": p_value,
                "decision": bool(p_value < 0.05),  # Cycle 14 L1-5 fix:
                "decision_at_5pct": bool(p_value < 0.05),  # kept for backward compat
                "alternative": "two_sided",  # Cycle 14 L1-5 fix:
                "correction_method": "nw",  # Cycle 14 L1-5 fix: DMP uses HAC kernel, no HLN
                "n_obs_stacked": n,
                "mean_loss_difference": mean_diff,
                "hac_kernel": hac_kernel,
            })
    return out


def _l6_nested_results(
    errors: pd.DataFrame,
    sub: dict[str, Any],
    leaf: dict[str, Any],
    l4_models: L4ModelArtifactsArtifact,
) -> dict[tuple[Any, ...], Any]:
    model_ids = sorted(errors["model_id"].unique()) if not errors.empty else []
    pairs = _l6_pair_list(
        {"nested_pair_strategy": sub.get("nested_pair_strategy")},
        leaf,
        model_ids,
        l4_models,
    )
    tests = (
        ["clark_west", "enc_new", "enc_t"]
        if sub.get("nested_test") == "multi"
        else [sub.get("nested_test")]
    )
    apply_cw = bool(sub.get("cw_adjustment", True))
    hac_kernel = leaf.get("dependence_correction", "newey_west")
    results: dict[tuple[Any, ...], Any] = {}
    for test_name in tests:
        for large_model, small_model in pairs:
            for (target, horizon), group in errors.groupby(["target", "horizon"]):
                small = group.loc[
                    group["model_id"] == small_model, ["origin", "squared", "forecast"]
                ].rename(columns={"squared": "loss_small", "forecast": "f_small"})
                large = group.loc[
                    group["model_id"] == large_model, ["origin", "squared", "forecast"]
                ].rename(columns={"squared": "loss_large", "forecast": "f_large"})
                joined = small.merge(large, on="origin", how="inner")
                improvement = joined["loss_small"] - joined["loss_large"]
                if test_name == "clark_west" and apply_cw:
                    adjustment = (joined["f_small"] - joined["f_large"]) ** 2
                    f_value = improvement + adjustment
                else:
                    f_value = improvement
                stat, p_value = _diebold_mariano_test(
                    f_value, horizon=int(horizon), hln=False, kernel=hac_kernel
                )
                # CW is a one-sided test (H_a: large model improves on small)
                p_value = (
                    (p_value / 2.0)
                    if (p_value is not None and stat is not None and stat > 0)
                    else p_value
                )
                results[(test_name, small_model, large_model, target, int(horizon))] = _L6ResultWithDeprecatedAlias({  # Cycle 15 M-3 fix: alias deprecation
                    "statistic": stat,
                    "p_value": p_value,
                    "decision": p_value is not None and p_value < 0.05,  # Cycle 14 L1-5 fix:
                    "decision_at_5pct": p_value is not None and p_value < 0.05,  # kept for backward compat
                    "alternative": "one_sided",  # Cycle 14 L1-5 fix: CW is one-sided (large improves on small)
                    "correction_method": "nw",  # Cycle 14 L1-5 fix: HAC NW kernel, no HLN for nested
                    "n_obs": int(f_value.notna().sum()),
                    "mean_adjusted_improvement": _float_or_none(f_value.mean())
                    if not f_value.empty
                    else None,
                    "cw_adjustment": apply_cw,
                })
    return results


def _gr_critical_value(window_ratio: float, alpha: float) -> float:
    """Issue #248 -- Giacomini-Rossi (2010) supremum-of-Brownian-bridge
    critical value, simulated per (m/T ratio, alpha).

    Each call uses a small Monte Carlo over standard Brownian-bridge
    increments to recover the alpha-th quantile of the supremum of the
    standardised cumulative deviation. Cached by (rounded ratio, alpha).
    """

    cache = _gr_critical_value._cache  # type: ignore[attr-defined]
    key = (round(float(window_ratio), 2), round(float(alpha), 4))
    if key in cache:
        return cache[key]
    rng = np.random.default_rng(
        int(round(float(window_ratio) * 100) + int(alpha * 1000))
    )
    # Vectorised simulation -- much faster than the row-by-row form.
    n_sims = 1000
    n_grid = 200
    m = max(1, int(round(window_ratio * n_grid)))
    paths = rng.normal(size=(n_sims, n_grid))
    # Cumulative window sums via convolution: rolling sum of length m.
    kernel = np.ones(m)
    rolling_sums = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), 1, paths
    )
    rolling_means = rolling_sums / m
    # Standardise: window std is approx 1/sqrt(m) under N(0, 1).
    rolling_stat = np.abs(rolling_means * np.sqrt(m))
    sup_stats = rolling_stat.max(axis=1)
    cv = float(np.quantile(sup_stats, 1.0 - float(alpha)))
    cache[key] = cv
    return cv


_gr_critical_value._cache = {}  # type: ignore[attr-defined]


def _l6_density_interval_results(
    l4_forecasts: L4ForecastsArtifact,
    l1_artifact: L1DataDefinitionArtifact,
    l3_features: L3FeaturesArtifact,
    sub: dict[str, Any],
) -> dict[tuple[Any, ...], Any]:
    """Issue #200 + #247 -- PIT-Berkowitz / KS / Christoffersen / Kupiec.

    Strict mode (default): require ``forecast_intervals`` from a
    quantile / density forecast. ``leaf_config.allow_residual_synth``
    opts back into the v0.2 residual-Gaussian synth path for legacy
    callers, but the published density tests are designed for real
    quantile / density forecasts.
    """

    actual = l3_features.y_final.metadata.values.get("data")
    if not isinstance(actual, pd.Series) or actual.dropna().empty:
        return {"status": "no_actuals", "tests": {}}

    allow_synth = bool(sub.get("allow_residual_synth", False))
    intervals = getattr(l4_forecasts, "forecast_intervals", {}) or {}
    if not intervals and not allow_synth:
        return {
            "status": "requires_quantile_or_density_forecast",
            "remediation": "set ``forecast_object: quantile`` on the fit_model node, or pass ``L6.E.allow_residual_synth: true`` to opt into the residual-Gaussian fallback (not the published procedure)",
            "tests": {},
        }
    by_model: dict[str, list[tuple[Any, float, float]]] = {}
    if intervals:
        # quantile mode -- compute PIT from per-(model, target, horizon, origin, q)
        # tuples. For each origin we approximate PIT by inverse-CDF
        # interpolation across the supplied quantile levels.
        for (model_id, target, horizon, origin, q), value in intervals.items():
            if origin in actual.index:
                by_model.setdefault(str(model_id), []).append(
                    (origin, float(q), float(value))
                )

    pit_series_by_model: dict[str, np.ndarray] = {}
    for model_id, rows in by_model.items():
        # Group by origin -> sorted quantile levels.
        by_origin: dict[Any, list[tuple[float, float]]] = {}
        for origin, q, value in rows:
            by_origin.setdefault(origin, []).append((q, value))
        pit: list[float] = []
        for origin, qvs in by_origin.items():
            qvs.sort()
            qs = np.asarray([qv[0] for qv in qvs])
            vs = np.asarray([qv[1] for qv in qvs])
            target_value = float(actual.loc[origin])
            if vs.size == 0:
                pit.append(0.5)
            else:
                pit.append(float(np.interp(target_value, vs, qs, left=0.0, right=1.0)))
        pit_series_by_model[model_id] = np.asarray(pit)
    if not pit_series_by_model:
        # Synthesize from residuals: Gaussian density centred at point
        # forecasts with std = residual std. PIT = Phi((y - f) / sigma).
        residuals_by_model: dict[str, list[float]] = {}
        for (
            model_id,
            target,
            horizon,
            origin,
        ), forecast in l4_forecasts.forecasts.items():
            if origin in actual.index:
                residuals_by_model.setdefault(str(model_id), []).append(
                    float(actual.loc[origin]) - float(forecast)
                )
        from scipy import stats as _stats  # type: ignore

        for model_id, resid in residuals_by_model.items():
            arr = np.asarray(resid, dtype=float)
            sigma = float(arr.std(ddof=1)) or 1.0
            pit_series_by_model[model_id] = _stats.norm.cdf(arr / sigma)

    # Run the four tests on each PIT series.
    out: dict[tuple[Any, ...], Any] = {}
    alpha = float(sub.get("alpha", 0.05))
    for model_id, pit in pit_series_by_model.items():
        if pit.size < 8:
            out[("density", model_id)] = {
                "status": "insufficient_data",
                "n": int(pit.size),
            }
            continue
        result = _density_interval_battery(pit, alpha=alpha)
        out[("density", model_id)] = result
    return out


def _density_interval_battery(
    pit: np.ndarray, *, alpha: float = 0.05
) -> dict[str, Any]:
    """Berkowitz / KS / Christoffersen / Kupiec battery."""

    from scipy import stats as _stats  # type: ignore

    pit = np.clip(np.asarray(pit, dtype=float), 1e-9, 1 - 1e-9)
    # Berkowitz (2001): inverse-normal of PIT; AR(1) on the transformed
    # series under H0 of i.i.d. N(0, 1).
    z = _stats.norm.ppf(pit)
    z_mean = z.mean()
    z_std = z.std(ddof=1) if z.size > 1 else 1.0
    berkowitz = {"mean": float(z_mean), "std": float(z_std)}
    # Likelihood ratio for H0: mu=0, sigma=1.
    if z_std > 0:
        ll_h1 = float(_stats.norm.logpdf(z, loc=z_mean, scale=z_std).sum())
        ll_h0 = float(_stats.norm.logpdf(z, loc=0.0, scale=1.0).sum())
        lr = -2.0 * (ll_h0 - ll_h1)
        berkowitz["lr_statistic"] = float(lr)
        berkowitz["p_value"] = float(1.0 - _stats.chi2.cdf(lr, df=2))
        berkowitz["reject"] = bool(berkowitz["p_value"] < alpha)
    # Kolmogorov-Smirnov against uniform.
    ks_stat, ks_pvalue = _stats.kstest(pit, "uniform")
    # Kupiec POF (proportion of failures) for VaR coverage at alpha.
    hits = (pit < alpha).astype(int)
    p_hat = float(hits.mean()) if hits.size else 0.0
    if 0 < p_hat < 1:
        ll_ratio = -2.0 * (
            hits.size * (alpha * np.log(alpha) + (1 - alpha) * np.log(1 - alpha))
            - hits.size * (p_hat * np.log(p_hat) + (1 - p_hat) * np.log(1 - p_hat))
        )
        kupiec_p = float(1.0 - _stats.chi2.cdf(ll_ratio, df=1))
    else:
        ll_ratio = 0.0
        kupiec_p = 1.0
    # Christoffersen independence: LR test on transitions of hits.
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(hits[:-1], hits[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(hits.size - 1, 1)
    if 0 < pi < 1 and 0 < pi01 < 1 and 0 < pi11 < 1:
        ll_ind = -2.0 * (
            (n01 + n11) * np.log(pi)
            + (n00 + n10) * np.log(1 - pi)
            - n01 * np.log(pi01)
            - n00 * np.log(1 - pi01)
            - n11 * np.log(pi11)
            - n10 * np.log(1 - pi11)
        )
        christoffersen_p = float(1.0 - _stats.chi2.cdf(ll_ind, df=1))
    else:
        ll_ind = 0.0
        christoffersen_p = 1.0
    # Issue #276 -- Engle-Manganelli (2004) Dynamic Quantile test.
    # Regress hits on (constant, lag1 hit, lag2 hit, lag3 hit) and chi-square
    # on the joint significance of the lagged regressors.
    dq_p = 1.0
    dq_stat = 0.0
    if hits.size >= 8:
        try:
            n_lags = 3
            X_dq = np.column_stack(
                [np.ones(hits.size - n_lags)]
                + [hits[n_lags - lag : -lag] for lag in range(1, n_lags + 1)]
            )
            y_dq = hits[n_lags:].astype(float)
            coef, _, _, _ = np.linalg.lstsq(X_dq, y_dq, rcond=None)
            preds = X_dq @ coef
            ssr = float(np.sum((y_dq - preds) ** 2))
            tss = float(np.sum((y_dq - y_dq.mean()) ** 2))
            r2 = 1.0 - ssr / tss if tss > 0 else 0.0
            dq_stat = float(hits.size * r2)
            dq_p = float(1.0 - _stats.chi2.cdf(dq_stat, df=n_lags))
        except Exception:
            dq_stat = 0.0
            dq_p = 1.0
    return {
        "berkowitz": berkowitz,
        "ks": {
            "statistic": float(ks_stat),
            "p_value": float(ks_pvalue),
            "reject": bool(ks_pvalue < alpha),
        },
        "kupiec_pof": {
            "hits_rate": p_hat,
            "lr_statistic": float(ll_ratio),
            "p_value": kupiec_p,
            "reject": bool(kupiec_p < alpha),
        },
        "christoffersen_independence": {
            "lr_statistic": float(ll_ind),
            "p_value": christoffersen_p,
            "reject": bool(christoffersen_p < alpha),
        },
        "engle_manganelli_dq": {
            "statistic": dq_stat,
            "p_value": dq_p,
            "reject": bool(dq_p < alpha),
            "n_lags": 3,
        },
        "n_obs": int(pit.size),
    }


def _l6_cpa_results(
    errors: pd.DataFrame, sub: dict[str, Any], l4_models: L4ModelArtifactsArtifact
) -> dict[tuple[Any, ...], Any]:
    """Issue #199 -- Giacomini & Rossi (2010) rolling-window fluctuation
    test + Rossi & Sekhposyan (2010) recursive-window variant.

    For each (model_a, model_b) pair and each (target, horizon):

    1. Compute the loss differential d_t = L(e^A_t) - L(e^B_t).
    2. Centre by the full-sample mean.
    3. Compute a rolling window mean of d_t with bandwidth ``m``,
       standardised by a Newey-West HAC estimate of the long-run variance.
    4. Report ``max_t |F_t|``: the maximum standardised rolling mean.

    The Giacomini-Rossi statistic is asymptotically supremum of a
    Brownian bridge functional; we report the published critical-value
    constant ``k_alpha = 2.7727`` for ``alpha = 0.05`` and ``m/T`` in
    [0.1, 0.5] (Giacomini-Rossi 2010 Table 1, two-sided test). Simulated
    critical values per ``m/T`` ratio remain a follow-up.

    For the recursive variant we replace the rolling window with an
    expanding window starting at ``m``.
    """

    results: dict[tuple[Any, ...], Any] = {}
    model_ids = sorted(errors["model_id"].unique()) if not errors.empty else []
    pairs = _l6_pair_list(
        {"model_pair_strategy": "vs_benchmark_only"}, {}, model_ids, l4_models
    )
    tests = (
        ["giacomini_rossi_2010", "rossi_sekhposyan"]
        if sub.get("cpa_test") == "multi"
        else [sub.get("cpa_test")]
    )
    window_ratio = float(sub.get("cpa_window_ratio", 0.25))
    alpha = float(sub.get("cpa_alpha", 0.05))
    # Issue #248 -- look up the simulated critical value at the matching
    # m/T ratio. Falls back to the v0.2 constant when the ratio is out of
    # the simulated grid.
    k_alpha = _gr_critical_value(window_ratio, alpha)

    def _newey_west_se(d: np.ndarray, lags: int) -> float:
        n = d.size
        if n == 0:
            return 1.0
        de = d - d.mean()
        gamma0 = float(np.dot(de, de) / n)
        var = gamma0
        for lag in range(1, lags + 1):
            weight = 1.0 - lag / (lags + 1)
            cov = float(np.dot(de[lag:], de[:-lag]) / n)
            var += 2.0 * weight * cov
        return float(np.sqrt(max(var / n, 1e-12)))

    for test_name in tests:
        for model_a, model_b in pairs:
            for (target, horizon), group in errors.groupby(["target", "horizon"]):
                left = group.loc[
                    group["model_id"] == model_a, ["origin", "squared"]
                ].rename(columns={"squared": "loss_a"})
                right = group.loc[
                    group["model_id"] == model_b, ["origin", "squared"]
                ].rename(columns={"squared": "loss_b"})
                joined = left.merge(right, on="origin", how="inner").sort_values(
                    "origin"
                )
                if joined.empty:
                    results[(test_name, (model_a, model_b), target, int(horizon))] = {
                        "statistic": None,
                        "critical_value": None,
                        "p_value": None,
                        "time_path": [],
                        "decision": None,
                        "window_size": 0,
                    }
                    continue
                d = joined["loss_a"].to_numpy() - joined["loss_b"].to_numpy()
                centered = d - d.mean()
                T = d.size
                m = max(4, int(round(window_ratio * T)))
                # Rolling (Giacomini-Rossi) or expanding (Rossi-Sekhposyan).
                stats: list[float] = []
                for t in range(m, T + 1):
                    if test_name == "giacomini_rossi_2010":
                        window = centered[t - m : t]
                    else:  # rossi_sekhposyan recursive
                        window = centered[:t]
                    nw_lags = max(1, int(np.ceil(m ** (1 / 3))))
                    se = _newey_west_se(window, nw_lags)
                    if se > 0:
                        stat_t = float(window.mean() / se)
                        stats.append(stat_t)
                supremum = float(max(abs(s) for s in stats)) if stats else 0.0
                decision = bool(abs(supremum) > k_alpha)
                results[(test_name, (model_a, model_b), target, int(horizon))] = {
                    "statistic": supremum,
                    "critical_value": k_alpha,
                    "p_value": None,  # finite-sample p-values via simulation = follow-up
                    "time_path": stats,
                    "decision": decision,
                    "window_size": m,
                    "n_obs": T,
                }
    return results


def _l6_multiple_model_results(
    metrics: pd.DataFrame,
    sub: dict[str, Any],
    *,
    per_origin_panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """MCS / SPA / Reality Check / StepM.

    When ``per_origin_panel`` carries a non-empty per-(model, target,
    horizon, origin) loss table, this runs the **Hansen (2005) stationary
    block bootstrap** on per-origin loss differentials -- the academic
    paper-grade procedure. SPA-Hansen, Reality-Check-White, and StepM
    (Romano-Wolf) share the same bootstrap pool with the appropriate
    studentization or step-down rule.

    When the panel is empty (legacy summary-only L5 path) the function
    falls back to a parametric Gaussian bootstrap on the cross-sectional
    spread of model-mean losses; the returned ``bootstrap_kind`` field
    flags which mode produced the numbers.
    """

    if per_origin_panel is not None and not per_origin_panel.empty:
        return _mcs_from_per_origin_panel(per_origin_panel, sub)
    return _mcs_from_summary_metrics(metrics, sub)


def _mcs_from_summary_metrics(
    metrics: pd.DataFrame, sub: dict[str, Any]
) -> dict[str, Any]:
    """Legacy fallback: parametric Gaussian bootstrap on cross-sectional
    model-mean losses (used when L5 didn't carry a per-origin panel)."""

    if metrics.empty:
        return {
            "mcs_inclusion": {},
            "spa_p_values": {},
            "reality_check_p_values": {},
            "stepm_rejected": {},
        }
    metric = (
        "mse"
        if "mse" in metrics.columns
        else metrics.select_dtypes("number").columns[0]
    )
    alpha = float(sub.get("mcs_alpha", 0.10))
    n_boot = int(sub.get("bootstrap_n_replications", 1000))
    block_length = sub.get("bootstrap_block_length", "auto")
    rng = np.random.default_rng(0)
    mcs: dict[tuple[Any, ...], set[str]] = {}
    spa: dict[tuple[Any, ...], float] = {}
    reality: dict[tuple[Any, ...], float] = {}
    stepm: dict[tuple[Any, ...], set[str]] = {}
    for (target, horizon), group in metrics.groupby(["target", "horizon"]):
        loss = group.set_index("model_id")[metric].astype(float)
        models = loss.index.tolist()
        if len(models) < 2:
            mcs[(target, int(horizon), alpha)] = set(models)
            spa[(target, int(horizon))] = 1.0
            reality[(target, int(horizon))] = 1.0
            stepm[(target, int(horizon), alpha)] = set()
            continue
        # Parametric Gaussian bootstrap over the cross-sectional spread of
        # model-mean losses (see warning in docstring -- this is *not* a
        # block bootstrap on per-origin loss differentials).
        diffs = loss.values - loss.values.mean()
        scale = float(np.std(diffs)) if np.std(diffs) > 0 else 1e-6
        boot_max = np.empty(n_boot)
        for b in range(n_boot):
            sample = rng.normal(loc=0.0, scale=scale, size=len(diffs))
            boot_max[b] = float(np.max(np.abs(sample)))
        observed = float(np.max(np.abs(diffs)))
        p_value = float((boot_max >= observed).mean())
        included = {
            model
            for model, value in zip(models, loss.values)
            if value <= loss.min() + scale * np.quantile(boot_max, 1 - alpha)
        }
        if not included:
            included = {loss.idxmin()}
        key = (target, int(horizon), alpha)
        mcs[key] = included
        spa[(target, int(horizon))] = float(p_value)
        reality[(target, int(horizon))] = float(p_value)
        stepm[key] = set(models) - included
    return {
        "mcs_inclusion": mcs,
        "spa_p_values": spa,
        "reality_check_p_values": reality,
        "stepm_rejected": stepm,
        "bootstrap_n_replications": n_boot,
        "block_length": block_length,
        "bootstrap_kind": "parametric_gaussian_cross_sectional",
    }


def _mcs_from_per_origin_panel(
    panel: pd.DataFrame, sub: dict[str, Any]
) -> dict[str, Any]:
    """Hansen (2005) MCS via Politis-Romano (1994) stationary block bootstrap
    on per-origin loss differentials.

    For each (target, horizon) slice we form an (origin x model) loss matrix
    L. The Hansen MCS test statistic is

        T_max = max_i  (L_bar_i - L_bar_mean) / sqrt(Var(L_bar_i))

    where L_bar_i is the time-mean of model i's losses and the variance is
    estimated from the bootstrap pool. Models whose deviation from the
    cross-model mean lands inside the (1 - alpha) bootstrap quantile of the
    null distribution (built by stationary block bootstrap on demeaned
    per-origin losses) are retained in the MCS. SPA / Reality Check / StepM
    reuse the bootstrap pool with their own studentization / step-down rule.
    """

    metric_col = (
        "squared_error"
        if sub.get("mmt_loss_function", "squared") == "squared"
        else "absolute_error"
    )
    if metric_col not in panel.columns:
        return _mcs_from_summary_metrics(pd.DataFrame(), sub)
    alpha = float(sub.get("mcs_alpha", 0.10))
    n_boot = int(sub.get("bootstrap_n_replications", 1000))
    block_length_axis = sub.get("bootstrap_block_length", "auto")
    bootstrap_method = sub.get("bootstrap_method", "stationary_bootstrap")
    rng = np.random.default_rng(0)

    mcs: dict[tuple[Any, ...], set[str]] = {}
    spa: dict[tuple[Any, ...], float] = {}
    reality: dict[tuple[Any, ...], float] = {}
    stepm: dict[tuple[Any, ...], set[str]] = {}
    block_lengths_used: dict[tuple[Any, ...], int] = {}

    for (target, horizon), slice_df in panel.groupby(["target", "horizon"]):
        wide = slice_df.pivot_table(
            index="origin", columns="model_id", values=metric_col, aggfunc="mean"
        ).sort_index()
        wide = wide.dropna(axis=0, how="any")
        if wide.shape[0] < 4 or wide.shape[1] < 2:
            mcs[(target, int(horizon), alpha)] = set(wide.columns)
            spa[(target, int(horizon))] = 1.0
            reality[(target, int(horizon))] = 1.0
            stepm[(target, int(horizon), alpha)] = set()
            continue

        L = wide.to_numpy(dtype=float)
        n_obs, n_models = L.shape
        block_length = _resolve_block_length(L, block_length_axis)
        block_lengths_used[(target, int(horizon))] = int(block_length)

        # Real losses centered model-by-model for the bootstrap null.
        means = L.mean(axis=0)
        centered = L - means

        # Stationary or fixed block bootstrap pool.
        boot_means = np.empty((n_boot, n_models))
        for b in range(n_boot):
            indices = (
                _stationary_bootstrap_indices(n_obs, block_length, rng)
                if bootstrap_method == "stationary_bootstrap"
                else _fixed_block_bootstrap_indices(n_obs, block_length, rng)
            )
            boot_means[b] = centered[indices].mean(axis=0)

        # MCS statistic: deviation from cross-model mean, studentized by
        # bootstrap variance.
        cross_mean = means.mean()
        observed_dev = means - cross_mean
        boot_dev = boot_means - boot_means.mean(axis=1, keepdims=True)
        boot_var = boot_dev.var(axis=0, ddof=1)
        boot_var = np.where(boot_var <= 0, 1e-12, boot_var)
        observed_t = observed_dev / np.sqrt(boot_var)
        boot_t = boot_dev / np.sqrt(boot_var)
        boot_t_max = boot_t.max(axis=1)
        critical = float(np.quantile(boot_t_max, 1 - alpha))
        mcs_set = {
            str(model) for model, t in zip(wide.columns, observed_t) if t <= critical
        }
        if not mcs_set:
            mcs_set = {str(wide.columns[int(np.argmin(observed_t))])}

        # SPA / Reality Check p-value via the same bootstrap pool: best-model
        # vs benchmark relative loss (using the cross-model best as the proxy).
        relative = boot_means - boot_means[:, [int(np.argmin(means))]]
        spa_stat = float((relative.max(axis=1) >= np.max(means - means.min())).mean())
        spa_p = float(spa_stat)

        # StepM (Romano-Wolf) iteratively trims models whose t exceeds the
        # bootstrap critical at each step.
        stepm_rejected: set[str] = set()
        active = list(wide.columns)
        active_idx = list(range(n_models))
        while len(active_idx) > 1:
            sub_means = means[active_idx]
            sub_centered = centered[:, active_idx]
            sub_boot = np.empty((n_boot, len(active_idx)))
            for b in range(n_boot):
                idx = _stationary_bootstrap_indices(n_obs, block_length, rng)
                sub_boot[b] = sub_centered[idx].mean(axis=0)
            sub_dev = sub_means - sub_means.mean()
            sub_boot_dev = sub_boot - sub_boot.mean(axis=1, keepdims=True)
            sub_boot_var = sub_boot_dev.var(axis=0, ddof=1)
            sub_boot_var = np.where(sub_boot_var <= 0, 1e-12, sub_boot_var)
            sub_t = sub_dev / np.sqrt(sub_boot_var)
            sub_boot_t_max = (sub_boot_dev / np.sqrt(sub_boot_var)).max(axis=1)
            critical_sub = float(np.quantile(sub_boot_t_max, 1 - alpha))
            worst_pos = int(np.argmax(sub_t))
            if sub_t[worst_pos] > critical_sub:
                stepm_rejected.add(str(active[worst_pos]))
                del active[worst_pos]
                del active_idx[worst_pos]
            else:
                break

        key = (target, int(horizon), alpha)
        mcs[key] = mcs_set
        spa[(target, int(horizon))] = spa_p
        reality[(target, int(horizon))] = spa_p
        stepm[key] = stepm_rejected

    return {
        "mcs_inclusion": mcs,
        "spa_p_values": spa,
        "reality_check_p_values": reality,
        "stepm_rejected": stepm,
        "bootstrap_n_replications": n_boot,
        "block_length": block_length_axis,
        "block_lengths_used": block_lengths_used,
        "bootstrap_kind": "stationary_block_bootstrap_per_origin",
    }


def _resolve_block_length(L: np.ndarray, axis_value: Any) -> int:
    """Politis-White (2004) automatic block length when axis_value=='auto',
    else the integer the recipe specified.

    The Politis-White formula picks the block length that minimizes
    asymptotic MSE of the long-run variance estimator. We approximate via the
    rule-of-thumb b_opt = floor(2 * (4 * n / 100) ** (1/3)) for stationarity,
    then floor it to >= 1 and clip to <= n // 2.
    """

    n = L.shape[0]
    if isinstance(axis_value, (int, np.integer)) and axis_value > 0:
        return max(1, min(int(axis_value), n // 2 if n > 1 else 1))
    if isinstance(axis_value, str) and axis_value.isdigit():
        return max(1, min(int(axis_value), n // 2 if n > 1 else 1))
    # Politis-White auto rule-of-thumb.
    block = max(1, int(np.floor(2 * (4 * n / 100) ** (1 / 3))))
    return min(block, max(1, n // 2))


def _stationary_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano (1994) stationary block bootstrap index draw.

    Each step: with probability 1/block_length, restart at a random index
    drawn uniformly from {0, ..., n-1}; otherwise advance one step (mod n).
    """

    if block_length <= 1:
        return rng.integers(0, n, size=n)
    p_restart = 1.0 / block_length
    indices = np.empty(n, dtype=np.int64)
    indices[0] = int(rng.integers(0, n))
    restarts = rng.random(n - 1) < p_restart
    new_starts = rng.integers(0, n, size=n - 1)
    for t in range(1, n):
        if restarts[t - 1]:
            indices[t] = int(new_starts[t - 1])
        else:
            indices[t] = (indices[t - 1] + 1) % n
    return indices


def _fixed_block_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Kunsch (1989) circular fixed-block bootstrap. Used when the recipe
    selects ``bootstrap_method = block``."""

    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    indices = np.empty(n_blocks * block_length, dtype=np.int64)
    for k, start in enumerate(starts):
        indices[k * block_length : (k + 1) * block_length] = (
            start + np.arange(block_length)
        ) % n
    return indices[:n]


def _l6_direction_results(
    errors: pd.DataFrame, sub: dict[str, Any], leaf: dict[str, Any]
) -> dict[tuple[Any, ...], Any]:
    threshold = (
        leaf.get("direction_threshold_value", 0.0)
        if sub.get("direction_threshold") == "user_defined"
        else 0.0
    )
    results: dict[tuple[Any, ...], Any] = {}
    tests = (
        ["pesaran_timmermann_1992", "henriksson_merton"]
        if sub.get("direction_test") == "multi"
        else [sub.get("direction_test")]
    )
    for test_name in tests:
        for (model_id, target, horizon), group in errors.groupby(
            ["model_id", "target", "horizon"]
        ):
            forecast_dir = (group["forecast"] - threshold).gt(0).astype(int).to_numpy()
            actual_dir = (group["actual"] - threshold).gt(0).astype(int).to_numpy()
            stat, p_value, success = _pesaran_timmermann_test(
                forecast_dir, actual_dir, test_name=test_name
            )
            results[(test_name, model_id, target, int(horizon))] = {
                "statistic": stat,
                "p_value": p_value,
                "success_ratio": success,
            }
    return results


def _pesaran_timmermann_test(
    forecast: np.ndarray, actual: np.ndarray, *, test_name: str
) -> tuple[float | None, float | None, float | None]:
    n = len(forecast)
    if n < 2:
        return None, None, None
    success = float((forecast == actual).mean())
    p_y = float(actual.mean())
    p_x = float(forecast.mean())
    p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
    if p_star <= 0 or p_star >= 1:
        return None, None, success
    var_p = (p_star * (1 - p_star)) / n
    var_p_star = (
        ((2 * p_y - 1) ** 2 * p_x * (1 - p_x)) / n
        + ((2 * p_x - 1) ** 2 * p_y * (1 - p_y)) / n
        + (4 * p_y * p_x * (1 - p_y) * (1 - p_x)) / (n * n)
    )
    denom = max(var_p - var_p_star, 1e-12)
    if denom <= 0:
        return None, None, success
    statistic = (success - p_star) / math.sqrt(denom)
    if test_name == "henriksson_merton":
        # HM: joint sum of conditional hit rates (up + down). Asymptotic Z under H0.
        up_correct = ((forecast == 1) & (actual == 1)).sum()
        down_correct = ((forecast == 0) & (actual == 0)).sum()
        n_up = max(int((actual == 1).sum()), 1)
        n_down = max(int((actual == 0).sum()), 1)
        joint = (up_correct / n_up) + (down_correct / n_down)
        statistic = (joint - 1.0) * math.sqrt(min(n_up, n_down))
    return float(statistic), _normal_two_sided_p(statistic), success


def _l6_residual_results(
    errors: pd.DataFrame, sub: dict[str, Any]
) -> dict[tuple[Any, ...], Any]:
    results: dict[tuple[Any, ...], Any] = {}
    tests = list(sub.get("residual_test", []))
    if "multi" in tests:
        tests = [
            "ljung_box_q",
            "arch_lm",
            "jarque_bera_normality",
            "breusch_godfrey_serial_correlation",
            "durbin_watson",
        ]
    lag = int(sub.get("residual_lag_count", 10))
    for (model_id, target, horizon), group in errors.groupby(
        ["model_id", "target", "horizon"]
    ):
        residuals = group.sort_values("origin")["error"].dropna()
        for test_name in tests:
            statistic, p_value = _residual_test_statistic(test_name, residuals, lag)
            results[(test_name, model_id, target, int(horizon))] = {
                "statistic": statistic,
                "p_value": p_value,
                "lag_used": min(lag, max(len(residuals) - 1, 0)),
            }
    return results


def _t_statistic(values: pd.Series) -> tuple[float | None, float | None]:
    clean = values.dropna()
    if len(clean) < 2:
        return None, None
    std = float(clean.std(ddof=1))
    if std == 0:
        stat = (
            0.0
            if float(clean.mean()) == 0
            else math.copysign(float("inf"), float(clean.mean()))
        )
    else:
        stat = float(clean.mean()) / (std / math.sqrt(len(clean)))
    return stat, _normal_two_sided_p(stat)


def _normal_two_sided_p(statistic: float | None) -> float | None:
    if statistic is None:
        return None
    if math.isinf(statistic):
        return 0.0
    return max(0.0, min(1.0, math.erfc(abs(statistic) / math.sqrt(2.0))))


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _sign_series(series: pd.Series) -> pd.Series:
    return series.map(_sign)


def _binomial_direction_stat(hit: pd.Series) -> tuple[float | None, float | None]:
    clean = hit.dropna()
    if clean.empty:
        return None, None
    p0 = 0.5
    stat = (float(clean.mean()) - p0) / math.sqrt(p0 * (1.0 - p0) / len(clean))
    return stat, _normal_two_sided_p(stat)


def _residual_test_statistic(
    test_name: str, residuals: pd.Series, lag: int
) -> tuple[float | None, float | None]:
    if residuals.empty:
        return None, None
    values = residuals.astype(float).dropna()
    if len(values) < 3:
        return None, None
    if test_name == "durbin_watson":
        from statsmodels.stats.stattools import durbin_watson

        return float(durbin_watson(values)), None
    if test_name == "jarque_bera_normality":
        from scipy import stats as scipy_stats

        jb, p_value = scipy_stats.jarque_bera(values)
        return float(jb), float(p_value)
    max_lag = min(lag, len(values) - 1)
    if max_lag < 1:
        return None, None
    if test_name == "ljung_box_q":
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(values, lags=[max_lag], return_df=True)
        return float(result["lb_stat"].iloc[0]), float(result["lb_pvalue"].iloc[0])
    if test_name == "breusch_godfrey_serial_correlation":
        try:
            from statsmodels.stats.diagnostic import acorr_breusch_godfrey
            from statsmodels.regression.linear_model import OLS

            x = np.column_stack([np.ones(len(values)), np.arange(len(values))])
            model = OLS(values.values, x).fit()
            stat, p_value, _, _ = acorr_breusch_godfrey(model, nlags=max_lag)
            return float(stat), float(p_value)
        except Exception:
            return None, None
    if test_name == "arch_lm":
        from statsmodels.stats.diagnostic import het_arch

        try:
            stat, p_value, _, _ = het_arch(values, nlags=max_lag)
            return float(stat), float(p_value)
        except Exception:
            return None, None
    return None, None


def _diebold_mariano_test(
    diff: pd.Series,
    *,
    horizon: int,
    hln: bool = True,
    kernel: str = "newey_west",
) -> tuple[float | None, float | None]:
    """Diebold-Mariano test with HAC standard error.

    Issue #259 -- ``kernel`` selects newey_west / andrews / parzen
    (default: newey_west). When ``hln=True`` (default) applies the
    Harvey-Leybourne-Newbold (1997) small-sample correction.
    """

    clean = diff.dropna()
    n = len(clean)
    if n < 3:
        return None, None
    mean = float(clean.mean())
    nw_lag = max(0, int(horizon) - 1)
    variance = _long_run_variance(clean.to_numpy() - mean, kernel=kernel, lag=nw_lag)
    if variance <= 0:
        return None, None
    statistic = mean / math.sqrt(variance / n)
    if hln:
        adjustment = math.sqrt(
            (n + 1 - 2 * (nw_lag + 1) + (nw_lag + 1) * (nw_lag) / n) / n
        )
        statistic *= adjustment if adjustment > 0 else 1.0
    return float(statistic), _normal_two_sided_p(statistic)


def _newey_west_variance(values: np.ndarray, *, lag: int) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    gamma_0 = float(np.dot(values, values) / n)
    variance = gamma_0
    for k in range(1, max(0, lag) + 1):
        weight = 1.0 - k / (lag + 1)
        gamma_k = float(np.dot(values[:-k], values[k:]) / n)
        variance += 2.0 * weight * gamma_k
    return variance


def _long_run_variance(
    values: np.ndarray, *, kernel: str = "newey_west", lag: int | None = None
) -> float:
    """Issue #259 -- HAC long-run variance with three published kernels.

    * ``newey_west`` (Newey-West 1987): Bartlett triangular kernel,
      ``w_k = 1 - k / (L + 1)``.
    * ``andrews``: Andrews (1991) data-driven Bartlett bandwidth from an
      AR(1) pre-whitening of the residuals; ``L = 1.1447 * (alpha * n)^(1/3)``.
    * ``parzen``: Parzen (1957) quartic-spectral kernel,
      ``w_k = 1 - 6(k/L)^2 + 6(k/L)^3`` for ``k <= L/2``, then
      ``2(1 - k/L)^3``.
    """

    n = len(values)
    if n == 0:
        return 0.0
    centered = values - values.mean() if abs(values.mean()) > 1e-12 else values
    gamma_0 = float(np.dot(centered, centered) / n)
    if kernel == "andrews":
        # AR(1) pre-whiten -> alpha = autocorrelation -> bandwidth.
        if n > 2:
            num = float(np.sum(centered[:-1] * centered[1:]))
            den = float(np.sum(centered[:-1] ** 2))
            alpha1 = num / den if den > 0 else 0.0
            alpha = (4 * alpha1**2) / (max(1 - alpha1**2, 1e-12) ** 2)
            L = max(1, int(np.floor(1.1447 * (alpha * n) ** (1 / 3))))
        else:
            L = 1
        kernel = (
            "newey_west"  # Andrews uses the Bartlett kernel with the data-driven L.
        )
        lag = L
    if lag is None:
        lag = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    L = max(0, int(lag))
    variance = gamma_0
    if kernel == "newey_west":
        for k in range(1, L + 1):
            weight = 1.0 - k / (L + 1)
            if n > k:
                gamma_k = float(np.dot(centered[:-k], centered[k:]) / n)
                variance += 2.0 * weight * gamma_k
        return float(variance)
    if kernel == "parzen":
        for k in range(1, L + 1):
            x = k / L
            if x <= 0.5:
                weight = 1 - 6 * x**2 + 6 * x**3
            else:
                weight = 2 * (1 - x) ** 3
            if n > k:
                gamma_k = float(np.dot(centered[:-k], centered[k:]) / n)
                variance += 2.0 * weight * gamma_k
        return float(variance)
    raise ValueError(f"unknown HAC kernel {kernel!r}")


def _autocorr(values: pd.Series, lag: int) -> float:
    if lag < 1 or len(values) <= lag:
        return 0.0
    left = values.iloc[lag:].astype(float).to_numpy()
    right = values.iloc[:-lag].astype(float).to_numpy()
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float((left**2).sum()) * float((right**2).sum()))
    if denom == 0:
        return 0.0
    return float((left * right).sum() / denom)


def materialize_l8_runtime(
    recipe_root: dict[str, Any],
    upstream_artifacts: dict[str, Any],
    *,
    runtime_durations: dict[str, float] | None = None,
    cell_resolved_axes: dict[str, dict[str, Any]] | None = None,
) -> tuple[L8ArtifactsArtifact, dict[str, Any]]:
    raw = recipe_root.get("8_output", {}) or {}
    context = l8_layer._recipe_context(recipe_root)
    report = l8_layer.validate_layer(raw, context=context)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    resolved = l8_layer.resolve_axes_from_raw(raw, context=context)
    axes = _plain_axes(resolved)
    output_directory = Path(axes["leaf_config"]["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary").mkdir(exist_ok=True)
    exported_files = _l8_export_artifacts(
        output_directory, axes, upstream_artifacts, recipe_root
    )
    git_sha, git_branch = _capture_git_state()
    package_version = _capture_package_version()
    runtime_env = _capture_full_runtime_environment()
    lockfile_content = _capture_dependency_lockfile_content()
    data_revision = _capture_data_revision_tag(recipe_root)
    # Cycle 14 K-3 fix: auto-capture FRED data_through and resolved sample dates
    _l1_art = upstream_artifacts.get("l1_data_definition_v1") if upstream_artifacts else None
    _fred_data_revision = ""
    if _l1_art is not None:
        _raw_meta = (_l1_art.raw_panel.metadata.values or {}) if hasattr(_l1_art, "raw_panel") and _l1_art.raw_panel else {}
        _dt = _raw_meta.get("data_through")
        if _dt:
            _dataset = getattr(_l1_art, "dataset", "") or ""
            if "fred_qd" in _dataset:
                _fred_data_revision = f"fred-qd@{_dt}"
            elif "fred_md" in _dataset or "fred_sd" in _dataset or _dataset.startswith("fred"):
                _fred_data_revision = f"fred-md@{_dt}"
            else:
                _fred_data_revision = f"current@{_dt}"
    if _fred_data_revision and not data_revision:
        data_revision = _fred_data_revision
    # Cycle 15 M-1 fix: post-L2 sample window
    # Prefer l2_clean_panel_v1.panel.data (post-window) over raw_panel
    _sample_start_resolved = None
    _sample_end_resolved = None
    import pandas as _pd_k3
    _l2_art = upstream_artifacts.get("l2_clean_panel_v1") if upstream_artifacts else None
    _post_window_idx = None
    if _l2_art is not None and hasattr(_l2_art, "panel") and _l2_art.panel is not None:
        _l2_panel_data = getattr(_l2_art.panel, "data", None)
        if _l2_panel_data is not None and hasattr(_l2_panel_data, "index") and len(_l2_panel_data.index):
            _post_window_idx = _l2_panel_data.index
    if _post_window_idx is None and _l1_art is not None and hasattr(_l1_art, "raw_panel") and _l1_art.raw_panel is not None:
        # fallback to raw_panel if no L2 artifact
        _idx_data = getattr(_l1_art.raw_panel, "data", None)
        if _idx_data is not None and hasattr(_idx_data, "index") and len(_idx_data.index):
            _post_window_idx = _idx_data.index
    if _post_window_idx is not None:
        _valid_idx = _post_window_idx[_post_window_idx.notna()]
        if len(_valid_idx):
            _sample_start_resolved = str(_valid_idx[0])
            _sample_end_resolved = str(_valid_idx[-1])
    seed_used = _capture_random_seed_used(recipe_root)
    # ``runtime_duration_per_layer`` and ``cells_summary[*].exported_files``
    # are non-deterministic across runs (wall-clock + tmp paths). Keep them
    # out of the hashed L8Manifest dataclass so bit-exact replicate still
    # passes; they are written to the on-disk JSON payload below.
    # F-P1-12 fix: stable cross-process hash (BREAKING -- prior manifest values differ)
    _canonical_json = json.dumps(
        _jsonable(recipe_root), sort_keys=True, default=str, separators=(",", ":")
    )
    _recipe_hash_hex = hashlib.sha256(_canonical_json.encode()).hexdigest()[:16]
    manifest = L8Manifest(
        recipe_hash=_recipe_hash_hex,
        package_version=package_version,
        python_version=platform.python_version(),
        r_version=_command_version_safe(("R", "--version")),
        julia_version=_command_version_safe(("julia", "--version")),
        dependency_lockfile_paths=_dependency_lockfile_paths(),
        git_commit_sha=git_sha,
        git_branch_name=git_branch,
        data_revision_tag=data_revision,
        random_seed_used=seed_used,
        runtime_environment=runtime_env,
        cells_summary=[
            {
                "cell_id": "cell_001",
                "status": "completed",
                "n_exported_files": len(exported_files),
            }
        ],
    )
    manifest_path = output_directory / (
        "manifest.jsonl"
        if axes.get("manifest_format") == "json_lines"
        else "manifest.json"
    )
    manifest_payload = {
        "manifest": _jsonable(manifest),
        "provenance_fields": axes.get("provenance_fields", []),
        "saved_objects": axes.get("saved_objects", []),
        "upstream_sinks": sorted(upstream_artifacts),
        "recipe_yaml_full": _stringify_recipe_root(recipe_root),
        "cell_resolved_axes": _jsonable(cell_resolved_axes or {}),
        "dependency_lockfile_content": lockfile_content,
        "runtime_duration_per_layer": dict(runtime_durations or {}),
        "exported_files": [file.path.as_posix() for file in exported_files],
        # F-P1-13 fix: cache_root provenance (additive, not breaking)
        "cache_root": recipe_root.get("1_data", {}).get("leaf_config", {}).get("cache_root"),
        # Cycle 14 K-3 fix: resolved sample dates
        "sample_start_resolved": _sample_start_resolved,
        "sample_end_resolved": _sample_end_resolved,
    }
    if axes.get("manifest_format") == "json_lines":
        manifest_path.write_text(
            json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif axes.get("manifest_format") == "yaml":
        try:
            import yaml as _yaml  # type: ignore
        except ImportError:
            manifest_path.write_text(
                json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        else:
            manifest_path = manifest_path.with_suffix(".yaml")
            manifest_path.write_text(
                _yaml.safe_dump(manifest_payload, sort_keys=True), encoding="utf-8"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    recipe_path = output_directory / "recipe.json"
    recipe_path.write_text(
        json.dumps(_jsonable(recipe_root), indent=2, sort_keys=True), encoding="utf-8"
    )
    exported_files.extend(
        [
            ExportedFile(
                path=manifest_path,
                artifact_type="manifest",
                source_sink="l8_artifacts_v1",
            ),
            ExportedFile(
                path=recipe_path, artifact_type="recipe", source_sink="recipe"
            ),
        ]
    )
    return (
        L8ArtifactsArtifact(
            output_directory=output_directory,
            manifest=manifest,
            exported_files=exported_files,
            artifact_count=len(exported_files),
            upstream_hashes={
                name: "runtime_unhashed" for name in sorted(upstream_artifacts)
            },
        ),
        axes,
    )


def _derive_saved_objects(
    recipe_root: dict[str, Any], upstream_artifacts: dict[str, Any]
) -> set[str]:
    """Issue #261 -- design rule: each active layer auto-adds its canonical
    saved-objects entries unless the user opts out via
    ``saved_objects_mode = explicit_only``."""

    derived: set[str] = set()
    if "1_data" in recipe_root:
        derived.update({"forecasts", "raw_panel"})
    l1 = recipe_root.get("1_data", {}) or {}
    fixed = l1.get("fixed_axes", {}) or {}
    if fixed.get("regime_definition", "none") != "none":
        derived.update({"regime_metrics", "regime_metadata"})
    if "2_preprocessing" in recipe_root:
        derived.update({"clean_panel", "cleaning_log"})
    if "3_feature_engineering" in recipe_root:
        derived.add("feature_metadata")
    if "4_forecasting_model" in recipe_root:
        derived.update({"model_artifacts", "training_metadata"})
    if "5_evaluation" in recipe_root:
        derived.update({"metrics", "ranking"})
    if "6_statistical_tests" in recipe_root:
        derived.add("tests")
    if "7_interpretation" in recipe_root:
        derived.update({"importance", "figures"})
    for diag_layer, suffix in (
        ("1_5_data_summary", "diagnostics_l1_5"),
        ("2_5_pre_post_preprocessing", "diagnostics_l2_5"),
        ("3_5_feature_diagnostics", "diagnostics_l3_5"),
        ("4_5_generator_diagnostics", "diagnostics_l4_5"),
    ):
        if diag_layer in recipe_root:
            derived.add(suffix)
    return derived


def _l8_export_artifacts(
    output_directory: Path,
    axes: dict[str, Any],
    upstream_artifacts: dict[str, Any],
    recipe_root: dict[str, Any],
) -> list[ExportedFile]:
    saved = set(axes.get("saved_objects", []))
    # Issue #261 -- derive default saved_objects from active layers when
    # the recipe omits them. ``saved_objects_mode = explicit_only`` opts
    # out of the auto-derivation.
    leaf = axes.get("leaf_config", {}) or {}
    if leaf.get("saved_objects_mode", "auto") != "explicit_only":
        saved = saved | _derive_saved_objects(recipe_root, upstream_artifacts)
    export_format = axes.get("export_format", "json_csv")
    formats = _l8_resolve_formats(export_format)
    granularity = axes.get("artifact_granularity", "per_cell")
    exported: list[ExportedFile] = []
    summary_dir = output_directory / "summary"
    summary_dir.mkdir(exist_ok=True)
    if granularity == "flat":
        cell_dir = output_directory
    else:
        cell_dir = output_directory / "cell_001"
    cell_dir.mkdir(exist_ok=True, parents=True)
    figures_dir = cell_dir / "figures"

    def add_dataframe(path: Path, frame: pd.DataFrame, source: str) -> None:
        if "csv" in formats:
            frame.to_csv(path.with_suffix(".csv"), index=True)
            exported.append(
                ExportedFile(
                    path=path.with_suffix(".csv"),
                    artifact_type="csv",
                    source_sink=source,
                )
            )
        if "parquet" in formats:
            try:
                frame.reset_index().to_parquet(path.with_suffix(".parquet"))
                exported.append(
                    ExportedFile(
                        path=path.with_suffix(".parquet"),
                        artifact_type="parquet",
                        source_sink=source,
                    )
                )
            except Exception:
                pass
        if "latex" in formats:
            latex = frame.to_latex(index=True)
            path.with_suffix(".tex").write_text(latex, encoding="utf-8")
            exported.append(
                ExportedFile(
                    path=path.with_suffix(".tex"),
                    artifact_type="latex",
                    source_sink=source,
                )
            )
        if "markdown" in formats:
            md = frame.to_markdown(index=True)
            path.with_suffix(".md").write_text(md, encoding="utf-8")
            exported.append(
                ExportedFile(
                    path=path.with_suffix(".md"),
                    artifact_type="markdown",
                    source_sink=source,
                )
            )

    def add_json(path: Path, payload: Any, source: str) -> None:
        if "json" not in formats:
            return
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
        )
        exported.append(
            ExportedFile(path=path, artifact_type="json", source_sink=source)
        )

    if "forecasts" in saved and "l4_forecasts_v1" in upstream_artifacts:
        # F-P1-10 fix: compute forecast_date (origin + horizon offset) and actual
        # Column order: model_id, target, horizon, origin, forecast, forecast_date, actual
        _l1_art = upstream_artifacts.get("l1_data_definition_v1")
        _l1_freq = getattr(_l1_art, "frequency", "monthly") if _l1_art is not None else "monthly"
        _raw_panel_data = (
            _l1_art.raw_panel.data
            if _l1_art is not None and hasattr(_l1_art, "raw_panel")
            else None
        )

        def _compute_forecast_date(origin, horizon: int, frequency: str):
            """Return the date when horizon-step-ahead forecast materializes."""
            try:
                months = horizon * (3 if frequency == "quarterly" else 1)
                return origin + pd.DateOffset(months=months)
            except Exception:
                return None

        def _lookup_actual(forecast_date, target: str, panel_data):
            """Return the realized value at forecast_date for target; NaN if unavailable."""
            if panel_data is None or forecast_date is None:
                return float("nan")
            if target not in panel_data.columns:
                return float("nan")
            try:
                # Try exact match first, then nearest
                col = panel_data[target]
                fd_ts = pd.Timestamp(forecast_date)
                if fd_ts in col.index:
                    val = col.loc[fd_ts]
                    return float(val) if pd.notna(val) else float("nan")
                # Try normalizing to period-end convention
                idx_ts = col.index.asof(fd_ts)
                if pd.isna(idx_ts):
                    return float("nan")
                val = col.loc[idx_ts]
                return float(val) if pd.notna(val) else float("nan")
            except Exception:
                return float("nan")

        rows = []
        for (model_id, target, horizon, origin), forecast in upstream_artifacts[
            "l4_forecasts_v1"
        ].forecasts.items():
            forecast_date = _compute_forecast_date(origin, int(horizon), _l1_freq)
            actual_val = _lookup_actual(forecast_date, target, _raw_panel_data)
            rows.append({
                "model_id": model_id,
                "target": target,
                "horizon": horizon,
                "origin": origin,
                "forecast": forecast,
                "forecast_date": forecast_date,
                "actual": actual_val,
            })
        forecasts_frame = pd.DataFrame(rows)
        if (
            granularity in {"per_target", "per_horizon", "per_target_horizon"}
            and not forecasts_frame.empty
        ):
            for sub_dir, sub_frame in _l8_split_by_granularity(
                forecasts_frame, granularity, cell_dir
            ):
                sub_dir.mkdir(parents=True, exist_ok=True)
                add_dataframe(
                    sub_dir / "forecasts",
                    sub_frame.reset_index(drop=True),
                    "l4_forecasts_v1",
                )
        else:
            add_dataframe(cell_dir / "forecasts", forecasts_frame, "l4_forecasts_v1")
    if "metrics" in saved and "l5_evaluation_v1" in upstream_artifacts:
        add_dataframe(
            summary_dir / "metrics_all_cells",
            upstream_artifacts["l5_evaluation_v1"].metrics_table,
            "l5_evaluation_v1",
        )
    if "ranking" in saved and "l5_evaluation_v1" in upstream_artifacts:
        add_dataframe(
            summary_dir / "ranking",
            upstream_artifacts["l5_evaluation_v1"].ranking_table,
            "l5_evaluation_v1",
        )
    if "tests" in saved and "l6_tests_v1" in upstream_artifacts:
        add_json(
            output_directory / "tests_summary.json",
            upstream_artifacts["l6_tests_v1"],
            "l6_tests_v1",
        )
    if "importance" in saved and "l7_importance_v1" in upstream_artifacts:
        importance_artifact = upstream_artifacts["l7_importance_v1"]
        add_json(
            output_directory / "importance_summary.json",
            importance_artifact,
            "l7_importance_v1",
        )
        # F-P1-11 fix: thread L7 figure axes (dpi, format, top_k, precision_digits) into render
        _l7_axis_resolved = (
            getattr(importance_artifact, "computation_metadata", {}) or {}
        ).get("axis_resolved", {})
        _fig_dpi = int(_l7_axis_resolved.get("figure_dpi", 300))
        _fig_fmt = str(_l7_axis_resolved.get("figure_format", "pdf")).lower()
        _top_k = int(_l7_axis_resolved.get("top_k_features_to_show", 20))
        _precision = int(_l7_axis_resolved.get("precision_digits", 4))
        # Map figure_format axis value to matplotlib format + file extension
        _fmt_map = {"pdf": "pdf", "png": "png", "svg": "svg", "eps": "eps"}
        _mpl_fmt = _fmt_map.get(_fig_fmt, "pdf")
        try:
            from .figures import render_default_for_op, render_us_state_choropleth

            figures_dir.mkdir(parents=True, exist_ok=True)
            sink_payloads = getattr(importance_artifact, "global_importance", {}) or {}
            for op_name, payload in sink_payloads.items():
                figure_path = figures_dir / f"{op_name}.{_mpl_fmt}"
                # Apply precision_digits formatting to importance column when available
                if isinstance(payload, __import__("pandas").DataFrame) and "importance" in payload.columns:
                    payload = payload.copy()
                    payload["importance"] = payload["importance"].map(
                        lambda v: round(float(v), _precision)
                    )
                rendered = render_default_for_op(
                    op_name, payload, output_path=figure_path, title=f"L7 {op_name}",
                    dpi=_fig_dpi, top_k=_top_k,
                )
                if rendered is not None:
                    exported.append(
                        ExportedFile(
                            path=rendered,
                            artifact_type=f"figure_{_mpl_fmt}",
                            source_sink="l7_importance_v1",
                        )
                    )
            # FRED-SD geographic visualization: when group_aggregate produced
            # per-state importance, render a US choropleth.
            group_payloads = getattr(importance_artifact, "group_importance", {}) or {}
            for op_name, payload in group_payloads.items():
                if isinstance(payload, pd.DataFrame) and "group" in payload.columns:
                    state_scores = {
                        row["group"]: float(row["importance"])
                        for _, row in payload.iterrows()
                        if isinstance(row.get("group"), str)
                        and len(str(row["group"])) == 2
                    }
                    if state_scores:
                        choropleth = figures_dir / f"{op_name}_state_choropleth.pdf"
                        render_us_state_choropleth(
                            state_scores,
                            output_path=choropleth,
                            title=f"L7 {op_name} (state)",
                        )
                        exported.append(
                            ExportedFile(
                                path=choropleth,
                                artifact_type="figure_pdf",
                                source_sink="l7_importance_v1",
                            )
                        )
        except Exception:
            # Figure rendering is best-effort; leave a json export and continue.
            pass
    if "feature_metadata" in saved and "l3_metadata_v1" in upstream_artifacts:
        add_json(
            cell_dir / "feature_metadata.json",
            upstream_artifacts["l3_metadata_v1"],
            "l3_metadata_v1",
        )
    if "clean_panel" in saved and "l2_clean_panel_v1" in upstream_artifacts:
        add_dataframe(
            cell_dir / "clean_panel",
            upstream_artifacts["l2_clean_panel_v1"].panel.data,
            "l2_clean_panel_v1",
        )
    if "raw_panel" in saved and "l1_data_definition_v1" in upstream_artifacts:
        add_dataframe(
            cell_dir / "raw_panel",
            upstream_artifacts["l1_data_definition_v1"].raw_panel.data,
            "l1_data_definition_v1",
        )
    for sink_name, artifact in upstream_artifacts.items():
        if sink_name.endswith("_diagnostic_v1"):
            object_name = f"diagnostics_{sink_name.split('_diagnostic_v1')[0]}"
            if object_name in saved or "diagnostics_all" in saved:
                diag_dir = output_directory / "diagnostics"
                diag_dir.mkdir(exist_ok=True)
                add_json(diag_dir / f"{sink_name}.json", artifact, sink_name)

    if export_format == "html_report":
        html_path = _l8_render_html_report(
            output_directory, axes, upstream_artifacts, recipe_root
        )
        if html_path is not None:
            exported.append(
                ExportedFile(
                    path=html_path,
                    artifact_type="html_report",
                    source_sink="l8_artifacts_v1",
                )
            )

    compression = axes.get("compression", "none")
    if compression in {"gzip", "zip"}:
        leaf = axes.get("leaf_config", {}) or {}
        try:
            level = int(leaf.get("compression_level", 6))
        except (TypeError, ValueError):
            level = 6
        if compression == "gzip":
            exported = _l8_apply_gzip(exported, level)
        else:  # zip
            exported = _l8_apply_zip(exported, output_directory, level)
    return exported


def _l8_render_html_report(
    output_directory: Path,
    axes: dict[str, Any],
    upstream_artifacts: dict[str, Any],
    recipe_root: dict[str, Any],
) -> Path | None:
    """Render a minimal self-contained HTML report summarising the run.

    Layout: study title, recipe digest, per-cell metrics table, embedded
    figure list. Uses pandas ``to_html`` for tables; no jinja dependency
    so we don't drag in an extra package for a small renderer.
    """

    from html import escape as _esc

    lines: list[str] = []
    lines.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    lines.append("<title>macroforecast study report</title>")
    lines.append(
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:90rem}"
        "h1,h2{border-bottom:1px solid #ccc;padding-bottom:.3rem}"
        "table{border-collapse:collapse;margin:1rem 0}"
        "td,th{border:1px solid #ddd;padding:.4rem .8rem}"
        "tr:nth-child(even){background:#f7f7f7}"
        "code{background:#f0f0f0;padding:.1rem .3rem;border-radius:.2rem}"
        "</style></head><body>"
    )
    lines.append("<h1>macroforecast study report</h1>")

    # Recipe digest -----------------------------------------------------
    target = ((recipe_root.get("1_data", {}) or {}).get("leaf_config", {}) or {}).get(
        "target"
    )
    if target:
        lines.append(f"<p><b>Target</b>: <code>{_esc(str(target))}</code></p>")
    family = None
    for node in (recipe_root.get("4_forecasting_model", {}) or {}).get(
        "nodes", []
    ) or []:
        if isinstance(node, dict) and node.get("op") == "fit_model":
            family = (node.get("params") or {}).get("family")
            break
    if family:
        lines.append(f"<p><b>Model family</b>: <code>{_esc(str(family))}</code></p>")

    # Metrics -----------------------------------------------------------
    if "l5_evaluation_v1" in upstream_artifacts:
        eval_artifact = upstream_artifacts["l5_evaluation_v1"]
        metrics = getattr(eval_artifact, "metrics_table", None)
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            lines.append("<h2>Metrics</h2>")
            try:
                lines.append(metrics.to_html(index=True, border=0))
            except Exception:
                lines.append("<p><i>(metrics table unavailable)</i></p>")

    # Figures (link to PDFs / PNGs in cell_001/figures) -----------------
    figures_dir = output_directory / "cell_001" / "figures"
    if figures_dir.exists():
        figures = sorted(p for p in figures_dir.iterdir() if p.is_file())
        if figures:
            lines.append("<h2>Figures</h2><ul>")
            for fig in figures:
                rel = fig.relative_to(output_directory).as_posix()
                lines.append(f"<li><a href='{_esc(rel)}'>{_esc(fig.name)}</a></li>")
            lines.append("</ul>")

    lines.append("</body></html>")
    html_path = output_directory / "report.html"
    html_path.write_text("\n".join(lines), encoding="utf-8")
    return html_path


def _l8_split_by_granularity(
    frame: pd.DataFrame, granularity: str, cell_dir: Path
) -> list[tuple[Path, pd.DataFrame]]:
    """Yield ``(sub_directory, group_frame)`` pairs for an artifact split by
    L8.D ``artifact_granularity`` (``per_target`` / ``per_horizon`` /
    ``per_target_horizon``). Frame must carry ``target`` and/or ``horizon``
    columns; missing values are coerced to a ``"_missing_"`` bucket."""

    pairs: list[tuple[Path, pd.DataFrame]] = []
    if granularity == "per_target":
        keys = ["target"]
    elif granularity == "per_horizon":
        keys = ["horizon"]
    else:
        keys = ["target", "horizon"]
    if not all(k in frame.columns for k in keys):
        return [(cell_dir, frame)]
    for keyvals, group in frame.groupby(keys, dropna=False):
        if not isinstance(keyvals, tuple):
            keyvals = (keyvals,)
        parts: list[str] = []
        for key, val in zip(keys, keyvals):
            safe = (
                "_missing_"
                if pd.isna(val)
                else str(val).replace("/", "_").replace(" ", "_")
            )
            parts.append(f"{key}={safe}")
        sub_dir = cell_dir
        for part in parts:
            sub_dir = sub_dir / part
        pairs.append((sub_dir, group))
    return pairs


def _l8_apply_gzip(exported: list[ExportedFile], level: int) -> list[ExportedFile]:
    """Replace each non-manifest export with its ``.gz`` form. Manifest /
    recipe files are left uncompressed (they are written *after*
    ``_l8_export_artifacts`` returns) -- gzipping every artifact in place
    keeps the per-file structure intact while shrinking the on-disk
    footprint."""

    import gzip
    import shutil

    rewritten: list[ExportedFile] = []
    for entry in exported:
        path = entry.path
        if not path.exists() or path.suffix == ".gz":
            rewritten.append(entry)
            continue
        gz_path = path.with_suffix(path.suffix + ".gz")
        with (
            path.open("rb") as src,
            gzip.open(gz_path, "wb", compresslevel=level) as dst,
        ):
            shutil.copyfileobj(src, dst)
        path.unlink()
        rewritten.append(
            ExportedFile(
                path=gz_path,
                artifact_type=f"{entry.artifact_type}_gz",
                source_sink=entry.source_sink,
            )
        )
    return rewritten


def _l8_apply_zip(
    exported: list[ExportedFile], output_directory: Path, level: int
) -> list[ExportedFile]:
    """Bundle every exported file into ``<output_dir>.zip`` and remove the
    originals; return a single ``ExportedFile`` describing the bundle."""

    import zipfile

    zip_path = output_directory / f"{output_directory.name}.zip"
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level
    ) as zf:
        for entry in exported:
            if not entry.path.exists():
                continue
            arcname = entry.path.relative_to(output_directory).as_posix()
            zf.write(entry.path, arcname=arcname)
    for entry in exported:
        try:
            if entry.path.exists() and entry.path != zip_path:
                entry.path.unlink()
        except OSError:
            pass
    return [
        ExportedFile(
            path=zip_path,
            artifact_type="zip_bundle",
            source_sink="l8_artifacts_v1",
        )
    ]


def _l8_resolve_formats(export_format: str) -> set[str]:
    if export_format == "all":
        return {"json", "csv", "parquet", "latex", "markdown"}
    if export_format == "json_csv":
        return {"json", "csv"}
    if export_format == "json_parquet":
        return {"json", "parquet"}
    if export_format == "latex_tables":
        return {"latex"}
    if export_format == "markdown_report":
        return {"markdown"}
    if export_format == "html_report":
        return {"json"}
    if export_format in {"json", "csv", "parquet"}:
        return {export_format}
    return {"json", "csv"}


def _dependency_lockfile_paths() -> dict[str, str]:
    paths: dict[str, str] = {}
    for candidate in ("uv.lock", "requirements.txt", "pyproject.toml"):
        path = Path(candidate)
        if path.exists():
            paths["python"] = path.as_posix()
            break
    return paths


def _capture_dependency_lockfile_content() -> dict[str, str]:
    """Read the first available python lockfile content (truncated). Mirrors
    ``_dependency_lockfile_paths`` but returns the file body so the manifest
    is self-contained for replication."""

    for candidate in ("uv.lock", "requirements.txt", "pyproject.toml"):
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if len(text) > 64_000:
            text = text[:64_000] + "\n# (truncated)\n"
        return {"python": text, "python_path": path.as_posix()}
    return {}


def _capture_git_state(start: Path | None = None) -> tuple[str | None, str | None]:
    """Best-effort ``(commit_sha, branch_name)``; both ``None`` outside a
    git checkout or when git is missing. Tolerant of detached HEAD."""

    import subprocess

    cwd = start or Path.cwd()
    try:
        sha = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None, None
    try:
        branch = subprocess.run(
            ("git", "rev-parse", "--abbrev-ref", "HEAD"),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        branch = None
    if branch == "HEAD":
        branch = None
    return sha or None, branch


def _command_version_safe(cmd: tuple[str, ...]) -> str | None:
    """Shell out to ``cmd`` and return the first stdout/stderr line, or
    ``None`` if the binary isn't present."""

    import subprocess

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    output = (proc.stdout or proc.stderr or "").strip().splitlines()
    return output[0] if output else None


def _capture_package_version() -> str:
    try:
        from .. import __version__
    except ImportError:  # pragma: no cover - defensive
        return "runtime-local"
    return str(__version__) if __version__ else "runtime-local"


def _capture_full_runtime_environment() -> RuntimeEnvironment:
    """Populate every available field on ``RuntimeEnvironment``. The
    ``manifest.capture_runtime_environment`` helper produces a richer record
    type that we can't reuse here without an import cycle, but we mirror the
    intent."""

    return RuntimeEnvironment(
        os_name=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu_info=f"{platform.machine()} ({platform.processor() or 'unknown'})",
        gpu_info=_detect_gpu(),
    )


def _detect_gpu() -> str | None:
    """Best-effort GPU descriptor: torch.cuda first, then nvidia-smi."""

    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore
    if torch is not None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                count = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(count)]
                return f"cuda x{count}: {', '.join(names)}"
        except Exception:  # pragma: no cover - defensive
            pass
    descriptor = _command_version_safe(
        ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    )
    return descriptor


def _capture_data_revision_tag(recipe_root: dict[str, Any]) -> str:
    """Mirror the FRED vintage tag (or generic data revision marker) for the
    L1 sub-graph so a replicate is locked to the source revision."""

    l1 = recipe_root.get("1_data", {}) or {}
    leaf = l1.get("leaf_config", {}) or {}
    tag = leaf.get("vintage_date_or_tag") or leaf.get("data_revision_tag")
    if tag:
        return str(tag)
    fixed = l1.get("fixed_axes", {}) or {}
    if fixed.get("vintage_policy"):
        return str(fixed["vintage_policy"])
    return ""


def _capture_random_seed_used(recipe_root: dict[str, Any]) -> int | None:
    l0 = recipe_root.get("0_meta", {}) or {}
    leaf = l0.get("leaf_config", {}) or {}
    fixed = l0.get("fixed_axes", {}) or {}
    if "random_seed" in leaf:
        try:
            return int(leaf["random_seed"])
        except (TypeError, ValueError):
            return None
    if "random_seed" in fixed:
        try:
            return int(fixed["random_seed"])
        except (TypeError, ValueError):
            return None
    repro = fixed.get("reproducibility_mode", "seeded_reproducible")
    return 0 if repro == "seeded_reproducible" else None


def _stringify_recipe_root(recipe_root: dict[str, Any]) -> str:
    """Canonical JSON form of the recipe root (full text) so the manifest
    carries the spec needed for replicate, even when the user passed a dict."""

    try:
        import yaml as _yaml  # type: ignore

        return _yaml.safe_dump(_jsonable(recipe_root), sort_keys=True)
    except ImportError:
        return json.dumps(_jsonable(recipe_root), indent=2, sort_keys=True)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    # Cycle 14 L3 fix: handle numpy scalar and array types
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _plain_axes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _plain_axes(item) for key, item in value.items() if key != "_active"
        }
    if isinstance(value, list):
        return [_plain_axes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_axes(item) for item in value)
    return value


def _iso_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _longest_missing_gap(series: pd.Series) -> int:
    longest = 0
    current = 0
    for is_missing in series.isna():
        if is_missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _ks_statistic(left: pd.Series, right: pd.Series) -> float | None:
    if left.empty or right.empty:
        return None
    values = sorted(set(left.tolist()) | set(right.tolist()))
    max_distance = 0.0
    for value in values:
        left_cdf = float((left <= value).mean())
        right_cdf = float((right <= value).mean())
        max_distance = max(max_distance, abs(left_cdf - right_cdf))
    return max_distance


def _load_raw_panel(resolved: dict[str, Any], leaf_config: dict[str, Any]) -> Panel:
    policy = resolved["custom_source_policy"]
    if policy == "official_only":
        raw_result = _load_official_raw_result(resolved, leaf_config)
        frame = raw_result.data.copy()
        metadata = {
            "stage": "l1_raw",
            "source": "official",
            "dataset": raw_result.dataset_metadata.dataset,
            "frequency": raw_result.dataset_metadata.frequency,
            "vintage": raw_result.dataset_metadata.vintage,
            "local_path": raw_result.artifact.local_path,
            "transform_codes": dict(raw_result.transform_codes),
        }
    elif policy in {"custom_panel_only", "official_plus_custom"}:
        if "custom_panel_inline" in leaf_config:
            frame = pd.DataFrame(leaf_config["custom_panel_inline"])
        elif "custom_panel_records" in leaf_config:
            frame = pd.DataFrame.from_records(leaf_config["custom_panel_records"])
        elif "custom_source_path" in leaf_config:
            frame = _read_custom_panel_path(Path(leaf_config["custom_source_path"]))
        else:
            raise ValueError(
                "custom panel runtime requires custom_panel_inline, custom_panel_records, or custom_source_path"
            )
        metadata = {"stage": "l1_raw", "source": "custom_panel"}
        if policy == "official_plus_custom":
            official = _load_official_raw_result(resolved, leaf_config)
            official_frame = official.data.copy()
            # merge: prefer custom values when both have a column
            frame = frame.join(official_frame, how="outer", rsuffix="__official")
            metadata.update(
                {
                    "source": "official_plus_custom",
                    "official_dataset": official.dataset_metadata.dataset,
                    "official_local_path": official.artifact.local_path,
                    "transform_codes": dict(
                        getattr(official, "transform_codes", {}) or {}
                    ),
                }
            )
    else:
        raise NotImplementedError(
            f"custom_source_policy={policy!r} core runtime loading is deferred"
        )
    frame = _normalize_datetime_index(frame, leaf_config)
    frame = _apply_sample_window(frame, resolved, leaf_config)
    _validate_targets_present(frame, leaf_config, resolved)
    return _panel_from_frame(frame, metadata=metadata)


def _load_official_raw_result(resolved: dict[str, Any], leaf_config: dict[str, Any]):
    dataset = resolved.get("dataset")
    vintage = leaf_config.get("vintage")
    cache_root = leaf_config.get("cache_root")
    local_source = leaf_config.get("local_raw_source") or leaf_config.get(
        "official_source_path"
    )
    if dataset == "fred_md":
        return load_fred_md(
            vintage=vintage, cache_root=cache_root, local_source=local_source
        )
    if dataset == "fred_qd":
        return load_fred_qd(
            vintage=vintage, cache_root=cache_root, local_source=local_source
        )
    if dataset == "fred_sd":
        states = _resolve_fred_sd_states(resolved, leaf_config)
        # F-P1-7 fix: resolve variable group to actual variable list before
        # passing to loader. Groups (e.g. "labor_market") are stored in
        # raw/fred_sd_groups.py; resolve_fred_sd_variable_group returns the
        # list (or None for "all_sd_variables").
        _sd_var_group = resolved.get("fred_sd_variable_group") or leaf_config.get("fred_sd_variable_group")
        if _sd_var_group and _sd_var_group not in ("all_sd_variables", None):
            _resolved_vars, _ = _resolve_fred_sd_variable_group(_sd_var_group, leaf_config)
            variables = _resolved_vars
        else:
            variables = leaf_config.get("fred_sd_variables") or leaf_config.get(
                "sd_variables"
            )
        return load_fred_sd(
            vintage=vintage,
            cache_root=cache_root,
            local_source=local_source,
            states=list(states) if states else None,
            variables=list(variables) if variables else None,
        )
    if dataset in {"fred_md+fred_sd", "fred_qd+fred_sd"}:
        national_loader = (
            load_fred_md if dataset.startswith("fred_md") else load_fred_qd
        )
        national = national_loader(
            vintage=vintage, cache_root=cache_root, local_source=local_source
        )
        states = _resolve_fred_sd_states(resolved, leaf_config)
        # F-P1-7 fix: same variable group resolution as above
        _sd_var_group = resolved.get("fred_sd_variable_group") or leaf_config.get("fred_sd_variable_group")
        if _sd_var_group and _sd_var_group not in ("all_sd_variables", None):
            _resolved_vars, _ = _resolve_fred_sd_variable_group(_sd_var_group, leaf_config)
            variables = _resolved_vars
        else:
            variables = leaf_config.get("fred_sd_variables") or leaf_config.get(
                "sd_variables"
            )
        regional = load_fred_sd(
            vintage=vintage,
            cache_root=cache_root,
            local_source=leaf_config.get("local_fred_sd_source"),
            states=list(states) if states else None,
            variables=list(variables) if variables else None,
        )
        merged = national.data.join(regional.data, how="outer")
        national.data.attrs.update(national.data.attrs)
        return type(national)(
            data=merged,
            dataset_metadata=national.dataset_metadata,
            artifact=national.artifact,
            transform_codes=national.transform_codes,
        )
    raise NotImplementedError(
        f"official dataset {dataset!r} is not supported by core L1 runtime yet"
    )


def _resolve_fred_sd_states(
    resolved: dict[str, Any], leaf_config: dict[str, Any]
) -> list[str] | None:
    # F-P1-6 fix: also check "sd_states" key (used by the L1 validator at
    # l1.py:716) in addition to "fred_sd_states" and "state_selection".
    explicit = (
        leaf_config.get("sd_states")
        or leaf_config.get("fred_sd_states")
        or leaf_config.get("state_selection")
    )
    if explicit and isinstance(explicit, (list, tuple)):
        return list(explicit)
    if explicit and isinstance(explicit, str) and explicit != "all_states" and explicit != "selected_states":
        return [explicit]
    group_key = resolved.get("fred_sd_state_group") or leaf_config.get(
        "fred_sd_state_group"
    )
    if group_key and group_key in FRED_SD_STATE_GROUPS:
        return list(FRED_SD_STATE_GROUPS[group_key])
    target_scope = resolved.get("target_geography_scope")
    if target_scope == "single_state":
        target = leaf_config.get("target_state")
        return [target] if target else None
    if target_scope == "selected_states":
        return list(leaf_config.get("selected_states", []) or [])
    return None


def _read_custom_panel_path(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        # F-P1-5 fix: detect FRED official CSV format (Transform: row) and
        # raise instead of silently corrupting the panel.
        with open(path, "r", encoding="utf-8", errors="replace") as _f:
            _first_line = _f.readline()
        _cells = _first_line.split(",")
        if _cells and (
            str(_cells[0]).strip().startswith("Transform:")
            or str(_cells[0]).strip().startswith("transform_code:")
        ):
            raise RuntimeError(
                f"Custom CSV at {path!r} appears to be an official FRED-MD/QD CSV "
                f"(first row starts with {_cells[0].strip()!r}). "
                "Use dataset=fred_md or dataset=fred_qd instead of a custom panel path."
            )
        return pd.read_csv(path)
    raise ValueError(
        f"unsupported custom panel format {path.suffix!r}; use CSV or Parquet"
    )


def _normalize_datetime_index(
    frame: pd.DataFrame, leaf_config: dict[str, Any]
) -> pd.DataFrame:
    date_column = leaf_config.get("date_column")
    if date_column is None:
        for candidate in ("date", "DATE", "timestamp", "time", "index"):
            if candidate in frame.columns:
                date_column = candidate
                break
    if date_column is not None:
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.pop(date_column))
    else:
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    # F-P1-4 fix: detect duplicate dates and raise instead of silently coalescing.
    if not frame.index.is_unique:
        dupes = frame.index[frame.index.duplicated()].tolist()
        raise RuntimeError(
            f"duplicate dates detected in custom panel: {dupes[:5]}"
            + (" (and more)" if len(dupes) > 5 else "")
        )
    return frame


def _apply_sample_window(
    frame: pd.DataFrame, resolved: dict[str, Any], leaf_config: dict[str, Any]
) -> pd.DataFrame:
    result = frame
    start_rule = resolved.get("sample_start_rule") or "max_balanced"
    end_rule = resolved.get("sample_end_rule") or "latest_available"
    if start_rule == "max_balanced":
        first_observed = (
            result.dropna(axis=0, how="any").index.min() if not result.empty else None
        )
        if first_observed is not None and pd.notna(first_observed):
            result = result.loc[first_observed:]
    elif start_rule == "fixed_date":
        _start_norm = l1_layer._normalize_iso_partial(leaf_config["sample_start_date"], end_of_period=False) or leaf_config["sample_start_date"]
        result = result.loc[pd.Timestamp(_start_norm) :]
    if end_rule == "latest_available":
        # default: keep as-is
        pass
    elif end_rule == "fixed_date":
        _end_norm = l1_layer._normalize_iso_partial(leaf_config["sample_end_date"], end_of_period=True) or leaf_config["sample_end_date"]
        result = result.loc[: pd.Timestamp(_end_norm)]
    return result


def _validate_targets_present(
    frame: pd.DataFrame, leaf_config: dict[str, Any], resolved: dict[str, Any]
) -> None:
    target = leaf_config.get("target")
    targets = tuple(leaf_config.get("targets", ()) or ((target,) if target else ()))
    missing = [name for name in targets if name not in frame.columns]
    if missing:
        raise ValueError(f"[L1/1_data.leaf_config.target] target columns missing from custom panel: {missing}")  # Cycle 14 L1-1 fix:
    if resolved.get("target_structure") == "single_target" and not target:
        raise ValueError("single_target runtime requires leaf_config.target")


def _default_chow_lin_indicator(
    frame: pd.DataFrame, monthly_cols: list[str]
) -> str | None:
    """Pick the monthly column with the highest absolute correlation to the
    target's quarterly observations -- used when ``chow_lin_indicator`` is
    not supplied."""

    if not monthly_cols:
        return None
    return monthly_cols[0]


def _chow_lin_disaggregate(
    quarterly: pd.Series, indicator_monthly: pd.Series
) -> pd.Series:
    """Issue #255 -- Chow-Lin (1971) regression-based disaggregation.

    Algorithm (constant-only intercept + AR(0) error variant -- the
    common ``chow_lin_litterman`` simplification used in mFilter / R's
    ``tempdisagg``):

    1. Aggregate the indicator to quarterly via mean.
    2. Regress the observed quarterly series on the aggregated indicator
       to estimate ``alpha`` + ``beta``.
    3. Disaggregate to monthly: ``y^M_t = alpha / 3 + beta * X^M_t +
       smoothed_residual_t`` where the smoothed residual distributes the
       quarterly residual evenly across its three months.

    Returns a monthly series aligned with ``indicator_monthly``'s index.
    """

    monthly_index = indicator_monthly.index
    if not isinstance(monthly_index, pd.DatetimeIndex):
        # Best-effort: bfill+ffill if we cannot do proper time aggregation.
        return quarterly.bfill().ffill()
    # Aggregate indicator to quarterly via mean. Use last-day-of-quarter as the
    # alignment point.
    indicator_q = indicator_monthly.resample("QE").mean()
    # Align quarterly observations: pull the value at each quarter-end.
    obs_q = quarterly.resample("QE").last().dropna()
    aligned = pd.concat([obs_q.rename("y"), indicator_q.rename("x")], axis=1).dropna()
    if aligned.shape[0] < 3:
        return quarterly.bfill().ffill()
    x = aligned["x"].to_numpy()
    y = aligned["y"].to_numpy()
    # OLS: y = alpha + beta * x
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    beta = float(((x - x_mean) * (y - y_mean)).sum() / denom) if denom > 0 else 0.0
    alpha = y_mean - beta * x_mean
    # Predicted quarterly series.
    pred_q = pd.Series(alpha + beta * indicator_q.to_numpy(), index=indicator_q.index)
    resid_q = (quarterly.resample("QE").last().reindex(pred_q.index) - pred_q).fillna(
        0.0
    )
    # Distribute alpha + beta * X^M with the quarterly residual smeared evenly.
    monthly = alpha / 3.0 + beta * indicator_monthly
    # Distribute each quarter's residual: each month in quarter Q gets
    # resid_q / 3.
    resid_monthly = pd.Series(0.0, index=monthly_index)
    for q_end, resid in resid_q.items():
        in_quarter = (monthly_index >= q_end - pd.Timedelta(days=92)) & (
            monthly_index <= q_end
        )
        n_months = int(in_quarter.sum())
        if n_months > 0:
            resid_monthly.loc[in_quarter] = float(resid) / n_months
    monthly = monthly + resid_monthly
    return monthly




# Cycle 17.5 LOW-A2 fix: enforce fred_sd_frequency_policy at validation time.
def _enforce_fred_sd_frequency_policy(
    policy: str,
    series_freq_map: dict,
    columns: list,
) -> None:
    """Raise ValueError when fred_sd_frequency_policy conditions are violated.

    Parameters
    ----------
    policy:
        The resolved fred_sd_frequency_policy axis value.
    series_freq_map:
        Mapping of column name to frequency string (e.g. "monthly", "quarterly").
        Only columns present in *columns* are examined.
    columns:
        Ordered list of column names in the panel (df.columns).

    Raises
    ------
    ValueError
        For reject_mixed_known_frequency when more than one distinct known
        frequency is detected across the panel columns.
        For require_single_known_frequency when more than one distinct known
        frequency is detected OR any column has an unknown/missing frequency.
    """
    if policy in ("report_only", "allow_mixed_frequency"):
        return  # No enforcement for permissive policies.

    known_freqs = set()
    unknown_cols = []
    for col in columns:
        raw = str(series_freq_map.get(col, "")).strip().lower()
        if raw in ("monthly", "quarterly"):
            known_freqs.add(raw)
        else:
            unknown_cols.append(col)

    if policy == "reject_mixed_known_frequency":
        if len(known_freqs) > 1:
            raise ValueError(
                "fred_sd_frequency_policy='reject_mixed_known_frequency': panel contains "
                f"columns at multiple known frequencies {sorted(known_freqs)!r}. "
                "Use allow_mixed_frequency or report_only to permit mixed panels, or "
                "filter to a single frequency with sd_series_frequency_filter."
            )

    elif policy == "require_single_known_frequency":
        errors = []
        if len(known_freqs) > 1:
            errors.append(
                f"columns at multiple known frequencies {sorted(known_freqs)!r}"
            )
        if unknown_cols:
            errors.append(
                f"{len(unknown_cols)} column(s) with unknown/missing frequency "
                f"(first few: {unknown_cols[:5]!r})"
            )
        if errors:
            raise ValueError(
                "fred_sd_frequency_policy='require_single_known_frequency': "
                + "; ".join(errors)
                + ". Use allow_mixed_frequency or report_only to permit such panels."
            )


def _apply_fred_sd_frequency_alignment(
    df: pd.DataFrame,
    resolved: dict[str, Any],
    l1_artifact: L1DataDefinitionArtifact,
    cleaning_log: dict[str, Any],
) -> pd.DataFrame:
    """Issue #202 -- align mixed-frequency FRED-SD panels.

    Reads ``sd_series_frequency_filter``, ``quarterly_to_monthly_rule``
    and ``monthly_to_quarterly_rule`` from the L2 resolved axes and the
    per-series frequency map from the L1 raw_panel metadata.
    """

    series_freq_map = (l1_artifact.raw_panel.metadata.values or {}).get(
        "series_frequency", {}
    ) or {}
    if not series_freq_map:
        # Without per-series frequency metadata we can't selectively align;
        # leave the panel untouched and record a no-op.
        cleaning_log.setdefault("steps", []).append(
            {
                "step": "fred_sd_frequency_alignment",
                "applied": False,
                "reason": "no series_frequency metadata",
            }
        )
        return df

    monthly_cols = [
        c for c in df.columns if str(series_freq_map.get(c, "")).lower() == "monthly"
    ]
    quarterly_cols = [
        c for c in df.columns if str(series_freq_map.get(c, "")).lower() == "quarterly"
    ]
    # Cycle 17.5 LOW-A2 fix: enforce fred_sd_frequency_policy BEFORE alignment.
    _enforce_fred_sd_frequency_policy(
        resolved.get("fred_sd_frequency_policy", "report_only"),
        series_freq_map,
        list(df.columns),
    )
    sd_filter = resolved.get("sd_series_frequency_filter", "both")
    qm_rule = resolved.get("quarterly_to_monthly_rule", "step_backward")
    mq_rule = resolved.get("monthly_to_quarterly_rule", "quarterly_average")

    if sd_filter == "monthly_only":
        df = df[
            monthly_cols
            + [
                c
                for c in df.columns
                if c not in monthly_cols and c not in quarterly_cols
            ]
        ]
        cleaning_log.setdefault("steps", []).append(
            {
                "step": "fred_sd_frequency_alignment",
                "filter": "monthly_only",
                "n_dropped": len(quarterly_cols),
            }
        )
        return df
    if sd_filter == "quarterly_only":
        df = df[
            quarterly_cols
            + [
                c
                for c in df.columns
                if c not in monthly_cols and c not in quarterly_cols
            ]
        ]
        cleaning_log.setdefault("steps", []).append(
            {
                "step": "fred_sd_frequency_alignment",
                "filter": "quarterly_only",
                "n_dropped": len(monthly_cols),
            }
        )
        return df

    # Both frequencies present -- harmonise to the dominant target
    # frequency. We default to monthly (most macroforecast recipes target
    # monthly horizons); the inverse path (monthly_to_quarterly) runs when
    # the target's frequency is quarterly.
    target_freq = (l1_artifact.frequency or "monthly").lower()
    if target_freq == "monthly" and quarterly_cols:
        for col in quarterly_cols:
            series = df[col]
            if qm_rule == "linear_interpolation":
                df[col] = series.interpolate(method="linear", limit_direction="both")
            elif qm_rule == "step_forward":
                df[col] = series.ffill()
            elif qm_rule == "chow_lin":
                # Issue #255 -- real Chow-Lin (1971) regression-based
                # disaggregation when a monthly indicator is supplied;
                # otherwise fall back to step_backward.
                indicator_col = (resolved.get("leaf_config") or {}).get(
                    "chow_lin_indicator"
                ) or _default_chow_lin_indicator(df, monthly_cols)
                if indicator_col and indicator_col in df.columns:
                    df[col] = _chow_lin_disaggregate(series, df[indicator_col])
                else:
                    df[col] = series.bfill().ffill()
            else:  # step_backward
                df[col] = series.bfill().ffill()
        cleaning_log.setdefault("steps", []).append(
            {
                "step": "fred_sd_frequency_alignment",
                "rule": qm_rule,
                "direction": "quarterly_to_monthly",
                "n_cols": len(quarterly_cols),
            }
        )
    elif target_freq == "quarterly" and monthly_cols:
        if not isinstance(df.index, pd.DatetimeIndex):
            cleaning_log.setdefault("steps", []).append(
                {
                    "step": "fred_sd_frequency_alignment",
                    "applied": False,
                    "reason": "non-datetime index",
                }
            )
            return df
        if mq_rule == "quarterly_average":
            agg = df[monthly_cols].resample("QE").mean()
        elif mq_rule == "quarterly_endpoint":
            agg = df[monthly_cols].resample("QE").last()
        else:  # quarterly_sum
            agg = df[monthly_cols].resample("QE").sum()
        # Align back to the (quarterly) target index by reindexing.
        df_q = df[
            quarterly_cols
            + [
                c
                for c in df.columns
                if c not in monthly_cols and c not in quarterly_cols
            ]
        ]
        df = df_q.join(agg, how="left").reindex(df_q.index)
        cleaning_log.setdefault("steps", []).append(
            {
                "step": "fred_sd_frequency_alignment",
                "rule": mq_rule,
                "direction": "monthly_to_quarterly",
                "n_cols": len(monthly_cols),
            }
        )
    return df


def _apply_transform(
    frame: pd.DataFrame,
    resolved: l2_layer.L2ResolvedAxes,
    l2_leaf: dict[str, Any],
    l1_leaf: dict[str, Any],
    cleaning_log: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    policy = resolved.get("transform_policy")
    if policy == "no_transform":
        cleaning_log["steps"].append({"transform": "no_transform"})
        return frame, {}
    tcode_map = dict(l1_leaf.get("official_tcode_map", {}))
    tcode_map.update(l1_leaf.get("custom_tcode_map", {}))
    tcode_map.update(l2_leaf.get("custom_tcode_map", {}))
    if policy == "apply_official_tcode" and not tcode_map:
        cleaning_log["steps"].append(
            {"transform": "apply_official_tcode", "fallback": "no_tcode_map_available"}
        )
        return frame, {}
    if policy == "custom_tcode" and not tcode_map:
        raise ValueError("custom_tcode runtime requires custom_tcode_map")
    # F-P1-14 fix: honour official_transform_scope (BREAKING)
    scope = resolved.get("transform_scope", "target_and_predictors") or "target_and_predictors"
    target_col = l1_leaf.get("target") or l2_leaf.get("target")
    transformed = frame.copy()
    applied: dict[str, int] = {}
    if scope in ("none", "not_applicable"):
        cleaning_log["steps"].append({"transform": policy, "applied": {}, "scope": scope})
        return transformed, applied
    for column, tcode in tcode_map.items():
        if column not in transformed.columns:
            continue
        if scope == "target_only" and column != target_col:
            continue
        if scope == "predictors_only" and column == target_col:
            continue
        transformed[column] = _apply_tcode(transformed[column], int(tcode))
        applied[column] = int(tcode)
    cleaning_log["steps"].append({"transform": policy, "applied": applied})
    return transformed, applied


def _apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    if tcode == 1:
        return series
    if tcode == 2:
        return series.diff()
    if tcode == 3:
        return series.diff().diff()
    if tcode == 4:
        return _safe_log(series)
    if tcode == 5:
        return _safe_log(series).diff()
    if tcode == 6:
        return _safe_log(series).diff().diff()
    if tcode == 7:
        return series.pct_change()
    raise ValueError(f"unsupported tcode {tcode}; expected 1..7")


def _safe_log(series: pd.Series) -> pd.Series:
    positive = series.where(series > 0)
    return positive.map(
        lambda value: pd.NA if pd.isna(value) else __import__("math").log(value)
    )


def _try_custom_l2_preprocessor(
    name: str, frame: pd.DataFrame, leaf_config: dict[str, Any]
) -> pd.DataFrame | None:
    """Issue #251 -- dispatch to a user-registered preprocessor when ``name``
    matches a registered entry. Returns ``None`` to indicate fall-through
    to built-in policies.

    Cycle 14 J-4 fix: silent skip bug removed.
    The previous bare ``except Exception: return None`` swallowed TypeError
    from a wrong function signature, silently skipping the preprocessor and
    preventing manifest provenance from reflecting truth.  Now TypeError
    surfaces as a clear ValueError.  Unexpected return types also raise
    instead of silently falling through.

    Contract: ``fn(X_train, y_train, X_test, context) -> (X_train, X_test)``
    or ``-> pd.DataFrame``. The runtime substitutes ``X_train = X_test =
    frame`` for the single-pass L2 hook."""

    try:
        from .. import custom as _custom_mod
    except ImportError:  # pragma: no cover
        return None
    if not _custom_mod.is_custom_preprocessor(str(name)):
        return None
    spec = _custom_mod.get_custom_preprocessor(str(name))
    # Cycle 15 M-2 fix: signature validation pre-call
    # Use inspect.signature to validate arity BEFORE calling spec.function.
    # This ensures a wrong-arity function raises ValueError with a clear hint,
    # while any TypeError raised INSIDE the user's function body propagates
    # naturally as TypeError (not misattributed to "wrong signature").
    try:
        sig = inspect.signature(spec.function)
        params = sig.parameters
        _positional_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
        )
        positional_count = len(
            [p for p in params.values() if p.kind in _positional_kinds]
        )
        accepts_4_args = has_var_positional or positional_count >= 4
        if not accepts_4_args:
            raise ValueError(
                f"custom preprocessor {name!r} has wrong signature. "
                "Expected fn(X_train, y_train, X_test, context) -> "
                "(X_train, X_test) or pd.DataFrame, "
                f"got {sig}."
            )
    except (ValueError, TypeError) as _sig_err:
        # inspect.signature may fail on C-implemented callables; re-raise
        # only if it is our own signature-hint ValueError.
        if isinstance(_sig_err, ValueError) and "wrong signature" in str(_sig_err):
            raise
        pass  # fall through to direct call for builtins/C callables
    # Documented contract: ``fn(X_train, y_train, X_test, context) ->
    # (X_train, X_test)``.  For the runtime hook we substitute
    # ``X_train = X_test = frame`` (single-pass L2 panel clean).
    # Body TypeErrors propagate naturally (no wrapping).
    result = spec.function(frame, None, frame, dict(leaf_config))
    if isinstance(result, tuple) and result:
        cleaned = result[0]
        if not isinstance(cleaned, pd.DataFrame):
            raise ValueError(
                f"custom preprocessor {name!r} returned a tuple whose "
                f"first element is not a DataFrame (got {type(cleaned).__name__!r})"
            )
        return cleaned
    if isinstance(result, pd.DataFrame):
        return result
    raise ValueError(
        f"custom preprocessor {name!r} returned an unexpected type "
        f"{type(result).__name__!r}; expected pd.DataFrame or "
        "(pd.DataFrame, pd.DataFrame)"
    )



def _apply_outlier_policy_per_origin(
    frame: pd.DataFrame,
    resolved: "l2_layer.L2ResolvedAxes",
    leaf_config: dict[str, Any],
    cutoff_ts: Any,
) -> pd.DataFrame:
    """Apply outlier policy using only data up to cutoff_ts for threshold computation.

    Prevents lookahead leakage for per_origin temporal rule.
    """
    policy = resolved.get("outlier_policy")
    action = resolved.get("outlier_action", "flag_as_nan")
    if policy == "none":
        return frame
    result = frame.copy()
    numeric = result.select_dtypes("number")
    if numeric.empty:
        return result
    # Stats computed on expanding window up to cutoff only
    train = numeric.loc[:cutoff_ts] if cutoff_ts is not None else numeric
    if train.empty:
        train = numeric
    if policy == "mccracken_ng_iqr":
        threshold = float(leaf_config.get("outlier_iqr_threshold", 10.0))
        median = train.median()
        iqr = train.quantile(0.75) - train.quantile(0.25)
        mask = (numeric - median).abs() > threshold * iqr.replace(0, pd.NA)
    elif policy == "zscore_threshold":
        threshold = float(leaf_config.get("zscore_threshold_value", 3.0))
        mask = (
            (numeric - train.mean()) / train.std(ddof=0).replace(0, pd.NA)
        ).abs() > threshold
    elif policy == "winsorize":
        low, high = leaf_config.get("winsorize_quantiles", [0.01, 0.99])
        lo_val = train.quantile(low)
        hi_val = train.quantile(high)
        clipped = numeric.clip(lo_val, hi_val, axis=1)
        result[numeric.columns] = clipped
        return result
    else:
        return frame
    if action == "flag_as_nan":
        result[numeric.columns] = numeric.mask(mask)
    elif action == "replace_with_median":
        result[numeric.columns] = numeric.mask(mask, train.median(), axis=1)
    elif action == "replace_with_cap_value":
        upper = train.quantile(0.99)
        lower = train.quantile(0.01)
        capped = numeric.clip(lower=lower, upper=upper, axis=1)
        result[numeric.columns] = numeric.where(~mask.fillna(False), capped)
    else:
        result[numeric.columns] = numeric.mask(mask)
    return result


def _apply_imputation_per_origin(
    frame: pd.DataFrame,
    resolved: "l2_layer.L2ResolvedAxes",
    cutoff_ts: Any,
) -> pd.DataFrame:
    """Apply imputation using stats computed only up to cutoff_ts.

    Prevents lookahead leakage for per_origin temporal rule.
    """
    policy = resolved.get("imputation_policy")
    if policy == "none_propagate":
        return frame
    # Stats/fit computed on expanding window up to cutoff only
    train = frame.loc[:cutoff_ts] if cutoff_ts is not None else frame
    if train.empty:
        train = frame
    if policy == "mean":
        fill_values = train.mean(numeric_only=True)
        return frame.fillna(fill_values)
    elif policy in {"em_factor", "em_multivariate"}:
        # For per-origin EM, fit on training window and apply imputation to full frame
        train_imputed = _pca_em_imputation(train, n_factors=8 if policy == "em_factor" else None, max_iter=20)
        # Build fill values from train imputed mean
        fill_values = train_imputed.mean(numeric_only=True)
        return frame.fillna(fill_values)
    elif policy == "forward_fill":
        return frame.ffill()
    elif policy == "linear_interpolation":
        return frame.interpolate(method="linear")
    else:
        return frame

def _apply_outlier_policy(
    frame: pd.DataFrame,
    resolved: l2_layer.L2ResolvedAxes,
    leaf_config: dict[str, Any],
    cleaning_log: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    policy = resolved.get("outlier_policy")
    action = resolved.get("outlier_action", "flag_as_nan")
    if policy == "none":
        cleaning_log["steps"].append({"outlier": "none"})
        return frame, 0
    result = frame.copy()
    numeric = result.select_dtypes("number")
    if numeric.empty:
        return result, 0
    if policy == "mccracken_ng_iqr":
        threshold = float(leaf_config.get("outlier_iqr_threshold", 10.0))
        median = numeric.median()
        iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
        mask = (numeric - median).abs() > threshold * iqr.replace(0, pd.NA)
    elif policy == "zscore_threshold":
        threshold = float(leaf_config.get("zscore_threshold_value", 3.0))
        mask = (
            (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, pd.NA)
        ).abs() > threshold
    elif policy == "winsorize":
        low, high = leaf_config.get("winsorize_quantiles", [0.01, 0.99])
        clipped = numeric.clip(numeric.quantile(low), numeric.quantile(high), axis=1)
        changed = int(
            (clipped.ne(numeric) & ~(clipped.isna() & numeric.isna())).sum().sum()
        )
        result[numeric.columns] = clipped
        cleaning_log["steps"].append(
            {
                "outlier": "winsorize",
                "action": action,
                "quantiles": [low, high],
                "capped": changed,
            }
        )
        return result, changed
    else:
        raise NotImplementedError(
            f"outlier_policy={policy!r} runtime is not implemented"
        )
    count = int(mask.fillna(False).sum().sum())
    if action == "flag_as_nan":
        result[numeric.columns] = numeric.mask(mask)
    elif action == "replace_with_median":
        result[numeric.columns] = numeric.mask(mask, numeric.median(), axis=1)
    elif action == "replace_with_cap_value":
        upper = numeric.quantile(0.99)
        lower = numeric.quantile(0.01)
        capped = numeric.clip(lower=lower, upper=upper, axis=1)
        result[numeric.columns] = numeric.where(~mask.fillna(False), capped)
    else:
        raise NotImplementedError(
            f"outlier_action={action!r} runtime is not implemented"
        )
    cleaning_log["steps"].append(
        {"outlier": policy, "action": action, "flagged": count}
    )
    return result, count


def _apply_imputation(
    frame: pd.DataFrame, resolved: l2_layer.L2ResolvedAxes, cleaning_log: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    policy = resolved.get("imputation_policy")
    missing_before = int(frame.isna().sum().sum())
    if policy == "none_propagate":
        cleaning_log["steps"].append({"imputation": "none_propagate"})
        return frame, 0
    if policy == "mean":
        result = frame.fillna(frame.mean(numeric_only=True))
        method = "mean"
    elif policy in {"em_factor", "em_multivariate"}:
        result = _pca_em_imputation(
            frame, n_factors=8 if policy == "em_factor" else None, max_iter=20
        )
        method = policy
    elif policy == "forward_fill":
        result = frame.ffill()
        method = "forward_fill"
    elif policy == "linear_interpolation":
        result = frame.interpolate(method="linear")
        method = "linear_interpolation"
    else:
        raise NotImplementedError(
            f"imputation_policy={policy!r} runtime is not implemented"
        )
    filled = missing_before - int(result.isna().sum().sum())
    cleaning_log["steps"].append({"imputation": method, "filled": filled})
    return result, filled


def _pca_em_imputation(
    frame: pd.DataFrame, *, n_factors: int | None, max_iter: int = 20, tol: float = 1e-4
) -> pd.DataFrame:
    """McCracken-Ng (2016) PCA-EM imputation.

    Iterates PCA reconstruction in place of missing values until convergence.
    """

    numeric = frame.select_dtypes("number")
    if numeric.empty:
        return frame
    matrix = numeric.to_numpy(dtype=float)
    mask = np.isnan(matrix)
    if not mask.any():
        return frame
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds[stds == 0] = 1.0
    filled = matrix.copy()
    for col_idx in range(matrix.shape[1]):
        col = filled[:, col_idx]
        col[np.isnan(col)] = means[col_idx]
        filled[:, col_idx] = col
    standardized = (filled - means) / stds
    rank = min(
        int(n_factors) if n_factors else min(matrix.shape) // 2, min(matrix.shape) - 1
    )
    if rank < 1:
        return frame.fillna(frame.mean(numeric_only=True))
    last = standardized.copy()
    for _ in range(max_iter):
        u, s, vt = np.linalg.svd(last, full_matrices=False)
        s[rank:] = 0.0
        approximation = (u * s) @ vt
        last_new = standardized.copy()
        last_new[mask] = approximation[mask]
        delta = float(np.linalg.norm(last_new - last)) / max(np.linalg.norm(last), 1e-9)
        last = last_new
        if delta < tol:
            break
    imputed = last * stds + means
    result = frame.copy()
    result[numeric.columns] = imputed
    # restore non-numeric columns untouched
    return result


def _apply_frame_edge(
    frame: pd.DataFrame, resolved: l2_layer.L2ResolvedAxes, cleaning_log: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    policy = resolved.get("frame_edge_policy")
    before = len(frame)
    if policy == "keep_unbalanced":
        result = frame
    elif policy == "truncate_to_balanced":
        result = frame.dropna(axis=0, how="any")
    elif policy == "drop_unbalanced_series":
        result = frame.dropna(axis=1, how="any")
    elif policy == "zero_fill_leading":
        result = frame.fillna(0)
    else:
        raise NotImplementedError(
            f"frame_edge_policy={policy!r} runtime is not implemented"
        )
    truncated = max(before - len(result), 0)
    cleaning_log["steps"].append({"frame_edge": policy, "truncated_rows": truncated})
    return result, truncated


def _panel_from_frame(frame: pd.DataFrame, metadata: dict[str, Any]) -> Panel:
    return Panel(
        data=frame,
        shape=frame.shape,
        column_names=tuple(str(column) for column in frame.columns),
        index=pd.DatetimeIndex(frame.index),
        metadata=PanelMetadata(values=metadata),
    )


def _minimal_l3_params(raw: dict[str, Any]) -> dict[str, Any]:
    lag_node = _first_node(raw, op="lag")
    target_node = _first_node(raw, op="target_construction")
    lag_params = lag_node.get("params", {}) if lag_node else {}
    target_params = target_node.get("params", {}) if target_node else {}
    return {
        "n_lag": lag_params.get("n_lag", 1),
        "horizon": target_params.get("horizon", 1),
    }


def _execute_l3_dag(dag, frame: pd.DataFrame, target_name: str) -> dict[str, Any]:
    """Issue #257 -- enforce ``cascade_max_depth`` and propagate
    ``pipeline_id`` through the executed step nodes.

    Cascade depth is the longest chain of ``step`` nodes from any source.
    The default cap is 10; override via the recipe's L3
    ``leaf_config.cascade_max_depth``. ``pipeline_id`` is sourced from
    each step's params (or inherited from its deepest input) so L7
    lineage / transformation_attribution can group steps without
    re-parsing the recipe.
    """

    cascade_max_depth = (
        int(getattr(dag, "leaf_config", {}).get("cascade_max_depth", 10))
        if hasattr(dag, "leaf_config")
        else 10
    )

    values: dict[str, Any] = {}
    depth_by_node: dict[str, int] = {}
    pipeline_by_node: dict[str, str] = {}

    for node in _topological_nodes(dag):
        # Cascade depth is one more than the deepest input's depth.
        input_depths = [depth_by_node.get(ref.node_id, 0) for ref in node.inputs]
        node_depth = (
            (max(input_depths) + 1) if (node.type == "step" and input_depths) else 0
        )
        depth_by_node[node.id] = node_depth
        if node_depth > cascade_max_depth:
            offending_chain = [node.id]
            cursor = node
            while cursor.inputs:
                deepest = max(
                    cursor.inputs, key=lambda ref: depth_by_node.get(ref.node_id, 0)
                )
                offending_chain.append(deepest.node_id)
                cursor = dag.nodes[deepest.node_id]
                if cursor.type != "step":
                    break
            raise ValueError(
                f"L3 cascade depth ({node_depth}) exceeds cascade_max_depth ({cascade_max_depth}); "
                f"offending chain: {' -> '.join(reversed(offending_chain))}"
            )
        # Resolve pipeline_id: explicit params.pipeline_id wins, otherwise
        # inherit from the deepest input.
        explicit_pipeline = (
            (node.params or {}).get("pipeline_id") if node.type == "step" else None
        )
        if explicit_pipeline:
            pipeline_by_node[node.id] = str(explicit_pipeline)
        elif node.inputs:
            inherited = next(
                (
                    pipeline_by_node[ref.node_id]
                    for ref in node.inputs
                    if ref.node_id in pipeline_by_node
                ),
                "",
            )
            if inherited:
                pipeline_by_node[node.id] = inherited

        if node.type == "source":
            values[node.id] = _execute_l3_source(node.selector, frame, target_name)
        elif node.op == "l3_feature_bundle":
            values[node.id] = tuple(values[ref.node_id] for ref in node.inputs)
        elif node.op == "l3_metadata_build":
            values[node.id] = None
        else:
            inputs = [values[ref.node_id] for ref in node.inputs]
            values[node.id] = _execute_l3_op(node.op, inputs, node.params, target_name)

    # Stash the pipeline-id breadcrumb on the dag for L3 metadata builders.
    try:
        dag.runtime_pipeline_by_node = dict(pipeline_by_node)
        dag.runtime_depth_by_node = dict(depth_by_node)
    except Exception:
        pass
    return values


def _execute_l3_source(
    selector, frame: pd.DataFrame, target_name: str
) -> pd.DataFrame | pd.Series:
    if selector is None:
        raise ValueError("L3 source node requires a selector")
    if selector.layer_ref != "l2" or selector.sink_name != "l2_clean_panel_v1":
        raise NotImplementedError(
            "minimal L3 runtime currently supports L2 clean panel sources only"
        )
    subset = selector.subset or {}
    role = subset.get("role")
    if role == "target":
        return frame[target_name].copy()
    if role == "predictors":
        return frame[
            [column for column in frame.columns if column != target_name]
        ].copy()
    if "variable_list" in subset:
        return frame[list(subset["variable_list"])].copy()
    if subset.get("raw") is True:
        return frame.copy()
    raise NotImplementedError(
        f"minimal L3 runtime does not support source subset {subset!r}"
    )


def _topological_nodes(dag) -> list[Any]:
    ordered = []
    pending = dict(dag.nodes)
    while pending:
        progressed = False
        for node_id, node in list(pending.items()):
            if all(ref.node_id not in pending for ref in node.inputs):
                ordered.append(node)
                pending.pop(node_id)
                progressed = True
        if not progressed:
            raise ValueError(
                f"{dag.layer_id}: DAG contains unresolved dependencies or a cycle"
            )
    return ordered


def _try_custom_l3_dispatch(
    op: str, inputs: list[Any], params: dict[str, Any]
) -> pd.DataFrame | pd.Series | None:
    """Issue #251 -- best-effort dispatch to a user-registered feature
    block / combiner. Returns ``None`` if no registered op matches,
    indicating the caller should fall through to the built-in branches.
    """

    try:
        from .. import custom as _custom_mod
    except ImportError:  # pragma: no cover
        return None
    spec = None
    is_combiner = False
    block_kind = params.get("block_kind") if isinstance(params, dict) else None
    if _custom_mod.is_custom_feature_combiner(op):
        spec = _custom_mod.get_custom_feature_combiner(op)
        is_combiner = True
    elif block_kind and _custom_mod.is_custom_feature_block(op, block_kind=block_kind):
        spec = _custom_mod.get_custom_feature_block(op, block_kind=block_kind)
    elif _custom_mod.is_custom_feature_block(op):
        for kind in ("temporal", "rotation", "factor"):
            if _custom_mod.is_custom_feature_block(op, block_kind=kind):
                spec = _custom_mod.get_custom_feature_block(op, block_kind=kind)
                break
    if spec is None:
        return None
    fn = spec.function
    try:
        if is_combiner:
            # Combiners receive the input list (multi-frame merge contract).
            return fn(inputs, dict(params))
        return fn(inputs[0] if inputs else None, dict(params))
    except Exception:  # pragma: no cover - fall through to built-in
        return None


def _execute_l3_op(
    op: str, inputs: list[Any], params: dict[str, Any], target_name: str
) -> pd.DataFrame | pd.Series:
    # Issue #251 -- if a user-registered feature_block / combiner matches
    # this op name, dispatch to it before the built-in handlers. The
    # contract is the v0.1 ``CustomFeatureBlock`` callable: receives a
    # context object and returns a ``FeatureBlockCallableResult``. For
    # a thin promotion we accept both that protocol and the simpler
    # callable-on-frame signature ``fn(frame, params) -> frame`` so
    # registered ops are usable in unit tests without the full L2
    # context plumbing.
    custom_result = _try_custom_l3_dispatch(op, inputs, params)
    if custom_result is not None:
        return custom_result
    if op == "identity" or op == "level":
        return inputs[0]
    if op == "lag":
        return _lagged_predictors(
            _as_frame(inputs[0]),
            n_lag=int(params.get("n_lag", 4)),
            include_contemporaneous=bool(params.get("include_contemporaneous", False)),
        )
    if op == "seasonal_lag":
        return _seasonal_lagged_predictors(
            _as_frame(inputs[0]),
            seasonal_period=int(params.get("seasonal_period", 12)),
            n_seasonal_lags=int(params.get("n_seasonal_lags", 1)),
        )
    if op == "ma_window":
        return (
            _as_frame(inputs[0])
            .rolling(
                window=int(params.get("window", 3)),
                min_periods=int(params.get("window", 3)),
            )
            .mean()
        )
    if op == "ma_increasing_order":
        return _ma_increasing_order(
            _as_frame(inputs[0]), max_order=int(params.get("max_order", 12))
        )
    if op == "maf_per_variable_pca":
        return _maf_per_variable_pca(
            _as_frame(inputs[0]),
            n_lags=int(params.get("n_lags", 12)),
            n_components_per_var=int(params.get("n_components_per_var", 2)),
        )
    if op == "cumsum":
        return inputs[0].cumsum()
    if op == "concat":
        return pd.concat([_as_frame(value) for value in inputs], axis=1)
    if op == "scale":
        return _scale_frame(_as_frame(inputs[0]), method=params.get("method", "zscore"))
    if op == "log":
        return _map_like(
            inputs[0],
            lambda value: (
                pd.NA if pd.isna(value) or value <= 0 else __import__("math").log(value)
            ),
        )
    if op == "diff":
        return _diff_like(inputs[0], periods=int(params.get("n_diff", 1)))
    if op == "log_diff":
        logged = _map_like(
            inputs[0],
            lambda value: (
                pd.NA if pd.isna(value) or value <= 0 else __import__("math").log(value)
            ),
        )
        return _diff_like(logged, periods=int(params.get("n_diff", 1)))
    if op == "pct_change":
        return _pct_change_like(inputs[0], periods=int(params.get("n_periods", 1)))
    if op in {"polynomial_expansion", "polynomial"}:
        return _polynomial_expansion(
            _as_frame(inputs[0]), degree=int(params.get("degree", 2))
        )
    if op == "interaction":
        return _interaction_terms(_as_frame(inputs[0]))
    if op == "season_dummy":
        return _season_dummy(_as_frame(inputs[0]))
    if op == "time_trend":
        frame = _as_frame(inputs[0])
        return pd.Series(range(1, len(frame) + 1), index=frame.index, name="time_trend")
    if op == "holiday":
        return _holiday_indicator(_as_frame(inputs[0]))
    if op == "regime_indicator":
        return _regime_indicator(inputs[0])
    if op in {"pca", "sparse_pca", "scaled_pca"}:
        # Phase A3 fix: accept ``n_components="all"`` sentinel — the
        # _pca_factors helper resolves it to ``min(T, N)`` at fit time.
        n_comp_param = params.get("n_components", 8)
        n_comp_resolved: int | str = (
            "all"
            if isinstance(n_comp_param, str) and n_comp_param == "all"
            else int(n_comp_param)
        )
        return _pca_factors(
            _as_frame(inputs[0]),
            n_components=n_comp_resolved,
            variant=op,
            target_signal=_first_series(inputs),
        )
    if op == "sparse_pca_chen_rohe":
        return _sparse_pca_chen_rohe(
            _as_frame(inputs[0]),
            n_components=int(params.get("n_components", 4)),
            zeta=float(params.get("zeta", 0.0)),  # 0.0 → default to J = n_components
            max_iter=int(params.get("max_iter", 200)),
            # v0.9.0F audit-fix + Phase B-14 paper-14 F2 closure: Rapach &
            # Zhou (2025) Strategy step 2 ("Fit a VAR(1) to the sparse
            # components; the fitted residuals constitute the set of
            # sparse macro-finance factors") is opt-in via
            # ``var_innovations``. Default False = sparse principal-
            # component scores only (the v0.9.0C-3 baseline).
            # Implementation is a true VAR(1) (cross-equation lags
            # retained) -- the v0.9.0F per-column AR(1) was the F2
            # finding now closed.
            var_innovations=bool(params.get("var_innovations", False)),
            random_state=int(params.get("random_state", 0)),
        )
    if op == "supervised_pca":
        return _supervised_pca(
            _as_frame(inputs[0]),
            target=_first_series(inputs),
            n_components=int(params.get("n_components", 4)),
            q=float(params.get("q", 0.5)),
        )
    if op == "varimax" or op == "varimax_rotation":
        return _varimax_rotation(_as_frame(inputs[0]))
    if op == "partial_least_squares":
        return _partial_least_squares(
            _as_frame(inputs[0]),
            target=_first_series(inputs),
            n_components=int(params.get("n_components", 4)),
        )
    if op == "random_projection":
        return _random_projection(
            _as_frame(inputs[0]), n_components=int(params.get("n_components", 8))
        )
    if op == "dfm":
        return _dfm_factors(
            _as_frame(inputs[0]), n_factors=int(params.get("n_factors", 3))
        )
    if op == "wavelet":
        return _wavelet_decomposition(
            _as_frame(inputs[0]), n_levels=int(params.get("n_levels", 1))
        )
    if op == "fourier":
        return _fourier_features(
            _as_frame(inputs[0]),
            n_terms=int(params.get("n_terms", 3)),
            period=int(params.get("period", 12)),
        )
    if op == "hp_filter":
        return _hp_filter(
            _as_frame(inputs[0]),
            lam=float(params.get("lambda_", params.get("lam", 1600.0))),
        )
    if op == "hamilton_filter":
        return _hamilton_filter(
            _as_frame(inputs[0]),
            n_lags=int(params.get("n_lags", 8)),
            n_horizon=int(params.get("n_horizon", 24)),
        )
    if op == "savitzky_golay_filter":
        return _savitzky_golay_filter(
            _as_frame(inputs[0]),
            window_length=int(params.get("window_length", 5)),
            polyorder=int(params.get("polyorder", 2)),
        )
    if op == "asymmetric_trim":
        return _asymmetric_trim(
            _as_frame(inputs[0]),
            smooth_window=int(params.get("smooth_window", 0)),
        )
    if op == "adaptive_ma_rf":
        return _adaptive_ma_rf(
            _as_frame(inputs[0]),
            n_estimators=int(params.get("n_estimators", 500)),
            min_samples_leaf=int(params.get("min_samples_leaf", 40)),
            sided=str(params.get("sided", "two")).lower(),
            random_state=int(params.get("random_state", 0)),
        )
    if op in {"kernel", "kernel_features"}:
        return _kernel_features(
            _as_frame(inputs[0]),
            kind=params.get("kind", "rbf"),
            gamma=float(params.get("gamma", 1.0)),
        )
    if op in {"nystroem", "nystroem_features"}:
        return _nystroem_features(
            _as_frame(inputs[0]), n_components=int(params.get("n_components", 32))
        )
    if op == "feature_selection":
        return _feature_selection(
            _as_frame(inputs[0]),
            target=_first_series(inputs),
            n_features=params.get("n_features", 0.5),
            method=params.get("method", "variance"),
        )
    if op == "hierarchical_pca":
        return _hierarchical_pca(
            inputs, n_components_per_block=int(params.get("n_components_per_block", 1))
        )
    if op == "weighted_concat":
        return _weighted_concat(inputs, weights=params.get("weights"))
    if op == "simple_average":
        return _simple_average(inputs)
    if op == "interact":
        return _interaction_terms(_as_frame(inputs[0]))
    if op == "target_construction":
        horizon = int(params.get("horizon", 1))
        mode = params.get("mode", "point_forecast")
        y_series = _as_series(inputs[0], name=target_name)
        if mode in {"cumulative_average", "path_average"}:
            target = _cumulative_average_target(y_series, horizon=horizon).rename(
                target_name
            )
            # Phase B-15 paper-15 F4: stash the source (un-averaged) y so
            # ``materialize_l4_minimal`` can fit the paper Eq. 4 path-average
            # estimator (h separate models on per-horizon shifted targets)
            # when ``forecast_strategy="path_average_eq4"`` is opted in.
            target.attrs["y_orig"] = y_series.copy()
        else:
            target = y_series.shift(-horizon).rename(target_name)
        target.attrs["horizon"] = horizon
        target.attrs["mode"] = mode
        return target
    if op == "u_midas":
        _u_midas_n_lags = params.get("n_lags_high", "bic")
        _u_midas_include_y_lag = bool(params.get("include_y_lag", False))
        # Resolve n_lags_high: keep as string for BIC/AIC, cast to int otherwise
        if isinstance(_u_midas_n_lags, str) and _u_midas_n_lags not in ("bic", "aic"):
            _u_midas_n_lags = int(_u_midas_n_lags)
        elif not isinstance(_u_midas_n_lags, str):
            _u_midas_n_lags = int(_u_midas_n_lags)
        # Extract y_series for AR(1) y-lag when include_y_lag=True
        _u_midas_y_series = None
        if _u_midas_include_y_lag and len(inputs) > 1:
            _u_midas_y_series = _first_series(inputs)
        return _u_midas(
            _as_frame(inputs[0]),
            freq_ratio=int(params.get("freq_ratio", 3)),
            n_lags_high=_u_midas_n_lags,
            target_freq=str(params.get("target_freq", "low")),
            include_y_lag=_u_midas_include_y_lag,
            y_series=_u_midas_y_series,
            ic_selector=str(_u_midas_n_lags) if isinstance(_u_midas_n_lags, str) else "bic",
        )
    if op == "midas":
        return _midas(
            _as_frame(inputs[0]),
            target=_first_series(inputs),
            weighting=str(params.get("weighting", "exp_almon")),
            polynomial_order=int(params.get("polynomial_order", 2)),
            freq_ratio=int(params.get("freq_ratio", 3)),
            n_lags_high=int(params.get("n_lags_high", 12)),
            sum_to_one=bool(params.get("sum_to_one", True)),
            max_iter=int(params.get("max_iter", 200)),
        )
    if op == "sliced_inverse_regression":
        n_comp_param = params.get("n_components", 2)
        n_comp_resolved: int = (
            int(n_comp_param)
            if not (isinstance(n_comp_param, str) and n_comp_param == "all")
            else max(1, int(_as_frame(inputs[0]).shape[1]))
        )
        return _sliced_inverse_regression(
            _as_frame(inputs[0]),
            target=_first_series(inputs),
            n_components=n_comp_resolved,
            n_slices=int(params.get("n_slices", 10)),
            scaling_method=str(params.get("scaling_method", "scaled_pca")),
        )
    raise NotImplementedError(f"L3 runtime does not support op {op!r}")


# ---------------------------------------------------------------------------
# Phase C M1 / M2 -- MIDAS helpers.
# ---------------------------------------------------------------------------


def _midas_lag_stack(
    frame: pd.DataFrame,
    *,
    freq_ratio: int,
    n_lags_high: int,
    target_freq: str = "low",
) -> pd.DataFrame:
    """Stack high-frequency lags as separate columns.

    For each column ``col`` in ``frame``, emit columns
    ``col_lag0, col_lag1, …, col_lag{K-1}`` where
    ``col_lagk[t_LF] = frame[col].iloc[t_LF · m − k]``. Rows with
    incomplete K-history are NaN. Output index is the LF subset of the
    original index (every m-th row when target_freq='low'); when
    target_freq='high' the LF subset is the full HF index.

    Paper reference: Foroni-Marcellino-Schumacher (2011/2015) §2.1 eq.(8).
    Bundesbank DP 35/2011; JRSS-A 178(1), 57-82, DOI 10.1111/rssa.12043.
    Design matrix structure (k=3 example, stock variable omega(L)=1):
      lag-0 col = x at the HF index coinciding with the LF boundary (eq.(10)
                  convention: direct-projection, lag-0 is the most recent HF obs).
      lag-j col = x at j HF steps before the LF boundary.
    """

    m = max(1, int(freq_ratio))
    K = max(1, int(n_lags_high))
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    if target_freq == "low":
        lf_positions = list(range(0, len(frame), m))
        out_index = frame.index[lf_positions]
    else:
        lf_positions = list(range(len(frame)))
        out_index = frame.index
    out: dict[str, np.ndarray] = {}
    for col in frame.columns:
        x = pd.Series(frame[col]).astype(float).to_numpy()
        for k in range(K):
            lagged = np.full(len(out_index), np.nan, dtype=float)
            for i, lf_pos in enumerate(lf_positions):
                hf_pos = lf_pos - k
                if 0 <= hf_pos < len(x):
                    lagged[i] = x[hf_pos]
            out[f"{col}_lag{k}"] = lagged
    return pd.DataFrame(out, index=out_index)


def _bic_select_k(
    frame_hf: pd.DataFrame,
    y_lf: "pd.Series | None",
    *,
    freq_ratio: int,
    include_y_lag: bool,
    ic: str = "bic",
) -> int:
    """BIC/AIC lag-order selection for U-MIDAS (paper §3.2 p.11 + §3.5).

    Searches K in {1, ..., K_max} where K_max = ceil(1.5 * freq_ratio).
    Fits OLS at each K, computes IC, returns K_star = argmin IC.

    Paper reference: Foroni-Marcellino-Schumacher (2011/2015) §3.2 eq.(20),
    §3.5. BIC formula: log(RSS/T_eff) + K_params * log(T_eff) / T_eff.
    Bundesbank DP 35/2011; JRSS-A 178(1), 57-82, DOI 10.1111/rssa.12043.
    """
    import math
    import warnings

    K_max = max(1, math.ceil(1.5 * freq_ratio))
    # Cycle 15.6 fix: BIC intractability warning
    _BIC_K_MAX_WARNING_THRESHOLD = 30  # empirical threshold above which BIC search becomes slow
    if K_max > _BIC_K_MAX_WARNING_THRESHOLD:
        warnings.warn(
            f"U-MIDAS BIC search will enumerate K_max={K_max} candidates "
            f"(freq_ratio={freq_ratio}). For freq_ratio > {_BIC_K_MAX_WARNING_THRESHOLD / 1.5:.0f}, "
            "this may take hours. Consider setting `n_lags_high` manually in the L3 op params "
            "to bypass BIC search, or use a coarser frequency representation.",
            UserWarning,
            stacklevel=2,
        )
    best_ic_val = np.inf
    best_K = 1
    any_valid = False

    for K_cand in range(1, K_max + 1):
        # Build lag-stacked design
        stacked = _midas_lag_stack(
            frame_hf, freq_ratio=freq_ratio, n_lags_high=K_cand, target_freq="low"
        )
        # Optionally prepend y-lag column
        if include_y_lag and y_lf is not None:
            y_aligned = y_lf.reindex(stacked.index)
            y_lag1 = y_aligned.shift(1)
            stacked = pd.concat([y_lag1.rename("y_lag1").to_frame(), stacked], axis=1)
        # Align y to LF index
        if y_lf is not None:
            y_target = y_lf.reindex(stacked.index)
        else:
            # Cannot compute IC without target; skip all candidates
            continue
        # Drop NaN rows
        combined = pd.concat(
            [y_target.rename("__y__"), stacked], axis=1
        ).dropna()
        T_eff = len(combined)
        n_cols = stacked.shape[1]  # K_cand * N_predictors (+ 1 if y_lag1)
        K_params = n_cols + 1  # +1 for intercept
        if T_eff < K_params + 2:
            continue  # insufficient degrees of freedom
        y_arr = combined["__y__"].to_numpy()
        X_arr = combined.drop(columns="__y__").to_numpy()
        # Add intercept column
        ones = np.ones((T_eff, 1))
        X_arr_int = np.hstack([ones, X_arr])
        # OLS via lstsq (numerically stable via SVD)
        beta, residuals, rank, _ = np.linalg.lstsq(X_arr_int, y_arr, rcond=None)
        if len(residuals) == 0:
            # lstsq did not return residuals (underdetermined); compute manually
            y_hat = X_arr_int @ beta
            rss = float(np.sum((y_arr - y_hat) ** 2))
        else:
            rss = float(residuals[0])
        if not np.isfinite(rss) or rss < 0:
            continue
        if rss == 0:
            # Perfect fit: BIC = -inf; accept this K (simplest perfect-fit model wins)
            return K_cand
        if ic == "bic":
            ic_val = np.log(rss / T_eff) + K_params * np.log(T_eff) / T_eff
        else:  # aic
            ic_val = np.log(rss / T_eff) + 2.0 * K_params / T_eff
        any_valid = True
        if ic_val < best_ic_val:
            best_ic_val = ic_val
            best_K = K_cand

    if not any_valid:
        import warnings as _w
        _w.warn(
            f"_bic_select_k: no K_candidate had sufficient degrees of freedom "
            f"(K_max={K_max}, freq_ratio={freq_ratio}); falling back to K_star=1.",
            stacklevel=3,
        )

    return best_K


def _u_midas(
    frame: pd.DataFrame,
    *,
    freq_ratio: int,
    n_lags_high: "int | str",
    target_freq: str = "low",
    include_y_lag: bool = False,
    y_series: "pd.Series | None" = None,
    ic_selector: str = "bic",
) -> pd.DataFrame:
    """Foroni-Marcellino-Schumacher (2015) Unrestricted MIDAS lag-stack op.
    Bundesbank DP 35/2011; JRSS-A 178(1), 57-82, DOI 10.1111/rssa.12043.

    Implements the design matrix for paper §3.2 eq.(20):
        y_{t×k} = μ₀ + μ₁ y_{t×k-k} + ψ(L) x_{t×k-1} + ε_{t×k}

    Steps:
    1. Resolve K: if n_lags_high is "bic" or "aic", run BIC/AIC selection
       over K ∈ {1, ..., ceil(1.5 * freq_ratio)} (paper §3.2 p.11 + §3.5).
    2. Build lag-stacked design via _midas_lag_stack (paper §2.1 eq.(8)).
    3. If include_y_lag=True, prepend y_lag1 column (paper eq.(20) μ₁ term).

    The L4 OLS fit (paper §3.2 p.11) is downstream and not part of this op.

    Parameters
    ----------
    frame : pd.DataFrame
        HF predictor panel.
    freq_ratio : int
        HF periods per LF period (e.g., 3 for monthly/quarterly).
    n_lags_high : int or str
        Number of HF lags. "bic"/"aic" triggers information-criterion
        selection; an integer fixes K directly.
    target_freq : str
        "low" (default) or "high". BIC only meaningful for "low".
    include_y_lag : bool, default False
        Include AR(1) y-lag term μ₁ y_{t×k-k} per paper §3.2 eq.(20).
        Defaults False at the primitive level; the u_midas() helper sets
        True by default.
    y_series : pd.Series or None
        LF target series (required when include_y_lag=True).
    ic_selector : str
        "bic" or "aic" (only used when n_lags_high is a string).
    """

    m = max(1, int(freq_ratio))

    # Step 1: Resolve K
    if isinstance(n_lags_high, str):
        if target_freq == "high":
            # BIC not meaningful for high-freq path (spec §Out of Scope item 5)
            import math
            K = max(1, math.ceil(1.5 * m))
        else:
            K = _bic_select_k(
                frame,
                y_series,
                freq_ratio=m,
                include_y_lag=include_y_lag,
                ic=ic_selector if n_lags_high == ic_selector else n_lags_high,
            )
    else:
        K = max(1, int(n_lags_high))

    # Step 2: Build lag-stacked design matrix
    stacked = _midas_lag_stack(
        frame, freq_ratio=m, n_lags_high=K, target_freq=target_freq
    )

    # Step 3: Optionally prepend y_lag1 column (paper eq.(20) μ₁ term)
    if include_y_lag and y_series is not None:
        y_lf = y_series.reindex(stacked.index)  # align to LF index
        y_lag1 = y_lf.shift(1)                  # 1 LF period lag
        y_lag1.name = "y_lag1"
        stacked = pd.concat([y_lag1.to_frame(), stacked], axis=1)

    return stacked


def _midas(
    frame_hf: pd.DataFrame,
    *,
    target: pd.Series | None,
    weighting: str = "exp_almon",
    polynomial_order: int = 2,
    freq_ratio: int = 3,
    n_lags_high: int = 12,
    sum_to_one: bool = True,
    max_iter: int = 200,
) -> pd.DataFrame:
    """Ghysels-Sinko-Valkanov (2007) MIDAS with parametric weighted lag polynomial.

    Three weighting schemes via ``weighting`` parameter:
    * ``almon``: polynomial w_k(θ) = Σ_q θ_q · k^q, optionally normalised
      to sum to one.
    * ``exp_almon``: w_k(θ) = exp(θ₀·k + θ₁·k²) / Σ_j exp(θ₀·j + θ₁·j²).
    * ``beta``: w_k(θ) ∝ x_k^(θ₀-1) · (1 - x_k)^(θ₁-1), x_k = (k+1)/(K+1).

    NLS fit via :func:`scipy.optimize.minimize` (method ``Nelder-Mead``;
    ``trust-constr`` was the spec recommendation but Nelder-Mead is
    derivative-free and robust on small samples).
    """

    K = max(1, int(n_lags_high))
    m = max(1, int(freq_ratio))
    poly_q = max(0, int(polynomial_order))
    stacked = _midas_lag_stack(frame_hf, freq_ratio=m, n_lags_high=K, target_freq="low")
    if stacked.empty or target is None:
        # Without a target we cannot fit weights; fall back to a uniform
        # average (equal weights = unweighted MA aggregation).
        out: dict[str, np.ndarray] = {}
        for col in frame_hf.columns:
            cols = [
                f"{col}_lag{k}" for k in range(K) if f"{col}_lag{k}" in stacked.columns
            ]
            if cols:
                out[col] = stacked[cols].mean(axis=1, skipna=True).to_numpy()
            else:
                out[col] = np.full(len(stacked.index), np.nan)
        return pd.DataFrame(out, index=stacked.index)

    from scipy.optimize import minimize  # type: ignore

    target_aligned = pd.Series(target).astype(float)

    def w_almon(theta: np.ndarray) -> np.ndarray:
        """Almon polynomial MIDAS weights.

        Computes w_k = sum_q theta[q] * k^q for k = 0..K-1, then:
        1. Clamps all weights to >= 0 (non-negativity for economic decay).
        2. If all clamped weights are zero, returns uniform 1/K (fallback).
        3. If sum_to_one=True, normalises the clamped weights by their sum.
        """
        kk = np.arange(K, dtype=float)
        w_raw = np.zeros(K, dtype=float)
        for q in range(poly_q + 1):
            w_raw = w_raw + theta[q] * (kk**q)
        # Non-negativity clamp: enforce economic constraint before normalisation.
        w_raw = np.maximum(w_raw, 0.0)
        if float(np.sum(w_raw)) == 0.0:
            # All weights clamped to zero: fall back to uniform.
            return np.full(K, 1.0 / K, dtype=float)
        if sum_to_one:
            denom = float(np.sum(w_raw))
            if abs(denom) > 1e-12:
                w_raw = w_raw / denom
        return w_raw

    def w_exp_almon(theta: np.ndarray) -> np.ndarray:
        kk = np.arange(K, dtype=float)
        z = float(theta[0]) * kk + float(theta[1]) * (kk**2)
        z = z - float(np.max(z))  # numerical-stability shift
        e = np.exp(z)
        s = float(np.sum(e))
        if s <= 0 or not np.isfinite(s):
            return np.full(K, 1.0 / K, dtype=float)
        return e / s

    def w_beta(theta: np.ndarray) -> np.ndarray:
        a, b = float(theta[0]), float(theta[1])
        a = max(a, 1e-3)
        b = max(b, 1e-3)
        kk = (np.arange(K, dtype=float) + 1.0) / (K + 1.0)
        w_raw = (kk ** (a - 1.0)) * ((1.0 - kk) ** (b - 1.0))
        s = float(np.sum(w_raw))
        if s <= 0 or not np.isfinite(s):
            return np.full(K, 1.0 / K, dtype=float)
        return w_raw / s

    if weighting == "almon":
        w_fn = w_almon
        theta0 = np.zeros(poly_q + 1, dtype=float)
        theta0[0] = 1.0
    elif weighting == "beta":
        w_fn = w_beta
        theta0 = np.array([1.0, 1.0], dtype=float)
    else:  # exp_almon (default)
        w_fn = w_exp_almon
        theta0 = np.array([0.0, 0.0], dtype=float)

    n_theta = theta0.size
    fit_info: dict[str, dict[str, Any]] = {}

    out: dict[str, np.ndarray] = {}
    for col in frame_hf.columns:
        lag_cols = [
            f"{col}_lag{k}" for k in range(K) if f"{col}_lag{k}" in stacked.columns
        ]
        if not lag_cols:
            out[col] = np.full(len(stacked.index), np.nan)
            continue
        Xk_full = stacked[lag_cols]
        # Align with target on common index (drop rows with any NaN).
        common_index = Xk_full.dropna(how="any").index
        y_aligned = target_aligned.reindex(common_index).dropna()
        common = Xk_full.index.intersection(y_aligned.index)
        if len(common) < n_theta + 2:
            # Insufficient training rows: fall back to equal weights.
            out[col] = Xk_full.mean(axis=1, skipna=True).to_numpy()
            continue
        Xk = Xk_full.loc[common].to_numpy(dtype=float)
        y = y_aligned.loc[common].to_numpy(dtype=float)

        def loss(params: np.ndarray) -> float:
            theta = params[:-2]
            alpha = float(params[-2])
            beta = float(params[-1])
            try:
                weights = w_fn(theta)
            except Exception:
                return 1e12
            agg = Xk @ weights
            resid = y - alpha - beta * agg
            return float(np.sum(resid * resid))

        x0 = np.concatenate([theta0, np.array([float(np.mean(y)), 1.0])])
        try:
            result = minimize(
                loss,
                x0=x0,
                method="Nelder-Mead",
                options={"maxiter": int(max_iter), "xatol": 1e-6, "fatol": 1e-8},
            )
            theta_hat = np.asarray(result.x[:-2], dtype=float)
            converged = bool(result.success)
        except Exception:  # pragma: no cover
            theta_hat = theta0.copy()
            converged = False
        try:
            weights_hat = w_fn(theta_hat)
        except Exception:  # pragma: no cover
            weights_hat = np.full(K, 1.0 / K)
        agg_full = Xk_full.to_numpy(dtype=float) @ weights_hat
        out[col] = agg_full
        fit_info[col] = {
            "theta_hat": theta_hat.tolist(),
            "weights": weights_hat.tolist(),
            "weighting": weighting,
            "converged": converged,
        }

    out_frame = pd.DataFrame(out, index=stacked.index)
    out_frame.attrs["midas_fit"] = fit_info
    return out_frame


# ---------------------------------------------------------------------------
# Phase C M3 -- Sliced Inverse Regression (Fan-Xue-Yao 2017) +
# Huang-Zhou (2022) sSUFF predictive scaling.
# ---------------------------------------------------------------------------


def _univariate_slope(x: pd.Series, y: pd.Series) -> float:
    """OLS slope of ``y`` on ``x`` (centered). Used by ``scaled_pca`` and
    ``sliced_inverse_regression`` (sSUFF variant).
    """

    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 2:
        return 0.0
    xc = x.loc[common].astype(float).to_numpy()
    yc = y.loc[common].astype(float).to_numpy()
    xc_dm = xc - float(np.mean(xc))
    yc_dm = yc - float(np.mean(yc))
    denom = float(np.dot(xc_dm, xc_dm))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc_dm, yc_dm) / denom)


def _sliced_inverse_regression(
    frame: pd.DataFrame,
    *,
    target: pd.Series | None,
    n_components: int,
    n_slices: int,
    scaling_method: str = "scaled_pca",
) -> pd.DataFrame:
    """Fan-Xue-Yao (2017) sliced inverse regression for factor models.

    Optional Huang-Zhou (2022) predictive scaling (``scaling_method=
    'scaled_pca'``) applies a univariate target-supervised slope per
    column before slicing -- the ``sSUFF`` variant.
    """

    K = max(1, int(n_components))
    H = max(2, int(n_slices))
    if target is None or frame.empty:
        return pd.DataFrame(
            np.zeros((len(frame), K)),
            index=frame.index,
            columns=[f"factor_{i + 1}" for i in range(K)],
        )
    common_idx = frame.index.intersection(target.index)
    X = frame.loc[common_idx].dropna(axis=0, how="any")
    y = pd.Series(target).reindex(X.index).dropna()
    X = X.loc[y.index]
    if X.empty or X.shape[1] == 0:
        return pd.DataFrame(
            np.zeros((len(frame), K)),
            index=frame.index,
            columns=[f"factor_{i + 1}" for i in range(K)],
        )
    K_eff = min(K, X.shape[1])

    # 1. Standardise X column-wise.
    mu_X = X.mean(axis=0)
    sd_X = X.std(axis=0, ddof=1).replace(0.0, 1.0)
    Xs = (X - mu_X) / sd_X

    # 2. Optional column-wise predictive scaling (sSUFF).
    beta_j: np.ndarray | None = None
    if scaling_method == "scaled_pca":
        beta_j = np.array(
            [_univariate_slope(Xs[c], y) for c in Xs.columns], dtype=float
        )
        Xs_scaled = Xs * beta_j
    elif scaling_method == "marginal_R2":
        beta_j = np.array(
            [_univariate_slope(Xs[c], y) for c in Xs.columns], dtype=float
        )
        # Marginal R² ≈ slope² · Var(x) / Var(y); under standardised x and
        # standardised y this collapses to slope². We use sign(beta) ·
        # |slope| as a robust proxy.
        signs = np.sign(beta_j)
        Xs_scaled = Xs * (signs * np.abs(beta_j))
    else:  # "none"
        Xs_scaled = Xs

    # 3. Sort by y, partition into H slices.
    Z = Xs_scaled.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    sort_idx = np.argsort(y_arr)
    Z_sorted = Z[sort_idx]
    n_total = Z_sorted.shape[0]
    if n_total < H:
        H = max(2, n_total)
    slice_size = n_total // H
    slice_means: list[np.ndarray] = []
    slice_weights: list[float] = []
    for h in range(H):
        start = h * slice_size
        end = (h + 1) * slice_size if h < H - 1 else n_total
        slc = Z_sorted[start:end]
        if slc.size == 0:
            slice_means.append(np.zeros(Z_sorted.shape[1]))
            slice_weights.append(0.0)
            continue
        slice_means.append(slc.mean(axis=0))
        slice_weights.append(slc.shape[0] / max(n_total, 1))
    M = np.vstack(slice_means)
    weights = np.asarray(slice_weights, dtype=float)

    # 5. Between-slice covariance Σ_S = M^⊤ · diag(n_h/n) · M.
    Sigma_S = (M * weights[:, None]).T @ M

    # 6. Eigendecomposition: top-K eigenvectors.
    try:
        vals, vecs = np.linalg.eigh(Sigma_S)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate
        vals = np.zeros(Sigma_S.shape[0])
        vecs = np.eye(Sigma_S.shape[0])
    order = np.argsort(-np.abs(vals))[:K_eff]
    V_K = vecs[:, order]

    # 7. Sign convention: enforce max-magnitude loading positive.
    for k in range(V_K.shape[1]):
        max_idx = int(np.argmax(np.abs(V_K[:, k])))
        if V_K[max_idx, k] < 0:
            V_K[:, k] = -V_K[:, k]

    # 8. Project all rows (re-using full standardised+scaled X for non-train rows).
    Xs_full = ((frame - mu_X) / sd_X).fillna(0.0)
    if scaling_method == "scaled_pca" and beta_j is not None:
        Xs_full = Xs_full * beta_j
    elif scaling_method == "marginal_R2" and beta_j is not None:
        signs = np.sign(beta_j)
        Xs_full = Xs_full * (signs * np.abs(beta_j))
    factors_arr = Xs_full.to_numpy(dtype=float) @ V_K
    if factors_arr.shape[1] < K:
        # Pad to requested K with zeros so downstream sees stable shape.
        pad = np.zeros((factors_arr.shape[0], K - factors_arr.shape[1]))
        factors_arr = np.hstack([factors_arr, pad])
    return pd.DataFrame(
        factors_arr,
        index=frame.index,
        columns=[f"factor_{i + 1}" for i in range(K)],
    )


def _first_series(inputs: list[Any]) -> pd.Series | None:
    for item in inputs[1:]:
        if isinstance(item, pd.Series):
            return item
        if isinstance(item, pd.DataFrame) and len(item.columns) == 1:
            return item.iloc[:, 0]
    return None


def _holiday_indicator(frame: pd.DataFrame) -> pd.DataFrame:
    """US federal-holiday indicator (sparse) over the input frame index."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame({"is_holiday": [0] * len(frame)}, index=frame.index)
    holidays = pd.tseries.offsets.USFederalHolidayCalendar().holidays(
        start=frame.index.min(), end=frame.index.max()
    )
    flags = frame.index.isin(holidays).astype(float)
    return pd.DataFrame({"is_holiday": flags}, index=frame.index)


def _regime_indicator(value: Any) -> pd.DataFrame:
    """Convert an L1 regime metadata artifact (or pandas Series) into 0/1 indicators per regime label."""

    if isinstance(value, pd.Series):
        series = value
    elif hasattr(value, "regime_series") and isinstance(
        getattr(value, "regime_series", None), pd.Series
    ):
        series = value.regime_series
    elif hasattr(value, "data") and isinstance(getattr(value, "data", None), pd.Series):
        series = value.data
    else:
        raise ValueError(
            "regime_indicator requires an L1 regime artifact or pandas Series input"
        )
    dummies = pd.get_dummies(series.astype(str), prefix="regime", dtype=float)
    return dummies


def _pca_factors(
    frame: pd.DataFrame,
    *,
    n_components: int | str,
    variant: str = "pca",
    target_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """In-sample PCA factor extraction.

    * ``variant='pca'`` -- standard PCA on the centred data matrix.
    * ``variant='sparse_pca'`` -- sklearn ``SparsePCA``
      (Zou-Hastie-Tibshirani 2006 via Mairal et al. 2009 dictionary
      learning); generic sparse-PCA primitive. NOTE: differs from
      Chen-Rohe (2023) SCA used in Rapach-Zhou (2025) Sparse Macro-
      Finance Factors paper -- planned as ``sparse_pca_chen_rohe`` in
      v0.9.x.
    * ``variant='scaled_pca'`` -- Huang/Jiang/Li/Tong/Zhou (2022)
      Scaled PCA (Management Science 68(3)): standardise X column-wise,
      compute per-column predictive slope ``β_j`` from a univariate
      OLS regression of target on the j-th standardised column, scale
      each column j by its signed ``β_j``, then run PCA on the scaled
      matrix. The signed column scaling means columns that strongly
      predict the target dominate the principal directions.
      v0.8.9 honesty-pass fix: prior implementations used a row-wise
      ``|target|`` weighting which is a different algorithm;
      ``_scaled_pca_huang_zhou`` below now matches the authors' MATLAB
      code (sPCAest.m).
    """

    from sklearn.decomposition import PCA, SparsePCA

    cleaned = frame.dropna(axis=0, how="any")
    if cleaned.empty:
        return pd.DataFrame(index=frame.index)
    # Phase A3 fix: ``n_components="all"`` sentinel resolves to the
    # effective rank ``min(T, N)`` of the cleaned input. We bypass the
    # historical ``-1`` safety margin (which capped explicit ints at
    # ``min(T, N) - 1``); the "all" semantics per Coulombe et al. (2022)
    # §3.2 paper §3.2 Eq. (18) is "we keep them all", so we let PCA keep
    # the full effective rank min(T, N).
    if isinstance(n_components, str) and n_components == "all":
        n_components = max(1, min(cleaned.shape))
    else:
        n_components = max(1, min(int(n_components), min(cleaned.shape) - 1))

    if variant == "scaled_pca":
        if target_signal is None:
            raise ValueError(
                "scaled_pca requires target_signal (per Huang/Zhou 2022 sPCAest)"
            )
        scores = _scaled_pca_huang_zhou(
            cleaned, n_components=n_components, target_signal=target_signal
        )
    elif variant == "sparse_pca":
        matrix = cleaned.to_numpy() - cleaned.to_numpy().mean(axis=0)
        model = SparsePCA(n_components=n_components, random_state=0)
        scores = model.fit_transform(matrix)
    else:
        matrix = cleaned.to_numpy() - cleaned.to_numpy().mean(axis=0)
        model = PCA(n_components=n_components, random_state=0)
        scores = model.fit_transform(matrix)
    factors = pd.DataFrame(
        scores,
        index=cleaned.index,
        columns=[f"factor_{idx + 1}" for idx in range(scores.shape[1])],
    )
    return factors.reindex(frame.index)


def _scaled_pca_huang_zhou(
    frame: pd.DataFrame,
    *,
    n_components: int,
    target_signal: pd.Series,
) -> np.ndarray:
    """Huang/Jiang/Li/Tong/Zhou (2022) Scaled PCA -- paper-faithful
    implementation per the authors' MATLAB ``sPCAest.m`` code.

    Algorithm (step-by-step from sPCAest.m):

        1. Standardise X column-wise: ``Xs = (X - mean(X)) / std(X)``
           with ddof=1 (sample std, matching MATLAB's std default).
        2. For each column j of Xs, fit a univariate OLS of target on
           [1, Xs[:, j]] and take the slope coefficient as ``β_j``.
        3. Scale each column j of Xs by ``β_j`` (signed; columns that
           positively predict target get positive weight, negatively
           predictive columns get negative weight, irrelevant columns
           get near-zero weight).
        4. Run PCA on the scaled matrix to extract ``n_components``
           factors.

    The authors' ``pc_T`` factor normalisation (``F'F/T = I``, factors
    have unit-T variance per column) differs from sklearn ``PCA``
    (which scales factors by the singular values). The factor
    *directions* are equivalent up to sign and a per-column scalar,
    so downstream regression (where factors feed into ridge / OLS)
    produces the same predictions modulo coefficient rescaling.
    Documented for transparency.

    Reference:
        Huang D., Jiang F., Li K., Tong G., Zhou G. (2022)
        "Scaled PCA: A New Approach to Dimension Reduction",
        Management Science 68(3), 1591-2376.
        Authors' MATLAB code: sPCAest.m.
    """

    from sklearn.decomposition import PCA

    # Align target with frame index; only keep rows where both are non-NaN.
    target_aligned = target_signal.reindex(frame.index)
    valid = target_aligned.notna() & frame.notna().all(axis=1)
    if not valid.any():
        return np.zeros((len(frame), n_components))
    X_in = frame.loc[valid].to_numpy(dtype=float)
    y_in = target_aligned.loc[valid].to_numpy(dtype=float)

    # Step 1: standardise columns (ddof=1 to match MATLAB std default).
    col_means = X_in.mean(axis=0)
    col_stds = X_in.std(axis=0, ddof=1)
    col_stds = np.where(col_stds > 1e-12, col_stds, 1.0)
    Xs = (X_in - col_means) / col_stds

    # Step 2: per-column predictive slope via univariate OLS
    # (target_t = α_j + β_j · Xs[t, j] + ε_t)
    # Closed-form: β_j = Cov(Xs[:, j], target) / Var(Xs[:, j])
    # Since Xs is standardised, Var(Xs[:, j]) ≈ 1 (modulo ddof), so
    # β_j ≈ Cov(Xs[:, j], target) = mean(Xs[:, j] · (target − target_mean)).
    y_centred = y_in - y_in.mean()
    denom = (Xs**2).sum(axis=0)  # since Xs is mean-zero standardised
    denom = np.where(denom > 1e-12, denom, 1.0)
    beta = (Xs * y_centred[:, None]).sum(axis=0) / denom

    # Step 3: column-wise signed scaling.
    scaleXs = Xs * beta[None, :]

    # Step 4: PCA on scaled matrix.
    pca = PCA(n_components=int(n_components), random_state=0)
    pca.fit(scaleXs)

    # Project the full frame (including any rows where target was NaN)
    # back through the same standardisation + scaling + PCA so the
    # output covers the input index. For rows missing target, the
    # standardisation of X is still well-defined, but the "scaled"
    # matrix uses the same per-column β learned from the valid subset.
    X_full = frame.to_numpy(dtype=float)
    X_full_filled = np.where(np.isfinite(X_full), X_full, col_means[None, :])
    Xs_full = (X_full_filled - col_means) / col_stds
    scaleXs_full = Xs_full * beta[None, :]
    scores_full = pca.transform(scaleXs_full)
    return scores_full


def _varimax_rotation(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.dropna(axis=0, how="any")
    if cleaned.empty:
        return frame
    matrix = cleaned.to_numpy(dtype=float)
    n_features = matrix.shape[1]
    rotation = np.eye(n_features)
    for _ in range(50):
        u, _, vh = np.linalg.svd(
            matrix.T
            @ (matrix**3 - matrix * (np.diag(matrix.T @ matrix) / matrix.shape[0]))
        )
        rotation = u @ vh
        matrix = matrix @ rotation
    rotated = pd.DataFrame(
        matrix,
        index=cleaned.index,
        columns=[f"varimax_{i + 1}" for i in range(n_features)],
    )
    return rotated.reindex(frame.index)


def _sparse_pca_chen_rohe(
    frame: pd.DataFrame,
    *,
    n_components: int,
    zeta: float = 0.0,
    max_iter: int = 200,
    var_innovations: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:
    """Chen-Rohe (2023) Sparse Component Analysis (SCA) -- non-diagonal D
    variant used by Rapach & Zhou (2025) Sparse Macro-Finance Factors.

    Solves the equivalent bilinear convex-hull form (Zhou-Rapach Eq. 4):

        ``max ‖Z' X Θ‖_F  s.t.  Z ∈ H(T,J),  Θ ∈ H(M,J),  ‖Θ‖_1 ≤ ζ``

    where ``H(n,r) = {V : V'V ⪯ I_r}``. Alternates SVD-projection of Z
    (fixing Θ) and L1-budget projection of Θ (fixing Z).

    ``zeta = 0.0`` (default) routes to ``zeta = J = n_components`` --
    the most-binding boundary the paper finds optimal in cross-
    validation (§2.3, "with respect to ζ").

    ``var_innovations=True`` activates Rapach & Zhou (2025) Strategy
    step 2 (paper §2.1): fit a *first-order vector autoregression*
    ``S_t = B S_{t-1} + e_t`` on the J × T panel of SCA scores via
    closed-form OLS ``B̂ = (S_lag' S_lag)^{-1} S_lag' S_now`` and
    return the fitted residuals ``ê_t = S_t − S_{t-1} B̂`` as the
    *sparse macro-finance factors* of the paper title (paper §2.3
    factor dynamics). Cross-equation lag effects are retained — this
    is a true VAR(1), not J independent AR(1)s. Default False returns
    the SCA scores themselves (sparse principal components, paper
    Strategy step 1).
    """

    cleaned = frame.dropna(how="any")
    if cleaned.empty or cleaned.shape[1] == 0:
        return pd.DataFrame(
            index=frame.index, columns=[f"sca_{i + 1}" for i in range(n_components)]
        )
    X = cleaned.to_numpy(dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    T, M = X.shape
    J = max(1, min(n_components, M, T))
    zeta_val = float(zeta) if zeta > 0 else float(J)
    rng = np.random.default_rng(random_state)

    # Initialise Z, Θ on the Stiefel manifold via QR of random matrices.
    Z = np.linalg.qr(rng.standard_normal((T, J)))[0]
    Theta = np.linalg.qr(rng.standard_normal((M, J)))[0]
    prev_obj = -np.inf
    for _ in range(max(1, max_iter)):
        # Update Z: max ‖Z' X Θ‖_F ⇒ Z = U V' from SVD(X Θ).
        U, _, Vt = np.linalg.svd(X @ Theta, full_matrices=False)
        Z = U @ Vt
        # Update Θ: SVD-orthonormalise X' Z, then soft-threshold to
        # honour the L1 budget. Bisection over the threshold τ ≥ 0.
        G = X.T @ Z  # M × J
        Uo, _, Vto = np.linalg.svd(G, full_matrices=False)
        Theta_unconstrained = Uo @ Vto
        # Soft-threshold rows of Theta_unconstrained until ‖Θ‖_1 <= ζ.
        if np.sum(np.abs(Theta_unconstrained)) <= zeta_val:
            Theta = Theta_unconstrained
        else:
            # Bisection on threshold τ. `hi` is the smallest τ observed
            # satisfying ‖Θ_st‖_1 ≤ ζ; using `hi` (not the final mid-point)
            # guarantees the budget. Paper Eq. (4) accepts Θ ∈ H(M,J), the
            # sub-orthonormal hull `Θ'Θ ⪯ I` — re-orthonormalising after
            # soft-thresholding would break ‖Θ‖_1 ≤ ζ, so we keep Θ_st.
            lo, hi = 0.0, float(np.max(np.abs(Theta_unconstrained)))
            for _ in range(50):
                tau = 0.5 * (lo + hi)
                Theta_st = np.sign(Theta_unconstrained) * np.maximum(
                    np.abs(Theta_unconstrained) - tau, 0.0
                )
                if np.sum(np.abs(Theta_st)) > zeta_val:
                    lo = tau
                else:
                    hi = tau
            Theta = np.sign(Theta_unconstrained) * np.maximum(
                np.abs(Theta_unconstrained) - hi, 0.0
            )
        obj = float(np.linalg.norm(Z.T @ X @ Theta, "fro"))
        if abs(obj - prev_obj) < 1e-9:
            break
        prev_obj = obj

    scores = X @ Theta  # (T, J)

    if var_innovations and scores.shape[0] > 2:
        # Rapach & Zhou (2025) Strategy step 2: fit a VAR(1) on the SCA
        # scores S_t = B S_{t-1} + e_t; return the fitted residuals as
        # the sparse macro-finance factors. Closed-form OLS on stacked
        # equations (statsmodels-free to avoid the dependency on this
        # hot path):
        #
        #     B̂ = (S_lag' S_lag)^{-1} (S_lag' S_now)        # J × J
        #     e_t = S_t − S_{t-1} B̂                         # row form
        #
        # NOTE: row form has S_lag rows post-multiplied by B̂, so the
        # solve target ``S_lag' S_now`` already gives B̂ in the
        # orientation expected by the residual formula. Cross-equation
        # lag effects are preserved (the previous per-column AR(1)
        # silently dropped them).
        S = scores
        S_lag = S[:-1]  # (T-1, J)
        S_now = S[1:]  # (T-1, J)
        gram = S_lag.T @ S_lag  # J × J
        rhs = S_lag.T @ S_now  # J × J
        try:
            B_hat = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            # Singular gram (rare: collinear or near-zero scores) —
            # fall back to least-squares minimum-norm solution.
            B_hat = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        innov = np.full_like(S, np.nan, dtype=float)
        innov[1:] = S_now - S_lag @ B_hat
        # First row has no lag → zero-fill so the output frame keeps
        # the original index length.
        innov[0] = 0.0
        scores = innov
        col_prefix = "scaf"  # sparse macro-finance factors
    else:
        col_prefix = "sca"

    out = pd.DataFrame(
        scores, index=cleaned.index, columns=[f"{col_prefix}_{i + 1}" for i in range(J)]
    )
    return out.reindex(frame.index)


def _supervised_pca(
    frame: pd.DataFrame,
    *,
    target: pd.Series | None,
    n_components: int,
    q: float = 0.5,
) -> pd.DataFrame:
    """Giglio-Xiu-Zhang (2025) Supervised PCA: screen panel columns by
    univariate correlation with the target, retain the top ``q · M``,
    run PCA on the screened sub-panel.

    Distinct from ``partial_least_squares`` (NIPALS over all columns)
    and ``scaled_pca`` (Huang-Jiang-Tu-Zhou 2022 column β-scaling on
    every column). SPCA hard-screens before factor extraction.
    """

    if target is None:
        raise ValueError("supervised_pca requires a target_signal input")
    aligned = pd.concat([frame, target.rename("__target__")], axis=1).dropna(how="any")
    if aligned.empty:
        return pd.DataFrame(
            index=frame.index, columns=[f"spca_{i + 1}" for i in range(n_components)]
        )
    g = aligned["__target__"].astype(float).to_numpy()
    R = aligned.drop(columns=["__target__"]).to_numpy(dtype=float)
    g_c = g - g.mean()
    R_c = R - R.mean(axis=0, keepdims=True)
    # Univariate correlations (Pearson).
    norm_R = np.linalg.norm(R_c, axis=0)
    norm_g = float(np.linalg.norm(g_c))
    corr = (R_c.T @ g_c) / (norm_R * norm_g + 1e-12)
    M = R.shape[1]
    n_keep = max(1, int(np.floor(np.clip(q, 1e-6, 1.0) * M)))
    keep = np.argsort(-np.abs(corr))[:n_keep]
    Rs = R_c[:, keep]
    # PCA on screened sub-panel via SVD.
    P = max(1, min(n_components, n_keep, Rs.shape[0]))
    U, S, Vt = np.linalg.svd(Rs, full_matrices=False)
    factors = U[:, :P] * S[:P]
    out = pd.DataFrame(
        factors, index=aligned.index, columns=[f"spca_{i + 1}" for i in range(P)]
    )
    return out.reindex(frame.index)


def _partial_least_squares(
    frame: pd.DataFrame, *, target: pd.Series | None, n_components: int
) -> pd.DataFrame:
    if target is None:
        raise ValueError("partial_least_squares requires a target Series input")
    from sklearn.cross_decomposition import PLSRegression

    aligned = pd.concat([frame, target.rename("__target__")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame(index=frame.index)
    n_components = max(
        1, min(int(n_components), min(aligned.shape[0] - 1, aligned.shape[1] - 1))
    )
    pls = PLSRegression(n_components=n_components)
    pls.fit(aligned.iloc[:, :-1], aligned.iloc[:, -1])
    scores = pls.transform(aligned.iloc[:, :-1])
    factors = pd.DataFrame(
        scores,
        index=aligned.index,
        columns=[f"pls_{idx + 1}" for idx in range(scores.shape[1])],
    )
    return factors.reindex(frame.index)


def _random_projection(frame: pd.DataFrame, *, n_components: int) -> pd.DataFrame:
    from sklearn.random_projection import GaussianRandomProjection

    cleaned = frame.dropna(axis=0, how="any")
    if cleaned.empty:
        return pd.DataFrame(index=frame.index)
    n_components = max(1, min(int(n_components), cleaned.shape[1]))
    proj = GaussianRandomProjection(n_components=n_components, random_state=0)
    scores = proj.fit_transform(cleaned.to_numpy())
    rp = pd.DataFrame(
        scores,
        index=cleaned.index,
        columns=[f"rp_{idx + 1}" for idx in range(scores.shape[1])],
    )
    return rp.reindex(frame.index)


def _dfm_factors(frame: pd.DataFrame, *, n_factors: int) -> pd.DataFrame:
    """Static dynamic factor approximation: PCA on standardized panel."""

    cleaned = frame.dropna(axis=0, how="any")
    if cleaned.empty:
        return pd.DataFrame(index=frame.index)
    standardized = (cleaned - cleaned.mean()) / cleaned.std(ddof=0).replace(0, np.nan)
    standardized = standardized.fillna(0.0)
    factors = _pca_factors(standardized, n_components=n_factors)
    factors.columns = [f"dfm_{idx + 1}" for idx in range(len(factors.columns))]
    return factors


def _wavelet_decomposition(frame: pd.DataFrame, *, n_levels: int) -> pd.DataFrame:
    """Daubechies-style multi-resolution decomposition via cumulative low-pass differences.

    Approximation: at each level, low-pass = rolling mean over 2**level window;
    detail = original minus low-pass at that level.
    """

    pieces: list[pd.DataFrame] = []
    for level in range(1, max(1, int(n_levels)) + 1):
        window = 2**level
        approx = (
            frame.rolling(window=window, min_periods=window)
            .mean()
            .add_suffix(f"_wA{level}")
        )
        detail = (
            frame - frame.rolling(window=window, min_periods=window).mean()
        ).add_suffix(f"_wD{level}")
        pieces.extend([approx, detail])
    return pd.concat(pieces, axis=1)


def _fourier_features(
    frame: pd.DataFrame, *, n_terms: int, period: int
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        positions = np.arange(len(frame))
    else:
        positions = (frame.index - frame.index[0]).days.values.astype(float)
    features = {}
    for k in range(1, max(1, int(n_terms)) + 1):
        features[f"fourier_sin_{k}"] = np.sin(2.0 * np.pi * k * positions / period)
        features[f"fourier_cos_{k}"] = np.cos(2.0 * np.pi * k * positions / period)
    return pd.DataFrame(features, index=frame.index)


def _hp_filter(frame: pd.DataFrame, *, lam: float) -> pd.DataFrame:
    from statsmodels.tsa.filters.hp_filter import hpfilter

    out = {}
    for column in frame.columns:
        series = frame[column].dropna()
        if len(series) < 4:
            out[f"{column}_hp_cycle"] = pd.Series(index=frame.index, dtype=float)
            continue
        cycle, _trend = hpfilter(series, lamb=lam)
        out[f"{column}_hp_cycle"] = cycle.reindex(frame.index)
    return pd.DataFrame(out, index=frame.index)


def _hamilton_filter(
    frame: pd.DataFrame, *, n_lags: int, n_horizon: int
) -> pd.DataFrame:
    """Hamilton (2018) regression-based filter: y_{t+h} on lagged values; residuals are the cycle."""

    from sklearn.linear_model import LinearRegression

    out: dict[str, pd.Series] = {}
    for column in frame.columns:
        series = frame[column].astype(float)
        lagged = pd.concat(
            [series.shift(n_horizon + lag) for lag in range(n_lags)], axis=1
        )
        lagged.columns = [f"lag{i}" for i in range(n_lags)]
        y = series.copy()
        aligned = pd.concat([lagged, y.rename("__y__")], axis=1).dropna()
        if aligned.empty or len(aligned) < n_lags + 2:
            out[f"{column}_hamilton_cycle"] = pd.Series(index=frame.index, dtype=float)
            continue
        reg = LinearRegression().fit(aligned.iloc[:, :-1], aligned.iloc[:, -1])
        predicted = pd.Series(reg.predict(aligned.iloc[:, :-1]), index=aligned.index)
        cycle = (aligned.iloc[:, -1] - predicted).rename(f"{column}_hamilton_cycle")
        out[f"{column}_hamilton_cycle"] = cycle.reindex(frame.index)
    return pd.DataFrame(out, index=frame.index)


def _asymmetric_trim(frame: pd.DataFrame, *, smooth_window: int = 0) -> pd.DataFrame:
    """Albacore-family rank-space transformation (Goulet Coulombe et al.
    2024 "Maximally Forward-Looking Core Inflation").

    For each row of ``frame`` (a (T x K) panel of contemporaneous
    component growth rates), sort the K values ascending and emit a
    matrix of the same shape where column ``rank_{r+1}`` is the r-th
    order statistic at each period. The downstream
    ``ridge(coefficient_constraint=nonneg)`` learns rank-weight vectors
    that produce *asymmetric* trimming -- the actual trimming pattern
    (e.g. drop the lower tail, emphasise the 60-75th percentile) is an
    emergent property of the supervised weights, not a preset bandit.

    Optional ``smooth_window > 0`` applies a centred moving average to
    each rank-position time series. Default 0 = no smoothing (paper
    notes 3-month MA in §3 implementation; users can chain ``ma_window``
    explicitly).
    """

    if frame.empty or frame.shape[1] == 0:
        return frame.copy()

    arr = frame.fillna(0.0).to_numpy(dtype=float)
    sorted_arr = np.sort(arr, axis=1)  # ascending per row
    cols = [f"rank_{r + 1}" for r in range(sorted_arr.shape[1])]
    out = pd.DataFrame(sorted_arr, index=frame.index, columns=cols)

    if smooth_window > 1:
        # Centred moving average on each rank-position series. Use
        # ``min_periods=1`` so edges still produce values.
        out = out.rolling(window=int(smooth_window), center=True, min_periods=1).mean()

    return out


def _savitzky_golay_filter(
    frame: pd.DataFrame, *, window_length: int, polyorder: int
) -> pd.DataFrame:
    """Savitzky-Golay (1964) polynomial smoothing filter.

    Wraps ``scipy.signal.savgol_filter`` column-wise. Used as the
    fixed-window baseline against which Coulombe & Klieber (2025) AlbaMA
    is compared (replication recipe in ``examples/recipes/`` once the
    AlbaMA primitive lands).
    """

    from scipy.signal import savgol_filter

    if window_length < 3:
        raise ValueError("savitzky_golay_filter: window_length must be >= 3")
    if polyorder >= window_length:
        raise ValueError("savitzky_golay_filter: polyorder must be < window_length")
    if window_length % 2 == 0:
        # scipy requires odd window; round up to next odd.
        window_length = window_length + 1

    out: dict[str, pd.Series] = {}
    for column in frame.columns:
        series = frame[column].astype(float)
        # Forward-fill NaNs so the filter has a contiguous signal; preserve
        # the original NaN mask on the output so callers can spot edges.
        filled = series.ffill().bfill().to_numpy()
        if len(filled) < window_length:
            out[f"{column}_savgol"] = pd.Series(filled, index=series.index)
            continue
        smoothed = savgol_filter(
            filled, window_length=window_length, polyorder=polyorder
        )
        out[f"{column}_savgol"] = pd.Series(smoothed, index=series.index)
    return pd.DataFrame(out, index=frame.index)


def _adaptive_ma_rf(
    frame: pd.DataFrame,
    *,
    n_estimators: int = 500,
    min_samples_leaf: int = 40,
    sided: str = "two",
    random_state: int = 0,
) -> pd.DataFrame:
    """AlbaMA -- Adaptive Moving Average via Random Forest with K=1
    (time index).

    Goulet Coulombe & Klieber (2025) "An Adaptive Moving Average for
    Macroeconomic Monitoring" (arXiv:2501.13222 §2). The estimator is a
    bagged-trees ensemble whose *sole regressor is the time index*: the
    CART splitting objective collapses to ``min_c [SSE(left) + SSE(right)]``
    over candidate cuts ``c``. Each prediction is a within-leaf average,
    so the forest prediction at time ``t`` is

        ``ŷ_t = (1/B) Σ_b T_b(t) = w_t · y``

    with leaf-membership-derived weights ``w_τt = (1/B) Σ_b w^b_τt``.
    The realised window length is *learned* per observation: rule-of-
    thumb ``min_samples_leaf = 40`` lower-bounds it. Paper recommends
    ``n_estimators = B = 500`` (paper p.8 line 351-352).

    Parameters
    ----------
    sided:
        ``"two"`` (default) fits the forest once on the full sample;
        each leaf may span past *and* future observations.
        ``"one"`` fits an *expanding-window* forest per time index t:
        for each t, the RF is fit on rows 0..t-1 and the leaf-mean at
        t is the prediction. Real-time / nowcasting variant per paper
        p.10 line 407.
    """

    from sklearn.ensemble import RandomForestRegressor

    if frame.empty:
        return frame.copy()
    sided = sided if sided in {"one", "two"} else "two"
    out: dict[str, pd.Series] = {}
    T = len(frame)
    t_index = np.arange(T, dtype=float).reshape(-1, 1)

    for column in frame.columns:
        y = frame[column].astype(float).to_numpy()
        if not np.any(np.isfinite(y)):
            out[f"{column}_albama"] = pd.Series(y, index=frame.index)
            continue
        smoothed = np.full(T, np.nan, dtype=float)
        if sided == "two":
            mask = np.isfinite(y)
            if mask.sum() < min_samples_leaf:
                # Insufficient data -- pass-through.
                smoothed[mask] = y[mask]
                out[f"{column}_albama"] = pd.Series(smoothed, index=frame.index)
                continue
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=1,
                bootstrap=True,
                random_state=random_state,
                n_jobs=1,
            )
            rf.fit(t_index[mask], y[mask])
            smoothed = rf.predict(t_index)
        else:
            # One-sided: per-t expanding window. The first
            # ``min_samples_leaf`` observations stay NaN (RF refuses to
            # fit smaller than the leaf bound).
            for end in range(min_samples_leaf, T + 1):
                mask = np.isfinite(y[:end])
                if mask.sum() < min_samples_leaf:
                    continue
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_features=1,
                    bootstrap=True,
                    random_state=random_state,
                    n_jobs=1,
                )
                rf.fit(t_index[:end][mask], y[:end][mask])
                smoothed[end - 1] = float(rf.predict(t_index[end - 1 : end])[0])
        out[f"{column}_albama"] = pd.Series(smoothed, index=frame.index)
    return pd.DataFrame(out, index=frame.index)


def _kernel_features(frame: pd.DataFrame, *, kind: str, gamma: float) -> pd.DataFrame:
    from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

    cleaned = frame.dropna(axis=0, how="any")
    if cleaned.empty:
        return pd.DataFrame(index=frame.index)
    matrix = cleaned.to_numpy()
    if kind == "polynomial":
        kernel = polynomial_kernel(matrix, degree=2, gamma=gamma)
    else:
        kernel = rbf_kernel(matrix, gamma=gamma)
    columns = [f"kernel_{i + 1}" for i in range(kernel.shape[1])]
    df = pd.DataFrame(kernel, index=cleaned.index, columns=columns)
    return df.reindex(frame.index)


def _nystroem_features(frame: pd.DataFrame, *, n_components: int) -> pd.DataFrame:
    from sklearn.kernel_approximation import Nystroem

    cleaned = frame.dropna(axis=0, how="any")
    if cleaned.empty:
        return pd.DataFrame(index=frame.index)
    n_components = max(1, min(int(n_components), cleaned.shape[0]))
    nys = Nystroem(n_components=n_components, random_state=0)
    scores = nys.fit_transform(cleaned.to_numpy())
    df = pd.DataFrame(
        scores,
        index=cleaned.index,
        columns=[f"nystroem_{idx + 1}" for idx in range(scores.shape[1])],
    )
    return df.reindex(frame.index)


def _feature_selection(
    frame: pd.DataFrame, *, target: pd.Series | None, n_features: Any, method: str
) -> pd.DataFrame:
    """Variance / correlation / lasso-based selection.

    `n_features` may be an integer (count) or a fraction (0 < f <= 1).
    """

    n_cols = frame.shape[1]
    if isinstance(n_features, float) and 0 < n_features <= 1:
        keep = max(1, int(n_features * n_cols))
    else:
        keep = max(1, min(int(n_features), n_cols))
    if method == "correlation" and target is not None:
        corr = frame.apply(lambda col: col.corr(target.reindex(col.index)))
        ordered = corr.abs().sort_values(ascending=False)
        return frame[list(ordered.index[:keep])]
    if method == "lasso" and target is not None:
        from sklearn.linear_model import LassoCV

        aligned = pd.concat([frame, target.rename("__y__")], axis=1).dropna()
        if aligned.empty:
            return frame.iloc[:, :keep]
        lasso = LassoCV(
            cv=min(5, max(2, len(aligned) // 4)), random_state=0, max_iter=20000
        )
        lasso.fit(aligned.iloc[:, :-1], aligned.iloc[:, -1])
        coefs = pd.Series(np.abs(lasso.coef_), index=frame.columns)
        ordered = coefs.sort_values(ascending=False)
        return frame[list(ordered.index[:keep])]
    variances = frame.var().sort_values(ascending=False)
    return frame[list(variances.index[:keep])]


def _hierarchical_pca(
    inputs: list[Any], *, n_components_per_block: int
) -> pd.DataFrame:
    blocks = []
    for block_index, item in enumerate(inputs):
        block_frame = _as_frame(item)
        block_factors = _pca_factors(block_frame, n_components=n_components_per_block)
        block_factors.columns = [
            f"hpca_block{block_index + 1}_f{i + 1}"
            for i in range(block_factors.shape[1])
        ]
        blocks.append(block_factors)
    return pd.concat(blocks, axis=1)


def _weighted_concat(inputs: list[Any], *, weights: Any) -> pd.DataFrame:
    frames = [_as_frame(item) for item in inputs]
    if weights is None:
        weights = [1.0] * len(frames)
    weights = list(weights)
    pieces = []
    for frame, w in zip(frames, weights):
        pieces.append(frame.multiply(float(w)))
    return pd.concat(pieces, axis=1)


def _simple_average(inputs: list[Any]) -> pd.DataFrame:
    frames = [_as_frame(item) for item in inputs]
    if not frames:
        return pd.DataFrame()
    aligned = pd.concat(
        [frame.add_suffix(f"_{i}") for i, frame in enumerate(frames)], axis=1
    )
    grouped: dict[str, list[pd.Series]] = {}
    for column in aligned.columns:
        base = column.rsplit("_", 1)[0]
        grouped.setdefault(base, []).append(aligned[column])
    averaged = {
        key: pd.concat(items, axis=1).mean(axis=1) for key, items in grouped.items()
    }
    return pd.DataFrame(averaged, index=frames[0].index)


def _cumulative_average_target(series: pd.Series, *, horizon: int) -> pd.Series:
    if horizon <= 0:
        return series.copy()
    rolled = series.rolling(window=horizon, min_periods=horizon).mean()
    return rolled.shift(-horizon)


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    raise TypeError(f"expected pandas DataFrame or Series, got {type(value).__name__}")


def _as_series(value: Any, *, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.rename(name)
    if isinstance(value, pd.DataFrame) and len(value.columns) == 1:
        return value.iloc[:, 0].rename(name)
    raise TypeError(f"expected single Series target, got {type(value).__name__}")


def _scale_frame(frame: pd.DataFrame, *, method: str) -> pd.DataFrame:
    """Standardise a Panel using one of the design-listed scale methods.

    Three operational methods (matches the L3 design table in
    ``plans/design/part2_l2_l3_l4.md`` § L3 Step library / ``scale``):

    * ``zscore`` / ``standard`` / ``standardize`` -- ``(x - mean) / std``
      (population std, ddof=0; matches sklearn ``StandardScaler``).
    * ``robust`` -- ``(x - median) / IQR`` where IQR is the 75th - 25th
      percentile gap (matches sklearn ``RobustScaler`` with default
      ``quantile_range=(25.0, 75.0)``).
    * ``minmax`` -- ``(x - min) / (max - min)`` over the column
      (matches sklearn ``MinMaxScaler`` with default ``feature_range=(0, 1)``).

    PR-E of the v0.1 honesty pass: ``robust`` and ``minmax`` were listed
    in the design as operational but only ``zscore`` was implemented;
    selecting either of the other two raised ``NotImplementedError``.
    """

    arr = frame.to_numpy(dtype=float)
    if method in {"zscore", "standard", "standardize"}:
        mean = frame.mean().to_numpy()
        std = frame.std(ddof=0).replace(0, pd.NA).to_numpy()
        scaled = (arr - mean) / std
    elif method == "robust":
        median = frame.median().to_numpy()
        iqr = (frame.quantile(0.75) - frame.quantile(0.25)).replace(0, pd.NA).to_numpy()
        scaled = (arr - median) / iqr
    elif method == "minmax":
        col_min = frame.min().to_numpy()
        col_max = frame.max().to_numpy()
        col_range = (
            pd.Series(col_max - col_min, index=frame.columns)
            .replace(0, pd.NA)
            .to_numpy()
        )
        scaled = (arr - col_min) / col_range
    else:
        raise NotImplementedError(
            f"L3 runtime does not support scale method {method!r}"
        )
    return pd.DataFrame(scaled, index=frame.index, columns=frame.columns)


def _map_like(value: pd.DataFrame | pd.Series, func) -> pd.DataFrame | pd.Series:
    if isinstance(value, pd.DataFrame):
        return value.map(func)
    if isinstance(value, pd.Series):
        return value.map(func)
    raise TypeError(f"expected pandas DataFrame or Series, got {type(value).__name__}")


def _diff_like(
    value: pd.DataFrame | pd.Series, *, periods: int
) -> pd.DataFrame | pd.Series:
    return value.diff(periods=periods)


def _pct_change_like(
    value: pd.DataFrame | pd.Series, *, periods: int
) -> pd.DataFrame | pd.Series:
    return value.pct_change(periods=periods)


def _minimal_train_size(params: dict[str, Any], *, n_obs: int, n_features: int) -> int:
    if n_obs < 3:
        raise ValueError("minimal L4 runtime requires at least 3 aligned observations")
    requested = params.get("min_train_size")
    if requested is not None:
        min_train_size = int(requested)
    else:
        min_train_size = min(n_obs - 1, max(2, min(n_features, n_obs - 1)))
    if min_train_size < 2:
        raise ValueError("minimal L4 runtime requires min_train_size >= 2")
    if min_train_size >= n_obs:
        raise ValueError(
            "minimal L4 runtime requires min_train_size < aligned observation count"
        )
    return min_train_size


def _lagged_predictors(
    frame: pd.DataFrame, n_lag: int, *, include_contemporaneous: bool = False
) -> pd.DataFrame:
    if n_lag < 1:
        raise ValueError("minimal L3 runtime requires n_lag >= 1")
    lagged = []
    first_lag = 0 if include_contemporaneous else 1
    for lag in range(first_lag, n_lag + 1):
        lagged.append(frame.shift(lag).add_suffix(f"_lag{lag}"))
    return pd.concat(lagged, axis=1)


def _seasonal_lagged_predictors(
    frame: pd.DataFrame, *, seasonal_period: int, n_seasonal_lags: int
) -> pd.DataFrame:
    if seasonal_period < 2:
        raise ValueError("minimal L3 runtime requires seasonal_period >= 2")
    if n_seasonal_lags < 1:
        raise ValueError("minimal L3 runtime requires n_seasonal_lags >= 1")
    lagged = []
    for lag in range(1, n_seasonal_lags + 1):
        periods = seasonal_period * lag
        lagged.append(frame.shift(periods).add_suffix(f"_s{seasonal_period}_lag{lag}"))
    return pd.concat(lagged, axis=1)


def _ma_increasing_order(frame: pd.DataFrame, *, max_order: int) -> pd.DataFrame:
    if max_order < 2:
        raise ValueError("minimal L3 runtime requires max_order >= 2")
    windows = []
    for order in range(2, max_order + 1):
        windows.append(
            frame.rolling(window=order, min_periods=order)
            .mean()
            .add_suffix(f"_ma{order}")
        )
    return pd.concat(windows, axis=1)


def _maf_per_variable_pca(
    frame: pd.DataFrame,
    *,
    n_lags: int = 12,
    n_components_per_var: int = 2,
) -> pd.DataFrame:
    """Coulombe et al. (2021 IJF) Eq. (7) per-variable PCA MAF.

    For each column k in ``frame``:
      1. Build the (T, n_lags+1) lag-panel: [X_k, L X_k, ..., L^n_lags X_k].
      2. Drop rows with any NaN (introduced by lagging).
      3. Run sklearn PCA with n_components = min(n_components_per_var,
         effective_rank) on the cleaned lag-panel.
      4. Project back to full T-length index (NaN rows remain NaN in output).
      5. Name output columns ``{col}_maf1``, ``{col}_maf2``, ...,
         ``{col}_maf{n_components_per_var}``.

    Returns a (T, K * n_components_per_var) DataFrame.

    Operational from v0.9.0 (phase-f16). Registered as L3 op
    ``maf_per_variable_pca`` in :mod:`macroforecast.core.ops.l3_ops`.
    """
    from sklearn.decomposition import PCA

    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    if n_components_per_var < 1:
        raise ValueError("n_components_per_var must be >= 1")

    pieces: list[pd.DataFrame] = []
    for col in frame.columns:
        series = frame[col]
        # Build lag-panel: shape (T, n_lags + 1)
        lag_cols = [series.shift(j).rename(f"lag{j}") for j in range(n_lags + 1)]
        lag_panel = pd.concat(lag_cols, axis=1)
        valid_rows = lag_panel.dropna()
        if valid_rows.empty or valid_rows.shape[0] < 2:
            # Degenerate: not enough data; fill with NaN columns
            n_comp = n_components_per_var
            empty = pd.DataFrame(
                np.nan,
                index=frame.index,
                columns=[f"{col}_maf{j + 1}" for j in range(n_comp)],
            )
            pieces.append(empty)
            continue
        n_comp = min(n_components_per_var, valid_rows.shape[0] - 1, valid_rows.shape[1])
        n_comp = max(n_comp, 1)
        pca = PCA(n_components=n_comp)
        factors_valid = pca.fit_transform(valid_rows.to_numpy(dtype=float))
        # Pad back to full T-length index
        factor_df = pd.DataFrame(
            factors_valid,
            index=valid_rows.index,
            columns=[f"{col}_maf{j + 1}" for j in range(n_comp)],
        )
        if n_comp < n_components_per_var:
            for j in range(n_comp, n_components_per_var):
                factor_df[f"{col}_maf{j + 1}"] = np.nan
        factor_df = factor_df.reindex(frame.index)
        pieces.append(factor_df)
    if not pieces:
        return pd.DataFrame(index=frame.index)
    return pd.concat(pieces, axis=1)


def _polynomial_expansion(frame: pd.DataFrame, *, degree: int) -> pd.DataFrame:
    if degree < 1:
        raise ValueError("minimal L3 runtime requires degree >= 1")
    pieces = [frame]
    for power in range(2, degree + 1):
        pieces.append(frame.pow(power).add_suffix(f"_pow{power}"))
    return pd.concat(pieces, axis=1)


def _interaction_terms(frame: pd.DataFrame) -> pd.DataFrame:
    terms: dict[str, pd.Series] = {}
    columns = list(frame.columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            terms[f"{left}_x_{right}"] = frame[left] * frame[right]
    return pd.DataFrame(terms, index=frame.index)


def _season_dummy(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        values = frame.index.month
        prefix = "month"
    else:
        values = pd.Series(range(len(frame)), index=frame.index) % 12 + 1
        prefix = "season"
    dummies = pd.get_dummies(values, prefix=prefix, dtype=float)
    dummies.index = frame.index
    return dummies


def _first_node(raw: dict[str, Any], *, op: str) -> dict[str, Any] | None:
    for node in raw.get("nodes", ()) or ():
        if isinstance(node, dict) and node.get("op") == op:
            return node
    return None


def _materialize_regime(
    resolved: dict[str, Any],
    leaf_config: dict[str, Any],
    sample_index: pd.DatetimeIndex,
) -> L1RegimeMetadataArtifact:
    """Populate L1 regime artifact, optionally loading external NBER dates.

    For ``external_nber`` we use the embedded NBER recession date list (months
    in recession get label ``recession``; everything else ``expansion``).
    For ``external_user_provided`` we read either a path or an inline date
    list from ``leaf_config.regime_dates_list``.
    """

    base = l1_layer._regime_artifact_from_resolved(resolved, leaf_config)
    definition = base.definition
    if definition == "external_nber" and len(sample_index):
        labels = _build_nber_regime_series(sample_index)
        return L1RegimeMetadataArtifact(
            definition=base.definition,
            n_regimes=2,
            regime_label_series=Series(
                shape=labels.shape,
                name="nber_recession",
                metadata=SeriesMetadata(
                    values={"data": labels, "source": "embedded_nber_recession_dates"}
                ),
            ),
            regime_probabilities=None,
            transition_matrix=None,
            estimation_temporal_rule=base.estimation_temporal_rule,
            estimation_metadata={
                **base.estimation_metadata,
                "n_recession_months": int((labels == "recession").sum()),
            },
        )
    if definition == "external_user_provided" and len(sample_index):
        labels = _build_user_provided_regime_series(leaf_config, sample_index)
        if labels is not None:
            return L1RegimeMetadataArtifact(
                definition=base.definition,
                n_regimes=int(labels.nunique()),
                regime_label_series=Series(
                    shape=labels.shape,
                    name="user_regime",
                    metadata=SeriesMetadata(values={"data": labels}),
                ),
                regime_probabilities=None,
                transition_matrix=None,
                estimation_temporal_rule=base.estimation_temporal_rule,
                estimation_metadata=base.estimation_metadata,
            )
    if definition == "estimated_markov_switching" and len(sample_index):
        target_name = leaf_config.get("regime_estimation_series") or leaf_config.get(
            "target"
        )
        n_regimes = int(leaf_config.get("n_regimes", 2))
        labels, probs, transition_matrix, metadata = _estimate_markov_switching_regime(
            sample_index, target_name, n_regimes, leaf_config
        )
        return L1RegimeMetadataArtifact(
            definition=base.definition,
            n_regimes=n_regimes,
            regime_label_series=Series(
                shape=labels.shape,
                name="hamilton_ms_regime",
                metadata=SeriesMetadata(
                    values={"data": labels, "source": "hamilton_1989_markov_regression"}
                ),
            ),
            regime_probabilities=probs,
            transition_matrix=transition_matrix,
            estimation_temporal_rule=base.estimation_temporal_rule,
            estimation_metadata={**base.estimation_metadata, **metadata},
        )
    if definition == "estimated_threshold" and len(sample_index):
        n_regimes = int(leaf_config.get("n_regimes", 2))
        labels, metadata = _estimate_threshold_regime(
            sample_index, n_regimes, leaf_config
        )
        return L1RegimeMetadataArtifact(
            definition=base.definition,
            n_regimes=n_regimes,
            regime_label_series=Series(
                shape=labels.shape,
                name="setar_regime",
                metadata=SeriesMetadata(
                    values={"data": labels, "source": "tong_1990_setar"}
                ),
            ),
            regime_probabilities=None,
            transition_matrix=None,
            estimation_temporal_rule=base.estimation_temporal_rule,
            estimation_metadata={**base.estimation_metadata, **metadata},
        )
    if definition == "estimated_structural_break" and len(sample_index):
        max_breaks = int(leaf_config.get("max_breaks", 3))
        labels, metadata = _estimate_structural_break_regime(
            sample_index, max_breaks, leaf_config
        )
        return L1RegimeMetadataArtifact(
            definition=base.definition,
            n_regimes=int(labels.nunique()),
            regime_label_series=Series(
                shape=labels.shape,
                name="bai_perron_regime",
                metadata=SeriesMetadata(
                    values={"data": labels, "source": "bai_perron_break_detection"}
                ),
            ),
            regime_probabilities=None,
            transition_matrix=None,
            estimation_temporal_rule=base.estimation_temporal_rule,
            estimation_metadata={**base.estimation_metadata, **metadata},
        )
    return base


def _estimate_threshold_regime(
    sample_index: pd.DatetimeIndex,
    n_regimes: int,
    leaf_config: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    """Issue #243 -- full Tong (1990) SETAR with grid-search threshold
    estimation + AR(p) per regime.

    For each candidate threshold ``r`` on the threshold variable's
    empirical grid, fit AR(``threshold_ar_p``) on the rows below ``r``
    and on the rows at-or-above ``r`` separately, then sum the residual
    SSR. The selected threshold minimises the joint SSR (equivalent to
    AIC under Gaussian residuals + identical regime sizes). For
    ``n_regimes > 2`` the procedure recurses on each partition.

    Falls back to the v0.2 quantile-split when the threshold series is
    absent or too short.
    """

    series_data = leaf_config.get("regime_target_values")
    n = len(sample_index)
    if series_data is None or len(series_data) < max(20, n_regimes * 8):
        # Fallback path -- documented to keep recipes runnable.
        idx_values = np.arange(n, dtype=float)
        cut_points = np.quantile(idx_values, np.linspace(0, 1, n_regimes + 1))
        thresholds = list(cut_points[1:-1])
        labels = [
            f"regime_{min(int(np.searchsorted(thresholds, v, side='right')), n_regimes - 1)}"
            for v in idx_values
        ]
        return pd.Series(labels, index=sample_index), {
            "method": "fallback_quantile_split",
            "reason": "threshold series unavailable or too short",
            "thresholds": [float(t) for t in thresholds],
            "n_regimes": n_regimes,
        }

    threshold_values = np.asarray(series_data, dtype=float)[:n]
    valid_mask = ~np.isnan(threshold_values)
    valid_values = threshold_values[valid_mask]
    ar_p = int(leaf_config.get("threshold_ar_p", 1))

    def _ar_ssr(values: np.ndarray, p: int) -> float:
        """OLS AR(p) residual SSR on the supplied series."""

        if values.size <= p + 2:
            return float("inf")
        y = values[p:]
        X = np.column_stack(
            [np.ones(len(y))]
            + [values[p - lag : -lag] if lag else values[p:] for lag in range(1, p + 1)]
        )
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")
        resid = y - X @ coef
        return float(np.dot(resid, resid))

    def _split_partition(values: np.ndarray) -> tuple[float, float] | None:
        """Find the threshold ``r`` that minimises SSR(below) + SSR(at-or-above).
        Returns ``(threshold, ssr_total)`` or ``None`` if no valid split."""

        if values.size < 2 * (ar_p + 3):
            return None
        # Grid: trim 10% / 90% quantiles to keep regime sizes balanced.
        grid = np.quantile(values, np.linspace(0.1, 0.9, 50))
        best: tuple[float, float] | None = None
        for r in grid:
            below = values[values < r]
            above = values[values >= r]
            if below.size < ar_p + 3 or above.size < ar_p + 3:
                continue
            ssr = _ar_ssr(below, ar_p) + _ar_ssr(above, ar_p)
            if best is None or ssr < best[1]:
                best = (float(r), float(ssr))
        return best

    selected_thresholds: list[float] = []
    if n_regimes >= 2:
        # Recursive partitioning -- find thresholds one at a time.
        partitions = [valid_values]
        for _ in range(n_regimes - 1):
            best_partition_idx: int | None = None
            best_split: tuple[float, float] | None = None
            for idx, part in enumerate(partitions):
                split = _split_partition(part)
                if split is None:
                    continue
                if best_split is None or split[1] < best_split[1]:
                    best_partition_idx = idx
                    best_split = split
            if best_partition_idx is None or best_split is None:
                break
            r, _ = best_split
            selected_thresholds.append(r)
            old = partitions.pop(best_partition_idx)
            partitions.extend([old[old < r], old[old >= r]])
        selected_thresholds = sorted(selected_thresholds)

    if not selected_thresholds:
        return _estimate_threshold_regime(
            sample_index,
            n_regimes,
            {**leaf_config, "regime_target_values": None},
        )

    labels: list[str] = []
    for v in threshold_values:
        if np.isnan(v):
            labels.append("regime_0")
            continue
        regime_idx = int(np.searchsorted(selected_thresholds, v, side="right"))
        labels.append(f"regime_{min(regime_idx, n_regimes - 1)}")
    return pd.Series(labels, index=sample_index), {
        "method": "tong_1990_setar_full_grid",
        "thresholds": [float(t) for t in selected_thresholds],
        "n_regimes": n_regimes,
        "threshold_ar_p": ar_p,
        "criterion": "min_joint_ssr",
    }


def _estimate_structural_break_regime(
    sample_index: pd.DatetimeIndex,
    max_breaks: int,
    leaf_config: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    """Bai-Perron (1998) global least-squares break detection.

    For a univariate target series ``y_t``, detect up to ``max_breaks``
    break points by minimising the sum of within-segment squared
    deviations from each segment's mean (the standard Bai-Perron
    criterion for a constant-only model). The number of breaks is
    selected by minimising BIC across ``k = 0, 1, ..., max_breaks``.

    When ``regime_target_values`` is missing we fall back to evenly-spaced
    breakpoints; the metadata records the path taken so callers can
    detect the fallback.
    """

    series_data = leaf_config.get("regime_target_values")
    n = len(sample_index)
    if series_data is None or len(series_data) < max(2 * (max_breaks + 1), 12):
        # Fallback: equal-spaced segments.
        edges = np.linspace(0, n, max_breaks + 2, dtype=int)
        labels = pd.Series(index=sample_index, dtype=object)
        for r in range(max_breaks + 1):
            labels.iloc[edges[r] : edges[r + 1]] = f"regime_{r}"
        return labels.fillna(f"regime_{max_breaks}"), {
            "method": "fallback_equal_spaced",
            "max_breaks": max_breaks,
        }

    y = np.asarray(series_data, dtype=float)[:n]
    valid = ~np.isnan(y)
    y_valid = y[valid]
    n_valid = y_valid.size
    if n_valid < 2 * (max_breaks + 1):
        edges = np.linspace(0, n, max_breaks + 2, dtype=int)
        labels = pd.Series(index=sample_index, dtype=object)
        for r in range(max_breaks + 1):
            labels.iloc[edges[r] : edges[r + 1]] = f"regime_{r}"
        return labels.fillna(f"regime_{max_breaks}"), {
            "method": "fallback_too_few_obs",
            "n_obs": int(n_valid),
        }

    # Issue #244 -- Bai (1997) dynamic-programming exact break search.
    #
    # Pre-compute the segment-SSR table ``S[i, j]`` = SSR(y[i:j]) for every
    # valid segment, then solve the recursion
    #
    #     SSR_k(j) = min_{m} { SSR_{k-1}(m) + S[m, j] }
    #
    # where ``SSR_k(j)`` is the minimum total SSR for a partition of the
    # first ``j`` observations into exactly ``k`` segments. Backtracking
    # recovers the optimal break dates per ``k``. The optimal ``k`` is then
    # picked by BIC across ``k = 1, ..., max_breaks + 1``.
    min_len = max(3, n_valid // (max_breaks * 2 + 4))
    cumsum = np.concatenate(([0.0], np.cumsum(y_valid)))
    cumsq = np.concatenate(([0.0], np.cumsum(y_valid**2)))

    def segment_ssr(start: int, end: int) -> float:
        length = end - start
        if length <= 0:
            return 0.0
        s = cumsum[end] - cumsum[start]
        sq = cumsq[end] - cumsq[start]
        return float(sq - (s * s) / length)

    INF = float("inf")
    K_max = max_breaks + 1  # number of segments
    # ssr_k_j[k, j] = min total SSR splitting the first j obs into k segments.
    ssr_k_j = np.full((K_max + 1, n_valid + 1), INF)
    prev = np.full((K_max + 1, n_valid + 1), -1, dtype=int)
    # Base case: k=1 segment from 0 to j.
    for j in range(min_len, n_valid + 1):
        ssr_k_j[1, j] = segment_ssr(0, j)
        prev[1, j] = 0
    for k in range(2, K_max + 1):
        for j in range(k * min_len, n_valid + 1):
            best = INF
            best_m = -1
            for m in range((k - 1) * min_len, j - min_len + 1):
                if ssr_k_j[k - 1, m] >= INF:
                    continue
                ssr_total = ssr_k_j[k - 1, m] + segment_ssr(m, j)
                if ssr_total < best:
                    best = ssr_total
                    best_m = m
            if best_m >= 0:
                ssr_k_j[k, j] = best
                prev[k, j] = best_m

    # BIC-select the optimal number of segments.
    best_k = 1
    best_bic = INF
    for k in range(1, K_max + 1):
        ssr = ssr_k_j[k, n_valid]
        if ssr >= INF:
            continue
        bic = n_valid * np.log(max(ssr / n_valid, 1e-9)) + k * np.log(n_valid)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    # Backtrack the optimal break points for best_k segments.
    breakpoints: list[int] = []
    j = n_valid
    for k in range(best_k, 0, -1):
        m = int(prev[k, j])
        if m == 0:
            break
        breakpoints.append(m)
        j = m
    breakpoints = sorted(breakpoints)

    boundaries = sorted([0] + breakpoints + [n_valid])
    valid_indices = np.where(valid)[0]
    labels = pd.Series(index=sample_index, dtype=object)
    for r in range(len(boundaries) - 1):
        valid_slice = valid_indices[boundaries[r] : boundaries[r + 1]]
        labels.iloc[valid_slice] = f"regime_{r}"
    labels = labels.fillna(f"regime_{best_k - 1}")
    metadata = {
        "method": "bai_perron_global_lse_dp",
        "n_breaks_selected": int(best_k - 1),
        "break_indices": [int(b) for b in breakpoints],
        "bic": float(best_bic),
        "min_segment_length": int(min_len),
    }
    return labels, metadata


def _estimate_markov_switching_regime(
    sample_index: pd.DatetimeIndex,
    target_name: str | None,
    n_regimes: int,
    leaf_config: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    """Hamilton (1989) Markov regression on the target series via
    ``statsmodels.tsa.regime_switching.MarkovRegression``.

    Returns:
        labels:    pd.Series of regime labels (``regime_0``, ``regime_1``, ...)
        probs:     pd.DataFrame of smoothed posterior probabilities
        transition_matrix: pd.DataFrame of estimated transition probabilities
        metadata:  dict of optimisation diagnostics

    Falls back to a piecewise-quantile rule when the target column is
    unavailable (custom-panel-only recipes that don't ship the target
    series at the L1 raw stage).
    """

    # Fall back to a deterministic split when no target series is reachable.
    if target_name is None:
        labels = pd.Series(
            ["regime_0"] * (len(sample_index) // 2)
            + ["regime_1"] * (len(sample_index) - len(sample_index) // 2),
            index=sample_index,
        )
        return (
            labels,
            None,
            None,
            {"method": "fallback_uniform_split", "reason": "target series unavailable"},
        )

    # In v0.1 the regime estimator only sees ``raw_panel`` via L1 path; here
    # we need to access the series. The simplest is to require the caller to
    # have provided it via leaf_config.regime_target_values; otherwise fall
    # back to a deterministic split. The L4 / L5 layers can apply the regime
    # downstream once the labels exist.
    series_data = leaf_config.get("regime_target_values")
    if series_data is None:
        # Fallback: alternate-half split. This is *not* Hamilton MS but it
        # produces something traceable that the rest of the pipeline can use
        # while still flagging the limitation in metadata.
        labels = pd.Series(
            ["regime_0"] * (len(sample_index) // 2)
            + ["regime_1"] * (len(sample_index) - len(sample_index) // 2),
            index=sample_index,
        )
        return (
            labels,
            None,
            None,
            {
                "method": "fallback_split_no_series",
                "warning": "regime_target_values not provided in leaf_config; "
                "supply the target series there to trigger the real Hamilton MS estimator",
            },
        )

    series = pd.Series(series_data, index=sample_index).dropna()
    if series.size < max(20, n_regimes * 5):
        labels = pd.Series(["regime_0"] * len(sample_index), index=sample_index)
        return (
            labels,
            None,
            None,
            {"method": "fallback_too_few_obs", "n_obs": int(series.size)},
        )
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        model = MarkovRegression(
            series.values,
            k_regimes=n_regimes,
            trend="c",
            switching_variance=True,
        )
        results = model.fit(disp=False)
        smoothed = pd.DataFrame(
            results.smoothed_marginal_probabilities,
            index=series.index,
            columns=[f"regime_{i}" for i in range(n_regimes)],
        )
        # Reindex back to the full sample (NaN-pad rows we dropped).
        smoothed_full = smoothed.reindex(sample_index)
        labels = smoothed_full.idxmax(axis=1).fillna("regime_0")
        # Estimated transition matrix.
        try:
            P = results.regime_transition[:, :, 0]  # (k, k) at the first time
            transition_matrix = pd.DataFrame(
                P,
                index=[f"from_regime_{i}" for i in range(n_regimes)],
                columns=[f"to_regime_{i}" for i in range(n_regimes)],
            )
        except Exception:
            transition_matrix = None
        metadata = {
            "method": "hamilton_1989_markov_regression",
            "log_likelihood": float(results.llf),
            "aic": float(results.aic),
            "bic": float(results.bic),
            "converged": bool(
                getattr(results, "mle_retvals", {}).get("converged", True)
            ),
        }
        return labels, smoothed_full, transition_matrix, metadata
    except Exception as exc:
        labels = pd.Series(["regime_0"] * len(sample_index), index=sample_index)
        return labels, None, None, {"method": "fallback_fit_failed", "error": str(exc)}


# NBER official US recession dates (start, end) inclusive, monthly.
# Source: nber.org/research/business-cycle-dating (peaks/troughs).
_NBER_RECESSIONS: tuple[tuple[str, str], ...] = (
    ("1948-11", "1949-10"),
    ("1953-07", "1954-05"),
    ("1957-08", "1958-04"),
    ("1960-04", "1961-02"),
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
)


def _build_nber_regime_series(index: pd.DatetimeIndex) -> pd.Series:
    in_recession = pd.Series(False, index=index)
    for start, end in _NBER_RECESSIONS:
        # Use the actual month-end so daily indices like 2009-06-30 stay inside
        # a recession that ends in 2009-06 (was: hard-coded "-28" which dropped
        # the last 1-3 days of 31/30-day months).
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
        mask = (index >= start_ts) & (index <= end_ts)
        in_recession |= mask
    labels = in_recession.map({True: "recession", False: "expansion"}).astype(str)
    labels.index = index
    return labels


def _build_user_provided_regime_series(
    leaf_config: dict[str, Any], index: pd.DatetimeIndex
) -> pd.Series | None:
    path = leaf_config.get("regime_indicator_path")
    if path:
        candidate = pd.read_csv(path, parse_dates=[0], index_col=0)
        if candidate.shape[1] >= 1:
            series = candidate.iloc[:, 0].astype(str)
            series.index = pd.DatetimeIndex(series.index)
            return series.reindex(index, method="nearest")
    dates_list = leaf_config.get("regime_dates_list")
    if dates_list:
        labels = pd.Series("baseline", index=index, dtype=object)
        for entry in dates_list:
            if isinstance(entry, dict):
                start = pd.Timestamp(entry.get("start"))
                end = pd.Timestamp(entry.get("end", entry.get("until")))
                label = str(entry.get("label", "alt"))
            else:
                start, end, label = (
                    pd.Timestamp(entry[0]),
                    pd.Timestamp(entry[1]),
                    str(entry[2] if len(entry) > 2 else "alt"),
                )
            labels.loc[(index >= start) & (index <= end)] = label
        return labels
    return None


def _l1_context(artifact: L1DataDefinitionArtifact) -> dict[str, Any]:
    return {
        "custom_source_policy": artifact.custom_source_policy,
        "dataset": artifact.dataset,
        "frequency": artifact.frequency,
        "custom_has_tcode_column": bool(artifact.leaf_config.get("custom_tcode_map")),
    }


def _l3_context(artifact: L1DataDefinitionArtifact) -> dict[str, Any]:
    return {
        "horizons": set(artifact.target_horizons),
        "regime_definition": artifact.regime_definition,
    }
