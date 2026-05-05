from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import math
import json
from pathlib import Path
import platform
from typing import Any, Literal

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
from ..raw.fred_sd_groups import FRED_SD_STATE_GROUPS
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

    root = parse_recipe_yaml(recipe_yaml_or_root) if isinstance(recipe_yaml_or_root, str) else recipe_yaml_or_root
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
        l2_5_artifact, l2_5_axes = materialize_l2_5_diagnostic(root, l1_artifact, l2_artifact)
        artifacts["l2_5_diagnostic_v1"] = l2_5_artifact
        resolved_axes["l2_5"] = l2_5_axes
    return RuntimeResult(
        artifacts=artifacts,
        resolved_axes=resolved_axes,
    )


def execute_minimal_forecast(recipe_yaml_or_root: str | dict[str, Any]) -> RuntimeResult:
    """Run the minimal L1-L5 runtime path for custom-panel ridge forecasts."""

    import time as _time

    root = parse_recipe_yaml(recipe_yaml_or_root) if isinstance(recipe_yaml_or_root, str) else recipe_yaml_or_root
    durations: dict[str, float] = {}

    def _timed(label: str, fn):
        clock = _time.perf_counter()
        result = fn()
        durations[label] = _time.perf_counter() - clock
        return result

    l1_artifact, regime_artifact, l1_axes = _timed("l1", lambda: materialize_l1(root))
    l2_artifact, l2_axes = _timed("l2", lambda: materialize_l2(root, l1_artifact))
    l3_features, l3_metadata = _timed("l3", lambda: materialize_l3_minimal(root, l1_artifact, l2_artifact))
    l4_forecasts, l4_models, l4_training = _timed("l4", lambda: materialize_l4_minimal(root, l3_features))
    l5_eval = _timed("l5", lambda: materialize_l5_minimal(root, l1_artifact, l3_features, l4_forecasts, l4_models))
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
    resolved_axes: dict[str, dict[str, Any]] = {"l1": l1_axes, "l2": dict(l2_axes), "l5": dict(l5_eval.l5_axis_resolved)}
    if "1_5_data_summary" in root:
        l1_5_artifact, l1_5_axes = _timed("l1_5", lambda: materialize_l1_5_diagnostic(root, l1_artifact))
        artifacts["l1_5_diagnostic_v1"] = l1_5_artifact
        resolved_axes["l1_5"] = l1_5_axes
    if "2_5_pre_post_preprocessing" in root:
        l2_5_artifact, l2_5_axes = _timed("l2_5", lambda: materialize_l2_5_diagnostic(root, l1_artifact, l2_artifact))
        artifacts["l2_5_diagnostic_v1"] = l2_5_artifact
        resolved_axes["l2_5"] = l2_5_axes
    if "3_5_feature_diagnostics" in root:
        l3_5_artifact, l3_5_axes = _timed(
            "l3_5", lambda: materialize_l3_5_diagnostic(root, l1_artifact, l2_artifact, l3_features, l3_metadata)
        )
        artifacts["l3_5_diagnostic_v1"] = l3_5_artifact
        resolved_axes["l3_5"] = l3_5_axes
    if "4_5_generator_diagnostics" in root:
        l4_5_artifact, l4_5_axes = _timed(
            "l4_5", lambda: materialize_l4_5_diagnostic(root, l3_features, l4_forecasts, l4_models, l4_training)
        )
        artifacts["l4_5_diagnostic_v1"] = l4_5_artifact
        resolved_axes["l4_5"] = l4_5_axes
    if "6_statistical_tests" in root:
        l6_tests, l6_axes = _timed(
            "l6", lambda: materialize_l6_runtime(root, l1_artifact, l3_features, l4_forecasts, l4_models, l5_eval)
        )
        artifacts["l6_tests_v1"] = l6_tests
        resolved_axes["l6"] = l6_axes
    if "7_interpretation" in root:
        l7_importance, l7_transform, l7_axes = _timed(
            "l7",
            lambda: materialize_l7_runtime(
                root, l3_features, l3_metadata, l4_forecasts, l4_models, l5_eval, artifacts.get("l6_tests_v1")
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


def materialize_l1(recipe_root: dict[str, Any]) -> tuple[L1DataDefinitionArtifact, L1RegimeMetadataArtifact, dict[str, Any]]:
    raw = recipe_root.get("1_data", {}) or {}
    report = l1_layer.validate_layer(raw)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))

    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    resolved = l1_layer.resolve_axes_from_raw(fixed_axes, leaf_config)
    raw_panel = _load_raw_panel(resolved, leaf_config)

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


def materialize_l2(recipe_root: dict[str, Any], l1_artifact: L1DataDefinitionArtifact) -> tuple[L2CleanPanelArtifact, l2_layer.L2ResolvedAxes]:
    raw = recipe_root.get("2_preprocessing", {}) or {}
    l1_context = _l1_context(l1_artifact)
    report = l2_layer.validate_layer(raw, l1_context=l1_context)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))

    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    resolved = l2_layer.resolve_axes_from_raw(fixed_axes, leaf_config, l1_context=l1_context)
    df = l1_artifact.raw_panel.data.copy()
    if df.empty:
        raise ValueError("L1 raw_panel is empty; L2 materialization requires custom panel data")

    cleaning_log: dict[str, Any] = {"runtime": "core_l1_l2_materialization", "steps": []}
    transform_map: dict[str, int] = {}
    n_outliers = 0
    n_imputed = 0

    l1_leaf_for_l2 = dict(l1_artifact.leaf_config)
    official_tcodes = (l1_artifact.raw_panel.metadata.values or {}).get("transform_codes", {})
    if official_tcodes:
        l1_leaf_for_l2["official_tcode_map"] = dict(official_tcodes)

    # Issue #202: FRED-SD frequency alignment. Applied *before* the
    # transform pipeline so downstream stages see a single-frequency
    # panel.
    if l1_artifact.dataset and "fred_sd" in str(l1_artifact.dataset):
        df = _apply_fred_sd_frequency_alignment(df, resolved, l1_artifact, cleaning_log)
    df, transform_map = _apply_transform(df, resolved, leaf_config, l1_leaf_for_l2, cleaning_log)
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

    panel = _panel_from_frame(df, metadata={"stage": "l2_clean", "source": "l1_raw_panel"})
    artifact = L2CleanPanelArtifact(
        panel=panel,
        shape=panel.shape,
        column_names=panel.column_names,
        index=panel.index,
        column_metadata={column: {"dtype": str(df[column].dtype)} for column in df.columns},
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


def materialize_l1_5_diagnostic(recipe_root: dict[str, Any], l1_artifact: L1DataDefinitionArtifact) -> tuple[DiagnosticArtifact, dict[str, Any]]:
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
        "univariate_summary": _diagnostic_univariate_summary(frame, axes.get("summary_metrics", [])),
        "missing_outlier_audit": _diagnostic_missing_outlier_audit(frame, axes.get("leaf_config", {})),
    }
    if axes.get("correlation_view") != "none":
        metadata["correlation"] = frame.corr(method=axes.get("correlation_method", "pearson"), numeric_only=True)
    stationarity_test = axes.get("stationarity_test", "none")
    if stationarity_test != "none":
        metadata["stationarity_tests"] = _diagnostic_stationarity_tests(
            frame=frame,
            test=stationarity_test,
            scope=axes.get("stationarity_test_scope", "target_and_predictors"),
            target=l1_artifact.target,
            targets=l1_artifact.targets,
            alpha=float((axes.get("leaf_config") or {}).get("stationarity_alpha", 0.05)),
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
        return {"statistic": float(stat), "p_value": float(pvalue), "reject_unit_root": bool(pvalue < alpha)}
    if name == "kpss":
        from statsmodels.tsa.stattools import kpss as _kpss
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, *_ = _kpss(series.values, regression="c", nlags="auto")
        # KPSS null = stationarity; "reject_stationarity" when p < alpha.
        return {"statistic": float(stat), "p_value": float(pvalue), "reject_stationarity": bool(pvalue < alpha)}
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
        - 0.5 * (lr_var - gamma0) * np.sqrt(n) * np.sqrt(XtX_inv[1, 1]) / np.sqrt(max(lr_var, 1e-12))
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
    recipe_root: dict[str, Any], l1_artifact: L1DataDefinitionArtifact, l2_artifact: L2CleanPanelArtifact
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
        "distribution_shift": _diagnostic_distribution_shift(raw_frame, clean_frame, axes.get("distribution_metric", [])),
        "cleaning_effect_summary": {
            "n_imputed_cells": l2_artifact.n_imputed_cells,
            "n_outliers_flagged": l2_artifact.n_outliers_flagged,
            "n_truncated_obs": l2_artifact.n_truncated_obs,
            "transform_map_applied": dict(l2_artifact.transform_map_applied),
            "cleaning_log": l2_artifact.cleaning_log,
        },
    }
    if axes.get("correlation_shift") != "none":
        metadata["correlation_shift"] = clean_frame.corr(numeric_only=True) - raw_frame.corr(numeric_only=True)
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
        metadata["per_decade_summary"] = _diagnostic_per_decade_summary(raw_frame, clean_frame)
    # ``t_code_application_log = per_series_detail`` -- per-series record
    # of which transform code was applied (from L2 transform_map).
    if axes.get("t_code_application_log") == "per_series_detail":
        metadata["t_code_log_per_series"] = {
            str(series): int(code) for series, code in l2_artifact.transform_map_applied.items()
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


def _diagnostic_per_decade_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, Any]:
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
                "mean": float(group.drop(columns="__decade").select_dtypes("number").mean(numeric_only=True).mean()),
                "std": float(group.drop(columns="__decade").select_dtypes("number").std(ddof=0, numeric_only=True).mean()),
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
        "comparison": _diagnostic_l3_comparison(raw_frame, clean_frame, feature_frame, l3_features),
        "feature_summary": _diagnostic_feature_summary(feature_frame),
        "lineage_summary": _diagnostic_l3_lineage_summary(l3_metadata),
        "factor_block": {"active": bool(context.get("has_factor_step")), "n_factors_to_show": axes.get("leaf_config", {}).get("n_factors_to_show", 8)},
        "lag_block": _diagnostic_l3_lag_summary(feature_frame, active=bool(context.get("has_lag_step"))),
        "selection_summary": {"active": bool(context.get("has_feature_selection_step"))},
    }
    if axes.get("feature_correlation") != "none":
        metadata["feature_correlation"] = feature_frame.corr(method=axes.get("correlation_method", "pearson"), numeric_only=True)
    # Issue #211: factor diagnostics. When the L3 panel has at least
    # ``n_factors_to_show`` columns, run a quick PCA so the diagnostic sink
    # carries the eigenvalue scree, loadings, and factor time series.
    n_factors_to_show = int((axes.get("leaf_config", {}) or {}).get("n_factors_to_show", 4))
    if feature_frame.shape[0] >= 4 and feature_frame.shape[1] >= 2:
        try:
            from sklearn.decomposition import PCA as _PCA

            n_comp = min(n_factors_to_show, feature_frame.shape[0] - 1, feature_frame.shape[1])
            pca = _PCA(n_components=n_comp)
            scores = pca.fit_transform(feature_frame.fillna(0.0).to_numpy())
            metadata["factor_diagnostics"] = {
                "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
                "eigenvalues": [float(v) for v in pca.explained_variance_],
                "cumulative_variance": [float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
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
        "fit_summary": _diagnostic_l4_fit_summary(l4_forecasts, actual if isinstance(actual, pd.Series) else None),
    }
    if axes.get("window_view") != "none":
        metadata["window_stability"] = _diagnostic_l4_window_summary(l4_training)
    # Issue #212: window stability + per-origin loss series for the rolling
    # training-loss curve diagnostic figure.
    if isinstance(actual, pd.Series) and l4_forecasts.forecasts:
        per_origin: dict[Any, float] = {}
        for (model_id, target, horizon, origin), forecast in l4_forecasts.forecasts.items():
            if origin not in actual.index:
                continue
            per_origin[origin] = (
                per_origin.get(origin, 0.0) + (float(actual.loc[origin]) - float(forecast)) ** 2
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
        for (model_id, target, horizon, origin), forecast in l4_forecasts.forecasts.items():
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
                    denom = float((centred ** 2).sum())
                    acf = []
                    for lag in range(1, 6):
                        if arr.size <= lag or denom <= 0:
                            acf.append(0.0)
                            continue
                        acf.append(float((centred[:-lag] * centred[lag:]).sum() / denom))
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
                for (m_id, _target, _horizon, origin), forecast in l4_forecasts.forecasts.items():
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
                for (m_id, _target, _horizon, origin), forecast in l4_forecasts.forecasts.items():
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
        "start": {column: _iso_or_none(frame[column].first_valid_index()) for column in frame.columns},
        "end": {column: _iso_or_none(frame[column].last_valid_index()) for column in frame.columns},
        "n_obs": frame.notna().sum().astype(int).to_dict(),
        "n_missing": frame.isna().sum().astype(int).to_dict(),
        "panel_shape": frame.shape,
    }


def _diagnostic_univariate_summary(frame: pd.DataFrame, metrics: list[str]) -> dict[str, dict[str, float | int | None]]:
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


def _diagnostic_missing_outlier_audit(frame: pd.DataFrame, leaf_config: dict[str, Any]) -> dict[str, Any]:
    numeric = frame.select_dtypes("number")
    threshold = float(leaf_config.get("outlier_threshold_iqr", 10.0))
    median = numeric.median()
    iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
    outlier_mask = (numeric - median).abs() > threshold * iqr.replace(0, pd.NA)
    return {
        "missing_count": frame.isna().sum().astype(int).to_dict(),
        "longest_gap": {column: _longest_missing_gap(frame[column]) for column in frame.columns},
        "iqr_outlier_count": outlier_mask.fillna(False).sum().astype(int).to_dict(),
    }


def _diagnostic_pre_post_comparison(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "raw_shape": raw_frame.shape,
        "clean_shape": clean_frame.shape,
        "raw_missing_total": int(raw_frame.isna().sum().sum()),
        "clean_missing_total": int(clean_frame.isna().sum().sum()),
        "common_columns": sorted(set(raw_frame.columns) & set(clean_frame.columns)),
    }


def _diagnostic_distribution_shift(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame, metrics: list[str]) -> dict[str, dict[str, float | None]]:
    common = [column for column in raw_frame.select_dtypes("number").columns if column in clean_frame.select_dtypes("number").columns]
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
                values[metric] = _float_or_none(clean.std() / raw_sd) if raw_sd else None
            elif metric == "skew_change":
                values[metric] = _float_or_none(clean.skew() - raw.skew())
            elif metric == "kurtosis_change":
                values[metric] = _float_or_none(clean.kurtosis() - raw.kurtosis())
            elif metric == "ks_statistic":
                values[metric] = _ks_statistic(raw.dropna(), clean.dropna())
        shifts[column] = values
    return shifts


def _diagnostic_l3_comparison(
    raw_frame: pd.DataFrame, clean_frame: pd.DataFrame, feature_frame: pd.DataFrame, l3_features: L3FeaturesArtifact
) -> dict[str, Any]:
    return {
        "raw_shape": raw_frame.shape,
        "clean_shape": clean_frame.shape,
        "feature_shape": feature_frame.shape,
        "y_shape": l3_features.y_final.shape,
        "sample_start": _iso_or_none(l3_features.sample_index[0]) if l3_features.sample_index is not None and len(l3_features.sample_index) else None,
        "sample_end": _iso_or_none(l3_features.sample_index[-1]) if l3_features.sample_index is not None and len(l3_features.sample_index) else None,
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
    pipeline_ids = sorted({lineage.pipeline_id for lineage in l3_metadata.column_lineage.values() if lineage.pipeline_id})
    return {
        "n_column_lineage": len(l3_metadata.column_lineage),
        "n_pipeline_definitions": len(l3_metadata.pipeline_definitions),
        "pipeline_ids": tuple(pipeline_ids),
        "source_variables": {key: tuple(value) for key, value in l3_metadata.source_variables.items()},
    }


def _diagnostic_l3_lag_summary(feature_frame: pd.DataFrame, *, active: bool) -> dict[str, Any]:
    lag_columns = [
        str(column)
        for column in feature_frame.columns
        if "_lag" in str(column) or "_ma" in str(column) or "_s" in str(column)
    ]
    return {"active": active, "lag_feature_count": len(lag_columns), "lag_features": tuple(lag_columns)}


def _diagnostic_l4_forecast_summary(l4_forecasts: L4ForecastsArtifact) -> dict[str, Any]:
    return {
        "n_forecasts": len(l4_forecasts.forecasts),
        "forecast_object": l4_forecasts.forecast_object,
        "model_ids": tuple(l4_forecasts.model_ids),
        "targets": tuple(l4_forecasts.targets),
        "horizons": tuple(l4_forecasts.horizons),
        "sample_start": _iso_or_none(l4_forecasts.sample_index[0]) if l4_forecasts.sample_index is not None and len(l4_forecasts.sample_index) else None,
        "sample_end": _iso_or_none(l4_forecasts.sample_index[-1]) if l4_forecasts.sample_index is not None and len(l4_forecasts.sample_index) else None,
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


def _diagnostic_l4_training_summary(l4_training: L4TrainingMetadataArtifact) -> dict[str, Any]:
    return {
        "n_forecast_origins": len(l4_training.forecast_origins),
        "forecast_origins": tuple(_iso_or_none(origin) for origin in l4_training.forecast_origins),
        "refit_origin_count": {model_id: len(origins) for model_id, origins in l4_training.refit_origins.items()},
        "training_window_count": len(l4_training.training_window_per_origin),
    }


def _diagnostic_l4_fit_summary(l4_forecasts: L4ForecastsArtifact, actual: pd.Series | None) -> dict[str, dict[str, float | int | None]]:
    if actual is None:
        return {}
    rows: list[dict[str, Any]] = []
    for (model_id, target, horizon, origin), forecast in l4_forecasts.forecasts.items():
        if origin not in actual.index:
            continue
        error = float(actual.loc[origin]) - float(forecast)
        rows.append({"model_id": model_id, "target": target, "horizon": horizon, "squared_error": error**2, "absolute_error": abs(error)})
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model_id", "target", "horizon"]).agg(
        n=("squared_error", "size"),
        mse=("squared_error", "mean"),
        mae=("absolute_error", "mean"),
    )
    return {
        f"{model_id}|{target}|h{horizon}": {"n": int(values["n"]), "mse": float(values["mse"]), "mae": float(values["mae"])}
        for (model_id, target, horizon), values in summary.iterrows()
    }


def _diagnostic_l4_window_summary(l4_training: L4TrainingMetadataArtifact) -> dict[str, Any]:
    by_model: dict[str, list[tuple[Any, Any, Any]]] = {}
    for (model_id, origin), window in l4_training.training_window_per_origin.items():
        by_model.setdefault(model_id, []).append((origin, window[0], window[1]))
    return {
        model_id: {
            "n_windows": len(windows),
            "first_window": tuple(_iso_or_none(value) for value in min(windows, key=lambda row: row[0])) if windows else (),
            "last_window": tuple(_iso_or_none(value) for value in max(windows, key=lambda row: row[0])) if windows else (),
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
        "inverse_transform": lambda values: transformer.inverse_transform_prediction(values, {}),
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
    return {key: float(state["inverse_transform"]([value])[0]) for key, value in forecasts.items()}


def materialize_l3_minimal(
    recipe_root: dict[str, Any], l1_artifact: L1DataDefinitionArtifact, l2_artifact: L2CleanPanelArtifact
) -> tuple[L3FeaturesArtifact, L3MetadataArtifact]:
    raw = recipe_root.get("3_feature_engineering", {}) or {}
    report = l3_layer.validate_layer(raw, recipe_context=_l3_context(l1_artifact))
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    dag = l3_layer.normalize_to_dag_form(raw)
    df = l2_artifact.panel.data.copy()
    target_name = l1_artifact.target or (l1_artifact.targets[0] if l1_artifact.targets else None)
    if not target_name or target_name not in df.columns:
        raise ValueError("minimal L3 runtime requires target column in L2 clean panel")

    node_values = _execute_l3_dag(dag, df, target_name)
    sink_node = dag.nodes.get(dag.sinks.get("l3_features_v1", ""))
    if sink_node is None or len(sink_node.inputs) < 2:
        raise ValueError("minimal L3 runtime requires l3_features_v1 sink with X_final and y_final")
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
    horizon = int((y.attrs or {}).get("horizon", l1_artifact.target_horizons[0] if l1_artifact.target_horizons else 1))
    y_meta = {"stage": "l3_y_final", "horizon": horizon, "data": y_aligned}
    if transformer_state is not None:
        y_meta["target_transformer"] = transformer_name
        y_meta["target_transformer_state"] = transformer_state
        # Cache the raw (pre-transform) y so L5 can evaluate forecasts on
        # the raw target scale per the contract.
        y_meta["raw_data"] = y.loc[aligned_index]
    return (
        L3FeaturesArtifact(
            X_final=_panel_from_frame(X_aligned, metadata={"stage": "l3_X_final", "runtime": "l3_dag"}),
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
    report = l4_layer.validate_layer(raw, recipe_context={"horizon_set": set(l3_features.horizon_set)})
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    fit_nodes = [node for node in raw.get("nodes", ()) or () if isinstance(node, dict) and node.get("op") == "fit_model"]
    if not fit_nodes:
        raise ValueError("L4 runtime requires a fit_model node")
    X = l3_features.X_final.data
    y = l3_features.y_final.metadata.values.get("data")
    if not isinstance(y, pd.Series):
        raise ValueError("L4 runtime requires L3 y_final series data")
    target = l3_features.y_final.name
    horizon = int(l3_features.horizon_set[0] if l3_features.horizon_set else 1)
    l0_seed = _resolve_l0_seed(recipe_root)
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
                fit_node, raw, X, y, target, horizon, l0_seed,
                parallel_origins=False, n_workers=n_workers,
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
            model_id, model, node_forecasts, node_origins, node_windows = _run_l4_fit_node(
                fit_node, raw, X, y, target, horizon, l0_seed,
                parallel_origins=True, n_workers=n_workers,
            )
            forecasts.update(node_forecasts)
            artifacts[model_id] = model
            benchmark_flags[model_id] = bool(fit_node.get("is_benchmark", False))
            refit_origins[model_id] = tuple(node_origins)
            training_windows.update(node_windows)
            model_ids.append(model_id)
        sample_index = pd.DatetimeIndex(sorted({key[3] for key in forecasts}))
        forecast_object = _resolve_forecast_object(fit_nodes)
        forecast_intervals = _emit_quantile_intervals(forecasts, fit_nodes, X=X, y=y) if forecast_object == "quantile" else {}
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
        rolling_window = int(params.get("rolling_window", max(24, min(len(X) // 2, 120))))
        refit_step = int(params.get("refit_step", 1)) if refit_policy == "every_n_origins" else 1
        # tuning hook: dispatch on search_algorithm (issue #217). Inject the
        # L4 leaf_config so the resolver can read tuning_grid /
        # tuning_distributions / tuning_budget / cv_path_alphas / GA settings.
        if params.get("search_algorithm") in {"cv_path", "grid_search", "random_search", "bayesian_optimization", "genetic_algorithm"}:
            params["_l4_leaf_config"] = raw.get("leaf_config", {}) or {}
            params = _resolve_l4_tuning(params, X, y)
            params.pop("_l4_leaf_config", None)
        min_train_size = _minimal_train_size(params, n_obs=len(X), n_features=len(X.columns))
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
            train_X = X.iloc[start:position]
            train_y = y.iloc[start:position]
            should_refit = (
                last_model is None
                or refit_policy == "every_origin"
                or (refit_policy == "every_n_origins" and (last_fit_position is None or position - last_fit_position >= refit_step))
            )
            if should_refit and refit_policy != "single_fit":
                last_model = _build_l4_model(family, params)
                last_model.fit(train_X, train_y)
                last_fit_position = position
            elif refit_policy == "single_fit" and last_model is None:
                last_model = _build_l4_model(family, params)
                last_model.fit(train_X, train_y)
                last_fit_position = position
            forecast_value = _l4_predict_one(last_model, X, position, forecast_strategy=forecast_strategy, horizon=horizon)
            forecasts[(model_id, target, horizon, origin)] = forecast_value
            origins.append(origin)
            training_windows[(model_id, origin)] = (train_X.index[0], train_X.index[-1])

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
                **{k: params[k] for k in ("alpha", "l1_ratio", "n_estimators", "max_depth", "C") if k in params},
            },
            feature_names=tuple(X.columns),
        )
        benchmark_flags[model_id] = bool(fit_node.get("is_benchmark", False))
        refit_origins[model_id] = tuple(origins)

    sample_index = pd.DatetimeIndex(sorted({key[3] for key in forecasts}))
    forecast_object = _resolve_forecast_object(fit_nodes)
    forecast_intervals: dict[tuple[str, str, int, Any, float], float] = {}
    if forecast_object == "quantile":
        forecast_intervals = _emit_quantile_intervals(forecasts, fit_nodes, X=X, y=y)
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


def _resolve_forecast_object(fit_nodes: list[dict[str, Any]]) -> Literal["point", "quantile", "density"]:
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
    quantile_engine = _native_quantile_engine(family) if quantile_capable else None
    if quantile_engine is not None and isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
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
                        out[(model_id, target, horizon, origin, float(q))] = float(preds[i])
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
    for (model_id, target, horizon, origin), point in forecasts.items():
        for q in levels:
            out[(model_id, target, horizon, origin, float(q))] = float(point + sigma * _stats.norm.ppf(q))
    return out


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
                return GradientBoostingRegressor(loss="quantile", alpha=float(q), n_estimators=100)

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
) -> tuple[str, ModelArtifact, dict[tuple[str, str, int, Any], float], list[Any], dict[tuple[str, Any], tuple[Any, Any]]]:
    """Run the same fit-loop the sequential path uses, but for a single
    ``fit_node``. Used by ``parallel_unit = models`` and (with
    ``parallel_origins = True``) ``parallel_unit = oos_dates``."""

    params = dict(fit_node.get("params", {}) or {})
    if l0_seed is not None and "random_state" not in params:
        params["random_state"] = l0_seed
    family = params.get("family", "ridge")
    forecast_strategy = params.get("forecast_strategy", "direct")
    training_start_rule = params.get("training_start_rule", "expanding")
    refit_policy = params.get("refit_policy", "every_origin")
    rolling_window = int(params.get("rolling_window", max(24, min(len(X) // 2, 120))))
    refit_step = int(params.get("refit_step", 1)) if refit_policy == "every_n_origins" else 1
    if params.get("search_algorithm") in {"cv_path", "grid_search", "random_search", "bayesian_optimization", "genetic_algorithm"}:
        params["_l4_leaf_config"] = raw.get("leaf_config", {}) or {}
        params = _resolve_l4_tuning(params, X, y)
        params.pop("_l4_leaf_config", None)
    min_train_size = _minimal_train_size(params, n_obs=len(X), n_features=len(X.columns))
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
        train_X = X.iloc[start:position]
        train_y = y.iloc[start:position]
        # Issue #279 -- give each origin worker a deterministic per-origin
        # seed derived from the cell-level seed so thread interleaving
        # cannot affect the per-origin RandomForest / xgboost RNG state.
        per_origin_params = dict(params)
        if "random_state" in per_origin_params:
            per_origin_params["random_state"] = (int(per_origin_params["random_state"]) + position) % (2 ** 31 - 1)
        model = _build_l4_model(family, per_origin_params)
        model.fit(train_X, train_y)
        forecast_value = _l4_predict_one(model, X, position, forecast_strategy=forecast_strategy, horizon=horizon)
        return origin, forecast_value, (train_X.index[0], train_X.index[-1])

    if parallel_origins and refit_policy in {"every_origin", "every_n_origins"}:
        # Issue #250 -- fan the walk-forward origin loop across threads.
        from concurrent.futures import ThreadPoolExecutor

        positions = list(range(min_train_size, len(X)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for position, (origin, forecast_value, window) in zip(positions, pool.map(_origin_step, positions)):
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
            train_X = X.iloc[start:position]
            train_y = y.iloc[start:position]
            should_refit = (
                last_model is None
                or refit_policy == "every_origin"
                or (refit_policy == "every_n_origins" and (last_fit_position is None or position - last_fit_position >= refit_step))
            )
            if should_refit and refit_policy != "single_fit":
                last_model = _build_l4_model(family, params)
                last_model.fit(train_X, train_y)
                last_fit_position = position
            elif refit_policy == "single_fit" and last_model is None:
                last_model = _build_l4_model(family, params)
                last_model.fit(train_X, train_y)
                last_fit_position = position
            forecast_value = _l4_predict_one(last_model, X, position, forecast_strategy=forecast_strategy, horizon=horizon)
            forecasts[(model_id, target, horizon, origin)] = forecast_value
            origins.append(origin)
            training_windows[(model_id, origin)] = (train_X.index[0], train_X.index[-1])
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
            **{k: params[k] for k in ("alpha", "l1_ratio", "n_estimators", "max_depth", "C") if k in params},
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
        return LassoCV(cv=int(params.get("cv", 5)), max_iter=int(params.get("max_iter", 20000)), random_state=seed)
    if family == "bayesian_ridge":
        return BayesianRidge()
    if family == "ar_p":
        return _LinearARModel(p=int(params.get("n_lag", params.get("p", 1))))
    if family == "factor_augmented_ar":
        return _FactorAugmentedAR(p=int(params.get("n_lag", 1)), n_factors=int(params.get("n_factors", 3)))
    if family == "factor_augmented_var":
        return _FactorAugmentedVAR(p=int(params.get("n_lag", 2)), n_factors=int(params.get("n_factors", 3)))
    if family == "principal_component_regression":
        return _PrincipalComponentRegression(n_components=int(params.get("n_components", 4)))
    if family == "decision_tree":
        return DecisionTreeRegressor(max_depth=params.get("max_depth"), random_state=seed)
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            random_state=seed,
            n_jobs=1,
        )
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
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
            raise NotImplementedError("xgboost family requires `pip install macroforecast[xgboost]`") from exc
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
            raise NotImplementedError("lightgbm family requires `pip install macroforecast[lightgbm]`") from exc
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
            raise NotImplementedError("catboost family requires `pip install macroforecast[catboost]`") from exc
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
    if family == "knn":
        n_neighbors = int(params.get("n_neighbors", 5))
        return _AutoClipKNN(n_neighbors=n_neighbors, weights=params.get("weights", "uniform"))
    if family == "huber":
        from sklearn.linear_model import HuberRegressor

        return HuberRegressor(epsilon=float(params.get("epsilon", 1.35)), max_iter=int(params.get("max_iter", 1000)))
    if family == "var":
        return _VARWrapper(p=int(params.get("n_lag", 1)))
    if family == "glmboost":
        return _GLMBoost(n_iter=int(params.get("n_estimators", 100)), learning_rate=float(params.get("learning_rate", 0.1)))
    if family in {"bvar_minnesota", "bvar_normal_inverse_wishart"}:
        return _BayesianVAR(
            p=int(params.get("n_lag", 2)),
            prior=family,
            lambda1=float(params.get("minnesota_lambda1", 0.2)),
            lambda_decay=float(params.get("minnesota_lambda_decay", 1.0)),
            lambda_cross=float(params.get("minnesota_lambda_cross", 0.5)),
        )
    if family == "macroeconomic_random_forest":
        # Coulombe (2024) MRF: linear-in-time random forest. Approx via random
        # forest on (X, time_trend) features.
        return _MRFWrapper(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            random_state=seed,
        )
    if family in {"mlp"}:
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (32, 16))),
            max_iter=int(params.get("max_iter", 500)),
            random_state=seed,
        )
    if family in {"lstm", "gru", "transformer"}:
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
    if family == "bagging":
        # Issue #282 -- generic bootstrap-aggregating meta-estimator.
        return _BaggingWrapper(
            base_family=str(params.get("base_family", "ridge")),
            n_estimators=int(params.get("n_estimators", 50)),
            max_samples=float(params.get("max_samples", 0.8)),
            random_state=seed,
            base_params={k: v for k, v in params.items() if k not in {"family", "base_family", "n_estimators", "max_samples"}},
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
            raise RuntimeError(f"custom model {self.spec.name!r} predict() called before fit()")
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


def _l4_predict_one(model, X: pd.DataFrame, position: int, *, forecast_strategy: str, horizon: int) -> float:
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
    if forecast_strategy == "path_average":
        # average h consecutive forecasts (path); equivalent to direct on a
        # cumulative_average target if L3 set it up that way.
        return float(model.predict(X.iloc[[position]])[0])
    return float(model.predict(X.iloc[[position]])[0])


def _resolve_l4_tuning(params: dict[str, Any], X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
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
                picker = LassoCV(alphas=alphas, cv=cv_folds, max_iter=20000, random_state=seed)
                picker.fit(X_filled, y)
                params["alpha"] = float(picker.alpha_)
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
                            trial_params[name] = trial.suggest_float(name, float(low), float(high))
                model = _build_l4_model(family, trial_params)
                from sklearn.model_selection import cross_val_score, TimeSeriesSplit

                scores = cross_val_score(
                    model, X_filled, y, cv=TimeSeriesSplit(n_splits=cv_folds), scoring="neg_mean_squared_error"
                )
                return -float(np.mean(scores))

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
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
                            individual[name] = float(rng.uniform(float(low), float(high)))
                return individual

            def fitness(ind: dict[str, Any]) -> float:
                model = _build_l4_model(family, ind)
                cv = TimeSeriesSplit(n_splits=cv_folds)
                try:
                    return -float(np.mean(cross_val_score(model, X_filled, y, cv=cv, scoring="neg_mean_squared_error")))
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
                    p1, p2 = top[rng.integers(0, len(top))], top[rng.integers(0, len(top))]
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
        return np.array([float(self.intercept_ + float(self.coef_ @ last_window))] * len(X))


class _FactorAugmentedAR:
    """Stock-Watson factor-augmented AR via PCA on X plus AR lags on y."""

    def __init__(self, p: int = 1, n_factors: int = 3) -> None:
        self.p = p
        self.n_factors = n_factors
        self._factor_loadings: np.ndarray | None = None
        self._regression: LinearRegression | None = None
        self._mean: np.ndarray | None = None
        self._last_y: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FactorAugmentedAR":
        from sklearn.decomposition import PCA

        if X.shape[0] < max(self.n_factors, self.p + 1):
            self._regression = LinearRegression().fit(X.fillna(0.0), y)
            self._last_y = np.asarray(y, dtype=float)
            return self
        n = min(self.n_factors, X.shape[1])
        self._mean = X.mean(axis=0).to_numpy()
        pca = PCA(n_components=n, random_state=0)
        factors = pca.fit_transform((X - self._mean).fillna(0.0).to_numpy())
        self._factor_loadings = pca.components_
        ar_lags = pd.concat([y.shift(lag) for lag in range(1, self.p + 1)], axis=1).fillna(0.0).to_numpy()
        design = np.column_stack([factors, ar_lags])
        self._regression = LinearRegression().fit(design, y.values)
        self._last_y = np.asarray(y, dtype=float)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._regression is None or self._factor_loadings is None or self._mean is None:
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
    ) -> None:
        self.p = max(1, int(p))
        self.prior = str(prior)
        self.lambda1 = float(lambda1)
        self.lambda_decay = float(lambda_decay)
        self.lambda_cross = float(lambda_cross)
        # Larger NIW default tightness reflects the parameter-uncertainty
        # adjustment the marginal predictive would apply.
        if self.prior == "bvar_normal_inverse_wishart":
            self.lambda1 *= 1.25
        self._mean: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._feature_names: tuple[str, ...] = ()
        self._fallback: float = 0.0

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

    def _prior(self, columns: tuple[str, ...], sigma2: float) -> tuple[np.ndarray, np.ndarray]:
        classification = self._classify_columns(columns)
        m = np.zeros(len(columns))
        v = np.zeros(len(columns))
        for i, (_base, lag, is_anchor) in enumerate(classification):
            scale = (self.lambda1 / max(1.0, lag) ** self.lambda_decay) ** 2
            if is_anchor:
                m[i] = 1.0
                v[i] = sigma2 * scale
            else:
                v[i] = sigma2 * scale * (self.lambda_cross ** 2 if lag > 0 else 1.0)
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
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            return np.full(len(X), self._fallback)
        # Re-align to the columns seen at fit time (silently zero unseen
        # columns so the predict path matches sklearn's expectation).
        Xmat = np.zeros((len(X), len(self._feature_names)), dtype=float)
        for i, col in enumerate(self._feature_names):
            if col in X.columns:
                Xmat[:, i] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        preds = self._intercept + Xmat @ self._coef
        return preds.astype(float)


class _TorchSequenceModel:
    """Tiny LSTM/GRU/Transformer regressor on lagged feature windows.

    Uses torch when available; otherwise falls back to a sklearn MLPRegressor
    so the operational schema status remains true (the recipe will still run
    end-to-end in lightweight installations).
    """

    def __init__(self, kind: str = "lstm", hidden_size: int = 32, n_epochs: int = 50, random_state: int = 0) -> None:
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
            cell = nn.LSTM(input_size=n_features, hidden_size=self.hidden_size, batch_first=True)
        elif self.kind == "gru":
            cell = nn.GRU(input_size=n_features, hidden_size=self.hidden_size, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(d_model=n_features, nhead=1, dim_feedforward=self.hidden_size, batch_first=True, dropout=0.1)
            cell = nn.TransformerEncoder(layer, num_layers=1)

        class _Wrapped(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cell = cell
                self.head = nn.Linear(self.hidden_size if self.kind != "transformer" else n_features, 1)
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
                self.head = nn.Linear(wrapped_hidden if wrapped_kind != "transformer" else n_features, 1)

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

    def _fit_mixed_frequency(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Issue #245 -- DynamicFactorMQ with monthly aggregator. Returns
        True when the MQ fit succeeded; False when the caller should fall
        back to the single-frequency path."""

        try:
            from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
        except ImportError:
            return False
        endog = pd.concat([y.rename("__y__"), X], axis=1).dropna(how="all")
        if endog.shape[0] < 12 or endog.shape[1] < 2:
            return False
        # Build the M / Q split per documented column_frequencies; default
        # to monthly when unspecified.
        monthly = []
        quarterly = []
        for col in endog.columns:
            if col == "__y__":
                # Honour the target's declared frequency.
                tag = self.column_frequencies.get(str(col)) or self.column_frequencies.get("target", "monthly")
            else:
                tag = self.column_frequencies.get(str(col), "monthly")
            (quarterly if str(tag).lower() == "quarterly" else monthly).append(col)
        if not monthly:  # MQ requires at least one monthly variable
            return False
        scaler_mean = endog.mean()
        scaler_std = endog.std(ddof=0).replace(0.0, 1.0)
        normalised = (endog - scaler_mean) / scaler_std
        # Issue #274 -- Mariano-Murasawa (2010) Eq. (4) specifies AR(1)
        # idiosyncratic errors. Try the published spec first; fall back
        # to the iid form when the optimisation diverges.
        for idiosyncratic_ar1 in (True, False):
            try:
                model = DynamicFactorMQ(
                    normalised[monthly],
                    endog_quarterly=normalised[quarterly] if quarterly else None,
                    k_endog_monthly=len(monthly),
                    factors=self.n_factors,
                    factor_orders=self.factor_order,
                    idiosyncratic_ar1=idiosyncratic_ar1,
                    standardize=False,
                )
                self._results = model.fit(disp=False, maxiter=20)
                self._idiosyncratic_ar1 = idiosyncratic_ar1
                break
            except Exception:  # pragma: no cover - DFMQ is fragile on small data
                continue
        else:
            return False
        self._scaler_mean = float(scaler_mean["__y__"])
        self._scaler_std = float(scaler_std["__y__"])
        self._mode = "mixed_frequency_mq"
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
                target_pred = float(forecast["__y__"].iloc[0]) * self._scaler_std + self._scaler_mean
            else:
                # MQ returns a Series-like; pull the first value directly.
                target_pred = float(np.asarray(forecast).ravel()[0]) * self._scaler_std + self._scaler_mean
        except Exception:
            target_pred = self._fallback
        return np.full(len(X), target_pred, dtype=float)


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
        out: dict[float, np.ndarray] = {q: np.empty(n_pred) for q in self.quantile_levels}
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
    quantile bands across the bag for ``predict_quantiles``."""

    def __init__(
        self,
        base_family: str = "ridge",
        n_estimators: int = 50,
        max_samples: float = 0.8,
        random_state: int = 0,
        base_params: dict[str, Any] | None = None,
    ) -> None:
        self.base_family = str(base_family)
        self.n_estimators = int(n_estimators)
        self.max_samples = float(max_samples)
        self.random_state = int(random_state)
        self.base_params = dict(base_params or {})
        self._models: list[Any] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BaggingWrapper":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        sample_size = max(2, int(round(self.max_samples * n)))
        for i in range(self.n_estimators):
            idx = rng.choice(n, size=sample_size, replace=True)
            sub_X = X.iloc[idx]
            sub_y = y.iloc[idx]
            params = dict(self.base_params)
            params["random_state"] = (self.random_state + i) % (2 ** 31 - 1)
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

    def predict_quantiles(self, X: pd.DataFrame, levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[float, np.ndarray]:
        if not self._models:
            return {q: np.zeros(len(X)) for q in levels}
        preds = np.column_stack([m.predict(X) for m in self._models])
        return {q: np.quantile(preds, q, axis=1) for q in levels}


class _MRFWrapper:
    """Macroeconomic random forest (Coulombe 2024) -- generalised time-varying
    parameter (GTVP) estimator.

    Issue #187. The published procedure trains a tree forest where the
    splitting features include a time / regime indicator, and *each leaf
    fits a local linear regression of y on X*. The forest prediction is
    the average of leaf-local linear predictions.

    Implementation:

    1. Augment ``X`` with a normalized time-trend column.
    2. Fit a sklearn ``RandomForestRegressor`` on the augmented design --
       this provides the (data-driven, time-aware) partitioning.
    3. For every (tree, leaf) pair, fit a local linear regression of
       ``y`` on the original ``X`` columns using the training rows that
       land in that leaf. Singleton / collinear leaves fall back to the
       leaf's mean target value.
    4. At predict time, route every sample through every tree to its
       leaf, evaluate the leaf-local linear model, and average across
       the forest.

    This is the GTVP form documented in the design table -- not the
    plain ``RandomForest + time_trend`` of v0.1. Promoted FUTURE ->
    OPERATIONAL.
    """

    def __init__(self, n_estimators: int = 200, max_depth: Any = None, random_state: int = 0) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self._forest: RandomForestRegressor | None = None
        self._n_train: int | None = None
        # Per-tree dict: leaf_id -> LinearRegression OR float (mean fallback)
        self._leaf_models: list[dict[int, Any]] = []
        self._feature_columns: tuple[str, ...] = ()
        self._global_fallback: float = 0.0

    def _augment(self, X: pd.DataFrame, offset: int) -> np.ndarray:
        time_trend = (np.arange(len(X), dtype=float) + offset).reshape(-1, 1)
        return np.concatenate([X.fillna(0.0).to_numpy(dtype=float), time_trend], axis=1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_MRFWrapper":
        self._feature_columns = tuple(X.columns)
        self._n_train = len(X)
        if self._n_train == 0:
            self._global_fallback = float(np.nan_to_num(y.mean(), nan=0.0))
            return self
        augmented = self._augment(X, offset=0)
        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=1,
        )
        target = np.asarray(y, dtype=float)
        self._forest.fit(augmented, target)
        self._global_fallback = float(target.mean())

        # Per (tree, leaf) local linear fit.
        train_leaves = self._forest.apply(augmented)
        self._leaf_models = []
        X_arr = X.fillna(0.0).to_numpy(dtype=float)
        for tree_idx in range(train_leaves.shape[1]):
            tree_leaves = train_leaves[:, tree_idx]
            leaf_dict: dict[int, Any] = {}
            for leaf_id in np.unique(tree_leaves):
                mask = tree_leaves == leaf_id
                count = int(mask.sum())
                # Need n_features + 1 rows for a stable linear fit; otherwise
                # store the leaf-mean so prediction stays well-defined.
                if count > X_arr.shape[1] + 1:
                    try:
                        lr = LinearRegression()
                        lr.fit(X_arr[mask], target[mask])
                        leaf_dict[int(leaf_id)] = lr
                    except Exception:
                        leaf_dict[int(leaf_id)] = float(target[mask].mean())
                else:
                    leaf_dict[int(leaf_id)] = float(target[mask].mean()) if count else self._global_fallback
            self._leaf_models.append(leaf_dict)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._forest is None or self._n_train is None:
            return np.full(len(X), self._global_fallback)
        # Re-align prediction matrix to the columns seen at fit time.
        aligned = pd.DataFrame(index=X.index, columns=self._feature_columns)
        for col in self._feature_columns:
            if col in X.columns:
                aligned[col] = pd.to_numeric(X[col], errors="coerce")
        aligned = aligned.fillna(0.0)
        augmented = self._augment(aligned, offset=self._n_train)
        leaves = self._forest.apply(augmented)
        n_pred = len(aligned)
        n_trees = leaves.shape[1]
        preds = np.zeros(n_pred, dtype=float)
        X_arr = aligned.to_numpy(dtype=float)
        for tree_idx in range(n_trees):
            leaf_models = self._leaf_models[tree_idx]
            tree_leaves = leaves[:, tree_idx]
            for i in range(n_pred):
                model = leaf_models.get(int(tree_leaves[i]))
                if isinstance(model, LinearRegression):
                    preds[i] += float(model.predict(X_arr[i : i + 1])[0])
                elif model is not None:
                    preds[i] += float(model)
                else:
                    preds[i] += self._global_fallback
        return preds / max(n_trees, 1)


def _add_l5_relative_metrics(metrics: pd.DataFrame, l4_models: L4ModelArtifactsArtifact | None) -> pd.DataFrame:
    if l4_models is None:
        return metrics
    benchmark_ids = [model_id for model_id, is_benchmark in l4_models.is_benchmark.items() if is_benchmark]
    if len(benchmark_ids) != 1:
        return metrics
    benchmark_id = benchmark_ids[0]
    benchmark = metrics.loc[metrics["model_id"] == benchmark_id, ["target", "horizon", "mse", "mae"]].rename(
        columns={"mse": "benchmark_mse", "mae": "benchmark_mae"}
    )
    if benchmark.empty:
        return metrics
    result = metrics.merge(benchmark, on=["target", "horizon"], how="left")
    result["relative_mse"] = result["mse"] / result["benchmark_mse"]
    result["r2_oos"] = 1.0 - result["relative_mse"]
    result["relative_mae"] = result["mae"] / result["benchmark_mae"]
    result["mse_reduction"] = result["benchmark_mse"] - result["mse"]
    return result


def _l5_ranking_metric(metrics: pd.DataFrame, resolved_axes: dict[str, Any]) -> str:
    if resolved_axes.get("ranking") == "by_relative_metric" and "relative_mse" in metrics.columns:
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
        return per_origin.assign(subperiod="full_oos").groupby(
            ["model_id", "target", "horizon", "subperiod"], as_index=False
        ).agg(mse=("squared_error", "mean"), mae=("absolute_error", "mean"))
    cuts = sorted(pd.to_datetime(boundaries))
    origins = pd.to_datetime(per_origin["origin"])
    edges = [pd.Timestamp.min, *cuts, pd.Timestamp.max]
    labels = [f"sp_{i}" for i in range(len(edges) - 1)]
    subperiod = pd.cut(origins, bins=edges, labels=labels, include_lowest=True)
    expanded = per_origin.assign(subperiod=subperiod.astype(str))
    return expanded.groupby(["model_id", "target", "horizon", "subperiod"], as_index=False).agg(
        mse=("squared_error", "mean"), mae=("absolute_error", "mean")
    )


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
        isinstance(X, pd.DataFrame) and isinstance(y, pd.Series) and not X.empty and len(y) > len(blocks) + 1
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
                    cols.extend(c for c in block_map[blocks[k]] if c in X_aligned.columns)
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
                        coalition_share = sum(sizes[k] for k in subset) / total_size if subset else 0.0
                        weight = 1.0 / (n * comb(n - 1, size))
                        for i in range(n):
                            if i in subset:
                                continue
                            new_share = (sum(sizes[k] for k in subset) + sizes[i]) / total_size
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
                    "block_mse_contribution": float(raw_share if can_refit else share * median_mse),
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
    actual = raw_actual if isinstance(raw_actual, pd.Series) else l3_features.y_final.metadata.values.get("data")
    if not isinstance(actual, pd.Series):
        raise ValueError("minimal L5 runtime requires L3 y_final series data")
    # Inverse-transform the L4 forecasts back to the raw scale.
    forecasts_raw = _apply_inverse_target_transform(l4_forecasts.forecasts, l3_features)
    rows: list[dict[str, Any]] = []
    for (model_id, target, horizon, origin), forecast in forecasts_raw.items():
        if origin not in actual.index:
            continue
        error = float(actual.loc[origin]) - float(forecast)
        rows.append(
            {
                "model_id": model_id,
                "target": target,
                "horizon": horizon,
                "origin": origin,
                "squared_error": error**2,
                "absolute_error": abs(error),
            }
        )
    if not rows:
        metrics = pd.DataFrame(columns=["model_id", "target", "horizon", "mse", "rmse", "mae"])
        per_origin = pd.DataFrame(columns=["model_id", "target", "horizon", "origin", "squared_error", "absolute_error"])
    else:
        errors = pd.DataFrame(rows)
        per_origin = errors[["model_id", "target", "horizon", "origin", "squared_error", "absolute_error"]].copy()
        metrics = errors.groupby(["model_id", "target", "horizon"], as_index=False).agg(mse=("squared_error", "mean"), mae=("absolute_error", "mean"))
        metrics["rmse"] = metrics["mse"] ** 0.5
        metrics = _add_l5_relative_metrics(metrics, l4_models)
    if metrics.empty:
        ranking = pd.DataFrame()
        resolved_axes = dict(l5_layer.resolve_axes_from_raw(raw.get("fixed_axes", {}) or {}, context=context))
    else:
        resolved_axes = dict(l5_layer.resolve_axes_from_raw(raw.get("fixed_axes", {}) or {}, context=context))
        ranking_metric = _l5_ranking_metric(metrics, resolved_axes)
        ranking = metrics.sort_values(ranking_metric, ascending=_l5_rank_ascending(ranking_metric)).assign(
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
                decomposition_tables["by_predictor_block"] = _l5_predictor_block_decomposition(
                    metrics,
                    block_map,
                    X=l3_features.X_final.data,
                    y=l3_features.y_final.metadata.values.get("data") if isinstance(l3_features.y_final.metadata.values.get("data"), pd.Series) else None,
                )
        if agg_horizon == "per_horizon_then_mean" and not metrics.empty:
            decomposition_tables["per_horizon_then_mean"] = (
                metrics.groupby("horizon", as_index=False).mean(numeric_only=True)
            )
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
        return L6TestsArtifact(test_metadata={"runtime": "core_l6_disabled"}, l6_axis_resolved=axes), axes

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
        equal_results = _l6_equal_predictive_results(errors, resolved["L6_A_equal_predictive"], raw.get("leaf_config", {}) or {}, l4_models)
    if resolved["L6_B_nested"]["enabled"]:
        nested_results = _l6_nested_results(errors, resolved["L6_B_nested"], raw.get("leaf_config", {}) or {}, l4_models)
    if resolved["L6_C_cpa"]["enabled"]:
        cpa_results = _l6_cpa_results(errors, resolved["L6_C_cpa"], l4_models)
    if resolved["L6_D_multiple_model"]["enabled"]:
        multiple_results = _l6_multiple_model_results(l5_eval.metrics_table, resolved["L6_D_multiple_model"], per_origin_panel=l5_eval.per_origin_loss_panel)
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
        direction_results = _l6_direction_results(errors, resolved["L6_F_direction"], raw.get("leaf_config", {}) or {})
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
    report = l7_layer.validate_layer(raw, recipe_context=l7_layer._recipe_context(recipe_root))
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
    values = _execute_l7_nodes(raw, l3_features, l3_metadata, l4_forecasts, l4_models, l5_eval, l6_tests)
    importance = L7ImportanceArtifact(computation_metadata={"runtime": "core_l7_minimal", "axis_resolved": axes})
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
                    horizon = value.attrs.get("horizon", l3_features.horizon_set[0] if l3_features.horizon_set else 1)
                    key = (model_id, target, int(horizon), method)
                    if "group" in value.columns or label.startswith("group"):
                        group_importance[key + (value.attrs.get("grouping", label),)] = value
                    elif "pipeline" in value.columns or label.startswith("pipeline"):
                        lineage_importance[key + (value.attrs.get("level", label),)] = value
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
            values[node.id] = _execute_l7_source(node.selector, l3_features, l3_metadata, l4_forecasts, l4_models, l5_eval, l6_tests)
        elif node.type == "step":
            inputs = [values[ref.node_id] for ref in node.inputs]
            values[node.id] = _execute_l7_step(node.op, inputs, node.params, l3_features, l3_metadata, l5_eval)
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
    raise NotImplementedError(f"minimal L7 runtime does not support source {selector.layer_ref}.{selector.sink_name}")


def _execute_l7_step(op: str, inputs: list[Any], params: dict[str, Any], l3_features: L3FeaturesArtifact, l3_metadata: L3MetadataArtifact, l5_eval: L5EvaluationArtifact) -> Any:
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
        frame = _partial_dependence_table(model, X, n_grid=int(params.get("n_grid", 20)))
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
        frame = _bootstrap_jackknife_frame(model, X, y, n_replications=int(params.get("n_replications", 50)))
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op == "rolling_recompute":
        model = _first_model_input(inputs)
        X, y = _l7_xy(inputs, l3_features)
        frame = _rolling_importance_table(model, X, y, window=int(params.get("window_size", max(8, len(X) // 4))))
        return _attach_l7_attrs(frame, model, op, l3_features)
    if op in {"forecast_decomposition"}:
        model = _first_model_input(inputs)
        X, _ = _l7_xy(inputs, l3_features)
        frame = _forecast_decomposition_frame(model, X)
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
        table = next((item for item in inputs if isinstance(item, pd.DataFrame)), pd.DataFrame())
        return _l7_group_aggregate(table, params)
    if op == "lineage_attribution":
        table = next((item for item in inputs if isinstance(item, pd.DataFrame)), pd.DataFrame())
        metadata = next((item for item in inputs if isinstance(item, L3MetadataArtifact)), l3_metadata)
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
    if op == "fevd" or op == "historical_decomposition" or op == "generalized_irf":
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


def _l7_xy(inputs: list[Any], l3_features: L3FeaturesArtifact) -> tuple[pd.DataFrame, pd.Series | None]:
    X = next((item for item in inputs if isinstance(item, pd.DataFrame)), l3_features.X_final.data)
    y_in = next((item for item in inputs if isinstance(item, pd.Series)), l3_features.y_final.metadata.values.get("data"))
    return X, y_in if isinstance(y_in, pd.Series) else None


def _shap_importance_frame(model: ModelArtifact, X: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """SHAP via the optional `shap` package; falls back to a coefficient or
    permutation proxy if `shap` is not installed."""

    try:
        import shap  # type: ignore
    except ImportError:
        if hasattr(model.fitted_object, "coef_"):
            return _linear_importance_frame(model, method=kind)
        return _permutation_importance_frame(model, X, None, method=kind)
    fitted = model.fitted_object
    try:
        if kind == "shap_linear" and hasattr(fitted, "coef_"):
            explainer = shap.LinearExplainer(fitted, X.fillna(0.0))
        elif kind in {"shap_tree", "shap_interaction"} and hasattr(fitted, "feature_importances_"):
            explainer = shap.TreeExplainer(fitted)
        else:
            explainer = shap.KernelExplainer(fitted.predict, shap.sample(X.fillna(0.0), min(50, len(X))))
        values = explainer.shap_values(X.fillna(0.0))
    except Exception:
        return _permutation_importance_frame(model, X, None, method=kind)
    importance = np.abs(values).mean(axis=0) if isinstance(values, np.ndarray) else np.abs(np.asarray(values)).mean(axis=0)
    return pd.DataFrame({"feature": list(model.feature_names), "importance": [float(v) for v in importance], "coefficient": None})


def _tree_importance_frame(model: ModelArtifact) -> pd.DataFrame:
    fitted = model.fitted_object
    importance = getattr(fitted, "feature_importances_", None)
    if importance is None:
        if hasattr(fitted, "coef_"):
            return _linear_importance_frame(model, method="tree_importance_proxy")
        return pd.DataFrame({"feature": list(model.feature_names), "importance": [0.0] * len(model.feature_names), "coefficient": None})
    return pd.DataFrame({"feature": list(model.feature_names), "importance": [float(v) for v in np.asarray(importance).ravel()], "coefficient": None})


def _partial_dependence_table(model: ModelArtifact, X: pd.DataFrame, *, n_grid: int) -> pd.DataFrame:
    fitted = model.fitted_object
    rows = []
    for column in X.columns:
        series = X[column].dropna()
        if series.empty:
            rows.append({"feature": column, "importance": 0.0, "coefficient": None})
            continue
        grid = np.linspace(series.quantile(0.05), series.quantile(0.95), max(2, int(n_grid)))
        responses = []
        for value in grid:
            edited = X.fillna(0.0).copy()
            edited[column] = value
            try:
                response = float(np.mean(fitted.predict(edited)))
            except Exception:
                response = 0.0
            responses.append(response)
        rows.append({"feature": column, "importance": float(max(responses) - min(responses)), "coefficient": None})
    return pd.DataFrame(rows)


def _ale_table(model: ModelArtifact, X: pd.DataFrame, *, n_quantiles: int) -> pd.DataFrame:
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
            rows.append({"feature": column, "importance": 0.0, "coefficient": None, "ale_function": []})
            continue
        quantiles = np.quantile(series, np.linspace(0, 1, max(3, int(n_quantiles) + 1)))
        bin_edges = np.unique(quantiles)
        if len(bin_edges) < 3:
            rows.append({"feature": column, "importance": 0.0, "coefficient": None, "ale_function": []})
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
                effect = float(np.mean(fitted.predict(edited_high) - fitted.predict(edited_low)))
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


def _friedman_h_table(model: ModelArtifact, X: pd.DataFrame, *, n_grid: int = 8) -> pd.DataFrame:
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
        left_grid = np.quantile(X_filled[left].to_numpy(), np.linspace(0, 1, max(2, n_grid)))
        for right in columns[i + 1 :]:
            right_grid = np.quantile(X_filled[right].to_numpy(), np.linspace(0, 1, max(2, n_grid)))
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
                        f_jk += _centred_pd((left, np.full(n_rows, v_l)), (right, np.full(n_rows, v_r)))
                        count += 1
                f_jk /= max(count, 1)
                num = float(np.sum((f_jk - f_j - f_k) ** 2))
                denom = float(np.sum(f_jk ** 2))
                h_sq = num / denom if denom > 0 else 0.0
            except Exception:
                h_sq = 0.0
            rows.append({"feature": f"{left}*{right}", "importance": float(np.sqrt(max(0.0, min(1.0, h_sq)))), "coefficient": None})
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
            {"feature": feature_names, "importance": [0.0] * len(feature_names), "coefficient": None}
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
    X_arr = aligned.drop(columns="__y__")[feature_names].fillna(0.0).to_numpy(dtype=float)
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
        boot_freq = boot_counts / boot_ok if boot_ok else (np.abs(coef_arr) > 1e-9).astype(float)
        roll_freq = roll_counts / roll_ok if roll_ok else (np.abs(coef_arr) > 1e-9).astype(float)
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
    sampling_meta_key = "n_rolling_windows_run" if sampling == "rolling" else "n_bootstraps_run"
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": inclusion.tolist(),
            "coefficient": [float(c) for c in coef_arr],
            sampling_meta_key: [successful] * len(feature_names),
        }
    )


def _cumulative_r2_frame(model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
    if y is None or not hasattr(model.fitted_object, "predict"):
        return _linear_importance_frame(model, method="cumulative_r2")
    aligned = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame({"feature": list(X.columns), "importance": [0.0] * len(X.columns), "coefficient": None})
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
        rows.append({"feature": column, "importance": float(max(0.0, r2 - cum_r2)), "coefficient": None})
        cum_r2 = max(cum_r2, r2)
    return pd.DataFrame(rows)


def _bootstrap_jackknife_frame(model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None, *, n_replications: int) -> pd.DataFrame:
    if y is None:
        return _linear_importance_frame(model, method="bootstrap_jackknife")
    aligned = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame({"feature": list(X.columns), "importance": [0.0] * len(X.columns), "coefficient": None})
    rng = np.random.default_rng(0)
    importances = []
    for _ in range(max(2, int(n_replications))):
        sample = aligned.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 2**32 - 1)))
        boot_frame = _permutation_importance_frame(model, sample[X.columns], sample["__y__"], method="permutation_importance")
        importances.append(boot_frame.set_index("feature")["importance"])
    matrix = pd.concat(importances, axis=1)
    summary = matrix.agg(["mean", "std"], axis=1).rename(columns={"mean": "importance", "std": "importance_std"}).reset_index()
    summary["coefficient"] = None
    return summary


def _rolling_importance_table(model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None, *, window: int) -> pd.DataFrame:
    if y is None or not hasattr(model.fitted_object, "predict"):
        return _linear_importance_frame(model, method="rolling_recompute")
    aligned = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    if len(aligned) <= window:
        return _permutation_importance_frame(model, X, y, method="rolling_recompute")
    importances: dict[Any, pd.Series] = {}
    for end in range(window, len(aligned) + 1, max(1, window // 4)):
        sub = aligned.iloc[max(0, end - window) : end]
        frame = _permutation_importance_frame(model, sub[X.columns], sub["__y__"], method="rolling_recompute")
        importances[sub.index[-1]] = frame.set_index("feature")["importance"]
    matrix = pd.DataFrame(importances)
    out = matrix.mean(axis=1).reset_index().rename(columns={0: "importance"})
    out.columns = ["feature", "importance"]
    out["coefficient"] = None
    return out


def _forecast_decomposition_frame(model: ModelArtifact, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contribution to the latest prediction (linear models only)."""

    fitted = model.fitted_object
    coef = getattr(fitted, "coef_", None)
    if coef is None:
        return _tree_importance_frame(model)
    last = X.iloc[[-1]].fillna(0.0).to_numpy().ravel()
    contributions = np.asarray(coef).ravel() * last
    return pd.DataFrame({"feature": list(model.feature_names), "importance": [float(abs(c)) for c in contributions], "coefficient": [float(v) for v in contributions]})


def _gradient_attribution_frame(model: ModelArtifact, X: pd.DataFrame, *, kind: str) -> pd.DataFrame:
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
    attribs = attributions.detach().cpu().numpy().reshape(x_arr.shape[0], x_arr.shape[1])
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

    For each prediction sample, route through every tree to its leaf,
    grab the leaf-local linear regression's coefficient vector, and
    average across the forest. This yields a per-row coefficient
    estimate β̂(t) for each feature -- the GTVP series the design
    promised.

    Importance: time-average of |β̂(t)| per feature. The full per-row
    coefficient table is exposed via the ``coefficient_path`` column for
    downstream renderers.
    """

    fitted = model.fitted_object
    feature_names = list(model.feature_names)
    forest = getattr(fitted, "_forest", None)
    leaf_models = getattr(fitted, "_leaf_models", None)
    n_train = getattr(fitted, "_n_train", None)
    if forest is None or leaf_models is None or n_train is None:
        # Not an _MRFWrapper -- fall back to tree importance.
        frame = _tree_importance_frame(model)
        if "status" not in frame.columns:
            frame["status"] = "fallback_not_mrf"
        return frame

    aligned = pd.DataFrame(index=X.index, columns=feature_names)
    for col in feature_names:
        if col in X.columns:
            aligned[col] = pd.to_numeric(X[col], errors="coerce")
    aligned = aligned.fillna(0.0)
    augmented = fitted._augment(aligned, offset=n_train)
    leaves = forest.apply(augmented)
    n_pred = len(aligned)
    n_features = len(feature_names)
    coef_path = np.zeros((n_pred, n_features), dtype=float)
    n_trees = leaves.shape[1]
    for tree_idx in range(n_trees):
        tree_models = leaf_models[tree_idx]
        tree_leaves = leaves[:, tree_idx]
        for i in range(n_pred):
            leaf = tree_models.get(int(tree_leaves[i]))
            if isinstance(leaf, LinearRegression):
                coef_path[i] += leaf.coef_
    coef_path /= max(n_trees, 1)
    importance = np.abs(coef_path).mean(axis=0)
    rows = []
    for j, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "importance": float(importance[j]),
                "coefficient": None,
                "coefficient_path": [float(v) for v in coef_path[:, j]],
                "status": "operational",
            }
        )
    return pd.DataFrame(rows)


def _var_impulse_frame(model: ModelArtifact, *, op_name: str, n_periods: int = 12) -> pd.DataFrame:
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
        target_index = list(fitted_results.names).index("__y__") if "__y__" in fitted_results.names else 0
    except Exception:
        target_index = 0

    rows: list[dict[str, Any]] = []
    try:
        if op_name == "fevd":
            fevd = fitted_results.fevd(int(n_periods))
            # decomp shape: (n_periods, n_vars, n_vars). Last axis is the
            # contribution of variable j to the variance of variable i.
            decomp = np.asarray(fevd.decomp, dtype=float)
            shares = decomp[:, target_index, :].mean(axis=0)  # avg over horizons
            for j, name in enumerate(fitted_results.names):
                if j >= len(shares):
                    break
                rows.append({"feature": str(name), "importance": float(shares[j]), "coefficient": None, "status": "operational"})
        elif op_name == "generalized_irf":
            irf = fitted_results.irf(int(n_periods))
            # Use orthogonalised IRFs (orth_irfs shape: (n_periods+1, n_vars, n_vars)).
            irfs = np.asarray(irf.orth_irfs, dtype=float)
            # Sum of absolute responses of target to each shock.
            response = np.abs(irfs[:, target_index, :]).sum(axis=0)
            for j, name in enumerate(fitted_results.names):
                if j >= len(response):
                    break
                rows.append({"feature": str(name), "importance": float(response[j]), "coefficient": None, "status": "operational"})
        else:  # historical_decomposition
            # statsmodels does not expose a ready-made historical
            # decomposition. We build one via the cumulative impact of
            # each shock on the target using orthogonalised impulses
            # weighted by the sample residuals.
            irf = fitted_results.irf(int(n_periods))
            irfs = np.asarray(irf.orth_irfs, dtype=float)
            resid = np.asarray(fitted_results.resid, dtype=float)
            # Contribution of shock j to target i: sum over time of
            # |IRF[h, i, j]| * |std(resid[:, j])|. Approximate.
            stds = resid.std(axis=0, ddof=0)
            response = (np.abs(irfs[:, target_index, :]).sum(axis=0)) * stds
            for j, name in enumerate(fitted_results.names):
                if j >= len(response):
                    break
                rows.append({"feature": str(name), "importance": float(response[j]), "coefficient": None, "status": "operational"})
    except Exception as exc:
        return pd.DataFrame(
            {
                "feature": feature_names[: 1],
                "importance": [0.0],
                "coefficient": [None],
                "status": [f"error: {type(exc).__name__}"],
            }
        )

    if not rows:
        return pd.DataFrame({"feature": [], "importance": [], "coefficient": [], "status": []})
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


def _permutation_importance_frame(model: ModelArtifact, X: pd.DataFrame, y: pd.Series | None, *, method: str) -> pd.DataFrame:
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
        rows.append({"feature": column, "importance": float(loss - baseline), "coefficient": None})
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
                bins = pd.qcut(X_eval[cond_col], q=n_bins, labels=False, duplicates="drop")
            except Exception:
                bins = pd.Series(np.zeros(len(X_eval), dtype=int), index=X_eval.index)
            for bin_id in bins.dropna().unique():
                mask = (bins == bin_id).values
                if mask.sum() <= 1:
                    continue
                values = permuted.loc[mask, column].values
                permuted.loc[mask, column] = rng.permutation(values)
        loss = float(((y_eval - model.fitted_object.predict(permuted)) ** 2).mean())
        rows.append({"feature": column, "importance": float(loss - baseline), "coefficient": None, "method": "strobl_conditional"})
    return pd.DataFrame(rows)


def _attach_l7_attrs(frame: pd.DataFrame, model: ModelArtifact, method: str, l3_features: L3FeaturesArtifact) -> pd.DataFrame:
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    frame.attrs.update({"method": method, "model_id": model.model_id, "target": l3_features.y_final.name, "horizon": l3_features.horizon_set[0] if l3_features.horizon_set else 1})
    return frame


def _l7_group_aggregate(table: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    grouping = params.get("grouping", "user_defined")
    result = table.copy()
    result["group"] = result["feature"].map(lambda feature: str(feature).split("_")[0])
    grouped = result.groupby("group", as_index=False)["importance"].sum()
    grouped.attrs.update(table.attrs)
    grouped.attrs["grouping"] = grouping
    return grouped


def _l7_lineage_attribution(table: pd.DataFrame, metadata: L3MetadataArtifact, params: dict[str, Any]) -> pd.DataFrame:
    level = params.get("level", "pipeline_name")
    rows = []
    for _, row in table.iterrows():
        lineage = metadata.column_lineage.get(str(row["feature"]))
        pipeline = lineage.pipeline_id if lineage and lineage.pipeline_id else "unknown"
        rows.append({"pipeline": pipeline, "importance": float(row["importance"])})
    result = pd.DataFrame(rows).groupby("pipeline", as_index=False)["importance"].sum() if rows else pd.DataFrame(columns=["pipeline", "importance"])
    result.attrs.update(table.attrs)
    result.attrs["level"] = level
    return result


def _l7_transformation_attribution(l5_eval: L5EvaluationArtifact, params: dict[str, Any]) -> L7TransformationAttributionArtifact:
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
            summary_table=pd.DataFrame(columns=["pipeline", "target", "horizon", "contribution"]),
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
        if method in {"shapley_over_pipelines", "shapley_over_pipelines_sampled"} and n <= 8 and method != "shapley_over_pipelines_sampled":
            shapley = np.zeros(n)
            indices = list(range(n))
            for size in range(n):
                for subset in combinations(indices, size):
                    subset_set = set(subset)
                    coalition_loss = float(np.mean([losses[k] for k in subset])) if subset else 0.0
                    weight = 1.0 / (n * comb(n - 1, size))
                    for i in indices:
                        if i in subset_set:
                            continue
                        new_subset = list(subset) + [i]
                        new_loss = float(np.mean([losses[k] for k in new_subset]))
                        # Marginal contribution: improvement (loss reduction).
                        shapley[i] += weight * (coalition_loss - new_loss) if subset else weight * (-new_loss)
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
                running_loss = 0.0
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
                {"pipeline": pipeline, "target": target, "horizon": int(horizon), "contribution": float(value)}
            )
            contributions[(target, int(horizon), pipeline)] = float(value)
    summary = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["pipeline", "target", "horizon", "contribution"])
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


def _l6_error_frame(l4_forecasts: L4ForecastsArtifact, actual: pd.Series) -> pd.DataFrame:
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


def _l6_pair_list(sub: dict[str, Any], leaf: dict[str, Any], model_ids: list[str], l4_models: L4ModelArtifactsArtifact) -> list[tuple[str, str]]:
    if sub.get("model_pair_strategy") == "user_list" or sub.get("nested_pair_strategy") == "user_list":
        key = "pair_user_list" if "model_pair_strategy" in sub else "nested_pair_user_list"
        return [tuple(pair) for pair in leaf.get(key, [])]
    benchmark_ids = [model_id for model_id, is_benchmark in l4_models.is_benchmark.items() if is_benchmark]
    if benchmark_ids:
        benchmark_id = benchmark_ids[0]
        return [(model_id, benchmark_id) for model_id in model_ids if model_id != benchmark_id]
    return [(left, right) for index, left in enumerate(model_ids) for right in model_ids[index + 1 :]]


def _l6_equal_predictive_results(
    errors: pd.DataFrame, sub: dict[str, Any], leaf: dict[str, Any], l4_models: L4ModelArtifactsArtifact
) -> dict[tuple[Any, ...], Any]:
    model_ids = sorted(errors["model_id"].unique()) if not errors.empty else []
    pairs = _l6_pair_list(sub, leaf, model_ids, l4_models)
    results: dict[tuple[Any, ...], Any] = {}
    tests = ["dm_diebold_mariano", "gw_giacomini_white"] if sub.get("equal_predictive_test") == "multi" else [sub.get("equal_predictive_test")]
    # Issue #283 -- when the recipe asks for the Diebold-Mariano-Pesaran
    # joint multi-horizon test, run it once per (model_a, model_b) pair
    # before the per-horizon DM loop and stash the results.
    dmp_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    if "dmp_multi_horizon" in tests or sub.get("equal_predictive_test") == "dmp_multi_horizon":
        dmp_results = _l6_dmp_multi_horizon(errors, pairs, leaf.get("dependence_correction", "newey_west"))
        tests = [t for t in tests if t != "dmp_multi_horizon"]
    loss_col = "absolute" if sub.get("loss_function") == "absolute" else "squared"
    # Issue #259 -- HAC kernel from L6 globals (newey_west / andrews / parzen).
    hac_kernel = leaf.get("dependence_correction", "newey_west")
    apply_hln = bool(sub.get("hln_correction", True))
    for test_name in tests:
        for model_a, model_b in pairs:
            for (target, horizon), group in errors.groupby(["target", "horizon"]):
                left = group.loc[group["model_id"] == model_a, ["origin", loss_col]].rename(columns={loss_col: "loss_a"})
                right = group.loc[group["model_id"] == model_b, ["origin", loss_col]].rename(columns={loss_col: "loss_b"})
                joined = left.merge(right, on="origin", how="inner")
                diff = joined["loss_a"] - joined["loss_b"]
                stat, p_value = _diebold_mariano_test(diff, horizon=int(horizon), hln=apply_hln, kernel=hac_kernel)
                results[(test_name, model_a, model_b, target, int(horizon))] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "decision_at_5pct": p_value is not None and p_value < 0.05,
                    "n_obs": int(diff.notna().sum()),
                    "mean_loss_difference": _float_or_none(diff.mean()) if not diff.empty else None,
                    "hln_correction": apply_hln,
                }
    # Stash the DMP joint-test results next to the DM per-horizon entries.
    for key, payload in dmp_results.items():
        results[key] = payload
    return results


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
                left = group.loc[group["model_id"] == model_a, ["origin", "squared"]].rename(columns={"squared": "loss_a"})
                right = group.loc[group["model_id"] == model_b, ["origin", "squared"]].rename(columns={"squared": "loss_b"})
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
            out[("dmp_multi_horizon", model_a, model_b, target)] = {
                "statistic": stat,
                "p_value": p_value,
                "decision_at_5pct": bool(p_value < 0.05),
                "n_obs_stacked": n,
                "mean_loss_difference": mean_diff,
                "hac_kernel": hac_kernel,
            }
    return out


def _l6_nested_results(
    errors: pd.DataFrame, sub: dict[str, Any], leaf: dict[str, Any], l4_models: L4ModelArtifactsArtifact
) -> dict[tuple[Any, ...], Any]:
    model_ids = sorted(errors["model_id"].unique()) if not errors.empty else []
    pairs = _l6_pair_list({"nested_pair_strategy": sub.get("nested_pair_strategy")}, leaf, model_ids, l4_models)
    tests = ["clark_west", "enc_new", "enc_t"] if sub.get("nested_test") == "multi" else [sub.get("nested_test")]
    apply_cw = bool(sub.get("cw_adjustment", True))
    hac_kernel = leaf.get("dependence_correction", "newey_west")
    results: dict[tuple[Any, ...], Any] = {}
    for test_name in tests:
        for large_model, small_model in pairs:
            for (target, horizon), group in errors.groupby(["target", "horizon"]):
                small = group.loc[group["model_id"] == small_model, ["origin", "squared", "forecast"]].rename(columns={"squared": "loss_small", "forecast": "f_small"})
                large = group.loc[group["model_id"] == large_model, ["origin", "squared", "forecast"]].rename(columns={"squared": "loss_large", "forecast": "f_large"})
                joined = small.merge(large, on="origin", how="inner")
                improvement = joined["loss_small"] - joined["loss_large"]
                if test_name == "clark_west" and apply_cw:
                    adjustment = (joined["f_small"] - joined["f_large"]) ** 2
                    f_value = improvement + adjustment
                else:
                    f_value = improvement
                stat, p_value = _diebold_mariano_test(f_value, horizon=int(horizon), hln=False, kernel=hac_kernel)
                # CW is a one-sided test (H_a: large model improves on small)
                p_value = (p_value / 2.0) if (p_value is not None and stat is not None and stat > 0) else p_value
                results[(test_name, small_model, large_model, target, int(horizon))] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "decision_at_5pct": p_value is not None and p_value < 0.05,
                    "n_obs": int(f_value.notna().sum()),
                    "mean_adjusted_improvement": _float_or_none(f_value.mean()) if not f_value.empty else None,
                    "cw_adjustment": apply_cw,
                }
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
    rng = np.random.default_rng(int(round(float(window_ratio) * 100) + int(alpha * 1000)))
    # Vectorised simulation -- much faster than the row-by-row form.
    n_sims = 1000
    n_grid = 200
    m = max(1, int(round(window_ratio * n_grid)))
    paths = rng.normal(size=(n_sims, n_grid))
    # Cumulative window sums via convolution: rolling sum of length m.
    kernel = np.ones(m)
    rolling_sums = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, paths)
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
                by_model.setdefault(str(model_id), []).append((origin, float(q), float(value)))

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
        for (model_id, target, horizon, origin), forecast in l4_forecasts.forecasts.items():
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
            out[("density", model_id)] = {"status": "insufficient_data", "n": int(pit.size)}
            continue
        result = _density_interval_battery(pit, alpha=alpha)
        out[("density", model_id)] = result
    return out


def _density_interval_battery(pit: np.ndarray, *, alpha: float = 0.05) -> dict[str, Any]:
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
        "ks": {"statistic": float(ks_stat), "p_value": float(ks_pvalue), "reject": bool(ks_pvalue < alpha)},
        "kupiec_pof": {"hits_rate": p_hat, "lr_statistic": float(ll_ratio), "p_value": kupiec_p, "reject": bool(kupiec_p < alpha)},
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


def _l6_cpa_results(errors: pd.DataFrame, sub: dict[str, Any], l4_models: L4ModelArtifactsArtifact) -> dict[tuple[Any, ...], Any]:
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

    from math import sqrt

    results: dict[tuple[Any, ...], Any] = {}
    model_ids = sorted(errors["model_id"].unique()) if not errors.empty else []
    pairs = _l6_pair_list({"model_pair_strategy": "vs_benchmark_only"}, {}, model_ids, l4_models)
    tests = ["giacomini_rossi_2010", "rossi_sekhposyan"] if sub.get("cpa_test") == "multi" else [sub.get("cpa_test")]
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
                left = group.loc[group["model_id"] == model_a, ["origin", "squared"]].rename(columns={"squared": "loss_a"})
                right = group.loc[group["model_id"] == model_b, ["origin", "squared"]].rename(columns={"squared": "loss_b"})
                joined = left.merge(right, on="origin", how="inner").sort_values("origin")
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


def _mcs_from_summary_metrics(metrics: pd.DataFrame, sub: dict[str, Any]) -> dict[str, Any]:
    """Legacy fallback: parametric Gaussian bootstrap on cross-sectional
    model-mean losses (used when L5 didn't carry a per-origin panel)."""

    if metrics.empty:
        return {"mcs_inclusion": {}, "spa_p_values": {}, "reality_check_p_values": {}, "stepm_rejected": {}}
    metric = "mse" if "mse" in metrics.columns else metrics.select_dtypes("number").columns[0]
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
        included = {model for model, value in zip(models, loss.values) if value <= loss.min() + scale * np.quantile(boot_max, 1 - alpha)}
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


def _mcs_from_per_origin_panel(panel: pd.DataFrame, sub: dict[str, Any]) -> dict[str, Any]:
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

    metric_col = "squared_error" if sub.get("mmt_loss_function", "squared") == "squared" else "absolute_error"
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
        wide = slice_df.pivot_table(index="origin", columns="model_id", values=metric_col, aggfunc="mean").sort_index()
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
            indices = _stationary_bootstrap_indices(n_obs, block_length, rng) if bootstrap_method == "stationary_bootstrap" else _fixed_block_bootstrap_indices(n_obs, block_length, rng)
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
        observed_t_max = observed_t.max()
        critical = float(np.quantile(boot_t_max, 1 - alpha))
        mcs_set = {str(model) for model, t in zip(wide.columns, observed_t) if t <= critical}
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


def _stationary_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
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


def _fixed_block_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Kunsch (1989) circular fixed-block bootstrap. Used when the recipe
    selects ``bootstrap_method = block``."""

    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    indices = np.empty(n_blocks * block_length, dtype=np.int64)
    for k, start in enumerate(starts):
        indices[k * block_length:(k + 1) * block_length] = (start + np.arange(block_length)) % n
    return indices[:n]


def _l6_direction_results(errors: pd.DataFrame, sub: dict[str, Any], leaf: dict[str, Any]) -> dict[tuple[Any, ...], Any]:
    threshold = leaf.get("direction_threshold_value", 0.0) if sub.get("direction_threshold") == "user_defined" else 0.0
    results: dict[tuple[Any, ...], Any] = {}
    tests = ["pesaran_timmermann_1992", "henriksson_merton"] if sub.get("direction_test") == "multi" else [sub.get("direction_test")]
    for test_name in tests:
        for (model_id, target, horizon), group in errors.groupby(["model_id", "target", "horizon"]):
            forecast_dir = (group["forecast"] - threshold).gt(0).astype(int).to_numpy()
            actual_dir = (group["actual"] - threshold).gt(0).astype(int).to_numpy()
            stat, p_value, success = _pesaran_timmermann_test(forecast_dir, actual_dir, test_name=test_name)
            results[(test_name, model_id, target, int(horizon))] = {"statistic": stat, "p_value": p_value, "success_ratio": success}
    return results


def _pesaran_timmermann_test(forecast: np.ndarray, actual: np.ndarray, *, test_name: str) -> tuple[float | None, float | None, float | None]:
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


def _l6_residual_results(errors: pd.DataFrame, sub: dict[str, Any]) -> dict[tuple[Any, ...], Any]:
    results: dict[tuple[Any, ...], Any] = {}
    tests = list(sub.get("residual_test", []))
    if "multi" in tests:
        tests = ["ljung_box_q", "arch_lm", "jarque_bera_normality", "breusch_godfrey_serial_correlation", "durbin_watson"]
    lag = int(sub.get("residual_lag_count", 10))
    for (model_id, target, horizon), group in errors.groupby(["model_id", "target", "horizon"]):
        residuals = group.sort_values("origin")["error"].dropna()
        for test_name in tests:
            statistic, p_value = _residual_test_statistic(test_name, residuals, lag)
            results[(test_name, model_id, target, int(horizon))] = {"statistic": statistic, "p_value": p_value, "lag_used": min(lag, max(len(residuals) - 1, 0))}
    return results


def _t_statistic(values: pd.Series) -> tuple[float | None, float | None]:
    clean = values.dropna()
    if len(clean) < 2:
        return None, None
    std = float(clean.std(ddof=1))
    if std == 0:
        stat = 0.0 if float(clean.mean()) == 0 else math.copysign(float("inf"), float(clean.mean()))
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


def _residual_test_statistic(test_name: str, residuals: pd.Series, lag: int) -> tuple[float | None, float | None]:
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
        adjustment = math.sqrt((n + 1 - 2 * (nw_lag + 1) + (nw_lag + 1) * (nw_lag) / n) / n)
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


def _long_run_variance(values: np.ndarray, *, kernel: str = "newey_west", lag: int | None = None) -> float:
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
            alpha = (4 * alpha1 ** 2) / (max(1 - alpha1 ** 2, 1e-12) ** 2)
            L = max(1, int(np.floor(1.1447 * (alpha * n) ** (1 / 3))))
        else:
            L = 1
        kernel = "newey_west"  # Andrews uses the Bartlett kernel with the data-driven L.
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
                weight = 1 - 6 * x ** 2 + 6 * x ** 3
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
    exported_files = _l8_export_artifacts(output_directory, axes, upstream_artifacts, recipe_root)
    git_sha, git_branch = _capture_git_state()
    package_version = _capture_package_version()
    runtime_env = _capture_full_runtime_environment()
    lockfile_content = _capture_dependency_lockfile_content()
    data_revision = _capture_data_revision_tag(recipe_root)
    seed_used = _capture_random_seed_used(recipe_root)
    # ``runtime_duration_per_layer`` and ``cells_summary[*].exported_files``
    # are non-deterministic across runs (wall-clock + tmp paths). Keep them
    # out of the hashed L8Manifest dataclass so bit-exact replicate still
    # passes; they are written to the on-disk JSON payload below.
    manifest = L8Manifest(
        recipe_hash=str(abs(hash(json.dumps(_jsonable(recipe_root), sort_keys=True)))),
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
        cells_summary=[{"cell_id": "cell_001", "status": "completed", "n_exported_files": len(exported_files)}],
    )
    manifest_path = output_directory / ("manifest.jsonl" if axes.get("manifest_format") == "json_lines" else "manifest.json")
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
    }
    if axes.get("manifest_format") == "json_lines":
        manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8")
    elif axes.get("manifest_format") == "yaml":
        try:
            import yaml as _yaml  # type: ignore
        except ImportError:
            manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
        else:
            manifest_path = manifest_path.with_suffix(".yaml")
            manifest_path.write_text(_yaml.safe_dump(manifest_payload, sort_keys=True), encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    recipe_path = output_directory / "recipe.json"
    recipe_path.write_text(json.dumps(_jsonable(recipe_root), indent=2, sort_keys=True), encoding="utf-8")
    exported_files.extend(
        [
            ExportedFile(path=manifest_path, artifact_type="manifest", source_sink="l8_artifacts_v1"),
            ExportedFile(path=recipe_path, artifact_type="recipe", source_sink="recipe"),
        ]
    )
    return (
        L8ArtifactsArtifact(
            output_directory=output_directory,
            manifest=manifest,
            exported_files=exported_files,
            artifact_count=len(exported_files),
            upstream_hashes={name: "runtime_unhashed" for name in sorted(upstream_artifacts)},
        ),
        axes,
    )


def _derive_saved_objects(recipe_root: dict[str, Any], upstream_artifacts: dict[str, Any]) -> set[str]:
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


def _l8_export_artifacts(output_directory: Path, axes: dict[str, Any], upstream_artifacts: dict[str, Any], recipe_root: dict[str, Any]) -> list[ExportedFile]:
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
            exported.append(ExportedFile(path=path.with_suffix(".csv"), artifact_type="csv", source_sink=source))
        if "parquet" in formats:
            try:
                frame.reset_index().to_parquet(path.with_suffix(".parquet"))
                exported.append(ExportedFile(path=path.with_suffix(".parquet"), artifact_type="parquet", source_sink=source))
            except Exception:
                pass
        if "latex" in formats:
            latex = frame.to_latex(index=True)
            path.with_suffix(".tex").write_text(latex, encoding="utf-8")
            exported.append(ExportedFile(path=path.with_suffix(".tex"), artifact_type="latex", source_sink=source))
        if "markdown" in formats:
            md = frame.to_markdown(index=True)
            path.with_suffix(".md").write_text(md, encoding="utf-8")
            exported.append(ExportedFile(path=path.with_suffix(".md"), artifact_type="markdown", source_sink=source))

    def add_json(path: Path, payload: Any, source: str) -> None:
        if "json" not in formats:
            return
        path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        exported.append(ExportedFile(path=path, artifact_type="json", source_sink=source))

    if "forecasts" in saved and "l4_forecasts_v1" in upstream_artifacts:
        rows = [
            {"model_id": model_id, "target": target, "horizon": horizon, "origin": origin, "forecast": forecast}
            for (model_id, target, horizon, origin), forecast in upstream_artifacts["l4_forecasts_v1"].forecasts.items()
        ]
        forecasts_frame = pd.DataFrame(rows)
        if granularity in {"per_target", "per_horizon", "per_target_horizon"} and not forecasts_frame.empty:
            for sub_dir, sub_frame in _l8_split_by_granularity(
                forecasts_frame, granularity, cell_dir
            ):
                sub_dir.mkdir(parents=True, exist_ok=True)
                add_dataframe(sub_dir / "forecasts", sub_frame.reset_index(drop=True), "l4_forecasts_v1")
        else:
            add_dataframe(cell_dir / "forecasts", forecasts_frame, "l4_forecasts_v1")
    if "metrics" in saved and "l5_evaluation_v1" in upstream_artifacts:
        add_dataframe(summary_dir / "metrics_all_cells", upstream_artifacts["l5_evaluation_v1"].metrics_table, "l5_evaluation_v1")
    if "ranking" in saved and "l5_evaluation_v1" in upstream_artifacts:
        add_dataframe(summary_dir / "ranking", upstream_artifacts["l5_evaluation_v1"].ranking_table, "l5_evaluation_v1")
    if "tests" in saved and "l6_tests_v1" in upstream_artifacts:
        add_json(output_directory / "tests_summary.json", upstream_artifacts["l6_tests_v1"], "l6_tests_v1")
    if "importance" in saved and "l7_importance_v1" in upstream_artifacts:
        importance_artifact = upstream_artifacts["l7_importance_v1"]
        add_json(output_directory / "importance_summary.json", importance_artifact, "l7_importance_v1")
        try:
            from .figures import render_default_for_op, render_us_state_choropleth

            figures_dir.mkdir(parents=True, exist_ok=True)
            sink_payloads = getattr(importance_artifact, "global_importance", {}) or {}
            for op_name, payload in sink_payloads.items():
                figure_path = figures_dir / f"{op_name}.pdf"
                rendered = render_default_for_op(op_name, payload, output_path=figure_path, title=f"L7 {op_name}")
                if rendered is not None:
                    exported.append(ExportedFile(path=rendered, artifact_type="figure_pdf", source_sink="l7_importance_v1"))
            # FRED-SD geographic visualization: when group_aggregate produced
            # per-state importance, render a US choropleth.
            group_payloads = getattr(importance_artifact, "group_importance", {}) or {}
            for op_name, payload in group_payloads.items():
                if isinstance(payload, pd.DataFrame) and "group" in payload.columns:
                    state_scores = {row["group"]: float(row["importance"]) for _, row in payload.iterrows() if isinstance(row.get("group"), str) and len(str(row["group"])) == 2}
                    if state_scores:
                        choropleth = figures_dir / f"{op_name}_state_choropleth.pdf"
                        render_us_state_choropleth(state_scores, output_path=choropleth, title=f"L7 {op_name} (state)")
                        exported.append(ExportedFile(path=choropleth, artifact_type="figure_pdf", source_sink="l7_importance_v1"))
        except Exception:
            # Figure rendering is best-effort; leave a json export and continue.
            pass
    if "feature_metadata" in saved and "l3_metadata_v1" in upstream_artifacts:
        add_json(cell_dir / "feature_metadata.json", upstream_artifacts["l3_metadata_v1"], "l3_metadata_v1")
    if "clean_panel" in saved and "l2_clean_panel_v1" in upstream_artifacts:
        add_dataframe(cell_dir / "clean_panel", upstream_artifacts["l2_clean_panel_v1"].panel.data, "l2_clean_panel_v1")
    if "raw_panel" in saved and "l1_data_definition_v1" in upstream_artifacts:
        add_dataframe(cell_dir / "raw_panel", upstream_artifacts["l1_data_definition_v1"].raw_panel.data, "l1_data_definition_v1")
    for sink_name, artifact in upstream_artifacts.items():
        if sink_name.endswith("_diagnostic_v1"):
            object_name = f"diagnostics_{sink_name.split('_diagnostic_v1')[0]}"
            if object_name in saved or "diagnostics_all" in saved:
                diag_dir = output_directory / "diagnostics"
                diag_dir.mkdir(exist_ok=True)
                add_json(diag_dir / f"{sink_name}.json", artifact, sink_name)

    if export_format == "html_report":
        html_path = _l8_render_html_report(output_directory, axes, upstream_artifacts, recipe_root)
        if html_path is not None:
            exported.append(ExportedFile(path=html_path, artifact_type="html_report", source_sink="l8_artifacts_v1"))

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
    target = ((recipe_root.get("1_data", {}) or {}).get("leaf_config", {}) or {}).get("target")
    if target:
        lines.append(f"<p><b>Target</b>: <code>{_esc(str(target))}</code></p>")
    family = None
    for node in (recipe_root.get("4_forecasting_model", {}) or {}).get("nodes", []) or []:
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
            safe = "_missing_" if pd.isna(val) else str(val).replace("/", "_").replace(" ", "_")
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
        with path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=level) as dst:
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
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
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
            ("git", "rev-parse", "HEAD"), cwd=cwd, capture_output=True, text=True, check=True, timeout=5
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

    import socket

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
    descriptor = _command_version_safe(("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"))
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
        return {key: _plain_axes(item) for key, item in value.items() if key != "_active"}
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
            raise ValueError("custom panel runtime requires custom_panel_inline, custom_panel_records, or custom_source_path")
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
                    "transform_codes": dict(getattr(official, "transform_codes", {}) or {}),
                }
            )
    else:
        raise NotImplementedError(f"custom_source_policy={policy!r} core runtime loading is deferred")
    frame = _normalize_datetime_index(frame, leaf_config)
    frame = _apply_sample_window(frame, resolved, leaf_config)
    _validate_targets_present(frame, leaf_config, resolved)
    return _panel_from_frame(frame, metadata=metadata)


def _load_official_raw_result(resolved: dict[str, Any], leaf_config: dict[str, Any]):
    dataset = resolved.get("dataset")
    vintage = leaf_config.get("vintage")
    cache_root = leaf_config.get("cache_root")
    local_source = leaf_config.get("local_raw_source") or leaf_config.get("official_source_path")
    if dataset == "fred_md":
        return load_fred_md(vintage=vintage, cache_root=cache_root, local_source=local_source)
    if dataset == "fred_qd":
        return load_fred_qd(vintage=vintage, cache_root=cache_root, local_source=local_source)
    if dataset == "fred_sd":
        states = _resolve_fred_sd_states(resolved, leaf_config)
        variables = leaf_config.get("fred_sd_variables") or leaf_config.get("sd_variables")
        return load_fred_sd(
            vintage=vintage,
            cache_root=cache_root,
            local_source=local_source,
            states=list(states) if states else None,
            variables=list(variables) if variables else None,
        )
    if dataset in {"fred_md+fred_sd", "fred_qd+fred_sd"}:
        national_loader = load_fred_md if dataset.startswith("fred_md") else load_fred_qd
        national = national_loader(vintage=vintage, cache_root=cache_root, local_source=local_source)
        states = _resolve_fred_sd_states(resolved, leaf_config)
        variables = leaf_config.get("fred_sd_variables") or leaf_config.get("sd_variables")
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
    raise NotImplementedError(f"official dataset {dataset!r} is not supported by core L1 runtime yet")


def _resolve_fred_sd_states(resolved: dict[str, Any], leaf_config: dict[str, Any]) -> list[str] | None:
    explicit = leaf_config.get("fred_sd_states") or leaf_config.get("state_selection")
    if explicit:
        return list(explicit)
    group_key = resolved.get("fred_sd_state_group") or leaf_config.get("fred_sd_state_group")
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
        return pd.read_csv(path)
    raise ValueError(f"unsupported custom panel format {path.suffix!r}; use CSV or Parquet")


def _normalize_datetime_index(frame: pd.DataFrame, leaf_config: dict[str, Any]) -> pd.DataFrame:
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
    return frame


def _apply_sample_window(frame: pd.DataFrame, resolved: dict[str, Any], leaf_config: dict[str, Any]) -> pd.DataFrame:
    result = frame
    start_rule = resolved.get("sample_start_rule") or "max_balanced"
    end_rule = resolved.get("sample_end_rule") or "latest_available"
    if start_rule == "max_balanced":
        first_observed = result.dropna(axis=0, how="any").index.min() if not result.empty else None
        if first_observed is not None and pd.notna(first_observed):
            result = result.loc[first_observed:]
    elif start_rule == "fixed_date":
        result = result.loc[pd.Timestamp(leaf_config["sample_start_date"]) :]
    if end_rule == "latest_available":
        # default: keep as-is
        pass
    elif end_rule == "fixed_date":
        result = result.loc[: pd.Timestamp(leaf_config["sample_end_date"])]
    return result


def _validate_targets_present(frame: pd.DataFrame, leaf_config: dict[str, Any], resolved: dict[str, Any]) -> None:
    target = leaf_config.get("target")
    targets = tuple(leaf_config.get("targets", ()) or ((target,) if target else ()))
    missing = [name for name in targets if name not in frame.columns]
    if missing:
        raise ValueError(f"target columns missing from custom panel: {missing}")
    if resolved.get("target_structure") == "single_target" and not target:
        raise ValueError("single_target runtime requires leaf_config.target")


def _default_chow_lin_indicator(frame: pd.DataFrame, monthly_cols: list[str]) -> str | None:
    """Pick the monthly column with the highest absolute correlation to the
    target's quarterly observations -- used when ``chow_lin_indicator`` is
    not supplied."""

    if not monthly_cols:
        return None
    return monthly_cols[0]


def _chow_lin_disaggregate(quarterly: pd.Series, indicator_monthly: pd.Series) -> pd.Series:
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
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    beta = float(((x - x_mean) * (y - y_mean)).sum() / denom) if denom > 0 else 0.0
    alpha = y_mean - beta * x_mean
    # Predicted quarterly series.
    pred_q = pd.Series(alpha + beta * indicator_q.to_numpy(), index=indicator_q.index)
    resid_q = (
        quarterly.resample("QE").last().reindex(pred_q.index) - pred_q
    ).fillna(0.0)
    # Distribute alpha + beta * X^M with the quarterly residual smeared evenly.
    monthly = alpha / 3.0 + beta * indicator_monthly
    # Distribute each quarter's residual: each month in quarter Q gets
    # resid_q / 3.
    resid_monthly = pd.Series(0.0, index=monthly_index)
    for q_end, resid in resid_q.items():
        in_quarter = (monthly_index >= q_end - pd.Timedelta(days=92)) & (monthly_index <= q_end)
        n_months = int(in_quarter.sum())
        if n_months > 0:
            resid_monthly.loc[in_quarter] = float(resid) / n_months
    monthly = monthly + resid_monthly
    return monthly


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

    series_freq_map = (l1_artifact.raw_panel.metadata.values or {}).get("series_frequency", {}) or {}
    if not series_freq_map:
        # Without per-series frequency metadata we can't selectively align;
        # leave the panel untouched and record a no-op.
        cleaning_log.setdefault("steps", []).append(
            {"step": "fred_sd_frequency_alignment", "applied": False, "reason": "no series_frequency metadata"}
        )
        return df

    monthly_cols = [c for c in df.columns if str(series_freq_map.get(c, "")).lower() == "monthly"]
    quarterly_cols = [c for c in df.columns if str(series_freq_map.get(c, "")).lower() == "quarterly"]
    sd_filter = resolved.get("sd_series_frequency_filter", "both")
    qm_rule = resolved.get("quarterly_to_monthly_rule", "step_backward")
    mq_rule = resolved.get("monthly_to_quarterly_rule", "quarterly_average")

    if sd_filter == "monthly_only":
        df = df[monthly_cols + [c for c in df.columns if c not in monthly_cols and c not in quarterly_cols]]
        cleaning_log.setdefault("steps", []).append(
            {"step": "fred_sd_frequency_alignment", "filter": "monthly_only", "n_dropped": len(quarterly_cols)}
        )
        return df
    if sd_filter == "quarterly_only":
        df = df[quarterly_cols + [c for c in df.columns if c not in monthly_cols and c not in quarterly_cols]]
        cleaning_log.setdefault("steps", []).append(
            {"step": "fred_sd_frequency_alignment", "filter": "quarterly_only", "n_dropped": len(monthly_cols)}
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
                indicator_col = (resolved.get("leaf_config") or {}).get("chow_lin_indicator") or _default_chow_lin_indicator(df, monthly_cols)
                if indicator_col and indicator_col in df.columns:
                    df[col] = _chow_lin_disaggregate(series, df[indicator_col])
                else:
                    df[col] = series.bfill().ffill()
            else:  # step_backward
                df[col] = series.bfill().ffill()
        cleaning_log.setdefault("steps", []).append(
            {"step": "fred_sd_frequency_alignment", "rule": qm_rule, "direction": "quarterly_to_monthly", "n_cols": len(quarterly_cols)}
        )
    elif target_freq == "quarterly" and monthly_cols:
        if not isinstance(df.index, pd.DatetimeIndex):
            cleaning_log.setdefault("steps", []).append(
                {"step": "fred_sd_frequency_alignment", "applied": False, "reason": "non-datetime index"}
            )
            return df
        if mq_rule == "quarterly_average":
            agg = df[monthly_cols].resample("QE").mean()
        elif mq_rule == "quarterly_endpoint":
            agg = df[monthly_cols].resample("QE").last()
        else:  # quarterly_sum
            agg = df[monthly_cols].resample("QE").sum()
        # Align back to the (quarterly) target index by reindexing.
        df_q = df[quarterly_cols + [c for c in df.columns if c not in monthly_cols and c not in quarterly_cols]]
        df = df_q.join(agg, how="left").reindex(df_q.index)
        cleaning_log.setdefault("steps", []).append(
            {"step": "fred_sd_frequency_alignment", "rule": mq_rule, "direction": "monthly_to_quarterly", "n_cols": len(monthly_cols)}
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
        cleaning_log["steps"].append({"transform": "apply_official_tcode", "fallback": "no_tcode_map_available"})
        return frame, {}
    if policy == "custom_tcode" and not tcode_map:
        raise ValueError("custom_tcode runtime requires custom_tcode_map")
    transformed = frame.copy()
    applied: dict[str, int] = {}
    for column, tcode in tcode_map.items():
        if column not in transformed.columns:
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
    return positive.map(lambda value: pd.NA if pd.isna(value) else __import__("math").log(value))


def _try_custom_l2_preprocessor(name: str, frame: pd.DataFrame, leaf_config: dict[str, Any]) -> pd.DataFrame | None:
    """Issue #251 -- dispatch to a user-registered preprocessor when ``name``
    matches a registered entry. Returns ``None`` to indicate fall-through
    to built-in policies."""

    try:
        from .. import custom as _custom_mod
    except ImportError:  # pragma: no cover
        return None
    if not _custom_mod.is_custom_preprocessor(str(name)):
        return None
    spec = _custom_mod.get_custom_preprocessor(str(name))
    try:
        # Documented contract: ``fn(X_train, y_train, X_test, context) ->
        # (X_train, X_test)``. For the runtime hook we substitute ``X_train
        # = X_test = frame`` so the callable can do a single-pass clean.
        result = spec.function(frame, None, frame, dict(leaf_config))
        if isinstance(result, tuple) and result:
            cleaned = result[0]
            return cleaned if isinstance(cleaned, pd.DataFrame) else None
        if isinstance(result, pd.DataFrame):
            return result
    except Exception:  # pragma: no cover
        return None
    return None


def _apply_outlier_policy(
    frame: pd.DataFrame, resolved: l2_layer.L2ResolvedAxes, leaf_config: dict[str, Any], cleaning_log: dict[str, Any]
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
        mask = ((numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, pd.NA)).abs() > threshold
    elif policy == "winsorize":
        low, high = leaf_config.get("winsorize_quantiles", [0.01, 0.99])
        clipped = numeric.clip(numeric.quantile(low), numeric.quantile(high), axis=1)
        changed = int((clipped.ne(numeric) & ~(clipped.isna() & numeric.isna())).sum().sum())
        result[numeric.columns] = clipped
        cleaning_log["steps"].append({"outlier": "winsorize", "action": action, "quantiles": [low, high], "capped": changed})
        return result, changed
    else:
        raise NotImplementedError(f"outlier_policy={policy!r} runtime is not implemented")
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
        raise NotImplementedError(f"outlier_action={action!r} runtime is not implemented")
    cleaning_log["steps"].append({"outlier": policy, "action": action, "flagged": count})
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
        result = _pca_em_imputation(frame, n_factors=8 if policy == "em_factor" else None, max_iter=20)
        method = policy
    elif policy == "forward_fill":
        result = frame.ffill()
        method = "forward_fill"
    elif policy == "linear_interpolation":
        result = frame.interpolate(method="linear")
        method = "linear_interpolation"
    else:
        raise NotImplementedError(f"imputation_policy={policy!r} runtime is not implemented")
    filled = missing_before - int(result.isna().sum().sum())
    cleaning_log["steps"].append({"imputation": method, "filled": filled})
    return result, filled


def _pca_em_imputation(frame: pd.DataFrame, *, n_factors: int | None, max_iter: int = 20, tol: float = 1e-4) -> pd.DataFrame:
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
    rank = min(int(n_factors) if n_factors else min(matrix.shape) // 2, min(matrix.shape) - 1)
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
        raise NotImplementedError(f"frame_edge_policy={policy!r} runtime is not implemented")
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

    cascade_max_depth = int(getattr(dag, "leaf_config", {}).get("cascade_max_depth", 10)) if hasattr(dag, "leaf_config") else 10

    values: dict[str, Any] = {}
    depth_by_node: dict[str, int] = {}
    pipeline_by_node: dict[str, str] = {}

    for node in _topological_nodes(dag):
        # Cascade depth is one more than the deepest input's depth.
        input_depths = [depth_by_node.get(ref.node_id, 0) for ref in node.inputs]
        node_depth = (max(input_depths) + 1) if (node.type == "step" and input_depths) else 0
        depth_by_node[node.id] = node_depth
        if node_depth > cascade_max_depth:
            offending_chain = [node.id]
            cursor = node
            while cursor.inputs:
                deepest = max(cursor.inputs, key=lambda ref: depth_by_node.get(ref.node_id, 0))
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
        explicit_pipeline = (node.params or {}).get("pipeline_id") if node.type == "step" else None
        if explicit_pipeline:
            pipeline_by_node[node.id] = str(explicit_pipeline)
        elif node.inputs:
            inherited = next(
                (pipeline_by_node[ref.node_id] for ref in node.inputs if ref.node_id in pipeline_by_node),
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


def _execute_l3_source(selector, frame: pd.DataFrame, target_name: str) -> pd.DataFrame | pd.Series:
    if selector is None:
        raise ValueError("L3 source node requires a selector")
    if selector.layer_ref != "l2" or selector.sink_name != "l2_clean_panel_v1":
        raise NotImplementedError("minimal L3 runtime currently supports L2 clean panel sources only")
    subset = selector.subset or {}
    role = subset.get("role")
    if role == "target":
        return frame[target_name].copy()
    if role == "predictors":
        return frame[[column for column in frame.columns if column != target_name]].copy()
    if "variable_list" in subset:
        return frame[list(subset["variable_list"])].copy()
    if subset.get("raw") is True:
        return frame.copy()
    raise NotImplementedError(f"minimal L3 runtime does not support source subset {subset!r}")


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
            raise ValueError(f"{dag.layer_id}: DAG contains unresolved dependencies or a cycle")
    return ordered


def _try_custom_l3_dispatch(op: str, inputs: list[Any], params: dict[str, Any]) -> pd.DataFrame | pd.Series | None:
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


def _execute_l3_op(op: str, inputs: list[Any], params: dict[str, Any], target_name: str) -> pd.DataFrame | pd.Series:
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
        return _lagged_predictors(_as_frame(inputs[0]), n_lag=int(params.get("n_lag", 4)), include_contemporaneous=bool(params.get("include_contemporaneous", False)))
    if op == "seasonal_lag":
        return _seasonal_lagged_predictors(
            _as_frame(inputs[0]),
            seasonal_period=int(params.get("seasonal_period", 12)),
            n_seasonal_lags=int(params.get("n_seasonal_lags", 1)),
        )
    if op == "ma_window":
        return _as_frame(inputs[0]).rolling(window=int(params.get("window", 3)), min_periods=int(params.get("window", 3))).mean()
    if op == "ma_increasing_order":
        return _ma_increasing_order(_as_frame(inputs[0]), max_order=int(params.get("max_order", 12)))
    if op == "cumsum":
        return inputs[0].cumsum()
    if op == "concat":
        return pd.concat([_as_frame(value) for value in inputs], axis=1)
    if op == "scale":
        return _scale_frame(_as_frame(inputs[0]), method=params.get("method", "zscore"))
    if op == "log":
        return _map_like(inputs[0], lambda value: pd.NA if pd.isna(value) or value <= 0 else __import__("math").log(value))
    if op == "diff":
        return _diff_like(inputs[0], periods=int(params.get("n_diff", 1)))
    if op == "log_diff":
        logged = _map_like(inputs[0], lambda value: pd.NA if pd.isna(value) or value <= 0 else __import__("math").log(value))
        return _diff_like(logged, periods=int(params.get("n_diff", 1)))
    if op == "pct_change":
        return _pct_change_like(inputs[0], periods=int(params.get("n_periods", 1)))
    if op in {"polynomial_expansion", "polynomial"}:
        return _polynomial_expansion(_as_frame(inputs[0]), degree=int(params.get("degree", 2)))
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
        return _pca_factors(_as_frame(inputs[0]), n_components=int(params.get("n_components", 8)), variant=op, target_signal=_first_series(inputs))
    if op == "varimax" or op == "varimax_rotation":
        return _varimax_rotation(_as_frame(inputs[0]))
    if op == "partial_least_squares":
        return _partial_least_squares(_as_frame(inputs[0]), target=_first_series(inputs), n_components=int(params.get("n_components", 4)))
    if op == "random_projection":
        return _random_projection(_as_frame(inputs[0]), n_components=int(params.get("n_components", 8)))
    if op == "dfm":
        return _dfm_factors(_as_frame(inputs[0]), n_factors=int(params.get("n_factors", 3)))
    if op == "wavelet":
        return _wavelet_decomposition(_as_frame(inputs[0]), n_levels=int(params.get("n_levels", 1)))
    if op == "fourier":
        return _fourier_features(_as_frame(inputs[0]), n_terms=int(params.get("n_terms", 3)), period=int(params.get("period", 12)))
    if op == "hp_filter":
        return _hp_filter(_as_frame(inputs[0]), lam=float(params.get("lambda_", params.get("lam", 1600.0))))
    if op == "hamilton_filter":
        return _hamilton_filter(_as_frame(inputs[0]), n_lags=int(params.get("n_lags", 8)), n_horizon=int(params.get("n_horizon", 24)))
    if op in {"kernel", "kernel_features"}:
        return _kernel_features(_as_frame(inputs[0]), kind=params.get("kind", "rbf"), gamma=float(params.get("gamma", 1.0)))
    if op in {"nystroem", "nystroem_features"}:
        return _nystroem_features(_as_frame(inputs[0]), n_components=int(params.get("n_components", 32)))
    if op == "feature_selection":
        return _feature_selection(_as_frame(inputs[0]), target=_first_series(inputs), n_features=params.get("n_features", 0.5), method=params.get("method", "variance"))
    if op == "hierarchical_pca":
        return _hierarchical_pca(inputs, n_components_per_block=int(params.get("n_components_per_block", 1)))
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
            target = _cumulative_average_target(y_series, horizon=horizon).rename(target_name)
        else:
            target = y_series.shift(-horizon).rename(target_name)
        target.attrs["horizon"] = horizon
        target.attrs["mode"] = mode
        return target
    raise NotImplementedError(f"L3 runtime does not support op {op!r}")


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
    elif hasattr(value, "regime_series") and isinstance(getattr(value, "regime_series", None), pd.Series):
        series = value.regime_series
    elif hasattr(value, "data") and isinstance(getattr(value, "data", None), pd.Series):
        series = value.data
    else:
        raise ValueError("regime_indicator requires an L1 regime artifact or pandas Series input")
    dummies = pd.get_dummies(series.astype(str), prefix="regime", dtype=float)
    return dummies


def _pca_factors(frame: pd.DataFrame, *, n_components: int, variant: str = "pca", target_signal: pd.Series | None = None) -> pd.DataFrame:
    """In-sample PCA on the centered data; target signal scaling for scaled_pca."""

    from sklearn.decomposition import PCA, SparsePCA

    cleaned = frame.dropna(axis=0, how="any")
    if cleaned.empty:
        return pd.DataFrame(index=frame.index)
    n_components = max(1, min(int(n_components), min(cleaned.shape) - 1))
    matrix = cleaned.to_numpy()
    matrix = matrix - matrix.mean(axis=0)
    if variant == "scaled_pca" and target_signal is not None:
        aligned = target_signal.reindex(cleaned.index).to_numpy()
        weights = np.where(np.isfinite(aligned), aligned - np.nanmean(aligned), 0.0)
        weights = np.abs(weights) / max(np.abs(weights).max(), 1e-9)
        matrix = matrix * (1.0 + weights[:, None])
    if variant == "sparse_pca":
        model = SparsePCA(n_components=n_components, random_state=0)
        scores = model.fit_transform(matrix)
    else:
        model = PCA(n_components=n_components, random_state=0)
        scores = model.fit_transform(matrix)
    factors = pd.DataFrame(
        scores,
        index=cleaned.index,
        columns=[f"factor_{idx + 1}" for idx in range(scores.shape[1])],
    )
    return factors.reindex(frame.index)


def _varimax_rotation(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.dropna(axis=0, how="any")
    if cleaned.empty:
        return frame
    matrix = cleaned.to_numpy(dtype=float)
    n_features = matrix.shape[1]
    rotation = np.eye(n_features)
    for _ in range(50):
        u, _, vh = np.linalg.svd(
            matrix.T @ (matrix**3 - matrix * (np.diag(matrix.T @ matrix) / matrix.shape[0]))
        )
        rotation = u @ vh
        matrix = matrix @ rotation
    rotated = pd.DataFrame(matrix, index=cleaned.index, columns=[f"varimax_{i+1}" for i in range(n_features)])
    return rotated.reindex(frame.index)


def _partial_least_squares(frame: pd.DataFrame, *, target: pd.Series | None, n_components: int) -> pd.DataFrame:
    if target is None:
        raise ValueError("partial_least_squares requires a target Series input")
    from sklearn.cross_decomposition import PLSRegression

    aligned = pd.concat([frame, target.rename("__target__")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame(index=frame.index)
    n_components = max(1, min(int(n_components), min(aligned.shape[0] - 1, aligned.shape[1] - 1)))
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
        approx = frame.rolling(window=window, min_periods=window).mean().add_suffix(f"_wA{level}")
        detail = (frame - frame.rolling(window=window, min_periods=window).mean()).add_suffix(f"_wD{level}")
        pieces.extend([approx, detail])
    return pd.concat(pieces, axis=1)


def _fourier_features(frame: pd.DataFrame, *, n_terms: int, period: int) -> pd.DataFrame:
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


def _hamilton_filter(frame: pd.DataFrame, *, n_lags: int, n_horizon: int) -> pd.DataFrame:
    """Hamilton (2018) regression-based filter: y_{t+h} on lagged values; residuals are the cycle."""

    from sklearn.linear_model import LinearRegression

    out: dict[str, pd.Series] = {}
    for column in frame.columns:
        series = frame[column].astype(float)
        lagged = pd.concat([series.shift(n_horizon + lag) for lag in range(n_lags)], axis=1)
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
    columns = [f"kernel_{i+1}" for i in range(kernel.shape[1])]
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


def _feature_selection(frame: pd.DataFrame, *, target: pd.Series | None, n_features: Any, method: str) -> pd.DataFrame:
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
        lasso = LassoCV(cv=min(5, max(2, len(aligned) // 4)), random_state=0, max_iter=20000)
        lasso.fit(aligned.iloc[:, :-1], aligned.iloc[:, -1])
        coefs = pd.Series(np.abs(lasso.coef_), index=frame.columns)
        ordered = coefs.sort_values(ascending=False)
        return frame[list(ordered.index[:keep])]
    variances = frame.var().sort_values(ascending=False)
    return frame[list(variances.index[:keep])]


def _hierarchical_pca(inputs: list[Any], *, n_components_per_block: int) -> pd.DataFrame:
    blocks = []
    for block_index, item in enumerate(inputs):
        block_frame = _as_frame(item)
        block_factors = _pca_factors(block_frame, n_components=n_components_per_block)
        block_factors.columns = [f"hpca_block{block_index + 1}_f{i + 1}" for i in range(block_factors.shape[1])]
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
    aligned = pd.concat([frame.add_suffix(f"_{i}") for i, frame in enumerate(frames)], axis=1)
    grouped: dict[str, list[pd.Series]] = {}
    for column in aligned.columns:
        base = column.rsplit("_", 1)[0]
        grouped.setdefault(base, []).append(aligned[column])
    averaged = {key: pd.concat(items, axis=1).mean(axis=1) for key, items in grouped.items()}
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
        col_range = pd.Series(col_max - col_min, index=frame.columns).replace(0, pd.NA).to_numpy()
        scaled = (arr - col_min) / col_range
    else:
        raise NotImplementedError(f"L3 runtime does not support scale method {method!r}")
    return pd.DataFrame(scaled, index=frame.index, columns=frame.columns)


def _map_like(value: pd.DataFrame | pd.Series, func) -> pd.DataFrame | pd.Series:
    if isinstance(value, pd.DataFrame):
        return value.map(func)
    if isinstance(value, pd.Series):
        return value.map(func)
    raise TypeError(f"expected pandas DataFrame or Series, got {type(value).__name__}")


def _diff_like(value: pd.DataFrame | pd.Series, *, periods: int) -> pd.DataFrame | pd.Series:
    return value.diff(periods=periods)


def _pct_change_like(value: pd.DataFrame | pd.Series, *, periods: int) -> pd.DataFrame | pd.Series:
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
        raise ValueError("minimal L4 runtime requires min_train_size < aligned observation count")
    return min_train_size


def _lagged_predictors(frame: pd.DataFrame, n_lag: int, *, include_contemporaneous: bool = False) -> pd.DataFrame:
    if n_lag < 1:
        raise ValueError("minimal L3 runtime requires n_lag >= 1")
    lagged = []
    first_lag = 0 if include_contemporaneous else 1
    for lag in range(first_lag, n_lag + 1):
        lagged.append(frame.shift(lag).add_suffix(f"_lag{lag}"))
    return pd.concat(lagged, axis=1)


def _seasonal_lagged_predictors(frame: pd.DataFrame, *, seasonal_period: int, n_seasonal_lags: int) -> pd.DataFrame:
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
        windows.append(frame.rolling(window=order, min_periods=order).mean().add_suffix(f"_ma{order}"))
    return pd.concat(windows, axis=1)


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


def _materialize_regime(resolved: dict[str, Any], leaf_config: dict[str, Any], sample_index: pd.DatetimeIndex) -> L1RegimeMetadataArtifact:
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
                metadata=SeriesMetadata(values={"data": labels, "source": "embedded_nber_recession_dates"}),
            ),
            regime_probabilities=None,
            transition_matrix=None,
            estimation_temporal_rule=base.estimation_temporal_rule,
            estimation_metadata={**base.estimation_metadata, "n_recession_months": int((labels == "recession").sum())},
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
        target_name = leaf_config.get("regime_estimation_series") or leaf_config.get("target")
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
                metadata=SeriesMetadata(values={"data": labels, "source": "hamilton_1989_markov_regression"}),
            ),
            regime_probabilities=probs,
            transition_matrix=transition_matrix,
            estimation_temporal_rule=base.estimation_temporal_rule,
            estimation_metadata={**base.estimation_metadata, **metadata},
        )
    if definition == "estimated_threshold" and len(sample_index):
        n_regimes = int(leaf_config.get("n_regimes", 2))
        labels, metadata = _estimate_threshold_regime(sample_index, n_regimes, leaf_config)
        return L1RegimeMetadataArtifact(
            definition=base.definition,
            n_regimes=n_regimes,
            regime_label_series=Series(
                shape=labels.shape,
                name="setar_regime",
                metadata=SeriesMetadata(values={"data": labels, "source": "tong_1990_setar"}),
            ),
            regime_probabilities=None,
            transition_matrix=None,
            estimation_temporal_rule=base.estimation_temporal_rule,
            estimation_metadata={**base.estimation_metadata, **metadata},
        )
    if definition == "estimated_structural_break" and len(sample_index):
        max_breaks = int(leaf_config.get("max_breaks", 3))
        labels, metadata = _estimate_structural_break_regime(sample_index, max_breaks, leaf_config)
        return L1RegimeMetadataArtifact(
            definition=base.definition,
            n_regimes=int(labels.nunique()),
            regime_label_series=Series(
                shape=labels.shape,
                name="bai_perron_regime",
                metadata=SeriesMetadata(values={"data": labels, "source": "bai_perron_break_detection"}),
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
            [np.ones(len(y))] + [values[p - lag : -lag] if lag else values[p:] for lag in range(1, p + 1)]
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
    cumsq = np.concatenate(([0.0], np.cumsum(y_valid ** 2)))

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
        return labels, None, None, {"method": "fallback_uniform_split", "reason": "target series unavailable"}

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
        return labels, None, None, {"method": "fallback_too_few_obs", "n_obs": int(series.size)}
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
            "converged": bool(getattr(results, "mle_retvals", {}).get("converged", True)),
        }
        return labels, smoothed_full, transition_matrix, metadata
    except Exception as exc:
        labels = pd.Series(["regime_0"] * len(sample_index), index=sample_index)
        return labels, None, None, {"method": "fallback_fit_failed", "error": str(exc)}


# NBER official US recession dates (start, end) inclusive, monthly.
# Source: nber.org/research/business-cycle-dating (peaks/troughs).
_NBER_RECESSIONS: tuple[tuple[str, str], ...] = (
    ("1948-11", "1949-10"), ("1953-07", "1954-05"), ("1957-08", "1958-04"),
    ("1960-04", "1961-02"), ("1969-12", "1970-11"), ("1973-11", "1975-03"),
    ("1980-01", "1980-07"), ("1981-07", "1982-11"), ("1990-07", "1991-03"),
    ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04"),
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


def _build_user_provided_regime_series(leaf_config: dict[str, Any], index: pd.DatetimeIndex) -> pd.Series | None:
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
                start, end, label = pd.Timestamp(entry[0]), pd.Timestamp(entry[1]), str(entry[2] if len(entry) > 2 else "alt")
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
