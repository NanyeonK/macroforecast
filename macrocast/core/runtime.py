from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import math
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.linear_model import Ridge

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
from ..raw import load_fred_md, load_fred_qd
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

    root = parse_recipe_yaml(recipe_yaml_or_root) if isinstance(recipe_yaml_or_root, str) else recipe_yaml_or_root
    l1_artifact, regime_artifact, l1_axes = materialize_l1(root)
    l2_artifact, l2_axes = materialize_l2(root, l1_artifact)
    l3_features, l3_metadata = materialize_l3_minimal(root, l1_artifact, l2_artifact)
    l4_forecasts, l4_models, l4_training = materialize_l4_minimal(root, l3_features)
    l5_eval = materialize_l5_minimal(root, l1_artifact, l3_features, l4_forecasts, l4_models)
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
        l1_5_artifact, l1_5_axes = materialize_l1_5_diagnostic(root, l1_artifact)
        artifacts["l1_5_diagnostic_v1"] = l1_5_artifact
        resolved_axes["l1_5"] = l1_5_axes
    if "2_5_pre_post_preprocessing" in root:
        l2_5_artifact, l2_5_axes = materialize_l2_5_diagnostic(root, l1_artifact, l2_artifact)
        artifacts["l2_5_diagnostic_v1"] = l2_5_artifact
        resolved_axes["l2_5"] = l2_5_axes
    if "3_5_feature_diagnostics" in root:
        l3_5_artifact, l3_5_axes = materialize_l3_5_diagnostic(root, l1_artifact, l2_artifact, l3_features, l3_metadata)
        artifacts["l3_5_diagnostic_v1"] = l3_5_artifact
        resolved_axes["l3_5"] = l3_5_axes
    if "4_5_generator_diagnostics" in root:
        l4_5_artifact, l4_5_axes = materialize_l4_5_diagnostic(root, l3_features, l4_forecasts, l4_models, l4_training)
        artifacts["l4_5_diagnostic_v1"] = l4_5_artifact
        resolved_axes["l4_5"] = l4_5_axes
    if "6_statistical_tests" in root:
        l6_tests, l6_axes = materialize_l6_runtime(root, l1_artifact, l3_features, l4_forecasts, l4_models, l5_eval)
        artifacts["l6_tests_v1"] = l6_tests
        resolved_axes["l6"] = l6_axes
    if "7_interpretation" in root:
        l7_importance, l7_transform, l7_axes = materialize_l7_runtime(root, l3_features, l3_metadata, l4_forecasts, l4_models, l5_eval, artifacts.get("l6_tests_v1"))
        artifacts["l7_importance_v1"] = l7_importance
        artifacts["l7_transformation_attribution_v1"] = l7_transform
        resolved_axes["l7"] = l7_axes
    if "8_output" in root:
        l8_artifacts, l8_axes = materialize_l8_runtime(root, artifacts)
        artifacts["l8_artifacts_v1"] = l8_artifacts
        resolved_axes["l8"] = l8_axes
    return RuntimeResult(
        artifacts=artifacts,
        resolved_axes=resolved_axes,
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
    return artifact, l1_layer._regime_artifact_from_resolved(resolved, leaf_config), resolved


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

    df, transform_map = _apply_transform(df, resolved, leaf_config, l1_leaf_for_l2, cleaning_log)
    df, n_outliers = _apply_outlier_policy(df, resolved, leaf_config, cleaning_log)
    df, n_imputed = _apply_imputation(df, resolved, cleaning_log)
    df, n_truncated = _apply_frame_edge(df, resolved, cleaning_log)

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
    return (
        DiagnosticArtifact(
            layer_hooked="l1",
            artifact_type="json",
            metadata=metadata,
            enabled=True,
        ),
        axes,
    )


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
    return (
        DiagnosticArtifact(
            layer_hooked="l1+l2",
            artifact_type="json",
            metadata=metadata,
            enabled=True,
        ),
        axes,
    )


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
    horizon = int((y.attrs or {}).get("horizon", l1_artifact.target_horizons[0] if l1_artifact.target_horizons else 1))
    return (
        L3FeaturesArtifact(
            X_final=_panel_from_frame(X_aligned, metadata={"stage": "l3_X_final", "runtime": "l3_dag"}),
            y_final=Series(
                shape=y_aligned.shape,
                name=target_name,
                metadata=SeriesMetadata(values={"stage": "l3_y_final", "horizon": horizon, "data": y_aligned}),
            ),
            sample_index=pd.DatetimeIndex(aligned_index),
            horizon_set=(horizon,),
        ),
        l3_layer.build_metadata_artifact(raw),
    )


def materialize_l4_minimal(
    recipe_root: dict[str, Any], l3_features: L3FeaturesArtifact
) -> tuple[L4ForecastsArtifact, L4ModelArtifactsArtifact, L4TrainingMetadataArtifact]:
    raw = recipe_root.get("4_forecasting_model", {}) or {}
    report = l4_layer.validate_layer(raw, recipe_context={"horizon_set": set(l3_features.horizon_set)})
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    fit_nodes = [node for node in raw.get("nodes", ()) or () if isinstance(node, dict) and node.get("op") == "fit_model"]
    if not fit_nodes:
        raise ValueError("minimal L4 runtime requires a fit_model node")
    X = l3_features.X_final.data
    y = l3_features.y_final.metadata.values.get("data")
    if not isinstance(y, pd.Series):
        raise ValueError("minimal L4 runtime requires L3 y_final series data")
    target = l3_features.y_final.name
    horizon = int(l3_features.horizon_set[0] if l3_features.horizon_set else 1)
    forecasts: dict[tuple[str, str, int, Any], float] = {}
    artifacts: dict[str, ModelArtifact] = {}
    benchmark_flags: dict[str, bool] = {}
    refit_origins: dict[str, tuple[Any, ...]] = {}
    training_windows: dict[tuple[str, Any], tuple[Any, Any]] = {}
    model_ids: list[str] = []
    for fit_node in fit_nodes:
        params = fit_node.get("params", {}) or {}
        family = params.get("family", "ridge")
        if family not in {"ols", "ridge", "lasso", "elastic_net"}:
            raise NotImplementedError("minimal L4 runtime currently supports linear sklearn families only")
        alpha = float(params.get("alpha", 1.0))
        min_train_size = _minimal_train_size(params, n_obs=len(X), n_features=len(X.columns))
        model_id = fit_node.get("id", "fit_model")
        model_ids.append(model_id)
        origins = []
        for position in range(min_train_size, len(X)):
            origin = X.index[position]
            train_X = X.iloc[:position]
            train_y = y.iloc[:position]
            origin_model = _build_l4_linear_model(family, params, alpha=alpha)
            origin_model.fit(train_X, train_y)
            forecast = float(origin_model.predict(X.iloc[[position]])[0])
            forecasts[(model_id, target, horizon, origin)] = forecast
            origins.append(origin)
            training_windows[(model_id, origin)] = (train_X.index[0], train_X.index[-1])

        model = _build_l4_linear_model(family, params, alpha=alpha)
        model.fit(X, y)
        artifacts[model_id] = ModelArtifact(
            model_id=model_id,
            family=family,
            fitted_object=model,
            framework="sklearn",
            fit_metadata={"alpha": alpha, "n_obs": len(X), "min_train_size": min_train_size, "runtime": "expanding_direct"},
            feature_names=tuple(X.columns),
        )
        benchmark_flags[model_id] = bool(fit_node.get("is_benchmark", False))
        refit_origins[model_id] = tuple(origins)

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


def _build_l4_linear_model(family: str, params: dict[str, Any], *, alpha: float):
    if family == "ols":
        return LinearRegression()
    if family == "ridge":
        return Ridge(alpha=alpha)
    if family == "lasso":
        return Lasso(alpha=alpha, max_iter=int(params.get("max_iter", 10000)))
    if family == "elastic_net":
        return ElasticNet(
            alpha=alpha,
            l1_ratio=float(params.get("l1_ratio", params.get("lambda1_ratio", 0.5))),
            max_iter=int(params.get("max_iter", 10000)),
        )
    raise NotImplementedError(f"minimal L4 runtime does not support family={family!r}")


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
    actual = l3_features.y_final.metadata.values.get("data")
    if not isinstance(actual, pd.Series):
        raise ValueError("minimal L5 runtime requires L3 y_final series data")
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
                "origin": origin,
                "squared_error": error**2,
                "absolute_error": abs(error),
            }
        )
    if not rows:
        metrics = pd.DataFrame(columns=["model_id", "target", "horizon", "mse", "rmse", "mae"])
    else:
        errors = pd.DataFrame(rows)
        metrics = errors.groupby(["model_id", "target", "horizon"], as_index=False).agg(mse=("squared_error", "mean"), mae=("absolute_error", "mean"))
        metrics["rmse"] = metrics["mse"] ** 0.5
        metrics = _add_l5_relative_metrics(metrics, l4_models)
    if metrics.empty:
        ranking = pd.DataFrame()
    else:
        resolved_axes = l5_layer.resolve_axes_from_raw(raw.get("fixed_axes", {}) or {}, context=context)
        ranking_metric = _l5_ranking_metric(metrics, resolved_axes)
        ranking = metrics.sort_values(ranking_metric, ascending=_l5_rank_ascending(ranking_metric)).assign(
            rank_method="by_primary_metric",
            rank_value=lambda frame: range(1, len(frame) + 1),
        )
    return L5EvaluationArtifact(
        metrics_table=metrics,
        ranking_table=ranking,
        l5_axis_resolved=dict(l5_layer.resolve_axes_from_raw(raw.get("fixed_axes", {}) or {}, context=context)),
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

    if resolved["L6_A_equal_predictive"]["enabled"]:
        equal_results = _l6_equal_predictive_results(errors, resolved["L6_A_equal_predictive"], raw.get("leaf_config", {}) or {}, l4_models)
    if resolved["L6_B_nested"]["enabled"]:
        nested_results = _l6_nested_results(errors, resolved["L6_B_nested"], raw.get("leaf_config", {}) or {}, l4_models)
    if resolved["L6_C_cpa"]["enabled"]:
        cpa_results = _l6_cpa_results(errors, resolved["L6_C_cpa"], l4_models)
    if resolved["L6_D_multiple_model"]["enabled"]:
        multiple_results = _l6_multiple_model_results(l5_eval.metrics_table, resolved["L6_D_multiple_model"])
    if resolved["L6_E_density_interval"]["enabled"]:
        # The minimal runtime currently materializes point forecasts only; schema validation rejects this path.
        raise NotImplementedError("L6 density and interval tests require quantile or density forecasts")
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
    if op in {"permutation_importance", "shap_kernel"}:
        model = _first_model_input(inputs)
        X = next((item for item in inputs if isinstance(item, pd.DataFrame)), l3_features.X_final.data)
        y = next((item for item in inputs if isinstance(item, pd.Series)), l3_features.y_final.metadata.values.get("data"))
        frame = _permutation_importance_frame(model, X, y if isinstance(y, pd.Series) else None, method=op)
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
    raise NotImplementedError(f"minimal L7 runtime does not support op {op!r}")


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
    metrics = l5_eval.metrics_table
    if metrics.empty:
        summary = pd.DataFrame(columns=["pipeline", "target", "horizon", "contribution"])
        contributions: dict[tuple[Any, ...], Any] = {}
    else:
        metric = params.get("loss_function", "mse")
        metric = metric if metric in metrics.columns else "mse"
        summary = metrics.assign(pipeline=metrics["model_id"]).rename(columns={metric: "contribution"})[["pipeline", "target", "horizon", "contribution"]]
        contributions = {(row.target, int(row.horizon), row.pipeline): float(row.contribution) for row in summary.itertuples()}
    return L7TransformationAttributionArtifact(
        pipeline_contributions=contributions,
        decomposition_method=params.get("decomposition_method", "shapley_over_pipelines"),
        loss_function=params.get("loss_function", "mse"),
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
    loss_col = "absolute" if sub.get("loss_function") == "absolute" else "squared"
    for test_name in tests:
        for model_a, model_b in pairs:
            for (target, horizon), group in errors.groupby(["target", "horizon"]):
                left = group.loc[group["model_id"] == model_a, ["origin", loss_col]].rename(columns={loss_col: "loss_a"})
                right = group.loc[group["model_id"] == model_b, ["origin", loss_col]].rename(columns={loss_col: "loss_b"})
                joined = left.merge(right, on="origin", how="inner")
                diff = joined["loss_a"] - joined["loss_b"]
                stat, p_value = _t_statistic(diff)
                results[(test_name, model_a, model_b, target, int(horizon))] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "decision_at_5pct": p_value is not None and p_value < 0.05,
                    "n_obs": int(diff.notna().sum()),
                    "mean_loss_difference": _float_or_none(diff.mean()) if not diff.empty else None,
                }
    return results


def _l6_nested_results(
    errors: pd.DataFrame, sub: dict[str, Any], leaf: dict[str, Any], l4_models: L4ModelArtifactsArtifact
) -> dict[tuple[Any, ...], Any]:
    model_ids = sorted(errors["model_id"].unique()) if not errors.empty else []
    pairs = _l6_pair_list({"nested_pair_strategy": sub.get("nested_pair_strategy")}, leaf, model_ids, l4_models)
    tests = ["clark_west", "enc_new", "enc_t"] if sub.get("nested_test") == "multi" else [sub.get("nested_test")]
    results: dict[tuple[Any, ...], Any] = {}
    for test_name in tests:
        for large_model, small_model in pairs:
            for (target, horizon), group in errors.groupby(["target", "horizon"]):
                small = group.loc[group["model_id"] == small_model, ["origin", "squared"]].rename(columns={"squared": "loss_small"})
                large = group.loc[group["model_id"] == large_model, ["origin", "squared"]].rename(columns={"squared": "loss_large"})
                joined = small.merge(large, on="origin", how="inner")
                improvement = joined["loss_small"] - joined["loss_large"]
                stat, p_value = _t_statistic(improvement)
                results[(test_name, small_model, large_model, target, int(horizon))] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "decision_at_5pct": p_value is not None and p_value < 0.05,
                    "n_obs": int(improvement.notna().sum()),
                    "mean_adjusted_improvement": _float_or_none(improvement.mean()) if not improvement.empty else None,
                }
    return results


def _l6_cpa_results(errors: pd.DataFrame, sub: dict[str, Any], l4_models: L4ModelArtifactsArtifact) -> dict[tuple[Any, ...], Any]:
    results: dict[tuple[Any, ...], Any] = {}
    model_ids = sorted(errors["model_id"].unique()) if not errors.empty else []
    pairs = _l6_pair_list({"model_pair_strategy": "vs_benchmark_only"}, {}, model_ids, l4_models)
    tests = ["giacomini_rossi_2010", "rossi_sekhposyan"] if sub.get("cpa_test") == "multi" else [sub.get("cpa_test")]
    for test_name in tests:
        for model_a, model_b in pairs:
            for (target, horizon), group in errors.groupby(["target", "horizon"]):
                left = group.loc[group["model_id"] == model_a, ["origin", "squared"]].rename(columns={"squared": "loss_a"})
                right = group.loc[group["model_id"] == model_b, ["origin", "squared"]].rename(columns={"squared": "loss_b"})
                joined = left.merge(right, on="origin", how="inner").sort_values("origin")
                diff = joined["loss_a"] - joined["loss_b"]
                centered = diff - diff.mean() if not diff.empty else diff
                path = centered.cumsum().tolist()
                statistic = float(max(abs(value) for value in path)) if path else None
                results[(test_name, (model_a, model_b), target, int(horizon))] = {
                    "statistic": statistic,
                    "p_value": None,
                    "time_path": path,
                    "decision": None,
                }
    return results


def _l6_multiple_model_results(metrics: pd.DataFrame, sub: dict[str, Any]) -> dict[str, Any]:
    if metrics.empty:
        return {"mcs_inclusion": {}, "spa_p_values": {}, "reality_check_p_values": {}, "stepm_rejected": {}}
    metric = "mse" if "mse" in metrics.columns else metrics.select_dtypes("number").columns[0]
    alpha = float(sub.get("mcs_alpha", 0.10))
    mcs: dict[tuple[Any, ...], set[str]] = {}
    spa: dict[tuple[Any, ...], float] = {}
    reality: dict[tuple[Any, ...], float] = {}
    stepm: dict[tuple[Any, ...], set[str]] = {}
    for (target, horizon), group in metrics.groupby(["target", "horizon"]):
        best = float(group[metric].min())
        tolerance = max(abs(best) * alpha, 1e-12)
        included = set(group.loc[group[metric] <= best + tolerance, "model_id"].astype(str))
        key = (target, int(horizon), alpha)
        mcs[key] = included
        spa[(target, int(horizon))] = 1.0 if len(included) == len(group) else alpha
        reality[(target, int(horizon))] = spa[(target, int(horizon))]
        stepm[key] = set(group.loc[group[metric] > best + tolerance, "model_id"].astype(str))
    return {"mcs_inclusion": mcs, "spa_p_values": spa, "reality_check_p_values": reality, "stepm_rejected": stepm}


def _l6_direction_results(errors: pd.DataFrame, sub: dict[str, Any], leaf: dict[str, Any]) -> dict[tuple[Any, ...], Any]:
    threshold = leaf.get("direction_threshold_value", 0.0) if sub.get("direction_threshold") == "user_defined" else 0.0
    results: dict[tuple[Any, ...], Any] = {}
    tests = ["pesaran_timmermann_1992", "henriksson_merton"] if sub.get("direction_test") == "multi" else [sub.get("direction_test")]
    for test_name in tests:
        for (model_id, target, horizon), group in errors.groupby(["model_id", "target", "horizon"]):
            hit = (_sign_series(group["forecast"] - threshold) == _sign_series(group["actual"] - threshold)).astype(float)
            success = float(hit.mean()) if len(hit) else None
            stat, p_value = _binomial_direction_stat(hit)
            results[(test_name, model_id, target, int(horizon))] = {"statistic": stat, "p_value": p_value, "success_ratio": success}
    return results


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
    values = residuals.astype(float)
    if test_name == "durbin_watson":
        denom = float((values**2).sum())
        return (float((values.diff().dropna() ** 2).sum() / denom) if denom else None), None
    if test_name == "jarque_bera_normality":
        if len(values) < 3:
            return None, None
        skew = float(values.skew())
        kurt = float(values.kurtosis() + 3.0)
        jb = len(values) / 6.0 * (skew**2 + ((kurt - 3.0) ** 2) / 4.0)
        return jb, math.exp(-jb / 2.0)
    max_lag = min(lag, len(values) - 1)
    if max_lag < 1:
        return None, None
    acfs = [_autocorr(values, k) for k in range(1, max_lag + 1)]
    acfs = [0.0 if pd.isna(value) else value for value in acfs]
    if test_name in {"ljung_box_q", "breusch_godfrey_serial_correlation"}:
        q = len(values) * (len(values) + 2) * sum((rho**2) / max(len(values) - k, 1) for k, rho in enumerate(acfs, start=1))
        return q, math.exp(-q / 2.0)
    if test_name == "arch_lm":
        squared = values**2
        rho = _autocorr(squared, 1) if len(squared) > 1 else 0.0
        stat = len(squared) * (0.0 if pd.isna(rho) else rho**2)
        return stat, math.exp(-stat / 2.0)
    return None, None


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


def materialize_l8_runtime(recipe_root: dict[str, Any], upstream_artifacts: dict[str, Any]) -> tuple[L8ArtifactsArtifact, dict[str, Any]]:
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
    manifest = L8Manifest(
        recipe_hash=str(abs(hash(json.dumps(_jsonable(recipe_root), sort_keys=True)))),
        package_version="runtime-local",
        python_version=platform.python_version(),
        random_seed_used=((recipe_root.get("0_meta", {}) or {}).get("fixed_axes", {}) or {}).get("random_seed"),
        runtime_environment=RuntimeEnvironment(os_name=platform.system(), python_version=platform.python_version(), cpu_info=platform.machine()),
        dependency_lockfile_paths=_dependency_lockfile_paths(),
        cells_summary=[{"cell_id": "cell_001", "status": "completed", "exported_files": [file.path.as_posix() for file in exported_files]}],
    )
    manifest_path = output_directory / ("manifest.jsonl" if axes.get("manifest_format") == "json_lines" else "manifest.json")
    manifest_payload = {
        "manifest": _jsonable(manifest),
        "provenance_fields": axes.get("provenance_fields", []),
        "saved_objects": axes.get("saved_objects", []),
        "upstream_sinks": sorted(upstream_artifacts),
    }
    if axes.get("manifest_format") == "json_lines":
        manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8")
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


def _l8_export_artifacts(output_directory: Path, axes: dict[str, Any], upstream_artifacts: dict[str, Any], recipe_root: dict[str, Any]) -> list[ExportedFile]:
    saved = set(axes.get("saved_objects", []))
    exported: list[ExportedFile] = []
    summary_dir = output_directory / "summary"
    cell_dir = output_directory / "cell_001"
    cell_dir.mkdir(exist_ok=True)

    def add_dataframe(path: Path, frame: pd.DataFrame, source: str) -> None:
        frame.to_csv(path, index=True)
        exported.append(ExportedFile(path=path, artifact_type="csv", source_sink=source))

    def add_json(path: Path, payload: Any, source: str) -> None:
        path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        exported.append(ExportedFile(path=path, artifact_type="json", source_sink=source))

    if "forecasts" in saved and "l4_forecasts_v1" in upstream_artifacts:
        rows = [
            {"model_id": model_id, "target": target, "horizon": horizon, "origin": origin, "forecast": forecast}
            for (model_id, target, horizon, origin), forecast in upstream_artifacts["l4_forecasts_v1"].forecasts.items()
        ]
        add_dataframe(cell_dir / "forecasts.csv", pd.DataFrame(rows), "l4_forecasts_v1")
    if "metrics" in saved and "l5_evaluation_v1" in upstream_artifacts:
        add_dataframe(summary_dir / "metrics_all_cells.csv", upstream_artifacts["l5_evaluation_v1"].metrics_table, "l5_evaluation_v1")
    if "ranking" in saved and "l5_evaluation_v1" in upstream_artifacts:
        add_dataframe(summary_dir / "ranking.csv", upstream_artifacts["l5_evaluation_v1"].ranking_table, "l5_evaluation_v1")
    if "tests" in saved and "l6_tests_v1" in upstream_artifacts:
        add_json(output_directory / "tests_summary.json", upstream_artifacts["l6_tests_v1"], "l6_tests_v1")
    if "importance" in saved and "l7_importance_v1" in upstream_artifacts:
        add_json(output_directory / "importance_summary.json", upstream_artifacts["l7_importance_v1"], "l7_importance_v1")
    if "feature_metadata" in saved and "l3_metadata_v1" in upstream_artifacts:
        add_json(cell_dir / "feature_metadata.json", upstream_artifacts["l3_metadata_v1"], "l3_metadata_v1")
    if "clean_panel" in saved and "l2_clean_panel_v1" in upstream_artifacts:
        add_dataframe(cell_dir / "clean_panel.csv", upstream_artifacts["l2_clean_panel_v1"].panel.data, "l2_clean_panel_v1")
    if "raw_panel" in saved and "l1_data_definition_v1" in upstream_artifacts:
        add_dataframe(cell_dir / "raw_panel.csv", upstream_artifacts["l1_data_definition_v1"].raw_panel.data, "l1_data_definition_v1")
    for sink_name, artifact in upstream_artifacts.items():
        if sink_name.endswith("_diagnostic_v1"):
            object_name = f"diagnostics_{sink_name.split('_diagnostic_v1')[0]}"
            if object_name in saved:
                diag_dir = output_directory / "diagnostics"
                diag_dir.mkdir(exist_ok=True)
                add_json(diag_dir / f"{sink_name}.json", artifact, sink_name)
    return exported


def _dependency_lockfile_paths() -> dict[str, str]:
    paths: dict[str, str] = {}
    for candidate in ("uv.lock", "requirements.txt", "pyproject.toml"):
        path = Path(candidate)
        if path.exists():
            paths["python"] = path.as_posix()
            break
    return paths


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
        if policy == "official_plus_custom":
            raise NotImplementedError("official_plus_custom core runtime loading is deferred")
        if "custom_panel_inline" in leaf_config:
            frame = pd.DataFrame(leaf_config["custom_panel_inline"])
        elif "custom_panel_records" in leaf_config:
            frame = pd.DataFrame.from_records(leaf_config["custom_panel_records"])
        elif "custom_source_path" in leaf_config:
            frame = _read_custom_panel_path(Path(leaf_config["custom_source_path"]))
        else:
            raise ValueError("custom panel runtime requires custom_panel_inline, custom_panel_records, or custom_source_path")
        metadata = {"stage": "l1_raw", "source": "custom_panel"}
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
    raise NotImplementedError(f"official dataset {dataset!r} is not supported by core L1 runtime yet")


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
    if resolved.get("sample_start_rule") == "fixed_date":
        result = result.loc[pd.Timestamp(leaf_config["sample_start_date"]) :]
    if resolved.get("sample_end_rule") == "fixed_date":
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
    if policy in {"mean", "em_factor", "em_multivariate"}:
        result = frame.fillna(frame.mean(numeric_only=True))
        method = policy if policy == "mean" else f"{policy}_mean_fallback"
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
    values: dict[str, Any] = {}
    for node in _topological_nodes(dag):
        if node.type == "source":
            values[node.id] = _execute_l3_source(node.selector, frame, target_name)
        elif node.op == "l3_feature_bundle":
            values[node.id] = tuple(values[ref.node_id] for ref in node.inputs)
        elif node.op == "l3_metadata_build":
            values[node.id] = None
        else:
            inputs = [values[ref.node_id] for ref in node.inputs]
            values[node.id] = _execute_l3_op(node.op, inputs, node.params, target_name)
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


def _execute_l3_op(op: str, inputs: list[Any], params: dict[str, Any], target_name: str) -> pd.DataFrame | pd.Series:
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
    if op == "cumsum":
        return inputs[0].cumsum()
    if op in {"polynomial_expansion", "polynomial"}:
        return _polynomial_expansion(_as_frame(inputs[0]), degree=int(params.get("degree", 2)))
    if op == "interaction":
        return _interaction_terms(_as_frame(inputs[0]))
    if op == "season_dummy":
        return _season_dummy(_as_frame(inputs[0]))
    if op == "time_trend":
        frame = _as_frame(inputs[0])
        return pd.Series(range(1, len(frame) + 1), index=frame.index, name="time_trend")
    if op == "target_construction":
        horizon = int(params.get("horizon", 1))
        y = _as_series(inputs[0], name=target_name).shift(-horizon).rename(target_name)
        y.attrs["horizon"] = horizon
        return y
    raise NotImplementedError(f"minimal L3 runtime does not support op {op!r}")


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
    if method not in {"zscore", "standard", "standardize"}:
        raise NotImplementedError(f"minimal L3 runtime does not support scale method {method!r}")
    mean = frame.mean().to_numpy()
    std = frame.std(ddof=0).replace(0, pd.NA).to_numpy()
    scaled = (frame.to_numpy() - mean) / std
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
