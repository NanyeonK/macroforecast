from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .layers import l1 as l1_layer
from .layers import l2 as l2_layer
from .types import L1DataDefinitionArtifact, L1RegimeMetadataArtifact, L2CleanPanelArtifact, Panel, PanelMetadata
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
    return RuntimeResult(
        artifacts={
            "l1_data_definition_v1": l1_artifact,
            "l1_regime_metadata_v1": regime_artifact,
            "l2_clean_panel_v1": l2_artifact,
        },
        resolved_axes={"l1": l1_axes, "l2": dict(l2_axes)},
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

    df, transform_map = _apply_transform(df, resolved, leaf_config, l1_artifact.leaf_config, cleaning_log)
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


def _load_raw_panel(resolved: dict[str, Any], leaf_config: dict[str, Any]) -> Panel:
    if resolved["custom_source_policy"] not in {"custom_panel_only", "official_plus_custom"}:
        raise NotImplementedError("official FRED runtime loading is deferred; use custom_panel_only for Runtime-1")
    if "custom_panel_inline" in leaf_config:
        frame = pd.DataFrame(leaf_config["custom_panel_inline"])
    elif "custom_panel_records" in leaf_config:
        frame = pd.DataFrame.from_records(leaf_config["custom_panel_records"])
    elif "custom_source_path" in leaf_config:
        frame = _read_custom_panel_path(Path(leaf_config["custom_source_path"]))
    else:
        raise ValueError("custom panel runtime requires custom_panel_inline, custom_panel_records, or custom_source_path")
    frame = _normalize_datetime_index(frame, leaf_config)
    frame = _apply_sample_window(frame, resolved, leaf_config)
    _validate_targets_present(frame, leaf_config, resolved)
    return _panel_from_frame(frame, metadata={"stage": "l1_raw", "source": "custom_panel"})


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
    tcode_map = dict(l1_leaf.get("custom_tcode_map", {}))
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
        result[numeric.columns] = numeric.clip(numeric.quantile(low), numeric.quantile(high), axis=1)
        cleaning_log["steps"].append({"outlier": "winsorize", "quantiles": [low, high]})
        return result, 0
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


def _l1_context(artifact: L1DataDefinitionArtifact) -> dict[str, Any]:
    return {
        "custom_source_policy": artifact.custom_source_policy,
        "dataset": artifact.dataset,
        "frequency": artifact.frequency,
        "custom_has_tcode_column": bool(artifact.leaf_config.get("custom_tcode_map")),
    }
