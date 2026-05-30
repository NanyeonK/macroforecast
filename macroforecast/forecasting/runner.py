from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pandas as pd

from macroforecast.data import DataBundle, as_panel, panel_info, spec as data_spec, validate_panel
from macroforecast.feature_engineering import FeatureSet, FeatureSpec, feature_spec
from macroforecast.meta import get_config
from macroforecast.models import ModelSpec, get_model
from macroforecast.preprocessing import FittedPreprocessor, PreprocessSpec
from macroforecast.selection import SearchSpec, select_params
from macroforecast.window import (
    Split,
    StagePolicy,
    WindowSpec,
    resolve_stage_policy,
    resolve_window,
    stage_index,
    stage_panel,
)
from macroforecast.forecasting.types import ForecastResult


@dataclass(frozen=True)
class _ModelRun:
    alias: str
    spec: ModelSpec


@dataclass(frozen=True)
class _PreparedStage:
    panel: pd.DataFrame
    fitted_preprocessing: FittedPreprocessor | None
    metadata: dict[str, Any] | None


def run(
    data: Any,
    model: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, Any],
    *,
    window: WindowSpec | str | None = None,
    preprocessing: PreprocessSpec | None = None,
    preprocessing_policy: StagePolicy | str | None = None,
    features: FeatureSpec | None = None,
    feature_policy: StagePolicy | str | None = None,
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None = None,
    selection_policy: StagePolicy | str | None = None,
    selection_metric: str | Callable[..., float] = "mse",
    maximize_selection: bool = False,
    preset: str | Mapping[str, str | None] | None = None,
    params: Mapping[str, Any] | None = None,
    target: str | None = None,
    horizon: int = 1,
) -> ForecastResult:
    """Run a windowed macro forecasting experiment.

    The runner composes small stage callables. ``window`` owns the temporal
    design, stage policies decide where preprocessing/features/selection are
    fitted, model specs fit predictors to targets, and the result records a
    run-level metadata ledger.
    """

    window_spec = resolve_window(window)
    config = get_config()
    model_runs = _resolve_model_runs(model, preset=preset, params=params)
    preprocessing_stage_policy = (
        resolve_stage_policy(
            preprocessing_policy,
            default_scope=str(config["default_preprocessing_scope"]),
        )
        if preprocessing is not None
        else None
    )
    feature_stage_policy = resolve_stage_policy(
        feature_policy,
        default_scope=str(config["default_feature_scope"]),
    )
    selection_stage_policy = resolve_stage_policy(
        selection_policy,
        default_scope=str(config["default_selection_scope"]),
    )
    _validate_runner_policies(
        preprocessing=preprocessing,
        preprocessing_policy=preprocessing_stage_policy,
        feature_policy=feature_stage_policy,
        selection_policy=selection_stage_policy,
    )

    if isinstance(data, FeatureSet):
        return _run_feature_set(
            data,
            model_runs=model_runs,
            window_spec=window_spec,
            selection=selection,
            selection_policy=selection_stage_policy,
            selection_metric=selection_metric,
            maximize_selection=maximize_selection,
            config=config,
        )

    panel = as_panel(pd.DataFrame(data).copy())
    validate_panel(panel)
    if features is None:
        if target is None:
            raise ValueError("target is required when data is not a FeatureSet")
        features = feature_spec(target=target, horizon=horizon)

    full_stage: _PreparedStage | None = None
    fixed_feature_builder: Any | None = None
    if preprocessing is not None and preprocessing_stage_policy is not None and preprocessing_stage_policy.scope == "full_panel":
        fitted = preprocessing.fit(_preprocessor_fit_input(panel, features), policy="origin_available")
        full_stage = _PreparedStage(
            panel=fitted.processed_train.panel,
            fitted_preprocessing=fitted,
            metadata=fitted.to_metadata(),
        )
    elif preprocessing is None:
        full_stage = _PreparedStage(panel=panel, fitted_preprocessing=None, metadata=None)

    if feature_stage_policy.scope in {"full_panel", "fixed_reference"}:
        if full_stage is None:
            raise ValueError(
                "feature_policy with scope='full_panel' or 'fixed_reference' requires "
                "preprocessing=None or preprocessing_policy scope='full_panel'"
            )
        feature_fit_panel = stage_panel(
            full_stage.panel,
            None,
            feature_stage_policy,
        )
        fixed_feature_builder = features.fit(feature_fit_panel)

    records: list[dict[str, Any]] = []
    model_param_cache: dict[str, dict[str, Any]] = {}
    selection_cache: dict[str, dict[str, Any] | None] = {}
    stage_records: list[dict[str, Any]] = []

    for item in window_spec.iter_origins(panel.index):
        prepared = (
            _prepare_origin_panel(
                panel,
                features=features,
                preprocessing=preprocessing,
                preprocessing_policy=preprocessing_stage_policy,
                item=item,
            )
            if full_stage is None
            else full_stage
        )
        if prepared.metadata is not None:
            stage_records.append(_origin_stage_record("preprocessing", item, prepared.metadata))

        fit_labels = panel.index[item["fit_idx"]]
        test_labels = panel.index[item["test_idx"]]
        selection_labels = stage_index(panel.index, item, selection_stage_policy)

        if fixed_feature_builder is None:
            feature_fit_labels = stage_index(panel.index, item, feature_stage_policy)
            feature_fit_panel = prepared.panel.reindex(feature_fit_labels).dropna(how="all")
            fitted_features = features.fit(feature_fit_panel)
        else:
            fitted_features = fixed_feature_builder

        train_features = fitted_features.transform(prepared.panel, index=fit_labels)
        test_features = _test_feature_builder(fitted_features).transform(prepared.panel, index=test_labels)
        selection_features = _test_feature_builder(fitted_features).transform(
            prepared.panel,
            index=selection_labels,
        )
        X_selection, y_selection = _align_feature_xy(
            selection_features.X,
            _single_target(selection_features.y),
        )
        selection_splits = _relative_splits_for_index(
            item.get("val_splits", []),
            X_selection.index,
            panel.index,
        )
        stage_records.append(_origin_stage_record("feature_engineering", item, fitted_features.to_metadata()))

        origin_item = {
            **item,
            "X_fit": train_features.X,
            "y_fit": _single_target(train_features.y),
            "X_selection": X_selection,
            "y_selection": y_selection,
            "selection_splits": selection_splits,
            "X_test": test_features.X,
            "y_test": _single_target(test_features.y),
        }
        origin_records = _fit_predict_origin(
            origin_item,
            model_runs=model_runs,
            selection=selection,
            selection_policy=selection_stage_policy,
            selection_metric=selection_metric,
            maximize_selection=maximize_selection,
            param_cache=model_param_cache,
            selection_cache=selection_cache,
            selection_random_state=config["random_seed"],
        )
        for record in origin_records:
            record["preprocessed"] = preprocessing is not None
        records.extend(origin_records)

    metadata = _result_metadata(
        input_panel=panel,
        window_spec=window_spec,
        model_runs=model_runs,
        features=features.to_dict(),
        preprocessing=preprocessing.to_dict() if preprocessing is not None else None,
        preprocessing_policy=preprocessing_stage_policy,
        feature_policy=feature_stage_policy,
        selection=selection,
        selection_policy=selection_stage_policy,
        stage_records=stage_records,
        n_forecasts=len(records),
        config=config,
    )
    return ForecastResult(pd.DataFrame.from_records(records), metadata=metadata)


def _run_feature_set(
    data: FeatureSet,
    *,
    model_runs: list[_ModelRun],
    window_spec: WindowSpec,
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    selection_metric: str | Callable[..., float],
    maximize_selection: bool,
    config: Mapping[str, Any],
) -> ForecastResult:
    X_all = data.X.copy()
    y_all = _single_target(data.y)
    validate_panel(X_all)
    records: list[dict[str, Any]] = []
    model_param_cache: dict[str, dict[str, Any]] = {}
    selection_cache: dict[str, dict[str, Any] | None] = {}
    for item in window_spec.iter_slices(X_all, y_all):
        X_selection, y_selection = _align_feature_xy(
            _select_existing_features(item, "X", selection_policy),
            _single_target(_select_existing_features(item, "y", selection_policy)),
        )
        item = {
            **item,
            "X_selection": X_selection,
            "y_selection": y_selection,
        }
        item["selection_splits"] = _relative_splits_for_index(
            item.get("val_splits", []),
            item["X_selection"].index,
            X_all.index,
        )
        records.extend(
            _fit_predict_origin(
                item,
                model_runs=model_runs,
                selection=selection,
                selection_policy=selection_policy,
                selection_metric=selection_metric,
                maximize_selection=maximize_selection,
                param_cache=model_param_cache,
                selection_cache=selection_cache,
                selection_random_state=config["random_seed"],
            )
        )
    metadata = _result_metadata(
        input_panel=X_all,
        window_spec=window_spec,
        model_runs=model_runs,
        features=data.metadata.get("feature_spec") or data.metadata.get("feature_engineering"),
        preprocessing=None,
        preprocessing_policy=None,
        feature_policy=None,
        selection=selection,
        selection_policy=selection_policy,
        stage_records=[],
        n_forecasts=len(records),
        config=config,
    )
    return ForecastResult(pd.DataFrame.from_records(records), metadata=metadata)


def _fit_predict_origin(
    item: dict[str, Any],
    *,
    model_runs: list[_ModelRun],
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    selection_metric: str | Callable[..., float],
    maximize_selection: bool,
    param_cache: dict[str, dict[str, Any]],
    selection_cache: dict[str, dict[str, Any] | None],
    selection_random_state: int | None,
) -> list[dict[str, Any]]:
    X_fit = item["X_fit"]
    y_fit = item["y_fit"]
    X_selection = item.get("X_selection", X_fit)
    y_selection = item.get("y_selection", y_fit)
    X_test = item["X_test"]
    y_test = item["y_test"]
    if X_fit.empty or X_test.empty:
        return []

    records: list[dict[str, Any]] = []
    row = item["row"]
    retune = bool(row.get("retune", True))
    for model_run in model_runs:
        model_spec = model_run.spec
        selected = _select_for_model(selection, model_run)
        should_select = selected is not None or bool(model_spec.search_spaces)
        selection_metadata: dict[str, Any] | None = None
        if should_select:
            if retune or model_run.alias not in param_cache:
                result = select_params(
                    model_spec,
                    X_selection,
                    y_selection,
                    search=selected,
                    splits=item.get("selection_splits"),
                    metric=selection_metric,
                    maximize=maximize_selection,
                    random_state=selection_random_state if selected is None else None,
                )
                param_cache[model_run.alias] = dict(result.best_params)
                selection_metadata = {
                    **result.to_metadata(),
                    "policy": selection_policy.to_dict(),
                    "retuned": True,
                }
                selection_cache[model_run.alias] = selection_metadata
            else:
                selection_metadata = selection_cache.get(model_run.alias)
                if selection_metadata is not None:
                    selection_metadata = {**selection_metadata, "retuned": False}
            best_params = dict(param_cache.get(model_run.alias, {}))
        else:
            best_params = {}
        fit = model_spec(X_fit, y_fit, **best_params)
        pred = _prediction_series(fit.predict(X_test), index=X_test.index)
        for date, value in pred.items():
            actual: Any = y_test.reindex([date]).iloc[0] if date in y_test.index else None
            actual_value = None if actual is None or pd.isna(actual) else float(actual)
            records.append(
                {
                    "date": date,
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "horizon": row.get("horizon"),
                    "model": model_run.alias,
                    "model_spec": model_spec.name,
                    "prediction": float(value),
                    "actual": actual_value,
                    "params": dict(best_params),
                    "selection": selection_metadata,
                    "window": row,
                }
            )
    return records


def _prepare_origin_panel(
    panel: pd.DataFrame,
    *,
    features: FeatureSpec,
    preprocessing: PreprocessSpec | None,
    preprocessing_policy: StagePolicy | None,
    item: dict[str, Any],
) -> _PreparedStage:
    if preprocessing is None or preprocessing_policy is None:
        return _PreparedStage(panel=panel, fitted_preprocessing=None, metadata=None)
    fit_panel = stage_panel(panel, item, preprocessing_policy)
    fit_policy = "fit_window" if preprocessing_policy.scope in {"fit_window", "fixed_reference"} else "origin_available"
    fitted = preprocessing.fit(
        _preprocessor_fit_input(fit_panel, features),
        policy=fit_policy,
    )
    apply_labels = _origin_apply_labels(panel.index, item)
    apply_panel = panel.reindex(apply_labels).loc[:, fitted.fit_panel.columns]
    if preprocessing_policy.scope in {"fit_window", "fixed_reference"}:
        transformed = fitted.transform(apply_panel, history=fitted.fit_panel, policy="fit_window")
        prepared_panel = transformed.panel
    else:
        test_panel = panel.iloc[item["test_idx"]].loc[:, fitted.fit_panel.columns]
        transformed = fitted.transform(test_panel, history=fitted.fit_panel, policy="origin_available")
        prepared_panel = _combine_processed_panels(fitted.processed_train.panel, transformed.panel)
    return _PreparedStage(
        panel=prepared_panel,
        fitted_preprocessing=fitted,
        metadata=fitted.to_metadata(),
    )


def _origin_apply_labels(index: pd.Index, item: dict[str, Any]) -> pd.Index:
    positions = np.unique(np.concatenate([item["estimation_idx"], item["fit_idx"], item["test_idx"]]))
    return index[positions]


def _single_target(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    frame = pd.DataFrame(y)
    if frame.shape[1] != 1:
        raise ValueError("forecasting runner currently expects exactly one target column")
    return frame.iloc[:, 0].rename(str(frame.columns[0]))


def _combine_processed_panels(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train_panel, test_panel], axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _test_feature_builder(builder: Any) -> Any:
    if not getattr(builder.spec, "drop_missing", True):
        return builder
    return replace(builder, spec=replace(builder.spec, drop_missing=False))


def _align_feature_xy(X: Any, y: Any) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(X).copy()
    target = _single_target(y)
    temp_name = "__macroforecast_selection_target__"
    while temp_name in frame.columns:
        temp_name = f"_{temp_name}"
    aligned = pd.concat([target.rename(temp_name), frame], axis=1).dropna()
    aligned_target = aligned.pop(temp_name)
    aligned_target.name = target.name
    return aligned, aligned_target


def _relative_splits_for_index(
    splits: Sequence[Split],
    selection_index: pd.Index,
    base_index: pd.Index,
) -> list[Split]:
    """Map absolute window positions onto a stage-local feature matrix."""

    if not splits:
        return []
    if not selection_index.is_unique:
        raise ValueError("selection feature index must be unique")
    out: list[Split] = []
    for split_id, (train_abs, val_abs) in enumerate(splits):
        train_labels = base_index[np.asarray(train_abs, dtype=int)]
        val_labels = base_index[np.asarray(val_abs, dtype=int)]
        train_pos = selection_index.get_indexer(train_labels)
        val_pos = selection_index.get_indexer(val_labels)
        train_pos = train_pos[train_pos >= 0]
        if len(train_pos) == 0:
            raise ValueError(
                f"validation split {split_id} has no train rows after feature alignment"
            )
        if (val_pos < 0).any():
            raise ValueError(
                f"validation split {split_id} is not contained in the selection feature index"
            )
        out.append((train_pos.astype(int, copy=False), val_pos.astype(int, copy=False)))
    return out


def _preprocessor_fit_input(fit_panel: pd.DataFrame, features: FeatureSpec) -> Any:
    target = features.target
    targets = features.targets or None
    if target is None and not targets:
        return fit_panel
    metadata = dict(fit_panel.attrs.get("macroforecast_metadata", {}))
    horizons: Any
    if features.horizons:
        horizons = features.horizons
    elif features.horizon is not None:
        horizons = features.horizon
    else:
        horizons = None
    predictors: Any = "all" if features.predictors is None else features.predictors
    return data_spec(
        DataBundle(fit_panel, metadata),
        target=target,
        targets=targets,
        horizons=horizons,
        predictors=predictors,
    )


def _resolve_model_runs(
    model: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, Any],
    *,
    preset: str | Mapping[str, str | None] | None,
    params: Mapping[str, Any] | None,
) -> list[_ModelRun]:
    if isinstance(model, Mapping):
        runs = []
        for alias, value in model.items():
            base = get_model(value)
            runs.append(
                _ModelRun(
                    alias=str(alias),
                    spec=get_model(
                        value,
                        preset=_preset_for_model(preset, str(alias), base.name),
                        params=_params_for_model(params, str(alias), base.name),
                    ),
                )
            )
    elif _is_model_sequence(model):
        runs = []
        seen: dict[str, int] = {}
        model_sequence = cast(Sequence[str | Callable[..., Any] | ModelSpec], model)
        for value in model_sequence:
            base = get_model(value)
            spec = get_model(
                value,
                preset=_preset_for_model(preset, None, base.name),
                params=_params_for_model(params, None, base.name),
            )
            count = seen.get(spec.name, 0) + 1
            seen[spec.name] = count
            alias = spec.name if count == 1 else f"{spec.name}_{count}"
            runs.append(_ModelRun(alias=alias, spec=spec))
    else:
        single_model = cast(str | Callable[..., Any] | ModelSpec, model)
        spec = get_model(
            single_model,
            preset=_preset_for_model(preset, None, None),
            params=_params_for_model(params, None, None),
        )
        runs = [_ModelRun(alias=spec.name, spec=spec)]
    if not runs:
        raise ValueError("model must contain at least one model")
    return runs


def _is_model_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _preset_for_model(
    preset: str | Mapping[str, str | None] | None,
    alias: str | None,
    model_name: str | None,
) -> str | None:
    if preset is None or isinstance(preset, str):
        return preset
    if alias is not None and alias in preset:
        return preset[alias]
    if model_name is not None and model_name in preset:
        return preset[model_name]
    return None


def _params_for_model(
    params: Mapping[str, Any] | None,
    alias: str | None,
    model_name: str | None,
) -> Mapping[str, Any] | None:
    if params is None:
        return None
    if alias is not None and isinstance(params.get(alias), Mapping):
        return params[alias]
    if model_name is not None and isinstance(params.get(model_name), Mapping):
        return params[model_name]
    if all(isinstance(value, Mapping) for value in params.values()):
        return None
    return params


def _select_for_model(
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    model_run: _ModelRun,
) -> SearchSpec | None:
    if selection is None or isinstance(selection, SearchSpec):
        return selection
    if model_run.alias in selection:
        return selection[model_run.alias]
    if model_run.spec.name in selection:
        return selection[model_run.spec.name]
    return None


def _prediction_series(prediction: Any, *, index: pd.Index) -> pd.Series:
    if isinstance(prediction, pd.Series):
        return prediction.reindex(index)
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError("model prediction DataFrame must have exactly one column")
        return prediction.iloc[:, 0].reindex(index)
    values = np.asarray(prediction).reshape(-1)
    if len(values) != len(index):
        raise ValueError("model prediction length does not match X_test")
    return pd.Series(values, index=index)


def _select_existing_features(item: dict[str, Any], prefix: str, policy: StagePolicy) -> Any:
    if policy.scope == "origin_available":
        return item[f"{prefix}_estimation"]
    return item[f"{prefix}_fit"]


def _validate_runner_policies(
    *,
    preprocessing: PreprocessSpec | None,
    preprocessing_policy: StagePolicy | None,
    feature_policy: StagePolicy,
    selection_policy: StagePolicy,
) -> None:
    policies = [policy for policy in (preprocessing_policy, feature_policy, selection_policy) if policy is not None]
    for policy in policies:
        if policy.scope == "custom":
            raise ValueError("custom stage policies require callable hooks and are not implemented in forecasting.run yet")
    if preprocessing is None:
        return
    if preprocessing_policy is None:
        return


def _origin_stage_record(stage: str, item: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    row = item["row"]
    return {
        "stage": stage,
        "origin": row.get("origin"),
        "origin_pos": row.get("origin_pos"),
        "fit_start": row.get("fit_start"),
        "fit_end": row.get("fit_end"),
        "test_start": row.get("test_start"),
        "test_end": row.get("test_end"),
        "metadata": metadata,
    }


def _result_metadata(
    *,
    input_panel: pd.DataFrame,
    window_spec: WindowSpec,
    model_runs: list[_ModelRun],
    features: Any,
    preprocessing: Any,
    preprocessing_policy: StagePolicy | None,
    feature_policy: StagePolicy | None,
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    stage_records: list[dict[str, Any]],
    n_forecasts: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_level = str(config.get("metadata_level", "standard"))
    return {
        "run": {
            "n_forecasts": int(n_forecasts),
            "n_models": len(model_runs),
            "config": dict(config),
        },
        "data": panel_info(DataBundle(input_panel, dict(input_panel.attrs.get("macroforecast_metadata", {})))),
        "window": window_spec.to_dict(),
        "stage_policies": {
            "preprocessing": None if preprocessing_policy is None else preprocessing_policy.to_dict(),
            "feature_engineering": None if feature_policy is None else feature_policy.to_dict(),
            "selection": selection_policy.to_dict(),
        },
        "preprocessing": preprocessing,
        "features": features,
        "selection": _selection_metadata(selection),
        "models": [
            {"alias": model_run.alias, "spec": model_run.spec.to_metadata()}
            for model_run in model_runs
        ],
        "stages": [] if metadata_level == "minimal" else stage_records,
    }


def _selection_metadata(selection: SearchSpec | Mapping[str, SearchSpec | None] | None) -> Any:
    if selection is None:
        return None
    if isinstance(selection, SearchSpec):
        return selection.to_dict()
    return {
        str(key): None if value is None else value.to_dict()
        for key, value in selection.items()
    }


__all__ = ["run"]
