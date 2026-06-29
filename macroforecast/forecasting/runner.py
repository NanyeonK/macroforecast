from __future__ import annotations

import warnings

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from macroforecast.data import (
    DataBundle,
    DataSpec,
    as_panel,
    panel_info,
    spec as data_spec,
    validate_panel,
)
from macroforecast.feature_engineering import FeatureSet, FeatureSpec, feature_spec
from macroforecast.feature_engineering.shared import TargetMode, TargetTransform
from macroforecast.meta import get_config
from macroforecast.model_ensemble import get_model_ensemble
from macroforecast.models import ModelSpec, get_model, save_fit
from macroforecast.preprocessing import FittedPreprocessor, PreprocessSpec
from macroforecast.preprocessing.cache import PreprocessorStore
from macroforecast.model_selection import (
    SearchSpec,
    select_by_information_criterion,
    select_params,
)
from macroforecast.window import (
    Split,
    StagePolicy,
    ValWindow,
    WindowSpec,
    make_splitter,
    resolve_stage_policy,
    resolve_window,
    stage_index,
    stage_panel,
)
from macroforecast.forecasting.combination import (
    CombinationSpec,
    apply_combinations,
    resolve_combinations,
)
from macroforecast.forecasting.types import ForecastResult
from macroforecast.forecasting.checkpoint import (
    LEAN_FORECAST_COLUMNS,
    append_origin_records,
    completed_origin_positions,
    load_checkpoint_frame,
)

ForecastPolicy = Literal["direct", "direct_average", "path_average", "recursive"]
FutureFeaturePolicy = Literal["target_lags", "observed_future"]


_FORECAST_TABLE_COLUMNS = (
    "date",
    "origin",
    "origin_pos",
    "horizon",
    "forecast_policy",
    "target_transform",
    "target",
    "model",
    "model_spec",
    "prediction",
    "variance_prediction",
    "quantile_predictions",
    "actual",
    "params",
    "model_selection",
    "stored_model",
    "window",
    "preprocessed",
    "combined",
    "combination",
)

_STAGE_RECORD_COLUMNS = (
    "stage",
    "origin",
    "origin_pos",
    "updated",
    "fit_start",
    "fit_end",
    "test_start",
    "test_end",
    "metadata",
)


@dataclass(frozen=True)
class _ModelRun:
    alias: str
    spec: ModelSpec


@dataclass(frozen=True)
class _PreparedStage:
    panel: pd.DataFrame
    fitted_preprocessing: FittedPreprocessor | None
    metadata: dict[str, Any] | None
    # Full per-origin panel metadata (the ``macroforecast_metadata`` payload that
    # ``_prepare_origin_panel`` would otherwise leave on ``panel.attrs``). It is
    # held here on the dataclass instead of on ``panel.attrs`` because the cached
    # ``panel`` is operated on repeatedly in the per-origin/per-arm hot loop, and
    # pandas deep-copies ``.attrs`` on every operation via ``__finalize__``. The
    # preprocessing metadata is large (EM/factor + standardization state, ~40 KB),
    # so carrying it on ``.attrs`` makes that deepcopy dominate runtime in
    # multi-horizon runs that share the panel across many origins. Consumers read
    # the metadata from this field and re-attach it only where it is needed.
    panel_metadata: dict[str, Any] | None = None


@dataclass
class _StageUpdateState:
    updated_once: bool = False
    last_origin: Any | None = None


def run(
    data: Any,
    model: str | Callable[..., Any] | ModelSpec,
    *,
    window: WindowSpec | str | None = None,
    preprocessing: PreprocessSpec | None = None,
    preprocessing_policy: StagePolicy | str | None = None,
    features: FeatureSpec | None = None,
    feature_policy: StagePolicy | str | None = None,
    model_selection: SearchSpec | Mapping[str, SearchSpec | None] | None = None,
    model_selection_policy: StagePolicy | str | None = None,
    model_selection_metric: str | Callable[..., float] = "mse",
    maximize_model_selection: bool = False,
    preset: str | Mapping[str, str | None] | None = None,
    params: Mapping[str, Any] | None = None,
    target: str | None = None,
    horizon: int = 1,
    horizons: Sequence[int] | int | None = None,
    forecast_policy: str = "direct",
    future_feature_policy: str | None = None,
    target_transform: str | None = None,
    combination: str
    | CombinationSpec
    | Sequence[str | CombinationSpec | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    save_models: bool = True,
    model_store: str | Path = "trained_model",
    preprocessing_cache: dict[Any, FittedPreprocessor | _PreparedStage] | None = None,
    preprocessing_store: PreprocessorStore | None = None,
    checkpoint_path: str | Path | None = None,
) -> ForecastResult:
    """Run a windowed macro forecasting experiment.

    The runner composes small stage callables. ``window`` owns the temporal
    design, stage policies decide where preprocessing, features, and model
    selection are fitted, model specs fit predictors to targets, and the result
    records a run-level metadata ledger.

    A ``run`` is ATOMIC: it fits exactly ONE model. ``model`` must be a single
    ``str`` model name, a ``Callable`` model factory, or a ``ModelSpec`` (a
    fit-time model-ensemble spec still counts as one model). Passing a sequence
    or a mapping of models raises ``TypeError`` -- run one model per call, or use
    the pipeline with one ``Arm`` per model when comparing models.
    """

    selection = model_selection
    selection_policy = model_selection_policy
    selection_metric = model_selection_metric
    maximize_selection = maximize_model_selection

    window_spec = resolve_window(window)
    horizon_values = _resolve_runner_horizons(horizon=horizon, horizons=horizons)
    policy = _normalize_forecast_policy(forecast_policy)
    future_policy = _normalize_future_feature_policy(
        future_feature_policy,
        forecast_policy=policy,
    )
    if isinstance(data, FeatureSet) and len(horizon_values) > 1:
        raise ValueError(
            "FeatureSet input is already target-constructed; run each prebuilt "
            "horizon-specific FeatureSet separately, or pass a panel with "
            "horizons=..."
        )
    if len(horizon_values) > 1:
        return _run_multiple_horizons(
            data,
            model,
            window=window_spec,
            preprocessing=preprocessing,
            preprocessing_policy=preprocessing_policy,
            features=features,
            feature_policy=feature_policy,
            selection=selection,
            selection_policy=selection_policy,
            selection_metric=selection_metric,
            maximize_selection=maximize_selection,
            preset=preset,
            params=params,
            target=target,
            horizons=horizon_values,
            forecast_policy=policy,
            future_feature_policy=future_policy,
            target_transform=target_transform,
            combination=combination,
            save_models=save_models,
            model_store=model_store,
            preprocessing_cache=preprocessing_cache,
            preprocessing_store=preprocessing_store,
            checkpoint_path=checkpoint_path,
        )
    config = get_config()
    # Namespace the checkpoint directory by horizon, mirroring the per-horizon
    # subdirectory that ``_run_multiple_horizons`` appends. Origin positions are
    # horizon-independent, so a single ``checkpoint_path`` reused across distinct
    # single-horizon runs of the SAME forecast cell (e.g. a per-horizon loop that
    # passes the same per-cell checkpoint dir) would otherwise collide: the first
    # horizon's lean records (carrying its own horizon/date) would be loaded as
    # "already done" for every later horizon, silently forecasting horizon 1 for
    # all of them. Anchoring on the resolved horizon keeps each horizon's origins
    # in their own ``h<h>`` subdirectory, so the single- and multi-horizon
    # checkpoint layouts are identical and never collide.
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path) / f"h{int(horizon_values[0])}"
    model_runs = _resolve_model_runs(model, preset=preset, params=params)
    _validate_selection_mapping(selection, model_runs)
    combination_specs = resolve_combinations(combination)
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
        if policy != "direct":
            raise ValueError(
                "FeatureSet input is already target-constructed; use "
                "forecast_policy='direct', or pass a panel plus FeatureSpec so "
                "the runner can build direct-average, path-average, or recursive targets."
            )
        _validate_feature_model_runs(model_runs)
        # preprocessing_store is intentionally not forwarded here: the FeatureSet
        # path bypasses _prepare_origin_panel (input is already preprocessed), so
        # there is no per-origin FittedPreprocessor to share.
        return _run_feature_set(
            data,
            model_runs=model_runs,
            window_spec=window_spec,
            selection=selection,
            selection_policy=selection_stage_policy,
            selection_metric=selection_metric,
            maximize_selection=maximize_selection,
            config=config,
            combination_specs=combination_specs,
            save_models=save_models,
            model_store=model_store,
            forecast_policy=policy,
            future_feature_policy=future_policy,
        )

    panel = _coerce_runner_panel(data)
    validate_panel(panel)
    if _all_panel_model_runs(model_runs):
        if policy == "recursive":
            raise ValueError(
                "recursive forecasting is only defined for feature-matrix models; "
                "panel-input models own their own multi-step prediction logic"
            )
        if features is not None:
            raise ValueError(
                "panel-input models consume the panel directly; pass features=None"
            )
        panel_target = _panel_runner_target(target, model_runs)
        # preprocessing_store is intentionally not forwarded here: panel-input
        # models bypass _prepare_origin_panel, so there is no per-origin
        # FittedPreprocessor to share via the disk store.
        return _run_panel_models(
            panel,
            target=panel_target,
            model_runs=model_runs,
            window_spec=_panel_window_for_horizon(window_spec, horizon_values[0]),
            preprocessing=preprocessing,
            preprocessing_policy=preprocessing_stage_policy,
            selection=selection,
            selection_policy=selection_stage_policy,
            combination_specs=combination_specs,
            config=config,
            save_models=save_models,
            model_store=model_store,
            forecast_policy=policy,
            future_feature_policy=future_policy,
        )
    _validate_feature_model_runs(model_runs)
    features = _feature_spec_for_policy(
        features,
        target=target,
        horizon=horizon_values[0],
        forecast_policy=policy,
        future_feature_policy=future_policy,
        target_transform=target_transform,
    )
    if policy == "recursive":
        _validate_recursive_feature_contract(
            features,
            future_feature_policy=future_policy,
        )

    full_stage: _PreparedStage | None = None
    fixed_feature_builder: Any | None = None
    if (
        preprocessing is not None
        and preprocessing_stage_policy is not None
        and preprocessing_stage_policy.scope == "full_panel"
    ):
        fitted = preprocessing.fit(
            _preprocessor_fit_input(panel, features), policy="origin_available"
        )
        full_panel = fitted.processed_train.panel
        full_panel_metadata = _detach_panel_metadata(full_panel)
        full_stage = _PreparedStage(
            panel=full_panel,
            fitted_preprocessing=fitted,
            metadata=fitted.to_metadata(),
            panel_metadata=full_panel_metadata,
        )
    elif preprocessing is None:
        # No preprocessing: the raw input panel's metadata is small and already
        # lives on ``panel.attrs``; mirror it onto ``panel_metadata`` so callers
        # have a single, consistent source for the per-origin metadata.
        full_stage = _PreparedStage(
            panel=panel,
            fitted_preprocessing=None,
            metadata=None,
            panel_metadata=dict(panel.attrs.get("macroforecast_metadata", {})),
        )

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
    selection_cache: dict[str, Any] = {}
    stage_records: list[dict[str, Any]] = []
    preprocessing_state = _StageUpdateState()
    feature_state = _StageUpdateState()
    fitted_preprocessing_cache: FittedPreprocessor | None = None
    fitted_feature_cache: Any | None = None

    execution_window = _feature_window_for_policy(window_spec, horizon_values[0])
    _validate_runner_window(execution_window, panel.index)

    # --- Incremental checkpoint setup (feature-matrix single-horizon path) ----
    # When checkpoint_path is set we persist each origin's LEAN records as soon as
    # the origin finishes, and on resume we skip origins already on disk. Skipping
    # an origin's COMPUTATION is only safe when no in-loop stage carries fitted
    # state forward across origins; otherwise a skipped early origin would leave a
    # later origin's cached fit unbuilt. We therefore detect that condition: when
    # both preprocessing and features are either fitted once before the loop
    # (full_panel / fixed_reference, which do not depend on the loop) or refit on
    # every origin (no carry-forward), computation-skipping is safe. Otherwise we
    # still COMPUTE every origin but skip the redundant WRITE, so correctness holds
    # on every policy while the common POOS configs get the resume speedup.
    completed_positions: set[int] = set()
    skip_computation = False
    if checkpoint_path is not None:
        completed_positions = completed_origin_positions(checkpoint_path)

        def _stage_is_independent(policy: StagePolicy | None, fitted_once: bool) -> bool:
            if fitted_once:
                return True  # full_panel / fixed_reference: built before the loop
            if policy is None:
                return True  # stage absent -> nothing carried forward
            return policy.update == "every_origin"

        preprocessing_independent = _stage_is_independent(
            preprocessing_stage_policy, full_stage is not None
        )
        feature_independent = _stage_is_independent(
            feature_stage_policy, fixed_feature_builder is not None
        )
        skip_computation = preprocessing_independent and feature_independent
    # -------------------------------------------------------------------------

    for origin_count, item in enumerate(execution_window.iter_origins(panel.index)):
        origin_pos = item["row"].get("origin_pos")
        already_done = (
            checkpoint_path is not None
            and origin_pos is not None
            and int(origin_pos) in completed_positions
        )
        if already_done and skip_computation:
            # This origin is on disk and the in-loop stages do not carry state
            # forward, so it is safe to skip computation entirely.
            continue

        preprocessing_updated = False
        if full_stage is None:
            preprocessing_updated = _stage_update_due(
                preprocessing_stage_policy,
                item,
                origin_count=origin_count,
                state=preprocessing_state,
            )
            prepared = _prepare_origin_panel(
                panel,
                features=features,
                preprocessing=preprocessing,
                preprocessing_policy=preprocessing_stage_policy,
                item=item,
                fitted_preprocessing=None
                if preprocessing_updated
                else fitted_preprocessing_cache,
                preprocessing_cache=preprocessing_cache,
                cache_key=_preprocessing_cache_key(item)
                if (preprocessing_cache is not None
                    or preprocessing_store is not None)
                else None,
                preprocessing_store=preprocessing_store,
                target=target,
            )
            if preprocessing_updated:
                fitted_preprocessing_cache = prepared.fitted_preprocessing
                _mark_stage_updated(preprocessing_state, item)
        else:
            prepared = full_stage
            preprocessing_updated = preprocessing is not None and origin_count == 0
        if prepared.metadata is not None:
            stage_records.append(
                _origin_stage_record(
                    "preprocessing",
                    item,
                    prepared.metadata,
                    updated=preprocessing_updated,
                )
            )

        # The preprocessing metadata is detached from ``prepared.panel.attrs`` (see
        # ``_detach_panel_metadata``) so the cached panel stays cheap to deep-copy
        # in this per-origin/per-arm loop. Pass it explicitly to feature fit and
        # transform so the feature stage still records full upstream provenance.
        prepared_metadata = prepared.panel_metadata

        fit_labels = panel.index[item["fit_idx"]]
        test_labels = panel.index[[int(item["test_idx"][0])]]
        selection_labels = stage_index(panel.index, item, selection_stage_policy)

        feature_updated = False
        if fixed_feature_builder is None:
            feature_updated = _stage_update_due(
                feature_stage_policy,
                item,
                origin_count=origin_count,
                state=feature_state,
            )
            if feature_updated or fitted_feature_cache is None:
                feature_fit_labels = stage_index(panel.index, item, feature_stage_policy)
                feature_fit_panel = prepared.panel.reindex(feature_fit_labels).dropna(
                    how="all"
                )
                fitted_feature_cache = features.fit(
                    feature_fit_panel, metadata=prepared_metadata
                )
                feature_updated = True
                _mark_stage_updated(feature_state, item)
            fitted_features = fitted_feature_cache
        else:
            fitted_features = fixed_feature_builder
            feature_updated = origin_count == 0

        feature_labels = _combined_feature_labels(
            fit_labels,
            selection_labels,
            test_labels,
        )
        all_features = _test_feature_builder(fitted_features).transform(
            prepared.panel,
            index=feature_labels,
            metadata=prepared_metadata,
        )
        train_features = _slice_feature_set(
            all_features,
            fit_labels,
            drop_missing=bool(getattr(fitted_features.spec, "drop_missing", True)),
        )
        test_features = _slice_feature_set(
            all_features,
            test_labels,
            drop_missing=False,
        )
        selection_features = _slice_feature_set(
            all_features,
            selection_labels,
            drop_missing=False,
        )
        if policy == "path_average":
            X_selection = selection_features.X
            y_selection = selection_features.y
            selection_splits = []
        else:
            X_selection, y_selection = _align_feature_xy(
                selection_features.X,
                _single_target(selection_features.y),
            )
            selection_splits = _relative_splits_for_index(
                item.get("val_splits", []),
                X_selection.index,
                panel.index,
                # Computed eagerly for every arm (even arms that never tune). A
                # single fold emptied by feature alignment at one origin/horizon
                # must not abort the whole consolidated multi-horizon run; drop the
                # unusable fold here and let select_params raise only for arms that
                # actually need tuning and are left with no usable folds.
                allow_degenerate=True,
            )
        stage_records.append(
            _origin_stage_record(
                "feature_engineering",
                item,
                fitted_features.to_metadata(),
                updated=feature_updated,
            )
        )

        origin_item = {
            **item,
            "base_index": panel.index,
            "forecast_horizon": horizon_values[0],
            "forecast_policy": policy,
            "future_feature_policy": future_policy,
            "window_spec": execution_window,
            "target_transform": features.target_transform,
            "target_name": _feature_target_name(features),
            "target_key": f"{policy}_h{horizon_values[0]}",
            "X_fit": train_features.X,
            "y_fit": train_features.y
            if policy == "path_average"
            else _single_target(train_features.y),
            "X_selection": X_selection,
            "y_selection": selection_features.y
            if policy == "path_average"
            else y_selection,
            "selection_splits": selection_splits,
            "absolute_val_splits": item.get("val_splits", []),
            "recursive_panel": prepared.panel,
            "actual_panel": prepared.panel,
            "recursive_builder": fitted_features,
            "X_test": test_features.X,
            "y_test": test_features.y
            if policy == "path_average"
            else _single_target(test_features.y),
            "preprocessed": preprocessing is not None,
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
            save_models=save_models,
            model_store=model_store,
        )
        if already_done:
            # Origin was recomputed only to keep stage caches consistent (state
            # carries forward); its records are already on disk, so do not add
            # them to the in-memory frame or rewrite them. The checkpoint-loaded
            # frame supplies this origin at merge time.
            continue
        records.extend(origin_records)
        if checkpoint_path is not None and origin_pos is not None:
            append_origin_records(checkpoint_path, origin_pos, origin_records)

    if checkpoint_path is not None:
        records = _merge_checkpoint_records(records, checkpoint_path)

    _assert_selection_was_possible(selection_cache, target_step=horizon_values[0])

    forecast_table = _forecast_table(records)
    combination_records = apply_combinations(forecast_table, combination_specs)
    if combination_records:
        records.extend(combination_records)
        forecast_table = _forecast_table(records)

    metadata = _result_metadata(
        input_path="panel_to_features",
        input_panel=panel,
        window_spec=execution_window,
        model_runs=model_runs,
        features=features.to_dict(),
        preprocessing=preprocessing.to_dict() if preprocessing is not None else None,
        preprocessing_policy=preprocessing_stage_policy,
        feature_policy=feature_stage_policy,
        selection=selection,
        selection_policy=selection_stage_policy,
        combination_specs=combination_specs,
        n_combination_forecasts=len(combination_records),
        stage_records=stage_records,
        n_forecasts=len(records),
        config=config,
        save_models=save_models,
        model_store=model_store,
        forecast_policy=policy,
        future_feature_policy=future_policy,
        horizons=horizon_values,
    )
    return ForecastResult(forecast_table, metadata=metadata)


def _run_multiple_horizons(
    data: Any,
    model: str | Callable[..., Any] | ModelSpec,
    *,
    window: WindowSpec,
    preprocessing: PreprocessSpec | None,
    preprocessing_policy: StagePolicy | str | None,
    features: FeatureSpec | None,
    feature_policy: StagePolicy | str | None,
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy | str | None,
    selection_metric: str | Callable[..., float],
    maximize_selection: bool,
    preset: str | Mapping[str, str | None] | None,
    params: Mapping[str, Any] | None,
    target: str | None,
    horizons: tuple[int, ...],
    forecast_policy: ForecastPolicy,
    future_feature_policy: FutureFeaturePolicy | None,
    target_transform: str | None,
    combination: str
    | CombinationSpec
    | Sequence[str | CombinationSpec | Mapping[str, Any]]
    | Mapping[str, Any]
    | None,
    save_models: bool,
    model_store: str | Path,
    preprocessing_cache: dict[Any, FittedPreprocessor | _PreparedStage] | None = None,
    preprocessing_store: PreprocessorStore | None = None,
    checkpoint_path: str | Path | None = None,
) -> ForecastResult:
    # The per-origin EM/factor fit is horizon-independent (it uses only the
    # origin_available rows), and the shared cache is keyed on origin_pos alone.
    # Forwarding the SAME cache to every horizon's single-horizon run() therefore
    # computes the EM once per origin and reuses it across all horizons (and arms),
    # which is the dominant cost in the GCLS pipeline.
    results = [
        run(
            data,
            model,
            window=window,
            preprocessing=preprocessing,
            preprocessing_policy=preprocessing_policy,
            features=features,
            feature_policy=feature_policy,
            model_selection=selection,
            model_selection_policy=selection_policy,
            model_selection_metric=selection_metric,
            maximize_model_selection=maximize_selection,
            preset=preset,
            params=params,
            target=target,
            horizon=horizon_value,
            horizons=None,
            forecast_policy=forecast_policy,
            future_feature_policy=future_feature_policy,
            target_transform=target_transform,
            combination=combination,
            save_models=save_models,
            model_store=model_store,
            preprocessing_cache=preprocessing_cache,
            # Forward the SAME disk store to every horizon so the per-origin
            # FittedPreprocessor (horizon-independent) is computed once and
            # reused across horizons -- and across worker processes -- exactly
            # like the in-memory cache above, just durable across run() calls.
            preprocessing_store=preprocessing_store,
            # Each horizon is a distinct forecast cell; the single-horizon run()
            # gives it its own ``h<h>`` checkpoint subdirectory (keyed on the
            # resolved horizon), so origins of different horizons never collide.
            # Pass the bare per-cell directory through and let run() namespace it
            # exactly once -- the same path a single-horizon run produces.
            checkpoint_path=checkpoint_path,
        )
        for horizon_value in horizons
    ]
    table = pd.concat([result.forecasts for result in results], ignore_index=True)
    first = results[0].metadata
    metadata = {
        **first,
        "run": {
            **first.get("run", {}),
            "n_forecasts": int(table.shape[0]),
            "horizons": list(horizons),
            "multi_horizon": True,
            "forecast_policy": forecast_policy,
        },
        "forecast_policy": {
            "method": forecast_policy,
            "horizons": list(horizons),
            "future_feature_policy": future_feature_policy,
            "target_transform": target_transform,
        },
        "per_horizon": {
            str(horizon_value): result.metadata
            for horizon_value, result in zip(horizons, results, strict=True)
        },
    }
    return ForecastResult(_forecast_table(table.to_dict(orient="records")), metadata=metadata)


def _resolve_runner_horizons(
    *,
    horizon: int,
    horizons: Sequence[int] | int | None,
) -> tuple[int, ...]:
    if horizons is not None and int(horizon) != 1:
        raise ValueError("provide either horizon or horizons, not both")
    raw_values: Sequence[int]
    if horizons is None:
        raw_values = (horizon,)
    elif isinstance(horizons, (int, np.integer)) and not isinstance(horizons, bool):
        raw_values = (int(horizons),)
    else:
        # horizons is a Sequence[int] here; the bool case is handled above.
        horizon_seq = cast("Sequence[int]", horizons)
        raw_values = tuple(int(value) for value in horizon_seq)
    values = tuple(int(value) for value in raw_values)
    if not values:
        raise ValueError("horizons must contain at least one horizon")
    if any(value < 1 for value in values):
        raise ValueError("forecast horizons must be positive")
    if len(set(values)) != len(values):
        raise ValueError("forecast horizons must be unique")
    return tuple(sorted(values))


def _normalize_forecast_policy(value: str) -> ForecastPolicy:
    aliases = {
        "direct": "direct",
        "single": "direct",
        "direct_average": "direct_average",
        "direct_avg": "direct_average",
        "average": "direct_average",
        "path_average": "path_average",
        "path_avg": "path_average",
        "path": "path_average",
        "recursive": "recursive",
        "iterated": "recursive",
    }
    if not isinstance(value, str):
        raise TypeError("forecast_policy must be a string")
    key = value.lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(
            "forecast_policy must be one of: direct, direct_average, "
            "path_average, recursive"
        )
    return cast(ForecastPolicy, aliases[key])


def _normalize_future_feature_policy(
    value: str | None,
    *,
    forecast_policy: ForecastPolicy,
) -> FutureFeaturePolicy | None:
    if forecast_policy != "recursive":
        if value is not None:
            raise ValueError("future_feature_policy is only used with recursive forecasting")
        return None
    if value is None:
        return "target_lags"
    aliases = {
        "target_lags": "target_lags",
        "target_lag": "target_lags",
        "target_only": "target_lags",
        "ar": "target_lags",
        "observed_future": "observed_future",
        "oracle": "observed_future",
        "actual_future": "observed_future",
    }
    if not isinstance(value, str):
        raise TypeError("future_feature_policy must be a string")
    key = value.lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(
            "future_feature_policy must be target_lags or observed_future"
        )
    return cast(FutureFeaturePolicy, aliases[key])


def _feature_spec_for_policy(
    features: FeatureSpec | None,
    *,
    target: str | None,
    horizon: int,
    forecast_policy: ForecastPolicy,
    future_feature_policy: FutureFeaturePolicy | None,
    target_transform: str | None,
) -> FeatureSpec:
    # _target_transform_for_policy returns plain str but only ever yields valid
    # TargetTransform literals; likewise target_mode is a TargetMode literal.
    transform = cast(
        TargetTransform,
        _target_transform_for_policy(
            forecast_policy,
            feature_transform=None if features is None else features.target_transform,
            explicit=target_transform,
        ),
    )
    target_mode: TargetMode = "path" if forecast_policy == "path_average" else "direct"
    if features is None:
        if target is None:
            raise ValueError("target is required when data is not a FeatureSet")
        if forecast_policy == "recursive":
            return feature_spec(
                target=target,
                horizon=1,
                predictors=[],
                lags=None,
                target_lags=(0, 1),
                target_mode="direct",
                target_transform=transform,
                metadata={"future_feature_policy": future_feature_policy},
            )
        return feature_spec(
            target=target,
            horizon=horizon,
            target_mode=target_mode,
            target_transform=transform,
        )
    if target is not None and features.target is not None and target != features.target:
        raise ValueError("target conflicts with the supplied FeatureSpec target")
    if len(features.targets) > 1:
        raise ValueError("forecasting.run currently supports one target per run")
    if features.horizons and len(features.horizons) > 1:
        raise ValueError(
            "FeatureSpec with multiple horizons should be passed through "
            "forecasting.run(..., horizons=...) so each horizon is fitted separately"
        )
    return replace(
        features,
        target=target or features.target,
        horizon=1 if forecast_policy == "recursive" else horizon,
        horizons=(),
        target_mode=target_mode,
        target_transform=transform,
    )


# Feature target-transforms that indicate the panel has ALREADY been
# differenced to stationarity. Only on such inputs does a change-based default
# target_transform double-difference; on raw/level panels it is correct.
_ALREADY_STATIONARY_TARGET_TRANSFORMS = frozenset(
    {
        "change",
        "growth",
        "log_growth",
        "log_change",
        "pct_change",
        "difference",
        "log_difference",
    }
)


def _warn_change_based_target_default(
    transform: str, feature_transform: str | None
) -> None:
    # Gate the warning on evidence that the panel is already stationary-
    # transformed. Firing on raw/level panels (where average_change/change is the
    # correct target) would be a false positive that trains users to ignore it.
    if feature_transform not in _ALREADY_STATIONARY_TARGET_TRANSFORMS:
        return
    warnings.warn(
        "forecast_policy yields a change-based target_transform "
        f"({transform!r}) by default while the features use an already-"
        f"stationary transform ({feature_transform!r}); this double-differences "
        "the target. Pass an explicit value-based target_transform "
        "('average_value' for direct_average, 'value' for path_average) to build "
        "averages from the one-period transformed series.",
        UserWarning,
        stacklevel=3,
    )


def _target_transform_for_policy(
    forecast_policy: ForecastPolicy,
    *,
    feature_transform: str | None,
    explicit: str | None,
) -> str:
    if forecast_policy == "direct_average":
        if explicit is not None:
            return explicit if explicit.startswith("average_") else f"average_{explicit}"
        if feature_transform and str(feature_transform).startswith("average_"):
            return feature_transform
        _warn_change_based_target_default("average_change", feature_transform)
        return "average_change"
    if forecast_policy == "path_average":
        if explicit is not None:
            return explicit
        if feature_transform and feature_transform != "level":
            return feature_transform
        _warn_change_based_target_default("change", feature_transform)
        return "change"
    if forecast_policy == "recursive":
        if explicit is not None:
            return explicit
        if feature_transform and str(feature_transform).startswith("average_"):
            raise ValueError("recursive forecasting does not support average_* target transforms")
        return feature_transform or "level"
    if explicit is not None:
        return explicit
    return feature_transform or "level"


def _validate_recursive_feature_contract(
    features: FeatureSpec,
    *,
    future_feature_policy: FutureFeaturePolicy | None,
) -> None:
    target = _feature_target_name(features)
    if target is None:
        raise ValueError("recursive forecasting requires exactly one target")
    transform = str(features.target_transform)
    if transform.startswith("average_"):
        raise ValueError("recursive forecasting does not support average_* target transforms")
    if future_feature_policy == "observed_future":
        return
    if features.predictors != ():
        raise ValueError(
            "recursive forecasting with future_feature_policy='target_lags' "
            "requires FeatureSpec predictors to be empty and target_lags to "
            "declare the autoregressive inputs. Use future_feature_policy="
            "'observed_future' for an explicit oracle/scenario path with "
            "exogenous future predictors."
        )
    if not features.target_lags:
        raise ValueError(
            "recursive forecasting with future_feature_policy='target_lags' "
            "requires FeatureSpec target_lags"
        )
    if 0 not in features.target_lags:
        raise ValueError(
            "recursive forecasting with future_feature_policy='target_lags' "
            "requires target_lags to include 0 so predicted target values can "
            "feed the next step under macroforecast's row-date convention"
        )
    if features.feature_steps:
        raise ValueError(
            "recursive target_lags currently supports FeatureSpec shortcut lags "
            "only; feature_steps need a future-step registry before they can be "
            "updated recursively"
        )
    if features.rolling_windows or features.pca_components is not None:
        raise ValueError(
            "recursive target_lags currently supports target lag features and "
            "optional deterministic time features only"
        )


def _horizon_val_window(val: "ValWindow", horizon: int) -> "ValWindow":
    """Re-derive the train/validation embargo for an h-step target.

    ``window.from_cutoffs`` defaults ``val_embargo`` to ``horizon - 1`` (the
    standard h-step purge that keeps training labels from realising inside the
    validation block). When a multi-horizon run injects the per-horizon test
    horizon into a base window built at horizon 1 (the consolidated-spec path used
    to share the per-origin EM across horizons), the validation embargo must be
    re-derived for the new horizon too -- otherwise the validation splits for h>1
    keep the h=1 purge (embargo 0) and leak the h-step training labels into the
    validation fold (and, downstream, fail feature alignment). We therefore set the
    val embargo to ``max(current, horizon - 1)`` so the per-horizon window matches
    a window that was built directly with ``from_cutoffs(horizon=h)``.
    """
    current = val.embargo if val.embargo is not None else 0
    return replace(val, embargo=max(int(current), max(0, int(horizon) - 1)))


def _feature_window_for_policy(window_spec: WindowSpec, horizon: int) -> WindowSpec:
    """Use h for origin cutoff while fitting one feature row per origin."""

    return replace(
        window_spec,
        test=replace(window_spec.test, horizon=int(horizon)),
        val=_horizon_val_window(window_spec.val, int(horizon)),
        horizon=int(horizon),
    )


def _panel_window_for_horizon(window_spec: WindowSpec, horizon: int) -> WindowSpec:
    return replace(
        window_spec,
        test=replace(window_spec.test, horizon=int(horizon)),
        val=_horizon_val_window(window_spec.val, int(horizon)),
        horizon=int(horizon),
    )


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
    combination_specs: Sequence[CombinationSpec],
    save_models: bool,
    model_store: str | Path,
    forecast_policy: ForecastPolicy,
    future_feature_policy: FutureFeaturePolicy | None,
) -> ForecastResult:
    X_all = data.X.copy()
    y_all = _single_target(data.y)
    validate_panel(X_all)
    _validate_runner_window(window_spec, X_all.index)
    records: list[dict[str, Any]] = []
    model_param_cache: dict[str, dict[str, Any]] = {}
    selection_cache: dict[str, Any] = {}
    for item in window_spec.iter_slices(X_all, y_all):
        selection_labels = stage_index(X_all.index, item, selection_policy)
        X_selection, y_selection = _align_feature_xy(
            X_all.reindex(selection_labels),
            y_all.reindex(selection_labels),
        )
        item = {
            **item,
            "base_index": X_all.index,
            "window_spec": window_spec,
            "X_selection": X_selection,
            "y_selection": y_selection,
            "forecast_horizon": data.horizons[0] if data.horizons else item["row"].get("horizon", 1),
            "forecast_policy": forecast_policy,
            "future_feature_policy": future_feature_policy,
            "target_name": data.target or (data.targets[0] if data.targets else None),
            "target_key": f"{forecast_policy}_h{data.horizons[0] if data.horizons else item['row'].get('horizon', 1)}",
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
                save_models=save_models,
                model_store=model_store,
            )
        )
    _assert_selection_was_possible(
        selection_cache,
        target_step=data.horizons[0] if data.horizons else 1,
    )
    forecast_table = _forecast_table(records)
    combination_records = apply_combinations(forecast_table, combination_specs)
    if combination_records:
        records.extend(combination_records)
        forecast_table = _forecast_table(records)

    metadata = _result_metadata(
        input_path="feature_set",
        input_panel=X_all,
        window_spec=window_spec,
        model_runs=model_runs,
        features=data.metadata.get("feature_spec")
        or data.metadata.get("feature_engineering"),
        preprocessing=None,
        preprocessing_policy=None,
        feature_policy=None,
        selection=selection,
        selection_policy=selection_policy,
        combination_specs=combination_specs,
        n_combination_forecasts=len(combination_records),
        stage_records=[],
        n_forecasts=len(records),
        config=config,
        save_models=save_models,
        model_store=model_store,
        forecast_policy=forecast_policy,
        future_feature_policy=future_feature_policy,
        horizons=data.horizons or (1,),
    )
    return ForecastResult(forecast_table, metadata=metadata)


def _run_panel_models(
    panel: pd.DataFrame,
    *,
    target: str,
    model_runs: list[_ModelRun],
    window_spec: WindowSpec,
    preprocessing: PreprocessSpec | None,
    preprocessing_policy: StagePolicy | None,
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    combination_specs: Sequence[CombinationSpec],
    config: Mapping[str, Any],
    save_models: bool,
    model_store: str | Path,
    forecast_policy: ForecastPolicy,
    future_feature_policy: FutureFeaturePolicy | None,
) -> ForecastResult:
    """Run models that fit on the canonical panel rather than engineered X/y."""

    _validate_panel_target(panel, target)
    _validate_runner_window(window_spec, panel.index)
    metadata = dict(panel.attrs.get("macroforecast_metadata", {}))
    records: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    full_stage: _PreparedStage | None = None
    if (
        preprocessing is not None
        and preprocessing_policy is not None
        and preprocessing_policy.scope == "full_panel"
    ):
        fitted = preprocessing.fit(
            _preprocessor_fit_input(panel, None),
            policy="origin_available",
        )
        full_stage = _PreparedStage(
            panel=fitted.processed_train.panel,
            fitted_preprocessing=fitted,
            metadata=fitted.to_metadata(),
        )
    elif preprocessing is None:
        full_stage = _PreparedStage(
            panel=panel,
            fitted_preprocessing=None,
            metadata=None,
        )

    preprocessing_state = _StageUpdateState()
    fitted_preprocessing_cache: FittedPreprocessor | None = None
    for origin_count, item in enumerate(window_spec.iter_origins(panel.index)):
        preprocessing_updated = False
        if full_stage is None:
            preprocessing_updated = _stage_update_due(
                preprocessing_policy,
                item,
                origin_count=origin_count,
                state=preprocessing_state,
            )
            prepared = _prepare_origin_panel(
                panel,
                features=None,
                preprocessing=preprocessing,
                preprocessing_policy=preprocessing_policy,
                item=item,
                include_target_pos=False,
                fitted_preprocessing=None
                if preprocessing_updated
                else fitted_preprocessing_cache,
            )
            if preprocessing_updated:
                fitted_preprocessing_cache = prepared.fitted_preprocessing
                _mark_stage_updated(preprocessing_state, item)
        else:
            prepared = full_stage
            preprocessing_updated = preprocessing is not None and origin_count == 0
        if prepared.metadata is not None:
            stage_records.append(
                _origin_stage_record(
                    "preprocessing",
                    item,
                    prepared.metadata,
                    updated=preprocessing_updated,
                )
            )
        # Metadata now lives on the dataclass (``panel_metadata``) rather than on
        # ``prepared.panel.attrs`` -- see ``_detach_panel_metadata`` -- so the
        # cached panel stays cheap to deep-copy in the per-origin loop. Re-attach
        # the full metadata onto the per-origin fit/test slices below, where the
        # model and forecast records actually consume it.
        prepared_metadata = dict(
            prepared.panel_metadata
            if prepared.panel_metadata is not None
            else prepared.panel.attrs.get("macroforecast_metadata", metadata)
        )
        fit_panel = prepared.panel.reindex(panel.index[item["fit_idx"]]).copy()
        test_panel = prepared.panel.reindex(panel.index[item["test_idx"]]).copy()
        fit_panel.attrs["macroforecast_metadata"] = prepared_metadata
        test_panel.attrs["macroforecast_metadata"] = prepared_metadata
        records.extend(
            _fit_predict_panel_origin(
                item,
                fit_panel=fit_panel,
                test_panel=test_panel,
                target=target,
                metadata=prepared_metadata,
                model_runs=model_runs,
                selection=selection,
                selection_policy=selection_policy,
                preprocessed=preprocessing is not None,
                save_models=save_models,
                model_store=model_store,
                forecast_policy=forecast_policy,
            )
        )

    forecast_table = _forecast_table(records)
    combination_records = apply_combinations(forecast_table, combination_specs)
    if combination_records:
        records.extend(combination_records)
        forecast_table = _forecast_table(records)

    result_metadata = _result_metadata(
        input_path="panel_model",
        input_panel=panel,
        window_spec=window_spec,
        model_runs=model_runs,
        features=None,
        preprocessing=preprocessing.to_dict() if preprocessing is not None else None,
        preprocessing_policy=preprocessing_policy,
        feature_policy=None,
        selection=selection,
        selection_policy=selection_policy,
        combination_specs=combination_specs,
        n_combination_forecasts=len(combination_records),
        stage_records=stage_records,
        n_forecasts=len(records),
        config=config,
        save_models=save_models,
        model_store=model_store,
        forecast_policy=forecast_policy,
        future_feature_policy=future_feature_policy,
        horizons=(int(window_spec.test.horizon),),
    )
    return ForecastResult(forecast_table, metadata=result_metadata)


def _fit_predict_panel_origin(
    item: dict[str, Any],
    *,
    fit_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    target: str,
    metadata: Mapping[str, Any],
    model_runs: list[_ModelRun],
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    preprocessed: bool,
    save_models: bool,
    model_store: str | Path,
    forecast_policy: ForecastPolicy,
) -> list[dict[str, Any]]:
    if fit_panel.empty or test_panel.empty:
        return []
    row = item["row"]
    records: list[dict[str, Any]] = []
    for model_run in model_runs:
        model_spec = model_run.spec
        _validate_panel_selection(selection, model_run)
        best_params = _panel_fit_params(model_spec, target=target)
        fit_params = _actual_model_params(model_spec, best_params)
        fit = model_spec(DataBundle(fit_panel, dict(metadata)), **best_params)
        stored_model = (
            _store_model_fit(
                fit,
                root=model_store,
                alias=model_run.alias,
                model_spec=model_spec,
                row=row,
                params=fit_params,
                selection_metadata=None,
            )
            if save_models
            else None
        )
        if model_spec.name == "dfm_unrestricted_midas" and hasattr(fit.estimator, "predict_from_panel"):
            pred_input = _panel_prediction_input_without_test_target(
                fit_panel,
                test_panel,
                target=target,
                metadata=metadata,
            )
            pred = _prediction_series(
                fit.estimator.predict_from_panel(
                    pred_input,
                    metadata=metadata,
                    anchor_dates=test_panel.index,
                ),
                index=test_panel.index,
            )
        else:
            pred = _prediction_series(fit.predict(test_panel), index=test_panel.index)
        y_test = test_panel[target] if target in test_panel.columns else pd.Series(dtype=float)
        for date, value in pred.items():
            step_horizon = _panel_prediction_horizon(
                date,
                origin=row.get("origin"),
                base_index=test_panel.index,
                default=int(row.get("horizon", 1)),
            )
            actual: Any = (
                y_test.reindex([date]).iloc[0] if date in y_test.index else None
            )
            records.append(
                {
                    "date": date,
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "horizon": step_horizon,
                    "forecast_policy": forecast_policy,
                    "target_transform": "level",
                    "target": target,
                    "model": model_run.alias,
                    "model_spec": model_spec.name,
                    "prediction": float(value),
                    "variance_prediction": None,
                    "quantile_predictions": None,
                    "actual": None if actual is None or pd.isna(actual) else float(actual),
                    "params": fit_params,
                    "model_selection": {
                        "policy": selection_policy.to_dict(),
                        "retuned": False,
                        "metadata": {
                            "note": "panel-input runner does not tune model parameters yet"
                        },
                    },
                    "stored_model": stored_model,
                    "window": row,
                    "preprocessed": bool(preprocessed),
                    "combined": False,
                    "combination": None,
                }
            )
    return records


def _fit_predict_origin(
    item: dict[str, Any],
    *,
    model_runs: list[_ModelRun],
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    selection_metric: str | Callable[..., float],
    maximize_selection: bool,
    param_cache: dict[str, dict[str, Any]],
    selection_cache: dict[str, Any],
    selection_random_state: int | None,
    save_models: bool,
    model_store: str | Path,
) -> list[dict[str, Any]]:
    if item.get("forecast_policy") == "path_average":
        return _fit_predict_path_average_origin(
            item,
            model_runs=model_runs,
            selection=selection,
            selection_policy=selection_policy,
            selection_metric=selection_metric,
            maximize_selection=maximize_selection,
            param_cache=param_cache,
            selection_cache=selection_cache,
            selection_random_state=selection_random_state,
            save_models=save_models,
            model_store=model_store,
        )
    if item.get("forecast_policy") == "recursive":
        return _fit_predict_recursive_origin(
            item,
            model_runs=model_runs,
            selection=selection,
            selection_policy=selection_policy,
            selection_metric=selection_metric,
            maximize_selection=maximize_selection,
            param_cache=param_cache,
            selection_cache=selection_cache,
            selection_random_state=selection_random_state,
            save_models=save_models,
            model_store=model_store,
        )

    X_fit = item["X_fit"]
    y_fit = item["y_fit"]
    X_selection = item.get("X_selection", X_fit)
    y_selection = item.get("y_selection", y_fit)
    X_test = item["X_test"]
    y_test = item["y_test"]
    if X_fit.empty or X_test.empty:
        return []

    records: list[dict[str, Any]] = []
    row = {
        **item["row"],
        "horizon": int(item.get("forecast_horizon", item["row"].get("horizon", 1))),
        "forecast_policy": item.get("forecast_policy", "direct"),
        "target_key": item.get("target_key"),
    }
    target_dates = _forecast_target_dates(
        X_test.index,
        base_index=item.get("base_index"),
        horizon=int(row["horizon"]),
    )
    target_step = int(row["horizon"])
    row.update(_target_availability_window_fields(item, target_step=target_step))
    X_fit, y_fit = _filter_xy_to_target_availability(
        X_fit,
        y_fit,
        item,
        target_step=target_step,
    )
    X_selection, y_selection = _filter_xy_to_target_availability(
        X_selection,
        y_selection,
        item,
        target_step=target_step,
    )
    selection_splits = _availability_safe_selection_splits(
        item,
        X_selection.index,
        target_step=target_step,
    )
    if X_fit.empty:
        return []
    retune = bool(row.get("retune", True))
    for model_run in model_runs:
        model_spec = model_run.spec
        selected, use_model_default_selection = _selection_for_model(
            selection, model_run
        )
        should_select = selected is not None or (
            use_model_default_selection and bool(model_spec.search_spaces)
        )
        selection_metadata: dict[str, Any] | None = None
        cache_key = _model_cache_key(model_run.alias, row.get("target_key"))
        uses_ic = str(getattr(model_spec, "selection_method", "cv")).lower() in (
            "bic", "aic", "aicc"
        )
        if should_select and uses_ic:
            # Information-criterion models (AR, FM) select their order by BIC/AIC on
            # the training sample, so they need no validation split. This is both
            # paper-faithful and what lets the autoregression run with a window that
            # carries no validation block.
            if retune or cache_key not in param_cache:
                result = select_by_information_criterion(
                    model_spec,
                    X_selection,
                    y_selection,
                    search=selected,
                    criterion=str(model_spec.selection_method).lower(),
                )
                param_cache[cache_key] = dict(result.best_params)
                selection_metadata = {
                    **result.to_metadata(),
                    "policy": selection_policy.to_dict(),
                    "retuned": True,
                }
                selection_cache[cache_key] = selection_metadata
                selection_cache[_SELECTION_TUNED_KEY] = True
                best_params = dict(param_cache.get(cache_key, {}))
            else:
                selection_metadata = selection_cache.get(cache_key)
                if selection_metadata is not None:
                    selection_metadata = {**selection_metadata, "retuned": False}
                best_params = dict(param_cache.get(cache_key, {}))
        elif should_select:
            if not selection_splits:
                best_params, selection_metadata = _resolve_degraded_selection(
                    cache_key=cache_key,
                    alias=model_run.alias,
                    target_step=target_step,
                    origin_label=row.get("origin"),
                    param_cache=param_cache,
                    selection_cache=selection_cache,
                    selection_policy=selection_policy,
                )
            elif retune or cache_key not in param_cache:
                result = select_params(
                    model_spec,
                    X_selection,
                    y_selection,
                    search=selected,
                    splits=selection_splits,
                    metric=selection_metric,
                    maximize=maximize_selection,
                    random_state=selection_random_state if selected is None else None,
                    allow_non_temporal_splits=_allow_non_temporal_selection_splits(item),
                )
                param_cache[cache_key] = dict(result.best_params)
                selection_metadata = {
                    **result.to_metadata(),
                    "policy": selection_policy.to_dict(),
                    "retuned": True,
                }
                selection_cache[cache_key] = selection_metadata
                selection_cache[_SELECTION_TUNED_KEY] = True
                best_params = dict(param_cache.get(cache_key, {}))
            else:
                selection_metadata = selection_cache.get(cache_key)
                if selection_metadata is not None:
                    selection_metadata = {**selection_metadata, "retuned": False}
                best_params = dict(param_cache.get(cache_key, {}))
        else:
            best_params = {}
        fit_params = _actual_model_params(model_spec, best_params)
        fit = model_spec(X_fit, y_fit, **best_params)
        stored_model = (
            _store_model_fit(
                fit,
                root=model_store,
                alias=model_run.alias,
                model_spec=model_spec,
                row=row,
                params=fit_params,
                selection_metadata=selection_metadata,
            )
            if save_models
            else None
        )
        pred = _prediction_series(fit.predict(X_test), index=X_test.index)
        variance_pred = _variance_series(fit, X_test=X_test, index=X_test.index)
        quantile_pred = _quantile_frame(fit, X_test=X_test, index=X_test.index)
        for date, value in pred.items():
            target_date = target_dates.loc[date]
            if pd.isna(target_date):
                continue
            actual: Any = (
                y_test.reindex([date]).iloc[0] if date in y_test.index else None
            )
            actual_value = None if actual is None or pd.isna(actual) else float(actual)
            variance_value = None
            if variance_pred is not None and date in variance_pred.index:
                variance_at_date = variance_pred.loc[date]
                variance_value = (
                    None if pd.isna(variance_at_date) else float(variance_at_date)
                )
            quantile_value = None
            if quantile_pred is not None and date in quantile_pred.index:
                quantile_row = quantile_pred.loc[date].dropna()
                quantile_value = {
                    str(level): float(quantile)
                    for level, quantile in quantile_row.items()
                }
            records.append(
                {
                    "date": target_date,
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "horizon": row.get("horizon"),
                    "forecast_policy": row.get("forecast_policy"),
                    "target_transform": item.get("target_transform"),
                    "target": item.get("target_name") or y_fit.name,
                    "model": model_run.alias,
                    "model_spec": model_spec.name,
                    "prediction": float(value),
                    "variance_prediction": variance_value,
                    "quantile_predictions": quantile_value,
                    "actual": actual_value,
                    "params": dict(fit_params),
                    "model_selection": selection_metadata,
                    "stored_model": stored_model,
                    "window": row,
                    "preprocessed": bool(item.get("preprocessed", False)),
                    "combined": False,
                    "combination": None,
                }
            )
    return records


def _fit_predict_recursive_origin(
    item: dict[str, Any],
    *,
    model_runs: list[_ModelRun],
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    selection_metric: str | Callable[..., float],
    maximize_selection: bool,
    param_cache: dict[str, dict[str, Any]],
    selection_cache: dict[str, Any],
    selection_random_state: int | None,
    save_models: bool,
    model_store: str | Path,
) -> list[dict[str, Any]]:
    X_fit = item["X_fit"]
    y_fit = item["y_fit"]
    X_selection = item.get("X_selection", X_fit)
    y_selection = item.get("y_selection", y_fit)
    if X_fit.empty:
        return []

    target = str(item.get("target_name") or getattr(y_fit, "name", "target"))
    transform = str(item.get("target_transform", "level"))
    horizon = int(item.get("forecast_horizon", item["row"].get("horizon", 1)))
    base_index = pd.Index(item["base_index"])
    origin_pos = int(item["row"]["origin_pos"])
    target_pos = origin_pos + horizon
    if target_pos >= len(base_index):
        return []
    origin_label = base_index[origin_pos]
    target_label = base_index[target_pos]
    recursive_panel = pd.DataFrame(item["recursive_panel"]).copy()
    recursive_builder = item["recursive_builder"]
    future_policy = cast(FutureFeaturePolicy, item.get("future_feature_policy") or "target_lags")
    row = {
        **item["row"],
        "horizon": horizon,
        "forecast_policy": "recursive",
        "target_key": item.get("target_key"),
        "future_feature_policy": future_policy,
    }
    origin_level = _target_level_at(recursive_panel, target, origin_label)
    actual_level = _target_level_at(pd.DataFrame(item.get("actual_panel", recursive_panel)), target, target_label)
    retune = bool(row.get("retune", True))
    records: list[dict[str, Any]] = []

    for model_run in model_runs:
        model_spec = model_run.spec
        selected, use_model_default_selection = _selection_for_model(selection, model_run)
        should_select = selected is not None or (
            use_model_default_selection and bool(model_spec.search_spaces)
        )
        selection_metadata: dict[str, Any] | None = None
        cache_key = _model_cache_key(model_run.alias, row.get("target_key"))
        recursive_splits = item.get("selection_splits")
        if should_select:
            if not recursive_splits:
                best_params, selection_metadata = _resolve_degraded_selection(
                    cache_key=cache_key,
                    alias=model_run.alias,
                    target_step=horizon,
                    origin_label=row.get("origin"),
                    param_cache=param_cache,
                    selection_cache=selection_cache,
                    selection_policy=selection_policy,
                )
            elif retune or cache_key not in param_cache:
                result = select_params(
                    model_spec,
                    X_selection,
                    y_selection,
                    search=selected,
                    splits=recursive_splits,
                    metric=selection_metric,
                    maximize=maximize_selection,
                    random_state=selection_random_state if selected is None else None,
                    allow_non_temporal_splits=_allow_non_temporal_selection_splits(item),
                )
                param_cache[cache_key] = dict(result.best_params)
                selection_metadata = {
                    **result.to_metadata(),
                    "policy": selection_policy.to_dict(),
                    "retuned": True,
                    "recursive": True,
                    "future_feature_policy": future_policy,
                }
                selection_cache[cache_key] = selection_metadata
                selection_cache[_SELECTION_TUNED_KEY] = True
                best_params = dict(param_cache.get(cache_key, {}))
            else:
                selection_metadata = selection_cache.get(cache_key)
                if selection_metadata is not None:
                    selection_metadata = {**selection_metadata, "retuned": False}
                best_params = dict(param_cache.get(cache_key, {}))
        else:
            best_params = {}
        fit_params = _actual_model_params(model_spec, best_params)
        fit = model_spec(X_fit, y_fit, **best_params)
        working_panel = recursive_panel.copy()
        step_predictions: list[float] = []
        step_levels: list[float] = []
        current_level = origin_level
        for step in range(1, horizon + 1):
            feature_label = base_index[origin_pos + step - 1]
            step_features = _test_feature_builder(recursive_builder).transform(
                working_panel,
                index=pd.Index([feature_label]),
            )
            X_step = step_features.X
            pred = _prediction_series(fit.predict(X_step), index=X_step.index)
            step_value = float(pred.iloc[0])
            current_level = _recursive_next_level(
                current_level,
                step_value,
                transform=transform,
            )
            step_predictions.append(step_value)
            # An ill-conditioned fit (e.g. near-collinear level predictors) can
            # produce a non-finite recursive level. Stop the recursion and report
            # the forecast as missing rather than poisoning the working panel
            # (panel validation rejects inf) or feeding NaN into the next predict.
            if not np.isfinite(current_level):
                current_level = float("nan")
                step_levels.append(current_level)
                break
            step_levels.append(current_level)
            update_pos = origin_pos + step
            if update_pos < len(base_index) and base_index[update_pos] in working_panel.index:
                working_panel.loc[base_index[update_pos], target] = current_level
        final_prediction = _recursive_output_value(
            origin_level,
            current_level,
            transform=transform,
        )
        actual_value = _recursive_output_value(
            origin_level,
            actual_level,
            transform=transform,
        )
        stored_model = (
            _store_model_fit(
                fit,
                root=model_store,
                alias=model_run.alias,
                model_spec=model_spec,
                row=row,
                params=fit_params,
                selection_metadata=selection_metadata,
            )
            if save_models
            else None
        )
        records.append(
            {
                "date": target_label,
                "origin": row.get("origin"),
                "origin_pos": row.get("origin_pos"),
                "horizon": horizon,
                "forecast_policy": "recursive",
                "target_transform": transform,
                "target": target,
                "model": model_run.alias,
                "model_spec": model_spec.name,
                "prediction": float(final_prediction),
                "variance_prediction": None,
                "quantile_predictions": None,
                "actual": float(actual_value),
                "params": {
                    **dict(fit_params),
                    "recursive": {
                        "future_feature_policy": future_policy,
                        "step_predictions": step_predictions,
                        "step_levels": step_levels,
                    },
                },
                "model_selection": selection_metadata,
                "stored_model": stored_model,
                "window": row,
                "preprocessed": bool(item.get("preprocessed", False)),
                "combined": False,
                "combination": None,
            }
        )
    return records


def _fit_predict_path_average_origin(
    item: dict[str, Any],
    *,
    model_runs: list[_ModelRun],
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    selection_metric: str | Callable[..., float],
    maximize_selection: bool,
    param_cache: dict[str, dict[str, Any]],
    selection_cache: dict[str, Any],
    selection_random_state: int | None,
    save_models: bool,
    model_store: str | Path,
) -> list[dict[str, Any]]:
    X_fit = pd.DataFrame(item["X_fit"])
    X_selection_base = pd.DataFrame(item.get("X_selection", X_fit))
    X_test = pd.DataFrame(item["X_test"])
    y_fit = pd.DataFrame(item["y_fit"])
    y_selection_base = pd.DataFrame(item.get("y_selection", y_fit))
    y_test = pd.DataFrame(item["y_test"])
    if X_fit.empty or X_test.empty:
        return []

    horizon = int(item.get("forecast_horizon", item["row"].get("horizon", 1)))
    if horizon < 1:
        raise ValueError("forecast horizon must be at least 1")
    step_columns = _path_step_columns(y_fit, horizon=horizon)
    row = {
        **item["row"],
        "horizon": horizon,
        "forecast_policy": "path_average",
        "target_key": item.get("target_key"),
        **_target_availability_window_fields(
            item,
            target_step=horizon,
            policy="path_step_specific",
        ),
    }
    row["target_availability_by_step"] = {
        str(step): _target_availability_window_fields(
            item,
            target_step=step,
            policy="row_pos_plus_step_lte_origin_pos",
        )
        for step in range(1, horizon + 1)
    }
    target_dates = _forecast_target_dates(
        X_test.index,
        base_index=item.get("base_index"),
        horizon=horizon,
    )
    retune = bool(row.get("retune", True))
    records: list[dict[str, Any]] = []

    for model_run in model_runs:
        model_spec = model_run.spec
        selected, use_model_default_selection = _selection_for_model(selection, model_run)
        should_select = selected is not None or (
            use_model_default_selection and bool(model_spec.search_spaces)
        )
        uses_ic = str(getattr(model_spec, "selection_method", "cv")).lower() in (
            "bic", "aic", "aicc"
        )
        predictions_by_step: dict[int, pd.Series] = {}
        stored_by_step: dict[str, Any] = {}
        selection_by_step: dict[str, Any] = {}
        params_by_step: dict[str, Any] = {}

        for step, column in enumerate(step_columns, start=1):
            y_fit_step = y_fit[column].rename(str(column))
            y_selection_step = y_selection_base[column].rename(str(column))
            X_fit_step, y_fit_aligned = _align_feature_xy(X_fit, y_fit_step)
            X_selection_step, y_selection_aligned = _align_feature_xy(
                X_selection_base,
                y_selection_step,
            )
            X_fit_step, y_fit_aligned = _filter_xy_to_target_availability(
                X_fit_step,
                y_fit_aligned,
                item,
                target_step=step,
            )
            X_selection_step, y_selection_aligned = _filter_xy_to_target_availability(
                X_selection_step,
                y_selection_aligned,
                item,
                target_step=step,
            )
            selection_splits = _availability_safe_selection_splits(
                item,
                X_selection_step.index,
                target_step=step,
            )
            if X_fit_step.empty:
                continue
            step_key = _model_cache_key(
                model_run.alias,
                f"{item.get('target_key', 'path')}_step{step}",
            )
            selection_metadata: dict[str, Any] | None = None
            if should_select and uses_ic:
                # Information-criterion models (AR, FM) select their order by
                # BIC/AIC on the full per-step training sample, exactly as the
                # direct path does. Routing them through the CV/validation-split
                # branch below would score the order on a truncated sample (the
                # validation block is held out) and pick a different order, which
                # breaks the horizon-1 direct==path invariant.
                if retune or step_key not in param_cache:
                    result = select_by_information_criterion(
                        model_spec,
                        X_selection_step,
                        y_selection_aligned,
                        search=selected,
                        criterion=str(model_spec.selection_method).lower(),
                    )
                    param_cache[step_key] = dict(result.best_params)
                    selection_metadata = {
                        **result.to_metadata(),
                        "policy": selection_policy.to_dict(),
                        "retuned": True,
                        "path_step": step,
                    }
                    selection_cache[step_key] = selection_metadata
                    selection_cache[_SELECTION_TUNED_KEY] = True
                    best_params = dict(param_cache.get(step_key, {}))
                else:
                    selection_metadata = selection_cache.get(step_key)
                    if selection_metadata is not None:
                        selection_metadata = {
                            **selection_metadata,
                            "retuned": False,
                            "path_step": step,
                        }
                    best_params = dict(param_cache.get(step_key, {}))
            elif should_select:
                if not selection_splits:
                    best_params, selection_metadata = _resolve_degraded_selection(
                        cache_key=step_key,
                        alias=f"{model_run.alias} (path step {step})",
                        target_step=step,
                        origin_label=row.get("origin"),
                        param_cache=param_cache,
                        selection_cache=selection_cache,
                        selection_policy=selection_policy,
                    )
                    if isinstance(selection_metadata, dict):
                        selection_metadata = {**selection_metadata, "path_step": step}
                elif retune or step_key not in param_cache:
                    result = select_params(
                        model_spec,
                        X_selection_step,
                        y_selection_aligned,
                        search=selected,
                        splits=selection_splits,
                        metric=selection_metric,
                        maximize=maximize_selection,
                        random_state=selection_random_state if selected is None else None,
                        allow_non_temporal_splits=_allow_non_temporal_selection_splits(item),
                    )
                    param_cache[step_key] = dict(result.best_params)
                    selection_metadata = {
                        **result.to_metadata(),
                        "policy": selection_policy.to_dict(),
                        "retuned": True,
                        "path_step": step,
                    }
                    selection_cache[step_key] = selection_metadata
                    selection_cache[_SELECTION_TUNED_KEY] = True
                    best_params = dict(param_cache.get(step_key, {}))
                else:
                    selection_metadata = selection_cache.get(step_key)
                    if selection_metadata is not None:
                        selection_metadata = {
                            **selection_metadata,
                            "retuned": False,
                            "path_step": step,
                        }
                    best_params = dict(param_cache.get(step_key, {}))
            else:
                best_params = {}
            fit_params = _actual_model_params(model_spec, best_params)
            fit = model_spec(X_fit_step, y_fit_aligned, **best_params)
            step_row = {**row, "target_key": f"{row.get('target_key')}_step{step}"}
            stored_by_step[str(step)] = (
                _store_model_fit(
                    fit,
                    root=model_store,
                    alias=model_run.alias,
                    model_spec=model_spec,
                    row=step_row,
                    params=fit_params,
                    selection_metadata=selection_metadata,
                )
                if save_models
                else None
            )
            predictions_by_step[step] = _prediction_series(
                fit.predict(X_test),
                index=X_test.index,
            )
            selection_by_step[str(step)] = selection_metadata
            params_by_step[str(step)] = dict(fit_params)

        if set(predictions_by_step) != set(range(1, horizon + 1)):
            continue

        prediction_frame = pd.concat(predictions_by_step, axis=1)
        actual_frame = y_test.reindex(columns=step_columns)
        for origin_label, path_values in prediction_frame.iterrows():
            target_date = target_dates.loc[origin_label]
            if pd.isna(target_date):
                continue
            actual_values = actual_frame.reindex([origin_label]).iloc[0]
            actual_value = (
                None
                if actual_values.isna().any()
                else float(actual_values.astype(float).mean())
            )
            records.append(
                {
                    "date": target_date,
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "horizon": horizon,
                    "forecast_policy": "path_average",
                    "target_transform": item.get("target_transform"),
                    "target": item.get("target_name"),
                    "model": model_run.alias,
                    "model_spec": model_spec.name,
                    "prediction": float(path_values.astype(float).mean()),
                    "variance_prediction": None,
                    "quantile_predictions": None,
                    "actual": actual_value,
                    "params": {"steps": params_by_step},
                    "model_selection": {
                        "policy": selection_policy.to_dict(),
                        "retuned": any(
                            bool(value and value.get("retuned"))
                            for value in selection_by_step.values()
                        ),
                        "steps": selection_by_step,
                    }
                    if should_select
                    else None,
                    "stored_model": {"steps": stored_by_step} if save_models else None,
                    "window": row,
                    "preprocessed": bool(item.get("preprocessed", False)),
                    "combined": False,
                    "combination": None,
                }
            )
    return records


def _preprocessing_cache_key(item: Mapping[str, Any]) -> Any:
    """Horizon-independent key for the shared per-origin preprocessing cache.

    The spec-level EM/factor fit at one origin uses only the ``origin_available``
    (estimation) rows, which are fully determined by the origin POSITION and do not
    depend on the forecast horizon (test block / target row are excluded from the
    fit). Keying the cache on ``origin_pos`` alone therefore lets arms AND horizons
    of the same target reuse the identical FittedPreprocessor without recomputing
    the EM SVD, while staying leak-free (the fit never sees future rows).
    """
    row = item.get("row")
    if isinstance(row, Mapping) and "origin_pos" in row:
        return ("origin_pos", int(row["origin_pos"]))
    # Fallback: estimation block identity (still horizon-independent).
    est = item.get("estimation_idx")
    if est is not None:
        arr = np.asarray(est, dtype=int)
        return ("estimation_span", int(arr[0]), int(arr[-1])) if arr.size else ("empty",)
    return ("origin_pos", item.get("origin_pos"))


def _origin_pos_for_store_key(cache_key: Any, item: Mapping[str, Any]) -> int | None:
    """Resolve the integer ``origin_pos`` used in the on-disk store key.

    The on-disk store keys on the EXACT ``(spec, target, origin_pos)`` triple, so
    it needs a stable integer origin position. We take it from the same source the
    in-memory key uses -- the origin row's ``origin_pos`` (preferred) -- so the two
    tiers agree on what "the same origin" means. If no integer origin position is
    available (only the ``estimation_span``/``empty`` fallback cache keys), we
    return ``None`` and the caller skips the disk tier entirely, falling back to a
    plain (correct) recompute rather than risk an ambiguous key.
    """

    row = item.get("row")
    if isinstance(row, Mapping):
        pos = row.get("origin_pos")
        if pos is not None:
            return int(pos)
    # Mirror the in-memory key's primary form: ("origin_pos", N).
    if (
        isinstance(cache_key, tuple)
        and len(cache_key) == 2
        and cache_key[0] == "origin_pos"
        and cache_key[1] is not None
    ):
        return int(cache_key[1])
    return None


def _prepare_origin_panel(
    panel: pd.DataFrame,
    *,
    features: FeatureSpec | None,
    preprocessing: PreprocessSpec | None,
    preprocessing_policy: StagePolicy | None,
    item: dict[str, Any],
    include_target_pos: bool = True,
    fitted_preprocessing: FittedPreprocessor | None = None,
    preprocessing_cache: dict[Any, FittedPreprocessor | _PreparedStage] | None = None,
    cache_key: Any | None = None,
    preprocessing_store: PreprocessorStore | None = None,
    target: str | None = None,
) -> _PreparedStage:
    if preprocessing is None or preprocessing_policy is None:
        return _PreparedStage(
            panel=panel,
            fitted_preprocessing=None,
            metadata=None,
            panel_metadata=dict(panel.attrs.get("macroforecast_metadata", {})),
        )
    # Cross-arm reuse of the TRANSFORMED panel. The leak-free origin_available
    # transform re-runs the (expensive) EM imputation on the apply window, and the
    # output panel depends only on (panel, the origin's apply/available labels,
    # preprocessing spec) -- NOT on the arm's features/model. Arms of the same
    # target therefore produce an identical prepared panel at each origin, so the
    # first arm computes it and the rest reuse the cached _PreparedStage. This
    # removes the dominant per-arm EM-transform redundancy (the fit cache alone
    # only saved the cheaper .fit(), not the per-origin transform).
    prepared_key = (
        ("prepared",) + tuple(cache_key) + (int(bool(include_target_pos)),
            int(_origin_target_pos(panel.index, item)))
        if (preprocessing_cache is not None and cache_key is not None)
        else None
    )
    if (
        prepared_key is not None
        and preprocessing_cache is not None
        and prepared_key in preprocessing_cache
    ):
        cached = preprocessing_cache[prepared_key]
        if isinstance(cached, _PreparedStage):
            return cached
    # Cross-arm/cross-horizon reuse of the per-origin FittedPreprocessor (the EM
    # FIT). The fit depends on target+horizons metadata only, not the arm's
    # predictors/model, and is horizon-independent (origin_available rows), so the
    # first caller fits and the rest reuse via the origin-keyed cache.
    if (
        fitted_preprocessing is None
        and preprocessing_cache is not None
        and cache_key is not None
        and cache_key in preprocessing_cache
    ):
        candidate = preprocessing_cache[cache_key]
        if isinstance(candidate, FittedPreprocessor):
            fitted_preprocessing = candidate
    # Second cache tier: the on-disk content-addressed store. Consulted ONLY on an
    # in-memory miss, and ONLY when a store was passed (store is None -> this whole
    # block is skipped and behaviour is byte-identical to the no-store path). The
    # store key is the EXACT (PreprocessSpec, target, origin_pos) triple that
    # identifies this fit, so a loaded preprocessor is reused only for an identical
    # spec+target+origin -- never served for a different one. A store hit is also
    # written back into the in-memory dict so later same-process lookups stay fast.
    # NOTE: the store key encodes (PreprocessSpec, target, origin_pos) but NOT
    # ``preprocessing_policy.scope`` (origin_available vs fit_window). Within a
    # single run() the scope is constant, so reuse is safe; do NOT share one store
    # directory across runs that use a different scope for the same spec (a
    # fit_window fit would be served where an origin_available fit is expected).
    store_key: str | None = None
    store_origin_pos: int | None = None
    if (
        fitted_preprocessing is None
        and preprocessing_store is not None
        and cache_key is not None
        and target is not None
    ):
        store_origin_pos = _origin_pos_for_store_key(cache_key, item)
        if store_origin_pos is not None:
            store_key = preprocessing_store.key(
                preprocessing,
                target=str(target),
                origin_pos=store_origin_pos,
            )
            loaded = preprocessing_store.get(store_key)
            if isinstance(loaded, FittedPreprocessor):
                fitted_preprocessing = loaded
                if preprocessing_cache is not None:
                    preprocessing_cache[cache_key] = loaded
    if fitted_preprocessing is None:
        fit_panel = stage_panel(panel, item, preprocessing_policy)
        fit_policy = (
            "fit_window"
            if preprocessing_policy.scope in {"fit_window", "fixed_reference"}
            else "origin_available"
        )
        fitted = preprocessing.fit(
            _preprocessor_fit_input(fit_panel, features),
            policy=fit_policy,
        )
        if preprocessing_cache is not None and cache_key is not None:
            preprocessing_cache[cache_key] = fitted
        # Persist the freshly computed fit to the on-disk store so other
        # processes / later run() calls reuse it instead of recomputing.
        if store_key is not None:
            preprocessing_store.put(store_key, fitted)
    else:
        fitted = fitted_preprocessing
    apply_labels = _origin_apply_labels(
        panel.index,
        item,
        include_target_pos=include_target_pos,
    )
    apply_panel = panel.reindex(apply_labels).loc[:, fitted.fit_panel.columns]
    if fitted.preprocessing_scope == "fit_window":
        transformed = fitted.transform(
            apply_panel, history=fitted.fit_panel, policy="fit_window"
        )
        prepared_panel = transformed.panel
    else:
        # Rows observable AT the forecast origin: every apply row whose position
        # is <= the origin position. This must exclude the ENTIRE forward test
        # block (origin+1 .. origin+horizon-1) and the appended target row, not
        # just the target row -- test_idx spans the whole horizon block, so a
        # naive estimation/fit/test union would still feed h-1 strictly-future
        # rows into the data-dependent preprocessing (EM imputation / outlier
        # statistics) fit and leak the future into training-row features.
        available_labels = _origin_available_labels(panel.index, item)
        transformed = fitted.transform(
            apply_panel,
            history=fitted.fit_panel,
            policy="origin_available",
            available=available_labels,
        )
        prepared_panel = transformed.panel
    # Move the heavy ``macroforecast_metadata`` payload off ``panel.attrs`` and
    # onto the dataclass. The transformed panel below is reindexed/dropna'd/fed to
    # feature fit+transform once per origin AND reused across arms/horizons from
    # the shared cache; pandas deep-copies ``.attrs`` on each of those operations
    # via ``__finalize__``, so leaving the ~40 KB preprocessing/standardization
    # metadata on ``.attrs`` makes that deepcopy dominate multi-horizon runtime.
    # Small keys that downstream code reads directly off the panel (e.g. the
    # transform-code map) are preserved on ``.attrs``; the full metadata is kept
    # on ``panel_metadata`` and re-attached by callers only where it is consumed.
    panel_metadata = _detach_panel_metadata(prepared_panel)
    stage = _PreparedStage(
        panel=prepared_panel,
        fitted_preprocessing=fitted,
        metadata=fitted.to_metadata(),
        panel_metadata=panel_metadata,
    )
    if prepared_key is not None and preprocessing_cache is not None:
        preprocessing_cache[prepared_key] = stage
    return stage


# Small ``.attrs`` keys that are cheap to deep-copy and are read directly off the
# panel by downstream stages (transform-code resolution, etc.). These stay on the
# panel; the large ``macroforecast_metadata`` payload is detached.
_LIGHT_PANEL_ATTR_KEYS = ("macroforecast_transform_codes",)


def _detach_panel_metadata(panel: pd.DataFrame) -> dict[str, Any] | None:
    """Strip the heavy ``macroforecast_metadata`` from ``panel.attrs`` in place.

    Returns the detached metadata dict (or ``None`` if absent). After this call
    ``panel.attrs`` retains only the small, deep-copy-cheap keys in
    ``_LIGHT_PANEL_ATTR_KEYS``, so the per-origin/per-arm pandas operations on the
    cached panel no longer pay the cost of deep-copying the full preprocessing
    metadata on every ``__finalize__``.
    """

    attrs = panel.attrs
    detached = attrs.get("macroforecast_metadata")
    light = {key: attrs[key] for key in _LIGHT_PANEL_ATTR_KEYS if key in attrs}
    panel.attrs = light
    return dict(detached) if isinstance(detached, Mapping) else detached


def _origin_target_pos(index: pd.Index, item: Mapping[str, Any]) -> int:
    """Position of the appended post-origin target row (origin_pos + horizon).

    Used only as part of the prepared-stage cache key so that two arms sharing the
    SAME (origin, horizon) reuse the identical transformed panel, while different
    horizons (different target rows) get distinct cache entries.
    """
    row = item.get("row", {})
    try:
        origin_pos = int(row.get("test_start_pos", item["test_idx"][0]))
        horizon = int(row.get("horizon", 1))
    except (TypeError, ValueError, IndexError, KeyError):
        return -1
    target_pos = origin_pos + horizon
    return target_pos if 0 <= target_pos < len(index) else -1


def _origin_apply_labels(
    index: pd.Index,
    item: dict[str, Any],
    *,
    include_target_pos: bool = True,
) -> pd.Index:
    positions = np.unique(
        np.concatenate([item["estimation_idx"], item["fit_idx"], item["test_idx"]])
    )
    if not include_target_pos:
        return index[positions]
    row = item.get("row", {})
    try:
        target_pos = int(row.get("test_start_pos", item["test_idx"][0])) + int(
            row.get("horizon", 1)
        )
    except (TypeError, ValueError, IndexError):
        target_pos = -1
    if 0 <= target_pos < len(index):
        positions = np.unique(np.concatenate([positions, np.asarray([target_pos])]))
    return index[positions]


def _origin_available_labels(index: pd.Index, item: dict[str, Any]) -> pd.Index:
    """Labels observable at the forecast origin: apply positions <= origin_pos.

    ``test_idx`` spans the whole forward horizon block for direct/path policies,
    so it contains rows strictly after the origin. Only rows at or before the
    origin position may inform the origin_available data-dependent preprocessing
    fit (imputation / outlier statistics); everything later is unrealised future.
    """
    positions = np.unique(
        np.concatenate([item["estimation_idx"], item["fit_idx"], item["test_idx"]])
    )
    row = item.get("row", {})
    try:
        origin_pos = int(row.get("test_start_pos", item["test_idx"][0]))
    except (TypeError, ValueError, IndexError):
        origin_pos = int(positions.max()) if len(positions) else -1
    available = positions[positions <= origin_pos]
    return index[available]


def _single_target(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    frame = pd.DataFrame(y)
    if frame.shape[1] != 1:
        raise ValueError(
            "forecasting runner currently expects exactly one target column"
        )
    return frame.iloc[:, 0].rename(str(frame.columns[0]))


def _filter_xy_to_target_availability(
    X: Any,
    y: Any,
    item: dict[str, Any],
    *,
    target_step: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X_aligned, y_aligned = _align_feature_xy(X, y)
    mask = _target_availability_mask(
        X_aligned.index,
        item,
        target_step=target_step,
    )
    return X_aligned.loc[mask], y_aligned.loc[mask]


def _target_availability_mask(
    labels: pd.Index,
    item: dict[str, Any],
    *,
    target_step: int,
) -> np.ndarray:
    base_index = _target_availability_base_index(item, labels)
    positions = base_index.get_indexer(pd.Index(labels))
    if (positions < 0).any():
        missing = pd.Index(labels)[positions < 0]
        raise ValueError(
            "forecast target-availability filtering requires feature labels "
            f"to be contained in base_index; missing labels: {list(missing[:3])}"
        )
    origin_pos = int(item["row"]["origin_pos"])
    cutoff_pos = origin_pos - int(target_step)
    return positions <= cutoff_pos


def _target_availability_base_index(
    item: dict[str, Any],
    fallback: pd.Index,
) -> pd.Index:
    raw = item.get("base_index")
    if raw is None:
        return pd.Index(fallback)
    return pd.Index(raw)


def _target_availability_window_fields(
    item: dict[str, Any],
    *,
    target_step: int,
    policy: str = "row_pos_plus_target_step_lte_origin_pos",
) -> dict[str, Any]:
    origin_pos = int(item["row"]["origin_pos"])
    cutoff_pos = origin_pos - int(target_step)
    base_index = _target_availability_base_index(item, pd.Index([]))
    cutoff_label: Any | None = None
    if 0 <= cutoff_pos < len(base_index):
        cutoff_label = base_index[cutoff_pos]
    return {
        "target_availability_policy": policy,
        "target_availability_lag": int(target_step),
        "target_availability_end": cutoff_label,
        "target_availability_end_pos": int(cutoff_pos),
    }


_SELECTION_TUNED_KEY = "__macroforecast_selection_ever_tuned__"
_SELECTION_DEGRADED_KEY = "__macroforecast_selection_ever_degraded__"


def _resolve_degraded_selection(
    *,
    cache_key: str,
    alias: str,
    target_step: int,
    origin_label: Any,
    param_cache: dict[str, dict[str, Any]],
    selection_cache: dict[str, Any],
    selection_policy: StagePolicy,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Degrade gracefully when no target-availability-safe split exists.

    A long pseudo-out-of-sample run must not abort entirely because ONE early
    origin lacks enough target-available data to form a tuning split. We reuse
    the model's last successfully-tuned parameters when available, otherwise
    fall back to the model's registered defaults (empty override), and emit a
    RuntimeWarning naming the origin/horizon/model. A run-level flag records the
    degradation so ``run`` can still raise if NO origin ever tunes (a genuinely
    impossible configuration rather than a sparse early origin).
    """

    selection_cache[_SELECTION_DEGRADED_KEY] = True
    reused = cache_key in param_cache
    best_params = dict(param_cache.get(cache_key, {}))
    source = "last-tuned params" if reused else "model defaults"
    warnings.warn(
        f"model selection found no target-availability-safe validation split for "
        f"model {alias!r} at origin {origin_label} (horizon {target_step}); "
        f"falling back to {source} for this origin instead of tuning",
        RuntimeWarning,
        stacklevel=3,
    )
    cached_metadata = selection_cache.get(cache_key)
    base_metadata = (
        dict(cached_metadata) if isinstance(cached_metadata, dict) else {}
    )
    selection_metadata = {
        **base_metadata,
        "policy": selection_policy.to_dict(),
        "retuned": False,
        "selection_degraded": True,
        "selection_degraded_reason": "no_availability_safe_split",
        "selection_degraded_source": source,
    }
    return best_params, selection_metadata


def _assert_selection_was_possible(
    selection_cache: dict[str, Any],
    *,
    target_step: int,
) -> None:
    """Raise if model selection degraded at every origin (impossible config).

    Graceful degradation tolerates sparse early origins, but a configuration
    that can NEVER form a tuning split for ANY origin is a genuine
    misconfiguration that should still surface as an error.
    """

    ever_degraded = bool(selection_cache.get(_SELECTION_DEGRADED_KEY))
    ever_tuned = bool(selection_cache.get(_SELECTION_TUNED_KEY))
    if ever_degraded and not ever_tuned:
        raise ValueError(
            "model selection has no target-availability-safe validation splits "
            f"for horizon {target_step} at ANY origin; increase the available "
            "sample, move the validation window earlier, reduce the validation "
            "size, or reduce the forecast horizon"
        )


def _availability_safe_selection_splits(
    item: dict[str, Any],
    selection_index: pd.Index,
    *,
    target_step: int,
) -> list[Split]:
    if len(selection_index) == 0:
        return []
    window_spec = item.get("window_spec")
    if isinstance(window_spec, WindowSpec):
        val = window_spec.val
        base_embargo = (
            val.embargo
            if val.embargo is not None
            else window_spec.estimation.embargo
        )
        embargo = max(int(base_embargo), max(int(target_step) - 1, 0))
        min_train_size = (
            window_spec.min_train_size
            if window_spec.min_train_size is not None
            else val.min_train_size
        )
        try:
            return make_splitter(
                val.method,
                len(selection_index),
                validation_size=val.size,
                validation_ratio=val.ratio,
                min_train_size=min_train_size,
                n_splits=val.n_splits,
                step=val.step,
                horizon=val.horizon,
                random_state=val.random_state,
                embargo=embargo,
            )
        except ValueError:
            return _availability_safe_explicit_splits(
                item,
                selection_index,
                target_step=target_step,
            )
    return _availability_safe_explicit_splits(
        item,
        selection_index,
        target_step=target_step,
    )


def _allow_non_temporal_selection_splits(item: dict[str, Any]) -> bool:
    window_spec = item.get("window_spec")
    return isinstance(window_spec, WindowSpec) and window_spec.val.method == "random_kfold"


def _availability_safe_explicit_splits(
    item: dict[str, Any],
    selection_index: pd.Index,
    *,
    target_step: int,
) -> list[Split]:
    absolute_splits = item.get("absolute_val_splits") or []
    if not absolute_splits:
        return []
    base_index = _target_availability_base_index(item, selection_index)
    out: list[Split] = []
    for train_abs, val_abs in absolute_splits:
        train_abs_arr = np.asarray(train_abs, dtype=int)
        val_abs_arr = np.asarray(val_abs, dtype=int)
        val_abs_arr = val_abs_arr[
            (val_abs_arr >= 0)
            & (val_abs_arr < len(base_index))
        ]
        if len(val_abs_arr) == 0:
            continue
        val_labels = base_index[val_abs_arr]
        val_rel = selection_index.get_indexer(val_labels)
        val_keep = val_rel >= 0
        if not val_keep.any():
            continue
        kept_val_abs = val_abs_arr[val_keep]
        val_rel = val_rel[val_keep]
        train_cutoff = int(kept_val_abs.min()) - int(target_step)
        train_abs_arr = train_abs_arr[
            (train_abs_arr >= 0)
            & (train_abs_arr < len(base_index))
            & (train_abs_arr <= train_cutoff)
        ]
        if len(train_abs_arr) == 0:
            continue
        train_labels = base_index[train_abs_arr]
        train_rel = selection_index.get_indexer(train_labels)
        train_rel = train_rel[train_rel >= 0]
        if len(train_rel) == 0:
            continue
        out.append((train_rel.astype(int, copy=False), val_rel.astype(int, copy=False)))
    return out


def _test_feature_builder(builder: Any) -> Any:
    if not getattr(builder.spec, "drop_missing", True):
        return builder
    return replace(builder, spec=replace(builder.spec, drop_missing=False))


def _combined_feature_labels(*indexes: Iterable[Any]) -> pd.Index:
    labels = pd.Index([])
    for index in indexes:
        labels = labels.append(pd.Index(index))
    return labels.unique()


def _slice_feature_set(
    features: FeatureSet,
    index: Iterable[Any],
    *,
    drop_missing: bool,
) -> FeatureSet:
    labels = pd.Index(index)
    X = features.X.reindex(labels)
    y = features.y.reindex(labels)
    if drop_missing:
        aligned = pd.concat([X, y], axis=1).dropna()
        X = aligned.loc[:, X.columns]
        y = aligned.loc[:, y.columns]
    X = X.copy()
    y = y.copy()
    X.attrs.update(features.X.attrs)
    y.attrs.update(features.y.attrs)
    return FeatureSet(
        X=X,
        y=y,
        metadata=dict(features.metadata),
        feature_metadata=features.feature_metadata.copy(),
        target_metadata=features.target_metadata.copy(),
        target=features.target,
        targets=features.targets,
        horizons=features.horizons,
        predictors=features.predictors,
    )


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
    *,
    allow_degenerate: bool = False,
) -> list[Split]:
    """Map absolute window positions onto a stage-local feature matrix.

    ``allow_degenerate`` controls what happens when feature alignment empties a
    fold's train or validation side. After data-dependent preprocessing (EM
    imputation that flags outliers as NaN) and lag-feature construction, some rows
    referenced by an absolute validation split may be dropped from the selection
    feature matrix, occasionally emptying one fold at a particular origin/horizon.
    With ``allow_degenerate=False`` (the default, used by the feature-set path) an
    empty fold is a hard error. With ``allow_degenerate=True`` (used by the
    per-origin selection-splits computation, which runs for every arm even those
    that never tune) the unusable fold is SKIPPED rather than aborting the whole
    multi-horizon run; ``select_params`` still raises downstream if an arm that
    needs tuning is left with no usable folds. Skipping an unusable CV fold is a
    standard, leak-free behaviour (it never feeds future rows into training).
    """

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
            if allow_degenerate:
                continue
            raise ValueError(
                f"validation split {split_id} has no train rows after feature alignment"
            )
        val_pos = val_pos[val_pos >= 0]
        if len(val_pos) == 0:
            if allow_degenerate:
                continue
            raise ValueError(
                f"validation split {split_id} has no validation rows after feature alignment"
            )
        out.append((train_pos.astype(int, copy=False), val_pos.astype(int, copy=False)))
    return out


def _preprocessor_fit_input(fit_panel: pd.DataFrame, features: FeatureSpec | None) -> Any:
    if features is None:
        metadata = dict(fit_panel.attrs.get("macroforecast_metadata", {}))
        return DataBundle(fit_panel, metadata)
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
    # Preprocessing may create columns that the downstream FeatureSpec uses.
    # Validate target/horizon metadata here, but defer predictor validation until
    # after preprocessing has produced the panel consumed by feature engineering.
    return data_spec(
        DataBundle(fit_panel, metadata),
        target=target,
        targets=targets,
        horizons=horizons,
        predictors="all",
    )


def _coerce_runner_panel(data: Any) -> pd.DataFrame:
    if isinstance(data, DataSpec):
        return as_panel(data.panel, metadata=data.metadata)
    if isinstance(data, DataBundle):
        return as_panel(data.panel, metadata=data.metadata)
    if (
        isinstance(data, tuple)
        and len(data) == 2
        and isinstance(data[0], pd.DataFrame)
        and isinstance(data[1], Mapping)
    ):
        return as_panel(data[0], metadata=data[1])
    if isinstance(data, pd.DataFrame):
        metadata = dict(data.attrs.get("macroforecast_metadata", {}))
        return as_panel(data, metadata=metadata)
    return as_panel(pd.DataFrame(data).copy())


def _all_panel_model_runs(model_runs: Sequence[_ModelRun]) -> bool:
    return all(model_run.spec.input_kind == "panel" for model_run in model_runs)


def _validate_feature_model_runs(model_runs: Sequence[_ModelRun]) -> None:
    panel_models = [
        model_run.alias
        for model_run in model_runs
        if model_run.spec.input_kind == "panel"
    ]
    if panel_models:
        raise ValueError(
            "panel-input models cannot be mixed with FeatureSet or feature-matrix "
            f"runner inputs: {panel_models}. Pass a panel/DataBundle with features=None."
        )


def _panel_runner_target(target: str | None, model_runs: Sequence[_ModelRun]) -> str:
    if target is not None:
        resolved = str(target)
        conflicts = [
            model_run.alias
            for model_run in model_runs
            if model_run.spec.params.get("target") is not None
            and str(model_run.spec.params["target"]) != resolved
        ]
        if conflicts:
            raise ValueError(
                "runner target conflicts with model-specific target parameters "
                f"for: {conflicts}"
            )
        return resolved
    model_targets = {
        str(model_run.spec.params["target"])
        for model_run in model_runs
        if model_run.spec.params.get("target") is not None
    }
    if len(model_targets) == 1:
        return next(iter(model_targets))
    raise ValueError(
        "target is required for panel-input forecasting unless every model spec "
        "sets the same target parameter"
    )


def _panel_fit_params(model_spec: ModelSpec, *, target: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if "target" in model_spec.default_params and model_spec.params.get("target") is None:
        params["target"] = target
    return params


def _panel_prediction_input_without_test_target(
    fit_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    *,
    target: str,
    metadata: Mapping[str, Any],
) -> DataBundle:
    """Return panel available at prediction time with test target values masked."""

    combined = pd.concat([fit_panel, test_panel], axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    if target in combined.columns:
        combined.loc[test_panel.index, target] = np.nan
    combined.attrs["macroforecast_metadata"] = dict(metadata)
    return DataBundle(combined, dict(metadata))


def _validate_panel_selection(
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    model_run: _ModelRun,
) -> None:
    selected, use_model_default_selection = _selection_for_model(selection, model_run)
    if selected is not None:
        raise ValueError(
            "panel-input forecasting does not tune model parameters yet; pass "
            f"model_selection={{'{model_run.alias}': None}} or model_selection=None"
        )
    if isinstance(selection, Mapping) and (
        model_run.alias in selection or model_run.spec.name in selection
    ):
        return
    if use_model_default_selection and model_run.spec.search_spaces:
        return


def _validate_runner_window(
    window_spec: WindowSpec,
    index: int | Sequence[Any] | pd.Index,
) -> None:
    report = window_spec.validate(index)
    if bool(report.get("ok")):
        return
    errors = [str(error) for error in report.get("errors", [])]
    if not errors:
        errors = ["unknown window validation error"]
    raise ValueError(f"window validation failed: {'; '.join(errors)}")


def _resolve_model_runs(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | Mapping[str, str | None] | None,
    params: Mapping[str, Any] | None,
) -> list[_ModelRun]:
    # ``run`` is ATOMIC: exactly one model per call. A mapping or a (non-str)
    # sequence used to fit several models in one run; that is now rejected at the
    # public boundary. The internal ``model_runs`` list is kept as a one-element
    # list so the downstream per-model loops iterate exactly once without churn.
    _reject_multi_model(model)
    single_model = cast(str | Callable[..., Any] | ModelSpec, model)
    base = _get_model_or_ensemble(single_model)
    spec = _get_model_or_ensemble(
        single_model,
        preset=_preset_for_model(preset, None, base.name),
        params=_params_for_model(params, None, base.name, model_spec=base),
    )
    runs = [_ModelRun(alias=spec.name, spec=spec)]
    _validate_preset_mapping(preset, runs)
    _validate_params_mapping(params, runs)
    return runs


def _reject_multi_model(model: Any) -> None:
    """Raise ``TypeError`` if ``model`` is a multi-model sequence or mapping.

    A ``ModelSpec`` is a mapping-like dataclass but is a SINGLE model, and a
    string is a single model name, so both are allowed. Anything else that is a
    ``Mapping`` or a non-string ``Sequence`` is a multi-model request and is no
    longer supported by the atomic ``run``.
    """
    if isinstance(model, ModelSpec):
        return
    if isinstance(model, Mapping) or _is_model_sequence(model):
        raise TypeError(
            "forecasting.run fits exactly ONE model per call; got a "
            f"{type(model).__name__} of models. Run one model per call "
            "(loop over single-model run() calls), or use the pipeline with one "
            "Arm per model to compare models in a single managed run."
        )


def _get_model_or_ensemble(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> ModelSpec:
    try:
        return get_model(model, preset=preset, params=params)
    except ValueError as model_error:
        try:
            return get_model_ensemble(model, preset=preset, params=params)
        except ValueError:
            raise model_error from None


def _is_model_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


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
    *,
    model_spec: ModelSpec | None = None,
) -> Mapping[str, Any] | None:
    if params is None:
        return None
    if alias is not None and isinstance(params.get(alias), Mapping):
        return params[alias]
    if model_name is not None and isinstance(params.get(model_name), Mapping):
        return params[model_name]
    if all(isinstance(value, Mapping) for value in params.values()):
        if model_spec is not None and set(params).issubset(
            _known_model_param_names(model_spec)
        ):
            return params
        return None
    return params


def _actual_model_params(model_spec: ModelSpec, params: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(model_spec.params), **dict(params)}


def _known_model_param_names(model_spec: ModelSpec) -> set[str]:
    return {
        *model_spec.default_params,
        *model_spec.params,
        *(parameter.name for parameter in model_spec.parameters),
    }


def _run_keys(model_runs: Sequence[_ModelRun]) -> set[str]:
    return {
        key
        for model_run in model_runs
        for key in (model_run.alias, model_run.spec.name)
    }


def _validate_preset_mapping(
    preset: str | Mapping[str, str | None] | None,
    model_runs: Sequence[_ModelRun],
) -> None:
    if preset is None or isinstance(preset, str):
        return
    unknown = set(preset) - _run_keys(model_runs)
    if unknown:
        allowed = ", ".join(sorted(_run_keys(model_runs)))
        raise ValueError(
            f"preset contains keys that do not match a model alias or spec: "
            f"{sorted(unknown)}. Available keys: {allowed}."
        )


def _validate_params_mapping(
    params: Mapping[str, Any] | None,
    model_runs: Sequence[_ModelRun],
) -> None:
    if params is None or not all(isinstance(value, Mapping) for value in params.values()):
        return
    keys = set(params)
    run_keys = _run_keys(model_runs)
    if keys & run_keys:
        unknown = keys - run_keys
        if unknown:
            allowed = ", ".join(sorted(run_keys))
            raise ValueError(
                f"params contains keys that do not match a model alias or spec: "
                f"{sorted(unknown)}. Available keys: {allowed}."
            )
        return
    if all(keys.issubset(_known_model_param_names(model_run.spec)) for model_run in model_runs):
        return
    allowed = ", ".join(sorted(run_keys))
    raise ValueError(
        "params looks model-keyed but no key matches a model alias or spec. "
        f"Use one of: {allowed}; or pass direct parameter names accepted by every model."
    )


def _validate_selection_mapping(
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    model_runs: Sequence[_ModelRun],
) -> None:
    if selection is None or isinstance(selection, SearchSpec):
        return
    unknown = set(selection) - _run_keys(model_runs)
    if unknown:
        allowed = ", ".join(sorted(_run_keys(model_runs)))
        raise ValueError(
            f"model_selection contains keys that do not match a model alias or spec: "
            f"{sorted(unknown)}. Available keys: {allowed}."
        )


def _selection_for_model(
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    model_run: _ModelRun,
) -> tuple[SearchSpec | None, bool]:
    if selection is None or isinstance(selection, SearchSpec):
        return selection, True
    if model_run.alias in selection:
        return selection[model_run.alias], selection[model_run.alias] is not None
    if model_run.spec.name in selection:
        return selection[model_run.spec.name], selection[
            model_run.spec.name
        ] is not None
    return None, True


def _prediction_series(prediction: Any, *, index: pd.Index) -> pd.Series:
    if isinstance(prediction, pd.Series):
        return _aligned_or_positional_series(
            prediction,
            index=index,
            label="model prediction",
        )
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError("model prediction DataFrame must have exactly one column")
        return _aligned_or_positional_series(
            prediction.iloc[:, 0],
            index=index,
            label="model prediction",
        )
    values = np.asarray(prediction).reshape(-1)
    if len(values) != len(index):
        raise ValueError("model prediction length does not match X_test")
    return pd.Series(values, index=index)


def _aligned_or_positional_series(
    values: pd.Series,
    *,
    index: pd.Index,
    label: str,
) -> pd.Series:
    if len(values) != len(index):
        raise ValueError(f"{label} length does not match X_test")
    if values.index.equals(index):
        return values.copy()
    if _is_default_position_index(values.index, len(index)):
        return pd.Series(values.to_numpy(), index=index, name=values.name)
    raise ValueError(
        f"{label} index does not match X_test. Return an array-like object for "
        "positional predictions, or return a pandas object indexed by X_test.index."
    )


def _is_default_position_index(index: pd.Index, n: int) -> bool:
    if isinstance(index, pd.RangeIndex):
        return index.equals(pd.RangeIndex(n))
    try:
        return bool(np.array_equal(index.to_numpy(dtype=int), np.arange(n)))
    except (TypeError, ValueError):
        return False


def _variance_series(
    fit: Any,
    *,
    X_test: pd.DataFrame | None = None,
    index: pd.Index,
) -> pd.Series | None:
    if not hasattr(fit, "predict_variance"):
        return None
    prediction = None
    if X_test is not None:
        try:
            prediction = fit.predict_variance(X_test)
        except TypeError:
            prediction = None
    try:
        if prediction is None:
            prediction = fit.predict_variance(horizon=len(index))
    except TypeError:
        prediction = fit.predict_variance(len(index))
    values = _positional_prediction_values(prediction, expected_len=len(index))
    return pd.Series(values, index=index, name="variance_prediction")


def _quantile_frame(
    fit: Any, *, X_test: pd.DataFrame, index: pd.Index
) -> pd.DataFrame | None:
    if not hasattr(fit, "predict_quantiles"):
        return None
    prediction = fit.predict_quantiles(X_test)
    if isinstance(prediction, pd.DataFrame):
        frame = prediction.copy()
        if len(frame) != len(index):
            raise ValueError("quantile prediction length does not match X_test")
        if frame.index.equals(index):
            return frame
        if _is_default_position_index(frame.index, len(index)):
            frame.index = index
            return frame
        raise ValueError(
            "quantile prediction index does not match X_test. Return a DataFrame "
            "indexed by X_test.index, use RangeIndex(len(X_test)), or return a mapping "
            "of array-like quantile predictions."
        )
    if isinstance(prediction, Mapping):
        columns: dict[str, np.ndarray] = {}
        for level, values in prediction.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if len(arr) != len(index):
                raise ValueError("quantile prediction length does not match X_test")
            columns[str(level)] = arr
        return pd.DataFrame(columns, index=index)
    raise TypeError("quantile predictions must be a DataFrame or mapping")


def _store_model_fit(
    fit: Any,
    *,
    root: str | Path,
    alias: str,
    model_spec: ModelSpec,
    row: Mapping[str, Any],
    params: Mapping[str, Any],
    selection_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model_dir = Path(root) / _safe_path_part(alias)
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = _model_store_stem(row)
    pickle_path = model_dir / f"{stem}.pkl"
    metadata_path = model_dir / f"{stem}.json"
    metadata = {
        "alias": alias,
        "model": model_spec.name,
        "model_spec": model_spec.to_metadata(),
        "params": dict(params),
        "model_selection": selection_metadata,
        "window": dict(row),
    }
    return save_fit(
        fit,
        pickle_path,
        metadata_path=metadata_path,
        metadata=metadata,
    ).to_dict()


def _model_store_stem(row: Mapping[str, Any]) -> str:
    origin_pos = row.get("origin_pos", "unknown")
    horizon = row.get("horizon", "unknown")
    origin = row.get("origin")
    if isinstance(origin, pd.Timestamp):
        origin_label = origin.strftime("%Y%m%d")
    else:
        origin_label = str(origin).replace(" ", "_").replace(":", "-")
    target_key = row.get("target_key")
    suffix = "" if target_key is None else f"_{_safe_path_part(target_key)}"
    return f"origin_{origin_pos}_h{horizon}_{_safe_path_part(origin_label)}{suffix}"


def _model_cache_key(alias: str, target_key: Any | None) -> str:
    if target_key is None:
        return str(alias)
    return f"{alias}::{target_key}"


def _forecast_target_dates(
    index: pd.Index,
    *,
    base_index: pd.Index | None,
    horizon: int,
) -> pd.Series:
    if base_index is None:
        return pd.Series(index, index=index)
    positions = base_index.get_indexer(index)
    target_positions = positions + int(horizon)
    valid = (positions >= 0) & (target_positions < len(base_index))
    values = pd.Series(pd.NA, index=index, dtype="object")
    if valid.any():
        valid_positions = target_positions[valid]
        values.iloc[np.flatnonzero(valid)] = list(base_index[valid_positions])
    return values


def _target_level_at(panel: pd.DataFrame, target: str, label: Any) -> float:
    if target not in panel.columns:
        raise ValueError(f"recursive target {target!r} is not present in the panel")
    if label not in panel.index:
        raise ValueError(f"recursive target date {label!r} is not present in the panel")
    value = panel.loc[label, target]
    if pd.isna(value):
        raise ValueError(f"recursive target {target!r} is missing at {label!r}")
    return float(value)


def _recursive_next_level(
    current_level: float,
    prediction: float,
    *,
    transform: str,
) -> float:
    if transform == "level":
        return float(prediction)
    if transform == "change":
        return float(current_level + prediction)
    if transform == "growth":
        return float(current_level * (1.0 + prediction))
    if transform == "log_growth":
        return float(current_level * np.exp(prediction))
    raise ValueError(
        "recursive forecasting supports target_transform level, change, growth, or log_growth"
    )


def _recursive_output_value(
    origin_level: float,
    final_level: float,
    *,
    transform: str,
) -> float:
    if transform == "level":
        return float(final_level)
    if transform == "change":
        return float(final_level - origin_level)
    if transform == "growth":
        if origin_level == 0:
            return float("nan")
        return float(final_level / origin_level - 1.0)
    if transform == "log_growth":
        if origin_level <= 0 or final_level <= 0:
            return float("nan")
        return float(np.log(final_level) - np.log(origin_level))
    raise ValueError(
        "recursive forecasting supports target_transform level, change, growth, or log_growth"
    )


def _path_step_columns(y: pd.DataFrame, *, horizon: int) -> list[str]:
    frame = pd.DataFrame(y)
    columns = [str(column) for column in frame.columns]
    selected: list[str] = []
    for step in range(1, int(horizon) + 1):
        suffix = f"_step{step}"
        matches = [column for column in columns if column.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                "path_average forecasting requires exactly one path target "
                f"column for step {step}; got {matches}"
            )
        selected.append(matches[0])
    return selected


def _feature_target_name(features: FeatureSpec) -> str | None:
    if features.target is not None:
        return features.target
    if len(features.targets) == 1:
        return features.targets[0]
    return None


def _panel_prediction_horizon(
    date: Any,
    *,
    origin: Any,
    base_index: pd.Index,
    default: int,
) -> int:
    origin_index = base_index.insert(0, origin) if origin not in base_index else base_index
    positions = origin_index.get_indexer(pd.Index([origin, date]))
    if (positions < 0).any():
        return int(default)
    # Horizon is the number of steps from the origin to the target date, i.e.
    # positions[1] - positions[0]. The previous "+ 1" overstated every panel
    # prediction's horizon by one, so horizon-keyed evaluation selected the
    # wrong rows.
    return max(1, int(positions[1] - positions[0]))


def _safe_path_part(value: Any) -> str:
    text = str(value)
    keep = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text]
    out = "".join(keep).strip("._")
    return out or "model"


def _positional_prediction_values(prediction: Any, *, expected_len: int) -> np.ndarray:
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError(
                "variance prediction DataFrame must have exactly one column"
            )
        values = prediction.iloc[:, 0].to_numpy(dtype=float)
    elif isinstance(prediction, pd.Series):
        values = prediction.to_numpy(dtype=float)
    else:
        values = np.asarray(prediction, dtype=float).reshape(-1)
    if len(values) != expected_len:
        raise ValueError("variance prediction length does not match X_test")
    return values


def _select_existing_features(
    item: dict[str, Any], prefix: str, policy: StagePolicy
) -> Any:
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
    policies = [
        policy
        for policy in (preprocessing_policy, feature_policy, selection_policy)
        if policy is not None
    ]
    for policy in policies:
        if policy.scope == "custom" and policy.selector is None:
            raise ValueError(
                "custom stage policies require callable selector hooks"
            )
    if preprocessing is None:
        return
    if preprocessing_policy is None:
        return


def _checkpoint_record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identity of a forecast row for de-duplicating checkpoint vs in-memory rows."""
    return (
        record.get("origin_pos"),
        record.get("model"),
        record.get("target"),
        record.get("horizon"),
        record.get("date"),
    )


def _merge_checkpoint_records(
    records: list[dict[str, Any]],
    checkpoint_path: str | Path,
) -> list[dict[str, Any]]:
    """Merge in-memory rich records with checkpoint-loaded lean records.

    Newly-computed origins are kept with their full (rich) columns. Origins that
    were resumed from disk (and therefore never recomputed) are supplied by the
    checkpoint as lean rows. Rows present in memory win; only checkpoint rows with
    a key not already in memory are appended, so a resumed run returns the
    complete frame across all origins without duplication.
    """
    in_memory_keys = {_checkpoint_record_key(record) for record in records}
    frame = load_checkpoint_frame(checkpoint_path)
    if frame.empty:
        return records
    merged = list(records)
    for lean_record in frame.to_dict(orient="records"):
        if _checkpoint_record_key(lean_record) in in_memory_keys:
            continue
        merged.append({column: lean_record.get(column) for column in LEAN_FORECAST_COLUMNS})
    return merged


def _forecast_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    for column in _FORECAST_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype=object)
    extra_columns = [
        str(column) for column in frame.columns if column not in _FORECAST_TABLE_COLUMNS
    ]
    return frame.loc[:, [*list(_FORECAST_TABLE_COLUMNS), *extra_columns]]


def _validate_panel_target(panel: pd.DataFrame, target: str) -> None:
    if target not in panel.columns:
        raise ValueError(
            f"target {target!r} is not present in panel columns; available columns: "
            f"{[str(column) for column in panel.columns]}"
        )


def _stage_update_due(
    policy: StagePolicy | None,
    item: dict[str, Any],
    *,
    origin_count: int,
    state: _StageUpdateState,
) -> bool:
    """Return whether a stateful stage should fit new state at this origin."""

    if policy is None:
        return False
    if not state.updated_once:
        return True
    update = policy.update
    if update == "every_origin":
        return True
    if update == "never":
        return False
    if update == "on_retrain":
        return bool(item["row"].get("retrain", True))
    if isinstance(update, int):
        return origin_count % update == 0
    if isinstance(update, DateOffset):
        origin = _origin_timestamp(item)
        last_origin = _coerce_last_update_timestamp(state.last_origin)
        return origin >= last_origin + update
    raise TypeError(f"unsupported stage policy update {update!r}")


def _mark_stage_updated(state: _StageUpdateState, item: dict[str, Any]) -> None:
    state.updated_once = True
    state.last_origin = item["row"].get("origin")


def _origin_timestamp(item: dict[str, Any]) -> pd.Timestamp:
    origin = item["row"].get("origin")
    try:
        timestamp = pd.Timestamp(origin)
    except Exception as exc:
        raise TypeError(
            "date-offset stage update requires datetime-like window origins"
        ) from exc
    if pd.isna(timestamp):
        raise TypeError("date-offset stage update requires datetime-like window origins")
    return timestamp


def _coerce_last_update_timestamp(origin: Any) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(origin)
    except Exception as exc:
        raise TypeError(
            "date-offset stage update requires datetime-like window origins"
        ) from exc
    if pd.isna(timestamp):
        raise TypeError("date-offset stage update requires datetime-like window origins")
    return timestamp


def _origin_stage_record(
    stage: str,
    item: dict[str, Any],
    metadata: dict[str, Any],
    *,
    updated: bool,
) -> dict[str, Any]:
    row = item["row"]
    return {
        "stage": stage,
        "origin": row.get("origin"),
        "origin_pos": row.get("origin_pos"),
        "updated": bool(updated),
        "fit_start": row.get("fit_start"),
        "fit_end": row.get("fit_end"),
        "test_start": row.get("test_start"),
        "test_end": row.get("test_end"),
        "metadata": metadata,
    }


def _result_metadata(
    *,
    input_path: str,
    input_panel: pd.DataFrame,
    window_spec: WindowSpec,
    model_runs: list[_ModelRun],
    features: Any,
    preprocessing: Any,
    preprocessing_policy: StagePolicy | None,
    feature_policy: StagePolicy | None,
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    combination_specs: Sequence[CombinationSpec],
    n_combination_forecasts: int,
    stage_records: list[dict[str, Any]],
    n_forecasts: int,
    config: Mapping[str, Any],
    save_models: bool,
    model_store: str | Path,
    forecast_policy: ForecastPolicy,
    future_feature_policy: FutureFeaturePolicy | None,
    horizons: Sequence[int],
) -> dict[str, Any]:
    metadata_level = str(config.get("metadata_level", "standard"))
    return {
        "metadata_schema": {
            "kind": "forecast_result",
            "version": 1,
            "input_path": input_path,
            "forecast_table_columns": list(_FORECAST_TABLE_COLUMNS),
            "stage_record_columns": list(_STAGE_RECORD_COLUMNS),
        },
        "run": {
            "n_forecasts": int(n_forecasts),
            "n_models": len(model_runs),
            "n_combinations": len(combination_specs),
            "n_combination_forecasts": int(n_combination_forecasts),
            "input_path": input_path,
            "panel_model_runner": input_path == "panel_model",
            "forecast_policy": forecast_policy,
            "future_feature_policy": future_feature_policy,
            "horizons": [int(value) for value in horizons],
            "multi_horizon": len(tuple(horizons)) > 1,
            "config": dict(config),
            "metadata_level": metadata_level,
            "save_models": bool(save_models),
            "model_store": str(model_store),
        },
        "data": panel_info(
            DataBundle(
                input_panel, dict(input_panel.attrs.get("macroforecast_metadata", {}))
            )
        ),
        "window": window_spec.to_dict(),
        "stage_policies": {
            "preprocessing": None
            if preprocessing_policy is None
            else preprocessing_policy.to_dict(),
            "feature_engineering": None
            if feature_policy is None
            else feature_policy.to_dict(),
            "model_selection": selection_policy.to_dict(),
        },
        "preprocessing": preprocessing,
        "features": features,
        "forecast_policy": {
            "method": forecast_policy,
            "future_feature_policy": future_feature_policy,
            "uses_observed_future_predictors": future_feature_policy == "observed_future",
            "horizons": [int(value) for value in horizons],
        },
        "model_selection": _selection_metadata(selection),
        "combination": [spec.to_dict() for spec in combination_specs],
        "models": [
            {"alias": model_run.alias, "spec": model_run.spec.to_metadata()}
            for model_run in model_runs
        ],
        "stages": [] if metadata_level == "minimal" else stage_records,
    }


def _selection_metadata(
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
) -> Any:
    if selection is None:
        return None
    if isinstance(selection, SearchSpec):
        return selection.to_dict()
    return {
        str(key): None if value is None else value.to_dict()
        for key, value in selection.items()
    }


__all__ = ["run"]
