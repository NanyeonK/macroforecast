from __future__ import annotations


from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from macroforecast.data import (
    DataBundle,
    DataSpec,
    as_panel,
    panel_info,
    validate_panel,
)
from macroforecast.feature_engineering import FeatureSet, FeatureSpec, FittedFeatureBuilder
from macroforecast.meta import get_config
from macroforecast.models import ModelSpec
from macroforecast.preprocessing import FittedPreprocessor, PreprocessSpec
from macroforecast.preprocessing.cache import PreprocessorStore
from macroforecast.model_selection import SearchSpec
from macroforecast.window import (
    StagePolicy,
    WindowSpec,
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
from macroforecast.forecasting.model_resolution import (
    _ModelRun,
    _actual_model_params,  # noqa: F401  (re-export)
    _get_model_or_ensemble,  # noqa: F401  (re-export)
    _is_model_sequence,  # noqa: F401  (re-export)
    _known_model_param_names,  # noqa: F401  (re-export)
    _params_for_model,  # noqa: F401  (re-export)
    _preset_for_model,  # noqa: F401  (re-export)
    _reject_multi_model,  # noqa: F401  (re-export)
    _resolve_model_runs,
    _run_keys,  # noqa: F401  (re-export)
    _validate_params_mapping,  # noqa: F401  (re-export)
    _validate_preset_mapping,  # noqa: F401  (re-export)
)
from macroforecast.forecasting.policy_config import (
    ForecastPolicy,
    FutureFeaturePolicy,
    _feature_spec_for_policy,
    _feature_target_name,
    _feature_window_for_policy,
    _horizon_val_window,  # noqa: F401  (re-export)
    _normalize_forecast_policy,
    _normalize_future_feature_policy,
    _panel_window_for_horizon,
    _target_transform_for_policy,  # noqa: F401  (re-export)
    _validate_recursive_feature_contract,
    _warn_change_based_target_default,  # noqa: F401  (re-export)
)
from macroforecast.forecasting.preprocessing_stage import (
    _PreparedStage,
    _detach_panel_metadata,
    _origin_apply_labels,  # noqa: F401  (re-export)
    _origin_available_labels,  # noqa: F401  (re-export)
    _origin_pos_for_store_key,  # noqa: F401  (re-export)
    _origin_target_pos,  # noqa: F401  (re-export)
    _prepare_origin_panel,
    _preprocessing_cache_key,
    _preprocessor_fit_input,
)
from macroforecast.forecasting.feature_stage import (
    _feature_cache_key,  # noqa: F401  (re-export)
    _fitted_feature_builder_for_origin,
)
from macroforecast.forecasting.selection_stage import (
    _SELECTION_DEGRADED_KEY,  # noqa: F401  (re-export)
    _SELECTION_TUNED_KEY,  # noqa: F401  (re-export)
    _align_feature_xy,
    _allow_non_temporal_selection_splits,  # noqa: F401  (re-export)
    _assert_selection_was_possible,
    _availability_safe_explicit_splits,  # noqa: F401  (re-export)
    _availability_safe_selection_splits,  # noqa: F401  (re-export)
    _filter_xy_to_target_availability,  # noqa: F401  (re-export)
    _relative_splits_for_index,
    _resolve_degraded_selection,  # noqa: F401  (re-export)
    _selection_for_model,  # noqa: F401  (re-export)
    _single_target,
    _target_availability_base_index,  # noqa: F401  (re-export)
    _target_availability_mask,  # noqa: F401  (re-export)
    _target_availability_window_fields,  # noqa: F401  (re-export)
    _validate_selection_mapping,
)


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
    preprocessing_cache: dict[Any, FittedPreprocessor | _PreparedStage | FittedFeatureBuilder] | None = None,
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
    # ``run`` is atomic (exactly one model per call, enforced by
    # ``_reject_multi_model`` in ``_resolve_model_runs``), so ``model_runs`` is
    # always a one-element list here; its spec name/``input_kind`` gate the
    # implicit default-feature-spec warning (see ``_feature_spec_for_policy``).
    features = _feature_spec_for_policy(
        features,
        target=target,
        horizon=horizon_values[0],
        forecast_policy=policy,
        future_feature_policy=future_policy,
        target_transform=target_transform,
        model_input_kind=model_runs[0].spec.input_kind,
        model_name=model_runs[0].spec.name,
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
                # Gap A: share the (often expensive PCA/MARX/SIR) feature-builder
                # fit across arms of the same target via the SAME shared cache dict
                # _prepare_origin_panel already uses for preprocessing -- see
                # feature_stage.py for the content-digest + exact fit-sample-bounds
                # key design that makes this safe regardless of why two arms' fit
                # sample might differ (window override, retrain cadence, scope).
                fitted_feature_cache = _fitted_feature_builder_for_origin(
                    features,
                    feature_fit_panel,
                    prepared_metadata=prepared_metadata,
                    feature_stage_policy=feature_stage_policy,
                    item=item,
                    preprocessing_cache=preprocessing_cache,
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
            _OriginRunConfig(
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
            ),
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
    preprocessing_cache: dict[Any, FittedPreprocessor | _PreparedStage | FittedFeatureBuilder] | None = None,
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
                _OriginRunConfig(
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
                ),
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
    _validate_runner_window(window_spec, panel.index, exclude_origin=True)
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
    for origin_count, item in enumerate(
        # exclude_origin=True: the panel test slice must not include the
        # origin date itself -- see WindowSpec.origins() and issue #423.
        window_spec.iter_origins(panel.index, exclude_origin=True)
    ):
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


def _fit_predict_origin(
    item: dict[str, Any],
    cfg: _OriginRunConfig,
) -> list[dict[str, Any]]:
    """Route one per-origin feature-matrix item to its policy strategy.

    The policy bodies live in ``macroforecast.forecasting.policies`` and are
    selected through the ``POLICY_FORECASTERS`` registry -- the direct policy
    is a registered strategy like every other policy (the historical
    inline-direct asymmetry of this function is gone). This name is kept in
    ``runner`` so existing spy/monkeypatch call sites keep working.
    """

    return _dispatch_policy(item, cfg)


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


def _validate_runner_window(
    window_spec: WindowSpec,
    index: int | Sequence[Any] | pd.Index,
    *,
    exclude_origin: bool = False,
) -> None:
    report = window_spec.validate(index, exclude_origin=exclude_origin)
    if bool(report.get("ok")):
        return
    errors = [str(error) for error in report.get("errors", [])]
    if not errors:
        errors = ["unknown window validation error"]
    raise ValueError(f"window validation failed: {'; '.join(errors)}")


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

    ``variance_prediction`` is one of :data:`LEAN_FORECAST_COLUMNS` (a plain
    float), so it survives this merge with no extra handling. Quantile
    predictions are stored on disk as wide ``q_<pct>`` columns;
    ``load_checkpoint_frame`` already reconstructs a ``quantile_predictions``
    mapping column from them (see ``forecasting/checkpoint.py``), matching the
    SAME ``{level_str: value}`` representation a freshly-computed origin's
    ``quantile_predictions`` carries, so a resumed run's forecast table never
    mixes two different quantile representations across rows.
    """
    in_memory_keys = {_checkpoint_record_key(record) for record in records}
    frame = load_checkpoint_frame(checkpoint_path)
    if frame.empty:
        return records
    merged = list(records)
    for lean_record in frame.to_dict(orient="records"):
        if _checkpoint_record_key(lean_record) in in_memory_keys:
            continue
        row = {column: lean_record.get(column) for column in LEAN_FORECAST_COLUMNS}
        row["quantile_predictions"] = lean_record.get("quantile_predictions")
        merged.append(row)
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


# ---------------------------------------------------------------------------
# Policy strategies (Phase 3 of the runner decomposition). Imported at the
# BOTTOM of the module on purpose: the policy bodies consume stage helpers
# still defined above in this module (they move out in Phase 4), so importing
# the package any earlier would be circular. Private ``_fit_predict_*`` names
# are re-exported so existing imports and monkeypatch targets keep working.
from macroforecast.forecasting.policies import (  # noqa: E402
    POLICY_FORECASTERS,  # noqa: F401  (re-export)
    dispatch as _dispatch_policy,
)
from macroforecast.forecasting.policies.base import (  # noqa: E402
    _FitOutcome,  # noqa: F401  (re-export)
    _OriginRunConfig,
    _aligned_or_positional_series,  # noqa: F401  (re-export)
    _fit_one_model_at_origin,  # noqa: F401  (re-export)
    _forecast_target_dates,  # noqa: F401  (re-export)
    _is_default_position_index,  # noqa: F401  (re-export)
    _model_cache_key,  # noqa: F401  (re-export)
    _model_store_stem,  # noqa: F401  (re-export)
    _positional_prediction_values,  # noqa: F401  (re-export)
    _prediction_series,  # noqa: F401  (re-export)
    _quantile_frame,  # noqa: F401  (re-export)
    _safe_path_part,  # noqa: F401  (re-export)
    _store_model_fit,  # noqa: F401  (re-export)
    _variance_series,  # noqa: F401  (re-export)
)
from macroforecast.forecasting.policies.direct import (  # noqa: E402
    forecast_direct_origin as _forecast_direct_origin,  # noqa: F401  (re-export)
)
from macroforecast.forecasting.policies.panel import (  # noqa: E402
    _panel_fit_params,  # noqa: F401  (re-export)
    _panel_prediction_horizon,  # noqa: F401  (re-export; tests import it from runner)
    _panel_prediction_input_without_test_target,  # noqa: F401  (re-export)
    _validate_panel_selection,  # noqa: F401  (re-export)
    forecast_panel_origin as _fit_predict_panel_origin,
)
from macroforecast.forecasting.policies.path_average import (  # noqa: E402
    _path_step_columns,  # noqa: F401  (re-export)
    forecast_path_average_origin as _fit_predict_path_average_origin,  # noqa: F401  (re-export)
)
from macroforecast.forecasting.policies.recursive import (  # noqa: E402
    _recursive_next_level,  # noqa: F401  (re-export)
    _recursive_output_value,  # noqa: F401  (re-export)
    _target_level_at,  # noqa: F401  (re-export)
    forecast_recursive_origin as _fit_predict_recursive_origin,  # noqa: F401  (re-export)
)
