from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import pandas as pd

from macroforecast.data import DataBundle, attach_metadata, panel_info
from macroforecast.preprocessing.clean import apply_standardization_state
from macroforecast.preprocessing.preprocess import (
    _coerce_input,
    _callable_name,
    _coerce_custom_preprocess_output,
    _json_ready,
    _normalize_impute,
    _normalize_outliers,
    reprocess,
)
from macroforecast.preprocessing.types import PreprocessedData, PreprocessInput

_REPROCESS_OPTIONS = {
    "frequency",
    "quarterly_to_monthly",
    "weekly_to_monthly",
    "monthly_to_quarterly",
    "weekly_to_quarterly",
    "transform_order",
    "transform",
    "transform_codes",
    "transform_code_overrides",
    "tcode_lag",
    "outliers",
    "outlier_action",
    "iqr_threshold",
    "zscore_threshold",
    "winsorize_quantiles",
    "impute",
    "em_n_factors",
    "em_factor_selection",
    "em_demean",
    "em_max_iter",
    "em_tolerance",
    "standardize",
    "standardize_columns",
    "standardize_ddof",
    "frame",
    "custom_steps",
    "warn_metadata",
}

_FIT_WINDOW_CUSTOM_WARNED: set[int] = set()


@dataclass(frozen=True)
class PreprocessSpec:
    """Reusable preprocessing callable for window-local forecasting runners."""

    options: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        data: PreprocessInput,
        *,
        metadata: dict[str, Any] | None = None,
        policy: str = "origin_available",
    ) -> FittedPreprocessor:
        """Fit preprocessing choices on a training panel."""

        base = _coerce_input(data, metadata=metadata)
        scope = _normalize_preprocessing_scope(policy)
        if scope == "fit_window":
            _validate_fit_window_options(self.options)
        custom_steps = _normalize_custom_steps(self.options.get("custom_steps"))
        if scope == "fit_window" and custom_steps:
            _warn_fit_window_custom_steps(self)
        fit_options = _reprocess_options(self.options)
        fit_options.setdefault("warn_metadata", False)
        processed = reprocess(data, metadata=metadata, **fit_options)
        processed = _apply_custom_preprocess_steps(processed, custom_steps)
        state_panel: pd.DataFrame | None = None
        outlier_state: dict[str, Any] | None = None
        impute_state: dict[str, Any] | None = None
        train_after_outlier: pd.DataFrame | None = None
        if scope == "fit_window":
            state_options = _state_base_options(self.options)
            state_panel = reprocess(data, metadata=metadata, **state_options).panel
            outlier_state = _fit_outlier_state(state_panel, self.options)
            train_after_outlier = _apply_outlier_state(state_panel, outlier_state)
            impute_state = _fit_impute_state(train_after_outlier, self.options)
        stage = {
            "options": _metadata_options(self.options),
            "preprocessing_scope": scope,
            "fit_input": panel_info(DataBundle(base.panel, base.metadata)),
            "fit_output": panel_info(DataBundle(processed.panel, processed.metadata)),
            "fit_period": _panel_period(base.panel),
            "processed_fit_period": _panel_period(processed.panel),
        }
        processed_panel = processed.panel.copy()
        processed_metadata = attach_metadata(processed.metadata, "preprocess_spec", stage)
        processed_panel.attrs["macroforecast_metadata"] = processed_metadata
        processed = PreprocessedData(
            panel=processed_panel,
            metadata=processed_metadata,
            target=processed.target,
            targets=processed.targets,
            horizons=processed.horizons,
            start=processed.start,
            end=processed.end,
            predictors=processed.predictors,
            steps=processed.steps,
        )
        return FittedPreprocessor(
            spec=self,
            fit_panel=base.panel.copy(),
            fit_metadata=dict(base.metadata),
            processed_train=processed,
            preprocessing_scope=scope,
            standardization_state=_standardization_state(processed),
            state_panel=None if state_panel is None else state_panel.copy(),
            outlier_state=outlier_state,
            impute_state=impute_state,
            train_after_outlier=None if train_after_outlier is None else train_after_outlier.copy(),
        )

    def fit_transform(
        self,
        data: PreprocessInput,
        *,
        metadata: dict[str, Any] | None = None,
        policy: str = "origin_available",
    ) -> PreprocessedData:
        """Fit on ``data`` and return the processed training panel."""

        return self.fit(data, metadata=metadata, policy=policy).processed_train

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready preprocessing choices."""

        return {"options": _metadata_options(self.options)}

    def to_metadata(self) -> dict[str, Any]:
        """Return compact metadata for runners."""

        return {"preprocessing": self.to_dict()}


@dataclass(frozen=True)
class FittedPreprocessor:
    """Preprocessing spec fitted on a training window."""

    spec: PreprocessSpec
    fit_panel: pd.DataFrame
    fit_metadata: dict[str, Any]
    processed_train: PreprocessedData
    preprocessing_scope: str = "origin_available"
    standardization_state: dict[str, Any] | None = None
    state_panel: pd.DataFrame | None = None
    outlier_state: dict[str, Any] | None = None
    impute_state: dict[str, Any] | None = None
    train_after_outlier: pd.DataFrame | None = None

    def transform(
        self,
        data: PreprocessInput,
        *,
        metadata: dict[str, Any] | None = None,
        history: pd.DataFrame | None = None,
        policy: str | None = None,
        available: pd.Index | None = None,
    ) -> PreprocessedData:
        """Transform new rows using the fitted training history.

        ``policy="origin_available"`` replays non-standardization steps on
        ``history + data``. ``policy="fit_window"`` applies states fitted on
        the fit window to the requested rows.

        ``available`` lists the row labels that were observable at the forecast
        origin. When supplied, origin_available fits the data-dependent steps
        (outlier statistics and EM/iterative imputation) using only those rows
        plus the training history, so a post-origin row appended to ``data``
        (e.g. the h-step target realization row) cannot leak future information
        into the imputed/flagged values of origin-available rows.
        """

        base = _coerce_input(data, metadata=metadata)
        scope = _normalize_preprocessing_scope(policy or self.preprocessing_scope)
        if scope == "fit_window":
            return self._transform_fit_window(base, history=history)

        return self._transform_origin_available(
            base, history=history, available=available
        )

    def _transform_origin_available(
        self,
        base: Any,
        *,
        history: pd.DataFrame | None,
        available: pd.Index | None = None,
    ) -> PreprocessedData:
        source_history = self.fit_panel if history is None else history
        combined = _combine_panels(source_history, base.panel)
        combined_metadata = dict(self.fit_metadata)
        combined_metadata.update(base.metadata)
        custom_steps = _normalize_custom_steps(self.spec.options.get("custom_steps"))
        transform_options = _reprocess_options(self.spec.options)
        transform_options.setdefault("warn_metadata", False)
        if self.standardization_state is not None:
            transform_options["standardize"] = "none"

        def _run(frame: pd.DataFrame) -> PreprocessedData:
            out = reprocess((frame, combined_metadata), **transform_options)
            out = _apply_fitted_standardization_state(out, self.standardization_state)
            return _apply_custom_preprocess_steps(out, custom_steps)

        processed = _run(combined)
        merged_panel = processed.panel
        # Leak-free origin_available: data-dependent steps (outlier thresholds,
        # EM/iterative imputation) for origin-available rows are recomputed using
        # only origin-available rows (training history + the rows observable at
        # the origin). Post-origin rows appended to ``data`` (e.g. the h-step
        # target realization row) therefore cannot influence the imputed or
        # flagged feature values of any origin-available row. Post-origin rows
        # keep the full-panel reconstruction (they are never used as training
        # features; the runner's target-availability mask excludes them).
        if available is not None:
            available_labels = pd.Index(source_history.index).union(pd.Index(available))
            available_rows = combined.index.intersection(available_labels)
            if len(available_rows) and len(available_rows) < len(combined.index):
                processed_available = _run(combined.loc[available_rows])
                merged_panel = merged_panel.copy()
                overlap = merged_panel.index.intersection(processed_available.panel.index)
                merged_panel.loc[overlap, :] = processed_available.panel.loc[
                    overlap, merged_panel.columns
                ]
        selected = merged_panel.reindex(base.panel.index)
        selected_metadata = dict(processed.metadata)
        selected_metadata = attach_metadata(
            selected_metadata,
            "preprocess_transform",
            {
                "options": _metadata_options(self.spec.options),
                "preprocessing_scope": "origin_available",
                "fit_rows": int(self.fit_panel.shape[0]),
                "transform_rows": int(base.panel.shape[0]),
                "history_rows": int(source_history.shape[0]),
                "fit_period": _panel_period(self.fit_panel),
                "history_period": _panel_period(source_history),
                "transform_period": _panel_period(base.panel),
                "output_period": _panel_period(selected),
                "input_panel": panel_info(DataBundle(base.panel, base.metadata)),
                "output_panel": panel_info(DataBundle(selected, selected_metadata)),
                "standardize_refit": False,
            },
        )
        selected.attrs["macroforecast_metadata"] = selected_metadata
        return PreprocessedData(
            panel=selected,
            metadata=selected_metadata,
            target=base.target or self.processed_train.target,
            targets=base.targets or self.processed_train.targets,
            horizons=base.horizons or self.processed_train.horizons,
            start=base.start or self.processed_train.start,
            end=base.end or self.processed_train.end,
            predictors=(
                self.processed_train.predictors
                if base.predictors == "all" and self.processed_train.predictors != "all"
                else base.predictors
            ),
            steps=processed.steps,
        )

    def _transform_fit_window(
        self,
        base: Any,
        *,
        history: pd.DataFrame | None,
    ) -> PreprocessedData:
        if self.preprocessing_scope != "fit_window":
            raise ValueError("policy='fit_window' requires fitting the preprocessor with policy='fit_window'")
        source_history = self.fit_panel if history is None else history
        combined = _combine_panels(source_history, base.panel)
        combined_metadata = dict(self.fit_metadata)
        combined_metadata.update(base.metadata)
        processed_base = reprocess((combined, combined_metadata), **_state_base_options(self.spec.options))
        selected = processed_base.panel.reindex(base.panel.index)
        selected = _apply_outlier_state(selected, self.outlier_state)
        selected = _apply_impute_state(
            selected,
            self.impute_state,
            train_context=self.train_after_outlier,
        )
        if self.standardization_state is not None:
            selected = apply_standardization_state(selected, self.standardization_state)
        custom_steps = _normalize_custom_steps(self.spec.options.get("custom_steps"))
        if custom_steps:
            custom_processed = _apply_custom_preprocess_steps(
                PreprocessedData(
                    panel=selected,
                    metadata=processed_base.metadata,
                    target=base.target or self.processed_train.target,
                    targets=base.targets or self.processed_train.targets,
                    horizons=base.horizons or self.processed_train.horizons,
                    start=base.start or self.processed_train.start,
                    end=base.end or self.processed_train.end,
                    predictors=base.predictors,
                    steps=tuple(processed_base.steps),
                ),
                custom_steps,
            )
            selected = custom_processed.panel
            processed_base = custom_processed
        selected_metadata = _fit_window_metadata(
            processed_base.metadata,
            self,
            transform_panel=base.panel,
            history=source_history,
            output=selected,
        )
        selected.attrs["macroforecast_metadata"] = selected_metadata
        return PreprocessedData(
            panel=selected,
            metadata=selected_metadata,
            target=base.target or self.processed_train.target,
            targets=base.targets or self.processed_train.targets,
            horizons=base.horizons or self.processed_train.horizons,
            start=base.start or self.processed_train.start,
            end=base.end or self.processed_train.end,
            predictors=(
                self.processed_train.predictors
                if base.predictors == "all" and self.processed_train.predictors != "all"
                else base.predictors
            ),
            steps=tuple(selected_metadata.get("preprocessing", {}).get("steps", ())),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return fit metadata for forecasting records."""

        return {
            "preprocess_spec": self.spec.to_dict(),
            "preprocessing_scope": self.preprocessing_scope,
            "fit_panel": panel_info(DataBundle(self.fit_panel, self.fit_metadata)),
            "processed_train": panel_info(
                DataBundle(self.processed_train.panel, self.processed_train.metadata)
            ),
            "standardization_state": self.standardization_state,
            "outlier_state": _metadata_state(self.outlier_state),
            "impute_state": _metadata_state(self.impute_state),
        }


def preprocess_spec(**options: Any) -> PreprocessSpec:
    """Create a reusable preprocessing specification.

    Keyword options are the same data-cleaning choices accepted by
    ``reprocess(...)``: frequency alignment, transform-code handling,
    outlier policy, imputation policy, standardization, frame-edge handling,
    and optional custom preprocessing steps. Stage timing and metadata are not
    accepted here; they are supplied later through ``PreprocessSpec.fit(...)``
    or by the forecasting/pipeline runner.

    Custom preprocessing callables are safe for in-memory use. Disk-backed
    preprocessing caches require a stable callable identity: use named
    functions and set ``func.__mf_digest__`` whenever cached reuse should span
    processes or runs. Anonymous lambdas without ``__mf_digest__`` are rejected
    because they cannot be distinguished by a stable content identity.

    With ``policy="fit_window"``, custom steps are re-executed on the apply
    window at each origin. Those steps must be row-local/stateless; a custom
    step that computes statistics from its whole input can read post-origin
    rows and leak future information.

    Returns
    -------
    PreprocessSpec
        Frozen preprocessing configuration. Call ``fit(data)`` to get a
        ``FittedPreprocessor`` or ``fit_transform(data)`` to obtain a
        ``PreprocessedData`` object for the training panel.

    Example
    -------
    >>> import macroforecast as mf
    >>> prep = mf.preprocessing.preprocess_spec(
    ...     transform="official",
    ...     outliers="iqr",
    ...     impute="em_factor",
    ...     standardize="zscore",
    ... )
    """

    unexpected = sorted(set(options) - _REPROCESS_OPTIONS)
    if unexpected:
        raise TypeError(
            "unexpected preprocess_spec option(s): "
            f"{unexpected}. Stage timing belongs to forecasting.run(...), "
            "and input metadata belongs to PreprocessSpec.fit(...)."
        )
    _reject_undigested_lambda_custom_steps(options.get("custom_steps"))
    return PreprocessSpec(options=dict(options))


def custom_preprocess_step(
    name: str,
    func: Callable[..., Any],
    **params: Any,
) -> dict[str, Any]:
    """Return a custom preprocessing step for ``preprocess_spec(custom_steps=...)``.

    For disk-backed preprocessing caches, set ``func.__mf_digest__`` to a stable
    string and update it when the callable's behavior changes. Without that
    opt-in digest, the runner skips disk get/put for specs containing the
    callable and recomputes instead of risking stale reuse.
    """

    if not callable(func):
        raise TypeError("custom preprocessing step func must be callable")
    if not name:
        raise ValueError("custom preprocessing step name must be non-empty")
    return {"name": str(name), "func": func, "params": dict(params)}


def _combine_panels(history: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([history, panel], axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _panel_period(panel: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(panel.shape[0]),
        "start": panel.index[0].strftime("%Y-%m-%d") if len(panel.index) else None,
        "end": panel.index[-1].strftime("%Y-%m-%d") if len(panel.index) else None,
    }


def _standardization_state(processed: PreprocessedData) -> dict[str, Any] | None:
    state = processed.metadata.get("preprocessing", {}).get("standardization_state", {})
    if isinstance(state, dict) and state:
        return dict(state)
    return None


def _normalize_preprocessing_scope(value: str) -> str:
    aliases = {
        "origin_available": "origin_available",
        "available": "origin_available",
        "expanding_available": "origin_available",
        "observed_available": "origin_available",
        "fit_window": "fit_window",
        "fit": "fit_window",
        "train_window": "fit_window",
        "estimation_window": "fit_window",
    }
    key = str(value).lower().replace("-", "_")
    if key not in aliases:
        raise ValueError("preprocessing policy must be 'origin_available' or 'fit_window'")
    return aliases[key]


def _state_base_options(options: dict[str, Any]) -> dict[str, Any]:
    out = _reprocess_options(options)
    out["outliers"] = "none"
    out["impute"] = "none"
    out["standardize"] = "none"
    out["frame"] = "keep"
    out.setdefault("warn_metadata", False)
    return out


def _validate_fit_window_options(options: dict[str, Any]) -> None:
    method = _normalize_impute(str(options.get("impute", "em_factor")))
    if method in {"em_factor", "em_multivariate", "linear"}:
        raise ValueError(
            "policy='fit_window' currently supports impute='none', "
            "impute='mean', impute='zero', and impute='forward_fill'. Use "
            "policy='origin_available' for EM or linear imputation -- it now "
            "fits imputation only on rows observable at each forecast origin, "
            "so it is leak-free for pseudo-out-of-sample runs."
        )


def _reprocess_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(options).items() if key != "custom_steps"}


def _metadata_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return _json_ready(dict(options))


def _warn_fit_window_custom_steps(spec: PreprocessSpec) -> None:
    key = id(spec)
    if key in _FIT_WINDOW_CUSTOM_WARNED:
        return
    _FIT_WINDOW_CUSTOM_WARNED.add(key)
    warnings.warn(
        "custom preprocessing steps with policy='fit_window' are re-executed on "
        "each apply window and must be row-local/stateless; data-dependent "
        "statistics inside custom steps can leak post-origin rows.",
        UserWarning,
        stacklevel=3,
    )


def _apply_fitted_standardization_state(
    data: PreprocessedData,
    state: dict[str, Any] | None,
) -> PreprocessedData:
    if state is None:
        return data
    panel = apply_standardization_state(data.panel, state)
    metadata = dict(data.metadata)
    preprocessing_stage = dict(metadata.get("preprocessing", {}))
    preprocessing_stage["standardize"] = state.get("method")
    preprocessing_stage["standardize_columns"] = list(state.get("columns", ()))
    preprocessing_stage["standardization_state"] = dict(state)
    preprocessing_stage["steps"] = _replace_standardize_step(
        preprocessing_stage.get("steps", data.steps),
        state,
    )
    metadata = attach_metadata(metadata, "preprocessing", preprocessing_stage)
    panel.attrs["macroforecast_metadata"] = metadata
    return PreprocessedData(
        panel=panel,
        metadata=metadata,
        target=data.target,
        targets=data.targets,
        horizons=data.horizons,
        start=data.start,
        end=data.end,
        predictors=data.predictors,
        steps=tuple(preprocessing_stage.get("steps", data.steps)),
    )


def _reject_undigested_lambda_custom_steps(value: Any) -> None:
    if value is None:
        return
    if callable(value):
        _reject_undigested_lambda(value, path="custom_steps")
        return
    if isinstance(value, Mapping):
        func = value.get("func", value.get("callable"))
        if callable(func):
            _reject_undigested_lambda(func, path=f"custom step {value.get('name', '<unnamed>')!r}")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_undigested_lambda_custom_steps(item)
        return


def _reject_undigested_lambda(func: Callable[..., Any], *, path: str) -> None:
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", ""))
    if "<lambda>" not in str(qualname):
        return
    if getattr(func, "__mf_digest__", None) is not None:
        return
    raise ValueError(
        f"{path} uses an anonymous lambda without __mf_digest__; use a named "
        "function or set a stable __mf_digest__ so disk preprocessing caches "
        "cannot collide or replay stale callable code."
    )


def _normalize_custom_steps(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if callable(value):
        return (custom_preprocess_step(_callable_name(value), value),)
    if isinstance(value, Mapping):
        return (_normalize_custom_step(value),)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_custom_step(item) for item in value)
    raise TypeError("custom_steps must be a callable, mapping, sequence, or None")


def _normalize_custom_step(value: Any) -> dict[str, Any]:
    if callable(value):
        return custom_preprocess_step(_callable_name(value), value)
    if not isinstance(value, Mapping):
        raise TypeError("each custom preprocessing step must be a mapping or callable")
    raw = dict(value)
    func = raw.pop("func", raw.pop("callable", None))
    if not callable(func):
        raise TypeError("custom preprocessing step requires callable 'func'")
    name = str(raw.pop("name", _callable_name(func)))
    params = dict(raw.pop("params", {}))
    params.update(raw)
    return custom_preprocess_step(name, func, **params)


def _apply_custom_preprocess_steps(
    data: PreprocessedData,
    steps: tuple[dict[str, Any], ...],
) -> PreprocessedData:
    if not steps:
        return data
    current = data
    for step in steps:
        func = step["func"]
        params = dict(step.get("params", {}))
        output = func(current.panel.copy(), metadata=dict(current.metadata), **params)
        panel, metadata = _coerce_custom_preprocess_output(output, current.metadata)
        record = {
            "name": str(step["name"]),
            "callable": _callable_name(func),
            "params": _json_ready(params),
            "input_panel": panel_info(DataBundle(current.panel, current.metadata)),
            "output_panel": panel_info(DataBundle(panel, metadata)),
        }
        updated = dict(metadata)
        updated["custom_preprocess_steps"] = [
            *current.metadata.get("custom_preprocess_steps", []),
            record,
        ]
        panel = panel.copy()
        panel.attrs["macroforecast_metadata"] = updated
        current = PreprocessedData(
            panel=panel,
            metadata=updated,
            target=current.target,
            targets=current.targets,
            horizons=current.horizons,
            start=current.start,
            end=current.end,
            predictors=current.predictors,
            steps=(*current.steps, {"step": "custom_preprocess", "name": str(step["name"])}),
        )
    return current


def _normalize_outlier_action(value: Any) -> str:
    aliases = {
        "flag_as_nan": "flag_as_nan",
        "nan": "flag_as_nan",
        "replace_with_median": "replace_with_median",
        "median": "replace_with_median",
        "replace_with_cap_value": "replace_with_cap_value",
        "cap": "replace_with_cap_value",
        "clip": "replace_with_cap_value",
    }
    key = str(value).lower()
    if key not in aliases:
        raise ValueError("outlier_action must be 'flag_as_nan', 'replace_with_median', or 'replace_with_cap_value'")
    return aliases[key]


def _fit_outlier_state(panel: pd.DataFrame, options: dict[str, Any]) -> dict[str, Any]:
    method = _normalize_outliers(str(options.get("outliers", "iqr")))
    if method == "none":
        return {"method": "none"}
    numeric = panel.select_dtypes("number")
    if numeric.empty:
        return {"method": method, "columns": []}
    state: dict[str, Any] = {"method": method, "columns": [str(column) for column in numeric.columns]}
    if method == "iqr":
        threshold = float(options.get("iqr_threshold", 10.0))
        if threshold <= 0:
            raise ValueError("iqr_threshold must be positive")
        median = numeric.median()
        iqr = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, np.nan)
        cap_low = numeric.quantile(0.01)
        cap_high = numeric.quantile(0.99)
        state.update(
            {
                "threshold": threshold,
                "action": _normalize_outlier_action(options.get("outlier_action", "flag_as_nan")),
                "median": _series_dict(median),
                "iqr": _series_dict(iqr),
                "cap_low": _series_dict(cap_low),
                "cap_high": _series_dict(cap_high),
            }
        )
    elif method == "zscore":
        threshold = float(options.get("zscore_threshold", 3.0))
        if threshold <= 0:
            raise ValueError("zscore_threshold must be positive")
        mean = numeric.mean()
        std = numeric.std(ddof=0).replace(0, np.nan)
        cap_low = numeric.quantile(0.01)
        cap_high = numeric.quantile(0.99)
        state.update(
            {
                "threshold": threshold,
                "action": _normalize_outlier_action(options.get("outlier_action", "flag_as_nan")),
                "mean": _series_dict(mean),
                "std": _series_dict(std),
                "cap_low": _series_dict(cap_low),
                "cap_high": _series_dict(cap_high),
            }
        )
    elif method == "winsorize":
        lower, upper = options.get("winsorize_quantiles", (0.01, 0.99))
        lower_value = float(lower)
        upper_value = float(upper)
        if not (0 <= lower_value < upper_value <= 1):
            raise ValueError("winsorize_quantiles must satisfy 0 <= lower < upper <= 1")
        state.update(
            {
                "lower_quantile": lower_value,
                "upper_quantile": upper_value,
                "lower": _series_dict(numeric.quantile(lower_value)),
                "upper": _series_dict(numeric.quantile(upper_value)),
            }
        )
    else:  # pragma: no cover - guarded by _normalize_outliers
        raise ValueError(f"unknown outlier method {method!r}")
    return state


def _apply_outlier_state(panel: pd.DataFrame, state: dict[str, Any] | None) -> pd.DataFrame:
    if not state or state.get("method") == "none":
        return panel.copy()
    method = str(state.get("method"))
    columns = _state_columns(panel, state)
    if not columns:
        return panel.copy()
    result = panel.copy()
    values = result.loc[:, columns].astype(float)
    if method == "iqr":
        median = _state_series(state, "median", columns)
        iqr = _state_series(state, "iqr", columns)
        threshold = float(state["threshold"])
        mask = ((values - median).abs() > threshold * iqr).fillna(False)
        result.loc[:, columns] = _apply_outlier_action(values, mask, state)
    elif method == "zscore":
        mean = _state_series(state, "mean", columns)
        std = _state_series(state, "std", columns)
        threshold = float(state["threshold"])
        mask = (((values - mean) / std).abs() > threshold).fillna(False)
        result.loc[:, columns] = _apply_outlier_action(values, mask, state)
    elif method == "winsorize":
        lower = _state_series(state, "lower", columns)
        upper = _state_series(state, "upper", columns)
        result.loc[:, columns] = values.clip(lower=lower, upper=upper, axis=1)
    else:  # pragma: no cover - guarded by fit
        raise ValueError(f"unknown outlier method {method!r}")
    result.attrs.update(dict(getattr(panel, "attrs", {}) or {}))
    return result


def _fit_impute_state(panel: pd.DataFrame, options: dict[str, Any]) -> dict[str, Any]:
    method = _normalize_impute(str(options.get("impute", "em_factor")))
    if method in {"em_factor", "em_multivariate", "linear"}:
        raise ValueError(
            "policy='fit_window' currently supports impute='none', "
            "impute='mean', impute='zero', and impute='forward_fill'. Use "
            "policy='origin_available' for EM or linear imputation -- it now "
            "fits imputation only on rows observable at each forecast origin, "
            "so it is leak-free for pseudo-out-of-sample runs."
        )
    if method == "none":
        return {"method": "none"}
    numeric = panel.select_dtypes("number")
    if method == "mean":
        return {"method": "mean", "means": _series_dict(numeric.mean())}
    if method == "zero":
        return {"method": "zero"}
    if method == "forward_fill":
        return {"method": "forward_fill", "columns": [str(column) for column in panel.columns]}
    raise ValueError(f"unknown imputation method {method!r}")


def _apply_impute_state(
    panel: pd.DataFrame,
    state: dict[str, Any] | None,
    *,
    train_context: pd.DataFrame | None,
) -> pd.DataFrame:
    if not state or state.get("method") == "none":
        return panel.copy()
    method = str(state.get("method"))
    result = panel.copy()
    if method == "mean":
        means = state.get("means", {})
        if not isinstance(means, dict):
            raise ValueError("mean imputation state is invalid")
        fill_values = {column: float(value) for column, value in means.items() if column in result.columns}
        return result.fillna(fill_values)
    if method == "zero":
        return result.fillna(0)
    if method == "forward_fill":
        if train_context is None:
            return result.ffill()
        filled = pd.concat([train_context, result], axis=0).ffill()
        return filled.reindex(result.index)
    raise ValueError(f"unknown imputation method {method!r}")


def _replace_standardize_step(steps: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    found = False
    for step in (steps if isinstance(steps, (list, tuple)) else ()):
        row = dict(step) if isinstance(step, dict) else {"step": str(step)}
        if row.get("step") == "standardize":
            row.update(
                {
                    "method": state.get("method"),
                    "columns": list(state.get("columns", ())),
                    "ddof": int(state.get("ddof", 0)),
                    "fitted_on": "train_window",
                }
            )
            found = True
        updated.append(row)
    if not found:
        updated.append(
            {
                "step": "standardize",
                "method": state.get("method"),
                "columns": list(state.get("columns", ())),
                "ddof": int(state.get("ddof", 0)),
                "fitted_on": "train_window",
            }
        )
    return updated


def _fit_window_metadata(
    metadata: dict[str, Any],
    fit: FittedPreprocessor,
    *,
    transform_panel: pd.DataFrame,
    history: pd.DataFrame,
    output: pd.DataFrame,
) -> dict[str, Any]:
    selected_metadata = dict(metadata)
    preprocessing_stage = dict(selected_metadata.get("preprocessing", {}))
    preprocessing_stage["preprocessing_scope"] = "fit_window"
    preprocessing_stage["outlier_state"] = _metadata_state(fit.outlier_state)
    preprocessing_stage["impute_state"] = _metadata_state(fit.impute_state)
    if fit.standardization_state is not None:
        preprocessing_stage["standardize"] = fit.standardization_state.get("method")
        preprocessing_stage["standardize_columns"] = list(fit.standardization_state.get("columns", ()))
        preprocessing_stage["standardization_state"] = dict(fit.standardization_state)
        preprocessing_stage["steps"] = _replace_standardize_step(
            preprocessing_stage.get("steps", ()),
            fit.standardization_state,
        )
    selected_metadata = attach_metadata(selected_metadata, "preprocessing", preprocessing_stage)
    return attach_metadata(
        selected_metadata,
        "preprocess_transform",
        {
            "options": dict(fit.spec.options),
            "preprocessing_scope": "fit_window",
            "fit_rows": int(fit.fit_panel.shape[0]),
            "transform_rows": int(transform_panel.shape[0]),
            "history_rows": int(history.shape[0]),
            "fit_period": _panel_period(fit.fit_panel),
            "history_period": _panel_period(history),
            "transform_period": _panel_period(transform_panel),
            "output_period": _panel_period(output),
            "input_panel": panel_info(DataBundle(transform_panel, fit.fit_metadata)),
            "output_panel": panel_info(DataBundle(output, metadata)),
            "stateful_steps": ["outliers", "impute", "standardize"],
        },
    )


def _metadata_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    out = dict(state)
    return out


def _series_dict(series: pd.Series) -> dict[str, float]:
    return {
        str(column): float(value)
        for column, value in series.items()
        if pd.notna(value)
    }


def _state_columns(panel: pd.DataFrame, state: dict[str, Any]) -> list[Any]:
    lookup = {str(column): column for column in panel.columns}
    return [lookup[column] for column in state.get("columns", ()) if column in lookup]


def _state_series(state: dict[str, Any], key: str, columns: list[Any]) -> pd.Series:
    values = state.get(key, {})
    if not isinstance(values, dict):
        raise ValueError(f"outlier state {key!r} must be a mapping")
    return pd.Series({column: float(values[str(column)]) for column in columns if str(column) in values})


def _apply_outlier_action(values: pd.DataFrame, mask: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    action = str(state.get("action", "flag_as_nan"))
    if action == "flag_as_nan":
        return values.mask(mask)
    if action == "replace_with_median":
        median = _state_series(state, "median", list(values.columns))
        return values.mask(mask, median, axis=1)
    cap_low = _state_series(state, "cap_low", list(values.columns))
    cap_high = _state_series(state, "cap_high", list(values.columns))
    capped = values.clip(lower=cap_low, upper=cap_high, axis=1)
    return values.where(~mask, capped)


__all__ = ["FittedPreprocessor", "PreprocessSpec", "custom_preprocess_step", "preprocess_spec"]
