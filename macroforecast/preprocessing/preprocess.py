from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd

from macroforecast.preprocessing.clean import (
    apply_standardization_state,
    apply_tcode_transform,
    drop_unbalanced_series_clean,
    em_factor_impute_clean,
    em_multivariate_impute_clean,
    fit_standardization_state,
    forward_fill_clean,
    iqr_outlier_clean,
    linear_interpolate_clean,
    mean_impute_clean,
    standardize_clean,
    truncate_to_balanced_clean,
    winsorize_clean,
    zero_fill_leading_clean,
    zscore_outlier_clean,
)
from macroforecast.data import (
    DataBundle,
    DataSpec,
    align_frequency,
    as_panel,
    attach_metadata,
    frequency_hardening_issues,
    infer_frequencies,
    panel_info,
    validate_panel,
)
from macroforecast.preprocessing.types import PreprocessedData, PreprocessInput, _InputBundle


FRED_SD_NATIONAL_ANALOG_TRANSFORM_CODES: dict[str, int] = {
    "CONS": 5,
    "FIRE": 5,
    "GOVT": 5,
    "ICLAIMS": 5,
    "INFO": 5,
    "LF": 5,
    "MFG": 5,
    "MFGHRS": 5,
    "MINNG": 5,
    "NA": 5,
    "PARTRATE": 2,
    "PSERV": 5,
    "UR": 2,
}

FRED_SD_MEDIUM_CONFIDENCE_TRANSFORM_CODES: dict[str, int] = {
    "BPPRIVSA": 5,
    "CONSTNQGSP": 5,
    "EXPORTS": 5,
    "FIRENQGSP": 5,
    "GOVNQGSP": 5,
    "IMPORTS": 5,
    "INFONQGSP": 5,
    "MANNQGSP": 5,
    "NATURNQGSP": 5,
    "NQGSP": 5,
    "OTOT": 5,
    "PSERVNQGSP": 5,
    "RENTS": 5,
    "STHPI": 5,
    "UTILNQGSP": 5,
}


def reprocess(
    data: PreprocessInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    frequency: str = "keep",
    quarterly_to_monthly: str = "step_forward",
    weekly_to_monthly: str = "mean",
    monthly_to_quarterly: str = "quarterly_average",
    weekly_to_quarterly: str = "mean",
    transform_order: str = "after_frequency",
    transform: str = "official",
    transform_codes: Mapping[str, int] | None = None,
    transform_code_overrides: Mapping[str, int] | None = None,
    tcode_lag: str = "drop",
    outliers: str = "iqr",
    outlier_action: str = "flag_as_nan",
    iqr_threshold: float = 10.0,
    zscore_threshold: float = 3.0,
    winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
    impute: str = "em_factor",
    em_n_factors: int = 8,
    em_factor_selection: str = "baing_p2",
    em_demean: int = 2,
    em_max_iter: int = 50,
    em_tolerance: float = 1e-6,
    standardize: str = "none",
    standardize_columns: str | Sequence[str] = "all",
    standardize_ddof: int = 0,
    frame: str = "keep",
    warn_metadata: bool = True,
) -> PreprocessedData:
    """Preprocess a canonical macroforecast panel.

    Parameters use user-facing names. Common legacy aliases such as
    ``apply_official_tcode`` and ``truncate_to_balanced`` are accepted, but
    returned metadata records the canonical direct-call names.
    """

    base = _coerce_input(data, metadata=metadata)
    if warn_metadata:
        _warn_if_no_data_metadata(base.metadata)
    panel = base.panel.copy()
    validate_panel(panel)
    input_info = panel_info(DataBundle(panel, base.metadata))

    steps: list[dict[str, Any]] = []
    frequency_method = _normalize_frequency(frequency)
    transform_order_method = _normalize_transform_order(transform_order)
    transform_method = _normalize_transform(transform)
    if transform_method == "official" and _is_fred_sd_metadata(base.metadata):
        raise ValueError(
            "FRED-SD has no official t-code map. Use transform='none' or "
            "transform='custom' with fred_sd_transform_codes(...)."
        )
    tcode_lag_method = _normalize_tcode_lag(tcode_lag)
    applied_codes: dict[str, int] = {}
    transform_state: dict[str, Any] = {}

    if transform_order_method == "before_frequency":
        panel, applied_codes, transform_state = _apply_transform_step(
            panel,
            metadata=base.metadata,
            transform_method=transform_method,
            transform_codes=transform_codes,
            transform_code_overrides=transform_code_overrides,
            steps=steps,
        )
        panel = _apply_tcode_lag_step(panel, method=tcode_lag_method, codes=applied_codes, steps=steps)
        panel = _apply_frequency_step(
            panel,
            method=frequency_method,
            quarterly_to_monthly=quarterly_to_monthly,
            weekly_to_monthly=weekly_to_monthly,
            monthly_to_quarterly=monthly_to_quarterly,
            weekly_to_quarterly=weekly_to_quarterly,
            steps=steps,
        )
    else:
        panel = _apply_frequency_step(
            panel,
            method=frequency_method,
            quarterly_to_monthly=quarterly_to_monthly,
            weekly_to_monthly=weekly_to_monthly,
            monthly_to_quarterly=monthly_to_quarterly,
            weekly_to_quarterly=weekly_to_quarterly,
            steps=steps,
        )
        panel, applied_codes, transform_state = _apply_transform_step(
            panel,
            metadata=base.metadata,
            transform_method=transform_method,
            transform_codes=transform_codes,
            transform_code_overrides=transform_code_overrides,
            steps=steps,
        )
        panel = _apply_tcode_lag_step(panel, method=tcode_lag_method, codes=applied_codes, steps=steps)

    outlier_method = _normalize_outliers(outliers)
    before_missing = int(panel.isna().sum().sum())
    panel = handle_outliers(
        panel,
        method=outlier_method,
        action=outlier_action,
        iqr_threshold=iqr_threshold,
        zscore_threshold=zscore_threshold,
        winsorize_quantiles=winsorize_quantiles,
    )
    steps.append(
        {
            "step": "outliers",
            "method": outlier_method,
            "action": outlier_action if outlier_method in {"iqr", "zscore"} else None,
            "missing_added": int(panel.isna().sum().sum()) - before_missing,
        }
    )

    impute_method = _normalize_impute(impute)
    before_missing = int(panel.isna().sum().sum())
    panel = impute_missing(
        panel,
        method=impute_method,
        em_n_factors=em_n_factors,
        em_factor_selection=em_factor_selection,
        em_demean=em_demean,
        em_max_iter=em_max_iter,
        em_tolerance=em_tolerance,
    )
    steps.append(
        {
            "step": "impute",
            "method": impute_method,
            "missing_filled": max(before_missing - int(panel.isna().sum().sum()), 0),
        }
    )

    standardize_method = _normalize_standardize(standardize)
    standardize_ddof_value = _normalize_standardize_ddof(standardize_ddof)
    standardization_state: dict[str, Any] = {}
    before_shape = tuple(int(value) for value in panel.shape)
    if standardize_method == "none":
        standardized_panel = panel.copy()
    else:
        columns_to_standardize = _resolve_standardize_columns(panel, base, standardize_columns)
        standardization_state = fit_standardization_state(
            panel.loc[:, columns_to_standardize],
            method=standardize_method,
            ddof=standardize_ddof_value,
        )
        standardized_panel = apply_standardization_state(panel, standardization_state)
    panel = standardized_panel
    steps.append(
        {
            "step": "standardize",
            "method": standardize_method,
            "columns": list(standardization_state.get("columns", ())),
            "ddof": standardize_ddof_value,
            "input_shape": before_shape,
            "output_shape": tuple(int(value) for value in panel.shape),
        }
    )

    frame_method = _normalize_frame(frame)
    before_shape = tuple(int(value) for value in panel.shape)
    panel = handle_frame_edges(panel, method=frame_method)
    steps.append(
        {
            "step": "frame",
            "method": frame_method,
            "input_shape": before_shape,
            "output_shape": tuple(int(value) for value in panel.shape),
        }
    )

    if panel.empty:
        raise ValueError("preprocessing leaves an empty panel")
    panel = as_panel(panel, metadata=base.metadata)
    output_info = panel_info(DataBundle(panel, base.metadata))
    stage = {
        "frequency": frequency_method,
        "transform_order": transform_order_method,
        "transform": transform_method,
        "transform_state": transform_state,
        "tcode_lag": tcode_lag_method,
        "outliers": outlier_method,
        "impute": impute_method,
        "standardize": standardize_method,
        "standardize_columns": list(standardization_state.get("columns", ())),
        "standardization_state": standardization_state,
        "frame": frame_method,
        "steps": steps,
        "input_panel": input_info,
        "output_panel": output_info,
    }
    updated_metadata = attach_metadata(base.metadata, "preprocessing", stage)
    if applied_codes:
        updated_metadata = attach_metadata(updated_metadata, "transform_codes_applied", applied_codes)
    panel.attrs["macroforecast_metadata"] = updated_metadata
    if applied_codes:
        # Store the final applied map, not merely the raw user input. This is
        # important when transform_code_overrides changes official metadata:
        # downstream diagnostics should see the exact transformations that were
        # actually used on this output panel.
        panel.attrs["macroforecast_transform_codes"] = dict(applied_codes)
    return PreprocessedData(
        panel=panel,
        metadata=updated_metadata,
        target=base.target,
        targets=base.targets,
        horizons=base.horizons,
        start=base.start,
        end=base.end,
        predictors=base.predictors,
        steps=tuple(steps),
    )


def apply_transform_codes(panel: pd.DataFrame, codes: Mapping[str, int]) -> pd.DataFrame:
    """Apply McCracken-Ng transform codes to matching panel columns."""

    if not codes:
        return panel.copy()
    return apply_tcode_transform(panel, dict(codes))


def custom_preprocess(
    data: PreprocessInput,
    func: Callable[..., Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    name: str | None = None,
    **params: Any,
) -> PreprocessedData:
    """Apply a user supplied preprocessing callable to a canonical panel."""

    if not callable(func):
        raise TypeError("custom preprocessing func must be callable")
    base = _coerce_input(data, metadata=metadata)
    output = func(base.panel.copy(), metadata=dict(base.metadata), **params)
    panel, output_metadata = _coerce_custom_preprocess_output(output, base.metadata)
    validate_panel(panel)
    step_name = name or _callable_name(func)
    stage = {
        "name": str(step_name),
        "callable": _callable_name(func),
        "params": _json_ready(params),
        "input_panel": panel_info(DataBundle(base.panel, base.metadata)),
        "output_panel": panel_info(DataBundle(panel, output_metadata)),
    }
    updated_metadata = attach_metadata(output_metadata, "custom_preprocess", stage)
    panel = panel.copy()
    panel.attrs["macroforecast_metadata"] = updated_metadata
    return PreprocessedData(
        panel=panel,
        metadata=updated_metadata,
        target=base.target,
        targets=base.targets,
        horizons=base.horizons,
        start=base.start,
        end=base.end,
        predictors=base.predictors,
        steps=(
            *(data.steps if isinstance(data, PreprocessedData) else ()),
            {"step": "custom_preprocess", "name": str(step_name)},
        ),
    )


def standardize_panel(
    panel: pd.DataFrame,
    *,
    method: str = "zscore",
    ddof: int = 0,
    standardize_scope: str = "fit_window",
    available: pd.Index | Sequence[Any] | None = None,
    columns: str | Sequence[str] = "all",
    predictors: str | Sequence[str] = "all",
    target: str | None = None,
    targets: Sequence[str] | None = None,
    nan_policy: Literal["propagate", "zero_after_standardize"] = "propagate",
    standardize_nan_fill: float | None = None,
) -> pd.DataFrame:
    """Standardize numeric columns with fitted parameters."""

    scope = _normalize_standardize_scope(standardize_scope)
    method_value = _normalize_standardize(method)
    nan_policy_value = _normalize_standardize_nan_policy(
        nan_policy,
        standardize_nan_fill,
    )
    if method_value == "none":
        return panel.copy()
    if (
        nan_policy_value == "propagate"
        and scope == "fit_window"
        and available is None
        and columns == "all"
        and predictors == "all"
        and target is None
        and targets is None
    ):
        return standardize_clean(panel, method=method_value, ddof=ddof)
    ddof_value = _normalize_standardize_ddof(ddof)
    if scope == "fit_window":
        fit_columns = _resolve_standardize_panel_columns(panel, columns)
        state = fit_standardization_state(
            panel.loc[:, fit_columns],
            method=method_value,
            ddof=ddof_value,
        )
        result = apply_standardization_state(panel, state)
        if nan_policy_value == "zero_after_standardize":
            return _fill_nonfinite_standardized_columns(
                result,
                _standardization_state_columns(state),
            )
        return result
    if available is None:
        raise ValueError(
            "standardize_scope='origin_available_predictors' requires available rows"
        )
    fit_rows = panel.index.intersection(pd.Index(available))
    if len(fit_rows) == 0:
        raise ValueError("available rows select no rows in panel")
    target_names = _target_column_names(target=target, targets=targets)
    predictor_columns = _resolve_predictor_standardize_columns(
        panel,
        predictors=predictors,
        targets=target_names,
        columns=columns,
    )
    state = fit_standardization_state(
        panel.loc[fit_rows, predictor_columns],
        method=method_value,
        ddof=ddof_value,
    )
    result = apply_standardization_state(panel, state)
    if nan_policy_value == "zero_after_standardize":
        return _fill_nonfinite_standardized_columns(
            result,
            _standardization_state_columns(state),
        )
    return result


def fred_sd_transform_codes(
    data: PreprocessInput,
    *,
    variable_codes: Mapping[str, int] | None = None,
    state_series_codes: Mapping[str, int] | None = None,
    use_national_analog_suggestions: bool = True,
    include_medium_confidence: bool = False,
    return_table: bool = False,
) -> dict[str, int] | tuple[dict[str, int], pd.DataFrame]:
    """Build FRED-SD t-code choices for state-series columns.

    FRED-SD does not publish official t-codes. The built-in suggestions are
    national-analog defaults, not official transformations.
    """

    base = _coerce_input(data)
    suggested: dict[str, tuple[int, str, str]] = {}
    if use_national_analog_suggestions:
        suggested.update(
            {
                key: (_validate_tcode(value, name=key), "national_analog", "high")
                for key, value in FRED_SD_NATIONAL_ANALOG_TRANSFORM_CODES.items()
            }
        )
    if include_medium_confidence:
        for key, value in FRED_SD_MEDIUM_CONFIDENCE_TRANSFORM_CODES.items():
            suggested.setdefault(key, (_validate_tcode(value, name=key), "national_analog", "medium"))
    if variable_codes:
        suggested.update(
            {
                str(key).upper(): (_validate_tcode(int(value), name=str(key)), "user_variable", "user")
                for key, value in variable_codes.items()
            }
        )

    series_overrides = {str(key): int(value) for key, value in (state_series_codes or {}).items()}
    expanded: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for column in base.panel.columns:
        name = str(column)
        variable, state = _fred_sd_column_parts(name)
        tcode: int | None = None
        source = "unassigned"
        suggestion_confidence = "none"
        if name in series_overrides:
            tcode = _validate_tcode(series_overrides[name], name=name)
            source = "user_state_series"
            suggestion_confidence = "user"
        elif variable.upper() in suggested:
            tcode, source, suggestion_confidence = suggested[variable.upper()]
        if tcode is not None:
            expanded[name] = tcode
        records.append(
            {
                "column": name,
                "sd_variable": variable,
                "state": state,
                "tcode": tcode,
                "source": source,
                "suggestion_confidence": suggestion_confidence,
            }
        )
    if return_table:
        table = pd.DataFrame.from_records(
            records,
            columns=["column", "sd_variable", "state", "tcode", "source", "suggestion_confidence"],
        )
        return expanded, table
    return expanded


def plan(
    data: PreprocessInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    frequency: str = "keep",
    transform_order: str = "after_frequency",
    transform: str = "official",
    transform_codes: Mapping[str, int] | None = None,
    transform_code_overrides: Mapping[str, int] | None = None,
    tcode_lag: str = "drop",
    outliers: str = "iqr",
    impute: str = "em_factor",
    standardize: str = "none",
    standardize_columns: str | Sequence[str] = "all",
    standardize_ddof: int = 0,
    frame: str = "keep",
) -> dict[str, Any]:
    """Return a dry-run summary of preprocessing choices and metadata provenance."""

    base = _coerce_input(data, metadata=metadata)
    validate_panel(base.panel)
    frequency_method = _normalize_frequency(frequency)
    transform_order_method = _normalize_transform_order(transform_order)
    transform_method = _normalize_transform(transform)
    if transform_method == "official" and _is_fred_sd_metadata(base.metadata):
        transform_error = "FRED-SD has no official t-code map"
        applied_codes: dict[str, int] = {}
        ignored_codes: dict[str, int] = {}
    else:
        transform_error = None
        codes = (
            {}
            if transform_method == "none"
            else _resolve_transform_codes(
                base.panel,
                base.metadata,
                transform_codes,
                transform_code_overrides=transform_code_overrides,
            )
        )
        applied_codes = {column: int(code) for column, code in codes.items() if column in base.panel.columns}
        ignored_codes = {column: int(code) for column, code in codes.items() if column not in base.panel.columns}
        if transform_method in {"official", "custom"} and not codes:
            transform_error = f"transform={transform_method!r} has no t-code map"
        elif transform_method in {"official", "custom"} and codes and not applied_codes:
            transform_error = f"transform={transform_method!r} has no t-code keys matching panel columns"
    frequency_map, frequency_source = infer_frequencies(base.panel)
    frequency_issues = frequency_hardening_issues(frequency_map)
    ddof_value = _normalize_standardize_ddof(standardize_ddof)
    return {
        "input_panel": panel_info(DataBundle(base.panel, base.metadata)),
        "metadata_warning": _data_metadata_warning_message(base.metadata),
        "steps": _ordered_step_names(transform_order_method),
        "frequency": {
            "method": frequency_method,
            "metadata_source": frequency_source,
            "native_frequencies": frequency_map,
            "issues": frequency_issues,
        },
        "transform": {
            "method": transform_method,
            "applied_codes": applied_codes,
            "ignored_codes": ignored_codes,
            "error": transform_error,
        },
        "tcode_lag": _normalize_tcode_lag(tcode_lag),
        "outliers": _normalize_outliers(outliers),
        "impute": _normalize_impute(impute),
        "standardize": _normalize_standardize(standardize),
        "standardize_ddof": ddof_value,
        "standardize_columns": _resolve_standardize_columns(base.panel, base, standardize_columns)
        if _normalize_standardize(standardize) != "none"
        else [],
        "frame": _normalize_frame(frame),
    }


def report(processed: PreprocessedData) -> dict[str, Any]:
    """Return a compact preprocessing report from a processed object."""

    if not isinstance(processed, PreprocessedData):
        raise TypeError("report() expects a PreprocessedData object")
    stage = dict(processed.metadata.get("preprocessing", {}))
    return {
        "input_panel": stage.get("input_panel"),
        "output_panel": stage.get("output_panel"),
        "steps": tuple(stage.get("steps", ())),
        "choices": {
            "frequency": stage.get("frequency"),
            "transform_order": stage.get("transform_order"),
            "transform": stage.get("transform"),
            "tcode_lag": stage.get("tcode_lag"),
            "outliers": stage.get("outliers"),
            "impute": stage.get("impute"),
            "standardize": stage.get("standardize"),
            "standardize_columns": stage.get("standardize_columns"),
            "standardize_ddof": _standardize_step_ddof(stage.get("steps", ())),
            "frame": stage.get("frame"),
        },
        "transform_state": stage.get("transform_state", {}),
        "standardization_state": stage.get("standardization_state", {}),
    }


def handle_tcode_lag(
    panel: pd.DataFrame,
    *,
    method: str = "drop",
    codes: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Handle missing values introduced by stationarity transforms."""

    method = _normalize_tcode_lag(method)
    if method == "keep":
        return panel.copy()
    if method == "drop":
        leading_loss = max((_tcode_leading_loss(code) for code in (codes or {}).values()), default=0)
        return panel.iloc[leading_loss:].copy() if leading_loss else panel.copy()
    if method == "drop_all_missing_rows":
        return panel.dropna(axis=0, how="all").copy()
    if method == "drop_any_missing_rows":
        return panel.dropna(axis=0, how="any").copy()
    raise ValueError(f"unknown transform missing method {method!r}")


def handle_outliers(
    panel: pd.DataFrame,
    *,
    method: str = "iqr",
    action: str = "flag_as_nan",
    iqr_threshold: float = 10.0,
    zscore_threshold: float = 3.0,
    winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Apply one outlier policy to a panel."""

    method = _normalize_outliers(method)
    if method == "none":
        return panel.copy()
    if method == "iqr":
        return iqr_outlier_clean(panel, threshold=iqr_threshold, action=action)
    if method == "zscore":
        return zscore_outlier_clean(panel, threshold=zscore_threshold, action=action)
    if method == "winsorize":
        lower, upper = winsorize_quantiles
        return winsorize_clean(panel, lower_quantile=lower, upper_quantile=upper)
    raise ValueError(f"unknown outlier method {method!r}")


def impute_missing(
    panel: pd.DataFrame,
    *,
    method: str = "em_factor",
    em_n_factors: int = 8,
    em_factor_selection: str = "baing_p2",
    em_demean: int = 2,
    em_max_iter: int = 50,
    em_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Fill missing panel values with the selected imputation method."""

    method = _normalize_impute(method)
    if method == "none":
        return panel.copy()
    if method == "mean":
        return mean_impute_clean(panel)
    if method == "zero":
        return zero_fill_leading_clean(panel)
    if method == "forward_fill":
        return forward_fill_clean(panel)
    if method == "linear":
        return linear_interpolate_clean(panel)
    if method == "em_factor":
        return em_factor_impute_clean(
            panel,
            n_factors=em_n_factors,
            max_iter=em_max_iter,
            tol=em_tolerance,
            factor_selection=em_factor_selection,
            demean=em_demean,
        )
    if method == "em_multivariate":
        return em_multivariate_impute_clean(panel, max_iter=em_max_iter, tol=em_tolerance)
    raise ValueError(f"unknown imputation method {method!r}")


def handle_frame_edges(panel: pd.DataFrame, *, method: str = "keep") -> pd.DataFrame:
    """Handle remaining unbalanced panel edges."""

    method = _normalize_frame(method)
    if method == "keep":
        return panel.copy()
    if method == "truncate":
        return truncate_to_balanced_clean(panel)
    if method == "drop_unbalanced_series":
        return drop_unbalanced_series_clean(panel)
    if method == "zero_fill":
        return zero_fill_leading_clean(panel)
    raise ValueError(f"unknown frame method {method!r}")


def _coerce_input(data: PreprocessInput, *, metadata: Mapping[str, Any] | None = None) -> _InputBundle:
    if isinstance(data, PreprocessedData):
        base = _InputBundle(
            panel=data.panel,
            metadata=dict(data.metadata),
            target=data.target,
            targets=data.targets,
            horizons=data.horizons,
            start=data.start,
            end=data.end,
            predictors=data.predictors,
        )
    elif isinstance(data, DataSpec):
        base = _InputBundle(
            panel=data.panel,
            metadata=dict(data.metadata),
            target=data.target,
            targets=data.targets,
            horizons=data.horizons,
            start=data.start,
            end=data.end,
            predictors=data.predictors,
        )
    elif isinstance(data, DataBundle):
        base = _InputBundle(panel=data.panel, metadata=dict(data.metadata))
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], pd.DataFrame):
        panel = as_panel(data[0], metadata=data[1])
        base = _InputBundle(panel=panel, metadata=dict(data[1]))
    elif isinstance(data, pd.DataFrame):
        existing = dict(data.attrs.get("macroforecast_metadata", {}))
        base = _InputBundle(panel=as_panel(data, metadata=existing), metadata=existing)
    else:
        raise TypeError("expected PreprocessedData, DataSpec, DataBundle, (panel, metadata), or pandas DataFrame")
    if metadata is None:
        return base
    merged = dict(base.metadata)
    merged.update(dict(metadata))
    panel = base.panel.copy()
    panel.attrs["macroforecast_metadata"] = merged
    return replace(base, panel=panel, metadata=merged)


def _resolve_transform_codes(
    panel: pd.DataFrame,
    metadata: Mapping[str, Any],
    transform_codes: Mapping[str, int] | None,
    *,
    transform_code_overrides: Mapping[str, int] | None = None,
) -> dict[str, int]:
    codes: dict[str, int]
    if transform_codes is not None:
        codes = {str(key): int(value) for key, value in transform_codes.items()}
    elif metadata.get("transform_codes"):
        codes = {str(key): int(value) for key, value in dict(metadata["transform_codes"]).items()}
    else:
        attr_codes = panel.attrs.get("macroforecast_transform_codes", {})
        codes = {str(key): int(value) for key, value in dict(attr_codes).items()}
    if transform_code_overrides:
        codes.update({str(key): int(value) for key, value in transform_code_overrides.items()})
    return {key: _validate_tcode(value, name=key) for key, value in codes.items()}


def _apply_frequency_step(
    panel: pd.DataFrame,
    *,
    method: str,
    quarterly_to_monthly: str,
    weekly_to_monthly: str,
    monthly_to_quarterly: str,
    weekly_to_quarterly: str,
    steps: list[dict[str, Any]],
) -> pd.DataFrame:
    before_shape = tuple(int(value) for value in panel.shape)
    _frequencies, frequency_source = infer_frequencies(panel)
    result = align_frequency(
        DataBundle(panel, dict(panel.attrs.get("macroforecast_metadata", {}))),
        method=method,
        quarterly_to_monthly=quarterly_to_monthly,
        weekly_to_monthly=weekly_to_monthly,
        monthly_to_quarterly=monthly_to_quarterly,
        weekly_to_quarterly=weekly_to_quarterly,
    ).panel
    steps.append(
        {
            "step": "frequency",
            "method": method,
            "metadata_source": frequency_source,
            "input_shape": before_shape,
            "output_shape": tuple(int(value) for value in result.shape),
        }
    )
    return result


def _apply_transform_step(
    panel: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    transform_method: str,
    transform_codes: Mapping[str, int] | None,
    transform_code_overrides: Mapping[str, int] | None,
    steps: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    if transform_method == "none":
        steps.append({"step": "transform", "method": "none", "applied": {}})
        return panel.copy(), {}, {}

    codes = _resolve_transform_codes(
        panel,
        metadata,
        transform_codes,
        transform_code_overrides=transform_code_overrides,
    )
    # Fail closed here. In macro forecasting, "official" and "custom" are
    # semantic promises that a t-code map exists. A silent no-op would leave
    # levels in place, break stationarity assumptions, and make the downstream
    # evaluation look better or worse for the wrong reason.
    if transform_method == "official" and not codes:
        raise ValueError("transform='official' requires transform_codes or metadata transform_codes")
    if transform_method == "custom" and not codes:
        raise ValueError("transform='custom' requires transform_codes")
    applied_codes = {column: int(code) for column, code in codes.items() if column in panel.columns}
    ignored_codes = {column: int(code) for column, code in codes.items() if column not in panel.columns}
    if ignored_codes and (transform_codes is not None or transform_code_overrides is not None):
        raise ValueError(f"transform code keys are not in the panel: {sorted(ignored_codes)}")
    if not applied_codes:
        raise ValueError(f"transform={transform_method!r} has no t-code keys matching panel columns")
    transform_state = _build_transform_state(panel, applied_codes)
    result = apply_transform_codes(panel, applied_codes) if applied_codes else panel.copy()
    steps.append(
        {
            "step": "transform",
            "method": transform_method,
            "applied": applied_codes,
            "ignored_metadata_codes": ignored_codes,
        }
    )
    return result, applied_codes, transform_state


def _apply_tcode_lag_step(
    panel: pd.DataFrame,
    *,
    method: str,
    codes: Mapping[str, int],
    steps: list[dict[str, Any]],
) -> pd.DataFrame:
    before_shape = tuple(int(value) for value in panel.shape)
    result = handle_tcode_lag(panel, method=method, codes=codes)
    steps.append(
        {
            "step": "tcode_lag",
            "method": method,
            "input_shape": before_shape,
            "output_shape": tuple(int(value) for value in result.shape),
            "rows_removed": before_shape[0] - int(result.shape[0]),
        }
    )
    return result


def _data_metadata_warning_message(metadata: Mapping[str, Any]) -> str | None:
    if metadata.get("dataset"):
        return None
    return (
        "reprocess() works best with metadata produced by macroforecast.data. "
        "Pass a DataBundle/DataSpec from mf.data.load_*(), mf.data.load_custom_*(), "
        "or a DataFrame with macroforecast_metadata attrs."
    )


def _warn_if_no_data_metadata(metadata: Mapping[str, Any]) -> None:
    message = _data_metadata_warning_message(metadata)
    if message:
        warnings.warn(message, UserWarning, stacklevel=3)


def _ordered_step_names(transform_order: str) -> tuple[str, ...]:
    if transform_order == "before_frequency":
        return ("transform", "tcode_lag", "frequency", "outliers", "impute", "standardize", "frame")
    return ("frequency", "transform", "tcode_lag", "outliers", "impute", "standardize", "frame")


def _normalize_frequency(value: str) -> str:
    aliases = {
        "keep": "keep",
        "native": "keep",
        "mixed": "keep",
        "monthly": "monthly",
        "align_monthly": "monthly",
        "to_monthly": "monthly",
        "quarterly": "quarterly",
        "align_quarterly": "quarterly",
        "to_quarterly": "quarterly",
        "drop_non_monthly": "drop_non_monthly",
        "monthly_only": "drop_non_monthly",
        "drop_non_quarterly": "drop_non_quarterly",
        "quarterly_only": "drop_non_quarterly",
    }
    return _lookup(value, aliases, "frequency")


def _normalize_transform_order(value: str) -> str:
    aliases = {
        "after_frequency": "after_frequency",
        "frequency_then_transform": "after_frequency",
        "frequency_first": "after_frequency",
        "default": "after_frequency",
        "before_frequency": "before_frequency",
        "transform_then_frequency": "before_frequency",
        "transform_first": "before_frequency",
    }
    return _lookup(value, aliases, "transform_order")


def _normalize_transform(value: str) -> str:
    aliases = {
        "none": "none",
        "no_transform": "none",
        "official": "official",
        "official_tcode": "official",
        "apply_official_tcode": "official",
        "custom": "custom",
        "custom_tcode": "custom",
    }
    return _lookup(value, aliases, "transform")


def _normalize_tcode_lag(value: str) -> str:
    aliases = {
        "keep": "keep",
        "none": "keep",
        "drop": "drop",
        "tcode_lag": "drop",
        "fred_md": "drop",
        "mccracken_ng_2016": "drop",
        "drop_all_missing_rows": "drop_all_missing_rows",
        "drop_all_na_rows": "drop_all_missing_rows",
        "drop_any_missing_rows": "drop_any_missing_rows",
        "drop_any_na_rows": "drop_any_missing_rows",
    }
    return _lookup(value, aliases, "tcode_lag")


def _normalize_outliers(value: str) -> str:
    aliases = {
        "none": "none",
        "iqr": "iqr",
        "mccracken_ng_iqr": "iqr",
        "zscore": "zscore",
        "zscore_threshold": "zscore",
        "winsorize": "winsorize",
    }
    return _lookup(value, aliases, "outliers")


def _normalize_impute(value: str) -> str:
    aliases = {
        "none": "none",
        "none_propagate": "none",
        "mean": "mean",
        "zero": "zero",
        "zeros": "zero",
        "zero_fill": "zero",
        "forward_fill": "forward_fill",
        "ffill": "forward_fill",
        "linear": "linear",
        "linear_interpolation": "linear",
        "em_factor": "em_factor",
        "em_multivariate": "em_multivariate",
    }
    return _lookup(value, aliases, "impute")


def _normalize_standardize(value: str) -> str:
    aliases = {
        "none": "none",
        "no": "none",
        "false": "none",
        "zscore": "zscore",
        "standard": "zscore",
        "standardize": "zscore",
        "standardized": "zscore",
        "robust": "robust",
        "iqr": "robust",
        "minmax": "minmax",
        "min_max": "minmax",
    }
    return _lookup(value, aliases, "standardize")


def _normalize_standardize_scope(value: str) -> str:
    aliases = {
        "fit_window": "fit_window",
        "fit": "fit_window",
        "train_window": "fit_window",
        "estimation_window": "fit_window",
        "origin_available_predictors": "origin_available_predictors",
        "available_predictors": "origin_available_predictors",
        "origin_predictors": "origin_available_predictors",
    }
    return _lookup(value, aliases, "standardize_scope")


def _normalize_standardize_nan_policy(
    nan_policy: str,
    standardize_nan_fill: float | None,
) -> Literal["propagate", "zero_after_standardize"]:
    policies = {"propagate", "zero_after_standardize"}
    if nan_policy not in policies:
        raise ValueError(
            "nan_policy must be one of ['propagate', 'zero_after_standardize']"
        )
    if standardize_nan_fill is None:
        return "zero_after_standardize" if nan_policy == "zero_after_standardize" else "propagate"
    if isinstance(standardize_nan_fill, bool) or standardize_nan_fill != 0.0:
        raise ValueError("standardize_nan_fill must be None or exactly 0.0")
    return "zero_after_standardize"


def _standardization_state_columns(state: Mapping[str, object]) -> Sequence[object]:
    columns = state.get("columns", ())
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise ValueError("standardization state columns must be a sequence")
    return columns


def _fill_nonfinite_standardized_columns(
    panel: pd.DataFrame,
    columns: Sequence[object],
) -> pd.DataFrame:
    result = panel.copy()
    column_lookup = {str(column): column for column in result.columns}
    actual_columns = [column_lookup[str(column)] for column in columns]
    values = result.loc[:, actual_columns]
    finite = pd.DataFrame(
        np.isfinite(values.to_numpy(dtype=float)),
        index=values.index,
        columns=values.columns,
    )
    result.loc[:, actual_columns] = values.where(finite, 0.0)
    return result


def _normalize_standardize_ddof(value: int) -> int:
    out = int(value)
    if out < 0:
        raise ValueError("standardize_ddof must be non-negative")
    return out


def _standardize_step_ddof(steps: Any) -> int | None:
    if not isinstance(steps, (list, tuple)):
        return None
    for step in steps:
        if isinstance(step, Mapping) and step.get("step") == "standardize":
            ddof = step.get("ddof")
            return int(ddof) if ddof is not None else None
    return None


def _resolve_standardize_columns(
    panel: pd.DataFrame,
    base: _InputBundle,
    columns: str | Sequence[str],
) -> list[str]:
    if isinstance(columns, str):
        key = columns.lower()
        if key == "all":
            selected = [str(column) for column in panel.columns]
        elif key in {"predictors", "x"}:
            if base.predictors == "all":
                targets = set(base.targets or ((base.target,) if base.target else ()))
                selected = [str(column) for column in panel.columns if str(column) not in targets]
            else:
                selected = [str(column) for column in base.predictors]
        elif key in {"targets", "target", "y"}:
            selected = [str(column) for column in (base.targets or ((base.target,) if base.target else ()))]
        else:
            raise ValueError(
                "standardize_columns must be 'all', 'predictors', 'targets', or a sequence of column names"
            )
    else:
        selected = [str(column) for column in columns]
    if not selected:
        raise ValueError("standardize_columns selects no columns")
    missing = [column for column in selected if column not in panel.columns]
    if missing:
        raise ValueError(f"standardize_columns are not in the panel: {missing}")
    return selected


def _resolve_standardize_panel_columns(
    panel: pd.DataFrame,
    columns: str | Sequence[str],
) -> list[str]:
    if isinstance(columns, str):
        key = columns.lower()
        if key == "all":
            selected = [str(column) for column in panel.columns]
        elif key in {"predictors", "x", "targets", "target", "y"}:
            raise ValueError(
                "standardize_panel() needs explicit predictors/targets metadata "
                "for semantic standardize columns"
            )
        else:
            raise ValueError(
                "columns must be 'all' or a sequence of column names for "
                "standardize_panel()"
            )
    else:
        selected = [str(column) for column in columns]
    if not selected:
        raise ValueError("columns selects no columns")
    missing = [column for column in selected if column not in panel.columns]
    if missing:
        raise ValueError(f"columns are not in the panel: {missing}")
    return selected


def _target_column_names(
    *,
    target: str | None,
    targets: Sequence[str] | None,
) -> set[str]:
    names: set[str] = set()
    if target:
        names.add(str(target))
    if targets:
        names.update(str(column) for column in targets)
    return names


def _resolve_predictor_standardize_columns(
    panel: pd.DataFrame,
    *,
    predictors: str | Sequence[str],
    targets: set[str],
    columns: str | Sequence[str],
) -> list[str]:
    if isinstance(predictors, str):
        key = predictors.lower()
        if key != "all":
            raise ValueError("predictors must be 'all' or a sequence of column names")
        if not targets:
            raise ValueError(
                "origin_available_predictors standardization requires target "
                "metadata when predictors='all'"
            )
        predictor_pool = [str(column) for column in panel.columns if str(column) not in targets]
    else:
        predictor_pool = [str(column) for column in predictors]
    if not predictor_pool:
        raise ValueError("origin_available_predictors selects no predictor columns")
    target_predictors = sorted(set(predictor_pool) & targets)
    if target_predictors:
        raise ValueError(
            "origin_available_predictors cannot standardize target columns: "
            f"{target_predictors}"
        )
    if isinstance(columns, str):
        key = columns.lower()
        if key in {"all", "predictors", "x"}:
            selected = predictor_pool
        elif key in {"targets", "target", "y"}:
            raise ValueError("origin_available_predictors cannot standardize targets")
        else:
            raise ValueError(
                "columns must be 'all', 'predictors', or a sequence of predictor names"
            )
    else:
        selected = [str(column) for column in columns]
        target_columns = sorted(set(selected) & targets)
        if target_columns:
            raise ValueError(
                "origin_available_predictors cannot standardize target columns: "
                f"{target_columns}"
            )
        non_predictors = sorted(set(selected) - set(predictor_pool))
        if non_predictors:
            raise ValueError(
                "origin_available_predictors columns must be predictor columns: "
                f"{non_predictors}"
            )
    if not selected:
        raise ValueError("origin_available_predictors selects no predictor columns")
    missing = [column for column in selected if column not in panel.columns]
    if missing:
        raise ValueError(f"predictor columns are not in the panel: {missing}")
    return selected


def _normalize_frame(value: str) -> str:
    aliases = {
        "keep": "keep",
        "keep_unbalanced": "keep",
        "truncate": "truncate",
        "truncate_to_balanced": "truncate",
        "drop_series": "drop_unbalanced_series",
        "drop_unbalanced_series": "drop_unbalanced_series",
        "zero_fill": "zero_fill",
        "zero_fill_leading": "zero_fill",
    }
    return _lookup(value, aliases, "frame")


def _lookup(value: str, aliases: Mapping[str, str], name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"{name} must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]


def _is_fred_sd_metadata(metadata: Mapping[str, Any]) -> bool:
    return str(metadata.get("dataset", "")).lower() == "fred_sd"


def _fred_sd_column_parts(column: str) -> tuple[str, str]:
    if "_" not in column:
        return column, ""
    variable, state = column.rsplit("_", 1)
    return variable, state


def _validate_tcode(code: int, *, name: str) -> int:
    value = int(code)
    if value not in {1, 2, 3, 4, 5, 6, 7}:
        raise ValueError(f"t-code for {name!r} must be in 1..7; got {code!r}")
    return value


def _tcode_leading_loss(code: int) -> int:
    code = _validate_tcode(int(code), name="tcode_lag")
    if code in {1, 4}:
        return 0
    if code in {2, 5}:
        return 1
    if code in {3, 6, 7}:
        return 2
    raise ValueError(f"t-code must be in 1..7; got {code!r}")


def _build_transform_state(panel: pd.DataFrame, codes: Mapping[str, int]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for column, code in codes.items():
        if column not in panel.columns:
            continue
        observed = panel[column].dropna().tail(2)
        state[str(column)] = {
            "tcode": int(code),
            "requires_log_inverse": int(code) in {4, 5, 6},
            "lag_count": _tcode_leading_loss(int(code)),
            "last_observed_dates": [pd.Timestamp(index).strftime("%Y-%m-%d") for index in observed.index],
            "last_observed_values": [float(value) for value in observed.tolist()],
        }
    return state


def _coerce_custom_preprocess_output(
    output: Any,
    base_metadata: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(output, PreprocessedData):
        return output.panel.copy(), dict(output.metadata)
    if isinstance(output, DataBundle):
        return output.panel.copy(), dict(output.metadata)
    if isinstance(output, tuple) and len(output) == 2:
        panel, metadata = output
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("custom preprocessing tuple output must be (DataFrame, metadata)")
        return as_panel(panel, metadata=metadata), dict(metadata)
    if isinstance(output, pd.DataFrame):
        existing = dict(output.attrs.get("macroforecast_metadata", {}))
        merged = dict(base_metadata)
        merged.update(existing)
        return as_panel(output, metadata=merged), merged
    raise TypeError(
        "custom preprocessing callable must return a DataFrame, DataBundle, "
        "PreprocessedData, or (DataFrame, metadata)"
    )


def _callable_name(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if callable(value):
        marker = getattr(value, "__mf_digest__", None)
        if marker is not None:
            return {"callable": _callable_name(value), "mf_digest": str(marker)}
        return _callable_name(value)
    return value


__all__ = [
    "PreprocessedData",
    "PreprocessInput",
    "reprocess",
    "custom_preprocess",
    "plan",
    "report",
    "apply_transform_codes",
    "standardize_panel",
    "fred_sd_transform_codes",
    "handle_tcode_lag",
    "handle_outliers",
    "impute_missing",
    "handle_frame_edges",
]
