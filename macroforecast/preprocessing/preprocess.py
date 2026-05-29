from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any
import warnings

import pandas as pd

from macroforecast.preprocessing.clean import (
    apply_tcode_transform,
    drop_unbalanced_series_clean,
    em_factor_impute_clean,
    em_multivariate_impute_clean,
    forward_fill_clean,
    iqr_outlier_clean,
    linear_interpolate_clean,
    mean_impute_clean,
    truncate_to_balanced_clean,
    winsorize_clean,
    zero_fill_leading_clean,
    zscore_outlier_clean,
)
from macroforecast.data import (
    DataBundle,
    DataSpec,
    as_panel,
    attach_metadata,
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
    quarterly_to_monthly: str = "step_backward",
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
    frame: str = "keep",
) -> PreprocessedData:
    """Preprocess a canonical macroforecast panel.

    Parameters use user-facing names. Common legacy aliases such as
    ``apply_official_tcode`` and ``truncate_to_balanced`` are accepted, but
    returned metadata records the canonical direct-call names.
    """

    base = _coerce_input(data, metadata=metadata)
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
            "transform='custom' with expand_fred_sd_transform_codes(...)."
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


preprocess = reprocess


def apply_transform_codes(panel: pd.DataFrame, codes: Mapping[str, int]) -> pd.DataFrame:
    """Apply McCracken-Ng transform codes to matching panel columns."""

    if not codes:
        return panel.copy()
    return apply_tcode_transform(panel, dict(codes))


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


def expand_fred_sd_transform_codes(
    data: PreprocessInput,
    *,
    variable_codes: Mapping[str, int] | None = None,
    state_series_codes: Mapping[str, int] | None = None,
    use_national_analog_suggestions: bool = True,
    include_medium_confidence: bool = False,
    return_table: bool = False,
) -> dict[str, int] | tuple[dict[str, int], pd.DataFrame]:
    """Alias for :func:`fred_sd_transform_codes`."""

    return fred_sd_transform_codes(
        data,
        variable_codes=variable_codes,
        state_series_codes=state_series_codes,
        use_national_analog_suggestions=use_national_analog_suggestions,
        include_medium_confidence=include_medium_confidence,
        return_table=return_table,
    )


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
    frequency_map, frequency_source = _column_frequency_map(base.panel)
    frequency_issues = _frequency_hardening_issues(frequency_map)
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
            "frame": stage.get("frame"),
        },
        "transform_state": stage.get("transform_state", {}),
    }


def handle_mixed_frequency(
    panel: pd.DataFrame,
    *,
    method: str = "keep",
    quarterly_to_monthly: str = "step_backward",
    weekly_to_monthly: str = "mean",
    monthly_to_quarterly: str = "quarterly_average",
    weekly_to_quarterly: str = "mean",
) -> pd.DataFrame:
    """Keep, filter, or align a mixed-frequency panel."""

    validate_panel(panel)
    method = _normalize_frequency(method)
    if method == "keep":
        return panel.copy()

    frequencies, _frequency_source = _column_frequency_map(panel)
    _warn_frequency_hardening_issues(frequencies)
    if method in {"drop_non_monthly", "drop_non_quarterly"}:
        target_frequency = "monthly" if method == "drop_non_monthly" else "quarterly"
        columns = [column for column, frequency in frequencies.items() if frequency == target_frequency]
        if not columns:
            raise ValueError(f"frequency={method!r} leaves no columns")
        result = panel[columns].copy()
        result.attrs.update(dict(getattr(panel, "attrs", {}) or {}))
        return result

    if method == "monthly":
        result = _align_to_monthly(
            panel,
            frequencies=frequencies,
            quarterly_to_monthly=quarterly_to_monthly,
            weekly_to_monthly=weekly_to_monthly,
        )
    elif method == "quarterly":
        result = _align_to_quarterly(
            panel,
            frequencies=frequencies,
            monthly_to_quarterly=monthly_to_quarterly,
            weekly_to_quarterly=weekly_to_quarterly,
        )
    else:  # pragma: no cover - guarded by _normalize_frequency
        raise ValueError(f"unknown frequency method {method!r}")

    result.attrs.update(dict(getattr(panel, "attrs", {}) or {}))
    return result


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
    _frequencies, frequency_source = _column_frequency_map(panel)
    result = handle_mixed_frequency(
        panel,
        method=method,
        quarterly_to_monthly=quarterly_to_monthly,
        weekly_to_monthly=weekly_to_monthly,
        monthly_to_quarterly=monthly_to_quarterly,
        weekly_to_quarterly=weekly_to_quarterly,
    )
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
    if metadata.get("dataset") and metadata.get("source_family"):
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
        return ("transform", "tcode_lag", "frequency", "outliers", "impute", "frame")
    return ("frequency", "transform", "tcode_lag", "outliers", "impute", "frame")


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
        "forward_fill": "forward_fill",
        "ffill": "forward_fill",
        "linear": "linear",
        "linear_interpolation": "linear",
        "em_factor": "em_factor",
        "em_multivariate": "em_multivariate",
    }
    return _lookup(value, aliases, "impute")


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
    return str(metadata.get("dataset", "")).lower() == "fred_sd" or str(metadata.get("source_family", "")).lower() == "fred-sd"


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


def _column_frequency_map(panel: pd.DataFrame) -> tuple[dict[str, str], str]:
    metadata_map = _frequency_map_from_metadata(panel)
    if metadata_map:
        return (
            {
                str(column): metadata_map.get(str(column), _infer_column_frequency(panel[column]))
                for column in panel.columns
            },
            "fred_sd_series_metadata",
        )
    return ({str(column): _infer_column_frequency(panel[column]) for column in panel.columns}, "observed_dates")


def _frequency_hardening_issues(frequencies: Mapping[str, str]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for frequency in ("unknown", "irregular", "annual"):
        columns = sorted(column for column, value in frequencies.items() if value == frequency)
        if columns:
            issues.append({"frequency": frequency, "columns": columns, "n_columns": len(columns)})
    return issues


def _warn_frequency_hardening_issues(frequencies: Mapping[str, str]) -> None:
    issues = _frequency_hardening_issues(frequencies)
    for issue in issues:
        if issue["frequency"] == "irregular":
            continue
        sample = ", ".join(issue["columns"][:5])
        if issue["n_columns"] > 5:
            sample = f"{sample}, ..."
        warnings.warn(
            "frequency inference found "
            f"{issue['frequency']} columns before alignment: {sample}. "
            "Use data metadata when the source frequency is known.",
            UserWarning,
            stacklevel=3,
        )


def _frequency_map_from_metadata(panel: pd.DataFrame) -> dict[str, str]:
    reports = getattr(panel, "attrs", {}).get("macrocast_reports", {})
    if not isinstance(reports, Mapping):
        return {}
    report = reports.get("fred_sd_series_metadata", {})
    if not isinstance(report, Mapping):
        return {}
    rows = report.get("series", ())
    if not isinstance(rows, (list, tuple)):
        return {}
    frequencies: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        column = row.get("column")
        frequency = row.get("native_frequency")
        if column is not None and frequency:
            frequencies[str(column)] = str(frequency)
    return frequencies


def _infer_column_frequency(series: pd.Series) -> str:
    observed = series.dropna()
    if observed.shape[0] < 2:
        return "unknown"
    index = pd.DatetimeIndex(observed.index).sort_values()
    day_deltas = [(right - left).days for left, right in zip(index[:-1], index[1:]) if right > left]
    if not day_deltas:
        return "unknown"
    median_days = float(pd.Series(day_deltas).median())
    if 5 <= median_days <= 10:
        return "weekly"
    if 25 <= median_days <= 35:
        return "monthly"
    if 80 <= median_days <= 100:
        return "quarterly"
    if 350 <= median_days <= 380:
        return "annual"
    return "irregular"


def _monthly_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    start = pd.DatetimeIndex(panel.index).min().to_period("M").to_timestamp()
    end = pd.DatetimeIndex(panel.index).max().to_period("M").to_timestamp()
    return pd.date_range(start, end, freq="MS", name="date")


def _quarterly_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    start = pd.DatetimeIndex(panel.index).min().to_period("Q").start_time
    end = pd.DatetimeIndex(panel.index).max().to_period("Q").start_time
    return pd.date_range(start, end, freq="QS", name="date")


def _align_to_monthly(
    panel: pd.DataFrame,
    *,
    frequencies: Mapping[str, str],
    quarterly_to_monthly: str,
    weekly_to_monthly: str,
) -> pd.DataFrame:
    index = _monthly_index(panel)
    columns: dict[str, pd.Series] = {}
    for column in panel.columns:
        name = str(column)
        series = panel[column].dropna()
        frequency = frequencies.get(name, "unknown")
        if frequency == "weekly":
            aligned = _aggregate_to_monthly(series, weekly_to_monthly)
        elif frequency == "quarterly":
            aligned = _quarterly_to_monthly(series, quarterly_to_monthly, index=index)
        else:
            aligned = series.resample("MS").last()
        columns[name] = aligned.reindex(index)
    return pd.DataFrame(columns, index=index)


def _align_to_quarterly(
    panel: pd.DataFrame,
    *,
    frequencies: Mapping[str, str],
    monthly_to_quarterly: str,
    weekly_to_quarterly: str,
) -> pd.DataFrame:
    index = _quarterly_index(panel)
    columns: dict[str, pd.Series] = {}
    for column in panel.columns:
        name = str(column)
        series = panel[column].dropna()
        frequency = frequencies.get(name, "unknown")
        if frequency == "weekly":
            aligned = _aggregate_to_quarterly(series, weekly_to_quarterly)
        elif frequency == "monthly":
            aligned = _aggregate_to_quarterly(series, monthly_to_quarterly)
        else:
            aligned = series.resample("QS").last()
        columns[name] = aligned.reindex(index)
    return pd.DataFrame(columns, index=index)


def _quarterly_to_monthly(series: pd.Series, rule: str, *, index: pd.DatetimeIndex) -> pd.Series:
    key = rule.lower()
    quarterly = series.copy()
    quarterly.index = pd.DatetimeIndex(quarterly.index).to_period("Q").to_timestamp()
    quarterly = quarterly.groupby(level=0).last().sort_index()
    if key in {"step_backward", "repeat_within_quarter", "repeat", "spread"}:
        by_quarter = dict(zip(pd.DatetimeIndex(quarterly.index).to_period("Q"), quarterly.to_numpy(), strict=False))
        values = [by_quarter.get(period) for period in pd.DatetimeIndex(index).to_period("Q")]
        return pd.Series(values, index=index, dtype="float64")
    if key in {"step_forward", "quarter_end_ffill", "ffill_from_quarter_end"}:
        observed = quarterly.copy()
        observed.index = pd.DatetimeIndex(observed.index).to_period("Q").asfreq("M", how="end").to_timestamp()
        return observed.reindex(index).ffill()
    if key in {"linear_interpolation", "linear"}:
        observed = quarterly.copy()
        observed.index = pd.DatetimeIndex(observed.index).to_period("Q").asfreq("M", how="end").to_timestamp()
        return observed.reindex(index.union(observed.index)).sort_index().interpolate(method="time").reindex(index)
    raise ValueError(
        "quarterly_to_monthly must be one of "
        "['step_backward', 'repeat_within_quarter', 'step_forward', 'quarter_end_ffill', 'linear_interpolation']"
    )


def _aggregate_to_monthly(series: pd.Series, rule: str) -> pd.Series:
    return _aggregate_resample(series, rule, frequency="MS", aliases={"last": "last", "endpoint": "last", "mean": "mean", "average": "mean", "sum": "sum"})


def _aggregate_to_quarterly(series: pd.Series, rule: str) -> pd.Series:
    aliases = {
        "quarterly_average": "mean",
        "average": "mean",
        "mean": "mean",
        "quarterly_endpoint": "last",
        "endpoint": "last",
        "last": "last",
        "quarterly_sum": "sum",
        "sum": "sum",
    }
    return _aggregate_resample(series, rule, frequency="QS", aliases=aliases)


def _aggregate_resample(series: pd.Series, rule: str, *, frequency: str, aliases: Mapping[str, str]) -> pd.Series:
    key = rule.lower()
    if key not in aliases:
        raise ValueError(f"aggregation rule must be one of {sorted(aliases)}; got {rule!r}")
    method = aliases[key]
    if method == "mean":
        return series.resample(frequency).mean()
    if method == "last":
        return series.resample(frequency).last()
    if method == "sum":
        return series.resample(frequency).sum(min_count=1)
    raise ValueError(f"unknown aggregation method {method!r}")



__all__ = [
    "PreprocessedData",
    "PreprocessInput",
    "reprocess",
    "preprocess",
    "plan",
    "report",
    "apply_transform_codes",
    "fred_sd_transform_codes",
    "expand_fred_sd_transform_codes",
    "handle_mixed_frequency",
    "handle_tcode_lag",
    "handle_outliers",
    "impute_missing",
    "handle_frame_edges",
]
