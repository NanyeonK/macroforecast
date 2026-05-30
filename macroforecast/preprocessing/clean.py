from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd


def iqr_outlier_clean(
    panel: pd.DataFrame,
    *,
    threshold: float = 10.0,
    action: str = "flag_as_nan",
) -> pd.DataFrame:
    """Flag or replace outliers with a per-column IQR rule."""

    _require_non_empty(panel)
    _validate_positive(threshold, "threshold")
    action = _normalize_outlier_action(action)
    result = panel.copy()
    numeric = result.select_dtypes("number")
    if numeric.empty:
        return result
    median = numeric.median()
    iqr = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, np.nan)
    mask = ((numeric - median).abs() > threshold * iqr).fillna(False)
    result[numeric.columns] = _apply_outlier_action(numeric, mask, action)
    return _copy_attrs(panel, result)


def zscore_outlier_clean(
    panel: pd.DataFrame,
    *,
    threshold: float = 3.0,
    action: str = "flag_as_nan",
) -> pd.DataFrame:
    """Flag or replace outliers with a per-column z-score rule."""

    _require_non_empty(panel)
    _validate_positive(threshold, "threshold")
    action = _normalize_outlier_action(action)
    result = panel.copy()
    numeric = result.select_dtypes("number")
    if numeric.empty:
        return result
    std = numeric.std(ddof=0).replace(0, np.nan)
    mask = (((numeric - numeric.mean()) / std).abs() > threshold).fillna(False)
    result[numeric.columns] = _apply_outlier_action(numeric, mask, action)
    return _copy_attrs(panel, result)


def winsorize_clean(
    panel: pd.DataFrame,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Clip numeric columns to quantile bounds."""

    _require_non_empty(panel)
    if not (0 <= lower_quantile < upper_quantile <= 1):
        raise ValueError("need 0 <= lower_quantile < upper_quantile <= 1")
    result = panel.copy()
    numeric = result.select_dtypes("number")
    if numeric.empty:
        return result
    result[numeric.columns] = numeric.clip(
        lower=numeric.quantile(lower_quantile),
        upper=numeric.quantile(upper_quantile),
        axis=1,
    )
    return _copy_attrs(panel, result)


def em_factor_impute_clean(
    panel: pd.DataFrame,
    *,
    n_factors: int = 8,
    max_iter: int = 50,
    tol: float = 1e-6,
    factor_selection: str = "baing_p2",
    demean: int = 2,
) -> pd.DataFrame:
    """Impute missing numeric cells with PCA-EM factor reconstruction."""

    _require_non_empty(panel)
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    _validate_positive(tol, "tol")
    if demean not in {0, 1, 2, 3}:
        raise ValueError("demean must be one of 0, 1, 2, 3")
    selection = factor_selection.lower()
    if selection in {"fixed", "fixed_rank"}:
        return _pca_em_imputation(panel, n_factors=n_factors, max_iter=max_iter, tol=tol)
    jj = _factor_selection_to_jj(selection)
    return _fred_md_em_factor_impute(panel, kmax=n_factors, jj=jj, demean=demean, max_iter=max_iter, tol=tol)


def em_multivariate_impute_clean(
    panel: pd.DataFrame,
    *,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> pd.DataFrame:
    """Impute missing numeric cells with an uncapped PCA-EM rank rule."""

    _require_non_empty(panel)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    _validate_positive(tol, "tol")
    return _pca_em_imputation(panel, n_factors=None, max_iter=max_iter, tol=tol)


def mean_impute_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace missing numeric cells with full-column means."""

    _require_non_empty(panel)
    return _copy_attrs(panel, panel.fillna(panel.mean(numeric_only=True)))


def forward_fill_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Carry each series' most recent observed value forward."""

    _require_non_empty(panel)
    return _copy_attrs(panel, panel.ffill())


def linear_interpolate_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Fill interior missing values by linear interpolation."""

    _require_non_empty(panel)
    # Fill only gaps bracketed by observed data. Leading/trailing NaNs often
    # encode a series' publication start/end, so extrapolating them would create
    # observations that were not available in the source panel.
    return _copy_attrs(panel, panel.interpolate(method="linear", limit_area="inside"))


def truncate_to_balanced_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with no missing values."""

    _require_non_empty(panel)
    return _copy_attrs(panel, panel.dropna(axis=0, how="any"))


def drop_unbalanced_series_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns with no missing values."""

    _require_non_empty(panel)
    return _copy_attrs(panel, panel.dropna(axis=1, how="any"))


def zero_fill_leading_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace remaining missing cells with zero."""

    _require_non_empty(panel)
    return _copy_attrs(panel, panel.fillna(0))


def fit_standardization_state(
    panel: pd.DataFrame,
    *,
    method: str = "zscore",
    ddof: int = 0,
) -> dict[str, object]:
    """Fit column-wise scaling parameters on a numeric panel."""

    _require_non_empty(panel)
    method_value = _normalize_standardization_method(method)
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    numeric = panel.select_dtypes("number")
    if numeric.empty:
        raise ValueError("standardization requires at least one numeric column")
    if method_value == "zscore":
        center = numeric.mean(axis=0)
        scale = numeric.std(axis=0, ddof=ddof)
    elif method_value == "robust":
        center = numeric.median(axis=0)
        scale = numeric.quantile(0.75, axis=0) - numeric.quantile(0.25, axis=0)
    else:
        center = numeric.min(axis=0)
        scale = numeric.max(axis=0) - numeric.min(axis=0)
    scale = scale.replace(0.0, np.nan)
    if scale.isna().any():
        bad = sorted(str(column) for column in scale.index[scale.isna()])
        raise ValueError(f"standardization has zero or invalid scale for columns: {bad}")
    return {
        "method": method_value,
        "ddof": int(ddof),
        "columns": [str(column) for column in numeric.columns],
        "center": {str(column): float(value) for column, value in center.items()},
        "scale": {str(column): float(value) for column, value in scale.items()},
    }


def apply_standardization_state(panel: pd.DataFrame, state: Mapping[str, object]) -> pd.DataFrame:
    """Apply fitted column-wise scaling parameters to a panel."""

    _require_non_empty(panel)
    raw_columns = state.get("columns", ())
    if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
        raise ValueError("standardization state columns must be a sequence")
    columns = [str(column) for column in raw_columns]
    if not columns:
        raise ValueError("standardization state has no columns")
    column_lookup = {str(column): column for column in panel.columns}
    missing = [column for column in columns if column not in column_lookup]
    if missing:
        raise ValueError(f"standardization columns are not in the panel: {missing}")
    raw_center = state.get("center", {})
    raw_scale = state.get("scale", {})
    if not isinstance(raw_center, Mapping) or not isinstance(raw_scale, Mapping):
        raise ValueError("standardization state center and scale must be mappings")
    center_map = cast(Mapping[str, Any], raw_center)
    scale_map = cast(Mapping[str, Any], raw_scale)
    missing_center = [column for column in columns if column not in center_map]
    missing_scale = [column for column in columns if column not in scale_map]
    if missing_center or missing_scale:
        raise ValueError(
            "standardization state is missing center/scale entries for columns: "
            f"{sorted(set(missing_center + missing_scale))}"
        )
    actual_columns = [column_lookup[column] for column in columns]
    center = pd.Series(
        {column_lookup[column]: float(center_map[column]) for column in columns}
    )
    scale = pd.Series(
        {column_lookup[column]: float(scale_map[column]) for column in columns}
    )
    result = panel.copy()
    result.loc[:, actual_columns] = (result.loc[:, actual_columns].astype(float) - center) / scale
    return _copy_attrs(panel, result)


def standardize_clean(
    panel: pd.DataFrame,
    *,
    method: str = "zscore",
    ddof: int = 0,
) -> pd.DataFrame:
    """Standardize numeric columns with full-sample fitted parameters."""

    state = fit_standardization_state(panel, method=method, ddof=ddof)
    return apply_standardization_state(panel, state)


def apply_tcode_transform(panel: pd.DataFrame, tcode_map: Mapping[str, int]) -> pd.DataFrame:
    """Apply McCracken-Ng transformation codes to matching columns."""

    _require_non_empty(panel)
    if not tcode_map:
        raise ValueError("tcode_map must not be empty")
    result = panel.copy()
    for column, code in tcode_map.items():
        name = str(column)
        if name not in result.columns:
            continue
        result[name] = _apply_tcode(result[name], int(code))
    return _copy_attrs(panel, result)


def freq_align_quarterly_to_monthly_clean(
    panel: pd.DataFrame,
    quarterly_columns: Sequence[str],
    *,
    rule: str = "step_backward",
) -> pd.DataFrame:
    """Align selected quarterly columns on the panel's monthly grid."""

    _require_datetime_panel(panel)
    aliases = {
        "step_backward": "step_backward",
        "repeat_within_quarter": "step_backward",
        "step_forward": "step_forward",
        "quarter_end_ffill": "step_forward",
        "linear_interpolation": "linear",
        "linear": "linear",
    }
    method = _lookup(rule, aliases, "rule")
    result = panel.copy()
    for column in quarterly_columns:
        if column not in result.columns:
            continue
        series = result[column]
        if method == "step_backward":
            result[column] = series.bfill().ffill()
        elif method == "step_forward":
            result[column] = series.ffill()
        else:
            result[column] = series.interpolate(method="linear", limit_direction="both")
    return _copy_attrs(panel, result)


def freq_align_monthly_to_quarterly_clean(
    panel: pd.DataFrame,
    monthly_columns: Sequence[str],
    *,
    rule: str = "quarterly_average",
) -> pd.DataFrame:
    """Aggregate selected monthly columns to a quarterly grid."""

    _require_datetime_panel(panel)
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
    method = _lookup(rule, aliases, "rule")
    result_parts: list[pd.Series] = []
    quarterly_index = pd.DatetimeIndex(panel.index).to_period("Q").to_timestamp()
    for column in panel.columns:
        series = panel[column].copy()
        if column not in monthly_columns:
            collapsed = series.groupby(quarterly_index).last()
        elif method == "mean":
            collapsed = series.groupby(quarterly_index).mean()
        elif method == "last":
            collapsed = series.groupby(quarterly_index).last()
        else:
            collapsed = series.groupby(quarterly_index).sum(min_count=1)
        collapsed.name = column
        result_parts.append(collapsed)
    result = pd.concat(result_parts, axis=1)
    result.index.name = panel.index.name
    return _copy_attrs(panel, result)


def _require_non_empty(panel: pd.DataFrame, *, name: str = "panel") -> None:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if panel.empty:
        raise ValueError(f"{name} must not be empty; got shape {panel.shape}")


def _require_datetime_panel(panel: pd.DataFrame) -> None:
    _require_non_empty(panel)
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("frequency alignment requires a DatetimeIndex")


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _copy_attrs(source: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    result.attrs.update(dict(getattr(source, "attrs", {}) or {}))
    return result


def _normalize_outlier_action(action: str) -> str:
    aliases = {
        "flag_as_nan": "flag_as_nan",
        "nan": "flag_as_nan",
        "replace_with_median": "replace_with_median",
        "median": "replace_with_median",
        "replace_with_cap_value": "replace_with_cap_value",
        "cap": "replace_with_cap_value",
        "clip": "replace_with_cap_value",
    }
    return str(_lookup(action, aliases, "action"))


def _normalize_standardization_method(method: str) -> str:
    aliases = {
        "zscore": "zscore",
        "standard": "zscore",
        "standardize": "zscore",
        "standardized": "zscore",
        "robust": "robust",
        "iqr": "robust",
        "minmax": "minmax",
        "min_max": "minmax",
    }
    return str(_lookup(method, aliases, "method"))


def _apply_outlier_action(numeric: pd.DataFrame, mask: pd.DataFrame, action: str) -> pd.DataFrame:
    if action == "flag_as_nan":
        return numeric.mask(mask)
    if action == "replace_with_median":
        return numeric.mask(mask, numeric.median(), axis=1)
    lower = numeric.quantile(0.01)
    upper = numeric.quantile(0.99)
    capped = numeric.clip(lower=lower, upper=upper, axis=1)
    return numeric.where(~mask, capped)


def _pca_em_imputation(
    panel: pd.DataFrame,
    *,
    n_factors: int | None,
    max_iter: int,
    tol: float,
) -> pd.DataFrame:
    numeric = panel.select_dtypes("number")
    if numeric.empty:
        return panel.copy()
    matrix = numeric.to_numpy(dtype=float)
    missing = np.isnan(matrix)
    if not missing.any():
        return panel.copy()
    if (missing.sum(axis=1) == matrix.shape[1]).any():
        # A fully missing date has no cross-sectional signal. The FRED-MD-style
        # EM path rejects it, and this generic PCA-EM helper follows the same
        # convention so direct callable behavior is not looser than reprocess().
        raise ValueError("PCA-EM cannot process an all-missing row")
    if (missing.sum(axis=0) == matrix.shape[0]).any():
        raise ValueError("PCA-EM cannot process an all-missing column")
    means = np.nanmean(matrix, axis=0)
    filled = matrix.copy()
    filled[missing] = np.take(means, np.where(missing)[1])
    rank = n_factors if n_factors is not None else max(1, min(filled.shape) // 2)
    rank = max(1, min(int(rank), min(filled.shape)))
    previous = filled.copy()
    for _ in range(max_iter):
        center = filled.mean(axis=0, keepdims=True)
        scale = filled.std(axis=0, ddof=0, keepdims=True)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        standardized = (filled - center) / scale
        u, s, vt = np.linalg.svd(standardized, full_matrices=False)
        k = min(rank, len(s))
        reconstructed = (u[:, :k] * s[:k]) @ vt[:k, :]
        predicted = reconstructed * scale + center
        filled[missing] = predicted[missing]
        denom = max(float(np.sum(previous * previous)), 1e-12)
        error = float(np.sum((filled - previous) ** 2) / denom)
        previous = filled.copy()
        if error <= tol:
            break
    result = panel.copy()
    result[numeric.columns] = filled
    return _copy_attrs(panel, result)


def _fred_md_em_factor_impute(
    panel: pd.DataFrame,
    *,
    kmax: int,
    jj: int,
    demean: int,
    max_iter: int,
    tol: float,
) -> pd.DataFrame:
    numeric = panel.select_dtypes("number")
    if numeric.empty:
        return panel.copy()
    matrix = numeric.to_numpy(dtype=float)
    missing = np.isnan(matrix)
    if not missing.any():
        return panel.copy()
    if (missing.sum(axis=1) == matrix.shape[1]).any():
        raise ValueError("em_factor cannot process an all-missing row")
    if (missing.sum(axis=0) == matrix.shape[0]).any():
        raise ValueError("em_factor cannot process an all-missing column")
    means = np.nanmean(matrix, axis=0)
    x2 = matrix.copy()
    x2[missing] = np.take(means, np.where(missing)[1])
    x3, center, scale = _em_transform_data(x2, demean)
    factor_count = _baing_factor_count(x3, kmax=kmax, jj=jj)
    chat = _pc2_chat(x3, factor_count)
    previous = chat.copy()
    for _ in range(max_iter):
        predicted = chat * scale + center
        x2 = matrix.copy()
        x2[missing] = predicted[missing]
        x3, center, scale = _em_transform_data(x2, demean)
        factor_count = _baing_factor_count(x3, kmax=kmax, jj=jj)
        chat = _pc2_chat(x3, factor_count)
        denom = max(float(np.sum(previous * previous)), 1e-12)
        error = float(np.sum((chat - previous) ** 2) / denom)
        previous = chat.copy()
        if error <= tol:
            break
    result = panel.copy()
    result[numeric.columns] = x2
    return _copy_attrs(panel, result)


def _factor_selection_to_jj(value: str) -> int:
    aliases = {
        "baing_p1": 1,
        "pc_p1": 1,
        "p1": 1,
        "baing_p2": 2,
        "pc_p2": 2,
        "p2": 2,
        "fred_md": 2,
        "mccracken_ng_2016": 2,
        "baing_p3": 3,
        "pc_p3": 3,
        "p3": 3,
    }
    return int(_lookup(value, aliases, "factor_selection"))


def _em_transform_data(matrix: np.ndarray, demean: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = matrix.shape
    if demean == 0:
        center = np.zeros((rows, cols))
        scale = np.ones((rows, cols))
        transformed = matrix.copy()
    elif demean == 1:
        center = np.tile(matrix.mean(axis=0), (rows, 1))
        scale = np.ones((rows, cols))
        transformed = matrix - center
    elif demean == 2:
        center = np.tile(matrix.mean(axis=0), (rows, 1))
        scale_values = matrix.std(axis=0, ddof=1)
        if np.any(~np.isfinite(scale_values)) or np.any(scale_values == 0):
            raise ValueError("em_factor requires finite non-zero column standard deviations")
        scale = np.tile(scale_values, (rows, 1))
        transformed = (matrix - center) / scale
    elif demean == 3:
        center = np.vstack([matrix[: row + 1].mean(axis=0) for row in range(rows)])
        scale_values = matrix.std(axis=0, ddof=1)
        if np.any(~np.isfinite(scale_values)) or np.any(scale_values == 0):
            raise ValueError("em_factor requires finite non-zero column standard deviations")
        scale = np.tile(scale_values, (rows, 1))
        transformed = (matrix - center) / scale
    else:
        raise ValueError("demean must be one of 0, 1, 2, 3")
    return transformed, center, scale


def _baing_factor_count(matrix: np.ndarray, *, kmax: int, jj: int) -> int:
    rows, cols = matrix.shape
    kmax_eff = max(1, min(int(kmax), rows, cols))
    total = rows * cols
    total_margin = rows + cols
    factors = np.arange(1, kmax_eff + 1)
    if jj == 1:
        penalty = np.log(total / total_margin) * factors * total_margin / total
    elif jj == 2:
        penalty = (total_margin / total) * np.log(min(rows, cols)) * factors
    elif jj == 3:
        gct = min(rows, cols)
        penalty = factors * np.log(gct) / gct
    else:
        raise ValueError("jj must be one of 1, 2, 3")
    fhat0, lambda0 = _pc_components(matrix)
    values: list[float] = []
    for count, term in zip(factors, penalty, strict=True):
        fhat = fhat0[:, :count]
        loadings = lambda0[:, :count]
        residual = matrix - fhat @ loadings.T
        sigma = float(np.sum(residual * residual) / total)
        values.append(np.log(max(sigma, 1e-300)) + float(term))
    sigma_zero = float(np.sum(matrix * matrix) / total)
    values.append(np.log(max(sigma_zero, 1e-300)))
    selected = int(np.argmin(values)) + 1
    return selected if selected <= kmax_eff else 0


def _pc_components(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = matrix.shape
    if rows < cols:
        u, _, _ = np.linalg.svd(matrix @ matrix.T, full_matrices=False)
        fhat0 = np.sqrt(rows) * u
        lambda0 = matrix.T @ fhat0 / rows
    else:
        u, _, _ = np.linalg.svd(matrix.T @ matrix, full_matrices=False)
        lambda0 = np.sqrt(cols) * u
        fhat0 = matrix @ lambda0 / cols
    return fhat0, lambda0


def _pc2_chat(matrix: np.ndarray, n_factors: int) -> np.ndarray:
    if n_factors <= 0:
        return np.zeros_like(matrix)
    cols = matrix.shape[1]
    u, _, _ = np.linalg.svd(matrix.T @ matrix, full_matrices=False)
    n_factors = min(n_factors, u.shape[1])
    loadings = u[:, :n_factors] * np.sqrt(cols)
    factors = matrix @ loadings / cols
    return factors @ loadings.T


def _apply_tcode(series: pd.Series, code: int) -> pd.Series:
    if code not in {1, 2, 3, 4, 5, 6, 7}:
        raise ValueError(f"t-code must be in 1..7; got {code!r}")
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 4:
        return _safe_log(x)
    if code == 5:
        return _safe_log(x).diff()
    if code == 6:
        return _safe_log(x).diff().diff()
    return x.pct_change(fill_method=None).diff()


def _safe_log(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index, name=series.name, dtype="float64")
    if float(series.min(skipna=True)) < 1e-6:
        return pd.Series(np.nan, index=series.index, name=series.name, dtype="float64")
    return np.log(series)


def _lookup(value: str, aliases: Mapping[str, str | int], name: str) -> str | int:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"{name} must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]


__all__ = [
    "iqr_outlier_clean",
    "zscore_outlier_clean",
    "winsorize_clean",
    "em_factor_impute_clean",
    "em_multivariate_impute_clean",
    "mean_impute_clean",
    "forward_fill_clean",
    "linear_interpolate_clean",
    "truncate_to_balanced_clean",
    "drop_unbalanced_series_clean",
    "zero_fill_leading_clean",
    "fit_standardization_state",
    "apply_standardization_state",
    "standardize_clean",
    "apply_tcode_transform",
    "freq_align_quarterly_to_monthly_clean",
    "freq_align_monthly_to_quarterly_clean",
]
