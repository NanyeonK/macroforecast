from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from macroforecast.data import attach_metadata, validate_panel
from macroforecast.feature_engineering.types import FeatureInput
from macroforecast.feature_engineering.shared import (
    FitPolicy,
    _coerce_input,
    _feature_matrix_records,
    _metadata_frame,
    _normalize_feature_matrix_specification,
    _normalize_fit_policy,
    _normalize_lags,
    _normalize_min_train_size,
    _prefix_columns,
    _prepend_zero_lag,
    _resolve_columns,
    _warn_if_full_sample_fit,
    _zscore_frame,
)
from macroforecast.feature_engineering.compose import pca_then_lags
from macroforecast.feature_engineering.transforms import lag, maf_features, moving_average_ladder

def feature_matrix(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    specification: str | Iterable[str] = "X",
    columns: Iterable[str] | None = None,
    level_data: FeatureInput | None = None,
    level_columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (0,),
    max_lag: int = 12,
    n_factors: int = 8,
    n_maf_components: int = 2,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    include_current_factor: bool = True,
    scale_factors: bool = True,
    scale_marx: bool = False,
    scale_maf: bool = False,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Build named macro-ML feature-matrix combinations.

    ``specification`` accepts combinations such as ``"X"``, ``"F"``,
    ``"MARX"``, ``"MAF"``, ``"F-X-MARX"``, or ``("F", "X", "MAF")``.
    ``X`` is the supplied panel, usually after preprocessing. ``LEVEL`` needs
    an explicit ``level_data`` panel because levels and stationarized ``X`` are
    different data objects in a clean pandas workflow.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    blocks = _normalize_feature_matrix_specification(specification)
    lag_values = _normalize_lags(lags, allow_zero=True)
    fit_value = _normalize_fit_policy(fit_policy)
    max_lag_value = int(max_lag)
    if max_lag_value < 0:
        raise ValueError("max_lag must be non-negative")
    if "MARX" in blocks and max_lag_value <= 0:
        raise ValueError("max_lag must be positive when MARX is requested")
    factor_lag_values = _prepend_zero_lag(lag_values) if include_current_factor else lag_values
    fitted_blocks = [
        block
        for block in blocks
        if block in {"F", "MAF"} or (block == "MARX" and scale_marx)
    ]
    if fitted_blocks:
        _warn_if_full_sample_fit(
            fit_value,
            context=f"feature_matrix(..., specification='{'-'.join(blocks)}')",
            enabled=warn_full_sample,
        )

    parts: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    block_summaries: list[dict[str, Any]] = []
    for block in blocks:
        if block == "X":
            frame = lag(panel, metadata=base.metadata, columns=selected, lags=lag_values)
        elif block == "F":
            frame = pca_then_lags(
                panel,
                metadata=base.metadata,
                columns=selected,
                n_components=n_factors,
                lags=factor_lag_values,
                fit_policy=fit_value,
                min_train_size=min_train_size,
                scale=scale_factors,
                prefix="F",
                include_pca=False,
                warn_full_sample=False,
            )
        elif block == "MARX":
            frame = _marx_ladder(
                panel,
                columns=selected,
                max_lag=max_lag_value,
                scale_lags=scale_marx,
                fit_policy=fit_value,
                min_train_size=min_train_size,
                warn_full_sample=False,
            )
        elif block == "MAF":
            frame = maf_features(
                panel,
                metadata=base.metadata,
                columns=selected,
                max_lag=max_lag_value,
                n_components=n_maf_components,
                fit_policy=fit_value,
                min_train_size=min_train_size,
                scale=scale_maf,
                prefix="maf",
                warn_full_sample=False,
            )
        elif block == "LEVEL":
            if level_data is None:
                raise ValueError("specification includes LEVEL, so level_data is required")
            level_base = _coerce_input(level_data)
            level_selected = _resolve_columns(
                level_base.panel,
                columns=level_columns if level_columns is not None else selected,
            )
            frame = lag(level_base.panel, metadata=level_base.metadata, columns=level_selected, lags=lag_values)
        else:
            raise ValueError(f"unsupported feature matrix block {block!r}")

        prefixed = _prefix_columns(frame, block)
        parts.append(prefixed)
        records.extend(
            _feature_matrix_records(
                prefixed,
                block=block,
                source_columns=level_selected if block == "LEVEL" else selected,
                fit_policy=fit_value if block in {"F", "MAF"} else None,
                scale_marx=scale_marx if block == "MARX" else None,
            )
        )
        block_summaries.append(
            {
                "block": block,
                "n_features": int(prefixed.shape[1]),
                "columns": [str(column) for column in prefixed.columns],
            }
        )

    result = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=panel.index)
    if result.columns.has_duplicates:
        duplicate_columns = result.columns[result.columns.duplicated()].unique()
        raise ValueError(f"feature matrix contains duplicate columns: {list(map(str, duplicate_columns))}")
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    updated_metadata = attach_metadata(
        base.metadata,
        "feature_engineering_feature_matrix",
        {
            "specification": "-".join(blocks),
            "blocks": block_summaries,
            "columns": list(selected),
            "lags": list(lag_values),
            "max_lag": max_lag_value,
            "n_factors": int(n_factors),
            "n_maf_components": int(n_maf_components),
            "fit_policy": fit_value,
            "min_train_size": min_train_size,
            "include_current_factor": bool(include_current_factor),
            "factor_lags": list(factor_lag_values) if "F" in blocks else None,
            "scale_factors": bool(scale_factors),
            "scale_marx": bool(scale_marx),
            "scale_maf": bool(scale_maf),
            "drop_missing": bool(drop_missing),
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_metadata"] = updated_metadata
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(records)
    return result


def _marx_ladder(
    panel: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    max_lag: int,
    scale_lags: bool,
    fit_policy: FitPolicy,
    min_train_size: int | None,
    warn_full_sample: bool,
) -> pd.DataFrame:
    if not scale_lags:
        return moving_average_ladder(
            panel,
            columns=columns,
            windows=range(1, int(max_lag) + 1),
            shift=1,
        )

    fit_value = _normalize_fit_policy(fit_policy)
    _warn_if_full_sample_fit(
        fit_value,
        context="feature_matrix(..., specification='MARX', scale_marx=True)",
        enabled=warn_full_sample,
    )
    min_size = _normalize_min_train_size(min_train_size, minimum=2)
    lag_values = tuple(range(1, int(max_lag) + 1))
    # Match the author-supplied R snippet: VAR datamat lag columns are ordered
    # as lag 1 for all variables, lag 2 for all variables, and so on. When
    # scale_lags=True, the whole lag matrix is z-scored with R's sample sd
    # convention before each lag-l column becomes mean(lag 1, ..., lag l).
    lag_matrix = pd.DataFrame(
        {
            f"{column}_lag{lag}": panel[column].shift(lag)
            for lag in lag_values
            for column in columns
        },
        index=panel.index,
    )
    scaled_lags = _zscore_frame(
        lag_matrix,
        fit_policy=fit_value,
        min_train_size=min_size,
        ddof=1,
    )
    result = pd.DataFrame(index=panel.index)
    for column in columns:
        for lag in lag_values:
            lag_columns = [f"{column}_lag{step}" for step in range(1, lag + 1)]
            result[f"{column}_ma{lag}_lag1"] = scaled_lags.loc[:, lag_columns].mean(axis=1, skipna=False)
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        dict(panel.attrs.get("macroforecast_metadata", {})),
        "feature_engineering_marx",
        {
            "columns": list(columns),
            "max_lag": int(max_lag),
            "scale_lags": True,
            "fit_policy": fit_value,
            "min_train_size": min_size,
            "note": (
                "MARX built from z-scored lag-matrix columns before taking "
                "increasing lag averages, matching the author R-code pattern."
            ),
        },
    )
    return result
