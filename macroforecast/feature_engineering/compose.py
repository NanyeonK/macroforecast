from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from macroforecast.data import attach_metadata, validate_panel
from macroforecast.feature_engineering.types import FeatureInput
from macroforecast.feature_engineering.shared import (
    FitPolicy,
    _coerce_input,
    _metadata_frame,
    _normalize_feature_method,
    _records_for_columns,
    _reject_extra_params,
    _resolve_columns,
    _step,
)
from macroforecast.feature_engineering.transforms import (
    group_pca,
    lag,
    maf_features,
    moving_average_ladder,
    pca_features,
    rolling_mean,
    scale_features,
    time_features,
)

def lag_step(
    *,
    name: str = "lag",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (1,),
    include: bool = True,
    drop_missing: bool = False,
) -> dict[str, Any]:
    """Return a reusable lag step for ``compose_features``."""

    return _step(
        name=name,
        method="lag",
        input=input,
        include=include,
        columns=columns,
        lags=lags,
        drop_missing=drop_missing,
    )


def rolling_step(
    *,
    name: str = "rolling_mean",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    windows: Iterable[int] | int = (3,),
    min_periods: int | None = None,
    shift: int = 0,
    include: bool = True,
    drop_missing: bool = False,
) -> dict[str, Any]:
    """Return a reusable rolling-mean step for ``compose_features``."""

    return _step(
        name=name,
        method="rolling_mean",
        input=input,
        include=include,
        columns=columns,
        windows=windows,
        min_periods=min_periods,
        shift=shift,
        drop_missing=drop_missing,
    )


def moving_average_step(
    *,
    name: str = "ma",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    windows: Iterable[int] | None = None,
    max_window: int = 12,
    min_periods: int | None = None,
    shift: int = 0,
    include: bool = True,
    drop_missing: bool = False,
) -> dict[str, Any]:
    """Return a reusable moving-average-ladder step for ``compose_features``."""

    return _step(
        name=name,
        method="moving_average_ladder",
        input=input,
        include=include,
        columns=columns,
        windows=windows,
        max_window=max_window,
        min_periods=min_periods,
        shift=shift,
        drop_missing=drop_missing,
    )


def maf_step(
    *,
    name: str = "maf",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    max_lag: int = 12,
    lags: Iterable[int] | None = None,
    n_components: int = 2,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = False,
    prefix: str | None = None,
    include: bool = True,
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable MAF step for ``compose_features``."""

    return _step(
        name=name,
        method="maf",
        input=input,
        include=include,
        columns=columns,
        max_lag=max_lag,
        lags=lags,
        n_components=n_components,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        scale=scale,
        prefix=name if prefix is None else prefix,
        drop_missing=drop_missing,
        random_state=random_state,
        warn_full_sample=warn_full_sample,
    )


def group_pca_step(
    *,
    groups: Mapping[str, Iterable[str]],
    name: str = "group_pca",
    input: str = "panel",
    n_components: int | Mapping[str, int] = 1,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str | None = None,
    include: bool = True,
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable grouped-PCA step for ``compose_features``."""

    return _step(
        name=name,
        method="group_pca",
        input=input,
        include=include,
        groups=groups,
        n_components=n_components,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        scale=scale,
        prefix=prefix,
        drop_missing=drop_missing,
        random_state=random_state,
        warn_full_sample=warn_full_sample,
    )


def pca_step(
    *,
    name: str = "pc",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    n_components: int = 1,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str | None = None,
    include: bool = True,
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable PCA step for ``compose_features``."""

    return _step(
        name=name,
        method="pca",
        input=input,
        include=include,
        columns=columns,
        n_components=n_components,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        scale=scale,
        prefix=name if prefix is None else prefix,
        drop_missing=drop_missing,
        random_state=random_state,
        warn_full_sample=warn_full_sample,
    )


def scale_step(
    *,
    name: str = "scale",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    method: str = "zscore",
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable scaling step for ``compose_features``."""

    return _step(
        name=name,
        method="scale",
        input=input,
        include=include,
        columns=columns,
        scale_method=method,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        drop_missing=drop_missing,
        warn_full_sample=warn_full_sample,
    )


def pca_then_lags(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 1,
    lags: Iterable[int] | int = (1,),
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str = "pc",
    include_pca: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create PCA factors and lagged PCA factors in one direct call."""

    return compose_features(
        data,
        [
            pca_step(
                name=prefix,
                columns=columns,
                n_components=n_components,
                fit_policy=fit_policy,
                min_train_size=min_train_size,
                scale=scale,
                prefix=prefix,
                include=include_pca,
                warn_full_sample=warn_full_sample,
            ),
            lag_step(name=f"{prefix}_lag", input=prefix, lags=lags),
        ],
        metadata=metadata,
        drop_missing=drop_missing,
    )


def lags_then_pca(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (0, 1),
    n_components: int = 1,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str = "lag_pc",
    include_lags: bool = False,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create lag block first, then PCA on that lag block."""

    return compose_features(
        data,
        [
            lag_step(name="lag_block", columns=columns, lags=lags, include=include_lags),
            pca_step(
                name=prefix,
                input="lag_block",
                n_components=n_components,
                fit_policy=fit_policy,
                min_train_size=min_train_size,
                scale=scale,
                prefix=prefix,
                warn_full_sample=warn_full_sample,
            ),
        ],
        metadata=metadata,
        drop_missing=drop_missing,
    )


def moving_average_pca_lags(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    windows: Iterable[int] | None = None,
    max_window: int = 12,
    n_components: int = 1,
    lags: Iterable[int] | int = (1,),
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str = "ma_pc",
    include_pca: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create moving-average block, PCA factors, then lags of those factors."""

    return compose_features(
        data,
        [
            moving_average_step(
                name="ma_block",
                columns=columns,
                windows=windows,
                max_window=max_window,
                include=False,
            ),
            pca_step(
                name=prefix,
                input="ma_block",
                n_components=n_components,
                fit_policy=fit_policy,
                min_train_size=min_train_size,
                scale=scale,
                prefix=prefix,
                include=include_pca,
                warn_full_sample=warn_full_sample,
            ),
            lag_step(name=f"{prefix}_lag", input=prefix, lags=lags),
        ],
        metadata=metadata,
        drop_missing=drop_missing,
    )


def compose_features(
    data: FeatureInput,
    steps: Iterable[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    include_original: bool = False,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Run named feature-engineering steps sequentially.

    Each step reads either the original ``panel`` or a prior step via
    ``input='step_name'``. This keeps the callable API composable while any
    future recipe wrapper can call the same functions.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    source_columns = _resolve_columns(panel, columns=columns)
    root = panel.loc[:, source_columns].copy()
    outputs: dict[str, pd.DataFrame] = {"panel": root}
    included: list[pd.DataFrame] = [root] if include_original else []
    step_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []

    for position, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            raise TypeError("each feature step must be a mapping")
        raw = dict(step)
        method = _normalize_feature_method(str(raw.pop("method", raw.pop("op", ""))))
        name = str(raw.pop("name", f"{method}_{position}"))
        if not name:
            raise ValueError("feature step name must be non-empty")
        if name in outputs:
            raise ValueError(f"duplicate feature step name: {name!r}")
        source_name = str(raw.pop("input", "panel"))
        if source_name not in outputs:
            raise ValueError(f"feature step {name!r} references unknown input {source_name!r}")
        include = bool(raw.pop("include", True))
        source = outputs[source_name]
        out = _run_feature_step(source, method=method, step_name=name, params=raw)
        if out.columns.has_duplicates:
            raise ValueError(f"feature step {name!r} produced duplicate columns")
        outputs[name] = out
        if include:
            included.append(out)
        step_record = {
            "step": name,
            "method": method,
            "input": source_name,
            "include": include,
            "columns": [str(column) for column in out.columns],
        }
        step_records.append(step_record)
        out_metadata = out.attrs.get("macroforecast_feature_metadata")
        if isinstance(out_metadata, pd.DataFrame) and not out_metadata.empty:
            records = out_metadata.to_dict("records")
        else:
            records = _records_for_columns(
                out,
                operation=method,
                sources=tuple(str(column) for column in source.columns),
                included=include,
            )
        for record in records:
            record["step"] = name
            record["included"] = include
            if not record.get("source"):
                record["source"] = source_name
            if not record.get("parameter"):
                record["parameter"] = name
        feature_records.extend(records)

    if not included:
        raise ValueError("compose_features() produced no included feature blocks")
    result = pd.concat(included, axis=1)
    duplicate_columns = result.columns[result.columns.duplicated()].unique()
    if len(duplicate_columns):
        raise ValueError(f"composed features contain duplicate columns: {list(map(str, duplicate_columns))}")
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    composed_metadata = attach_metadata(
        base.metadata,
        "feature_engineering_compose",
        {
            "input_columns": list(source_columns),
            "include_original": bool(include_original),
            "drop_missing": bool(drop_missing),
            "steps": step_records,
            "output_columns": [str(column) for column in result.columns],
        },
    )
    result.attrs["macroforecast_metadata"] = composed_metadata
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(feature_records)
    return result


def _run_feature_step(
    source: pd.DataFrame,
    *,
    method: str,
    step_name: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    columns = params.pop("columns", None)
    if method == "lag":
        lag_values = params.pop("lags", params.pop("n_lag", (1,)))
        drop_missing = bool(params.pop("drop_missing", False))
        _reject_extra_params(params, step_name)
        return lag(source, columns=columns, lags=lag_values, drop_missing=drop_missing)
    if method == "rolling_mean":
        windows = params.pop("windows", params.pop("window", (3,)))
        min_periods = params.pop("min_periods", None)
        shift = int(params.pop("shift", 0))
        drop_missing = bool(params.pop("drop_missing", False))
        _reject_extra_params(params, step_name)
        return rolling_mean(
            source,
            columns=columns,
            windows=windows,
            min_periods=min_periods,
            shift=shift,
            drop_missing=drop_missing,
        )
    if method == "moving_average_ladder":
        windows = params.pop("windows", None)
        max_window = int(params.pop("max_window", 12))
        min_periods = params.pop("min_periods", None)
        shift = int(params.pop("shift", 0))
        drop_missing = bool(params.pop("drop_missing", False))
        _reject_extra_params(params, step_name)
        return moving_average_ladder(
            source,
            columns=columns,
            windows=windows,
            max_window=max_window,
            min_periods=min_periods,
            shift=shift,
            drop_missing=drop_missing,
        )
    if method == "scale":
        scale_method = str(params.pop("scale_method", params.pop("method_name", "zscore")))
        fit_policy = params.pop("fit_policy", "expanding")
        min_train_size = params.pop("min_train_size", None)
        drop_missing = bool(params.pop("drop_missing", False))
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        return scale_features(
            source,
            columns=columns,
            method=scale_method,
            fit_policy=fit_policy,
            min_train_size=min_train_size,
            drop_missing=drop_missing,
            warn_full_sample=warn_full_sample,
        )
    if method == "pca":
        n_components = int(params.pop("n_components", 1))
        fit_policy = params.pop("fit_policy", "expanding")
        min_train_size = params.pop("min_train_size", None)
        scale = bool(params.pop("scale", True))
        prefix = str(params.pop("prefix", step_name))
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", None)
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        return pca_features(
            source,
            columns=columns,
            n_components=n_components,
            fit_policy=fit_policy,
            min_train_size=min_train_size,
            scale=scale,
            prefix=prefix,
            drop_missing=drop_missing,
            random_state=random_state,
            warn_full_sample=warn_full_sample,
        )
    if method == "group_pca":
        groups = params.pop("groups", None)
        if groups is None:
            raise ValueError(f"feature step {step_name!r} requires groups")
        n_components = params.pop("n_components", 1)
        fit_policy = params.pop("fit_policy", "expanding")
        min_train_size = params.pop("min_train_size", None)
        scale = bool(params.pop("scale", True))
        prefix = params.pop("prefix", None)
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", None)
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        return group_pca(
            source,
            groups=groups,
            n_components=n_components,
            fit_policy=fit_policy,
            min_train_size=min_train_size,
            scale=scale,
            prefix=prefix,
            drop_missing=drop_missing,
            random_state=random_state,
            warn_full_sample=warn_full_sample,
        )
    if method == "maf":
        max_lag = int(params.pop("max_lag", 12))
        lags = params.pop("lags", None)
        n_components = int(params.pop("n_components", 2))
        fit_policy = params.pop("fit_policy", "expanding")
        min_train_size = params.pop("min_train_size", None)
        scale = bool(params.pop("scale", False))
        prefix = str(params.pop("prefix", "maf"))
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", None)
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        return maf_features(
            source,
            columns=columns,
            max_lag=max_lag,
            lags=lags,
            n_components=n_components,
            fit_policy=fit_policy,
            min_train_size=min_train_size,
            scale=scale,
            prefix=prefix,
            drop_missing=drop_missing,
            random_state=random_state,
            warn_full_sample=warn_full_sample,
        )
    if method == "time":
        trend = bool(params.pop("trend", True))
        month = bool(params.pop("month", False))
        quarter = bool(params.pop("quarter", False))
        year = bool(params.pop("year", False))
        _reject_extra_params(params, step_name)
        return time_features(source, trend=trend, month=month, quarter=quarter, year=year)
    raise ValueError(f"unsupported feature method {method!r}")
