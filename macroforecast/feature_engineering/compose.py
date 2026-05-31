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
    fourier_features,
    group_pca,
    hamilton_filter_features,
    interaction_features,
    lag,
    maf_features,
    moving_average_ladder,
    nystroem_features,
    pca_features,
    polynomial_features,
    random_projection_features,
    rolling_mean,
    scale_features,
    season_dummy,
    seasonal_lag,
    sparse_pca_chen_rohe_features,
    time_features,
    transform_features,
    varimax_features,
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


def marx_step(
    *,
    name: str = "marx",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    max_lag: int = 12,
    scale_lags: bool = False,
    min_train_size: int | None = None,
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable MARX step for ``compose_features`` or ``feature_spec``.

    MARX uses increasing averages of lagged predictors. With
    ``scale_lags=True``, lag-matrix columns are z-scored before averaging,
    matching the author R-code variant.
    """

    return _step(
        name=name,
        method="marx",
        input=input,
        include=include,
        columns=columns,
        max_lag=max_lag,
        scale_lags=scale_lags,
        min_train_size=min_train_size,
        drop_missing=drop_missing,
        warn_full_sample=warn_full_sample,
    )


def transform_step(
    *,
    transform: str,
    name: str | None = None,
    input: str = "panel",
    columns: Iterable[str] | None = None,
    periods: int = 1,
    include: bool = True,
    drop_missing: bool = False,
) -> dict[str, Any]:
    """Return a reusable deterministic column transform step."""

    return _step(
        name=transform if name is None else name,
        method="transform",
        input=input,
        include=include,
        columns=columns,
        transform=transform,
        periods=periods,
        drop_missing=drop_missing,
    )


def seasonal_lag_step(
    *,
    name: str = "seasonal_lag",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    season_length: int = 12,
    lags: Iterable[int] | int = (1,),
    include: bool = True,
    drop_missing: bool = False,
) -> dict[str, Any]:
    """Return a reusable seasonal-lag step."""

    return _step(
        name=name,
        method="seasonal_lag",
        input=input,
        include=include,
        columns=columns,
        season_length=season_length,
        lags=lags,
        drop_missing=drop_missing,
    )


def season_dummy_step(
    *,
    name: str = "season_dummy",
    input: str = "panel",
    frequency: str = "auto",
    drop_first: bool = False,
    include: bool = True,
) -> dict[str, Any]:
    """Return a reusable date-index seasonal-dummy step."""

    return _step(
        name=name,
        method="season_dummy",
        input=input,
        include=include,
        frequency=frequency,
        drop_first=drop_first,
    )


def fourier_step(
    *,
    name: str = "fourier",
    input: str = "panel",
    period: int = 12,
    order: int = 2,
    prefix: str = "fourier",
    include: bool = True,
) -> dict[str, Any]:
    """Return a reusable Fourier seasonal-term step."""

    return _step(
        name=name,
        method="fourier",
        input=input,
        include=include,
        period=period,
        order=order,
        prefix=prefix,
    )


def time_step(
    *,
    name: str = "time",
    input: str = "panel",
    trend: bool = True,
    month: bool = False,
    quarter: bool = False,
    year: bool = False,
    include: bool = True,
) -> dict[str, Any]:
    """Return a reusable deterministic trend/month/quarter/year step."""

    return _step(
        name=name,
        method="time",
        input=input,
        include=include,
        trend=trend,
        month=month,
        quarter=quarter,
        year=year,
    )


def polynomial_step(
    *,
    name: str = "polynomial",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    degree: int = 2,
    include_bias: bool = False,
    interaction_only: bool = False,
    include: bool = True,
    drop_missing: bool = False,
) -> dict[str, Any]:
    """Return a reusable polynomial-expansion step."""

    return _step(
        name=name,
        method="polynomial",
        input=input,
        include=include,
        columns=columns,
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
        drop_missing=drop_missing,
    )


def interaction_step(
    *,
    name: str = "interaction",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    order: int = 2,
    include: bool = True,
    drop_missing: bool = False,
) -> dict[str, Any]:
    """Return a reusable pure-interaction step."""

    return _step(
        name=name,
        method="interaction",
        input=input,
        include=include,
        columns=columns,
        order=order,
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


def partial_least_squares_step(
    *,
    name: str = "pls",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    n_components: int = 3,
    min_train_size: int | None = None,
    prefix: str | None = None,
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a target-aware PLS step for ``feature_spec``.

    ``compose_features`` has no target contract; use the direct
    ``partial_least_squares_features`` callable for full-sample manual use.
    """

    return _step(
        name=name,
        method="partial_least_squares",
        input=input,
        include=include,
        columns=columns,
        n_components=n_components,
        min_train_size=min_train_size,
        prefix=name if prefix is None else prefix,
        drop_missing=drop_missing,
        warn_full_sample=warn_full_sample,
    )


def sliced_inverse_regression_step(
    *,
    name: str = "sir",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    n_components: int = 3,
    n_slices: int = 10,
    scaling_policy: str = "scaled_pca",
    min_train_size: int | None = None,
    prefix: str | None = None,
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a target-aware SIR step for ``feature_spec``."""

    return _step(
        name=name,
        method="sliced_inverse_regression",
        input=input,
        include=include,
        columns=columns,
        n_components=n_components,
        n_slices=n_slices,
        scaling_policy=scaling_policy,
        min_train_size=min_train_size,
        prefix=name if prefix is None else prefix,
        drop_missing=drop_missing,
        warn_full_sample=warn_full_sample,
    )


def feature_selection_step(
    *,
    name: str = "select",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    n_features: int | float = 0.5,
    method: str = "variance",
    lasso_alpha: float = 0.001,
    min_train_size: int | None = None,
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a runner-safe feature-selection step for ``feature_spec``."""

    return _step(
        name=name,
        method="feature_selection",
        input=input,
        include=include,
        columns=columns,
        n_features=n_features,
        selection_method=method,
        lasso_alpha=lasso_alpha,
        min_train_size=min_train_size,
        drop_missing=drop_missing,
        warn_full_sample=warn_full_sample,
    )


def sparse_pca_chen_rohe_step(
    *,
    name: str = "sca",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    n_components: int = 4,
    zeta: float = 0.0,
    max_iter: int = 200,
    var_innovations: bool = False,
    prefix: str | None = None,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    include: bool = True,
    drop_missing: bool = False,
    random_state: int | None = 0,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable Chen-Rohe sparse component step."""

    return _step(
        name=name,
        method="sparse_pca_chen_rohe",
        input=input,
        include=include,
        columns=columns,
        n_components=n_components,
        zeta=zeta,
        max_iter=max_iter,
        var_innovations=var_innovations,
        prefix=name if prefix is None else prefix,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        drop_missing=drop_missing,
        random_state=random_state,
        warn_full_sample=warn_full_sample,
    )


def varimax_step(
    *,
    name: str = "varimax",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
    prefix: str | None = None,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable orthogonal varimax-rotation step."""

    return _step(
        name=name,
        method="varimax",
        input=input,
        include=include,
        columns=columns,
        max_iter=max_iter,
        tol=tol,
        prefix=name if prefix is None else prefix,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        drop_missing=drop_missing,
        warn_full_sample=warn_full_sample,
    )


def hamilton_step(
    *,
    name: str = "hamilton",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    h: int = 8,
    p: int = 4,
    component: str = "cycle",
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    missing: str = "drop",
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable Hamilton-filter step for ``compose_features``."""

    return _step(
        name=name,
        method="hamilton_filter",
        input=input,
        include=include,
        columns=columns,
        h=h,
        p=p,
        component=component,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        missing=missing,
        drop_missing=drop_missing,
        warn_full_sample=warn_full_sample,
    )


def random_projection_step(
    *,
    name: str = "rp",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    n_components: int = 2,
    random_state: int | None = None,
    prefix: str | None = None,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable Gaussian random-projection step."""

    return _step(
        name=name,
        method="random_projection",
        input=input,
        include=include,
        columns=columns,
        n_components=n_components,
        random_state=random_state,
        prefix=name if prefix is None else prefix,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        drop_missing=drop_missing,
        warn_full_sample=warn_full_sample,
    )


def nystroem_step(
    *,
    name: str = "nys",
    input: str = "panel",
    columns: Iterable[str] | None = None,
    n_components: int = 10,
    kernel: str = "rbf",
    gamma: float | None = None,
    random_state: int | None = None,
    prefix: str | None = None,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    include: bool = True,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> dict[str, Any]:
    """Return a reusable Nystroem kernel-approximation step."""

    return _step(
        name=name,
        method="nystroem",
        input=input,
        include=include,
        columns=columns,
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        random_state=random_state,
        prefix=name if prefix is None else prefix,
        fit_policy=fit_policy,
        min_train_size=min_train_size,
        drop_missing=drop_missing,
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
        method_value = str(raw.pop("method")) if "method" in raw else str(raw.pop("op", ""))
        method = _normalize_feature_method(method_value)
        if method == "transform" and "transform" not in raw and method_value.lower() not in {
            "transform",
            "transform_features",
            "feature_transform",
        }:
            raw["transform"] = method_value
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
    if method == "marx":
        max_lag = int(params.pop("max_lag", 12))
        scale_lags = bool(params.pop("scale_lags", False))
        min_train_size = params.pop("min_train_size", None)
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, step_name)
        selected = _resolve_columns(source, columns=columns)
        if max_lag <= 0:
            raise ValueError("max_lag must be positive")
        if not scale_lags:
            result = moving_average_ladder(
                source,
                columns=selected,
                windows=range(1, max_lag + 1),
                shift=1,
                drop_missing=drop_missing,
            )
            result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
                _records_for_columns(result, operation="marx", sources=selected, included=True)
            )
            return result
        min_size = 2 if min_train_size is None else int(min_train_size)
        if min_size < 2:
            raise ValueError("min_train_size must be >= 2")
        lag_values = tuple(range(1, max_lag + 1))
        lag_matrix = pd.DataFrame(
            {
                f"{column}_lag{lag_value}": source[column].shift(lag_value)
                for lag_value in lag_values
                for column in selected
            },
            index=source.index,
        )
        complete = lag_matrix.dropna()
        if len(complete) < min_size:
            raise ValueError(f"feature step {step_name!r} has fewer than {min_size} complete rows to fit MARX")
        center = complete.mean(axis=0)
        divisor = complete.std(axis=0, ddof=1).replace(0.0, float("nan"))
        scaled = (lag_matrix - center) / divisor
        result = pd.DataFrame(index=source.index)
        for column in selected:
            for lag_order in lag_values:
                lag_columns = [f"{column}_lag{step}" for step in range(1, lag_order + 1)]
                result[f"{column}_ma{lag_order}_lag1"] = scaled.loc[:, lag_columns].mean(axis=1, skipna=False)
        if drop_missing:
            result = result.dropna()
        result.index.name = "date"
        result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
            _records_for_columns(result, operation="marx", sources=selected, included=True)
        )
        return result
    if method == "transform":
        transform = str(params.pop("transform", step_name))
        periods = int(params.pop("periods", 1))
        drop_missing = bool(params.pop("drop_missing", False))
        _reject_extra_params(params, step_name)
        return transform_features(
            source,
            columns=columns,
            transform=transform,
            periods=periods,
            drop_missing=drop_missing,
        )
    if method == "seasonal_lag":
        season_length = int(params.pop("season_length", 12))
        lag_values = params.pop("lags", (1,))
        drop_missing = bool(params.pop("drop_missing", False))
        _reject_extra_params(params, step_name)
        return seasonal_lag(
            source,
            columns=columns,
            season_length=season_length,
            lags=lag_values,
            drop_missing=drop_missing,
        )
    if method == "season_dummy":
        frequency = str(params.pop("frequency", "auto"))
        drop_first = bool(params.pop("drop_first", False))
        _reject_extra_params(params, step_name)
        return season_dummy(source, frequency=frequency, drop_first=drop_first)
    if method == "fourier":
        period = int(params.pop("period", 12))
        order = int(params.pop("order", 2))
        prefix = str(params.pop("prefix", "fourier"))
        _reject_extra_params(params, step_name)
        return fourier_features(source, period=period, order=order, prefix=prefix)
    if method == "polynomial":
        degree = int(params.pop("degree", 2))
        include_bias = bool(params.pop("include_bias", False))
        interaction_only = bool(params.pop("interaction_only", False))
        drop_missing = bool(params.pop("drop_missing", False))
        _reject_extra_params(params, step_name)
        return polynomial_features(
            source,
            columns=columns,
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only,
            drop_missing=drop_missing,
        )
    if method == "interaction":
        order = int(params.pop("order", 2))
        drop_missing = bool(params.pop("drop_missing", False))
        _reject_extra_params(params, step_name)
        return interaction_features(
            source,
            columns=columns,
            order=order,
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
    if method == "sparse_pca_chen_rohe":
        n_components = int(params.pop("n_components", 4))
        zeta = float(params.pop("zeta", 0.0))
        max_iter = int(params.pop("max_iter", 200))
        var_innovations = bool(params.pop("var_innovations", False))
        prefix = str(params.pop("prefix", step_name))
        min_train_size = params.pop("min_train_size", None)
        params.pop("fit_policy", None)
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", 0)
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        return sparse_pca_chen_rohe_features(
            source,
            columns=columns,
            n_components=n_components,
            zeta=zeta,
            max_iter=max_iter,
            var_innovations=var_innovations,
            prefix=prefix,
            min_train_size=min_train_size,
            drop_missing=drop_missing,
            random_state=None if random_state is None else int(random_state),
            warn_full_sample=warn_full_sample,
        )
    if method == "varimax":
        max_iter = int(params.pop("max_iter", 50))
        tol = float(params.pop("tol", 1e-7))
        prefix = str(params.pop("prefix", step_name))
        min_train_size = params.pop("min_train_size", None)
        params.pop("fit_policy", None)
        drop_missing = bool(params.pop("drop_missing", False))
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        return varimax_features(
            source,
            columns=columns,
            max_iter=max_iter,
            tol=tol,
            prefix=prefix,
            min_train_size=min_train_size,
            drop_missing=drop_missing,
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
    if method == "hamilton_filter":
        h = int(params.pop("h", 8))
        p = int(params.pop("p", 4))
        component = str(params.pop("component", "cycle"))
        fit_policy = params.pop("fit_policy", "expanding")
        min_train_size = params.pop("min_train_size", None)
        missing = str(params.pop("missing", "drop"))
        drop_missing = bool(params.pop("drop_missing", False))
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        return hamilton_filter_features(
            source,
            columns=columns,
            h=h,
            p=p,
            component=component,
            fit_policy=fit_policy,
            min_train_size=min_train_size,
            missing=missing,
            drop_missing=drop_missing,
            warn_full_sample=warn_full_sample,
        )
    if method == "random_projection":
        n_components = int(params.pop("n_components", 2))
        random_state = params.pop("random_state", None)
        prefix = str(params.pop("prefix", step_name))
        min_train_size = params.pop("min_train_size", None)
        params.pop("fit_policy", None)
        drop_missing = bool(params.pop("drop_missing", False))
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        if min_train_size is not None and len(source.loc[:, _resolve_columns(source, columns=columns)].dropna()) < int(
            min_train_size
        ):
            raise ValueError(f"feature step {step_name!r} has fewer than {int(min_train_size)} complete rows")
        return random_projection_features(
            source,
            columns=columns,
            n_components=n_components,
            random_state=None if random_state is None else int(random_state),
            prefix=prefix,
            drop_missing=drop_missing,
            warn_full_sample=warn_full_sample,
        )
    if method == "nystroem":
        n_components = int(params.pop("n_components", 10))
        kernel = str(params.pop("kernel", "rbf"))
        gamma = params.pop("gamma", None)
        random_state = params.pop("random_state", None)
        prefix = str(params.pop("prefix", step_name))
        min_train_size = params.pop("min_train_size", None)
        params.pop("fit_policy", None)
        drop_missing = bool(params.pop("drop_missing", False))
        warn_full_sample = bool(params.pop("warn_full_sample", True))
        _reject_extra_params(params, step_name)
        if min_train_size is not None and len(source.loc[:, _resolve_columns(source, columns=columns)].dropna()) < int(
            min_train_size
        ):
            raise ValueError(f"feature step {step_name!r} has fewer than {int(min_train_size)} complete rows")
        return nystroem_features(
            source,
            columns=columns,
            n_components=n_components,
            kernel=kernel,
            gamma=None if gamma is None else float(gamma),
            random_state=None if random_state is None else int(random_state),
            prefix=prefix,
            drop_missing=drop_missing,
            warn_full_sample=warn_full_sample,
        )
    if method in {"partial_least_squares", "sliced_inverse_regression", "feature_selection"}:
        raise ValueError(
            f"target-aware feature step {method!r} requires feature_spec(). "
            "Use the direct *_features callable when fitting on a full panel manually."
        )
    if method == "time":
        trend = bool(params.pop("trend", True))
        month = bool(params.pop("month", False))
        quarter = bool(params.pop("quarter", False))
        year = bool(params.pop("year", False))
        _reject_extra_params(params, step_name)
        return time_features(source, trend=trend, month=month, quarter=quarter, year=year)
    raise ValueError(f"unsupported feature method {method!r}")
