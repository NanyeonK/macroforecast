from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import Rule, register_op
from ..types import DataType, Factor, ForecastArtifact, L4ForecastsArtifact, LaggedPanel, MappingArtifact, Panel, Series


def _values(item):
    """Extract a numeric pandas object from a Panel/Series/DataType wrapper."""

    if hasattr(item, "data") and isinstance(getattr(item, "data", None), pd.DataFrame):
        return item.data
    if hasattr(item, "metadata") and isinstance(getattr(item, "metadata", None), object):
        meta = getattr(item.metadata, "values", {}) if hasattr(item.metadata, "values") else {}
        if isinstance(meta, dict) and "data" in meta:
            return meta["data"]
    if isinstance(item, (pd.DataFrame, pd.Series)):
        return item
    return item


@register_op(
    name="identity",
    layer_scope="universal",
    input_types={"default": (Panel, Series, LaggedPanel, Factor)},
    output_type=(Panel, Series, LaggedPanel, Factor),
)
def identity(input_data, params):
    return input_data[0] if isinstance(input_data, list) and input_data else input_data


@register_op(
    name="concat",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    params_schema={"axis": {"type": str, "default": "column", "sweepable": False}},
    hard_rules=(
        Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "concat requires at least 2 inputs"),
    ),
)
def concat(inputs, params):
    frames = [_values(item) for item in inputs]
    axis = 0 if params.get("axis") == "row" else 1
    return pd.concat(frames, axis=axis)


@register_op(
    name="lag",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series, Factor)},
    output_type=LaggedPanel,
    params_schema={
        "n_lag": {"type": int, "default": 4, "sweepable": True},
        "include_contemporaneous": {"type": bool, "default": False},
    },
    hard_rules=(
        Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("n_lag", 4) >= 1, "n_lag must be >= 1"),
    ),
    soft_rules=(
        Rule(
            "soft",
            lambda dag, nref: "T" not in dag.layer_globals or dag.node(nref.node_id).params.get("n_lag", 4) <= dag.layer_globals["T"] / 10,
            "high lag/observation ratio, overfitting risk",
        ),
    ),
)
def lag(input_data, params):
    source = input_data[0] if isinstance(input_data, list) else input_data
    n_lag = int(params.get("n_lag", 4))
    include_now = bool(params.get("include_contemporaneous", False))
    if isinstance(source, Panel):
        base_columns = source.column_names
        lag_columns = [
            f"{column}_lag{lag}"
            for column in base_columns
            for lag in range(0 if include_now else 1, n_lag + 1)
        ]
        width = len(lag_columns)
        return LaggedPanel(
            shape=(source.shape[0], width) if source.shape else None,
            column_names=tuple(lag_columns),
            n_lag=n_lag,
        )
    if isinstance(source, Series):
        lag_columns = [
            f"{source.name}_lag{lag}"
            for lag in range(0 if include_now else 1, n_lag + 1)
        ]
        return LaggedPanel(
            shape=(source.shape[0], len(lag_columns)) if source.shape else None,
            column_names=tuple(lag_columns),
            n_lag=n_lag,
        )
    raise TypeError(f"lag expected Panel or Series, got {type(source).__name__}")


@register_op(
    name="level",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
)
def level(input_data, params):
    return _values(input_data[0] if isinstance(input_data, list) else input_data)


@register_op(
    name="log",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series, LaggedPanel, Factor)},
    output_type=(Panel, Series, LaggedPanel, Factor),
)
def log(input_data, params):
    series = _values(input_data[0] if isinstance(input_data, list) else input_data)
    if isinstance(series, (pd.DataFrame, pd.Series)):
        return np.log(series.where(series > 0))
    return series


@register_op(
    name="diff",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
    params_schema={"n_diff": {"type": int, "default": 1, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("n_diff", 1) >= 1, "n_diff must be >= 1"),),
)
def diff(input_data, params):
    series = _values(input_data[0] if isinstance(input_data, list) else input_data)
    return series.diff(periods=int(params.get("n_diff", 1))) if isinstance(series, (pd.DataFrame, pd.Series)) else series


@register_op(
    name="log_diff",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
    params_schema={"n_diff": {"type": int, "default": 1, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("n_diff", 1) >= 1, "n_diff must be >= 1"),),
)
def log_diff(input_data, params):
    series = _values(input_data[0] if isinstance(input_data, list) else input_data)
    if isinstance(series, (pd.DataFrame, pd.Series)):
        return np.log(series.where(series > 0)).diff(periods=int(params.get("n_diff", 1)))
    return series


@register_op(
    name="pct_change",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
    params_schema={"n_periods": {"type": int, "default": 1, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("n_periods", 1) >= 1, "n_periods must be >= 1"),),
)
def pct_change(input_data, params):
    series = _values(input_data[0] if isinstance(input_data, list) else input_data)
    return series.pct_change(periods=int(params.get("n_periods", 1))) if isinstance(series, (pd.DataFrame, pd.Series)) else series


@register_op(
    name="interact",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) == 2, "interact requires exactly 2 inputs"),),
)
def interact(inputs, params):
    a = _values(inputs[0])
    b = _values(inputs[1])
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        cols = {f"{ca}__x__{cb}": a[ca] * b[cb] for ca in a.columns for cb in b.columns}
        return pd.DataFrame(cols, index=a.index)
    return a * b


@register_op(
    name="hierarchical_pca",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Factor,
    params_schema={
        "n_components_per_block": {"type": int, "default": 4, "sweepable": True},
        "n_components_top": {"type": int, "default": 4, "sweepable": True},
    },
)
def hierarchical_pca(inputs, params):
    from sklearn.decomposition import PCA

    n_per = int(params.get("n_components_per_block", 4))
    n_top = int(params.get("n_components_top", 4))
    block_factors: list[pd.DataFrame] = []
    for index, item in enumerate(inputs):
        frame = _values(item)
        if not isinstance(frame, pd.DataFrame):
            continue
        cleaned = frame.dropna(axis=0, how="any")
        if cleaned.empty:
            continue
        n = max(1, min(n_per, min(cleaned.shape) - 1))
        scores = PCA(n_components=n, random_state=0).fit_transform(cleaned.to_numpy() - cleaned.to_numpy().mean(axis=0))
        block_factors.append(
            pd.DataFrame(
                scores,
                index=cleaned.index,
                columns=[f"block{index + 1}_f{i + 1}" for i in range(scores.shape[1])],
            )
        )
    if not block_factors:
        return pd.DataFrame()
    stacked = pd.concat(block_factors, axis=1).dropna()
    n_top = max(1, min(n_top, min(stacked.shape) - 1))
    top_scores = PCA(n_components=n_top, random_state=0).fit_transform(stacked.to_numpy() - stacked.to_numpy().mean(axis=0))
    return pd.DataFrame(
        top_scores,
        index=stacked.index,
        columns=[f"hpca_top_{i + 1}" for i in range(top_scores.shape[1])],
    )


@register_op(
    name="weighted_concat",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    params_schema={"weights": {"type": list, "default": [], "sweepable": False}},
    hard_rules=(
        Rule(
            "hard",
            lambda dag, nref: not dag.node(nref.node_id).params.get("weights")
            or len(dag.node(nref.node_id).params.get("weights", ())) == len(dag.node(nref.node_id).inputs),
            "weighted_concat weight count must match input count",
        ),
    ),
)
def weighted_concat(inputs, params):
    weights = list(params.get("weights") or [1.0] * len(inputs))
    pieces = []
    for item, weight in zip(inputs, weights):
        frame = _values(item)
        if isinstance(frame, (pd.DataFrame, pd.Series)):
            pieces.append(frame * float(weight))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1)


@register_op(
    name="simple_average",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "simple_average requires at least 2 inputs"),),
)
def simple_average(inputs, params):
    frames = [_values(item) for item in inputs if isinstance(_values(item), (pd.DataFrame, pd.Series))]
    if not frames:
        return pd.DataFrame()
    if all(isinstance(frame, pd.Series) for frame in frames):
        return pd.concat(frames, axis=1).mean(axis=1)
    aligned = pd.concat([frame.reset_index(drop=True) if isinstance(frame, pd.DataFrame) else frame.to_frame().reset_index(drop=True) for frame in frames], axis=1).groupby(level=0, axis=1).mean()
    aligned.index = frames[0].index
    return aligned


@register_op(
    name="weighted_average_forecast",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    params_schema={
        "weights_method": {
            "type": str,
            "default": "dmsfe",
            "sweepable": True,
            "options": ["equal", "dmsfe", "inverse_msfe", "mallows_cp", "sic_weights", "granger_ramanathan", "cv_optimized"],
        },
        "temporal_rule": {"type": str, "default": "expanding_window_per_origin", "sweepable": True},
        "dmsfe_theta": {"type": float, "default": 0.95, "sweepable": True},
        "cv_optimized_window": {"type": int, "default": 60, "sweepable": True},
        "granger_ramanathan_constraint": {"type": str, "default": "sum_to_one", "sweepable": True},
    },
    hard_rules=(
        Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "weighted_average_forecast requires at least 2 inputs"),
        Rule(
            "hard",
            lambda dag, nref: dag.node(nref.node_id).params.get("temporal_rule", "expanding_window_per_origin") != "full_sample_once",
            "full_sample_once is rejected for forecast combination temporal_rule",
        ),
    ),
)
def weighted_average_forecast(inputs, params):
    return _combine_forecast_artifacts(inputs, method="weighted_average", params=params)


@register_op(
    name="median_forecast",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "median_forecast requires at least 2 inputs"),),
)
def median_forecast(inputs, params):
    return _combine_forecast_artifacts(inputs, method="median", params=params)


@register_op(
    name="trimmed_mean_forecast",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    params_schema={"trim_pct": {"type": float, "default": 0.1, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "trimmed_mean_forecast requires at least 2 inputs"),),
)
def trimmed_mean_forecast(inputs, params):
    return _combine_forecast_artifacts(inputs, method="trimmed_mean", params=params)


@register_op(
    name="bma_forecast",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    params_schema={"prior_method": {"type": str, "default": "uniform", "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "bma_forecast requires at least 2 inputs"),),
)
def bma_forecast(inputs, params):
    return _combine_forecast_artifacts(inputs, method="bma", params=params)


@register_op(
    name="bivariate_ardl_combination",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    params_schema={"combination_weights_temporal_rule": {"type": str, "default": "expanding_window_per_origin", "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) == 2, "bivariate_ardl_combination requires exactly 2 inputs"),),
)
def bivariate_ardl_combination(inputs, params):
    return _combine_forecast_artifacts(inputs[:2], method="bivariate_ardl", params=params)


def _combine_forecast_artifacts(inputs, *, method: str, params: dict):
    """Combine multiple ForecastArtifacts (or L4ForecastsArtifacts) into one.

    Operates on the per-(target, horizon, origin) forecast tensor. Each method
    computes one combined forecast value per origin from the member values.
    """

    members = [item for item in inputs if hasattr(item, "forecasts")]
    if not members:
        return inputs[0] if inputs else None
    rows: dict[tuple, list[float]] = {}
    for member in members:
        for (model_id, target, horizon, origin), value in member.forecasts.items():
            rows.setdefault((target, horizon, origin), []).append(float(value))
    weights = params.get("weights")
    weights_method = params.get("weights_method", "equal")
    combined: dict[tuple[str, str, int, object], float] = {}
    for (target, horizon, origin), values in rows.items():
        arr = np.asarray(values, dtype=float)
        if method == "median":
            combined[("combined", target, int(horizon), origin)] = float(np.median(arr))
        elif method == "trimmed_mean":
            trim = float(params.get("trim_pct", 0.1))
            n_trim = max(0, int(len(arr) * trim))
            sorted_arr = np.sort(arr)
            kept = sorted_arr[n_trim : len(arr) - n_trim] if n_trim and len(arr) > 2 * n_trim else sorted_arr
            combined[("combined", target, int(horizon), origin)] = float(np.mean(kept))
        elif method == "bma":
            combined[("combined", target, int(horizon), origin)] = float(np.mean(arr))
        elif method == "bivariate_ardl":
            combined[("combined", target, int(horizon), origin)] = float(0.5 * arr[0] + 0.5 * arr[1])
        else:
            if weights_method == "equal" or not weights:
                w = np.full(len(arr), 1.0 / len(arr))
            else:
                weight_arr = np.asarray(weights, dtype=float)
                weight_arr = weight_arr / max(weight_arr.sum(), 1e-9)
                w = weight_arr if len(weight_arr) == len(arr) else np.full(len(arr), 1.0 / len(arr))
            combined[("combined", target, int(horizon), origin)] = float(np.dot(w, arr))
    base = members[0]
    return type(base)(
        forecasts=combined,
        forecast_object=getattr(base, "forecast_object", "point"),
        sample_index=base.sample_index,
        targets=base.targets,
        horizons=base.horizons,
        model_ids=("combined",),
        upstream_hashes={},
    )


@register_op(
    name="layer_meta_aggregate",
    layer_scope="universal",
    input_types={"default": DataType},
    output_type=MappingArtifact,
)
def layer_meta_aggregate(inputs, params):
    return {"inputs": inputs, **params}
