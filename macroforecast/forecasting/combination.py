from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CombinationSpec:
    """Forecast-combination request consumed by ``forecasting.run``."""

    method: str
    name: str
    models: tuple[str, ...] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    func: Callable[..., Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "name": self.name,
            "models": list(self.models) if self.models is not None else None,
            "params": dict(self.params),
            "callable": _callable_name(self.func),
        }


def combination_spec(
    method: str,
    *,
    name: str | None = None,
    models: Sequence[str] | None = None,
    **params: Any,
) -> CombinationSpec:
    """Build a runner-compatible forecast-combination spec."""

    canonical = _normalize_method(method)
    return CombinationSpec(
        method=canonical,
        name=name or f"combined_{canonical}",
        models=tuple(str(model) for model in models) if models is not None else None,
        params=dict(params),
    )


def custom_combination(
    name: str,
    func: Callable[..., Any],
    *,
    models: Sequence[str] | None = None,
    **params: Any,
) -> CombinationSpec:
    """Build a custom forecast-combination spec for ``forecasting.run``."""

    if not name:
        raise ValueError("custom combination name must be non-empty")
    if not callable(func):
        raise TypeError("custom combination func must be callable")
    return CombinationSpec(
        method="custom",
        name=str(name),
        models=tuple(str(model) for model in models) if models is not None else None,
        params=dict(params),
        func=func,
    )


def combine_mean(forecasts: Any) -> pd.Series:
    """Equal-weight average forecast."""

    frame = _forecast_frame(forecasts)
    return frame.mean(axis=1).rename("combined")


def combine_median(forecasts: Any) -> pd.Series:
    """Cross-model median forecast."""

    frame = _forecast_frame(forecasts)
    return frame.median(axis=1).rename("combined")


def combine_trimmed_mean(forecasts: Any, *, trim: float = 0.1) -> pd.Series:
    """Trim extreme model forecasts before averaging."""

    frame = _forecast_frame(forecasts)
    if not 0 <= trim < 0.5:
        raise ValueError("trim must satisfy 0 <= trim < 0.5")
    values = np.sort(frame.to_numpy(dtype=float), axis=1)
    n_models = values.shape[1]
    cut = int(np.floor(trim * n_models))
    if cut:
        values = values[:, cut:-cut]
    return pd.Series(np.nanmean(values, axis=1), index=frame.index, name="combined")


def combine_winsorized_mean(forecasts: Any, *, limits: tuple[float, float] = (0.1, 0.1)) -> pd.Series:
    """Winsorize cross-model forecasts before averaging."""

    frame = _forecast_frame(forecasts)
    lower, upper = limits
    if lower < 0 or upper < 0 or lower + upper >= 1:
        raise ValueError("limits must be non-negative and sum to less than 1")
    q_low = frame.quantile(lower, axis=1)
    q_high = frame.quantile(1 - upper, axis=1)
    clipped = frame.clip(lower=q_low, upper=q_high, axis=0)
    return clipped.mean(axis=1).rename("combined")


def combine_inverse_mspe(
    forecasts: Any,
    y_true: Any,
    *,
    discount: float = 1.0,
    min_weight: float = 1e-12,
    horizon: int = 1,
) -> pd.Series:
    """Combine forecasts with inverse discounted MSPE weights.

    For an h-step forecast the weights at a target date are decided at the
    origin (h target-date steps earlier), so only forecast errors realised on or
    before that origin are observable. The error history is therefore lagged by
    ``horizon`` target-date rows (a 1-row lag for one-step forecasts), preventing
    the use of not-yet-realised errors for multi-step combinations.
    """

    frame = _forecast_frame(forecasts)
    target = pd.Series(y_true).reindex(frame.index).astype(float)
    if not 0 < discount <= 1:
        raise ValueError("discount must satisfy 0 < discount <= 1")
    lag = max(1, int(horizon))
    errors = frame.sub(target, axis=0) ** 2
    weights = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    running = pd.Series(0.0, index=frame.columns, dtype=float)
    has_history = False
    for step, date in enumerate(frame.index):
        src = step - lag
        if src >= 0:
            current = errors.iloc[src]
            # Update only models with an observed error at this date; a model
            # missing a forecast carries its accumulated error forward unchanged
            # rather than being imputed with the cross-model mean (which mixes
            # scales and mis-weights a model that simply lacks one forecast).
            observed = current.notna()
            updated = discount * running + current.fillna(0.0)
            running = running.where(~observed, updated)
            has_history = True
        if not has_history or float(running.sum()) <= 0:
            weights.loc[date, :] = 1.0 / len(frame.columns)
        else:
            inv = 1.0 / running.clip(lower=min_weight)
            weights.loc[date, :] = inv / inv.sum()
    combined = (frame * weights).sum(axis=1)
    return combined.rename("combined")


combine_dmspe = combine_inverse_mspe


def combine_best_n(forecasts: Any, y_true: Any, *, n: int = 3, horizon: int = 1) -> pd.Series:
    """Average the historically best ``n`` models by MSPE.

    The ranking at each target date uses only errors observable at the forecast
    origin, so the expanding MSPE is lagged by ``horizon`` target-date rows
    (a 1-row lag for one-step forecasts).
    """

    frame = _forecast_frame(forecasts)
    target = pd.Series(y_true).reindex(frame.index).astype(float)
    n_value = int(n)
    if n_value < 1:
        raise ValueError("n must be at least 1")
    lag = max(1, int(horizon))
    mspe = frame.sub(target, axis=0).pow(2).expanding(min_periods=1).mean().shift(lag)
    output = pd.Series(index=frame.index, dtype=float, name="combined")
    for date in frame.index:
        historical = mspe.loc[date]
        if historical.isna().all():
            best = frame.columns[:n_value]
        else:
            best = historical.fillna(float("inf")).sort_values().index[:n_value]
        output.loc[date] = frame.loc[date, best].mean()
    return output



def _recursive_combination(
    forecasts: Any,
    y_true: Any,
    *,
    horizon: int,
    min_periods: int,
    weight_fn: "Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]]",
    window: int | None = None,
    shrink_to_equal: float | None = None,
) -> pd.Series:
    """Leak-free recursive combiner.

    At each target date the weights are fit by ``weight_fn`` on the (forecast,
    realised) history whose errors are observable at the forecast origin (lagged
    by ``horizon`` target-date rows), then applied to the current forecasts.
    Falls back to the equal-weight mean until ``min_periods`` of history exist.
    """
    frame = _forecast_frame(forecasts)
    target = pd.Series(y_true).reindex(frame.index).astype(float)
    lag = max(1, int(horizon))
    n_models = frame.shape[1]
    F = frame.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    combined = pd.Series(index=frame.index, dtype=float, name="combined")
    for step in range(len(frame.index)):
        row = F[step]
        finite_row = np.isfinite(row)
        avail = step - lag  # last history row whose realised error is observable
        used_weights = False
        if avail + 1 >= int(min_periods) and finite_row.all():
            lo = 0 if window is None else max(0, avail + 1 - int(window))
            Fh = F[lo : avail + 1]
            yh = y[lo : avail + 1]
            mask = np.isfinite(yh) & np.all(np.isfinite(Fh), axis=1)
            if int(mask.sum()) >= int(min_periods):
                weights, intercept = weight_fn(Fh[mask], yh[mask])
                if shrink_to_equal is not None:
                    weights = _shrink(weights, float(shrink_to_equal))
                if np.all(np.isfinite(weights)):
                    combined.iloc[step] = float(intercept + row @ weights)
                    used_weights = True
        if not used_weights:
            combined.iloc[step] = (
                float(np.nanmean(row[finite_row])) if finite_row.any() else np.nan
            )
    return combined


def combine_bates_granger(
    forecasts: Any, y_true: Any, *, horizon: int = 1, min_periods: int = 10,
    window: int | None = None, shrink_to_equal: float | None = None,
) -> pd.Series:
    """Bates-Granger (1969) minimum error-variance combination (full covariance)."""

    from macroforecast.models._weight_solvers import min_variance_weights

    def _wf(Fh: np.ndarray, yh: np.ndarray) -> "tuple[np.ndarray, float]":
        return min_variance_weights(Fh - yh[:, None]), 0.0

    return _recursive_combination(
        forecasts, y_true, horizon=horizon, min_periods=min_periods,
        weight_fn=_wf, window=window, shrink_to_equal=shrink_to_equal,
    )


def combine_granger_ramanathan(
    forecasts: Any, y_true: Any, *, variant: str = "constrained", horizon: int = 1,
    min_periods: int = 10, window: int | None = None, shrink_to_equal: float | None = None,
) -> pd.Series:
    """Granger-Ramanathan (1984) regression combination.

    ``variant``: ``"ols"`` (with intercept), ``"no_intercept"``, or
    ``"constrained"`` (no intercept, weights sum to one).
    """

    from macroforecast.models._weight_solvers import regression_weights

    key = str(variant).lower()
    if key not in {"ols", "no_intercept", "constrained"}:
        raise ValueError("variant must be 'ols', 'no_intercept', or 'constrained'")

    def _wf(Fh: np.ndarray, yh: np.ndarray) -> "tuple[np.ndarray, float]":
        return regression_weights(
            Fh, yh, intercept=(key == "ols"), sum_to_one=(key == "constrained")
        )

    return _recursive_combination(
        forecasts, y_true, horizon=horizon, min_periods=min_periods,
        weight_fn=_wf, window=window, shrink_to_equal=shrink_to_equal,
    )


def combine_constrained_ls(
    forecasts: Any, y_true: Any, *, horizon: int = 1, min_periods: int = 10,
    window: int | None = None, shrink_to_equal: float | None = None,
) -> pd.Series:
    """Non-negative weights summing to one minimising squared combination error."""

    from macroforecast.models._weight_solvers import constrained_ls_weights

    def _wf(Fh: np.ndarray, yh: np.ndarray) -> "tuple[np.ndarray, float]":
        return constrained_ls_weights(Fh, yh), 0.0

    return _recursive_combination(
        forecasts, y_true, horizon=horizon, min_periods=min_periods,
        weight_fn=_wf, window=window, shrink_to_equal=shrink_to_equal,
    )


def _shrink(weights: np.ndarray, shrinkage: float) -> np.ndarray:
    from macroforecast.models._weight_solvers import shrink_weights

    return shrink_weights(weights, shrinkage)



def combine_eigenvector(
    forecasts: Any, y_true: Any, *, horizon: int = 1, min_periods: int = 10,
    window: int | None = None, shrink_to_equal: float | None = None,
) -> pd.Series:
    """Eigenvector (principal-component) combination (Hsiao-Wan)."""

    from macroforecast.models._weight_solvers import eigenvector_weights

    def _wf(Fh: np.ndarray, yh: np.ndarray) -> "tuple[np.ndarray, float]":
        return eigenvector_weights(Fh - yh[:, None]), 0.0

    return _recursive_combination(
        forecasts, y_true, horizon=horizon, min_periods=min_periods,
        weight_fn=_wf, window=window, shrink_to_equal=shrink_to_equal,
    )


def combine_regularized(
    forecasts: Any, y_true: Any, *, penalty: str = "ridge", alpha: float = 1.0,
    intercept: bool = True, horizon: int = 1, min_periods: int = 10,
    window: int | None = None, shrink_to_equal: float | None = None,
) -> pd.Series:
    """Ridge/Lasso-penalised regression combination (high-dimensional weights)."""

    from macroforecast.models._weight_solvers import regularized_weights

    def _wf(Fh: np.ndarray, yh: np.ndarray) -> "tuple[np.ndarray, float]":
        return regularized_weights(
            Fh, yh, penalty=penalty, alpha=alpha, intercept=intercept
        )

    return _recursive_combination(
        forecasts, y_true, horizon=horizon, min_periods=min_periods,
        weight_fn=_wf, window=window, shrink_to_equal=shrink_to_equal,
    )



def combine_linear_pool(
    means: Any, sds: Any | None = None, *, weights: Any | None = None
) -> pd.DataFrame:
    """Linear opinion pool of (Gaussian) density forecasts.

    The combined density is the mixture ``sum_i w_i N(mu_i, sigma_i^2)``. ``means``
    and ``sds`` are frames (rows = dates, cols = models); ``weights`` default to
    equal. Returns a frame with the pooled ``mean``, ``variance`` and ``sd``. The
    mixture variance exceeds the weighted component variance, capturing model
    disagreement. If ``sds`` is omitted the pool reduces to the weighted mean.
    """
    mu = _forecast_frame(means)
    w = _pool_weights(weights, mu.columns)
    pooled_mean = mu.mul(w, axis=1).sum(axis=1)
    out = pd.DataFrame({"mean": pooled_mean}, index=mu.index)
    if sds is not None:
        sd = _forecast_frame(sds).reindex(index=mu.index, columns=mu.columns)
        var_comp = (sd ** 2).mul(w, axis=1).sum(axis=1)
        second = (mu ** 2).mul(w, axis=1).sum(axis=1)
        mixture_var = var_comp + second - pooled_mean ** 2
        out["variance"] = mixture_var.clip(lower=0.0)
        out["sd"] = np.sqrt(out["variance"])
    return out


def combine_log_pool(means: Any, sds: Any, *, weights: Any | None = None) -> pd.DataFrame:
    """Logarithmic opinion pool of Gaussian density forecasts.

    The combined density is proportional to ``prod_i f_i^{w_i}``. For Gaussians the
    pool is itself Gaussian with precision ``tau = sum_i w_i / sigma_i^2`` and mean
    ``(sum_i w_i mu_i / sigma_i^2) / tau``. The log pool is sharper (smaller
    variance) than the linear pool. Returns a frame with pooled ``mean``,
    ``variance`` and ``sd``.
    """
    mu = _forecast_frame(means)
    sd = _forecast_frame(sds).reindex(index=mu.index, columns=mu.columns)
    if not np.all(np.asarray(sd.to_numpy(), dtype=float) > 0):
        raise ValueError("log pool requires strictly positive forecast standard deviations")
    w = _pool_weights(weights, mu.columns)
    precision_i = sd.pow(-2.0)
    tau = precision_i.mul(w, axis=1).sum(axis=1)
    weighted_mean = (mu * precision_i).mul(w, axis=1).sum(axis=1)
    pooled_var = 1.0 / tau
    pooled_mean = weighted_mean * pooled_var
    out = pd.DataFrame(
        {"mean": pooled_mean, "variance": pooled_var, "sd": np.sqrt(pooled_var)},
        index=mu.index,
    )
    return out


def _pool_weights(weights: Any, columns: pd.Index) -> pd.Series:
    if weights is None:
        return pd.Series(1.0 / len(columns), index=columns)
    w = pd.Series(weights)
    if not w.index.equals(columns):
        w = pd.Series(np.asarray(weights, dtype=float).ravel(), index=columns)
    total = float(w.sum())
    if total <= 0:
        return pd.Series(1.0 / len(columns), index=columns)
    return (w / total).astype(float)


def resolve_combinations(value: Any) -> list[CombinationSpec]:
    """Normalize runner ``combination=...`` input into concrete specs."""

    if value is None or value is False:
        return []
    if isinstance(value, CombinationSpec):
        return [value]
    if isinstance(value, str):
        return [combination_spec(value)]
    if isinstance(value, Mapping):
        if "method" in value:
            return [_mapping_to_spec(value)]
        return [_mapping_value_to_spec(alias, spec) for alias, spec in value.items()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        specs: list[CombinationSpec] = []
        for item in value:
            specs.extend(resolve_combinations(item))
        return specs
    raise TypeError("combination must be a string, spec, sequence, mapping, or None")


def apply_combinations(
    forecasts: pd.DataFrame,
    specs: Sequence[CombinationSpec],
) -> list[dict[str, Any]]:
    """Create combined forecast records from base model forecast rows."""

    if not specs or forecasts.empty:
        return []
    required = {"date", "origin", "origin_pos", "horizon", "model", "prediction"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast table missing required columns: {sorted(missing)}")
    if "combined" in forecasts.columns:
        base = forecasts.loc[~forecasts["combined"].fillna(False).astype(bool)].copy()
    else:
        base = forecasts.copy()
    out: list[dict[str, Any]] = []
    for spec in specs:
        selected = _selected_model_rows(base, spec)
        if selected.empty:
            raise ValueError(
                f"combination {spec.name!r} has no matching base model forecasts"
            )
        for _, group in selected.groupby(["horizon"], dropna=False, sort=True):
            out.extend(_combined_group_records(group, spec))
    return out


def _combined_group_records(
    group: pd.DataFrame,
    spec: CombinationSpec,
) -> list[dict[str, Any]]:
    index_columns = ["date", "origin", "origin_pos", "horizon"]
    wide = (
        group.pivot_table(
            index=index_columns,
            columns="model",
            values="prediction",
            aggfunc="first",
        )
        .sort_index()
        .astype(float)
    )
    if wide.shape[1] < 1:
        return []
    actual = (
        group.drop_duplicates(index_columns)
        .set_index(index_columns)["actual"]
        .reindex(wide.index)
    )
    combined = _apply_combination_method(wide, actual, spec)
    templates = (
        group.sort_values(["date", "origin_pos", "model"])
        .drop_duplicates(index_columns)
        .set_index(index_columns)
    )
    records: list[dict[str, Any]] = []
    for key, prediction in combined.items():
        template = templates.loc[key].to_dict()
        record = {
            **template,
            "date": key[0],
            "origin": key[1],
            "origin_pos": key[2],
            "horizon": key[3],
            "model": spec.name,
            "model_spec": "forecast_combination",
            "prediction": None if pd.isna(prediction) else float(prediction),
            "variance_prediction": None,
            "quantile_predictions": None,
            "actual": None if pd.isna(actual.loc[key]) else float(actual.loc[key]),
            "params": dict(spec.params),
            "model_selection": None,
            "stored_model": None,
            "combined": True,
            "combination": spec.to_dict(),
        }
        records.append(record)
    return records


def _apply_combination_method(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    spec: CombinationSpec,
) -> pd.Series:
    params = dict(spec.params)
    method = spec.method
    if spec.func is not None:
        output = spec.func(forecasts.copy(), actual=actual.copy(), **params)
        return _coerce_combination_output(output, forecasts.index, name="combined")
    if method == "mean":
        return combine_mean(forecasts)
    if method == "median":
        return combine_median(forecasts)
    if method == "trimmed_mean":
        return combine_trimmed_mean(forecasts, **params)
    if method == "winsorized_mean":
        return combine_winsorized_mean(forecasts, **params)
    _estimated = {"inverse_mspe", "dmspe", "best_n", "bates_granger",
                  "granger_ramanathan", "constrained_ls", "eigenvector", "regularized"}
    if method in _estimated:
        if "horizon" in (forecasts.index.names or []):
            levels = forecasts.index.get_level_values("horizon")
            if len(levels):
                params.setdefault("horizon", int(levels[0]))
    if method in {"inverse_mspe", "dmspe"}:
        return combine_inverse_mspe(forecasts, actual, **params)
    if method == "best_n":
        return combine_best_n(forecasts, actual, **params)
    if method == "bates_granger":
        return combine_bates_granger(forecasts, actual, **params)
    if method == "granger_ramanathan":
        return combine_granger_ramanathan(forecasts, actual, **params)
    if method == "constrained_ls":
        return combine_constrained_ls(forecasts, actual, **params)
    if method == "eigenvector":
        return combine_eigenvector(forecasts, actual, **params)
    if method == "regularized":
        return combine_regularized(forecasts, actual, **params)
    raise ValueError(f"unsupported combination method {method!r}")


def _mapping_to_spec(value: Mapping[str, Any]) -> CombinationSpec:
    method = str(value["method"])
    reserved = {"method", "name", "models", "params", "func", "callable"}
    params = dict(value.get("params", {}))
    params.update({key: item for key, item in value.items() if key not in reserved})
    func = value.get("func", value.get("callable"))
    if _normalize_method(method) == "custom":
        if not callable(func):
            raise TypeError("custom combination mapping requires callable 'func'")
        return custom_combination(
            str(value.get("name") or "custom_combination"),
            func,
            models=value.get("models"),
            **params,
        )
    return combination_spec(
        method,
        name=None if value.get("name") is None else str(value["name"]),
        models=value.get("models"),
        **params,
    )


def _mapping_value_to_spec(alias: Any, value: Any) -> CombinationSpec:
    if value is None:
        return combination_spec(str(alias))
    if isinstance(value, str):
        return combination_spec(value, name=str(alias))
    if isinstance(value, CombinationSpec):
        if value.name == f"combined_{value.method}":
            return CombinationSpec(
                method=value.method,
                name=str(alias),
                models=value.models,
                params=dict(value.params),
                func=value.func,
            )
        return value
    if isinstance(value, Mapping):
        spec = _mapping_to_spec({"name": str(alias), **value})
        return spec
    raise TypeError("mapping values must be strings, specs, mappings, or None")


def _selected_model_rows(frame: pd.DataFrame, spec: CombinationSpec) -> pd.DataFrame:
    base = frame.loc[frame["model"] != spec.name]
    if spec.models is None:
        return base
    return base.loc[base["model"].isin(spec.models)]


def _normalize_method(method: str) -> str:
    key = method.lower().replace("-", "_")
    aliases = {
        "average": "mean",
        "equal_weight": "mean",
        "equal_weighted": "mean",
        "trimmed": "trimmed_mean",
        "winsorized": "winsorized_mean",
        "inverse_mse": "inverse_mspe",
        "inverse_msfe": "inverse_mspe",
        "discounted_mspe": "dmspe",
        "best": "best_n",
        "bates": "bates_granger",
        "bg": "bates_granger",
        "granger_ramanathan": "granger_ramanathan",
        "gr": "granger_ramanathan",
        "regression": "granger_ramanathan",
        "constrained": "constrained_ls",
        "nnls": "constrained_ls",
        "eigen": "eigenvector",
        "pc": "eigenvector",
        "ridge": "regularized",
        "lasso": "regularized",
    }
    key = aliases.get(key, key)
    allowed = {
        "mean",
        "median",
        "trimmed_mean",
        "winsorized_mean",
        "inverse_mspe",
        "dmspe",
        "best_n",
        "bates_granger",
        "granger_ramanathan",
        "constrained_ls",
        "eigenvector",
        "regularized",
        "custom",
    }
    if key not in allowed:
        raise ValueError(f"unknown combination method {method!r}")
    return key


def _forecast_frame(forecasts: Any) -> pd.DataFrame:
    if isinstance(forecasts, pd.Series):
        frame = forecasts.to_frame()
    else:
        frame = pd.DataFrame(forecasts).copy()
    if frame.empty:
        raise ValueError("forecasts must not be empty")
    return frame.astype(float)


def _coerce_combination_output(output: Any, index: pd.Index, *, name: str) -> pd.Series:
    if isinstance(output, pd.Series):
        series = output.copy()
    else:
        values = np.asarray(output, dtype=float).reshape(-1)
        series = pd.Series(values)
    if not series.index.equals(index):
        if len(series) != len(index):
            raise ValueError("custom combination output length does not match forecast rows")
        series.index = index
    return series.astype(float).rename(name)


def _callable_name(func: Callable[..., Any] | None) -> str | None:
    if func is None:
        return None
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


__all__ = [
    "CombinationSpec",
    "apply_combinations",
    "combine_best_n",
    "combine_constrained_ls",
    "combine_regularized",
    "combine_log_pool",
    "combine_linear_pool",
    "combine_eigenvector",
    "combine_granger_ramanathan",
    "combine_bates_granger",
    "combine_dmspe",
    "combine_inverse_mspe",
    "combine_mean",
    "combine_median",
    "combine_trimmed_mean",
    "combine_winsorized_mean",
    "combination_spec",
    "custom_combination",
    "resolve_combinations",
]
