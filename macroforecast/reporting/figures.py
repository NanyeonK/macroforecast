from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_PLOTS_EXTRA_MESSAGE = (
    'macroforecast reporting figures require matplotlib; install it with '
    'pip install "macroforecast[plots]".'
)
_OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#000000",
)
_MASTER_REQUIRED = {"target", "horizon", "actual", "prediction", "contender"}


def cumulative_loss_differential_plot(
    report: Any,
    *,
    target: Any,
    horizon: int,
    benchmark: str | None = None,
    contenders: Sequence[str] | None = None,
    loss: str | Callable[[Any, Any], Any] = "squared",
    ax: Any | None = None,
    shade: Sequence[tuple[Any, Any]] | None = None,
    savefig: str | Path | None = None,
    dpi: int = 300,
) -> Any:
    """Plot cumulative benchmark-minus-contender loss differentials.

    This is the cumulative squared-error differential exhibit when
    ``loss="squared"``. Upward slopes mean the contender is beating the
    benchmark over that interval because benchmark loss is larger than
    contender loss.
    """

    plt = _load_pyplot()
    frame = _filter_master(_coerce_master(report), target=target, horizon=horizon)
    date_column = _date_column(frame)
    resolved_benchmark = _resolve_benchmark(report, frame, target, horizon, benchmark)
    selected = _resolve_contenders(frame, resolved_benchmark, contenders)
    fig, axis, created = _figure_and_axis(plt, ax)
    if created:
        _set_default_cycle(axis)

    plotted = 0
    for contender in selected:
        series = _loss_differential_series(
            frame,
            benchmark=resolved_benchmark,
            contender=contender,
            date_column=date_column,
            loss=loss,
        )
        if series.empty:
            continue
        axis.plot(series.index, series.cumsum(), label=str(contender), linewidth=1.8)
        plotted += 1
    if plotted == 0:
        raise ValueError(
            "no aligned benchmark/contender forecasts are available for the "
            f"target={target!r}, horizon={horizon!r} selection"
        )

    axis.axhline(0.0, color="0.35", linewidth=0.9)
    _shade_intervals(axis, shade)
    axis.set_xlabel("Forecast target date")
    axis.set_ylabel("Cumulative loss differential")
    axis.set_title(f"{target}, h={horizon}: benchmark {resolved_benchmark}")
    _set_y_major_grid(axis)
    axis.legend(frameon=False)
    _save_figure(fig, savefig=savefig, dpi=dpi)
    return axis if ax is not None else fig


def fluctuation_test_plot(
    report: Any,
    *,
    target: Any,
    horizon: int,
    benchmark: str | None = None,
    contender: str | None = None,
    contenders: Sequence[str] | None = None,
    loss: str | Callable[[Any, Any], Any] = "squared",
    window_ratio: float = 0.5,
    dmv_fullsample: bool = True,
    lag_truncate: int | None = None,
    alpha: float = 0.05,
    ax: Any | None = None,
    shade: Sequence[tuple[Any, Any]] | None = None,
    savefig: str | Path | None = None,
    dpi: int = 300,
) -> Any:
    """Plot Giacomini-Rossi rolling fluctuation test paths.

    The statistic path, window size, and critical value are produced by
    :func:`macroforecast.tests.conditional_predictive_ability_test`; this
    function only aligns forecast losses and draws the path with two-sided
    critical-value bands.
    """

    plt = _load_pyplot()
    frame = _filter_master(_coerce_master(report), target=target, horizon=horizon)
    date_column = _date_column(frame)
    resolved_benchmark = _resolve_benchmark(report, frame, target, horizon, benchmark)
    selected = _resolve_contenders(
        frame,
        resolved_benchmark,
        _merge_contender_args(contender=contender, contenders=contenders),
    )
    resolved_lag = (
        min(max(int(horizon) - 1, 0), 5)
        if lag_truncate is None
        else int(lag_truncate)
    )
    fig, axis, created = _figure_and_axis(plt, ax)
    if created:
        _set_default_cycle(axis)

    critical_value: float | None = None
    plotted = 0
    for candidate in selected:
        losses = _aligned_losses(
            frame,
            benchmark=resolved_benchmark,
            contender=candidate,
            date_column=date_column,
            loss=loss,
        )
        if losses.empty:
            continue
        from macroforecast.tests import conditional_predictive_ability_test

        result = conditional_predictive_ability_test(
            losses["contender_loss"],
            losses["benchmark_loss"],
            method="giacomini_rossi",
            window_ratio=window_ratio,
            dmv_fullsample=dmv_fullsample,
            lag_truncate=resolved_lag,
            alpha=alpha,
        )
        path = [float(value) for value in result.get("time_path", [])]
        window_size = int(result.get("window_size") or 0)
        if not path or window_size < 1:
            continue
        x_values = losses.index[window_size - 1 : window_size - 1 + len(path)]
        axis.plot(x_values, path, label=str(candidate), linewidth=1.8)
        critical_value = _coerce_optional_float(result.get("critical_value"))
        plotted += 1

    if plotted == 0:
        raise ValueError(
            "no Giacomini-Rossi fluctuation path could be computed for the "
            f"target={target!r}, horizon={horizon!r} selection"
        )
    if critical_value is not None:
        axis.axhline(critical_value, color="0.35", linestyle="--", linewidth=1.0)
        axis.axhline(-critical_value, color="0.35", linestyle="--", linewidth=1.0)
    axis.axhline(0.0, color="0.55", linewidth=0.8)
    _shade_intervals(axis, shade)
    axis.set_xlabel("Forecast target date")
    axis.set_ylabel("GR fluctuation statistic")
    axis.set_title(f"{target}, h={horizon}: benchmark {resolved_benchmark}")
    _set_y_major_grid(axis)
    axis.legend(frameon=False)
    _save_figure(fig, savefig=savefig, dpi=dpi)
    return axis if ax is not None else fig


def pit_histogram_plot(
    report: Any,
    *,
    target: Any,
    horizon: int,
    model: str,
    bins: int = 10,
    ax: Any | None = None,
    savefig: str | Path | None = None,
    dpi: int = 300,
) -> Any:
    """Plot a probability integral transform histogram with a binomial band."""

    plt = _load_pyplot()
    frame = _filter_master(_coerce_master(report), target=target, horizon=horizon)
    cgroup = frame.loc[frame["contender"].astype(str) == str(model)].copy()
    if cgroup.empty:
        raise ValueError(f"model {model!r} is not present for target={target!r}, horizon={horizon!r}")
    _require_density_columns(cgroup)
    for column in ("actual", "prediction", "variance_prediction"):
        cgroup[column] = pd.to_numeric(cgroup[column], errors="coerce")

    from macroforecast.pipeline.evaluate import _gaussian_pit
    from macroforecast.tests import pit_histogram

    pit = _gaussian_pit(cgroup)
    if pit is None or pit.empty:
        raise ValueError(
            f"model {model!r} has no finite positive variance_prediction rows "
            f"for target={target!r}, horizon={horizon!r}"
        )
    histogram = pit_histogram(pit, n_bins=bins)
    fig, axis, _created = _figure_and_axis(plt, ax)
    centers = (histogram["lower"].to_numpy(float) + histogram["upper"].to_numpy(float)) / 2.0
    widths = histogram["upper"].to_numpy(float) - histogram["lower"].to_numpy(float)
    counts = histogram["count"].to_numpy(float)
    n_obs = float(counts.sum())
    expected = n_obs / float(bins)
    band_half_width = 1.96 * np.sqrt(n_obs * (1.0 / bins) * (1.0 - 1.0 / bins))
    lower = max(0.0, expected - band_half_width)
    upper = expected + band_half_width

    axis.axhspan(lower, upper, color="#56B4E9", alpha=0.18, linewidth=0.0)
    axis.bar(
        centers,
        counts,
        width=widths * 0.9,
        align="center",
        color="#0072B2",
        edgecolor="white",
        linewidth=0.8,
    )
    axis.axhline(expected, color="0.25", linestyle="--", linewidth=1.0)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("PIT")
    axis.set_ylabel("Count")
    axis.set_title(f"{target}, h={horizon}: {model} PIT")
    _set_y_major_grid(axis)
    _save_figure(fig, savefig=savefig, dpi=dpi)
    return axis if ax is not None else fig


def forecast_path_plot(
    report: Any,
    *,
    target: Any,
    horizon: int,
    models: Sequence[str] | None = None,
    start: Any | None = None,
    end: Any | None = None,
    variance_band: str | None = None,
    ax: Any | None = None,
    savefig: str | Path | None = None,
    dpi: int = 300,
) -> Any:
    """Plot actual values and selected forecast paths over target dates."""

    plt = _load_pyplot()
    frame = _filter_master(_coerce_master(report), target=target, horizon=horizon)
    date_column = _date_column(frame)
    frame = _filter_date_window(frame, date_column=date_column, start=start, end=end)
    selected = _resolve_models(frame, models)
    fig, axis, created = _figure_and_axis(plt, ax)
    if created:
        _set_default_cycle(axis)

    actual = (
        frame[[date_column, "actual"]]
        .dropna()
        .groupby(date_column, sort=True)["actual"]
        .first()
    )
    if actual.empty:
        raise ValueError(
            f"no actual values are available for target={target!r}, horizon={horizon!r}"
        )
    axis.plot(
        actual.index,
        actual.to_numpy(float),
        label="Actual",
        color="0.15",
        linewidth=2.0,
    )
    for model_name in selected:
        path = (
            frame.loc[frame["contender"].astype(str) == str(model_name), [date_column, "prediction"]]
            .dropna()
            .groupby(date_column, sort=True)["prediction"]
            .mean()
        )
        if path.empty:
            continue
        axis.plot(
            path.index,
            path.to_numpy(float),
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            label=str(model_name),
        )
        if variance_band is not None and str(model_name) == str(variance_band):
            _draw_variance_band(axis, frame, date_column=date_column, model=model_name)

    axis.set_xlabel("Forecast target date")
    axis.set_ylabel(str(target))
    axis.set_title(f"{target}, h={horizon}: forecast path")
    _set_y_major_grid(axis)
    axis.legend(frameon=False)
    _save_figure(fig, savefig=savefig, dpi=dpi)
    return axis if ax is not None else fig


def _load_pyplot() -> Any:
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(_PLOTS_EXTRA_MESSAGE) from exc


def _coerce_master(report: Any) -> pd.DataFrame:
    if isinstance(report, pd.DataFrame):
        return report.copy()
    forecasts = getattr(report, "forecasts", None)
    if forecasts is not None:
        return pd.DataFrame(forecasts).copy()
    to_frame = getattr(report, "to_frame", None)
    if callable(to_frame):
        return pd.DataFrame(to_frame()).copy()
    if isinstance(report, dict) and "forecasts" in report:
        return pd.DataFrame(report["forecasts"]).copy()
    raise TypeError(
        "report must be a PipelineReport-like object with .forecasts/.to_frame() "
        "or a master forecast DataFrame"
    )


def _filter_master(frame: pd.DataFrame, *, target: Any, horizon: int) -> pd.DataFrame:
    missing = sorted(_MASTER_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"forecast master frame is missing required column(s): {missing}")
    out = frame.loc[(frame["target"] == target) & (frame["horizon"] == horizon)].copy()
    if out.empty:
        raise ValueError(f"no forecasts found for target={target!r}, horizon={horizon!r}")
    return out


def _date_column(frame: pd.DataFrame) -> str:
    for column in ("date", "target_date", "origin"):
        if column in frame.columns:
            return column
    raise ValueError("forecast master frame must contain 'date' or 'origin'")


def _resolve_benchmark(
    report: Any,
    frame: pd.DataFrame,
    target: Any,
    horizon: int,
    benchmark: str | None,
) -> str:
    resolved: str | None
    if benchmark is not None:
        resolved = str(benchmark)
    else:
        resolved = _benchmark_from_report(report)
        if resolved is None:
            resolved = _benchmark_from_accuracy(report, target=target, horizon=horizon)
        if resolved is None and "is_benchmark" in frame.columns:
            values = frame.loc[frame["is_benchmark"].fillna(False), "contender"].dropna().astype(str)
            if not values.empty:
                resolved = str(values.iloc[0])
        if resolved is None:
            contenders = sorted(frame["contender"].dropna().astype(str).unique())
            if not contenders:
                raise ValueError("forecast master frame has no contender values")
            resolved = contenders[0]
    if resolved not in set(frame["contender"].dropna().astype(str)):
        raise ValueError(
            f"benchmark {resolved!r} is not present for target={target!r}, horizon={horizon!r}"
        )
    return resolved


def _benchmark_from_report(report: Any) -> str | None:
    provenance = getattr(report, "provenance", None)
    if isinstance(provenance, dict) and provenance.get("benchmark") is not None:
        return str(provenance["benchmark"])
    spec = getattr(report, "spec", None)
    evaluation = getattr(spec, "evaluation", None)
    benchmark = getattr(evaluation, "benchmark", None)
    if benchmark is not None:
        return str(benchmark)
    return None


def _benchmark_from_accuracy(report: Any, *, target: Any, horizon: int) -> str | None:
    accuracy = getattr(report, "accuracy", None)
    if accuracy is None:
        return None
    frame = pd.DataFrame(accuracy)
    if frame.empty or "contender" not in frame.columns:
        return None
    mask = pd.Series(True, index=frame.index)
    if "target" in frame.columns:
        mask &= frame["target"] == target
    if "horizon" in frame.columns:
        mask &= frame["horizon"] == horizon
    if "is_benchmark" in frame.columns:
        values = frame.loc[mask & frame["is_benchmark"].fillna(False), "contender"].dropna()
        if not values.empty:
            return str(values.iloc[0])
    return None


def _resolve_contenders(
    frame: pd.DataFrame,
    benchmark: str,
    contenders: Sequence[str] | None,
) -> list[str]:
    available = [str(value) for value in frame["contender"].dropna().astype(str).unique()]
    requested = list(contenders) if contenders is not None else available
    selected = [str(value) for value in requested if str(value) != str(benchmark)]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"contender(s) not present in forecast frame: {missing}")
    if not selected:
        raise ValueError("at least one non-benchmark contender is required")
    return selected


def _merge_contender_args(
    *,
    contender: str | None,
    contenders: Sequence[str] | None,
) -> Sequence[str] | None:
    if contender is not None and contenders is not None:
        raise ValueError("pass either contender= or contenders=, not both")
    if contender is not None:
        return (contender,)
    return contenders


def _resolve_models(frame: pd.DataFrame, models: Sequence[str] | None) -> list[str]:
    available = [str(value) for value in frame["contender"].dropna().astype(str).unique()]
    selected = [str(value) for value in (models if models is not None else available)]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"model(s) not present in forecast frame: {missing}")
    if not selected:
        raise ValueError("at least one model is required")
    return selected


def _aligned_losses(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    contender: str,
    date_column: str,
    loss: str | Callable[[Any, Any], Any],
) -> pd.DataFrame:
    actual = frame.groupby(date_column, sort=True)["actual"].first()
    pivot = frame.pivot_table(
        index=date_column,
        columns="contender",
        values="prediction",
        aggfunc="mean",
    )
    if benchmark not in pivot.columns or contender not in pivot.columns:
        return pd.DataFrame(columns=["benchmark_loss", "contender_loss"])
    common = pd.DataFrame(
        {
            "actual": actual,
            "benchmark_prediction": pivot[benchmark],
            "contender_prediction": pivot[contender],
        }
    ).dropna()
    if common.empty:
        return pd.DataFrame(columns=["benchmark_loss", "contender_loss"])
    benchmark_loss = _loss_values(common["actual"], common["benchmark_prediction"], loss)
    contender_loss = _loss_values(common["actual"], common["contender_prediction"], loss)
    return pd.DataFrame(
        {
            "benchmark_loss": benchmark_loss,
            "contender_loss": contender_loss,
        },
        index=common.index,
    )


def _loss_differential_series(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    contender: str,
    date_column: str,
    loss: str | Callable[[Any, Any], Any],
) -> pd.Series:
    losses = _aligned_losses(
        frame,
        benchmark=benchmark,
        contender=contender,
        date_column=date_column,
        loss=loss,
    )
    if losses.empty:
        return pd.Series(dtype=float)
    return losses["benchmark_loss"] - losses["contender_loss"]


def _loss_values(
    actual: Any,
    prediction: Any,
    loss: str | Callable[[Any, Any], Any],
) -> np.ndarray:
    actual_values = np.asarray(actual, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if callable(loss):
        values = np.asarray(loss(actual_values, prediction_values), dtype=float)
        if values.shape == ():
            values = np.full(actual_values.shape, float(values))
        return values.reshape(-1)
    key = str(loss).lower().replace("-", "_")
    error = actual_values - prediction_values
    if key in {"squared", "squared_error", "mse", "quadratic"}:
        return np.square(error)
    if key in {"absolute", "absolute_error", "mae", "abs"}:
        return np.abs(error)
    raise ValueError("loss must be 'squared', 'absolute', or a callable")


def _require_density_columns(frame: pd.DataFrame) -> None:
    if "variance_prediction" not in frame.columns:
        raise ValueError(
            "variance_prediction column is required for a PIT histogram; no arm "
            "in this forecast frame emits a predictive variance (see "
            "macroforecast.forecasting.policies.base._variance_series)."
        )


def _filter_date_window(
    frame: pd.DataFrame,
    *,
    date_column: str,
    start: Any | None,
    end: Any | None,
) -> pd.DataFrame:
    if start is None and end is None:
        return frame
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    return frame.loc[mask].copy()


def _draw_variance_band(axis: Any, frame: pd.DataFrame, *, date_column: str, model: str) -> None:
    if "variance_prediction" not in frame.columns:
        raise ValueError(
            f"variance_band={model!r} requires a variance_prediction column in the forecast frame"
        )
    work = frame.loc[
        frame["contender"].astype(str) == str(model),
        [date_column, "prediction", "variance_prediction"],
    ].copy()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["variance_prediction"] = pd.to_numeric(work["variance_prediction"], errors="coerce")
    band = work.dropna().groupby(date_column, sort=True).mean(numeric_only=True)
    band = band.loc[band["variance_prediction"] > 0.0]
    if band.empty:
        raise ValueError(f"variance_band model {model!r} has no positive variance predictions")
    x_values = band.index
    prediction = band["prediction"].to_numpy(float)
    radius = 1.96 * np.sqrt(band["variance_prediction"].to_numpy(float))
    axis.fill_between(
        x_values,
        prediction - radius,
        prediction + radius,
        color="#0072B2",
        alpha=0.16,
        linewidth=0.0,
        label=f"{model} 95% band",
    )


def _figure_and_axis(plt: Any, ax: Any | None) -> tuple[Any, Any, bool]:
    if ax is not None:
        return ax.figure, ax, False
    fig, axis = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    return fig, axis, True


def _set_default_cycle(axis: Any) -> None:
    axis.set_prop_cycle(color=list(_OKABE_ITO))


def _set_y_major_grid(axis: Any) -> None:
    axis.grid(False, axis="x")
    axis.grid(axis="y", which="major", color="0.90", linewidth=0.8)


def _shade_intervals(axis: Any, shade: Sequence[tuple[Any, Any]] | None) -> None:
    if shade is None:
        return
    for start, end in shade:
        axis.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="0.85", alpha=0.6, linewidth=0.0)


def _save_figure(fig: Any, *, savefig: str | Path | None, dpi: int) -> None:
    if savefig is None:
        return
    path = Path(savefig)
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".png"}:
        raise ValueError("savefig path must end in .pdf or .png")
    kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if suffix == ".png":
        kwargs["dpi"] = int(dpi)
    fig.savefig(path, **kwargs)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


__all__ = [
    "cumulative_loss_differential_plot",
    "fluctuation_test_plot",
    "forecast_path_plot",
    "pit_histogram_plot",
]
