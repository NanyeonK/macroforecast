from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline


class _VarianceFit:
    def __init__(self, level: float):
        self._level = level

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self._level)

    def predict_variance(self, X: Any) -> np.ndarray:
        return np.full(len(X), 0.25)


def _variance_model(X: Any, y: Any) -> _VarianceFit:
    return _VarianceFit(float(np.asarray(y, dtype=float).mean()))


def _bundle(n: int = 96):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05,
            "x1": x,
        },
        index=idx,
    )
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )


def _feats():
    return mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        lags=1,
        target_lags=(0, 1),
    )


@pytest.fixture(scope="module")
def figure_report():
    """Tiny real PipelineReport, matching the cheap pipeline recipe used in
    tests/pipeline/test_density_pipeline.py and tests/pipeline/test_run_pipeline.py."""

    variance_spec = mf.models.ModelSpec(
        name="var_model",
        family="test",
        fit_func=_variance_model,
    )
    spec = pipeline_spec(
        data=_bundle(),
        targets=["y"],
        horizons=[1],
        window=_window(),
        arms=[
            Arm("AR", model="ar", features=_feats(), is_benchmark=True),
            Arm("OLS", model="ols", features=_feats(), nested_in_benchmark=True),
            Arm("VAR_MODEL", model=variance_spec, features=_feats()),
        ],
        evaluation=EvalSpec(
            benchmark="AR",
            metrics=("rmse", "crps"),
            tests=("dm", "gr"),
        ),
        save_models=False,
    )
    return run_pipeline(spec)


def test_cumulative_loss_differential_plot_smoke_and_final_value(
    figure_report: Any,
    tmp_path,
) -> None:
    path = tmp_path / "cssed.png"

    fig = mf.reporting.cumulative_loss_differential_plot(
        figure_report,
        target="y",
        horizon=1,
        benchmark="AR",
        contenders=("OLS",),
        savefig=path,
    )

    assert isinstance(fig, Figure)
    assert path.exists()
    axis = fig.axes[0]
    assert axis.get_xlabel() == "Forecast target date"
    assert axis.get_ylabel() == "Cumulative loss differential"
    assert len(axis.lines) == 2

    forecasts = figure_report.forecasts
    sub = forecasts.loc[(forecasts["target"] == "y") & (forecasts["horizon"] == 1)]
    actual = sub.groupby("date", sort=True)["actual"].first()
    pivot = sub.pivot_table(index="date", columns="contender", values="prediction", aggfunc="mean")
    common = pd.DataFrame({"actual": actual, "AR": pivot["AR"], "OLS": pivot["OLS"]}).dropna()
    expected = np.square(common["actual"] - common["AR"]) - np.square(common["actual"] - common["OLS"])
    assert axis.lines[0].get_ydata()[-1] == pytest.approx(float(expected.sum()))
    plt.close(fig)


def test_fluctuation_test_plot_reuses_public_gr_path(figure_report: Any) -> None:
    fig = mf.reporting.fluctuation_test_plot(
        figure_report,
        target="y",
        horizon=1,
        benchmark="AR",
        contender="OLS",
    )

    assert isinstance(fig, Figure)
    axis = fig.axes[0]
    assert axis.get_ylabel() == "GR fluctuation statistic"
    assert len(axis.lines) == 4

    forecasts = figure_report.forecasts
    sub = forecasts.loc[(forecasts["target"] == "y") & (forecasts["horizon"] == 1)]
    actual = sub.groupby("date", sort=True)["actual"].first()
    pivot = sub.pivot_table(index="date", columns="contender", values="prediction", aggfunc="mean")
    common = pd.DataFrame({"actual": actual, "AR": pivot["AR"], "OLS": pivot["OLS"]}).dropna()
    loss_benchmark = np.square(common["actual"] - common["AR"])
    loss_contender = np.square(common["actual"] - common["OLS"])

    expected = mf.tests.conditional_predictive_ability_test(
        loss_contender,
        loss_benchmark,
        method="giacomini_rossi",
        window_ratio=0.5,
        dmv_fullsample=True,
        lag_truncate=0,
        alpha=0.05,
    )
    np.testing.assert_allclose(axis.lines[0].get_ydata(), expected["time_path"])
    plt.close(fig)


def test_pit_histogram_plot_smoke_and_counts_sum(figure_report: Any) -> None:
    fig = mf.reporting.pit_histogram_plot(
        figure_report,
        target="y",
        horizon=1,
        model="VAR_MODEL",
        bins=5,
    )

    assert isinstance(fig, Figure)
    axis = fig.axes[0]
    assert axis.get_xlabel() == "PIT"
    assert axis.get_ylabel() == "Count"
    heights = [patch.get_height() for patch in axis.containers[0].patches]
    expected_n = int(
        figure_report.forecasts.loc[
            (figure_report.forecasts["target"] == "y")
            & (figure_report.forecasts["horizon"] == 1)
            & (figure_report.forecasts["contender"] == "VAR_MODEL"),
            "variance_prediction",
        ]
        .dropna()
        .gt(0.0)
        .sum()
    )
    assert sum(heights) == expected_n
    plt.close(fig)


def test_forecast_path_plot_smoke_ax_return_and_variance_band(figure_report: Any) -> None:
    fig, ax = plt.subplots()

    got = mf.reporting.forecast_path_plot(
        figure_report,
        target="y",
        horizon=1,
        models=("AR", "VAR_MODEL"),
        variance_band="VAR_MODEL",
        ax=ax,
    )

    assert got is ax
    assert ax.get_xlabel() == "Forecast target date"
    assert ax.get_ylabel() == "y"
    assert len(ax.lines) == 3
    assert len(ax.collections) >= 1
    plt.close(fig)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (
            mf.reporting.cumulative_loss_differential_plot,
            {"target": "y", "horizon": 1},
        ),
        (
            mf.reporting.fluctuation_test_plot,
            {"target": "y", "horizon": 1},
        ),
        (
            mf.reporting.pit_histogram_plot,
            {"target": "y", "horizon": 1, "model": "AR"},
        ),
        (
            mf.reporting.forecast_path_plot,
            {"target": "y", "horizon": 1, "models": ("AR",)},
        ),
    ],
)
def test_figure_functions_raise_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
    func: Callable[..., Any],
    kwargs: dict[str, Any],
) -> None:
    import macroforecast.reporting.figures as figures

    original_import_module = figures.importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "matplotlib.pyplot":
            raise ImportError("missing matplotlib")
        return original_import_module(name, package)

    monkeypatch.setattr(figures.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match='pip install "macroforecast\\[plots\\]"'):
        func(pd.DataFrame(), **kwargs)
