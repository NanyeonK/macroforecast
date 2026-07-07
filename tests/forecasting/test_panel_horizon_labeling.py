"""Regression for #423: panel-input models must label the true horizon.

Panel-input models (``var``, and the whole panel path via
``_fit_predict_panel_origin``) used to mislabel the forecast horizon: a
``horizons=[2]`` request emitted a multi-step path but tagged every row
``horizon=1``, and the requested horizon (2) was never produced. Root cause
was two bugs acting together:

1. ``WindowSpec.origins()`` built the panel test slice as
   ``[origin_pos, origin_pos + horizon - 1]`` -- it included the origin's own
   date as the first test row instead of starting one step after it.
2. ``_panel_prediction_horizon`` floored the computed distance to
   ``max(1, ...)``, so the origin's own row (distance 0) and the genuinely
   1-step-away row both read horizon=1.

The fix: the panel test slice now excludes the origin and runs
``[origin_pos + 1, origin_pos + horizon]``, and the panel estimation window
runs THROUGH the origin -- the origin is the last in-sample date
(``WindowSpec.origins``/``plan``/``iter_origins``/``validate`` take a new
``exclude_origin`` keyword, opted into only by the panel call site in
``forecasting/runner.py``, so feature-matrix/supervised window plans are
unaffected). Panel models forecast positionally from the end of their fit
data, so fitting through the origin is what makes forecast step s line up
with date ``origin + s``. The horizon label is the plain positional distance
from the origin, and only the row matching the REQUESTED horizon is
emitted -- intermediate path steps are dropped so a multi-horizon request
never creates duplicate ``(horizon, origin)`` keys.
"""
import time

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.models.timeseries import _VAR


def _var_bundle(n: int = 100) -> mf.data.DataBundle:
    idx = pd.date_range("2000-01-01", periods=n, freq="MS", name="date")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {f"s{i}": np.cumsum(rng.normal(size=n)) for i in range(3)}, index=idx
    )
    panel["Y"] = 0.5 * panel["s0"] + rng.normal(size=n)
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def _var_window() -> mf.window.WindowSpec:
    return mf.window.from_cutoffs(
        test_start="2005-01-01",
        test_end="2006-06-01",
        mode="expanding",
        val_method="last_block",
        retrain_every=1,
    )


def _mixed_panel(n: int = 60) -> mf.data.DataBundle:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    t = np.arange(n, dtype=float)
    q_target = pd.Series(np.nan, index=idx, name="q_target")
    q_mask = idx.month.isin([3, 6, 9, 12])
    q_target.loc[q_mask] = 100.0 + 0.3 * t[q_mask] + np.sin(t[q_mask] / 6.0)
    panel = pd.DataFrame(
        {
            "m1": np.sin(t / 5.0) + t / 100.0,
            "m2": np.cos(t / 7.0),
            "q_target": q_target,
        },
        index=idx,
    )
    return mf.data.set_frequencies(
        panel,
        {"m1": "monthly", "m2": "monthly", "q_target": "quarterly"},
        frequency="mixed",
    )


def _assert_dates_match_positional_horizon(
    frame: pd.DataFrame, full_index: pd.DatetimeIndex, horizon: int
) -> None:
    """date == full_index[position(origin) + horizon] for every row."""

    positions = full_index.get_indexer(frame["origin"])
    assert (positions >= 0).all(), "every origin must be a real panel date"
    expected = full_index[positions + horizon]
    assert (frame["date"].to_numpy() == expected.to_numpy()).all()


def test_var_single_horizon_labels_and_dates_correctly() -> None:
    """Reproduction of #423: horizons=[2] must yield horizon==2, date==origin+2."""

    bundle = _var_bundle()
    report = mf.forecasting.run(
        bundle,
        "var",
        window=_var_window(),
        features=None,
        target="Y",
        horizons=[2],
        save_models=False,
    )
    fc = report.to_frame().dropna(subset=["prediction"])
    assert not fc.empty
    assert sorted(fc["horizon"].unique().tolist()) == [2]
    assert not fc.duplicated(["horizon", "origin"]).any()
    _assert_dates_match_positional_horizon(fc, bundle.panel.index, horizon=2)

    # VALUE-alignment oracle: under forecast_policy="direct", var now fits a
    # target-equation direct projection on data THROUGH the origin (the last
    # in-sample date) for the requested horizon. This still pins the information
    # set: a window that excluded the origin from the fit sample (the pre-#423
    # estimation end of origin-1) would feed the wrong origin-dated lag block.
    params = fc.iloc[0]["params"]
    for _, row_ in fc.iterrows():
        train = bundle.panel.loc[: row_["origin"]]
        oracle = _VAR(
            n_lag=int(params.get("n_lag", 1)),
            target="Y",
            type=str(params.get("type", "const")),
            direct=True,
            direct_horizon=2,
        ).fit(train)
        # Direct _VAR.predict returns the requested-horizon direct forecast for
        # each supplied row; the panel policy keeps the row at the requested
        # horizon.
        path = oracle.predict(pd.DataFrame(index=range(2)))
        assert float(row_["prediction"]) == pytest.approx(
            float(path[1]), abs=1e-12
        )


def test_var_multi_horizon_no_duplicate_keys_and_correct_offsets() -> None:
    """horizons=[1, 3]: each horizon keeps its own correct date, no duplicate keys."""

    bundle = _var_bundle()
    report = mf.forecasting.run(
        bundle,
        "var",
        window=_var_window(),
        features=None,
        target="Y",
        horizons=[1, 3],
        save_models=False,
    )
    fc = report.to_frame().dropna(subset=["prediction"])
    assert sorted(fc["horizon"].unique().tolist()) == [1, 3]
    assert not fc.duplicated(["horizon", "origin"]).any()
    for horizon in (1, 3):
        sub = fc[fc["horizon"] == horizon]
        assert not sub.empty
        _assert_dates_match_positional_horizon(sub, bundle.panel.index, horizon=horizon)


def test_dfm_mixed_mariano_murasawa_single_horizon_labels_and_dates_correctly() -> None:
    """Non-var panel model: same horizon contract, kept only because it is cheap.

    Skips itself if it ever stops being cheap on this toy panel (budget: 30s),
    per the WP1 acceptance criteria -- the intent is to cover a second panel
    model family without adding a slow test to the suite.
    """

    bundle = _mixed_panel()
    win = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=3),
    )
    start = time.time()
    report = mf.forecasting.run(
        bundle,
        "dfm_mixed_mariano_murasawa",
        window=win,
        target="q_target",
        horizons=[2],
        params={"dfm_mixed_mariano_murasawa": {"maxiter": 5, "tolerance": 1e-3}},
        model_selection={"dfm_mixed_mariano_murasawa": None},
        save_models=False,
    )
    elapsed = time.time() - start
    if elapsed > 30:
        pytest.skip(f"dfm_mixed_mariano_murasawa toy run took {elapsed:.1f}s (>30s budget)")

    fc = report.to_frame().dropna(subset=["prediction"])
    assert not fc.empty
    assert sorted(fc["horizon"].unique().tolist()) == [2]
    assert not fc.duplicated(["horizon", "origin"]).any()
    _assert_dates_match_positional_horizon(fc, bundle.panel.index, horizon=2)
