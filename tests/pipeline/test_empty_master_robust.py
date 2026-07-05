"""Robustness: evaluation must not raise on an empty / column-less master.

Regression for the silent-drop bug surfaced in the ML-Useful replication. When a
(target, horizon) cell produces ZERO forecast rows the master frame can be empty
(no rows, hence no columns); ``accuracy_table`` / ``significance_table`` /
``mcs_table`` / ``apply_combinations`` previously raised ``KeyError: 'target'`` on
the ``groupby(["target","horizon"])``. They must instead return empty result
frames with the right columns, and a spec whose cells all produce zero rows must
flow through ``run_pipeline`` without raising while surfacing the empty cells.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, CombinationContender, EvalSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.evaluate import (
    accuracy_table,
    apply_combinations,
    evaluate,
    mcs_table,
    significance_table,
)


def _spec_for_eval():
    """A minimal spec; only its ``evaluation``/``combinations`` are used here."""
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    idx = pd.date_range("2000-01-31", periods=96, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, 96)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x, "x1": x}, index=idx)
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )
    return pipeline_spec(
        data=bundle, targets=["y"], horizons=[1], window=w,
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
        combinations=[CombinationContender(name="POOL", method="mean")],
    )


@pytest.mark.parametrize(
    "master",
    [
        pd.DataFrame(),  # fully empty: no rows, no columns
        pd.DataFrame(columns=["origin", "prediction", "actual"]),  # missing group keys
    ],
    ids=["empty", "missing-keys"],
)
def test_evaluation_tables_do_not_raise_on_empty_master(master):
    """The four group-by tables return empty frames instead of KeyError."""
    spec = _spec_for_eval()

    acc = accuracy_table(master, spec)
    assert isinstance(acc, pd.DataFrame) and acc.empty
    assert {"target", "horizon", "contender"}.issubset(acc.columns)

    sig = significance_table(master, spec)
    assert isinstance(sig, pd.DataFrame) and sig.empty
    assert {"target", "horizon", "contender"}.issubset(sig.columns)

    mcs = mcs_table(master, spec)
    assert isinstance(mcs, pd.DataFrame) and mcs.empty
    assert {"target", "horizon", "contender", "in_mcs"}.issubset(mcs.columns)

    combined = apply_combinations(master, spec)
    assert isinstance(combined, pd.DataFrame) and combined.empty


def test_evaluate_on_empty_master_returns_empty_frames():
    """``evaluate`` (the orchestrator) survives an empty master end to end."""
    spec = _spec_for_eval()
    out = evaluate(pd.DataFrame(), spec)
    assert set(out) == {
        "forecasts", "accuracy", "significance", "mcs", "density", "calibration",
    }
    for key in ("forecasts", "accuracy", "significance", "mcs", "density", "calibration"):
        assert isinstance(out[key], pd.DataFrame)
        assert out[key].empty


def test_run_pipeline_over_zero_row_spec_does_not_raise():
    """A spec whose every cell produces zero rows yields empty eval frames.

    Reproduces the ML-Useful zero-row long-horizon mechanism in miniature: a
    ``log_growth`` direct-average target computed on a series that takes
    non-positive values makes ``log(x)`` NaN, so the h-period-average target is
    all-NaN (``_average_future_path`` uses ``skipna=False``) and every fit row is
    dropped -> ``X_fit`` empty -> zero forecast rows, with NO exception. The
    longer horizon (12) guarantees every averaging window straddles a sign change.
    ``run_pipeline`` must complete, returning empty accuracy/significance/mcs
    frames rather than raising, and must surface the zero-row cell on
    ``empty_cells`` (Task 2 observability).
    """
    n = 96
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    # y oscillates around zero (so log_growth on it is undefined for most windows),
    # exactly the differenced-series condition that bit INDPRO/CPI.
    t = np.arange(n)
    y = np.sin(t / 3.0) * 0.5  # spans negative and positive
    x1 = np.cos(t / 4.0)
    frame = pd.DataFrame({"y": y, "x1": x1}, index=idx)
    # transform_codes=1 (level) -> the panel column is used as-is; the
    # log_growth direct-average TARGET is then built on this signed series.
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    w = mf.window.from_cutoffs(
        test_start=idx[60], horizon=12, mode="expanding",
        val_method="expanding", val_min_train_size=24,
    )
    spec = pipeline_spec(
        data=bundle,
        targets=[mf.pipeline.TargetSpec("y", transform="log_growth", policy="direct_average")],
        horizons=[12], window=w,
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = run_pipeline(spec)  # must NOT raise

    assert report.forecasts.empty
    assert report.accuracy.empty
    assert report.significance.empty
    assert report.mcs.empty

    # Task 2: the zero-row (target, horizon) cell is surfaced, not silent.
    assert report.empty_cells, "empty (target, horizon) cells must be recorded"
    rec = report.empty_cells[0]
    assert rec["target"] == "y" and rec["horizon"] == 12
    assert set(rec["arms"]) == {"AR", "OLS"}
    # mirrored into the leakage audit and a RuntimeWarning emitted
    assert report.leakage_audit.get("empty_cells")
    assert any(
        issubclass(w.category, RuntimeWarning) and "ZERO" in str(w.message)
        for w in caught
    )


def test_run_pipeline_long_horizon_positive_series_produces_rows():
    """Control for the zero-row test: a POSITIVE level series (HOUST analogue)
    yields non-zero rows at the same long horizon, so the empty-cell signal is a
    genuine data condition rather than a horizon bug that drops rows wholesale."""
    n = 96
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    t = np.arange(n)
    # strictly positive level (like HOUST's log-level): log_growth well-defined.
    y = 100.0 + 10.0 * np.sin(t / 6.0) + t * 0.1
    x1 = np.cos(t / 4.0)
    frame = pd.DataFrame({"y": y, "x1": x1}, index=idx)
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    w = mf.window.from_cutoffs(
        test_start=idx[60], horizon=12, mode="expanding",
        val_method="expanding", val_min_train_size=24,
    )
    spec = pipeline_spec(
        data=bundle,
        targets=[mf.pipeline.TargetSpec("y", transform="log_growth", policy="direct_average")],
        horizons=[12], window=w,
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    report = run_pipeline(spec)
    assert not report.forecasts.empty
    assert not report.empty_cells  # positive series -> no zero-row cell
