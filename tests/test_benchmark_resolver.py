"""Phase 4 - benchmark_resolver behavioural tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.execution.evaluation.benchmark_resolver import (
    BenchmarkResolverError,
    BenchmarkSpec,
    resolve_benchmark_forecasts,
    resolve_benchmark_suite,
)


def _synthetic_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01", periods=n, freq="MS")
    y = pd.Series(np.cumsum(rng.standard_normal(n) * 0.5) + 0.1 * np.arange(n), index=idx, name="y")
    return y, idx


def test_four_variants_produce_different_forecasts():
    y, idx = _synthetic_series()
    origins = idx[60:90]
    specs = (
        BenchmarkSpec(benchmark_model="historical_mean", estimation_window="expanding"),
        BenchmarkSpec(benchmark_model="rolling_mean", estimation_window="rolling", window_len=20),
        BenchmarkSpec(benchmark_model="random_walk"),
        BenchmarkSpec(benchmark_model="ar_bic", max_p=4),
    )
    preds = []
    for spec in specs:
        out = resolve_benchmark_forecasts(target_series=y, horizon=1, spec=spec, train_origins=origins)
        assert len(out) == len(origins)
        preds.append(out["benchmark_pred"].to_numpy())
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            assert not np.allclose(preds[i], preds[j]), f"variants {i} vs {j} should differ"


def test_rolling_window_len_changes_forecasts():
    y, idx = _synthetic_series()
    origins = idx[100:130]
    spec_short = BenchmarkSpec(benchmark_model="rolling_mean", estimation_window="rolling", window_len=20)
    spec_long = BenchmarkSpec(benchmark_model="rolling_mean", estimation_window="rolling", window_len=80)
    out_s = resolve_benchmark_forecasts(target_series=y, horizon=1, spec=spec_short, train_origins=origins)
    out_l = resolve_benchmark_forecasts(target_series=y, horizon=1, spec=spec_long, train_origins=origins)
    assert not np.allclose(out_s["benchmark_pred"], out_l["benchmark_pred"])


def test_resolve_benchmark_suite_stacks_rows():
    y, idx = _synthetic_series()
    origins = idx[60:65]
    suite = (
        BenchmarkSpec(benchmark_model="historical_mean"),
        BenchmarkSpec(benchmark_model="random_walk"),
        BenchmarkSpec(benchmark_model="rolling_mean", estimation_window="rolling", window_len=20),
    )
    out = resolve_benchmark_suite(target_series=y, horizon=1, suite=list(suite), train_origins=origins)
    assert len(out) == len(suite) * len(origins)
    names = set(out["benchmark_name"].unique())
    expected = set(("historical_mean", "random_walk", "rolling_mean"))
    assert names == expected


def test_invalid_benchmark_model_raises():
    y, idx = _synthetic_series(n=100)
    spec = BenchmarkSpec(benchmark_model="not_a_real_benchmark")
    with pytest.raises(BenchmarkResolverError):
        resolve_benchmark_forecasts(target_series=y, horizon=1, spec=spec, train_origins=idx[50:55])


def test_survey_forecast_stub_raises_notimplemented():
    y, idx = _synthetic_series(n=100)
    spec = BenchmarkSpec(benchmark_model="survey_forecast")
    with pytest.raises(NotImplementedError):
        resolve_benchmark_forecasts(target_series=y, horizon=1, spec=spec, train_origins=idx[50:55])


def test_paper_specific_benchmark_stub_raises_notimplemented():
    y, idx = _synthetic_series(n=100)
    spec = BenchmarkSpec(benchmark_model="paper_specific_benchmark")
    with pytest.raises(NotImplementedError):
        resolve_benchmark_forecasts(target_series=y, horizon=1, spec=spec, train_origins=idx[50:55])


def test_multi_benchmark_suite_via_single_dispatch_raises():
    y, idx = _synthetic_series(n=100)
    spec = BenchmarkSpec(benchmark_model="multi_benchmark_suite")
    with pytest.raises(BenchmarkResolverError):
        resolve_benchmark_forecasts(target_series=y, horizon=1, spec=spec, train_origins=idx[50:55])


def test_fixed_estimation_window_returns_constant_train():
    y, idx = _synthetic_series(n=120)
    origins = idx[60:80]
    spec = BenchmarkSpec(benchmark_model="historical_mean", estimation_window="fixed", window_len=40)
    out = resolve_benchmark_forecasts(target_series=y, horizon=1, spec=spec, train_origins=origins)
    # Fixed window means train slice never moves, so all forecasts should equal mean of first 40.
    expected = float(y.iloc[:40].mean())
    assert np.allclose(out["benchmark_pred"], expected)
