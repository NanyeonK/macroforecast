"""Regression test for the dfm_unrestricted_midas prediction-anchor NaN bug.

2026-07-04 (WP3): ``predict_from_panel`` rebuilt the composite DFM-factor
design at prediction time and hard-rejected any NaN in it. With the default
``factor_lags=(0,)`` and ``target_frequency="quarterly"`` /
``anchor_position="period_end"``, the lag-0 factor lookup date is the
anchor's *enclosing quarter-end* month. On a plain MONTHLY (single-frequency)
panel -- the common case exercised here, and by
``.dev-notes/policy_matrix_scan.py`` -- that projected quarter-end month can
fall past the last date the fitted DFM's own predictors cover (e.g.
forecasting April with h=2 projects the anchor to the June quarter-end), so
``dfm_factor1_lag0`` was NaN and the ``direct``/``direct_average``/
``path_average`` policies all failed with:
    ValueError: DFM unrestricted MIDAS prediction design contains missing
    values; missing columns: ['dfm_factor1_lag0']
This is fixed by extending the fitted DynamicFactorMQ state forward to the
anchor date (via the model's own ``.extend()`` Kalman-filter forecast)
instead of reindexing to NaN. ``recursive`` is unaffected because panel-input
models use their native panel multi-step path under the recursive label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _monthly_panel(n: int = 140) -> mf.data.DataBundle:
    rng = np.random.default_rng(20260703)
    factors = np.cumsum(rng.normal(size=(n, 2)) * 0.3, axis=0)
    loadings = rng.normal(size=(6, 2))
    predictors = factors @ loadings.T + rng.normal(size=(n, 6)) * 0.2
    y = 0.02 * factors[:, 0] - 0.015 * factors[:, 1] + rng.normal(size=n) * 0.05
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    cols = {f"x{i}": predictors[:, i] for i in range(6)}
    cols["Y"] = y
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def _window():
    return mf.window.from_cutoffs(
        test_start="2006-01-01", test_end="2006-04-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )


@pytest.mark.parametrize(
    "policy", ["direct", "direct_average", "path_average"]
)
def test_dfm_unrestricted_midas_prediction_anchor_no_longer_nans(policy) -> None:
    report = mf.forecasting.run(
        _monthly_panel(), "dfm_unrestricted_midas", window=_window(), features=None,
        target="Y", horizons=[2], forecast_policy=policy,
    )
    fc = report.to_frame()

    assert not fc.empty
    assert np.isfinite(fc["prediction"].to_numpy(dtype=float)).all()


def test_dfm_unrestricted_midas_recursive_uses_panel_path() -> None:
    report = mf.forecasting.run(
        _monthly_panel(), "dfm_unrestricted_midas", window=_window(), features=None,
        target="Y", horizons=[2], forecast_policy="recursive",
    )
    fc = report.to_frame()

    assert not fc.empty
    assert set(fc["forecast_policy"]) == {"recursive"}
    assert np.isfinite(fc["prediction"].to_numpy(dtype=float)).all()
