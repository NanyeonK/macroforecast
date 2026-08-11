"""ADD: MASE + seasonal-naive scale + ACF1 (forecast::accuracy parity)."""
from __future__ import annotations
import numpy as np, pandas as pd
import pytest
import macroforecast as mf


def test_mase_matches_definition():
    y_train = pd.Series(np.arange(1, 25, dtype=float))            # naive(m=1) MAE = 1
    y_true = pd.Series([25.0, 26.0, 27.0])
    y_pred = pd.Series([25.5, 25.0, 27.5])                        # |err| = .5,1,.5
    assert mf.metrics.seasonal_naive_mae(y_train, m=1) == 1.0
    assert mf.metrics.mase(y_true, y_pred, y_train, m=1) == (0.5 + 1.0 + 0.5) / 3
    # seasonal scaling uses the m-step naive denominator
    assert mf.metrics.seasonal_naive_mae(y_train, m=12) == 12.0
    assert mf.metrics.mase(y_true, y_pred, y_train, m=12) == ((0.5 + 1.0 + 0.5) / 3) / 12.0


def test_mase_registered_and_error_metric():
    assert mf.metrics.get_metric("mase") is mf.metrics.mase
    # lower-is-better -> ascending ranking (not in higher-is-better set)
    from macroforecast.metrics import _metric_ascending
    assert _metric_ascending("mase") is True


def test_acf1_canonical():
    assert mf.metrics.acf1([1, 2, 3, 4]) == 0.25  # gamma_1/gamma_0 = 1.25/5.0
    assert np.isnan(mf.metrics.acf1([5.0]))


def test_seasonal_naive_mae_keeps_original_time_positions():
    """A gap must not close: dropping NaN first invents an adjacent pair.

    ``[1, NaN, 3]`` at ``m=1`` has pairs (1,0) and (2,1), both invalid, so the
    scale is undefined. Compressing to ``[1, 3]`` would report ``|3 - 1| = 2``
    from two observations that are two periods apart.
    """
    assert np.isnan(mf.metrics.seasonal_naive_mae([1.0, np.nan, 3.0], m=1))
    # every pair invalid -> NaN, not a finite denominator
    assert np.isnan(mf.metrics.seasonal_naive_mae([np.nan, np.nan, np.nan], m=1))
    assert np.isnan(mf.metrics.seasonal_naive_mae([1.0, 2.0], m=5))


def test_seasonal_naive_mae_uses_only_valid_original_position_pairs():
    """With at least one genuine lag-m pair, only those pairs are averaged."""
    # positions: 0=1, 1=NaN, 2=3, 3=NaN, 4=5.  m=2 pairs: (2,0)=2, (3,1)=NaN, (4,2)=2
    assert mf.metrics.seasonal_naive_mae([1.0, np.nan, 3.0, np.nan, 5.0], m=2) == 2.0
    # compressing to [1, 3, 5] and taking m=2 would give |5 - 1| = 4
    assert mf.metrics.seasonal_naive_mae([1.0, np.nan, 3.0, np.nan, 5.0], m=2) != 4.0
    # one valid adjacent pair among gaps
    assert mf.metrics.seasonal_naive_mae([1.0, 4.0, np.nan, 9.0], m=1) == 3.0


def test_seasonal_naive_mae_accepts_pandas_missing_markers():
    """``pd.NA`` must be scored, not raise, and must not close the gap.

    The pre-fix implementation dropped missing values before casting, so
    ``pd.NA`` never reached ``astype(float)``. Removing that compression left the
    cast exposed and a nullable input began raising ``TypeError``. The marker is
    now mapped to NaN in place, so length and order -- and therefore the
    positional pairing -- are preserved.
    """
    assert np.isnan(mf.metrics.seasonal_naive_mae([1.0, pd.NA, 3.0], m=1))
    assert mf.metrics.seasonal_naive_mae([1.0, pd.NA, 3.0, pd.NA, 5.0], m=2) == 2.0

    # every pandas missing marker behaves the same way, including the nullable
    # extension dtypes, and none of them shortens the series
    assert np.isnan(mf.metrics.seasonal_naive_mae([1.0, None, 3.0], m=1))
    assert np.isnan(
        mf.metrics.seasonal_naive_mae(pd.array([1, pd.NA, 3], dtype="Int64"), m=1)
    )
    assert (
        mf.metrics.seasonal_naive_mae(
            pd.array([1.0, pd.NA, 3.0, pd.NA, 5.0], dtype="Float64"), m=2
        )
        == 2.0
    )


def test_seasonal_naive_mae_still_rejects_non_numeric_input():
    """Accepting missing markers must not turn into coercing anything at all."""
    with pytest.raises(ValueError):
        mf.metrics.seasonal_naive_mae(["a", "b", "c"], m=1)
    with pytest.raises(ValueError):
        mf.metrics.seasonal_naive_mae([1.0, "x", 3.0], m=1)


def test_mase_is_nan_when_the_scale_is_undefined():
    """An unusable denominator must not silently become a finite MASE."""
    value = mf.metrics.mase([1.0, 2.0], [1.1, 1.9], [1.0, np.nan, 3.0], m=1)
    assert np.isnan(value)


def test_table_level_mase_is_an_explicit_unsupported_contract():
    """Requesting MASE from the table API names the reason, not a quantile column.

    The forecast table has no training-sample column, so the in-sample naive
    scale cannot be formed from it. Before this contract the request raised a
    ``quantile_predictions is required`` error, and supplying that column made
    the metric vanish from the result instead.
    """
    table = pd.DataFrame(
        {
            "model": ["a", "a"],
            "actual": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        mf.metrics.evaluate_forecasts(table, metrics=("mase",), by=("model",))
    message = str(excinfo.value)
    assert "mase" in message
    assert "training" in message
    assert "quantile" not in message.lower()

    # ... and it stays an error when a quantile column happens to be present,
    # rather than being dropped from the output.
    with_quantiles = table.assign(
        quantile_predictions=[{0.1: 0.9, 0.9: 1.2}, {0.1: 1.8, 0.9: 2.2}]
    )
    with pytest.raises(ValueError, match="no table-level evaluation"):
        mf.metrics.evaluate_forecasts(
            with_quantiles, metrics=("mase",), by=("model",)
        )

    # the metric-family classification is unchanged: MASE is a point metric
    assert mf.metrics.metric_kind("mase") == "point"
