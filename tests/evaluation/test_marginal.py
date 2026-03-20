"""Tests for evaluation/marginal.py — OOS pseudo-R² and marginal contribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.evaluation.marginal import (
    MarginalEffect,
    _Z_95,
    marginal_contribution,
    marginal_contribution_all,
    oos_r2_panel,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_N_DATES = 40
_HORIZONS = [1, 3]
_TARGETS = ["GDP", "CPI"]
_MODELS = ["m1", "m2"]
_FEATURE_SETS = ["F", "F-MARX"]


def _make_result_df(
    seed: int = 42,
    marx_advantage: float = 0.2,
) -> pd.DataFrame:
    """Synthetic result table with controlled F vs F-MARX signal.

    F-MARX systematically produces lower squared errors than F by
    ``marx_advantage`` units of variance, so alpha_MARX should be > 0.

    Columns match ResultSet.to_dataframe() output:
        model_id, feature_set, horizon, forecast_date, y_hat, y_true,
        target_scheme, target
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01", periods=_N_DATES, freq="MS")
    rows = []

    for target in _TARGETS:
        # Common signal for this target
        y_true_arr = rng.standard_normal(_N_DATES)

        for horizon in _HORIZONS:
            for model_id in _MODELS:
                for fs in _FEATURE_SETS:
                    # F-MARX has tighter forecasts than F
                    noise_std = 0.3 if fs == "F-MARX" else (0.3 + marx_advantage)
                    y_hat_arr = y_true_arr + rng.standard_normal(_N_DATES) * noise_std

                    for i, d in enumerate(dates):
                        rows.append(
                            {
                                "model_id": model_id,
                                "feature_set": fs,
                                "horizon": horizon,
                                "forecast_date": d,
                                "y_hat": y_hat_arr[i],
                                "y_true": y_true_arr[i],
                                "target_scheme": "direct",
                                "target": target,
                            }
                        )

    return pd.DataFrame(rows)


def _make_result_with_r2(seed: int = 42) -> pd.DataFrame:
    """Convenience: result_df with oos_r2 column already added."""
    df = _make_result_df(seed=seed)
    return oos_r2_panel(df)


# ---------------------------------------------------------------------------
# Tests for oos_r2_panel
# ---------------------------------------------------------------------------


def test_oos_r2_panel_shape():
    """Output has same number of rows as input and adds oos_r2 column."""
    df = _make_result_df()
    out = oos_r2_panel(df)

    assert out.shape[0] == df.shape[0]
    assert "oos_r2" in out.columns
    # No other columns should be dropped
    for col in df.columns:
        assert col in out.columns


def test_oos_r2_perfect_forecast():
    """A perfect forecast (y_hat == y_true) should yield oos_r2 == 1.0."""
    rng = np.random.default_rng(0)
    n = 30
    y = rng.standard_normal(n)
    df = pd.DataFrame(
        {
            "model_id": "perfect",
            "feature_set": "F",
            "horizon": 1,
            "forecast_date": pd.date_range("2010-01", periods=n, freq="MS"),
            "y_hat": y,
            "y_true": y,
            "target": "GDP",
        }
    )
    out = oos_r2_panel(df)
    # All R² values should be exactly 1
    np.testing.assert_allclose(out["oos_r2"].values, 1.0, atol=1e-12)


def test_oos_r2_mean_forecast():
    """Forecasting the mean gives mean(oos_r2) == 0 over the OOS window.

    The pseudo-R² is defined per-row as 1 - e_t² / sigma²_{v,h}.  When
    y_hat_t = ybar for all t, the average squared error equals sigma²_{v,h}
    (by definition of variance), so the *mean* of the per-row R² values
    is 0.  Individual rows will generally not equal 0.
    """
    rng = np.random.default_rng(1)
    n = 30
    y = rng.standard_normal(n)
    y_mean = np.full(n, y.mean())
    df = pd.DataFrame(
        {
            "model_id": "mean",
            "feature_set": "F",
            "horizon": 1,
            "forecast_date": pd.date_range("2010-01", periods=n, freq="MS"),
            "y_hat": y_mean,
            "y_true": y,
            "target": "GDP",
        }
    )
    out = oos_r2_panel(df)
    # Mean across all rows: mean(1 - e_t²/sigma²) = 1 - MSE/sigma² = 1 - 1 = 0
    # The mean of (y - ybar)² equals the population variance (ddof=0) exactly.
    np.testing.assert_allclose(out["oos_r2"].mean(), 0.0, atol=1e-10)


def test_oos_r2_zero_variance_group_is_nan():
    """Groups with constant y_true should produce NaN R²."""
    n = 20
    df = pd.DataFrame(
        {
            "model_id": "m",
            "feature_set": "F",
            "horizon": 1,
            "forecast_date": pd.date_range("2010-01", periods=n, freq="MS"),
            "y_hat": np.zeros(n),
            "y_true": np.ones(n),  # constant => variance == 0
            "target": "GDP",
        }
    )
    out = oos_r2_panel(df)
    assert out["oos_r2"].isna().all()


def test_oos_r2_panel_multiple_groups():
    """sigma² is computed per (target, horizon) group, not globally."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2010-01", periods=20, freq="MS")
    rows = []
    for target, scale in [("GDP", 1.0), ("CPI", 10.0)]:
        y = rng.standard_normal(20) * scale
        y_hat = y + rng.standard_normal(20) * 0.1 * scale
        for d, yt, yh in zip(dates, y, y_hat):
            rows.append(
                {
                    "model_id": "m",
                    "feature_set": "F",
                    "horizon": 1,
                    "forecast_date": d,
                    "y_hat": yh,
                    "y_true": yt,
                    "target": target,
                }
            )
    df = pd.DataFrame(rows)
    out = oos_r2_panel(df)

    # Both groups should have well-defined R² close to 1 (tight forecasts)
    for target in ["GDP", "CPI"]:
        r2_grp = out.loc[out["target"] == target, "oos_r2"]
        assert r2_grp.notna().all()
        assert r2_grp.mean() > 0.5


# ---------------------------------------------------------------------------
# Tests for marginal_contribution
# ---------------------------------------------------------------------------


def test_marginal_contribution_requires_oos_r2():
    """marginal_contribution raises ValueError if oos_r2 column is absent."""
    df = _make_result_df()  # no oos_r2 column
    with pytest.raises(ValueError, match="oos_r2"):
        marginal_contribution(df, feature="MARX")


def test_marginal_contribution_positive_when_feature_helps():
    """alpha_MARX > 0 when F-MARX consistently beats F."""
    df = _make_result_with_r2(seed=0)
    result = marginal_contribution(df, feature="MARX", hac_bandwidth=3)

    assert not result.empty
    # All estimated alphas should be positive
    assert (result["alpha"] > 0).all(), f"Expected all alpha>0, got:\n{result}"


def test_marginal_contribution_shape():
    """Return DataFrame has all required columns, one row per (model, horizon)."""
    df = _make_result_with_r2()
    result = marginal_contribution(df, feature="MARX", hac_bandwidth=3)

    required_cols = {"feature", "model", "horizon", "alpha", "se", "ci_low", "ci_high", "n_obs"}
    assert required_cols.issubset(set(result.columns))

    # Expect one row per (model_id × horizon) combination that had matched pairs
    expected_cells = len(_MODELS) * len(_HORIZONS)
    assert len(result) == expected_cells, (
        f"Expected {expected_cells} rows, got {len(result)}"
    )


def test_marginal_contribution_ci_consistent():
    """ci_low and ci_high are alpha ± _Z_95 * se."""
    df = _make_result_with_r2()
    result = marginal_contribution(df, feature="MARX", hac_bandwidth=3)

    np.testing.assert_allclose(
        result["ci_low"].values,
        result["alpha"].values - _Z_95 * result["se"].values,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        result["ci_high"].values,
        result["alpha"].values + _Z_95 * result["se"].values,
        rtol=1e-10,
    )


def test_marginal_contribution_feature_column():
    """The feature column should equal the requested feature name."""
    df = _make_result_with_r2()
    result = marginal_contribution(df, feature="MARX", hac_bandwidth=3)
    assert (result["feature"] == "MARX").all()


def test_marginal_contribution_explicit_pairs():
    """Explicitly passing feature_pairs overrides auto-detection."""
    df = _make_result_with_r2()
    result = marginal_contribution(
        df,
        feature="MARX",
        feature_pairs=[("F-MARX", "F")],
        hac_bandwidth=3,
    )
    assert not result.empty
    assert (result["alpha"] > 0).all()


def test_marginal_contribution_no_matching_pairs_returns_empty():
    """When no matching pairs exist, return empty DataFrame with correct columns."""
    df = _make_result_with_r2()
    result = marginal_contribution(
        df,
        feature="MARX",
        feature_pairs=[("NONEXISTENT-A", "NONEXISTENT-B")],
        hac_bandwidth=3,
    )
    assert result.empty
    required_cols = {"feature", "model", "horizon", "alpha", "se", "ci_low", "ci_high", "n_obs"}
    assert required_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Tests for marginal_contribution_all
# ---------------------------------------------------------------------------


def test_marginal_contribution_all_stacks():
    """Result has rows for each feature × model × horizon cell present."""
    df = _make_result_with_r2()
    # Add F feature_set alternative (for "F" feature detection)
    # The synthetic data already has "F" and "F-MARX"; for "MAF" and "F" feature
    # to contribute we need the matching pairs too.
    # With only F and F-MARX in data, only MARX should yield non-empty results.
    result = marginal_contribution_all(df, features=("MARX",), hac_bandwidth=3)

    assert not result.empty
    assert "MARX" in result["feature"].values


def test_marginal_contribution_all_multiple_features():
    """Adding more feature_sets produces rows for multiple features."""
    # Build a richer synthetic df with F, F-MARX, X, F-X, F-MAF, X-MAF
    rng = np.random.default_rng(99)
    dates = pd.date_range("2010-01", periods=_N_DATES, freq="MS")
    feature_sets = ["F", "F-MARX", "X", "F-X", "F-MAF", "X-MAF"]
    rows = []

    y_true_arr = rng.standard_normal(_N_DATES)
    for fs in feature_sets:
        noise = 0.3 + 0.1 * (hash(fs) % 5)
        y_hat_arr = y_true_arr + rng.standard_normal(_N_DATES) * noise
        for i, d in enumerate(dates):
            rows.append(
                {
                    "model_id": "m1",
                    "feature_set": fs,
                    "horizon": 1,
                    "forecast_date": d,
                    "y_hat": y_hat_arr[i],
                    "y_true": y_true_arr[i],
                    "target_scheme": "direct",
                    "target": "GDP",
                }
            )
    df = pd.DataFrame(rows)
    df = oos_r2_panel(df)

    result = marginal_contribution_all(df, features=("MARX", "MAF", "F"), hac_bandwidth=3)

    # All three features should appear
    features_found = set(result["feature"].unique())
    assert "MARX" in features_found
    assert "MAF" in features_found
    assert "F" in features_found


def test_marginal_contribution_all_empty_features_returns_empty():
    """Passing an empty feature list returns an empty DataFrame."""
    df = _make_result_with_r2()
    result = marginal_contribution_all(df, features=(), hac_bandwidth=3)
    assert result.empty


def test_marginal_contribution_all_n_obs_positive():
    """n_obs must be a positive integer for every row in the result."""
    df = _make_result_with_r2()
    result = marginal_contribution_all(df, features=("MARX",), hac_bandwidth=3)
    assert (result["n_obs"] > 0).all()


# ---------------------------------------------------------------------------
# Tests for path_avg feature
# ---------------------------------------------------------------------------


def _make_path_avg_df(seed: int = 7) -> pd.DataFrame:
    """Synthetic result table with both path_average and direct target_scheme rows.

    path_average rows have systematically lower squared errors than direct,
    so alpha_path_avg should be > 0.
    """
    rng = np.random.default_rng(seed)
    n_dates = 40
    dates = pd.date_range("2010-01", periods=n_dates, freq="MS")
    rows = []

    for model_id in ["m1", "m2"]:
        for horizon in [1, 3]:
            y_true_arr = rng.standard_normal(n_dates)
            # direct: noisier forecasts
            y_hat_direct = y_true_arr + rng.standard_normal(n_dates) * 0.5
            # path_average: tighter forecasts (lower sq error)
            y_hat_path = y_true_arr + rng.standard_normal(n_dates) * 0.1

            for i, d in enumerate(dates):
                rows.append(
                    {
                        "model_id": model_id,
                        "feature_set": "F",
                        "horizon": horizon,
                        "forecast_date": d,
                        "y_hat": y_hat_direct[i],
                        "y_true": y_true_arr[i],
                        "target_scheme": "direct",
                        "target": "GDP",
                    }
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "feature_set": "F",
                        "horizon": horizon,
                        "forecast_date": d,
                        "y_hat": y_hat_path[i],
                        "y_true": y_true_arr[i],
                        "target_scheme": "path_average",
                        "target": "GDP",
                    }
                )

    return pd.DataFrame(rows)


def test_marginal_contribution_path_avg_shape():
    """path_avg result has required columns and at least one row."""
    df = _make_path_avg_df()
    df = oos_r2_panel(df)

    result = marginal_contribution(df, feature="path_avg", hac_bandwidth=3)

    required_cols = {"feature", "model", "horizon", "alpha", "se", "ci_low", "ci_high", "n_obs"}
    assert required_cols.issubset(set(result.columns))
    assert not result.empty


def test_marginal_contribution_path_avg_positive_alpha():
    """alpha > 0 when path_average forecasts are systematically more accurate."""
    df = _make_path_avg_df()
    df = oos_r2_panel(df)

    result = marginal_contribution(df, feature="path_avg", hac_bandwidth=3)

    assert not result.empty
    assert (result["alpha"] > 0).all(), f"Expected all alpha>0, got:\n{result}"


def test_marginal_contribution_path_avg_warns_missing_column():
    """A UserWarning is emitted when target_scheme column is absent."""
    df = _make_result_with_r2()  # no target_scheme column in this fixture variant
    # Remove target_scheme to ensure the warning path is exercised
    df = df.drop(columns=["target_scheme"], errors="ignore")

    with pytest.warns(UserWarning, match="target_scheme"):
        result = marginal_contribution(df, feature="path_avg", hac_bandwidth=3)

    assert result.empty
