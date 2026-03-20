"""Tests for evaluation/horserace.py."""

import numpy as np
import pandas as pd
import pytest

from macrocast.evaluation.horserace import (
    HorseRaceResult,
    best_spec_table,
    dm_vs_benchmark_table,
    horserace_summary,
    mcs_membership_table,
    relative_msfe_table,
)


@pytest.fixture()
def synthetic_result_df() -> pd.DataFrame:
    """Synthetic result_df with known MSFE structure.

    Two models: benchmark AR (linear, none) and KRR (nonlinear).
    Two feature_sets: "" (AR) and "F".
    Two horizons: 1, 3.
    20 OOS dates.

    KRR errors are systematically smaller than AR errors so that RMSFE < 1
    for KRR is guaranteed regardless of random draws.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01", periods=20, freq="MS")

    rows = []
    for h in [1, 3]:
        y_true = rng.standard_normal(20) * 0.5
        # AR benchmark: larger errors
        ar_errors = rng.standard_normal(20) * 0.3
        # KRR: smaller errors (better model)
        krr_errors = rng.standard_normal(20) * 0.2

        for i, d in enumerate(dates):
            # AR benchmark with empty feature_set
            rows.append(
                {
                    "model_id": "ar_benchmark",
                    "nonlinearity": "linear",
                    "regularization": "none",
                    "feature_set": "",
                    "horizon": h,
                    "forecast_date": d,
                    "y_true": y_true[i],
                    "y_hat": y_true[i] + ar_errors[i],
                }
            )
            # KRR with "F" feature_set
            rows.append(
                {
                    "model_id": "krr_model",
                    "nonlinearity": "krr",
                    "regularization": "none",
                    "feature_set": "F",
                    "horizon": h,
                    "forecast_date": d,
                    "y_true": y_true[i],
                    "y_hat": y_true[i] + krr_errors[i],
                }
            )

    return pd.DataFrame(rows)


class TestRelativeMsfeTable:
    def test_shape(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = relative_msfe_table(synthetic_result_df)
        assert set(tbl.columns) == {1, 3}
        assert len(tbl) == 2  # 2 (model_id, feature_set) pairs

    def test_index_names(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = relative_msfe_table(synthetic_result_df)
        assert tbl.index.names == ["model_id", "feature_set"]

    def test_benchmark_is_one(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = relative_msfe_table(synthetic_result_df)
        # AR benchmark should have relative MSFE == 1.0 by definition
        assert tbl.loc[("ar_benchmark", ""), 1] == pytest.approx(1.0)
        assert tbl.loc[("ar_benchmark", ""), 3] == pytest.approx(1.0)

    def test_krr_below_one(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = relative_msfe_table(synthetic_result_df)
        # KRR has systematically smaller error scale so RMSFE < 1
        assert tbl.loc[("krr_model", "F"), 1] < 1.0

    def test_explicit_benchmark_id(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = relative_msfe_table(synthetic_result_df, benchmark_id="ar_benchmark")
        assert tbl.loc[("ar_benchmark", ""), 1] == pytest.approx(1.0)

    def test_horizon_subset(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = relative_msfe_table(synthetic_result_df, horizons=[1])
        assert list(tbl.columns) == [1]

    def test_bad_benchmark_raises(self, synthetic_result_df: pd.DataFrame) -> None:
        df = synthetic_result_df.copy()
        # Remove the unique benchmark identifier so auto-detection fails
        df = df[df["model_id"] != "ar_benchmark"]
        with pytest.raises(ValueError):
            relative_msfe_table(df)

    def test_all_values_finite(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = relative_msfe_table(synthetic_result_df)
        assert tbl.notna().all().all()


class TestBestSpecTable:
    def test_shape(self, synthetic_result_df: pd.DataFrame) -> None:
        rmsfe = relative_msfe_table(synthetic_result_df)
        best = best_spec_table(rmsfe)
        assert len(best) == 2  # one row per horizon
        assert set(best["horizon"]) == {1, 3}

    def test_columns(self, synthetic_result_df: pd.DataFrame) -> None:
        rmsfe = relative_msfe_table(synthetic_result_df)
        best = best_spec_table(rmsfe)
        assert list(best.columns) == ["horizon", "model_id", "feature_set", "rmsfe"]

    def test_best_is_krr(self, synthetic_result_df: pd.DataFrame) -> None:
        rmsfe = relative_msfe_table(synthetic_result_df)
        best = best_spec_table(rmsfe)
        # KRR has smaller RMSFE so it should win for both horizons
        h1_row = best[best["horizon"] == 1].iloc[0]
        assert h1_row["model_id"] == "krr_model"

    def test_rmsfe_values_are_positive(self, synthetic_result_df: pd.DataFrame) -> None:
        rmsfe = relative_msfe_table(synthetic_result_df)
        best = best_spec_table(rmsfe)
        assert (best["rmsfe"] > 0).all()


class TestMcsMembershipTable:
    def test_shape(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = mcs_membership_table(synthetic_result_df, n_bootstrap=100)
        assert set(tbl.columns) == {1, 3}
        assert len(tbl) == 2

    def test_index_names(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = mcs_membership_table(synthetic_result_df, n_bootstrap=100)
        assert tbl.index.names == ["model_id", "feature_set"]

    def test_values_are_bool_or_nan(self, synthetic_result_df: pd.DataFrame) -> None:
        # All specs present in all horizons, so all values should be bool (True/False)
        tbl = mcs_membership_table(synthetic_result_df, n_bootstrap=100)
        for col in tbl.columns:
            for val in tbl[col]:
                assert val is True or val is False or val is None

    def test_at_least_one_true_per_horizon(
        self, synthetic_result_df: pd.DataFrame
    ) -> None:
        # The MCS always contains at least one model
        tbl = mcs_membership_table(synthetic_result_df, n_bootstrap=100)
        for h in tbl.columns:
            assert tbl[h].any(), f"No model in MCS for horizon {h}"

    def test_horizon_subset(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = mcs_membership_table(synthetic_result_df, horizons=[1], n_bootstrap=100)
        assert list(tbl.columns) == [1]


class TestDmVsBenchmarkTable:
    def test_shape(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = dm_vs_benchmark_table(synthetic_result_df)
        assert set(tbl.columns) == {1, 3}
        # 2 (model_id, feature_set) pairs
        assert len(tbl) == 2

    def test_index_names(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = dm_vs_benchmark_table(synthetic_result_df)
        assert tbl.index.names == ["model_id", "feature_set"]

    def test_benchmark_is_nan(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = dm_vs_benchmark_table(synthetic_result_df)
        # Benchmark vs itself cannot be tested: should be NaN for all horizons
        assert pd.isna(tbl.loc[("ar_benchmark", ""), 1])
        assert pd.isna(tbl.loc[("ar_benchmark", ""), 3])

    def test_non_benchmark_has_p_value(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = dm_vs_benchmark_table(synthetic_result_df)
        # KRR should have a valid p-value in [0, 1]
        p = tbl.loc[("krr_model", "F"), 1]
        assert not pd.isna(p)
        assert 0.0 <= p <= 1.0

    def test_explicit_benchmark_id(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = dm_vs_benchmark_table(
            synthetic_result_df, benchmark_id="ar_benchmark"
        )
        assert pd.isna(tbl.loc[("ar_benchmark", ""), 1])

    def test_horizon_subset(self, synthetic_result_df: pd.DataFrame) -> None:
        tbl = dm_vs_benchmark_table(synthetic_result_df, horizons=[3])
        assert list(tbl.columns) == [3]


class TestHorseraceSummary:
    def test_returns_horserace_result(self, synthetic_result_df: pd.DataFrame) -> None:
        result = horserace_summary(synthetic_result_df, mcs_alpha=0.10)
        assert isinstance(result, HorseRaceResult)

    def test_tables_not_empty(self, synthetic_result_df: pd.DataFrame) -> None:
        result = horserace_summary(synthetic_result_df, mcs_alpha=0.10)
        assert not result.rmsfe_table.empty
        assert not result.best_specs.empty
        assert not result.mcs_table.empty
        assert not result.dm_table.empty

    def test_consistent_index(self, synthetic_result_df: pd.DataFrame) -> None:
        # rmsfe_table, mcs_table, dm_table should share the same index
        result = horserace_summary(synthetic_result_df, mcs_alpha=0.10)
        assert result.rmsfe_table.index.equals(result.mcs_table.index)
        assert result.rmsfe_table.index.equals(result.dm_table.index)

    def test_consistent_columns(self, synthetic_result_df: pd.DataFrame) -> None:
        result = horserace_summary(synthetic_result_df, mcs_alpha=0.10)
        assert set(result.rmsfe_table.columns) == set(result.mcs_table.columns)
        assert set(result.rmsfe_table.columns) == set(result.dm_table.columns)

    def test_best_specs_has_one_row_per_horizon(
        self, synthetic_result_df: pd.DataFrame
    ) -> None:
        result = horserace_summary(synthetic_result_df, mcs_alpha=0.10)
        assert len(result.best_specs) == len(result.rmsfe_table.columns)

    def test_explicit_benchmark_and_horizons(
        self, synthetic_result_df: pd.DataFrame
    ) -> None:
        result = horserace_summary(
            synthetic_result_df,
            benchmark_id="ar_benchmark",
            horizons=[1],
            mcs_alpha=0.10,
        )
        assert list(result.rmsfe_table.columns) == [1]
        assert len(result.best_specs) == 1
