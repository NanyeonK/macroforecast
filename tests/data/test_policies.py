from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _bundle() -> mf.data.DataBundle:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x1": [10.0, 11.0, 12.0, 13.0, 14.0],
                "x2": [20.0, 21.0, 22.0, 23.0, 24.0],
            }
        ),
        date="date",
        metadata={"dataset": "custom", "frequency": "monthly"},
    )
    return mf.data.DataBundle(panel, {"dataset": "custom", "frequency": "monthly"})


def test_availability_lag_delays_selected_columns_and_records_metadata() -> None:
    lagged = mf.data.availability_lag(_bundle(), columns=["x1"], lags=1)

    assert np.isnan(lagged.panel["x1"].iloc[0])
    assert lagged.panel["x1"].iloc[1] == 10.0
    assert lagged.panel["x2"].iloc[1] == 21.0
    assert lagged.metadata["data_availability_lag"]["lags"] == {"x1": 1}


def test_same_period_predictors_can_lag_or_drop_predictors() -> None:
    data_spec = mf.data.spec(_bundle(), target="y", predictors=["x1", "x2"])

    lagged = mf.data.same_period_predictors(data_spec, policy="lag", columns=["x1"])
    dropped = mf.data.same_period_predictors(data_spec, policy="drop", columns=["x2"])

    assert np.isnan(lagged.panel["x1"].iloc[0])
    assert lagged.panel["x1"].iloc[1] == 10.0
    assert dropped.predictors == ("x1",)
    assert "x2" not in dropped.panel.columns


def test_same_period_predictors_forbid_raises_when_predictors_present() -> None:
    data_spec = mf.data.spec(_bundle(), target="y", predictors=["x1"])

    with pytest.raises(ValueError, match="policy='forbid'"):
        mf.data.same_period_predictors(data_spec, policy="forbid")


def test_define_regime_attaches_threshold_regime_metadata_and_optional_column() -> None:
    bundle = mf.data.define_regime(
        _bundle(),
        name="high_x1",
        column="x1",
        threshold=12.0,
        append=True,
    )

    assert "high_x1_regime" in bundle.panel.columns
    assert bundle.panel["high_x1_regime"].iloc[-1] == 1.0
    assert bundle.metadata["regimes"]["high_x1"]["n_regime"] == 2
    assert bundle.metadata["data_regime"]["available_regimes"] == ["high_x1"]


def test_align_frequency_can_align_weekly_and_monthly_to_monthly() -> None:
    dates = pd.to_datetime(
        [
            "2020-01-01",
            "2020-01-08",
            "2020-01-15",
            "2020-01-22",
            "2020-02-01",
            "2020-02-08",
            "2020-02-15",
            "2020-02-22",
        ]
    )
    panel = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    panel["weekly"] = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
    panel["monthly"] = [100.0, np.nan, np.nan, np.nan, 200.0, np.nan, np.nan, np.nan]

    aligned = mf.data.align_frequency(
        panel,
        method="monthly",
        weekly_to_monthly="mean",
    )

    assert list(aligned.panel.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-02-01"]
    assert aligned.panel["weekly"].tolist() == [2.5, 25.0]
    assert aligned.panel["monthly"].tolist() == [100.0, 200.0]
    assert aligned.metadata["frequency"] == "monthly"
    assert aligned.metadata["data_frequency_alignment"]["method"] == "monthly"


def test_align_frequency_warns_when_frequency_is_inferred_unknown() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "sparse": [1.0, np.nan, np.nan],
                "monthly": [10.0, 11.0, 12.0],
            }
        ),
        date="date",
    )

    with pytest.warns(UserWarning, match="unknown columns"):
        aligned = mf.data.align_frequency(panel, method="monthly")

    assert list(aligned.panel.columns) == ["sparse", "monthly"]


def test_align_frequency_quarterly_to_monthly_matches_data_combine() -> None:
    metadata = {"dataset": "fred_sd", "source_family": "fred-sd", "frequency": "state_monthly"}
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "M_CA": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "Q_CA": [np.nan, np.nan, 10.0, np.nan, np.nan, 20.0],
            }
        ),
        date="date",
        metadata=metadata,
    )
    panel.attrs["macrocast_reports"] = {
        "fred_sd_series_metadata": {
            "series": [
                {"column": "M_CA", "native_frequency": "monthly"},
                {"column": "Q_CA", "native_frequency": "quarterly"},
            ]
        }
    }

    aligned = mf.data.align_frequency(
        panel,
        method="monthly",
        quarterly_to_monthly="repeat_within_quarter",
    )
    alias = mf.data.align_frequency(
        panel,
        method="monthly",
        quarterly_to_monthly="step_backward",
    )

    national = mf.data.DataBundle(
        panel[["M_CA"]],
        {"dataset": "monthly_source", "source_family": "test", "frequency": "monthly"},
    )
    regional = mf.data.DataBundle(panel[["Q_CA"]], metadata)
    regional.panel.attrs["macrocast_reports"] = panel.attrs["macrocast_reports"]
    with pytest.warns(UserWarning, match="quarterly variables were aligned to monthly"):
        combined = mf.data.combine(national, regional, dataset="combo", frequency="monthly")

    assert aligned.panel["Q_CA"].tolist() == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    pd.testing.assert_series_equal(aligned.panel["Q_CA"], alias.panel["Q_CA"])
    pd.testing.assert_series_equal(aligned.panel["Q_CA"], combined.panel["Q_CA"])


def test_chow_lin_disaggregate_conserves_low_frequency_mean() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    indicator = pd.Series(np.linspace(1.0, 12.0, 12), index=dates, name="indicator")
    quarterly = (1.0 + 2.0 * indicator.resample("QS").mean()).rename("quarterly")

    disaggregated = mf.data.chow_lin_disaggregate(
        quarterly,
        indicator,
        aggregation="mean",
        rho=0.0,
    )

    pd.testing.assert_index_equal(disaggregated.index, indicator.index)
    reconstructed = disaggregated.resample("QS").mean()
    pd.testing.assert_series_equal(reconstructed, quarterly, check_names=False)
    assert mf.chow_lin_disaggregate is mf.data.chow_lin_disaggregate


def test_align_frequency_supports_chow_lin_quarterly_to_monthly() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    indicator = pd.Series(np.linspace(1.0, 12.0, 12), index=dates)
    quarterly = 1.0 + 2.0 * indicator.resample("QS").mean()
    panel = pd.DataFrame({"monthly": indicator, "quarterly": quarterly.reindex(dates)}, index=dates)
    panel.index.name = "date"
    bundle = mf.data.set_frequencies(
        panel,
        {"monthly": "monthly", "quarterly": "quarterly"},
        frequency="mixed",
    )

    aligned = mf.data.align_frequency(
        bundle,
        method="monthly",
        quarterly_to_monthly="chow_lin",
        chow_lin_indicator="monthly",
        chow_lin_aggregation="mean",
        chow_lin_rho=0.0,
    )

    reconstructed = aligned.panel["quarterly"].resample("QS").mean()
    pd.testing.assert_series_equal(reconstructed, quarterly, check_names=False)
    alignment = aligned.metadata["data_frequency_alignment"]
    assert alignment["quarterly_to_monthly"] == "chow_lin"
    assert alignment["chow_lin_indicator"] == "monthly"


def test_align_frequency_uses_native_metadata_before_observed_inference() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "Q_CA": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "M_CA": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            }
        ),
        date="date",
    )
    bundle = mf.data.set_frequencies(
        panel,
        {"Q_CA": "quarterly", "M_CA": "monthly"},
        frequency="mixed",
    )

    filtered = mf.data.align_frequency(bundle, method="drop_non_quarterly")

    assert list(filtered.panel.columns) == ["Q_CA"]
    assert filtered.metadata["data_frequency_alignment"]["input_frequency_source"] == "native_frequency_by_column"
