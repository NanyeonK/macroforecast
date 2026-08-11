from __future__ import annotations

import json

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
    metadata = {"dataset": "fred_sd", "frequency": "mixed"}
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


# --------------------------------------------------------------------------- #
# F-013: Chow-Lin never returns a non-Chow-Lin result
# --------------------------------------------------------------------------- #

def _monthly(periods: int = 6) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=periods, freq="MS")


def test_chow_lin_rejects_a_non_datetime_low_frequency_index() -> None:
    """It used to return ``reindex().bfill().ffill()`` — a repeat-the-nearest-value
    series — from a function whose contract is a regression-distribution estimate.

    The caller got numbers, the docstring's conservation promise did not hold, and
    nothing said the method had not run.
    """
    low = pd.Series([10.0, 20.0], index=[0, 1])
    high = pd.Series(np.arange(6, dtype=float), index=range(6))

    with pytest.raises(TypeError, match="DatetimeIndex"):
        mf.data.chow_lin_disaggregate(low, high)


def test_chow_lin_rejects_a_non_datetime_indicator_index() -> None:
    dates = _monthly(2)
    low = pd.Series([10.0, 20.0], index=dates)
    high = pd.Series(np.arange(6, dtype=float), index=range(6))

    with pytest.raises(TypeError, match="DatetimeIndex"):
        mf.data.chow_lin_disaggregate(low, high)


def test_chow_lin_rejects_an_insufficient_sample() -> None:
    """One usable low-frequency observation cannot identify the regression."""
    dates = _monthly(3)
    one_quarter = pd.Series([10.0], index=pd.DatetimeIndex([dates[-1]]))
    indicator = pd.Series([1.0, 2.0, 3.0], index=dates)

    with pytest.raises(ValueError, match="at least two usable"):
        mf.data.chow_lin_disaggregate(one_quarter, indicator)


def test_chow_lin_refusal_names_the_deterministic_alternative() -> None:
    """A caller who wanted repetition should be told where repetition lives.

    That policy exists and is spelled ``align_frequency(...,
    quarterly_to_monthly='step_forward')`` — the refusal points at it rather than
    quietly being it.
    """
    dates = _monthly(3)
    one_quarter = pd.Series([10.0], index=pd.DatetimeIndex([dates[-1]]))
    indicator = pd.Series([1.0, 2.0, 3.0], index=dates)

    with pytest.raises(ValueError, match="step_forward"):
        mf.data.chow_lin_disaggregate(one_quarter, indicator)


def test_chow_lin_still_conserves_on_valid_input() -> None:
    """The refusals must not cost the working path its aggregation contract."""
    monthly = _monthly(6)
    quarterly = pd.DatetimeIndex(["2020-01-01", "2020-04-01"])
    low = pd.Series([10.0, 20.0], index=quarterly)
    indicator = pd.Series(np.arange(1.0, 7.0), index=monthly)

    result = mf.data.chow_lin_disaggregate(low, indicator, aggregation="mean")

    assert len(result) == len(monthly)
    for period_start, value in low.items():
        window = result.loc[period_start : period_start + pd.DateOffset(months=2)]
        assert window.mean() == pytest.approx(value, rel=1e-6)


# --------------------------------------------------------------------------- #
# F-014: regime missingness and append identity
# --------------------------------------------------------------------------- #

def _regime_bundle(periods: int = 5) -> mf.data.DataBundle:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=periods, freq="MS"),
                "y": np.arange(periods, dtype=float),
                "x": np.arange(periods, dtype=float) + 10.0,
            }
        ),
        date="date",
        metadata={"dataset": "regime", "frequency": "monthly"},
    )
    return mf.data.DataBundle(panel, {"dataset": "regime", "frequency": "monthly"})


def test_missing_threshold_predictor_stays_missing_in_the_regime() -> None:
    """A comparison against NaN is False in pandas, so a missing predictor landed in
    the "not in regime" half — the regime claiming to know a period whose input was
    absent."""
    base = _regime_bundle()
    holey = base.panel.copy()
    holey.loc[holey.index[2], "x"] = np.nan

    result = mf.data.define_regime(holey, column="x", threshold=11.0, append=True)

    assert pd.isna(result.panel.loc[holey.index[2], "regime_regime"])
    assert result.metadata["regimes"]["regime"]["n_observations"] == 4


def test_unmatched_series_values_stay_missing() -> None:
    """``astype(bool)`` made every unmatched row True, because ``bool(nan)`` is True."""
    base = _regime_bundle()
    partial = pd.Series([True], index=base.panel.index[:1])

    result = mf.data.define_regime(base, values=partial, append=True)

    assert result.panel["regime_regime"].iloc[0] == 1.0
    assert result.panel["regime_regime"].iloc[1:].isna().all()
    assert result.metadata["regimes"]["regime"]["n_observations"] == 1


def test_missing_sequence_values_stay_missing() -> None:
    """Same trap one level down: a ``None`` in a plain sequence went through
    ``bool(...)`` and became in-regime."""
    base = _regime_bundle()
    values = [True, None, False, True, np.nan]

    result = mf.data.define_regime(base, values=values, append=True)

    column = result.panel["regime_regime"]
    assert column.iloc[0] == 1.0
    assert pd.isna(column.iloc[1])
    assert column.iloc[2] == 0.0
    assert pd.isna(column.iloc[4])
    assert result.metadata["regimes"]["regime"]["n_observations"] == 3


def test_explicit_dates_remain_fully_observed() -> None:
    """Membership is knowable for every row, so nothing here is missing."""
    base = _regime_bundle()

    result = mf.data.define_regime(base, dates=[base.panel.index[1]], append=True)

    assert result.panel["regime_regime"].notna().all()
    assert result.metadata["regimes"]["regime"]["n_observations"] == len(base.panel)
    assert result.metadata["regimes"]["regime"]["n_regime"] == 1


@pytest.mark.parametrize("existing", ["y", "x"])
def test_append_refuses_to_overwrite_an_existing_column(existing: str) -> None:
    """Appending assigned straight into the panel, so a name that already existed was
    overwritten in place — including a target or a predictor, values gone, nothing
    recorded."""
    base = _regime_bundle()

    with pytest.raises(ValueError, match="already exists"):
        mf.data.define_regime(
            base, values=[True] * len(base.panel), append=True, output_column=existing
        )


def test_serialized_regime_series_describes_observed_values_only() -> None:
    base = _regime_bundle()
    partial = pd.Series([True], index=base.panel.index[:1])

    entry = mf.data.define_regime(base, values=partial).metadata["regimes"]["regime"]

    assert len(entry["series"]) == 1
    assert entry["n_regime"] == 1


# --------------------------------------------------------------------------- #
# F-015: exact lag type and a valid returned panel
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lag", [1.0, 1.9, True, np.bool_(True), "2"])
def test_availability_lag_rejects_non_integral_lags(lag) -> None:
    """``int(lag)`` truncated ``1.9`` to one period and parsed ``"2"`` into two, so the
    information set moved by a different amount than the caller asked for."""
    with pytest.raises((TypeError, ValueError), match="integer"):
        mf.data.availability_lag(_regime_bundle(), lags=lag)


def test_availability_lag_rejects_non_integral_mapping_values() -> None:
    with pytest.raises((TypeError, ValueError), match="integer"):
        mf.data.availability_lag(_regime_bundle(), columns=["x"], lags={"x": 1.9})


def test_availability_lag_stores_numpy_integers_as_plain_int() -> None:
    """Accepting a numpy integer and keeping it leaves a non-JSON-serialisable scalar
    in run metadata."""
    result = mf.data.availability_lag(_regime_bundle(), columns=["x"], lags=np.int64(1))

    stored = result.metadata["data_availability_lag"]["lags"]["x"]
    assert type(stored) is int
    assert json.loads(json.dumps({"lag": stored})) == {"lag": 1}


def test_zero_lag_is_still_accepted() -> None:
    result = mf.data.availability_lag(_regime_bundle(), columns=["x"], lags=0)

    assert result.metadata["data_availability_lag"]["lags"]["x"] == 0


def test_availability_lag_rejects_an_empty_post_drop_panel() -> None:
    """A two-row panel lagged by two keeps nothing, and the empty frame used to be
    returned as a valid bundle that failed later, somewhere else."""
    with pytest.raises(ValueError, match="panel must not be empty"):
        mf.data.availability_lag(_regime_bundle(2), lags=2, drop_missing=True)


@pytest.mark.parametrize("lag", [1.0, 1.9, True, np.bool_(True), "2"])
def test_same_period_predictors_rejects_non_integral_lags(lag) -> None:
    spec = mf.data.spec(_regime_bundle(), target="y", predictors=["x"])

    with pytest.raises((TypeError, ValueError), match="integer"):
        mf.data.same_period_predictors(spec, policy="lag", lag=lag)


def test_same_period_predictors_rejects_an_empty_post_drop_panel() -> None:
    spec = mf.data.spec(_regime_bundle(), target="y", predictors=["x"])

    with pytest.raises(ValueError, match="panel must not be empty"):
        mf.data.same_period_predictors(spec, policy="lag", lag=5, drop_missing=True)


# --------------------------------------------------------------------------- #
# F-016: one strict frequency vocabulary
# --------------------------------------------------------------------------- #

def test_infer_frequencies_rejects_an_unknown_metadata_label() -> None:
    """This module kept a second, permissive copy of the vocabulary that returned an
    unrecognised string unchanged, so ``"montly"`` survived inference and became a
    frequency label nothing downstream could match."""
    base = _regime_bundle()
    panel = base.panel.copy()
    panel.attrs["macroforecast_metadata"] = {
        "dataset": "regime",
        "frequency": "mixed",
        "native_frequency_by_column": {"y": "montly", "x": "monthly"},
    }

    with pytest.raises(ValueError, match="frequency"):
        mf.data.infer_frequencies(panel)


@pytest.mark.parametrize(
    "alias, expected",
    [("m", "monthly"), ("month", "monthly"), ("state_monthly", "monthly"),
     ("q", "quarterly"), ("quarterly", "quarterly"), ("a", "annual"), ("yearly", "annual")],
)
def test_supported_frequency_aliases_still_resolve(alias: str, expected: str) -> None:
    """Sharing the vocabulary must not cost the aliases it already accepted."""
    base = _regime_bundle()
    panel = base.panel.copy()
    panel.attrs["macroforecast_metadata"] = {
        "dataset": "regime",
        "native_frequency_by_column": {"y": alias, "x": alias},
    }

    frequencies, source = mf.data.infer_frequencies(panel)

    assert frequencies["y"] == expected
    assert source == "native_frequency_by_column"


def test_observed_date_inference_is_unchanged_without_metadata() -> None:
    base = _regime_bundle()

    frequencies, source = mf.data.infer_frequencies(base.panel)

    assert source == "observed_dates"
    assert set(frequencies) == {"y", "x"}


def test_numeric_regime_values_keep_ordinary_truthiness() -> None:
    """Observed numeric values are not missing, so they read as truthiness: 0 is out of
    regime, anything else is in. Only rows the caller did not supply are missing."""
    base = _regime_bundle()
    numeric = pd.Series([0.0, 2.0], index=base.panel.index[:2])

    result = mf.data.define_regime(base, name="numeric", values=numeric, append=True)

    assert result.panel["numeric_regime"].iloc[:2].tolist() == [0.0, 1.0]
    assert result.panel["numeric_regime"].iloc[2:].isna().all()
    assert result.metadata["regimes"]["numeric"]["n_regime"] == 1
    assert result.metadata["regimes"]["numeric"]["n_observations"] == 2


def test_regime_missingness_survives_every_flavour_of_missing() -> None:
    """``isinstance(value, float) and pd.isna(value)`` is not enough.

    ``np.float32("nan")`` is not a Python ``float``, and ``bool(pd.NA)`` raises instead
    of returning anything, so a narrower missing test lets both through — one as True,
    the other as an exception.
    """
    base = _regime_bundle()
    sequence = [False, 2, np.float32(np.nan), pd.NA, True]

    result = mf.data.define_regime(base, name="sequence", values=sequence, append=True)

    column = result.panel["sequence_regime"]
    assert column.iloc[[0, 1, 4]].tolist() == [0.0, 1.0, 1.0]
    assert column.iloc[[2, 3]].isna().all()
    assert result.metadata["regimes"]["sequence"]["n_regime"] == 2
    assert result.metadata["regimes"]["sequence"]["n_observations"] == 3


def test_chow_lin_rejects_when_only_one_period_maps() -> None:
    """The design branch, which the length check never reaches.

    ``_chow_lin_design`` returns ``None`` when fewer than TWO low-frequency periods map
    onto the indicator with a finite value -- not only when none do. Here the indicator
    covers Q1 2020 only, so the second observation has nowhere to map and one usable row
    is built. Both series still clear the length check, so this reaches the design branch
    and nothing else, and the refusal has to say "fewer than two" rather than "no period
    maps".
    """
    monthly = pd.date_range("2020-01-01", periods=3, freq="MS")
    one_maps = pd.Series([10.0, 20.0], index=pd.DatetimeIndex(["2020-01-01", "2021-07-01"]))
    indicator = pd.Series([1.0, 2.0, 3.0], index=monthly)

    with pytest.raises(ValueError, match="fewer than two low-frequency periods"):
        mf.data.chow_lin_disaggregate(one_maps, indicator)


def test_chow_lin_design_refusal_does_not_claim_there_is_no_overlap() -> None:
    """A false claim in an error message is a defect in its own right.

    There IS an overlapping period here -- the first observation maps fine. What is
    missing is a second one, which is what the message now says.
    """
    monthly = pd.date_range("2020-01-01", periods=3, freq="MS")
    one_maps = pd.Series([10.0, 20.0], index=pd.DatetimeIndex(["2020-01-01", "2021-07-01"]))
    indicator = pd.Series([1.0, 2.0, 3.0], index=monthly)

    with pytest.raises(ValueError) as exc:
        mf.data.chow_lin_disaggregate(one_maps, indicator)

    message = str(exc.value)
    assert "no low-frequency period maps" not in message
    assert "step_forward" in message
