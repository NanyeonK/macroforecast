from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "target": [1.0, 2.0, 4.0, 8.0, 16.0],
            "x": [10.0, 11.0, np.nan, 13.0, 1000.0],
        }
    )


def test_preprocess_accepts_data_spec_and_preserves_choices():
    metadata = {"dataset": "custom", "source_family": "custom", "frequency": "monthly", "transform_codes": {"target": 2}}
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)
    data_spec = mf.data.spec(bundle, target="target", horizons=[1, 3], predictors=["x"])

    result = mf.preprocessing.preprocess(
        data_spec,
        transform="none",
        outliers="none",
        impute="mean",
        frame="keep",
    )

    assert isinstance(result, mf.preprocessing.PreprocessedData)
    assert result.target == "target"
    assert result.targets == ("target",)
    assert result.horizons == (1, 3)
    assert result.predictors == ("x",)
    assert result.panel.index.name == "date"
    assert result.panel.isna().sum().sum() == 0
    assert "preprocessing" in result.metadata


def test_preprocess_applies_custom_transform_codes_and_metadata():
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date"), {"dataset": "custom", "source_family": "custom"})

    result = mf.preprocessing.preprocess(
        bundle,
        transform="custom",
        transform_codes={"target": 2},
        tcode_lag="keep",
        outliers="none",
        impute="mean",
        frame="keep",
    )

    assert result.panel["target"].iloc[0] == 3.75
    assert result.panel["target"].iloc[1:].tolist() == [1.0, 2.0, 4.0, 8.0]
    stage = result.metadata["preprocessing"]
    assert stage["transform"] == "custom"
    assert stage["steps"][1]["applied"] == {"target": 2}


def test_preprocess_default_handles_tcode_lag_after_tcodes():
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date"), {"dataset": "custom", "source_family": "custom"})

    result = mf.preprocessing.preprocess(
        bundle,
        transform="custom",
        transform_codes={"target": 3, "x": 2},
        outliers="none",
        impute="mean",
        frame="keep",
    )

    assert len(result.panel) == 3
    assert result.steps[2]["step"] == "tcode_lag"
    assert result.steps[2]["method"] == "drop"
    assert result.steps[2]["rows_removed"] == 2


def test_preprocess_dispatches_outlier_impute_and_frame_steps():
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date"), {"dataset": "custom", "source_family": "custom"})

    result = mf.preprocessing.preprocess(
        bundle,
        transform="none",
        outliers="zscore",
        zscore_threshold=1.0,
        impute="forward_fill",
        frame="truncate",
    )

    assert len(result.panel) > 0
    assert result.steps[3]["method"] == "zscore"
    assert result.steps[4]["method"] == "forward_fill"
    assert result.steps[5]["method"] == "truncate"


def test_step_helpers_are_public():
    panel = mf.data.as_panel(_panel(), date="date")

    transformed = mf.preprocessing.apply_transform_codes(panel, {"target": 2})
    assert transformed["target"].isna().iloc[0]

    imputed = mf.preprocessing.impute_missing(transformed, method="mean")
    assert not imputed["target"].isna().any()

    balanced = mf.preprocessing.handle_frame_edges(imputed, method="keep")
    pd.testing.assert_frame_equal(balanced, imputed)


def test_transform_code_overrides_replace_metadata_codes():
    metadata = {"dataset": "custom", "source_family": "custom", "transform_codes": {"target": 2, "x": 1}}
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)

    result = mf.preprocessing.preprocess(
        bundle,
        transform="official",
        transform_code_overrides={"target": 1},
        tcode_lag="keep",
        outliers="none",
        impute="none",
        frame="keep",
    )

    assert result.steps[1]["applied"] == {"target": 1, "x": 1}
    assert result.panel["target"].tolist() == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_fred_sd_requires_explicit_transform_choice():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "UR_CA": [4.0, 4.1, 4.2, 4.3],
            }
        ),
        date="date",
        metadata={"dataset": "fred_sd", "source_family": "fred-sd"},
    )

    with np.testing.assert_raises_regex(ValueError, "FRED-SD has no official t-code map"):
        mf.preprocessing.preprocess(panel)


def test_expand_fred_sd_transform_codes_uses_suggestions_and_overrides():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "UR_CA": [4.0, 4.1, 4.2, 4.3],
                "ICLAIMS_TX": [10.0, 11.0, 12.0, 13.0],
                "CUSTOM_NY": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        date="date",
        metadata={"dataset": "fred_sd", "source_family": "fred-sd"},
    )

    codes = mf.preprocessing.expand_fred_sd_transform_codes(
        panel,
        variable_codes={"CUSTOM": 1},
        state_series_codes={"UR_CA": 1},
    )

    assert codes == {"UR_CA": 1, "ICLAIMS_TX": 5, "CUSTOM_NY": 1}


def test_handle_mixed_frequency_can_align_weekly_and_monthly_to_monthly():
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

    aligned = mf.preprocessing.handle_mixed_frequency(panel, method="monthly", weekly_to_monthly="mean")

    assert list(aligned.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-02-01"]
    assert aligned["weekly"].tolist() == [2.5, 25.0]
    assert aligned["monthly"].tolist() == [100.0, 200.0]


def test_tcode_seven_matches_fred_md_first_difference_of_percent_change():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "x": [1.0, 2.0, 4.0, 10.0],
            }
        ),
        date="date",
    )

    transformed = mf.preprocessing.apply_transform_codes(panel, {"x": 7})

    expected = pd.Series([np.nan, np.nan, 0.0, 0.5], index=panel.index, name="x")
    pd.testing.assert_series_equal(transformed["x"], expected)


def test_tcode_log_guard_matches_fred_md_whole_series_nan():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "x": [1.0, -1.0, 3.0],
            }
        ),
        date="date",
    )

    transformed = mf.preprocessing.apply_transform_codes(panel, {"x": 4})

    assert transformed["x"].isna().all()


def test_em_factor_default_fills_missing_with_baing_path():
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": dates,
                "a": np.linspace(1.0, 12.0, 12),
                "b": np.linspace(2.0, 24.0, 12),
                "c": np.sin(np.arange(12, dtype=float)) + 5.0,
            }
        ),
        date="date",
    )
    panel.iloc[2, 0] = np.nan
    panel.iloc[5, 1] = np.nan

    imputed = mf.preprocessing.impute_missing(panel, method="em_factor")

    assert imputed.shape == panel.shape
    assert imputed.isna().sum().sum() == 0


def test_preprocess_warns_without_data_metadata():
    with pytest.warns(UserWarning, match="macroforecast.data"):
        result = mf.preprocessing.preprocess(
            _panel(),
            transform="none",
            outliers="none",
            impute="none",
            frame="keep",
        )

    assert result.metadata["preprocessing"]["input_panel"]["n_columns"] == 2


def test_preprocess_can_transform_before_frequency():
    metadata = {
        "dataset": "custom",
        "source_family": "custom",
        "frequency": "monthly",
        "transform_codes": {"target": 2},
    }
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)

    result = mf.preprocessing.preprocess(
        bundle,
        frequency="quarterly",
        transform_order="before_frequency",
        transform="official",
        outliers="none",
        impute="none",
        frame="keep",
    )

    assert [step["step"] for step in result.steps[:3]] == ["transform", "tcode_lag", "frequency"]
    assert result.metadata["preprocessing"]["transform_order"] == "before_frequency"


def test_preprocess_stores_inverse_transform_state():
    metadata = {"dataset": "custom", "source_family": "custom", "frequency": "monthly"}
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)

    result = mf.preprocessing.preprocess(
        bundle,
        transform="custom",
        transform_codes={"target": 5},
        tcode_lag="keep",
        outliers="none",
        impute="none",
        frame="keep",
    )

    state = result.metadata["preprocessing"]["transform_state"]["target"]
    assert state["tcode"] == 5
    assert state["requires_log_inverse"] is True
    assert state["lag_count"] == 1
    assert state["last_observed_values"] == [8.0, 16.0]


def test_handle_mixed_frequency_uses_fred_sd_metadata_before_inference():
    metadata = {"dataset": "fred_sd", "source_family": "fred-sd", "frequency": "state_monthly"}
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "Q_CA": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "M_CA": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            }
        ),
        date="date",
        metadata=metadata,
    )
    panel.attrs["macrocast_reports"] = {
        "fred_sd_series_metadata": {
            "series": [
                {"column": "Q_CA", "native_frequency": "quarterly"},
                {"column": "M_CA", "native_frequency": "monthly"},
            ]
        }
    }

    filtered = mf.preprocessing.handle_mixed_frequency(panel, method="drop_non_quarterly")

    assert list(filtered.columns) == ["Q_CA"]


def test_expand_fred_sd_transform_codes_can_return_provenance_table():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "UR_CA": [4.0, 4.1, 4.2, 4.3],
                "CUSTOM_NY": [1.0, 2.0, 3.0, 4.0],
                "UNKNOWN_TX": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        date="date",
        metadata={"dataset": "fred_sd", "source_family": "fred-sd"},
    )

    codes, table = mf.preprocessing.fred_sd_transform_codes(
        panel,
        variable_codes={"CUSTOM": 1},
        state_series_codes={"UR_CA": 2},
        return_table=True,
    )

    assert codes == {"UR_CA": 2, "CUSTOM_NY": 1}
    assert list(table.columns) == ["column", "sd_variable", "state", "tcode", "source", "suggestion_confidence"]
    sources = dict(zip(table["column"], table["source"]))
    assert sources["UR_CA"] == "user_state_series"
    assert sources["CUSTOM_NY"] == "user_variable"
    assert sources["UNKNOWN_TX"] == "unassigned"
    suggestion_confidence = dict(zip(table["column"], table["suggestion_confidence"]))
    assert suggestion_confidence["UR_CA"] == "user"
    assert suggestion_confidence["UNKNOWN_TX"] == "none"


def test_plan_and_report_summarize_preprocessing_choices():
    metadata = {
        "dataset": "custom",
        "source_family": "custom",
        "frequency": "monthly",
        "transform_codes": {"target": 2},
    }
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)

    dry_run = mf.preprocessing.plan(bundle, transform="official")
    assert dry_run["metadata_warning"] is None
    assert dry_run["steps"] == ("frequency", "transform", "tcode_lag", "outliers", "impute", "frame")
    assert dry_run["transform"]["applied_codes"] == {"target": 2}

    result = mf.preprocessing.preprocess(
        bundle,
        transform="official",
        outliers="none",
        impute="none",
        frame="keep",
    )
    summary = mf.preprocessing.report(result)

    assert summary["choices"]["transform"] == "official"
    assert summary["output_panel"]["n_columns"] == 2
