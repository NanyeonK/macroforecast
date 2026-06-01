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

    result = mf.preprocessing.reprocess(
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


def test_legacy_preprocess_alias_is_not_public():
    assert not hasattr(mf, "preprocess")
    assert not hasattr(mf.preprocessing, "preprocess")
    removed_frequency_api = "handle_" + "mixed_frequency"
    assert not hasattr(mf.preprocessing, removed_frequency_api)


def test_preprocess_applies_custom_transform_codes_and_metadata():
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date"), {"dataset": "custom", "source_family": "custom"})

    result = mf.preprocessing.reprocess(
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

    result = mf.preprocessing.reprocess(
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

    result = mf.preprocessing.reprocess(
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
    assert result.steps[6]["method"] == "truncate"


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

    result = mf.preprocessing.reprocess(
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
    assert result.panel.attrs["macroforecast_transform_codes"] == {"target": 1, "x": 1}
    assert result.metadata["transform_codes_applied"] == {"target": 1, "x": 1}


def test_official_transform_requires_a_matching_tcode_map():
    bundle = mf.data.DataBundle(
        mf.data.as_panel(_panel(), date="date"),
        {"dataset": "custom", "source_family": "custom", "frequency": "monthly"},
    )

    with pytest.raises(ValueError, match="transform='official' requires transform_codes"):
        mf.preprocessing.reprocess(bundle, transform="official")


def test_explicit_transform_codes_reject_unknown_columns():
    bundle = mf.data.DataBundle(
        mf.data.as_panel(_panel(), date="date"),
        {"dataset": "custom", "source_family": "custom", "frequency": "monthly"},
    )

    with pytest.raises(ValueError, match="not in the panel"):
        mf.preprocessing.reprocess(
            bundle,
            transform="custom",
            transform_codes={"target": 2, "missing_series": 1},
            outliers="none",
            impute="none",
            frame="keep",
        )


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
        mf.preprocessing.reprocess(panel)


def test_fred_sd_transform_codes_uses_suggestions_and_overrides():
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

    codes = mf.preprocessing.fred_sd_transform_codes(
        panel,
        variable_codes={"CUSTOM": 1},
        state_series_codes={"UR_CA": 1},
    )

    assert codes == {"UR_CA": 1, "ICLAIMS_TX": 5, "CUSTOM_NY": 1}


def test_reprocess_default_pipeline_matches_fred_md_order():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "level": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "diff": [10.0, 11.0, 13.0, 16.0, 20.0, 25.0],
                "diff2": [1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
                "logdiff": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            }
        ),
        date="date",
    )
    metadata = {
        "dataset": "fred_md",
        "frequency": "monthly",
        "transform_codes": {"level": 1, "diff": 2, "diff2": 3, "logdiff": 5},
    }
    bundle = mf.data.DataBundle(panel, metadata)

    result = mf.preprocessing.reprocess(bundle)

    assert [step["step"] for step in result.steps] == [
        "frequency",
        "transform",
        "tcode_lag",
        "outliers",
        "impute",
        "standardize",
        "frame",
    ]
    assert result.steps[2]["rows_removed"] == 2
    assert result.panel.index[0] == pd.Timestamp("2020-03-01")
    assert result.metadata["preprocessing"]["transform"] == "official"
    assert result.metadata["preprocessing"]["outliers"] == "iqr"
    assert result.metadata["preprocessing"]["impute"] == "em_factor"
    assert result.metadata["preprocessing"]["standardize"] == "none"
    assert result.metadata["preprocessing"]["frame"] == "keep"
    assert result.panel.isna().sum().sum() == 0


def test_reprocess_can_standardize_after_imputation():
    metadata = {"dataset": "custom", "source_family": "custom", "frequency": "monthly"}
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)

    result = mf.preprocessing.reprocess(
        bundle,
        transform="none",
        outliers="none",
        impute="mean",
        standardize="zscore",
        frame="keep",
    )

    numeric = result.panel[["target", "x"]]
    assert np.allclose(numeric.mean().to_numpy(), 0.0)
    assert np.allclose(numeric.std(ddof=0).to_numpy(), 1.0)
    stage = result.metadata["preprocessing"]
    assert stage["standardize"] == "zscore"
    assert set(stage["standardization_state"]["columns"]) == {"target", "x"}


def test_reprocess_can_standardize_predictors_only_from_data_spec():
    metadata = {"dataset": "custom", "source_family": "custom", "frequency": "monthly"}
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)
    spec = mf.data.spec(bundle, target="target", horizons=[1], predictors=["x"])

    result = mf.preprocessing.reprocess(
        spec,
        transform="none",
        outliers="none",
        impute="mean",
        standardize="zscore",
        standardize_columns="predictors",
        frame="keep",
    )

    assert np.allclose(result.panel["x"].mean(), 0.0)
    assert np.allclose(result.panel["x"].std(ddof=0), 1.0)
    assert result.panel["target"].tolist() == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert result.metadata["preprocessing"]["standardize_columns"] == ["x"]


def test_preprocess_spec_reuses_train_standardization_state_for_transform():
    metadata = {"dataset": "custom", "source_family": "custom", "frequency": "monthly"}
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "x": [1.0, 2.0, 3.0, 100.0, 101.0, 102.0],
            }
        ),
        date="date",
        metadata=metadata,
    )
    pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="none",
        standardize="zscore",
        frame="keep",
    )

    fitted = pre.fit((panel.iloc[:3], metadata))
    transformed = fitted.transform((panel.iloc[3:4], metadata), history=panel.iloc[:3])

    assert np.isclose(fitted.processed_train.panel["x"].mean(), 0.0)
    assert np.isclose(fitted.processed_train.panel["x"].std(ddof=0), 1.0)
    assert transformed.panel["x"].iloc[0] > 50.0
    assert transformed.metadata["preprocess_transform"]["standardize_refit"] is False


def test_preprocess_spec_preserves_data_spec_choices_and_predictor_scaling():
    metadata = {"dataset": "custom", "source_family": "custom", "frequency": "monthly"}
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)
    spec = mf.data.spec(bundle, target="target", horizons=[1, 3], predictors=["x"])
    pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="mean",
        standardize="zscore",
        standardize_columns="predictors",
        frame="keep",
    )

    fitted = pre.fit(spec)
    transformed = fitted.transform(spec.panel.iloc[-1:])

    assert fitted.processed_train.target == "target"
    assert fitted.processed_train.horizons == (1, 3)
    assert fitted.processed_train.predictors == ("x",)
    assert transformed.target == "target"
    assert transformed.horizons == (1, 3)
    assert transformed.predictors == ("x",)
    assert transformed.metadata["preprocessing"]["standardize"] == "zscore"
    assert transformed.metadata["preprocessing"]["standardize_columns"] == ["x"]
    transform_steps = transformed.metadata["preprocessing"]["steps"]
    standardize_step = next(step for step in transform_steps if step["step"] == "standardize")
    assert standardize_step["method"] == "zscore"
    assert standardize_step["fitted_on"] == "train_window"
    assert fitted.standardization_state is not None
    assert fitted.standardization_state["columns"] == ["x"]


def test_preprocess_spec_fit_window_policy_applies_outlier_and_mean_state():
    metadata = {"dataset": "custom", "source_family": "custom", "frequency": "monthly"}
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "x": [0.0, 1.0, 2.0, 3.0, 100.0],
            }
        ),
        date="date",
        metadata=metadata,
    )
    pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="iqr",
        iqr_threshold=1.5,
        impute="mean",
        frame="keep",
    )

    fitted = pre.fit((panel.iloc[:4], metadata), policy="fit_window")
    transformed = fitted.transform((panel.iloc[4:], metadata), policy="fit_window")

    assert np.isclose(transformed.panel["x"].iloc[0], 1.5)
    assert fitted.to_metadata()["preprocessing_scope"] == "fit_window"
    assert transformed.metadata["preprocess_transform"]["preprocessing_scope"] == "fit_window"
    assert transformed.metadata["preprocessing"]["outlier_state"]["method"] == "iqr"
    assert transformed.metadata["preprocessing"]["impute_state"]["method"] == "mean"
    transform_stage = transformed.metadata["preprocess_transform"]
    assert transform_stage["fit_period"]["end"] == "2020-04-01"
    assert transform_stage["transform_period"]["start"] == "2020-05-01"
    assert transform_stage["output_period"]["n_rows"] == 1


def test_preprocess_spec_rejects_non_preprocessing_options_early():
    with pytest.raises(TypeError, match="unexpected preprocess_spec option"):
        mf.preprocessing.preprocess_spec(transform="none", preprocessing_policy="fit_window")


def test_preprocess_spec_fit_window_rejects_em_imputation():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "x": [0.0, 1.0, np.nan, 3.0, 4.0],
            }
        ),
        date="date",
        metadata={"dataset": "custom", "source_family": "custom", "frequency": "monthly"},
    )
    pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="em_factor",
        frame="keep",
    )

    with pytest.raises(ValueError, match="origin_available"):
        pre.fit(panel, policy="fit_window")


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


def test_linear_interpolation_keeps_leading_and_trailing_edges_missing():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "x": [np.nan, 1.0, np.nan, 3.0, np.nan],
            }
        ),
        date="date",
    )

    imputed = mf.preprocessing.impute_missing(panel, method="linear")

    assert np.isnan(imputed["x"].iloc[0])
    assert imputed["x"].iloc[2] == 2.0
    assert np.isnan(imputed["x"].iloc[-1])


def test_em_multivariate_rejects_all_missing_rows_like_em_factor():
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "a": [1.0, np.nan, 3.0],
                "b": [2.0, np.nan, 4.0],
            }
        ),
        date="date",
    )

    with pytest.raises(ValueError, match="all-missing row"):
        mf.preprocessing.impute_missing(panel, method="em_multivariate")


def test_preprocess_warns_without_data_metadata():
    with pytest.warns(UserWarning, match="macroforecast.data"):
        result = mf.preprocessing.reprocess(
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

    result = mf.preprocessing.reprocess(
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

    result = mf.preprocessing.reprocess(
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


def test_fred_sd_transform_codes_can_return_provenance_table():
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

    dry_run = mf.preprocessing.plan(bundle, transform="official", standardize="zscore", standardize_ddof=1)
    assert dry_run["metadata_warning"] is None
    assert dry_run["steps"] == (
        "frequency",
        "transform",
        "tcode_lag",
        "outliers",
        "impute",
        "standardize",
        "frame",
    )
    assert dry_run["transform"]["applied_codes"] == {"target": 2}
    assert dry_run["standardize"] == "zscore"
    assert dry_run["standardize_ddof"] == 1

    result = mf.preprocessing.reprocess(
        bundle,
        transform="official",
        outliers="none",
        impute="none",
        standardize="zscore",
        frame="keep",
    )
    summary = mf.preprocessing.report(result)

    assert summary["choices"]["transform"] == "official"
    assert summary["choices"]["standardize"] == "zscore"
    assert summary["standardization_state"]["method"] == "zscore"
    assert summary["output_panel"]["n_columns"] == 2


def test_reprocess_returns_preprocessed_data():
    metadata = {
        "dataset": "custom",
        "source_family": "custom",
        "frequency": "monthly",
        "transform_codes": {"target": 2},
    }
    bundle = mf.data.DataBundle(mf.data.as_panel(_panel(), date="date", metadata=metadata), metadata)

    result = mf.preprocessing.reprocess(
        bundle,
        transform="official",
        outliers="none",
        impute="none",
        frame="keep",
    )

    assert isinstance(result, mf.preprocessing.PreprocessedData)
    assert result.metadata["preprocessing"]["transform"] == "official"
