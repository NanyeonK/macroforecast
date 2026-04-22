from __future__ import annotations

import json
import pytest
from pathlib import Path

import pandas as pd

from macrocast import (
    CompileValidationError,
    DEFAULT_PROFILE_NAME,
    Experiment,
    ExperimentRunResult,
    ExperimentSweepResult,
    clear_custom_extensions,
    compile_recipe_dict,
    custom_model,
    custom_preprocessor,
    forecast,
    get_custom_target_transformer,
    list_custom_target_transformers,
    target_transformer,
)
from macrocast.raw.types import RawArtifactRecord, RawDatasetMetadata, RawLoadResult

FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")
FIXTURE_START = "2000-01"
FIXTURE_END = "2000-10"


def _raw_result(dataset: str, frequency: str, data: pd.DataFrame, transform_codes: dict[str, int] | None = None) -> RawLoadResult:
    return RawLoadResult(
        data=data,
        dataset_metadata=RawDatasetMetadata(
            dataset=dataset,
            source_family=dataset.replace("_", "-"),
            frequency=frequency,
            version_mode="current",
            vintage=None,
            data_through=data.index[-1].strftime("%Y-%m"),
            support_tier="stable",
        ),
        artifact=RawArtifactRecord(
            dataset=dataset,
            version_mode="current",
            vintage=None,
            source_url=f"memory://{dataset}",
            local_path=f"memory://{dataset}",
            file_format="csv" if dataset != "fred_sd" else "xlsx",
            downloaded_at="2026-01-01T00:00:00+00:00",
            file_sha256=dataset,
            file_size_bytes=1,
            cache_hit=True,
            manifest_version="v1",
        ),
        transform_codes=dict(transform_codes or {}),
    )


def test_forecast_result_facade_exposes_common_outputs(tmp_path: Path) -> None:
    result = forecast(
        "fred_md",
        target="INDPRO",
        horizons=[1, 3],
        start=FIXTURE_START,
        end=FIXTURE_END,
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
    )

    assert isinstance(result, ExperimentRunResult)
    assert result.file_path("predictions.csv").is_file()
    assert result.manifest["default_profile"] == DEFAULT_PROFILE_NAME
    assert result.metrics_json["target"] == "INDPRO"
    assert set(result.metrics["horizon"]) == {"h1", "h3"}
    assert "msfe" in result.metrics.columns
    assert "relative_msfe" in result.comparison.columns
    assert "y_pred" in result.predictions.columns
    assert len(result.forecasts) == len(result.predictions)


def test_experiment_mvp_public_contract_single_run(tmp_path: Path) -> None:
    exp = Experiment(
        dataset="fred_md",
        target="INDPRO",
        horizons=[1, 3],
        start=FIXTURE_START,
        end=FIXTURE_END,
    )

    recipe = exp.to_recipe_dict()
    assert recipe["path"]["0_meta"]["leaf_config"]["default_profile"] == DEFAULT_PROFILE_NAME
    assert recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] == "fred_md"
    assert recipe["path"]["1_data_task"]["fixed_axes"]["frequency"] == "monthly"
    assert recipe["path"]["1_data_task"]["leaf_config"]["sample_start_date"] == FIXTURE_START
    assert recipe["path"]["1_data_task"]["leaf_config"]["sample_end_date"] == FIXTURE_END
    assert recipe["path"]["3_training"]["fixed_axes"]["model_family"] == "ar"

    result = exp.run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    assert isinstance(result, ExperimentRunResult)
    assert result.artifact_path.is_dir()
    assert result.file_path("manifest.json").is_file()
    assert result.file_path("predictions.csv").is_file()
    assert result.manifest["default_profile"] == DEFAULT_PROFILE_NAME
    assert set(result.metrics["horizon"]) == {"h1", "h3"}
    assert "y_pred" in result.forecasts.columns


def test_experiment_mvp_public_contract_model_sweep(tmp_path: Path) -> None:
    result = (
        Experiment(
            dataset="fred_md",
            target="INDPRO",
            horizons=[1],
            start=FIXTURE_START,
            end=FIXTURE_END,
        )
        .sweep({"models": ["ridge", "lasso"]})
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )

    assert isinstance(result, ExperimentSweepResult)
    assert result.size == 2
    assert result.successful_count == 2
    assert result.failed_count == 0
    assert set(result.metrics["status"]) == {"success"}
    assert result.variants["3_training.model_family"].nunique() == 2
    comparison = result.compare("msfe")
    assert comparison["msfe"].notna().all()


def test_experiment_mvp_blocks_preprocessing_sweep_until_layer_audit(tmp_path: Path) -> None:
    exp = (
        Experiment(
            dataset="fred_md",
            target="INDPRO",
            horizons=[1],
            start=FIXTURE_START,
            end=FIXTURE_END,
        )
        .sweep({"scaling": ["none", "standard"]})
    )

    with pytest.raises(CompileValidationError, match="preprocessing sweeps are not executable"):
        exp.run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)


def test_experiment_mvp_requires_explicit_sample_period() -> None:
    with pytest.raises(ValueError, match="start is required"):
        Experiment(dataset="fred_md", target="INDPRO", start="", end=FIXTURE_END).to_recipe_dict()

    with pytest.raises(ValueError, match="end is required"):
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end="").to_recipe_dict()


def test_forecast_default_runs_and_records_default_profile(tmp_path: Path) -> None:
    result = forecast(
        "fred_md",
        target="INDPRO",
        horizons=[1, 3],
        start=FIXTURE_START,
        end=FIXTURE_END,
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
    )

    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert manifest["default_profile"] == DEFAULT_PROFILE_NAME
    assert manifest["compiler"]["leaf_config"]["default_profile"] == DEFAULT_PROFILE_NAME
    assert manifest["model_spec"]["model_family"] == "ar"
    assert manifest["benchmark_name"] == "zero_change"
    assert manifest["evaluation_spec"]["primary_metric"] == "msfe"
    assert manifest["data_task_spec"]["sample_start_date"] == FIXTURE_START
    assert manifest["data_task_spec"]["sample_end_date"] == FIXTURE_END
    assert manifest["preprocess_contract"]["tcode_policy"] == "tcode_only"
    assert manifest["preprocess_contract"]["target_transform_policy"] == "tcode_transformed"
    assert manifest["preprocess_contract"]["x_transform_policy"] == "dataset_tcode_transformed"
    assert manifest["data_reports"]["tcode"]["applied"] is True
    assert manifest["data_reports"]["availability"]["target_leading_missing"]


def test_experiment_to_recipe_dict_uses_model_sweep_for_compare_models() -> None:
    recipe = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1, 3])
        .compare_models(["ar", "ridge"])
        .to_recipe_dict()
    )

    assert recipe["path"]["0_meta"]["fixed_axes"]["research_design"] == "controlled_variation"
    assert recipe["path"]["3_training"]["sweep_axes"]["model_family"] == ["ar", "ridge"]
    assert "model_family" not in recipe["path"]["3_training"]["fixed_axes"]
    assert recipe["path"]["1_data_task"]["fixed_axes"]["frequency"] == "monthly"
    assert recipe["path"]["1_data_task"]["leaf_config"]["sample_start_date"] == FIXTURE_START
    assert recipe["path"]["1_data_task"]["leaf_config"]["sample_end_date"] == FIXTURE_END


def test_fred_sd_requires_frequency_when_used_alone() -> None:
    with pytest.raises(ValueError, match="frequency is required"):
        Experiment(
            dataset="fred_sd",
            target="CAUR",
            start="2000-01",
            end="2001-12",
            horizons=[1],
        ).to_recipe_dict()


def test_fred_md_and_qd_resolve_dataset_frequency() -> None:
    md_recipe = Experiment(
        dataset="fred_md",
        target="INDPRO",
        start=FIXTURE_START,
        end=FIXTURE_END,
        horizons=[1],
    ).to_recipe_dict()
    qd_recipe = Experiment(dataset="fred_qd", target="GDPC1", start="2000-01", end="2002-12", horizons=[1]).to_recipe_dict()

    assert md_recipe["path"]["1_data_task"]["fixed_axes"]["frequency"] == "monthly"
    assert qd_recipe["path"]["1_data_task"]["fixed_axes"]["frequency"] == "quarterly"


def test_fred_sd_combination_uses_md_or_qd_frequency() -> None:
    md_recipe = Experiment(
        dataset="fred_sd+fred_md",
        target="INDPRO",
        start=FIXTURE_START,
        end=FIXTURE_END,
        horizons=[1],
    ).to_recipe_dict()
    qd_recipe = Experiment(
        dataset="fred_sd+fred_qd",
        target="GDPC1",
        start="2000-01",
        end="2002-12",
        horizons=[1],
    ).to_recipe_dict()

    assert md_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] == "fred_md+fred_sd"
    assert md_recipe["path"]["1_data_task"]["fixed_axes"]["frequency"] == "monthly"
    assert qd_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] == "fred_qd+fred_sd"
    assert qd_recipe["path"]["1_data_task"]["fixed_axes"]["frequency"] == "quarterly"

    with pytest.raises(ValueError, match="fred_md and fred_qd cannot be combined"):
        Experiment(dataset="fred_md+fred_qd", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END).to_recipe_dict()

    with pytest.raises(ValueError, match="conflicts"):
        Experiment(
            dataset="fred_sd+fred_qd",
            target="GDPC1",
            start="2000-01",
            end="2002-12",
            frequency="monthly",
        ).to_recipe_dict()


def test_fred_qd_with_sd_converts_monthly_state_data_to_quarterly(monkeypatch, tmp_path: Path) -> None:
    q_dates = pd.date_range("2000-01-01", periods=10, freq="QS")
    qd_frame = pd.DataFrame(
        {
            "GDPC1": [100.0 + i for i in range(len(q_dates))],
            "QX": [10.0 + i for i in range(len(q_dates))],
        },
        index=q_dates,
    )
    qd_frame.index.name = "date"
    m_dates = pd.date_range("2000-01-01", periods=30, freq="MS")
    sd_frame = pd.DataFrame({"STATE_CA": [float(i + 1) for i in range(len(m_dates))]}, index=m_dates)
    sd_frame.index.name = "date"

    monkeypatch.setattr(
        "macrocast.execution.build.load_fred_qd",
        lambda **_: _raw_result("fred_qd", "quarterly", qd_frame, {"GDPC1": 1, "QX": 1}),
    )
    monkeypatch.setattr(
        "macrocast.execution.build.load_fred_sd",
        lambda **_: _raw_result("fred_sd", "state_monthly", sd_frame),
    )

    result = forecast(
        "fred_sd+fred_qd",
        target="GDPC1",
        start="2000-01",
        end="2002-06",
        horizons=[1],
        output_root=tmp_path,
    )

    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert manifest["raw_dataset"] == "fred_qd+fred_sd"
    assert manifest["data_task_spec"]["frequency"] == "quarterly"
    assert manifest["data_reports"]["combined_dataset"]["components"] == ["fred_qd", "fred_sd"]
    sd_report = manifest["data_reports"]["components"]["fred_sd"]["frequency_conversion"]
    assert sd_report["method"] == "monthly_to_quarterly_3_month_average"

    preview = pd.read_csv(Path(result.artifact_dir) / "data_preview.csv", index_col=0)
    assert preview.loc["2000-01-01", "STATE_CA"] == pytest.approx(2.0)


def test_sd_inferred_tcodes_are_opt_in_for_composite_dataset(monkeypatch, tmp_path: Path) -> None:
    q_dates = pd.date_range("2000-01-01", periods=10, freq="QS")
    qd_frame = pd.DataFrame(
        {
            "GDPC1": [100.0 + i for i in range(len(q_dates))],
            "QX": [10.0 + i for i in range(len(q_dates))],
        },
        index=q_dates,
    )
    qd_frame.index.name = "date"
    m_dates = pd.date_range("2000-01-01", periods=30, freq="MS")
    sd_frame = pd.DataFrame(
        {
            "BPPRIVSA_CA": [100.0 + float(i) for i in range(len(m_dates))],
            "UR_CA": [5.0 + 0.1 * float(i) for i in range(len(m_dates))],
        },
        index=m_dates,
    )
    sd_frame.index.name = "date"

    monkeypatch.setattr(
        "macrocast.execution.build.load_fred_qd",
        lambda **_: _raw_result("fred_qd", "quarterly", qd_frame, {"GDPC1": 1, "QX": 1}),
    )
    monkeypatch.setattr(
        "macrocast.execution.build.load_fred_sd",
        lambda **_: _raw_result("fred_sd", "state_monthly", sd_frame),
    )

    result = (
        Experiment(
            dataset="fred_sd+fred_qd",
            target="GDPC1",
            start="2000-01",
            end="2002-06",
            horizons=[1],
        )
        .use_sd_inferred_tcodes()
        .run(output_root=tmp_path)
    )

    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    report = manifest["data_reports"]["sd_inferred_tcodes"]
    assert report["official"] is False
    assert report["map_version"] == "sd-analog-v0.1"
    assert report["frequency"] == "quarterly"
    assert report["applied"]["BPPRIVSA_CA"] == 5
    assert report["applied"]["UR_CA"] == 2
    assert manifest["data_reports"]["tcode"]["columns"]["BPPRIVSA_CA"] == 5
    assert manifest["data_reports"]["tcode"]["columns"]["UR_CA"] == 2
    assert any("not official FRED-SD" in warning for warning in manifest["data_warnings"])


def test_sd_inferred_tcodes_are_not_applied_by_default(monkeypatch, tmp_path: Path) -> None:
    q_dates = pd.date_range("2000-01-01", periods=10, freq="QS")
    qd_frame = pd.DataFrame({"GDPC1": [100.0 + i for i in range(len(q_dates))]}, index=q_dates)
    qd_frame.index.name = "date"
    m_dates = pd.date_range("2000-01-01", periods=30, freq="MS")
    sd_frame = pd.DataFrame({"BPPRIVSA_CA": [100.0 + float(i) for i in range(len(m_dates))]}, index=m_dates)
    sd_frame.index.name = "date"

    monkeypatch.setattr(
        "macrocast.execution.build.load_fred_qd",
        lambda **_: _raw_result("fred_qd", "quarterly", qd_frame, {"GDPC1": 1}),
    )
    monkeypatch.setattr(
        "macrocast.execution.build.load_fred_sd",
        lambda **_: _raw_result("fred_sd", "state_monthly", sd_frame),
    )

    result = forecast(
        "fred_sd+fred_qd",
        target="GDPC1",
        start="2000-01",
        end="2002-06",
        horizons=[1],
        output_root=tmp_path,
    )

    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert "sd_inferred_tcodes" not in manifest["data_reports"]
    assert manifest["data_reports"]["tcode"]["columns"]["BPPRIVSA_CA"] == 1


def test_experiment_use_sd_inferred_tcodes_lowers_to_leaf_config() -> None:
    recipe = (
        Experiment(dataset="fred_sd+fred_qd", target="GDPC1", start="2000-01", end="2002-06", horizons=[1])
        .use_sd_inferred_tcodes(statuses=["tentative_accept"])
        .to_recipe_dict()
    )

    leaf = recipe["path"]["2_preprocessing"]["leaf_config"]
    assert leaf["sd_tcode_policy"] == "inferred_v0_1"
    assert leaf["sd_tcode_map_version"] == "sd-analog-v0.1"
    assert leaf["sd_tcode_allowed_statuses"] == ["tentative_accept"]


def test_experiment_sweep_alias_maps_to_internal_axis() -> None:
    recipe = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1])
        .sweep({"scaling": ["none", "standard"]})
        .to_recipe_dict()
    )

    assert recipe["path"]["0_meta"]["fixed_axes"]["research_design"] == "controlled_variation"
    assert recipe["path"]["2_preprocessing"]["sweep_axes"]["scaling_policy"] == ["none", "standard"]
    assert "scaling_policy" not in recipe["path"]["2_preprocessing"]["fixed_axes"]


def test_experiment_compare_models_runs_sweep(tmp_path: Path) -> None:
    result = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1])
        .compare_models(["ridge", "lasso"])
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )

    assert isinstance(result, ExperimentSweepResult)
    assert result.successful_count == 2
    assert result.failed_count == 0
    assert result.size == 2
    assert result.manifest["research_design"] == "controlled_variation"
    assert result.predictions["variant_id"].nunique() == 2
    assert set(result.metrics["status"]) == {"success"}
    assert "msfe" in result.metrics.columns
    assert "3_training.model_family" in result.metrics.columns
    comparison = result.compare("msfe")
    assert comparison["msfe"].notna().all()
    variant = result.variant(result.per_variant_results[0].variant_id)
    assert "y_pred" in variant.predictions.columns


def test_custom_model_runs_as_single_experiment(tmp_path: Path) -> None:
    clear_custom_extensions()

    @custom_model("last_lag_custom")
    def last_lag_custom(X_train, y_train, X_test, context):
        return X_test[0, 0]

    result = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1])
        .compare_models(["last_lag_custom"])
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )

    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "last_lag_custom"
    assert manifest["model_spec"]["custom_model"] is True
    assert manifest["forecast_engine"] == "custom_model:last_lag_custom:autoreg_lagged_target_v0"


def test_custom_model_can_compare_with_builtin_model(tmp_path: Path) -> None:
    clear_custom_extensions()

    @custom_model("mean_custom")
    def mean_custom(X_train, y_train, X_test, context):
        return float(y_train.mean())

    result = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1])
        .compare_models(["ridge", "mean_custom"])
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )

    assert result.successful_count == 2
    assert result.failed_count == 0


def test_custom_preprocessor_runs_as_x_only_extension(tmp_path: Path) -> None:
    clear_custom_extensions()

    @custom_preprocessor("center_x")
    def center_x(X_train, y_train, X_test, context):
        location = X_train.mean(axis=0)
        return X_train - location, X_test - location

    result = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1], model_family="ridge")
        .use_preprocessor("center_x")
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )

    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert manifest["training_spec"]["custom_preprocessor"] == "center_x"
    assert "custom_preprocessor_transforms_y" not in manifest["training_spec"]
    assert "custom_preprocessor_prediction_scale" not in manifest["training_spec"]


def test_custom_preprocessor_alias_lowers_to_axis() -> None:
    recipe = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1])
        .sweep({"preprocessor": ["one", "two"]})
        .to_recipe_dict()
    )
    assert recipe["path"]["2_preprocessing"]["sweep_axes"]["custom_preprocessor"] == ["one", "two"]


def test_target_transformer_runs_autoreg_path_on_raw_forecast_scale(tmp_path: Path) -> None:
    clear_custom_extensions()

    @target_transformer("standardize_y")
    class StandardizeY:
        def fit(self, y_train, context):
            self.mean_ = float(y_train.mean())
            self.scale_ = float(y_train.std(ddof=0)) or 1.0
            return self

        def transform(self, y, context):
            return (y - self.mean_) / self.scale_

        def inverse_transform_prediction(self, y_pred, context):
            return float(y_pred) * self.scale_ + self.mean_

    assert list_custom_target_transformers() == ("standardize_y",)
    assert get_custom_target_transformer("standardize_y").evaluation_scale == "raw"

    recipe = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1])
        .use_target_transformer("standardize_y")
        .to_recipe_dict()
    )
    assert recipe["path"]["2_preprocessing"]["fixed_axes"]["target_transformer"] == "standardize_y"

    compiled = compile_recipe_dict(recipe).compiled
    assert compiled.execution_status == "executable"

    result = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1], model_family="ridge")
        .use_target_transformer("standardize_y")
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )
    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert manifest["training_spec"]["target_transformer"] == "standardize_y"
    assert manifest["target_transformer"]["name"] == "standardize_y"
    assert manifest["target_transformer"]["forecast_scale"] == "raw"
    assert manifest["target_transformer"]["evaluation_scale"] == "raw"

    predictions = pd.read_csv(Path(result.artifact_dir) / "predictions.csv")
    assert set(predictions["target_transformer"]) == {"standardize_y"}
    assert set(predictions["model_target_scale"]) == {"transformed"}
    assert set(predictions["forecast_scale"]) == {"raw"}
    assert "y_pred_model_scale" in predictions


def test_target_transformer_runs_raw_panel_on_raw_forecast_scale(tmp_path: Path) -> None:
    clear_custom_extensions()

    @target_transformer("identity_y")
    class IdentityY:
        def fit(self, y_train, context):
            return self

        def transform(self, y, context):
            return y

        def inverse_transform_prediction(self, y_pred, context):
            return y_pred

    recipe = (
        Experiment(
            dataset="fred_md",
            target="INDPRO",
            start=FIXTURE_START,
            end=FIXTURE_END,
            horizons=[1],
            model_family="ridge",
            feature_builder="raw_feature_panel",
        )
        .use_target_transformer("identity_y")
        .to_recipe_dict()
    )
    compiled = compile_recipe_dict(recipe).compiled
    assert compiled.execution_status == "executable"

    result = (
        Experiment(
            dataset="fred_md",
            target="INDPRO",
            start=FIXTURE_START,
            end=FIXTURE_END,
            horizons=[1],
            model_family="ridge",
            feature_builder="raw_feature_panel",
        )
        .use_target_transformer("identity_y")
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )
    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert manifest["training_spec"]["target_transformer"] == "identity_y"
    assert manifest["target_transformer"]["runtime"] == "raw_panel_v1"
    predictions = pd.read_csv(Path(result.artifact_dir) / "predictions.csv")
    assert set(predictions["target_transformer"]) == {"identity_y"}
    assert set(predictions["model_target_scale"]) == {"transformed"}
    assert set(predictions["forecast_scale"]) == {"raw"}


def test_target_transformer_blocks_unsupported_raw_panel_model() -> None:
    clear_custom_extensions()

    @target_transformer("identity_y")
    class IdentityY:
        def fit(self, y_train, context):
            return self

        def transform(self, y, context):
            return y

        def inverse_transform_prediction(self, y_pred, context):
            return y_pred

    recipe = (
        Experiment(
            dataset="fred_md",
            target="INDPRO",
            start=FIXTURE_START,
            end=FIXTURE_END,
            horizons=[1],
            model_family="randomforest",
            feature_builder="raw_feature_panel",
        )
        .use_target_transformer("identity_y")
        .to_recipe_dict()
    )
    compiled = compile_recipe_dict(recipe).compiled
    assert compiled.execution_status == "blocked_by_incompatibility"
    assert any("target_transformer raw-panel runtime currently supports" in reason for reason in compiled.blocked_reasons)
