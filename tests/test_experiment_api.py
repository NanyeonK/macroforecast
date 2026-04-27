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
from macrocast.execution.build import _fred_sd_frequency_report_from_metadata
from macrocast.execution.errors import ExecutionError
from macrocast.raw.types import RawArtifactRecord, RawDatasetMetadata, RawLoadResult

FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")
FIXTURE_SD_CSV = Path("tests/fixtures/fred_sd_sample.csv")
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
    assert recipe["path"]["1_data_task"]["fixed_axes"]["official_transform_policy"] == "apply_official_tcode"
    assert recipe["path"]["1_data_task"]["fixed_axes"]["official_transform_scope"] == "target_and_predictors"
    assert recipe["path"]["1_data_task"]["leaf_config"]["sample_start_date"] == FIXTURE_START
    assert recipe["path"]["1_data_task"]["leaf_config"]["sample_end_date"] == FIXTURE_END
    assert "tcode_policy" not in recipe["path"]["2_preprocessing"]["fixed_axes"]
    assert "target_transform_policy" not in recipe["path"]["2_preprocessing"]["fixed_axes"]
    assert "x_transform_policy" not in recipe["path"]["2_preprocessing"]["fixed_axes"]
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
    assert manifest["data_task_spec"]["official_transform_policy"] == "apply_official_tcode"
    assert manifest["data_task_spec"]["official_transform_scope"] == "target_and_predictors"
    assert manifest["preprocess_contract"]["tcode_policy"] == "official_tcode_only"
    assert manifest["preprocess_contract"]["target_transform_policy"] == "official_tcode_transformed"
    assert manifest["preprocess_contract"]["x_transform_policy"] == "official_tcode_transformed"
    layer2_spec = manifest["layer2_representation_spec"]
    assert layer2_spec["runtime_effect"] == "provenance_plus_runtime_block_dispatch"
    assert layer2_spec["source_bridge"]["feature_builder"] == "target_lag_features"
    assert layer2_spec["feature_blocks"]["feature_block_set"]["value"] == "target_lags_only"
    assert manifest["compiler"]["layer2_representation_spec"] == layer2_spec
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


def test_fred_md_sd_composite_runs_with_local_csv_sd_fixture(tmp_path: Path) -> None:
    result = forecast(
        "fred_md+fred_sd",
        target="INDPRO",
        start=FIXTURE_START,
        end=FIXTURE_END,
        horizons=[1],
        output_root=tmp_path,
        local_raw_source={"fred_md": FIXTURE_RAW, "fred_sd": FIXTURE_SD_CSV},
    )

    artifact_dir = Path(result.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    layer1_contract = json.loads((artifact_dir / "layer1_official_frame.json").read_text())

    assert manifest["raw_dataset"] == "fred_md+fred_sd"
    assert manifest["data_reports"]["combined_dataset"]["components"] == ["fred_md", "fred_sd"]
    preview = pd.read_csv(artifact_dir / "data_preview.csv", index_col=0)
    assert "BPPRIVSA_CA" in preview.columns

    source_contract = layer1_contract["source_availability_contract"]
    assert source_contract["component_count"] == 2
    components = {component["component"]: component for component in source_contract["component_source_contracts"]}
    assert components["fred_md"]["source_url_kind"] == "local_path"
    assert components["fred_sd"]["source_url_kind"] == "local_path"
    assert components["fred_sd"]["uses_local_source"] is True
    assert components["fred_sd"]["artifact_file_format"] == "csv"
    assert components["fred_sd"]["artifact_file_size_bytes"] > 0


def test_fred_sd_selection_filters_component_before_composite_run(tmp_path: Path) -> None:
    result = (
        Experiment(
            dataset="fred_md+fred_sd",
            target="INDPRO",
            start=FIXTURE_START,
            end=FIXTURE_END,
            horizons=[1],
        )
        .use_fred_sd_selection(states=["CA"], variables=["UR"])
        .run(
            output_root=tmp_path,
            local_raw_source={"fred_md": FIXTURE_RAW, "fred_sd": FIXTURE_SD_CSV},
        )
    )

    artifact_dir = Path(result.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    layer1_contract = json.loads((artifact_dir / "layer1_official_frame.json").read_text())
    metadata_contract = json.loads((artifact_dir / "fred_sd_series_metadata.json").read_text())
    frequency_report = json.loads((artifact_dir / "fred_sd_frequency_report.json").read_text())
    artifact_manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text())
    preview = pd.read_csv(artifact_dir / "data_preview.csv", index_col=0)

    assert manifest["data_task_spec"]["state_selection"] == "selected_states"
    assert manifest["data_task_spec"]["sd_variable_selection"] == "selected_sd_variables"
    assert manifest["data_task_spec"]["fred_sd_frequency_policy"] == "report_only"
    assert manifest["data_task_spec"]["sd_states"] == ["CA"]
    assert manifest["data_task_spec"]["sd_variables"] == ["UR"]
    assert "UR_CA" in preview.columns
    assert "UR_TX" not in preview.columns
    assert "BPPRIVSA_CA" not in preview.columns
    assert layer1_contract["data_task_spec"]["sd_states"] == ["CA"]
    assert layer1_contract["data_task_spec"]["sd_variables"] == ["UR"]
    assert manifest["fred_sd_series_metadata_contract"] == "fred_sd_series_metadata_v1"
    assert manifest["fred_sd_series_metadata_file"] == "fred_sd_series_metadata.json"
    assert manifest["fred_sd_series_metadata_summary"] == {
        "schema_version": "fred_sd_series_metadata_v1",
        "contract_version": "fred_sd_series_metadata_v1",
        "series_count": 1,
        "state_count": 1,
        "sd_variable_count": 1,
        "native_frequency_counts": {"monthly": 1},
    }
    assert manifest["fred_sd_frequency_report_contract"] == "fred_sd_frequency_report_v1"
    assert manifest["fred_sd_frequency_report_file"] == "fred_sd_frequency_report.json"
    assert manifest["fred_sd_frequency_report_summary"] == {
        "schema_version": "fred_sd_frequency_report_v1",
        "contract_version": "fred_sd_frequency_report_v1",
        "series_count": 1,
        "native_frequency_counts": {"monthly": 1},
        "frequency_status": "single_frequency",
        "has_monthly_quarterly_mix": False,
        "requires_mixed_frequency_decision": False,
    }
    assert metadata_contract["selector"] == {"states": ["CA"], "variables": ["UR"]}
    assert metadata_contract["series"][0]["column"] == "UR_CA"
    assert frequency_report["source_series_metadata_contract"] == "fred_sd_series_metadata_v1"
    assert frequency_report["frequency_status"] == "single_frequency"
    assert frequency_report["by_sd_variable"] == {"UR": {"monthly": 1}}
    assert layer1_contract["data_reports"]["components"]["fred_sd"]["fred_sd_series_metadata"]["series_count"] == 1
    assert layer1_contract["data_reports"]["fred_sd_frequency_report"]["frequency_status"] == "single_frequency"
    assert layer1_contract["data_reports"]["fred_sd_frequency_policy"]["policy"] == "report_only"
    assert layer1_contract["data_reports"]["fred_sd_frequency_policy"]["decision"] == "allowed"
    assert "fred_sd_series_metadata.json" in {
        item["path"] for item in artifact_manifest["artifacts"] if item["artifact_type"] == "fred_sd_series_metadata"
    }
    assert "fred_sd_frequency_report.json" in {
        item["path"] for item in artifact_manifest["artifacts"] if item["artifact_type"] == "fred_sd_frequency_report"
    }


def test_fred_sd_frequency_report_marks_mixed_panel() -> None:
    report = _fred_sd_frequency_report_from_metadata(
        {
            "contract_version": "fred_sd_series_metadata_v1",
            "selector": {"states": ["CA"], "variables": ["UR", "NQGSP", "X"]},
            "series_count": 3,
            "state_count": 1,
            "sd_variable_count": 3,
            "series": [
                {"column": "UR_CA", "sd_variable": "UR", "state": "CA", "native_frequency": "monthly"},
                {"column": "NQGSP_CA", "sd_variable": "NQGSP", "state": "CA", "native_frequency": "quarterly"},
                {"column": "X_CA", "sd_variable": "X", "state": "CA", "native_frequency": "unknown"},
            ],
        }
    )

    assert report is not None
    assert report["contract_version"] == "fred_sd_frequency_report_v1"
    assert report["native_frequency_counts"] == {"monthly": 1, "quarterly": 1, "unknown": 1}
    assert report["known_native_frequency_counts"] == {"monthly": 1, "quarterly": 1}
    assert report["frequency_status"] == "mixed_frequency_with_unknown"
    assert report["has_monthly_quarterly_mix"] is True
    assert report["requires_mixed_frequency_decision"] is True
    assert report["by_state"] == {"CA": {"monthly": 1, "quarterly": 1, "unknown": 1}}
    assert report["by_sd_variable"]["NQGSP"] == {"quarterly": 1}


def test_experiment_fred_sd_frequency_policy_lowers_to_layer1_axis() -> None:
    recipe = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2001-12",
            horizons=[1],
            frequency="monthly",
        )
        .use_fred_sd_frequency_policy("require_single_known_frequency")
        .to_recipe_dict()
    )

    assert (
        recipe["path"]["1_data_task"]["fixed_axes"]["fred_sd_frequency_policy"]
        == "require_single_known_frequency"
    )


def test_experiment_fred_sd_mixed_frequency_representation_lowers_to_layer2_axis() -> None:
    recipe = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2001-12",
            horizons=[1],
            frequency="monthly",
        )
        .use_fred_sd_mixed_frequency_representation("drop_non_target_native_frequency")
        .to_recipe_dict()
    )

    assert (
        recipe["path"]["2_preprocessing"]["fixed_axes"][
            "fred_sd_mixed_frequency_representation"
        ]
        == "drop_non_target_native_frequency"
    )


def test_fred_sd_frequency_policy_blocks_mixed_native_panel(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-01", periods=12, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(12)],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(12)],
        }
    ).to_csv(source, index=False)

    exp = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2000-12",
            horizons=[1],
            frequency="monthly",
        )
        .use_fred_sd_selection(states=["CA"], variables=["UR", "NQGSP"])
        .use_fred_sd_frequency_policy("require_single_known_frequency")
    )

    with pytest.raises(ExecutionError, match="fred_sd_frequency_policy='require_single_known_frequency'"):
        exp.run(output_root=tmp_path / "runs", local_raw_source=source)


def test_fred_sd_mixed_frequency_representation_drops_non_target_native_frequency(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-01", periods=18, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)

    result = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2001-06",
            horizons=[1],
            frequency="monthly",
        )
        .use_fred_sd_selection(states=["CA"], variables=["UR", "NQGSP"])
        .use_fred_sd_mixed_frequency_representation("drop_non_target_native_frequency")
        .run(output_root=tmp_path / "runs", local_raw_source=source)
    )

    artifact_dir = Path(result.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    representation = json.loads((artifact_dir / "fred_sd_mixed_frequency_representation.json").read_text())
    preview = pd.read_csv(artifact_dir / "data_preview.csv", index_col=0)

    assert manifest["layer2_representation_spec"]["input_panel"][
        "fred_sd_mixed_frequency_representation"
    ] == "drop_non_target_native_frequency"
    assert manifest["fred_sd_mixed_frequency_representation_contract"] == "fred_sd_mixed_frequency_representation_v1"
    assert manifest["fred_sd_mixed_frequency_representation_file"] == "fred_sd_mixed_frequency_representation.json"
    assert representation["owner_layer"] == "2_preprocessing"
    assert representation["dropped_fred_sd_columns"] == ["NQGSP_CA"]
    assert representation["dropped_by_native_frequency"] == {"quarterly": 1}
    assert "UR_CA" in preview.columns
    assert "NQGSP_CA" not in preview.columns


def test_fred_sd_mixed_frequency_representation_does_not_drop_target(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-01", periods=18, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)

    exp = (
        Experiment(
            dataset="fred_sd",
            target="NQGSP_CA",
            start="2000-01",
            end="2001-06",
            horizons=[1],
            frequency="monthly",
        )
        .use_fred_sd_selection(states=["CA"], variables=["UR", "NQGSP"])
        .use_fred_sd_mixed_frequency_representation("drop_non_target_native_frequency")
    )

    with pytest.raises(ExecutionError, match="would drop target columns"):
        exp.run(output_root=tmp_path / "runs", local_raw_source=source)


def test_fred_sd_native_frequency_block_payload_runs_with_custom_model(tmp_path: Path) -> None:
    clear_custom_extensions()
    dates = pd.date_range("2000-01-01", periods=18, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)
    calls: list[dict[str, object]] = []

    @custom_model("fred_sd_block_custom")
    def _fred_sd_block_custom(X_train, y_train, X_test, context):
        auxiliary = context["auxiliary_payloads"]
        block_payload = auxiliary["fred_sd_native_frequency_block_payload"]
        assert block_payload["contract_version"] == "fred_sd_native_frequency_block_payload_v1"
        assert "monthly" in block_payload["blocks"]
        assert "quarterly" in block_payload["blocks"]
        assert block_payload["column_to_native_frequency"]["NQGSP_CA"] == "quarterly"
        assert context["alignment"]["fred_sd_native_frequency_block_payload_contract"] == (
            "fred_sd_native_frequency_block_payload_v1"
        )
        calls.append(block_payload)
        return float(y_train[-1])

    try:
        result = (
            Experiment(
                dataset="fred_sd",
                target="UR_CA",
                start="2000-01",
                end="2001-06",
                horizons=[1],
                frequency="monthly",
                model_family="fred_sd_block_custom",
                feature_builder="raw_feature_panel",
                benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
            )
            .use_fred_sd_selection(states=["CA"], variables=["UR", "NQGSP"])
            .use_fred_sd_native_frequency_blocks()
            .run(output_root=tmp_path / "runs", local_raw_source=source)
        )

        artifact_dir = Path(result.artifact_dir)
        manifest = json.loads((artifact_dir / "manifest.json").read_text())
        block_payload = json.loads((artifact_dir / "fred_sd_native_frequency_block_payload.json").read_text())

        assert calls
        assert manifest["fred_sd_native_frequency_block_payload_contract"] == (
            "fred_sd_native_frequency_block_payload_v1"
        )
        assert manifest["fred_sd_native_frequency_block_payload_file"] == (
            "fred_sd_native_frequency_block_payload.json"
        )
        assert manifest["model_spec"]["executor_name"].endswith(
            "fred_sd_native_frequency_block_payload_v1"
        )
        assert block_payload["blocks"]["quarterly"]["columns"] == ["NQGSP_CA"]
        assert (
            manifest["layer2_representation_contract_metadata"]["auxiliary_payloads"][
                "fred_sd_native_frequency_block_payload"
            ]["contract_version"]
            == "fred_sd_native_frequency_block_payload_v1"
        )
    finally:
        clear_custom_extensions()


def test_fred_sd_mixed_frequency_adapter_runs_with_custom_model(tmp_path: Path) -> None:
    clear_custom_extensions()
    dates = pd.date_range("2000-01-01", periods=18, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)

    @custom_model("fred_sd_adapter_custom")
    def _fred_sd_adapter_custom(X_train, y_train, X_test, context):
        adapter = context["auxiliary_payloads"]["fred_sd_mixed_frequency_model_adapter"]
        assert adapter["contract_version"] == "fred_sd_mixed_frequency_model_adapter_v1"
        assert adapter["input_payload_contract"] == "fred_sd_native_frequency_block_payload_v1"
        return float(y_train.mean())

    try:
        result = (
            Experiment(
                dataset="fred_sd",
                target="UR_CA",
                start="2000-01",
                end="2001-06",
                horizons=[1],
                frequency="monthly",
                model_family="fred_sd_adapter_custom",
                feature_builder="raw_feature_panel",
                benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
            )
            .use_fred_sd_selection(states=["CA"], variables=["UR", "NQGSP"])
            .use_fred_sd_mixed_frequency_adapter()
            .run(output_root=tmp_path / "runs", local_raw_source=source)
        )

        manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())

        assert manifest["fred_sd_mixed_frequency_model_adapter_contract"] == (
            "fred_sd_mixed_frequency_model_adapter_v1"
        )
        assert manifest["fred_sd_mixed_frequency_model_adapter_file"] == (
            "fred_sd_mixed_frequency_model_adapter.json"
        )
        assert manifest["model_spec"]["executor_name"].endswith(
            "fred_sd_mixed_frequency_model_adapter_v1"
        )
    finally:
        clear_custom_extensions()


def test_fred_sd_midas_almon_runs_as_builtin_mixed_frequency_model(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-01", periods=18, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "UR_TX": [4.5 + idx / 20 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)

    result = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2001-06",
            horizons=[1],
            frequency="monthly",
            model_family="midas_almon",
            feature_builder="raw_feature_panel",
            benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        )
        .use_fred_sd_selection(states=["CA", "TX"], variables=["UR", "NQGSP"])
        .use_fred_sd_mixed_frequency_adapter()
        .run(output_root=tmp_path / "runs", local_raw_source=source)
    )

    artifact_dir = Path(result.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    adapter_payload = json.loads((artifact_dir / "fred_sd_mixed_frequency_model_adapter.json").read_text())

    assert manifest["model_spec"]["model_family"] == "midas_almon"
    assert manifest["model_spec"]["executor_name"] == (
        "midas_almon:fred_sd_mixed_frequency_model_adapter_v1"
    )
    assert manifest["model_spec"]["fred_sd_mixed_frequency_custom_model_required"] is False
    assert manifest["model_spec"]["fred_sd_mixed_frequency_builtin_model"] is True
    assert adapter_payload["contract_version"] == "fred_sd_mixed_frequency_model_adapter_v1"
    metadata = manifest["layer2_representation_contract_metadata"]
    assert metadata["alignment"]["midas_almon_contract"] == "midas_almon_direct_v1"
    assert metadata["alignment"]["midas_max_lag"] == 3
    assert "midas_almon:monthly" in metadata["block_order"]
    assert "midas_almon:quarterly" in metadata["block_order"]


def test_fred_sd_midasr_nealmon_runs_as_builtin_midasr_slice(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-01", periods=30, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "UR_TX": [4.5 + idx / 20 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)

    result = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2002-06",
            horizons=[1],
            frequency="monthly",
            model_family="midasr_nealmon",
            feature_builder="raw_feature_panel",
            benchmark_config={"minimum_train_size": 12, "rolling_window_size": 12},
        )
        .use_fred_sd_selection(states=["CA", "TX"], variables=["UR", "NQGSP"])
        .use_fred_sd_mixed_frequency_adapter()
        .run(output_root=tmp_path / "runs", local_raw_source=source)
    )

    artifact_dir = Path(result.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    adapter_payload = json.loads((artifact_dir / "fred_sd_mixed_frequency_model_adapter.json").read_text())

    assert manifest["model_spec"]["model_family"] == "midasr_nealmon"
    assert manifest["model_spec"]["executor_name"] == (
        "midasr_nealmon:fred_sd_mixed_frequency_model_adapter_v1"
    )
    assert manifest["model_spec"]["fred_sd_mixed_frequency_custom_model_required"] is False
    assert manifest["model_spec"]["fred_sd_mixed_frequency_builtin_model"] is True
    assert manifest["model_spec"]["fred_sd_mixed_frequency_builtin_model_family"] == "midasr_nealmon"
    assert adapter_payload["contract_version"] == "fred_sd_mixed_frequency_model_adapter_v1"
    metadata = manifest["layer2_representation_contract_metadata"]
    assert metadata["alignment"]["midasr_nealmon_contract"] == "midasr_nealmon_direct_v1"
    assert metadata["alignment"]["midasr_reference_package"] == "midasr"
    assert metadata["alignment"]["midasr_reference_function"] == "midas_r + nealmon"
    assert metadata["alignment"]["midas_max_lag"] == 3
    assert "midasr_nealmon:monthly" in metadata["block_order"]
    assert "midasr_nealmon:quarterly" in metadata["block_order"]


def test_fred_sd_midasr_almonp_runs_as_builtin_weight_family(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-01", periods=30, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "UR_TX": [4.5 + idx / 20 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)

    result = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2002-06",
            horizons=[1],
            frequency="monthly",
            model_family="midasr",
            feature_builder="raw_feature_panel",
            benchmark_config={"minimum_train_size": 12, "rolling_window_size": 12},
        )
        .sweep({"midasr_weight_family": "almonp"})
        .use_fred_sd_selection(states=["CA", "TX"], variables=["UR", "NQGSP"])
        .use_fred_sd_mixed_frequency_adapter()
        .run(output_root=tmp_path / "runs", local_raw_source=source)
    )

    artifact_dir = Path(result.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())

    assert manifest["model_spec"]["model_family"] == "midasr"
    assert manifest["model_spec"]["executor_name"] == "midasr:fred_sd_mixed_frequency_model_adapter_v1"
    assert manifest["model_spec"]["fred_sd_mixed_frequency_builtin_model"] is True
    assert manifest["model_spec"]["fred_sd_mixed_frequency_builtin_model_family"] == "midasr"
    metadata = manifest["layer2_representation_contract_metadata"]
    assert metadata["alignment"]["midasr_contract"] == "midasr_restricted_direct_v1"
    assert metadata["alignment"]["midasr_weight_family"] == "almonp"
    assert metadata["alignment"]["midasr_reference_function"] == "midas_r + almonp"
    assert "midasr:monthly" in metadata["block_order"]
    assert "midasr:quarterly" in metadata["block_order"]


@pytest.mark.parametrize(
    ("weight_family", "expected_lag"),
    [
        ("nbeta", 3),
        ("genexp", 3),
        ("harstep", 20),
    ],
)
def test_fred_sd_midasr_runs_remaining_builtin_weight_families(
    tmp_path: Path,
    weight_family: str,
    expected_lag: int,
) -> None:
    dates = pd.date_range("2000-01-01", periods=55, freq="MS")
    source = tmp_path / f"mixed_fred_sd_{weight_family}.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "UR_TX": [4.5 + idx / 20 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)

    result = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2004-07",
            horizons=[1],
            frequency="monthly",
            model_family="midasr",
            feature_builder="raw_feature_panel",
            benchmark_config={"minimum_train_size": 30, "rolling_window_size": 30},
        )
        .sweep({"midasr_weight_family": weight_family})
        .use_fred_sd_selection(states=["CA", "TX"], variables=["UR", "NQGSP"])
        .use_fred_sd_mixed_frequency_adapter()
        .run(output_root=tmp_path / f"runs_{weight_family}", local_raw_source=source)
    )

    artifact_dir = Path(result.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    metadata = manifest["layer2_representation_contract_metadata"]

    assert manifest["model_spec"]["model_family"] == "midasr"
    assert metadata["alignment"]["midasr_weight_family"] == weight_family
    assert metadata["alignment"]["midasr_reference_function"] == f"midas_r + {weight_family}"
    assert metadata["alignment"]["midas_max_lag"] == expected_lag
    assert metadata["alignment"]["midasr_param_width"] in {3, 4}


def test_fred_sd_advanced_mixed_frequency_requires_custom_or_midas_model(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-01", periods=18, freq="MS")
    source = tmp_path / "mixed_fred_sd.csv"
    pd.DataFrame(
        {
            "date": dates,
            "UR_CA": [5.0 + idx / 10 for idx in range(len(dates))],
            "NQGSP_CA": [100.0 + idx if idx % 3 == 0 else None for idx in range(len(dates))],
        }
    ).to_csv(source, index=False)

    exp = (
        Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2001-06",
            horizons=[1],
            frequency="monthly",
            model_family="ridge",
            feature_builder="raw_feature_panel",
        )
        .use_fred_sd_selection(states=["CA"], variables=["UR", "NQGSP"])
        .use_fred_sd_native_frequency_blocks()
    )

    with pytest.raises(CompileValidationError, match="registered custom Layer 3 model or model_family in"):
        exp.run(output_root=tmp_path / "runs", local_raw_source=source)


def test_experiment_fred_sd_selection_lowers_to_layer1_axes() -> None:
    recipe = (
        Experiment(dataset="fred_sd+fred_qd", target="GDPC1", start="2000-01", end="2002-06", horizons=[1])
        .use_fred_sd_selection(states=["CA", "TX"], variables=["UR"])
        .to_recipe_dict()
    )

    data_task = recipe["path"]["1_data_task"]
    assert data_task["fixed_axes"]["state_selection"] == "selected_states"
    assert data_task["fixed_axes"]["sd_variable_selection"] == "selected_sd_variables"
    assert data_task["leaf_config"]["sd_states"] == ["CA", "TX"]
    assert data_task["leaf_config"]["sd_variables"] == ["UR"]


def test_experiment_fred_sd_groups_lower_to_layer1_axes() -> None:
    recipe = (
        Experiment(dataset="fred_md+fred_sd", target="INDPRO", start="2000-01", end="2002-06", horizons=[1])
        .use_fred_sd_groups(state_group="census_region_west", variable_group="labor_market_core")
        .to_recipe_dict()
    )

    data_task = recipe["path"]["1_data_task"]
    assert data_task["fixed_axes"]["fred_sd_state_group"] == "census_region_west"
    assert data_task["fixed_axes"]["fred_sd_variable_group"] == "labor_market_core"


def test_fred_sd_group_selection_filters_component_before_composite_run(tmp_path: Path) -> None:
    result = (
        Experiment(
            dataset="fred_md+fred_sd",
            target="INDPRO",
            start=FIXTURE_START,
            end=FIXTURE_END,
            horizons=[1],
        )
        .use_fred_sd_groups(state_group="census_region_west", variable_group="labor_market_core")
        .run(
            output_root=tmp_path,
            local_raw_source={"fred_md": FIXTURE_RAW, "fred_sd": FIXTURE_SD_CSV},
        )
    )

    artifact_dir = Path(result.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    layer1_contract = json.loads((artifact_dir / "layer1_official_frame.json").read_text())
    metadata_contract = json.loads((artifact_dir / "fred_sd_series_metadata.json").read_text())
    preview = pd.read_csv(artifact_dir / "data_preview.csv", index_col=0)

    assert manifest["data_task_spec"]["fred_sd_state_group"] == "census_region_west"
    assert manifest["data_task_spec"]["fred_sd_variable_group"] == "labor_market_core"
    assert manifest["data_task_spec"]["state_selection"] == "selected_states"
    assert manifest["data_task_spec"]["sd_variable_selection"] == "selected_sd_variables"
    assert "CA" in manifest["data_task_spec"]["sd_states"]
    assert "TX" not in manifest["data_task_spec"]["sd_states"]
    assert manifest["data_task_spec"]["sd_variables"] == ["ICLAIMS", "LF", "NA", "PARTRATE", "UR"]
    assert "UR_CA" in preview.columns
    assert "UR_TX" not in preview.columns
    assert "BPPRIVSA_CA" not in preview.columns
    assert metadata_contract["selector"]["states"] == manifest["data_task_spec"]["sd_states"]
    assert metadata_contract["selector"]["variables"] == manifest["data_task_spec"]["sd_variables"]
    assert layer1_contract["data_task_spec"]["fred_sd_state_group"] == "census_region_west"


def test_experiment_fred_sd_selection_accepts_single_string_values() -> None:
    recipe = (
        Experiment(dataset="fred_md+fred_sd", target="INDPRO", start="2000-01", end="2002-06", horizons=[1])
        .use_fred_sd_selection(states="ca", variables="UR")
        .to_recipe_dict()
    )

    data_task = recipe["path"]["1_data_task"]
    assert data_task["leaf_config"]["sd_states"] == ["CA"]
    assert data_task["leaf_config"]["sd_variables"] == ["UR"]


def test_experiment_fred_sd_selection_rejects_empty_values() -> None:
    exp = Experiment(dataset="fred_md+fred_sd", target="INDPRO", start="2000-01", end="2002-06", horizons=[1])

    with pytest.raises(ValueError, match="empty values"):
        exp.use_fred_sd_selection(states=["CA", " "])


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


def test_sd_variable_global_empirical_tcodes_are_explicit_opt_in(monkeypatch, tmp_path: Path) -> None:
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
            "STHPI_CA": [200.0 + 2.0 * float(i) for i in range(len(m_dates))],
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
        .use_sd_empirical_tcodes(unit="variable_global")
        .run(output_root=tmp_path)
    )

    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    report = manifest["data_reports"]["sd_inferred_tcodes"]
    assert report["policy"] == "variable_global_stationarity_v0_1"
    assert report["map_version"] == "sd-variable-global-stationarity-v0.1"
    assert report["decision_unit"] == "sd_variable"
    assert report["applied"]["BPPRIVSA_CA"] == 2
    assert report["applied"]["STHPI_CA"] == 6
    assert manifest["data_reports"]["tcode"]["columns"]["BPPRIVSA_CA"] == 2
    assert manifest["data_reports"]["tcode"]["columns"]["STHPI_CA"] == 6


def test_sd_state_series_empirical_tcodes_use_explicit_column_map(monkeypatch, tmp_path: Path) -> None:
    q_dates = pd.date_range("2000-01-01", periods=10, freq="QS")
    qd_frame = pd.DataFrame({"GDPC1": [100.0 + i for i in range(len(q_dates))]}, index=q_dates)
    qd_frame.index.name = "date"
    m_dates = pd.date_range("2000-01-01", periods=30, freq="MS")
    sd_frame = pd.DataFrame(
        {
            "BPPRIVSA_CA": [100.0 + float(i) for i in range(len(m_dates))],
            "BPPRIVSA_TX": [120.0 + 2.0 * float(i) for i in range(len(m_dates))],
        },
        index=m_dates,
    )
    sd_frame.index.name = "date"

    monkeypatch.setattr(
        "macrocast.execution.build.load_fred_qd",
        lambda **_: _raw_result("fred_qd", "quarterly", qd_frame, {"GDPC1": 1}),
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
        .use_sd_empirical_tcodes(
            unit="state_series",
            code_map={"BPPRIVSA_CA": 2, "BPPRIVSA_TX": 5},
            audit_uri="artifacts/state_series_audit.csv",
        )
        .run(output_root=tmp_path)
    )

    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    report = manifest["data_reports"]["sd_inferred_tcodes"]
    assert report["policy"] == "state_series_stationarity_override_v0_1"
    assert report["map_version"] == "sd-state-series-stationarity-override-v0.1"
    assert report["decision_unit"] == "sd_variable_x_state"
    assert report["audit_uri"] == "artifacts/state_series_audit.csv"
    assert report["applied"]["BPPRIVSA_CA"] == 2
    assert report["applied"]["BPPRIVSA_TX"] == 5


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


def test_experiment_use_sd_empirical_tcodes_lowers_to_leaf_config() -> None:
    variable_recipe = (
        Experiment(dataset="fred_sd+fred_qd", target="GDPC1", start="2000-01", end="2002-06", horizons=[1])
        .use_sd_empirical_tcodes(unit="variable_global")
        .to_recipe_dict()
    )
    variable_leaf = variable_recipe["path"]["2_preprocessing"]["leaf_config"]
    assert variable_leaf["sd_tcode_policy"] == "variable_global_stationarity_v0_1"
    assert variable_leaf["sd_tcode_map_version"] == "sd-variable-global-stationarity-v0.1"

    state_recipe = (
        Experiment(dataset="fred_sd+fred_qd", target="GDPC1", start="2000-01", end="2002-06", horizons=[1])
        .use_sd_empirical_tcodes(unit="state_series", code_map={"UR_CA": 2}, audit_uri="audit.csv")
        .to_recipe_dict()
    )
    state_leaf = state_recipe["path"]["2_preprocessing"]["leaf_config"]
    assert state_leaf["sd_tcode_policy"] == "state_series_stationarity_override_v0_1"
    assert state_leaf["sd_tcode_map_version"] == "sd-state-series-stationarity-override-v0.1"
    assert state_leaf["sd_tcode_code_map"] == {"UR_CA": 2}
    assert state_leaf["sd_tcode_audit_uri"] == "audit.csv"


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
    assert manifest["forecast_engine"] == "custom_model:last_lag_custom:target_lag_features_v0"


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
    assert "custom_preprocessor" not in manifest["training_spec"]
    assert manifest["layer2_representation_spec"]["frame_conditioning"]["custom_preprocessor"] == "center_x"
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

    @target_transformer("standardize_target")
    class StandardizeTarget:
        def fit(self, target_train, context):
            self.mean_ = float(target_train.mean())
            self.scale_ = float(target_train.std(ddof=0)) or 1.0
            return self

        def transform(self, target, context):
            return (target - self.mean_) / self.scale_

        def inverse_transform_prediction(self, target_pred, context):
            return float(target_pred) * self.scale_ + self.mean_

    assert list_custom_target_transformers() == ("standardize_target",)
    assert get_custom_target_transformer("standardize_target").evaluation_scale == "raw"

    recipe = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1])
        .use_target_transformer("standardize_target")
        .to_recipe_dict()
    )
    assert recipe["path"]["2_preprocessing"]["fixed_axes"]["target_transformer"] == "standardize_target"

    compiled = compile_recipe_dict(recipe).compiled
    assert compiled.execution_status == "executable"

    result = (
        Experiment(dataset="fred_md", target="INDPRO", start=FIXTURE_START, end=FIXTURE_END, horizons=[1], model_family="ridge")
        .use_target_transformer("standardize_target")
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )
    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert "target_transformer" not in manifest["training_spec"]
    assert manifest["layer2_representation_spec"]["target_representation"]["target_transformer"] == "standardize_target"
    assert manifest["target_transformer"]["name"] == "standardize_target"
    assert manifest["target_transformer"]["forecast_scale"] == "raw"
    assert manifest["target_transformer"]["evaluation_scale"] == "raw"

    predictions = pd.read_csv(Path(result.artifact_dir) / "predictions.csv")
    assert set(predictions["target_transformer"]) == {"standardize_target"}
    assert set(predictions["model_target_scale"]) == {"custom_transformer_scale"}
    assert set(predictions["forecast_scale"]) == {"original_target_scale"}
    assert "y_pred_model_scale" in predictions


def test_target_transformer_runs_raw_panel_on_raw_forecast_scale(tmp_path: Path) -> None:
    clear_custom_extensions()

    @target_transformer("identity_target")
    class IdentityTarget:
        def fit(self, target_train, context):
            return self

        def transform(self, target, context):
            return target

        def inverse_transform_prediction(self, target_pred, context):
            return target_pred

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
        .use_target_transformer("identity_target")
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
        .use_target_transformer("identity_target")
        .run(output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    )
    manifest = json.loads((Path(result.artifact_dir) / "manifest.json").read_text())
    assert "target_transformer" not in manifest["training_spec"]
    assert manifest["layer2_representation_spec"]["target_representation"]["target_transformer"] == "identity_target"
    assert manifest["target_transformer"]["runtime"] == "raw_panel_v1"
    predictions = pd.read_csv(Path(result.artifact_dir) / "predictions.csv")
    assert set(predictions["target_transformer"]) == {"identity_target"}
    assert set(predictions["model_target_scale"]) == {"custom_transformer_scale"}
    assert set(predictions["forecast_scale"]) == {"original_target_scale"}


def test_target_transformer_compile_uses_feature_runtime_for_factor_bridge() -> None:
    clear_custom_extensions()

    @target_transformer("identity_target_factor_bridge")
    class IdentityTargetFactorBridge:
        def fit(self, target_train, context):
            return self

        def transform(self, target, context):
            return target

        def inverse_transform_prediction(self, target_pred, context):
            return target_pred

    recipe = (
        Experiment(
            dataset="fred_md",
            target="INDPRO",
            start=FIXTURE_START,
            end=FIXTURE_END,
            horizons=[1],
            model_family="ridge",
            feature_builder="pca_factor_features",
        )
        .use_target_transformer("identity_target_factor_bridge")
        .to_recipe_dict()
    )

    compiled = compile_recipe_dict(recipe).compiled

    assert compiled.execution_status == "executable"


def test_target_transformer_blocks_unsupported_raw_panel_model() -> None:
    clear_custom_extensions()

    @target_transformer("identity_target")
    class IdentityTarget:
        def fit(self, target_train, context):
            return self

        def transform(self, target, context):
            return target

        def inverse_transform_prediction(self, target_pred, context):
            return target_pred

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
        .use_target_transformer("identity_target")
        .to_recipe_dict()
    )
    compiled = compile_recipe_dict(recipe).compiled
    assert compiled.execution_status == "blocked_by_incompatibility"
    assert any("target_transformer raw-panel runtime currently supports" in reason for reason in compiled.blocked_reasons)
