from __future__ import annotations

import gzip
import json
import zipfile

import pandas as pd

import macroforecast as mf


class _DummyEstimator:
    def predict(self, X):
        return [0.0] * len(X)


def _forecast_result() -> mf.forecasting.ForecastResult:
    forecasts = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "origin": [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-31")],
            "horizon": [1, 1],
            "prediction": [1.0, 1.2],
            "actual": [1.1, 1.3],
            "model": ["ridge", "ridge"],
            "model_spec": ["ridge", "ridge"],
        }
    )
    return mf.forecasting.ForecastResult(
        forecasts,
        metadata={"metadata_schema": {"kind": "forecast_result", "version": 1}},
    )


def test_write_artifacts_writes_forecast_result_and_manifest(tmp_path) -> None:
    forecasts = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")],
            "origin": [pd.Timestamp("2019-12-31")],
            "prediction": [1.0],
            "actual": [1.1],
            "model": ["ridge"],
        }
    )
    result = mf.forecasting.ForecastResult(forecasts, metadata={"model": "ridge"})

    manifest = mf.output.write_artifacts(result, tmp_path)

    assert (tmp_path / "forecast_result.json").exists()
    assert (tmp_path / "forecast_result_forecasts.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert "forecast_result.json" in manifest.artifacts
    assert manifest.metadata_schema == {"kind": "artifact_manifest", "version": 1}
    assert [record.kind for record in manifest.records] == ["forecast_result", "forecast_table"]
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["metadata_schema"]["kind"] == "artifact_manifest"
    assert payload["provenance"]["macroforecast_version"] == mf.__version__
    assert payload["records"][0]["source"] == "forecast_result"
    assert payload["records"][0]["metadata"]["forecast_rows"] == 1


def test_write_artifacts_supports_named_dataframes(tmp_path) -> None:
    table = pd.DataFrame({"a": [1.0, 2.0]})
    table.attrs["macroforecast_metadata_schema"] = {"kind": "forecast_metrics", "version": 1}

    manifest = mf.output.write_artifacts({"metrics": table}, tmp_path, formats=("json", "csv"))

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert manifest.artifacts["metrics.csv"].endswith("metrics.csv")
    assert manifest.records[0].metadata["metadata_schema"]["kind"] == "forecast_metrics"
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metadata_schema"]["kind"] == "dataframe_artifact"
    assert payload["attrs"]["macroforecast_metadata_schema"]["kind"] == "forecast_metrics"


def test_write_artifacts_supports_manifest_table_formats(tmp_path) -> None:
    table = pd.DataFrame({"a": [1.0, 2.0]})

    manifest = mf.output.write_artifacts(
        {"metrics": table},
        tmp_path,
        formats=("csv",),
        manifest_format="csv",
        include_provenance=False,
    )

    assert (tmp_path / "manifest.csv").exists()
    assert manifest.provenance == {}
    manifest_table = pd.read_csv(tmp_path / "manifest.csv")
    assert manifest_table.loc[0, "kind"] == "dataframe"
    assert manifest_table.loc[0, "format"] == "csv"
    assert manifest.records[0].metadata["path_exists"] is True
    assert isinstance(manifest.records[0].metadata["sha256"], str)


def test_write_artifacts_records_forecast_result_stored_models(tmp_path) -> None:
    model_path = tmp_path / "trained_model" / "ridge" / "origin_0_h1_20200131.pkl"
    metadata_path = model_path.with_suffix(".json")
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    metadata_path.write_text("{}\n", encoding="utf-8")
    stored_model = {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "save_error": None,
    }
    forecasts = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")],
            "origin": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31")],
            "origin_pos": [0, 0],
            "horizon": [1, 1],
            "model": ["ridge", "ridge"],
            "model_spec": ["ridge", "ridge"],
            "prediction": [1.0, 1.2],
            "actual": [1.1, 1.3],
            "stored_model": [stored_model, stored_model],
        }
    )
    result = mf.forecasting.ForecastResult(forecasts, metadata={"model": "ridge"})

    manifest = mf.output.write_artifacts(result, tmp_path / "artifacts")

    stored_records = [
        record for record in manifest.records if record.kind.startswith("stored_model_")
    ]
    assert [record.kind for record in stored_records] == [
        "stored_model_pickle",
        "stored_model_metadata",
    ]
    assert stored_records[0].path == str(model_path)
    assert stored_records[0].metadata["path_exists"] is True
    assert stored_records[0].metadata["model"] == "ridge"
    assert len(stored_records) == 2


def test_write_artifacts_supports_gzip_and_provenance_filter(tmp_path) -> None:
    table = pd.DataFrame({"a": [1.0, 2.0]})

    manifest = mf.output.write_artifacts(
        {"metrics": table},
        tmp_path,
        formats=("csv",),
        compression="gzip",
        provenance_fields=("macroforecast_version", "git"),
    )

    assert (tmp_path / "metrics.csv.gz").exists()
    assert not (tmp_path / "metrics.csv").exists()
    assert set(manifest.provenance) == {"macroforecast_version", "git"}
    record = manifest.records[0]
    assert record.name == "metrics.csv.gz"
    assert record.metadata["compression"] == "gzip"
    assert record.metadata["path_exists"] is True
    assert isinstance(record.metadata["sha256"], str)
    with gzip.open(tmp_path / "metrics.csv.gz", "rt", encoding="utf-8") as handle:
        assert "a" in handle.read()


def test_write_artifacts_supports_zip_bundle(tmp_path) -> None:
    table = pd.DataFrame({"a": [1.0, 2.0]})

    manifest = mf.output.write_artifacts(
        {"metrics": table},
        tmp_path,
        formats=("json", "csv"),
        compression="zip",
        include_provenance=False,
    )

    bundle = tmp_path / "artifact_bundle.zip"
    assert bundle.exists()
    bundle_records = [record for record in manifest.records if record.kind == "artifact_bundle"]
    assert len(bundle_records) == 1
    assert bundle_records[0].metadata["compression"] == "zip"
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert {"metrics.json", "metrics.csv"} <= names


def test_output_generation_tables_and_summary() -> None:
    result = _forecast_result()
    report = mf.evaluation.EvaluationReport(
        scores=pd.DataFrame(
            {"model": ["ridge"], "horizon": [1], "metric": ["rmse"], "value": [0.1]}
        ),
        ranking=pd.DataFrame({"model": ["ridge"], "rank": [1]}),
    )

    forecasts = mf.output.forecast_table(result)
    metrics = mf.output.metric_table(report)
    ranking = mf.output.ranking_table(report)
    summary = mf.output.run_summary(result, evaluation=report, metadata={"run": "demo"})

    assert list(forecasts["model"].unique()) == ["ridge"]
    assert forecasts.attrs["macroforecast_metadata"]["source"] == "ForecastResult"
    assert metrics.attrs["macroforecast_metadata_schema"]["kind"] == "metric_table"
    assert metrics.attrs["macroforecast_metadata"]["field"] == "scores"
    assert ranking.attrs["macroforecast_metadata_schema"]["kind"] == "ranking_table"
    assert ranking.attrs["macroforecast_metadata"]["field"] == "ranking"
    assert summary["metadata_schema"]["kind"] == "run_summary"
    assert summary["forecasts"]["models"] == ["ridge"]
    assert summary["evaluation"]["ranking_rows"] == 1


def test_output_writes_forecast_result_anatomy_sidecar(tmp_path) -> None:
    anatomy = mf.interpretation.AnatomyPipelineResult(
        anatomy=None,
        explanations={
            "forecast": pd.DataFrame(
                {
                    "model_set": ["ridge"],
                    "index": [pd.Timestamp("2020-01-31")],
                    "feature": ["x1"],
                    "contribution": [0.25],
                    "is_base": [False],
                }
            )
        },
        variable_importance=pd.DataFrame(
            {"model_set": ["ridge"], "feature": ["x1"], "importance": [0.25]}
        ),
        performance_values={
            "rmse": pd.DataFrame(
                {
                    "model_set": ["ridge"],
                    "index": ["global"],
                    "feature": ["x1"],
                    "contribution": [-0.1],
                    "is_base": [False],
                }
            )
        },
        metadata={"kind": "anatomy_pipeline", "models": ["ridge"]},
    )
    result = _forecast_result().with_sidecar("anatomy", anatomy)

    tables = mf.output.forecast_shapley_tables(anatomy, prefix="demo")
    backend_tables = mf.output.anatomy_tables(anatomy, prefix="backend")
    bundle = mf.output.bundle_outputs(forecasts=result, include_summary=True)
    manifest = mf.output.write_artifacts(
        result,
        tmp_path,
        formats=("json",),
        include_provenance=False,
    )

    assert result.sidecar_names() == ("anatomy",)
    assert result.get_sidecar("anatomy") is anatomy
    assert "sidecars" in result.to_dict()
    assert "demo_variable_importance" in tables
    assert "backend_variable_importance" in backend_tables
    assert "anatomy_anatomy_variable_importance" in bundle.artifacts
    assert bundle.artifacts["summary"]["sidecars"]["names"] == ["anatomy"]
    assert (tmp_path / "forecast_result_anatomy_summary.json").exists()
    assert (tmp_path / "forecast_result_anatomy_variable_importance.json").exists()
    assert any(
        record.source == "forecast_result.sidecars.anatomy"
        for record in manifest.records
    )


def test_output_writes_forecast_result_dual_sidecar(tmp_path) -> None:
    X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_train = pd.Series([0.0, 1.0, 2.0])
    X_test = pd.DataFrame({"x": [1.5]})
    result = _forecast_result().with_dual(
        None,
        X_train,
        y_train,
        X_test,
        method="ridge",
        lambda_=0.1,
        ridge_penalty_scale="none",
        top_n=1,
    )

    bundle = mf.output.bundle_outputs(forecasts=result, include_summary=True)
    manifest = mf.output.write_artifacts(
        result,
        tmp_path,
        formats=("json",),
        include_provenance=False,
        layout="grouped",
    )

    assert result.sidecar_names() == ("dual",)
    assert "dual_dual_observation_weights" in bundle.artifacts
    assert (tmp_path / "interpretation" / "dual" / "forecast_result_dual_summary.json").exists()
    assert (
        tmp_path
        / "interpretation"
        / "dual"
        / "forecast_result_dual_observation_weights.json"
    ).exists()
    assert (
        "interpretation/dual/forecast_result_dual_observation_weights.json"
        in manifest.artifacts
    )
    assert any(
        record.source == "forecast_result.sidecars.dual"
        and record.metadata["group"] == "interpretation/dual"
        for record in manifest.records
    )


def test_output_generation_supports_tests_models_selection_and_metadata() -> None:
    test_result = mf.tests.dm_test([1.0, 1.2, 0.9], [1.1, 1.1, 1.0])
    fit = mf.models.ModelFit(
        estimator=_DummyEstimator(),
        model="ridge",
        feature_names=("x1", "x2"),
        target_name="y",
        metadata={"alpha": 1.0},
    )
    search = mf.model_selection.SearchResult(
        best_params={"alpha": 1.0},
        best_score=0.1,
        trials=pd.DataFrame({"alpha": [0.1, 1.0], "score": [0.2, 0.1], "status": ["ok", "ok"]}),
        metric="rmse",
        method="grid",
        window="last_block",
    )

    tests = mf.output.test_table({"dm": test_result})
    models = mf.output.model_table({"ridge_fit": fit})
    search_table = mf.output.model_selection_table(search)
    metadata = mf.output.metadata_table({"data": {"source": "FRED-MD"}, "seed": 42})

    assert tests.loc[0, "name"] == "dm"
    assert tests.loc[0, "statistic_type"] == "t"
    assert tests.loc[0, "p_value_status"] == "available"
    assert tests.loc[0, "r_reference"] == "forecast/R/DM2.R::dm.test"
    assert models.loc[0, "model"] == "ridge"
    assert models.loc[0, "n_features"] == 2
    assert search_table.shape[0] == 2
    assert "data.source" in set(metadata["path"])


def test_bundle_select_name_index_and_write_artifacts(tmp_path) -> None:
    result = _forecast_result()
    report = mf.evaluation.EvaluationReport(
        scores=pd.DataFrame({"model": ["ridge"], "metric": ["rmse"], "value": [0.1]}),
        ranking=pd.DataFrame({"model": ["ridge"], "rank": [1]}),
    )

    bundle = mf.output.bundle_outputs(
        forecasts=result,
        evaluation=report,
        metadata={"run": "demo"},
        include_summary=True,
    )
    assert isinstance(bundle, mf.output.OutputBundle)
    assert {"forecasts", "metrics", "ranking", "metadata", "summary"} <= set(bundle.artifacts)

    selected = mf.output.select_outputs(bundle, objects=("forecasts", "summary"))
    renamed = mf.output.name_outputs(selected, convention="prefixed", prefix="demo")
    index = mf.output.artifact_index(renamed)
    manifest = mf.output.write_artifacts(renamed, tmp_path, formats=("json",), include_provenance=False)

    assert list(renamed.artifacts) == ["demo_forecasts", "demo_summary"]
    assert set(index["name"]) == {"demo_forecasts", "demo_summary"}
    assert (tmp_path / "demo_forecasts.json").exists()
    assert (tmp_path / "demo_summary.json").exists()
    assert len(manifest.records) == 2


def test_interpretation_outputs_and_grouped_artifact_layout(tmp_path) -> None:
    result = _forecast_result()
    report = mf.evaluation.EvaluationReport(
        scores=pd.DataFrame({"model": ["ridge"], "metric": ["rmse"], "value": [0.1]}),
        ranking=pd.DataFrame({"model": ["ridge"], "rank": [1]}),
    )
    shap = pd.DataFrame({"feature": ["x1"], "importance": [0.3]})
    shap.attrs["macroforecast_metadata_schema"] = {
        "kind": "shap_importance",
        "version": 1,
    }
    X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_train = pd.Series([0.0, 1.0, 2.0])
    X_test = pd.DataFrame({"x": [1.5]})
    dual = mf.interpretation.dual.dual_interpretation(
        None,
        X_train,
        y_train,
        X_test,
        method="ridge",
        lambda_=0.1,
        ridge_penalty_scale="none",
        top_n=1,
    )

    interp_bundle = mf.output.interpretation_outputs(
        {"shap": shap, "dual": dual},
        prefix="interp",
    )
    bundle = mf.output.bundle_outputs(
        forecasts=result,
        evaluation=report,
        interpretation=interp_bundle.artifacts,
        metadata={"run": "demo"},
    )
    manifest = mf.output.write_artifacts(
        bundle,
        tmp_path,
        formats=("json",),
        include_provenance=False,
        layout="grouped",
    )

    assert "interp_shap" in interp_bundle.artifacts
    assert "interp_dual_observation_weights" in interp_bundle.artifacts
    assert "interp_dual_forecast_diagnostics" in interp_bundle.artifacts
    assert (tmp_path / "forecasts" / "forecasts.json").exists()
    assert (tmp_path / "evaluation" / "metrics.json").exists()
    assert (tmp_path / "evaluation" / "ranking.json").exists()
    assert (tmp_path / "interpretation" / "interpretation_interp_shap.json").exists()
    assert (
        tmp_path
        / "interpretation"
        / "dual"
        / "interpretation_interp_dual_observation_weights.json"
    ).exists()
    assert (
        tmp_path
        / "interpretation"
        / "dual"
        / "interpretation_interp_dual_forecast_diagnostics.json"
    ).exists()
    assert (tmp_path / "metadata" / "metadata.json").exists()
    assert (tmp_path / "metadata" / "summary.json").exists()
    groups = {record.source: record.metadata["group"] for record in manifest.records}
    assert groups["forecasts"] == "forecasts"
    assert groups["metrics"] == "evaluation"
    assert groups["interpretation_interp_shap"] == "interpretation"
    assert (
        groups["interpretation_interp_dual_observation_weights"]
        == "interpretation/dual"
    )
    assert groups["metadata"] == "metadata"
    assert "interpretation/interpretation_interp_shap.json" in manifest.artifacts
    assert (
        "interpretation/dual/interpretation_interp_dual_observation_weights.json"
        in manifest.artifacts
    )
