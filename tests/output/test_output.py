from __future__ import annotations

import json

import pandas as pd

import macroforecast as mf


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
