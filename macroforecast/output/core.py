from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
import gzip
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Literal
import zipfile

import numpy as np
import pandas as pd

from macroforecast import __version__
from macroforecast.forecasting import ForecastResult

ExportFormat = Literal["json", "csv", "parquet", "markdown"]
ManifestFormat = Literal["json", "csv", "parquet"]
CompressionFormat = Literal["none", "gzip", "zip"]


@dataclass(frozen=True)
class ArtifactRecord:
    """One written artifact in a manifest."""

    name: str
    path: str
    kind: str
    format: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "format": self.format,
            "source": self.source,
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class ArtifactManifest:
    """Manifest returned by ``write_artifacts``."""

    output_dir: str
    artifacts: dict[str, str] = field(default_factory=dict)
    records: list[ArtifactRecord] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata_schema: dict[str, Any] = field(
        default_factory=lambda: {"kind": "artifact_manifest", "version": 1}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_schema": dict(self.metadata_schema),
            "created_at": self.created_at,
            "output_dir": self.output_dir,
            "artifacts": dict(self.artifacts),
            "records": [record.to_dict() for record in self.records],
            "provenance": _json_ready(self.provenance),
        }

    def to_frame(self) -> pd.DataFrame:
        """Return artifact records as a table."""

        rows: list[dict[str, Any]] = []
        for record in self.records:
            row = record.to_dict()
            row["metadata"] = json.dumps(
                _json_ready(row["metadata"]),
                ensure_ascii=False,
                sort_keys=True,
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def to_json(self, path: str | Path | None = None, *, indent: int | None = 2) -> str:
        """Return JSON text, and optionally write it to ``path``."""

        text = json.dumps(_json_ready(self.to_dict()), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


@dataclass(frozen=True)
class OutputBundle:
    """Named output objects produced before artifact writing."""

    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_schema: dict[str, Any] = field(
        default_factory=lambda: {"kind": "output_bundle", "version": 1}
    )

    def to_artifacts(self) -> dict[str, Any]:
        """Return a shallow copy suitable for ``write_artifacts``."""

        return dict(self.artifacts)

    def select(self, objects: tuple[str, ...] | list[str]) -> "OutputBundle":
        """Return a bundle with selected artifact names."""

        return select_outputs(self, objects=objects)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready description of the bundle."""

        return {
            "metadata_schema": dict(self.metadata_schema),
            "metadata": _json_ready(self.metadata),
            "artifacts": {
                name: _bundle_artifact_summary(value)
                for name, value in self.artifacts.items()
            },
        }


def forecast_table(result: ForecastResult | pd.DataFrame) -> pd.DataFrame:
    """Return the standard forecast table output."""

    if isinstance(result, ForecastResult):
        table = result.to_frame()
        metadata = {"source": "ForecastResult", "metadata_keys": sorted(result.metadata)}
    elif isinstance(result, pd.DataFrame):
        table = result.copy()
        metadata = {"source": "DataFrame"}
    else:
        raise TypeError("forecast_table expects a ForecastResult or DataFrame")
    return _attach_output_schema(table, kind="forecast_table", metadata=metadata)


def metric_table(report: Any) -> pd.DataFrame:
    """Return the main metric score table from an evaluation output."""

    if hasattr(report, "scores"):
        table = report.scores.copy()
        metadata = {"source": type(report).__name__, "field": "scores"}
    elif isinstance(report, ForecastResult):
        table = report.evaluate()
        metadata = {"source": "ForecastResult.evaluate"}
    elif isinstance(report, pd.DataFrame):
        table = report.copy()
        metadata = {"source": "DataFrame"}
    else:
        raise TypeError("metric_table expects an EvaluationReport, ForecastResult, or DataFrame")
    return _attach_output_schema(table, kind="metric_table", metadata=metadata)


def ranking_table(report: Any) -> pd.DataFrame:
    """Return a ranking table from an evaluation output."""

    if hasattr(report, "ranking"):
        table = report.ranking.copy()
        metadata = {"source": type(report).__name__, "field": "ranking"}
    elif isinstance(report, pd.DataFrame):
        table = report.copy()
        metadata = {"source": "DataFrame"}
    else:
        raise TypeError("ranking_table expects an EvaluationReport or DataFrame")
    return _attach_output_schema(table, kind="ranking_table", metadata=metadata)


def test_table(results: Any) -> pd.DataFrame:
    """Return a flat table from one or more forecast test results."""

    if isinstance(results, pd.DataFrame):
        table = results.copy()
    elif hasattr(results, "to_dict") and _looks_like_test_result(results):
        table = pd.DataFrame([_test_record("test", results)])
    elif isinstance(results, Mapping):
        table = pd.DataFrame(
            [
                _test_record(str(name), result)
                for name, result in results.items()
            ]
        )
    elif isinstance(results, (list, tuple)):
        table = pd.DataFrame(
            [
                _test_record(f"test_{pos}", result)
                for pos, result in enumerate(results)
            ]
        )
    else:
        raise TypeError("test_table expects a TestResult, DataFrame, mapping, or sequence")
    return _attach_output_schema(table, kind="test_table")


def model_table(models: Any) -> pd.DataFrame:
    """Return a compact table of fitted model metadata or stored model paths."""

    if isinstance(models, ForecastResult):
        table = _model_table_from_forecasts(models.to_frame())
        metadata = {"source": "ForecastResult"}
    elif isinstance(models, pd.DataFrame):
        table = _model_table_from_forecasts(models)
        metadata = {"source": "DataFrame"}
    elif isinstance(models, Mapping):
        table = pd.DataFrame(
            [
                _model_record(model, alias=str(alias))
                for alias, model in models.items()
            ]
        )
        metadata = {"source": "mapping"}
    elif isinstance(models, (list, tuple)):
        table = pd.DataFrame([_model_record(model) for model in models])
        metadata = {"source": "sequence"}
    else:
        table = pd.DataFrame([_model_record(models)])
        metadata = {"source": "object"}
    return _attach_output_schema(table, kind="model_table", metadata=metadata)


def model_selection_table(model_selection: Any) -> pd.DataFrame:
    """Return model-selection trial or metadata output."""

    if isinstance(model_selection, pd.DataFrame):
        table = model_selection.copy()
        metadata = {"source": "DataFrame"}
    elif hasattr(model_selection, "to_frame"):
        table = model_selection.to_frame()
        metadata = {
            "source": type(model_selection).__name__,
            "best_params": _json_ready(getattr(model_selection, "best_params", None)),
            "best_score": _json_ready(getattr(model_selection, "best_score", None)),
            "method": _json_ready(getattr(model_selection, "method", None)),
        }
    elif hasattr(model_selection, "to_metadata"):
        table = pd.DataFrame([model_selection.to_metadata()])
        metadata = {"source": type(model_selection).__name__}
    elif isinstance(model_selection, Mapping):
        table = pd.DataFrame([dict(model_selection)])
        metadata = {"source": "mapping"}
    else:
        raise TypeError("model_selection_table expects a SearchResult, SearchSpec, DataFrame, or mapping")
    return _attach_output_schema(table, kind="model_selection_table", metadata=metadata)


def interpretation_table(value: Any) -> pd.DataFrame:
    """Return a standardized interpretation output table."""

    if isinstance(value, pd.DataFrame):
        table = value.copy()
        metadata = {"source": "DataFrame"}
    elif isinstance(value, Mapping):
        table = pd.DataFrame([dict(value)])
        metadata = {"source": "mapping"}
    else:
        raise TypeError("interpretation_table expects a DataFrame or mapping")
    return _attach_output_schema(table, kind="interpretation_table", metadata=metadata)


def metadata_table(value: Any, *, prefix: str = "") -> pd.DataFrame:
    """Flatten metadata from a result, report, bundle, mapping, or object."""

    if isinstance(value, ForecastResult):
        metadata = value.metadata
    elif isinstance(value, OutputBundle):
        metadata = {"bundle": value.metadata, "artifacts": value.to_dict()["artifacts"]}
    elif isinstance(value, ArtifactManifest):
        metadata = value.to_dict()
    elif hasattr(value, "metadata"):
        metadata = getattr(value, "metadata")
    elif isinstance(value, Mapping):
        metadata = value
    else:
        metadata = _object_metadata(value)
    rows = [
        {"path": path, "value": _json_ready(item), "type": type(item).__name__}
        for path, item in _flatten_mapping(metadata, prefix=prefix)
    ]
    return _attach_output_schema(pd.DataFrame(rows), kind="metadata_table")


def run_summary(
    result: ForecastResult | pd.DataFrame | None = None,
    *,
    evaluation: Any | None = None,
    tests: Any | None = None,
    model_selection: Any | None = None,
    models: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact JSON output summary for a study run."""

    out: dict[str, Any] = {
        "metadata_schema": {"kind": "run_summary", "version": 1},
        "created_at": datetime.now(UTC).isoformat(),
    }
    if result is not None:
        forecasts = forecast_table(result)
        out["forecasts"] = {
            "rows": int(len(forecasts)),
            "columns": [str(column) for column in forecasts.columns],
            "models": sorted(forecasts["model"].dropna().astype(str).unique().tolist())
            if "model" in forecasts
            else [],
            "horizons": sorted(_json_ready(forecasts["horizon"].dropna().unique().tolist()))
            if "horizon" in forecasts
            else [],
        }
    if evaluation is not None:
        metrics = metric_table(evaluation)
        out["evaluation"] = {"metric_rows": int(len(metrics))}
        if hasattr(evaluation, "ranking"):
            out["evaluation"]["ranking_rows"] = int(len(evaluation.ranking))
    if tests is not None:
        out["tests"] = {"rows": int(len(test_table(tests)))}
    if model_selection is not None:
        table = model_selection_table(model_selection)
        out["model_selection"] = {
            "rows": int(len(table)),
            "best_params": _json_ready(getattr(model_selection, "best_params", None)),
            "best_score": _json_ready(getattr(model_selection, "best_score", None)),
        }
    if models is not None:
        out["models"] = {"rows": int(len(model_table(models)))}
    if metadata is not None:
        out["metadata"] = _json_ready(dict(metadata))
    return out


def artifact_index(value: Any) -> pd.DataFrame:
    """Return an index table for a manifest, output bundle, or artifact mapping."""

    if isinstance(value, ArtifactManifest):
        table = value.to_frame()
    elif isinstance(value, OutputBundle):
        table = pd.DataFrame(
            [
                {
                    "name": name,
                    "kind": _object_kind(artifact),
                    "object_type": f"{type(artifact).__module__}.{type(artifact).__name__}",
                    "metadata_schema": _json_ready(_object_schema(artifact)),
                }
                for name, artifact in value.artifacts.items()
            ]
        )
    elif isinstance(value, Mapping):
        table = artifact_index(OutputBundle(dict(value)))
    else:
        raise TypeError("artifact_index expects an ArtifactManifest, OutputBundle, or mapping")
    return _attach_output_schema(table, kind="artifact_index")


def bundle_outputs(
    *,
    forecasts: ForecastResult | pd.DataFrame | None = None,
    evaluation: Any | None = None,
    tests: Any | None = None,
    models: Any | None = None,
    model_selection: Any | None = None,
    interpretation: Mapping[str, Any] | pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_summary: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> OutputBundle:
    """Build a named bundle of output tables and JSON-ready summaries."""

    artifacts: dict[str, Any] = {}
    if forecasts is not None:
        artifacts["forecasts"] = forecast_table(forecasts)
    if evaluation is not None:
        artifacts["metrics"] = metric_table(evaluation)
        if hasattr(evaluation, "ranking"):
            artifacts["ranking"] = ranking_table(evaluation)
        for name, table in getattr(evaluation, "aggregations", {}).items():
            artifacts[f"metrics_{_safe_name(name)}"] = _attach_output_schema(
                table.copy(),
                kind="metric_aggregation_table",
                metadata={"aggregation": str(name)},
            )
        if getattr(evaluation, "benchmark", None) is not None:
            artifacts["benchmark"] = _attach_output_schema(evaluation.benchmark.copy(), kind="benchmark_table")
        if getattr(evaluation, "regime", None) is not None:
            artifacts["regime"] = _attach_output_schema(evaluation.regime.copy(), kind="regime_table")
        if getattr(evaluation, "decomposition", None) is not None:
            artifacts["decomposition"] = _attach_output_schema(evaluation.decomposition.copy(), kind="decomposition_table")
    if tests is not None:
        artifacts["tests"] = test_table(tests)
    if models is not None:
        artifacts["models"] = model_table(models)
    elif forecasts is not None:
        model_out = model_table(forecasts)
        if not model_out.empty:
            artifacts["models"] = model_out
    if model_selection is not None:
        artifacts["model_selection"] = model_selection_table(model_selection)
    if interpretation is not None:
        if isinstance(interpretation, Mapping):
            for name, table in interpretation.items():
                artifacts[f"interpretation_{_safe_name(name)}"] = interpretation_table(table)
        else:
            artifacts["interpretation"] = interpretation_table(interpretation)
    if metadata is not None:
        artifacts["metadata"] = metadata_table(metadata)
    if include_summary:
        artifacts["summary"] = run_summary(
            forecasts,
            evaluation=evaluation,
            tests=tests,
            model_selection=model_selection,
            models=models,
            metadata=metadata,
        )
    if extra:
        artifacts.update({str(name): value for name, value in extra.items()})
    return OutputBundle(
        artifacts=artifacts,
        metadata={
            "n_artifacts": int(len(artifacts)),
            "artifact_names": list(artifacts),
        },
    )


def select_outputs(
    bundle: OutputBundle | Mapping[str, Any],
    *,
    objects: tuple[str, ...] | list[str],
) -> OutputBundle:
    """Select named outputs from a bundle or mapping."""

    artifact_map = bundle.artifacts if isinstance(bundle, OutputBundle) else dict(bundle)
    missing = [name for name in objects if name not in artifact_map]
    if missing:
        raise KeyError(f"output objects not found: {missing}")
    selected = {name: artifact_map[name] for name in objects}
    metadata = dict(bundle.metadata) if isinstance(bundle, OutputBundle) else {}
    metadata.update({"selected_objects": list(objects), "n_artifacts": len(selected)})
    return OutputBundle(artifacts=selected, metadata=metadata)


def name_outputs(
    bundle: OutputBundle | Mapping[str, Any],
    *,
    convention: Literal["identity", "descriptive", "kind", "prefixed"] = "descriptive",
    prefix: str | None = None,
) -> OutputBundle:
    """Rename output objects before writing artifacts."""

    if convention not in {"identity", "descriptive", "kind", "prefixed"}:
        raise ValueError("convention must be 'identity', 'descriptive', 'kind', or 'prefixed'")
    artifact_map = bundle.artifacts if isinstance(bundle, OutputBundle) else dict(bundle)
    renamed: dict[str, Any] = {}
    for name, value in artifact_map.items():
        if convention == "identity":
            new_name = str(name)
        elif convention == "kind":
            new_name = _object_kind(value)
        elif convention == "prefixed":
            new_name = f"{prefix or 'output'}_{name}"
        else:
            new_name = f"{prefix + '_' if prefix else ''}{_object_kind(value)}_{name}"
        unique = _unique_name(_safe_name(new_name), renamed)
        renamed[unique] = value
    metadata = dict(bundle.metadata) if isinstance(bundle, OutputBundle) else {}
    metadata.update({"naming_convention": convention, "prefix": prefix})
    return OutputBundle(artifacts=renamed, metadata=metadata)


def write_artifacts(
    artifacts: Mapping[str, Any] | ForecastResult | pd.DataFrame | OutputBundle,
    output_dir: str | Path,
    *,
    formats: tuple[ExportFormat, ...] = ("json", "csv"),
    manifest_format: ManifestFormat = "json",
    include_provenance: bool = True,
    provenance_fields: tuple[str, ...] | None = None,
    compression: CompressionFormat = "none",
) -> ArtifactManifest:
    """Write forecast/package artifacts and a reproducibility manifest."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if compression not in {"none", "gzip", "zip"}:
        raise ValueError("compression must be 'none', 'gzip', or 'zip'")
    artifact_map = _normalize_artifacts(artifacts)
    paths: dict[str, str] = {}
    records: list[ArtifactRecord] = []

    def add_record(
        *,
        file_name: str,
        path: str,
        kind: str,
        fmt: str,
        source: str,
        metadata: Mapping[str, Any] | None = None,
        compress_artifact: bool = True,
    ) -> None:
        record_compression = compression if compress_artifact else "none"
        final_path = _compress_file(Path(path), compression=record_compression)
        final_name = Path(final_path).name
        paths[final_name] = final_path
        records.append(
            ArtifactRecord(
                name=final_name,
                path=final_path,
                kind=kind,
                format=fmt,
                source=source,
                metadata={
                    **dict(metadata or {}),
                    **_file_metadata(final_path),
                    "compression": record_compression if record_compression == "gzip" else "none",
                },
            )
        )

    for name, value in artifact_map.items():
        safe = _safe_name(name)
        if isinstance(value, ForecastResult):
            file_name = f"{safe}.json"
            add_record(
                file_name=file_name,
                path=_write_json(out / file_name, value.to_dict()),
                kind="forecast_result",
                fmt="json",
                source=name,
                metadata=_forecast_result_metadata(value),
            )
            forecasts = value.to_frame()
            file_name = f"{safe}_forecasts.csv"
            add_record(
                file_name=file_name,
                path=_write_frame(out / file_name, forecasts),
                kind="forecast_table",
                fmt="csv",
                source=name,
                metadata={**_dataframe_metadata(forecasts), "parent_kind": "forecast_result"},
            )
            for model_record in _stored_model_records(forecasts, source=name):
                add_record(
                    file_name=model_record.name,
                    path=model_record.path,
                    kind=model_record.kind,
                    fmt=model_record.format,
                    source=model_record.source,
                    metadata=model_record.metadata,
                    compress_artifact=False,
                )
        elif isinstance(value, pd.DataFrame):
            for fmt in formats:
                suffix = "md" if fmt == "markdown" else fmt
                file_name = f"{safe}.{suffix}"
                add_record(
                    file_name=file_name,
                    path=_write_value(out / file_name, value, fmt),
                    kind="dataframe",
                    fmt=fmt,
                    source=name,
                    metadata=_dataframe_metadata(value),
                )
        else:
            file_name = f"{safe}.json"
            add_record(
                file_name=file_name,
                path=_write_json(out / file_name, value),
                kind="json",
                fmt="json",
                source=name,
                metadata=_object_metadata(value),
            )
    if compression == "zip" and records:
        zip_record = _write_zip_bundle(out, records)
        paths[zip_record.name] = zip_record.path
        records.append(zip_record)
    provenance = (
        collect_provenance(fields=provenance_fields) if include_provenance else {}
    )
    manifest = ArtifactManifest(
        output_dir=str(out),
        artifacts=paths,
        records=records,
        provenance=provenance,
    )
    _write_manifest(out, manifest, manifest_format)
    return manifest


def collect_provenance(
    *,
    cwd: str | Path | None = None,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Collect lightweight package, Python, platform, and git provenance."""

    root = Path(cwd or Path.cwd())
    provenance = {
        "macroforecast_version": __version__,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(root),
        "git": _git_provenance(root),
        "packages": {
            package: _package_version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels")
        },
    }
    if fields is None:
        return provenance
    return {field: provenance[field] for field in fields if field in provenance}


def _attach_output_schema(
    table: pd.DataFrame,
    *,
    kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    table.attrs["macroforecast_metadata_schema"] = {
        "kind": kind,
        "version": 1,
        "columns": [str(column) for column in table.columns],
        "n_rows": int(len(table)),
    }
    table.attrs["macroforecast_metadata"] = _json_ready(dict(metadata or {}))
    return table


def _bundle_artifact_summary(value: Any) -> dict[str, Any]:
    summary = {
        "kind": _object_kind(value),
        "object_type": f"{type(value).__module__}.{type(value).__name__}",
        "metadata_schema": _json_ready(_object_schema(value)),
    }
    if isinstance(value, pd.DataFrame):
        summary.update({"shape": [int(value.shape[0]), int(value.shape[1])]})
    elif isinstance(value, ForecastResult):
        summary.update({"forecast_rows": int(len(value.to_frame()))})
    elif isinstance(value, Mapping):
        summary.update({"keys": sorted(str(key) for key in value)})
    elif isinstance(value, (list, tuple, set)):
        summary.update({"length": int(len(value))})
    return summary


def _looks_like_test_result(value: Any) -> bool:
    if all(hasattr(value, field) for field in ("statistic", "p_value", "decision")):
        return True
    if hasattr(value, "to_dict"):
        try:
            payload = value.to_dict()
        except TypeError:
            return False
        return {"statistic", "p_value", "decision"} <= set(payload)
    return False


def _test_record(name: str, result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = _object_metadata(result)
    metadata = _json_ready(payload.get("metadata", {}))
    return {
        "name": name,
        "statistic": payload.get("statistic"),
        "p_value": payload.get("p_value"),
        "decision": payload.get("decision"),
        "alternative": payload.get("alternative"),
        "correction_policy": payload.get("correction_policy"),
        "n_obs": payload.get("n_obs"),
        "statistic_type": metadata.get("statistic_type") if isinstance(metadata, Mapping) else None,
        "p_value_status": metadata.get("p_value_status") if isinstance(metadata, Mapping) else None,
        "p_value_reference": metadata.get("p_value_reference") if isinstance(metadata, Mapping) else None,
        "null_hypothesis": metadata.get("null_hypothesis") if isinstance(metadata, Mapping) else None,
        "critical_value": metadata.get("critical_value") if isinstance(metadata, Mapping) else None,
        "source_reference": metadata.get("source_reference") if isinstance(metadata, Mapping) else None,
        "external_reference": metadata.get("external_reference") if isinstance(metadata, Mapping) else None,
        "r_reference": metadata.get("r_reference") if isinstance(metadata, Mapping) else None,
        "r_alignment": metadata.get("r_alignment") if isinstance(metadata, Mapping) else None,
        "metadata": metadata,
    }


def _model_table_from_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if "model" not in forecasts.columns:
        return pd.DataFrame(
            columns=[
                "model",
                "model_spec",
                "n_forecasts",
                "n_horizons",
                "stored_model_count",
            ]
        )
    group_columns = ["model"]
    if "model_spec" in forecasts.columns:
        group_columns.append("model_spec")
    rows: list[dict[str, Any]] = []
    for keys, group in forecasts.groupby(group_columns, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = {
            "model": key_tuple[0],
            "model_spec": key_tuple[1] if len(key_tuple) > 1 else None,
            "n_forecasts": int(len(group)),
            "n_horizons": int(group["horizon"].nunique()) if "horizon" in group else None,
            "stored_model_count": _stored_model_count(group),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _stored_model_count(forecasts: pd.DataFrame) -> int:
    if "stored_model" not in forecasts:
        return 0
    return int(
        sum(isinstance(item, Mapping) and bool(item) for item in forecasts["stored_model"])
    )


def _model_record(model: Any, *, alias: str | None = None) -> dict[str, Any]:
    if hasattr(model, "to_metadata"):
        metadata = model.to_metadata()
    elif hasattr(model, "to_dict"):
        metadata = model.to_dict()
    elif isinstance(model, Mapping):
        metadata = dict(model)
    else:
        metadata = _object_metadata(model)
    fit = metadata.get("fit", metadata) if isinstance(metadata, Mapping) else {}
    return {
        "alias": alias,
        "model": getattr(model, "model", fit.get("model")),
        "estimator": fit.get("estimator")
        or f"{type(getattr(model, 'estimator', model)).__module__}."
        f"{type(getattr(model, 'estimator', model)).__name__}",
        "n_features": fit.get("n_features", len(getattr(model, "feature_names", ()))),
        "target_name": fit.get("target_name", getattr(model, "target_name", None)),
        "metadata": _json_ready(metadata),
    }


def _flatten_mapping(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []

    def walk(current: Any, path: str) -> None:
        if isinstance(current, Mapping):
            if not current and path:
                rows.append((path, {}))
            for key, item in current.items():
                next_path = f"{path}.{key}" if path else str(key)
                walk(item, next_path)
        elif isinstance(current, (list, tuple)):
            if not current and path:
                rows.append((path, []))
            for pos, item in enumerate(current):
                walk(item, f"{path}[{pos}]")
        else:
            rows.append((path or prefix or "value", current))

    walk(value, prefix)
    return rows


def _object_kind(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        schema = _object_schema(value)
        return str(schema.get("kind", "dataframe")) if isinstance(schema, Mapping) else "dataframe"
    if isinstance(value, ForecastResult):
        return "forecast_result"
    if isinstance(value, ArtifactManifest):
        return "artifact_manifest"
    if isinstance(value, OutputBundle):
        return "output_bundle"
    if isinstance(value, Mapping):
        return "json"
    if isinstance(value, (list, tuple, set)):
        return "sequence"
    return type(value).__name__.lower()


def _object_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        return _json_ready(dict(value.attrs).get("macroforecast_metadata_schema")) or {}
    if hasattr(value, "metadata_schema"):
        return _json_ready(getattr(value, "metadata_schema")) or {}
    if isinstance(value, ForecastResult):
        return _json_ready(value.metadata.get("metadata_schema")) or {}
    return {}


def _unique_name(name: str, existing: Mapping[str, Any]) -> str:
    if name not in existing:
        return name
    pos = 2
    while f"{name}_{pos}" in existing:
        pos += 1
    return f"{name}_{pos}"


def _normalize_artifacts(
    artifacts: Mapping[str, Any] | ForecastResult | pd.DataFrame | OutputBundle,
) -> dict[str, Any]:
    if isinstance(artifacts, OutputBundle):
        return artifacts.to_artifacts()
    if isinstance(artifacts, ForecastResult):
        return {"forecast_result": artifacts}
    if isinstance(artifacts, pd.DataFrame):
        return {"dataframe": artifacts}
    return {str(key): value for key, value in artifacts.items()}


def _write_value(path: Path, value: pd.DataFrame, fmt: ExportFormat) -> str:
    if fmt == "json":
        return _write_json(path, _dataframe_payload(value))
    if fmt == "csv":
        return _write_frame(path, value)
    if fmt == "parquet":
        value.to_parquet(path, index=True)
        return str(path)
    if fmt == "markdown":
        try:
            text = value.to_markdown(index=True)
        except ImportError as exc:
            raise ImportError(
                "markdown export requires tabulate; install macroforecast[markdown]"
            ) from exc
        path.write_text(text + "\n", encoding="utf-8")
        return str(path)
    raise ValueError("formats must contain only 'json', 'csv', 'parquet', or 'markdown'")


def _write_manifest(out: Path, manifest: ArtifactManifest, manifest_format: ManifestFormat) -> None:
    if manifest_format == "json":
        _write_json(out / "manifest.json", manifest.to_dict())
        return
    if manifest_format == "csv":
        manifest.to_frame().to_csv(out / "manifest.csv", index=False)
        return
    if manifest_format == "parquet":
        manifest.to_frame().to_parquet(out / "manifest.parquet", index=False)
        return
    raise ValueError("manifest_format must be 'json', 'csv', or 'parquet'")


def _write_frame(path: Path, value: pd.DataFrame) -> str:
    value.to_csv(path, index=True)
    return str(path)


def _write_json(path: Path, value: Any) -> str:
    path.write_text(json.dumps(_json_ready(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def _compress_file(path: Path, *, compression: CompressionFormat) -> str:
    if compression != "gzip":
        return str(path)
    if not path.exists():
        return str(path)
    gz_path = path.with_name(path.name + ".gz")
    with path.open("rb") as source, gzip.open(gz_path, "wb") as target:
        target.write(source.read())
    path.unlink()
    return str(gz_path)


def _write_zip_bundle(out: Path, records: list[ArtifactRecord]) -> ArtifactRecord:
    zip_path = out / "artifact_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in records:
            path = Path(record.path)
            if path.exists() and path.is_file():
                archive.write(path, arcname=path.name)
    return ArtifactRecord(
        name=zip_path.name,
        path=str(zip_path),
        kind="artifact_bundle",
        format="zip",
        source="write_artifacts",
        metadata={**_file_metadata(zip_path), "compression": "zip"},
    )


def _file_metadata(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return {"path_exists": False, "size_bytes": None, "sha256": None}
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path_exists": True,
        "size_bytes": int(file_path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _safe_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(name))
    return safe.strip("_") or "artifact"


def _git_provenance(cwd: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(cwd),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(run_git("status", "--porcelain")),
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _forecast_result_metadata(value: ForecastResult) -> dict[str, Any]:
    forecasts = value.to_frame()
    stored_model_count = (
        len(_stored_model_records(forecasts, source="forecast_result"))
        if "stored_model" in forecasts.columns
        else 0
    )
    return {
        "object_type": f"{type(value).__module__}.{type(value).__name__}",
        "forecast_rows": int(len(forecasts)),
        "forecast_columns": [str(column) for column in forecasts.columns],
        "stored_model_artifacts": int(stored_model_count),
        "metadata_keys": sorted(str(key) for key in value.metadata),
        "metadata_schema": _json_ready(value.metadata.get("metadata_schema")),
    }


def _dataframe_metadata(value: pd.DataFrame) -> dict[str, Any]:
    attrs = dict(value.attrs)
    return {
        "object_type": "pandas.DataFrame",
        "shape": [int(value.shape[0]), int(value.shape[1])],
        "columns": [str(column) for column in value.columns],
        "index_name": None if value.index.name is None else str(value.index.name),
        "attrs": _json_ready(attrs),
        "metadata_schema": _json_ready(attrs.get("macroforecast_metadata_schema")),
    }


def _dataframe_payload(value: pd.DataFrame) -> dict[str, Any]:
    return {
        "metadata_schema": {"kind": "dataframe_artifact", "version": 1},
        "shape": [int(value.shape[0]), int(value.shape[1])],
        "columns": [str(column) for column in value.columns],
        "index_name": None if value.index.name is None else str(value.index.name),
        "index": [_json_ready(item) for item in value.index],
        "attrs": _json_ready(dict(value.attrs)),
        "data": _json_ready(value.to_dict(orient="records")),
    }


def _object_metadata(value: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"object_type": f"{type(value).__module__}.{type(value).__name__}"}
    if isinstance(value, Mapping):
        metadata["keys"] = sorted(str(key) for key in value)
    elif isinstance(value, (list, tuple, set)):
        metadata["length"] = len(value)
    return metadata


def _stored_model_records(forecasts: pd.DataFrame, *, source: str) -> list[ArtifactRecord]:
    if "stored_model" not in forecasts.columns:
        return []
    records: list[ArtifactRecord] = []
    seen: set[tuple[str, str]] = set()
    for row_pos, row in forecasts.iterrows():
        stored = row.get("stored_model")
        if not isinstance(stored, Mapping):
            continue
        model_label = str(row.get("model", "model"))
        origin_pos = row.get("origin_pos", row_pos)
        horizon = row.get("horizon", "unknown")
        base_metadata = {
            "model": model_label,
            "model_spec": row.get("model_spec"),
            "origin": row.get("origin"),
            "origin_pos": origin_pos,
            "horizon": horizon,
            "save_error": stored.get("save_error"),
            "stored_model": dict(stored),
        }
        for key, kind, fmt in (
            ("model_path", "stored_model_pickle", "pickle"),
            ("metadata_path", "stored_model_metadata", "json"),
        ):
            path_value = stored.get(key)
            if not path_value:
                continue
            path_text = str(path_value)
            seen_key = (kind, path_text)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            path = Path(path_text)
            record_name = (
                f"stored_model_{_safe_name(model_label)}_"
                f"origin_{_safe_name(origin_pos)}_h{_safe_name(horizon)}_"
                f"{'fit' if key == 'model_path' else 'metadata'}"
                f"{path.suffix or ''}"
            )
            records.append(
                ArtifactRecord(
                    name=record_name,
                    path=path_text,
                    kind=kind,
                    format=fmt,
                    source=f"{source}.stored_model",
                    metadata={**base_metadata, "path_exists": path.exists()},
                )
            )
    return records


def _json_ready(value: Any) -> Any:
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return _dataframe_payload(value)
    if isinstance(value, pd.Series):
        return [_json_ready(item) for item in value.to_list()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "ArtifactManifest",
    "ArtifactRecord",
    "CompressionFormat",
    "ExportFormat",
    "ManifestFormat",
    "OutputBundle",
    "artifact_index",
    "bundle_outputs",
    "collect_provenance",
    "forecast_table",
    "interpretation_table",
    "metadata_table",
    "metric_table",
    "model_table",
    "name_outputs",
    "ranking_table",
    "run_summary",
    "model_selection_table",
    "select_outputs",
    "test_table",
    "write_artifacts",
]
