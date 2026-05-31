from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast import __version__
from macroforecast.forecasting import ForecastResult

ExportFormat = Literal["json", "csv", "parquet", "markdown"]
ManifestFormat = Literal["json", "csv", "parquet"]


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


def write_artifacts(
    artifacts: Mapping[str, Any] | ForecastResult | pd.DataFrame,
    output_dir: str | Path,
    *,
    formats: tuple[ExportFormat, ...] = ("json", "csv"),
    manifest_format: ManifestFormat = "json",
    include_provenance: bool = True,
) -> ArtifactManifest:
    """Write forecast/package artifacts and a reproducibility manifest."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    ) -> None:
        paths[file_name] = path
        records.append(
            ArtifactRecord(
                name=file_name,
                path=path,
                kind=kind,
                format=fmt,
                source=source,
                metadata=dict(metadata or {}),
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
    provenance = collect_provenance() if include_provenance else {}
    manifest = ArtifactManifest(
        output_dir=str(out),
        artifacts=paths,
        records=records,
        provenance=provenance,
    )
    _write_manifest(out, manifest, manifest_format)
    return manifest


def collect_provenance(*, cwd: str | Path | None = None) -> dict[str, Any]:
    """Collect lightweight package, Python, platform, and git provenance."""

    root = Path(cwd or Path.cwd())
    return {
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


def _normalize_artifacts(artifacts: Mapping[str, Any] | ForecastResult | pd.DataFrame) -> dict[str, Any]:
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
    "ExportFormat",
    "ManifestFormat",
    "collect_provenance",
    "write_artifacts",
]
