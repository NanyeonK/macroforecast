"""User-facing result facades for ``Experiment.run`` outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..execution import ExecutionResult, SweepResult, VariantResult


JsonDict = dict[str, Any]


def _read_json(path: Path) -> JsonDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {"raw": data}


def _scalar_top_level(payload: JsonDict) -> JsonDict:
    scalars: JsonDict = {}
    skip = {
        "metrics_by_horizon",
        "metrics_by_target",
        "comparison_by_horizon",
        "comparison_by_target",
        "model_spec",
        "benchmark_spec",
    }
    for key, value in payload.items():
        if key in skip:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            scalars[key] = value
    return scalars


def _horizon_sort_key(item: tuple[str, Any]) -> tuple[int, int | str]:
    key = str(item[0])
    if key.startswith("h") and key[1:].isdigit():
        return (0, int(key[1:]))
    return (1, key)


def _rows_from_horizon_block(
    payload: JsonDict,
    *,
    block_name: str,
    target_block_name: str,
) -> list[JsonDict]:
    if not payload:
        return []

    target_block = payload.get(target_block_name)
    if isinstance(target_block, dict):
        rows: list[JsonDict] = []
        parent = _scalar_top_level(payload)
        for target, target_payload in sorted(target_block.items()):
            if not isinstance(target_payload, dict):
                continue
            for row in _rows_from_horizon_block(
                target_payload,
                block_name=block_name,
                target_block_name=target_block_name,
            ):
                merged = dict(parent)
                merged.update(row)
                merged["target"] = row.get("target", str(target))
                rows.append(merged)
        return rows

    horizon_block = payload.get(block_name)
    base = _scalar_top_level(payload)
    if not isinstance(horizon_block, dict):
        return [base] if base else []

    rows = []
    for horizon, values in sorted(horizon_block.items(), key=_horizon_sort_key):
        row = dict(base)
        row["horizon"] = str(horizon)
        if isinstance(values, dict):
            row.update(values)
        else:
            row["value"] = values
        rows.append(row)
    return rows


def _metrics_table(metrics: JsonDict) -> pd.DataFrame:
    return pd.DataFrame(
        _rows_from_horizon_block(
            metrics,
            block_name="metrics_by_horizon",
            target_block_name="metrics_by_target",
        )
    )


def _comparison_table(comparison: JsonDict) -> pd.DataFrame:
    return pd.DataFrame(
        _rows_from_horizon_block(
            comparison,
            block_name="comparison_by_horizon",
            target_block_name="comparison_by_target",
        )
    )


def _variant_base_row(variant: VariantResult) -> JsonDict:
    row: JsonDict = {
        "variant_id": variant.variant_id,
        "status": variant.status,
        "artifact_dir": variant.artifact_dir,
        "runtime_seconds": variant.runtime_seconds,
        "error": variant.error,
    }
    row.update(dict(variant.axis_values))
    return row


@dataclass(frozen=True)
class ExperimentRunResult:
    """Convenience wrapper for one executed experiment run.

    The underlying execution artifacts stay on disk. This facade only gives a
    researcher direct access to the files they usually inspect first.
    """

    artifact_dir: str
    execution: ExecutionResult | None = None

    @classmethod
    def from_execution(cls, execution: ExecutionResult) -> "ExperimentRunResult":
        return cls(artifact_dir=execution.artifact_dir, execution=execution)

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "ExperimentRunResult":
        return cls(artifact_dir=str(artifact_dir), execution=None)

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_dir)

    @property
    def spec(self):
        if self.execution is None:
            raise AttributeError("spec is available only on direct comparison-cell results")
        return self.execution.spec

    @property
    def run(self):
        if self.execution is None:
            raise AttributeError("run is available only on direct comparison-cell results")
        return self.execution.run

    @property
    def raw_result(self):
        if self.execution is None:
            raise AttributeError("raw_result is available only on direct comparison-cell results")
        return self.execution.raw_result

    def file_path(self, filename: str) -> Path:
        return self.artifact_path / filename

    def read_json(self, filename: str) -> JsonDict:
        return _read_json(self.file_path(filename))

    @property
    def predictions(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path("predictions.csv"))

    @property
    def forecasts(self) -> pd.DataFrame:
        return self.predictions

    @property
    def metrics_json(self) -> JsonDict:
        return self.read_json("metrics.json")

    @property
    def metrics(self) -> pd.DataFrame:
        return _metrics_table(self.metrics_json)

    @property
    def comparison_json(self) -> JsonDict:
        return self.read_json("comparison_summary.json")

    @property
    def comparison(self) -> pd.DataFrame:
        return _comparison_table(self.comparison_json)

    @property
    def manifest(self) -> JsonDict:
        return self.read_json("manifest.json")

    @property
    def summary_text(self) -> str:
        return self.file_path("summary.txt").read_text(encoding="utf-8")

    def summary(self) -> pd.DataFrame:
        return self.metrics


@dataclass(frozen=True)
class ExperimentSweepResult:
    """Convenience wrapper for a model/preprocessing sweep."""

    sweep: SweepResult

    @property
    def study_id(self) -> str:
        return self.sweep.study_id

    @property
    def output_root(self) -> str:
        return self.sweep.output_root

    @property
    def output_path(self) -> Path:
        return Path(self.sweep.output_root)

    @property
    def manifest_path(self) -> str:
        return self.sweep.manifest_path

    @property
    def per_variant_results(self) -> tuple[VariantResult, ...]:
        return self.sweep.per_variant_results

    @property
    def successful_count(self) -> int:
        return self.sweep.successful_count

    @property
    def failed_count(self) -> int:
        return self.sweep.failed_count

    @property
    def size(self) -> int:
        return self.sweep.size

    @property
    def manifest(self) -> JsonDict:
        return _read_json(Path(self.sweep.manifest_path))

    @property
    def variants(self) -> pd.DataFrame:
        return pd.DataFrame([_variant_base_row(variant) for variant in self.per_variant_results])

    def variant(self, variant_id: str) -> ExperimentRunResult:
        for variant in self.per_variant_results:
            if variant.variant_id == variant_id:
                if variant.artifact_dir is None:
                    raise ValueError(f"variant {variant_id!r} has no artifact directory; status={variant.status!r}")
                return ExperimentRunResult.from_artifact_dir(variant.artifact_dir)
        raise KeyError(f"unknown variant_id {variant_id!r}")

    @property
    def predictions(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for variant in self.per_variant_results:
            if variant.status != "success" or variant.artifact_dir is None:
                continue
            frame = ExperimentRunResult.from_artifact_dir(variant.artifact_dir).predictions
            frame = frame.copy()
            for key, value in reversed(tuple(variant.axis_values.items())):
                column = key if key not in frame.columns else f"axis:{key}"
                frame.insert(0, column, value)
            frame.insert(0, "variant_id", variant.variant_id)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @property
    def forecasts(self) -> pd.DataFrame:
        return self.predictions

    @property
    def metrics(self) -> pd.DataFrame:
        rows: list[JsonDict] = []
        for variant in self.per_variant_results:
            base = _variant_base_row(variant)
            metric_rows = _rows_from_horizon_block(
                dict(variant.metrics_summary or {}),
                block_name="metrics_by_horizon",
                target_block_name="metrics_by_target",
            )
            if not metric_rows:
                rows.append(base)
                continue
            for metric_row in metric_rows:
                row = dict(base)
                row.update(metric_row)
                rows.append(row)
        return pd.DataFrame(rows)

    def compare(self, metric: str = "msfe", *, ascending: bool = True) -> pd.DataFrame:
        table = self.metrics
        if table.empty or metric not in table.columns:
            return table

        preferred = [
            "variant_id",
            "status",
            "target",
            "horizon",
            metric,
            "relative_msfe",
            "oos_r2",
            "benchmark_msfe",
            "n_predictions",
            "model_name",
            "benchmark_name",
            "artifact_dir",
            "runtime_seconds",
            "error",
        ]
        axis_columns = [
            key
            for variant in self.per_variant_results
            for key in variant.axis_values
            if key in table.columns
        ]
        ordered = []
        for column in [*preferred[:2], *axis_columns, *preferred[2:]]:
            if column in table.columns and column not in ordered:
                ordered.append(column)
        remainder = [column for column in table.columns if column not in ordered]
        ranked = table.loc[:, [*ordered, *remainder]]
        return ranked.sort_values(metric, ascending=ascending, kind="stable", na_position="last").reset_index(drop=True)

    @property
    def comparison(self) -> pd.DataFrame:
        return self.compare()

    def summary(self) -> pd.DataFrame:
        return self.compare()
