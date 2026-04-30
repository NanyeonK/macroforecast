from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import getpass
import json
import platform
from pathlib import Path
import socket
import subprocess
import traceback as traceback_module
from typing import Any, Literal
from uuid import uuid4

import yaml

from .cache import canonical_dict, recipe_hash
from .dag import LayerId
from .recipe import Recipe
from .sweep import Cell, SweepCombination
from .validator import ValidationReport, validate_recipe
from .yaml import RecipeMetadata

MANIFEST_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class CpuInfo:
    processor: str
    machine: str
    cores: int | None = None


@dataclass(frozen=True)
class GpuInfo:
    devices: tuple[dict[str, Any], ...] = ()
    cuda_version: str | None = None
    cudnn_version: str | None = None
    cublas_version: str | None = None
    deterministic_flags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeEnvironment:
    os_name: str
    os_version: str
    python_version: str
    r_version: str | None
    julia_version: str | None
    cpu_info: CpuInfo
    gpu_info: GpuInfo | None
    hostname: str
    user: str
    working_directory: str


@dataclass(frozen=True)
class CustomPackage:
    name: str
    path: str
    version: str | None = None
    hash: str | None = None


@dataclass(frozen=True)
class DependencyManifest:
    python_lockfile_content: str
    python_lockfile_path: str
    r_lockfile_content: str | None = None
    r_lockfile_path: str | None = None
    julia_manifest: str | None = None
    julia_manifest_path: str | None = None
    custom_packages: tuple[CustomPackage, ...] = ()


@dataclass(frozen=True)
class ResolvedAxis:
    value: Any
    source: Literal["explicit", "derived", "dynamic_default", "package_default"]


@dataclass(frozen=True)
class CellSummary:
    cell_id: str
    sweep_values: dict[str, Any]
    cell_hash: str
    status: Literal["completed", "failed", "skipped"]
    layer_hashes: dict[LayerId, str] = field(default_factory=dict)
    resolved_axes: dict[LayerId, dict[str, ResolvedAxis]] = field(default_factory=dict)
    output_subdirectory: str = ""
    output_files: tuple[str, ...] = ()
    runtime_per_layer: dict[LayerId, float] = field(default_factory=dict)
    total_runtime_seconds: float = 0.0
    peak_memory_mb: float = 0.0


@dataclass(frozen=True)
class FailedCellSummary:
    cell_id: str
    error: str
    traceback: str | None = None


@dataclass(frozen=True)
class LayerExecutionRecord:
    layer_id: LayerId
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    status: Literal["completed", "failed", "skipped_disabled", "skipped_diagnostic_off"]
    nodes_executed: int = 0
    nodes_cache_hit: int = 0
    nodes_cache_miss: int = 0
    produced_sinks: tuple[str, ...] = ()
    sink_hashes: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class ExportedFile:
    path: str
    kind: str
    hash: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class EnvironmentDiff:
    differences: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    critical_keys: tuple[str, ...] = ("os_name", "python_version")

    @property
    def has_critical_diff(self) -> bool:
        return any(key in self.differences for key in self.critical_keys)


@dataclass(frozen=True)
class ReplicationResult:
    original_manifest: "Manifest"
    new_manifest: "Manifest"
    environment_diff: EnvironmentDiff
    matching_recipe_hash: bool


@dataclass(frozen=True)
class Manifest:
    manifest_id: str
    macrocast_version: str
    schema_version: str
    created_at: datetime
    recipe_yaml_full: str
    recipe_hash: str
    recipe_metadata: RecipeMetadata
    runtime_environment: RuntimeEnvironment
    dependencies: DependencyManifest
    sweep_specification: SweepCombination
    cells_summary: tuple[CellSummary, ...] = ()
    failed_cells: tuple[FailedCellSummary, ...] = ()
    layer_execution_log: dict[LayerId, LayerExecutionRecord] = field(default_factory=dict)
    output_directory: str = ""
    exported_files: tuple[ExportedFile, ...] = ()
    validation_report: ValidationReport = field(default_factory=ValidationReport)
    soft_warnings_acknowledged: bool = False

    @classmethod
    def initialize(cls, recipe: Recipe, recipe_yaml_full: str = "", project_root: Path | None = None) -> "Manifest":
        dags = recipe.to_dag_form()
        return cls(
            manifest_id=str(uuid4()),
            macrocast_version=_macrocast_version(),
            schema_version=MANIFEST_SCHEMA_VERSION,
            created_at=datetime.now(timezone.utc),
            recipe_yaml_full=recipe_yaml_full,
            recipe_hash=recipe_hash(dags),
            recipe_metadata=recipe.metadata,
            runtime_environment=capture_runtime_environment(project_root),
            dependencies=capture_dependency_manifest(project_root or Path.cwd()),
            sweep_specification=recipe.sweep_combination,
            validation_report=validate_recipe(recipe),
        )

    def with_layer_record(self, record: LayerExecutionRecord) -> "Manifest":
        log = dict(self.layer_execution_log)
        log[record.layer_id] = record
        return _replace_manifest(self, layer_execution_log=log)

    def with_cell_summary(self, summary: CellSummary) -> "Manifest":
        return _replace_manifest(self, cells_summary=self.cells_summary + (summary,))

    def with_failed_cell(self, summary: FailedCellSummary) -> "Manifest":
        return _replace_manifest(self, failed_cells=self.failed_cells + (summary,))

    def finalize(self, output_directory: Path | str | None = None) -> "Manifest":
        output_dir = str(output_directory) if output_directory is not None else self.output_directory
        return _replace_manifest(self, output_directory=output_dir)

    def to_dict(self) -> dict[str, Any]:
        return _manifest_to_jsonable(self)

    def write_to_disk(self, output_directory: Path, write_yaml: bool = True, json_lines: bool = False) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        lock_dir = output_directory / "lockfiles"
        cells_dir = output_directory / "cells"
        lock_dir.mkdir(exist_ok=True)
        cells_dir.mkdir(exist_ok=True)

        finalized = self.finalize(output_directory)
        (output_directory / "manifest.json").write_text(json.dumps(finalized.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        if write_yaml:
            (output_directory / "manifest.yaml").write_text(yaml.safe_dump(finalized.to_dict(), sort_keys=True), encoding="utf-8")
        if json_lines:
            with (output_directory / "manifest.jsonl").open("w", encoding="utf-8") as fh:
                for cell in finalized.cells_summary:
                    fh.write(json.dumps(_manifest_to_jsonable(cell), sort_keys=True) + "\n")
        (output_directory / "recipe.yaml").write_text(finalized.recipe_yaml_full, encoding="utf-8")
        if finalized.dependencies.python_lockfile_content:
            (lock_dir / Path(finalized.dependencies.python_lockfile_path).name).write_text(
                finalized.dependencies.python_lockfile_content,
                encoding="utf-8",
            )
        if finalized.dependencies.r_lockfile_content and finalized.dependencies.r_lockfile_path:
            (lock_dir / Path(finalized.dependencies.r_lockfile_path).name).write_text(
                finalized.dependencies.r_lockfile_content,
                encoding="utf-8",
            )
        if finalized.dependencies.julia_manifest and finalized.dependencies.julia_manifest_path:
            (lock_dir / Path(finalized.dependencies.julia_manifest_path).name).write_text(
                finalized.dependencies.julia_manifest,
                encoding="utf-8",
            )
        for cell in finalized.cells_summary:
            cell_dir = cells_dir / cell.cell_id
            cell_dir.mkdir(exist_ok=True)
            (cell_dir / "cell_manifest.json").write_text(
                json.dumps(_manifest_to_jsonable(cell), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    @classmethod
    def load(cls, manifest_path: Path | str) -> "Manifest":
        path = Path(manifest_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema_version {raw.get('schema_version')!r}")
        return _manifest_from_dict(raw)


def capture_runtime_environment(project_root: Path | None = None) -> RuntimeEnvironment:
    return RuntimeEnvironment(
        os_name=platform.system(),
        os_version=platform.platform(),
        python_version=platform.python_version(),
        r_version=_command_version(("R", "--version")),
        julia_version=_command_version(("julia", "--version")),
        cpu_info=CpuInfo(processor=platform.processor(), machine=platform.machine(), cores=_cpu_count()),
        gpu_info=_capture_gpu_info(),
        hostname=socket.gethostname(),
        user=getpass.getuser(),
        working_directory=str(project_root or Path.cwd()),
    )


def capture_dependency_manifest(project_root: Path) -> DependencyManifest:
    uv_lock = project_root / "uv.lock"
    renv_lock = project_root / "renv.lock"
    julia_manifest = project_root / "Manifest.toml"
    return DependencyManifest(
        python_lockfile_content=_read_optional(uv_lock),
        python_lockfile_path=str(uv_lock) if uv_lock.exists() else "",
        r_lockfile_content=_read_optional(renv_lock) or None,
        r_lockfile_path=str(renv_lock) if renv_lock.exists() else None,
        julia_manifest=_read_optional(julia_manifest) or None,
        julia_manifest_path=str(julia_manifest) if julia_manifest.exists() else None,
    )


def compare_environments(expected: RuntimeEnvironment, actual: RuntimeEnvironment) -> EnvironmentDiff:
    differences: dict[str, tuple[Any, Any]] = {}
    for key in ("os_name", "python_version", "hostname", "working_directory"):
        expected_value = getattr(expected, key)
        actual_value = getattr(actual, key)
        if expected_value != actual_value:
            differences[key] = (expected_value, actual_value)
    return EnvironmentDiff(differences=differences)


def replicate(manifest_path: Path | str) -> ReplicationResult:
    manifest = Manifest.load(manifest_path)
    current_env = capture_runtime_environment(Path.cwd())
    env_diff = compare_environments(manifest.runtime_environment, current_env)
    recipe = Recipe.from_yaml(manifest.recipe_yaml_full)
    new_manifest = Manifest.initialize(recipe, manifest.recipe_yaml_full, Path.cwd())
    return ReplicationResult(
        original_manifest=manifest,
        new_manifest=new_manifest,
        environment_diff=env_diff,
        matching_recipe_hash=manifest.recipe_hash == new_manifest.recipe_hash,
    )


def layer_record_from_exception(layer_id: LayerId, started_at: datetime, exc: BaseException) -> LayerExecutionRecord:
    finished = datetime.now(timezone.utc)
    return LayerExecutionRecord(
        layer_id=layer_id,
        started_at=started_at,
        finished_at=finished,
        duration_seconds=(finished - started_at).total_seconds(),
        status="failed",
        error=str(exc),
        traceback="".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__)),
    )


def cell_summary_from_cell(cell: Cell, status: Literal["completed", "failed", "skipped"] = "completed") -> CellSummary:
    layer_hashes = {
        layer_id: recipe_hash({layer_id: dag})
        for layer_id, dag in cell.concrete_dag.items()
    }
    return CellSummary(
        cell_id=cell.cell_id,
        sweep_values=cell.sweep_values,
        cell_hash=cell.cache_hash,
        status=status,
        layer_hashes=layer_hashes,
        output_subdirectory=f"cells/{cell.cell_id}",
    )


def _macrocast_version() -> str:
    try:
        from macrocast import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _command_version(command: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).splitlines()
    return text[0] if text else None


def _capture_gpu_info() -> GpuInfo | None:
    try:
        result = subprocess.run(
            ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    devices = tuple({"raw": line.strip()} for line in result.stdout.splitlines() if line.strip())
    return GpuInfo(devices=devices)


def _cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def _read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _replace_manifest(manifest: Manifest, **changes: Any) -> Manifest:
    values = {field_name: getattr(manifest, field_name) for field_name in manifest.__dataclass_fields__}
    values.update(changes)
    return Manifest(**values)


def _manifest_to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: _manifest_to_jsonable(child) for key, child in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _manifest_to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_manifest_to_jsonable(child) for child in value]
    return canonical_dict(value)


def _manifest_from_dict(raw: dict[str, Any]) -> Manifest:
    metadata = raw.get("recipe_metadata", {})
    env = raw["runtime_environment"]
    deps = raw["dependencies"]
    return Manifest(
        manifest_id=raw["manifest_id"],
        macrocast_version=raw["macrocast_version"],
        schema_version=raw["schema_version"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        recipe_yaml_full=raw["recipe_yaml_full"],
        recipe_hash=raw["recipe_hash"],
        recipe_metadata=RecipeMetadata(**{key: metadata.get(key, "") for key in ("name", "description", "author", "created_at")}),
        runtime_environment=RuntimeEnvironment(
            os_name=env["os_name"],
            os_version=env["os_version"],
            python_version=env["python_version"],
            r_version=env.get("r_version"),
            julia_version=env.get("julia_version"),
            cpu_info=CpuInfo(**env["cpu_info"]),
            gpu_info=GpuInfo(**env["gpu_info"]) if env.get("gpu_info") else None,
            hostname=env["hostname"],
            user=env["user"],
            working_directory=env["working_directory"],
        ),
        dependencies=DependencyManifest(
            python_lockfile_content=deps.get("python_lockfile_content", ""),
            python_lockfile_path=deps.get("python_lockfile_path", ""),
            r_lockfile_content=deps.get("r_lockfile_content"),
            r_lockfile_path=deps.get("r_lockfile_path"),
            julia_manifest=deps.get("julia_manifest"),
            julia_manifest_path=deps.get("julia_manifest_path"),
        ),
        sweep_specification=SweepCombination(
            mode=raw.get("sweep_specification", {}).get("mode", "grid"),
            groups=tuple(raw.get("sweep_specification", {}).get("groups", ())),
        ),
        cells_summary=tuple(_cell_summary_from_dict(item) for item in raw.get("cells_summary", ())),
        failed_cells=tuple(FailedCellSummary(**item) for item in raw.get("failed_cells", ())),
        output_directory=raw.get("output_directory", ""),
        exported_files=tuple(ExportedFile(**item) for item in raw.get("exported_files", ())),
        validation_report=ValidationReport(),
        soft_warnings_acknowledged=bool(raw.get("soft_warnings_acknowledged", False)),
    )


def _cell_summary_from_dict(raw: dict[str, Any]) -> CellSummary:
    return CellSummary(
        cell_id=raw["cell_id"],
        sweep_values=raw.get("sweep_values", {}),
        cell_hash=raw["cell_hash"],
        status=raw["status"],
        layer_hashes=raw.get("layer_hashes", {}),
        output_subdirectory=raw.get("output_subdirectory", ""),
        output_files=tuple(raw.get("output_files", ())),
        runtime_per_layer=raw.get("runtime_per_layer", {}),
        total_runtime_seconds=float(raw.get("total_runtime_seconds", 0.0)),
        peak_memory_mb=float(raw.get("peak_memory_mb", 0.0)),
    )
