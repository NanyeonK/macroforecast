"""End-to-end recipe execution: cell loop, sweep expansion, replicate.

Wraps :func:`macrocast.core.runtime.execute_minimal_forecast` with sweep
iteration, seed propagation, failure capture and bit-exact replication.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import traceback as traceback_module
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cache import canonical_dict
from .runtime import RuntimeResult, execute_minimal_forecast
from .yaml import parse_recipe_yaml


L0_KEY = "0_meta"


def _walk_sweep_paths(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], list[Any]]]:
    """Find every ``{sweep: [...]}`` marker in a recipe-root tree.

    Returns a list of ``(path, sweep_values)`` in document order. ``path`` is a
    tuple of dict-keys / list-indices to navigate into the tree.
    """

    found: list[tuple[tuple[Any, ...], list[Any]]] = []
    if isinstance(value, dict):
        sweep_values = value.get("sweep")
        if isinstance(sweep_values, list) and len(value) == 1:
            found.append((path, list(sweep_values)))
            return found
        for key, child in value.items():
            found.extend(_walk_sweep_paths(child, path + (key,)))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found.extend(_walk_sweep_paths(child, path + (idx,)))
    return found


def _set_at(root: Any, path: tuple[Any, ...], value: Any) -> None:
    cursor = root
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def _path_label(path: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in path)


def _expand_cells(recipe_root: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[tuple[Any, ...], Any]]]:
    """Expand every ``{sweep: [...]}`` marker into concrete recipe roots.

    Returns ``(concrete_roots, sweep_paths)``. When the recipe has no sweep
    markers, returns ``([deep_copy_of_root], [])`` so the caller can iterate a
    single cell.
    """

    paths = _walk_sweep_paths(recipe_root)
    if not paths:
        return [copy.deepcopy(recipe_root)], []

    combo_mode = ((recipe_root.get("sweep_combination", {}) or {}).get("mode")) or "grid"
    options = [values for _, values in paths]

    if combo_mode == "zip":
        lengths = {len(values) for values in options}
        if len(lengths) != 1:
            raise ValueError("zip sweep combination requires equal sweep lengths")
        combos = [tuple(values[idx] for values in options) for idx in range(next(iter(lengths)))]
    elif combo_mode == "grid":
        combos = []
        from itertools import product

        for combo in product(*options):
            combos.append(combo)
    else:
        raise ValueError(f"unsupported sweep_combination.mode={combo_mode!r}")

    concrete: list[dict[str, Any]] = []
    for combo in combos:
        cell = copy.deepcopy(recipe_root)
        for (path, _values), value in zip(paths, combo):
            _set_at(cell, path, value)
        concrete.append(cell)
    return concrete, [(path, None) for path, _ in paths]


def _generate_cell_id(index: int, sweep_values: dict[str, Any], naming: str = "descriptive", template: str | None = None) -> str:
    if naming == "cell_id" or not sweep_values:
        return f"cell_{index:03d}"
    if template:
        safe = {key.replace(".", "_").replace("-", "_"): value for key, value in sweep_values.items()}
        try:
            return template.format(**safe)
        except KeyError:
            return f"cell_{index:03d}"
    parts = []
    for key, value in sorted(sweep_values.items()):
        safe_key = key.split(".")[-1]
        safe_value = str(value).replace(" ", "_").replace("/", "_")
        parts.append(f"{safe_key}-{safe_value}")
    return "__".join(parts) or f"cell_{index:03d}"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _resolve_seed(recipe_root: dict[str, Any]) -> int | None:
    l0 = recipe_root.get(L0_KEY, {}) or {}
    leaf = (l0.get("leaf_config", {}) or {})
    fixed = (l0.get("fixed_axes", {}) or {})
    if "random_seed" in leaf:
        return int(leaf["random_seed"])
    repro = fixed.get("reproducibility_mode", "seeded_reproducible")
    return 0 if repro == "seeded_reproducible" else None


def _resolve_cache_root(
    recipe_root: dict[str, Any],
    explicit: str | Path | None,
    output_directory: str | Path | None,
) -> Path | None:
    """Pick the effective raw-cache directory for this study.

    Resolution order (first non-None wins):

    1. ``explicit`` (the ``cache_root=`` argument to ``execute_recipe``)
    2. ``recipe['1_data']['leaf_config']['cache_root']`` (recipe-level override)
    3. ``output_directory / '.raw_cache'``
    4. ``None`` (let the raw loader use its package default)
    """

    if explicit is not None:
        return Path(explicit)
    l1 = recipe_root.get("1_data", {}) or {}
    leaf = l1.get("leaf_config", {}) or {}
    if "cache_root" in leaf and leaf["cache_root"]:
        return Path(leaf["cache_root"])
    if output_directory is not None:
        return Path(output_directory) / ".raw_cache"
    return None


def _inject_cache_root(recipe_root: dict[str, Any], cache_root: Path) -> None:
    """Force ``cache_root`` into ``recipe['1_data']['leaf_config']`` so every
    materialize_l1 call sees the same directory regardless of cell index."""

    l1 = recipe_root.setdefault("1_data", {})
    leaf = l1.setdefault("leaf_config", {})
    leaf["cache_root"] = str(cache_root)


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed_int))
    # Propagate to torch when available so lstm/gru/transformer recipes are
    # bit-exact replicable. Best-effort: torch is an optional dependency.
    try:
        import torch  # type: ignore
    except ImportError:
        return
    torch.manual_seed(seed_int)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


# ---------------------------------------------------------------------------
# Sink hashing (for bit-exact replication)
# ---------------------------------------------------------------------------

def _stable_repr(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(f"{value:.10g}")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    from datetime import date

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _stable_repr(value.item())
    if isinstance(value, np.ndarray):
        return [_stable_repr(item) for item in value.tolist()]
    if isinstance(value, pd.DatetimeIndex):
        return {"kind": "DatetimeIndex", "values": [ts.isoformat() for ts in value]}
    if isinstance(value, pd.Index):
        return {"kind": "Index", "values": [_stable_repr(item) for item in value.tolist()]}
    if isinstance(value, pd.DataFrame):
        return {
            "kind": "DataFrame",
            "columns": [str(col) for col in value.columns],
            "index": [_stable_repr(idx) for idx in value.index],
            "values": [[_stable_repr(cell) for cell in row] for row in value.itertuples(index=False, name=None)],
        }
    if isinstance(value, pd.Series):
        return {
            "kind": "Series",
            "name": str(value.name) if value.name is not None else None,
            "index": [_stable_repr(idx) for idx in value.index],
            "values": [_stable_repr(item) for item in value.tolist()],
        }
    if isinstance(value, dict):
        return {str(_stable_repr(k)): _stable_repr(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        # set iteration order depends on PYTHONHASHSEED, so identical artifacts
        # would otherwise serialize differently across processes and break the
        # bit-exact replicate guarantee. Sort by string repr for stability.
        return sorted((_stable_repr(item) for item in value), key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [_stable_repr(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _stable_repr(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
            if field_name not in {"fitted_object", "raw_panel", "panel"}
            # exclude unhashable fitted_object; panel data is captured via shape/columns elsewhere
        }
    if isinstance(value, Path):
        return value.as_posix()
    return repr(value)


def _hash_sink(sink: Any) -> str:
    payload = json.dumps(_stable_repr(sink), sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _canonicalize_keys(value: Any) -> Any:
    """Recursively sort dict keys so that authoring/serialization order does
    not bleed into runtime determinism.

    Preserves list order (lists are positional); only dict keys are sorted.
    """

    if isinstance(value, dict):
        return {key: _canonicalize_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_keys(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    from dataclasses import asdict as _asdict, is_dataclass as _is_dataclass
    from datetime import date, datetime as _dt

    if _is_dataclass(value) and not isinstance(value, type):
        return _json_safe(_asdict(value))
    if isinstance(value, (date, _dt, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(_json_safe(k)) if not isinstance(k, str) else k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


# ---------------------------------------------------------------------------
# Per-cell provenance helpers (used by ManifestExecutionResult.to_manifest_dict)
# ---------------------------------------------------------------------------

def _per_layer_durations(cell: "CellExecutionResult") -> dict[str, float]:
    rt = cell.runtime_result
    if rt is None:
        return {}
    return dict(getattr(rt, "runtime_durations", {}) or {})


def _per_cell_resolved_axes(cell: "CellExecutionResult") -> dict[str, Any]:
    rt = cell.runtime_result
    if rt is None:
        return {}
    return _json_safe(dict(getattr(rt, "resolved_axes", {}) or {}))


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellExecutionResult:
    cell_id: str
    index: int
    sweep_values: dict[str, Any]
    duration_seconds: float
    runtime_result: RuntimeResult | None = None
    sink_hashes: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.runtime_result is not None and self.error is None


@dataclass(frozen=True)
class ManifestExecutionResult:
    recipe_root: dict[str, Any]
    cells: tuple[CellExecutionResult, ...]
    failure_policy: str
    sweep_paths: tuple[str, ...] = ()
    duration_seconds: float = 0.0
    started_at: str = ""
    cache_root: str | None = None

    @property
    def succeeded(self) -> tuple[CellExecutionResult, ...]:
        return tuple(cell for cell in self.cells if cell.succeeded)

    @property
    def failed(self) -> tuple[CellExecutionResult, ...]:
        return tuple(cell for cell in self.cells if not cell.succeeded)

    def to_manifest_dict(self) -> dict[str, Any]:
        from .runtime import (
            _capture_data_revision_tag,
            _capture_dependency_lockfile_content,
            _capture_full_runtime_environment,
            _capture_git_state,
            _capture_package_version,
            _capture_random_seed_used,
            _command_version_safe,
            _dependency_lockfile_paths,
        )

        git_sha, git_branch = _capture_git_state()
        provenance = {
            "package_version": _capture_package_version(),
            "python_version": platform.python_version(),
            "r_version": _command_version_safe(("R", "--version")),
            "julia_version": _command_version_safe(("julia", "--version")),
            "git_commit_sha": git_sha,
            "git_branch_name": git_branch,
            "data_revision_tag": _capture_data_revision_tag(self.recipe_root),
            "random_seed_used": _capture_random_seed_used(self.recipe_root),
            "dependency_lockfile_paths": _dependency_lockfile_paths(),
            "dependency_lockfile_content": _capture_dependency_lockfile_content(),
            "runtime_environment": _json_safe(_capture_full_runtime_environment()),
        }
        return {
            "schema_version": "0.1.0",
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "failure_policy": self.failure_policy,
            "sweep_paths": list(self.sweep_paths),
            "recipe_root": _json_safe(self.recipe_root),
            "cache_root": self.cache_root,
            "runtime_environment": {
                "python_version": platform.python_version(),
                "os_name": platform.system(),
                "machine": platform.machine(),
            },
            "provenance": provenance,
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "index": cell.index,
                    "sweep_values": canonical_dict(cell.sweep_values),
                    "duration_seconds": cell.duration_seconds,
                    "succeeded": cell.succeeded,
                    "sink_hashes": dict(cell.sink_hashes),
                    "error": cell.error,
                    "runtime_duration_per_layer": _per_layer_durations(cell),
                    "cell_resolved_axes": _per_cell_resolved_axes(cell),
                }
                for cell in self.cells
            ],
        }

    def write_manifest(self, output_directory: str | Path, json_lines: bool | None = None) -> Path:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_format = self._resolve_manifest_format()
        if json_lines is True:
            manifest_format = "json_lines"
        payload = _json_safe(self.to_manifest_dict())
        if manifest_format == "json_lines":
            target = directory / "manifest.jsonl"
            with target.open("w", encoding="utf-8") as fh:
                base = {key: value for key, value in payload.items() if key != "cells"}
                fh.write(json.dumps(base, sort_keys=True) + "\n")
                for cell in payload["cells"]:
                    fh.write(json.dumps(cell, sort_keys=True) + "\n")
        elif manifest_format == "yaml":
            try:
                import yaml as _yaml  # type: ignore
            except ImportError:
                target = directory / "manifest.json"
                target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            else:
                target = directory / "manifest.yaml"
                target.write_text(_yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        else:  # default json
            target = directory / "manifest.json"
            target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def _resolve_manifest_format(self) -> str:
        l8 = (self.recipe_root.get("8_output") or {})
        for source in ("fixed_axes", "leaf_config"):
            block = l8.get(source) or {}
            if "manifest_format" in block:
                return str(block["manifest_format"])
        return "json"


@dataclass(frozen=True)
class ReplicationResult:
    manifest_path: Path
    recipe_match: bool
    sink_hashes_match: bool
    per_cell_match: dict[str, bool]
    new_result: ManifestExecutionResult


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def execute_recipe(
    recipe: str | dict[str, Any] | Path,
    *,
    output_directory: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> ManifestExecutionResult:
    """Execute every sweep cell defined by ``recipe`` and return a manifest result.

    Accepted ``recipe`` shapes:

    * ``dict`` -- already-parsed recipe-root mapping (deep-copied internally).
    * :class:`pathlib.Path` -- YAML file to read and parse.
    * ``str`` -- inline YAML text. **As of v0.1 the str-as-path heuristic is
      deprecated**: a string that does not contain a newline and that names
      an existing file still loads from disk for back-compat, but a
      ``DeprecationWarning`` is raised. Pass a :class:`pathlib.Path` (or call
      :func:`execute_recipe_file`) for the file path; pass plain ``str`` for
      inline YAML.

    Honors L0 ``failure_policy`` so that a single failing cell does not abort
    the rest of the sweep when the policy is ``continue_on_failure``.

    Parameters
    ----------
    recipe
        YAML string, parsed recipe dict, or path to a YAML file.
    output_directory
        Directory to write ``manifest.json`` and per-cell artifacts into.
    cache_root
        Shared raw-data cache directory. When provided, every sweep cell's
        L1 raw loader (FRED-MD/QD/SD) reuses the same on-disk cache so a
        large multi-cell sweep does not redownload the same vintage N times.
        Resolution order:

        1. Explicit ``cache_root`` argument (this kwarg)
        2. ``recipe['1_data']['leaf_config']['cache_root']`` already set in YAML
        3. ``output_directory / ".raw_cache"`` (when ``output_directory`` given)
        4. The raw loader's package default

        Items 1 and 2 are mutually exclusive — if both are set, the explicit
        argument wins and we override the in-recipe value (a soft warning is
        attached to the manifest's environment metadata).

    Cell loop concurrency is controlled by L0 ``compute_mode``:

    * ``serial`` (default) -- iterate cells in-process.
    * ``parallel`` -- dispatch cells to a ``ProcessPoolExecutor`` of size
      ``leaf_config.n_workers`` (default = ``min(8, os.cpu_count() - 1)``).
      Each worker re-applies ``base_seed + cell_index`` so determinism is
      preserved. ``parallel_unit`` is accepted at the schema level but
      currently always interpreted as cell-level (sub-cell parallelism is
      tracked as a follow-up).
    """

    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()

    if isinstance(recipe, Path):
        recipe_root = parse_recipe_yaml(recipe.read_text(encoding="utf-8"))
    elif isinstance(recipe, str):
        if "\n" not in recipe and Path(recipe).exists():
            import warnings

            warnings.warn(
                "execute_recipe: passing a file path as `str` is deprecated -- "
                "pass `Path(...)` (or call `execute_recipe_file`) for files; "
                "plain `str` is reserved for inline YAML in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            recipe_root = parse_recipe_yaml(Path(recipe).read_text(encoding="utf-8"))
        else:
            recipe_root = parse_recipe_yaml(recipe)
    else:
        recipe_root = copy.deepcopy(recipe)

    # Canonicalize key order so the same recipe produces bit-identical artifacts
    # regardless of YAML/JSON authoring order or roundtrip representation.
    recipe_root = _canonicalize_keys(recipe_root)

    # Resolve the effective raw cache root and inject it into every cell's L1
    # leaf_config so the raw loader picks it up.
    effective_cache_root = _resolve_cache_root(recipe_root, cache_root, output_directory)
    if effective_cache_root is not None:
        _inject_cache_root(recipe_root, effective_cache_root)

    fixed_axes = (recipe_root.get(L0_KEY, {}) or {}).get("fixed_axes", {}) or {}
    leaf = (recipe_root.get(L0_KEY, {}) or {}).get("leaf_config", {}) or {}
    failure_policy = fixed_axes.get("failure_policy") or "fail_fast"
    compute_mode = fixed_axes.get("compute_mode") or "serial"
    n_workers = int(leaf.get("n_workers", _default_n_workers()))
    base_seed = _resolve_seed(recipe_root)

    concrete_roots, sweep_paths = _expand_cells(recipe_root)
    sweep_paths_str = tuple(_path_label(path) for path, _ in sweep_paths)

    parallel_active = compute_mode == "parallel" and n_workers > 1 and len(concrete_roots) > 1

    cell_jobs: list[tuple[int, dict[str, Any], dict[str, Any], str, int | None]] = []
    for index, concrete in enumerate(concrete_roots, start=1):
        sweep_values = _extract_sweep_values(concrete, recipe_root, sweep_paths)
        cell_id = _generate_cell_id(index, sweep_values)
        seed_for_cell = base_seed if base_seed is None else int(base_seed) + (index - 1)
        cell_jobs.append((index, concrete, sweep_values, cell_id, seed_for_cell))

    if parallel_active:
        cells = _run_cells_parallel(cell_jobs, failure_policy=failure_policy, n_workers=n_workers)
    else:
        cells = _run_cells_serial(cell_jobs, failure_policy=failure_policy)

    total_duration = time.perf_counter() - started_clock
    result = ManifestExecutionResult(
        recipe_root=recipe_root,
        cells=tuple(cells),
        failure_policy=failure_policy,
        sweep_paths=sweep_paths_str,
        duration_seconds=total_duration,
        started_at=started_at,
        cache_root=str(effective_cache_root) if effective_cache_root is not None else None,
    )
    if output_directory is not None:
        result.write_manifest(output_directory)
    return result


def _default_n_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu - 1))


def _run_single_cell(
    index: int,
    concrete_root: dict[str, Any],
    sweep_values: dict[str, Any],
    cell_id: str,
    seed: int | None,
) -> CellExecutionResult:
    """Execute one cell and return its CellExecutionResult.

    Lives at module scope so it is picklable for ProcessPoolExecutor.
    """

    _apply_seed(seed)
    clock = time.perf_counter()
    try:
        runtime_result = execute_minimal_forecast(concrete_root)
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - clock
        return CellExecutionResult(
            cell_id=cell_id,
            index=index,
            sweep_values=sweep_values,
            duration_seconds=duration,
            error=str(exc),
            traceback=traceback_module.format_exc(),
        )
    duration = time.perf_counter() - clock
    sink_hashes = {name: _hash_sink(value) for name, value in runtime_result.artifacts.items()}
    return CellExecutionResult(
        cell_id=cell_id,
        index=index,
        sweep_values=sweep_values,
        duration_seconds=duration,
        runtime_result=runtime_result,
        sink_hashes=sink_hashes,
    )


def _run_cells_serial(jobs, *, failure_policy: str) -> list[CellExecutionResult]:
    cells: list[CellExecutionResult] = []
    for index, concrete, sweep_values, cell_id, seed in jobs:
        result = _run_single_cell(index, concrete, sweep_values, cell_id, seed)
        cells.append(result)
        if not result.succeeded and failure_policy == "fail_fast":
            raise RuntimeError(f"cell {cell_id} failed: {result.error}")
    return cells


def _run_cells_parallel(jobs, *, failure_policy: str, n_workers: int) -> list[CellExecutionResult]:
    """Dispatch cells via ProcessPoolExecutor; preserves cell-index order in
    the returned list."""

    from concurrent.futures import ProcessPoolExecutor, as_completed

    pending: dict = {}
    cells_by_index: dict[int, CellExecutionResult] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for index, concrete, sweep_values, cell_id, seed in jobs:
            future = pool.submit(_run_single_cell, index, concrete, sweep_values, cell_id, seed)
            pending[future] = (index, cell_id)
        for future in as_completed(pending):
            index, cell_id = pending[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = CellExecutionResult(
                    cell_id=cell_id,
                    index=index,
                    sweep_values={},
                    duration_seconds=0.0,
                    error=str(exc),
                    traceback=traceback_module.format_exc(),
                )
            cells_by_index[index] = result
            if not result.succeeded and failure_policy == "fail_fast":
                # Cancel remaining futures and stop. Any worker already
                # running will finish but we discard its result.
                for other in pending:
                    if other is not future:
                        other.cancel()
                raise RuntimeError(f"cell {cell_id} failed in parallel mode: {result.error}")
    return [cells_by_index[i] for i in sorted(cells_by_index)]


def _extract_sweep_values(
    concrete: dict[str, Any],
    original: dict[str, Any],
    sweep_paths: list[tuple[tuple[Any, ...], Any]],
) -> dict[str, Any]:
    if not sweep_paths:
        return {}
    sweep_values: dict[str, Any] = {}
    for path, _ in sweep_paths:
        cursor = concrete
        for key in path:
            cursor = cursor[key]
        sweep_values[_path_label(path)] = cursor
    return sweep_values


def execute_recipe_file(
    path: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> ManifestExecutionResult:
    """Read a YAML recipe from disk and run it through :func:`execute_recipe`.

    Equivalent to ``execute_recipe(Path(path), output_directory=...)`` but
    spelled out so callers cannot accidentally trigger the deprecated
    str-path heuristic. Prefer this over passing a path string to
    ``execute_recipe`` -- a future release will remove that heuristic and a
    plain ``str`` will be interpreted as inline YAML only.
    """

    return execute_recipe(Path(path), output_directory=output_directory)


def replicate_recipe(manifest_path: str | Path) -> ReplicationResult:
    """Re-execute the recipe stored in ``manifest_path`` and verify hashes match.

    The manifest must have been produced by :func:`execute_recipe`. Returns a
    :class:`ReplicationResult` describing whether every per-cell sink hash
    matched the original execution.
    """

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe_root = payload.get("recipe_root")
    if not isinstance(recipe_root, dict):
        raise ValueError(f"manifest at {manifest_path} is missing recipe_root")
    new_result = execute_recipe(recipe_root)

    per_cell_match: dict[str, bool] = {}
    expected_cells = {cell["cell_id"]: cell for cell in payload.get("cells", [])}
    for cell in new_result.cells:
        expected = expected_cells.get(cell.cell_id)
        if expected is None:
            per_cell_match[cell.cell_id] = False
            continue
        expected_hashes = expected.get("sink_hashes", {})
        per_cell_match[cell.cell_id] = (
            expected.get("succeeded", True) == cell.succeeded
            and dict(expected_hashes) == dict(cell.sink_hashes)
        )

    sink_match = bool(per_cell_match) and all(per_cell_match.values())
    recipe_match = canonical_dict(recipe_root) == canonical_dict(new_result.recipe_root)

    return ReplicationResult(
        manifest_path=manifest_path,
        recipe_match=recipe_match,
        sink_hashes_match=sink_match,
        per_cell_match=per_cell_match,
        new_result=new_result,
    )


__all__ = [
    "CellExecutionResult",
    "ManifestExecutionResult",
    "ReplicationResult",
    "execute_recipe",
    "execute_recipe_file",
    "replicate_recipe",
]
