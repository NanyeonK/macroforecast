# Reproducibility Policy

> Closes phase-00 issue #7. Pinned by the regression batteries in
> `tests/core/test_seed_policy.py`,
> `tests/core/test_deterministic_replay.py`, and
> `tests/core/test_execution_cache.py`.

macrocast v0.1 promises that **the same recipe produces the same artifacts
bit-for-bit**, on the same machine and across machines that share the
package version + dependency lockfile. This page documents what
"reproducible" means in practice, what knobs control it, and what is
deliberately *out of scope*.

## Public API

```python
import macrocast

# Run any recipe (inline YAML, dict, or Path).
result = macrocast.run("recipe.yaml", output_directory="out/")

# Re-execute the stored manifest and verify per-cell sink hashes match.
replication = macrocast.replicate("out/manifest.json")
assert replication.recipe_match
assert replication.sink_hashes_match
```

## Seed-policy modes (L0)

The L0 layer's `reproducibility_mode` axis selects one of two regimes:

| Mode | When | Seed source | Best for |
|------|------|-------------|----------|
| `seeded_reproducible` *(default)* | every run is a deterministic replay | `0_meta.leaf_config.random_seed` (default `0`) | paper replication, regression tests, multi-cell sweeps |
| `exploratory` | seed is left to whatever process state happens to be | none | one-off interactive runs where determinism doesn't matter |

`strict` and any other unknown value are rejected by the L0 schema
validator. Pass `random_seed` explicitly when you want a non-zero base.

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 42
```

`_resolve_seed(recipe_root)` returns:

* the explicit `leaf_config.random_seed` if present,
* `0` for the default `seeded_reproducible` mode,
* `None` for `exploratory` (or any other non-`seeded_reproducible` value).

## What `_apply_seed` actually seeds

A best-effort propagation that covers every RNG macrocast or its
dependencies are likely to touch:

| Library | Call |
|---------|------|
| Python `random` module | `random.seed(seed)` |
| NumPy global state | `np.random.seed(seed % 2**32)` |
| Process env (hash-seed-sensitive iteration) | `os.environ.setdefault("PYTHONHASHSEED", str(seed))` |
| PyTorch (when installed) | `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` |

`scikit-learn` estimators receive `random_state=seed_int` from the L4
recipe params (`_build_l4_model`) -- the global numpy seed isn't enough
for sklearn because most estimators capture `random_state=None` and call
`check_random_state` once. **Pin `random_state` per estimator if you need
deterministic ensembles.**

## Cell-index seed schedule

A multi-cell sweep is *not* run with the same seed in every cell. The
sweep loop applies `base_seed + (cell_index - 1)` so:

* Cell 1 uses `random_seed`.
* Cell 2 uses `random_seed + 1`.
* ... cell N uses `random_seed + N - 1`.

This means two cells of the same recipe with different `{sweep: [...]}`
values produce *different* RNG streams (bug-catching: see
`test_distinct_cells_get_distinct_seeds`), but a re-run of the same
sweep produces *identical* streams cell-by-cell.

## Bit-exact replicate

`macrocast.replicate(manifest_path)` reads the stored manifest, expands
the same sweep, and re-executes every cell. The returned
`ReplicationResult` carries:

* `recipe_match: bool` -- the canonicalized recipe dict round-trips
  identically (key order, sweep marker placement, etc.).
* `sink_hashes_match: bool` -- every cell's per-sink SHA-256 matches
  the original.
* `per_cell_match: dict[str, bool]` -- per-cell breakdown.

Two sinks are exempt from the strict equality check because they
legitimately encode environmental data:

* `l1_data_definition_v1` -- carries `leaf_config.cache_root` which
  depends on the local filesystem layout.
* `l8_artifacts_v1` -- records the absolute paths of exported files.

The other eight sinks (L1 regime, L2, L3 features + metadata, L4
forecasts + models + training, L5 evaluation, plus L6 / L7 / L8 outputs
when produced) are byte-equal across runs.

## Shared raw cache (`cache_root`)

Multi-cell sweeps that hit the same FRED vintage many times share the
on-disk raw cache when you pass `cache_root=`:

```python
macrocast.run(
    "recipe.yaml",
    output_directory="out/sweep_a",
    cache_root="/var/macrocast/raw_cache",
)
```

Resolution order (first non-None wins):

1. The explicit `cache_root=` argument.
2. `recipe['1_data']['leaf_config']['cache_root']` (recipe-level
   override).
3. `output_directory / ".raw_cache"` (auto-derived).
4. The raw loader's package default.

The effective value is recorded in `manifest.json[ "cache_root"]` and on
the `ManifestExecutionResult.cache_root` attribute, so a follow-up run
or a downstream auditor can verify exactly which cache backed the
artifacts.

## Determinism boundaries

| Boundary | Guarantee | Caveats |
|----------|-----------|---------|
| Two re-runs of the same recipe in the same Python session | byte-identical sinks (excluding `l1_data_definition_v1` + `l8_artifacts_v1` when output paths differ) | -- |
| Two re-runs in **different processes** with the same package + lockfile | byte-identical sinks (validated by `tests/core/test_v01_1_hot_patch.py::test_hash_sink_with_set_payload_is_stable_across_processes`, which rotates `PYTHONHASHSEED`) | -- |
| `compute_mode = parallel` cell loop | byte-identical sinks vs. serial run for the same cells (validated by `tests/core/test_compute_mode_parallel.py::test_parallel_matches_serial_sink_hashes`) | `l8_artifacts_v1` legitimately differs because of output paths |
| Across machines with the same package version + lockfile | numerical equality at machine epsilon | floating-point summation order across BLAS implementations can drift on the last bit |
| Across `xgboost` / `lightgbm` / `catboost` versions | best-effort | C++ trees are sensitive to library upgrades; pin via the lockfile |
| Deep-NN families (`lstm` / `gru` / `transformer`) | seeded (we call `torch.manual_seed`) but **not** guaranteed bit-exact across torch versions or CUDA driver versions | install `torch[cpu]` for tighter portability |
| Across `shap` versions or with `shap` not installed | best-effort | the L7 SHAP path falls back to a coefficient / permutation proxy when `shap` is missing; the proxy is itself deterministic |

## Worked examples

### Single-path recipe -> identical artifacts twice

```python
import macrocast
from pathlib import Path

a = macrocast.run("recipe.yaml", output_directory=Path("out/a"))
b = macrocast.run("recipe.yaml", output_directory=Path("out/b"))

# Every cell's sink hashes match (excluding path-dependent l1, l8).
for left, right in zip(a.cells, b.cells):
    for sink_name in left.sink_hashes:
        if sink_name in {"l1_data_definition_v1", "l8_artifacts_v1"}:
            continue
        assert left.sink_hashes[sink_name] == right.sink_hashes[sink_name]
```

### Sweep variant ID -> distinct seed

```python
recipe = """
0_meta:
  fixed_axes: {reproducibility_mode: seeded_reproducible}
  leaf_config: {random_seed: 100}
3_feature_engineering:
  nodes:
    - {id: lag_x, type: step, op: lag, params: {n_lag: {sweep: [1, 2, 3, 4]}}, ...}
"""
result = macrocast.run(recipe)
# Cells get seeds 100, 101, 102, 103.
```

### Replicate the manifest

```python
import macrocast

primary = macrocast.run("paper_recipe.yaml", output_directory="paper_out/")
replication = macrocast.replicate("paper_out/manifest.json")
assert replication.sink_hashes_match
```

## Out of scope

* GPU determinism beyond `torch.manual_seed`. Set
  `torch.use_deterministic_algorithms(True)` and the relevant cuDNN
  flags yourself if you need bit-exact CUDA output -- that is a
  platform-specific decision.
* Reproducibility across BLAS implementations (OpenBLAS vs. MKL vs.
  Apple Accelerate). The L4 estimators are deterministic given fixed
  parameters, but floating-point reductions are not associative.
* Reproducibility across Python versions. The package targets
  `python>=3.10`; minor versions are tested in CI but cross-version
  hash equality is not guaranteed.

## Related tests

| Test file | Pins |
|-----------|------|
| `tests/core/test_seed_policy.py` | `_resolve_seed`, `_apply_seed` contract |
| `tests/core/test_deterministic_replay.py` | identical recipe twice -> identical sinks + byte-identical CSVs |
| `tests/core/test_execution_cache.py` | `cache_root` precedence + shared cache + independence |
| `tests/core/test_v01_1_hot_patch.py` | `set` hashing across `PYTHONHASHSEED` |
| `tests/core/test_compute_mode_parallel.py` | parallel run matches serial run |
| `tests/core/test_execute_recipe_dispatch.py` | str-vs-Path dispatch + deprecation |

## Related issues

* #4 -- `cache_root` parameter on `execute_recipe`
* #6 -- determinism regression battery
* #167 -- L6/L7 numerical golden tests
* #169 -- explicit dispatch in `execute_recipe`
