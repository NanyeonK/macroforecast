# Reproducibility

Reproducibility is a first-class package feature. Every run produces a
bit-exact artifact record that can be independently verified by re-execution.

## Run contract

`mf.run(recipe)` returns a `ManifestExecutionResult`. Each cell in
`ManifestExecutionResult.cells` is a `CellExecutionResult` with a
`sink_hashes: dict[str, str]` field recording the SHA-256 hash of every
output artifact produced by that cell.

```python
result = mf.run("recipe.yaml", output_directory="out/")
for cell in result.cells:
    print(cell.cell_id, cell.sink_hashes)
```

## Replication contract

`mf.replicate(manifest_path)` re-executes the stored recipe from the
manifest and returns a `ReplicationResult`. The field
`ReplicationResult.sink_hashes_match: bool` is `True` when every per-cell
hash matches the original run bit-for-bit.

```python
replication = mf.replicate("out/manifest.json")
assert replication.sink_hashes_match
```

The `ReplicationResult` also carries `per_cell_match` for cell-level
granularity and `recipe_match` to confirm the stored recipe was not altered.

## Seed propagation

Set `reproducibility_mode: seeded_reproducible` and `random_seed` in
`leaf_config` (L0 `0_meta`) to propagate a deterministic seed through the
entire run. The runtime derives per-cell seeds from the base seed plus cell
position so sweeps remain independently reproducible.

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 42
```

## Experiment export

`Experiment.to_yaml()` exports a fully resolved recipe YAML string. This
string can be saved to disk and passed back to `mf.run` to reproduce the
exact study configuration, including all defaults that were applied at
construction time.

```python
import macroforecast as mf

exp = mf.Experiment(target="INDPRO", horizons=[1, 3, 6])
print(exp.to_yaml())  # resolved recipe YAML
```

## What is recorded

The manifest written to `output_directory/manifest.json` carries:

- the resolved recipe (all axes, leaf_config, and sweep markers);
- per-cell `sink_hashes` for all output artifacts;
- per-cell sweep variant values (`sweep_values`);
- run timing (`started_at`, `duration_seconds`);
- the cache root path used during the run.

Custom model and preprocessor names registered via `macroforecast.custom`
are recorded in the manifest so provenance is auditable even when extension
callables are used.
