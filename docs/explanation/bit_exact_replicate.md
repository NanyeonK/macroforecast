# Bit-Exact Replication

In academic forecasting research, a result is only as credible as its
reproducibility. macroforecast makes a specific and verifiable promise: given
the same manifest, re-running the study produces byte-identical artifacts.

This page explains what "bit-exact" means in practice, how the guarantee is
achieved, and what can break it.

---

## What "Bit-Exact" Means

Bit-exact means that the SHA-256 hash of every output artifact matches between
the original run and the replicate run. Not "the same numbers to eight decimal
places." Not "the same algorithm run again." The actual bytes written to disk
are identical.

This is a stronger claim than *results-equivalent* (same values to N decimal
places after rounding) or *method-equivalent* (same algorithm applied by a
different implementation). Results-equivalence allows for floating-point
accumulation differences and silent changes in default parameters.
Method-equivalence allows for entirely different code paths that happen to
produce similar numbers. Bit-exactness allows neither.

The choice of SHA-256 per artifact — rather than a single aggregate checksum
of the output directory — is deliberate. A single changed byte in a forecast
CSV propagates to a hash mismatch on that file alone, making the source of
drift visible. If the metric summary changed but the raw forecasts did not,
the per-artifact hash comparison tells you exactly that.

---

## The Two-Step Contract

macroforecast's replication guarantee operates in two steps.

**Step 1: `mf.run()`.** When a study completes, the runtime returns a
`ManifestExecutionResult`. Each element in `result.cells` is a
`CellExecutionResult` carrying a `sink_hashes` dictionary that maps artifact
names to their SHA-256 hexadecimal digests. The manifest is written to
`output_directory/manifest.json` and contains the complete resolved recipe
along with these hashes.

**Step 2: `mf.replicate()`.** Given the path to a manifest file,
`mf.replicate` reads the stored recipe, re-executes the study from scratch,
computes the SHA-256 hash of every artifact produced in the new run, and
compares them against the stored hashes. The field
`ReplicationResult.sink_hashes_match` is `True` when every per-cell hash
matches bit-for-bit.

A minimal illustration of the contract (not a step-by-step how-to):

```python
# Original run: produces artifacts and writes manifest.json
result = mf.run("recipe.yaml", output_directory="out/")
# result.cells[0].sink_hashes -> {"l4_forecasts_v1": "abc123...", ...}

# Replication: re-executes, verifies hashes
rep = mf.replicate("out/manifest.json")
assert rep.sink_hashes_match  # True iff bit-exact
```

`ReplicationResult` also exposes `per_cell_match` (a list of booleans, one
per cell) and `recipe_match` (confirming the stored recipe was not altered
between runs). These fields provide cell-level granularity when a study spans
many sweep cells and only one cell's output changed.

---

## How Seed Propagation Works

Many forecasting algorithms involve randomness: random forests, gradient
boosting with subsampling, neural networks, and bootstrap tests all require a
random number generator. If different cells in a sweep share a random state,
their outputs interfere with each other and the result is not reproducible
even within a single run.

macroforecast's seed propagation solves this with a per-cell deterministic
derivation. The recipe's `0_meta.leaf_config.random_seed` is the base seed.
The runtime derives a per-cell seed as `base_seed + cell_position`, where
`cell_position` is the zero-indexed position of each cell in the expanded
cell grid. A study that sweeps three lag orders and two model families
produces six cells; each receives a distinct seed (base, base+1, ..., base+5),
ensuring that cells cannot interfere with each other's random state.

To enable seed propagation, the recipe must set:

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 42
```

Without `reproducibility_mode: seeded_reproducible`, stochastic layers
receive no deterministic seed and the replication guarantee does not hold.
The manifest records the mode that was in effect, so a reader auditing a
manifest can immediately see whether the run was seeded.

---

## What the Manifest Records

The manifest is not just a log — it is the complete recipe for replication.
A reader who has only the manifest (and the data) can reproduce the study
without any other documentation.

Fields recorded in `output_directory/manifest.json`:

- **Resolved recipe**: all layer keys, all axis values (including defaults
  that were applied implicitly), all sweep markers, and all `leaf_config`
  entries. The recipe is stored in its fully resolved form, not in the
  abbreviated form the user originally wrote.
- **Per-cell `sink_hashes`**: a dictionary from artifact names to SHA-256
  hexadecimal digests for every artifact each cell produced.
- **Per-cell `sweep_values`**: the specific combination of sweep axis values
  that each cell used, so cell IDs can be interpreted.
- **Run timing**: `started_at` and `duration_seconds` for the full run.
- **Cache root path**: the content-addressed cache directory used during the run.
- **Python and package version**: the macroforecast version and a marker for
  the Python runtime.
- **Custom model and preprocessor names**: any callables registered via
  `macroforecast.custom` are listed by name. This is what enables
  `mf.replicate` to fail gracefully: if a custom model from the original run
  is not re-registered in the new process, the validator raises a clear error
  before any computation begins, rather than silently producing incorrect output.

---

## Why This Matters for Research

Three scenarios illustrate when bit-exact reproducibility is not merely
desirable but necessary.

**Sharing results with a collaborator.** The manifest plus the data is
sufficient for an independent check of any result in the paper. The
collaborator does not need to inspect the code, re-read the recipe, or trust
that the same defaults were applied. They run `mf.replicate("manifest.json")`
and observe whether `sink_hashes_match` is `True`.

**Revising a paper after referee comments.** If a revision changes one model
comparison, the pre-revision and post-revision manifests can be compared
artifact by artifact. The researcher can confirm that the results changed
exactly where the recipe changed and nowhere else. Without per-artifact
hashing, it is not possible to make this statement with confidence.

**Long-running studies after a package update.** If a study is re-run after
a macroforecast update, the hash comparison immediately reveals whether any
artifact changed. If the study should have been stable but a hash mismatches,
the version fields in the manifest make it straightforward to identify which
version change caused the drift.

---

## What Can Break the Guarantee

Bit-exact reproducibility is a property of the full stack, not just of the
macroforecast package. Several factors outside the package's control can
cause a legitimate run with an identical recipe to produce different bytes.

**A different macroforecast version.** The package records its version in the
manifest. Any version that changes how artifacts are computed or serialized
may change the bytes.

**A different Python or NumPy/scikit-learn version.** Floating-point arithmetic
in scientific Python is deterministic for a given platform and library version
but not guaranteed across versions. A NumPy upgrade that changes the BLAS
backend or the implementation of a routine can produce different floating-point
results.

**Operating system or BLAS differences.** On rare occasions, the same NumPy
version compiled against different BLAS libraries (OpenBLAS versus MKL versus
Accelerate) produces different floating-point results in matrix operations.

**Custom models with unmanaged randomness.** A custom model registered via
`macroforecast.custom` that uses randomness not seeded through the runtime's
seed propagation — for example, by calling `numpy.random.rand()` directly
without setting a seed — will produce non-reproducible output regardless of
the recipe's `random_seed` setting.

The manifest deliberately records version metadata to make these failure
modes auditable. When `sink_hashes_match` is `False`, the first diagnostic
step is to compare the manifest's `python_version`, `macroforecast_version`,
and `custom_model_names` fields between the original and replicate runs.

---

## Further Reading

- [Reproducibility](../reference/architecture/reproducibility.md) — the full
  API contract for `mf.run` and `mf.replicate`, including field-level details.
- [Artifacts and Manifest](../reference/architecture/artifacts_and_manifest.md)
  — the complete manifest field list and artifact directory layout.
