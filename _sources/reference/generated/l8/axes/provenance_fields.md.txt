# `provenance_fields`

[Back to L8](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``provenance_fields`` on sub-layer ``L8_C_provenance`` (layer ``l8``).

## Sub-layer

**L8_C_provenance**

## Axis metadata

- Default: `None`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 14 option(s)
- Future: 0 option(s)

## Options

### `cell_resolved_axes`  --  operational

Per-cell resolved axis values from sweep expansion.

For sweep-expanded cells, records the (axis → value) mapping that produced each cell. Without this field, interpreting which cell ran which configuration requires re-expanding the sweep.

**When to use**

Default-on when sweeps are active.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `data_revision_tag`  --  operational

FRED vintage / data revision tag.

When the L1 raw is FRED-MD / -QD / -SD, captures the vintage tag (e.g. ``2024-09``) so future re-runs against an updated FRED snapshot can detect that the input data has revised.

**When to use**

Default-on when raw data is FRED-MD / -QD / -SD.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `dependency_lockfile`  --  operational

Lockfile contents (pip freeze / poetry.lock / conda env).

Verbatim contents of the active environment's lockfile. Critical for reproducing the same package versions on a different machine.

**When to use**

Default-on; needed for environment replication.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `git_branch_name`  --  operational

Git branch name.

From ``git rev-parse --abbrev-ref HEAD``. Default-on with git_commit_sha; documents which feature branch produced the run.

**When to use**

Default-on with git_commit_sha.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `git_commit_sha`  --  operational

Git commit SHA of the active checkout.

From ``git rev-parse HEAD``. Default-on when the run executes inside a git working tree; provides exact code traceability.

**When to use**

Default-on when the run executes inside a git tree.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `julia_version`  --  operational

Julia version (when Julia-backed steps are active).

Captured via ``julia`` Python bridge when any pipeline step calls into Julia. Optional.

**When to use**

Recipes that call Julia. Selecting ``julia_version`` on ``l8.provenance_fields`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `package_version`  --  operational

macroforecast version string.

From ``macroforecast.__version__``. Lets ``replicate()`` warn the user when the manifest was produced by a different package version.

**When to use**

Default-on. Selecting ``package_version`` on ``l8.provenance_fields`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`python_version`](#python-version), [`r_version`](#r-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `python_version`  --  operational

Python interpreter version (3-tuple major.minor.patch).

From ``sys.version_info``. Lets ``replicate()`` warn when running on a different interpreter.

**When to use**

Default-on. Selecting ``python_version`` on ``l8.provenance_fields`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`r_version`](#r-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `r_version`  --  operational

R version (when R-backed steps are active).

Captured via ``rpy2`` when any L3 / L4 / L6 / L7 op calls into R. Optional -- only emitted when R is actually used.

**When to use**

Recipes that call R (e.g. for arima or robust statistics).

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `random_seed_used`  --  operational

Resolved random seeds (L0 + per-cell propagation).

The exact seed values used by every numpy / sklearn / torch RNG. Required for bit-exact replication; the seed-propagation system in v0.2 ensures every non-deterministic op receives a deterministic seed.

**When to use**

Default-on; required for bit-exact replication.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `recipe_hash`  --  operational

SHA-256 hash of the canonicalised recipe.

Cheap consistency check. Compares against the recipe hash from the original run during ``replicate()``; mismatch triggers a hard error before any compute is wasted.

**When to use**

Default-on; cheap consistency check.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`package_version`](#package-version), [`python_version`](#python-version), [`r_version`](#r-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `recipe_yaml_full`  --  operational

Full recipe YAML embedded in the manifest.

Verbatim copy of the recipe as supplied by the user (post-canonicalisation). Required for ``replicate()`` to reconstruct the exact run; without it the manifest is descriptive but not replayable.

**When to use**

Default-on; required for replication.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version), [`r_version`](#r-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `runtime_duration`  --  operational

Wall-clock duration per cell.

Per-cell timings; useful for cost-tracking and detecting slow cells in a sweep.

Configures the ``provenance_fields`` axis on ``L8_C_provenance`` (layer ``l8``); the ``runtime_duration`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Default-on; useful for cost-tracking.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._

### `runtime_environment`  --  operational

Hostname / OS / CPU summary string.

Captured at run start; useful for diagnosing performance regressions across machines (laptop vs cluster).

**When to use**

Default-on. Selecting ``runtime_environment`` on ``l8.provenance_fields`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`recipe_yaml_full`](#recipe-yaml-full), [`recipe_hash`](#recipe-hash), [`package_version`](#package-version), [`python_version`](#python-version)

_Last reviewed 2026-05-05 by macroforecast author._
