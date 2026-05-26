# Migration Guide

This guide covers breaking changes and migration steps for macroforecast
across major version transitions. For a complete change log, see
`CHANGELOG.md` at the repository root. For the full deprecation removal
timeline, see [Deprecation Timeline](deprecation_timeline.md).

---

## v0.9.x to v0.10.0 (Upcoming)

The following parameter aliases were renamed in v0.9.x. They emit
`DeprecationWarning` in all v0.9.x releases and will raise `TypeError`
in v0.10.0.

### Experiment constructor and forecast() function

**`model_family=` removed — use `model=`**

```python
# Before (deprecated, removed in v0.10.0)
mf.Experiment(..., model_family="ridge", ...)
mf.forecast(..., model_family="ridge", ...)

# After
mf.Experiment(..., model="ridge", ...)
mf.forecast(..., model="ridge", ...)
```

**`model_families=` removed — use `models=`**

```python
# Before (deprecated)
mf.Experiment(..., model_families=["ridge", "lasso"], ...)

# After
mf.Experiment(..., models=["ridge", "lasso"], ...)
```

**`benchmark_family=` removed — use `benchmark_model=`**

```python
# Before (deprecated)
mf.Experiment(..., benchmark_family="ar_p", ...)

# After
mf.Experiment(..., benchmark_model="ar_p", ...)
```

All three shims are centralized in `macroforecast.api._deprecations`
(`resolve_model`, `resolve_models`, `resolve_benchmark_model`). No value
transformation is required — the values (model family strings) are identical.

### Constants in macroforecast.layers.l4_models.ops

```python
# Before (deprecated)
from macroforecast.layers.l4_models.ops import OPERATIONAL_MODEL_FAMILIES
from macroforecast.layers.l4_models.ops import FUTURE_MODEL_FAMILIES

# After
from macroforecast.layers.l4_models.ops import OPERATIONAL_MODELS
from macroforecast.layers.l4_models.ops import FUTURE_MODELS
```

### L6 result dict key alias

```python
# Before (deprecated)
result["decision_at_5pct"]   # emits DeprecationWarning

# After
result["decision"]
```

The `decision_at_5pct` key is excluded from `keys()`, `iter()`, and `len()`
but remains accessible via `in` and `.get()` during the deprecation window.

---

## v0.1 / v0.8.x to v0.9.x: Axis Renames

The following recipe axis renames are **hard changes** — no alias helpers
exist at the YAML recipe level. Users receive an explicit `unknown axis`
error at recipe parse time when a stale key is present.

| Old name (v0.8.x / v0.1) | New name (v0.9.x+) | Layer | Context |
|--------------------------|-------------------|-------|---------|
| `custom_source_policy` | `panel_composition` | L1.A (`1_data.fixed_axes`) | Recipe YAML |
| `forecast_strategy` | `forecast_policy` | L4 node `params:` | Recipe YAML |
| `quarterly_to_monthly_rule` | `quarterly_to_monthly_policy` | L2.A (`2_preprocessing.fixed_axes`) | Recipe YAML |
| `monthly_to_quarterly_rule` | `monthly_to_quarterly_policy` | L2.A (`2_preprocessing.fixed_axes`) | Recipe YAML |
| `reproducibility_mode` | `reproducibility_policy` | L0 (`0_meta.fixed_axes`) | Recipe YAML |
| `fit_model` | `fit` | L4 node `id:` (user-chosen) | Recipe YAML node id (node ids are user-defined; `op: fit` was always the correct op name) |

These are intentional hard changes. Silent acceptance of deprecated axis names
in recipes could hide bugs and misroute configuration. An explicit
`unknown axis` error at recipe parse time is a sharper signal than a
deprecation warning that could be suppressed.

No value changes are required — only key names changed.

---

## Module Path Migrations

### Phase 3 restructuring (v0.9.x)

| Old import | New import | Status |
|-----------|-----------|--------|
| `from macroforecast.scaffold import ...` | `from tools.docgen import ...` | HARD CHANGE — `macroforecast.scaffold` contains only `option_docs/`; no Python CLI in that namespace |
| `python -m macroforecast.scaffold encyclopedia <dir>` | `python -m tools.docgen encyclopedia <dir>` | CLI command redirect |
| `macroforecast.defaults.DEFAULT_FORECAST_POLICY` | `macroforecast.api.defaults.DEFAULT_FORECAST_STRATEGY` | Constant renamed and relocated (see PR7) |

### Shim-backed paths (backward-compatible in v0.9.x)

| Old import | Canonical import | Status |
|-----------|-----------------|--------|
| `import macroforecast.interpretation` | `from macroforecast.layers.l7_interpretation import ...` | SHIM EXISTS in v0.9.x — works but emits deprecation; use canonical |
| `import macroforecast.recipes` | `from macroforecast.layers import ...` | SHIM EXISTS in v0.9.x |

### Always-current paths

| Module | Notes |
|--------|-------|
| `macroforecast.raw` | Unchanged — FRED-MD/QD/SD adapters, vintage manager |
| `macroforecast.tuning` | Unchanged — HP search engines; used internally by L4 |
| `macroforecast.core` | Unchanged — execution engine, dag, sweep, manifest |
| `macroforecast.api` | Unchanged — `run`, `replicate`, `Experiment`, `forecast()` |

---

## Removed Features

| Feature | Removed in | Replacement |
|---------|-----------|-------------|
| Interactive scaffold wizard (`macroforecast scaffold` interactive mode) | v0.9.5 Phase 0 | Use a recipe template from `examples/recipes/`; see [Recipe authors guide](../for_recipe_authors/index.md) |
| Navigator pages (`docs/reference/api/navigator*`) | v0.6.2 docs reorg | See `docs/reference/` index |
| Audience-tree doc paths (`getting_started/`, `user_guide/`, `fred_dataset/`, `simple/`, `detail/`, `dev/`) | v0.6.2 docs reorg | See `docs/tutorial/`, `docs/how_to/`, `docs/reference/` |
| `scripts/v01_smoke_check.py` | v0.9.5 (deep-audit) | Run `python -m pytest tests/ -x -q` for smoke validation |
| `examples/recipes/l2_fred_sd_alignment.yaml` | v0.9.5 (deep-audit PR4) | Renamed to `l2_preprocessing_minimal.yaml` — functionally identical |

---

## CLI Entry Point Change

The `macroforecast` command now routes through `macroforecast.__main__` (the
package itself) rather than `tools.docgen.cli` directly. The subcommands and
their arguments are unchanged:

```bash
macroforecast run recipe.yaml -o out/
macroforecast replicate out/manifest.json
macroforecast validate recipe.yaml
macroforecast scaffold -o recipe.yaml
macroforecast encyclopedia docs/reference/encyclopedia/
```

The `python -m macroforecast` form also works and is equivalent.

If you have a stale installation that points `macroforecast` to
`macroforecast.scaffold.cli` (from a pre-v0.9 install), reinstall:

```bash
pip install -e .
```

This ensures `pyproject.toml` `[project.scripts]` resolves to
`macroforecast.__main__:main`, which forwards to `tools.docgen.cli:main`.

---

## Recipe YAML Migration Checklist

If you have recipes written against v0.1 or early v0.8.x, apply these
substitutions before running:

```
0_meta.fixed_axes.reproducibility_mode    → reproducibility_policy
1_data.fixed_axes.custom_source_policy    → panel_composition
2_preprocessing.fixed_axes.quarterly_to_monthly_rule → quarterly_to_monthly_policy
2_preprocessing.fixed_axes.monthly_to_quarterly_rule → monthly_to_quarterly_policy
4_forecasting_model: nodes: op: fit_model → op: fit
```

Run `macroforecast validate your_recipe.yaml` after migration to catch
remaining schema errors before execution. The validator reports every
unknown axis with the layer name and allowed values, making it easier to
spot residual stale keys.

---

## Deprecation Alias Removal Timeline

See [Deprecation Timeline](deprecation_timeline.md) for the full table.
Summary:

| Deprecated | Replacement | Removed in |
|-----------|-------------|-----------|
| `model_family=` | `model=` | v0.10.0 |
| `model_families=` | `models=` | v0.10.0 |
| `benchmark_family=` | `benchmark_model=` | v0.10.0 |
| `OPERATIONAL_MODEL_FAMILIES` | `OPERATIONAL_MODELS` | v0.10.0 |
| `FUTURE_MODEL_FAMILIES` | `FUTURE_MODELS` | v0.10.0 |
| `decision_at_5pct` result key | `decision` | v0.10.0 |

---

## Where to Get Help

- **Encyclopedia** (`docs/reference/encyclopedia/`) — per-option reference for
  every axis, op, and model family.
- **Recipe schema** (`docs/reference/recipe_schema/`) — full YAML schema
  with allowed keys and value constraints.
- **Architecture explanation** (`docs/explanation/architecture/`) — layer
  design rationale and layer boundary contract.
- **Troubleshooting guide** (`docs/how_to/troubleshooting.md`) — common
  error messages and resolutions.
- **CHANGELOG** (`CHANGELOG.md`) — per-PR change history with exact
  migration diffs.
