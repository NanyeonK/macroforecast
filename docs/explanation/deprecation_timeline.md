# Deprecation Timeline

This page lists all deprecated public API parameters and their scheduled
removal versions.

## v0.10.0 removals

The following parameter aliases were renamed in v0.9.x. They emit
`DeprecationWarning` in all v0.9.x releases and will raise `TypeError`
in v0.10.0.

### Experiment constructor and forecast() function

| Deprecated | Replacement | Notes |
|-----------|-------------|-------|
| `model_family=` | `model=` | All Experiment constructors and `mf.forecast()` |
| `model_families=` | `models=` | Multi-model Experiment constructors |
| `benchmark_family=` | `benchmark_model=` | Experiment benchmark parameter |

All three shims are centralized in `macroforecast.api._deprecations`
(`resolve_model`, `resolve_models`, `resolve_benchmark_model`).

### Constants in macroforecast.models.ops

| Deprecated | Replacement |
|-----------|-------------|
| `OPERATIONAL_MODEL_FAMILIES` | `OPERATIONAL_MODELS` |
| `FUTURE_MODEL_FAMILIES` | `FUTURE_MODELS` |

Access via the deprecated names emits `DeprecationWarning` through the
module-level `__getattr__` shim in `l4_models/ops.py`.

### L6 result dict key alias

| Deprecated key | Replacement key |
|---------------|----------------|
| `decision_at_5pct` | `decision` |

Accessing `decision_at_5pct` on a DM/CW result dict emits
`DeprecationWarning`. The key is excluded from `keys()`, `iter()`, and
`len()` but remains findable via `in` and `.get()` for backward
compatibility.

### Migration

Replace all `model_family=` arguments with `model=`. No value
transformation is required -- the values are identical.

**Before (deprecated):**

```python
mf.Experiment(..., model_family="ridge", ...)
mf.forecast(..., model_family="ridge", ...)
```

**After:**

```python
mf.Experiment(..., model="ridge", ...)
mf.forecast(..., model="ridge", ...)
```

## Axis renames -- hard changes (no alias support)

The following recipe axis renames from v0.9.x are HARD CHANGES. There are
no silent alias helpers for these at the YAML recipe level. Users of v0.8.x
recipes must update axis names manually.

| Old name (v0.8.x) | New name (v0.9.x+) | Layer |
|-------------------|-------------------|-------|
| `custom_source_policy` | `panel_composition` | L1.A |
| `forecast_strategy` | `forecast_policy` | L4 |
| `quarterly_to_monthly_rule` | `quarterly_to_monthly_policy` | preprocessing frequency alignment |
| `monthly_to_quarterly_rule` | `monthly_to_quarterly_policy` | preprocessing frequency alignment |

These are intentional hard changes. Silent acceptance of deprecated axis
names in recipes could hide bugs and misroute configuration. Users receive
an explicit `unknown axis` error at recipe parse time, which is a sharper
signal than a deprecation warning that could be suppressed.

To migrate, open your recipe YAML and replace old axis keys with new ones
in the appropriate layer block. No value changes are required -- only the
key names changed.
