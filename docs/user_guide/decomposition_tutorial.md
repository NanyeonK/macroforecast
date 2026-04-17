# Decomposition tutorial (§4.5 identity)

macrocast's horse race does not stop at "which variant won"; Phase 7 layers
on top a reviewer-defensible attribution: **how much of the sweep's
forecast-error variance is explained by each component of the pipeline?**

This page walks a concrete synthetic example end-to-end. For the
mathematical definition see [Decomposition attribution](../math/decomposition_attribution.md).
For API reference see [API: decomposition](../api/decomposition.md).

## When to reach for it

- You ran a horse-race sweep and want a paper-ready figure showing the
  share of variance attributable to, say, `preprocessing` vs `nonlinearity`.
- You want to sanity-check a result: if the "model family" share is tiny,
  the conclusion "model X beats model Y" might be a noise artifact.
- You need a reproducible number for review — Phase 7 guarantees the same
  manifest produces the same `decomposition_result.parquet` byte-for-byte.

## Minimum viable example

```python
from macrocast.compiler.sweep_plan import compile_sweep_plan
from macrocast.execution.sweep_runner import execute_sweep
from macrocast import DecompositionPlan, run_decomposition

# 1. Run a horse race — Phase 1 territory.
recipe = {...}   # see sweep_recipes.md for a full example
plan = compile_sweep_plan(recipe)
sweep = execute_sweep(plan=plan, output_root="out/horse-race")

# 2. Decompose — Phase 7 entry point.
dplan = DecompositionPlan(
    study_manifest_path=sweep.manifest_path,
    components_to_decompose=("preprocessing", "nonlinearity"),
    primary_metric="msfe",
)
result = run_decomposition(dplan)

print(result.per_component_shares)
# e.g. {"preprocessing": 0.74, "nonlinearity": 0.09}
```

`decomposition_result.parquet` and `decomposition_report.json` both land in
the same directory as the sweep's `study_manifest.json` by default; pass
`output_dir=` to route them elsewhere.

## Component attribution rules

Phase 7 ships 8 components:

| component | v0.9 axes |
|---|---|
| `preprocessing` | `scaling_policy`, `dimensionality_reduction_policy`, `feature_selection_policy`, `target_transform_policy`, `x_transform_policy`, `tcode_policy` |
| `nonlinearity` | `model_family` |
| `feature_builder` | `feature_builder` |
| `benchmark` | `benchmark_family` |
| `importance` | `importance_method` |
| `regularization` | (reserved — no v0.9 axis) |
| `cv_scheme` | (reserved — no v0.9 axis) |
| `loss` | (reserved — no v0.9 axis) |

Any axis not tagged goes to the residual (axes with `component=None`). Shares
therefore do **not** sum to 1 — the residual includes noise and
unattributed axes.

## Custom component subsets

```python
# Only decompose preprocessing and nonlinearity; ignore everything else.
DecompositionPlan(..., components_to_decompose=("preprocessing", "nonlinearity"))
```

```python
# Full catalogue (default).
from macrocast.decomposition import COMPONENT_NAMES
DecompositionPlan(..., components_to_decompose=COMPONENT_NAMES)
```

## Determinism

Two consecutive `run_decomposition` calls over the same manifest produce
byte-identical `decomposition_result.parquet`. See
`tests/test_decomposition_stability.py`.

## Limitations (v0.9)

- Aggregate (per-variant × horizon) ANOVA only — per-`oos_date` attribution
  is a v1.1 enhancement. If your sweep has one horizon and few variants,
  the F-test power is limited.
- ANOVA one-way: interaction decomposition is not modelled (you'll see each
  axis attributed independently; two-axis interactions fall into the
  residual).
- Shapley attribution (`attribution_method="shapley"`) is out of scope for
  v1.0 per [ADR-002](../../plans/infra/adr/ADR-002-anova-before-shapley.md).

These land in Phase 10 / v1.1.
