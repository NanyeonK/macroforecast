# Reproducibility Policy

`execute_recipe()` routes every randomness-consuming call site through a
single seed resolver, so two calls with the same recipe and reproducibility
spec produce byte-identical predictions and metrics. The resolver reads the
`reproducibility_spec` that `execute_recipe()` extracts from the provenance
payload and returns concrete integer seeds.

This page describes the four reproducibility modes, the public API, and the
range over which determinism is guaranteed.

## Modes

The active mode is selected by `reproducibility_spec["reproducibility_mode"]`.

| Mode | Behavior | Typical use |
|------|----------|-------------|
| `seeded_reproducible` | Returns `reproducibility_spec["seed"]` (default 42) everywhere. | Single-path runs, package-level defaults. |
| `strict_reproducible` | Seed derived from `SHA-256("{recipe_id}\|{variant_id or 'main'}\|{model_family}")`. Distinct variants get distinct seeds; identical variants stay bit-identical. | Horse-race sweeps where between-variant seed independence matters for statistical validity. |
| `best_effort` | Same as `seeded_reproducible`. Marks runs where BLAS/GPU non-determinism is explicitly tolerated. | GPU-backed training, parallel BLAS with non-deterministic reductions. |
| `exploratory` | Fresh non-deterministic seed per call. | Ad-hoc sanity checks where variability is desired. |

Unknown modes raise `ValueError`.

## Public API

```python
from macrocast.execution.seed_policy import resolve_seed

seed = resolve_seed(
    recipe_id="my_recipe",
    variant_id="variant-A",           # optional
    reproducibility_spec={"reproducibility_mode": "strict_reproducible"},
    model_family="randomforest",      # optional
)
```

`resolve_seed` is pure — same inputs, same output.

### Context-aware convenience wrapper

Execution-path code calls `current_seed(model_family=...)` instead of
threading `recipe_id` / `reproducibility_spec` through every executor.
`execute_recipe()` installs a `ReproducibilityContext` at entry; every
`current_seed` call inside the execution reads from that context.

```python
from macrocast.execution.seed_policy import current_seed

# Inside an executor that only has access to ``recipe``:
model = RandomForestRegressor(random_state=current_seed(model_family="randomforest"))
```

Outside `execute_recipe()` (no context installed), `current_seed` falls back
to the `seeded_reproducible` default with `seed=42`, matching pre-Phase-0
behavior.

## Guarantee scope

| Scope | Guarantee |
|-------|-----------|
| Two calls with identical recipe, identical preprocess, identical `reproducibility_spec` | Byte-identical `predictions.csv`, `metrics.json`, `comparison_summary.json`. |
| Two variants with `strict_reproducible` and distinct `variant_id` | Distinct seeds → statistically independent model draws. |
| Two variants with `strict_reproducible` and identical `variant_id` | Byte-identical artifacts. |
| Across machines with identical Python / sklearn / numpy versions | Reproducible under `strict_reproducible` and `seeded_reproducible`. Not guaranteed under `best_effort` (BLAS / GPU nondeterminism). |
| Under `exploratory` | No reproducibility guarantee. |

Determinism is gated on the model library honouring the supplied seed.
Deep-learning-specific determinism flags (`torch.backends.cudnn`,
`torch.use_deterministic_algorithms`) are wired in Phase 5a.

## Single-path vs sweep-variant

Single-path callers — the common case — pass no `variant_id`. The default
`reproducibility_mode` is `seeded_reproducible`, so the whole run uses seed
`42` and reproduces across repeated invocations.

Sweep variants will be introduced in Phase 1. The sweep executor will set
`reproducibility_mode: strict_reproducible` and pass a stable `variant_id`
per variant in the provenance payload. Distinct variants therefore obtain
distinct, hash-derived seeds automatically, without the sweep code touching
any model instantiation.

## Tests

Three regression tests enforce the contract:

- `tests/test_seed_policy.py` — per-mode semantics, context wiring, unknown-mode handling.
- `tests/test_deterministic_replay.py` — `execute_recipe()` round-trip produces byte-identical predictions; strict mode with distinct `variant_id` diverges; strict mode with identical `variant_id` reproduces.
- `tests/test_execution_cache.py` — `cache_root` parameter accepted; shared `cache_root` reuses raw data across runs; single `manifest.json` write per call.

## References

- Infra spec: [`plans/infra/seed_policy.md`](https://github.com/NanyeonK/macrocast/blob/main/plans/infra/seed_policy.md)
- Phase plan: [`plans/phases/phase_00_stability.md`](https://github.com/NanyeonK/macrocast/blob/main/plans/phases/phase_00_stability.md)
