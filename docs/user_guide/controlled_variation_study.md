# controlled_variation_study

`controlled_variation_study` is the study mode that macrocast's horse
race sweep runner was designed for. It holds every design decision
constant except the axes you explicitly sweep, so any difference in
per-variant forecast performance is attributable to those axes.

## When to reach for it

- Comparing candidate models with everything else (framework, features,
  preprocessing, metric) pinned down.
- Measuring the sensitivity of one decision (e.g. `scaling_policy`)
  while the rest of the pipeline stays locked.
- Producing a paper-ready bundle where the reported comparison is
  defensibly fair.

For open-ended hyperparameter search, use the tuning engine
(`grid_search`, `random_search`, `bayesian_optimization`,
`genetic_algorithm`). The sweep runner is a *comparison* engine, not a
search engine.

## Declaring the study mode

```yaml
0_meta:
  fixed_axes:
    study_mode: controlled_variation_study
```

This is now operational in v0.3 (promoted from `registry_only` in
Phase 1).

## Runtime contract

When `execute_sweep` walks a plan:

1. One shared FRED cache at `<output_root>/.raw_cache_shared/` — every
   variant sees the same raw snapshot.
2. Per-variant seed via `reproducibility_mode=strict_reproducible` —
   same axis values produce the same seed, deterministically.
3. Per-variant manifest at `<output_root>/variants/<vid>/manifest.json`
   with the full existing single-path provenance.
4. One study manifest at `<output_root>/study_manifest.json` (Schema
   v1) aggregating every variant's status, axis values, metrics, and
   runtime.

The per-variant output layout is identical to a single-path run, so
downstream tooling (importance, decomposition, bundles) keeps working
unchanged.

## Failure handling

`execute_sweep(..., fail_fast=False)` (default) records failed variants
in the study manifest with their exception text and keeps going. Use
`fail_fast=True` when you want the first failure to abort the sweep —
useful during development.

## Replication

A sweep's `study_id` is a canonical hash of the sweep plan. Two plans
produced from the same recipe (even on different machines) always share
the same `study_id`, and their variants share the same `variant_id`s in
the same order. This is the foundation for the Phase 6 replication
runner.

## Phase 6 specialisations

Two wrapper patterns on top of `controlled_variation_study` landed in
v0.8:

- `execute_ablation` — baseline + N drop-one variants (one per
  component) in a single sweep, with a per-component delta report at
  `<output_root>/ablation_report.json`. See
  [ablation cookbook](ablation_cookbook.md).
- `execute_replication` — frozen recipe + overrides + diff report,
  including a byte-identical round-trip under `seeded_reproducible`.
  See [replication cookbook](replication_cookbook.md) and the
  [synthetic round-trip example](../examples/synthetic_replication_roundtrip.md).

Both are thin layers over the Phase 1 sweep runner and respect the same
`study_id` / `variant_id` stability guarantees; ablation uses explicit
`v-baseline` for the reference variant and hash-based IDs for the
drop-one variants.
