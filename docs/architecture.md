# Architecture

[Back to User Guide](guide/index.md)

This page records what the package's layering is *for*, so a change that breaks
it is a decision rather than an accident. It is enforced by
`tests/architecture/test_import_boundaries.py`, which reads import statements
and fails on any new upward one.

## The layers

A lower layer never imports a higher one.

| layer | packages | may import |
|---|---|---|
| 0 | `data`, `window`, `metrics`, `filters`, `meta`, `tests` | nothing else in the package |
| 1 | `preprocessing`, `feature_engineering`, `models`, `model_selection`, `model_ensemble`, `data_analysis` | layer 0 |
| 2 | `forecasting`, `interpretation`, `feature_analysis`, `feature_diagnostic` | layers 0-1 |
| 3 | `pipeline`, `forecast_analysis`, `forecast_diagnostic` | layers 0-2 |
| 4 | `evaluation`, `output` | layers 0-3 |
| 5 | `reporting` | layers 0-4 |

These levels were derived by reading what each package actually imports, not
assigned by taste. `metrics`, `filters`, `meta` and `tests` (the statistical
tests) import nothing from the package at all, which is why they sit at the
bottom despite being used everywhere.

## Why it matters

Two properties depend on it, and both are load-bearing:

**`forecasting.run()` is usable on its own.** Every replication under
`docs/replication` drives one model at a time through the runner without
constructing a pipeline. That only works while `forecasting` knows nothing about
`pipeline`.

**A model can be read without reading the orchestrator.** `models/` is where a
reader checks whether an estimator does what a paper says. If it reached upward,
answering that question would mean understanding cell scheduling too.

A single upward import ends both quietly — the package still imports, the tests
still pass, and the coupling is only discovered when someone tries to use the
lower layer alone.

## Two known exceptions

Both are function-local, so neither creates an import cycle, and both are listed
in `KNOWN_EXCEPTIONS` so the ratchet applies to everything else. A test also
fails if one is *fixed* without being removed from the list, so the list cannot
drift into fiction.

| where | what | status |
|---|---|---|
| `data/vintage.py` | imports `pipeline.run._panel_fingerprint` | a layer-0 module reaching into a layer-3 **private** function. The fingerprinting belongs lower; moving it is behaviour-preserving and left for its own change. |
| `pipeline/run.py` | imports `output.collect_provenance` | genuinely a question of whether `output` sits above `pipeline` or beside it. Recorded so the answer is decided rather than accreted. |

## The two-stage forecasting structure

```
pipeline.run_pipeline()      one study: targets x arms x horizons
        |
        v
forecasting.run()            one model, origin by origin
        |
        v
preprocessing -> features -> model selection -> fit -> predict
```

`pipeline` owns *which* cells exist and how their results are combined and
scored. `forecasting` owns *one* cell end to end. Forecast policies (`direct`,
`direct_average`, `path_average`, `recursive`) are strategies inside the second
stage, not branches in the first.

## Where a forecast task is resolved today

A single task's identity — target, horizon, policy, transform — is currently
assembled in more than one place:

| step | file |
|---|---|
| declared | `pipeline/spec.py` (`TargetSpec`, horizons, arm overrides) |
| resolved per cell | `pipeline/run.py` |
| re-resolved for the policy | `forecasting/policy_config.py` (`_feature_spec_for_policy`) |
| materialized | `feature_engineering/specs.py` |

`[GAP]` This is a known weakness rather than a defect with a reproduction: a
task's identity is re-derived rather than resolved once and passed down, which
is the kind of arrangement in which `TargetSpec.transform` and
`FeatureSpec.target_transform` can drift apart. Consolidating it behind a single
resolved task object is the next structural change; nothing here claims a
current disagreement between those surfaces.

## Evaluation reads data

`[GAP]` `pipeline/evaluate.py` calls `load_fred_series()` to resolve a named
subsample mask (for example `nber_recession`). Evaluation is otherwise pure
computation over a forecast table, so this makes the evaluation of one fixed
table depend on network and cache state. Resolving named masks earlier — at spec
time, alongside their provenance — would restore that purity.
