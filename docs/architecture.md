# Architecture

[Back to User Guide](guide/index.md)

This page records what the package's layering is *for*, so a change that breaks
it is a decision rather than an accident. It is enforced by
`tests/architecture/test_import_boundaries.py`, which reads import statements
and fails on any upward one.

## The layers

A lower layer never imports a higher one.

| layer | packages | may import |
|---|---|---|
| 0 | `data`, `window`, `metrics`, `filters`, `meta`, `tests`, `analysis` | nothing else in the package |
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

## No known exceptions

`KNOWN_EXCEPTIONS` is empty: no module imports a layer above its own. Two entries
sat there until 2026-08-09, and both were resolved by moving the function down
rather than by tolerating the import.

| the old upward import | resolved by |
|---|---|
| `data/vintage.py` reaching `pipeline.run._panel_fingerprint`, a layer-0 module calling a layer-3 **private** function | the fingerprint now lives in `data/identity.py`, since it is a property of the data rather than of a study |
| `pipeline/run.py` reaching `output.collect_provenance` | the probe now lives in `meta/provenance.py`, since reading git and the environment is not artifact writing |

Neither move changed behaviour: the digest is byte-identical, and
`output.collect_provenance` re-exports the same object, so the public name is
unchanged.

Keep the list empty. An entry added back is a layering violation the project has
decided to live with, which needs a reason recorded here rather than a tuple in a
test. A test also fails if a listed exception is *fixed* without being removed
from the list, so the list cannot drift into fiction.

## The guard does not see same-layer cycles

`[GAP]` The check rejects an *upward* import and permits imports within a layer,
so a cycle among same-level packages passes it. Layer 1 contains such a cycle:
`feature_engineering` -> `model_selection` -> `model_ensemble` -> `models` ->
`feature_engineering`. Every edge but one is module-level; the edge that closes
the loop, `feature_engineering/_sparse_ic.py` reaching `model_selection`, is
function-local, so the four packages still import independently and no test is
failing. It is recorded because "no upward imports" is a weaker property than an
acyclic package graph, and only the first one is enforced.

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

## Where a forecast task is resolved

A single task's identity — which target, under which policy, from which
features — is resolved once and passed down:

| step | file |
|---|---|
| declared | `pipeline/spec.py` (`TargetSpec`, horizons, arm policy overrides) |
| resolved | `forecasting/task.py` (`resolve_forecast_tasks` → `ResolvedForecastTask`) |
| resolved once per cell | `pipeline/run.py` (`_run_cells`) |
| shaped for the policy | `forecasting/policy_config.py` (`_feature_spec_for_policy`) |
| materialized | `feature_engineering/specs.py` |

`ResolvedForecastTask` is the single answer, and `_run_cells` resolves each
(target, arm) cell once and hands the same objects to every consumer:
execution, the result-store digest, the checkpoint cell manifests, and the
provenance echo. Those four used to derive the policy override and the feature
retarget independently, and two of them disagreed about what an unretargetable
feature spec means — a digest could describe a forecast that was never run.
`tests/pipeline/test_shared_cell_task.py` counts the retargets over a
production run so the coincidence that hid it cannot come back.

`_feature_spec_for_policy` still builds a fresh `FeatureSpec` per (horizon,
policy), but it bakes in the horizon and transform the resolved task already
fixed rather than deciding again which forecast this is.

## Evaluation does not read data

`pipeline/evaluate.py` is computation over a forecast table and nothing else:
the same table and the same spec produce the same tables whatever the network,
the FRED cache, and the filesystem are doing. One input used to break that. A
named subsample mask (`nber_recession`, `nber_expansion`) is a *name*, and
turning it into a boolean state series meant a `load_fred_series()` call from
inside the evaluator, so scoring one fixed table could disagree between runs, or
fail offline, for reasons that had nothing to do with the forecasts.

Named masks are resolved before evaluation now, once per evaluation operation:

| step | file |
|---|---|
| declared | `pipeline/spec.py` (`SubsampleWindow.mask`) |
| resolved | `pipeline/evaluation_inputs.py` (`resolve_evaluation_inputs`) |
| aligned and applied | `pipeline/evaluate.py` |

`run_pipeline` and `rescore` call the resolver on the master forecast frame and
pass what comes back to `evaluate(..., inputs=...)`. The evaluator aligns that
state series against the forecast target dates, enforces the same strict
overlap/coverage/NaN errors it always did, and publishes the provenance it was
handed. It never calls a loader, which is why `evaluate()` invoked directly with
a named mask and no resolved inputs raises and says how to resolve them rather
than quietly reaching for the network. Masks a user supplies as a Series or
mapping never needed data and are unchanged.

The split follows the question each half answers: which series a name means at
the frame's frequency is a data question, and whether that series covers the
forecast dates is a question about the frame. Resolving also de-duplicates —
both NBER indicators read one series per frequency (`USREC` monthly, `USRECQ`
quarterly) and differ only in polarity, so an evaluation using recession *and*
expansion now loads once rather than twice.

This is enforced by `tests/architecture/test_evaluation_purity.py`, from both
ends: the evaluator's source may import no loader and no I/O module, and a full
named-mask evaluation over already-resolved inputs must still run — twice,
identically — with the loader replaced by a function that raises.
