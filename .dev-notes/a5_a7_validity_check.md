# Validity check — architecture review items A5, A6, A7

**Status: measured. Recommendation is to close A5–A7 without implementing them.**

A0–A4 of the architecture review found real defects and have all landed
(#523, #524, #525, #526, #527, #528, #529, #530, #531). A5–A7 were checked the same
way A4's `CompiledFeaturePlan` was checked — by measurement rather than by reading
the proposal — and none of them survives.

## A5 — `CellKey` / `CellBatch` / `CellResult`

### `CellResult` (replace `DataFrame.attrs`) — premise refuted

The review's case is that results ride on `DataFrame.attrs` (22 sites across
`pipeline/` and `forecasting/`: failed cells, empty cells, vintage boundary audits,
vintage sources) and that `attrs` is fragile.

Measured on both the declared floor and the current release:

| operation | pandas 2.2.2 (floor) | pandas 3.0.2 |
|---|---|---|
| `copy()` | KEPT | KEPT |
| boolean filter | KEPT | KEPT |
| `sort_values` | KEPT | KEPT |
| `reset_index` | KEPT | KEPT |
| `groupby().sum()` | KEPT | KEPT |
| `concat` | KEPT | KEPT |
| `merge` | KEPT | KEPT |

`attrs` survives every operation a caller is likely to perform, on every pandas this
package supports. There is no observed loss to fix. A `CellResult` type would be a
new public surface justified by a failure mode that does not occur here.

### `CellKey` vs `CellBatch` — real conflation, already pinned

`_Cell` is `(target_idx, arm_idx, horizons)` and its shape does depend on the
backend: serial groups every horizon into one cell, parallel splits one horizon per
cell. So the scientific unit and the execution unit ARE the same object, which is
what the review objects to.

But the consequence the split would prevent is already prevented by tests:

- `tests/pipeline/test_auto_parallelism.py::test_auto_forecasts_identical_to_serial_and_parallel`
- `tests/pipeline/test_arm_tags.py::test_arm_tags_propagate_to_master_serial_and_parallel`

and `_enumerate_cells` documents that with `result_store` enabled BOTH backends split
by horizon, so the persisted unit is identical either way. Separating the types would
be a naming improvement over a behaviour that is already fixed by contract. That is
not nothing, but it is not a defect, and it touches the hottest file in the repo.

## A6 — split `pipeline/run.py`, `forecasting/runner.py`, `pipeline/spec.py`

The files are large (2093 / 2867 / 1609 lines). The review offers size as the
argument; no defect is attached.

Against it, measured: of the branches not yet merged into `main`,

| file | unmerged branches touching it |
|---|---|
| `macroforecast/pipeline/run.py` | 8 |
| `macroforecast/forecasting/runner.py` | 9 |
| `macroforecast/pipeline/spec.py` | 9 |

A physical split would put every one of those into conflict, and would do it for a
change that fixes nothing observable. If a split is wanted later, the moment to do it
is when the in-flight work has drained — not while it is at its peak.

## A7 — evaluation kernel, and named subsamples at spec resolution

### Named subsamples — already there

The review asks for named subsample resolution to move from execution into spec
resolution. It is already in spec resolution: `_parse_subsample_date`,
`_coerce_subsample_mask_value`, `_canonicalize_subsample_mask` and
`_normalize_subsamples` all live in `macroforecast/pipeline/spec.py` (lines 379–472).

### Evaluation kernel — nothing entangled to separate

`macroforecast/evaluation/` is `__init__.py` + `report.py` (625 lines). `report.py` is
already function-first: one `EvaluationReport`, one `evaluate_report` entry point, five
public composable functions (`filter_oos_period`, `error_decomposition`,
`aggregate_scores`, `benchmark_comparison`, `regime_scores`) and private helpers
beneath them. The metric kernel is a separate top-level module already. There is no
kernel-inside-reporting to extract.

## Why A5–A7 read as plausible

Two of the three were true when the review was written and stopped being true as
A0–A4 landed:

- A7's subsample half describes code that spec resolution already owns.
- A5's `CellResult` half assumes `attrs` loss that this pandas does not exhibit.

The remaining items (A5's type split, A6's file split) are restructurings whose
benefit is legibility. Both are defensible one day; neither is defensible now, at the
cost of conflicting with 8–9 in-flight branches each, with no failure they prevent.

## What was checked and rejected as a finding

While examining A5 it looked as though a STATEFUL custom preprocessing step never
reached the result identity: `_custom_preprocessing_digests` digests only
`step[func]`, so a step defined with `fit_func`/`transform_func` returns `[]`.

End-to-end that is not true, and it was measured before it was filed. Two specs
differing only in a stateful step's quantile produce different cell identities, and
the difference is digest-based rather than address-based:

```
-  mf_digest: fit-0.01    +  mf_digest: fit-0.05
-  mf_digest: tr-0.01     +  mf_digest: tr-0.05
```

So the identity is correct and stable across processes. No issue filed.
