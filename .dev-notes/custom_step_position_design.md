# Design note — custom preprocess step position (#453)

**Status: proposal. Nothing implemented.**

#453 reports two things in one sentence. They have different costs and different
risks, and the note's first recommendation is to stop treating them as one feature.

## What is true today

`PreprocessSpec.fit` calls `reprocess(...)` — the whole built-in chain — and only then
applies custom steps:

```python
processed = reprocess(data, metadata=metadata, **fit_options)
custom_states = _fit_custom_preprocess_states(processed, custom_steps)
processed = _apply_custom_preprocess_steps(processed, custom_steps, custom_states)
```

So custom steps are pinned last, unconditionally. Both halves of #453 follow from
that one line: a custom outlier filter cannot run before imputation, and a column a
custom step creates appears after the transform stage has already finished, so no
tcode is ever applied to it.

## (1) Position — implementable, and the boundaries already exist

`reprocess` is not opaque inside. It runs named stages and records each one in a step
ledger, and those recorded names are the natural vocabulary:

```
frequency -> transform -> tcode_lag -> outliers -> impute -> standardize -> frame
```

(`preprocess.py`, `steps.append({"step": ...})` at lines 186, 206, 229, 243, 777, 797,
824, 844.)

So `position="before:impute"` is not inventing a concept — it names a boundary that
already exists and is already reported in the artifact users read. Proposed shape:

```python
mf.preprocessing.custom_preprocess_step(
    "winsorize_by_hand", func=..., position="before:impute",
)
```

with `position="last"` the default, so every spec written to date keeps its exact
current meaning.

Two constraints the implementation inherits and must not quietly drop:

- **#502's fit-window contract.** A step declared as aggregating is rejected under
  `policy="fit_window"`. Position does not change that, and an earlier position makes
  it *more* load-bearing: a step inserted before `impute` sees a panel with holes, and
  a step that aggregates across rows there is computing a statistic on a sample whose
  composition changes with the origin.
- **The ledger must record the position.** A custom step at `before:impute` and the
  same step at `last` are different preprocessing pipelines. If the recorded steps do
  not distinguish them, two runs that differ become indistinguishable in their own
  provenance — and the on-disk preprocessing cache keys off the spec, so they would
  also collide.

## (2) tcodes on custom-created series — not plumbing, and should be split out

The second half asks for "explicit opt-in for tcode application to custom-created
columns." This is a different kind of change and the note recommends filing it
separately.

A tcode is not a formatting flag; it is an assertion about a *raw* series — that
`INDPRO` is I(1) in logs and therefore enters as a log difference. A series that a
custom step creates does not have a raw form that anyone published a tcode for. There
are three cases, and they do not want the same answer:

1. **A rescaling of one existing column** (a hand winsorized `INDPRO`). It inherits the
   source column's tcode, and that inheritance is defensible because the series is
   still that series.
2. **A combination of several** (a spread, a ratio). Its transform is a *modelling
   choice*, and taking any parent's tcode would be arbitrary — a spread of two I(1)
   series need not be I(1).
3. **Something new** (an external indicator injected by the step). No parent, no
   inheritance; the user must state the tcode.

Only case 1 has a safe default. So the opt-in should be explicit and per-column —
`tcode=` on the step, or a mapping — never a blanket "apply tcodes to custom output"
switch, which would silently pick the wrong answer for cases 2 and 3.

There is also an ordering consequence: applying a tcode to a custom column requires
that column to exist *before* the transform stage, so this half is only meaningful
together with `position="before:transform"`. That is another reason to land position
first and on its own.

## Recommended split

- **#453a — `position`.** Named boundaries, default `"last"`, recorded in the step
  ledger and in the cache key, with #502's aggregation rejection still enforced.
  Self-contained.
- **#453b — tcodes for custom-created series.** Depends on 453a. Needs a decision on
  the three cases above before it has an API.

## Test that would pin 453a

A custom step that clips values, at `position="before:impute"` and at `"last"`, on a
panel with holes. The two must produce **different** panels — clipping before
imputation changes what the imputer sees, so a run where position was accepted and
ignored is caught rather than passing. Plus: the step ledger reports the position; two
specs differing only in position get different cache keys; and an aggregating step at
an early position under `fit_window` is still rejected.
