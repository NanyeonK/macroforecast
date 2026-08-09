# Design note — `retrain_every > 1` under vintage-aware runs (#450)

**Status: proposal. Nothing implemented.** #450 asks for a design note before
implementation, because the question is not "remove the guard" but "decide what a
frozen fit *means* when every origin carries its own panel."

Current behaviour, `macroforecast/forecasting/runner.py:1850`:

```python
if window_spec.estimation.retrain_every != 1:
    raise ValueError("vintage-aware runs require retrain_every=1 in Phase 1")
```

## What `retrain_every` does today (non-vintage)

`window/core.py` builds the origin table. At each origin it computes an estimation
window `[estimation_start_pos, estimation_end_pos]`, and then:

```python
if retrain:
    retrain_group += 1
    fit_start_pos = estimation_start_pos
    fit_end_pos   = estimation_end_pos
```

Non-retrain origins inherit the previous `fit_start_pos`/`fit_end_pos`. So
`retrain_every` freezes **row positions**, and `window/core.py:844` warns exactly
that: *"fit window lags estimation window when retrain_every > 1"*.

With a single static panel, frozen positions and a frozen fit sample are the same
thing, because those positions hold the same numbers forever. That identity is what
breaks under vintages.

## The fork

At a non-retrain origin *t* whose last retrain was *r*, the fit window is positions
`[s, e]` fixed at *r*. Two readings, and they are not close:

**(A) Freeze the vintage too.** Fill `[s, e]` from the panel as it stood at *r*. The
model a forecaster trained in March is the model they had in June — revisions
included, because they had not looked again.

**(B) Freeze only the positions.** Fill `[s, e]` from the panel as it stands at *t*.
Same rows, revised values.

**(B) is not a leak** — revisions known at *t* are legitimately available at *t* — but
it is a refit: same window, different numbers, different coefficients. That is
precisely the thing `retrain_every` exists to *not* do, so (B) delivers a cadence
parameter that silently ignores its own cadence.

**Recommendation: (A).** It is the only reading under which "retrain quarterly, forecast
monthly" describes what the run actually did, and it is what the convention #450 cites
(quarterly/annual retraining in long real-time studies) means operationally.

## What (A) requires

1. **The fit sample must be resolved against `vintage(r)`, not `vintage(t)`.** The
   origin→vintage map already exists (`metadata["vintage_source"]["origin_vintage_map"]`);
   the fit path needs the retrain origin's entry, not the current origin's. The window
   table already carries `retrain_group`, which identifies *r* for every *t*.

2. **Features at the origin still use `vintage(t)`.** Only the *fit sample* is frozen.
   A forecaster in June predicts from June's data using March's coefficients. Mixing
   these up in either direction is the failure mode this note exists to prevent.

3. **The boundary audit must record both.** `vintage_boundary_audit` currently
   describes one vintage per origin. Under (A) an origin has two: the fit vintage and
   the origin vintage. Reporting only one would make the audit wrong rather than
   incomplete — and for a vintage-aware run the audit is the artifact people trust.

4. **No new fit-caching work.** `_fit_reuse_key` (#505) keys on the *content* of the
   fit sample, not on window bounds. Under (A) every origin in a retrain group
   resolves the same rows from the same vintage, so the content hash matches and the
   group fits once. Under (B) the content differs at every origin and the cache would
   correctly miss every time — another way of saying (B) saves nothing, which is the
   original complaint in #452(2).

## Test that would pin it

Two vintages of one panel differing only in a revision to a row inside the fit
window, `retrain_every=3`, and three origins in one retrain group:

- under (A) the three origins produce **identical coefficients**, and they equal the
  coefficients from a single fit on `vintage(r)`;
- the forecasts still differ across the three origins, because the features advance;
- an origin in the *next* retrain group picks up the revision;
- the audit reports `fit_vintage=vintage(r)` and `origin_vintage=vintage(t)` separately.

The first and third assertions are what distinguish (A) from (B); the second stops the
test passing on a run that froze the origin panel as well.

## Scope note

This is Phase 1's guard being lifted deliberately, not an oversight being fixed. The
guard should not be removed until (3) is implemented, because a vintage run whose
audit under-reports its own inputs is worse than one that refuses to start.
