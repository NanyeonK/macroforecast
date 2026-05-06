# `tuning_view`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``tuning_view`` on sub-layer ``L4_5_D_tuning_history`` (layer ``l4_5``).

## Sub-layer

**L4_5_D_tuning_history**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `cv_score_distribution`  --  operational

Distribution of CV scores at each iteration.

L4.5.D tuning view ``cv_score_distribution``.

This option configures the ``tuning_view`` axis on the ``L4_5_D_tuning_history`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_D_tuning_history/`` alongside the other selected views.

**When to use**

Detecting high-variance objective surfaces; wide distributions suggest the search has not converged.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`objective_trace`](#objective-trace), [`hyperparameter_path`](#hyperparameter-path), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `hyperparameter_path`  --  operational

Sequence of hyperparameter values explored.

L4.5.D tuning view ``hyperparameter_path``.

This option configures the ``tuning_view`` axis on the ``L4_5_D_tuning_history`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_D_tuning_history/`` alongside the other selected views.

**When to use**

Diagnosing search behaviour -- e.g. detecting Bayesian optimisation getting stuck on a local minimum.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`objective_trace`](#objective-trace), [`cv_score_distribution`](#cv-score-distribution), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Produce all tuning-history views together.

L4.5.D tuning view ``multi``.

This option configures the ``tuning_view`` axis on the ``L4_5_D_tuning_history`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_D_tuning_history/`` alongside the other selected views.

**When to use**

Comprehensive tuning audit. Activates the ``multi`` branch on L4.5.tuning_view; combine with related options on the same sub-layer for a comprehensive diagnostic.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`objective_trace`](#objective-trace), [`hyperparameter_path`](#hyperparameter-path), [`cv_score_distribution`](#cv-score-distribution)

_Last reviewed 2026-05-05 by macroforecast author._

### `objective_trace`  --  operational

Tuning-objective trace over iterations.

L4.5.D tuning view ``objective_trace``.

This option configures the ``tuning_view`` axis on the ``L4_5_D_tuning_history`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_D_tuning_history/`` alongside the other selected views.

**When to use**

Default convergence audit; monotone decrease confirms good search behaviour.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`hyperparameter_path`](#hyperparameter-path), [`cv_score_distribution`](#cv-score-distribution), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
