# SPA/RC/StepM dependent-loss size investigation

Date: 2026-07-07
Lane: `mf-eval`

## Question

The existing strict-xfail MC test for `superior_predictive_ability_test` shows
roughly 1.5-2x nominal over-rejection under an equal-predictive-ability null
when model losses have nonzero AR(1) dependence. The requested timebox was to
check whether there is a clean wrapper-side fix, with block-length resolution as
the prime suspect.

## Findings

- The public SPA/RC/StepM wrappers share
  `tests.py::_arch_benchmark_multiple_comparison`.
- That helper validates the long loss panel, pivots to benchmark and candidate
  loss matrices, resolves `block_length`, and delegates the actual bootstrap
  statistic/p-value computation to `arch.bootstrap.SPA`,
  `arch.bootstrap.RealityCheck`, or `arch.bootstrap.StepM`.
- The existing MC note in `tests/mc/test_spa_size.py` already rules out the
  simple block-length hypothesis: the distortion is stable across
  `block_length` in `{1, 5, 10, 20, "auto"}`, p-value type, studentization, and
  sample length, while the iid-null companion is correctly sized.
- No clean local fix is apparent without replacing or materially wrapping the
  arch statistic/bootstrap algorithm itself. That is outside this lane's
  disclosure-focused scope and would need a separate validation design.

## Action taken

- Added a machine-readable `size_caveat` payload to SPA/RC/StepM results and
  per-record metadata.
- Added docstring and reference-doc disclosures recommending `uspa`/`aspa` or
  `model_confidence_set` under dependent losses.
- Added strict-xfail dependent-null MC pins for RC and StepM mirroring the
  existing SPA design.

## Status

No implementation fix in this lane; disclosure plus MC coverage is the intended
stop point.
