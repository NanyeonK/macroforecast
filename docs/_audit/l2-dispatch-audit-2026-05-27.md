# L2 Full-Sample Dispatch Audit

**Date**: 2026-05-27
**PR**: PR7 (Chan Review remediation batch)
**Auditor**: StatsClaw builder agent

---

## Scope

This audit traces every `imputation_temporal_rule` option in the L2 schema
through the runtime dispatch logic to identify which combinations invoke
full-sample cleaning functions (`_apply_outlier_policy`,
`_apply_imputation`) versus deferred per-origin cleaning.

---

## Dispatch Matrix

| `imputation_temporal_rule` | `_apply_outlier_policy` called at L2 stage? | `_apply_imputation` called at L2 stage? | Per-origin closure in L3? | Leak risk |
|---|---|---|---|---|
| `expanding_window_per_origin` (default) | No — deferred | No — deferred | Yes — `_apply_outlier_policy_per_origin` + `_apply_imputation_per_origin` in `materialize_l3_minimal` line ~1533 | None — per-origin is leak-free |
| `rolling_window_per_origin` | YES — full-sample | YES — full-sample | No | LEAK for any stateful policy (runtime has no per-origin rolling-window implementation) |
| `block_recompute` | YES — full-sample | YES — full-sample | No | Partial — block may span post-origin observations for stateful policies |
| `full_sample_once` | Validator-rejected (hard error) | Validator-rejected (hard error) | N/A | N/A — blocked at schema level |

---

## Per-Policy Leak Analysis (full-sample branch)

When `rolling_window_per_origin` or `block_recompute` is active, the
full-sample functions are called. The leak depends on the cleaning policy:

### Imputation

| `imputation_policy` | Leak when full-sample? | Reason |
|---|---|---|
| `none_propagate` | None | No imputation performed |
| `forward_fill` | None | `ffill()` only looks backward in time |
| `mean` | Partial | Mean includes post-origin rows |
| `linear_interpolation` | Critical | Bidirectional by pandas default; PR6 fix limited to `limit_direction=forward` but full panel is still used |
| `em_factor` / `em_multivariate` | Partial | PCA-EM factors fitted on full panel |

### Outlier detection

| `outlier_policy` | Leak when full-sample? | Reason |
|---|---|---|
| `none` | None | No outlier detection |
| `mccracken_ng_iqr` | Yes | Median/IQR thresholds from full panel |
| `zscore_threshold` | Yes | Mean/std thresholds from full panel |
| `winsorize` | Yes | Clip bounds from full panel quantiles |

---

## Identified Gaps and Fixes Applied (PR7)

### Gap 1/2: `rolling_window_per_origin` + `linear_interpolation` — CRITICAL

**Problem**: The option name `rolling_window_per_origin` implies per-origin
safety but the runtime has no per-origin rolling-window implementation.
`rolling_window_per_origin` falls into the full-sample else-branch
(runtime.py line 420-422), calling `_apply_imputation` with
`linear_interpolation`. PR6 fixed the bidirectional default to
`limit_direction=forward`, but the full panel is still used as the data
source, which is a lookahead leak. The option name creates a false sense of
causal safety.

**Fix applied**: Hard validator rejection in `_validate_imputation`
(schema.py). Any recipe with `imputation_temporal_rule: rolling_window_per_origin`
and `imputation_policy: linear_interpolation` now raises a `ValueError` at
recipe parse time with a message directing users to the safe alternatives
(`forward_fill` or `expanding_window_per_origin` with `linear_interpolation`).

**Implementation note**: Full per-origin rolling-window dispatch is deferred
to v0.4. The schema option is retained to avoid breaking recipe files; the
runtime behaviour for rolling_window_per_origin (other than the linear_interpolation
hard reject) is equivalent to block_recompute (full-sample with documented warning
for stateful policies).

### Gap 3: `block_recompute` + stateful policies — SOFT warning

**Problem**: `block_recompute` is a legitimate full-sample-at-block-boundary
approach. The name does not imply per-origin safety. However, stateful
outlier/imputation policies used with `block_recompute` may use statistics
computed over observations that post-date any given forecast origin.

**Fix applied**: SOFT warning (not hard error) emitted by `validate_layer`
when the user explicitly sets a stateful policy alongside `block_recompute`.
Only explicitly-set policies trigger the warning; package-default values do
not, to avoid spurious warnings for causal-safe combos like
`forward_fill + block_recompute`.

The warning message directs users to `expanding_window_per_origin` for
strictly causal evaluation.

---

## Files Modified

| File | Change |
|---|---|
| `macroforecast/layers/l2_preprocessing/schema.py` | `_validate_imputation`: added hard rejection for `rolling_window_per_origin + linear_interpolation`. `validate_layer`: added SOFT warning for `block_recompute + stateful policies`. |
| `tests/layers/test_l2_temporal_dispatch_audit.py` | New file — 20 TDD tests covering all dispatch matrix combinations. |

---

## Remaining Items (deferred to v0.4)

- Full per-origin rolling-window dispatch: implement `_apply_outlier_policy_per_origin`
  and `_apply_imputation_per_origin` with a rolling window size parameter, wired
  into the `materialize_l3_minimal` per-origin closure when
  `imputation_temporal_rule == "rolling_window_per_origin"`.
- `block_recompute` runtime audit: verify the block-boundary logic in
  `materialize_l2` correctly restricts cleaning to each block's date range.
