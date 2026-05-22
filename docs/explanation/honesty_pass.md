# The Honesty Pass

A forecasting package names procedures. If a package says it implements
Diebold-Mariano with HLN correction, a researcher who cites that package in
a paper is implicitly claiming that the published HLN formula was executed.
macroforecast takes this claim literally.

This page explains the project's two-value status vocabulary, why intermediate
values were abandoned, and the sequence of honesty passes that brought every
schema item to one of the two canonical values.

---

## The Two-Value Vocabulary

The `macroforecast.core.status` module defines exactly two values:

**`operational`** means the runtime executes the full design-specification
procedure. The output matches the published method named in the design
document. A researcher who cites a result produced by an operational item
is citing the named published procedure.

**`future`** means schema-only. The validator raises a hard error at recipe
validation time if any selected option has `future` status. The runtime raises
`NotImplementedError`. The item is tracked in the GitHub issue tracker.

These are not stages on a development timeline. They are epistemic states:
`operational` means "we can stand behind this," and `future` means "we
cannot yet."

Why only two values? The grey band between these two — "runs, but is an
approximation of the named procedure" — is itself a form of false advertising.
A researcher using an item in that grey band receives numbers that look like
the named procedure's output but are not. If they cite the package as
implementing the named procedure, they are claiming something false. The
design's position is that it is more honest to reject a recipe with a clear
error than to silently return numbers that are approximately right.

---

## Why Intermediate Values Were Abandoned

In the v0.1 era, the schema used several intermediate values: `planned`,
`approximation`, `simplified`, `registry_only`, and `contract_defined_gated`.
Each was intended to carry different information about the relationship between
the named procedure and the current implementation.

The problem shared by all of them was that a user could not determine from
the status alone whether the numbers they received were from the named
published procedure or from a placeholder. `simplified` might mean "we ran
OLS instead of the full Kalman smoother." `approximation` might mean "we used
a first-order Taylor expansion instead of the exact formula." In each case,
a citation of the package as implementing the named method would be misleading.

The `normalize_status` function in `macroforecast.core.status` retains all
legacy values as deprecated aliases and collapses them to `future`. This
ensures that old recipes and serialized manifests from the v0.1 era continue
to parse correctly. New code writes only `operational` or `future`.

---

## The No-Proxy Policy

A proxy is an implementation that produces numbers under the name of a
published procedure but executes a different algorithm. The distinction from
an approximation is not sharp in general, but the operative test is simple:
if a researcher reads the procedure name in the docs and would cite the
corresponding published paper, the runtime must execute that paper's procedure.

In v0.1, the `bvar_minnesota` item carried `operational` status but its
runtime returned OLS estimates with no Minnesota prior. A researcher citing
the result as "Minnesota-prior BVAR forecasts" would be citing OLS results.
This was a proxy, and it was the reason for the first honesty pass.

The policy today: any item whose runtime does not execute the procedure named
in the design document is labeled `future` and rejected at recipe validation
time. The error message includes a link to the tracking issue so users can
follow the implementation progress.

---

## Version History: The Four Passes

### v0.1 — First Design Pass

The package was released with the full 12-layer schema. The codex review
on PR #163 flagged 19 families and ops whose v0.1 runtime did not match the
published procedure named in the design. They were demoted from `operational`
to `future`. The affected items spanned three areas: regime estimation in L1.G
(three estimators), forecasting model families in L4 (five families), and
interpretation ops in L7 (eleven ops).

This pass established the honesty vocabulary by naming the problem: items
that carry `operational` status imply a specific published procedure, and
the runtime must deliver that procedure.

### v0.2 — First Honesty Pass

All 19 demotions from v0.1 were re-promoted with real implementations. Issues
#184 through #198 each closed one item. `FUTURE_MODEL_FAMILIES` dropped from
19 to 0 for the first time. This was the first state in which the package
could honestly say that every `operational` item delivered its named procedure.

The v0.2 release also added 18 new capabilities (L0 seed propagation, L8
manifest provenance, additional L6 tests, sub-cell parallelism) that were
new additions rather than honesty-pass promotions. Those additions are in the
changelog under "v0.2 design-coverage additions."

### v0.25 — Second Honesty Pass

A subsequent audit found 19 items that carried `operational` status but had
been implemented as minimum-viable proxies relative to the full published
procedure. The classification changed: these were not rejected at validation
time (so they were technically `operational`), but their runtime did not
execute the procedure described in the source paper.

Examples of the gap: the Phillips-Perron test used the `arch` package's
implementation rather than the native OLS + Newey-West HAC formula specified
in Phillips and Perron (1988). The Tong SETAR estimator used a simple
quantile split rather than the full grid-search over the joint-SSR objective.

v0.25 promoted all 19 items to their full published procedures. The criterion
was strict: the implementation must execute the formula as stated in the paper
cited in the design document, not a numerically close variant.

### v0.3 — Third Honesty Pass Plus New Features

A third audit found 15 items plus added new model families. The most notable
promotions: the MacKinnon (2010) p-value for the Phillips-Perron test
(finite-sample table interpolation replacing the asymptotic approximation),
the Engle-Manganelli DQ test for density forecast evaluation, and the
Diebold-Mariano-Pesaran joint multi-horizon test. After v0.3, the schema
was audited clean.

### v0.9.3 (C49 and C50) — Final Algorithmic Cycle

Two schema items remained non-operational through v0.3: `realized_garch`
in L4 and `generalized_irf` via `lstm_hidden_state` in L7. These were
promoted in C49 and C50 respectively. After the C50 merge:

```
FUTURE_MODEL_FAMILIES = ()
FUTURE_OPS = ()
```

The schema has no remaining `future` items. Every item in the design document
corresponds to an `operational` runtime implementation.

---

## Operational Coverage Today

Following C50, the operational inventory is:

- 35+ L4 forecasting families: linear models, tree ensembles, boosting,
  SVM, kNN, MLP, deep neural networks (LSTM/GRU/Transformer with
  `[deep]` extra), AR(p), BVAR Minnesota and Normal-Inverse-Wishart,
  FAVAR, Macroeconomic Random Forest (GTVP), Mariano-Murasawa DFM,
  Quantile Regression Forest, and bagging meta-estimator.
- Approximately 30 L7 importance ops: SHAP family, partial dependence,
  ALE, Friedman H-interaction, lasso inclusion frequency, cumulative R2,
  bootstrap/jackknife, rolling recompute, forecast decomposition, FEVD,
  historical decomposition, generalized IRF, group aggregate, lineage
  attribution, transformation attribution, MRF GTVP importance.
- 6 L1.G regime estimators: none, NBER, user-supplied, Hamilton
  Markov-switching, Tong SETAR, and Bai-Perron structural breaks.

See the [Architecture Index](../reference/architecture/index.md) for
per-layer operational coverage detail.

---

## The Validator Enforcement

At recipe validation time, the validator checks every selected option against
the layer's `AxisSpec`. If any selected value has status `future`, the
validator raises a hard error before any computation begins. No output
directory is written. No partial artifacts are produced.

The user-facing error message names the item, gives its current status, and
includes a link to the tracking issue. This means a researcher cannot
accidentally run a future item and receive proxy numbers. The failure is
visible at the entry point, not buried in a log file.

---

## Further Reading

- [Foundation Core](../reference/architecture/foundation.md) — where the core
  vocabulary (`OPERATIONAL`, `FUTURE`, `ItemStatus`, `normalize_status`,
  `is_runnable`) is defined.
- `macroforecast.core.status` — the Python module that is the single source
  of truth for the two-value vocabulary and all legacy alias mappings.
