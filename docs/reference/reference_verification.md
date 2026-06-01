# Reference Verification

[Back to reference](index.md)

`macroforecast` separates ordinary unit tests from reference-style verification
anchors.

| Test class | Location | Purpose |
| --- | --- | --- |
| Unit/API tests | `tests/<module>/` | Public callable behavior, validation, metadata, and regressions. |
| Reference anchors | `tests/reference/` | Paper/formula invariants, known synthetic designs, and cross-module preservation checks. |
| Future external reference checks | `tests/reference/` or a gated verification suite | Comparisons to paper authors' code, known-DGP simulations, or pinned external outputs. |

The reference suite is intentionally small by default. It should catch drift in
core formulas and contracts without becoming a slow simulation lab.

Run it with:

```bash
uv run pytest tests/reference -q
```

## Current Anchors

| Anchor | File | What it checks |
| --- | --- | --- |
| DM antisymmetry | `tests/reference/test_reference_verification.py` | `dm_test(loss_a, loss_b)` has the opposite statistic and same p-value as the reversed comparison. |
| Blocked reality check sign | `tests/reference/test_reference_verification.py` | A synthetic candidate with lower loss has positive `mean_diff` and rejects no-improvement against the benchmark. |
| Iterative MCS elimination | `tests/reference/test_reference_verification.py` | A clearly worse benchmark is removed and the best synthetic candidate remains in the MCS. |
| Reporting/output metadata | `tests/reference/test_reference_verification.py` | Report-table metadata survives output bundling and artifact writing. |

## Expansion Rules

Add a reference test when any of these are true:

| Trigger | Example |
| --- | --- |
| A callable implements a named paper formula. | MCS, MIDAS weights, supervised scaled PCA, MARX/MAF transforms. |
| A callable claims compatibility with original source code. | Macro Random Forest, supervised scaled PCA support code. |
| A result is sensitive to look-ahead leakage. | Runner preprocessing/feature fit policies. |
| Metadata must survive across modules. | Feature lineage through interpretation, output, and reporting. |

Do not put long Monte Carlo studies in the default suite. Use a gated command
or separate verification artifact for slow known-DGP studies.

## Status

| Area | Current status | Next useful check |
| --- | --- | --- |
| Forecast tests | Core anchors added. | Add pinned small-sample checks for MCS variants if an external reference output is accepted. |
| Feature formulas | Covered by module tests, not yet reference-tagged. | Add MARX/MAF and MIDAS weight reference anchors. |
| Models | Covered by callable tests. | Add author-code or known synthetic checks for supervised scaled PCA and Macro Random Forest when needed. |
| Runner leakage | Covered by runner tests. | Add explicit reference anchors for full-sample vs origin-local preprocessing/feature policies. |
| Reporting/output | Initial metadata anchor added. | Add table preset checks after reporting presets are designed. |
