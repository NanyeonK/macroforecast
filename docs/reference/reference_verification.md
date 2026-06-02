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
| Direct and path targets | `tests/reference/test_reference_verification.py` | Direct, average, and path target columns use the intended future-step formulas. |
| MARX moving-average loop | `tests/reference/test_reference_verification.py` | `feature_matrix(..., specification="MARX")` matches the author-style cumulative lag-average loop. |
| MAF variable-specific PCA | `tests/reference/test_reference_verification.py` | `maf_features()` equals PCA applied separately to each variable's own lag panel. |
| MIDAS weights | `tests/reference/test_reference_verification.py` | Almon, beta, and step MIDAS metadata weights match the pinned shape formulas. |
| Runner stage policies | `tests/reference/test_reference_verification.py` | Fit-window preprocessing/features stop at each origin fit window, while explicit full-panel policies fit once on the full sample. |

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
| Feature formulas | MARX, MAF, and target formulas are reference-tagged. | Add external source-code fixtures if accepted. |
| Models | Covered by callable tests. | Add author-code or known synthetic checks for supervised scaled PCA and Macro Random Forest when needed. |
| Runner leakage | Runner tests and reference anchors cover full-sample vs origin-local preprocessing/feature policies. | Add known-DGP leakage simulations only if needed. |
| Reporting/output | Metadata anchors and reporting preset tests are covered. | Add table-style fixtures after paper-specific style presets are accepted. |
