# Examples

End-to-end runnable examples demonstrating macrocast workflows.

:::{admonition} Work in progress
:class: note
The example gallery is being written. The table below lists the intended
coverage; individual pages will land in subsequent releases. In the meantime,
see [Getting Started](../getting_started/index.md) for a runnable tutorial and
`examples/recipes/` in the repository for live recipe YAML files.
:::

| Example | Difficulty | Concepts | Time |
|---------|-----------|----------|------|
| Basic Benchmark | Beginner | AR vs benchmark, MSFE, OOS R2 | 5 min |
| Preprocessing Ablation | Intermediate | raw vs impute+scale, governance | 10 min |
| Feature Builder Comparison | Intermediate | autoreg vs raw_panel, sweep | 10 min |
| Custom Benchmark | Intermediate | Plugin benchmark callable | 10 min |
| Statistical Test Gallery | Advanced | DM, CW, MCS, diagnostics | 15 min |
| Importance Gallery | Advanced | SHAP, PDP, permutation | 15 min |
| Tuning Study | Advanced | Grid, Bayesian optimization | 15 min |
| Multi-Target | Advanced | INDPRO + UNRATE + CPI | 10 min |
| Real-Time Vintage | Advanced | Explicit vintage, real-time | 10 min |

All examples will use test fixtures from `tests/fixtures/` for reproducibility.

**See also:** [Getting Started](../getting_started/index.md) | [User Guide](../user_guide/index.md)

## Phase 4 patterns

- [CLSS replication pattern](clss_replication_pattern.md) - relative-RMSE horse race using FRED-MD INDPRO
