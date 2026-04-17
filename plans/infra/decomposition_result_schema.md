# decomposition_result.parquet — schema v1.0

Emitted by `macrocast.decomposition.run_decomposition` alongside a
`decomposition_report.json`. Parquet is chosen over CSV because (a) per-axis
rows are narrow and downstream consumers (plotting, paper-bundle aggregation
in Phase 8) want typed columns, and (b) the [parquet] extra already ships
`pyarrow`.

## Columns (order is stable)

| column | type | nullable | meaning |
|---|---|:---:|---|
| `component` | string | no | Component bucket (e.g. `preprocessing`, `nonlinearity`) |
| `axis_name` | string | no | Registry axis name (e.g. `scaling_policy`) |
| `ss_between` | float64 | no | Sum of squares explained by this axis |
| `ss_total` | float64 | no | Total sum of squares across the whole sweep |
| `share` | float64 | no | `ss_between / ss_total`, 0 if `ss_total == 0` |
| `n_variants` | int64 | no | Distinct variant IDs contributing to this axis |
| `n_groups` | int64 | no | Distinct axis values used in the sweep |
| `significance_p` | float64 | yes | ANOVA F-test p-value; `NaN` when undefined (1-group / zero-within-SS) |

An empty sweep (no variants) or an empty `components_to_decompose` produces a
parquet with zero rows and the schema above.

## Per-component aggregation

`per_component_shares` in the JSON report is Σ(`ss_between` over axes in
component) / `ss_total`. Shares do **not** sum to 1 — axes tagged as `None`
(unassigned) and the within-group residual carry the rest.

## Determinism

Row order is stable under repeated runs on the same `study_manifest.json`
because:

- axis names are iterated in `sorted()` order
- variant rows within each ANOVA group are sorted by `variant_id`
- all arithmetic uses `numpy.float64`
- pyarrow's default encoding is deterministic for primitive columns

Two consecutive `run_decomposition` calls over the same manifest produce
byte-identical parquet files (covered by
`tests/test_decomposition_stability.py`).

## Schema evolution

Future enhancements that would bump the schema version:

- Per-`oos_date` attribution (v1.1) — new `horizon`, `oos_date` columns.
- Shapley / interaction attribution (v1.1) — new `attribution_method` column
  and wider row set covering interaction terms.
- Bootstrap CIs — new `share_ci_low`, `share_ci_high` columns.

Additions are column-append only; existing columns keep their types and
positions. A major bump (`2.0`) is reserved for a breaking restructure.
