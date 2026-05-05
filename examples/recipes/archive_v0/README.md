# Pre-v0.0.0 archive

These recipes use the **old 8-layer schema** that was deprecated when
the 12-layer canonical design landed at the v0.0.0 reset
(`recipe_id` + `path:` wrapper, with layer keys like `1_data_task`,
`3_training`, `4_evaluation`, `5_output_provenance`).

The runtime no longer parses this format -- attempting
`macrocast.run("examples/recipes/archive_v0/...")` will fail with a
schema-validation error. They are kept here only as a **historical
reference** for users porting old recipes to the current schema; do
not start new work from these files.

## Migration notes

| Old key | New canonical key |
|---|---|
| `path: { 1_data_task: { ... } }` (top-level wrapper) | `1_data:` (no wrapper, no `recipe_id`) |
| `2_preprocessing_task` | `2_preprocessing` |
| `3_training` | `3_feature_engineering` + `4_forecasting_model` (split) |
| `4_evaluation` | `5_evaluation` |
| `5_horse_race`, `5_output_provenance` | `5_evaluation` + `8_output` |
| `6_decomposition` | `7_interpretation` |
| `7_export` | `8_output` |
| `8_audit` | `1_5_data_summary` / `2_5_pre_post_preprocessing` / etc. (diagnostic layers) |

For a runnable end-to-end example see ``examples/recipes/l4_minimal_ridge.yaml``
or ``examples/recipes/l4_bagging.yaml``.

The Goulet-Coulombe (2021) FRED-MD ridge replication has been ported
to the new format at
``examples/recipes/goulet_coulombe_2021_replication.yaml``.
