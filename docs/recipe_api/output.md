# Understanding Recipe Output

Layer 8 writes the run directory. The default L8 output directory is `./macrocast_output/default_recipe/timestamp/` unless `8_output.leaf_config.output_directory` overrides it.

Common files:

```text
<output_directory>/
  manifest.json
  recipe.json
  summary/
    metrics_all_cells.csv
    ranking.csv
  cell_001/
    forecasts.csv
    feature_metadata.json
    clean_panel.csv
    raw_panel.csv
    figures/
      *.pdf
  diagnostics/
    *_diagnostic_v1.json
  tests_summary.json
  importance_summary.json
  report.html
```

Files appear only when the corresponding layer ran and the object is saved. Core saved objects include `forecasts`, `metrics`, `ranking`, `tests`, `importance`, `feature_metadata`, `clean_panel`, `raw_panel`, and `diagnostics_all`.

Use metric names exactly as emitted by L5: `mse`, `rmse`, `mae`, `medae`, `theil_u1`, `theil_u2`, `success_ratio`, `pesaran_timmermann_metric`, `relative_mse`, `r2_oos`, `relative_mae`, `mse_reduction`, `log_score`, `crps`, `interval_score`, `coverage_rate`.
