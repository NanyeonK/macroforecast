# How to add a custom dataset

Use your own panel data (CSV or Parquet) instead of FRED.

---

## File format

Your CSV must have a date column first, then numeric columns:

```text
date,my_target,x1,x2
1990-01-01,1.23,0.45,2.10
1990-02-01,1.31,0.47,2.05
```

Rules:

- Date format: `YYYY-MM-DD` (monthly or quarterly).
- The column you name as `target` in the recipe must exist in the CSV.
- No multi-level headers.

---

## Recipe snippet for a CSV file

```yaml
data:
  fixed_axes:
    panel_composition: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: my_target
    target_horizons: [1]
    custom_source_path: data/my_monthly_panel.csv
    sample_start_date: "1990-01"
    sample_end_date: "2019-12"
```

Run it:

```python
import macroforecast as mf
result = mf.run("my_study.yaml")
```

The path in `custom_source_path` is resolved relative to the working directory
where you call `mf.run()`.

---

## Using inline data

For quick tests or reproducible examples, embed the panel directly in the recipe.
This removes any dependency on external files:

```yaml
data:
  fixed_axes:
    panel_composition: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01, 2020-05-01, 2020-06-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
```

Inline data is embedded in the manifest, so `mf.replicate()` works without
any external files.

---

## Adding custom series to FRED

To run FRED-MD plus your own extra series, use `panel_composition: official_plus_custom`.
The `custom_merge_rule` controls how the two panels are aligned:

```yaml
data:
  fixed_axes:
    panel_composition: official_plus_custom
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: my_target
    target_horizons: [1, 3]
    custom_source_path: data/extra_series.csv
    custom_merge_rule: left_join    # keep all FRED dates; fill custom series with NaN
    # Alternatives: inner_join (shared dates only), outer_join (all dates)
```

---

## Common pitfalls

| Problem | Symptom | Fix |
|---|---|---|
| Date column not first | Parse error at L1 | Move date column to position 0 |
| Target column name mismatch | `RuntimeError: single_target requires leaf_config.target string` | Set `target:` to the exact column name that appears in your CSV header |
| Relative path resolution | `FileNotFoundError` | Run `mf.run()` from the directory that contains your data folder, or use an absolute path |

---

See {doc}`../tutorial/01_first_forecast` for a full tutorial using inline data.
