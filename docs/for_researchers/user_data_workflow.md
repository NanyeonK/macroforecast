# Bring Your Own Data

macroforecast works with any time-series panel you supply. This guide covers
monthly and quarterly CSV / Parquet files.

If you prefer the official FRED-MD or FRED-QD panels, start with
[FRED-MD](fred_datasets/fred_md.md) or [FRED-QD](fred_datasets/fred_qd.md)
instead.

> **FRED-MD/QD format note**: the raw FRED CSV files include a `Transform:`
> header row above the data. Your custom CSV must **not** include that row --
> it is an artefact of the official FRED format and is stripped automatically
> only when `dataset=fred_md` / `fred_qd` uses the built-in adapter. Custom
> CSV files are plain panels: date index + numeric columns only.

## When to use this guide

Use this guide when you have:

- A proprietary indicator panel (e.g., firm-level surveys, regional prices).
- A monthly or quarterly series not available in FRED.
- A country-specific macro panel.

If you have a few additional series you want to **add on top of** the official
FRED panel, see [Merging with FRED-MD or FRED-QD](#merging-with-fred-md-or-fred-qd).

## File format contract

### Monthly CSV

```csv
date,my_target,x1,x2
1990-01-01,1.23,0.45,2.10
1990-02-01,1.31,0.47,2.05
1990-03-01,1.29,0.46,1.99
```

Rules:

- First column: date, parseable by pandas (`YYYY-MM-DD` is the safest format;
  `YYYY-MM` also works when the day is not meaningful).
- Remaining columns: numeric. Non-numeric cells are coerced to `NaN`;
  columns that are entirely `NaN` are dropped silently.
- No `Transform:` row. No multi-level headers. No trailing metadata rows.
- The column you name as `target` in the recipe must be present.

### Quarterly CSV

Same rules. Use `YYYY-01-01`, `YYYY-04-01`, `YYYY-07-01`, `YYYY-10-01` as
quarterly date stamps, or any convention pandas parses to quarterly periods.
The recipe axis `frequency: quarterly` tells the runtime to interpret the
dates as quarterly.

### Parquet

Same schema as CSV. The Parquet file may have either a `DatetimeIndex` or
a date column as its first column. Column names and numeric typing rules
are identical.

## Running with your own data

### Option A: YAML recipe (recommended)

Set `custom_source_policy: custom_panel_only` and point `custom_source_path`
at your file. The runtime infers CSV vs Parquet from the file extension
(`.csv` -> CSV loader; `.parquet` or `.pq` -> Parquet loader).

**Monthly example**

```yaml
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible

1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    dataset: fred_md          # labels the panel as "monthly" in the runtime
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: my_target
    target_horizons: [1, 3, 6]
    custom_source_path: data/my_monthly_panel.csv
    sample_start_date: "1990-01"
    sample_end_date: "2019-12"

2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced

3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto

4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, min_train_size: 24, forecast_strategy: direct,
               training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict_ridge, type: step, op: predict, inputs: [fit_ridge, src_X]}
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: fit_ridge
    l4_training_metadata_v1: auto

5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]

8_output:
  fixed_axes:
    saved_objects: [forecasts, metrics, ranking]
  leaf_config:
    output_directory: ./output/my_study/
```

Run it:

```python
import macroforecast as mf
result = mf.run("my_study.yaml", output_directory="output/my_study/")
print(result.cells[0].sink_hashes)
```

**Quarterly example**

Change two lines:

```yaml
    dataset: fred_qd          # labels the panel as "quarterly"
    frequency: quarterly
```

Everything else stays the same. The quarterly panel uses the same date-index
format rules as monthly; the runtime resolves the frequency from `dataset`.

### Option B: Python helper functions

`mf.load_custom_csv` and `mf.load_custom_parquet` load your file and return
a `RawLoadResult` you can inspect before running a full study.

```python
import macroforecast as mf

# Monthly panel
result = mf.load_custom_csv("data/my_monthly_panel.csv", dataset="fred_md")
print(result.data.head())           # pandas DataFrame, date index
print(result.dataset_metadata)      # frequency, data_through, etc.

# Quarterly panel
result_q = mf.load_custom_csv("data/my_quarterly_panel.csv", dataset="fred_qd")

# Parquet
result_pq = mf.load_custom_parquet("data/my_panel.parquet", dataset="fred_md")
```

`dataset` must be one of `"fred_md"` (monthly), `"fred_qd"` (quarterly), or
`"fred_sd"` (state-level monthly). It labels the schema downstream -- it does
not require your columns to match FRED mnemonics.

These helper functions are for inspection only. To run a full study, use the
YAML recipe path (Option A).

## Merging with FRED-MD or FRED-QD

If you want McCracken-Ng's curated 126 monthly (or 245 quarterly) series
**plus** a few custom series, use `official_plus_custom`:

```yaml
1_data:
  fixed_axes:
    custom_source_policy: official_plus_custom
    dataset: fred_md
    frequency: monthly
  leaf_config:
    target: CPIAUCSL
    target_horizons: [1, 3, 6]
    custom_source_path: data/my_extra_series.csv
    custom_merge_rule: left_join    # inner_join / left_join / outer_join
    sample_start_date: "1990-01"
    sample_end_date: "2019-12"
```

`custom_merge_rule` is required. Choose:

| Rule | Keeps dates from |
|---|---|
| `inner_join` | Rows present in **both** FRED and your file |
| `left_join` | All FRED dates; your series gets `NaN` where missing |
| `outer_join` | All dates in either file |

The custom file must have the same date column format. Duplicate column names
(same mnemonic as a FRED series) will be suffixed by the runtime; rename
before merging if the intent is to replace a FRED series.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `RawParseError: must have a parseable date index` | Date column is not the first column, or the date format is not parseable. | Move the date column first; use ISO format `YYYY-MM-DD`. |
| Target column is silently missing from the panel | Column name in `target:` does not match the CSV header (case-sensitive). | Check column names with `pd.read_csv("file.csv").columns`. |
| All-NaN columns dropped silently | A series has no numeric values after type coercion. | Inspect the raw file for text entries or hidden characters. |
| `official_transform_policy` has no effect | `custom_panel_only` disables FRED T-code application. | Apply your own transforms in `2_preprocessing` via `transform_policy: tcode` and a custom T-code map, or use `no_transform` and handle it upstream. |
| `custom_source_path` not found at runtime | Relative path resolves from where `mf.run()` is called, not from the YAML location. | Use an absolute path or change your working directory to the project root before calling `mf.run()`. |
| `official_plus_custom` fails with date mismatch | Your extra file's date range does not overlap the FRED vintage dates. | Use `outer_join` or trim your sample dates to the intersection. |

For FRED-MD / FRED-QD column definitions and T-code reference, see
[FRED-MD](fred_datasets/fred_md.md) and [FRED-QD](fred_datasets/fred_qd.md).

## See also

- [FRED-MD column dictionary](fred_datasets/fred_md.md) -- 126 monthly series, T-codes, groups
- [FRED-QD column dictionary](fred_datasets/fred_qd.md) -- 245 quarterly series, T-codes
- [Custom function quickstart](../for_recipe_authors/custom_function_quickstart.md) -- bring your own model, preprocessor, or target transformer
- [Quickstart](quickstart.md) -- minimal recipe walkthrough
- [First study](first_study.md) -- full study with diagnostics, tests, and output
