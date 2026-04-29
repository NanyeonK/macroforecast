# Source & Frame (1.1)

Declares **where the data comes from and which information-set regime applies**. The user-facing axes answer: which FRED source panel is used, whether a custom file replaces or augments it, at what frequency, and under which real-time regime — before the target-structure choice (1.2) or the evaluation window (1.3) is fixed.

| Section | axis | Role |
|---|---|---|
| 1.1.1 | [`dataset`](#111-dataset) | Which FRED source family to load |
| 1.1.2 | [`custom_source_policy`](#112-custom_source_policy) | FRED data only, custom data only, or FRED plus custom data |
| 1.1.3 | [`custom_source_format`](#113-custom_source_format) | Custom file parser |
| 1.1.4 | [`custom_source_schema`](#114-custom_source_schema) | FRED-family schema the custom file follows |
| 1.1.5 | [`frequency`](#115-frequency) | Series frequency (monthly/quarterly); dataset-derived |
| 1.1.6 | [`information_set_type`](#116-information_set_type) | Real-time regime (revised vs. vintage-aware) |

**Note**: `data_domain` axis was dropped entirely in this pass — every FRED dataset implies `domain=macro` via its own source_family metadata, so a separate axis was pure duplication (same rationale as 0.5 `registry_type` drop).
**At a glance (defaults):**
- `dataset` — no default; you pick one FRED panel: `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, or `fred_qd+fred_sd`.
- `custom_source_policy = official_only` — default; choose `custom_panel_only` for custom data only, or `official_plus_custom` for FRED data plus custom data.
- `custom_source_format = none` — only means no custom file. Once custom data is selected, Navigator treats this as an explicit required choice: `csv` or `parquet`.
- `custom_source_schema = none` — only means no custom file. Once custom data is selected, Navigator treats this as an explicit required choice: `fred_md`, `fred_qd`, or `fred_sd`.
- `frequency` — derived from `dataset` for MD/QD/composites. Standalone `fred_sd` requires an explicit monthly/quarterly choice.
- `information_set_type = final_revised_data` — post-revision truth. Pick `pseudo_oos_on_revised_data` only when you want synthetic release-lag masking.

**Most research runs need only `dataset` + `information_set_type`.** The other source-frame choices auto-derive, except standalone `fred_sd` and custom files.


---

## 1.1.1 `dataset`

**Selects the FRED source family loaded.** Every recipe picks exactly one FRED source panel. Custom files are configured separately so the recipe can say whether the file replaces the FRED panel or is appended to it.

### Value catalog

| Value | Status | Loader | Content |
|---|---|---|---|
| `fred_md` | operational | `macrocast.raw.load_fred_md` | FRED-MD monthly macro panel (McCracken & Ng 2016) — see [fred_md.md](datasets/fred_md.md) |
| `fred_qd` | operational | `macrocast.raw.load_fred_qd` | FRED-QD quarterly macro panel (McCracken & Ng 2020) — see [fred_qd.md](datasets/fred_qd.md) |
| `fred_sd` | operational | `macrocast.raw.load_fred_sd` | FRED-SD state-level real-time panel (Bokun, Jackson, Kliesen, Owyang 2022) — see [fred_sd.md](datasets/fred_sd.md) |
| `fred_md+fred_sd` | operational | composite loader | FRED-MD monthly panel plus FRED-SD state panel converted to monthly when needed. |
| `fred_qd+fred_sd` | operational | composite loader | FRED-QD quarterly panel plus FRED-SD state panel converted to quarterly when needed. |

Each base dataset has its own dedicated documentation page covering citation, download path, variable groups, transformation codes, and changes from the original working paper to the current vintage. Composite dataset values are compiler/runtime contracts over those base loaders: the registry entry is used by the compiler, the composite loader chooses the relevant FRED loaders at run time, and the merged panel flows into every downstream axis.

### Functions & features

- `macrocast.load_fred_md()` / `load_fred_qd()` / `load_fred_sd()` — public loaders.
- `macrocast.raw.datasets.fred_md` / `fred_qd` / `fred_sd` — per-dataset modules with cache + manifest logic.
- Compiler reads `dataset` via `_selection_value(selection_map, "dataset")` → propagated into `CompiledRecipeSpec.dataset` and every downstream spec.
- `_DATASET_DEFAULT_FREQUENCY` in `compiler/build.py` maps MD/QD/composite datasets to their default `frequency` value.
- Compile guard: standalone `dataset=fred_sd` requires explicit `frequency`; `fred_md+fred_sd` requires monthly; `fred_qd+fred_sd` requires quarterly.

## 1.1.2 `custom_source_policy`

**Controls whether a custom file is used with the FRED source panel.**

| Value | Status | Meaning |
|---|---|---|
| `official_only` | operational | FRED data only. Use the selected FRED source panel. |
| `custom_panel_only` | operational | Custom data only. Load a custom file instead of the selected single FRED panel. |
| `official_plus_custom` | operational | FRED plus custom data. Load the selected FRED panel and append custom columns after frequency alignment. |

`custom_panel_only` supports only one FRED panel (`fred_md`, `fred_qd`, or `fred_sd`). It requires `custom_source_schema` to match the selected `dataset`.

`official_plus_custom` supports single or composite FRED panels. The custom file can declare `custom_source_schema: fred_md`, `fred_qd`, or `fred_sd`; runtime converts it to the selected Layer 1 frequency before appending.

## 1.1.3 `custom_source_format`

**Declares the custom file type.**

| Value | Status | Meaning |
|---|---|---|
| `none` | operational | No custom file is used. Valid only with `custom_source_policy: official_only`. |
| `csv` | operational | Use `macrocast.raw.load_custom_csv`. |
| `parquet` | operational | Use `macrocast.raw.load_custom_parquet`. |

When `custom_source_policy` is not `official_only`:

- **`leaf_config.custom_source_path`** is **required**. Compile-time validation in `compile_recipe_dict` raises `CompileValidationError` if missing.
- **`custom_source_schema`** is **required** in `fixed_axes` and declares the schema the custom file must conform to (`fred_md` / `fred_qd` / `fred_sd`). The loader validates the schema label and labels the resulting panel accordingly.
- **CSV shape**: first column is a date index (parseable by `pandas.read_csv(..., parse_dates=True)`); remaining columns are numeric with series IDs as headers. Optional FRED-style T-code row is not consumed automatically — pre-strip it or use `tcode_policy: raw_only`.
- **Parquet shape**: DatetimeIndex OR first column parseable as date; numeric columns.
- **Frequency shape**: the file only needs to satisfy the selected route's frequency contract. Monthly routes need rows that are monthly or can be converted to monthly; quarterly routes need rows that are quarterly or can be converted/aggregated to quarterly. The `custom_source_schema` tells Layer 1 how to align the custom file.
- **No caching** — the custom loader reads the file fresh each time (no vintage / no cache key). For reproducibility, users should treat the path as part of the recipe provenance and keep the file pinned.
- **`support_tier = provisional`** on the returned `RawLoadResult` — signals the user-supplied panel has not been through FRED's QC / vintage pipeline.

> **Warning:** The package can validate enum choices and file parseability, but
> it cannot certify the economic meaning of custom columns, vintage discipline,
> release lags, or whether a T-code row was stripped correctly. Treat the file
> path and file hash as part of study provenance.

## 1.1.4 `custom_source_schema`

**Declares which FRED-style panel shape the custom file follows.** This is a
decision axis because the Navigator must show it only when custom data is used.
The file path remains in `leaf_config.custom_source_path` because paths are
study-specific payload, not an enum.

| Value | Status | Required shape |
|---|---|---|
| `none` | operational | No custom source file is used. Valid only with `custom_source_policy: official_only`. |
| `fred_md` | operational | Monthly date index, national macro series columns, normally FRED-style series IDs. |
| `fred_qd` | operational | Quarterly date index, national macro series columns, normally FRED-style series IDs. |
| `fred_sd` | operational | State-level series columns. Current v1 custom-source runtime treats the file as monthly (`state_monthly`) before Layer 1 alignment. |

Frequency behavior:

| Custom schema | If selected Layer 1 `frequency` is monthly | If selected Layer 1 `frequency` is quarterly |
|---|---|---|
| `fred_md` | Used at native monthly frequency. | Aggregated to quarterly by 3-month average when appended to a quarterly route. |
| `fred_qd` | Interpolated to monthly when appended to a monthly route. | Used at native quarterly frequency. |
| `fred_sd` | Used as monthly state-level data. | Aggregated to quarterly by 3-month average. |

For `custom_panel_only`, the schema must match `dataset`: `dataset: fred_md`
requires `custom_source_schema: fred_md`, and so on. For `official_plus_custom`,
the schema may differ from the FRED panel because Layer 1 aligns the custom
file to the selected route frequency before appending.

### Functions & features

- `macrocast.load_custom_csv(path, *, dataset, cache_root=None)` — direct call.
- `macrocast.load_custom_parquet(path, *, dataset, cache_root=None)` — direct call (requires pyarrow or fastparquet).
- Dispatcher: `_load_raw_for_recipe` in `macrocast/execution/build.py` dispatches from `custom_source_policy` and `custom_source_format`.
- Compile guard: custom source selected + missing `custom_source_schema` or `leaf_config.custom_source_path` -> `CompileValidationError`.

### Recipe usage

Canonical FRED (most common):

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      custom_source_policy: official_only
      custom_source_format: none
      custom_source_schema: none
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
```

Replace FRED-MD with a user-supplied CSV:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      custom_source_policy: custom_panel_only
      custom_source_format: csv
      custom_source_schema: fred_md
      frequency: monthly
    leaf_config:
      target: INDPRO
      horizons: [1, 3]
      custom_source_path: /path/to/custom_fred_md_extract.csv
```

Append custom Parquet columns to FRED-QD:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_qd
      custom_source_policy: official_plus_custom
      custom_source_format: parquet
      custom_source_schema: fred_qd
      frequency: quarterly
    leaf_config:
      target: GDPC1
      horizons: [1, 2, 4]
      custom_source_path: /path/to/custom_fred_qd_extract.parquet
```

---

## 1.1.5 `frequency`

**Series frequency of the dataset panel.** Dataset-derived for MD/QD/composite runs; required for standalone FRED-SD runs.

### Value catalog

| Value | Status | Which dataset uses this |
|---|---|---|
| `monthly` | operational | `fred_md`, `fred_md+fred_sd`, standalone `fred_sd` when requested |
| `quarterly` | operational | `fred_qd`, `fred_qd+fred_sd`, standalone `fred_sd` when requested |

### Functions & features

- Compiler default: `dataset=fred_md` implies `frequency=monthly`, `dataset=fred_qd` implies `frequency=quarterly`, `dataset=fred_md+fred_sd` implies `monthly`, and `dataset=fred_qd+fred_sd` implies `quarterly`.
- Standalone `dataset=fred_sd` has no safe default because the source contains monthly and quarterly state series. The compiler requires `frequency` in `fixed_axes`.
- Runtime frequency conversion is active: monthly data requested as quarterly are converted by 3-month average; quarterly data requested as monthly are linearly interpolated. Both conversions append warnings/reports to provenance.
- Composite rules are strict: `fred_md+fred_sd` is monthly and `fred_qd+fred_sd` is quarterly. Explicit conflicting `frequency` values raise `CompileValidationError`.

### Dropped values

- `daily`, `weekly`, `yearly`, `mixed_frequency` — FRED-MD/QD only expose monthly and quarterly cadences. Daily/weekly/yearly would require new loaders. FRED-SD can be loaded and converted to monthly or quarterly, and the package has a narrow first-class mixed-frequency route through native-frequency payloads, custom adapters, `midas_almon`, and `midasr` with `nealmon` / `almonp` / `nbeta` / `genexp` / `harstep`; state-space infrastructure remains future.
- Manifest records `frequency` for provenance.

### Recipe usage

Usually omitted for MD/QD/composites because the dataset implies the frequency. Required for standalone FRED-SD.

---

## 1.1.6 `information_set_type`

**Real-time regime** that governs which version of each observation the model is allowed to see at each forecast origin. Fully wired — this is the only 1.1 axis with compile-time validation AND runtime dispatch across its operational values.

### Value catalog

| Value | Status | Contract |
|---|---|---|
| `final_revised_data` | operational | Latest revised values (post-revision truth). Default. |
| `pseudo_oos_on_revised_data` | operational | Pseudo out-of-sample: latest revised values but masked according to (fake) release-lag discipline. |

### Functions & features

- Runtime: loaders (`raw/datasets/fred_md.py` etc.) pick the correct data source based on the dataset axis; `information_set_type` shapes the downstream pseudo-OOS masking when set.
- Canonical API: recipes use `information_set_type` directly; removed information-set aliases are rejected at compile time.

### Dropped values

- `real_time_vintage`, `release_calendar_aware`, `publication_lag_aware` — real-time vintage / release-calendar / publication-lag stacks require data-infrastructure that is outside v1.0 scope. The associated `vintage_policy` axis was dropped in the 1.5 cleanup for the same reason.
- `pseudo_oos_vintage_aware` — metadata-only label whose intended semantics (publication-lag-aware pseudo OOS) is already expressible through `release_lag_rule` (1.5, `fixed_lag_all_series` / `series_specific_lag`). Dropped to remove the duplicate surface.

### Recipe usage

```yaml
# Revised (post-revision) default
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO

# Pseudo-OOS on revised data
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: pseudo_oos_on_revised_data
```

---

## Source & Frame (1.1) takeaways

- **`dataset`** and **`information_set_type`** are the two axes the user usually decides in 1.1; standalone FRED-SD also requires `frequency`.
- **`frequency`** is executable: it controls conversion and is compile-checked for FRED-SD composites.
- **`data_domain`** axis dropped entirely (pure duplication of `dataset.source_family`).

Next group: [1.2 Target Structure](target_structure.md) (coming) — what exactly is being forecast.
