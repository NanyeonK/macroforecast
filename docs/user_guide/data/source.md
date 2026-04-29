# Source & Frame (1.1)

Declares **where the data comes from and which information-set regime applies**. The user-facing axes answer: which official source panel, whether a custom file replaces or augments it, at what frequency, and under which real-time regime — before the target-structure choice (1.2) or the evaluation window (1.3) is fixed.

| Section | axis | Role |
|---|---|---|
| 1.1.1 | [`dataset`](#111-dataset) | Which official source family to load |
| 1.1.2 | [`custom_source_mode`](#112-custom_source_mode) / [`custom_source_format`](#113-custom_source_format) | Whether a custom file replaces or augments the official panel |
| 1.1.3 | [`frequency`](#114-frequency) | Series frequency (monthly/quarterly); dataset-derived |
| 1.1.4 | [`information_set_type`](#115-information_set_type) | Real-time regime (revised vs. vintage-aware) |

**Note**: `data_domain` axis was dropped entirely in this pass — every FRED dataset implies `domain=macro` via its own source_family metadata, so a separate axis was pure duplication (same rationale as 0.5 `registry_type` drop).
**At a glance (defaults):**
- `dataset` — no default; you pick one official panel: `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, or `fred_qd+fred_sd`.
- `custom_source_mode = no_custom_source` — default; pick `replace_official_panel` or `append_to_official_panel` only when using a custom file.
- `custom_source_format = none` — default; must be `csv` or `parquet` when a custom source is selected.
- `frequency` — derived from `dataset` for MD/QD/composites. Standalone `fred_sd` requires an explicit monthly/quarterly choice.
- `information_set_type = revised` — post-revision truth. Pick `pseudo_oos_on_revised_data` only when you want synthetic release-lag masking.

**Most research runs need only `dataset` + `information_set_type`.** The other source-frame choices auto-derive, except standalone `fred_sd` and custom files.


---

## 1.1.1 `dataset`

**Selects the official source family loaded.** Every recipe picks exactly one official source panel. Custom files are configured separately so the recipe can say whether the file replaces the official panel or is appended to it.

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

## 1.1.2 `custom_source_mode`

**Controls whether a user-supplied file is used with the official source panel.**

| Value | Status | Meaning |
|---|---|---|
| `no_custom_source` | operational | Use only the selected official source panel. |
| `replace_official_panel` | operational | Load a custom file instead of the selected single official panel. |
| `append_to_official_panel` | operational | Load the selected official panel and append custom columns after frequency alignment. |

`replace_official_panel` supports only one official panel (`fred_md`, `fred_qd`, or `fred_sd`). It requires `leaf_config.custom_dataset_schema` to match the selected `dataset`.

`append_to_official_panel` supports single or composite official panels. The custom file can declare `custom_dataset_schema: fred_md`, `fred_qd`, or `fred_sd`; runtime converts it to the selected Layer 1 frequency before appending.

## 1.1.3 `custom_source_format`

**Declares the custom file type.**

| Value | Status | Meaning |
|---|---|---|
| `none` | operational | No custom source file is used. |
| `csv` | operational | Use `macrocast.raw.load_custom_csv`. |
| `parquet` | operational | Use `macrocast.raw.load_custom_parquet`. |

When `custom_source_mode` is not `no_custom_source`:

- **`leaf_config.custom_data_path`** is **required**. Compile-time validation in `compile_recipe_dict` raises `CompileValidationError` if missing.
- **`leaf_config.custom_dataset_schema`** is **required** and declares the schema the custom file must conform to (`fred_md` / `fred_qd` / `fred_sd`). The loader validates the schema label and labels the resulting panel accordingly.
- **CSV shape**: first column is a date index (parseable by `pandas.read_csv(..., parse_dates=True)`); remaining columns are numeric with series IDs as headers. Optional FRED-style T-code row is not consumed automatically — pre-strip it or use `tcode_policy: raw_only`.
- **Parquet shape**: DatetimeIndex OR first column parseable as date; numeric columns.
- **No caching** — the custom loader reads the file fresh each time (no vintage / no cache key). For reproducibility, users should treat the path as part of the recipe provenance and keep the file pinned.
- **`support_tier = provisional`** on the returned `RawLoadResult` — signals the user-supplied panel has not been through FRED's QC / vintage pipeline.

### Functions & features

- `macrocast.load_custom_csv(path, *, dataset, cache_root=None)` — direct call.
- `macrocast.load_custom_parquet(path, *, dataset, cache_root=None)` — direct call (requires pyarrow or fastparquet).
- Dispatcher: `_load_raw_for_recipe` in `macrocast/execution/build.py` dispatches from `custom_source_mode` and `custom_source_format`.
- Compile guard: custom source selected + missing `leaf_config.custom_dataset_schema` or `leaf_config.custom_data_path` -> `CompileValidationError`.

### Recipe usage

Canonical FRED (most common):

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
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
      custom_source_mode: replace_official_panel
      custom_source_format: csv
      frequency: monthly
    leaf_config:
      target: INDPRO
      horizons: [1, 3]
      custom_dataset_schema: fred_md
      custom_data_path: /path/to/my_fred_md_extract.csv
```

Append custom Parquet columns to FRED-QD:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_qd
      custom_source_mode: append_to_official_panel
      custom_source_format: parquet
      frequency: quarterly
    leaf_config:
      target: GDPC1
      horizons: [1, 2, 4]
      custom_dataset_schema: fred_qd
      custom_data_path: /path/to/my_fred_qd_extract.parquet
```

---

## 1.1.4 `frequency`

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

## 1.1.3 `information_set_type`

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
