# Source & Frame (1.1)

Declares **where the data comes from and which information-set regime applies**. These four axes together answer: which dataset, at what frequency, and under which real-time regime — before the target-structure choice (1.2) or the evaluation window (1.3) is fixed.

| Section | axis | Role |
|---|---|---|
| 1.1.1 | [`dataset`](#111-dataset) | Which FRED-family dataset schema to load |
| 1.1.2 | [`source_adapter`](#112-source_adapter) | Which loader to use (FRED canonical / user CSV / user Parquet) |
| 1.1.3 | [`frequency`](#113-frequency) | Series frequency (monthly/quarterly); dataset-derived |
| 1.1.4 | [`information_set_type`](#114-information_set_type) | Real-time regime (revised vs. vintage-aware) |

**Note**: `data_domain` axis was dropped entirely in this pass — every FRED dataset implies `domain=macro` via its own source_family metadata, so a separate axis was pure duplication (same rationale as 0.5 `registry_type` drop).
**At a glance (defaults):**
- `dataset` — no default; you pick one of `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd`.
- `source_adapter` — mirrors `dataset` automatically (canonical FRED loader). Override only for `custom_csv` / `custom_parquet`.
- `frequency` — derived from `dataset` for MD/QD/composites. Standalone `fred_sd` requires an explicit monthly/quarterly choice.
- `information_set_type = revised` — post-revision truth. Pick `pseudo_oos_revised` only when you want synthetic release-lag masking.

**Most research runs need only `dataset` + `information_set_type`.** The other two auto-derive.


---

## 1.1.1 `dataset`

**Selects the schema of the dataset loaded.** Every recipe picks exactly one.

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

### Recipe usage

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
```

---

## 1.1.2 `source_adapter`

**Selects which loader is used.** Orthogonal to `dataset` (which declares the schema). The canonical FRED values call the corresponding `load_fred_*` loader; the two `custom_*` values load a user-supplied file conforming to the declared schema.

### Value catalog

| Value | Status | Loader | What it does |
|---|---|---|---|
| `fred_md` | operational | `load_fred_md` | FRED-MD from St. Louis Fed (default for `dataset=fred_md`) |
| `fred_qd` | operational | `load_fred_qd` | FRED-QD from St. Louis Fed |
| `fred_sd` | operational | `load_fred_sd` | FRED-SD from St. Louis Fed |
| `custom_csv` | operational | `load_custom_csv` | **User-supplied CSV** at `leaf_config.custom_data_path`; must conform to `dataset` schema |
| `custom_parquet` | operational | `load_custom_parquet` | **User-supplied Parquet** at `leaf_config.custom_data_path`; same schema rules |

### What was dropped in this pass

The previous iteration carried 14 reserved source-adapter labels (`bea`, `bls`, `census`, `oecd`, `imf_ifs`, `ecb_sdw`, `bis`, `world_bank`, `wrds_macro_finance`, `survey_spf`, `fred_api_custom`, `market_prices`, `news_text`, `custom_sql`) as registry_only or future. None had a concrete adapter roadmap for v1.0 / v1.1 that justified keeping them in the registry, so they were dropped outright. If and when a third-party adapter ships, the corresponding value can be re-registered.

### Custom CSV / Parquet — implementation contract

When `source_adapter` is `custom_csv` or `custom_parquet`:

- **`leaf_config.custom_data_path`** is **required**. Compile-time validation in `compile_recipe_dict` raises `CompileValidationError` if missing.
- **`dataset`** still declares the schema the custom file must conform to (`fred_md` / `fred_qd` / `fred_sd`). The loader validates the schema label and labels the resulting panel accordingly.
- **CSV shape**: first column is a date index (parseable by `pandas.read_csv(..., parse_dates=True)`); remaining columns are numeric with series IDs as headers. Optional FRED-style T-code row is not consumed automatically — pre-strip it or use `tcode_policy: raw_only`.
- **Parquet shape**: DatetimeIndex OR first column parseable as date; numeric columns.
- **No caching** — the custom loader reads the file fresh each time (no vintage / no cache key). For reproducibility, users should treat the path as part of the recipe provenance and keep the file pinned.
- **`support_tier = provisional`** on the returned `RawLoadResult` — signals the user-supplied panel has not been through FRED's QC / vintage pipeline.

### Functions & features

- `macrocast.load_custom_csv(path, *, dataset, cache_root=None)` — direct call.
- `macrocast.load_custom_parquet(path, *, dataset, cache_root=None)` — direct call (requires pyarrow or fastparquet).
- Dispatcher: `_load_raw_for_recipe` in `macrocast/execution/build.py` checks `recipe.data_task_spec["source_adapter"]` first; falls through to the FRED-canonical path otherwise.
- Compile guard: `source_adapter ∈ {custom_csv, custom_parquet}` + no `leaf_config.custom_data_path` → `CompileValidationError`.
- Compatibility: legacy recipes may still use `dataset_source`; the compiler treats it as an alias for `source_adapter` and rejects conflicting old/new choices.

### Recipe usage

Canonical FRED (most common):

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      # source_adapter: fred_md  (omit — defaults to dataset value)
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
```

User-supplied CSV:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md          # schema declaration
      source_adapter: custom_csv
    leaf_config:
      target: INDPRO
      horizons: [1, 3]
      custom_data_path: /path/to/my_fred_md_extract.csv
```

User-supplied Parquet:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_qd
      source_adapter: custom_parquet
    leaf_config:
      target: GDPC1
      horizons: [1, 2, 4]
      custom_data_path: /path/to/my_fred_qd_extract.parquet
```

---

## 1.1.3 `frequency`

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

- `daily`, `weekly`, `yearly`, `mixed_frequency` — FRED-MD/QD only expose monthly and quarterly cadences. Daily/weekly/yearly would require new loaders. FRED-SD can be loaded and converted to monthly or quarterly, and the package has a narrow first-class mixed-frequency route through native-frequency payloads, custom adapters, `midas_almon`, and `midasr` with `nealmon` / `almonp`; state-space infrastructure remains future.
- Manifest records `frequency` for provenance.

### Recipe usage

Usually omitted for MD/QD/composites because the dataset implies the frequency. Required for standalone FRED-SD.

---

## 1.1.4 `information_set_type`

**Real-time regime** that governs which version of each observation the model is allowed to see at each forecast origin. Fully wired — this is the only 1.1 axis with compile-time validation AND runtime dispatch across its operational values.

### Value catalog

| Value | Status | Contract |
|---|---|---|
| `revised` | operational | Latest revised values (post-revision truth). Default. |
| `pseudo_oos_revised` | operational | Pseudo out-of-sample: latest revised values but masked according to (fake) release-lag discipline. |

### Functions & features

- Runtime: loaders (`raw/datasets/fred_md.py` etc.) pick the correct data source based on the dataset axis; `information_set_type` shapes the downstream pseudo-OOS masking when set.
- Compat mirror: the older recipe alias `info_set` is canonicalised to `information_set_type` (compiler/build.py alias map).

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
      information_set_type: revised
    leaf_config:
      target: INDPRO

# Pseudo-OOS on revised data
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: pseudo_oos_revised
```

---

## Source & Frame (1.1) takeaways

- **`dataset`** and **`information_set_type`** are the two axes the user usually decides in 1.1; standalone FRED-SD also requires `frequency`.
- **`source_adapter`** now carries actual loader dispatch: FRED canonical (default) vs `custom_csv` vs `custom_parquet`. 14 reserved third-party adapter labels dropped. Legacy `dataset_source` remains a recipe alias during the compatibility window.
- **`frequency`** is executable: it controls conversion and is compile-checked for FRED-SD composites.
- **`data_domain`** axis dropped entirely (pure duplication of `dataset.source_family`).

Next group: [1.2 Target Structure](target_structure.md) (coming) — what exactly is being forecast.
