# FRED-SD

Real-time U.S. **state-level** macroeconomic panel maintained by the Federal Reserve Bank of St. Louis. Unlike FRED-MD and FRED-QD — both flat national panels — FRED-SD is organized along two axes simultaneously: **variable × state**. It is also the only dataset in this section that mixes native frequencies (monthly + quarterly) within the same file, which drives several of its quirks.

## At a glance

| Field | Value |
|---|---|
| File format | Single Excel workbook — **one sheet per variable**, within each sheet **columns are states** (50 + DC), rows are dates |
| Native frequencies | **Mixed**: monthly for most labour / housing series, quarterly for state GDP / personal income |
| Coverage start | Depends on variable (labour ~1976, state GDP ~2005Q1) — each sheet has its own start |
| Coverage end | Current — refreshed per vintage |
| Panel shape (after loader reshape) | Wide DataFrame with columns `{variable}_{state}` (e.g. `PAYEMS_CA`, `UNRATE_TX`) |
| Maintainer | St. Louis Fed — Bokun, Jackson, Kliesen, Owyang |
| Reference | Bokun, Jackson, Kliesen, Owyang (2022), *Int. J. Forecasting* 38(4) |
| macrocast loader | `macrocast.load_fred_sd()` |
| Recipe selector | `path.1_data_task.fixed_axes.dataset: fred_sd` |
| Support tier | `provisional` — newer than FRED-MD / FRED-QD, fewer downstream axes are wired |

## Citation & authoritative source

- **Paper**: Kathryn O. Bokun, Laura E. Jackson, Kevin L. Kliesen, Michael T. Owyang, "FRED-SD: A Real-Time Database for State-Level Data with Forecasting Applications," *International Journal of Forecasting* **38**(4): 1376–1399, 2022 (accepted 2021).
- **Working paper**: [St. Louis Fed WP 2020-031](https://research.stlouisfed.org/wp/more/2020-031) (first version December 2020).
- **Landing page**: [St. Louis Fed — FRED-SD](https://www.stlouisfed.org/research/economists/owyang/fred-sd).
- **Data page / appendix**: [Research Data — FRED-SD](https://research.stlouisfed.org/data/owyang/fred-sd/). Per-variable metadata and the authoritative variable list live here.

If you cite results produced with FRED-SD, cite Bokun et al. (2022) and record the **vintage label** (see §6 below). State-level series — particularly state GDP — see meaningful revisions, so vintage pinning matters more here than for FRED-MD / FRED-QD.

## What macrocast downloads

- **Current vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/FRED_SD.xlsx`
- **Historical vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/{vintage}.xlsx` where `vintage` follows the publication-month convention (`YYYY-MM`).

Both URLs are referenced verbatim in `macrocast.raw.datasets.fred_sd` as `_CURRENT_URL` / `_VINTAGE_URL`. Cache layout:

```
~/.cache/macrocast/raw/fred_sd/<version>.xlsx
```

**The workbook is laid out 2-dimensionally.** Each sheet is one macro variable; within that sheet, columns are state abbreviations (`AL`, `AK`, …, `WY`, `DC`) and rows are dates indexed by the variable's native frequency (monthly or quarterly — see §4).

The loader reshapes this into a single wide DataFrame on load:

```python
from macrocast import load_fred_sd

# Load everything (~28 variables × 51 states = ~1,400 columns)
result = load_fred_sd()

# Or filter at load time
result = load_fred_sd(
    vintage="2022-03",
    states=["CA", "TX", "NY", "FL"],
    variables=["PAYEMS", "UNRATE"],
)
# result.data.columns -> ["PAYEMS_CA", "PAYEMS_TX", ..., "UNRATE_FL"]
```

Both `states` and `variables` are *loader-level* selectors — they do not currently flow through a Layer 1 axis (see §9 limitations).

## Variable groups — and the frequency split

Per Bokun et al. (2022), FRED-SD provides approximately **28 variables per state**, grouped into three broad categories. Unlike FRED-MD's 8 categories or FRED-QD's 14, FRED-SD's category count is small because state-level real-time data is scarce.

**Frequencies are mixed within the workbook**: each sheet has its own native frequency. The loader does not reconcile them — resulting monthly columns have NaN at positions where a quarterly column has a value, and vice versa.

| Category | Native frequency | Representative variables | Publication lag |
|---|---|---|---|
| **Labour market** | Monthly | State payroll employment (`PAYEMS`-state), state unemployment rate (`UNRATE`-state), labour force (`CLF`), labour force participation, average weekly hours | ~25 days (LAUS / CES state release) |
| **Housing** | Monthly | State housing starts (`HOUST`), permits (`PERMIT`), house price index (`HPI`), per-state regional breakdowns | ~20–30 days (Census / FHFA) |
| **Output / income** | **Quarterly** | State real GDP (`GDP`-state), state personal income (`PI`-state), per-component breakdowns, coincident activity indices | ~90 days (BEA state release) |

**Rule of thumb**: if you want a single-frequency analysis, pick one of the first two rows (monthly). If you need output / GDP, the whole panel becomes quarterly.

Per-variable and per-state coverage drift across vintages — the [data appendix](https://research.stlouisfed.org/data/owyang/fred-sd/) on the St. Louis Fed site tracks the current roster. macrocast does not redistribute the appendix.

### State coverage

FRED-SD covers all **50 U.S. states plus DC** = 51 state-level columns per variable. Some variables have shorter histories in some states because the underlying Fed / BEA / BLS release did not start publishing for that state until later. Missing-state handling:

- **The loader keeps states present in at least one sheet**, even if other sheets have them missing. Missing observations become NaN in the wide DataFrame.
- Filter at load time with `states=["CA", "TX", ...]` to only load a subset.

## Transformation codes

**FRED-SD does not ship T-codes.** The workbook has no header row encoding per-series stationarity transforms the way FRED-MD / FRED-QD do. Consequences:

- `tcode_policy: apply_tcodes` is a **no-op** for FRED-SD (nothing to apply).
- Users who want stationarity transforms must consult the paper or the variable's underlying FRED ID on the FRED website, then apply the transform in user-side preprocessing.
- Monthly labour-market series at the state level broadly mirror their national counterparts' T-codes (e.g. log-diff payrolls, first-diff unemployment rate), but this is not guaranteed per state.

## Vintage & revisions

FRED-SD is explicitly designed as a **real-time** database. Each `.xlsx` file is the dataset *as known at a specific publication month* — every value in the file reflects only revisions available up to that vintage date.

How the vintage maps to `information_set_type`:

| `information_set_type` | Vintage choice | What the panel represents |
|---|---|---|
| `revised` *(default)* | `FRED_SD.xlsx` (current) | Post-revision levels — the "true" number as currently published. No information-leakage protection. |
| `pseudo_oos_revised` | `FRED_SD.xlsx` (current) | Post-revision levels + release-lag masking downstream. Still uses today's revised numbers. |
| `real_time_vintage` | `{vintage}.xlsx` (per `leaf_config.data_vintage`) | Workbook-level vintage. All sheets reflect the same vintage date. |

Unlike FRED-MD (where `real_time_vintage` pulls one monthly CSV snapshot), FRED-SD's `real_time_vintage` pulls one workbook snapshot — all variables across all states at the same vintage. This is coarser than a per-series vintage but consistent with how the authors publish the data.

**Why vintage pinning matters more for FRED-SD**: state GDP and state personal income (the quarterly BEA series) can be revised substantially, sometimes quarters later. A study using `information_set_type: revised` on FRED-SD is implicitly assuming those revisions don't drive results; for nowcasting / real-time studies that's not a safe assumption.

## Real-time release lag

The default `release_lag_rule` for FRED-SD is `fixed_lag_one_period` (one month), which is too tight for the quarterly BEA series and roughly right for the monthly BLS / FHFA series. Representative latencies:

| Series class | Native frequency | Typical first release | Notes |
|---|---|---|---|
| State payroll employment (CES-state) | Monthly | ~25 days | Later than national CES |
| State unemployment rate (LAUS) | Monthly | ~25 days | Model-based; LAUS release |
| State housing starts / permits | Monthly | ~20–30 days | Census per-state breakdowns |
| State house price index (FHFA) | Monthly | ~60 days | |
| State real GDP (BEA) | Quarterly | ~90 days | 1 quarter after national advance GDP |
| State personal income (BEA) | Quarterly | ~90 days | Tied to GDP release |
| Coincident activity indices | Monthly | ~45 days | Model-based; constructed by authors |

For per-series overrides, see [Policies (1.5) — `release_lag_rule`](../user_guide/data/policies.md#release_lag_rule). A state-specific release-lag registry is a v1.1 candidate.

## Schema contract for `custom_csv` / `custom_parquet`

**Limited in v1.0**: FRED-SD's native format is an Excel workbook with per-variable sheets, and macrocast does not currently accept a user-supplied workbook of the same shape through `dataset_source: custom_csv` / `custom_parquet`. If you want a custom state-level panel:

- Flatten externally to a single wide CSV with columns `{variable}_{state}` (matching what the loader produces), a `date` first column, and no T-code row.
- Load via `dataset_source: custom_csv` — but only if the frequencies are uniform across all columns. Mixed frequency is not supported on the custom path.
- A proper FRED-SD-shaped custom route (workbook-in, state + variable metadata preserved) is a v1.1 candidate.

See [Source & Frame (1.1) — Custom CSV / Parquet](../user_guide/data/source.md) for the axis-level mechanics.

## Known quirks / breaking changes

1. **Mixed frequency not reconciled** — quarterly sheets (GDP, PI) sit alongside monthly sheets in the same DataFrame after the loader runs. Monthly-indexed rows at non-quarter-end dates have NaN for the quarterly columns; quarterly-indexed rows (when the quarterly sheet has its own separate index) don't line up with the monthly index at all. Practical patterns:
   - Filter with `variables=[...]` to a same-frequency subset.
   - Use Layer 2 preprocessing (`x_missing_policy`) to forward-fill quarterly observations into monthly gaps.
   - Wait for the mixed-frequency adapter (`mixed_frequency` value in the `frequency` axis, currently `future`, v2 / Phase 11).
2. **No state-selection axis in v1.0** — the loader's `states=[...]` argument is not surfaced through recipe YAML. Recipe-level state filtering is v1.1.
3. **No variable-selection axis for FRED-SD** — the `variable_universe` axis (1.4) operates on FRED-MD / FRED-QD column names; its mapping to FRED-SD sheet names is not wired.
4. **No T-codes** — see §5. Any preprocessing that calls for FRED-MD-style per-series transforms needs user-side work.
5. **Metadata frequency label is `state_monthly` even when quarterly sheets are present** — `RawDatasetMetadata.frequency` is always `state_monthly` regardless of which sheets you loaded. Don't rely on this field to distinguish monthly-only from mixed-frequency subsets.
6. **`support_tier = provisional`** — FRED-SD ingestion is younger than FRED-MD / FRED-QD. Expect rougher edges; pin a vintage for any study intended to be replicable.
7. **Per-variable start dates** — the loader concatenates all sheets with `pd.concat(axis=1)`, so the resulting DataFrame's earliest date is driven by the oldest sheet. Shorter-history sheets have leading NaN. Filter `variables=[...]` to align start dates if this matters.
8. **Excel parsing cost** — reading the entire workbook on every fresh load is noticeably slower than FRED-MD / FRED-QD's CSV parse. Cache hits are fast; the first pull after a vintage change can be several seconds.

## See also

- [FRED-MD](fred_md.md) — sister monthly national database (127 series, 8 categories).
- [FRED-QD](fred_qd.md) — sister quarterly national database (248 series, 14 categories).
- [Source & Frame (1.1)](../user_guide/data/source.md) — `dataset` / `dataset_source` / `frequency` / `information_set_type` axis semantics.
- [Policies (1.5)](../user_guide/data/policies.md) — `release_lag_rule` per-series overrides.
