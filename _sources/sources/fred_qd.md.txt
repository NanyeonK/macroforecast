# FRED-QD

Quarterly U.S. macroeconomic panel maintained by the Federal Reserve Bank of St. Louis. Sister dataset to [FRED-MD](fred_md.md) — same maintainer, same two-row CSV format, same upstream hosting — but quarterly, extended with balance-sheet aggregates that are only available at quarterly frequency. Designed for longer-horizon forecasting, factor models requiring real-GDP-style series, and studies of credit and household wealth.

## At a glance

| Field | Value |
|---|---|
| Frequency | Quarterly (observations timestamped to quarter end) |
| Coverage start | 1959:Q1 |
| Coverage end | Advance release — updated at each BEA advance GDP release (roughly Q+30 days) |
| Panel size | ~248 series across 14 categories (current vintage) |
| File format | Single CSV with two-row header |
| Maintainer | St. Louis Fed — Michael McCracken and team |
| Reference | McCracken & Ng (2021), *FRB St. Louis Review* 103(1) |
| macrocast loader | `macrocast.load_fred_qd()` |
| Recipe selector | `path.1_data_task.fixed_axes.dataset: fred_qd` |

## Citation & authoritative source

- **Paper**: Michael W. McCracken and Serena Ng, "FRED-QD: A Quarterly Database for Macroeconomic Research," *Federal Reserve Bank of St. Louis Review* **103**(1): 1–44, 2021.
- **Working paper**: [NBER WP 26872](https://www.nber.org/papers/w26872) (March 2020). St. Louis Fed WP: [2020-005](https://research.stlouisfed.org/wp/more/2020-005).
- **Landing page**: [St. Louis Fed — FRED-MD & FRED-QD](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) (documentation, appendix, vintages — shared with FRED-MD).
- **Variable appendix (current)**: `FRED-QD_updated_appendix.pdf` linked from the landing page. Authoritative list of every series, T-code, source FRED ID, and description. macrocast does not redistribute the appendix.

If you cite results produced with FRED-QD, cite McCracken & Ng (2021) and record the **vintage label** (see §6 below) so the exact panel can be reproduced.

## What macrocast downloads

Hosted under the `fred-md/quarterly/` path on St. Louis Fed servers (yes — FRED-QD lives under the `fred-md` namespace; the two datasets share hosting infrastructure):

- **Current vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv`
- **Historical vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/{vintage}.csv` where `vintage` is `YYYY-MM` (e.g., `2020-06`).

Both URLs are referenced verbatim in `macrocast.raw.datasets.fred_qd` as `_CURRENT_URL` / `_VINTAGE_URL`.

CSV layout:

- **Row 1** (header): series IDs (first column literal `sasdate`).
- **Row 2**: T-code per series (first column blank / numeric placeholder — shared helper `parse_fred_csv` strips it into a separate `tcodes` record).
- **Rows 3+**: quarterly observations, first column = quarter-end date.

Cache layout (see `macrocast.raw.cache.get_raw_file_path`):

```
~/.cache/macrocast/raw/fred_qd/<version>.csv
```

Where `<version>` is `current` or `YYYY-MM` for pinned vintages. The first call downloads; subsequent calls hit cache. Pass `force=True` to `load_fred_qd()` to re-download.

## Variable groups

FRED-QD is benchmarked to the 200-series quarterly panel of Stock & Watson (2012a) and extended with ~48 additional series. McCracken & Ng (2021) organise the panel into **14 categories**. The category label is in the appendix, not the CSV file — the CSV itself has no category column.

| # | Category | Focus | Representative series |
|:-:|---|---|---|
| 1 | **NIPA** | Real GDP and components | `GDPC1` (real GDP), `PCECC96` (real consumption), `GPDIC1` (real private investment), `EXPGSC1` / `IMPGSC1` (real exports / imports), `GCEC1` (real government spending), `GDPDEF` (GDP deflator) |
| 2 | **Industrial Production** | IP indices at aggregate and sectoral levels | `INDPRO` (total IP), final products / intermediate / materials sub-indices |
| 3 | **Employment and Unemployment** | Quarterly labour aggregates | `PAYEMS` (total nonfarm payrolls), `UNRATE` (civilian unemployment), `CE16OV` (civilian employment), `CLF16OV` (labour force), hours, participation |
| 4 | **Housing** | Quarterly housing activity and tenure | `HOUST` (housing starts), `PERMIT` (building permits), existing home sales, owner-occupier equity (quarterly-only) |
| 5 | **Inventories, Orders, and Sales** | Supply-chain aggregates | Manufacturing inventories, new orders, wholesale / retail sales |
| 6 | **Prices** | Price indices and deflators | `CPIAUCSL` (CPI all items), `PCEPI` (PCE price index), core variants, `OILPRICEx` (WTI crude), commodity price indices |
| 7 | **Earnings and Productivity** | Quarterly-only productivity / compensation | `OPHNFB` (nonfarm business productivity), `ULCNFB` (unit labour costs), `COMPNFB` (real compensation per hour), `HOANBS` (hours) |
| 8 | **Interest Rates** | Short, medium, long rates + spreads | `FEDFUNDS` (Fed funds), `TB3MS` (3-month T-bill), `GS10` (10-year yield), `AAA` / `BAA` (corporate bond yields) |
| 9 | **Money and Credit** | Monetary aggregates, bank credit | `M1SL`, `M2SL`, `BUSLOANS` (C&I loans), bank reserves, total credit to the non-financial sector |
| 10 | **Household Balance Sheets** | Household wealth and obligations | Household net worth, financial obligations ratios, debt service ratios |
| 11 | **Exchange Rates** | FX indices and bilateral rates | `TWEXBMTH` (trade-weighted dollar), major bilateral rates |
| 12 | **Stock Markets** | Equity and volatility | `SP500`, dividend yield, realised-volatility proxies |
| 13 | **Non-Household Balance Sheets** *(new in FRED-QD)* | Corporate and bank balance sheets — the headline FRED-QD extension over Stock–Watson (2012a) | Corporate liabilities, non-corporate business liabilities, bank-sector balance sheet aggregates |
| 14 | **Other** | Miscellaneous series the paper treats separately | Survey expectations, residual series |

**Exact membership and counts per group evolve across vintages** — the authoritative list is the appendix PDF. The representative members above are stable across the recent window (2020 paper through current).

When macrocast uses `variable_universe = category:<name>` (see [Benchmark & Predictor Universe (1.4)](../user_guide/data/benchmark.md#variable_universe)), it reads the group mapping from the loader's metadata; group labels follow the appendix naming verbatim so a recipe can say `category:nipa` and hit the NIPA subset.

## Transformation codes (T-codes)

FRED-QD uses the same seven-code table as FRED-MD, plus an optional eighth code introduced in the 2021 paper:

| T-code | Transform | Typical use |
|:---:|---|---|
| 1 | $x_t$ | Already stationary (e.g. unemployment rate when no unit root) |
| 2 | $\Delta x_t$ | First difference — rates |
| 3 | $\Delta^2 x_t$ | Second difference — rarely used outside specific price series |
| 4 | $\log x_t$ | Log levels — rarely a final choice, usually intermediate |
| 5 | $\Delta \log x_t$ | Log growth rate — most real activity / price series |
| 6 | $\Delta^2 \log x_t$ | Second-difference of logs — price acceleration |
| 7 | $\Delta (x_t / x_{t-1} - 1)$ | Change in growth rate |
| 8 | Special volatility transform | Market-volatility series where T-codes 1–7 don't produce stationarity per unit-root test |

The T-code for each series is declared in the CSV header (row 2) and in the appendix. macrocast reads and preserves them through the loader; applying the transform is the job of the `tcode_policy` axis under preprocessing (Layer 2, not yet user-doc-complete) — the Stage 1 loader intentionally does **not** apply T-codes itself, so the raw level data remains inspectable.

McCracken & Ng (2021) flag that their unit-root-test-driven T-code choice sometimes differs from prior literature; the appendix marks these cases.

## Vintage & revisions

FRED-QD ships one CSV per vintage. Three ways to pin:

1. **`current.csv`** — the latest published vintage. This is what `load_fred_qd()` fetches by default (`vintage=None`). Reproducible only if you record the vintage label from the returned `RawDatasetMetadata`.
2. **`YYYY-MM.csv`** — a labelled historical vintage, e.g. `2020-06.csv`. Each FRED-QD vintage typically follows the BEA advance GDP release; vintage labels align with the first-month-of-quarter naming the maintainers use.
3. **`local_source=<path>`** — bypass download and load a user-provided copy. Still populates the manifest.

How the vintage maps to `information_set_type`:

| `information_set_type` | Vintage choice | What the panel represents |
|---|---|---|
| `revised` *(default)* | `current.csv` | Post-revision levels — the "true" number as currently published. No information-leakage protection. Use for studies where revisions don't drive results. |
| `pseudo_oos_revised` | `current.csv` | Post-revision levels, but release-lag masking is applied downstream (`release_lag_rule` axis) so an origin at time *t* only sees data that would have been public at *t*. Not a real-time vintage — still uses today's revised numbers. |

FRED-QD does **not** support a true `real_time_vintage` regime in the same way FRED-MD does, because the quarterly vintage cadence is coarse (one vintage per quarter). If you need genuine quarterly real-time data, [FRED-SD](fred_sd.md) or the ALFRED vintage archive are better fits.

## Real-time release lag

FRED-QD's release-lag structure is coarser and more heterogeneous than FRED-MD's. Representative latencies (measured from quarter end to first public release):

| Series class | Typical first release | Notes |
|---|---|---|
| GDP components (NIPA) | ~30 days (advance) | Revised at second (+60) and third (+90) estimates |
| Industrial production | ~15 days | Consolidated from monthly FRED-MD |
| Employment | ~5 days after quarter end for last month | Quarterly average lag depends on month-within-quarter |
| Prices (CPI, PCE) | ~15 days | |
| Productivity / unit labour costs | ~35–40 days | Tied to GDP release |
| Corporate profits | ~60 days | Flow-of-funds dependent |
| Household balance sheets | ~75–90 days | Financial Accounts Z.1 release |
| Non-household balance sheets | ~75–90 days | Financial Accounts Z.1 release |
| Interest rates / stock / FX | same-quarter | Daily series aggregated to quarter-end |

macrocast's default `release_lag_rule` for FRED-QD is the same `fixed_lag_one_period` (one quarter) that FRED-MD uses — conservative for most series but too loose for interest-rate / stock-market series and too tight for balance-sheet series. See [Policies (1.5)](../user_guide/data/policies.md#release_lag_rule) for per-series overrides.

## Schema contract for `custom_csv` / `custom_parquet`

To pass as `dataset: fred_qd` via `dataset_source: custom_csv` (or `custom_parquet`), a user-provided file must match the FRED-QD schema:

- **First column**: `sasdate` (CSV) or `date` (Parquet). Quarter-end dates, parseable by pandas.
- **Row 1 (CSV) / first record (Parquet)**: T-codes per series, integer 1–8. Missing T-codes are treated as code 1 (no transform).
- **Columns 2+**: series IDs. Naming convention does **not** need to match FRED series IDs; macrocast treats them as opaque labels. If you want `variable_universe = category:<name>` to work, supply a side-car appendix mapping when loading.
- **Frequency**: quarterly. If the `frequency` axis disagrees with the dates in the file, the loader raises a schema error.
- **Time range**: at least one observation required. No minimum span enforced at load time — downstream `min_train_size` (Horizon, 1.3) will fail compatibly if you don't have enough.

See [Source & Frame (1.1) — Custom CSV / Parquet](../user_guide/data/source.md) for the axis-level mechanics.

## Known quirks / breaking changes

- **2021 paper vs. current vintage series count**: the paper reports 248 series. The current vintage count drifts slightly as the maintainers retire consolidated FRED IDs and substitute replacements. A recipe that pins `vintage: current` on a different day may get a slightly different column set — this is the point of the vintage label.
- **COVID outliers**: 2020:Q2 observations are extreme for most real-activity series (real GDP -31%, payrolls collapse). McCracken & Ng's published factor models used robust estimation; macrocast's default preprocessing treats these as normal observations. If your model is sensitive to outliers, consider the COVID-outlier handling options under Preprocessing (not yet user-doc-complete; see `plans/`).
- **T-code revisions**: the authors re-run unit-root tests periodically and may change a T-code between vintages. Bit-identical replication requires pinning the vintage label.
- **"fred-md" in the URL path is not a typo**: the St. Louis Fed hosts both FRED-MD and FRED-QD under the `fred-md/` path prefix. The `/quarterly/` subdirectory distinguishes FRED-QD from FRED-MD's `/monthly/`.
- **No per-variable metadata surface in v1.0**: macrocast does not ship an inline variable-description database. If your study needs human-readable series names, load the appendix PDF separately or use a side-car mapping.

## See also

- [FRED-MD](fred_md.md) — sister monthly database; same format, monthly frequency, 127 series.
- [FRED-SD](fred_sd.md) — state-level real-time panel.
- [Source & Frame (1.1)](../user_guide/data/source.md) — `dataset` / `dataset_source` / `frequency` / `information_set_type` axis semantics.
- [Benchmark & Predictor Universe (1.4)](../user_guide/data/benchmark.md) — `variable_universe` reads group labels from this page's §4.
- [Policies (1.5)](../user_guide/data/policies.md) — `release_lag_rule` per-series overrides.
