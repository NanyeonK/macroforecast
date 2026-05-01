# FRED-MD

Monthly U.S. macroeconomic panel maintained by the Federal Reserve Bank of St. Louis. The canonical monthly benchmark for empirical factor-model and macro-forecasting research since McCracken & Ng (2016). Updated at the start of every month; historical vintages archived back to the 2015 release.

## At a glance

| Field | Value |
|---|---|
| Frequency | Monthly (observations timestamped to month end) |
| Coverage start | 1959:01 |
| Coverage end | Current — refreshed at the start of each month by the maintainers |
| Panel size | ~127 series across 8 categories (current vintage); 134 in the 2015 paper |
| File format | Single CSV with two-row header |
| Maintainer | St. Louis Fed — Michael McCracken and team |
| Reference | McCracken & Ng (2016), *JBES* 34(4) |
| macrocast loader | `macrocast.load_fred_md()` |
| Recipe selector | `path.1_data_task.fixed_axes.dataset: fred_md` |

## Citation & authoritative source

- **Paper**: Michael W. McCracken and Serena Ng, "FRED-MD: A Monthly Database for Macroeconomic Research," *Journal of Business & Economic Statistics* **34**(4): 574–589, 2016.
- **Working paper**: [St. Louis Fed WP 2015-012](https://research.stlouisfed.org/wp/more/2015-012) (October 2015).
- **Landing page**: [St. Louis Fed — FRED-MD & FRED-QD](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) (shared with FRED-QD — documentation, appendix, historical vintages).
- **Variable appendix (current)**: `FRED-MD_updated_appendix.pdf` linked from the landing page. Authoritative list of every series, T-code, FRED source ID, and description. macrocast does not redistribute the appendix.

If you cite results produced with FRED-MD, cite McCracken & Ng (2016) and record the **vintage label** (see §6 below) so the exact panel can be reproduced.

## What macrocast downloads

- **Current vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv`
- **Historical vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/{vintage}.csv` where `vintage` is `YYYY-MM` (e.g., `2020-06`).
- **Historical zip**: the maintainers periodically publish a ZIP containing all past vintages. `load_fred_md(vintage="YYYY-MM", local_zip_source=<path-to-zip>)` extracts a specific vintage from the zip without a network call — unique to FRED-MD (FRED-QD / FRED-SD don't ship a vintage bundle).

All URLs are referenced verbatim in `macrocast.raw.datasets.fred_md` as `_CURRENT_URL` / `_VINTAGE_URL`.

CSV layout:

- **Row 1** (header): series IDs (first column literal `sasdate`).
- **Row 2**: T-code per series (first column blank / numeric placeholder — shared helper `parse_fred_csv` strips it into a separate `tcodes` record).
- **Rows 3+**: monthly observations, first column = month-end date.

Cache layout (see `macrocast.raw.cache.get_raw_file_path`):

```
~/.cache/macrocast/raw/fred_md/<version>.csv
```

Where `<version>` is `current` or `YYYY-MM`. First call downloads; subsequent calls hit cache. Pass `force=True` to `load_fred_md()` to re-download.

## Variable groups

McCracken & Ng (2016) organise the panel into **8 categories**. Membership is maintained in the appendix — the CSV itself has no category column. The 8-category structure has been stable since the 2016 publication (unlike FRED-QD, which grew to 14 categories in the 2021 paper with the addition of balance-sheet groups).

| # | Category | Focus | Representative series |
|:-:|---|---|---|
| 1 | **Output and Income** | Industrial production and real income aggregates | `INDPRO` (total IP), `IPFINAL` (final products), `IPMANSICS` (manufacturing), `IPDCONGD` (durable consumer goods), `RPI` (real personal income), `W875RX1` (real PI ex transfers), capacity utilization |
| 2 | **Labor Market** | Payrolls, hours, unemployment, earnings | `PAYEMS` (total nonfarm payrolls), `UNRATE` (civilian unemployment), `CE16OV` (civilian employment), `CLF16OV` (labour force), `AWHMAN` (avg weekly hours), `CES0600000008` (avg hourly earnings), `ICSA` (initial claims) |
| 3 | **Housing** | Starts, permits at national and regional level | `HOUST` (national housing starts), `PERMIT` (building permits), regional `HOUSTNE` / `HOUSTMW` / `HOUSTS` / `HOUSTW` and corresponding `PERMIT*` breakdowns |
| 4 | **Consumption, Orders, and Inventories** | Retail sales, new orders, sentiment | `DPCERA3M086SBEA` (real PCE), `RETAILx` (retail sales), `AMDMNOx` (new orders durable goods), `AMDMUOx` (unfilled orders), `BUSINVx` (business inventories), `UMCSENTx` (consumer sentiment) |
| 5 | **Money and Credit** | Monetary aggregates, bank credit, commercial paper | `M1SL`, `M2SL`, `M2REAL`, `TOTRESNS` (total reserves), `NONBORRES` (non-borrowed reserves), `BUSLOANS` (C&I loans), `REALLN` (real estate loans), `COMPAPFFx` (CP spread) |
| 6 | **Interest and Exchange Rates** | Yields, spreads, FX | `FEDFUNDS` (Fed funds), `TB3MS`/`TB6MS`/`GS1`/`GS5`/`GS10` (Treasury yields), `AAA`/`BAA` (corporate bonds), `AAAFFM`/`BAAFFM` (spreads vs Fed funds), `TWEXAFEGSMTHx` (trade-weighted USD), bilateral rates (JPY/USD, GBP/USD, CAD/USD, CHF/USD) |
| 7 | **Prices** | Price indices (CPI, PCE, PPI) and commodity prices | `CPIAUCSL` (headline CPI), `CPILFESL` (core CPI), `PCEPI` (PCE price index), `PPIACO` (all-commodity PPI), sectoral PPIs, `OILPRICEx` (WTI crude) |
| 8 | **Stock Market** | S&P 500 and equity aggregates | `S&P 500` (index level), `S&P: indust`, `S&P div yield`, `S&P PE ratio`, `VXOCLSx` (volatility) |

**Exact membership per group evolves across vintages** — a series may be added, removed, or moved between categories. The authoritative list is the appendix PDF; the representative members above are stable across the recent window (2016 paper through current).

When macrocast uses `variable_universe = category:<name>` (see [Benchmark & Predictor Universe (1.4)](../user_guide/data/benchmark.md#variable_universe)), it reads the group mapping from the loader's metadata; group labels follow the appendix naming so a recipe can say `category:prices` and hit the Prices subset.

## Transformation codes (T-codes)

FRED-MD uses the seven-code stationarity table from the 2016 paper:

| T-code | Transform | Typical use |
|:---:|---|---|
| 1 | $x_t$ | Already stationary (e.g. unemployment rate when no unit root) |
| 2 | $\Delta x_t$ | First difference — rates |
| 3 | $\Delta^2 x_t$ | Second difference — rare |
| 4 | $\log x_t$ | Log levels — usually intermediate, rarely final |
| 5 | $\Delta \log x_t$ | Log growth rate — most real activity / price series |
| 6 | $\Delta^2 \log x_t$ | Second-difference of logs — price acceleration |
| 7 | $\Delta (x_t / x_{t-1} - 1)$ | Change in growth rate |

(FRED-MD does not use T-code 8 — that's FRED-QD-only, for a specific volatility transform.)

The T-code for each series is declared in the CSV header (row 2) and in the appendix. macrocast reads and preserves them through the loader; applying the transform is the job of the `tcode_policy` axis under preprocessing (Layer 2, not yet user-doc-complete) — the Stage 1 loader intentionally does **not** apply T-codes itself, so the raw level data remains inspectable.

## Vintage & revisions

FRED-MD publishes one CSV per month. Three ways to pin a vintage:

1. **`current.csv`** — the latest published vintage. This is what `load_fred_md()` fetches by default (`vintage=None`). Reproducible only if you record the vintage label from the returned `RawDatasetMetadata`.
2. **`YYYY-MM.csv`** — a labelled historical vintage, e.g. `2020-06.csv`. Each FRED-MD vintage is named by the month the maintainers released it.
3. **`local_source=<path>`** or **`local_zip_source=<zip>`** — bypass download. `local_zip_source` extracts a specific vintage from the maintainers' historical zip bundle.

How the vintage maps to `information_set_type`:

| `information_set_type` | Vintage choice | What the panel represents |
|---|---|---|
| `revised` *(default)* | `current.csv` | Post-revision levels — the "true" number as currently published. No information-leakage protection. Use for studies where revisions don't drive results. |
| `pseudo_oos_revised` | `current.csv` | Post-revision levels, but release-lag masking is applied downstream (`release_lag_rule` axis) so an origin at time *t* only sees data that would have been public at *t*. Not a real-time vintage — still uses today's revised numbers. |
| `real_time_vintage` | `YYYY-MM.csv` (per the `leaf_config.data_vintage` field) | The panel *as published* at a specific month. Every value in the file reflects only revisions available up to that vintage date. |

FRED-MD is the only dataset in this section that supports genuine monthly real-time vintages via `real_time_vintage`: FRED-QD's quarterly cadence is too coarse, and FRED-SD has its own workbook-level vintage mechanism (see [FRED-SD](fred_sd.md)).

## Real-time release lag

Representative latencies, measured from month end to first public release:

| Series class | Typical first release | Notes |
|---|---|---|
| Industrial production | ~15 days (mid-month Friday) | `INDPRO` and sectoral breakdowns |
| Employment situation | ~3–7 days (first-Friday release) | Payrolls, unemployment, hours |
| Initial claims | ~4–5 days | Weekly data aggregated to monthly |
| Retail sales | ~15 days | |
| New orders (durable goods) | ~25 days | |
| Prices (CPI, PPI) | ~10–15 days | |
| Housing starts / permits | ~15–20 days | |
| Personal income / real PCE | ~30 days | |
| Monetary aggregates (M1, M2) | ~20 days | |
| Interest rates / FX / stock | same-month | Daily data aggregated to month end; no material lag |

macrocast's default `release_lag_rule` for FRED-MD is `fixed_lag_one_period` (one month) — conservative for real-activity series, zero-lag-realistic for financial series. Fine-grained per-series overrides live under [Policies (1.5)](../user_guide/data/policies.md#release_lag_rule).

## Schema contract for `custom_csv` / `custom_parquet`

To pass as `dataset: fred_md` via `dataset_source: custom_csv` (or `custom_parquet`), a user-provided file must match the FRED-MD schema:

- **First column**: `sasdate` (CSV) or `date` (Parquet). Month-end dates, parseable by pandas.
- **Row 1 (CSV) / first record (Parquet)**: T-codes per series, integer 1–7. Missing T-codes are treated as code 1 (no transform).
- **Columns 2+**: series IDs. Naming does **not** need to match FRED series IDs; macrocast treats them as opaque labels. If you want `variable_universe = category:<name>` to work, supply a side-car appendix mapping.
- **Frequency**: monthly. If the `frequency` axis disagrees with the dates in the file, the loader raises a schema error.
- **Time range**: at least one observation required. No minimum span enforced at load time — downstream `min_train_size` (Horizon, 1.3) will fail compatibly if you don't have enough.

See [Source & Frame (1.1) — Custom CSV / Parquet](../user_guide/data/source.md) for the axis-level mechanics.

## Known quirks / breaking changes

- **2015 paper vs. current vintage series count**: the paper reports 134 series; the current vintage sits around 127 after the maintainers retired some FRED IDs that Census / BLS / BEA have consolidated. A recipe that pins `vintage: current` on a different day may see slight column drift — this is the point of the vintage label.
- **COVID outliers**: April–June 2020 observations are extreme for most real-activity series (payrolls -21M month-over-month, claims spike, IP collapse). macrocast's default preprocessing treats these as normal observations. Robust alternatives live under Preprocessing (not yet user-doc-complete; see `plans/`).
- **T-code revisions**: St. Louis Fed periodically re-runs unit-root tests and may change a T-code between vintages. Bit-identical replication requires pinning the vintage label *and* the T-code column from that vintage.
- **Series with embedded spaces in the ID**: `S&P 500`, `S&P: indust`, `S&P div yield`, `S&P PE ratio` use literal spaces and `&` / `:` characters. macrocast treats them as opaque column names, but if you write a recipe that references them by name, you'll need to quote them.
- **Historical zip format is not stable**: the historical-vintages zip the maintainers ship has changed its internal file-naming convention at least once. `load_fred_md(local_zip_source=...)` handles both `{vintage}.csv` and `{vintage}-md.csv` names, but a third convention may require a loader patch.
- **No per-variable metadata surface in v1.0**: macrocast does not ship an inline variable-description database. Query FRED's REST API directly if you need units / descriptions / source URLs.

## See also

- [FRED-QD](fred_qd.md) — sister quarterly database; same format, quarterly frequency, 14 categories.
- [FRED-SD](fred_sd.md) — state-level real-time database.
- [Source & Frame (1.1)](../user_guide/data/source.md) — `dataset` / `dataset_source` / `frequency` / `information_set_type` axis semantics.
- [Benchmark & Predictor Universe (1.4)](../user_guide/data/benchmark.md) — `variable_universe` reads group labels from this page's §4.
- [Policies (1.5)](../user_guide/data/policies.md) — `release_lag_rule` per-series overrides.
