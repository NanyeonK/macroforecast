# FRED-QD

Quarterly U.S. macroeconomic panel maintained by the Federal Reserve Bank of St. Louis. Sister dataset to [FRED-MD](fred_md.md), designed for longer-horizon forecasting and for factor models requiring quarterly frequency (GDP, productivity, balance-sheet aggregates). Loaded via `macroforecast.load_fred_qd()` when `path.1_data_task.fixed_axes.dataset == "fred_qd"`.

## Citation & authoritative source

- **Original paper**: Michael W. McCracken and Serena Ng, "FRED-QD: A Quarterly Database for Macroeconomic Research," *Federal Reserve Bank of St. Louis Review* **103**(1): 1–44, 2021 (Q1). NBER working paper: [NBER WP 26872](https://www.nber.org/papers/w26872) (March 2020). St. Louis Fed WP: [2020-005](https://research.stlouisfed.org/wp/more/2020-005).
- **Official landing page**: [St. Louis Fed — FRED-MD & FRED-QD](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) (documentation, appendix, vintages — same page as FRED-MD).
- **Variable appendix (current)**: [`FRED-QD_updated_appendix.pdf`](https://research.stlouisfed.org/econ/mccracken/fred-databases/) — the authoritative list of every series, T-code, source FRED ID, and variable description. The package does not redistribute the appendix.

## What macroforecast downloads

- **Current vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-qd/quarterly/current.csv` (referenced in `macroforecast/raw/datasets/fred_qd.py`). Replaced by the maintainers at each quarter's advance release.
- **Historical vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-qd/quarterly/{vintage}.csv` with `vintage` as `YYYY-Q` (e.g., `2020-06`). Accessed when the recipe pins `information_set_type == "real_time_vintage"` plus `leaf_config.data_vintage`.

Same two-row header format as FRED-MD: row 1 holds the T-code per series, rows 2+ are quarterly observations starting 1959:Q1.

## Structure — 14 variable categories

FRED-QD is benchmarked to the 200-series quarterly panel of Stock and Watson (2012a), extended with 48 additional series (net: 248 series), with emphasis on non-household balance-sheet aggregates and credit data that the original Stock–Watson panel did not cover. The current panel is grouped into 14 categories; the group membership is maintained in the appendix and evolves over time.

Representative group coverage (exact membership: see appendix):

1. **NIPA** (National Income and Product Accounts) — real GDP (`GDPC1`), personal consumption expenditures (`PCECC96`), gross private investment, net exports, government spending.
2. **Industrial Production** — IP total and sectoral indices (`INDPRO`, sector-level from FRED-MD consolidated).
3. **Employment and Unemployment** — quarterly averages of payrolls, unemployment rate, participation rate, hours.
4. **Housing** — starts, permits, existing sales (quarterly averages plus quarterly-only series like owner-occupier equity).
5. **Inventories, Orders, and Sales** — manufacturing inventories, new orders, wholesale / retail sales.
6. **Prices** — CPI, PCE price, core variants, commodity price indices.
7. **Earnings and Productivity** — nonfarm-business labor productivity (`OPHNFB`), unit labor costs, average hourly earnings.
8. **Interest Rates** — federal funds, Treasury yields, credit spreads.
9. **Money and Credit** — monetary aggregates, bank reserves, total credit to the nonfinancial sector.
10. **Household Balance Sheets** — household net worth, financial obligations ratios.
11. **Exchange Rates** — trade-weighted dollar, bilateral major currencies.
12. **Stock Markets** — S&P 500, dividend yield, realised volatility proxies.
13. **Non-Household Balance Sheets** (new in FRED-QD) — corporate and non-corporate business liabilities, bank-sector balance sheets. This is the primary extension over the Stock–Watson 2012a panel.
14. **Other** — miscellaneous series the paper treats separately from the 13 numbered groups.

Exact category counts vary across vintages — St. Louis Fed tracks it in the appendix.

## Transformation codes (T-codes)

FRED-QD extends the FRED-MD T-code table with one additional entry (Case 8), reflecting a quarterly-specific transform pattern:

| T-code | Transform |
|:---:|---|
| 1 | No transformation |
| 2 | First difference $\\Delta x_t$ |
| 3 | Second difference $\\Delta^2 x_t$ |
| 4 | Natural logarithm $\\log x_t$ |
| 5 | First difference of logs $\\Delta \\log x_t$ |
| 6 | Second difference of logs $\\Delta^2 \\log x_t$ |
| 7 | First difference of percent change $\\Delta (x_t / x_{t-1} - 1)$ |
| **8** | **GARCH-type volatility transform** (for specific market-volatility series only) |

Case 8 is used sparingly — the paper notes that unit-root-test-driven T-code selection sometimes differs from codes used by prior literature, and the appendix flags these cases explicitly.

Flow into macroforecast is via the same `tcode_policy` axis as FRED-MD (1.1 does not override it).

## Changes from the 2020 working paper / 2021 Fed Review publication to current

The working paper (NBER WP 26872, March 2020) and the 2021 Fed Review publication documented **248 series**. Since then, the St. Louis Fed has maintained the panel; typical maintenance patterns are:

- **Series discontinuation** when underlying FRED IDs retire (e.g., when a BEA table or Fed H.8 breakdown consolidates). Such discontinuations are flagged in the appendix's history log; the composite series remains but its upstream mapping may move to a replacement FRED ID.
- **T-code revision** for individual series when a new unit-root test outcome contradicts the paper-era code. The authors' stated protocol is to revise T-codes based on ongoing unit-root test re-evaluation; revisions are documented in the appendix.
- **Additions** when a new Fed release adds a directly useful aggregate — rare but non-zero. The 14-category structure accommodates additions without a group rename.
- **Balance-sheet refinements** — the "Non-Household Balance Sheets" group was the working paper's main contribution over Stock–Watson (2012a); refinements to this group (e.g., updating a "banks vs non-banks" split) occur as Fed source data gets restructured.

As with FRED-MD, the authoritative change log lives in the appendix PDF; macroforecast does not mirror it. Bit-identical replication of a published study requires pinning the exact vintage via `leaf_config.data_vintage`.

## Loader behaviour

Mirror of FRED-MD:

- Caches download at `~/.cache/macroforecast/raw/`.
- Parsing via `parse_fred_csv` (same shared helper).
- No data redistribution. Network access or `local_source` path required on first load.
- Schema: date index (quarter-end) + numeric columns with FRED series IDs as column names.

## Known limitations in macroforecast v1.0

- Same as FRED-MD: no per-variable metadata surface, no auto T-code validation across vintages, `data_vintage` required for real-time_vintage mode.
- **Quarterly / monthly alignment**: when a study mixes FRED-QD with FRED-MD, aligning them requires the `alignment_rule` axis (1.5). v1.0 implements month-to-quarter aggregation (operational) but mixed-frequency evaluation is limited to single-frequency panels at each horizon.

## See also

- [FRED-MD](fred_md.md) — sister monthly database.
- [FRED-SD](fred_sd.md) — state-level real-time database.
- [Source & Frame (1.1)](../source.md) — dataset/frequency/information_set_type interactions.
