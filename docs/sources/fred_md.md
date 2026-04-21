# FRED-MD

Monthly U.S. macroeconomic panel maintained by the Federal Reserve Bank of St. Louis. Loaded via `macrocast.load_fred_md()` when `path.1_data_task.fixed_axes.dataset == "fred_md"`.

## Citation & authoritative source

- **Original paper**: Michael W. McCracken and Serena Ng, "FRED-MD: A Monthly Database for Macroeconomic Research," *Journal of Business & Economic Statistics* **34**(4): 574–589, 2016. Working paper: [Federal Reserve Bank of St. Louis WP 2015-012](https://research.stlouisfed.org/wp/more/2015-012).
- **Official landing page**: [St. Louis Fed — FRED-MD & FRED-QD](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) (current documentation, appendix, and historical vintages).
- **Variable appendix (current)**: [`FRED-MD_updated_appendix.pdf`](https://research.stlouisfed.org/econ/mccracken/fred-databases/) — authoritative list of every series, its T-code, and its source. **The macrocast package does not redistribute this appendix**; users who need the exact current variable list should fetch it from St. Louis Fed.

## What macrocast downloads

- **Current vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv` (this exact URL is used by `macrocast/raw/datasets/fred_md.py`). Replaced at the start of every month by the maintainers.
- **Historical vintage**: per-month CSVs at `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/{vintage}.csv` where `vintage` is e.g. `2020-06`. Accessed when the recipe sets `leaf_config.data_vintage` and `information_set_type == "real_time_vintage"`.
- **Bundle (historical)**: St. Louis Fed periodically publishes a ZIP of all past vintages. macrocast supports extraction from such a zip via the `local_zip_source` loader argument.

The CSV uses two header rows: the first is the transformation code (T-code) per series; subsequent rows are the observations indexed by month.

## Structure — 8 variable categories

The paper organises the panel into eight groups, unchanged since the 2016 publication:

1. **Output and income** — industrial production aggregates and sectoral indices (e.g., `INDPRO`, `IPFINAL`, `IPMANSICS`), real personal income (`RPI`).
2. **Labor market** — nonfarm payrolls (`PAYEMS`), unemployment rate (`UNRATE`), hours, earnings, initial claims.
3. **Housing** — housing starts and permits at national and regional level (`HOUST`, `PERMIT`).
4. **Consumption, orders, and inventories** — retail sales, new orders for durable goods, wholesale / retail inventories, consumer sentiment (`UMCSENTx`).
5. **Money and credit** — monetary aggregates (`M1SL`, `M2SL`), bank reserves, consumer and real-estate loans, commercial paper outstanding.
6. **Interest and exchange rates** — fed funds rate (`FEDFUNDS`), T-bill and Treasury yields across the curve (`GS1`, `GS5`, `GS10`), credit spreads, exchange rates against major currencies.
7. **Prices** — CPI (`CPIAUCSL`), PCE price index, PPI, commodity prices (oil, metals).
8. **Stock market** — S&P 500 (`S&P 500`), dividend yield, P/E ratio, aggregate market returns.

Exact membership of each group at any given point in time is in the appendix PDF — macrocast does not encode it. For code that needs category-aware feature grouping, the `feature_grouping` axis in Layer 2 / 3 will eventually surface a `fred_category` value (reserved for v1.1).

## Transformation codes (T-codes)

The first row of `current.csv` encodes the recommended stationarity transform for each series. From the 2016 paper, appendix Table 1:

| T-code | Transform |
|:---:|---|
| 1 | No transformation |
| 2 | First difference $\\Delta x_t$ |
| 3 | Second difference $\\Delta^2 x_t$ |
| 4 | Natural logarithm $\\log x_t$ |
| 5 | First difference of logs $\\Delta \\log x_t$ |
| 6 | Second difference of logs $\\Delta^2 \\log x_t$ |
| 7 | First difference of percent change $\\Delta (x_t / x_{t-1} - 1)$ |

In macrocast these codes flow into Layer 2 preprocessing via the `tcode_policy` axis:

- `tcode_policy: raw_only` → ignore T-codes, keep raw levels (default).
- `tcode_policy: apply_tcodes` → apply the CSV's per-series transform before downstream preprocessing.
- `tcode_policy: custom_override` → user-supplied per-series override.

## Changes from the 2015–2016 working paper to current

The 2015 working paper documented **134 series**. The current panel (circa 2024–2026) has evolved through monthly maintenance — the exact current count fluctuates because St. Louis Fed:

- **Drops series** when the underlying FRED ID is discontinued (e.g., an index whose source survey is retired). Example pattern: some housing-permits breakdowns were trimmed when the source Census tables consolidated.
- **Adds series** when FRED adds a directly comparable index or when a new sub-indicator becomes useful for factor estimation. Additions are rare and flagged in the appendix's change log.
- **Re-codes T-codes** when a series' stationarity profile visibly changes (e.g., a regime shift in a price index warranting a log-diff instead of log-level). Such changes are also flagged in the appendix.
- **Renames** source FRED IDs when the Fed updates its own taxonomy. The paper-era name remains in the first-row header for backward compatibility.

The authoritative change log is maintained by the St. Louis Fed in the appendix PDF (the "change history" section); macrocast does not attempt to mirror it. If a user needs bit-identical replication of a published study that cites FRED-MD, they should pin `information_set_type: real_time_vintage` + `leaf_config.data_vintage: "YYYY-MM"` where `YYYY-MM` is the vintage the study used.

## Loader behaviour — things to know

- **Download is cached** at `~/.cache/macrocast/raw/` (override with `cache_root` on the loader). The cache key is `(dataset, vintage, source_url)`.
- **No data redistribution** — the package never bundles the CSV. Network access or a user-provided `local_source` path is required on first load.
- **Parsing**: `parse_fred_csv` at `macrocast/raw/shared_csv.py` separates the T-code header row from the observation rows and returns both (T-codes surface only if Layer 2 preprocessing consumes them).
- **Schema conformance**: FRED-MD's column naming follows FRED series IDs (`INDPRO`, `CPIAUCSL`, …). Any user-side CSV used with `dataset_source: custom_csv` must match the same schema (date index + numeric columns named with FRED IDs) for the downstream pipeline to align.

## Known limitations in macrocast v1.0

- **No variable-level metadata surface** — the package does not expose each FRED ID's description / units / source URL. Users who want that enrichment should query FRED's REST API directly.
- **No automated T-code validation** — if St. Louis Fed changes a T-code in a new vintage, `tcode_policy: apply_tcodes` will use the new code silently. For strict reproducibility pin the vintage.
- **`data_vintage` required** for `information_set_type=real_time_vintage`; bare `fred_md` assumes `information_set_type=revised` (latest available revision).

## See also

- [FRED-QD](fred_qd.md) — sister quarterly database.
- [FRED-SD](fred_sd.md) — state-level real-time database.
- [Source & Frame (1.1)](../source.md) — how `dataset`, `dataset_source`, `frequency`, `information_set_type` interact at recipe compile time.
