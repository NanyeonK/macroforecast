# FRED-MD

[Back to FRED Datasets](index.md)

FRED-MD is the monthly national macroeconomic panel used by `macroforecast`.
It is loaded with `mf.data.load_fred_md()` and returned as a canonical
`DataBundle`.

## What This Dataset Is

| Item | Value |
| --- | --- |
| Dataset | FRED-MD |
| Native frequency | Monthly |
| Package loader | `macroforecast.data.load_fred_md()` |
| Package dataset id | `"fred_md"` |
| Package source family | `"fred-md"` |
| Current package output | `DataBundle(panel, metadata)` |
| Panel index | `DatetimeIndex` named `date` |
| Panel columns | FRED-MD series mnemonics |
| Official t-codes | Yes |
| Official groups | Yes, 8 numbered groups |
| Package default horizons | `(1, 3, 6, 12)` |

Use FRED-MD when the target, predictors, or evaluation unit are monthly. For
monthly state analysis with national controls, use `load_fred_md_sd()`.

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()
spec = mf.data.spec(bundle, target="INDPRO", horizons=[1, 3, 6, 12])
processed = mf.preprocessing.reprocess(spec)
```

## Official Sources

| Source | What it provides | URL |
| --- | --- | --- |
| FRED-MD/FRED-QD landing page | Current and vintage CSV links, appendix zip, Matlab code | <https://www.stlouisfed.org/research/economists/mccracken/fred-databases> |
| FRED-MD appendix zip | Official group labels, t-codes, descriptions, legacy comparable labels | linked from the landing page |
| FRED-Databases Matlab code | Reference preprocessing/factor code used by the St. Louis Fed distribution | linked from the landing page |
| FRED API `fred/series` | Per-series FRED metadata for direct FRED mnemonics, such as frequency and units | <https://fred.stlouisfed.org/docs/api/fred/series.html> |
| FRED API `fred/series/release` | Release metadata for direct FRED mnemonics | <https://fred.stlouisfed.org/docs/api/fred/series_release.html> |

The package does not reconstruct FRED-MD by calling the FRED API series by
series. It reads the official St. Louis Fed FRED-MD CSV, because that file is
the curated dataset contract. Some FRED-MD columns are adjusted or spliced
series, so users should not assume every FRED-MD mnemonic is a raw FRED API
series id.

## Current Snapshot Checked For This Page

This page's counts and catalog were checked against the St. Louis Fed landing
page and updated appendix on 2026-06-01.

| Item | Checked value |
| --- | --- |
| Landing-page current CSV label | `2026-04-md.csv` |
| Official data date range in that file | 1959-01 through 2026-03 |
| Official data rows | 807 monthly observations |
| Official series columns | 126 |
| Official appendix used for groups | `FRED-MD_updated_appendix.csv` from the appendix zip |
| Official appendix series count | 126 |

The package records the exact downloaded source in
`bundle.metadata["artifact"]["source_url"]`, so users can verify which raw file
produced any given run.

## Loader

```python
macroforecast.data.load_fred_md(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
    local_zip_source: str | pathlib.Path | None = None,
) -> DataBundle
```

## Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `vintage` | `str | None` | `None` | Vintage label in `YYYY-MM` form. `None` loads the package current source. |
| `force` | `bool` | `False` | Re-download or re-copy the raw file even if it already exists in cache. |
| `cache_root` | path-like or `None` | `None` | Root directory for raw-file cache and manifest. |
| `local_source` | path-like or `None` | `None` | Local CSV file to use instead of the online current/vintage CSV. |
| `local_zip_source` | path-like or `None` | `None` | Local historical zip file used to extract a vintage CSV. |

## Output

`load_fred_md()` returns a `DataBundle`.

| Field | Value |
| --- | --- |
| `bundle.panel` | Monthly canonical panel with `DatetimeIndex` named `date`. |
| `bundle.metadata["dataset"]` | `"fred_md"` |
| `bundle.metadata["source_family"]` | `"fred-md"` |
| `bundle.metadata["frequency"]` | `"monthly"` |
| `bundle.metadata["version_mode"]` | `"current"` or `"vintage"` |
| `bundle.metadata["vintage"]` | Requested vintage label, or `None` for current. |
| `bundle.metadata["data_through"]` | Last non-missing date parsed from the panel. |
| `bundle.metadata["artifact"]["source_url"]` | Exact URL or local path used by the loader. |
| `bundle.metadata["artifact"]["file_sha256"]` | SHA-256 hash of the raw file. |
| `bundle.metadata["transform_codes"]` | Official FRED-MD t-code map parsed from the CSV transform row. |
| `bundle.panel.attrs["macroforecast_transform_codes"]` | Same t-code map for pandas-native handoff. |

The loader appends the raw artifact metadata to the raw manifest when
`cache_root` is supplied.

## Frequency Contract

FRED-MD is monthly. The raw CSV date column is `sasdate`; the package parses it
to a pandas monthly `DatetimeIndex` named `date`.

Important consequences:

- A one-step horizon means one month ahead.
- Default horizons are `(1, 3, 6, 12)`.
- FRED-MD should be the default national panel for monthly forecasting.
- FRED-QD should be preferred for quarterly targets.
- If FRED-MD is combined into a quarterly panel, the package allows it but
  records a not-recommended parse note and frequency-conversion metadata.

## Official T-Codes

FRED-MD stores stationarity transform codes in the first row of the official
CSV. The package parses that row and stores it in
`metadata["transform_codes"]`.

| T-code | Formula | Meaning |
| ---: | --- | --- |
| 1 | `x_t` | Level, no transformation. |
| 2 | `x_t - x_{t-1}` | First difference. |
| 3 | `(x_t - x_{t-1}) - (x_{t-1} - x_{t-2})` | Second difference. |
| 4 | `log(x_t)` | Log level. |
| 5 | `log(x_t) - log(x_{t-1})` | First difference of log. |
| 6 | `(log(x_t) - log(x_{t-1})) - (log(x_{t-1}) - log(x_{t-2}))` | Second difference of log. |
| 7 | `(x_t / x_{t-1} - 1) - (x_{t-1} / x_{t-2} - 1)` | First difference of percent change. |

`mf.preprocessing.reprocess(..., transform="official")` applies these codes.
FRED-MD t-code 3 is part of the official codebook, but the current official
updated appendix has no FRED-MD series assigned to code 3.

## Group Summary

The official updated appendix has 126 series and 8 groups.

| Group | Name | Series count |
| ---: | --- | ---: |
| 1 | Output and income | 16 |
| 2 | Labor market | 31 |
| 3 | Housing | 10 |
| 4 | Consumption, orders, and inventories | 10 |
| 5 | Money and credit | 13 |
| 6 | Interest and exchange rates | 22 |
| 7 | Prices | 20 |
| 8 | Stock market | 4 |

## T-Code Summary

| T-code | Series count |
| ---: | ---: |
| 1 | 11 |
| 2 | 19 |
| 4 | 10 |
| 5 | 52 |
| 6 | 33 |
| 7 | 1 |

## T-Codes By Group

| Group | Name | Code 1 | Code 2 | Code 4 | Code 5 | Code 6 | Code 7 | Total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Output and income | 0 | 1 | 0 | 15 | 0 | 0 | 16 |
| 2 | Labor market | 2 | 5 | 0 | 21 | 3 | 0 | 31 |
| 3 | Housing | 0 | 0 | 10 | 0 | 0 | 0 | 10 |
| 4 | Consumption, orders, and inventories | 0 | 2 | 0 | 8 | 0 | 0 | 10 |
| 5 | Money and credit | 0 | 1 | 0 | 1 | 10 | 1 | 13 |
| 6 | Interest and exchange rates | 8 | 9 | 0 | 5 | 0 | 0 | 22 |
| 7 | Prices | 0 | 0 | 0 | 0 | 20 | 0 | 20 |
| 8 | Stock market | 1 | 1 | 0 | 2 | 0 | 0 | 4 |

## FRED API Source Boundary

FRED-MD is a curated dataset, not a direct one-call FRED API object.

For direct FRED series, users can query FRED API metadata with the series
mnemonic:

```text
https://api.stlouisfed.org/fred/series?series_id=INDPRO&api_key=...
https://api.stlouisfed.org/fred/series/release?series_id=INDPRO&api_key=...
```

Use this for per-series frequency, units, seasonal adjustment, and release
metadata. Use the FRED-MD CSV and appendix for package-level FRED-MD group and
t-code truth. If a FRED-MD mnemonic is adjusted, spliced, or otherwise curated,
the FRED API series metadata is not sufficient to reconstruct the FRED-MD
column.

## Full Series Coverage Catalog

This table joins two sources. `Group`, `T-code`, and `Description` come
from the official updated FRED-MD appendix. `Latest start`, `Latest end`,
and `Latest obs.` are computed directly from the landing-page current CSV
`2026-04-md.csv`. These coverage columns can change when a new
vintage is released, even if the official appendix table has not changed.

| ID | Series | Group | T-code | Latest start | Latest end | Latest obs. | Description |
| ---: | --- | --- | ---: | --- | --- | ---: | --- |
| 1 | `RPI` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | Real Personal Income |
| 2 | `W875RX1` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | Real personal income ex transfer receipts |
| 3 | `DPCERA3M086SBEA` | 4: Consumption, orders, and inventories | 5 | 1959-01 | 2026-03 | 807 | Real personal consumption expenditures |
| 4 | `CMRMTSPLx` | 4: Consumption, orders, and inventories | 5 | 1959-01 | 2026-02 | 806 | Real Manu.  and Trade Industries Sales |
| 5 | `RETAILx` | 4: Consumption, orders, and inventories | 5 | 1959-01 | 2026-03 | 807 | Retail and Food Services Sales |
| 6 | `INDPRO` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP Index |
| 7 | `IPFPNSS` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Final Products and Nonindustrial Supplies |
| 8 | `IPFINAL` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Final Products (Market Group) |
| 9 | `IPCONGD` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Consumer Goods |
| 10 | `IPDCONGD` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Durable Consumer Goods |
| 11 | `IPNCONGD` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Nondurable Consumer Goods |
| 12 | `IPBUSEQ` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Business Equipment |
| 13 | `IPMAT` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Materials |
| 14 | `IPDMAT` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Durable Materials |
| 15 | `IPNMAT` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Nondurable Materials |
| 16 | `IPMANSICS` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Manufacturing (SIC) |
| 17 | `IPB51222s` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Residential Utilities |
| 18 | `IPFUELS` | 1: Output and income | 5 | 1959-01 | 2026-03 | 807 | IP: Fuels |
| 20 | `CUMFNS` | 1: Output and income | 2 | 1959-01 | 2026-03 | 807 | Capacity Utilization:  Manufacturing |
| 21 | `HWI` | 2: Labor market | 2 | 1959-01 | 2026-02 | 806 | Help-Wanted Index for United States |
| 22 | `HWIURATIO` | 2: Labor market | 2 | 1959-01 | 2026-02 | 805 | Ratio of Help Wanted/No.  Unemployed |
| 23 | `CLF16OV` | 2: Labor market | 5 | 1959-01 | 2026-03 | 806 | Civilian Labor Force |
| 24 | `CE16OV` | 2: Labor market | 5 | 1959-01 | 2026-03 | 806 | Civilian Employment |
| 25 | `UNRATE` | 2: Labor market | 2 | 1959-01 | 2026-03 | 806 | Civilian Unemployment Rate |
| 26 | `UEMPMEAN` | 2: Labor market | 2 | 1959-01 | 2026-03 | 806 | Average Duration of Unemployment (Weeks) |
| 27 | `UEMPLT5` | 2: Labor market | 5 | 1959-01 | 2026-03 | 806 | Civilians Unemployed - Less Than 5 Weeks |
| 28 | `UEMP5TO14` | 2: Labor market | 5 | 1959-01 | 2026-03 | 806 | Civilians Unemployed for 5-14 Weeks |
| 29 | `UEMP15OV` | 2: Labor market | 5 | 1959-01 | 2026-03 | 806 | Civilians Unemployed - 15 Weeks & Over |
| 30 | `UEMP15T26` | 2: Labor market | 5 | 1959-01 | 2026-03 | 806 | Civilians Unemployed for 15-26 Weeks |
| 31 | `UEMP27OV` | 2: Labor market | 5 | 1959-01 | 2026-03 | 806 | Civilians Unemployed for 27 Weeks and Over |
| 32 | `CLAIMSx` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | Initial Claims |
| 33 | `PAYEMS` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Total nonfarm |
| 34 | `USGOOD` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Goods-Producing Industries |
| 35 | `CES1021000001` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Mining and Logging:  Mining |
| 36 | `USCONS` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Construction |
| 37 | `MANEMP` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Manufacturing |
| 38 | `DMANEMP` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Durable goods |
| 39 | `NDMANEMP` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Nondurable goods |
| 40 | `SRVPRD` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Service-Providing Industries |
| 41 | `USTPU` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Trade, Transportation & Utilities |
| 42 | `USWTRADE` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Wholesale Trade |
| 43 | `USTRADE` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Retail Trade |
| 44 | `USFIRE` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Financial Activities |
| 45 | `USGOVT` | 2: Labor market | 5 | 1959-01 | 2026-03 | 807 | All Employees:  Government |
| 46 | `CES0600000007` | 2: Labor market | 1 | 1959-01 | 2026-03 | 807 | Avg Weekly Hours :  Goods-Producing |
| 47 | `AWOTMAN` | 2: Labor market | 2 | 1959-01 | 2026-03 | 807 | Avg Weekly Overtime Hours :  Manufacturing |
| 48 | `AWHMAN` | 2: Labor market | 1 | 1959-01 | 2026-03 | 807 | Avg Weekly Hours :  Manufacturing |
| 50 | `HOUST` | 3: Housing | 4 | 1959-01 | 2026-03 | 807 | Housing Starts:  Total New Privately Owned |
| 51 | `HOUSTNE` | 3: Housing | 4 | 1959-01 | 2026-03 | 807 | Housing Starts, Northeast |
| 52 | `HOUSTMW` | 3: Housing | 4 | 1959-01 | 2026-03 | 807 | Housing Starts, Midwest |
| 53 | `HOUSTS` | 3: Housing | 4 | 1959-01 | 2026-03 | 807 | Housing Starts, South |
| 54 | `HOUSTW` | 3: Housing | 4 | 1959-01 | 2026-03 | 807 | Housing Starts, West |
| 55 | `PERMIT` | 3: Housing | 4 | 1960-01 | 2026-03 | 795 | New Private Housing Permits (SAAR) |
| 56 | `PERMITNE` | 3: Housing | 4 | 1960-01 | 2026-03 | 795 | New Private Housing Permits, Northeast (SAAR) |
| 57 | `PERMITMW` | 3: Housing | 4 | 1960-01 | 2026-03 | 795 | New Private Housing Permits, Midwest (SAAR) |
| 58 | `PERMITS` | 3: Housing | 4 | 1960-01 | 2026-03 | 795 | New Private Housing Permits, South (SAAR) |
| 59 | `PERMITW` | 3: Housing | 4 | 1960-01 | 2026-03 | 795 | New Private Housing Permits, West (SAAR) |
| 64 | `ACOGNO` | 4: Consumption, orders, and inventories | 5 | 1992-02 | 2026-02 | 409 | New Orders for Consumer Goods |
| 65 | `AMDMNOx` | 4: Consumption, orders, and inventories | 5 | 1959-01 | 2026-03 | 807 | New Orders for Durable Goods |
| 66 | `ANDENOx` | 4: Consumption, orders, and inventories | 5 | 1968-02 | 2026-03 | 698 | New Orders for Nondefense Capital Goods |
| 67 | `AMDMUOx` | 4: Consumption, orders, and inventories | 5 | 1959-01 | 2026-03 | 807 | Unfilled Orders for Durable Goods |
| 68 | `BUSINVx` | 4: Consumption, orders, and inventories | 5 | 1959-01 | 2026-02 | 806 | Total Business Inventories |
| 69 | `ISRATIOx` | 4: Consumption, orders, and inventories | 2 | 1959-01 | 2026-02 | 806 | Total Business:  Inventories to Sales Ratio |
| 70 | `M1SL` | 5: Money and credit | 6 | 1959-01 | 2026-03 | 807 | M1 Money Stock |
| 71 | `M2SL` | 5: Money and credit | 6 | 1959-01 | 2026-03 | 807 | M2 Money Stock |
| 72 | `M2REAL` | 5: Money and credit | 5 | 1959-01 | 2026-03 | 806 | Real M2 Money Stock |
| 73 | `BOGMBASE` | 5: Money and credit | 6 | 1959-01 | 2026-03 | 807 | Monetary Base |
| 74 | `TOTRESNS` | 5: Money and credit | 6 | 1959-01 | 2026-03 | 807 | Total Reserves of Depository Institutions |
| 75 | `NONBORRES` | 5: Money and credit | 7 | 1959-01 | 2026-03 | 807 | Reserves Of Depository Institutions |
| 76 | `BUSLOANS` | 5: Money and credit | 6 | 1959-01 | 2026-03 | 807 | Commercial and Industrial Loans |
| 77 | `REALLN` | 5: Money and credit | 6 | 1959-01 | 2026-03 | 807 | Real Estate Loans at All Commercial Banks |
| 78 | `NONREVSL` | 5: Money and credit | 6 | 1959-01 | 2026-02 | 806 | Total Nonrevolving Credit |
| 79 | `CONSPI` | 5: Money and credit | 2 | 1959-01 | 2026-02 | 806 | Nonrevolving consumer credit to Personal Income |
| 80 | `S&P 500` | 8: Stock market | 5 | 1959-01 | 2026-03 | 807 | S&P's Common Stock Price Index: Composite |
| 82 | `S&P div yield` | 8: Stock market | 2 | 1959-01 | 2026-03 | 807 | S&P's Composite Common Stock: Dividend Yield |
| 83 | `S&P PE ratio` | 8: Stock market | 5 | 1959-01 | 2026-03 | 807 | S&P's Composite Common Stock: Price-Earnings Ratio |
| 84 | `FEDFUNDS` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 807 | Effective Federal Funds Rate |
| 85 | `CP3Mx` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 806 | 3-Month AA Financial Commercial Paper Rate |
| 86 | `TB3MS` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 807 | 3-Month Treasury Bill: |
| 87 | `TB6MS` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 807 | 6-Month Treasury Bill: |
| 88 | `GS1` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 807 | 1-Year Treasury Rate |
| 89 | `GS5` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 807 | 5-Year Treasury Rate |
| 90 | `GS10` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 807 | 10-Year Treasury Rate |
| 91 | `AAA` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 807 | Moody's Seasoned Aaa Corporate Bond Yield |
| 92 | `BAA` | 6: Interest and exchange rates | 2 | 1959-01 | 2026-03 | 807 | Moody's Seasoned Baa Corporate Bond Yield |
| 93 | `COMPAPFFx` | 6: Interest and exchange rates | 1 | 1959-01 | 2026-03 | 806 | 3-Month Commercial Paper Minus FEDFUNDS |
| 94 | `TB3SMFFM` | 6: Interest and exchange rates | 1 | 1959-01 | 2026-03 | 807 | 3-Month Treasury C Minus FEDFUNDS |
| 95 | `TB6SMFFM` | 6: Interest and exchange rates | 1 | 1959-01 | 2026-03 | 807 | 6-Month Treasury C Minus FEDFUNDS |
| 96 | `T1YFFM` | 6: Interest and exchange rates | 1 | 1959-01 | 2026-03 | 807 | 1-Year Treasury C Minus FEDFUNDS |
| 97 | `T5YFFM` | 6: Interest and exchange rates | 1 | 1959-01 | 2026-03 | 807 | 5-Year Treasury C Minus FEDFUNDS |
| 98 | `T10YFFM` | 6: Interest and exchange rates | 1 | 1959-01 | 2026-03 | 807 | 10-Year Treasury C Minus FEDFUNDS |
| 99 | `AAAFFM` | 6: Interest and exchange rates | 1 | 1959-01 | 2026-03 | 807 | Moody's Aaa Corporate Bond Minus FEDFUNDS |
| 100 | `BAAFFM` | 6: Interest and exchange rates | 1 | 1959-01 | 2026-03 | 807 | Moody's Baa Corporate Bond Minus FEDFUNDS |
| 101 | `TWEXAFEGSMTHx` | 6: Interest and exchange rates | 5 | 1973-01 | 2026-03 | 639 | Trade Weighted U.S. Dollar Index |
| 102 | `EXSZUSx` | 6: Interest and exchange rates | 5 | 1959-01 | 2026-03 | 807 | Switzerland / U.S. Foreign Exchange Rate |
| 103 | `EXJPUSx` | 6: Interest and exchange rates | 5 | 1959-01 | 2026-03 | 807 | Japan / U.S. Foreign Exchange Rate |
| 104 | `EXUSUKx` | 6: Interest and exchange rates | 5 | 1959-01 | 2026-03 | 807 | U.S. / U.K. Foreign Exchange Rate |
| 105 | `EXCAUSx` | 6: Interest and exchange rates | 5 | 1959-01 | 2026-03 | 807 | Canada / U.S. Foreign Exchange Rate |
| 106 | `WPSFD49207` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | PPI: Finished Goods |
| 107 | `WPSFD49502` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | PPI: Finished Consumer Goods |
| 108 | `WPSID61` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | PPI: Intermediate Materials |
| 109 | `WPSID62` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | PPI: Crude Materials |
| 110 | `OILPRICEx` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | Crude Oil, spliced WTI and Cushing |
| 111 | `PPICMM` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | PPI: Metals and metal products: |
| 113 | `CPIAUCSL` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : All Items |
| 114 | `CPIAPPSL` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : Apparel |
| 115 | `CPITRNSL` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : Transportation |
| 116 | `CPIMEDSL` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : Medical Care |
| 117 | `CUSR0000SAC` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : Commodities |
| 118 | `CUSR0000SAD` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : Durables |
| 119 | `CUSR0000SAS` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : Services |
| 120 | `CPIULFSL` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : All Items Less Food |
| 121 | `CUSR0000SA0L2` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : All items less shelter |
| 122 | `CUSR0000SA0L5` | 7: Prices | 6 | 1959-01 | 2026-03 | 806 | CPI : All items less medical care |
| 123 | `PCEPI` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | Personal Cons.  Expend.:  Chain Index |
| 124 | `DDURRG3M086SBEA` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | Personal Cons.  Exp:  Durable goods |
| 125 | `DNDGRG3M086SBEA` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | Personal Cons.  Exp:  Nondurable goods |
| 126 | `DSERRG3M086SBEA` | 7: Prices | 6 | 1959-01 | 2026-03 | 807 | Personal Cons.  Exp:  Services |
| 127 | `CES0600000008` | 2: Labor market | 6 | 1959-01 | 2026-03 | 807 | Avg Hourly Earnings :  Goods-Producing |
| 128 | `CES2000000008` | 2: Labor market | 6 | 1959-01 | 2026-03 | 807 | Avg Hourly Earnings :  Construction |
| 129 | `CES3000000008` | 2: Labor market | 6 | 1959-01 | 2026-03 | 807 | Avg Hourly Earnings :  Manufacturing |
| 130 | `UMCSENTx` | 4: Consumption, orders, and inventories | 2 | 1959-05 | 2026-03 | 653 | Consumer Sentiment Index |
| 132 | `DTCOLNVHFNM` | 5: Money and credit | 6 | 1959-01 | 2026-02 | 806 | Consumer Motor Vehicle Loans Outstanding |
| 133 | `DTCTHFNM` | 5: Money and credit | 6 | 1959-01 | 2026-02 | 806 | Total Consumer Loans and Leases Outstanding |
| 134 | `INVEST` | 5: Money and credit | 6 | 1959-01 | 2026-03 | 807 | Securities in Bank Credit at All Commercial Banks |
| 135 | `VIXCLSx` | 8: Stock market | 1 | 1962-07 | 2026-03 | 765 | VIX |
## Combined With FRED-SD

`load_fred_md_sd()` loads FRED-MD and FRED-SD and combines them.

```python
bundle = mf.data.load_fred_md_sd(
    states=["CA", "TX"],
    variables=["UR", "ICLAIMS", "NQGSP"],
    frequency="monthly",
)
```

FRED-SD includes both monthly and quarterly state series. If a selected FRED-SD
series is quarterly, the default rule
`quarterly_to_monthly="repeat_within_quarter"` assigns the quarterly value to
each month in that quarter. The function emits `UserWarning` and records the
conversion in `metadata["frequency_conversion_warnings"]`.
