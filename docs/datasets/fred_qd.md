# FRED-QD

[Back to FRED Datasets](index.md)

FRED-QD is the quarterly national macroeconomic panel used by `macroforecast`.
It is loaded with `mf.data.load_fred_qd()` and returned as a canonical
`DataBundle`.

## What This Dataset Is

| Item | Value |
| --- | --- |
| Dataset | FRED-QD |
| Native frequency | Quarterly |
| Package loader | `macroforecast.data.load_fred_qd()` |
| Metadata dataset | `"fred_qd"` |
| Current package output | `DataBundle(panel, metadata)` |
| Panel index | `DatetimeIndex` named `date` |
| Panel columns | FRED-QD series mnemonics |
| Official t-codes | Yes |
| Official groups | Yes, 14 numbered groups |
| Raw factor row | Yes, first row named `factors` |
| Package default horizons | `(1, 2, 4, 8)` |

Use FRED-QD when the target, outcome, or evaluation unit is quarterly. For
quarterly state analysis with national controls, use `load_fred_qd_sd()`.

```python
import macroforecast as mf

bundle = mf.data.load_fred_qd()
spec = mf.data.spec(bundle, target="GDPC1", horizons=[1, 2, 4, 8])
processed = mf.preprocessing.reprocess(spec)
```

## Official Sources

| Source | What it provides | URL |
| --- | --- | --- |
| FRED-MD/FRED-QD landing page | Current and vintage CSV links, appendix zip, Matlab code | <https://www.stlouisfed.org/research/economists/mccracken/fred-databases> |
| FRED-QD article | Dataset motivation, quarterly design, and construction notes | <https://www.stlouisfed.org/publications/review/2021/01/14/fred-qd-a-quarterly-database-for-macroeconomic-research> |
| FRED-QD appendix zip | Official group labels, t-codes, SW factor flags, descriptions, and SW mnemonics | linked from the landing page |
| FRED-Databases Matlab code | Reference preprocessing/factor code used by the St. Louis Fed distribution | linked from the landing page |
| FRED API `fred/series` | Per-series FRED metadata for direct FRED mnemonics, such as frequency and units | <https://fred.stlouisfed.org/docs/api/fred/series.html> |
| FRED API `fred/series/release` | Release metadata for direct FRED mnemonics | <https://fred.stlouisfed.org/docs/api/fred/series_release.html> |

The package does not reconstruct FRED-QD by calling the FRED API series by
series. It reads the official St. Louis Fed FRED-QD CSV, because that file is
the curated quarterly dataset contract. Some FRED-QD columns ending in `x` are
modified variants, not raw FRED API series.

## Current Snapshot Checked For This Page

This page was checked against the St. Louis Fed landing page and updated
appendix on 2026-06-01.

| Item | Checked value |
| --- | --- |
| Landing-page current CSV label | `2026-04-qd.csv` |
| Official data date range in that file | 1959-03 through 2026-03 |
| Official data rows | 269 quarterly observations |
| Official series columns | 245 |
| Official appendix used for groups | `FRED-QD_updated_appendix.csv` from the appendix zip |
| Official appendix series count | 245 |
| Latest factor row counts | `{0: 120, 1: 125}` |
| Latest t-code row counts | `{1: 22, 2: 32, 5: 140, 6: 50, 7: 1}` |

The latest CSV factor row and the updated appendix factor flag are not fully
identical. Difference found in this check: `S&P 500` appendix=0, latest=1.
The package currently parses official t-codes from the raw CSV; this page
reports both appendix and latest factor flags for auditability.

## Loader

```python
macroforecast.data.load_fred_qd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
) -> DataBundle
```

## Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `vintage` | `str | None` | `None` | Vintage label in `YYYY-MM` form. `None` loads the package current source. |
| `force` | `bool` | `False` | Re-download or re-copy the raw file even if it already exists in cache. |
| `cache_root` | path-like or `None` | `None` | Root directory for raw-file cache and manifest. |
| `local_source` | path-like or `None` | `None` | Local CSV file to use instead of the online current/vintage CSV. |

## Output

`load_fred_qd()` returns a `DataBundle`.

| Field | Value |
| --- | --- |
| `bundle.panel` | Quarterly canonical panel with `DatetimeIndex` named `date`. |
| `bundle.metadata["dataset"]` | `"fred_qd"` |
| `bundle.metadata["frequency"]` | `"quarterly"` |
| `bundle.metadata["version_mode"]` | `"current"` or `"vintage"` |
| `bundle.metadata["vintage"]` | Requested vintage label, or `None` for current. |
| `bundle.metadata["data_through"]` | Last non-missing date parsed from the panel. |
| `bundle.metadata["artifact"]["source_url"]` | Exact URL or local path used by the loader. |
| `bundle.metadata["artifact"]["file_sha256"]` | SHA-256 hash of the raw file. |
| `bundle.metadata["transform_codes"]` | Official FRED-QD t-code map parsed from the CSV transform row. |
| `bundle.panel.attrs["macroforecast_transform_codes"]` | Same t-code map for pandas-native handoff. |

The loader appends the raw artifact metadata to the raw manifest when
`cache_root` is supplied.

## Frequency Contract

FRED-QD is quarterly. The raw CSV date column is `sasdate`; the current
St. Louis Fed file labels quarters by the first day of the quarter-ending
month, such as `3/1`, `6/1`, `9/1`, and `12/1`. The package parses those
labels to a pandas `DatetimeIndex` named `date`.

Important consequences:

- A one-step horizon means one quarter ahead.
- Default horizons are `(1, 2, 4, 8)`.
- FRED-QD should be the default national panel for quarterly forecasting.
- FRED-MD should be preferred for monthly targets.
- If FRED-QD is combined into a monthly panel, the package allows it but
  records a not-recommended parse note and frequency-conversion metadata.

## Official T-Codes

FRED-QD stores factor flags in the first row and stationarity transform codes
in the second row of the official CSV. The package parses the `transform` row
and stores it in `metadata["transform_codes"]`.

| T-code | Formula | Meaning |
| ---: | --- | --- |
| 1 | `x_t` | Level, no transformation. |
| 2 | `x_t - x_{t-1}` | First difference. |
| 3 | `(x_t - x_{t-1}) - (x_{t-1} - x_{t-2})` | Second difference. |
| 4 | `log(x_t)` | Log level. |
| 5 | `log(x_t) - log(x_{t-1})` | First difference of log. |
| 6 | `(log(x_t) - log(x_{t-1})) - (log(x_{t-1}) - log(x_{t-2}))` | Second difference of log. |
| 7 | `(x_t / x_{t-1} - 1) - (x_{t-1} / x_{t-2} - 1)` | First difference of percent change. |

The updated appendix includes code 3 and code 4 in the official codebook, but
the latest checked FRED-QD CSV has no series assigned to either code. Latest
t-code counts are `{1: 22, 2: 32, 5: 140, 6: 50, 7: 1}`.

## Group Summary

The official updated appendix has 245 series and 14 groups.

| Group | Name | Series count |
| ---: | --- | ---: |
| 1 | NIPA | 23 |
| 2 | Industrial Production | 16 |
| 3 | Employment and Unemployment | 50 |
| 4 | Housing | 14 |
| 5 | Inventories, Orders, and Sales | 9 |
| 6 | Prices | 48 |
| 7 | Earnings and Productivity | 14 |
| 8 | Interest Rates | 20 |
| 9 | Money and Credit | 15 |
| 10 | Household Balance Sheets | 9 |
| 11 | Exchange Rates | 6 |
| 12 | Other | 2 |
| 13 | Stock Markets | 6 |
| 14 | Non-Household Balance Sheets | 13 |

## T-Code Summary

| T-code | Series count |
| ---: | ---: |
| 1 | 22 |
| 2 | 32 |
| 5 | 140 |
| 6 | 50 |
| 7 | 1 |

## T-Codes By Group

| Group | Name | Code 1 | Code 2 | Code 5 | Code 6 | Code 7 | Total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | NIPA | 2 | 2 | 19 | 0 | 0 | 23 |
| 2 | Industrial Production | 2 | 0 | 14 | 0 | 0 | 16 |
| 3 | Employment and Unemployment | 2 | 12 | 36 | 0 | 0 | 50 |
| 4 | Housing | 0 | 0 | 14 | 0 | 0 | 14 |
| 5 | Inventories, Orders, and Sales | 0 | 1 | 8 | 0 | 0 | 9 |
| 6 | Prices | 0 | 0 | 3 | 45 | 0 | 48 |
| 7 | Earnings and Productivity | 0 | 0 | 13 | 1 | 0 | 14 |
| 8 | Interest Rates | 10 | 10 | 0 | 0 | 0 | 20 |
| 9 | Money and Credit | 1 | 0 | 9 | 4 | 1 | 15 |
| 10 | Household Balance Sheets | 1 | 1 | 7 | 0 | 0 | 9 |
| 11 | Exchange Rates | 0 | 0 | 6 | 0 | 0 | 6 |
| 12 | Other | 1 | 1 | 0 | 0 | 0 | 2 |
| 13 | Stock Markets | 1 | 1 | 4 | 0 | 0 | 6 |
| 14 | Non-Household Balance Sheets | 2 | 4 | 7 | 0 | 0 | 13 |

## FRED API Source Boundary

FRED-QD is a curated quarterly dataset, not a direct one-call FRED API object.
For direct FRED series, users can query FRED API metadata with the series
mnemonic:

```text
https://api.stlouisfed.org/fred/series?series_id=GDPC1&api_key=...
https://api.stlouisfed.org/fred/series/release?series_id=GDPC1&api_key=...
```

Use the FRED API for per-series frequency, units, seasonal adjustment, and
release metadata. Use the FRED-QD CSV and appendix for package-level FRED-QD
group, t-code, factor-row, and quarterly coverage truth. If a FRED-QD mnemonic
ends in `x`, the appendix treats it as a modified variant rather than a direct
FRED pull.

## Full Series Coverage Catalog

This table joins two sources. `Group`, `T-code`, `Appendix factor`, `SW
mnemonic`, and `Description` come from the official updated FRED-QD appendix.
`Latest factor`, `Latest start`, `Latest end`, and `Latest obs.` are computed
directly from the landing-page current CSV `2026-04-qd.csv`. These
coverage columns can change when a new vintage is released, even if the
official appendix table has not changed.

| ID | Series | Group | T-code | Latest factor | Appendix factor | Latest start | Latest end | Latest obs. | SW mnemonic | Description |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 1 | `GDPC1` | 1: NIPA | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | GDP | Real Gross Domestic Product, 3 Decimal (Billions of Chained 2017 Dollars) |
| 2 | `PCECC96` | 1: NIPA | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Consumption | Real Personal Consumption Expenditures (Billions of Chained 2017 Dollars) |
| 3 | `PCDGx` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Cons:Dur | Real personal consumption expenditures: Durable goods (Billions of Chained 2017 Dollars), deflated using its own price index |
| 4 | `PCESVx` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Cons:Svc | Real Personal Consumption Expenditures: Services (Billions of 2017 Dollars), deflated using its own price index |
| 5 | `PCNDx` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Cons:NonDur | Real Personal Consumption Expenditures: Nondurable Goods (Billions of 2017 Dollars), deflated using its own price index |
| 6 | `GPDIC1` | 1: NIPA | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Investment | Real Gross Private Domestic Investment, 3 decimal (Billions of Chained 2017 Dollars) |
| 7 | `FPIx` | 1: NIPA | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | FixedInv | Real private fixed investment (Billions of Chained 2017 Dollars), deflated using its own price index |
| 8 | `Y033RC1Q027SBEAx` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Inv:Equip&Software | Real Gross Private Domestic Investment: Fixed Investment: Nonresidential: Equipment (Billions of Chained 2017 Dollars), deflated using its own price index |
| 9 | `PNFIx` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | FixInv:NonRes | Real private fixed investment: Nonresidential (Billions of Chained 2017 Dollars), deflated using its own price index |
| 10 | `PRFIx` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | FixedInv:Res | Real private fixed investment: Residential (Billions of Chained 2017 Dollars), deflated using its own price index |
| 11 | `A014RE1Q156NBEA` | 1: NIPA | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | Inv:Inventories | Shares of gross domestic product: Gross private domestic investment: Change in private inventories (Percent) |
| 12 | `GCEC1` | 1: NIPA | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Gov.Spending | Real Government Consumption Expenditures & Gross Investment (Billions of Chained 2017 Dollars) |
| 13 | `A823RL1Q225SBEA` | 1: NIPA | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | Gov:Fed | Real Government Consumption Expenditures and Gross Investment: Federal (Percent Change from Preceding Period) |
| 14 | `FGRECPTx` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | Real Gov Receipts | Real Federal Government Current Receipts (Billions of Chained 2017 Dollars), deflated using its own price index |
| 15 | `SLCEx` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Gov:State&Local | Real government state and local consumption expenditures (Billions of Chained 2017 Dollars), deflated using its own price index |
| 16 | `EXPGSC1` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Exports | Real Exports of Goods & Services, 3 Decimal (Billions of Chained 2017 Dollars) |
| 17 | `IMPGSC1` | 1: NIPA | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Imports | Real Imports of Goods & Services, 3 Decimal (Billions of Chained 2017 Dollars) |
| 18 | `DPIC96` | 1: NIPA | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Disp-Income | Real Disposable Personal Income (Billions of Chained 2017 Dollars) |
| 19 | `OUTNFB` | 1: NIPA | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 | Ouput:NFB | Nonfarm Business Sector: Real Output (Index 2017=100) |
| 20 | `OUTBS` | 1: NIPA | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 | Output:Bus | Business Sector: Real Output (Index 2017=100) |
| 21 | `OUTMS` | 1: NIPA | 5 | 0 | 0 | 1987-03 | 2025-12 | 156 | Output:Manuf | Manufacturing Sector: Real Output (Index 2017=100) |
| 22 | `INDPRO` | 2: Industrial Production | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | IP:Total index | Industrial Production Index (Index 2017=100) |
| 23 | `IPFINAL` | 2: Industrial Production | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | IP:Final products | Industrial Production: Final Products (Market Group) (Index 2017=100) |
| 24 | `IPCONGD` | 2: Industrial Production | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | IP:Consumer goods | Industrial Production: Consumer Goods (Index 2017=100) |
| 25 | `IPMAT` | 2: Industrial Production | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | IP:Materials | Industrial Production: Materials (Index 2017=100) |
| 26 | `IPDMAT` | 2: Industrial Production | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | IP:Dur gds materials | Industrial Production: Durable Materials (Index 2017=100) |
| 27 | `IPNMAT` | 2: Industrial Production | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | IP:Nondur gds materials | Industrial Production: Nondurable Materials (Index 2017=100) |
| 28 | `IPDCONGD` | 2: Industrial Production | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | IP:Dur Cons. Goods | Industrial Production: Durable Consumer Goods (Index 2017=100) |
| 29 | `IPB51110SQ` | 2: Industrial Production | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | IP:Auto  | Industrial Production: Durable Goods: Automotive products (Index 2017=100) |
| 30 | `IPNCONGD` | 2: Industrial Production | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | IP:NonDur Cons God | Industrial Production: Nondurable Consumer Goods (Index 2017=100) |
| 31 | `IPBUSEQ` | 2: Industrial Production | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | IP:Bus Equip | Industrial Production: Business Equipment (Index 2017=100) |
| 32 | `IPB51220SQ` | 2: Industrial Production | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | IP:Energy Prds | Industrial Production: Consumer energy products (Index 2017=100) |
| 33 | `TCU` | 2: Industrial Production | 1 | 1 | 1 | 1967-03 | 2026-03 | 237 | Capu Tot | Capacity Utilization: Total Industry (Percent of Capacity) |
| 34 | `CUMFNS` | 2: Industrial Production | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | Capu Man. | Capacity Utilization: Manufacturing (SIC) (Percent of Capacity) |
| 35 | `PAYEMS` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Emp:Nonfarm | All Employees: Total nonfarm (Thousands of Persons) |
| 36 | `USPRIV` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Emp:Private | All Employees: Total Private Industries (Thousands of Persons) |
| 37 | `MANEMP` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Emp:mfg | All Employees: Manufacturing (Thousands of Persons) |
| 38 | `SRVPRD` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Emp:Services | All Employees: Service-Providing Industries (Thousands of Persons) |
| 39 | `USGOOD` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Emp:Goods | All Employees: Goods-Producing Industries (Thousands of Persons) |
| 40 | `DMANEMP` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:DurGoods | All Employees: Durable goods (Thousands of Persons) |
| 41 | `NDMANEMP` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Emp:Nondur Goods | All Employees: Nondurable goods (Thousands of Persons) |
| 42 | `USCONS` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Const | All Employees: Construction (Thousands of Persons) |
| 43 | `USEHS` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Edu&Health | All Employees: Education & Health Services (Thousands of Persons) |
| 44 | `USFIRE` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Finance | All Employees: Financial Activities (Thousands of Persons) |
| 45 | `USINFO` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Infor | All Employees: Information Services (Thousands of Persons) |
| 46 | `USPBS` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Bus Serv | All Employees: Professional & Business Services (Thousands of Persons) |
| 47 | `USLAH` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Leisure | All Employees: Leisure & Hospitality (Thousands of Persons) |
| 48 | `USSERV` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:OtherSvcs | All Employees: Other Services (Thousands of Persons) |
| 49 | `USMINE` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Mining/NatRes | All Employees: Mining and logging (Thousands of Persons) |
| 50 | `USTPU` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Trade&Trans | All Employees: Trade, Transportation & Utilities (Thousands of Persons) |
| 51 | `USGOVT` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Emp:Gov | All Employees: Government (Thousands of Persons) |
| 52 | `USTRADE` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Retail | All Employees: Retail Trade (Thousands of Persons) |
| 53 | `USWTRADE` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Wholesal | All Employees: Wholesale Trade (Thousands of Persons) |
| 54 | `CES9091000001` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Gov(Fed) | All Employees: Government: Federal (Thousands of Persons) |
| 55 | `CES9092000001` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Gov (State) | All Employees: Government: State Government (Thousands of Persons) |
| 56 | `CES9093000001` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:Gov (Local) | All Employees: Government: Local Government (Thousands of Persons) |
| 57 | `CE16OV` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Emp:Total (HHSurve) | Civilian Employment (Thousands of Persons) |
| 58 | `CIVPART` | 3: Employment and Unemployment | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | LF Part Rate | Civilian Labor Force Participation Rate (Percent) |
| 59 | `UNRATE` | 3: Employment and Unemployment | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | Unemp Rate | Civilian Unemployment Rate (Percent) |
| 60 | `UNRATESTx` | 3: Employment and Unemployment | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | Urate_ST | Unemployment Rate less than 27 weeks (Percent) |
| 61 | `UNRATELTx` | 3: Employment and Unemployment | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | Urate_LT | Unemployment Rate for more than 27 weeks (Percent) |
| 62 | `LNS14000012` | 3: Employment and Unemployment | 2 | 1 | 1 | 1959-03 | 2026-03 | 269 | Urate:Age16-19 | Unemployment Rate - 16 to 19 years (Percent) |
| 63 | `LNS14000025` | 3: Employment and Unemployment | 2 | 1 | 1 | 1959-03 | 2026-03 | 269 | Urate:Age>20 Men | Unemployment Rate - 20 years and over, Men (Percent) |
| 64 | `LNS14000026` | 3: Employment and Unemployment | 2 | 1 | 1 | 1959-03 | 2026-03 | 269 | Urate:Age>20 Women | Unemployment Rate - 20 years and over, Women (Percent) |
| 65 | `UEMPLT5` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | U:Dur<5wks | Number of Civilians Unemployed - Less Than 5 Weeks (Thousands of Persons) |
| 66 | `UEMP5TO14` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | U:Dur5-14wks | Number of Civilians Unemployed for 5 to 14 Weeks (Thousands of Persons) |
| 67 | `UEMP15T26` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | U:dur>15-26wks | Number of Civilians Unemployed for 15 to 26 Weeks (Thousands of Persons) |
| 68 | `UEMP27OV` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | U:Dur>27wks | Number of Civilians Unemployed for 27 Weeks and Over (Thousands of Persons) |
| 69 | `LNS13023621` | 3: Employment and Unemployment | 5 | 1 | 1 | 1967-03 | 2026-03 | 237 | U:Job losers | Unemployment Level - Job Losers (Thousands of Persons) |
| 70 | `LNS13023557` | 3: Employment and Unemployment | 5 | 1 | 1 | 1967-03 | 2026-03 | 237 | U:LF Reenty | Unemployment Level - Reentrants to Labor Force (Thousands of Persons) |
| 71 | `LNS13023705` | 3: Employment and Unemployment | 5 | 1 | 1 | 1967-03 | 2026-03 | 237 | U:Job Leavers | Unemployment Level - Job Leavers (Thousands of Persons) |
| 72 | `LNS13023569` | 3: Employment and Unemployment | 5 | 1 | 1 | 1967-03 | 2026-03 | 237 | U:New Entrants | Unemployment Level - New Entrants (Thousands of Persons) |
| 73 | `LNS12032194` | 3: Employment and Unemployment | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Emp:SlackWk | Employment Level - Part-Time for Economic Reasons, All Industries (Thousands of Persons) |
| 74 | `HOABS` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 | EmpHrs:Bus Sec | Business Sector: Hours of All Persons (Index 2017=100) |
| 75 | `HOAMS` | 3: Employment and Unemployment | 5 | 0 | 0 | 1987-03 | 2025-12 | 156 | EmpHrs:mfg | Manufacturing Sector: Hours of All Persons (Index 2017=100) |
| 76 | `HOANBS` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 | EmpHrs:nfb | Nonfarm Business Sector: Hours of All Persons (Index 2017=100) |
| 77 | `AWHMAN` | 3: Employment and Unemployment | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | AWH Man | Average Weekly Hours of Production and Nonsupervisory Employees: Manufacturing (Hours) |
| 78 | `AWHNONAG` | 3: Employment and Unemployment | 2 | 1 | 1 | 1964-03 | 2026-03 | 249 | AWH Privat | Average Weekly Hours Of Production And Nonsupervisory Employees: Total private (Hours) |
| 79 | `AWOTMAN` | 3: Employment and Unemployment | 2 | 1 | 1 | 1959-03 | 2026-03 | 269 | AWH Overtime | Average Weekly Overtime Hours of Production and Nonsupervisory Employees: Manufacturing (Hours) |
| 80 | `HWIx` | 3: Employment and Unemployment | 1 | 0 | 0 | 1959-03 | 2026-03 | 269 | HelpWnted | Help-Wanted Index |
| 81 | `HOUST` | 4: Housing | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Hstarts | Housing Starts: Total: New Privately Owned Housing Units Started (Thousands of Units) |
| 82 | `HOUST5F` | 4: Housing | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Hstarts >5units | Privately Owned Housing Starts: 5-Unit Structures or More (Thousands of Units) |
| 83 | `PERMIT` | 4: Housing | 5 | 1 | 1 | 1960-03 | 2026-03 | 265 | Hpermits | New Private Housing Units Authorized by Building Permits (Thousands of Units) |
| 84 | `HOUSTMW` | 4: Housing | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Hstarts:MW | Housing Starts in Midwest Census Region (Thousands of Units) |
| 85 | `HOUSTNE` | 4: Housing | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Hstarts:NE | Housing Starts in Northeast Census Region (Thousands of Units) |
| 86 | `HOUSTS` | 4: Housing | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Hstarts:S | Housing Starts in South Census Region (Thousands of Units) |
| 87 | `HOUSTW` | 4: Housing | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Hstarts:W | Housing Starts in West Census Region (Thousands of Units) |
| 88 | `CMRMTSPLx` | 5: Inventories, Orders, and Sales | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | MT Sales | Real Manufacturing and Trade Industries Sales (Millions of Chained 2017 Dollars) |
| 89 | `RSAFSx` | 5: Inventories, Orders, and Sales | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Ret. Sale | Real Retail and Food Services Sales (Millions of Chained 2017 Dollars), deflated by Core PCE |
| 90 | `AMDMNOx` | 5: Inventories, Orders, and Sales | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Orders (DurMfg) | Real Manufacturers' New Orders: Durable Goods (Millions of 2017 Dollars), deflated by Core PCE |
| 91 | `ACOGNOx` | 5: Inventories, Orders, and Sales | 5 | 1 | 1 | 1992-03 | 2026-03 | 137 | Orders(ConsumerGoods/Mat.) | Real Value of Manufacturers' New Orders for Consumer Goods Industries (Million of 2017 Dollars), deflated by Core PCE  |
| 92 | `AMDMUOx` | 5: Inventories, Orders, and Sales | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | UnfOrders(DurGds) | Real Value of Manufacturers' Unfilled Orders for Durable Goods Industries (Million of 2017 Dollars), deflated by Core PCE  |
| 93 | `ANDENOx` | 5: Inventories, Orders, and Sales | 5 | 1 | 1 | 1968-03 | 2026-03 | 233 | Orders(NonDefCap) | Real Value of Manufacturers' New Orders for Capital Goods: Nondefense Capital Goods Industries (Million of 2017 Dollars), deflated by Core PCE  |
| 94 | `INVCQRMTSPL` | 5: Inventories, Orders, and Sales | 5 | 1 | 1 | 1967-03 | 2025-12 | 236 | MT Invent | Real Manufacturing and Trade Inventories (Millions of 2017 Dollars) |
| 95 | `PCECTPI` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PCED | Personal Consumption Expenditures: Chain-type Price Index (Index 2017=100) |
| 96 | `PCEPILFE` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PCED_LFE | Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index) (Index 2017=100) |
| 97 | `GDPCTPI` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | GDP Defl | Gross Domestic Product: Chain-type Price Index (Index 2017=100) |
| 98 | `GPDICTPI` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | GPDI Defl | Gross Private Domestic Investment: Chain-type Price Index (Index 2017=100) |
| 99 | `IPDBS` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2025-12 | 268 | BusSec Defl | Business Sector: Implicit Price Deflator (Index 2017=100) |
| 100 | `DGDSRG3Q086SBEA` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PCED_Goods | Personal consumption expenditures: Goods (chain-type price index) |
| 101 | `DDURRG3Q086SBEA` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PCED_DurGoods | Personal consumption expenditures: Durable goods (chain-type price index) |
| 102 | `DSERRG3Q086SBEA` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PCED_Serv | Personal consumption expenditures: Services (chain-type price index) |
| 103 | `DNDGRG3Q086SBEA` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PCED_NDurGoods | Personal consumption expenditures: Nondurable goods (chain-type price index) |
| 104 | `DHCERG3Q086SBEA` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PCED_HouseholdServ. | Personal consumption expenditures: Services: Household consumption expenditures (chain-type price index) |
| 105 | `DMOTRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_MotorVec | Personal consumption expenditures: Durable goods: Motor vehicles and parts (chain-type price index) |
| 106 | `DFDHRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_DurHousehold | Personal consumption expenditures: Durable goods: Furnishings and durable household equipment (chain-type price index) |
| 107 | `DREQRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_Recreation | Personal consumption expenditures: Durable goods: Recreational goods and vehicles (chain-type price index) |
| 108 | `DODGRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_OthDurGds | Personal consumption expenditures: Durable goods: Other durable goods (chain-type price index) |
| 109 | `DFXARG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_Food_Bev | Personal consumption expenditures: Nondurable goods: Food and beverages purchased for off-premises consumption (chain-type price index) |
| 110 | `DCLORG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_Clothing | Personal consumption expenditures: Nondurable goods: Clothing and footwear (chain-type price index) |
| 111 | `DGOERG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_Gas_Enrgy | Personal consumption expenditures: Nondurable goods: Gasoline and other energy goods (chain-type price index) |
| 112 | `DONGRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_OthNDurGds | Personal consumption expenditures: Nondurable goods: Other nondurable goods (chain-type price index) |
| 113 | `DHUTRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_Housing-Utilities | Personal consumption expenditures: Services: Housing and utilities (chain-type price index) |
| 114 | `DHLCRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_HealthCare | Personal consumption expenditures: Services: Health care (chain-type price index) |
| 115 | `DTRSRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_TransSvg | Personal consumption expenditures: Transportation services (chain-type price index) |
| 116 | `DRCARG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_RecServices | Personal consumption expenditures: Recreation services (chain-type price index) |
| 117 | `DFSARG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_FoodServ_Acc. | Personal consumption expenditures: Services: Food services and accommodations (chain-type price index) |
| 118 | `DIFSRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_FIRE | Personal consumption expenditures: Financial services and insurance (chain-type price index) |
| 119 | `DOTSRG3Q086SBEA` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PCED_OtherServices | Personal consumption expenditures: Other services (chain-type price index) |
| 120 | `CPIAUCSL` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | CPI | Consumer Price Index for All Urban Consumers: All Items (Index 1982-84=100) |
| 121 | `CPILFESL` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | CPI_LFE | Consumer Price Index for All Urban Consumers: All Items Less Food & Energy (Index 1982-84=100) |
| 122 | `WPSFD49207` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PPI:FinGds | Producer Price Index by Commodity for Finished Goods (Index 1982=100) |
| 123 | `PPIACO` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 | PPI | Producer Price Index for All Commodities (Index 1982=100) |
| 124 | `WPSFD49502` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PPI:FinConsGds | Producer Price Index by Commodity for Finished Consumer Goods (Index 1982=100) |
| 125 | `WPSFD4111` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PPI:FinConsGds(Food) | Producer Price Index by Commodity for Finished Consumer Foods (Index 1982=100) |
| 126 | `PPIIDC` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PPI:IndCom | Producer Price Index by Commodity Industrial Commodities (Index 1982=100) |
| 127 | `WPSID61` | 6: Prices | 6 | 1 | 1 | 1959-03 | 2026-03 | 269 | PPI:IntMat | Producer Price Index by Commodity Intermediate Materials: Supplies & Components (Index 1982=100) |
| 128 | `WPU0531` | 6: Prices | 5 | 1 | 1 | 1967-03 | 2026-03 | 237 | Real Price:NatGas | Producer Price Index by Commodity for Fuels and Related Products and Power: Natural Gas (Index 1982=100) |
| 129 | `WPU0561` | 6: Prices | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Real Price:Oil | Producer Price Index by Commodity for Fuels and Related Products and Power: Crude Petroleum (Domestic Production) (Index 1982=100) |
| 130 | `OILPRICEx` | 6: Prices | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Real Crudeoil Price | Real Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma (2017 Dollars per Barrel), deflated by Core PCE |
| 131 | `AHETPIx` | 7: Earnings and Productivity | 5 | 0 | 0 | 1964-03 | 2026-03 | 249 | Real AHE:PrivInd | Real Average Hourly Earnings of Production and Nonsupervisory Employees: Total Private (2017 Dollars per Hour), deflated by Core PCE |
| 132 | `CES2000000008x` | 7: Earnings and Productivity | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Real AHE:Const | Real Average Hourly Earnings of Production and Nonsupervisory Employees: Construction (2017 Dollars per Hour), deflated by Core PCE |
| 133 | `CES3000000008x` | 7: Earnings and Productivity | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Real AHE:MFG | Real Average Hourly Earnings of Production and Nonsupervisory Employees: Manufacturing (2017 Dollars per Hour), deflated by Core PCE |
| 134 | `COMPRMS` | 7: Earnings and Productivity | 5 | 1 | 1 | 1987-03 | 2025-12 | 156 | CPH:Mfg | Manufacturing Sector: Real Compensation Per Hour (Index 2017=100) |
| 135 | `COMPRNFB` | 7: Earnings and Productivity | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | CPH:NFB | Nonfarm Business Sector: Real Compensation Per Hour (Index 2017=100) |
| 136 | `RCPHBS` | 7: Earnings and Productivity | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | CPH:Bus | Business Sector: Real Compensation Per Hour (Index 2017=100) |
| 137 | `OPHMFG` | 7: Earnings and Productivity | 5 | 1 | 1 | 1987-03 | 2025-12 | 156 | OPH:mfg | Manufacturing Sector: Real Output Per Hour of All Persons (Index 2017=100) |
| 138 | `OPHNFB` | 7: Earnings and Productivity | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | OPH:nfb | Nonfarm Business Sector: Real Output Per Hour of All Persons (Index 2017=100) |
| 139 | `OPHPBS` | 7: Earnings and Productivity | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 | OPH:Bus | Business Sector: Real Output Per Hour of All Persons (Index 2017=100) |
| 140 | `ULCBS` | 7: Earnings and Productivity | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 | ULC:Bus | Business Sector: Unit Labor Cost (Index 2017=100) |
| 141 | `ULCMFG` | 7: Earnings and Productivity | 5 | 1 | 1 | 1987-03 | 2025-12 | 156 | ULC:Mfg | Manufacturing Sector: Unit Labor Cost (Index 2017=100) |
| 142 | `ULCNFB` | 7: Earnings and Productivity | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | ULC:NFB | Nonfarm Business Sector: Unit Labor Cost (Index 2017=100) |
| 143 | `UNLPNBS` | 7: Earnings and Productivity | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | UNLPay:nfb | Nonfarm Business Sector: Unit Nonlabor Payments (Index 2017=100) |
| 144 | `FEDFUNDS` | 8: Interest Rates | 2 | 1 | 1 | 1959-03 | 2026-03 | 269 | FedFunds | Effective Federal Funds Rate (Percent) |
| 145 | `TB3MS` | 8: Interest Rates | 2 | 1 | 1 | 1959-03 | 2026-03 | 269 | TB-3Mth | 3-Month Treasury Bill: Secondary Market Rate (Percent) |
| 146 | `TB6MS` | 8: Interest Rates | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | TM-6MTH | 6-Month Treasury Bill: Secondary Market Rate (Percent) |
| 147 | `GS1` | 8: Interest Rates | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | TB-1YR | 1-Year Treasury Constant Maturity Rate (Percent) |
| 148 | `GS10` | 8: Interest Rates | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | TB-10YR | 10-Year Treasury Constant Maturity Rate (Percent) |
| 149 | `MORTGAGE30US` | 8: Interest Rates | 2 | 0 | 0 | 1971-06 | 2026-03 | 220 | Mort-30Yr | 30-Year Conventional Mortgage Rate (Percent) |
| 150 | `AAA` | 8: Interest Rates | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | AAA Bond | Moody's Seasoned Aaa Corporate Bond Yield (Percent) |
| 151 | `BAA` | 8: Interest Rates | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 | BAA Bond | Moody's Seasoned Baa Corporate Bond Yield (Percent) |
| 152 | `BAA10YM` | 8: Interest Rates | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | BAA_GS10 | Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity (Percent) |
| 153 | `MORTG10YRx` | 8: Interest Rates | 1 | 1 | 1 | 1971-06 | 2026-03 | 220 | MRTG_GS10 | 30-Year Conventional Mortgage Rate Relative to 10-Year Treasury Constant Maturity (Percent) |
| 154 | `TB6M3Mx` | 8: Interest Rates | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | tb6m_tb3m | 6-Month Treasury Bill Minus 3-Month Treasury Bill, secondary market (Percent) |
| 155 | `GS1TB3Mx` | 8: Interest Rates | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | GS1_tb3m | 1-Year Treasury Constant Maturity Minus 3-Month Treasury Bill, secondary market (Percent) |
| 156 | `GS10TB3Mx` | 8: Interest Rates | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | GS10_tb3m | 10-Year Treasury Constant Maturity Minus 3-Month Treasury Bill, secondary market (Percent) |
| 157 | `CPF3MTB3Mx` | 8: Interest Rates | 1 | 1 | 1 | 1959-03 | 2026-03 | 269 | CP_Tbill Spread | 3-Month Commercial Paper Minus 3-Month Treasury Bill, secondary market (Percent) |
| 158 | `BOGMBASEREALx` | 9: Money and Credit | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Real Mbase | St. Louis Adjusted Monetary Base (Billions of 1982-84 Dollars), deflated by CPI |
| 160 | `M1REAL` | 9: Money and Credit | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Real m1 | Real M1 Money Stock (Billions of 1982-84 Dollars), deflated by CPI |
| 161 | `M2REAL` | 9: Money and Credit | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Real m2 | Real M2 Money Stock (Billions of 1982-84 Dollars), deflated by CPI |
| 163 | `BUSLOANSx` | 9: Money and Credit | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Real C&Lloand | Real Commercial and Industrial Loans, All Commercial Banks (Billions of 2017 U.S. Dollars), deflated by Core PCE  |
| 164 | `CONSUMERx` | 9: Money and Credit | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Real ConsLoans | Real Consumer Loans at All Commercial Banks (Billions of 2017 U.S. Dollars), deflated by Core PCE  |
| 165 | `NONREVSLx` | 9: Money and Credit | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Real NonRevCredit | Total Real Nonrevolving Credit Owned and Securitized, Outstanding (Billions of Dollars), deflated by Core PCE  |
| 166 | `REALLNx` | 9: Money and Credit | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Real LoansRealEst | Real Real Estate Loans, All Commercial Banks (Billions of 2017 U.S. Dollars), deflated by Core PCE  |
| 167 | `REVOLSLx` | 9: Money and Credit | 5 | 1 | 1 | 1968-03 | 2026-03 | 233 | Real RevolvCredit | Total Real Revolving Credit Owned and Securitized, Outstanding (Billions of 2017 Dollars), deflated by Core PCE  |
| 168 | `TOTALSLx` | 9: Money and Credit | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 | Real ConsuCred | Total Consumer Credit Outstanding, deflated by Core PCE  |
| 169 | `DRIWCIL` | 9: Money and Credit | 1 | 1 | 1 | 1982-06 | 2026-03 | 176 | FRBSLO_Consumers | FRB Senior Loans Officer Opions. Net Percentage of Domestic Respondents Reporting Increased Willingness to Make Consumer Installment Loans |
| 170 | `TABSHNOx` | 10: Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 | Real HHW:TASA | Real Total Assets of Households and Nonprofit Organizations (Billions of 2017 Dollars), deflated by Core PCE |
| 171 | `TLBSHNOx` | 10: Household Balance Sheets | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | Real HHW:LiabSA | Real Total Liabilities of Households and Nonprofit Organizations (Billions of 2017 Dollars), deflated by Core PCE |
| 172 | `LIABPIx` | 10: Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 | liab_PDISA | Liabilities of Households and Nonprofit Organizations Relative to Personal Disposable Income (Percent) |
| 173 | `TNWBSHNOx` | 10: Household Balance Sheets | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | Real HHW:WSA | Real Net Worth of Households and Nonprofit Organizations (Billions of 2017 Dollars), deflated by Core PCE |
| 174 | `NWPIx` | 10: Household Balance Sheets | 1 | 0 | 0 | 1959-03 | 2025-12 | 268 | W_PDISA | Net Worth of Households and Nonprofit Organizations Relative to  Disposable Personal Income (Percent) |
| 175 | `TARESAx` | 10: Household Balance Sheets | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | Real HHW:TA_RESA | Real Assets of Households and Nonprofit Organizations excluding Real Estate Assets (Billions of 2017 Dollars), deflated by Core PCE |
| 176 | `HNOREMQ027Sx` | 10: Household Balance Sheets | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | Real HHW:RESA | Real Real Estate Assets of Households and Nonprofit Organizations (Billions of 2017 Dollars), deflated by Core PCE |
| 177 | `TFAABSHNOx` | 10: Household Balance Sheets | 5 | 1 | 1 | 1959-03 | 2025-12 | 268 | Real HHW:FinSA | Real Total Financial Assets of Households and Nonprofit Organizations (Billions of 2017 Dollars), deflated by Core PCE |
| 178 | `VIXCLSx` | 13: Stock Markets | 1 | 1 | 1 | 1962-09 | 2026-03 | 255 | VIX | CBOE Volatility Index: VIX |
| 179 | `USSTHPI` | 4: Housing | 5 | 1 | 1 | 1975-03 | 2025-12 | 204 | Real Hprice:OFHEO | All-Transactions House Price Index for the United States (Index 1980 Q1=100) |
| 180 | `SPCS10RSA` | 4: Housing | 5 | 1 | 1 | 1987-03 | 2026-03 | 157 | Real CS_10 | S&P/Case-Shiller 10-City Composite Home Price Index (Index January 2000 = 100) |
| 181 | `SPCS20RSA` | 4: Housing | 5 | 1 | 1 | 2000-03 | 2026-03 | 105 | Real CS_20 | S&P/Case-Shiller 20-City Composite Home Price Index (Index January 2000 = 100) |
| 182 | `TWEXAFEGSMTHx` | 11: Exchange Rates | 5 | 1 | 1 | 1973-03 | 2026-03 | 213 | Ex rate:major | Trade Weighted U.S. Dollar Index: Major Currencies (Index March 1973=100) |
| 183 | `EXUSEU` | 11: Exchange Rates | 5 | 1 | 1 | 1999-03 | 2026-03 | 109 | Ex rate:Euro | U.S. / Euro Foreign Exchange Rate (U.S. Dollars to One Euro) |
| 184 | `EXSZUSx` | 11: Exchange Rates | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Ex rate:Switz | Switzerland / U.S. Foreign Exchange Rate |
| 185 | `EXJPUSx` | 11: Exchange Rates | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Ex rate:Japan | Japan / U.S. Foreign Exchange Rate |
| 186 | `EXUSUKx` | 11: Exchange Rates | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | Ex rate:UK | U.S. / U.K. Foreign Exchange Rate |
| 187 | `EXCAUSx` | 11: Exchange Rates | 5 | 1 | 1 | 1959-03 | 2026-03 | 269 | EX rate:Canada | Canada / U.S. Foreign Exchange Rate |
| 188 | `UMCSENTx` | 12: Other | 1 | 1 | 1 | 1959-06 | 2026-03 | 267 | Cons. Expectations | University of Michigan: Consumer Sentiment (Index 1st Quarter 1966=100) |
| 189 | `USEPUINDXM` | 12: Other | 2 | 1 | 1 | 1985-03 | 2026-03 | 165 | PoilcyUncertainty | Economic Policy Uncertainty Index for United States |
| 190 | `B020RE1Q156NBEA` | 1: NIPA | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Shares of gross domestic product: Exports of goods and services (Percent) |
| 191 | `B021RE1Q156NBEA` | 1: NIPA | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Shares of gross domestic product: Imports of goods and services (Percent) |
| 192 | `GFDEGDQ188S` | 14: Non-Household Balance Sheets | 2 | 0 | 0 | 1966-03 | 2025-12 | 240 |  | Federal Debt: Total Public Debt as Percent of GDP (Percent) |
| 193 | `GFDEBTNx` | 14: Non-Household Balance Sheets | 2 | 0 | 0 | 1966-03 | 2025-12 | 240 |  | Real Federal Debt: Total Public Debt (Millions of 2017 Dollars), deflated by PCE |
| 194 | `IPMANSICS` | 2: Industrial Production | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Industrial Production: Manufacturing (SIC) (Index 2017=100) |
| 195 | `IPB51222S` | 2: Industrial Production | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Industrial Production: Residential Utilities (Index 2017=100) |
| 196 | `IPFUELS` | 2: Industrial Production | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Industrial Production: Fuels (Index 2017=100) |
| 197 | `UEMPMEAN` | 3: Employment and Unemployment | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Average (Mean) Duration of Unemployment (Weeks) |
| 198 | `CES0600000007` | 3: Employment and Unemployment | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Average Weekly Hours of Production and Nonsupervisory Employees: Goods-Producing |
| 199 | `TOTRESNS` | 9: Money and Credit | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Total Reserves of Depository Institutions (Billions of Dollars) |
| 200 | `NONBORRES` | 9: Money and Credit | 7 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Reserves Of Depository Institutions, Nonborrowed  (Millions of Dollars) |
| 201 | `GS5` | 8: Interest Rates | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | 5-Year Treasury Constant Maturity Rate |
| 202 | `TB3SMFFM` | 8: Interest Rates | 1 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | 3-Month Treasury Constant Maturity Minus Federal Funds Rate |
| 203 | `T5YFFM` | 8: Interest Rates | 1 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | 5-Year Treasury Constant Maturity Minus Federal Funds Rate |
| 204 | `AAAFFM` | 8: Interest Rates | 1 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Moody's Seasoned Aaa Corporate Bond Minus Federal Funds Rate |
| 205 | `WPSID62` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Producer Price Index: Crude Materials for Further Processing (Index 1982=100) |
| 206 | `PPICMM` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Producer Price Index: Commodities: Metals and metal products: Primary nonferrous metals (Index 1982=100) |
| 207 | `CPIAPPSL` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: Apparel (Index 1982-84=100) |
| 208 | `CPITRNSL` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: Transportation (Index 1982-84=100) |
| 209 | `CPIMEDSL` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: Medical Care (Index 1982-84=100) |
| 210 | `CUSR0000SAC` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: Commodities (Index 1982-84=100) |
| 211 | `CUSR0000SAD` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: Durables (Index 1982-84=100) |
| 212 | `CUSR0000SAS` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: Services (Index 1982-84=100) |
| 213 | `CPIULFSL` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: All Items Less Food (Index 1982-84=100) |
| 214 | `CUSR0000SA0L2` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: All items less shelter (Index 1982-84=100) |
| 215 | `CUSR0000SA0L5` | 6: Prices | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Price Index for All Urban Consumers: All items less medical care (Index 1982-84=100) |
| 216 | `CES0600000008` | 7: Earnings and Productivity | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Average Hourly Earnings of Production and Nonsupervisory Employees: Goods-Producing (Dollars per Hour) |
| 217 | `DTCOLNVHFNM` | 9: Money and Credit | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Consumer Motor Vehicle Loans Outstanding Owned by Finance Companies (Millions of Dollars) |
| 218 | `DTCTHFNM` | 9: Money and Credit | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Total Consumer Loans and Leases Outstanding Owned and Securitized by Finance Companies (Millions of Dollars) |
| 219 | `INVEST` | 9: Money and Credit | 6 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Securities in Bank Credit at All Commercial Banks (Billions of Dollars) |
| 220 | `HWIURATIOx` | 3: Employment and Unemployment | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Ratio of Help Wanted/No. Unemployed |
| 221 | `CLAIMSx` | 3: Employment and Unemployment | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Initial Claims |
| 222 | `BUSINVx` | 5: Inventories, Orders, and Sales | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Total Business Inventories (Millions of Dollars) |
| 223 | `ISRATIOx` | 5: Inventories, Orders, and Sales | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Total Business: Inventories to Sales Ratio |
| 224 | `CONSPIx` | 10: Household Balance Sheets | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Nonrevolving consumer credit to Personal Income |
| 225 | `CP3M` | 8: Interest Rates | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | 3-Month AA Financial Commercial Paper Rate |
| 226 | `COMPAPFF` | 8: Interest Rates | 1 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | 3-Month Commercial Paper Minus Federal Funds Rate |
| 227 | `PERMITNE` | 4: Housing | 5 | 0 | 0 | 1960-03 | 2026-03 | 265 |  | New Private Housing Units Authorized by Building Permits in the Northeast Census Region (Thousands, SAAR) |
| 228 | `PERMITMW` | 4: Housing | 5 | 0 | 0 | 1960-03 | 2026-03 | 265 |  | New Private Housing Units Authorized by Building Permits in the Midwest Census Region (Thousands, SAAR) |
| 229 | `PERMITS` | 4: Housing | 5 | 0 | 0 | 1960-03 | 2026-03 | 265 |  | New Private Housing Units Authorized by Building Permits in the South Census Region (Thousands, SAAR) |
| 230 | `PERMITW` | 4: Housing | 5 | 0 | 0 | 1960-03 | 2026-03 | 265 |  | New Private Housing Units Authorized by Building Permits in the West Census Region (Thousands, SAAR) |
| 231 | `NIKKEI225` | 13: Stock Markets | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | Nikkei Stock Average |
| 232 | `NASDAQCOM` | 13: Stock Markets | 5 | 0 | 0 | 1971-03 | 2026-03 | 221 |  | NASDAQ Composite (Index Feb 5, 1971=100) |
| 233 | `CUSR0000SEHC` | 6: Prices | 6 | 0 | 0 | 1983-03 | 2026-03 | 173 |  | CPI for All Urban Consumers: Owners' equivalent rent of residences (Index Dec 1982=100) |
| 234 | `TLBSNNCBx` | 14: Non-Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Real Nonfinancial Corporate Business Sector Liabilities (Billions of 2017 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS |
| 235 | `TLBSNNCBBDIx` | 14: Non-Household Balance Sheets | 1 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Nonfinancial Corporate Business Sector Liabilities to Disposable Business Income (Percent) |
| 236 | `TTAABSNNCBx` | 14: Non-Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Real Nonfinancial Corporate Business Sector Assets  (Billions of 2017 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS  |
| 237 | `TNWMVBSNNCBx` | 14: Non-Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Real Nonfinancial Corporate Business Sector Net Worth  (Billions of 2017 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS  |
| 238 | `TNWMVBSNNCBBDIx` | 14: Non-Household Balance Sheets | 2 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Nonfinancial Corporate Business Sector Net Worth to Disposable Business Income (Percent) |
| 239 | `TLBSNNBx` | 14: Non-Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Real Nonfinancial Noncorporate Business Sector Liabilities  (Billions of 2017 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS |
| 240 | `TLBSNNBBDIx` | 14: Non-Household Balance Sheets | 1 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Nonfinancial Noncorporate Business Sector Liabilities to Disposable Business Income (Percent) |
| 241 | `TABSNNBx` | 14: Non-Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Real Nonfinancial Noncorporate Business Sector Assets  (Billions of 2017 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS |
| 242 | `TNWBSNNBx` | 14: Non-Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Real Nonfinancial Noncorporate Business Sector Net Worth  (Billions of 2017 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS |
| 243 | `TNWBSNNBBDIx` | 14: Non-Household Balance Sheets | 2 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Nonfinancial Noncorporate Business Sector Net Worth to Disposable Business Income (Percent) |
| 244 | `CNCFx` | 14: Non-Household Balance Sheets | 5 | 0 | 0 | 1959-03 | 2025-12 | 268 |  | Real Disposable Business Income, Billions of 2017 Dollars (Corporate cash flow with IVA minus taxes on corporate income, deflated by Implicit Price Deflator for Business Sector IPDBS) |
| 245 | `S&P 500` | 13: Stock Markets | 5 | 1 | 0 | 1959-03 | 2026-03 | 269 |  | S&P's Common Stock Price Index: Composite |
| 247 | `S&P div yield` | 13: Stock Markets | 2 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | S&P's Composite Common Stock: Dividend Yield |
| 248 | `S&P PE ratio` | 13: Stock Markets | 5 | 0 | 0 | 1959-03 | 2026-03 | 269 |  | S&P's Composite Common Stock: Price-Earnings Ratio |

## Combined With FRED-SD

`load_fred_qd_sd()` loads FRED-QD and FRED-SD and combines them.

```python
bundle = mf.data.load_fred_qd_sd(
    states=["CA", "TX"],
    variables=["UR", "NQGSP"],
    frequency="quarterly",
    monthly_to_quarterly="quarterly_average",
)
```

FRED-SD includes both monthly and quarterly state series. If a selected FRED-SD
series is monthly, the default rule
`monthly_to_quarterly="quarterly_average"` averages monthly observations
within each quarter. The function emits `UserWarning` and records the
conversion in `metadata["frequency_conversion_warnings"]`.
