# 5.3 FRED-SD

- Parent: [5. FRED-Dataset](index.md)
- Current dataset: FRED-SD

FRED-SD is the state-level panel used by `dataset=fred_sd` and by composite
routes such as `fred_md+fred_sd` or `fred_qd+fred_sd`.

macroforecast uses the official **Data by Series** workbook. The workbook vintage
used for this generated page is `2026-03`:

`https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-03.xlsx`

Generated: `2026-04-30`. Current data through: `2026-03`. Current
generated column count: `1428`. Native-frequency counts:
monthly: 861, quarterly: 546, unknown: 21.

## Column Contract

- Workbook sheets are FRED-SD variables.
- Sheet columns are state abbreviations.
- macroforecast generated columns use `{sd_variable}_{state}`.
- Example: sheet `UR`, state `CA` becomes `UR_CA`.
- Layer 1 owns state/series source selection. Layer 2 owns any mixed-frequency
  representation after the source frame exists.

## FRED-SD Variables / Workbook Sheets

| FRED-SD variable / workbook sheet | Generated state columns | Native-frequency profile |
|---|---:|---|
| `BPPRIVSA` | 51 | monthly: 51 |
| `CONS` | 51 | monthly: 51 |
| `CONSTNQGSP` | 51 | quarterly: 49, unknown: 2 |
| `EXPORTS` | 51 | monthly: 51 |
| `FIRE` | 51 | monthly: 51 |
| `FIRENQGSP` | 51 | quarterly: 51 |
| `GOVNQGSP` | 51 | quarterly: 51 |
| `GOVT` | 51 | monthly: 51 |
| `ICLAIMS` | 51 | monthly: 51 |
| `IMPORTS` | 51 | monthly: 51 |
| `INFO` | 51 | monthly: 51 |
| `INFONQGSP` | 51 | quarterly: 51 |
| `LF` | 51 | monthly: 51 |
| `MANNQGSP` | 51 | quarterly: 49, unknown: 2 |
| `MFG` | 51 | monthly: 51 |
| `MFGHRS` | 51 | monthly: 51 |
| `MINNG` | 51 | monthly: 48, unknown: 3 |
| `NA` | 51 | monthly: 51 |
| `NATURNQGSP` | 51 | quarterly: 40, unknown: 11 |
| `NQGSP` | 51 | quarterly: 51 |
| `OTOT` | 51 | quarterly: 51 |
| `PARTRATE` | 51 | monthly: 51 |
| `PSERV` | 51 | monthly: 51 |
| `PSERVNQGSP` | 51 | quarterly: 51 |
| `RENTS` | 51 | monthly: 48, unknown: 3 |
| `STHPI` | 51 | quarterly: 51 |
| `UR` | 51 | monthly: 51 |
| `UTILNQGSP` | 51 | quarterly: 51 |

## States

- `AK`: Alaska
- `AL`: Alabama
- `AR`: Arkansas
- `AZ`: Arizona
- `CA`: California
- `CO`: Colorado
- `CT`: Connecticut
- `DC`: District of Columbia
- `DE`: Delaware
- `FL`: Florida
- `GA`: Georgia
- `HI`: Hawaii
- `IA`: Iowa
- `ID`: Idaho
- `IL`: Illinois
- `IN`: Indiana
- `KS`: Kansas
- `KY`: Kentucky
- `LA`: Louisiana
- `MA`: Massachusetts
- `MD`: Maryland
- `ME`: Maine
- `MI`: Michigan
- `MN`: Minnesota
- `MO`: Missouri
- `MS`: Mississippi
- `MT`: Montana
- `NC`: North Carolina
- `ND`: North Dakota
- `NE`: Nebraska
- `NH`: New Hampshire
- `NJ`: New Jersey
- `NM`: New Mexico
- `NV`: Nevada
- `NY`: New York
- `OH`: Ohio
- `OK`: Oklahoma
- `OR`: Oregon
- `PA`: Pennsylvania
- `RI`: Rhode Island
- `SC`: South Carolina
- `SD`: South Dakota
- `TN`: Tennessee
- `TX`: Texas
- `UT`: Utah
- `VA`: Virginia
- `VT`: Vermont
- `WA`: Washington
- `WI`: Wisconsin
- `WV`: West Virginia
- `WY`: Wyoming

## All Current Generated Columns

| Column | FRED-SD variable | State | Native frequency | Observed window | Non-missing obs |
|---|---|---|---|---|---:|
| `BPPRIVSA_AK` | `BPPRIVSA` | `AK` (Alaska) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_AL` | `BPPRIVSA` | `AL` (Alabama) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_AR` | `BPPRIVSA` | `AR` (Arkansas) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_AZ` | `BPPRIVSA` | `AZ` (Arizona) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_CA` | `BPPRIVSA` | `CA` (California) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_CO` | `BPPRIVSA` | `CO` (Colorado) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_CT` | `BPPRIVSA` | `CT` (Connecticut) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_DC` | `BPPRIVSA` | `DC` (District of Columbia) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_DE` | `BPPRIVSA` | `DE` (Delaware) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_FL` | `BPPRIVSA` | `FL` (Florida) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_GA` | `BPPRIVSA` | `GA` (Georgia) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_HI` | `BPPRIVSA` | `HI` (Hawaii) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_IA` | `BPPRIVSA` | `IA` (Iowa) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_ID` | `BPPRIVSA` | `ID` (Idaho) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_IL` | `BPPRIVSA` | `IL` (Illinois) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_IN` | `BPPRIVSA` | `IN` (Indiana) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_KS` | `BPPRIVSA` | `KS` (Kansas) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_KY` | `BPPRIVSA` | `KY` (Kentucky) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_LA` | `BPPRIVSA` | `LA` (Louisiana) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_MA` | `BPPRIVSA` | `MA` (Massachusetts) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_MD` | `BPPRIVSA` | `MD` (Maryland) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_ME` | `BPPRIVSA` | `ME` (Maine) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_MI` | `BPPRIVSA` | `MI` (Michigan) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_MN` | `BPPRIVSA` | `MN` (Minnesota) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_MO` | `BPPRIVSA` | `MO` (Missouri) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_MS` | `BPPRIVSA` | `MS` (Mississippi) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_MT` | `BPPRIVSA` | `MT` (Montana) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_NC` | `BPPRIVSA` | `NC` (North Carolina) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_ND` | `BPPRIVSA` | `ND` (North Dakota) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_NE` | `BPPRIVSA` | `NE` (Nebraska) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_NH` | `BPPRIVSA` | `NH` (New Hampshire) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_NJ` | `BPPRIVSA` | `NJ` (New Jersey) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_NM` | `BPPRIVSA` | `NM` (New Mexico) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_NV` | `BPPRIVSA` | `NV` (Nevada) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_NY` | `BPPRIVSA` | `NY` (New York) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_OH` | `BPPRIVSA` | `OH` (Ohio) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_OK` | `BPPRIVSA` | `OK` (Oklahoma) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_OR` | `BPPRIVSA` | `OR` (Oregon) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_PA` | `BPPRIVSA` | `PA` (Pennsylvania) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_RI` | `BPPRIVSA` | `RI` (Rhode Island) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_SC` | `BPPRIVSA` | `SC` (South Carolina) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_SD` | `BPPRIVSA` | `SD` (South Dakota) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_TN` | `BPPRIVSA` | `TN` (Tennessee) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_TX` | `BPPRIVSA` | `TX` (Texas) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_UT` | `BPPRIVSA` | `UT` (Utah) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_VA` | `BPPRIVSA` | `VA` (Virginia) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_VT` | `BPPRIVSA` | `VT` (Vermont) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_WA` | `BPPRIVSA` | `WA` (Washington) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_WI` | `BPPRIVSA` | `WI` (Wisconsin) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_WV` | `BPPRIVSA` | `WV` (West Virginia) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `BPPRIVSA_WY` | `BPPRIVSA` | `WY` (Wyoming) | monthly | 1988-01-01 to 2025-08-01 | 452 |
| `CONS_AK` | `CONS` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_AL` | `CONS` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_AR` | `CONS` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_AZ` | `CONS` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_CA` | `CONS` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_CO` | `CONS` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_CT` | `CONS` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_DC` | `CONS` | `DC` (District of Columbia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_DE` | `CONS` | `DE` (Delaware) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_FL` | `CONS` | `FL` (Florida) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_GA` | `CONS` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_HI` | `CONS` | `HI` (Hawaii) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_IA` | `CONS` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_ID` | `CONS` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_IL` | `CONS` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_IN` | `CONS` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_KS` | `CONS` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_KY` | `CONS` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_LA` | `CONS` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_MA` | `CONS` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_MD` | `CONS` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_ME` | `CONS` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_MI` | `CONS` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_MN` | `CONS` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_MO` | `CONS` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_MS` | `CONS` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_MT` | `CONS` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_NC` | `CONS` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_ND` | `CONS` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_NE` | `CONS` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_NH` | `CONS` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_NJ` | `CONS` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_NM` | `CONS` | `NM` (New Mexico) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_NV` | `CONS` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_NY` | `CONS` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_OH` | `CONS` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_OK` | `CONS` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_OR` | `CONS` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_PA` | `CONS` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_RI` | `CONS` | `RI` (Rhode Island) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_SC` | `CONS` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_SD` | `CONS` | `SD` (South Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_TN` | `CONS` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_TX` | `CONS` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_UT` | `CONS` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_VA` | `CONS` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_VT` | `CONS` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_WA` | `CONS` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_WI` | `CONS` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_WV` | `CONS` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONS_WY` | `CONS` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `CONSTNQGSP_AK` | `CONSTNQGSP` | `AK` (Alaska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_AL` | `CONSTNQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_AR` | `CONSTNQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_AZ` | `CONSTNQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_CA` | `CONSTNQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_CO` | `CONSTNQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_CT` | `CONSTNQGSP` | `CT` (Connecticut) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_DC` | `CONSTNQGSP` | `DC` (District of Columbia) | unknown | - | 0 |
| `CONSTNQGSP_DE` | `CONSTNQGSP` | `DE` (Delaware) | quarterly | 2006-01-01 to 2025-07-01 | 79 |
| `CONSTNQGSP_FL` | `CONSTNQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_GA` | `CONSTNQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_HI` | `CONSTNQGSP` | `HI` (Hawaii) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_IA` | `CONSTNQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_ID` | `CONSTNQGSP` | `ID` (Idaho) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_IL` | `CONSTNQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_IN` | `CONSTNQGSP` | `IN` (Indiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_KS` | `CONSTNQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_KY` | `CONSTNQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_LA` | `CONSTNQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_MA` | `CONSTNQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_MD` | `CONSTNQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_ME` | `CONSTNQGSP` | `ME` (Maine) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_MI` | `CONSTNQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_MN` | `CONSTNQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_MO` | `CONSTNQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_MS` | `CONSTNQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_MT` | `CONSTNQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_NC` | `CONSTNQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_ND` | `CONSTNQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_NE` | `CONSTNQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_NH` | `CONSTNQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_NJ` | `CONSTNQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_NM` | `CONSTNQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_NV` | `CONSTNQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_NY` | `CONSTNQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_OH` | `CONSTNQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_OK` | `CONSTNQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_OR` | `CONSTNQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_PA` | `CONSTNQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_RI` | `CONSTNQGSP` | `RI` (Rhode Island) | unknown | - | 0 |
| `CONSTNQGSP_SC` | `CONSTNQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_SD` | `CONSTNQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_TN` | `CONSTNQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_TX` | `CONSTNQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_UT` | `CONSTNQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_VA` | `CONSTNQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_VT` | `CONSTNQGSP` | `VT` (Vermont) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_WA` | `CONSTNQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_WI` | `CONSTNQGSP` | `WI` (Wisconsin) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_WV` | `CONSTNQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `CONSTNQGSP_WY` | `CONSTNQGSP` | `WY` (Wyoming) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `EXPORTS_AK` | `EXPORTS` | `AK` (Alaska) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_AL` | `EXPORTS` | `AL` (Alabama) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_AR` | `EXPORTS` | `AR` (Arkansas) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_AZ` | `EXPORTS` | `AZ` (Arizona) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_CA` | `EXPORTS` | `CA` (California) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_CO` | `EXPORTS` | `CO` (Colorado) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_CT` | `EXPORTS` | `CT` (Connecticut) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_DC` | `EXPORTS` | `DC` (District of Columbia) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_DE` | `EXPORTS` | `DE` (Delaware) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_FL` | `EXPORTS` | `FL` (Florida) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_GA` | `EXPORTS` | `GA` (Georgia) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_HI` | `EXPORTS` | `HI` (Hawaii) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_IA` | `EXPORTS` | `IA` (Iowa) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_ID` | `EXPORTS` | `ID` (Idaho) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_IL` | `EXPORTS` | `IL` (Illinois) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_IN` | `EXPORTS` | `IN` (Indiana) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_KS` | `EXPORTS` | `KS` (Kansas) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_KY` | `EXPORTS` | `KY` (Kentucky) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_LA` | `EXPORTS` | `LA` (Louisiana) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_MA` | `EXPORTS` | `MA` (Massachusetts) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_MD` | `EXPORTS` | `MD` (Maryland) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_ME` | `EXPORTS` | `ME` (Maine) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_MI` | `EXPORTS` | `MI` (Michigan) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_MN` | `EXPORTS` | `MN` (Minnesota) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_MO` | `EXPORTS` | `MO` (Missouri) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_MS` | `EXPORTS` | `MS` (Mississippi) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_MT` | `EXPORTS` | `MT` (Montana) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_NC` | `EXPORTS` | `NC` (North Carolina) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_ND` | `EXPORTS` | `ND` (North Dakota) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_NE` | `EXPORTS` | `NE` (Nebraska) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_NH` | `EXPORTS` | `NH` (New Hampshire) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_NJ` | `EXPORTS` | `NJ` (New Jersey) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_NM` | `EXPORTS` | `NM` (New Mexico) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_NV` | `EXPORTS` | `NV` (Nevada) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_NY` | `EXPORTS` | `NY` (New York) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_OH` | `EXPORTS` | `OH` (Ohio) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_OK` | `EXPORTS` | `OK` (Oklahoma) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_OR` | `EXPORTS` | `OR` (Oregon) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_PA` | `EXPORTS` | `PA` (Pennsylvania) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_RI` | `EXPORTS` | `RI` (Rhode Island) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_SC` | `EXPORTS` | `SC` (South Carolina) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_SD` | `EXPORTS` | `SD` (South Dakota) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_TN` | `EXPORTS` | `TN` (Tennessee) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_TX` | `EXPORTS` | `TX` (Texas) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_UT` | `EXPORTS` | `UT` (Utah) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_VA` | `EXPORTS` | `VA` (Virginia) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_VT` | `EXPORTS` | `VT` (Vermont) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_WA` | `EXPORTS` | `WA` (Washington) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_WI` | `EXPORTS` | `WI` (Wisconsin) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_WV` | `EXPORTS` | `WV` (West Virginia) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `EXPORTS_WY` | `EXPORTS` | `WY` (Wyoming) | monthly | 1995-08-01 to 2026-01-01 | 366 |
| `FIRE_AK` | `FIRE` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_AL` | `FIRE` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_AR` | `FIRE` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_AZ` | `FIRE` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_CA` | `FIRE` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_CO` | `FIRE` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_CT` | `FIRE` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_DC` | `FIRE` | `DC` (District of Columbia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_DE` | `FIRE` | `DE` (Delaware) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_FL` | `FIRE` | `FL` (Florida) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_GA` | `FIRE` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_HI` | `FIRE` | `HI` (Hawaii) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_IA` | `FIRE` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_ID` | `FIRE` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_IL` | `FIRE` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_IN` | `FIRE` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_KS` | `FIRE` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_KY` | `FIRE` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_LA` | `FIRE` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_MA` | `FIRE` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_MD` | `FIRE` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_ME` | `FIRE` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_MI` | `FIRE` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_MN` | `FIRE` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_MO` | `FIRE` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_MS` | `FIRE` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_MT` | `FIRE` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_NC` | `FIRE` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_ND` | `FIRE` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_NE` | `FIRE` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_NH` | `FIRE` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_NJ` | `FIRE` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_NM` | `FIRE` | `NM` (New Mexico) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_NV` | `FIRE` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_NY` | `FIRE` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_OH` | `FIRE` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_OK` | `FIRE` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_OR` | `FIRE` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_PA` | `FIRE` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_RI` | `FIRE` | `RI` (Rhode Island) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_SC` | `FIRE` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_SD` | `FIRE` | `SD` (South Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_TN` | `FIRE` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_TX` | `FIRE` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_UT` | `FIRE` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_VA` | `FIRE` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_VT` | `FIRE` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_WA` | `FIRE` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_WI` | `FIRE` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_WV` | `FIRE` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRE_WY` | `FIRE` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `FIRENQGSP_AK` | `FIRENQGSP` | `AK` (Alaska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_AL` | `FIRENQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_AR` | `FIRENQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_AZ` | `FIRENQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_CA` | `FIRENQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_CO` | `FIRENQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_CT` | `FIRENQGSP` | `CT` (Connecticut) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_DC` | `FIRENQGSP` | `DC` (District of Columbia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_DE` | `FIRENQGSP` | `DE` (Delaware) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_FL` | `FIRENQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_GA` | `FIRENQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_HI` | `FIRENQGSP` | `HI` (Hawaii) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_IA` | `FIRENQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_ID` | `FIRENQGSP` | `ID` (Idaho) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_IL` | `FIRENQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_IN` | `FIRENQGSP` | `IN` (Indiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_KS` | `FIRENQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_KY` | `FIRENQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_LA` | `FIRENQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_MA` | `FIRENQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_MD` | `FIRENQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_ME` | `FIRENQGSP` | `ME` (Maine) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_MI` | `FIRENQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_MN` | `FIRENQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_MO` | `FIRENQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_MS` | `FIRENQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_MT` | `FIRENQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_NC` | `FIRENQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_ND` | `FIRENQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_NE` | `FIRENQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_NH` | `FIRENQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_NJ` | `FIRENQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_NM` | `FIRENQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_NV` | `FIRENQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_NY` | `FIRENQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_OH` | `FIRENQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_OK` | `FIRENQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_OR` | `FIRENQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_PA` | `FIRENQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_RI` | `FIRENQGSP` | `RI` (Rhode Island) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_SC` | `FIRENQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_SD` | `FIRENQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_TN` | `FIRENQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_TX` | `FIRENQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_UT` | `FIRENQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_VA` | `FIRENQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_VT` | `FIRENQGSP` | `VT` (Vermont) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_WA` | `FIRENQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_WI` | `FIRENQGSP` | `WI` (Wisconsin) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_WV` | `FIRENQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `FIRENQGSP_WY` | `FIRENQGSP` | `WY` (Wyoming) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_AK` | `GOVNQGSP` | `AK` (Alaska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_AL` | `GOVNQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_AR` | `GOVNQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_AZ` | `GOVNQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_CA` | `GOVNQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_CO` | `GOVNQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_CT` | `GOVNQGSP` | `CT` (Connecticut) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_DC` | `GOVNQGSP` | `DC` (District of Columbia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_DE` | `GOVNQGSP` | `DE` (Delaware) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_FL` | `GOVNQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_GA` | `GOVNQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_HI` | `GOVNQGSP` | `HI` (Hawaii) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_IA` | `GOVNQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_ID` | `GOVNQGSP` | `ID` (Idaho) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_IL` | `GOVNQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_IN` | `GOVNQGSP` | `IN` (Indiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_KS` | `GOVNQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_KY` | `GOVNQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_LA` | `GOVNQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_MA` | `GOVNQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_MD` | `GOVNQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_ME` | `GOVNQGSP` | `ME` (Maine) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_MI` | `GOVNQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_MN` | `GOVNQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_MO` | `GOVNQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_MS` | `GOVNQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_MT` | `GOVNQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_NC` | `GOVNQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_ND` | `GOVNQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_NE` | `GOVNQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_NH` | `GOVNQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_NJ` | `GOVNQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_NM` | `GOVNQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_NV` | `GOVNQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_NY` | `GOVNQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_OH` | `GOVNQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_OK` | `GOVNQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_OR` | `GOVNQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_PA` | `GOVNQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_RI` | `GOVNQGSP` | `RI` (Rhode Island) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_SC` | `GOVNQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_SD` | `GOVNQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_TN` | `GOVNQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_TX` | `GOVNQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_UT` | `GOVNQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_VA` | `GOVNQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_VT` | `GOVNQGSP` | `VT` (Vermont) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_WA` | `GOVNQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_WI` | `GOVNQGSP` | `WI` (Wisconsin) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_WV` | `GOVNQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVNQGSP_WY` | `GOVNQGSP` | `WY` (Wyoming) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `GOVT_AK` | `GOVT` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_AL` | `GOVT` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_AR` | `GOVT` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_AZ` | `GOVT` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_CA` | `GOVT` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_CO` | `GOVT` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_CT` | `GOVT` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_DC` | `GOVT` | `DC` (District of Columbia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_DE` | `GOVT` | `DE` (Delaware) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_FL` | `GOVT` | `FL` (Florida) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_GA` | `GOVT` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_HI` | `GOVT` | `HI` (Hawaii) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_IA` | `GOVT` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_ID` | `GOVT` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_IL` | `GOVT` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_IN` | `GOVT` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_KS` | `GOVT` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_KY` | `GOVT` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_LA` | `GOVT` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_MA` | `GOVT` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_MD` | `GOVT` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_ME` | `GOVT` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_MI` | `GOVT` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_MN` | `GOVT` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_MO` | `GOVT` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_MS` | `GOVT` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_MT` | `GOVT` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_NC` | `GOVT` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_ND` | `GOVT` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_NE` | `GOVT` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_NH` | `GOVT` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_NJ` | `GOVT` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_NM` | `GOVT` | `NM` (New Mexico) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_NV` | `GOVT` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_NY` | `GOVT` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_OH` | `GOVT` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_OK` | `GOVT` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_OR` | `GOVT` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_PA` | `GOVT` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_RI` | `GOVT` | `RI` (Rhode Island) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_SC` | `GOVT` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_SD` | `GOVT` | `SD` (South Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_TN` | `GOVT` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_TX` | `GOVT` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_UT` | `GOVT` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_VA` | `GOVT` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_VT` | `GOVT` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_WA` | `GOVT` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_WI` | `GOVT` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_WV` | `GOVT` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `GOVT_WY` | `GOVT` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `ICLAIMS_AK` | `ICLAIMS` | `AK` (Alaska) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_AL` | `ICLAIMS` | `AL` (Alabama) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_AR` | `ICLAIMS` | `AR` (Arkansas) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_AZ` | `ICLAIMS` | `AZ` (Arizona) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_CA` | `ICLAIMS` | `CA` (California) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_CO` | `ICLAIMS` | `CO` (Colorado) | monthly | 1985-09-28 to 2026-03-07 | 487 |
| `ICLAIMS_CT` | `ICLAIMS` | `CT` (Connecticut) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_DC` | `ICLAIMS` | `DC` (District of Columbia) | monthly | 1986-01-04 to 2026-03-07 | 483 |
| `ICLAIMS_DE` | `ICLAIMS` | `DE` (Delaware) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_FL` | `ICLAIMS` | `FL` (Florida) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_GA` | `ICLAIMS` | `GA` (Georgia) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_HI` | `ICLAIMS` | `HI` (Hawaii) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_IA` | `ICLAIMS` | `IA` (Iowa) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_ID` | `ICLAIMS` | `ID` (Idaho) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_IL` | `ICLAIMS` | `IL` (Illinois) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_IN` | `ICLAIMS` | `IN` (Indiana) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_KS` | `ICLAIMS` | `KS` (Kansas) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_KY` | `ICLAIMS` | `KY` (Kentucky) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_LA` | `ICLAIMS` | `LA` (Louisiana) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_MA` | `ICLAIMS` | `MA` (Massachusetts) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_MD` | `ICLAIMS` | `MD` (Maryland) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_ME` | `ICLAIMS` | `ME` (Maine) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_MI` | `ICLAIMS` | `MI` (Michigan) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_MN` | `ICLAIMS` | `MN` (Minnesota) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_MO` | `ICLAIMS` | `MO` (Missouri) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_MS` | `ICLAIMS` | `MS` (Mississippi) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_MT` | `ICLAIMS` | `MT` (Montana) | monthly | 1985-10-05 to 2026-03-07 | 486 |
| `ICLAIMS_NC` | `ICLAIMS` | `NC` (North Carolina) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_ND` | `ICLAIMS` | `ND` (North Dakota) | monthly | 1985-10-05 to 2026-03-07 | 486 |
| `ICLAIMS_NE` | `ICLAIMS` | `NE` (Nebraska) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_NH` | `ICLAIMS` | `NH` (New Hampshire) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_NJ` | `ICLAIMS` | `NJ` (New Jersey) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_NM` | `ICLAIMS` | `NM` (New Mexico) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_NV` | `ICLAIMS` | `NV` (Nevada) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_NY` | `ICLAIMS` | `NY` (New York) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_OH` | `ICLAIMS` | `OH` (Ohio) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_OK` | `ICLAIMS` | `OK` (Oklahoma) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_OR` | `ICLAIMS` | `OR` (Oregon) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_PA` | `ICLAIMS` | `PA` (Pennsylvania) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_RI` | `ICLAIMS` | `RI` (Rhode Island) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_SC` | `ICLAIMS` | `SC` (South Carolina) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_SD` | `ICLAIMS` | `SD` (South Dakota) | monthly | 1985-09-28 to 2026-03-07 | 487 |
| `ICLAIMS_TN` | `ICLAIMS` | `TN` (Tennessee) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_TX` | `ICLAIMS` | `TX` (Texas) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_UT` | `ICLAIMS` | `UT` (Utah) | monthly | 1985-09-28 to 2026-03-07 | 487 |
| `ICLAIMS_VA` | `ICLAIMS` | `VA` (Virginia) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_VT` | `ICLAIMS` | `VT` (Vermont) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_WA` | `ICLAIMS` | `WA` (Washington) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_WI` | `ICLAIMS` | `WI` (Wisconsin) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_WV` | `ICLAIMS` | `WV` (West Virginia) | monthly | 1986-02-01 to 2026-03-07 | 482 |
| `ICLAIMS_WY` | `ICLAIMS` | `WY` (Wyoming) | monthly | 1985-09-28 to 2026-03-07 | 487 |
| `IMPORTS_AK` | `IMPORTS` | `AK` (Alaska) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_AL` | `IMPORTS` | `AL` (Alabama) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_AR` | `IMPORTS` | `AR` (Arkansas) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_AZ` | `IMPORTS` | `AZ` (Arizona) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_CA` | `IMPORTS` | `CA` (California) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_CO` | `IMPORTS` | `CO` (Colorado) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_CT` | `IMPORTS` | `CT` (Connecticut) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_DC` | `IMPORTS` | `DC` (District of Columbia) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_DE` | `IMPORTS` | `DE` (Delaware) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_FL` | `IMPORTS` | `FL` (Florida) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_GA` | `IMPORTS` | `GA` (Georgia) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_HI` | `IMPORTS` | `HI` (Hawaii) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_IA` | `IMPORTS` | `IA` (Iowa) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_ID` | `IMPORTS` | `ID` (Idaho) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_IL` | `IMPORTS` | `IL` (Illinois) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_IN` | `IMPORTS` | `IN` (Indiana) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_KS` | `IMPORTS` | `KS` (Kansas) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_KY` | `IMPORTS` | `KY` (Kentucky) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_LA` | `IMPORTS` | `LA` (Louisiana) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_MA` | `IMPORTS` | `MA` (Massachusetts) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_MD` | `IMPORTS` | `MD` (Maryland) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_ME` | `IMPORTS` | `ME` (Maine) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_MI` | `IMPORTS` | `MI` (Michigan) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_MN` | `IMPORTS` | `MN` (Minnesota) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_MO` | `IMPORTS` | `MO` (Missouri) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_MS` | `IMPORTS` | `MS` (Mississippi) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_MT` | `IMPORTS` | `MT` (Montana) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_NC` | `IMPORTS` | `NC` (North Carolina) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_ND` | `IMPORTS` | `ND` (North Dakota) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_NE` | `IMPORTS` | `NE` (Nebraska) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_NH` | `IMPORTS` | `NH` (New Hampshire) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_NJ` | `IMPORTS` | `NJ` (New Jersey) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_NM` | `IMPORTS` | `NM` (New Mexico) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_NV` | `IMPORTS` | `NV` (Nevada) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_NY` | `IMPORTS` | `NY` (New York) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_OH` | `IMPORTS` | `OH` (Ohio) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_OK` | `IMPORTS` | `OK` (Oklahoma) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_OR` | `IMPORTS` | `OR` (Oregon) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_PA` | `IMPORTS` | `PA` (Pennsylvania) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_RI` | `IMPORTS` | `RI` (Rhode Island) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_SC` | `IMPORTS` | `SC` (South Carolina) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_SD` | `IMPORTS` | `SD` (South Dakota) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_TN` | `IMPORTS` | `TN` (Tennessee) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_TX` | `IMPORTS` | `TX` (Texas) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_UT` | `IMPORTS` | `UT` (Utah) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_VA` | `IMPORTS` | `VA` (Virginia) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_VT` | `IMPORTS` | `VT` (Vermont) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_WA` | `IMPORTS` | `WA` (Washington) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_WI` | `IMPORTS` | `WI` (Wisconsin) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_WV` | `IMPORTS` | `WV` (West Virginia) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `IMPORTS_WY` | `IMPORTS` | `WY` (Wyoming) | monthly | 2008-01-01 to 2026-01-01 | 217 |
| `INFO_AK` | `INFO` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_AL` | `INFO` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_AR` | `INFO` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_AZ` | `INFO` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_CA` | `INFO` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_CO` | `INFO` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_CT` | `INFO` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_DC` | `INFO` | `DC` (District of Columbia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_DE` | `INFO` | `DE` (Delaware) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_FL` | `INFO` | `FL` (Florida) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_GA` | `INFO` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_HI` | `INFO` | `HI` (Hawaii) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_IA` | `INFO` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_ID` | `INFO` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_IL` | `INFO` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_IN` | `INFO` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_KS` | `INFO` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_KY` | `INFO` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_LA` | `INFO` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_MA` | `INFO` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_MD` | `INFO` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_ME` | `INFO` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_MI` | `INFO` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_MN` | `INFO` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_MO` | `INFO` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_MS` | `INFO` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_MT` | `INFO` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_NC` | `INFO` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_ND` | `INFO` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_NE` | `INFO` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_NH` | `INFO` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_NJ` | `INFO` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_NM` | `INFO` | `NM` (New Mexico) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_NV` | `INFO` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_NY` | `INFO` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_OH` | `INFO` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_OK` | `INFO` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_OR` | `INFO` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_PA` | `INFO` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_RI` | `INFO` | `RI` (Rhode Island) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_SC` | `INFO` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_SD` | `INFO` | `SD` (South Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_TN` | `INFO` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_TX` | `INFO` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_UT` | `INFO` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_VA` | `INFO` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_VT` | `INFO` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_WA` | `INFO` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_WI` | `INFO` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_WV` | `INFO` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFO_WY` | `INFO` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `INFONQGSP_AK` | `INFONQGSP` | `AK` (Alaska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_AL` | `INFONQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_AR` | `INFONQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_AZ` | `INFONQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_CA` | `INFONQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_CO` | `INFONQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_CT` | `INFONQGSP` | `CT` (Connecticut) | quarterly | 2005-01-01 to 2025-01-01 | 81 |
| `INFONQGSP_DC` | `INFONQGSP` | `DC` (District of Columbia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_DE` | `INFONQGSP` | `DE` (Delaware) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_FL` | `INFONQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_GA` | `INFONQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_HI` | `INFONQGSP` | `HI` (Hawaii) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_IA` | `INFONQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_ID` | `INFONQGSP` | `ID` (Idaho) | quarterly | 2005-01-01 to 2025-01-01 | 81 |
| `INFONQGSP_IL` | `INFONQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_IN` | `INFONQGSP` | `IN` (Indiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_KS` | `INFONQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_KY` | `INFONQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_LA` | `INFONQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_MA` | `INFONQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_MD` | `INFONQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_ME` | `INFONQGSP` | `ME` (Maine) | quarterly | 2005-01-01 to 2025-01-01 | 81 |
| `INFONQGSP_MI` | `INFONQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_MN` | `INFONQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_MO` | `INFONQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_MS` | `INFONQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_MT` | `INFONQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_NC` | `INFONQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_ND` | `INFONQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_NE` | `INFONQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_NH` | `INFONQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_NJ` | `INFONQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_NM` | `INFONQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_NV` | `INFONQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_NY` | `INFONQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_OH` | `INFONQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_OK` | `INFONQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_OR` | `INFONQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_PA` | `INFONQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_RI` | `INFONQGSP` | `RI` (Rhode Island) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_SC` | `INFONQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_SD` | `INFONQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_TN` | `INFONQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_TX` | `INFONQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_UT` | `INFONQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_VA` | `INFONQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_VT` | `INFONQGSP` | `VT` (Vermont) | quarterly | 2005-01-01 to 2025-01-01 | 81 |
| `INFONQGSP_WA` | `INFONQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_WI` | `INFONQGSP` | `WI` (Wisconsin) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_WV` | `INFONQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `INFONQGSP_WY` | `INFONQGSP` | `WY` (Wyoming) | quarterly | 2005-01-01 to 2025-01-01 | 81 |
| `LF_AK` | `LF` | `AK` (Alaska) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_AL` | `LF` | `AL` (Alabama) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_AR` | `LF` | `AR` (Arkansas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_AZ` | `LF` | `AZ` (Arizona) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_CA` | `LF` | `CA` (California) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_CO` | `LF` | `CO` (Colorado) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_CT` | `LF` | `CT` (Connecticut) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_DC` | `LF` | `DC` (District of Columbia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_DE` | `LF` | `DE` (Delaware) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_FL` | `LF` | `FL` (Florida) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_GA` | `LF` | `GA` (Georgia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_HI` | `LF` | `HI` (Hawaii) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_IA` | `LF` | `IA` (Iowa) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_ID` | `LF` | `ID` (Idaho) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_IL` | `LF` | `IL` (Illinois) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_IN` | `LF` | `IN` (Indiana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_KS` | `LF` | `KS` (Kansas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_KY` | `LF` | `KY` (Kentucky) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_LA` | `LF` | `LA` (Louisiana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_MA` | `LF` | `MA` (Massachusetts) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_MD` | `LF` | `MD` (Maryland) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_ME` | `LF` | `ME` (Maine) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_MI` | `LF` | `MI` (Michigan) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_MN` | `LF` | `MN` (Minnesota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_MO` | `LF` | `MO` (Missouri) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_MS` | `LF` | `MS` (Mississippi) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_MT` | `LF` | `MT` (Montana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_NC` | `LF` | `NC` (North Carolina) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_ND` | `LF` | `ND` (North Dakota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_NE` | `LF` | `NE` (Nebraska) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_NH` | `LF` | `NH` (New Hampshire) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_NJ` | `LF` | `NJ` (New Jersey) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_NM` | `LF` | `NM` (New Mexico) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_NV` | `LF` | `NV` (Nevada) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_NY` | `LF` | `NY` (New York) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_OH` | `LF` | `OH` (Ohio) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_OK` | `LF` | `OK` (Oklahoma) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_OR` | `LF` | `OR` (Oregon) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_PA` | `LF` | `PA` (Pennsylvania) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_RI` | `LF` | `RI` (Rhode Island) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_SC` | `LF` | `SC` (South Carolina) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_SD` | `LF` | `SD` (South Dakota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_TN` | `LF` | `TN` (Tennessee) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_TX` | `LF` | `TX` (Texas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_UT` | `LF` | `UT` (Utah) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_VA` | `LF` | `VA` (Virginia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_VT` | `LF` | `VT` (Vermont) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_WA` | `LF` | `WA` (Washington) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_WI` | `LF` | `WI` (Wisconsin) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_WV` | `LF` | `WV` (West Virginia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `LF_WY` | `LF` | `WY` (Wyoming) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `MANNQGSP_AK` | `MANNQGSP` | `AK` (Alaska) | quarterly | 2005-01-01 to 2025-07-01 | 79 |
| `MANNQGSP_AL` | `MANNQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_AR` | `MANNQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_AZ` | `MANNQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_CA` | `MANNQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_CO` | `MANNQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_CT` | `MANNQGSP` | `CT` (Connecticut) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_DC` | `MANNQGSP` | `DC` (District of Columbia) | unknown | - | 0 |
| `MANNQGSP_DE` | `MANNQGSP` | `DE` (Delaware) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_FL` | `MANNQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_GA` | `MANNQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_HI` | `MANNQGSP` | `HI` (Hawaii) | quarterly | 2005-01-01 to 2025-01-01 | 77 |
| `MANNQGSP_IA` | `MANNQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_ID` | `MANNQGSP` | `ID` (Idaho) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_IL` | `MANNQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_IN` | `MANNQGSP` | `IN` (Indiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_KS` | `MANNQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_KY` | `MANNQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_LA` | `MANNQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_MA` | `MANNQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_MD` | `MANNQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_ME` | `MANNQGSP` | `ME` (Maine) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_MI` | `MANNQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_MN` | `MANNQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_MO` | `MANNQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_MS` | `MANNQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_MT` | `MANNQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_NC` | `MANNQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_ND` | `MANNQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_NE` | `MANNQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_NH` | `MANNQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_NJ` | `MANNQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_NM` | `MANNQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_NV` | `MANNQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_NY` | `MANNQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_OH` | `MANNQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_OK` | `MANNQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_OR` | `MANNQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_PA` | `MANNQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_RI` | `MANNQGSP` | `RI` (Rhode Island) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_SC` | `MANNQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_SD` | `MANNQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_TN` | `MANNQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_TX` | `MANNQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_UT` | `MANNQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_VA` | `MANNQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_VT` | `MANNQGSP` | `VT` (Vermont) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_WA` | `MANNQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_WI` | `MANNQGSP` | `WI` (Wisconsin) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_WV` | `MANNQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `MANNQGSP_WY` | `MANNQGSP` | `WY` (Wyoming) | unknown | - | 0 |
| `MFG_AK` | `MFG` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_AL` | `MFG` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_AR` | `MFG` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_AZ` | `MFG` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_CA` | `MFG` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_CO` | `MFG` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_CT` | `MFG` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_DC` | `MFG` | `DC` (District of Columbia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_DE` | `MFG` | `DE` (Delaware) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_FL` | `MFG` | `FL` (Florida) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_GA` | `MFG` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_HI` | `MFG` | `HI` (Hawaii) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_IA` | `MFG` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_ID` | `MFG` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_IL` | `MFG` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_IN` | `MFG` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_KS` | `MFG` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_KY` | `MFG` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_LA` | `MFG` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_MA` | `MFG` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_MD` | `MFG` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_ME` | `MFG` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_MI` | `MFG` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_MN` | `MFG` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_MO` | `MFG` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_MS` | `MFG` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_MT` | `MFG` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_NC` | `MFG` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_ND` | `MFG` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_NE` | `MFG` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_NH` | `MFG` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_NJ` | `MFG` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_NM` | `MFG` | `NM` (New Mexico) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_NV` | `MFG` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_NY` | `MFG` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_OH` | `MFG` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_OK` | `MFG` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_OR` | `MFG` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_PA` | `MFG` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_RI` | `MFG` | `RI` (Rhode Island) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_SC` | `MFG` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_SD` | `MFG` | `SD` (South Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_TN` | `MFG` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_TX` | `MFG` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_UT` | `MFG` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_VA` | `MFG` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_VT` | `MFG` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_WA` | `MFG` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_WI` | `MFG` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_WV` | `MFG` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFG_WY` | `MFG` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MFGHRS_AK` | `MFGHRS` | `AK` (Alaska) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_AL` | `MFGHRS` | `AL` (Alabama) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_AR` | `MFGHRS` | `AR` (Arkansas) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_AZ` | `MFGHRS` | `AZ` (Arizona) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_CA` | `MFGHRS` | `CA` (California) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_CO` | `MFGHRS` | `CO` (Colorado) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_CT` | `MFGHRS` | `CT` (Connecticut) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_DC` | `MFGHRS` | `DC` (District of Columbia) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_DE` | `MFGHRS` | `DE` (Delaware) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_FL` | `MFGHRS` | `FL` (Florida) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_GA` | `MFGHRS` | `GA` (Georgia) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_HI` | `MFGHRS` | `HI` (Hawaii) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_IA` | `MFGHRS` | `IA` (Iowa) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_ID` | `MFGHRS` | `ID` (Idaho) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_IL` | `MFGHRS` | `IL` (Illinois) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_IN` | `MFGHRS` | `IN` (Indiana) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_KS` | `MFGHRS` | `KS` (Kansas) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_KY` | `MFGHRS` | `KY` (Kentucky) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_LA` | `MFGHRS` | `LA` (Louisiana) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_MA` | `MFGHRS` | `MA` (Massachusetts) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_MD` | `MFGHRS` | `MD` (Maryland) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_ME` | `MFGHRS` | `ME` (Maine) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_MI` | `MFGHRS` | `MI` (Michigan) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_MN` | `MFGHRS` | `MN` (Minnesota) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_MO` | `MFGHRS` | `MO` (Missouri) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_MS` | `MFGHRS` | `MS` (Mississippi) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_MT` | `MFGHRS` | `MT` (Montana) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_NC` | `MFGHRS` | `NC` (North Carolina) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_ND` | `MFGHRS` | `ND` (North Dakota) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_NE` | `MFGHRS` | `NE` (Nebraska) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_NH` | `MFGHRS` | `NH` (New Hampshire) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_NJ` | `MFGHRS` | `NJ` (New Jersey) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_NM` | `MFGHRS` | `NM` (New Mexico) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_NV` | `MFGHRS` | `NV` (Nevada) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_NY` | `MFGHRS` | `NY` (New York) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_OH` | `MFGHRS` | `OH` (Ohio) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_OK` | `MFGHRS` | `OK` (Oklahoma) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_OR` | `MFGHRS` | `OR` (Oregon) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_PA` | `MFGHRS` | `PA` (Pennsylvania) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_RI` | `MFGHRS` | `RI` (Rhode Island) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_SC` | `MFGHRS` | `SC` (South Carolina) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_SD` | `MFGHRS` | `SD` (South Dakota) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_TN` | `MFGHRS` | `TN` (Tennessee) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_TX` | `MFGHRS` | `TX` (Texas) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_UT` | `MFGHRS` | `UT` (Utah) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_VA` | `MFGHRS` | `VA` (Virginia) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_VT` | `MFGHRS` | `VT` (Vermont) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_WA` | `MFGHRS` | `WA` (Washington) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_WI` | `MFGHRS` | `WI` (Wisconsin) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_WV` | `MFGHRS` | `WV` (West Virginia) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MFGHRS_WY` | `MFGHRS` | `WY` (Wyoming) | monthly | 2007-01-01 to 2025-12-01 | 228 |
| `MINNG_AK` | `MINNG` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_AL` | `MINNG` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_AR` | `MINNG` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_AZ` | `MINNG` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_CA` | `MINNG` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_CO` | `MINNG` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_CT` | `MINNG` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_DC` | `MINNG` | `DC` (District of Columbia) | unknown | - | 0 |
| `MINNG_DE` | `MINNG` | `DE` (Delaware) | unknown | - | 0 |
| `MINNG_FL` | `MINNG` | `FL` (Florida) | monthly | 2002-01-01 to 2025-12-01 | 288 |
| `MINNG_GA` | `MINNG` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_HI` | `MINNG` | `HI` (Hawaii) | unknown | - | 0 |
| `MINNG_IA` | `MINNG` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_ID` | `MINNG` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_IL` | `MINNG` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_IN` | `MINNG` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_KS` | `MINNG` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_KY` | `MINNG` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_LA` | `MINNG` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_MA` | `MINNG` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_MD` | `MINNG` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_ME` | `MINNG` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_MI` | `MINNG` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_MN` | `MINNG` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_MO` | `MINNG` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_MS` | `MINNG` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_MT` | `MINNG` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_NC` | `MINNG` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_ND` | `MINNG` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_NE` | `MINNG` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_NH` | `MINNG` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_NJ` | `MINNG` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_NM` | `MINNG` | `NM` (New Mexico) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_NV` | `MINNG` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_NY` | `MINNG` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_OH` | `MINNG` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_OK` | `MINNG` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_OR` | `MINNG` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_PA` | `MINNG` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_RI` | `MINNG` | `RI` (Rhode Island) | monthly | 2009-01-01 to 2025-12-01 | 204 |
| `MINNG_SC` | `MINNG` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_SD` | `MINNG` | `SD` (South Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_TN` | `MINNG` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_TX` | `MINNG` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_UT` | `MINNG` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_VA` | `MINNG` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_VT` | `MINNG` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_WA` | `MINNG` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_WI` | `MINNG` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_WV` | `MINNG` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `MINNG_WY` | `MINNG` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_AK` | `NA` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_AL` | `NA` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_AR` | `NA` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_AZ` | `NA` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_CA` | `NA` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_CO` | `NA` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_CT` | `NA` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_DC` | `NA` | `DC` (District of Columbia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_DE` | `NA` | `DE` (Delaware) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_FL` | `NA` | `FL` (Florida) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_GA` | `NA` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_HI` | `NA` | `HI` (Hawaii) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_IA` | `NA` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_ID` | `NA` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_IL` | `NA` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_IN` | `NA` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_KS` | `NA` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_KY` | `NA` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_LA` | `NA` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_MA` | `NA` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_MD` | `NA` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_ME` | `NA` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_MI` | `NA` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_MN` | `NA` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_MO` | `NA` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_MS` | `NA` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_MT` | `NA` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_NC` | `NA` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_ND` | `NA` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_NE` | `NA` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_NH` | `NA` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_NJ` | `NA` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_NM` | `NA` | `NM` (New Mexico) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_NV` | `NA` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_NY` | `NA` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_OH` | `NA` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_OK` | `NA` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_OR` | `NA` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_PA` | `NA` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_RI` | `NA` | `RI` (Rhode Island) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_SC` | `NA` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_SD` | `NA` | `SD` (South Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_TN` | `NA` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_TX` | `NA` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_UT` | `NA` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_VA` | `NA` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_VT` | `NA` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_WA` | `NA` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_WI` | `NA` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_WV` | `NA` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NA_WY` | `NA` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `NATURNQGSP_AK` | `NATURNQGSP` | `AK` (Alaska) | unknown | - | 0 |
| `NATURNQGSP_AL` | `NATURNQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_AR` | `NATURNQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_AZ` | `NATURNQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_CA` | `NATURNQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_CO` | `NATURNQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_CT` | `NATURNQGSP` | `CT` (Connecticut) | unknown | - | 0 |
| `NATURNQGSP_DC` | `NATURNQGSP` | `DC` (District of Columbia) | unknown | - | 0 |
| `NATURNQGSP_DE` | `NATURNQGSP` | `DE` (Delaware) | unknown | - | 0 |
| `NATURNQGSP_FL` | `NATURNQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_GA` | `NATURNQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_HI` | `NATURNQGSP` | `HI` (Hawaii) | unknown | - | 0 |
| `NATURNQGSP_IA` | `NATURNQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_ID` | `NATURNQGSP` | `ID` (Idaho) | unknown | - | 0 |
| `NATURNQGSP_IL` | `NATURNQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_IN` | `NATURNQGSP` | `IN` (Indiana) | unknown | - | 0 |
| `NATURNQGSP_KS` | `NATURNQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_KY` | `NATURNQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_LA` | `NATURNQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_MA` | `NATURNQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 63 |
| `NATURNQGSP_MD` | `NATURNQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_ME` | `NATURNQGSP` | `ME` (Maine) | unknown | - | 0 |
| `NATURNQGSP_MI` | `NATURNQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_MN` | `NATURNQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_MO` | `NATURNQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_MS` | `NATURNQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_MT` | `NATURNQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_NC` | `NATURNQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_ND` | `NATURNQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_NE` | `NATURNQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_NH` | `NATURNQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_NJ` | `NATURNQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_NM` | `NATURNQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_NV` | `NATURNQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_NY` | `NATURNQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_OH` | `NATURNQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_OK` | `NATURNQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_OR` | `NATURNQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_PA` | `NATURNQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_RI` | `NATURNQGSP` | `RI` (Rhode Island) | unknown | - | 0 |
| `NATURNQGSP_SC` | `NATURNQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_SD` | `NATURNQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_TN` | `NATURNQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_TX` | `NATURNQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_UT` | `NATURNQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_VA` | `NATURNQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_VT` | `NATURNQGSP` | `VT` (Vermont) | unknown | - | 0 |
| `NATURNQGSP_WA` | `NATURNQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_WI` | `NATURNQGSP` | `WI` (Wisconsin) | unknown | - | 0 |
| `NATURNQGSP_WV` | `NATURNQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NATURNQGSP_WY` | `NATURNQGSP` | `WY` (Wyoming) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_AK` | `NQGSP` | `AK` (Alaska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_AL` | `NQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_AR` | `NQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_AZ` | `NQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_CA` | `NQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_CO` | `NQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_CT` | `NQGSP` | `CT` (Connecticut) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_DC` | `NQGSP` | `DC` (District of Columbia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_DE` | `NQGSP` | `DE` (Delaware) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_FL` | `NQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_GA` | `NQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_HI` | `NQGSP` | `HI` (Hawaii) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_IA` | `NQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_ID` | `NQGSP` | `ID` (Idaho) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_IL` | `NQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_IN` | `NQGSP` | `IN` (Indiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_KS` | `NQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_KY` | `NQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_LA` | `NQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_MA` | `NQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_MD` | `NQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_ME` | `NQGSP` | `ME` (Maine) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_MI` | `NQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_MN` | `NQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_MO` | `NQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_MS` | `NQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_MT` | `NQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_NC` | `NQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_ND` | `NQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_NE` | `NQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_NH` | `NQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_NJ` | `NQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_NM` | `NQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_NV` | `NQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_NY` | `NQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_OH` | `NQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_OK` | `NQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_OR` | `NQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_PA` | `NQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_RI` | `NQGSP` | `RI` (Rhode Island) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_SC` | `NQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_SD` | `NQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_TN` | `NQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_TX` | `NQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_UT` | `NQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_VA` | `NQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_VT` | `NQGSP` | `VT` (Vermont) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_WA` | `NQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_WI` | `NQGSP` | `WI` (Wisconsin) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_WV` | `NQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `NQGSP_WY` | `NQGSP` | `WY` (Wyoming) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `OTOT_AK` | `OTOT` | `AK` (Alaska) | quarterly | 1950-01-01 to 2025-07-01 | 303 |
| `OTOT_AL` | `OTOT` | `AL` (Alabama) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_AR` | `OTOT` | `AR` (Arkansas) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_AZ` | `OTOT` | `AZ` (Arizona) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_CA` | `OTOT` | `CA` (California) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_CO` | `OTOT` | `CO` (Colorado) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_CT` | `OTOT` | `CT` (Connecticut) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_DC` | `OTOT` | `DC` (District of Columbia) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_DE` | `OTOT` | `DE` (Delaware) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_FL` | `OTOT` | `FL` (Florida) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_GA` | `OTOT` | `GA` (Georgia) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_HI` | `OTOT` | `HI` (Hawaii) | quarterly | 1950-01-01 to 2025-07-01 | 303 |
| `OTOT_IA` | `OTOT` | `IA` (Iowa) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_ID` | `OTOT` | `ID` (Idaho) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_IL` | `OTOT` | `IL` (Illinois) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_IN` | `OTOT` | `IN` (Indiana) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_KS` | `OTOT` | `KS` (Kansas) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_KY` | `OTOT` | `KY` (Kentucky) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_LA` | `OTOT` | `LA` (Louisiana) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_MA` | `OTOT` | `MA` (Massachusetts) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_MD` | `OTOT` | `MD` (Maryland) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_ME` | `OTOT` | `ME` (Maine) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_MI` | `OTOT` | `MI` (Michigan) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_MN` | `OTOT` | `MN` (Minnesota) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_MO` | `OTOT` | `MO` (Missouri) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_MS` | `OTOT` | `MS` (Mississippi) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_MT` | `OTOT` | `MT` (Montana) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_NC` | `OTOT` | `NC` (North Carolina) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_ND` | `OTOT` | `ND` (North Dakota) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_NE` | `OTOT` | `NE` (Nebraska) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_NH` | `OTOT` | `NH` (New Hampshire) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_NJ` | `OTOT` | `NJ` (New Jersey) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_NM` | `OTOT` | `NM` (New Mexico) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_NV` | `OTOT` | `NV` (Nevada) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_NY` | `OTOT` | `NY` (New York) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_OH` | `OTOT` | `OH` (Ohio) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_OK` | `OTOT` | `OK` (Oklahoma) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_OR` | `OTOT` | `OR` (Oregon) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_PA` | `OTOT` | `PA` (Pennsylvania) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_RI` | `OTOT` | `RI` (Rhode Island) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_SC` | `OTOT` | `SC` (South Carolina) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_SD` | `OTOT` | `SD` (South Dakota) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_TN` | `OTOT` | `TN` (Tennessee) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_TX` | `OTOT` | `TX` (Texas) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_UT` | `OTOT` | `UT` (Utah) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_VA` | `OTOT` | `VA` (Virginia) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_VT` | `OTOT` | `VT` (Vermont) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_WA` | `OTOT` | `WA` (Washington) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_WI` | `OTOT` | `WI` (Wisconsin) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_WV` | `OTOT` | `WV` (West Virginia) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `OTOT_WY` | `OTOT` | `WY` (Wyoming) | quarterly | 1948-01-01 to 2025-07-01 | 311 |
| `PARTRATE_AK` | `PARTRATE` | `AK` (Alaska) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_AL` | `PARTRATE` | `AL` (Alabama) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_AR` | `PARTRATE` | `AR` (Arkansas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_AZ` | `PARTRATE` | `AZ` (Arizona) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_CA` | `PARTRATE` | `CA` (California) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_CO` | `PARTRATE` | `CO` (Colorado) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_CT` | `PARTRATE` | `CT` (Connecticut) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_DC` | `PARTRATE` | `DC` (District of Columbia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_DE` | `PARTRATE` | `DE` (Delaware) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_FL` | `PARTRATE` | `FL` (Florida) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_GA` | `PARTRATE` | `GA` (Georgia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_HI` | `PARTRATE` | `HI` (Hawaii) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_IA` | `PARTRATE` | `IA` (Iowa) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_ID` | `PARTRATE` | `ID` (Idaho) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_IL` | `PARTRATE` | `IL` (Illinois) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_IN` | `PARTRATE` | `IN` (Indiana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_KS` | `PARTRATE` | `KS` (Kansas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_KY` | `PARTRATE` | `KY` (Kentucky) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_LA` | `PARTRATE` | `LA` (Louisiana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_MA` | `PARTRATE` | `MA` (Massachusetts) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_MD` | `PARTRATE` | `MD` (Maryland) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_ME` | `PARTRATE` | `ME` (Maine) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_MI` | `PARTRATE` | `MI` (Michigan) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_MN` | `PARTRATE` | `MN` (Minnesota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_MO` | `PARTRATE` | `MO` (Missouri) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_MS` | `PARTRATE` | `MS` (Mississippi) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_MT` | `PARTRATE` | `MT` (Montana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_NC` | `PARTRATE` | `NC` (North Carolina) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_ND` | `PARTRATE` | `ND` (North Dakota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_NE` | `PARTRATE` | `NE` (Nebraska) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_NH` | `PARTRATE` | `NH` (New Hampshire) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_NJ` | `PARTRATE` | `NJ` (New Jersey) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_NM` | `PARTRATE` | `NM` (New Mexico) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_NV` | `PARTRATE` | `NV` (Nevada) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_NY` | `PARTRATE` | `NY` (New York) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_OH` | `PARTRATE` | `OH` (Ohio) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_OK` | `PARTRATE` | `OK` (Oklahoma) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_OR` | `PARTRATE` | `OR` (Oregon) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_PA` | `PARTRATE` | `PA` (Pennsylvania) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_RI` | `PARTRATE` | `RI` (Rhode Island) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_SC` | `PARTRATE` | `SC` (South Carolina) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_SD` | `PARTRATE` | `SD` (South Dakota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_TN` | `PARTRATE` | `TN` (Tennessee) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_TX` | `PARTRATE` | `TX` (Texas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_UT` | `PARTRATE` | `UT` (Utah) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_VA` | `PARTRATE` | `VA` (Virginia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_VT` | `PARTRATE` | `VT` (Vermont) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_WA` | `PARTRATE` | `WA` (Washington) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_WI` | `PARTRATE` | `WI` (Wisconsin) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_WV` | `PARTRATE` | `WV` (West Virginia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PARTRATE_WY` | `PARTRATE` | `WY` (Wyoming) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `PSERV_AK` | `PSERV` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_AL` | `PSERV` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_AR` | `PSERV` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_AZ` | `PSERV` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_CA` | `PSERV` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_CO` | `PSERV` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_CT` | `PSERV` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_DC` | `PSERV` | `DC` (District of Columbia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_DE` | `PSERV` | `DE` (Delaware) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_FL` | `PSERV` | `FL` (Florida) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_GA` | `PSERV` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_HI` | `PSERV` | `HI` (Hawaii) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_IA` | `PSERV` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_ID` | `PSERV` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_IL` | `PSERV` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_IN` | `PSERV` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_KS` | `PSERV` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_KY` | `PSERV` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_LA` | `PSERV` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_MA` | `PSERV` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_MD` | `PSERV` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_ME` | `PSERV` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_MI` | `PSERV` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_MN` | `PSERV` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_MO` | `PSERV` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_MS` | `PSERV` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_MT` | `PSERV` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_NC` | `PSERV` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_ND` | `PSERV` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_NE` | `PSERV` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_NH` | `PSERV` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_NJ` | `PSERV` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_NM` | `PSERV` | `NM` (New Mexico) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_NV` | `PSERV` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_NY` | `PSERV` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_OH` | `PSERV` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_OK` | `PSERV` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_OR` | `PSERV` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_PA` | `PSERV` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_RI` | `PSERV` | `RI` (Rhode Island) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_SC` | `PSERV` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_SD` | `PSERV` | `SD` (South Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_TN` | `PSERV` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_TX` | `PSERV` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_UT` | `PSERV` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_VA` | `PSERV` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_VT` | `PSERV` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_WA` | `PSERV` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_WI` | `PSERV` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_WV` | `PSERV` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERV_WY` | `PSERV` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `PSERVNQGSP_AK` | `PSERVNQGSP` | `AK` (Alaska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_AL` | `PSERVNQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_AR` | `PSERVNQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_AZ` | `PSERVNQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_CA` | `PSERVNQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_CO` | `PSERVNQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_CT` | `PSERVNQGSP` | `CT` (Connecticut) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_DC` | `PSERVNQGSP` | `DC` (District of Columbia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_DE` | `PSERVNQGSP` | `DE` (Delaware) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_FL` | `PSERVNQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_GA` | `PSERVNQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_HI` | `PSERVNQGSP` | `HI` (Hawaii) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_IA` | `PSERVNQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_ID` | `PSERVNQGSP` | `ID` (Idaho) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_IL` | `PSERVNQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_IN` | `PSERVNQGSP` | `IN` (Indiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_KS` | `PSERVNQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_KY` | `PSERVNQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_LA` | `PSERVNQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_MA` | `PSERVNQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_MD` | `PSERVNQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_ME` | `PSERVNQGSP` | `ME` (Maine) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_MI` | `PSERVNQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_MN` | `PSERVNQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_MO` | `PSERVNQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_MS` | `PSERVNQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_MT` | `PSERVNQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_NC` | `PSERVNQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_ND` | `PSERVNQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_NE` | `PSERVNQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_NH` | `PSERVNQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_NJ` | `PSERVNQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_NM` | `PSERVNQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_NV` | `PSERVNQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_NY` | `PSERVNQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_OH` | `PSERVNQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_OK` | `PSERVNQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_OR` | `PSERVNQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_PA` | `PSERVNQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_RI` | `PSERVNQGSP` | `RI` (Rhode Island) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_SC` | `PSERVNQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_SD` | `PSERVNQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_TN` | `PSERVNQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_TX` | `PSERVNQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_UT` | `PSERVNQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_VA` | `PSERVNQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_VT` | `PSERVNQGSP` | `VT` (Vermont) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_WA` | `PSERVNQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_WI` | `PSERVNQGSP` | `WI` (Wisconsin) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_WV` | `PSERVNQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `PSERVNQGSP_WY` | `PSERVNQGSP` | `WY` (Wyoming) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `RENTS_AK` | `RENTS` | `AK` (Alaska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_AL` | `RENTS` | `AL` (Alabama) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_AR` | `RENTS` | `AR` (Arkansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_AZ` | `RENTS` | `AZ` (Arizona) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_CA` | `RENTS` | `CA` (California) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_CO` | `RENTS` | `CO` (Colorado) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_CT` | `RENTS` | `CT` (Connecticut) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_DC` | `RENTS` | `DC` (District of Columbia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_DE` | `RENTS` | `DE` (Delaware) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_FL` | `RENTS` | `FL` (Florida) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_GA` | `RENTS` | `GA` (Georgia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_HI` | `RENTS` | `HI` (Hawaii) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_IA` | `RENTS` | `IA` (Iowa) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_ID` | `RENTS` | `ID` (Idaho) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_IL` | `RENTS` | `IL` (Illinois) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_IN` | `RENTS` | `IN` (Indiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_KS` | `RENTS` | `KS` (Kansas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_KY` | `RENTS` | `KY` (Kentucky) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_LA` | `RENTS` | `LA` (Louisiana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_MA` | `RENTS` | `MA` (Massachusetts) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_MD` | `RENTS` | `MD` (Maryland) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_ME` | `RENTS` | `ME` (Maine) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_MI` | `RENTS` | `MI` (Michigan) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_MN` | `RENTS` | `MN` (Minnesota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_MO` | `RENTS` | `MO` (Missouri) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_MS` | `RENTS` | `MS` (Mississippi) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_MT` | `RENTS` | `MT` (Montana) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_NC` | `RENTS` | `NC` (North Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_ND` | `RENTS` | `ND` (North Dakota) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_NE` | `RENTS` | `NE` (Nebraska) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_NH` | `RENTS` | `NH` (New Hampshire) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_NJ` | `RENTS` | `NJ` (New Jersey) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_NM` | `RENTS` | `NM` (New Mexico) | unknown | - | 0 |
| `RENTS_NV` | `RENTS` | `NV` (Nevada) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_NY` | `RENTS` | `NY` (New York) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_OH` | `RENTS` | `OH` (Ohio) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_OK` | `RENTS` | `OK` (Oklahoma) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_OR` | `RENTS` | `OR` (Oregon) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_PA` | `RENTS` | `PA` (Pennsylvania) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_RI` | `RENTS` | `RI` (Rhode Island) | unknown | - | 0 |
| `RENTS_SC` | `RENTS` | `SC` (South Carolina) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_SD` | `RENTS` | `SD` (South Dakota) | unknown | - | 0 |
| `RENTS_TN` | `RENTS` | `TN` (Tennessee) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_TX` | `RENTS` | `TX` (Texas) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_UT` | `RENTS` | `UT` (Utah) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_VA` | `RENTS` | `VA` (Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_VT` | `RENTS` | `VT` (Vermont) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_WA` | `RENTS` | `WA` (Washington) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_WI` | `RENTS` | `WI` (Wisconsin) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_WV` | `RENTS` | `WV` (West Virginia) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `RENTS_WY` | `RENTS` | `WY` (Wyoming) | monthly | 1990-01-01 to 2025-12-01 | 432 |
| `STHPI_AK` | `STHPI` | `AK` (Alaska) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_AL` | `STHPI` | `AL` (Alabama) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_AR` | `STHPI` | `AR` (Arkansas) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_AZ` | `STHPI` | `AZ` (Arizona) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_CA` | `STHPI` | `CA` (California) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_CO` | `STHPI` | `CO` (Colorado) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_CT` | `STHPI` | `CT` (Connecticut) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_DC` | `STHPI` | `DC` (District of Columbia) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_DE` | `STHPI` | `DE` (Delaware) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_FL` | `STHPI` | `FL` (Florida) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_GA` | `STHPI` | `GA` (Georgia) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_HI` | `STHPI` | `HI` (Hawaii) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_IA` | `STHPI` | `IA` (Iowa) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_ID` | `STHPI` | `ID` (Idaho) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_IL` | `STHPI` | `IL` (Illinois) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_IN` | `STHPI` | `IN` (Indiana) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_KS` | `STHPI` | `KS` (Kansas) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_KY` | `STHPI` | `KY` (Kentucky) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_LA` | `STHPI` | `LA` (Louisiana) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_MA` | `STHPI` | `MA` (Massachusetts) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_MD` | `STHPI` | `MD` (Maryland) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_ME` | `STHPI` | `ME` (Maine) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_MI` | `STHPI` | `MI` (Michigan) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_MN` | `STHPI` | `MN` (Minnesota) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_MO` | `STHPI` | `MO` (Missouri) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_MS` | `STHPI` | `MS` (Mississippi) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_MT` | `STHPI` | `MT` (Montana) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_NC` | `STHPI` | `NC` (North Carolina) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_ND` | `STHPI` | `ND` (North Dakota) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_NE` | `STHPI` | `NE` (Nebraska) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_NH` | `STHPI` | `NH` (New Hampshire) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_NJ` | `STHPI` | `NJ` (New Jersey) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_NM` | `STHPI` | `NM` (New Mexico) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_NV` | `STHPI` | `NV` (Nevada) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_NY` | `STHPI` | `NY` (New York) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_OH` | `STHPI` | `OH` (Ohio) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_OK` | `STHPI` | `OK` (Oklahoma) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_OR` | `STHPI` | `OR` (Oregon) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_PA` | `STHPI` | `PA` (Pennsylvania) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_RI` | `STHPI` | `RI` (Rhode Island) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_SC` | `STHPI` | `SC` (South Carolina) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_SD` | `STHPI` | `SD` (South Dakota) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_TN` | `STHPI` | `TN` (Tennessee) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_TX` | `STHPI` | `TX` (Texas) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_UT` | `STHPI` | `UT` (Utah) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_VA` | `STHPI` | `VA` (Virginia) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_VT` | `STHPI` | `VT` (Vermont) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_WA` | `STHPI` | `WA` (Washington) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_WI` | `STHPI` | `WI` (Wisconsin) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_WV` | `STHPI` | `WV` (West Virginia) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `STHPI_WY` | `STHPI` | `WY` (Wyoming) | quarterly | 1975-01-01 to 2025-10-01 | 204 |
| `UR_AK` | `UR` | `AK` (Alaska) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_AL` | `UR` | `AL` (Alabama) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_AR` | `UR` | `AR` (Arkansas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_AZ` | `UR` | `AZ` (Arizona) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_CA` | `UR` | `CA` (California) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_CO` | `UR` | `CO` (Colorado) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_CT` | `UR` | `CT` (Connecticut) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_DC` | `UR` | `DC` (District of Columbia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_DE` | `UR` | `DE` (Delaware) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_FL` | `UR` | `FL` (Florida) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_GA` | `UR` | `GA` (Georgia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_HI` | `UR` | `HI` (Hawaii) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_IA` | `UR` | `IA` (Iowa) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_ID` | `UR` | `ID` (Idaho) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_IL` | `UR` | `IL` (Illinois) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_IN` | `UR` | `IN` (Indiana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_KS` | `UR` | `KS` (Kansas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_KY` | `UR` | `KY` (Kentucky) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_LA` | `UR` | `LA` (Louisiana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_MA` | `UR` | `MA` (Massachusetts) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_MD` | `UR` | `MD` (Maryland) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_ME` | `UR` | `ME` (Maine) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_MI` | `UR` | `MI` (Michigan) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_MN` | `UR` | `MN` (Minnesota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_MO` | `UR` | `MO` (Missouri) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_MS` | `UR` | `MS` (Mississippi) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_MT` | `UR` | `MT` (Montana) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_NC` | `UR` | `NC` (North Carolina) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_ND` | `UR` | `ND` (North Dakota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_NE` | `UR` | `NE` (Nebraska) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_NH` | `UR` | `NH` (New Hampshire) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_NJ` | `UR` | `NJ` (New Jersey) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_NM` | `UR` | `NM` (New Mexico) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_NV` | `UR` | `NV` (Nevada) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_NY` | `UR` | `NY` (New York) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_OH` | `UR` | `OH` (Ohio) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_OK` | `UR` | `OK` (Oklahoma) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_OR` | `UR` | `OR` (Oregon) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_PA` | `UR` | `PA` (Pennsylvania) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_RI` | `UR` | `RI` (Rhode Island) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_SC` | `UR` | `SC` (South Carolina) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_SD` | `UR` | `SD` (South Dakota) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_TN` | `UR` | `TN` (Tennessee) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_TX` | `UR` | `TX` (Texas) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_UT` | `UR` | `UT` (Utah) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_VA` | `UR` | `VA` (Virginia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_VT` | `UR` | `VT` (Vermont) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_WA` | `UR` | `WA` (Washington) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_WI` | `UR` | `WI` (Wisconsin) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_WV` | `UR` | `WV` (West Virginia) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UR_WY` | `UR` | `WY` (Wyoming) | monthly | 1976-01-01 to 2025-08-01 | 596 |
| `UTILNQGSP_AK` | `UTILNQGSP` | `AK` (Alaska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_AL` | `UTILNQGSP` | `AL` (Alabama) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_AR` | `UTILNQGSP` | `AR` (Arkansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_AZ` | `UTILNQGSP` | `AZ` (Arizona) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_CA` | `UTILNQGSP` | `CA` (California) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_CO` | `UTILNQGSP` | `CO` (Colorado) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_CT` | `UTILNQGSP` | `CT` (Connecticut) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_DC` | `UTILNQGSP` | `DC` (District of Columbia) | quarterly | 2005-01-01 to 2025-07-01 | 79 |
| `UTILNQGSP_DE` | `UTILNQGSP` | `DE` (Delaware) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_FL` | `UTILNQGSP` | `FL` (Florida) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_GA` | `UTILNQGSP` | `GA` (Georgia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_HI` | `UTILNQGSP` | `HI` (Hawaii) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_IA` | `UTILNQGSP` | `IA` (Iowa) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_ID` | `UTILNQGSP` | `ID` (Idaho) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_IL` | `UTILNQGSP` | `IL` (Illinois) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_IN` | `UTILNQGSP` | `IN` (Indiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_KS` | `UTILNQGSP` | `KS` (Kansas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_KY` | `UTILNQGSP` | `KY` (Kentucky) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_LA` | `UTILNQGSP` | `LA` (Louisiana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_MA` | `UTILNQGSP` | `MA` (Massachusetts) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_MD` | `UTILNQGSP` | `MD` (Maryland) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_ME` | `UTILNQGSP` | `ME` (Maine) | quarterly | 2005-01-01 to 2025-07-01 | 79 |
| `UTILNQGSP_MI` | `UTILNQGSP` | `MI` (Michigan) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_MN` | `UTILNQGSP` | `MN` (Minnesota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_MO` | `UTILNQGSP` | `MO` (Missouri) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_MS` | `UTILNQGSP` | `MS` (Mississippi) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_MT` | `UTILNQGSP` | `MT` (Montana) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_NC` | `UTILNQGSP` | `NC` (North Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_ND` | `UTILNQGSP` | `ND` (North Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_NE` | `UTILNQGSP` | `NE` (Nebraska) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_NH` | `UTILNQGSP` | `NH` (New Hampshire) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_NJ` | `UTILNQGSP` | `NJ` (New Jersey) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_NM` | `UTILNQGSP` | `NM` (New Mexico) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_NV` | `UTILNQGSP` | `NV` (Nevada) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_NY` | `UTILNQGSP` | `NY` (New York) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_OH` | `UTILNQGSP` | `OH` (Ohio) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_OK` | `UTILNQGSP` | `OK` (Oklahoma) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_OR` | `UTILNQGSP` | `OR` (Oregon) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_PA` | `UTILNQGSP` | `PA` (Pennsylvania) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_RI` | `UTILNQGSP` | `RI` (Rhode Island) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_SC` | `UTILNQGSP` | `SC` (South Carolina) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_SD` | `UTILNQGSP` | `SD` (South Dakota) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_TN` | `UTILNQGSP` | `TN` (Tennessee) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_TX` | `UTILNQGSP` | `TX` (Texas) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_UT` | `UTILNQGSP` | `UT` (Utah) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_VA` | `UTILNQGSP` | `VA` (Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_VT` | `UTILNQGSP` | `VT` (Vermont) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_WA` | `UTILNQGSP` | `WA` (Washington) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_WI` | `UTILNQGSP` | `WI` (Wisconsin) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_WV` | `UTILNQGSP` | `WV` (West Virginia) | quarterly | 2005-01-01 to 2025-07-01 | 83 |
| `UTILNQGSP_WY` | `UTILNQGSP` | `WY` (Wyoming) | quarterly | 2005-01-01 to 2025-07-01 | 75 |
