# FRED-SD

[Back to FRED Datasets](index.md)

FRED-SD is the package's state-level macroeconomic data source. In
`macroforecast`, it is loaded through `mf.data.load_fred_sd()` and returned as a
canonical `DataBundle`.

The official source is the St. Louis Fed FRED-SD page:
<https://www.stlouisfed.org/research/economists/owyang/fred-sd>.

## What FRED-SD Contains

FRED-SD is state-level and mixed-frequency. The official FRED-SD page describes
the database as state-level monthly and quarterly observations. The package
loads the official Data by Series workbook and infers each column's native
frequency from its observed dates.

This page is intentionally kept separate from the national FRED-MD/FRED-QD
pages. FRED-SD needs its own audit because state-series frequencies and edge
cases can differ by variable and state, including weekly or otherwise
non-monthly/non-quarterly observed spacing in raw files.

## Current Snapshot Checked For This Page

This page was checked against the St. Louis Fed FRED-SD page and the latest
official Data by Series workbook on 2026-06-01.

| Item | Checked value |
| --- | --- |
| Latest workbook label | `series-2026-04.xlsx` |
| Latest workbook URL | <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-04.xlsx> |
| Panel shape after package load | 1133 rows x 1428 state-series columns |
| State count | 51 |
| FRED-SD variable count | 28 |
| Last date present in panel | 2026-04 |
| Official page | <https://www.stlouisfed.org/research/economists/owyang/fred-sd> |

The latest workbook has these native frequency counts:

| Native frequency | Number of state-series columns |
| --- | ---: |
| monthly | 861 |
| quarterly | 546 |
| unknown | 21 |

These are aggregate counts across all state-series columns. Because FRED-SD is
state-level, the state axis must also be checked. The latest workbook has 28
state-series columns for every state/DC, but the monthly/quarterly/unknown mix
differs by state when some state-variable cells have too few observations for
reliable frequency inference.

The latest workbook has these date-anchor counts:

| Date anchor | Number of state-series columns | Meaning |
| --- | ---: | --- |
| `month_start` | 810 | Monthly observations dated on the first day of the month. |
| `monthly_weekday_anchor` | 51 | Monthly observations dated on a repeated weekday pattern rather than month start. In the latest workbook this is `ICLAIMS`. |
| `quarter_start` | 546 | Quarterly observations dated on the first day of the quarter. |
| `none` | 21 | Too few non-missing observations to infer an anchor. |

The `unknown` frequency category appears when a state-series has too few
observed points to infer a reliable date spacing. `ICLAIMS` is a special case:
its dates look weekly because they fall on Saturdays, but each state has one
observation per month in the latest workbook, so the package records
`native_frequency="monthly"` and `date_anchor="monthly_weekday_anchor"`.

## Loader

```python
macroforecast.data.load_fred_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
) -> DataBundle
```

## Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `vintage` | <code>str &#124; None</code> | `None` | Vintage label in `YYYY-MM` form. `None` loads the current official workbook. |
| `force` | `bool` | `False` | Re-download or re-copy the raw file even if it already exists in cache. |
| `cache_root` | path-like or `None` | `None` | Root directory for raw-file cache and manifest. |
| `local_source` | path-like or `None` | `None` | Local FRED-SD workbook or CSV source. |
| `states` | <code>list[str] &#124; None</code> | `None` | State subset. Example: `["CA", "TX"]`. |
| `variables` | <code>list[str] &#124; None</code> | `None` | FRED-SD variable subset. Example: `["UR", "ICLAIMS"]`. |

## Output

`load_fred_sd()` returns a `DataBundle`.

| Field | Value |
| --- | --- |
| `bundle.panel` | Wide state-level panel with columns such as `UR_CA` or `NQGSP_TX`. |
| `bundle.metadata["dataset"]` | `"fred_sd"` |
| `bundle.metadata["frequency"]` | `"monthly"`, `"quarterly"`, `"weekly"`, `"annual"`, `"mixed"`, or `"unknown"` depending on loaded columns. |
| `bundle.metadata["native_frequency_by_column"]` | Per-column native frequency map. |
| `bundle.metadata["native_frequency_counts"]` | Per-column native frequency counts. |
| `bundle.metadata["date_anchor_by_column"]` | Per-column date-anchor map. |
| `bundle.metadata["date_anchor_counts"]` | Per-column date-anchor counts. |
| `bundle.metadata["state_summary"]` | Per-state coverage, frequency counts, date-anchor counts, observed span, and unknown variables. |
| `bundle.metadata["transform_codes"]` | `{}` because FRED-SD has no official t-code map. |

FRED-SD series metadata is stored in:

```python
bundle.panel.attrs["macrocast_reports"]["fred_sd_series_metadata"]
```

That report contains:

| Key | Meaning |
| --- | --- |
| `series_count` | Number of loaded state-series columns. |
| `state_count` | Number of states represented in the loaded panel. |
| `sd_variable_count` | Number of FRED-SD variable groups represented. |
| `native_frequency_counts` | Count of monthly, quarterly, weekly, annual, irregular, unknown, or other inferred frequencies. |
| `date_anchor_counts` | Count of date-anchor patterns such as `month_start`, `quarter_start`, or `monthly_weekday_anchor`. |
| `state_summary` | State-level coverage summary with frequency and date-anchor counts by state. |
| `series` | Row-level metadata for each state-series column. |

Each `series` row contains `column`, `sd_variable`, `state`, `source_sheet`,
`native_frequency`, `date_anchor`, `observed_start`, `observed_end`, and
`non_missing_observation_count`.

Each `state_summary` row contains `state`, `series_count`, `sd_variable_count`,
`native_frequency_counts`, `date_anchor_counts`, `observed_start`,
`observed_end`, and `unknown_variables`.

## State Coverage In Latest Workbook

This table is computed from `series-2026-04.xlsx`. The key audit surface is not
only the aggregate FRED-SD panel shape, but also whether each state/DC has the
same number of columns and which state-variable pairs fall into `unknown`
frequency.

| State | State-series columns | Monthly | Quarterly | Unknown | Date-anchor pattern | Earliest obs. | Latest obs. | Unknown variables |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `AK` | 28 | 17 | 10 | 1 | month_start: 16, monthly_weekday_anchor: 1, none: 1, quarter_start: 10 | 1950-01-01 | 2026-04-04 | NATURNQGSP |
| `AL` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `AR` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `AZ` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `CA` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `CO` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `CT` | 28 | 17 | 10 | 1 | month_start: 16, monthly_weekday_anchor: 1, none: 1, quarter_start: 10 | 1948-01-01 | 2026-04-04 | NATURNQGSP |
| `DC` | 28 | 16 | 8 | 4 | month_start: 15, monthly_weekday_anchor: 1, none: 4, quarter_start: 8 | 1948-01-01 | 2026-04-04 | CONSTNQGSP, MANNQGSP, MINNG, NATURNQGSP |
| `DE` | 28 | 16 | 10 | 2 | month_start: 15, monthly_weekday_anchor: 1, none: 2, quarter_start: 10 | 1948-01-01 | 2026-04-04 | MINNG, NATURNQGSP |
| `FL` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `GA` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `HI` | 28 | 16 | 10 | 2 | month_start: 15, monthly_weekday_anchor: 1, none: 2, quarter_start: 10 | 1950-01-01 | 2026-04-04 | MINNG, NATURNQGSP |
| `IA` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `ID` | 28 | 17 | 10 | 1 | month_start: 16, monthly_weekday_anchor: 1, none: 1, quarter_start: 10 | 1948-01-01 | 2026-04-04 | NATURNQGSP |
| `IL` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `IN` | 28 | 17 | 10 | 1 | month_start: 16, monthly_weekday_anchor: 1, none: 1, quarter_start: 10 | 1948-01-01 | 2026-04-04 | NATURNQGSP |
| `KS` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `KY` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `LA` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `MA` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `MD` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `ME` | 28 | 17 | 10 | 1 | month_start: 16, monthly_weekday_anchor: 1, none: 1, quarter_start: 10 | 1948-01-01 | 2026-04-04 | NATURNQGSP |
| `MI` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `MN` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `MO` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `MS` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `MT` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `NC` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `ND` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `NE` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `NH` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `NJ` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `NM` | 28 | 16 | 11 | 1 | month_start: 15, monthly_weekday_anchor: 1, none: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | RENTS |
| `NV` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `NY` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `OH` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `OK` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `OR` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `PA` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `RI` | 28 | 16 | 9 | 3 | month_start: 15, monthly_weekday_anchor: 1, none: 3, quarter_start: 9 | 1948-01-01 | 2026-04-04 | CONSTNQGSP, NATURNQGSP, RENTS |
| `SC` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `SD` | 28 | 16 | 11 | 1 | month_start: 15, monthly_weekday_anchor: 1, none: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | RENTS |
| `TN` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `TX` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `UT` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `VA` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `VT` | 28 | 17 | 10 | 1 | month_start: 16, monthly_weekday_anchor: 1, none: 1, quarter_start: 10 | 1948-01-01 | 2026-04-04 | NATURNQGSP |
| `WA` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `WI` | 28 | 17 | 10 | 1 | month_start: 16, monthly_weekday_anchor: 1, none: 1, quarter_start: 10 | 1948-01-01 | 2026-04-04 | NATURNQGSP |
| `WV` | 28 | 17 | 11 | 0 | month_start: 16, monthly_weekday_anchor: 1, quarter_start: 11 | 1948-01-01 | 2026-04-04 | - |
| `WY` | 28 | 17 | 10 | 1 | month_start: 16, monthly_weekday_anchor: 1, none: 1, quarter_start: 10 | 1948-01-01 | 2026-04-04 | MANNQGSP |

## Variable Coverage In Latest Workbook

This table is computed from `series-2026-04.xlsx`. Frequency and date-anchor
patterns are column counts across states. Start/end dates are the widest
observed range across states for that variable. `Unknown states` lists states
whose state-series has too few observations for reliable frequency inference.

| Variable | Native frequency pattern | Date-anchor pattern | States | Earliest obs. | Latest obs. | Non-missing obs. | Unknown states |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| `BPPRIVSA` | monthly: 51 | month_start: 51 | 51 | 1988-01-01 | 2025-08-01 | 23052 | - |
| `CONS` | monthly: 51 | month_start: 51 | 51 | 1990-01-01 | 2026-02-01 | 22034 | - |
| `CONSTNQGSP` | quarterly: 49, unknown: 2 | none: 2, quarter_start: 49 | 51 | 2005-01-01 | 2025-10-01 | 4112 | DC, RI |
| `EXPORTS` | monthly: 51 | month_start: 51 | 51 | 1995-08-01 | 2026-02-01 | 18717 | - |
| `FIRE` | monthly: 51 | month_start: 51 | 51 | 1990-01-01 | 2025-12-01 | 22032 | - |
| `FIRENQGSP` | quarterly: 51 | quarter_start: 51 | 51 | 2005-01-01 | 2025-10-01 | 4284 | - |
| `GOVNQGSP` | quarterly: 51 | quarter_start: 51 | 51 | 2005-01-01 | 2025-10-01 | 4284 | - |
| `GOVT` | monthly: 51 | month_start: 51 | 51 | 1990-01-01 | 2025-12-01 | 22032 | - |
| `ICLAIMS` | monthly: 51 | monthly_weekday_anchor: 51 | 51 | 1985-09-28 | 2026-04-04 | 24662 | - |
| `IMPORTS` | monthly: 51 | month_start: 51 | 51 | 2008-01-01 | 2026-02-01 | 11118 | - |
| `INFO` | monthly: 51 | month_start: 51 | 51 | 1990-01-01 | 2025-12-01 | 22032 | - |
| `INFONQGSP` | quarterly: 51 | quarter_start: 51 | 51 | 2005-01-01 | 2025-10-01 | 4244 | - |
| `LF` | monthly: 51 | month_start: 51 | 51 | 1976-01-01 | 2025-08-01 | 30396 | - |
| `MANNQGSP` | quarterly: 49, unknown: 2 | none: 2, quarter_start: 49 | 51 | 2005-01-01 | 2025-10-01 | 4100 | DC, WY |
| `MFG` | monthly: 51 | month_start: 51 | 51 | 1990-01-01 | 2025-12-01 | 22032 | - |
| `MFGHRS` | monthly: 51 | month_start: 51 | 51 | 2007-01-01 | 2025-12-01 | 11628 | - |
| `MINNG` | monthly: 48, unknown: 3 | month_start: 48, none: 3 | 51 | 1990-01-01 | 2025-12-01 | 20364 | DC, DE, HI |
| `NA` | monthly: 51 | month_start: 51 | 51 | 1990-01-01 | 2026-02-01 | 22048 | - |
| `NATURNQGSP` | quarterly: 40, unknown: 11 | none: 11, quarter_start: 40 | 51 | 2005-01-01 | 2025-10-01 | 3340 | AK, CT, DC, DE, HI, ID, IN, ME, RI, VT, WI |
| `NQGSP` | quarterly: 51 | quarter_start: 51 | 51 | 2005-01-01 | 2025-10-01 | 4284 | - |
| `OTOT` | quarterly: 51 | quarter_start: 51 | 51 | 1948-01-01 | 2025-10-01 | 15896 | - |
| `PARTRATE` | monthly: 51 | month_start: 51 | 51 | 1976-01-01 | 2025-08-01 | 30396 | - |
| `PSERV` | monthly: 51 | month_start: 51 | 51 | 1990-01-01 | 2025-12-01 | 22032 | - |
| `PSERVNQGSP` | quarterly: 51 | quarter_start: 51 | 51 | 2005-01-01 | 2025-10-01 | 4284 | - |
| `RENTS` | monthly: 48, unknown: 3 | month_start: 48, none: 3 | 51 | 1990-01-01 | 2025-12-01 | 20736 | NM, RI, SD |
| `STHPI` | quarterly: 51 | quarter_start: 51 | 51 | 1975-01-01 | 2025-10-01 | 10404 | - |
| `UR` | monthly: 51 | month_start: 51 | 51 | 1976-01-01 | 2025-08-01 | 30396 | - |
| `UTILNQGSP` | quarterly: 51 | quarter_start: 51 | 51 | 2005-01-01 | 2025-10-01 | 4268 | - |

The official FRED-SD page includes a correction noting that `OTOT` and `STHPI`
were previously described as monthly, but are quarterly. The latest workbook
confirms both as quarterly in all 51 states.

## No Official T-Codes

FRED-SD does not provide official stationarity transformation codes. Therefore:

```python
mf.preprocessing.reprocess(fred_sd_bundle)
```

raises unless the user explicitly chooses `transform="none"` or supplies custom
codes.

Recommended explicit paths:

```python
processed = mf.preprocessing.reprocess(fred_sd_bundle, transform="none")
```

or:

```python
codes, provenance = mf.preprocessing.fred_sd_transform_codes(
    fred_sd_bundle,
    variable_codes={"UR": 2, "ICLAIMS": 5},
    return_table=True,
)

processed = mf.preprocessing.reprocess(
    fred_sd_bundle,
    transform="custom",
    transform_codes=codes,
)
```

`fred_sd_transform_codes()` can use user choices and package suggestions based
on national FRED-MD/FRED-QD analogs. These are suggestions, not official
metadata. The provenance table uses `suggestion_confidence` to distinguish:

| Value | Meaning |
| --- | --- |
| `user` | The code was supplied by the user. |
| `high` | Built-in national-analog suggestion with close variable match. |
| `medium` | Built-in national-analog suggestion with weaker match. |
| `none` | No t-code was assigned. |

## Related Combined Loaders

Use FRED-SD alone when the analysis only needs state-level variables. Combined
national/state loaders are documented separately:

- [FRED-MD + FRED-SD](fred_md_sd.md) for monthly state analysis with national controls.
- [FRED-QD + FRED-SD](fred_qd_sd.md) for quarterly state analysis with national controls.

FRED-SD state series can have different observed frequencies and date anchors.
The current loader records that explicitly:

- `native_frequency_by_column` records whether each state column is monthly,
  quarterly, weekly, or unknown before any combination.
- `date_anchor_by_column` records whether dates are first-of-month,
  first-of-quarter, monthly weekday/Saturday-style anchors, or missing.
- combined loaders preserve those maps and raise/record conversion warnings
  when selected FRED-SD columns are aligned to the MD or QD output frequency.

Use the combined pages for the default monthly and quarterly policies. Use
`frequency="native"` or `mf.data.align_frequency()` when you need to inspect or
manually govern unusual state-series timing.

## Example

```python
import macroforecast as mf

sd = mf.data.load_fred_sd(states=["CA", "TX"], variables=["UR", "ICLAIMS"])
codes = mf.preprocessing.fred_sd_transform_codes(
    sd,
    variable_codes={"UR": 2, "ICLAIMS": 5},
)
processed = mf.preprocessing.reprocess(
    sd,
    transform="custom",
    transform_codes=codes,
)
```

## Official URLs

| Source | URL |
| --- | --- |
| FRED-SD official page | <https://www.stlouisfed.org/research/economists/owyang/fred-sd> |
| Latest Data by Series workbook checked here | <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-04.xlsx> |
