# FRED-SD

[Back to reference](index.md)

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

The current official workbook checked during this implementation pass had:

| Native frequency | Number of state-series columns |
| --- | ---: |
| monthly | 861 |
| quarterly | 546 |
| unknown | 21 |

The `unknown` category appears when a state-series has too few observed points
to infer a reliable date spacing.

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
| `bundle.metadata["source_family"]` | `"fred-sd"` |
| `bundle.metadata["frequency"]` | `"state_monthly"` |
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
| `native_frequency_counts` | Count of monthly, quarterly, unknown, or other inferred frequencies. |
| `series` | Row-level metadata for each state-series column. |

Each `series` row contains `column`, `sd_variable`, `state`, `source_sheet`,
`native_frequency`, `observed_start`, `observed_end`, and
`non_missing_observation_count`.

## Monthly Variables

The current official workbook check identified these variables as monthly in
all loaded states:

| Variable | Meaning in package |
| --- | --- |
| `BPPRIVSA` | Building permits, private housing units, seasonally adjusted. |
| `CONS` | Construction employment. |
| `EXPORTS` | State exports. |
| `FIRE` | Financial activities employment. |
| `GOVT` | Government employment. |
| `ICLAIMS` | Initial unemployment insurance claims. |
| `IMPORTS` | State imports. |
| `INFO` | Information employment. |
| `LF` | Labor force. |
| `MFG` | Manufacturing employment. |
| `MFGHRS` | Manufacturing hours. |
| `NA` | Nonfarm payroll employment. |
| `PARTRATE` | Labor force participation rate. |
| `PSERV` | Private service-providing employment. |
| `UR` | Unemployment rate. |

## Quarterly Variables

The current official workbook check identified these variables as quarterly in
all loaded states:

| Variable | Meaning in package |
| --- | --- |
| `FIRENQGSP` | Financial activities contribution to nominal GSP. |
| `GOVNQGSP` | Government contribution to nominal GSP. |
| `INFONQGSP` | Information contribution to nominal GSP. |
| `NQGSP` | Nominal gross state product. |
| `OTOT` | Total personal income. |
| `PSERVNQGSP` | Private service-providing contribution to nominal GSP. |
| `STHPI` | State house price index. |
| `UTILNQGSP` | Utilities contribution to nominal GSP. |

The official FRED-SD page includes a correction noting that `OTOT` and `STHPI`
were previously described as monthly, but are quarterly.

## Variables With Some Unknown State-Series

Some variables are mostly monthly or quarterly, but have a small number of
state-series columns with too few observations to infer frequency in the
current workbook check.

| Variable | Observed pattern |
| --- | --- |
| `CONSTNQGSP` | Mostly quarterly, with some unknown state-series. |
| `MANNQGSP` | Mostly quarterly, with some unknown state-series. |
| `MINNG` | Mostly monthly, with some unknown state-series. |
| `NATURNQGSP` | Mostly quarterly, with some unknown state-series. |
| `RENTS` | Mostly monthly, with some unknown state-series. |

The loader records the exact per-column result in `fred_sd_series_metadata`.
Downstream alignment uses that per-column metadata first and falls back to
observed-date inference only when needed.

## No Official T-Codes

FRED-SD does not provide official stationarity transformation codes. Therefore:

```python
mf.preprocessing.preprocess(fred_sd_bundle)
```

raises unless the user explicitly chooses `transform="none"` or supplies custom
codes.

Recommended explicit paths:

```python
processed = mf.preprocessing.preprocess(fred_sd_bundle, transform="none")
```

or:

```python
codes, provenance = mf.preprocessing.fred_sd_transform_codes(
    fred_sd_bundle,
    variable_codes={"UR": 2, "ICLAIMS": 5},
    return_table=True,
)

processed = mf.preprocessing.preprocess(
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

## Combined With National Data

Use FRED-SD alone when the analysis only needs state-level variables. Use a
combined loader when national controls should be attached.

Monthly state analysis:

```python
bundle = mf.data.load_fred_md_sd(
    states=["CA", "TX"],
    variables=["UR", "ICLAIMS", "NQGSP"],
)
```

Quarterly state analysis:

```python
bundle = mf.data.load_fred_qd_sd(
    states=["CA", "TX"],
    variables=["UR", "NQGSP"],
)
```

If FRED-SD native frequencies do not match the combined panel frequency, the
data module emits `UserWarning` and records the conversion in
`metadata["frequency_conversion_warnings"]`.

Examples:

- `fred_md+fred_sd` monthly panel with quarterly `NQGSP_CA`: quarterly to
  monthly using `quarterly_to_monthly`.
- `fred_qd+fred_sd` quarterly panel with monthly `UR_CA`: monthly to quarterly
  using `monthly_to_quarterly`.

## Example

```python
import macroforecast as mf

sd = mf.data.load_fred_sd(states=["CA", "TX"], variables=["UR", "ICLAIMS"])
codes = mf.preprocessing.fred_sd_transform_codes(
    sd,
    variable_codes={"UR": 2, "ICLAIMS": 5},
)
processed = mf.preprocessing.preprocess(
    sd,
    transform="custom",
    transform_codes=codes,
)
```
