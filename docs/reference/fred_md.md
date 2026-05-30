# FRED-MD

[Back to reference](index.md)

FRED-MD is the package's monthly national macroeconomic data source. In
`macroforecast`, it is loaded through `mf.data.load_fred_md()` and returned as a
canonical `DataBundle`.

The official source is the St. Louis Fed FRED-MD/FRED-QD page:
<https://www.stlouisfed.org/research/economists/mccracken/fred-databases>.

## When To Use

Use FRED-MD when the forecasting target, panel, or evaluation unit is monthly.
Typical examples are monthly industrial production, inflation, unemployment,
or state-level monthly outcomes that need national controls.

For combined national and state-level analysis, use:

```python
bundle = mf.data.load_fred_md_sd(
    states=["CA", "TX"],
    variables=["UR", "ICLAIMS"],
)
```

This returns `dataset="fred_md+fred_sd"` and defaults to `frequency="monthly"`.
That is the recommended combined dataset for monthly state analysis.

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
| `vintage` | <code>str &#124; None</code> | `None` | Vintage label in `YYYY-MM` form. `None` loads the current official CSV. |
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
| `bundle.metadata["transform_codes"]` | Official FRED-MD t-code map parsed from the CSV transform row. |
| `bundle.panel.attrs["macroforecast_transform_codes"]` | Same t-code map for pandas-native handoff. |

The loader stores raw-file provenance in `metadata["artifact"]`: source URL or
local path, cache path, file format, download/copy time, SHA-256, file size, and
cache-hit status.

## Transform Codes

FRED-MD provides official transformation codes in the raw CSV. The loader does
not manually maintain a separate package table. It parses the CSV transform row
and stores the result in `metadata["transform_codes"]`.

`mf.preprocessing.reprocess(..., transform="official")` uses that map.

## Frequency Contract

FRED-MD is monthly. `metadata["frequency"] == "monthly"` and default data-spec
horizons are `(1, 3, 6, 12)`.

If FRED-MD is combined into a quarterly panel, `mf.data.combine()` allows it,
but this is not the recommended default. Prefer FRED-QD for quarterly targets.
When this conversion occurs, the combined bundle records a note in
`metadata["parse_notes"]` and records converted columns in
`metadata["frequency_conversion_warnings"]`.

## Combined With FRED-SD

`load_fred_md_sd()` loads FRED-MD and FRED-SD and combines them.

```python
bundle = mf.data.load_fred_md_sd(
    states=["CA", "TX"],
    variables=["UR", "ICLAIMS", "NQGSP"],
    frequency="monthly",
)
```

FRED-SD includes both monthly and quarterly series. If a selected FRED-SD
series is quarterly, the default rule `quarterly_to_monthly="repeat_within_quarter"`
assigns the quarterly value to each month in that quarter. The function emits
`UserWarning` and records the conversion in
`metadata["frequency_conversion_warnings"]`.

## Example

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()
spec = mf.data.spec(bundle, target="INDPRO", horizons=[1, 3, 6, 12])
processed = mf.preprocessing.reprocess(spec)
```
