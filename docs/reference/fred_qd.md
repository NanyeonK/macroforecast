# FRED-QD

[Back to reference](index.md)

FRED-QD is the package's quarterly national macroeconomic data source. In
`macroforecast`, it is loaded through `mf.data.load_fred_qd()` and returned as a
canonical `DataBundle`.

The official source is the St. Louis Fed FRED-MD/FRED-QD page:
<https://www.stlouisfed.org/research/economists/mccracken/fred-databases>.

## When To Use

Use FRED-QD when the target, outcome, or evaluation unit is quarterly. This is
the natural source for quarterly national controls, especially when the state
panel is also being collapsed to quarterly outcomes.

For combined national and state-level analysis, use:

```python
bundle = mf.data.load_fred_qd_sd(
    states=["CA", "TX"],
    variables=["UR", "NQGSP"],
)
```

This returns `dataset="fred_qd+fred_sd"` and defaults to
`frequency="quarterly"`. That is the recommended combined dataset for quarterly
state-level analysis.

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
| `vintage` | <code>str &#124; None</code> | `None` | Vintage label in `YYYY-MM` form. `None` loads the current official CSV. |
| `force` | `bool` | `False` | Re-download or re-copy the raw file even if it already exists in cache. |
| `cache_root` | path-like or `None` | `None` | Root directory for raw-file cache and manifest. |
| `local_source` | path-like or `None` | `None` | Local CSV file to use instead of the online current/vintage CSV. |

## Output

`load_fred_qd()` returns a `DataBundle`.

| Field | Value |
| --- | --- |
| `bundle.panel` | Quarterly canonical panel with `DatetimeIndex` named `date`. |
| `bundle.metadata["dataset"]` | `"fred_qd"` |
| `bundle.metadata["source_family"]` | `"fred-qd"` |
| `bundle.metadata["frequency"]` | `"quarterly"` |
| `bundle.metadata["transform_codes"]` | Official FRED-QD t-code map parsed from the CSV transform row. |
| `bundle.panel.attrs["macroforecast_transform_codes"]` | Same t-code map for pandas-native handoff. |

The loader stores raw-file provenance in `metadata["artifact"]`: source URL or
local path, cache path, file format, download/copy time, SHA-256, file size, and
cache-hit status.

## Transform Codes

FRED-QD provides official transformation codes in the raw CSV. The loader does
not maintain a separate hand-coded table. It parses the CSV transform row and
stores the result in `metadata["transform_codes"]`.

`mf.preprocessing.reprocess(..., transform="official")` uses that map.

The current official CSV was manually checked during this implementation pass:
the raw `transform` row, `_parse_fred_csv()` output, and
`DataBundle.metadata["transform_codes"]` matched with zero mismatches. The
current check found 245 series and transform-code counts `{1: 22, 2: 32,
5: 140, 6: 50, 7: 1}`.

## Frequency Contract

FRED-QD is quarterly. `metadata["frequency"] == "quarterly"` and default
data-spec horizons are `(1, 2, 4, 8)`.

If FRED-QD is combined into a monthly panel, `mf.data.combine()` allows it, but
this is not the recommended default. Prefer FRED-MD for monthly targets. When
this conversion occurs, the combined bundle records a note in
`metadata["parse_notes"]` and records converted columns in
`metadata["frequency_conversion_warnings"]`.

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

FRED-SD includes both monthly and quarterly series. If a selected FRED-SD
series is monthly, the default rule `monthly_to_quarterly="quarterly_average"`
averages monthly observations within each quarter. The function emits
`UserWarning` and records the conversion in
`metadata["frequency_conversion_warnings"]`.

## Example

```python
import macroforecast as mf

bundle = mf.data.load_fred_qd()
spec = mf.data.spec(bundle, target="GDPC1", horizons=[1, 2, 4, 8])
processed = mf.preprocessing.reprocess(spec)
```
