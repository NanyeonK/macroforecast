# Real-Time Vintages

[Back to User Guide](index.md)

Most pseudo-out-of-sample studies use one final data panel and pretend each
forecast origin only saw the rows dated before that origin. That is useful, but
it is not the same as a real-time study. A real-time study also fixes the data
release vintage available at each origin, so a model fitted in March 2010 sees
the March 2010 snapshot, not the panel after later benchmark revisions.

`macroforecast` handles this with a `VintageSource`: a lazy source that resolves
one ordinary `DataBundle` for each forecast origin. Each resolved bundle still
uses the same canonical panel contract as any other dataset. The runner changes
which panel is used at each origin; feature engineering, preprocessing, models,
and evaluation keep their usual APIs.

## FRED-MD or FRED-QD vintages

Use the vintage source factory, then wrap it in `VintagePanelSpec` with an
explicit reference calendar. The reference calendar is the schedule of possible
origins and target dates; the source controls which data snapshot each origin
can observe.

```python
import pandas as pd
import macroforecast as mf

source = mf.data.fred_md_vintages(start="1999-01", end="2024-12")
reference = pd.date_range("1999-01-01", "2024-12-01", freq="MS", name="date")

data = mf.data.VintagePanelSpec(
    source=source,
    reference_calendar=reference,
    actuals_vintage="latest",
)
```

Then pass `data` anywhere you would normally pass a bundle:

```python
spec = mf.pipeline.pipeline_spec(
    data=data,
    targets=["INDPRO"],
    horizons=[1, 3, 6],
    window=mf.window.from_cutoffs(test_start="2005-01-01", horizon=1),
    arms=[mf.pipeline.Arm("AR", model="ar", is_benchmark=True)],
    evaluation=mf.pipeline.EvalSpec(benchmark="AR"),
    checkpoint_dir="ckpt/vintage_indpro",
)

report = mf.pipeline.run_pipeline(spec)
```

Forecast rows include `vintage_id`, the snapshot used for the origin, and
`actuals_vintage_id`, the snapshot used for the realized value.

FRED-MD and FRED-QD source files date observations at the start of the month or
quarter. Build the reference calendar with matching labels (`freq="MS"` for
monthly FRED-MD, `freq="QS"` for quarterly FRED-QD). A month-end calendar such
as `freq="ME"` has no labels in common with FRED-MD vintages and the vintage
runner raises a calendar-anchor error before feature construction.

## Custom vintages

`custom_vintages` accepts three common real-time data shapes.

A callable can query a database or load a file by origin date:

```python
def resolve_snapshot(origin_date):
    panel = read_snapshot_from_store(origin_date)
    return mf.data.custom_dataset(panel, dataset="my_realtime", frequency="monthly")

source = mf.data.custom_vintages(resolve_snapshot, vintage_id=lambda d: str(d.date()))
```

A mapping is useful when snapshots have already been downloaded:

```python
snapshots = {
    pd.Timestamp("2020-01-31"): panel_2020_01,
    pd.Timestamp("2020-02-29"): panel_2020_02,
}
source = mf.data.custom_vintages(snapshots, dataset="my_archive", frequency="monthly")
```

A grouped-wide ALFRED-style frame can be grouped by vintage. It must contain a
vintage column, a date column, and one numeric column per series within each
vintage snapshot:

```python
source = mf.data.custom_vintages(
    realtime_frame,
    vintage_column="realtime_start",
    date_column="date",
    dataset="alfred_extract",
    frequency="monthly",
)
```

Every custom snapshot is normalized through `as_panel` and `validate_panel`.
Sources are memoized by the resolved vintage identifier. If a callable source is
non-deterministic, run without persistent preprocessing cache so stale fitted
state cannot be reused against changing content.

For `actuals_vintage="first_release"`, the source must report timestamp-parsable
`available_vintages()`. Callable-only custom sources do not provide a release
calendar, so use a mapping or grouped-wide frame when first-release scoring is
needed.

## Static extras

If the real-time core should be joined with genuinely non-revised columns, wrap
the source once:

```python
extras = pd.DataFrame(
    {"policy_dummy": policy_dummy},
    index=reference,
)

source = mf.data.with_static_extras(
    mf.data.fred_md_vintages(start="1999-01", end="2024-12"),
    extras,
    join="outer",
)
```

The static panel's SHA-256 fingerprint is included in each resolved vintage ID.
Changing the extras therefore changes cache identity and provenance. Static
extras are truncated to rows strictly before the forecast origin before joining,
so a full-span calendar dummy panel does not create post-origin rows. Only use
this wrapper for deterministic columns or columns genuinely known in advance at
the origin.

## Latest vs first-release actuals

`actuals_vintage="latest"` scores every forecast against the latest snapshot
available to the run. This is the usual referee convention when the target is
defined as the current best measurement.

`actuals_vintage="first_release"` scores each target date `d` against the first
available vintage strictly after `d` that contains a non-missing value for that
observation. The resolver walks forward across later vintages when the first
post-date vintage does not yet contain the release, up to
`first_release_max_vintages` probes on `VintagePanelSpec` (default `12`). This is
useful when the estimand is the initial public release. It also makes
`actuals_vintage_id` row-varying because each realized date can come from a
different release.

Choose the convention before running the pipeline. `rescore()` re-evaluates the
forecast rows already written to checkpoints; it cannot retroactively switch
`actuals_vintage`. To compare latest and first-release scoring, run two specs
that differ only in `VintagePanelSpec(actuals_vintage=...)`.

## Target-transform warning

For `change`, `growth`, `log_growth`, and their `average_*` variants, the target
object spans multiple rows. Under revised data, that can make a macroforecast-side
target transform sensitive to adjacent-date vintage conventions. The vintage
runner emits a `UserWarning` for these transforms.

If the intended estimand is a real-time growth or change series, pre-transform
the target inside each vintage snapshot and pass a level/value target to
`macroforecast`. The package warns rather than trying to reconstruct a
cross-vintage transform automatically.

## Cache and provenance

Vintage-aware runs can use `n_jobs > 1`. Preprocessing cache keys include both
the origin and the resolved `vintage_id`, so two origins with the same calendar
position but different data content do not share fitted state. In parallel mode,
each worker receives the `VintagePanelSpec.data` payload once and reuses its
source object across cells, so mapping-backed custom vintage sources keep their
per-worker memoization instead of reparsing every snapshot for every cell.

Pipeline reports include `provenance["vintage_source"]` with:

- `kind`: source family or class name.
- `actuals_vintage`: `"latest"` or `"first_release"`.
- `reference_calendar`: start, end, and count.
- `origin_vintage_map`: origin to vintage ID.

Maps with 500 origins or fewer are stored inline. Larger maps are written as
`vintage_map.json` next to `checkpoint_dir`, and provenance stores the sidecar
`path`, `sha256`, and `n_origins`.
