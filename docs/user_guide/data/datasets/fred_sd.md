# FRED-SD

Real-time U.S. state-level macroeconomic panel maintained by the Federal Reserve Bank of St. Louis. Provides per-state variables (labor market, production, housing) with vintage history, enabling real-time forecasting exercises at the state level. Loaded via `macrocast.load_fred_sd()` when `path.1_data_task.fixed_axes.dataset == "fred_sd"`.

FRED-SD differs structurally from FRED-MD / FRED-QD: the file format is an Excel workbook, and the panel **mixes monthly and quarterly series** — a user-facing complication discussed in v1.0 limitations below.

## Citation & authoritative source

- **Original paper**: Kathryn O. Bokun, Laura E. Jackson, Kevin L. Kliesen, Michael T. Owyang, "FRED-SD: A Real-Time Database for State-Level Data with Forecasting Applications," *International Journal of Forecasting* **38**(4): 1376–1399, 2022 (accepted 2021). St. Louis Fed working paper: [2020-031](https://research.stlouisfed.org/wp/more/2020-031) (first version December 2020).
- **Official landing page**: [St. Louis Fed — FRED-SD](https://www.stlouisfed.org/research/economists/owyang/fred-sd).
- **Data page**: [Research Data — FRED-SD](https://research.stlouisfed.org/data/owyang/fred-sd/).

## What macrocast downloads

- **Current vintage**: macrocast reads the official FRED-SD landing page and
  selects the latest **Data by Series** workbook link, for example
  `.../fred-sd/series/series-YYYY-MM.xlsx`. This is the workbook layout the
  loader expects: tabs are variables, columns are states.
- **Historical vintage**: macrocast first tries the direct **Data by Series**
  workbook URL, `.../fred-sd/series/series-YYYY-MM.xlsx`. If a vintage is only
  distributed inside an official by-series archive, it falls back to the
  appropriate `fredsd_byseries_*.zip` file and extracts the requested workbook.

The workbook is laid out with **one sheet per variable**; within each sheet, **columns are states** (50 states plus DC) and **rows are observations** indexed by date. The macrocast loader (`macrocast/raw/datasets/fred_sd.py`) accepts two selectors:

```python
load_fred_sd(
    vintage=None,
    states=None,     # list of state abbreviations, e.g. ["CA", "TX", "NY"]
    variables=None,  # list of variable names (sheet names)
)
```

When both selectors are `None`, the loader ingests every sheet and every state, producing a wide panel with column names of the form `{variable}_{state}` (e.g., `PAYEMS_CA`).

## Structure — ~28 variables per state

Per the 2022 paper, FRED-SD provides approximately 28 variables per state, grouped into three broad categories:

1. **Labor market** — payroll employment (`PAYEMS`-analogues), unemployment rate, labor force participation, average weekly hours.
2. **Production / output** — state personal income, coincident activity index, industrial production proxies where available.
3. **Housing** — housing starts, permits, house price indices.

Exact variable list and state coverage at a given vintage: see the [data appendix](https://research.stlouisfed.org/data/owyang/fred-sd/) on the St. Louis Fed site. macrocast does not redistribute the per-variable metadata.

## Mixed frequency — the core structural issue

Unlike FRED-MD (purely monthly) or FRED-QD (purely quarterly), FRED-SD **mixes** series at different native frequencies:

- **Monthly series**: labor market indicators (unemployment, payroll employment), housing starts / permits.
- **Quarterly series**: state GDP, state personal income (only from certain BEA releases), some productivity measures.

Within a single workbook, the monthly series have 12 observations per year while the quarterly series have 4. macrocast now separates this into two explicit decisions:

- Layer 1 reports and optionally gates the selected native-frequency mix through
  `fred_sd_frequency_report_v1` and `fred_sd_frequency_policy`.
- Layer 2 chooses how the selected FRED-SD panel enters the representation
  through `fred_sd_mixed_frequency_representation`.

A **proper mixed-frequency model adapter** (MIDAS-style or state-space filling)
is still future work. The current executable surface is a panel-shaping
contract, not a mixed-frequency likelihood/model.

## Real-time vintage discipline

FRED-SD is explicitly designed as a *real-time* database — each `.xlsx` file is the dataset as known at a specific publication month. This matters for studies that want to avoid contaminating forecasts with data revisions:

- `information_set_type: real_time_vintage` + `leaf_config.data_vintage: "2022-03"` loads the March-2022 vintage exactly.
- `information_set_type: revised` (default) loads the latest official **Data by Series** workbook discovered from the St. Louis Fed landing page.

The authors emphasise in the 2022 paper that some state-level series see substantial revisions (especially GDP by state), so real-time studies on FRED-SD generally should pin a vintage.

## State and variable selection

The loader supports direct source selectors:

```python
from macrocast import load_fred_sd
result = load_fred_sd(states=["CA", "TX", "NY", "FL"], variables=["PAYEMS"])
```

Recipe/runtime selection is available through Layer 1 and the simple API:

```python
import macrocast as mc

exp = (
    mc.Experiment(
        dataset="fred_md+fred_sd",
        target="INDPRO",
        start="1985-01",
        end="2019-12",
        horizons=[1, 3, 6],
    )
.use_fred_sd_selection(states=["CA", "TX"], variables=["UR", "BPPRIVSA"])
)
```

Built-in groups are also Layer 1 source-load selectors:

```python
exp = (
    mc.Experiment(
        dataset="fred_md+fred_sd",
        target="INDPRO",
        start="1985-01",
        end="2019-12",
        horizons=[1, 3, 6],
    )
    .use_fred_sd_groups(
        state_group="census_region_west",
        variable_group="labor_market_core",
    )
)
```

FRED-SD has two different variable concepts:

- `sd_variable_selection` chooses workbook sheets before loading, via
  `leaf_config.sd_variables`.
- `fred_sd_variable_group` is a recipe-level shortcut that resolves to
  `sd_variable_selection=selected_sd_variables` before loading.
- `variable_universe` remains the generic post-load column universe filter,
  after columns have canonical `VARIABLE_STATE` names.

The built-in state groups are Census regions and divisions plus
`contiguous_48_plus_dc`. Full recipes may also use
`fred_sd_state_group=custom_state_group` with either
`leaf_config.sd_state_group_members` or
`leaf_config.sd_state_groups` + `leaf_config.sd_state_group_name`.

Built-in SD-variable groups are:

- `labor_market_core`: `ICLAIMS`, `LF`, `NA`, `PARTRATE`, `UR`
- `employment_sector`: `CONS`, `FIRE`, `GOVT`, `INFO`, `MFG`, `MFGHRS`,
  `MINNG`, `PSERV`
- `gsp_output`: aggregate and sector GSP/output variables
- `housing`: `BPPRIVSA`, `RENTS`, `STHPI`
- `trade`: `EXPORTS`, `IMPORTS`
- `income`: `OTOT`
- t-code-review groups: `direct_analog_high_confidence`,
  `provisional_analog_medium`, `semantic_review_outputs`,
  `no_reliable_analog`

Use `fred_sd_variable_group=custom_sd_variable_group` with
`leaf_config.sd_variable_group_members` or
`leaf_config.sd_variable_groups` + `leaf_config.sd_variable_group_name` for
paper-specific bundles. Group selectors and explicit list selectors are
mutually exclusive for the same side; use one source of truth per state or
variable selection.

## Series metadata contract

FRED-SD runs write `fred_sd_series_metadata.json` when the selected dataset
includes FRED-SD. The contract version is `fred_sd_series_metadata_v1` and its
owner is Layer 1. It records one row per loaded FRED-SD column:

- canonical column name, `sd_variable`, state, and source sheet
- inferred native frequency (`monthly`, `quarterly`, `annual`, `irregular`, or
  `unknown`) from the non-missing observation calendar
- observed start/end dates and non-missing observation count
- selected states, selected SD variables, and native-frequency counts

For composite datasets, the same contract is stored from the FRED-SD component
before generic post-load `variable_universe` filtering.

## Frequency composition report

FRED-SD runs also write `fred_sd_frequency_report.json`. The contract version
is `fred_sd_frequency_report_v1` and its owner is Layer 1. It is derived from
`fred_sd_series_metadata_v1` and makes the selected native-frequency mix
explicit before Layer 2 chooses any post-frame representation.

The report records:

- `native_frequency_counts` and `known_native_frequency_counts`
- `frequency_status`: `empty`, `unknown_only`, `single_frequency`,
  `single_frequency_with_unknown`, `mixed_frequency`, or
  `mixed_frequency_with_unknown`
- `has_monthly_quarterly_mix`
- `requires_mixed_frequency_decision`
- frequency counts by state and by SD variable

This report feeds the Layer 1 `fred_sd_frequency_policy` axis. The default is
`report_only`, which records the selected panel without changing execution.
Full recipes can instead choose:

- `allow_mixed_frequency`: explicitly accept mixed or unknown native
  frequencies.
- `reject_mixed_known_frequency`: block selected panels with more than one
  known native frequency, such as monthly plus quarterly.
- `require_single_known_frequency`: require exactly one known native frequency
  and no unknown-frequency selected series.

The policy runs before Layer 2 representation construction. Use the strict
policy for same-frequency FRED-SD studies where quarterly-vs-monthly conversion
should not be silently folded into the downstream representation.

## Layer 2 mixed-frequency representation

After Layer 1 has selected the FRED-SD series and written the frequency report,
Layer 2 can shape the panel with
`fred_sd_mixed_frequency_representation`:

| Value | Runtime behavior |
|---|---|
| `calendar_aligned_frame` | Default. Keep the selected FRED-SD columns on the recipe target calendar after generic monthly/quarterly conversion. This preserves the current behavior and records a Layer 2 report. |
| `drop_unknown_native_frequency` | Drop selected FRED-SD columns whose inferred native frequency is `unknown`; keep known monthly, quarterly, annual, and irregular classes. |
| `drop_non_target_native_frequency` | Keep only selected FRED-SD columns whose inferred native frequency matches the recipe frequency (`monthly` or `quarterly`). This is the strict same-frequency representation choice. |
| `native_frequency_block_payload` | Planned. Reserved for future separate monthly/quarterly/unknown block payloads. |
| `mixed_frequency_model_adapter` | Planned. Reserved for future MIDAS/state-space style adapters consumed with compatible Layer 3 models. |

Simple API:

```python
exp = (
    mc.Experiment(
        dataset="fred_md+fred_sd",
        target="INDPRO",
        start="1985-01",
        end="2019-12",
        horizons=[1],
    )
    .use_fred_sd_groups(variable_group="labor_market_core")
    .use_fred_sd_mixed_frequency_representation("drop_non_target_native_frequency")
)
```

Runtime writes `fred_sd_mixed_frequency_representation.json` with owner
`2_preprocessing`, the selected policy, target frequency, kept/dropped FRED-SD
columns, and dropped counts by native frequency. Non-FRED-SD columns in a
composite dataset are preserved. A FRED-SD target column is never silently
dropped; runtime raises an execution error if the selected representation would
remove it.

## Changes from the 2020 working paper to current

Compared with FRED-MD / FRED-QD the FRED-SD maintenance history is shorter (first release late 2020):

- **Coverage expansion** — the initial release focused on the subset of states for which all three category groups were available in real-time; subsequent vintages added state-series coverage as Fed / BEA / BLS backfilled missing vintage history.
- **Methodology refinement** — the authors' methodology for constructing real-time coincident activity indices evolved between the 2020 working paper and the 2022 published version; the data appendix on the St. Louis Fed site tracks the exact method currently in use.
- **Series additions** — a small number of series have been added (e.g., certain housing-quality indices for states where BLS released new data). Always consult the live data appendix for the current variable roster.

## Loader behaviour — things to know

- **Excel parsing** via `openpyxl`. Each sheet is read independently; `pd.read_excel(..., sheet_name=None)` returns `dict[str, DataFrame]` and the loader concatenates wide-form.
- **Cache**: same mechanism as FRED-MD / FRED-QD (`~/.cache/macrocast/raw/`).
- **support_tier = "provisional"** on the returned `RawDatasetMetadata` — this now reflects remaining mixed-frequency and study-design edge cases, not lack of a live/vintage loader or t-code policy surface.
- **Series metadata** — runtime runs that include FRED-SD write `fred_sd_series_metadata.json`, which makes the selected state/variable panel and native-frequency mix auditable.
- **Frequency report** — runtime also writes `fred_sd_frequency_report.json`, which reduces the selected panel to a Layer 1 frequency-composition contract for downstream policy decisions.
- **Mixed-frequency representation report** — runtime writes `fred_sd_mixed_frequency_representation.json` for FRED-SD runs; this is the Layer 2 panel-shaping contract consumed before t-code preprocessing.
- **No T-code row** — the FRED-SD workbook does not encode stationarity codes per variable the way FRED-MD / FRED-QD do. FRED-SD transformation codes are therefore a research decision, not source metadata.
- **T-code policy choices** — state panels create a real choice between national-analog t-codes, one empirically selected code per SD variable, or independent state-by-series codes. The default is no FRED-SD t-code. The reviewed national-analog map is opt-in via `Experiment.use_sd_inferred_tcodes()`. The empirical variable-global map is opt-in via `Experiment.use_sd_empirical_tcodes(unit="variable_global")`. State-by-series empirical codes require an explicit column map via `Experiment.use_sd_empirical_tcodes(unit="state_series", code_map={...})`. All three record `official=false` in runtime reports.

## Known limitations in macrocast v1.0

1. **Mixed-frequency modeling is not implemented** — monthly-to-quarterly and quarterly-to-monthly conversion, strict same-frequency filtering, and unknown-frequency filtering are available and reported, but MIDAS/state-space mixed-frequency modeling is not implemented.
2. **State and SD-variable groups are recipe-level selectors** —
   `fred_sd_state_group` and `fred_sd_variable_group` resolve into explicit
   `sd_states` / `sd_variables` before loading. They are not post-load
   `variable_universe` filters.
3. **Generic `variable_universe` is post-load** — use `sd_variable_selection` for workbook-sheet selection and `variable_universe` for loaded `VARIABLE_STATE` columns.
4. **No official T-code row** — `official_transform_policy: dataset_tcode` has no FRED-SD workbook T-code row to consume. FRED-SD inferred T-codes are macrocast research metadata and must be opted into separately.
5. **`support_tier = provisional`** — keep this label until the future
   native-frequency block payload / mixed-frequency model-adapter path is
   implemented.

## See also

- [FRED-MD](fred_md.md), [FRED-QD](fred_qd.md) — sister databases.
- [Source & Frame (1.1)](../source.md) — `dataset` / `information_set_type` / `frequency` axis interaction.
- [Horizon & evaluation window (1.3)](../horizon.md) — how mixed-frequency panels interact with horizon / OOS structure.
