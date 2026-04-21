# FRED-SD

Real-time U.S. state-level macroeconomic panel maintained by the Federal Reserve Bank of St. Louis. Provides per-state variables (labor market, production, housing) with vintage history, enabling real-time forecasting exercises at the state level. Loaded via `macrocast.load_fred_sd()` when `path.1_data_task.fixed_axes.dataset == "fred_sd"`.

FRED-SD differs structurally from FRED-MD / FRED-QD: the file format is an Excel workbook, and the panel **mixes monthly and quarterly series** — a user-facing complication discussed in v1.0 limitations below.

## Citation & authoritative source

- **Original paper**: Kathryn O. Bokun, Laura E. Jackson, Kevin L. Kliesen, Michael T. Owyang, "FRED-SD: A Real-Time Database for State-Level Data with Forecasting Applications," *International Journal of Forecasting* **38**(4): 1376–1399, 2022 (accepted 2021). St. Louis Fed working paper: [2020-031](https://research.stlouisfed.org/wp/more/2020-031) (first version December 2020).
- **Official landing page**: [St. Louis Fed — FRED-SD](https://www.stlouisfed.org/research/economists/owyang/fred-sd).
- **Data page**: [Research Data — FRED-SD](https://research.stlouisfed.org/data/owyang/fred-sd/).

## What macrocast downloads

- **Current vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/FRED_SD.xlsx` (referenced in `macrocast/raw/datasets/fred_sd.py`). Excel workbook.
- **Historical vintage**: `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/{vintage}.xlsx` where `vintage` follows the publication-month convention.

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

Within a single workbook, the monthly series have 12 observations per year while the quarterly series have 4. macrocast does not currently reconcile this automatically:

- The loader reads every sheet into a single wide DataFrame. Monthly columns have NaN at positions where quarterly columns have values (and vice versa), depending on the underlying sheet's native index.
- Downstream handling is the user's responsibility in v1.0. Two practical patterns:
  - Pick `variables=[...]` to include only same-frequency series (pure-monthly or pure-quarterly).
  - Let missing values propagate and use a Layer 2 preprocessing imputation (`x_missing_policy`) to fill quarterly observations into monthly gaps.

A **proper mixed-frequency adapter** (MIDAS-style or state-space filling) is scheduled for v1.1 / Phase 10. See 1.3 Horizon & evaluation window and the `frequency` registry entry (`mixed_frequency` is `future` status, v2 Phase 11).

## Real-time vintage discipline

FRED-SD is explicitly designed as a *real-time* database — each `.xlsx` file is the dataset as known at a specific publication month. This matters for studies that want to avoid contaminating forecasts with data revisions:

- `information_set_type: real_time_vintage` + `leaf_config.data_vintage: "2022-03"` loads the March-2022 vintage exactly.
- `information_set_type: revised` (default) loads the latest `FRED_SD.xlsx`.

The authors emphasise in the 2022 paper that some state-level series see substantial revisions (especially GDP by state), so real-time studies on FRED-SD generally should pin a vintage.

## State selection — no registry axis yet in v1.0

The loader supports `states=["CA", "TX", ...]` but macrocast has **no Layer 1 axis that carries this list** in v1.0. Users who need a state subset should pass it via the loader directly:

```python
from macrocast import load_fred_sd
result = load_fred_sd(states=["CA", "TX", "NY", "FL"], variables=["PAYEMS"])
```

A proper `states` leaf_config field (parallel to `target` / `targets` for variable selection) is planned for v1.1.

## Variable selection

Same caveat — `variables=[...]` works at the loader level but there is no Layer 1 axis for it in v1.0. The existing `variable_universe` axis (1.4) operates on FRED-MD / FRED-QD column names; its mapping to FRED-SD sheet names is not wired.

## Changes from the 2020 working paper to current

Compared with FRED-MD / FRED-QD the FRED-SD maintenance history is shorter (first release late 2020):

- **Coverage expansion** — the initial release focused on the subset of states for which all three category groups were available in real-time; subsequent vintages added state-series coverage as Fed / BEA / BLS backfilled missing vintage history.
- **Methodology refinement** — the authors' methodology for constructing real-time coincident activity indices evolved between the 2020 working paper and the 2022 published version; the data appendix on the St. Louis Fed site tracks the exact method currently in use.
- **Series additions** — a small number of series have been added (e.g., certain housing-quality indices for states where BLS released new data). Always consult the live data appendix for the current variable roster.

## Loader behaviour — things to know

- **Excel parsing** via `openpyxl`. Each sheet is read independently; `pd.read_excel(..., sheet_name=None)` returns `dict[str, DataFrame]` and the loader concatenates wide-form.
- **Cache**: same mechanism as FRED-MD / FRED-QD (`~/.cache/macrocast/raw/`).
- **support_tier = "provisional"** on the returned `RawDatasetMetadata` — signals that FRED-SD ingestion is newer than FRED-MD / FRED-QD and has edge cases still being worked out.
- **No T-code row** — the FRED-SD workbook does not encode stationarity codes per variable the way FRED-MD / FRED-QD do. Users applying T-codes must consult the paper's appendix or the variable's underlying FRED ID.

## Known limitations in macrocast v1.0

1. **No state-selection axis** — loader-only selection. Recipe-level state filtering is v1.1.
2. **No variable-selection axis for FRED-SD** — `variable_universe` values do not map to FRED-SD sheet names.
3. **Mixed frequency not reconciled** — user is responsible for picking same-frequency subsets or using missing-value imputation.
4. **No T-code row** — preprocessing that relies on FRED-MD / FRED-QD T-codes via `tcode_policy: apply_tcodes` is a no-op for FRED-SD.
5. **`support_tier = provisional`** — expect rougher edges than FRED-MD / FRED-QD; pin a vintage for any study intended to be replicable.

## See also

- [FRED-MD](fred_md.md), [FRED-QD](fred_qd.md) — sister databases.
- [Source & Frame (1.1)](../source.md) — `dataset` / `information_set_type` / `frequency` axis interaction.
- [1.3 horizon & evaluation window](../window.md) (coming) — how mixed-frequency panels interact with horizon / OOS structure.
