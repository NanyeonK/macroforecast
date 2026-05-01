# Data

Layer 1 answers: **what source frame, target y, and predictor x universe does
this study start from?** Layer 0 fixes runtime policy. Layer 1 defines source
data, forecast-time information, target y, and candidate predictor x columns.

Layer 1 contains canonical registry axes plus a small number of hidden
compatibility/helper selectors. The primary Navigator tree shows the
user-facing decisions; `state_selection` and `sd_variable_selection` are hidden
lower selectors used by FRED-SD helper resolution.

| Group | Axes | Focus |
|---|---:|---|
| Data Source Mode / Frequency | 3 | `custom_source_policy`, `dataset`, `frequency` |
| Forecast-Time Information | 3 | `information_set_type`, `release_lag_rule`, `contemporaneous_x_rule` |
| Target (y) Definition | 1 | `target_structure`; target IDs, horizons, and dates live in `leaf_config` |
| Predictor (x) Definition | 1 | `variable_universe`; x column lists and mappings live in `leaf_config` |
| FRED-SD Predictor Scope | 5 | `fred_sd_frequency_policy`, `fred_sd_state_group`, `fred_sd_variable_group`, hidden `state_selection`, hidden `sd_variable_selection` |
| Raw Source Quality | 2 | `raw_missing_policy`, `raw_outlier_policy` before FRED transforms/T-codes |
| FRED Transform / Frame Availability | 3 | `official_transform_policy`, `official_transform_scope`, `missing_availability` |

The first Layer 1 decision is `custom_source_policy`:

- `official_only`: choose a FRED Source Panel.
- `custom_panel_only`: provide a custom file path and analysis frequency; do
  not choose a FRED Source Panel.
- `official_plus_custom`: choose a FRED Source Panel and append custom columns.

Layer 1 does not choose model family, benchmark, researcher preprocessing,
feature representation, or evaluation metrics. Those belong to Layer 2 and
later.

## Relation To Layer Contracts

[Layer Contract Design](../design.md) defines the current L0-L8 map. L1 fixes
the data task: source, target, predictor universe, geography, sample window,
horizons, and regimes. Layer 0 handles runtime policy plus derived manifest
metadata.

## Operational Status

Kept in Layer 1:

- data source mode, FRED source panel, analysis frequency;
- data revision / vintage regime, publication lag, same-period x rule;
- target y cardinality and predictor x universe;
- FRED-SD predictor scope;
- raw-source missing/outlier handling before official transforms;
- FRED transform scope and frame availability.

Not owned by Layer 1:

- Layer 2: raw-to-clean preprocessing.
- Layer 3: feature engineering and target construction.
- Layer 4: model fitting, forecasts, benchmarks, ensembles, and tuning.
- Layer 5+: evaluation, statistical tests, interpretation, and output.

## Data Sources

Built-in FRED sources have dedicated pages:

- [FRED-MD](datasets/fred_md.md) — monthly U.S. macro panel.
- [FRED-QD](datasets/fred_qd.md) — quarterly U.S. macro panel.
- [FRED-SD](datasets/fred_sd.md) — state-level mixed-frequency panel.

```{toctree}
:maxdepth: 1
:hidden:

source
target_structure
horizon
benchmark
policies
datasets/fred_md
datasets/fred_qd
datasets/fred_sd
```
