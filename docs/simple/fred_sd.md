# FRED-SD

FRED-SD can be used by itself or together with FRED-MD/FRED-QD.

The important rule is simple:

- FRED-MD and FRED-QD publish official transformation codes.
- FRED-SD does not publish official transformation codes.
- macrocast therefore leaves FRED-SD source values untransformed by default.
- Reviewed and empirical FRED-SD t-codes are available only when you opt in.

## Frequency

FRED-MD fixes the experiment to monthly frequency:

```python
import macrocast as mc

result = mc.forecast(
    "fred_md+fred_sd",
    target="INDPRO",
    start="1985-01",
    end="2019-12",
    horizons=[1, 3, 6],
)
```

FRED-QD fixes the experiment to quarterly frequency:

```python
result = mc.forecast(
    "fred_qd+fred_sd",
    target="GDPC1",
    start="1985-01",
    end="2019-12",
    horizons=[1, 2, 4],
)
```

FRED-SD alone requires `frequency`:

```python
result = mc.forecast(
    "fred_sd",
    target="UR_CA",
    start="1985-01",
    end="2019-12",
    horizons=[1],
    frequency="monthly",
)
```

## State And Variable Selection

Use `use_fred_sd_selection()` to restrict the FRED-SD component before the
workbook is widened into `VARIABLE_STATE` columns:

```python
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

This is different from `variable_universe`, which filters already-loaded
columns.

## Mixed-Frequency Representation

FRED-SD mixes monthly and quarterly state series. By default macrocast keeps the
selected FRED-SD columns on the experiment calendar and records the decision:

```python
exp = (
    mc.Experiment(
        dataset="fred_md+fred_sd",
        target="INDPRO",
        start="1985-01",
        end="2019-12",
        horizons=[1],
    )
    .use_fred_sd_selection(states=["CA"], variables=["UR", "NQGSP"])
    .use_fred_sd_mixed_frequency_representation("drop_non_target_native_frequency")
)
```

Useful choices:

- `calendar_aligned_frame`: default, keep selected FRED-SD columns.
- `drop_unknown_native_frequency`: drop columns whose native frequency could
  not be inferred.
- `drop_non_target_native_frequency`: keep only FRED-SD columns matching the
  experiment frequency.
- `native_frequency_block_payload`: keep the panel and give a registered
  custom model, `midas_almon`, `midasr`, or `midasr_nealmon`
  monthly/quarterly/unknown block metadata.
- `mixed_frequency_model_adapter`: same block payload plus a mixed-frequency
  adapter contract.

The last two choices require `feature_builder="raw_feature_panel"`, a
registered custom `model_family`, `model_family="midas_almon"`, or
`model_family="midasr"` / `"midasr_nealmon"`, and direct forecasts. Runtime writes
`fred_sd_mixed_frequency_representation.json`; block runs also write
`fred_sd_native_frequency_block_payload.json`, and adapter runs write
`fred_sd_mixed_frequency_model_adapter.json`.

Use `model_family="midas_almon"` for the built-in narrow MIDAS-style baseline.
It constructs Almon distributed-lag basis features from the selected FRED-SD
raw panel and fits a ridge direct forecast. Full MIDAS/state-space estimator
families remain future work.

Use `model_family="midasr"` when you want the package-owned Python slice
aligned to the R `midasr` package's restricted MIDAS surface. Set
`midasr_weight_family` to `nealmon`, `almonp`, `nbeta`, `genexp`, or
`harstep`; `midasr_nealmon` is kept as a legacy alias for the `nealmon`
branch. `harstep` uses the R `midasr` 20-lag HAR shape, so leave
`midas_max_lag` unset or set it to `20`. This is still an operational-narrow
research route, so keep the selected predictor set explicit.

## Inferred Transforms

Use inferred SD t-codes only when you want the reviewed national-analog
research layer:

```python
exp = (
    mc.Experiment(
        dataset="fred_qd+fred_sd",
        target="GDPC1",
        start="1985-01",
        end="2019-12",
        horizons=[1, 2, 4],
    )
    .use_sd_inferred_tcodes()
)

result = exp.run()
```

The manifest records `data_reports.sd_inferred_tcodes` with:

- `official: false`
- map version
- allowed review statuses
- columns that received inferred codes
- skipped columns and their review status

## Empirical Transforms

Use the empirical variable-global policy when you want the 2026-04-26
stationarity audit choice: one code per FRED-SD variable, shared across states.

```python
exp = (
    mc.Experiment(
        dataset="fred_qd+fred_sd",
        target="GDPC1",
        start="1985-01",
        end="2019-12",
        horizons=[1, 2, 4],
    )
    .use_sd_empirical_tcodes(unit="variable_global")
)
```

Use the state-series policy only with an explicit audited column map:

```python
exp.use_sd_empirical_tcodes(
    unit="state_series",
    code_map={"UR_CA": 2, "UR_TX": 5},
    audit_uri="artifacts/sd_state_series_audit.csv",
)
```

Both policies record `official: false` and the map/audit source in
`data_reports.sd_inferred_tcodes`.

## BPPRIVSA And STHPI

`BPPRIVSA` is frequency-specific:

- monthly experiments use the MD permits/housing analog code `4`
- quarterly experiments use the QD permits/housing analog code `5`

`STHPI` follows the quarterly QD `USSTHPI` analog code `5`. If a monthly
experiment later interpolates STHPI, the transform is still justified from the
quarterly source-frequency analog.

## Defaults

For ordinary use, do nothing. The default profile uses official FRED-MD/QD
t-codes and leaves FRED-SD inferred/empirical t-codes off.
