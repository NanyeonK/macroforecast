# Use FRED-SD

FRED-SD can be used alone or together with FRED-MD/FRED-QD.

FRED-MD fixes the experiment frequency to monthly:

```python
import macroforecast as mf

result = mf.forecast(
    "fred_md+fred_sd",
    target="INDPRO",
    start="1985-01",
    end="2019-12",
    horizons=[1, 3, 6],
)
```

> **Date formats**: `start` / `end` accept ISO date strings: full `YYYY-MM-DD`, or partial `YYYY-MM` (normalized to first/last of month), or `YYYY` (normalized to year-start/year-end).

FRED-QD fixes the experiment frequency to quarterly:

```python
import macroforecast as mf

result = mf.forecast(
    "fred_qd+fred_sd",
    target="GDPC1",
    start="1985-01",
    end="2019-12",
    horizons=[1, 2, 4],
)
```

FRED-SD alone requires `frequency`:

```python
import macroforecast as mf

result = mf.forecast(
    "fred_sd",
    target="UR_CA",
    frequency="monthly",
    start="1985-01",
    end="2019-12",
    horizons=[1],
)
```

Use `Experiment` to restrict the FRED-SD component to selected states or variables:

```python
import macroforecast as mf

result = (
    mf.Experiment(
        dataset="fred_md+fred_sd",
        target="INDPRO",
        start="1985-01",
        end="2019-12",
        horizons=[1, 3, 6],
    )
    .use_fred_sd_selection(states=["CA", "TX"], variables=["UR", "BPPRIVSA"])
    .run(output_directory="outputs/indpro_fred_sd_selection")
)
```

FRED-SD does not publish official transformation codes. The default path leaves FRED-SD inferred or empirical t-code policies off. Opt into inferred national-analog t-codes only when that is part of the study design:

```python
import macroforecast as mf

result = (
    mf.Experiment(
        dataset="fred_qd+fred_sd",
        target="GDPC1",
        start="1985-01",
        end="2019-12",
        horizons=[1, 2, 4],
    )
    .use_sd_inferred_tcodes()
    .run(output_directory="outputs/gdpc1_fred_sd_inferred")
)
```

For empirical FRED-SD t-codes, choose the unit explicitly:

```python
import macroforecast as mf

result = (
    mf.Experiment(
        dataset="fred_qd+fred_sd",
        target="GDPC1",
        start="1985-01",
        end="2019-12",
        horizons=[1, 2, 4],
    )
    .use_sd_empirical_tcodes(unit="variable_global", audit_uri="artifacts/sd_tcodes.csv")
    .run(output_directory="outputs/gdpc1_fred_sd_empirical")
)
```

To drop FRED-SD columns whose native frequency does not match the experiment calendar:

```python
import macroforecast as mf

result = (
    mf.Experiment(
        dataset="fred_md+fred_sd",
        target="INDPRO",
        start="1985-01",
        end="2019-12",
        horizons=[1],
    )
    .use_fred_sd_selection(states=["CA"], variables=["UR", "NQGSP"])
    .use_mixed_frequency_representation("drop_non_target_native_frequency")
    .run(output_directory="outputs/indpro_fred_sd_frequency_drop")
)
```
