# FRED-SD

FRED-SD can be used by itself or together with FRED-MD/FRED-QD.

The important rule is simple:

- FRED-MD and FRED-QD publish official transformation codes.
- FRED-SD does not publish official transformation codes.
- macrocast therefore leaves FRED-SD source values untransformed by default.
- Reviewed FRED-SD inferred t-codes are available only when you opt in.

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

## Inferred Transforms

Use inferred SD t-codes only when you want the reviewed research layer:

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

## BPPRIVSA And STHPI

`BPPRIVSA` is frequency-specific:

- monthly experiments use the MD permits/housing analog code `4`
- quarterly experiments use the QD permits/housing analog code `5`

`STHPI` follows the quarterly QD `USSTHPI` analog code `5`. If a monthly
experiment later interpolates STHPI, the transform is still justified from the
quarterly source-frequency analog.

## Defaults

For ordinary use, do nothing. The default profile uses official FRED-MD/QD
t-codes and leaves FRED-SD inferred t-codes off.
