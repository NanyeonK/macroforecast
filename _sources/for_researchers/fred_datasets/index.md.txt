# FRED datasets

- Architecture: [layer contracts](../../architecture/index.md)
- This page: FRED-MD / FRED-QD / FRED-SD column dictionary
- API: [public API reference](../../reference/public_api.md)

This section is the dataset dictionary for macroforecast's official FRED-backed
source panels. It is separate from Layer 1 because Layer 1 should decide the
source contract, target y, predictor x universe, and timing rules. The raw
dataset definitions belong here.

Generated: `2026-04-30` from current official FRED-MD/FRED-QD CSV files and
the current FRED-SD by-series workbook.

## Current Snapshot

| Dataset | macroforecast `dataset` value | Frequency | Current source count | Data through | Column definition |
|---|---|---|---:|---|---|
| FRED-MD | `fred_md` | monthly | 126 columns | 2025-09 | one column per official current CSV mnemonic |
| FRED-QD | `fred_qd` | quarterly | 245 columns | 2025-09 | one column per official current CSV mnemonic |
| FRED-SD | `fred_sd` | mixed state monthly/quarterly | 1428 generated columns | 2026-03 | `{sd_variable}_{state}` from by-series workbook sheets and state columns |

## How This Connects To Layer 1

- `dataset=fred_md` activates the FRED-MD monthly panel.
- `dataset=fred_qd` activates the FRED-QD quarterly panel.
- `dataset=fred_sd` activates the FRED-SD state-level panel and then requires
  `frequency` plus optional FRED-SD state/series scope choices.
- `dataset=fred_md+fred_sd` and `dataset=fred_qd+fred_sd` combine a national
  MD/QD panel with selected FRED-SD state-level columns.
- `variable_universe` is a FRED-MD/QD predictor-universe axis. FRED-SD uses
  State Scope/List and Series Scope/List before the source frame is loaded.

## Pages

```{toctree}
:maxdepth: 1

fred_md
fred_qd
fred_sd/index
```
