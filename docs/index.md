# macrocast

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

macrocast is a research-oriented forecasting package with **opinionated defaults** and **auditable choice points**. Most studies don't need to touch 90% of the axes — the defaults match the empirical-macro baseline. When you *do* want to deviate, there's exactly one place that documents each choice, and the choice lands in `manifest.json` so you can verify it at the artifact level.

## What is complete (v0.9.4)

- **Stage 0 — Design** (6 axes, 31 operational values). Recipe grammar: runner dispatch, sweep shape, reproducibility, compute.
- **Stage 1 — Data** (20 axes, 73 operational values). Dataset + task + forecast object + time windows + benchmark + predictors + break policy.

Stages 2 through 7 (preprocessing, training, evaluation, provenance, stat tests, importance) are still in active development and are deliberately **not** exposed in the user-facing docs until their per-axis walk lands.

## Documentation

| Section | Description |
|---------|-------------|
| [Installation](install.md) | Install macrocast and optional dependencies |
| [Getting Started](getting_started/index.md) | Quickstart + Stages Reference cheat sheet |
| [User Guide — Design (Stage 0)](user_guide/design.md) | Six axes that decide study shape |
| [User Guide — Data (Stage 1)](user_guide/data/index.md) | Twenty axes that decide data, task, and evaluation window |
| [Sources](sources/index.md) | FRED-MD / FRED-QD / FRED-SD — what macrocast actually downloads, variable groups, T-codes, vintages |
| [API Reference](api/index.md) | Function signatures and class documentation |
| [Docs Conventions](CONVENTIONS.md) | Rules that every docs page follows |

## Core design principles

1. **One recipe = one fully specified study.** No hidden defaults, no implicit preprocessing.
2. **Defaults match empirical-macro baseline.** Most users don't read docs until they want to deviate.
3. **Single source of truth per axis.** Each axis is documented on exactly one page.
4. **Docs and code stay in sync.** Every value cites the exact dispatch function.
5. **Verify through the manifest.** `manifest.json` records every resolved axis value.

```{toctree}
:hidden:
:maxdepth: 1

install
getting_started/index
user_guide/index
sources/index
api/index
CONVENTIONS
```
