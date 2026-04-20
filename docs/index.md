# macrocast

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

## Why macrocast?

Macroeconomic forecasting studies often suffer from a common problem: models are compared under conditions that are not fully aligned. Differences in sample periods, information sets, preprocessing rules, or benchmark definitions make it difficult to determine whether one method truly outperforms another.

macrocast addresses this by enforcing a **recipe-based experimental grammar** where every design choice is explicit. One YAML recipe defines one complete forecasting study. Comparisons are fair because the comparison environment is held fixed while only the forecasting tool varies.

**Core design principles:**

1. **One recipe = one fully specified study.** No hidden defaults, no implicit preprocessing.
2. **Grammar first, content later.** The study language is fixed before registry inventories are filled.
3. **Represent before execute.** The registry can express more choices than the runtime currently supports.
4. **Fair comparison by construction.** Preprocessing, splits, benchmarks, and metrics are governed explicitly.

## Documentation

| Section | Description |
|---------|-------------|
| [Installation](install.md) | Install macrocast and optional dependencies |
| [Getting Started](getting_started/index.md) | Your first forecasting study in 5 minutes |
| [User Guide](user_guide/index.md) | In-depth guide to every package layer |
| [Examples](examples/index.md) | End-to-end runnable example gallery |
| [API Reference](api/index.md) | Function signatures and class documentation |
| [Mathematical Background](math/index.md) | Formal definitions for statistical tests, metrics, and importance methods |
| [Developer Guide](dev/index.md) | Architecture, contributing, extending the package |

## Package surfaces

macrocast has eight layers, executed in canonical order:

| Layer | Module | Purpose |
|-------|--------|---------|
| Stage 0 (Design) | [`macrocast.design`](user_guide/design.md) | Study grammar: fixed/varying design, comparison contract |
| Stage 1 | [`macrocast.raw`](user_guide/raw.md) | FRED-MD/QD/SD raw data loading and provenance |
| Stage 2 | [`macrocast.recipes`](user_guide/recipes.md) | Declarative recipe and run specification |
| Stage 3 | [`macrocast.preprocessing`](user_guide/preprocessing.md) | Preprocessing contract and governance |
| Stage 4 | [`macrocast.registry`](user_guide/registry.md) | Per-axis choice-space registry |
| Stage 5 | [`macrocast.compiler`](user_guide/compiler.md) | Recipe compilation and execution eligibility |
| Stage 6 | [`macrocast.execution`](user_guide/execution.md) | Runtime: models, benchmarks, metrics, artifacts |
| Stage 7 | [`macrocast.tuning`](user_guide/tuning.md) | Hyperparameter tuning engine |

## Current operational scale

- **24 model families** from AR to MLP, including linear boosting, factor models, SVR, tree ensembles
- **20 statistical tests** covering equal/conditional predictive ability, nested models, multiple comparison, diagnostics
- **12 importance methods** including SHAP, permutation, LIME, PDP/ICE/ALE, grouped, stability
- **4 tuning algorithms**: grid search, random search, Bayesian optimization, genetic algorithm
- **125 registry axes**, 717 values, 310 operational (43%)

**See also:** [Getting Started: Quickstart](getting_started/quickstart.md) | [API Reference](api/index.md)

```{toctree}
:hidden:
:maxdepth: 2

install
getting_started/index
user_guide/index
examples/index
api/index
math/index
dev/index
```
