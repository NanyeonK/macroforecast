# macrocast

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

## Current capabilities

- **27 operational model families** plus 3 operational-narrow MIDAS routes: AR through MLP, including linear boosting, factor models, SVR, tree ensembles, CatBoost, and MIDAS adapters
- **20 statistical tests**: DM variants, Clark-West, MCS, nested model tests, conditional ability, residual diagnostics
- **12 importance methods**: SHAP (tree/kernel/linear), permutation, LIME, PDP/ICE/ALE, grouped, stability
- **4 tuning algorithms**: grid search, random search, Bayesian optimization, genetic algorithm
- **5 export formats**: JSON, CSV, Parquet, JSON+CSV, all
- **3 datasets**: FRED-MD (stable), FRED-QD (stable), FRED-SD (provisional)

## Structure

```
macrocast/       # package source
  stage0/        # study grammar
  raw/           # dataset loading
  recipes/       # recipe specification
  preprocessing/ # preprocessing governance
  registry/      # current public per-axis choice registry (147 axes)
  core/          # Phase 0 DAG foundation scaffold (next execution surface)
  compiler/      # recipe -> execution bridge
  execution/     # runtime engine
  tuning/        # HP optimization
tests/           # ~195 tests
docs/            # public documentation
plans/           # internal planning
```

## Development rules

- `docs/` is public-facing only; internal planning stays in `plans/`
- Every new code surface must be documented in `docs/`
- Grammar is fixed before registry content is expanded
- One path = one fully specified forecasting study
- `macrocast/registry` remains authoritative for the current public runtime;
  `macrocast/core` is the next DAG execution foundation until migration
  adapters are complete
