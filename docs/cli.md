# Command-Line Interface

macrocast ships with a CLI for running forecast experiments without writing Python code. The entry point is the `macrocast` command.

---

## Overview

```
macrocast --help
macrocast init [--output experiment.yaml] [--force]
macrocast info CONFIG.yaml
macrocast run CONFIG.yaml [--summary]
```

Three commands are available:

| Command | Purpose |
|---------|---------|
| `init` | Write a default YAML config template to disk |
| `info` | Print a resolved config summary without running the experiment |
| `run` | Execute a forecast experiment from a YAML config file |

Add `--verbose` / `-v` before the subcommand to enable DEBUG-level logging.

---

## `macrocast init`

Writes a fully-annotated YAML template to the specified output path.

```bash
macrocast init                        # writes experiment.yaml
macrocast init --output my_run.yaml
macrocast init --output my_run.yaml --force   # overwrite if exists
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `experiment.yaml` | Output file path |
| `--force` | false | Overwrite existing file |

---

## `macrocast info`

Parses and resolves a YAML config, prints a human-readable summary, and exits without running any models. Useful for verifying config syntax before committing to a long experiment.

```bash
macrocast info experiment.yaml
```

Example output:

```
Experiment ID:  my_indpro_experiment
Output dir:     /home/user/.macrocast/results/my_indpro_experiment
Dataset:        fred_md
Target:         INDPRO
Vintage:        current
Horizons:       [1, 3, 6, 12]
Window:         expanding
OOS start:      2000-01-01
OOS end:        auto
n_jobs:         -1
Models (3):
  - krr__factors__KFoldCV(k=5)__l2
  - rf__none__KFoldCV(k=5)__l2
  - xgboost__none__KFoldCV(k=5)__l2
Features:
  n_factors=8, n_lags=4, use_factors=True, lookback=12
```

---

## `macrocast run`

Executes the full experiment and writes results to parquet.

```bash
macrocast run experiment.yaml
macrocast run experiment.yaml --summary    # print MSFE table on completion
macrocast -v run experiment.yaml           # verbose (DEBUG) logging
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--summary` | Print per-model MSFE table to stdout after the run |

The experiment writes a parquet file to `{output_dir}/{experiment_id}.parquet` on completion. Pass that path to `ResultSet.from_parquet()` for downstream evaluation.

---

## YAML Config Format

The template produced by `macrocast init` is reproduced below with inline annotations.

```yaml
experiment:
  id: "my_experiment"          # optional; UUID auto-generated if absent
  output_dir: "~/.macrocast/results"

data:
  dataset: "fred_md"           # fred_md | fred_qd  (fred_sd not supported via CLI)
  vintage: null                # null = current release; "YYYY-MM" for a specific vintage
  target: "INDPRO"             # column name in the dataset
  cache_dir: "~/.macrocast/cache"  # optional; defaults to ~/.macrocast/cache

features:
  n_factors: 8                 # PCA factors; ignored if use_factors=false
  n_lags: 4                    # AR lags of target appended to feature matrix
  use_factors: true            # false = AR-only mode (data-poor baseline)
  standardize_X: true          # standardize predictor panel before PCA
  standardize_Z: false         # standardize output feature matrix
  lookback: 12                 # LSTM look-back window; ignored for cross-sectional models

experiment:
  horizons: [1, 3, 6, 12]     # forecast horizons h; one model is trained per h
  window: "expanding"          # expanding | rolling
  rolling_size: null           # required when window = rolling
  oos_start: "2000-01-01"     # null = default (80th percentile of sample)
  oos_end: null                # null = last date minus max horizon
  n_jobs: -1                   # -1 = all cores

models:
  - name: krr
    regularization: factors    # none|ridge|lasso|adaptive_lasso|group_lasso|elastic_net|factors|tvp_ridge|booging
    cv_scheme: kfold           # bic | poos | kfold
    kfold_k: 5                 # only used when cv_scheme = kfold
    loss_function: l2          # l2 | epsilon_insensitive
    kwargs:
      alpha_grid: [0.001, 0.01, 0.1, 1.0, 10.0]
      gamma_grid: [0.01, 0.1, 1.0]
      cv_folds: 5

  - name: rf
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      n_estimators: 500

  - name: xgboost
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2

  - name: nn
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      hidden_dims: [64, 128]
      max_epochs: 200
      patience: 20

  - name: lstm
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      hidden_dims: [32, 64]
      max_epochs: 200
```

**Notes:**

- `fred_sd` (state-level data) is not supported via the CLI. Load it programmatically using `mc.load_fred_sd()`.
- R-side models (Ridge, LASSO, ARDI, etc.) are invoked through `macrocastR` and are not yet integrated into the CLI. Use the Python API to run mixed Python/R experiments.
- Results are written to `{output_dir}/{experiment_id}.parquet`. If `output_dir` is not specified in the config, results are not persisted automatically.

---

## Cross-References

- [ForecastExperiment](pipeline/experiment.md) — programmatic equivalent of `macrocast run`
- [FeatureSpec](pipeline/experiment.md#featurespec) — feature configuration details
- [Pipeline Layer](pipeline/index.md) — model zoo and component reference
