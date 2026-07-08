# Windows

[Back to User Guide](../index.md)

`macroforecast.window` defines the estimation/val/test time frame. A `WindowSpec`
is the object passed between data, feature engineering, model selection, models,
and evaluation. It answers six questions:

- how the pre-test estimation sample expands or rolls;
- how validation splits are created inside it for model selection;
- where the final test origins start and end;
- how far each test target horizon runs;
- when the model is retrained versus reused;
- where each runner stage may fit stateful operations.

## Estimation modes

The estimation window controls how the pre-test training data grows across origins:

- **Expanding** (`mode="expanding"`): the training sample grows by one period at
  each test origin. This is the standard POOS convention.
- **Rolling** (`mode="rolling"`): a fixed-size trailing window moves forward. An
  `estimation_size` must be specified.
- **Fixed** (`mode="fixed"`): the estimation sample is anchored between fixed
  start and end dates.

## Validation designs

The inner validation window determines how hyperparameters are selected at each
retune origin. Available designs include `last_block` (one final holdout block
inside the estimation sample), `poos` (pseudo-out-of-sample one-step tail
splits), `expanding` (walk-forward expanding train/val splits), `rolling_blocks`
(consecutive tail time blocks), and `blocked_kfold` (chronological blocked
folds). Use `random_kfold` only when reproducing papers that explicitly used
random iid folds.

## Model selection routes

`SearchSpec` controls the candidate grid and, when needed, the selection route.
The default route scores every Cartesian grid point on the window's validation
splits. This keeps ordinary holdout and POOS designs on the `WindowSpec`:

```python
window = mf.window.from_cutoffs(
    test_start="1985-01-01",
    val_method="poos",
    val_size=24,
    retune_every=12,
)
search = mf.model_selection.grid({"alpha": [0.01, 0.1, 1.0]})
```

Use a validation-splitter override when the validation scheme belongs to the
selection rule rather than to the outer estimation/test window. Explicit fold
boundaries are end-exclusive positions inside the current selection sample;
date labels map to the position just after that label. With
`within_fold="expanding"`, each validation block becomes one single-observation
split whose training block grows through the fold:

```python
search = mf.model_selection.grid(
    {"K": [1, 2, 3], "qN": [0, 1]},
    validation_splitter=mf.model_selection.explicit_folds(
        [80, 130, 190, 240],
        within_fold="expanding",
    ),
)
```

Use information-criterion selection when candidates should be scored on the fit
sample instead of a held-out validation block. The fitted model must expose
`ssr_`, `nobs_`, and `n_params_`; AR/FAR-style RSS models do, while generic
supervised regressors do not promise that interface:

```python
search = mf.model_selection.SearchSpec(
    method="information_criterion",
    criterion="bic",
    param_grid={"n_lag": (1, 2, 4, 6, 12)},
)
```

## Per-arm windows and no-validation fits

An arm may declare its own per-arm window via `Arm(window=...)`. When an arm has
no tunable hyperparameters — the AR benchmark is the typical case — no validation
split is consumed and the model fits on the full estimation window, regardless of
the window's `val_method`.

## Retrain and retune cadence

`retrain_every` controls how often the model is refit from scratch (default: at
every origin). `retune_every` controls how often hyperparameters are re-selected.
Both accept a positive integer (number of origins) or a pandas offset string such
as `"12ME"` (every 12 month-ends). Setting `retune_on_retrain=True` (the
default) ensures retuning only happens when a new model is also fit.

## Key Callable

`mf.window.from_cutoffs` is the most common entry point. It builds a full
`WindowSpec` from named estimation/test cutoff dates and a validation design.

```python
import macroforecast as mf

# Standard POOS setup: expanding estimation, last-block val, monthly step.
window = mf.window.from_cutoffs(
    test_start="1985-01-01",
    test_end="2019-12-01",
    mode="expanding",
    val_method="last_block",
    val_ratio=0.2,
    horizon=1,
    step=1,
)

# Annual retraining with quarterly retuning.
window_annual = mf.window.from_cutoffs(
    test_start="1985-01-01",
    mode="expanding",
    val_method="last_block",
    horizon=12,
    retrain_every=12,     # refit every 12 origins
    retune_every=3,       # retune every 3 origins
    retune_on_retrain=True,
)
```

## Executed walkthrough

A monthly window from 1985-01 to 2019-12 produces one test origin per month:

```python
import pandas as pd
origins = pd.date_range("1985-01-01", "2019-12-01", freq="MS")
print("n test origins:", len(origins))
```

```text
n test origins: 420
```

`mf.window.split_table` materializes the inner validation splits for a given
design. The `last_block` design over a 300-observation estimation sample holds
out the final 20 percent as one validation block:

```python
st = mf.window.split_table("last_block", 300, validation_ratio=0.2)
print(st[["split", "n_train", "n_validation", "train_end_pos", "validation_end_pos"]])
```

```text
   split  n_train  n_validation  train_end_pos  validation_end_pos
0      0      240            60            239                 299
```

Walk-forward designs return many splits instead of one. Calling `split_table`
with `"poos"` over a 120-observation sample yields a sequence of one-step tail
splits, each adding one observation to the training block. The estimation mode
and the retrain/retune cadence then govern how often models are refit across the
420 origins.

## Reference

- [Window reference page](../../reference/window.md) — `WindowSpec`, `EstimationWindow`, `ValWindow`, `TestWindow`, `from_cutoffs`, `spec`, `plan`, and the full configuration axis table.
