# Layer 3 Training Audit

Date: 2026-04-22

Layer 3 is the forecast-generator layer. It consumes the feature matrix created
by Layer 2 and produces forecasts.

## Canonical Role

Layer 3 owns estimator and training protocol choices:

- model family;
- benchmark family;
- direct versus iterated forecast generation;
- forecast object, such as mean, median, or quantile;
- training window, minimum training size, training-start rule, and refit policy;
- model-order choices that are estimator behavior, such as AR lag selection;
- validation split, hyperparameter search, tuning objective, and tuning budget;
- model seed, early stopping, convergence handling, cache, checkpointing, and
  execution backend.

Layer 3 does not own the research feature representation grammar. It should
receive `Z_train` and `Z_pred` from Layer 2, then fit and predict.

## Boundary With Layer 2

Canonical Layer 2 ownership now includes:

- `feature_builder`;
- `predictor_family`;
- `data_richness_mode`;
- `factor_count`;
- future `feature_block_set` and feature-block primitive axes.

These axes decide how `H`, `X`, and `Y` become `Z`; they are not model
estimator choices.

Legacy runtime code still uses these names for executor dispatch. That is a
compatibility shape, not the canonical boundary. Future implementation should
split the current coarse names into explicit Layer 2 feature blocks and leave
Layer 3 with only model/training execution.

## Current Layer 3 Axes

The canonical Layer 3 registry surface is:

| Group | Axes |
|---|---|
| Forecast generator | `model_family`, `benchmark_family`, `forecast_type`, `forecast_object`, `horizon_modelization` |
| Training window | `min_train_size`, `training_start_rule`, `outer_window`, `refit_policy`, `lookback` |
| Model order | `y_lag_count` for AR/model-order selection; future target-lag feature blocks should move to Layer 2 |
| Validation/search | `validation_size_rule`, `validation_location`, `embargo_gap`, `split_family`, `shuffle_rule`, `alignment_fairness`, `search_algorithm`, `tuning_objective`, `tuning_budget`, `hp_space_style` |
| Runtime discipline | `seed_policy`, `early_stopping`, `convergence_handling`, `logging_level`, `checkpointing`, `cache_policy`, `execution_backend` |

## Compatibility Items To Clean Later

- `feature_builder`: keep accepted for runtime dispatch, but split into Layer 2
  feature-block grammar.
- `predictor_family`: keep accepted for existing guards, but express as Layer 2
  block/input selection.
- `data_richness_mode`: keep accepted while `feature_block_set` is introduced.
- `factor_count`: keep accepted while factor block axes are introduced.
- `factor_ar_lags`: split into target-lag feature block dimensions versus model
  lag-order selection.
