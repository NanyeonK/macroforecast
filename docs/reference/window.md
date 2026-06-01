# Window

[Back to reference](index.md)

`macroforecast.window` defines the estimation/val/test time frame. It is the object
passed between data, feature engineering, model selection, models, and evaluation to
answer five questions:

- how the pre-test estimation sample expands or rolls
- how validation splits are created inside the estimation sample for model selection
- where the final test origins start and end
- how far each test target horizon runs
- when the model is retrained versus reused
- where each runner stage may fit stateful operations

The public unit is one `WindowSpec`. Internally it is composed in this order:
`EstimationWindow`, `ValWindow`, `TestWindow`, then `AlignmentWindow`.

## Configuration Axes

All major macro-forecasting time-frame choices are explicit:

| Question | Setting | Where |
| --- | --- | --- |
| Should the pre-test estimation sample expand, roll, or stay fixed? | `mode` | `estimation_expanding()`, `estimation_rolling()`, `estimation_fixed()` |
| How long is a rolling estimation sample? | `size` | `estimation_rolling(size=...)` |
| What is the minimum estimation sample before a test origin is allowed? | `min_size` | `estimation_expanding(min_size=...)`, `estimation_rolling(min_size=...)`, `estimation_fixed(min_size=...)` |
| How often does the final test origin move? | `step` | `test_origins(step=...)` |
| Is test-origin movement row-count based or calendar based? | positive integer or pandas offset | `test_origins(step=1)`, `test_origins(step="1ME")`, `test_origins(step=pd.offsets.MonthEnd(3))` |
| How far does each final test target run? | `horizon` | `test_origins(horizon=...)` |
| Which inner validation design is used for model selection? | `method` | `val_last_block()`, `val_poos()`, `val_expanding()`, `val_rolling_blocks()`, `val_blocked_kfold()` |
| How many time blocks are used for tail-block validation? | `n_blocks` | `val_rolling_blocks(n_blocks=...)` |
| How large is each time block? | `block_size` | `val_rolling_blocks(block_size=...)` |
| How many chronological CV folds are used? | `n_splits` | `val_blocked_kfold(n_splits=...)` |
| How often is the model refit? | positive integer or pandas offset | `estimation_* (retrain_every=12)`, `estimation_* (retrain_every="12ME")` |
| How often are hyperparameters reselected? | positive integer or pandas offset | `val_* (retune_every=12)`, `val_* (retune_every="12ME")` |
| Should retuning happen only when the model is refit? | `retune_on_retrain` | `val_* (retune_on_retrain=True/False)` |
| Can skipped retune origins reuse the previous selected parameters? | `reuse_params` | `val_* (reuse_params=True/False)` |
| Where may preprocessing, feature engineering, or model selection fit state? | `scope` | `stage_policy("full_panel")`, `stage_policy("origin_available")`, `stage_policy("fit_window")`, `stage_policy("fixed_reference")` |

Validation choices are time-aware:

| Function | Design | Typical use |
| --- | --- | --- |
| `val_last_block(size=...)` | One final holdout block inside the estimation sample. | Simple holdout model selection. |
| `val_poos(size=...)` | Pseudo-out-of-sample one-step tail splits. | Recursive historical model selection with many tail origins. |
| `val_expanding(min_train_size=..., step=..., horizon=...)` | Expanding inner training sample and forward validation blocks. | Walk-forward validation inside each estimation window. |
| `val_rolling_blocks(n_blocks=..., block_size=...)` | Several consecutive tail time blocks. | Time-block model selection over recent history. |
| `val_blocked_kfold(n_splits=...)` | Chronological blocked folds with only past data used for training. | Time-aware CV. This is not random iid k-fold. |

```python
window = mf.window.spec(
    estimation=mf.window.estimation_rolling(
        size=120,
        embargo=1,
        retrain_every="12ME",
    ),
    val=mf.window.val_expanding(
        min_train_size=80,
        horizon=12,
        step=1,
        retune_every="12ME",
        retune_on_retrain=True,
        reuse_params=True,
    ),
    test=mf.window.test_origins(
        first_origin="2000-01-31",
        last_origin="2023-12-31",
        horizon=12,
        step="1ME",
    ),
    alignment=mf.window.alignment_drop_incomplete(),
)
```

Rolling estimation with time-block validation:

```python
window = mf.window.spec(
    estimation=mf.window.estimation_rolling(
        size=120,
        min_size=80,
        embargo=1,
        retrain_every="12ME",
    ),
    val=mf.window.val_rolling_blocks(
        n_blocks=4,
        block_size=12,
        embargo=1,
        retune_every="12ME",
        retune_on_retrain=True,
        reuse_params=True,
    ),
    test=mf.window.test_origins(
        first_origin="2000-01-31",
        last_origin="2023-12-31",
        horizon=12,
        step="1ME",
    ),
)
```

Expanding estimation with chronological blocked CV:

```python
window = mf.window.spec(
    estimation=mf.window.estimation_expanding(
        min_size=120,
        embargo=1,
        retrain_every=1,
    ),
    val=mf.window.val_blocked_kfold(
        n_splits=5,
        embargo=1,
        retune_every=1,
    ),
    test=mf.window.test_origins(
        first_origin="2000-01-31",
        last_origin="2023-12-31",
        horizon=4,
        step="1QE",
    ),
)
```

## Public Functions

| Task | Functions |
| --- | --- |
| Compose full window | `spec()` |
| Build from common cutoffs | `from_cutoffs()` |
| Configure estimation | `estimation_expanding()`, `estimation_rolling()`, `estimation_fixed()` |
| Configure val | `val_last_block()`, `val_poos()`, `val_expanding()`, `val_rolling_blocks()`, `val_blocked_kfold()` |
| Configure test | `test_origins()` |
| Configure alignment | `alignment_drop_incomplete()`, `alignment_keep_missing()` |
| Configure runner stage timing | `stage_policy()`, `custom_stage_policy()`, `stage_index()`, `stage_panel()` |
| Shortcut windows | `last_block()`, `poos()`, `expanding()`, `rolling_blocks()`, `blocked_kfold()` |
| Inspect windows | `WindowSpec.plan()`, `WindowSpec.origins()`, `WindowSpec.test_mask()`, `WindowSpec.align()`, `WindowSpec.to_table()` |
| Runner handoff | `WindowSpec.val_splits_for_origin()`, `WindowSpec.iter_origins()`, `WindowSpec.iter_slices()` |
| Low-level split generators | `last_block_split()`, `poos_split()`, `expanding_split()`, `rolling_blocks_split()`, `blocked_kfold_split()` |

## StagePolicy

```python
macroforecast.window.stage_policy(
    scope="fit_window",
    update="every_origin",
    reference_start=None,
    reference_end=None,
    apply_to=("fit", "test"),
    metadata=None,
)
```

`StagePolicy` is the runner-facing timing rule for stateful stages. It is used
as `preprocessing_policy`, `feature_policy`, and `selection_policy` in
`macroforecast.forecasting.run(...)`.

| Scope | Meaning |
| --- | --- |
| `full_panel` | Fit the stage on the complete panel once. |
| `origin_available` | Fit the stage on rows available by each origin. |
| `fit_window` | Fit the stage only on the model fit window. |
| `fixed_reference` | Fit the stage on a fixed reference period and reuse that state. |
| `custom` | Use a user selector callable to choose the allowed labels. Build this with `custom_stage_policy(...)`. |

`update` controls how often a runner refits stateful preprocessing or feature
engineering stages. Current accepted values are:

| Update | Meaning |
| --- | --- |
| `"every_origin"` | Refit the stage at every test origin. |
| `"on_retrain"` | Refit when the window row has `retrain=True`. |
| `"never"` | Fit once at the first origin and reuse the fitted state. |
| Positive integer | Refit every N origins. |
| Pandas date offset string, such as `"12ME"` | Refit when the current origin is at least the offset after the last update. |

Selection retuning follows the validation window's `retune_every` setting. The
stage `update` field is mainly for fitted preprocessing state, fixed PCA
loadings, and other feature-engineering states.

`stage_index(index, item, policy)` and `stage_panel(panel, item, policy)` are
the low-level handoff helpers used by runners. They resolve the exact rows
allowed by a policy for an origin item returned by `WindowSpec.iter_origins()`.
This keeps policy-to-index logic in `macroforecast.window`, not in
preprocessing, feature engineering, model, or model selection code.

```python
feature_policy = mf.window.stage_policy(
    "fixed_reference",
    reference_start="2000-01-31",
    reference_end="2019-12-31",
    update="never",
)
```

### custom_stage_policy

Build a `StagePolicy` whose rows are selected by user code.

```python
macroforecast.window.custom_stage_policy(
    selector,
    *,
    update="every_origin",
    apply_to=("fit", "test"),
    metadata=None,
) -> StagePolicy
```

The selector receives the full index and the current origin item:

```python
selector(index: pandas.Index, *, item: dict, policy: StagePolicy)
```

It may return:

| Return type | Meaning |
| --- | --- |
| Boolean `Series` or boolean `ndarray` | Mask over the full index. |
| `slice` | Positional slice into the full index. |
| Integer positions | Positional labels from the full index. |
| Index labels | Labels to keep. |

The selected labels must not be empty. The runner stores the policy under
`ForecastResult.metadata["stage_policies"]`; the callable itself is recorded by
name in metadata.

```python
def last_half_of_fit(index, *, item, policy):
    fit_idx = item["fit_idx"]
    return fit_idx[len(fit_idx) // 2 :]

result = mf.forecasting.run(
    panel,
    "ridge",
    window=window,
    features=features,
    model_selection_policy=mf.window.custom_stage_policy(last_half_of_fit),
)
```

## WindowSpec

```python
macroforecast.window.WindowSpec(
    method="expanding",
    estimation=EstimationWindow(...),
    val=ValWindow(...),
    test=TestWindow(...),
    alignment=AlignmentWindow(...),
    validation_size=None,
    validation_ratio=0.2,
    min_train_size=None,
    n_splits=5,
    step=1,
    horizon=1,
    embargo=0,
    metadata=None,
)
```

The explicit component fields are the preferred API. The scalar validation
fields remain to preserve the older model-selection split behavior.

Output methods:

| Method | Output | Meaning |
| --- | --- | --- |
| `split(n_samples)` | list of inner train/val integer positions | Validation splits for model-parameter selection. |
| `to_table(n_samples, index=None)` | DataFrame | Inspectable inner train/val split ranges. |
| `plan(index)` | DataFrame | Combined estimation/val/test execution plan. |
| `origins(index)` | DataFrame | Test-origin rows with estimation, fit, and test ranges. |
| `val_splits_for_origin(index, origin)` | list of train/val positions | Absolute-position inner validation splits for one test origin. |
| `iter_origins(index)` | iterator of dicts | Origin metadata plus absolute `estimation_idx`, `fit_idx`, `test_idx`, and retune-time `val_splits`. |
| `iter_slices(X, y=None)` | iterator of dicts | Same origin metadata plus sliced `X_estimation`, `X_fit`, `X_test`, `y_estimation`, `y_fit`, and `y_test`. |
| `validate(index)` | dict | User-facing validation report with `ok`, counts, errors, and warnings. |
| `test_mask(index)` | Series[bool] | Dates included in the final test region. |
| `align(X, y=None)` | DataFrame or `(X, y)` | Feature/target index alignment. |
| `to_dict()` | dict | JSON-ready metadata. |

`origins(index)` returns:

| Column | Meaning |
| --- | --- |
| `origin`, `origin_pos` | Test origin label and position. |
| `estimation_start`, `estimation_end` | Full pre-test sample available at that origin. |
| `fit_start`, `fit_end` | Sample actually used by the current fitted model. This can lag `estimation_*` when `retrain_every > 1`. |
| `test_start`, `test_end` | Test label range produced at the origin. |
| `*_pos` | Integer positions. |
| `horizon`, `step`, `test_step`, `n_estimation`, `n_fit`, `n_test` | Window sizes and test-origin movement metadata. |
| `retrain`, `retrain_group` | Whether this origin refits the model and the refit group id. |
| `retrain_cadence`, `estimation_mode` | Refit cadence metadata and estimation-window mode. |

`plan(index)` adds:

| Column | Meaning |
| --- | --- |
| `val_method` | Inner validation splitter used when retuning. |
| `retune`, `retune_group` | Whether hyperparameters are retuned at this origin and the retune group id. |
| `retune_cadence` | Hyperparameter retuning cadence metadata. |
| `retune_on_retrain` | Whether scheduled retunes are allowed only at retrain origins. |
| `reuse_params` | Whether non-retune origins may reuse the last selected parameters. |
| `selection_start`, `selection_end` | Label range of the estimation sample used for the active model-parameter selection run. Non-retune origins reuse the previous range. |
| `selection_start_pos`, `selection_end_pos`, `n_selection` | Integer positions and length for the active model-selection sample. |
| `n_val_splits` | Number of inner train/val splits evaluated at this origin. Zero when `retune=False`. |
| `val_start`, `val_end` | Label range covered by the validation folds at retune origins. |
| `val_start_pos`, `val_end_pos` | Integer positions for the validation-fold label range. |

All methods that consume an index require unique, monotonic increasing labels.
This keeps time order explicit and avoids silent reordering.

`retrain_every` and `retune_every` are separate cadences:

- `retrain_every` controls when model coefficients are refit.
- `retune_every` controls when hyperparameters are reselected.
- Positive integers count emitted test origins. `retrain_every=12` means every
  twelfth emitted origin, regardless of the calendar distance between labels.
- Pandas offsets use calendar time. `retrain_every="12ME"` means the first
  emitted origin retrains, then the next origin on or after last retrain date
  plus 12 month ends retrains.
- Calendar `retrain_every` and `retune_every` require a `DatetimeIndex`.
- With `retune_on_retrain=True`, retuning is allowed only at origins that also retrain.
- With `reuse_params=True`, skipped retune origins reuse the last selected parameters.
- With `reuse_params=False`, `validate(index)` requires every emitted origin to retune.

`TestWindow.step` can be either row-count based or calendar based:

- `step=1` means every emitted observation in the supplied index.
- `step=3` means every third emitted observation.
- `step="1ME"` means one month-end calendar move between test origins.
- `step="1QE"` means one quarter-end calendar move.
- `step=pd.DateOffset(months=3)` or a pandas offset object such as
  `pd.offsets.MonthEnd(3)` is also accepted.

Calendar/date-offset steps require a `DatetimeIndex`. If the offset lands
between two available labels, the next available label is used. For example, on
an irregular monthly panel, a target date of May 29 moves to the first available
index label on or after May 29. This calendar option applies to final test
origins only; `ValWindow.step` and the low-level validation splitters remain
row-count based because they operate inside an already selected estimation
sample.

Current row-count-only settings:

| Setting | Meaning |
| --- | --- |
| `TestWindow.horizon` | Number of rows in the final test horizon. |
| `EstimationWindow.embargo` | Number of rows between estimation end and test origin. |
| `ValWindow.step` | Row-count movement between inner validation splits. |
| `ValWindow.horizon` | Row-count length of each inner validation target block. |
| Low-level splitters | All low-level splitters operate on integer positions. |

Irregular calendar example:

```python
idx = pd.date_range("2000-01-31", periods=18, freq="ME").delete([7, 10, 14])
X = pd.DataFrame({"x": range(len(idx))}, index=idx)

window = mf.window.spec(
    estimation=mf.window.estimation_expanding(
        min_size=3,
        retrain_every="6ME",
    ),
    val=mf.window.val_last_block(
        size=2,
        retune_every="3ME",
        retune_on_retrain=False,
    ),
    test=mf.window.test_origins(
        first_origin=idx[4],
        last_origin=idx[-1],
        horizon=1,
        step="3ME",
    ),
)

plan = window.plan(X.index)
plan[[
    "origin",
    "test_step",
    "retrain",
    "retrain_cadence",
    "retune",
    "retune_cadence",
]]
```

If `idx[4] + 3ME` lands on a missing label, the next available index label is
used. The cadence metadata columns make the effective runner plan auditable
without reopening the original `WindowSpec`.

```python
plan = window.plan(X.index)
report = window.validate(X.index)

for origin in window.iter_slices(X, y):
    X_fit = origin["X_fit"]
    y_fit = origin["y_fit"]
    X_test = origin["X_test"]
    if origin["row"]["retune"]:
        inner_splits = origin["val_splits"]
```

Minimal runner loop:

```python
selected_params = None
fit = None

for origin in window.iter_slices(X, y):
    row = origin["row"]
    if row["retune"]:
        selected_params = your_selection_function(
            origin["X_fit"],
            origin["y_fit"],
            val_splits=origin["val_splits"],
        )
    if row["retrain"]:
        fit = your_model_function(origin["X_fit"], origin["y_fit"], **selected_params)
    prediction = fit.predict(origin["X_test"])
```

## from_cutoffs

```python
macroforecast.window.from_cutoffs(
    test_start,
    test_end=None,
    estimation_start=None,
    mode="expanding",
    estimation_size=None,
    estimation_min_size=None,
    embargo=0,
    retrain_every=1,
    val_method="last_block",
    val_size=None,
    val_ratio=0.2,
    val_min_train_size=None,
    val_n_splits=5,
    val_horizon=None,
    val_step=1,
    val_embargo=None,
    retune_every=1,
    retune_on_retrain=True,
    reuse_params=True,
    horizon=1,
    step=1,
)
```

This helper builds the same `WindowSpec` from common cutoff choices.

```python
window = mf.window.from_cutoffs(
    estimation_start="1960-01-31",
    test_start="2000-01-31",
    test_end="2023-12-31",
    mode="rolling",
    estimation_size=120,
    val_method="last_block",
    val_size=24,
    horizon=12,
    retrain_every="12ME",
    retune_every="12ME",
    retune_on_retrain=True,
    reuse_params=True,
    step="1ME",
)
```

## Components

### EstimationWindow

```python
macroforecast.window.EstimationWindow(
    mode="expanding",
    start=None,
    end=None,
    min_size=None,
    size=None,
    embargo=0,
    retrain_every=1,
)
```

Builders:

```python
mf.window.estimation_expanding(
    start=None,
    min_size=None,
    embargo=0,
    retrain_every=1,
)
mf.window.estimation_rolling(
    start=None,
    size=120,
    min_size=None,
    embargo=0,
    retrain_every=1,
)
mf.window.estimation_fixed(
    start=None,
    end=None,
    min_size=None,
    embargo=0,
    retrain_every=1,
)
```

Input:

| Argument | Meaning |
| --- | --- |
| `mode` | `expanding`, `rolling`, or `fixed`. |
| `start` | First allowed estimation label or integer position. |
| `end` | Last allowed estimation label or integer position for fixed bounds. |
| `min_size` | Minimum estimation observations required before an origin is emitted. |
| `size` | Rolling estimation length. Required for `estimation_rolling()`. |
| `embargo` | Gap between the last estimation observation and the test origin. |
| `retrain_every` | Refit cadence. Positive integers count emitted test origins; pandas offset strings or `DateOffset` objects use calendar time and require a `DatetimeIndex`. |

### ValWindow

```python
macroforecast.window.ValWindow(
    method="expanding",
    size=None,
    ratio=0.2,
    min_train_size=None,
    n_splits=5,
    horizon=1,
    step=1,
    embargo=None,
    retune_every=1,
    retune_on_retrain=True,
    reuse_params=True,
)
```

Builders:

```python
mf.window.val_last_block(
    size=None,
    ratio=0.2,
    embargo=None,
    retune_every=1,
    retune_on_retrain=True,
    reuse_params=True,
)
mf.window.val_poos(
    size=None,
    ratio=0.25,
    embargo=None,
    retune_every=1,
    retune_on_retrain=True,
    reuse_params=True,
)
mf.window.val_expanding(
    min_train_size=None,
    step=1,
    horizon=1,
    embargo=None,
    retune_every=1,
    retune_on_retrain=True,
    reuse_params=True,
)
mf.window.val_rolling_blocks(
    n_blocks=3,
    block_size=None,
    embargo=None,
    retune_every=1,
    retune_on_retrain=True,
    reuse_params=True,
)
mf.window.val_blocked_kfold(
    n_splits=5,
    embargo=None,
    retune_every=1,
    retune_on_retrain=True,
    reuse_params=True,
)
```

Input:

| Argument | Meaning |
| --- | --- |
| `method` | Validation splitter: `last_block`, `poos`, `expanding`, `rolling_blocks`, or `blocked_kfold`. |
| `size` | Explicit validation size for holdout-style splitters. |
| `ratio` | Validation ratio when `size` is absent. |
| `min_train_size` | Minimum inner training size for expanding validation inside the estimation window. |
| `n_splits` | Number of validation folds or blocks. |
| `horizon` | Validation target length. Defaults to one-step validation. |
| `step` | Validation split movement. |
| `embargo` | Validation-specific embargo. If absent, estimation embargo is used. |
| `retune_every` | Hyperparameter retuning cadence. Positive integers count emitted test origins; pandas offset strings or `DateOffset` objects use calendar time and require a `DatetimeIndex`. |
| `retune_on_retrain` | If `True`, a scheduled retune happens only when the same origin retrains the model. |
| `reuse_params` | If `True`, skipped retune origins reuse the most recent selected parameters. If `False`, validation fails unless every origin retunes. |

### TestWindow

```python
macroforecast.window.TestWindow(
    first_origin=None,
    last_origin=None,
    horizon=1,
    step=1,
    drop_incomplete=True,
    exclude=(),
)
```

Builder:

```python
mf.window.test_origins(
    first_origin=None,
    last_origin=None,
    horizon=1,
    step=1,
    drop_incomplete=True,
    exclude=(),
)
```

Input:

| Argument | Meaning |
| --- | --- |
| `first_origin` | First final-test origin label or integer position. |
| `last_origin` | Last final-test origin label or integer position. |
| `horizon` | Test horizon length in rows. |
| `step` | Origin movement. Positive integers move by row count. Pandas offset strings or `DateOffset` objects move by calendar time and require a `DatetimeIndex`. |
| `drop_incomplete` | Drop origins whose full horizon exceeds the available index. |
| `exclude` | Sequence of `(start, end)` windows removed from the test mask. |

### AlignmentWindow

```python
macroforecast.window.AlignmentWindow(
    join="inner",
    drop_missing=True,
    require_full_horizon=True,
)
```

Builders:

```python
mf.window.alignment_drop_incomplete(join="inner", require_full_horizon=True)
mf.window.alignment_keep_missing(join="inner", require_full_horizon=True)
```

`alignment_drop_incomplete()` removes rows with missing features or missing
targets. `alignment_keep_missing()` keeps missing feature rows after index
alignment, but with `require_full_horizon=True` it still drops rows whose target
horizon is incomplete.

## Shortcuts

Shortcuts create a full `WindowSpec` and still support model-selection-style use.

### last_block

```python
macroforecast.window.last_block(validation_size=None, validation_ratio=0.2, embargo=0)
```

One final validation block.

### poos

```python
macroforecast.window.poos(validation_size=None, validation_ratio=0.25, embargo=0)
```

Pseudo-out-of-sample one-step tail validation splits.

### expanding

```python
macroforecast.window.expanding(min_train_size=None, step=1, horizon=1, embargo=0)
```

Expanding inner-train window with forward validation blocks.

### rolling_blocks

```python
macroforecast.window.rolling_blocks(n_blocks=3, block_size=None, embargo=0)
```

Consecutive validation blocks over the sample tail.

### blocked_kfold

```python
macroforecast.window.blocked_kfold(n_splits=5, embargo=0)
```

Chronological blocked folds with past-only training.

## Low-Level Splitters

Low-level splitters return iterators of `(train_idx, val_idx)`.

```python
macroforecast.window.last_block_split(n_samples, validation_size=None, validation_ratio=0.2, embargo=0)
macroforecast.window.poos_split(n_samples, validation_size=None, validation_ratio=0.25, embargo=0)
macroforecast.window.expanding_split(n_samples, min_train_size=None, step=1, horizon=1, embargo=0)
macroforecast.window.rolling_blocks_split(n_samples, n_blocks=3, block_size=None, embargo=0)
macroforecast.window.blocked_kfold_split(n_samples, n_splits=5, embargo=0)
```

## split_table

```python
macroforecast.window.split_table(
    window,
    n_samples,
    *,
    index=None,
    validation_size=None,
    validation_ratio=0.2,
    min_train_size=None,
    n_splits=5,
    step=1,
    horizon=1,
    embargo=0,
)
```

Output columns:

| Column | Meaning |
| --- | --- |
| `split` | Split id. |
| `n_train`, `n_validation` | Split sizes. |
| `train_start`, `train_end` | Labels for the training range. |
| `validation_start`, `validation_end` | Labels for the validation range. |
| `*_pos` | Integer positions for the same ranges. |

## Aliases

`normalize_window_name()` and `resolve_window()` accept these aliases:

| Alias | Canonical |
| --- | --- |
| `last`, `holdout` | `last_block` |
| `poos`, `pseudo_out_of_sample` | `poos` |
| `expanding`, `expanding_walk_forward` | `expanding` |
| `time_series_split` | `expanding` |
| `rolling`, `rolling_walk_forward` | `rolling_blocks` |
| `blocked_kfold`, `block_cv`, `kfold` | `blocked_kfold` |
