# Window

[Back to reference](index.md)

`macroforecast.window` defines temporal train/validation windows. It is shared
by selection, forecasting, and later evaluation code.

## Public Functions

| Task | Functions |
| --- | --- |
| Build reusable specs | `last_block()`, `poos()`, `expanding()`, `rolling_blocks()`, `blocked_kfold()` |
| Inspect splits | `split_table()`, `WindowSpec.to_table()` |
| Low-level split generators | `last_block_split()`, `poos_split()`, `expanding_split()`, `rolling_blocks_split()`, `blocked_kfold_split()` |
| Resolve names | `resolve_window()`, `normalize_window_name()`, `make_splitter()` |

## WindowSpec

```python
macroforecast.window.WindowSpec(
    method="expanding",
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

Output methods:

| Method | Output |
| --- | --- |
| `split(n_samples)` | `list[tuple[np.ndarray, np.ndarray]]` train/validation positions. |
| `to_table(n_samples, index=None)` | pandas DataFrame with split ranges. |
| `to_dict()` | JSON-ready metadata. |

## Builders

### last_block

```python
macroforecast.window.last_block(validation_size=None, validation_ratio=0.2, embargo=0)
```

One final holdout block.

### poos

```python
macroforecast.window.poos(validation_size=None, validation_ratio=0.25, embargo=0)
```

Pseudo-out-of-sample one-step tail splits.

### expanding

```python
macroforecast.window.expanding(min_train_size=None, step=1, horizon=1, embargo=0)
```

Expanding train window with forward validation blocks.

### rolling_blocks

```python
macroforecast.window.rolling_blocks(n_blocks=3, block_size=None, embargo=0)
```

Consecutive validation blocks over the sample tail.

### blocked_kfold

```python
macroforecast.window.blocked_kfold(n_splits=5, embargo=0)
```

Chronological blocked k-fold using only past observations for training.

## Low-Level Splitters

Low-level splitters return iterators of `(train_idx, validation_idx)`.

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
| `n_train`, `n_validation` | Number of observations in each block. |
| `train_start`, `train_end` | Label range for training. |
| `validation_start`, `validation_end` | Label range for validation. |
| `*_pos` | Integer positions. |

## Name Aliases

`normalize_window_name()` and `resolve_window()` accept these aliases:

| Alias | Canonical |
| --- | --- |
| `last`, `holdout` | `last_block` |
| `poos`, `pseudo_out_of_sample` | `poos` |
| `expanding`, `time_series_split` | `expanding` |
| `rolling`, `rolling_walk_forward` | `rolling_blocks` |
| `blocked_kfold`, `block_cv`, `kfold` | `blocked_kfold` |

Example:

```python
window = mf.window.last_block(validation_size=24)
table = window.to_table(len(X), index=X.index)
```
