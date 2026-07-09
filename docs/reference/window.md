# macroforecast.window

[Back to reference](index.md)

Estimation, validation, test-window, split, and stage-policy definitions.

Guide context: [../guide/concepts/windows.md](../guide/concepts/windows.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `AlignmentWindow` | class | Feature/target alignment rule before model fitting. |
| `EstimationWindow` | class | Pre-test estimation-sample rule applied at each test origin. |
| `Split` | callable | Built-in immutable sequence. |
| `StagePolicy` | class | Fit/apply timing rule for one forecasting-run stage. |
| `TestWindow` | class | Final test-origin and horizon rule. |
| `ValWindow` | class | Validation rule used for model and hyperparameter selection. |
| `WindowSpec` | class | Macro forecasting time frame passed across selection/model/evaluation. |
| `alignment_drop_incomplete` | function | Alignment rule that drops rows with missing feature or target values. |
| `alignment_keep_missing` | function | Alignment rule that preserves missing rows after index alignment. |
| `blocked_kfold` | function | Configure chronological blocked k-fold validation. |
| `blocked_kfold_split` | function | Yield chronological blocked-fold splits using only past data for training. |
| `custom_stage_policy` | function | Create a stage policy whose sample labels are supplied by a callable. |
| `estimation_expanding` | function | Estimation rule that expands from ``start`` through each test origin. |
| `estimation_fixed` | function | Estimation rule with a fixed start and optional fixed end bound. |
| `estimation_rolling` | function | Estimation rule with a trailing sample size at each test origin. |
| `expanding` | function | Configure expanding-window train/val splits. |
| `expanding_split` | function | Yield expanding-window validation splits. |
| `from_cutoffs` | function | Build a window from common estimation/test cutoff dates. |
| `last_block` | function | Configure one final val block. |
| `last_block_split` | function | Yield one split with the last block held out for validation. |
| `make_splitter` | function | Build validation splits from a validation method name. |
| `normalize_window_name` | function | Return the canonical window method name for a method or alias. |
| `poos` | function | Configure pseudo-out-of-sample one-step tail splits. |
| `poos_split` | function | Yield pseudo-out-of-sample one-step validation splits over the tail block. |
| `random_kfold` | function | Configure randomly assigned iid-style K-fold validation. |
| `random_kfold_split` | function | Yield randomly assigned K-fold splits. |
| `resolve_window` | function | Return a ``WindowSpec`` from a spec, method name, or default. |
| `resolve_stage_policy` | function | Return a ``StagePolicy`` from a policy object, scope name, or default. |
| `rolling_blocks` | function | Configure consecutive validation blocks over the sample tail. |
| `rolling_blocks_split` | function | Yield consecutive validation blocks with all prior observations as training data. |
| `spec` | function | Compose a full estimation/val/test macro window from component windows. |
| `stage_index` | function | Return labels allowed by one stage policy for one origin item. |
| `stage_panel` | function | Return panel rows allowed by one stage policy for one origin item. |
| `stage_policy` | function | Create a reusable stage timing policy. |
| `split_table` | function | Return validation splits as an inspectable table. |
| `test_origins` | function | Final test-origin rule for model-stage out-of-sample runs. |
| `val_blocked_kfold` | function | Validation rule with chronological blocked folds. |
| `val_expanding` | function | Validation rule with expanding train windows. |
| `val_last_block` | function | Validation rule with one final holdout block. |
| `val_poos` | function | Validation rule with one-step pseudo-out-of-sample tail splits. |
| `val_random_kfold` | function | Validation rule with randomly assigned iid-style folds. |
| `val_rolling_blocks` | function | Validation rule with consecutive validation blocks over the sample tail. |

## Callable And Class Reference

### AlignmentWindow

Qualified name: `macroforecast.window.core.AlignmentWindow`

#### Signature

```python
macroforecast.window.AlignmentWindow(join: str = "inner", drop_missing: bool = True, require_full_horizon: bool = True) -> None
```

#### Description

Feature/target alignment rule before model fitting.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `join` | positional or keyword | `str` | `"inner"` |
| `drop_missing` | positional or keyword | `bool` | `True` |
| `require_full_horizon` | positional or keyword | `bool` | `True` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.window.AlignmentWindow(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `join` | `str` | `"inner"` |
| `drop_missing` | `bool` | `True` |
| `require_full_horizon` | `bool` | `True` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### EstimationWindow

Qualified name: `macroforecast.window.core.EstimationWindow`

#### Signature

```python
macroforecast.window.EstimationWindow(mode: str = "expanding", start: Any | None = None, end: Any | None = None, min_size: int | None = None, size: int | None = None, size_rule: Callable[[int, int], int] | None = None, size_by_horizon: Mapping[int, int] | None = None, embargo: int = 0, retrain_every: TemporalCadence = 1) -> None
```

#### Description

Pre-test estimation-sample rule applied at each test origin.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `mode` | positional or keyword | `str` | `"expanding"` |
| `start` | positional or keyword | `Any \| None` | `None` |
| `end` | positional or keyword | `Any \| None` | `None` |
| `min_size` | positional or keyword | `int \| None` | `None` |
| `size` | positional or keyword | `int \| None` | `None` |
| `size_rule` | positional or keyword | `Callable[[int, int], int] \| None` | `None` |
| `size_by_horizon` | positional or keyword | `Mapping[int, int] \| None` | `None` |
| `embargo` | positional or keyword | `int` | `0` |
| `retrain_every` | positional or keyword | `TemporalCadence` | `1` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.window.EstimationWindow(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `mode` | `str` | `"expanding"` |
| `start` | `Any \| None` | `None` |
| `end` | `Any \| None` | `None` |
| `min_size` | `int \| None` | `None` |
| `size` | `int \| None` | `None` |
| `size_rule` | `Callable[[int, int], int] \| None` | `None` |
| `size_by_horizon` | `Mapping[int, int] \| None` | `None` |
| `embargo` | `int` | `0` |
| `retrain_every` | `TemporalCadence` | `1` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### Split

Qualified name: `builtins.tuple`

#### Signature

```python
macroforecast.window.Split(*args, **kwargs)
```

#### Description

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `unspecified` | `required` |
| `kwargs` | var keyword | `unspecified` | `required` |

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.Split(...)
```
### StagePolicy

Qualified name: `macroforecast.window.policy.StagePolicy`

#### Signature

```python
macroforecast.window.StagePolicy(scope: StageScope = "fit_window", update: StageUpdate = "every_origin", reference_start: Any | None = None, reference_end: Any | None = None, apply_to: tuple[str, ...] = ('fit', 'test'), metadata: dict[str, Any] = <factory>, selector: Callable[..., Any] | None = None) -> None
```

#### Description

Fit/apply timing rule for one forecasting-run stage.

``scope`` decides what sample a stateful stage may use. ``update`` decides
when a runner should refit or reuse that stage state across forecast
origins.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `scope` | positional or keyword | `StageScope` | `"fit_window"` |
| `update` | positional or keyword | `StageUpdate` | `"every_origin"` |
| `reference_start` | positional or keyword | `Any \| None` | `None` |
| `reference_end` | positional or keyword | `Any \| None` | `None` |
| `apply_to` | positional or keyword | `tuple[str, ...]` | `("fit", "test")` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `selector` | positional or keyword | `Callable[..., Any] \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.window.StagePolicy(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `scope` | `StageScope` | `"fit_window"` |
| `update` | `StageUpdate` | `"every_origin"` |
| `reference_start` | `Any \| None` | `None` |
| `reference_end` | `Any \| None` | `None` |
| `apply_to` | `tuple[str, ...]` | `("fit", "test")` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `selector` | `Callable[..., Any] \| None` | `None` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a JSON-ready policy description. |
### TestWindow

Qualified name: `macroforecast.window.core.TestWindow`

#### Signature

```python
macroforecast.window.TestWindow(first_origin: Any | None = None, last_origin: Any | None = None, horizon: int = 1, step: TestStep = 1, drop_incomplete: bool = True, exclude: tuple[tuple[Any | None, Any | None], ...] = ()) -> None
```

#### Description

Final test-origin and horizon rule.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `first_origin` | positional or keyword | `Any \| None` | `None` |
| `last_origin` | positional or keyword | `Any \| None` | `None` |
| `horizon` | positional or keyword | `int` | `1` |
| `step` | positional or keyword | `TestStep` | `1` |
| `drop_incomplete` | positional or keyword | `bool` | `True` |
| `exclude` | positional or keyword | `tuple[tuple[Any \| None, Any \| None], ...]` | `()` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.window.TestWindow(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `first_origin` | `Any \| None` | `None` |
| `last_origin` | `Any \| None` | `None` |
| `horizon` | `int` | `1` |
| `step` | `TestStep` | `1` |
| `drop_incomplete` | `bool` | `True` |
| `exclude` | `tuple[tuple[Any \| None, Any \| None], ...]` | `()` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### ValWindow

Qualified name: `macroforecast.window.core.ValWindow`

#### Signature

```python
macroforecast.window.ValWindow(method: str = "expanding", size: int | None = None, ratio: float = 0.2, min_train_size: int | None = None, n_splits: int = 5, horizon: int = 1, step: int = 1, embargo: int | None = None, random_state: int | None = None, retune_every: TemporalCadence = 1, retune_on_retrain: bool = True, reuse_params: bool = True) -> None
```

#### Description

Validation rule used for model and hyperparameter selection.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `method` | positional or keyword | `str` | `"expanding"` |
| `size` | positional or keyword | `int \| None` | `None` |
| `ratio` | positional or keyword | `float` | `0.2` |
| `min_train_size` | positional or keyword | `int \| None` | `None` |
| `n_splits` | positional or keyword | `int` | `5` |
| `horizon` | positional or keyword | `int` | `1` |
| `step` | positional or keyword | `int` | `1` |
| `embargo` | positional or keyword | `int \| None` | `None` |
| `random_state` | positional or keyword | `int \| None` | `None` |
| `retune_every` | positional or keyword | `TemporalCadence` | `1` |
| `retune_on_retrain` | positional or keyword | `bool` | `True` |
| `reuse_params` | positional or keyword | `bool` | `True` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.window.ValWindow(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `method` | `str` | `"expanding"` |
| `size` | `int \| None` | `None` |
| `ratio` | `float` | `0.2` |
| `min_train_size` | `int \| None` | `None` |
| `n_splits` | `int` | `5` |
| `horizon` | `int` | `1` |
| `step` | `int` | `1` |
| `embargo` | `int \| None` | `None` |
| `random_state` | `int \| None` | `None` |
| `retune_every` | `TemporalCadence` | `1` |
| `retune_on_retrain` | `bool` | `True` |
| `reuse_params` | `bool` | `True` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### WindowSpec

Qualified name: `macroforecast.window.core.WindowSpec`

#### Signature

```python
macroforecast.window.WindowSpec(method: str = "expanding", estimation: EstimationWindow = <factory>, val: ValWindow = <factory>, test: TestWindow = <factory>, alignment: AlignmentWindow = <factory>, validation_size: int | None = None, validation_ratio: float = 0.2, min_train_size: int | None = None, n_splits: int = 5, step: int = 1, horizon: int = 1, embargo: int = 0, metadata: dict[str, Any] | None = None) -> None
```

#### Description

Macro forecasting time frame passed across selection/model/evaluation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `method` | positional or keyword | `str` | `"expanding"` |
| `estimation` | positional or keyword | `EstimationWindow` | `<factory>` |
| `val` | positional or keyword | `ValWindow` | `<factory>` |
| `test` | positional or keyword | `TestWindow` | `<factory>` |
| `alignment` | positional or keyword | `AlignmentWindow` | `<factory>` |
| `validation_size` | positional or keyword | `int \| None` | `None` |
| `validation_ratio` | positional or keyword | `float` | `0.2` |
| `min_train_size` | positional or keyword | `int \| None` | `None` |
| `n_splits` | positional or keyword | `int` | `5` |
| `step` | positional or keyword | `int` | `1` |
| `horizon` | positional or keyword | `int` | `1` |
| `embargo` | positional or keyword | `int` | `0` |
| `metadata` | positional or keyword | `dict[str, Any] \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.window.WindowSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `method` | `str` | `"expanding"` |
| `estimation` | `EstimationWindow` | `default_factory` |
| `val` | `ValWindow` | `default_factory` |
| `test` | `TestWindow` | `default_factory` |
| `alignment` | `AlignmentWindow` | `default_factory` |
| `validation_size` | `int \| None` | `None` |
| `validation_ratio` | `float` | `0.2` |
| `min_train_size` | `int \| None` | `None` |
| `n_splits` | `int` | `5` |
| `step` | `int` | `1` |
| `horizon` | `int` | `1` |
| `embargo` | `int` | `0` |
| `metadata` | `dict[str, Any] \| None` | `None` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `align` | `align(self, X: pd.DataFrame \| pd.Series, y: pd.Series \| pd.DataFrame \| None = None) -> pd.DataFrame \| tuple[pd.DataFrame, pd.Series \| pd.DataFrame]` | Align feature and target objects according to the alignment rule. |
| `iter_origins` | `iter_origins(self, index: int \| Sequence[Any] \| pd.Index, *, exclude_origin: bool = False) -> Iterator[dict[str, Any]]` | Yield origin metadata and absolute-position slices for model runners. |
| `iter_slices` | `iter_slices(self, X: pd.DataFrame \| pd.Series, y: pd.Series \| pd.DataFrame \| None = None) -> Iterator[dict[str, Any]]` | Yield origin metadata with already sliced ``X`` and optional ``y``. |
| `origins` | `origins(self, index: int \| Sequence[Any] \| pd.Index, *, exclude_origin: bool = False) -> pd.DataFrame` | Return test-origin rows with train and test ranges. |
| `plan` | `plan(self, index: int \| Sequence[Any] \| pd.Index, *, exclude_origin: bool = False) -> pd.DataFrame` | Return an execution plan with estimation, val, and test metadata. |
| `split` | `split(self, n_samples: int) -> list[Split]` | Return train/val index splits for ``n_samples``. |
| `test_mask` | `test_mask(self, index: int \| Sequence[Any] \| pd.Index) -> pd.Series` | Return a boolean final-test mask indexed by the supplied labels. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a metadata representation of the window. |
| `to_table` | `to_table(self, n_samples: int, *, index: pd.Index \| None = None) -> pd.DataFrame` | Return this window's train/val splits as an inspectable table. |
| `val_splits_for_origin` | `val_splits_for_origin(self, index: int \| Sequence[Any] \| pd.Index, origin: Any, *, exclude_origin: bool = False) -> list[Split]` | Return absolute-position inner train/val splits for one test origin. |
| `validate` | `validate(self, index: int \| Sequence[Any] \| pd.Index, *, exclude_origin: bool = False) -> dict[str, Any]` | Return a validation report for this time-frame specification. |
### alignment_drop_incomplete

Qualified name: `macroforecast.window.core.alignment_drop_incomplete`

#### Signature

```python
macroforecast.window.alignment_drop_incomplete(*, join: str = "inner", require_full_horizon: bool = True) -> AlignmentWindow
```

#### Description

Alignment rule that drops rows with missing feature or target values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `join` | keyword only | `str` | `"inner"` |
| `require_full_horizon` | keyword only | `bool` | `True` |

#### Returns

`AlignmentWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.alignment_drop_incomplete(...)
```
### alignment_keep_missing

Qualified name: `macroforecast.window.core.alignment_keep_missing`

#### Signature

```python
macroforecast.window.alignment_keep_missing(*, join: str = "inner", require_full_horizon: bool = True) -> AlignmentWindow
```

#### Description

Alignment rule that preserves missing rows after index alignment.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `join` | keyword only | `str` | `"inner"` |
| `require_full_horizon` | keyword only | `bool` | `True` |

#### Returns

`AlignmentWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.alignment_keep_missing(...)
```
### blocked_kfold

Qualified name: `macroforecast.window.core.blocked_kfold`

#### Signature

```python
macroforecast.window.blocked_kfold(*, n_splits: int = 5, embargo: int = 0) -> WindowSpec
```

#### Description

Configure chronological blocked k-fold validation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_splits` | keyword only | `int` | `5` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.blocked_kfold(...)
```
### blocked_kfold_split

Qualified name: `macroforecast.window.core.blocked_kfold_split`

#### Signature

```python
macroforecast.window.blocked_kfold_split(n_samples: int, *, n_splits: int = 5, embargo: int = 0) -> Iterator[Split]
```

#### Description

Yield chronological blocked-fold splits using only past data for training.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_samples` | positional or keyword | `int` | `required` |
| `n_splits` | keyword only | `int` | `5` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`Iterator[Split]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.blocked_kfold_split(...)
```
### custom_stage_policy

Qualified name: `macroforecast.window.policy.custom_stage_policy`

#### Signature

```python
macroforecast.window.custom_stage_policy(selector: Callable[..., Any], *, update: StageUpdate = "every_origin", apply_to: tuple[str, ...] | list[str] = ('fit', 'test'), metadata: Mapping[str, Any] | None = None) -> StagePolicy
```

#### Description

Create a stage policy whose sample labels are supplied by a callable.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `selector` | positional or keyword | `Callable[..., Any]` | `required` |
| `update` | keyword only | `StageUpdate` | `"every_origin"` |
| `apply_to` | keyword only | `tuple[str, ...] \| list[str]` | `("fit", "test")` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`StagePolicy`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.custom_stage_policy(...)
```
### estimation_expanding

Qualified name: `macroforecast.window.core.estimation_expanding`

#### Signature

```python
macroforecast.window.estimation_expanding(*, start: Any | None = None, min_size: int | None = None, embargo: int = 0, retrain_every: TemporalCadence = 1) -> EstimationWindow
```

#### Description

Estimation rule that expands from ``start`` through each test origin.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `start` | keyword only | `Any \| None` | `None` |
| `min_size` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int` | `0` |
| `retrain_every` | keyword only | `TemporalCadence` | `1` |

#### Returns

`EstimationWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.estimation_expanding(...)
```
### estimation_fixed

Qualified name: `macroforecast.window.core.estimation_fixed`

#### Signature

```python
macroforecast.window.estimation_fixed(*, start: Any | None = None, end: Any | None = None, min_size: int | None = None, embargo: int = 0, retrain_every: TemporalCadence = 1) -> EstimationWindow
```

#### Description

Estimation rule with a fixed start and optional fixed end bound.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `start` | keyword only | `Any \| None` | `None` |
| `end` | keyword only | `Any \| None` | `None` |
| `min_size` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int` | `0` |
| `retrain_every` | keyword only | `TemporalCadence` | `1` |

#### Returns

`EstimationWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.estimation_fixed(...)
```
### estimation_rolling

Qualified name: `macroforecast.window.core.estimation_rolling`

#### Signature

```python
macroforecast.window.estimation_rolling(*, start: Any | None = None, size: int | None = None, size_rule: Callable[[int, int], int] | None = None, size_by_horizon: Mapping[int, int] | None = None, min_size: int | None = None, embargo: int = 0, retrain_every: TemporalCadence = 1) -> EstimationWindow
```

#### Description

Estimation rule with a trailing sample size at each test origin.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `start` | keyword only | `Any \| None` | `None` |
| `size` | keyword only | `int \| None` | `None` |
| `size_rule` | keyword only | `Callable[[int, int], int] \| None` | `None` |
| `size_by_horizon` | keyword only | `Mapping[int, int] \| None` | `None` |
| `min_size` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int` | `0` |
| `retrain_every` | keyword only | `TemporalCadence` | `1` |

#### Returns

`EstimationWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.estimation_rolling(...)
```
### expanding

Qualified name: `macroforecast.window.core.expanding`

#### Signature

```python
macroforecast.window.expanding(*, min_train_size: int | None = None, step: int = 1, horizon: int = 1, embargo: int = 0) -> WindowSpec
```

#### Description

Configure expanding-window train/val splits.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `step` | keyword only | `int` | `1` |
| `horizon` | keyword only | `int` | `1` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.expanding(...)
```
### expanding_split

Qualified name: `macroforecast.window.core.expanding_split`

#### Signature

```python
macroforecast.window.expanding_split(n_samples: int, *, min_train_size: int | None = None, step: int = 1, horizon: int = 1, embargo: int = 0) -> Iterator[Split]
```

#### Description

Yield expanding-window validation splits.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_samples` | positional or keyword | `int` | `required` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `step` | keyword only | `int` | `1` |
| `horizon` | keyword only | `int` | `1` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`Iterator[Split]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.expanding_split(...)
```
### from_cutoffs

Qualified name: `macroforecast.window.core.from_cutoffs`

#### Signature

```python
macroforecast.window.from_cutoffs(*, test_start: Any, test_end: Any | None = None, estimation_start: Any | None = None, mode: str = "expanding", estimation_size: int | None = None, estimation_size_rule: Callable[[int, int], int] | None = None, estimation_size_by_horizon: Mapping[int, int] | None = None, estimation_min_size: int | None = None, embargo: int = 0, retrain_every: TemporalCadence = 1, val_method: str = "last_block", val_size: int | None = None, val_ratio: float = 0.2, val_min_train_size: int | None = None, val_n_splits: int = 5, val_horizon: int | None = None, val_step: int = 1, val_embargo: int | None = None, val_random_state: int | None = None, retune_every: TemporalCadence = 1, retune_on_retrain: bool = True, reuse_params: bool = True, horizon: int = 1, step: TestStep = 1, drop_incomplete: bool = True, exclude: Sequence[tuple[Any | None, Any | None]] = (), alignment: AlignmentWindow | None = None, metadata: dict[str, Any] | None = None) -> WindowSpec
```

#### Description

Build a window from common estimation/test cutoff dates.

Embargo conventions for h-step targets (``horizon > 1``)
A row at position ``p`` carries the direct h-step target realised at
``p + horizon``. Two boundaries are embargoed independently:

* Estimation/test boundary (``embargo``, default ``0``). With ``embargo=0``
  the production model for origin ``t`` is fit on rows up to ``t - 1`` whose
  targets realise at ``t + horizon - 1`` -- i.e. after the forecast is
  issued. This is the standard *pseudo-out-of-sample* convention used on a
  fixed (final-vintage) dataset, and it maximises the training sample. For a
  *strict real-time* protocol, where every training label must be observable
  at the origin, pass ``embargo=horizon - 1`` (the last training label then
  realises at the origin). The default is deliberately the pseudo-OOS choice;
  it does not enforce real-time observability.

* Train/validation boundary (``val_embargo``, default ``horizon - 1``). This
  purges the gap between the last training label and the first validation
  input. The default ``horizon - 1`` leaves the single boundary timestamp
  shared between the last training label and the first validation feature;
  pass ``val_embargo=horizon`` for a fully disjoint purge.

Both defaults are conventions, not guarantees of real-time observability;
set the embargoes explicitly when the forecasting protocol requires it.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `test_start` | keyword only | `Any` | `required` |
| `test_end` | keyword only | `Any \| None` | `None` |
| `estimation_start` | keyword only | `Any \| None` | `None` |
| `mode` | keyword only | `str` | `"expanding"` |
| `estimation_size` | keyword only | `int \| None` | `None` |
| `estimation_size_rule` | keyword only | `Callable[[int, int], int] \| None` | `None` |
| `estimation_size_by_horizon` | keyword only | `Mapping[int, int] \| None` | `None` |
| `estimation_min_size` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int` | `0` |
| `retrain_every` | keyword only | `TemporalCadence` | `1` |
| `val_method` | keyword only | `str` | `"last_block"` |
| `val_size` | keyword only | `int \| None` | `None` |
| `val_ratio` | keyword only | `float` | `0.2` |
| `val_min_train_size` | keyword only | `int \| None` | `None` |
| `val_n_splits` | keyword only | `int` | `5` |
| `val_horizon` | keyword only | `int \| None` | `None` |
| `val_step` | keyword only | `int` | `1` |
| `val_embargo` | keyword only | `int \| None` | `None` |
| `val_random_state` | keyword only | `int \| None` | `None` |
| `retune_every` | keyword only | `TemporalCadence` | `1` |
| `retune_on_retrain` | keyword only | `bool` | `True` |
| `reuse_params` | keyword only | `bool` | `True` |
| `horizon` | keyword only | `int` | `1` |
| `step` | keyword only | `TestStep` | `1` |
| `drop_incomplete` | keyword only | `bool` | `True` |
| `exclude` | keyword only | `Sequence[tuple[Any \| None, Any \| None]]` | `()` |
| `alignment` | keyword only | `AlignmentWindow \| None` | `None` |
| `metadata` | keyword only | `dict[str, Any] \| None` | `None` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.from_cutoffs(...)
```
### last_block

Qualified name: `macroforecast.window.core.last_block`

#### Signature

```python
macroforecast.window.last_block(*, validation_size: int | None = None, validation_ratio: float = 0.2, embargo: int = 0) -> WindowSpec
```

#### Description

Configure one final val block.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `validation_size` | keyword only | `int \| None` | `None` |
| `validation_ratio` | keyword only | `float` | `0.2` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.last_block(...)
```
### last_block_split

Qualified name: `macroforecast.window.core.last_block_split`

#### Signature

```python
macroforecast.window.last_block_split(n_samples: int, *, validation_size: int | None = None, validation_ratio: float = 0.2, embargo: int = 0) -> Iterator[Split]
```

#### Description

Yield one split with the last block held out for validation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_samples` | positional or keyword | `int` | `required` |
| `validation_size` | keyword only | `int \| None` | `None` |
| `validation_ratio` | keyword only | `float` | `0.2` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`Iterator[Split]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.last_block_split(...)
```
### make_splitter

Qualified name: `macroforecast.window.core.make_splitter`

#### Signature

```python
macroforecast.window.make_splitter(validation: str, n_samples: int, *, validation_size: int | None = None, validation_ratio: float = 0.2, min_train_size: int | None = None, n_splits: int = 5, step: int = 1, horizon: int = 1, random_state: int | None = None, embargo: int = 0) -> list[Split]
```

#### Description

Build validation splits from a validation method name.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `validation` | positional or keyword | `str` | `required` |
| `n_samples` | positional or keyword | `int` | `required` |
| `validation_size` | keyword only | `int \| None` | `None` |
| `validation_ratio` | keyword only | `float` | `0.2` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `n_splits` | keyword only | `int` | `5` |
| `step` | keyword only | `int` | `1` |
| `horizon` | keyword only | `int` | `1` |
| `random_state` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`list[Split]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.make_splitter(...)
```
### normalize_window_name

Qualified name: `macroforecast.window.core.normalize_window_name`

#### Signature

```python
macroforecast.window.normalize_window_name(window: str) -> str
```

#### Description

Return the canonical window method name for a method or alias.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `window` | positional or keyword | `str` | `required` |

#### Returns

`str`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.normalize_window_name(...)
```
### poos

Qualified name: `macroforecast.window.core.poos`

#### Signature

```python
macroforecast.window.poos(*, validation_size: int | None = None, validation_ratio: float = 0.25, embargo: int = 0) -> WindowSpec
```

#### Description

Configure pseudo-out-of-sample one-step tail splits.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `validation_size` | keyword only | `int \| None` | `None` |
| `validation_ratio` | keyword only | `float` | `0.25` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.poos(...)
```
### poos_split

Qualified name: `macroforecast.window.core.poos_split`

#### Signature

```python
macroforecast.window.poos_split(n_samples: int, *, validation_size: int | None = None, validation_ratio: float = 0.25, embargo: int = 0) -> Iterator[Split]
```

#### Description

Yield pseudo-out-of-sample one-step validation splits over the tail block.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_samples` | positional or keyword | `int` | `required` |
| `validation_size` | keyword only | `int \| None` | `None` |
| `validation_ratio` | keyword only | `float` | `0.25` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`Iterator[Split]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.poos_split(...)
```
### random_kfold

Qualified name: `macroforecast.window.core.random_kfold`

#### Signature

```python
macroforecast.window.random_kfold(*, n_splits: int = 5, random_state: int | None = 0) -> WindowSpec
```

#### Description

Configure randomly assigned iid-style K-fold validation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_splits` | keyword only | `int` | `5` |
| `random_state` | keyword only | `int \| None` | `0` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.random_kfold(...)
```
### random_kfold_split

Qualified name: `macroforecast.window.core.random_kfold_split`

#### Signature

```python
macroforecast.window.random_kfold_split(n_samples: int, *, n_splits: int = 5, random_state: int | None = 0) -> Iterator[Split]
```

#### Description

Yield randomly assigned K-fold splits.

Each fold trains on all non-validation positions. This intentionally does
not enforce temporal ordering, so use it only when reproducing methods that
used random iid folds, not as the default macro forecast validation design.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_samples` | positional or keyword | `int` | `required` |
| `n_splits` | keyword only | `int` | `5` |
| `random_state` | keyword only | `int \| None` | `0` |

#### Returns

`Iterator[Split]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.random_kfold_split(...)
```
### resolve_window

Qualified name: `macroforecast.window.core.resolve_window`

#### Signature

```python
macroforecast.window.resolve_window(window: WindowSpec | str | None = None) -> WindowSpec
```

#### Description

Return a ``WindowSpec`` from a spec, method name, or default.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `window` | positional or keyword | `WindowSpec \| str \| None` | `None` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.resolve_window(...)
```
### resolve_stage_policy

Qualified name: `macroforecast.window.policy.resolve_stage_policy`

#### Signature

```python
macroforecast.window.resolve_stage_policy(policy: StagePolicy | str | None, *, default_scope: StageScope = "fit_window") -> StagePolicy
```

#### Description

Return a ``StagePolicy`` from a policy object, scope name, or default.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `policy` | positional or keyword | `StagePolicy \| str \| None` | `required` |
| `default_scope` | keyword only | `StageScope` | `"fit_window"` |

#### Returns

`StagePolicy`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.resolve_stage_policy(...)
```
### rolling_blocks

Qualified name: `macroforecast.window.core.rolling_blocks`

#### Signature

```python
macroforecast.window.rolling_blocks(*, n_blocks: int = 3, block_size: int | None = None, embargo: int = 0) -> WindowSpec
```

#### Description

Configure consecutive validation blocks over the sample tail.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_blocks` | keyword only | `int` | `3` |
| `block_size` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.rolling_blocks(...)
```
### rolling_blocks_split

Qualified name: `macroforecast.window.core.rolling_blocks_split`

#### Signature

```python
macroforecast.window.rolling_blocks_split(n_samples: int, *, n_blocks: int = 3, block_size: int | None = None, embargo: int = 0) -> Iterator[Split]
```

#### Description

Yield consecutive validation blocks with all prior observations as training data.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_samples` | positional or keyword | `int` | `required` |
| `n_blocks` | keyword only | `int` | `3` |
| `block_size` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`Iterator[Split]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.rolling_blocks_split(...)
```
### spec

Qualified name: `macroforecast.window.core.spec`

#### Signature

```python
macroforecast.window.spec(*, estimation: EstimationWindow | None = None, val: ValWindow | None = None, test: TestWindow | None = None, alignment: AlignmentWindow | None = None, method: str = "expanding", metadata: dict[str, Any] | None = None) -> WindowSpec
```

#### Description

Compose a full estimation/val/test macro window from component windows.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `estimation` | keyword only | `EstimationWindow \| None` | `None` |
| `val` | keyword only | `ValWindow \| None` | `None` |
| `test` | keyword only | `TestWindow \| None` | `None` |
| `alignment` | keyword only | `AlignmentWindow \| None` | `None` |
| `method` | keyword only | `str` | `"expanding"` |
| `metadata` | keyword only | `dict[str, Any] \| None` | `None` |

#### Returns

`WindowSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.spec(...)
```
### stage_index

Qualified name: `macroforecast.window.policy.stage_index`

#### Signature

```python
macroforecast.window.stage_index(index: Any, item: Mapping[str, Any] | None, policy: StagePolicy | str | None) -> pd.Index
```

#### Description

Return labels allowed by one stage policy for one origin item.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `index` | positional or keyword | `Any` | `required` |
| `item` | positional or keyword | `Mapping[str, Any] \| None` | `required` |
| `policy` | positional or keyword | `StagePolicy \| str \| None` | `required` |

#### Returns

`pd.Index`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.stage_index(...)
```
### stage_panel

Qualified name: `macroforecast.window.policy.stage_panel`

#### Signature

```python
macroforecast.window.stage_panel(panel: pd.DataFrame, item: Mapping[str, Any] | None, policy: StagePolicy | str | None) -> pd.DataFrame
```

#### Description

Return panel rows allowed by one stage policy for one origin item.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `item` | positional or keyword | `Mapping[str, Any] \| None` | `required` |
| `policy` | positional or keyword | `StagePolicy \| str \| None` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.stage_panel(...)
```
### stage_policy

Qualified name: `macroforecast.window.policy.stage_policy`

#### Signature

```python
macroforecast.window.stage_policy(scope: StageScope = "fit_window", *, update: StageUpdate = "every_origin", reference_start: Any | None = None, reference_end: Any | None = None, apply_to: tuple[str, ...] | list[str] = ('fit', 'test'), metadata: Mapping[str, Any] | None = None, selector: Callable[..., Any] | None = None) -> StagePolicy
```

#### Description

Create a reusable stage timing policy.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `scope` | positional or keyword | `StageScope` | `"fit_window"` |
| `update` | keyword only | `StageUpdate` | `"every_origin"` |
| `reference_start` | keyword only | `Any \| None` | `None` |
| `reference_end` | keyword only | `Any \| None` | `None` |
| `apply_to` | keyword only | `tuple[str, ...] \| list[str]` | `("fit", "test")` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `selector` | keyword only | `Callable[..., Any] \| None` | `None` |

#### Returns

`StagePolicy`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.stage_policy(...)
```
### split_table

Qualified name: `macroforecast.window.core.split_table`

#### Signature

```python
macroforecast.window.split_table(validation: str, n_samples: int, *, index: pd.Index | None = None, validation_size: int | None = None, validation_ratio: float = 0.2, min_train_size: int | None = None, n_splits: int = 5, step: int = 1, horizon: int = 1, random_state: int | None = None, embargo: int = 0) -> pd.DataFrame
```

#### Description

Return validation splits as an inspectable table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `validation` | positional or keyword | `str` | `required` |
| `n_samples` | positional or keyword | `int` | `required` |
| `index` | keyword only | `pd.Index \| None` | `None` |
| `validation_size` | keyword only | `int \| None` | `None` |
| `validation_ratio` | keyword only | `float` | `0.2` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `n_splits` | keyword only | `int` | `5` |
| `step` | keyword only | `int` | `1` |
| `horizon` | keyword only | `int` | `1` |
| `random_state` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int` | `0` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.split_table(...)
```
### test_origins

Qualified name: `macroforecast.window.core.test_origins`

#### Signature

```python
macroforecast.window.test_origins(*, first_origin: Any | None = None, last_origin: Any | None = None, horizon: int = 1, step: TestStep = 1, drop_incomplete: bool = True, exclude: Sequence[tuple[Any | None, Any | None]] = ()) -> TestWindow
```

#### Description

Final test-origin rule for model-stage out-of-sample runs.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `first_origin` | keyword only | `Any \| None` | `None` |
| `last_origin` | keyword only | `Any \| None` | `None` |
| `horizon` | keyword only | `int` | `1` |
| `step` | keyword only | `TestStep` | `1` |
| `drop_incomplete` | keyword only | `bool` | `True` |
| `exclude` | keyword only | `Sequence[tuple[Any \| None, Any \| None]]` | `()` |

#### Returns

`TestWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.test_origins(...)
```
### val_blocked_kfold

Qualified name: `macroforecast.window.core.val_blocked_kfold`

#### Signature

```python
macroforecast.window.val_blocked_kfold(*, n_splits: int = 5, embargo: int | None = None, retune_every: TemporalCadence = 1, retune_on_retrain: bool = True, reuse_params: bool = True) -> ValWindow
```

#### Description

Validation rule with chronological blocked folds.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_splits` | keyword only | `int` | `5` |
| `embargo` | keyword only | `int \| None` | `None` |
| `retune_every` | keyword only | `TemporalCadence` | `1` |
| `retune_on_retrain` | keyword only | `bool` | `True` |
| `reuse_params` | keyword only | `bool` | `True` |

#### Returns

`ValWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.val_blocked_kfold(...)
```
### val_expanding

Qualified name: `macroforecast.window.core.val_expanding`

#### Signature

```python
macroforecast.window.val_expanding(*, min_train_size: int | None = None, step: int = 1, horizon: int = 1, embargo: int | None = None, retune_every: TemporalCadence = 1, retune_on_retrain: bool = True, reuse_params: bool = True) -> ValWindow
```

#### Description

Validation rule with expanding train windows.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `step` | keyword only | `int` | `1` |
| `horizon` | keyword only | `int` | `1` |
| `embargo` | keyword only | `int \| None` | `None` |
| `retune_every` | keyword only | `TemporalCadence` | `1` |
| `retune_on_retrain` | keyword only | `bool` | `True` |
| `reuse_params` | keyword only | `bool` | `True` |

#### Returns

`ValWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.val_expanding(...)
```
### val_last_block

Qualified name: `macroforecast.window.core.val_last_block`

#### Signature

```python
macroforecast.window.val_last_block(*, size: int | None = None, ratio: float = 0.2, embargo: int | None = None, retune_every: TemporalCadence = 1, retune_on_retrain: bool = True, reuse_params: bool = True) -> ValWindow
```

#### Description

Validation rule with one final holdout block.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `size` | keyword only | `int \| None` | `None` |
| `ratio` | keyword only | `float` | `0.2` |
| `embargo` | keyword only | `int \| None` | `None` |
| `retune_every` | keyword only | `TemporalCadence` | `1` |
| `retune_on_retrain` | keyword only | `bool` | `True` |
| `reuse_params` | keyword only | `bool` | `True` |

#### Returns

`ValWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.val_last_block(...)
```
### val_poos

Qualified name: `macroforecast.window.core.val_poos`

#### Signature

```python
macroforecast.window.val_poos(*, size: int | None = None, ratio: float = 0.25, embargo: int | None = None, retune_every: TemporalCadence = 1, retune_on_retrain: bool = True, reuse_params: bool = True) -> ValWindow
```

#### Description

Validation rule with one-step pseudo-out-of-sample tail splits.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `size` | keyword only | `int \| None` | `None` |
| `ratio` | keyword only | `float` | `0.25` |
| `embargo` | keyword only | `int \| None` | `None` |
| `retune_every` | keyword only | `TemporalCadence` | `1` |
| `retune_on_retrain` | keyword only | `bool` | `True` |
| `reuse_params` | keyword only | `bool` | `True` |

#### Returns

`ValWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.val_poos(...)
```
### val_random_kfold

Qualified name: `macroforecast.window.core.val_random_kfold`

#### Signature

```python
macroforecast.window.val_random_kfold(*, n_splits: int = 5, random_state: int | None = 0, retune_every: TemporalCadence = 1, retune_on_retrain: bool = True, reuse_params: bool = True) -> ValWindow
```

#### Description

Validation rule with randomly assigned iid-style folds.

This is useful for reproducing papers that explicitly used random K-fold
CV. It is not the default macro-forecasting validation rule because train
folds can contain observations later than their validation folds.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_splits` | keyword only | `int` | `5` |
| `random_state` | keyword only | `int \| None` | `0` |
| `retune_every` | keyword only | `TemporalCadence` | `1` |
| `retune_on_retrain` | keyword only | `bool` | `True` |
| `reuse_params` | keyword only | `bool` | `True` |

#### Returns

`ValWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.val_random_kfold(...)
```
### val_rolling_blocks

Qualified name: `macroforecast.window.core.val_rolling_blocks`

#### Signature

```python
macroforecast.window.val_rolling_blocks(*, n_blocks: int = 3, block_size: int | None = None, embargo: int | None = None, retune_every: TemporalCadence = 1, retune_on_retrain: bool = True, reuse_params: bool = True) -> ValWindow
```

#### Description

Validation rule with consecutive validation blocks over the sample tail.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_blocks` | keyword only | `int` | `3` |
| `block_size` | keyword only | `int \| None` | `None` |
| `embargo` | keyword only | `int \| None` | `None` |
| `retune_every` | keyword only | `TemporalCadence` | `1` |
| `retune_on_retrain` | keyword only | `bool` | `True` |
| `reuse_params` | keyword only | `bool` | `True` |

#### Returns

`ValWindow`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.val_rolling_blocks(...)
```
