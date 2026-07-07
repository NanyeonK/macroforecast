# macroforecast.meta

[Back to reference](index.md)

Package-wide defaults such as random seed, worker count, metadata level, and context-managed overrides.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `DEFAULT_RANDOM_SEED` | data | int([x]) -> integer |
| `configure` | function | Update package-wide execution defaults and return the active config. |
| `get_config` | function | Return a copy of the current package-wide execution defaults. |
| `get_option` | function | Return one setting from the current package-wide execution defaults. |
| `resolve_n_jobs` | function | Return the configured worker count, resolving ``'auto'`` to the CPU count. |
| `MetaConfig` | class | dict() -> new empty dictionary |
| `MetadataLevel` | callable | No public docstring is available. |
| `NJobs` | callable | No public docstring is available. |
| `OnError` | callable | No public docstring is available. |
| `StageDefaultScope` | callable | No public docstring is available. |
| `reset_config` | function | Reset package-wide execution defaults to their initial values. |
| `use_config` | function | Temporarily update package-wide execution defaults inside a context. |

## Data And Module Values

### `DEFAULT_RANDOM_SEED`

Kind: `data`

```python
DEFAULT_RANDOM_SEED = 42
```

## Callable And Class Reference

### configure

Qualified name: `macroforecast.meta.config.configure`

#### Signature

```python
macroforecast.meta.configure(*, random_seed: int | None | object = <UNSET>, n_jobs: NJobs | object = <UNSET>, on_error: OnError | object = <UNSET>, verbose: int | object = <UNSET>, default_preprocessing_scope: StageDefaultScope | object = <UNSET>, default_feature_scope: StageDefaultScope | object = <UNSET>, default_selection_scope: StageDefaultScope | object = <UNSET>, metadata_level: MetadataLevel | object = <UNSET>) -> MetaConfig
```

#### Description

Update package-wide execution defaults and return the active config.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `random_seed` | keyword only | `int \| None \| object` | `<UNSET>` |
| `n_jobs` | keyword only | `NJobs \| object` | `<UNSET>` |
| `on_error` | keyword only | `OnError \| object` | `<UNSET>` |
| `verbose` | keyword only | `int \| object` | `<UNSET>` |
| `default_preprocessing_scope` | keyword only | `StageDefaultScope \| object` | `<UNSET>` |
| `default_feature_scope` | keyword only | `StageDefaultScope \| object` | `<UNSET>` |
| `default_selection_scope` | keyword only | `StageDefaultScope \| object` | `<UNSET>` |
| `metadata_level` | keyword only | `MetadataLevel \| object` | `<UNSET>` |

#### Returns

`MetaConfig`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.meta.configure(...)
```
### get_config

Qualified name: `macroforecast.meta.config.get_config`

#### Signature

```python
macroforecast.meta.get_config() -> MetaConfig
```

#### Description

Return a copy of the current package-wide execution defaults.

#### Returns

`MetaConfig`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.meta.get_config(...)
```
### get_option

Qualified name: `macroforecast.meta.config.get_option`

#### Signature

```python
macroforecast.meta.get_option(name: str) -> Any
```

#### Description

Return one setting from the current package-wide execution defaults.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.meta.get_option(...)
```
### resolve_n_jobs

Qualified name: `macroforecast.meta.config.resolve_n_jobs`

#### Signature

```python
macroforecast.meta.resolve_n_jobs() -> int
```

#### Description

Return the configured worker count, resolving ``'auto'`` to the CPU count.

This is the single resolution point so that ``meta.configure(n_jobs=...)``
actually controls parallelism in callers that opt in (e.g. tree ensembles).

#### Returns

`int`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.meta.resolve_n_jobs(...)
```
### MetaConfig

Qualified name: `macroforecast.meta.config.MetaConfig`

#### Signature

```python
macroforecast.meta.MetaConfig(...)
```

#### Description

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.meta.MetaConfig(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `clear` | `clear(...)` | D.clear() -> None.  Remove all items from D. |
| `copy` | `copy(...)` | D.copy() -> a shallow copy of D |
| `fromkeys` | `fromkeys(iterable, value=None, /)` | Create a new dictionary with keys from iterable and values set to value. |
| `get` | `get(self, key, default=None, /)` | Return the value for key if key is in the dictionary, else default. |
| `items` | `items(...)` | D.items() -> a set-like object providing a view on D's items |
| `keys` | `keys(...)` | D.keys() -> a set-like object providing a view on D's keys |
| `pop` | `pop(...)` | D.pop(k[,d]) -> v, remove specified key and return the corresponding value. |
| `popitem` | `popitem(self, /)` | Remove and return a (key, value) pair as a 2-tuple. |
| `setdefault` | `setdefault(self, key, default=None, /)` | Insert key with a value of default if key is not in the dictionary. |
| `update` | `update(...)` | D.update([E, ]**F) -> None.  Update D from dict/iterable E and F. |
| `values` | `values(...)` | D.values() -> an object providing a view on D's values |
### MetadataLevel

Qualified name: `typing.Literal`

#### Signature

```python
macroforecast.meta.MetadataLevel(*args, **kwargs)
```

#### Description

No public docstring is available.

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
# mf.meta.MetadataLevel(...)
```
### NJobs

Qualified name: `typing.Union`

#### Signature

```python
macroforecast.meta.NJobs(*args, **kwargs)
```

#### Description

No public docstring is available.

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
# mf.meta.NJobs(...)
```
### OnError

Qualified name: `typing.Literal`

#### Signature

```python
macroforecast.meta.OnError(*args, **kwargs)
```

#### Description

No public docstring is available.

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
# mf.meta.OnError(...)
```
### StageDefaultScope

Qualified name: `typing.Literal`

#### Signature

```python
macroforecast.meta.StageDefaultScope(*args, **kwargs)
```

#### Description

No public docstring is available.

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
# mf.meta.StageDefaultScope(...)
```
### reset_config

Qualified name: `macroforecast.meta.config.reset_config`

#### Signature

```python
macroforecast.meta.reset_config() -> MetaConfig
```

#### Description

Reset package-wide execution defaults to their initial values.

#### Returns

`MetaConfig`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.meta.reset_config(...)
```
### use_config

Qualified name: `macroforecast.meta.config.use_config`

#### Signature

```python
macroforecast.meta.use_config(*, random_seed: int | None | object = <UNSET>, n_jobs: NJobs | object = <UNSET>, on_error: OnError | object = <UNSET>, verbose: int | object = <UNSET>, default_preprocessing_scope: StageDefaultScope | object = <UNSET>, default_feature_scope: StageDefaultScope | object = <UNSET>, default_selection_scope: StageDefaultScope | object = <UNSET>, metadata_level: MetadataLevel | object = <UNSET>) -> Iterator[MetaConfig]
```

#### Description

Temporarily update package-wide execution defaults inside a context.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `random_seed` | keyword only | `int \| None \| object` | `<UNSET>` |
| `n_jobs` | keyword only | `NJobs \| object` | `<UNSET>` |
| `on_error` | keyword only | `OnError \| object` | `<UNSET>` |
| `verbose` | keyword only | `int \| object` | `<UNSET>` |
| `default_preprocessing_scope` | keyword only | `StageDefaultScope \| object` | `<UNSET>` |
| `default_feature_scope` | keyword only | `StageDefaultScope \| object` | `<UNSET>` |
| `default_selection_scope` | keyword only | `StageDefaultScope \| object` | `<UNSET>` |
| `metadata_level` | keyword only | `MetadataLevel \| object` | `<UNSET>` |

#### Returns

`Iterator[MetaConfig]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.meta.use_config(...)
```
