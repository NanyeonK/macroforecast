# Getting Started

## 1. Preferred runtime grammar

The preferred package entry path is now recipe-first:
- `recipes/baselines/minimal_fred_md.yaml`
- `recipes/papers/clss2021.yaml`

Compile with:

```python
from macrocast.specs.compiler import compile_experiment_spec_from_recipe

compiled = compile_experiment_spec_from_recipe(
    'baselines/minimal_fred_md.yaml',
    preset_id='researcher_explicit',
)
```

## 2. Legacy config compatibility

`macrocast.config` still exists, but it should now be treated as a compatibility layer.
It remains available for older YAML formats and transitional execution paths.
It is not the target long-run canonical package grammar.

## 3. Current stable migration pieces

- recipe schema
- recipe loaders/validators
- recipe-aware compiled spec entry path
- benchmark family/options resolution
- path-aware output layout
- compiled spec recipe resolution metadata

## 4. Current transitional pieces

- legacy nested/flat config parsing
- some direct `config/*.yaml` operational storage
- some paper-specific replication helpers retained as migration scaffolding
