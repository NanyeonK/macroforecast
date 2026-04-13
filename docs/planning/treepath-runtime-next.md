# Tree-Path Runtime Next Plan

## Goal

Move from migration bridge state to tree-path-native runtime.

## Remaining blocking gaps

### G1. Recipe compile still lowers to legacy runtime config
Current:
- `recipe -> recipe_to_runtime_config() -> legacy config dict -> old resolver`

Target:
- `recipe -> taxonomy/registries aware resolver -> compiled spec`

### G2. `macrocast/config.py` still acts as primary runtime language
Current:
- nested/flat legacy config remains the primary executable grammar

Target:
- recipe/path becomes first-class runtime grammar
- legacy config becomes compatibility shim or adapter layer

### G3. Registries layer is not operational truth yet
Current:
- `config/*.yaml` remains the effective runtime truth

Target:
- `registries/` becomes package-facing operational truth
- `config/*.yaml` either migrates or becomes compatibility storage only

### G4. Execution does not fully default to path-aware runs
Current:
- path-aware runs layout exists but callers must opt into it

Target:
- resolved recipe/path metadata automatically organizes outputs under `runs/`

### G5. CLSS helper scaffolding still exists
Target:
- recipe/path verification route replaces helper-centered route
- helper code reduced to thin shim or removed

## Next implementation sequence

1. TP-010: registries operationalization
   - start moving live defaults from `config/*.yaml` into `registries/`
   - add registry resolution helpers used by compiler/resolver

2. TP-011: tree-path-native compiler
   - replace `recipe_to_runtime_config()` bridge with taxonomy/registries aware resolver
   - compiled spec should be built from recipe path directly

3. TP-012: legacy config compatibility layer
   - demote `macrocast/config.py` to compatibility adapter
   - explicit note that legacy config is not canonical

4. TP-013: path-aware execution default
   - runtime should automatically use recipe/path metadata for output layout and manifest

5. TP-014: CLSS shim reduction
   - move verification usage to `recipes/papers/clss2021.yaml`
   - keep only minimal compatibility wrappers if still needed
