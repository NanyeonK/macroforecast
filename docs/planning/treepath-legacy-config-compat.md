# TP-012 Legacy Config Compatibility

This step does not remove `macrocast.config`.
Instead it explicitly demotes it to a compatibility layer.

Current rule:
- preferred grammar = recipes/paths
- compatibility grammar = legacy nested/flat experiment config YAML

This is an architectural clarification step so new work does not continue treating legacy config as canonical package structure.
