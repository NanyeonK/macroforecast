# TP-013 Path-Aware Runtime Default

Runtime output layout should default to recipe/path-aware organization whenever recipe metadata exists.

First-pass implementation target:
- if `ForecastExperiment` receives `recipe_id`, use recipe-based `runs/recipes/...` layout
- else if it receives `taxonomy_path`, use path-based `runs/paths/...` layout
- else keep ad hoc fallback
