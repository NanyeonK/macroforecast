# Recipe-to-Compiled-Spec Integration

Current state:
- recipe compile now builds ExperimentConfig directly from recipe metadata
- recipe compile resolves through resolve_experiment_spec_from_experiment_config()
- shared feature/model constructors now live in macrocast/construction.py
- config.py remains a compatibility shell, not the only home of constructor semantics

Remaining gap:
- some runtime/storage truth still lives outside registries/ canonical files
- deeper runtime provenance still needs refinement
