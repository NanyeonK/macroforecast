# Artifacts And Manifest

Artifacts are the durable outputs of an experiment. The manifest is the audit trail.

`Experiment.run()` wraps these files with:

- `ExperimentRunResult` for one executable recipe
- `ExperimentSweepResult` for controlled variation sweeps

The facade does not replace artifacts. It reads saved files and exposes
`forecasts`, `metrics`, `comparison`, and `manifest` directly to researchers.
Use `metrics_json` and `comparison_json` when the exact artifact payload matters.

This page should eventually document:

- output directory layout
- forecasts artifact
- metrics artifact
- comparison summary
- sweep manifest
- raw-data manifest
- custom method provenance
- default profile provenance
- package version and environment metadata

Design requirements:

- every run records resolved defaults
- every custom method records its registered name
- every sweep variant is traceable to axis values
- artifacts are readable without rerunning the experiment
- result objects point back to artifact paths
