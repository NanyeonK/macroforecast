# Reproducibility

Reproducibility is a first-class package feature, not an afterthought.

This page should eventually document:

- recipe export
- default profile versioning
- run identifiers
- sweep variant identifiers
- raw data cache and manifest behavior
- seeds and deterministic modes
- optional dependency versions
- custom extension provenance

Design requirements:

- an `Experiment` can be exported to a resolved recipe
- a run records the default profile and every override
- a sweep records every variant
- custom methods are named in the manifest
- artifacts are sufficient to audit what was compared
