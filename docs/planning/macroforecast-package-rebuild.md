# Macroforecast Package Rebuild Plan

This document explains the machine-readable planning truth stored under `config/plans/package_rebuild/`.

Planning rule:
- package-first architecture
- YAML/code/tests as truth
- replication used for verification only
- Meta Layer 0 locked before deeper data-task expansion

Artifact guide:
- `00_meta_layer.yaml`: Meta Layer 0 lock
- `01_whole_package_plan.yaml`: package objective, scope, layers, closure criteria
- `02_separate_plans.yaml`: stream-by-stream plan
- `03_implementation_issues.yaml`: execution-ready issue stack
- `04_reverse_plan_check.yaml`: dependency and gap audit
- `05_registry_map.yaml`: registry families and inheritance order
- `06_contracts.yaml`: runtime I/O contracts
- `07_test_matrix.yaml`: required test coverage
- `08_e2e_plan.yaml`: end-to-end closure path

First construction rule:
- implement `0.1 experiment unit` and `0.2 fixed vs sweep axes` first
- then lock inheritance, fit scope, and failure policy
- only then expand dataset/task/preprocessing/model YAML detail

Current server1 status when this plan was written:
- meta/data/preprocessing/design/evaluation/interpretation/output/verification skeleton exists
- focused package skeleton suite is passing
- planning artifacts here define the remaining package-first completion path
