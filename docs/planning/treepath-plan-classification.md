# plan_2026_04_09_2358 classification

Source of record:
- archive/legacy-plans/source/plan_2026_04_09_2358.md

This file classifies plan items into three buckets:
- implemented
- in_progress
- taxonomy_only

## 1. Implemented

### Architecture spine
- top-level tree split: engine modules vs experiment tree
- taxonomy / registries / recipes / runs buckets exist
- one path = one study direction is active
- CLSS now lives as recipe artifact + recipe-native example, not helper runtime path

### Compiler/runtime basics
- benchmark family + options + resolved id structure exists
- recipe-native compile path exists
- recipe compile now constructs ExperimentConfig directly from recipe metadata
- shared constructor layer exists in `macrocast/construction.py`
- path-aware output layout exists and is default when recipe/path metadata is present
- manifest supports recipe_id / taxonomy_path / tree_context

### Docs/policy
- docs present generic tree-path package first
- legacy config explicitly demoted to compatibility role
- CLSS helper runtime path removed from active package code

## 2. In progress

### Registries truth migration
- selected registry files now embed inline canonical payloads
- remaining operational storage still partly lives under config/*.yaml
- canonical-source migration is not complete for all live domains

### Runtime provenance / contract depth
- tree_context exists in manifests
- recipe/path metadata propagates further than before
- fixed-vs-sweep / richer tree semantics are still shallow in manifests/results

### Direct tree-path compiler completion
- recipe compile no longer primarily depends on load_config_from_dict()
- remaining bridge is smaller, but direct taxonomy/registries-only compilation is not fully complete end-to-end

### Docs synchronization
- active docs mostly reflect current state
- some migration/planning docs still need consolidation after implementation stabilizes

## 3. Taxonomy only

### Broad option universe from source plan
These remain mainly at taxonomy/master-registry vision level, not implemented package capability:
- many experiment-unit modes beyond current macro forecasting path
  - multi_output_joint_model
  - hierarchical_forecasting_run
  - panel_forecasting_run
  - state_space_run
  - benchmark_suite
  - ablation_study
- many compute modes
  - gpu_multi
  - distributed_cluster
- broad dataset adapter universe
  - oecd
  - imf_ifs
  - ecb_sdw
  - world_bank
  - news_text
  - satellite_proxy
  - custom_sql and many others
- richer exact real-time / release-calendar / ragged-edge policy space
- broader task / evaluation / stat / importance option universe beyond current active implementation

## Current assessment
- core architecture intent from source plan: ~90%
- full option-universe implementation from source plan: far lower
- main remaining engineering work is generic package completion, not CLSS-specific work
