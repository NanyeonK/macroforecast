# Goulet-Coulombe (2021) — bundled paper baseline

> Status: bundled paper-baseline replication. The recipe runs end-to-end
> on a stock install with bit-exact replicate. Real FRED-MD figure
> replication requires swapping the bundled 36-month sample panel for
> the official 1960-onwards FRED-MD vintage (one-line edit in the
> recipe; see "Real FRED-MD" section).

This page documents what the bundled
[`examples/recipes/goulet_coulombe_2021_replication.yaml`](https://github.com/NanyeonK/macroforecast/blob/main/examples/recipes/goulet_coulombe_2021_replication.yaml)
recipe does today. It is **not** a full paper replication on its own —
the bundled panel is a 36-month sample (1960-1962) for smoke
testing — but the recipe shape, layer choices, and replicate contract
are paper-faithful.

## Paper

- **Title**: "The Macroeconomy as a Random Forest"
- **Author**: Goulet-Coulombe, P. (2021)
- **Source**: Working paper / JAE 2024
- **Recipe ports the ridge baseline** of the paper's FRED-MD INDPRO
  forecasting horse race. The full MRF / GTVP route lives at
  [`examples/recipes/l4_mrf_placeholder.yaml`](https://github.com/NanyeonK/macroforecast/blob/main/examples/recipes/l4_mrf_placeholder.yaml).

## Recipe shape

| Layer | Choice |
|---|---|
| L0 | `failure_policy=fail_fast`, `reproducibility_mode=seeded_reproducible`, `random_seed=42` |
| L1 | `target=INDPRO`, `target_horizons=[1, 3, 6, 12]`, `information_set_type=final_revised_data`, custom_panel_inline (smoke; see below) |
| L2 | McCracken-Ng official tcodes (`apply_official_tcode`) |
| L3 | Lag features + `target_construction(direct, h)` |
| L4 | Ridge (`alpha=1.0`) + AR(BIC) benchmark, expanding-window walk-forward, `refit_policy=every_origin`, `forecast_strategy=direct` |
| L5 | MSE primary + relative metrics (vs AR benchmark) |
| L6 | DM (HLN-corrected) horse-race test, `dependence_correction=newey_west` |
| L8 | Manifest + per-cell artifacts |

## Today's reproduce + replicate

```bash
macroforecast run examples/recipes/goulet_coulombe_2021_replication.yaml -o out/gc2021
macroforecast replicate out/gc2021/manifest.json
```

**Verified on v0.8.6** (server1, 2026-05-06):

```
run:        1.40s, cells=1, all_ok=True
replicate:  recipe_match=True, sink_hashes_match=True

cell sink hashes (deterministic across runs):
  l1_data_definition_v1:   d164d15be5b3135c
  l1_regime_metadata_v1:   41cb7c29ec77d7c1
  l2_clean_panel_v1:       effec065444a0ac4
  l3_features_v1:          139853d1515cab8a
  l3_metadata_v1:          7c2368d756c4a66a
  l4_forecasts_v1:         395df3d0f5979dac
  l4_model_artifacts_v1:   0a2b7f63e1df2a31
  l4_training_metadata_v1: 62abd040e502ebdd
  l5_evaluation_v1:        f5ac78c21ae68f7e
  l6_tests_v1:             bc383c20f93143dd
```

Smoke regression: covered by
[`tests/test_examples_smoke.py::test_recipe_runs_end_to_end[goulet_coulombe_2021_replication.yaml]`](https://github.com/NanyeonK/macroforecast/blob/main/tests/test_examples_smoke.py).

## Real FRED-MD (full paper replication)

To replicate the actual paper figures (1960-onwards walk-forward over
the full FRED-MD vintage), swap two YAML keys:

```diff
 1_data:
   fixed_axes:
-    custom_source_policy: custom_panel_only
+    custom_source_policy: official_only
+    dataset: fred_md
     frequency: monthly
     horizon_set: custom_list
     ...
   leaf_config:
     target: INDPRO
     target_horizons: [1, 3, 6, 12]
-    custom_panel_inline:
-      ...  # 36-month smoke sample
+    sample_start_rule: max_balanced
+    sample_end_rule: latest_available
```

After the swap, the L1 loader pulls real FRED-MD via the vintage
manager (see
[`docs/architecture/layer1/index.md`](../architecture/layer1/index.md)
for vintage configuration) and the recipe runs the actual paper
horse race.

To extend to MRF (the paper's headline result), add another L4 fit
node with `family: macroeconomic_random_forest` alongside the ridge
(`is_benchmark: true`). See
[encyclopedia L4](../encyclopedia/l4/index.md) for the full family
list.

## What this page proves vs. what is still TBD

- ✅ Recipe shape is paper-faithful at the schema level (L0/L1/L2/L3/L4/L5/L6 axes match the paper's experimental design).
- ✅ End-to-end runtime is wired: every layer materialises, every sink hashes deterministically, replicate bit-exact succeeds.
- ✅ Stock-install runnable: no extras (`pip install macroforecast` is enough).
- ⏳ Full paper figures from real FRED-MD: requires the one-line swap above plus a real vintage configured. Not gated by this page; the user runs it.
- ⏳ Numerical match against the paper's reported MSEs: requires the real-FRED run plus access to the paper's appendix tables for cross-check.

## See also

- [`example_walkthrough.md`](example_walkthrough.md) — line-by-line
  layer reading using the smaller `l4_minimal_ridge.yaml`.
- [Recipe gallery](../recipe_api/gallery.md) — full
  list of bundled recipes including GC2021.
- [`encyclopedia/l4/axes/family.md`](../encyclopedia/l4/axes/family.md)
  — every L4 model family available for horse-race extension.
