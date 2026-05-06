# Troubleshooting & FAQ

When a macroforecast recipe / API call fails, walk through these
checks before filing an issue. Most failures hit one of the patterns
below.

## Recipe fails immediately at L1

```
RuntimeError: cell cell_001 failed: single_target requires leaf_config.target string
```

**Cause**: L1 `target_structure=single_target` (the default) but
`leaf_config.target` is missing.

**Fix**: add `leaf_config.target: <column_name>` under `1_data` in your
recipe. For inline panels, the column name must exist in
`custom_panel_inline`.

```yaml
1_data:
  fixed_axes: {target_structure: single_target}
  leaf_config:
    target: INDPRO              # required
    target_horizons: [1, 3, 6]  # required when horizon_set != standard_*
```

---

## `pip install macroforecast` installs a version older than 0.6

You're probably hitting the older `macroforecast 0.1.0` artifact (a
maintainer-uploaded placeholder before the v0.6 series). Upgrade:

```bash
pip install --upgrade macroforecast
python -c "import macroforecast; print(macroforecast.__version__)"
```

Should print `0.8.6` or later. If it still shows `0.1.0`, you have a
stale cached install — `pip install --force-reinstall macroforecast`.

---

## `compare_models([...]).compare(...)` doesn't find the L4 fit node

Symptom:

```python
exp.compare_models(["ridge", "lasso"]).compare(
    "4_forecasting_model.nodes.fit_main.params.alpha", [0.1, 1.0]
)
```

works on **v0.8.6+**. On v0.8.0 / v0.8.5 the L4 fit node had
auto-generated id `fit_1_<family>` (e.g. `fit_1_ridge`), so chained
`.compare()` calls broke.

**Fix**: upgrade to v0.8.6+ where `fit_main` is the stable id, OR use
the auto-generated id explicitly. Verify with:

```python
print(exp.to_recipe_dict()["4_forecasting_model"]["nodes"])
```

---

## `mf.replicate(manifest_path).sink_hashes_match` is False

Bit-exact replication is guaranteed only when:

- **same package version** (manifest carries the macroforecast version
  it was produced with).
- **same dependency versions** (lockfile pinned: pandas / numpy /
  scikit-learn / statsmodels / scipy).
- **same recipe** (the manifest stores the canonical-key-ordered
  recipe; replicate compares `recipe_match=True` first).
- **same raw data / cache** (FRED data revisions break the contract; pin
  the cache via `cache_root=` and re-use across runs).
- **deterministic model families** — torch / xgboost / lightgbm /
  catboost may introduce nondeterminism on GPU / threaded backends. CPU
  + single-thread is the deterministic baseline.

**Debug**:

```python
replication = mf.replicate("out/manifest.json")
print("recipe_match:", replication.recipe_match)
for cell_id, ok in replication.per_cell_match.items():
    print(f"  {cell_id}: {ok}")
```

If `recipe_match=False`, the recipe was edited after the manifest was
written. If `per_cell_match[<cell_id>]=False`, that specific cell's
sink hashes drifted — check the original cell's `sink_hashes` against
the new run's.

---

## Custom preprocessor / model not registered

```
RuntimeError: custom callable 'my_preproc' not registered
```

**Cause**: the decorator (`@mf.custom_preprocessor("my_preproc")`) ran
in a Python process / session different from the one that calls
`Experiment.run()`. macroforecast looks up callables by name in the
in-process registry.

**Fix**: import the module that defines the callable in the same
process before calling `run()`:

```python
import my_methods  # the file with @mf.custom_preprocessor decorators
import macroforecast as mf
exp = mf.Experiment(...).use_preprocessor("my_preproc")
exp.run()
```

For batch / CI runs, ensure the recipe runner imports your method
module before calling `mf.run("recipe.yaml")`. See
[`docs/for_recipe_authors/custom_hooks.md`](for_recipe_authors/custom_hooks.md)
for the full registration contract.

---

## `mixed_frequency_representation` gate doesn't fire

The axis is gated: it is only active when L1 produces a mixed-frequency
panel. Triggers:

- `dataset` includes `fred_sd` (e.g. `fred_md+fred_sd`).
- A custom panel inlines mixed monthly + quarterly columns and L1 has
  detected it.

If your `dataset=fred_md` and you set
`mixed_frequency_representation=...`, the validator emits a soft
warning and the axis is ignored.

---

## A bundled example recipe fails with `ImportError: xgboost`

Some recipes need extras. Check the
[Recipe gallery](for_researchers/recipe_gallery.md) "Extras required"
column. To install everything:

```bash
pip install "macroforecast[all]"
```

Or specific extras:

```bash
pip install "macroforecast[xgboost,lightgbm,catboost,shap]"
```

---

## Sphinx encyclopedia drift CI fails on PR

```
docs/encyclopedia/ is out of sync. Run: python -m macroforecast.scaffold encyclopedia docs/encyclopedia/
```

You edited a `LayerImplementationSpec` axis or an `OPTION_DOCS` entry,
and the encyclopedia source-committed tree no longer matches the
generator output.

**Fix**:

```bash
python -m macroforecast.scaffold encyclopedia docs/encyclopedia/
git add docs/encyclopedia/
git commit -m "docs(encyclopedia): regen after axis/OptionDoc edit"
```

See [`.github/RELEASE_CHECKLIST.md`](https://github.com/NanyeonK/macroforecast/blob/main/.github/RELEASE_CHECKLIST.md).

---

## Want to inspect L1 / L2 / L3 outputs without running L4-L8

```python
from macroforecast.core import materialize_l1, materialize_l2, materialize_l3_minimal
from macroforecast.core.yaml import parse_recipe_yaml

recipe = parse_recipe_yaml(open("recipe.yaml").read())
l1 = materialize_l1(recipe)
print(l1[0].raw_panel.head())                     # raw L1 panel

l2, _ = materialize_l2(recipe, l1[0])
print(l2.cleaned_panel.head())                    # post-McCracken-Ng panel
print(l2.cleaning_log["steps"])                   # which stages applied what

l3 = materialize_l3_minimal(recipe, l1[0], l2)
print(l3[0].X_final.head())                       # L3 features
print(l3[0].y_final.head())                       # L3 target
```

Full guide:
[`docs/for_recipe_authors/partial_layer_execution.md`](for_recipe_authors/partial_layer_execution.md).

---

## Where to ask

- [GitHub issues](https://github.com/NanyeonK/macroforecast/issues) —
  bugs / feature requests
- [GitHub discussions](https://github.com/NanyeonK/macroforecast/discussions) —
  recipe / method questions
- Include in your report: macroforecast `__version__`, Python version,
  the recipe (or a minimal repro), the full traceback, and what you
  expected.
