# How to replicate a study

Take a manifest from a previous run and verify bit-exact replication.

---

## What you need

A `manifest.json` produced by a previous `mf.run(..., output_directory="...")` call.
The manifest records the recipe, data, per-cell artifact hashes, and environment provenance.

```text
my_study_output/
  manifest.json       <- this is what you need
  recipe.json
  summary/
    metrics_all_cells.csv
  cell_001/
    forecasts.csv
```

```{note}
The `manifest.json` usable with `mf.replicate()` is written when you pass
`output_directory=` to `mf.run()`. The L8 artifact manifest is a separate file
and does not support replication.
```

---

## Run the replication

```python
import macroforecast as mf

replication = mf.replicate("my_study_output/manifest.json")

print("Recipe unchanged:", replication.recipe_match)
print("All artifacts match:", replication.sink_hashes_match)

# Per-cell breakdown
for cell_id, matched in replication.per_cell_match.items():
    status = "OK" if matched else "MISMATCH"
    print(f"  {cell_id}: {status}")
```

If `sink_hashes_match` is `True`, the replication produced bit-identical artifacts.
If a cell shows `MISMATCH`, see the debugging section below.

---

## Debugging a mismatch

A mismatch means one of: different package version, different dependency versions,
the recipe was edited after the manifest was written, or different raw data.

**1. Check the package version:**

```python
import macroforecast
print(macroforecast.__version__)
# Must match the version recorded in manifest.json
```

Open `manifest.json` and look for `"macroforecast_version"` in the provenance fields.

**2. Check recipe match:**

If `replication.recipe_match` is `False`, the recipe changed. Do not re-run with
a modified recipe. Restore the original recipe from `recipe.json` in the output
directory:

```python
import json, pathlib

recipe_path = pathlib.Path("my_study_output/recipe.json")
original_recipe = json.loads(recipe_path.read_text())
result = mf.run(original_recipe, output_directory="my_study_output_v2/")
```

**3. Check data:**

If `panel_composition: official_only`, FRED data may have been revised since
the original run. Use `custom_panel_inline` or a fixed CSV to avoid vintage drift.

**4. Pin the lockfile:**

```bash
pip freeze > requirements.txt
```

Record this at run time and restore it before replication.

---

## Sharing a study for external replication

Provide three things:

1. `manifest.json` — the full provenance record.
2. `requirements.txt` — the dependency lockfile at run time.
3. The raw data — if using `custom_source_path` or `custom_panel_inline`, the
   inline data is embedded in the manifest already.

---

See {doc}`reproducibility_policy` for the full bit-exact replication contract.
