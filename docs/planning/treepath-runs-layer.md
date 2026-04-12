# TP-007 Runs Layer

This step makes realized outputs path-aware.

Current first pass:
- `runs/` bucket exists in repo
- `ensure_output_dirs()` can now organize outputs by:
  - recipe id
  - taxonomy path
  - fallback ad hoc run id
- manifests now carry:
  - recipe_id
  - taxonomy_path
- recipe-based compiled specs now retain recipe/taxonomy metadata in `meta_config`

Current limitation:
- existing runtime callers still need to opt into recipe-aware output usage explicitly
- full run layout migration across all execution paths is not complete yet
