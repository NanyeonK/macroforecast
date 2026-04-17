# Synthetic replication round-trip

The round-trip is Phase 6's Acceptance Gate P0 test — run a recipe, replay it with no overrides, verify the predictions are bit-identical.

Under `seeded_reproducible` mode with a fixed `seed`, the sweep / runner infrastructure is deterministic enough that `predictions.csv` SHA-256s match across two independent `execute_recipe` calls. `execute_replication` exposes this as a single API:

```python
from macrocast.compiler.build import compile_recipe_dict
from macrocast.execution.build import execute_recipe
from macrocast import execute_replication
import yaml, hashlib
from pathlib import Path

RECIPE = yaml.safe_load(open("examples/recipes/replication-synthetic.yaml"))
PROVENANCE = {"compiler": {"reproducibility_spec": {
    "reproducibility_mode": "seeded_reproducible", "seed": 42,
}}}

compiled = compile_recipe_dict(RECIPE).compiled
src = execute_recipe(
    recipe=compiled.recipe_spec,
    preprocess=compiled.preprocess_contract,
    output_root="out/src",
    local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
    provenance_payload=PROVENANCE,
)

rep = execute_replication(
    source_recipe_dict=RECIPE,
    overrides={},
    source_artifact_dir=src.artifact_dir,
    output_root="out/replay",
    local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
    provenance_payload=PROVENANCE,
)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

src_pred = Path(src.artifact_dir) / "predictions.csv"
rep_pred = Path(rep.execution_result.artifact_dir) / "predictions.csv"
assert sha(src_pred) == sha(rep_pred)
assert rep.byte_identical_predictions is True
```

The integration test in `tests/test_replication_end_to_end.py` exercises this flow as part of every CI run.
