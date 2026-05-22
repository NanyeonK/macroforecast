# Recipe Gallery

The runnable gallery lives in `examples/recipes/`. Gallery recipes must use current top-level layer keys, not the legacy `path: 1_data_task` wrapper.

Use:

```python
from pathlib import Path
import macroforecast as mf

result = mf.run(Path("examples/recipes/l4_minimal_ridge.yaml"))
```

or:

```bash
macroforecast run examples/recipes/l4_minimal_ridge.yaml -o out/l4_minimal_ridge
```

Recommended first recipes:

| Recipe | Use it for |
| --- | --- |
| `l4_minimal_ridge.yaml` | Small end-to-end custom-panel recipe. |
| `l6_standard.yaml` | AR-p benchmark versus ridge with DM-HLN and MCS blocks. |
| `l6_full_replication.yaml` | Broader L6 statistical-test coverage. |
| `goulet_coulombe_2021_replication.yaml` | Paper-style replication scaffold. |

Do not describe optional extras unless the recipe actually imports that family. Keep this page generated or smoke-test-backed.
