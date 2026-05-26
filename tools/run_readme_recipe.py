#!/usr/bin/env python3
"""Extract the minimal recipe from README.md and run it via mf.run().

Used by CI to prevent README drift: any change to README.md or macroforecast/
triggers this check. Exits non-zero if the recipe errors or produces no cells.
"""
import os
import re
import sys
import tempfile
import warnings

warnings.simplefilter("ignore")

import macroforecast as mf

README_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")

with open(README_PATH) as f:
    readme = f.read()

match = re.search(r"A minimal recipe:\n\n```yaml\n(.*?)\n```", readme, re.DOTALL)
if not match:
    print("ERROR: Could not find 'A minimal recipe:' yaml block in README.md", file=sys.stderr)
    sys.exit(1)

recipe_yaml = match.group(1)

with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write(recipe_yaml)
    fname = f.name

try:
    result = mf.run(fname)
    n_cells = len(result.cells)
    if n_cells == 0:
        print("ERROR: README minimal recipe ran but produced 0 cells", file=sys.stderr)
        sys.exit(1)
    print(f"README minimal recipe: OK (cells={n_cells})")
    sys.exit(0)
except Exception as exc:
    print(f"README minimal recipe: FAILED — {exc}", file=sys.stderr)
    sys.exit(1)
finally:
    os.unlink(fname)
