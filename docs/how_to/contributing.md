# For contributors

You want to modify the package source, add a layer / op / model, fix a bug, or
ship a release. Start here.

## If you want to ...

| Goal | Page |
|---|---|
| Understand the bit-exact replicate contract every change must preserve | [Reproducibility policy](reproducibility_policy.md) |
| Decide where to start a function or feature review | [Feature review router](#feature-review-router) |
| Read the canonical layer design before touching `core/` | [Architecture index](../explanation/architecture/index.md) |
| Read the per-layer contract before touching one layer | [Architecture: layer pages](../explanation/architecture/index.md) |
| Run the test suite locally | see [`CLAUDE.md`](../../CLAUDE.md) at the repo root |
| Cut a release | [`.github/RELEASE_CHECKLIST.md`](https://github.com/NanyeonK/macroforecast/blob/main/.github/RELEASE_CHECKLIST.md) |
| Write an axis page that follows the docs convention | [`docs/CONVENTIONS.md`](../CONVENTIONS.md) |

## Code map

```
macroforecast/
  __init__.py             # lazy-export top-level surface
  api/                    # run / replicate / forecast / Experiment public API
  meta/                   # L0 study setup, seed, reproducibility, compute policy
  data/                   # L1 data sources, FRED adapters, vintages, manifests
  preprocessing/          # callable pandas preprocessing helpers
  features/               # L3 feature graph ops, transforms, selectors
  models/                 # L4 model families, paper helpers, tuning engines
  evaluation/             # L5 metrics and evaluation ops
  stat_tests/             # L6 forecast-comparison tests
  interpretation/         # L7 importance, attribution, IRF, interpretation methods
  output/                 # L8 artifact/provenance/export ops
  diagnostics/            # L1.5/L2.5/L3.5/L4.5 diagnostic schemas
  core/                   # runtime, pipeline.py execution, registry, cache, manifest, validator
  layers/                 # compatibility aliases only; do not add new implementation here
  functions/              # backward-compatible shim to api/functions
  recipes/                # recipe-orchestration public namespace
  _vendor/                # vendored third-party implementations
tools/docgen/             # RecipeBuilder + OptionDoc + encyclopedia generation
tests/                    # test suite (counts vary by extras; see CI badges)
examples/recipes/         # YAML recipe examples per layer
```

## Tests

```bash
python -m pytest tests/ -x -q -m "not deep"
```

Expected: tests pass / some skipped (counts vary by extras and Python version; see CI badges). The `[deep]` extra (and
its tests) require torch.

## Typecheck

```bash
python -m pip install -e ".[typecheck]"
mypy
```

The mypy gate is a baseline, not a strict package-wide cleanup. It runs over the
package entrypoint in `pyproject.toml` and explicitly marks legacy dynamic
runtime/layer modules as typed-debt via `tool.mypy.overrides`. New typed modules
should pass without adding to that override list.

## Release

The release workflow (`release.yml`) publishes to PyPI on tag push. The
checklist at `.github/RELEASE_CHECKLIST.md` enumerates the version-bump,
CHANGELOG, README, docs/install.md, and tag-vs-CI steps. The `ci-core` workflow
has stale-tag checks that will reject a release if `README.md` or
`docs/install.md` still advertise an older `@vN.N.N` tag.

```{toctree}
:hidden:
:maxdepth: 1

reproducibility_policy
```
