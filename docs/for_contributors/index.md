# For contributors

You want to modify the package source, add a layer / op / model, fix a bug, or
ship a release. Start here.

## If you want to ...

| Goal | Page |
|---|---|
| Understand the bit-exact replicate contract every change must preserve | [Reproducibility policy](reproducibility_policy.md) |
| Read the canonical 12-layer design before touching `core/` | [Architecture index](../architecture/index.md) |
| Read the per-layer contract before touching one layer | [Architecture: layer pages](../architecture/index.md) |
| Run the test suite locally | see [`CLAUDE.md`](../../CLAUDE.md) at the repo root |
| Cut a release | [`.github/RELEASE_CHECKLIST.md`](https://github.com/NanyeonK/macroforecast/blob/main/.github/RELEASE_CHECKLIST.md) |
| Write an axis page that follows the docs convention | [`docs/CONVENTIONS.md`](../CONVENTIONS.md) |

## Code map

```
macroforecast/
  __init__.py             # lazy-export top-level surface
  api.py                  # macroforecast.run / macroforecast.replicate
  core/
    execution.py          # cell loop + replicate_recipe (bit-exact)
    runtime.py            # per-layer materialize_l{1..8}_minimal helpers
    figures.py            # matplotlib backend + US choropleth
    layers/               # l0..l8 + l1_5/l2_5/l3_5/l4_5 schema definitions
    ops/                  # universal/l3/l4/l5/l6/l7/l8/diagnostic op registry
    cache.py, dag.py, sweep.py, manifest.py, validator.py, yaml.py, types.py
  raw/                    # FRED-MD/QD/SD adapters + vintage manager
  preprocessing/          # contract helpers (legacy support)
  custom.py               # register_model / register_preprocessor / ...
  scaffold/               # python -m macroforecast scaffold wizard
  defaults.py             # default profile dict template
  tuning/                 # HP search engines (optional, integrated via L4)
plans/design/             # 4-part canonical design (source of truth)
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
