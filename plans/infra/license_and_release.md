# Infra: License and Release

## 1. What

MIT license for the codebase plus a PyPI-only release policy executed via
GitHub Actions Trusted Publishing (OIDC). Data sources have per-source terms
documented in the top-of-file docstring of every adapter module, and the
`macrocast` package itself never redistributes third-party datasets. Release
versions flow through normal PyPI minor numbers during pre-1.0, with `v1.0.0`
reserved as the stable marker. This infra doc is the single place that lists
the canonical source table and the release command set.

## 2. Used by phases

- all phases — every new data adapter file references the source table here
- phase-09 — activates the release workflow and performs the first PyPI
  publication of `macrocast`

## 3. API spec

Code license — MIT, declared in `pyproject.toml`:

```toml
[project]
name = "macrocast"
license = {text = "MIT"}
```

Adapter docstring template (every file under `macrocast/raw/datasets/`):

```python
# macrocast/raw/datasets/fred_md.py
"""FRED-MD dataset adapter.

Source: St. Louis Fed (https://research.stlouisfed.org/econ/mccracken/fred-databases/)
Terms: Public domain; citation requested (McCracken & Ng 2016)

This adapter downloads and caches locally but does not redistribute.
Users are responsible for compliance with source terms.
"""
```

Source table:

- FRED-MD / FRED-QD / FRED-SD — public domain (phase-00, already operational)
- BEA / BLS — public domain (phase-10)
- OECD — subscription for bulk access, free for research (phase-10)
- ECB SDW — free with registration (phase-10)
- IMF IFS — free with registration (phase-10)
- SPF — public, Philadelphia Fed (phase-10)
- WRDS — institutional subscription (phase-11)
- BIS — varying terms per dataset (phase-11)

PyPI release command set:

```bash
python -m build                 # sdist + wheel
# upload is done by the release.yml workflow via Trusted Publishing
```

Installation (README snippet):

```bash
pip install macrocast
pip install macrocast[deep]
pip install macrocast[all]
pip install macrocast[dev]
```

## 4. Implementation notes

Release channel is **PyPI only** — no conda-forge recipe in this repo, no
private registry. Trusted Publishing is configured once on the PyPI side by
binding the `macrocast` project to the `macrocast` GitHub repo and the
`release.yml` workflow. After that binding, tag pushes produce authenticated
uploads with no stored secrets.

Pre-release versions flow as normal PyPI minor numbers: v0.2.x, v0.3.x, ...,
v0.9.x. `v1.0.0` is reserved as the stable marker and will only be tagged
once the public API is frozen and documented. No `aN/bN/rcN` suffixes during
normal minor bumps; a release candidate would use `v1.0.0rc1` style when we
approach v1.0.

Data license compliance — the package never ships raw data. Adapters
download on demand into the user's cache directory (see cache discipline),
and the adapter docstring is the user-facing notice of source terms. The
`pyproject.toml` `package-data` field excludes `*.csv`, `*.parquet`, and
similar extensions as a belt-and-braces measure.

## 5. Test requirements

Release hygiene tests (`tests/test_release_hygiene.py`):

- `pyproject.toml` declares `license = {text = "MIT"}`
- Every file under `macrocast/raw/datasets/` has a top-of-module docstring
  containing `"Source:"` and `"Terms:"` lines
- Built wheel contains no `*.csv` or `*.parquet` files (check on the built
  artifact in `ci-core`)
- `macrocast.__version__` matches the git tag on release builds (workflow
  step inside `release.yml`)

Smoke test the published artifact from a clean venv after release:

```bash
pip install macrocast==<version>
python -c "import macrocast; print(macrocast.__version__)"
```

## 6. Owner / ADR references

Owned by phase-09 (release activation). Aligns with Resolved Decisions #5
(MIT license) and #7 (PyPI-only distribution). Every phase's adapter
additions must reference the source table in section 3 of this doc and
include the adapter docstring template verbatim.
