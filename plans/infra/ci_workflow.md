# Infra: CI Workflow

## 1. What

GitHub Actions workflow design covering four workflows: `ci-core`, `ci-deep`,
`ci-docs`, and `release`. `ci-core` runs on every push and PR and gates the core
test suite across supported Python versions. `ci-deep` runs nightly and on
opt-in PR labels so that GPU/torch-heavy tests do not block every change.
`ci-docs` rebuilds Sphinx on docs changes. `release` publishes to PyPI via
Trusted Publishing (OIDC) on version tags only. Together these four workflows
form the automation surface referenced by every phase's PR checklist.

## 2. Used by phases

- all phases — every phase PR runs `ci-core` as the baseline gate
- phase-00 — delivers the initial `ci-core.yml`
- phase-05a — adds `ci-deep.yml` when deep model support lands
- phase-09 — activates `release.yml` for v0.3.0 tag push

## 3. API spec

`ci-core.yml` (every push and PR, phase-00 deliverable):

```yaml
name: ci-core
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        python: ['3.10', '3.11', '3.12']
        os: [ubuntu-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: ${{ matrix.python }}}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -x -q --ignore=tests/deep
```

`ci-deep.yml` (phase-05a and later; nightly + PR label `needs-deep-test`):

```yaml
name: ci-deep
on:
  schedule: [cron: '0 3 * * *']
  pull_request:
    types: [labeled]
jobs:
  test:
    if: github.event_name == 'schedule' || contains(github.event.pull_request.labels.*.name, 'needs-deep-test')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[dev,deep]"
      - run: pytest tests/ -m deep -x
```

`ci-docs.yml` (docs/** changes on push and PR):

```yaml
name: ci-docs
on:
  push:
    paths: ['docs/**']
  pull_request:
    paths: ['docs/**']
jobs:
  sphinx:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[docs]"
      - run: sphinx-build -W -b html docs docs/_build
```

`release.yml` (tag push `v*.*.*`, activated phase-09):

```yaml
name: release
on:
  push:
    tags: ['v*.*.*']
permissions:
  id-token: write  # OIDC for PyPI Trusted Publishing
jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install build twine
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

## 4. Implementation notes

Gate rules:

- `ci-core` must be green on every PR — branch protection enforces this
- `ci-deep` is required on PRs that touch `macrocast/models/deep/**` or declare
  the `needs-deep-test` label; phase-05a adds a CODEOWNERS-style check
- `ci-docs` is required on PRs that modify `docs/**`
- `release` fires only on `v*.*.*` tag pushes and never on PRs

Pytest markers: `@pytest.mark.deep` marks GPU/torch-heavy tests. The core
workflow uses `--ignore=tests/deep` plus marker exclusion; the deep workflow
selects `-m deep`. The marker is registered in `pyproject.toml` under
`[tool.pytest.ini_options].markers`.

PyPI Trusted Publishing requires a one-time project-side configuration on
PyPI that binds the repository and workflow. No long-lived API token is
stored; `id-token: write` permission is the only secret requirement.

## 5. Test requirements

Workflows are validated by:

- `actionlint` on every PR that changes `.github/workflows/**` (runs inside
  `ci-core` as a lint step added in phase-00)
- A dry-run of `pytest -m deep --collect-only` in `ci-core` that asserts the
  deep marker is registered but no deep tests execute
- A local `make test-core` and `make test-deep` that mirror the CI invocation
  so contributors reproduce gate results before pushing

## 6. Owner / ADR references

Owned by phase-00 (`ci-core.yml` initial delivery), phase-05a (`ci-deep.yml`
real activation with genuine deep tests), and phase-09 (`release.yml`
activation for PyPI publication). No dedicated ADR — this doc is the
reference for branch protection configuration.
