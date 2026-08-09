# Contributing

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # pytest + every optional backend EXCEPT deep
pip install -e ".[dev,deep]"     # add torch/captum
pip install -e ".[ci]"           # what CI installs: parquet, markdown, interpretation
pip install pytest-timeout       # see below -- not in the pip extras
```

`dev` resolves to `macroforecast[all]`, and `all` deliberately omits `deep`, so
`.[dev]` alone gives you no `torch`.

`uv.lock` records the repository development environment for reproducible local setup
and dependency audits. CI installs from `pyproject.toml` extras and does **not** enforce
the lockfile, so adding a dependency means editing `pyproject.toml`, not only the lock.

Optional backends live behind extras — `xgboost`, `lightgbm`, `catboost`, `arch`,
`plots`, `macro_random_forest`, `interpretation`, `deep`. Code that needs one must raise
a clear `ImportError` naming the extra when it is absent, so that installing the extra
is an instruction rather than a guess.

## Running the tests

```bash
pytest tests/ -q -m 'not slow and not rparity and not mc'      # what ci-core runs
```

Three marker-gated groups are excluded by default and opt in individually. A
fourth marker, `reference`, exists but runs by default:

| marker | what it is | opt in |
|---|---|---|
| `slow` | realistic-shape integration tests | `pytest -m slow` |
| `rparity` | R cross-reference parity — needs `Rscript` + R packages | `pytest -m rparity` |
| `mc` | Monte Carlo size/power validation | `pytest -m mc` |
| `reference` | anchors for paper/formula behaviour (runs by default) | — |

The full suite is ~30 minutes serially. `pytest-xdist` is **not** a declared dependency;
if you install it yourself, four tests fail under `-n` and pass serially, so deselect
them:

```
tests/models/test_default_cost_budget.py
tests/mc/test_mcs_coverage.py
tests/analysis/test_contribution.py::test_axis_contribution_dk_serial_mc_tracks_empirical_se
tests/models/test_standard_estimators.py::test_ucsv_default_draws_runtime_guard_on_500_obs
```

Unattended runs need a timeout — `-x` does not catch a hang:

```bash
pip install pytest-timeout      # NOT in the pip `dev` extra
pytest tests/ -q --timeout=90 --timeout-method=thread
```

`pytest-timeout` is declared in `[dependency-groups].dev`, which `pip install -e
".[dev]"` does not read. `uv sync` picks it up; pip users install it explicitly.

## What CI checks

| workflow | trigger | what fails it |
|---|---|---|
| `ci-core` | push, PR | the suite on 3.10/3.11/3.12, plus a guard that the package never hard-codes a seed instead of threading the configured RNG |
| `ci-docs` | push, PR | `sphinx -W`, and `python -m tools.docgen --check docs/reference` |
| `ci-readme` | push, PR | the README's minimal recipe actually executes |
| `ci-typecheck` | push, PR | `mypy` across modules |
| `ci-os-smoke` | push, PR | serial == parallel, cache round-trip, and model save/reload on Windows and macOS |
| `ci-deep` | nightly, manual | installs `.[ci,deep]` and runs `tests/models` + `tests/forecasting` |
| `ci-extras` | weekly, manual | optional backends |

**`docs/reference/` is generated but source-committed.** Changing a public signature or
docstring puts it out of date and fails both `ci-docs` and `tests/test_docgen.py`.
Regenerate before pushing:

```bash
python -m tools.docgen docs/reference
```

## Branches and merges

- Branch from `main`; never commit to `main` directly.
- **Project convention is the squash merge**, with the PR number in the subject:
  `fix(window): seed the k-fold shuffle (#515)`. Repository settings currently permit
  merge commits and rebases too, so this is a convention the reviewer upholds rather
  than something the platform enforces.
- Long-running or parallel work belongs in a `git worktree`, one per branch — two agents
  or sessions must not share a branch or a file scope.
- **Do not touch `CHANGELOG.md` or `logs/file_usage_log.md` in a feature branch.** Every
  branch that edits them conflicts with every other one, and each conflict restarts a
  45–60 minute CI cycle.

## Writing tests

The bar is that a test **fails on the unfixed code**. Two examples from this repo where
the obvious version did not:

- A cache-release test asserted that a new target starts with an empty cache. The broken
  code also gave each target its own initially-empty dict, so it passed either way — the
  bug was that the *finished* dict stayed reachable, not that it was shared.
- A custom-interpretation test checked `isinstance` and the column names. Both were true
  of the broken one-row output; it took asserting the row count and that the cells are
  scalars to see the defect.

So: write the test, confirm it fails without the change, then apply the change. For
numerical work, prefer an oracle derived independently of the implementation — a closed
form written out separately, or an axiom (a Shapley row sums to prediction minus base
value) — over a golden value copied from a previous run.

Pin column names to the real schema rather than positional fallbacks. A schema change
should fail loudly, not silently redirect an assertion to a different column.

## Reporting a defect

Include a minimal reproduction, the version or commit, and what you expected. If a
number is wrong, say how wrong and against what — "0.0405 vs a published 0.0389" is
actionable in a way that "the results look off" is not. If you worked around it, say so;
the workaround is information about the defect.
