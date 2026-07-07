# Env Drift Diagnosis: PR #456 Docgen Gate

Date: 2026-07-07
Worktree: `/home/nanyeon99/project/mf-docgen`
Branch: `fix/docgen-421`

## CI Install Lines Checked

- `ci-core`: `.github/workflows/ci-core.yml` installs `pip install -e ".[ci]"`.
- `ci-docs` Sphinx job: `.github/workflows/ci-docs.yml` installs `pip install -e ".[docs]"`.

## Environments Built

- Minimal CI venv: `.venv-ci-min`, built with `python3 -m venv .venv-ci-min` and `.venv-ci-min/bin/pip install -e '.[ci]'`.
- Docs-only venv: `.venv-docs-min`, built with `python3 -m venv .venv-docs-min` and `.venv-docs-min/bin/pip install -e '.[docs]'`.
- Full venv: `~/project/macroforecast/.venv/bin/python`.

Optional dependency availability:

| Package | `.[ci]` venv | `.[docs]` venv | Full venv |
| --- | --- | --- | --- |
| `xgboost` | no | no | yes |
| `lightgbm` | no | no | yes |
| `catboost` | no | no | yes |
| `arch` | no | no | yes |
| `torch` | no | no | yes |
| `shap` | no | no | yes |
| `anatomy` | no | no | yes |
| `matplotlib` | no | no | yes |

## Reproduction Result Before Fix

Commands run before any code edits in this fix pass:

```bash
.venv-ci-min/bin/python -m tools.docgen /tmp/mf-docgen-ci-min.rmHATJ
diff -ru docs/reference /tmp/mf-docgen-ci-min.rmHATJ

.venv-docs-min/bin/python -m tools.docgen /tmp/mf-docgen-docs-min.OEAZb3
diff -ru docs/reference /tmp/mf-docgen-docs-min.OEAZb3

~/project/macroforecast/.venv/bin/python -m tools.docgen /tmp/mf-docgen-full.I0Yiat
diff -ru /tmp/mf-docgen-ci-min.rmHATJ /tmp/mf-docgen-full.I0Yiat
diff -ru /tmp/mf-docgen-docs-min.OEAZb3 /tmp/mf-docgen-full.I0Yiat
```

All four `diff -ru` commands exited 0 on current `HEAD` (`413dcb6d`).
`python -m tools.docgen --check docs/reference` also passed in both minimal
CI and docs-only venvs before code edits.

## Differing Pages

No pages differed in the current worktree reproduction.

The CI-red symptom from the fix brief, including the example
`SearchError | macroforecast.model_selection | class` row, did not reproduce
from current `HEAD`: `macroforecast.model_selection.__all__` exports
`SearchError` unconditionally in both the full and minimal environments.

## Root-Cause Class Still Present

Even though current pages were byte-identical in this reproduction, the renderer
still derived each module page from the live runtime `module.__all__`. That is
the environment-sensitive mechanism described in the brief: if an optional
dependency path adds or removes a symbol from a module's runtime export list,
the generated page and reference-verification counts can drift between a full
developer venv and CI's smaller extras.

Hardening target: use the source-declared public surface for module pages when a
literal `__all__` is present, and ignore runtime-only export mutations. Runtime
objects are still used for signatures and docstrings after the source-declared
name has been selected.
