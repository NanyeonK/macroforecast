# Sphinx Warning Audit — 2026-05-27

**Command**: `python3 -m sphinx -W -b html docs docs/_build/html_strict`  
**Date**: 2026-05-27  
**Branch**: `deep-audit/pr7-sphinx-howto`  
**Sphinx version**: see `python3 -m sphinx --version`

---

## Result

**Warnings before fix**: 0 (Sphinx -W build was already clean)  
**Warnings after fix**: 0  
**Exit code**: 0 (`build succeeded`)  

The Sphinx `-W` flag (treat warnings as errors) passed without any fixes  
beyond the import correction in Sub-task A.  
The CI gate at `.github/workflows/ci-docs.yml:58-60` already enforces  
`sphinx-build -W -b html docs docs/_build/html` on every PR.

---

## CI gate status

**Gate already present**: YES  
**Location**: `.github/workflows/ci-docs.yml`, step "Build Sphinx HTML (fail on warning)"  

```yaml
- name: Build Sphinx HTML (fail on warning)
  run: |
    sphinx-build -W -b html docs docs/_build/html
```

No new CI step was added because the gate already existed.

---

## Makefile target added

A `docs-strict` target was added to `docs/Makefile` so developers can run  
the same check locally before opening a PR:

```makefile
docs-strict:
    sphinx-build -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html_strict" $(SPHINXOPTS) $(O)
```

Usage: `cd docs && make docs-strict`

---

## Build output (final lines)

```
writing output... [100%] user_guide
generating indices... genindex done
highlighting module code...
writing additional pages... search done
dumping search index in English (code: en)... done
dumping object inventory... done
build succeeded.

The HTML pages are in docs/_build/html_strict.
```
