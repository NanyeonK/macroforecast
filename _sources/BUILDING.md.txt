# Building the Documentation

## Quick start

```bash
# From the repo root:
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
# then open docs/_build/html/index.html
```

Or using the Makefile from the `docs/` directory:

```bash
cd docs
make html
```

## Clean build (required after a version bump)

When the package version changes (`pyproject.toml` or
`macroforecast/__init__.py` bumped), always run a **clean** Sphinx build.
Sphinx's incremental builder only re-renders pages whose source `.md` file
has a newer modification time than its cached `.doctrees` entry. A version
bump does not touch any source file, so unchanged pages will continue to
embed the old version string in the HTML footer and `docsearch:version`
meta tag.

```bash
# From docs/:
make cleanhtml

# Equivalent without make:
rm -rf docs/_build/html
sphinx-build -b html docs docs/_build/html
```

Both CI (GitHub Actions `ci-docs.yml`) and ReadTheDocs always run clean
builds because they start from a fresh checkout. The incremental-build
problem can only occur during local development.

## Dependencies

Install the `[docs]` extras:

```bash
pip install -e ".[docs]"
```

Dependencies include `sphinx`, `pydata-sphinx-theme`, `myst-parser`,
`sphinx-copybutton`, and `sphinx-design`. See `pyproject.toml`
`[project.optional-dependencies]` for the full list.

## CI

The `ci-docs.yml` workflow runs `sphinx-build -W -b html docs docs/_build/html`
on every pull request that touches `docs/` or `macroforecast/`. Warnings
are treated as errors (`-W` flag). The built HTML is uploaded as an artifact
(`docs-html`) and, on pushes to `main`, deployed to the `gh-pages` branch.
