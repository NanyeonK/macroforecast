# Release Checklist

Run before tagging a new release. The version-consistency CI step in
``ci-core.yml`` will fail if any of the version updates are skipped.

## Pre-tag

- [ ] ``pyproject.toml`` ``version`` bumped
- [ ] ``macrocast/__init__.py`` ``__version__`` bumped
- [ ] ``README.md`` install commands point at the new ``@vX.Y.Z`` tag
- [ ] ``README.md`` citation line uses the new ``vX.Y.Z`` (badges are
      now real CI badges; no version badge to bump)
- [ ] ``docs/install.md`` install commands and verify-comment use the
      new tag
- [ ] ``CHANGELOG.md`` entry added under ``## [X.Y.Z] -- YYYY-MM-DD``
- [ ] Local pytest pass:
      ```bash
      python -m pytest tests/ -x -q -m "not deep"
      ```

## Tag and push

- [ ] ``git push origin main`` (CI must be green for the commit)
- [ ] ``git tag -a vX.Y.Z -m "..."``
- [ ] ``git push origin vX.Y.Z``

## After tag

- [ ] ``ci-core`` and ``ci-docs`` green for the tag commit
- [ ] ``release.yml`` workflow ran (PyPI publish; will be a no-op
      until the PyPI ``macrocast`` namespace is resolved — see
      ``release.yml`` NOTE)
- [ ] GitHub Pages (``gh-pages``) docs deploy reflects the new tag
      (``docs/install.md`` and ``README.md`` show ``vX.Y.Z`` in the
      published HTML, not just in the source)
- [ ] Verify clean install on a throwaway venv:
      ```bash
      python -m venv /tmp/macrocast-vX.Y.Z
      /tmp/macrocast-vX.Y.Z/bin/pip install \
          "git+https://github.com/NanyeonK/macrocast.git@vX.Y.Z"
      /tmp/macrocast-vX.Y.Z/bin/python -c "import macrocast; print(macrocast.__version__)"
      /tmp/macrocast-vX.Y.Z/bin/macrocast --help
      ```

## PyPI namespace status

The PyPI ``macrocast`` namespace is held by an unrelated 2017 package
(``macrocast 0.0.2`` by Amir Sani). Until ownership / rename is
resolved, the supported install path is GitHub-tag (above), not
``pip install macrocast``. README and docs/install.md carry warnings.

If/when the namespace is freed up:

- [ ] Add ``PYPI_API_TOKEN`` to repo secrets (Settings → Secrets and
      variables → Actions → New repository secret)
- [ ] Re-run ``release.yml`` for the most recent tag
      (``gh run rerun <id> --repo NanyeonK/macrocast``)
