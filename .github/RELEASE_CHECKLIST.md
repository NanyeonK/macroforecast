# Release Checklist

Run before tagging a new release. The version-consistency CI step in
``ci-core.yml`` will fail if any of the version updates are skipped.

## Pre-tag

- [ ] ``pyproject.toml`` ``version`` bumped
- [ ] ``macroforecast/__init__.py`` ``__version__`` bumped
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
- [ ] If you edited any ``OptionDoc`` content or any
      ``LayerImplementationSpec`` (axes / options / sub-layers),
      regenerate the encyclopedia and stage the diff:
      ```bash
      python -m macroforecast.scaffold encyclopedia docs/encyclopedia/
      git add docs/encyclopedia/
      ```
      The ``ci-docs`` Encyclopedia drift check fails the build if this
      step is skipped.

## Tag and push

- [ ] ``git push origin main`` (CI must be green for the commit)
- [ ] ``git tag -a vX.Y.Z -m "..."``
- [ ] ``git push origin vX.Y.Z``

## After tag

- [ ] ``ci-core`` and ``ci-docs`` green for the tag commit
- [ ] ``release.yml`` workflow ran successfully (publishes to
      https://pypi.org/project/macroforecast/; first release under the
      ``macroforecast`` PyPI namespace was v0.6.0)
- [ ] GitHub Pages (``gh-pages``) docs deploy reflects the new tag
      (``docs/install.md`` and ``README.md`` show ``vX.Y.Z`` in the
      published HTML, not just in the source)
- [ ] PyPI ``latest`` shows the new version:
      ```bash
      python -m pip index versions macroforecast
      ```
- [ ] Verify clean install on a throwaway venv:
      ```bash
      python -m venv /tmp/macroforecast-vX.Y.Z
      /tmp/macroforecast-vX.Y.Z/bin/pip install macroforecast==X.Y.Z
      /tmp/macroforecast-vX.Y.Z/bin/python -c "import macroforecast; print(macroforecast.__version__)"
      /tmp/macroforecast-vX.Y.Z/bin/macroforecast --help
      ```

## PyPI status

The project maintainer owns the ``macroforecast`` PyPI namespace
(see https://pypi.org/project/macroforecast/). v0.6.0 was the first
release published under this namespace; v0.5.x tags pre-date the
rename and were never published to PyPI.

The ``PYPI_API_TOKEN`` secret must remain registered in repo
Settings → Secrets and variables → Actions, scoped to the
``macroforecast`` project, for ``release.yml`` to publish on tag push.
