# Release Checklist

Run before publishing a new release. ``release.yml`` is manual and validates
that its ``version`` input is final-release form ``vX.Y.Z`` and matches
``pyproject.toml`` before any build or publish step runs.

## Pre-tag

- [ ] ``pyproject.toml`` ``version`` bumped
- [ ] ``macroforecast/__init__.py`` ``__version__`` bumped
- [ ] ``CITATION.cff`` ``version`` and ``date-released`` updated
- [ ] ``CHANGELOG.md`` entry added under ``## [X.Y.Z] -- YYYY-MM-DD``
- [ ] Local pytest pass:
      ```bash
      python -m pytest tests/ -q -m "not slow and not rparity and not mc"
      ```
- [ ] If you changed public docstrings, signatures, ``__all__``, or public
      modules, regenerate and check reference docs:
      ```bash
      python -m tools.docgen docs/reference
      python -m tools.docgen --check docs/reference
      ```
      The ``ci-docs`` reference drift check fails the build if this step is
      skipped.

## Publish

- [ ] ``git push origin main`` (CI must be green for the commit)
- [ ] Run the ``release.yml`` workflow manually with ``version=vX.Y.Z``
- [ ] Confirm ``release.yml`` validation, build, twine check, artifact upload,
      and PyPI publish steps are green
- [ ] ``git tag -a vX.Y.Z -m "..."``
- [ ] ``git push origin vX.Y.Z`` (source tag only; tag push does not publish)

## After publish

- [ ] ``ci-core``, ``ci-docs``, ``ci-typecheck``, and scheduled optional
      workflow expectations are green for the release commit
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
``macroforecast`` project, for the manual ``release.yml`` workflow to publish.
