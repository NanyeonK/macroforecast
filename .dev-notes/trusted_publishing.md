# Trusted Publishing for the release workflow — proposal (#Phase 3)

**Status: proposal, deliberately NOT merged.** This one cannot be landed by an agent
in the middle of the night, because merging it *before* PyPI is configured breaks
releases, and merging it *after* is a change to how the project's publishing identity
works. Both halves belong to whoever owns the PyPI namespace.

## What is there now

`.github/workflows/release.yml`:

```yaml
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

A long-lived API token in repo secrets. It works, and the workflow around it is
already careful — manual dispatch only, version validated against `pyproject.toml`,
tag push no longer publishes on its own.

## What the token costs

- **It does not expire.** Its blast radius is the whole `macroforecast` project on
  PyPI, for as long as it exists, and revocation is a manual act nobody is prompted to
  perform.
- **It is exfiltratable.** Any workflow change that can read secrets can read this one.
  The repo runs workflows on `pull_request`, so the review of a workflow diff is the
  only thing standing between a contributor and the token — which is a lot to ask of a
  diff review.
- **It says nothing about provenance.** A package published with a token proves only
  that whoever ran the job had the token.

## What Trusted Publishing changes

OIDC: GitHub mints a short-lived token for this repository, this workflow, this
environment, at publish time. PyPI verifies the claim and accepts the upload. Nothing
long-lived exists to leak, and the resulting release carries a verifiable statement of
*which workflow in which repository* built it.

## The change

```yaml
  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi          # gate the step, not just the repo
    permissions:
      id-token: write          # the OIDC claim; nothing else
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v8.0.1
        with:
          name: dist
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

Three lines added, one removed. `permissions: id-token: write` is scoped to the
`publish` job only — the top-level `permissions: contents: read` stays as it is, so no
other job gains anything.

## Order of operations — this is the part that matters

The PyPI side must exist first, or the first release after merge fails:

1. On PyPI → `macroforecast` → *Publishing* → add a trusted publisher:
   - Owner `NanyeonK`, repository `macroforecast`
   - Workflow `release.yml`
   - Environment `pypi`
2. In GitHub → Settings → Environments → create `pypi`. Adding a required reviewer here
   is the cheap win: publishing then needs an explicit human approval click, which the
   token flow never had.
3. Merge this change.
4. Publish one release and confirm it lands.
5. **Then** revoke `PYPI_API_TOKEN` on PyPI and delete the repo secret. Not before — it
   is the rollback.

Step 5 is the one that is easy to skip and is the whole point: an un-revoked token
leaves the old blast radius intact while looking as if it were closed.

## Why an agent should not merge this

Steps 1 and 2 are account actions on PyPI and in repo settings — outside what this
session can or should do. Merging step 3 without them turns the next release into a
failed job. So the workflow diff is prepared and reviewable here, and the merge waits
for the person who can do 1 and 2.
