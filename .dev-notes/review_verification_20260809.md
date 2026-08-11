# External review 2026-08-09 — verification and plan

**14 checkable claims verified. 14 correct. 0 wrong.** Every one was checked against the
repository, not against memory.

## A. My own artifacts — all wrong as reported, all now fixed

| # | Claim | Verified how | Fix |
|---|---|---|---|
| 8.1 | `.[dev]` is not "everything" — `all` omits `deep` | `tomllib` read of `pyproject.toml`: `all = [...macro_random_forest,interpretation]`, no `deep` | #517 |
| 8.2 | `pytest-timeout` is in `[dependency-groups].dev`, not the pip extra | line 112 under `[dependency-groups]`; `dev` extra = `['pytest>=8.0','macroforecast[all]']` | #517 |
| 8.3 | "four marker-gated groups excluded" — `reference` runs by default | CI runs `-m 'not slow and not rparity and not mc'` | #517 |
| 8.4 | `ci-deep` does not run the gated groups | installs `.[ci,deep]`, runs `tests/models tests/forecasting` | #517 |
| 8.5 | Squash is convention, not enforcement | `gh api`: `squash=true merge=true rebase=true` | #517 |
| 9.1 | `tomllib` on a matrix including 3.10 | matrix `['3.10','3.11','3.12']`, `import tomllib` at line 48 | #506 |
| 9.2 | `pip check \|\| true` defeats the job | line 77 | #506 |
| 9.3 | `--no-deps pytest` in a job that never runs pytest | line 76 | #506 |
| 10.1 | `Split` is `mf.window.Split`, a `GenericAlias` | **my own inventory said so**; prose contradicted it | #518 |
| 10.2 | `BoogingRegressor` is Booging, not a typo | `model_ensemble/core.py:841` cites arXiv:2008.07063 and the R `Booging(...)` | #518 |
| 11 | Architecture guard skips **all** relative imports | `if node.level: continue` at `test_import_boundaries.py:105` | open |
| — | 6 repro branches, not 5 | `git ls-remote`: I omitted `repro/mrf-2024` | this file |

## B. Repository state — confirmed

| Claim | Verified |
|---|---|
| `main` trust notes point at scripts not on `main` | Medeiros note line 43 says `python3 scripts/replication/medeiros_2021_pipeline/run_block.py`; that path does not exist on `main` |
| Hardcoded absolute paths block a fresh clone | `medeiros .../prepare_data.py:16` → `/home/nanyeon99/...zip`; `gcls_2022_pipeline/data.py:35` → `~/second_brain/...`; `zww .../build_stage1_target.py:532` → `~/second_brain/_pipeline_state/...` |
| `handoff_20260808.md` is not on `main` | committed to `repro/gcls-2022` only |
| `cd_publishing_design.md` does not exist | the file is `.dev-notes/trusted_publishing.md`; **I cited a wrong filename in an earlier report** |
| GCLS verdict overclaimed | said "verified faithful and correct" while the same doc said 4/5 targets superseded |

## C. Done this pass

- **#517** — five factual corrections, pushed
- **#506** — three workflow defects fixed (tomli fallback + conditional install, hard `pip check`, drop `--no-deps pytest`), pushed
- **#518** (new) — `Split` and `BoogingRegressor` corrections, plus a written caveat that the inventory resolves owners by `dir()` order and must be rebuilt from `_LAZY_EXPORTS` / `__all__` / `object.__module__` before any ownership-dependent decision
- **`repro/gcls-2022`** — verdict downgraded to **PROVISIONAL** with a four-item completion table, pushed

## D. Not done — ordered

1. **Architecture guard blind spot** (§11). Resolve `node.level` + `node.module` against the
   current module path instead of skipping. Add the three cases the reviewer names.
2. **`docs/replication/README.md` index** (§2). Add branch, immutable SHA, package SHA,
   data manifest, smoke command per paper. Branch names alone are mutable and insufficient.
3. **Fresh-clone executability** (§4). Replace the three hardcoded paths with `--archive` /
   `--data-dir` CLI options; add acquisition manifests (source URL, SHA-256, member name,
   expected shape).
4. **Replication manifest schema** (§16). `macroforecast-replication-v1` per directory.
5. **Trust-rating decomposition** (§6). Split the single P1 label into
   `headline_replication` / `table_cell_parity` / `protocol_fidelity` / `unresolved_gap`.
   Medeiros is currently `STRONG` on the index while its own note puts UCSV outside
   tolerance at h=3, 6, 12.
6. **Hounyo–Li wording** (§7). The current phrasing asserts the authors' *intent* and the
   *effect on published numbers*. Narrow it to what was observed in the published code,
   plus archive SHA, file hash, exact lines, and a minimal counterfactual reproducer.
   This is a claim about someone else's published finding and carries a higher evidence
   bar than package documentation.
7. **`repro/mrf-2024`** (§15). Determine whether it is a replication artifact or a package
   source lane; rename or archive.
8. **`lags` documentation** (§12) — explicit `feature_steps` do not inherit the shortcut.
9. **`SECURITY.md` scope** (§13) — narrow "pickle loading … out of scope" to expected vs
   unexpected unpickling.
10. **Known architecture exceptions** (§11) — extract `panel_fingerprint` and
    `collect_provenance` into lower shared modules, then drive the exception list to zero.

## E. Decisions, with the review's positions

| Item | Review's recommendation | Blocker? |
|---|---|---|
| #506 floors | raise pandas **and** scipy **and** `statsmodels>=0.14.2`, citing the NumPy-2 compatibility notes | **yes** |
| #516 search seed | adopt deterministic default; better still, derive from the pipeline seed by (target, horizon, arm, stage), with explicit `None` meaning intentionally nondeterministic | **yes** |
| #511 Trusted Publishing | create the PyPI publisher and GitHub `pypi` environment first | yes, before release |
| #507 aliases | publish canonical paths only; do not mass-remove; fix the inventory first | no |
| #450 / #453 | adopt as designed; implement position before tcodes | no |

The statsmodels floor is an addition to what I had measured — I checked pandas and scipy
against `numpy==2.0` but not statsmodels. The review's `>=0.14.2` is consistent with the
0.14.2 release notes.
