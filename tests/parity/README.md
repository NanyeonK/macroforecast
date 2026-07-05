# R-parity tests (`tests/parity/`)

Independent, external-reference verification of the package's most
inference-critical hand-rolled statistics/models against R. Produced as
WP-V1 of the macroforecast verification program (see
`.dev-notes/anchor_coverage/summary.md` + `matrix.csv` for the inventory
that motivated this work, and `.dev-notes/anchor_coverage/v1_parity_results.md`
for the per-item outcome table).

## Running

```
pytest tests/parity/ -m rparity
```

The whole directory collects normally under plain `pytest tests/` (it is
*not* excluded from collection), but every individual test calls
`require_r(...)` first and self-skips if `Rscript` or a needed R package is
missing -- so on a runner without R (e.g. default CI), everything in this
directory reports as `skipped`, not `error`. `ci-core.yml` additionally
excludes `-m 'not rparity'` explicitly as a belt-and-suspenders measure.

## Bridge: subprocess-Rscript, not rpy2

The deleted C59 suite this restores (`git show 7050f5d2:tests/core/test_r_crossref_c59.py`)
used `rpy2`. On this host (R 4.3.3 at `/usr/bin/R`), the only `rpy2` wheel
`pip`/`uv pip` resolves (3.6.7) fails to import against this R build in
*both* API and ABI mode:

```
ffi.error: symbol 'R_getVar' not found in library '/usr/lib/R/lib/libR.so'
```

This persists even with `R_HOME`/`LD_LIBRARY_PATH` set explicitly to the R
install directory -- it is an rpy2-cffi/R-ABI version mismatch (rpy2 3.6.x
assumes internal R C-API symbols not present/exported in this R 4.3.3
build), not a missing-library problem. Pinning an older rpy2 build was out
of scope for this WP (flagged as follow-up infra work, not a macroforecast
bug).

Instead, every parity test in this directory goes through
`conftest.run_rscript(script_body)`: it writes a temp R script wrapped in a
small preamble (an `emit(name, numeric_vector)` / `emit_str(name,
character_vector)` helper that appends `key=value` lines to a results
file), invokes `Rscript` synchronously with a bounded timeout
(`R_TIMEOUT_SECONDS = 180`), and parses the results file back into Python.
Any non-zero R exit raises `RuntimeError` with full stdout/stderr so R-side
errors surface loudly instead of as silent empty results. This has no ABI
dependency, and the exact R call is plain text inline in each test --
arguably easier to audit than an rpy2 call graph.

## R packages needed

Installed into a user library (`~/R/x86_64-pc-linux-gnu-library/4.3`,
via `R_LIBS_USER`) because the default `/usr/local/lib/R/site-library` is
root-owned/read-only on this host:

```
Rscript -e 'install.packages(c("forecast","rugarch","midasr","MCS","sandwich",
  "lmtest","scoringRules","earth","Boruta"), repos="https://cloud.r-project.org",
  lib=Sys.getenv("R_LIBS_USER"))'
```

`require_r("pkg")` checks `requireNamespace(pkg, quietly=TRUE)` per-test, so
partial installs degrade to per-test skips rather than an all-or-nothing
gate.

## Mission-critical convention: a MISMATCH is a finding

Per the WP-V1 brief: a parity mismatch is never resolved by loosening a
tolerance. Every test's tolerance is fixed at time of authoring from either
(a) the deleted C59 test-spec's own stated thresholds (unchanged), or (b)
a documented closed-form/optimizer-precision justification. When a test
disagrees with R, the failure is diagnosed (bug-in-macroforecast vs.
documented-convention-difference vs. R-package quirk) and either:

- filed as `pytest.mark.xfail(reason="...", strict=True)` pointing at the
  finding in `v1_parity_results.md`, if the mismatch reflects a genuine
  behavioral difference worth tracking as a live regression trip-wire, or
- left as a hard `MISMATCH` with a comment above the assertion and an entry
  in `v1_parity_results.md`, if turning it into a permanently-skipped xfail
  would hide it from view.

Do not "fix" a mismatch by loosening the assertion without an explicit,
comment-documented reason grounded in what the *reference formula* actually
requires.

## Fixtures

Deterministic: seeded `numpy` RNGs with fixed seeds committed in the test
bodies. Every cross-language comparison materializes its fixture to a CSV
in a pytest `tmp_path` at runtime and has R read that file, so both
languages consume the literal same numbers off disk instead of
regenerating a DGP twice and risking language-level RNG drift. (No
fixtures are committed as static CSVs yet; if one is ever needed, put it
under `tests/parity/data/`.)
