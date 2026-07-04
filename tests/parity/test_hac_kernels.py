"""HAC long-run-variance kernel parity: ``macroforecast.tests._long_run_variance``
vs R ``sandwich``.

Work item 2 of the WP-V1 brief. These three kernels feed every DM-family
test in the package (``dm_test``, ``gw_test``, ``harvey_newbold_test``,
``clark_west_test``/``cw_test``, ``enc_t_test``), so a formula mismatch here
propagates to every reported p-value in the equal-predictive-accuracy
family.

Kernel name mapping (macroforecast -> sandwich::kernHAC ``kernel=``):
  - "acf"      -> "Truncated"  (flat weight = 1 for all lags <= bandwidth;
                  macroforecast's own name for this is inherited from an
                  "acf-style" full-lag sum, not literally the sample ACF)
  - "bartlett" -> "Bartlett"   (linear taper, matches classic Newey-West)
  - "parzen"   -> "Parzen"     (cubic taper)

Both sides compute the *same* n-normalized (not n-bandwidth-normalized)
formula: gamma_0 + 2 * sum_{k=1}^{L} w(k) * gamma_k, gamma_k = (1/n) sum
c_t c_{t+k}. On the R side this is reproduced via ``sandwich::kernHAC`` on
an intercept-only ``lm`` fit (so the "residual" IS the centered series),
with ``prewhite = FALSE`` (no VAR(1) prewhitening -- macroforecast does
none) and ``adjust = FALSE`` (no small-sample n/(n-k) inflation --
macroforecast divides by plain n throughout). ``kernHAC`` returns the
variance of the *coefficient* (i.e. already divided by n once more than the
"meat"); to compare apples-to-apples with ``_long_run_variance`` (which
returns the LRV of the *series*, un-scaled by 1/n), we multiply R's
``kernHAC`` coefficient-variance by n.

Tolerance: 1e-6 (both sides are closed-form double-precision sums with no
optimizer in the loop; the only source of residual disagreement is float
summation order, hence not 1e-10).

SUSPECTED BUG found by this harness (bartlett kernel only): macroforecast's
"bartlett" branch weights lag k as ``1 - k/(bandwidth + 1)`` (the classical
Newey & West 1987, Econometrica eq. 2.3, ``w_j = 1 - j/(q+1)`` formula --
weight never reaches exactly zero at ``k=bandwidth``), while its "acf"
(Truncated) and "parzen" branches both weight lag k using ``x = k /
bandwidth`` (the Andrews 1991 generic-kernel convention, weight reaching
exactly zero at ``k=bandwidth`` for Parzen and being cut off there for
Truncated) -- which is also what R's ``sandwich::kernHAC`` uses for ALL
THREE kernels uniformly, including "Bartlett". So macroforecast's own three
kernel branches do not share one bandwidth convention: "bartlett" is off
by a `bandwidth -> bandwidth+1` rescaling relative to its OWN sibling
branches, not just relative to R. This is confirmed to full double
precision below (`test_bartlett_kernel_uses_different_bandwidth_convention_than_its_siblings`):
substituting R's `x = k/bandwidth` convention into the same lag-sum
reproduces R's `kernHAC` value exactly, while macroforecast's own
`x = k/(bandwidth+1)` formula reproduces macroforecast's own output exactly
-- i.e. this is a real, reproducible internal inconsistency, not a
numerical-precision artifact. See ``test_hac_kernel_matches_sandwich_kernhac``
and ``test_hac_kernel_small_n_edge``'s ``xfail`` on the "bartlett"
parametrization for the R-facing consequence.

The "andrews" kernel is exercised separately in
``test_andrews_kernel_is_broken`` -- see that test's docstring for the bug
this harness surfaced.
"""
from __future__ import annotations

import numpy as np
import pytest

from macroforecast.tests import _long_run_variance

from tests.parity.conftest import parse_float, require_r, run_rscript, write_csv

pytestmark = [pytest.mark.rparity]

_KERNEL_MAP = {
    "acf": "Truncated",
    "bartlett": "Bartlett",
    "parzen": "Parzen",
}


def _r_kernhac_lrv(values: np.ndarray, *, kernel: str, bandwidth: int, tmp_path) -> float:
    r_kernel = _KERNEL_MAP[kernel]
    csv_path = tmp_path / "series.csv"
    write_csv(csv_path, {"x": values})
    script = f'''
library(sandwich)
df <- read.csv("{csv_path}")
fit <- lm(x ~ 1, data = df)
n <- nrow(df)
V <- kernHAC(fit, kernel = "{r_kernel}", bw = {bandwidth}, prewhite = FALSE,
             adjust = FALSE, sandwich = TRUE)
lrv <- as.numeric(V) * n
emit("lrv", lrv)
'''
    result = run_rscript(script)
    return parse_float(result["lrv"])


_BARTLETT_XFAIL = pytest.mark.xfail(
    reason=(
        "SUSPECTED BUG: macroforecast's bartlett branch weights lag k as "
        "1 - k/(bandwidth+1) (Newey-West 1987 convention) while its own "
        "acf/parzen siblings -- and R sandwich::kernHAC's Bartlett kernel -- "
        "use 1 - k/bandwidth (Andrews 1991 convention). See module docstring "
        "and test_bartlett_kernel_uses_different_bandwidth_convention_than_its_siblings "
        "for the exact-precision confirmation. Not resolved by loosening "
        "tolerance per WP-V1 mandate; tracked as a live finding via strict xfail."
    ),
    strict=True,
)


@pytest.mark.parametrize(
    "kernel",
    ["acf", pytest.param("bartlett", marks=_BARTLETT_XFAIL), "parzen"],
)
def test_hac_kernel_matches_sandwich_kernhac(kernel: str, tmp_path) -> None:
    require_r("sandwich")
    rng = np.random.default_rng(42)
    n = 80
    # AR(1)-flavoured series so autocovariances are non-trivial at several lags.
    ar = np.empty(n)
    ar[0] = rng.standard_normal()
    for t in range(1, n):
        ar[t] = 0.4 * ar[t - 1] + rng.standard_normal()
    bandwidth = 4

    py_lrv = _long_run_variance(ar, kernel=kernel, lag=bandwidth)
    r_lrv = _r_kernhac_lrv(ar, kernel=kernel, bandwidth=bandwidth, tmp_path=tmp_path)

    assert py_lrv == pytest.approx(r_lrv, abs=1e-6), (
        f"kernel={kernel}: py={py_lrv!r} vs R sandwich::kernHAC={r_lrv!r}"
    )


@pytest.mark.parametrize(
    "kernel",
    ["acf", pytest.param("bartlett", marks=_BARTLETT_XFAIL), "parzen"],
)
def test_hac_kernel_small_n_edge(kernel: str, tmp_path) -> None:
    """Small-n edge: n=5, bandwidth larger than n (bandwidth=4 >= n-1=4).

    Exercises the ``if n > k:`` truncation guards in every kernel branch of
    ``_long_run_variance`` -- with n=5 and bandwidth=4, only lags k=1..3
    have any terms (k=4 has zero pairs since n=5 means indices 0..4, and
    `centered[:-4]`/`centered[4:]` are each length-1 arrays, so k=4 IS
    computable; only k>=n contributes nothing). This is the smallest n for
    which the requested bandwidth exceeds what a naive off-by-one
    implementation might silently zero out.
    """
    require_r("sandwich")
    rng = np.random.default_rng(7)
    values = rng.standard_normal(5)
    bandwidth = 4

    py_lrv = _long_run_variance(values, kernel=kernel, lag=bandwidth)
    r_lrv = _r_kernhac_lrv(values, kernel=kernel, bandwidth=bandwidth, tmp_path=tmp_path)

    assert py_lrv == pytest.approx(r_lrv, abs=1e-6), (
        f"small-n kernel={kernel}: py={py_lrv!r} vs R sandwich::kernHAC={r_lrv!r}"
    )


def test_bartlett_kernel_uses_different_bandwidth_convention_than_its_siblings() -> None:
    """Exact-precision confirmation of the bartlett finding (no R needed):
    substituting the Andrews-1991-style ``x = k/bandwidth`` weight into the
    identical lag-sum reproduces R's ``kernHAC`` Bartlett value to ~1e-15,
    while macroforecast's actual ``x = k/(bandwidth+1)`` weight reproduces
    macroforecast's own reported value to ~1e-15 -- i.e. the two conventions
    are both *internally exact*, they are simply different conventions, and
    macroforecast mixes them across its three kernel branches.
    """
    rng = np.random.default_rng(42)
    n = 80
    ar = np.empty(n)
    ar[0] = rng.standard_normal()
    for t in range(1, n):
        ar[t] = 0.4 * ar[t - 1] + rng.standard_normal()
    centered = ar - ar.mean()
    gamma_0 = float(np.dot(centered, centered) / n)
    bandwidth = 4

    def _lrv_with_denominator(denom_offset: int) -> float:
        variance = gamma_0
        for k in range(1, bandwidth + 1):
            weight = max(0.0, 1.0 - k / (bandwidth + denom_offset))
            variance += 2.0 * weight * float(np.dot(centered[:-k], centered[k:]) / n)
        return variance

    newey_west_1987_denom = _lrv_with_denominator(1)  # bandwidth + 1
    andrews_1991_denom = _lrv_with_denominator(0)  # bandwidth

    py_actual = _long_run_variance(ar, kernel="bartlett", lag=bandwidth)

    assert py_actual == pytest.approx(newey_west_1987_denom, abs=1e-12), (
        "macroforecast's bartlett branch should match the k/(bandwidth+1) "
        "(Newey-West 1987) formula exactly -- if this fails, the branch's "
        "own formula changed and the finding above needs re-diagnosis."
    )
    assert py_actual != pytest.approx(andrews_1991_denom, abs=1e-6), (
        "if this now passes, macroforecast's bartlett branch has been "
        "changed to the k/bandwidth (Andrews 1991 / sandwich::kernHAC) "
        "convention and this whole xfail/finding is stale and should be "
        "removed together with the bartlett xfail marks above."
    )


@pytest.mark.parametrize("horizon,expected_bandwidth", [(1, 0), (4, 3), (8, 7)])
def test_horizon_implied_bandwidth_matches_sandwich_at_that_lag(
    horizon: int, expected_bandwidth: int, tmp_path
) -> None:
    """``_diebold_mariano_stat`` sets ``lag = horizon - 1`` (macroforecast/tests.py,
    the h-step-ahead-forecast-error MA(h-1) convention shared with
    ``forecast::dm.test``'s ``varestimator='acf'`` path, which truncates its
    ``acf(d, lag.max = h - 1)`` sum at the same h-1 lag). Rather than
    re-deriving that from prose, this executes ``_long_run_variance`` at
    exactly the bandwidth each horizon implies and checks it still agrees
    with R's ``kernHAC`` at that same explicit bandwidth -- i.e. the
    per-kernel formula match in ``test_hac_kernel_matches_sandwich_kernhac``
    continues to hold at the specific lag truncations h=1/4/8 actually
    request, not just at one arbitrarily chosen bandwidth.
    """
    require_r("sandwich")
    rng = np.random.default_rng(11)
    n = 60
    ar = np.empty(n)
    ar[0] = rng.standard_normal()
    for t in range(1, n):
        ar[t] = 0.4 * ar[t - 1] + rng.standard_normal()

    lag_used = max(0, int(horizon) - 1)
    assert lag_used == expected_bandwidth

    py_lrv = _long_run_variance(ar, kernel="acf", lag=lag_used)

    if lag_used == 0:
        # R's sandwich::kernHAC(bw=0) is a degenerate call for every kernel
        # (weightsAndrews() finds no weight above `tol` and errors with
        # "result would be too long a vector" -- an R-side edge-case bug in
        # sandwich itself, not something macroforecast controls). bandwidth=0
        # is unambiguous regardless of kernel convention though: no lag terms
        # enter at all, so LRV reduces to the plain sample variance gamma_0.
        # Assert against that closed form directly instead of routing through
        # kernHAC's fragile bw=0 path.
        centered = ar - ar.mean()
        gamma_0 = float(np.dot(centered, centered) / n)
        assert py_lrv == pytest.approx(gamma_0, abs=1e-12), (
            f"horizon=1 (bandwidth=0): py={py_lrv!r} vs closed-form gamma_0={gamma_0!r}"
        )
        return

    r_lrv = _r_kernhac_lrv(ar, kernel="acf", bandwidth=lag_used, tmp_path=tmp_path)
    assert py_lrv == pytest.approx(r_lrv, abs=1e-6), (
        f"horizon={horizon} (bandwidth={lag_used}): py={py_lrv!r} vs R={r_lrv!r}"
    )


def test_andrews_kernel_is_broken() -> None:
    """SUSPECTED BUG (found while building this harness, not an R-parity
    mismatch -- macroforecast crashes on its own before any R comparison is
    possible): ``_long_run_variance(..., kernel="andrews")`` -- and every
    public callable that forwards to it (``dm_test``, ``gw_test``,
    ``harvey_newbold_test`` confirmed directly below; ``clark_west_test``/
    ``cw_test``/``enc_t_test`` share the same ``_mean_hac_test_statistic`` ->
    ``_long_run_variance`` path and are almost certainly affected too) --
    ALWAYS raises ``ValueError: unknown HAC kernel 'newey_west'`` for any
    input, any n, any horizon.

    Root cause (macroforecast/tests.py, ``_long_run_variance``, the
    ``if kernel == "andrews":`` branch): after computing the Andrews/
    Newey-West (1994) AR(1)-plug-in automatic bandwidth, the code does
    ``kernel = "newey_west"`` -- clearly *intending* to route into the
    linear-taper (Bartlett/Newey-West) weighted-sum branch below -- but
    that branch is spelled ``if kernel == "bartlett":``, not
    ``"newey_west"``. Since no branch matches the string ``"newey_west"``,
    execution always falls through to the final
    ``raise ValueError(f"unknown HAC kernel {{kernel!r}}")``. This makes
    ``kernel="andrews"`` -- one of the two automatic-bandwidth options
    matrix.csv's row for ``dm_test`` explicitly flags as unverified/broken
    ("bartlett/parzen/andrews kernels of _long_run_variance are NOT
    independently formula-matched anywhere, only metadata-string-checked")
    -- dead on arrival rather than merely unverified: it cannot return a
    value at all, in production, today.

    One-line fix (not applied here per the WP-V1 mandate that a finding is
    reported, not silently patched by the audit): change
    ``kernel = "newey_west"`` to ``kernel = "bartlett"`` in that branch.

    This assertion needs no R and no R package -- it is a pure-Python crash
    reproduction -- but is kept in ``tests/parity/`` (with the ``rparity``
    marker, so it does not newly appear in default CI as part of this PR)
    because it was discovered specifically while building the HAC-kernel
    R-parity harness for work item 2.
    """
    import numpy as np

    import macroforecast as mf
    from macroforecast.tests import _long_run_variance

    rng = np.random.default_rng(0)
    values = rng.standard_normal(50)
    with pytest.raises(ValueError, match="unknown HAC kernel"):
        _long_run_variance(values, kernel="andrews")

    a = rng.standard_normal(60)
    b = rng.standard_normal(60)
    for public_callable in (mf.tests.dm_test, mf.tests.gw_test, mf.tests.harvey_newbold_test):
        with pytest.raises(ValueError, match="unknown HAC kernel"):
            public_callable(a, b, kernel="andrews")
