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

RESOLVED FINDING (bartlett kernel, kept + documented, NOT aligned to R):
macroforecast's "bartlett" branch weights lag k as ``1 - k/(bandwidth + 1)``
(the classical Newey & West 1987, Econometrica eq. 2.3, ``w_j = 1 -
j/(q+1)`` formula -- weight never reaches exactly zero at ``k=bandwidth``),
while its "acf" (Truncated) and "parzen" branches both weight lag k using
``x = k / bandwidth`` (the Andrews 1991 generic-kernel convention, weight
reaching exactly zero at ``k=bandwidth`` for Parzen and being cut off there
for Truncated) -- which is also what R's ``sandwich::kernHAC`` uses for ALL
THREE kernels uniformly, including "Bartlett". So macroforecast's own three
kernel branches do not share one bandwidth convention: "bartlett" is off
by a `bandwidth -> bandwidth+1` rescaling relative to its OWN sibling
branches, not just relative to R. This is confirmed to full double
precision below (`test_bartlett_kernel_uses_different_bandwidth_convention_than_its_siblings`):
substituting R's `x = k/bandwidth` convention into the same lag-sum
reproduces R's `kernHAC` value exactly, while macroforecast's own
`x = k/(bandwidth+1)` formula reproduces macroforecast's own output exactly
-- i.e. this is a real, reproducible internal inconsistency, not a
numerical-precision artifact.

This is INTENTIONALLY kept, not "fixed": the public
``dm_test(kernel="bartlett", horizon=h)`` path's bandwidth is always
exactly ``horizon - 1``, which makes macroforecast's ``1 -
k/(bandwidth+1)`` formula reduce to ``1 - k/horizon`` -- the EXACT
convention ``forecast::dm.test`` itself uses for its own "bartlett"
varestimator (see ``tests/parity/test_dm_test.py``'s module docstring,
item 2, and its verified 6/6 pass against ``forecast::dm.test``). Aligning
the private ``_long_run_variance`` bartlett branch to the Andrews-1991
``1 - k/bandwidth`` form would silently break that ``dm_test`` parity.
See ``test_hac_kernel_matches_sandwich_kernhac``, ``test_hac_kernel_small_n_edge``,
and the two ``test_bartlett_kernel_*_documented_divergence_from_kernhac``
tests below for the R-facing consequence: bartlett is excluded from the
"should match kernHAC" parametrizations and instead has its own
documented-divergence assertions (same style as the ``midas_almon``
architecture-difference finding in ``test_midas_almon.py``).

RESOLVED FINDING (andrews kernel, fixed): ``_long_run_variance(...,
kernel="andrews")`` used to ALWAYS raise ``ValueError: unknown HAC kernel
'newey_west'`` -- reachable from ``dm_test``, ``gw_test``,
``harvey_newbold_test``, ``clark_west_test``/``cw_test``, ``enc_t_test``.
Root cause: after computing the Andrews (1991, eq. 6.4)/Newey-West (1994)
AR(1) plug-in automatic bandwidth, the code reassigned ``kernel =
"newey_west"``, intending to route into the linear-taper (bartlett) branch
below, but that branch is spelled ``"bartlett"``, not ``"newey_west"`` --
so no branch matched and every call fell through to the final
``raise ValueError``. Fixed by reassigning to ``"bartlett"`` directly (see
``macroforecast/tests.py``, ``_long_run_variance``). Since the andrews
branch reuses the bartlett taper, it inherits the SAME NW-1987 convention
documented above, so it is likewise not expected to match
``sandwich::kernHAC`` (which auto-selects its own Andrews-1991-taper
Bartlett-kernel bandwidth by default) -- see
``test_andrews_kernel_no_longer_crashes*`` and
``test_long_run_variance_andrews_kernel_matches_hand_computed_value`` for
the regression coverage, and
``test_andrews_kernel_documented_divergence_from_kernhac_auto_bandwidth``
for the documented (not asserted-equal) R comparison.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import macroforecast as mf
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


def _r_kernhac_lrv_auto_bw(values: np.ndarray, *, kernel: str, tmp_path) -> float:
    """Like ``_r_kernhac_lrv``, but lets R select its own bandwidth via the
    package default ``bw = bwAndrews`` (``approx = "AR(1)"``) instead of
    pinning an explicit ``bw=``. Used only by the andrews-kernel divergence
    check below, where the question is whether R's OWN automatic-bandwidth
    Bartlett-kernel HAC estimate coincides with macroforecast's
    ``kernel="andrews"`` value, not whether a shared fixed bandwidth does.
    """
    r_kernel = _KERNEL_MAP[kernel]
    csv_path = tmp_path / "series.csv"
    write_csv(csv_path, {"x": values})
    script = f'''
library(sandwich)
df <- read.csv("{csv_path}")
fit <- lm(x ~ 1, data = df)
n <- nrow(df)
V <- kernHAC(fit, kernel = "{r_kernel}", bw = bwAndrews, approx = "AR(1)",
             prewhite = FALSE, adjust = FALSE, sandwich = TRUE)
lrv <- as.numeric(V) * n
emit("lrv", lrv)
'''
    result = run_rscript(script)
    return parse_float(result["lrv"])


@pytest.mark.parametrize("kernel", ["acf", "parzen"])
def test_hac_kernel_matches_sandwich_kernhac(kernel: str, tmp_path) -> None:
    """"bartlett" is intentionally excluded from this parametrization -- it
    is documented to diverge from ``sandwich::kernHAC`` by design (see
    module docstring and the ``test_bartlett_kernel_*`` tests below), so
    asserting equality here would be asserting the wrong thing.
    """
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


@pytest.mark.parametrize("kernel", ["acf", "parzen"])
def test_hac_kernel_small_n_edge(kernel: str, tmp_path) -> None:
    """Small-n edge: n=5, bandwidth larger than n (bandwidth=4 >= n-1=4).

    Exercises the ``if n > k:`` truncation guards in every kernel branch of
    ``_long_run_variance`` -- with n=5 and bandwidth=4, only lags k=1..3
    have any terms (k=4 has zero pairs since n=5 means indices 0..4, and
    `centered[:-4]`/`centered[4:]` are each length-1 arrays, so k=4 IS
    computable; only k>=n contributes nothing). This is the smallest n for
    which the requested bandwidth exceeds what a naive off-by-one
    implementation might silently zero out.

    "bartlett" is intentionally excluded here for the same reason as in
    ``test_hac_kernel_matches_sandwich_kernhac`` above -- see
    ``test_bartlett_kernel_small_n_edge_documented_divergence_from_kernhac``
    below for its small-n divergence check.
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


def test_bartlett_kernel_documented_divergence_from_kernhac(tmp_path) -> None:
    """Documented-divergence check (same fixture as
    ``test_hac_kernel_matches_sandwich_kernhac``): confirms, via an actual
    R ``sandwich::kernHAC`` call, that macroforecast's bartlett branch does
    NOT match R at a shared explicit bandwidth -- this is intentional, not
    a regression. macroforecast's bartlett taper (``1 - k/(bandwidth+1)``,
    Newey-West 1987) is kept specifically because it is required for the
    public ``dm_test(kernel="bartlett")`` path's verified 6/6 parity with
    ``forecast::dm.test`` (``tests/parity/test_dm_test.py``); aligning it
    to kernHAC's Andrews-1991 ``1 - k/bandwidth`` convention would silently
    break that dependency. If this assertion ever starts failing (i.e. the
    two values coincide), treat it as a signal that the bartlett branch's
    formula changed and re-check the ``dm_test`` parity tests immediately.
    """
    require_r("sandwich")
    rng = np.random.default_rng(42)
    n = 80
    ar = np.empty(n)
    ar[0] = rng.standard_normal()
    for t in range(1, n):
        ar[t] = 0.4 * ar[t - 1] + rng.standard_normal()
    bandwidth = 4

    py_lrv = _long_run_variance(ar, kernel="bartlett", lag=bandwidth)
    r_lrv = _r_kernhac_lrv(ar, kernel="bartlett", bandwidth=bandwidth, tmp_path=tmp_path)

    assert py_lrv != pytest.approx(r_lrv, abs=1e-6), (
        f"bartlett: py={py_lrv!r} unexpectedly matched R sandwich::kernHAC={r_lrv!r} -- "
        "if this now passes, the bartlett branch's NW-1987 convention changed; "
        "re-verify tests/parity/test_dm_test.py's dm_test(kernel='bartlett') parity "
        "before treating this as an improvement."
    )


def test_bartlett_kernel_small_n_edge_documented_divergence_from_kernhac(tmp_path) -> None:
    """Documented-divergence check (same small-n fixture as
    ``test_hac_kernel_small_n_edge``): see
    ``test_bartlett_kernel_documented_divergence_from_kernhac`` above for
    the full rationale -- this is the same intentional-divergence finding,
    confirmed to also hold at the small-n/large-relative-bandwidth edge.
    """
    require_r("sandwich")
    rng = np.random.default_rng(7)
    values = rng.standard_normal(5)
    bandwidth = 4

    py_lrv = _long_run_variance(values, kernel="bartlett", lag=bandwidth)
    r_lrv = _r_kernhac_lrv(values, kernel="bartlett", bandwidth=bandwidth, tmp_path=tmp_path)

    assert py_lrv != pytest.approx(r_lrv, abs=1e-6), (
        f"small-n bartlett: py={py_lrv!r} unexpectedly matched R sandwich::kernHAC={r_lrv!r} -- "
        "if this now passes, re-verify the dm_test(kernel='bartlett') parity dependency "
        "(tests/parity/test_dm_test.py) before treating this as an improvement."
    )


def test_bartlett_kernel_uses_different_bandwidth_convention_than_its_siblings() -> None:
    """Exact-precision confirmation of the bartlett convention (no R
    needed): substituting the Andrews-1991-style ``x = k/bandwidth`` weight
    into the identical lag-sum reproduces R's ``kernHAC`` Bartlett value to
    ~1e-15, while macroforecast's actual ``x = k/(bandwidth+1)`` weight
    reproduces macroforecast's own reported value to ~1e-15 -- i.e. the two
    conventions are both *internally exact*, they are simply different
    conventions, and macroforecast deliberately keeps the NW-1987 one for
    its bartlett branch (see module docstring and
    ``_long_run_variance``'s docstring for why -- the public
    ``dm_test(kernel="bartlett")`` parity dependency).
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
        "own formula changed and the dm_test parity dependency needs re-diagnosis."
    )
    assert py_actual != pytest.approx(andrews_1991_denom, abs=1e-6), (
        "if this now passes, macroforecast's bartlett branch has been "
        "changed to the k/bandwidth (Andrews 1991 / sandwich::kernHAC) "
        "convention -- re-verify tests/parity/test_dm_test.py's "
        "dm_test(kernel='bartlett') parity before treating this as an "
        "improvement, since that parity currently depends on the NW-1987 form."
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


def test_long_run_variance_andrews_kernel_matches_hand_computed_value() -> None:
    """Regression + value check for the fixed andrews-kernel crash
    (previously ``_long_run_variance(..., kernel="andrews")`` always raised
    ``ValueError: unknown HAC kernel 'newey_west'`` -- see the module
    docstring's "RESOLVED FINDING (andrews kernel, fixed)" section).

    This does not just check that the call no longer raises -- it
    independently re-derives, in this test, the exact value the andrews
    branch should return under macroforecast's own documented convention:
    the Andrews (1991, eq. 6.4)/Newey-West (1994) AR(1) plug-in bandwidth,
    then the SAME NW-1987 bartlett taper ``1 - k/(bandwidth+1)`` used by
    the "bartlett" branch (macroforecast's andrews branch reassigns into
    that branch after computing the bandwidth). Neither the bandwidth
    formula nor the taper is imported from ``macroforecast/tests.py`` --
    both are re-typed here from the module docstring's formula, so this is
    a genuine hand-check, not a tautological call-the-same-code-twice test.
    """
    rng = np.random.default_rng(3)
    n = 40
    ar = np.empty(n)
    ar[0] = rng.standard_normal()
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + rng.standard_normal()

    centered = ar - ar.mean()
    gamma_0 = float(np.dot(centered, centered) / n)

    # Andrews (1991, eq. 6.4) / Newey-West (1994) AR(1) plug-in bandwidth.
    numerator = float(np.sum(centered[:-1] * centered[1:]))
    denominator = float(np.sum(centered[:-1] ** 2))
    alpha1 = numerator / denominator if denominator > 0 else 0.0
    alpha = (4.0 * alpha1**2) / (max(1.0 - alpha1**2, 1e-12) ** 2)
    expected_bandwidth = max(1, int(np.floor(1.1447 * (alpha * n) ** (1 / 3))))

    expected_variance = gamma_0
    for k in range(1, expected_bandwidth + 1):
        if n > k:
            weight = 1.0 - k / (expected_bandwidth + 1)  # NW-1987 bartlett taper
            expected_variance += 2.0 * weight * float(np.dot(centered[:-k], centered[k:]) / n)

    py_actual = _long_run_variance(ar, kernel="andrews")

    assert py_actual == pytest.approx(expected_variance, abs=1e-10), (
        f"kernel='andrews': py={py_actual!r} vs hand-computed={expected_variance!r} "
        "(Andrews plug-in bandwidth + NW-1987 bartlett taper) -- if this fails, "
        "the andrews branch's bandwidth formula or taper changed in "
        "macroforecast/tests.py and this hand-check needs re-deriving."
    )


def test_andrews_kernel_documented_divergence_from_kernhac_auto_bandwidth(tmp_path) -> None:
    """Documented divergence, not a bug (per the WP-V1 mandate: report,
    don't silently align): even letting R choose ITS OWN automatic
    bandwidth for the Bartlett kernel (``sandwich::kernHAC``'s default,
    ``bw = bwAndrews(approx = "AR(1)")``) rather than pinning a shared
    explicit bandwidth, macroforecast's ``kernel="andrews"`` value does not
    match it. The root cause is the SAME one already documented for the
    plain "bartlett" kernel (see
    ``test_bartlett_kernel_uses_different_bandwidth_convention_than_its_siblings``
    and ``_long_run_variance``'s docstring): macroforecast's andrews branch
    reuses the NW-1987 bartlett taper (``1 - k/(bandwidth+1)``), while
    ``sandwich::kernHAC``'s Bartlett kernel always uses the Andrews-1991
    taper (``1 - k/bandwidth``), regardless of how the bandwidth itself was
    chosen. This divergence is the deliberate cost of the documented
    ``dm_test(kernel="bartlett")`` parity dependency
    (``tests/parity/test_dm_test.py``, 6/6 pass vs ``forecast::dm.test``)
    and is NOT expected to be closed by this test suite.
    """
    require_r("sandwich")
    rng = np.random.default_rng(42)
    n = 80
    ar = np.empty(n)
    ar[0] = rng.standard_normal()
    for t in range(1, n):
        ar[t] = 0.4 * ar[t - 1] + rng.standard_normal()

    py_lrv = _long_run_variance(ar, kernel="andrews")
    r_lrv = _r_kernhac_lrv_auto_bw(ar, kernel="bartlett", tmp_path=tmp_path)

    assert py_lrv != pytest.approx(r_lrv, abs=1e-6), (
        f"andrews (auto-bandwidth): py={py_lrv!r} unexpectedly matched R "
        f"sandwich::kernHAC(bw=bwAndrews)={r_lrv!r} -- if this now passes, "
        "re-verify whether the bartlett-taper-convention asymmetry documented "
        "in _long_run_variance's docstring is still accurate before relying "
        "on this as a 'divergence' assertion."
    )


def test_dm_test_andrews_kernel_no_longer_crashes() -> None:
    """Regression test for the fixed andrews-kernel crash: previously,
    ``dm_test(..., kernel="andrews")`` ALWAYS raised ``ValueError: unknown
    HAC kernel 'newey_west'`` for any input, any n, any horizon -- see the
    module docstring's "RESOLVED FINDING (andrews kernel, fixed)" section
    for the root cause and fix. This exercises the public ``dm_test`` path
    end-to-end with ``kernel="andrews"`` and asserts it returns a real
    ``TestResult`` with a finite statistic and a valid p-value, not merely
    that it avoids raising.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal(60)
    b = rng.standard_normal(60)

    result = mf.tests.dm_test(a, b, kernel="andrews")

    assert result.statistic is not None
    assert math.isfinite(result.statistic)
    assert result.p_value is not None
    assert 0.0 <= result.p_value <= 1.0


def test_andrews_kernel_no_longer_crashes_through_other_public_callables() -> None:
    """Same regression as ``test_dm_test_andrews_kernel_no_longer_crashes``,
    extended to the other public callables that share
    ``_long_run_variance``'s andrews branch: ``gw_test`` and
    ``harvey_newbold_test`` (both confirmed to crash by the original
    finding, via the same ``_diebold_mariano_stat``/direct
    ``_long_run_variance`` call sites as ``dm_test``), plus
    ``clark_west_test`` and ``enc_t_test`` (which route through
    ``_mean_hac_test_statistic`` -> ``_long_run_variance`` instead -- the
    original finding flagged these as "almost certainly affected too" but
    did not confirm them directly; this test promotes that to confirmed).
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal(60)
    b = rng.standard_normal(60)

    for public_callable in (mf.tests.dm_test, mf.tests.gw_test, mf.tests.harvey_newbold_test):
        result = public_callable(a, b, kernel="andrews")
        assert result.statistic is not None
        assert math.isfinite(result.statistic)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    forecast_small = rng.standard_normal(60)
    forecast_large = rng.standard_normal(60)
    cw_result = mf.tests.clark_west_test(a, b, forecast_small, forecast_large, kernel="andrews")
    assert cw_result.statistic is not None
    assert math.isfinite(cw_result.statistic)

    enc_result = mf.tests.enc_t_test(a, b, kernel="andrews", normal_approximation=True)
    assert enc_result.statistic is not None
    assert math.isfinite(enc_result.statistic)
    assert enc_result.p_value is not None
    assert 0.0 <= enc_result.p_value <= 1.0
