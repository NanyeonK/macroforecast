"""``macroforecast.tests.dm_test`` vs R ``forecast::dm.test`` -- statistic AND
p-value, for h=1/h=4, both HLN correction and none, power=1 and power=2.

Work item 3 of the WP-V1 brief: promotes the docstring claim
(``r_reference: "forecast/R/DM2.R::dm.test"``) to an executable check.

Two conventions confirmed by reading ``forecast:::dm.test``'s R body
(``Rscript -e 'library(forecast); print(forecast:::dm.test)'``, this R
install's forecast version) rather than assumed from memory:

1. ``forecast::dm.test`` ALWAYS applies the Harvey-Leybourne-Newbold (1997)
   small-sample correction factor ``k`` -- there is no "raw DM statistic"
   mode in the installed version. So macroforecast's ``correction="none"``
   has no direct R comparator; instead we ask R to also emit its own ``k``
   factor (computed with the exact formula printed in the R source) and
   divide R's corrected statistic back out, then re-derive R's own p-value
   for that raw statistic via ``pt()`` with the same df=n-1 R itself uses.
   This keeps the comparison "R's numbers, decomposed with R's own
   documented formula" rather than "R's numbers vs. a formula we typed from
   memory."

2. ``forecast::dm.test``'s "bartlett" varestimator weights lag k as
   ``1 - k/h`` (h = the forecast horizon, not h-1) -- which is algebraically
   IDENTICAL to macroforecast's own bartlett formula ``1 - k/(bandwidth+1)``
   once you substitute macroforecast's ``bandwidth = horizon - 1`` (so
   ``bandwidth + 1 = horizon = h``). This is the opposite conclusion from
   ``test_hac_kernels.py``'s bartlett finding: that finding is about calling
   the *private* ``_long_run_variance(kernel="bartlett", lag=<arbitrary>)``
   directly with an arbitrary bandwidth (where macroforecast's convention
   diverges from R sandwich::kernHAC's generic-kernel convention); THIS
   file is about the *public* ``dm_test(kernel="bartlett", horizon=h)`` path
   specifically, where the bandwidth is always exactly ``h - 1`` by
   construction, and that happens to make macroforecast's formula coincide
   with forecast::dm.test's own bartlett convention. Both findings are
   independently verified against R source/output; they are not in tension
   with each other, they are about two different call sites.
"""
from __future__ import annotations

import numpy as np
import pytest

import macroforecast as mf

from tests.parity.conftest import parse_float, require_r, run_rscript, write_csv

pytestmark = [pytest.mark.rparity]


def _r_dm_test(e1: np.ndarray, e2: np.ndarray, *, h: int, power: float, varestimator: str, tmp_path) -> dict:
    e1_path = tmp_path / "e1.csv"
    e2_path = tmp_path / "e2.csv"
    write_csv(e1_path, {"x": e1})
    write_csv(e2_path, {"x": e2})
    script = f'''
library(forecast)
e1 <- read.csv("{e1_path}")$x
e2 <- read.csv("{e2_path}")$x
res <- dm.test(e1, e2, h = {h}, power = {power}, varestimator = "{varestimator}")
n <- length(e1)
h_val <- {h}
k <- sqrt((n + 1 - 2 * h_val + (h_val / n) * (h_val - 1)) / n)
raw_stat <- unname(res$statistic) / k
raw_pvalue <- 2 * pt(-abs(raw_stat), df = n - 1)
emit("statistic_hln", unname(res$statistic))
emit("p_value_hln", res$p.value)
emit("k", k)
emit("statistic_raw", raw_stat)
emit("p_value_raw", raw_pvalue)
'''
    result = run_rscript(script)
    return {
        "statistic_hln": parse_float(result["statistic_hln"]),
        "p_value_hln": parse_float(result["p_value_hln"]),
        "statistic_raw": parse_float(result["statistic_raw"]),
        "p_value_raw": parse_float(result["p_value_raw"]),
    }


def _fixture(n: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal(n) * 1.0
    e2 = rng.standard_normal(n) * 1.2 + 0.1  # slightly worse/biased forecaster
    return e1, e2


@pytest.mark.parametrize(
    "horizon,kernel",
    [(1, "acf"), (4, "acf"), (4, "bartlett")],
)
@pytest.mark.parametrize("power", [1.0, 2.0])
def test_dm_test_statistic_and_pvalue_match_forecast_dm_test(
    horizon: int, kernel: str, power: float, tmp_path
) -> None:
    require_r("forecast")
    e1, e2 = _fixture(n=50, seed=int(horizon * 100 + power))

    r = _r_dm_test(e1, e2, h=horizon, power=power, varestimator=kernel, tmp_path=tmp_path)

    py_hln = mf.tests.dm_test(
        e1, e2, horizon=horizon, correction="hln", kernel=kernel, input_type="error", power=power
    )
    assert py_hln.statistic == pytest.approx(r["statistic_hln"], rel=1e-6, abs=1e-8), (
        f"HLN statistic mismatch h={horizon} kernel={kernel} power={power}: "
        f"py={py_hln.statistic!r} vs R={r['statistic_hln']!r}"
    )
    assert py_hln.p_value == pytest.approx(r["p_value_hln"], rel=1e-6, abs=1e-8), (
        f"HLN p-value mismatch h={horizon} kernel={kernel} power={power}: "
        f"py={py_hln.p_value!r} vs R={r['p_value_hln']!r}"
    )

    py_raw = mf.tests.dm_test(
        e1, e2, horizon=horizon, correction="none", kernel=kernel, input_type="error", power=power
    )
    assert py_raw.statistic == pytest.approx(r["statistic_raw"], rel=1e-6, abs=1e-8), (
        f"raw (correction=none) statistic mismatch h={horizon} kernel={kernel} power={power}: "
        f"py={py_raw.statistic!r} vs R(statistic_hln/k)={r['statistic_raw']!r}"
    )
    assert py_raw.p_value == pytest.approx(r["p_value_raw"], rel=1e-6, abs=1e-8), (
        f"raw (correction=none) p-value mismatch h={horizon} kernel={kernel} power={power}: "
        f"py={py_raw.p_value!r} vs R={r['p_value_raw']!r}"
    )
