"""MC anchors for ``multi_horizon_spa_test``.

Anchor tier: MC + hand-oracle only, no external parity. On this host,
``Rscript -e 'install.packages("MultiHorizonSPA")'`` against the configured
CRAN mirror (https://cloud.r-project.org) reported that the package is not
available for R 4.3.3, and ``available.packages()`` did not list it.
"""
from __future__ import annotations

import numpy as np
import pytest

import macroforecast as mf

from tests.mc.conftest import clopper_pearson, record, spawn_generators

ALPHA = 0.10
N_OBS = 120
N_HORIZONS = 3
N_BOOT = 149
N_REPS = 200


def _ar1_panel(rng: np.random.Generator, *, rho: float = 0.4) -> np.ndarray:
    panel = np.empty((N_OBS, N_HORIZONS), dtype=float)
    panel[0] = rng.normal(size=N_HORIZONS) / np.sqrt(1.0 - rho**2)
    for idx in range(1, N_OBS):
        panel[idx] = rho * panel[idx - 1] + rng.normal(size=N_HORIZONS)
    return panel


def _mc_rejection_rate(
    *,
    statistic: str,
    design: str,
    salt: int,
) -> int:
    n_reject = 0
    for rep, rng in enumerate(spawn_generators(N_REPS, salt=salt)):
        if design == "iid_null":
            diff = rng.normal(size=(N_OBS, N_HORIZONS))
        elif design == "ar_null":
            diff = _ar1_panel(rng)
        elif design == "one_horizon":
            diff = rng.normal(size=(N_OBS, N_HORIZONS))
            diff[:, 0] += 0.55
        elif design == "all_horizons":
            diff = rng.normal(size=(N_OBS, N_HORIZONS)) + 0.30
        else:  # pragma: no cover - test bug guard
            raise ValueError(design)
        result = mf.tests.multi_horizon_spa_test(
            diff,
            statistic=statistic,
            alpha=ALPHA,
            n_boot=N_BOOT,
            block_length=3,
            random_state=81_000_000 + salt + rep,
        )
        n_reject += int(bool(result.decision))
    return n_reject


@pytest.mark.mc
@pytest.mark.timeout(180)
@pytest.mark.parametrize("statistic", ["uspa", "aspa"])
def test_multi_horizon_spa_size_equal_losses_iid(statistic: str) -> None:
    n_reject = _mc_rejection_rate(
        statistic=statistic,
        design="iid_null",
        salt=7_100_000 + (0 if statistic == "uspa" else 1_000),
    )
    lo, hi = clopper_pearson(n_reject, N_REPS, conf=0.99)
    in_band = lo <= ALPHA <= hi
    record(
        test=f"multi_horizon_spa_test[{statistic}]",
        design=f"equal predictive ability across horizons, iid losses, n={N_OBS}, H={N_HORIZONS}",
        nominal_alpha=ALPHA,
        n_reps=N_REPS,
        n_rejections=n_reject,
        verdict="PASS (in band)" if in_band else ("OVERSIZED" if ALPHA < lo else "UNDERSIZED"),
    )
    assert in_band


@pytest.mark.mc
@pytest.mark.timeout(180)
@pytest.mark.parametrize("statistic", ["uspa", "aspa"])
def test_multi_horizon_spa_size_equal_losses_autocorrelated(statistic: str) -> None:
    n_reject = _mc_rejection_rate(
        statistic=statistic,
        design="ar_null",
        salt=7_200_000 + (0 if statistic == "uspa" else 1_000),
    )
    lo, hi = clopper_pearson(n_reject, N_REPS, conf=0.99)
    in_band = lo <= ALPHA <= hi
    record(
        test=f"multi_horizon_spa_test[{statistic}]",
        design=f"equal predictive ability across horizons, AR(1) loss differences, n={N_OBS}, H={N_HORIZONS}",
        nominal_alpha=ALPHA,
        n_reps=N_REPS,
        n_rejections=n_reject,
        verdict="PASS (in band)" if in_band else ("OVERSIZED" if ALPHA < lo else "UNDERSIZED"),
    )
    assert in_band


@pytest.mark.mc
@pytest.mark.timeout(180)
def test_multi_horizon_spa_power_one_horizon_average_spa() -> None:
    n_reject = _mc_rejection_rate(statistic="aspa", design="one_horizon", salt=7_300_000)
    rate = n_reject / N_REPS
    record(
        test="multi_horizon_spa_test[aspa]",
        design=f"one-horizon positive loss differential, n={N_OBS}, H={N_HORIZONS}",
        nominal_alpha=ALPHA,
        n_reps=N_REPS,
        n_rejections=n_reject,
        verdict="PASS (power > 0.60)" if rate > 0.60 else "LOW POWER",
    )
    assert rate > 0.60


@pytest.mark.mc
@pytest.mark.timeout(180)
@pytest.mark.parametrize("statistic", ["uspa", "aspa"])
def test_multi_horizon_spa_power_all_horizons(statistic: str) -> None:
    n_reject = _mc_rejection_rate(
        statistic=statistic,
        design="all_horizons",
        salt=7_400_000 + (0 if statistic == "uspa" else 1_000),
    )
    rate = n_reject / N_REPS
    record(
        test=f"multi_horizon_spa_test[{statistic}]",
        design=f"all-horizon positive loss differential, n={N_OBS}, H={N_HORIZONS}",
        nominal_alpha=ALPHA,
        n_reps=N_REPS,
        n_rejections=n_reject,
        verdict="PASS (power > 0.60)" if rate > 0.60 else "LOW POWER",
    )
    assert rate > 0.60
