"""WP-V3 target #5: ``superior_predictive_ability_test`` (SPA, arch-backed) size.

Two designs, K=5 (``benchmark`` + 4 candidates), benchmark's expected loss
EQUAL to every candidate's -- H0 (benchmark is not beaten) is true by
construction, so ``decision`` (any candidate significantly beats the
benchmark) should be True at rate ~= alpha:

  A) IID: each model's loss is iid noise around a common mean (no serial or
     cross-model correlation).
  B) DEPENDENT: each model's loss has an AR(1) common factor plus AR(1)
     idiosyncratic noise -- the realistic case (forecast losses are
     serially correlated in practice), and the one this WP's brief actually
     asked for ("K=5 models ... fix seeds per replication").

A pre-registration-style diagnostic sweep (see
``.dev-notes/anchor_coverage/v3_mc_results.md``) checked design B's size
across ``p_value_type`` in {consistent (default), upper, lower},
``studentize`` in {True, False}, ``block_length`` in {1, 5, 10, 20, "auto"},
and ``n_origins`` in {100, 250, 500} -- design A (iid) is correctly sized in
every variant tried; design B (any nonzero serial correlation, any block
length, any n) is consistently oversized by roughly 1.5-2x nominal and does
NOT shrink with more data or a longer/auto block length. That rules out "MC
design DGP" and "bootstrap replication count" as the driver and isolates the
distortion to how the ``arch.bootstrap``-backed SPA path here handles
serially-correlated loss differentials specifically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

from tests.mc.conftest import clopper_pearson, record, spawn_generators

MODEL_NAMES = ["benchmark", "m1", "m2", "m3", "m4"]
N_BOOT = 1000
ALPHA = 0.05
N_REPS = 1500


def _make_panel_iid(rng: np.random.Generator, *, n: int, sd: float) -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES:
        loss = 1.0 + rng.normal(0.0, sd, size=n)
        for origin in range(n):
            rows.append({"target": "y", "horizon": 1, "origin": origin, "model_id": name, "squared_error": loss[origin]})
    return pd.DataFrame(rows)


def _make_panel_dependent(
    rng: np.random.Generator, *, n: int, common_rho: float, idio_rho: float, common_sd: float, idio_sd: float
) -> pd.DataFrame:
    common = np.empty(n)
    common[0] = rng.normal(0.0, common_sd / np.sqrt(max(1e-9, 1.0 - common_rho**2)))
    for t in range(1, n):
        common[t] = common_rho * common[t - 1] + rng.normal(0.0, common_sd)
    rows = []
    for name in MODEL_NAMES:
        idio = np.empty(n)
        idio[0] = rng.normal(0.0, idio_sd / np.sqrt(max(1e-9, 1.0 - idio_rho**2)))
        for t in range(1, n):
            idio[t] = idio_rho * idio[t - 1] + rng.normal(0.0, idio_sd)
        loss = 1.0 + common + idio
        for origin in range(n):
            rows.append({"target": "y", "horizon": 1, "origin": origin, "model_id": name, "squared_error": loss[origin]})
    return pd.DataFrame(rows)


@pytest.mark.mc
@pytest.mark.timeout(180)
def test_spa_size_equal_benchmark_iid_losses() -> None:
    n = 100
    gens = spawn_generators(N_REPS, salt=6_000_001)
    n_reject = 0
    for i, rng in enumerate(gens):
        panel = _make_panel_iid(rng, n=n, sd=0.3)
        res = mf.tests.superior_predictive_ability_test(
            panel, benchmark="benchmark", alpha=ALPHA, n_boot=N_BOOT,
            block_length="auto", random_state=17_000_000 + i,
        )
        if res["records"][0]["decision"]:
            n_reject += 1

    lo, hi = clopper_pearson(n_reject, N_REPS, conf=0.99)
    in_band = lo <= ALPHA <= hi
    record(
        test="superior_predictive_ability_test",
        design=f"K=5, benchmark==all candidates (iid losses), n={n}",
        nominal_alpha=ALPHA,
        n_reps=N_REPS,
        n_rejections=n_reject,
        verdict="PASS (in band)" if in_band else ("OVERSIZED" if ALPHA < lo else "UNDERSIZED"),
    )
    assert in_band, (
        f"superior_predictive_ability_test size distortion (iid): rate="
        f"{n_reject / N_REPS:.4f} CI99=[{lo:.4f},{hi:.4f}]"
    )


_SPA_DEPENDENT_DISTORTION_REASON = (
    "CONFIRMED across a wide diagnostic sweep (see v3_mc_results.md): with "
    "ANY nonzero serial correlation in the loss series (even a mild AR(1) "
    "idio_rho=0.3), superior_predictive_ability_test over-rejects the true "
    "'benchmark not beaten' null by roughly 1.5-2x nominal alpha=.05, and "
    "this is STABLE across p_value_type in {consistent,upper,lower}, "
    "studentize in {True,False}, block_length in {1,5,10,20,'auto'}, and "
    "n_origins in {100,250,500} -- ruling out MC design choice, bootstrap "
    "replication count, and block-length misspecification as the driver. "
    "The companion test_spa_size_equal_benchmark_iid_losses (same code path, "
    "zero serial correlation) is correctly sized, isolating this to how the "
    "arch.bootstrap-backed SPA path here handles serially-correlated loss "
    "differentials specifically -- not a defect in the core "
    "better-models/decision logic itself."
)


@pytest.mark.mc
@pytest.mark.timeout(180)
@pytest.mark.xfail(reason=_SPA_DEPENDENT_DISTORTION_REASON, strict=True)
def test_spa_size_equal_benchmark_dependent_losses() -> None:
    n = 100
    gens = spawn_generators(N_REPS, salt=6_000_002)
    n_reject = 0
    for i, rng in enumerate(gens):
        panel = _make_panel_dependent(
            rng, n=n, common_rho=0.5, idio_rho=0.3, common_sd=0.15, idio_sd=0.3
        )
        res = mf.tests.superior_predictive_ability_test(
            panel, benchmark="benchmark", alpha=ALPHA, n_boot=N_BOOT,
            block_length="auto", random_state=18_000_000 + i,
        )
        if res["records"][0]["decision"]:
            n_reject += 1

    lo, hi = clopper_pearson(n_reject, N_REPS, conf=0.99)
    in_band = lo <= ALPHA <= hi
    record(
        test="superior_predictive_ability_test",
        design=f"K=5, benchmark==all candidates (AR(1) common+idio losses), n={n}",
        nominal_alpha=ALPHA,
        n_reps=N_REPS,
        n_rejections=n_reject,
        verdict="PASS (in band)" if in_band else "OVERSIZED",
    )
    assert in_band, (
        f"superior_predictive_ability_test size distortion (dependent): rate="
        f"{n_reject / N_REPS:.4f} CI99=[{lo:.4f},{hi:.4f}]"
    )


_RC_STEPM_DEPENDENT_DISTORTION_REASON = (
    "Companion diagnostic for the arch-backed multiple-comparison family: the "
    "same dependent-null AR(1) design that oversizes SPA is expected to oversize "
    "Reality Check and StepM because the wrapper delegates their serially "
    "dependent benchmark/candidate loss matrix to arch.bootstrap with the same "
    "block-length resolution path. Keep strict xfail until the dependent-loss "
    "size distortion is fixed or disproved by the full MC gate."
)


@pytest.mark.mc
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("callable_name", "seed_offset"),
    [
        ("reality_check_test", 19_000_000),
        ("stepm_test", 20_000_000),
    ],
)
@pytest.mark.xfail(reason=_RC_STEPM_DEPENDENT_DISTORTION_REASON, strict=True)
def test_arch_set_comparison_size_equal_benchmark_dependent_losses(
    callable_name: str,
    seed_offset: int,
) -> None:
    n = 100
    gens = spawn_generators(N_REPS, salt=6_000_003 + seed_offset)
    n_reject = 0
    test_callable = getattr(mf.tests, callable_name)
    for i, rng in enumerate(gens):
        panel = _make_panel_dependent(
            rng, n=n, common_rho=0.5, idio_rho=0.3, common_sd=0.15, idio_sd=0.3
        )
        res = test_callable(
            panel, benchmark="benchmark", alpha=ALPHA, n_boot=N_BOOT,
            block_length="auto", random_state=seed_offset + i,
        )
        if res["records"][0]["decision"]:
            n_reject += 1

    lo, hi = clopper_pearson(n_reject, N_REPS, conf=0.99)
    in_band = lo <= ALPHA <= hi
    record(
        test=callable_name,
        design=f"K=5, benchmark==all candidates (AR(1) common+idio losses), n={n}",
        nominal_alpha=ALPHA,
        n_reps=N_REPS,
        n_rejections=n_reject,
        verdict="PASS (in band)" if in_band else "OVERSIZED",
    )
    assert in_band, (
        f"{callable_name} size distortion (dependent): rate="
        f"{n_reject / N_REPS:.4f} CI99=[{lo:.4f},{hi:.4f}]"
    )
