"""``macroforecast.tests.model_confidence_set`` (and its private engine,
``_mcs_statistic``/``_mcs_loss_differences``) vs R ``MCS::MCSprocedure`` --
work item 4 of the WP-V1 brief.

Two layers of comparison, chosen deliberately after reading the *installed*
``MCS`` package's actual R source (``getAnywhere("GetD")``, ``print(MCS::
MCSprocedure)`` -- both dumped and inspected before writing any assertion
here, not assumed from memory or the R help page):

1. ``test_mcs_loss_differences_matches_r_getd_exactly`` -- the DETERMINISTIC
   piece. ``GetD()``'s pairwise loss differentials (``mD_ij_bar``) and
   per-model average differential vs. the rest of the active set
   (``vD_i_bar = sum(mD_ij_bar[i,]) / (iM-1)``) involve no bootstrap
   randomness at all -- they are plain sample means. macroforecast's
   ``_mcs_loss_differences`` implements the identical formula. This is
   checked to 1e-10 with NO bootstrap involved on either side.

2. ``test_mcs_survivor_set_matches_r_mcsprocedure`` -- the bootstrap-facing
   piece, restricted to what the WP-V1 brief calls out as legitimately
   comparable across independent RNG streams: the final MCS survivor
   ("included") and rejected ("excluded") SETS (unordered), with block
   length fixed identically on both sides and a large B, on a fixture with
   large loss-level separation relative to noise.

A FULL elimination-ORDER comparison (which model gets removed 1st, 2nd,
...) was attempted and deliberately dropped -- not because it "failed to
pass" but because it is not a well-posed cross-implementation invariant,
for two independently-verified reasons (see
``test_mcs_elimination_order_is_not_a_meaningful_cross_language_invariant``
for the executable demonstration):

  (a) R's ``MCSprocedure`` sorts its final table by ascending cumulative
      "MCS p-Value" (``mTab[order(mTab[, "MCS p-Value"]), ]``) AFTER the
      elimination while-loop finishes; it does not report the raw removal
      sequence. Under the kind of large, clean separation needed to make a
      survivor SET robust to bootstrap draws, per-step bootstrap p-values
      routinely collapse to exactly 0 for several eliminated models
      simultaneously (the observed Tmax exceeds literally every bootstrap
      resample). R's ``order()`` breaks such ties by *original column
      position in the input matrix*, not true elimination chronology -- so
      "excluded order" in this regime silently reduces to "input column
      order among the tied models," independent of which model was
      actually removed first. Confirmed directly: feeding a clean 3-model
      fixture (means 0/1/2, noise sd 0.05) to R across 5 different
      bootstrap seeds (1, 2, 3, 42, 99) produced the identical excluded
      order ``m_b, m_c`` every time -- NOT because the order is a
      genuinely stable emergent property of the statistic, but because
      the tie-break (original column order) does not depend on the seed
      at all.
  (b) Even setting tie-breaking aside, the per-model elimination statistic
      is ``t_i = D_i_bar / sqrt(var(D_i_bar))`` where ``D_i_bar`` is
      model i's average loss differential *against the other models
      currently in the active set*, not model i's raw mean loss. For the
      3-model fixture above, hand-deriving from ``GetD``'s own formula:
      ``D_i_bar`` for the MIDDLE model (m_b, mean loss exactly halfway
      between m_a and m_c) is theoretically 0 by symmetry, i.e. the
      middle-ranked model can have the SMALLEST |t_i| of the group despite
      being neither the best nor the worst by raw mean loss -- so "worst
      raw mean loss eliminated first" is not even the right mental model
      for which model ``which.max(ti)`` removes at a given step.

Bootstrap-method alignment: macroforecast's ``bootstrap_method="mcs_fixed_block"``
is commented in ``macroforecast/tests.py`` (``_mcs_bootstrap_indices``) as a
direct port of ``MCS/R/internalFunctions.R::GetIndices`` (block starts drawn
uniformly, fixed contiguous blocks concatenated and truncated to n) -- the
correct value to request for an MCS-package-comparable draw (as opposed to
``stationary_bootstrap``, Politis-Romano geometric restarts, not expected to
agree even in distribution).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.tests import _mcs_loss_differences

from tests.parity.conftest import parse_float_list, parse_str_list, require_r, run_rscript

pytestmark = [pytest.mark.rparity]

_MODEL_BASES = {"m1": 1.15, "m2": 0.95, "m3": 0.30, "m4": 0.50, "m5": 0.80}
_ALPHA = 0.10
_N_BOOT = 3000
_BLOCK_LENGTH = 5
_SEED = 123


def _make_fixture(n_obs: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = 0.02 * np.sin(np.arange(n_obs) / 5.0)
    data = {
        name: base + common + rng.normal(0.0, 0.05, size=n_obs)
        for name, base in _MODEL_BASES.items()
    }
    return pd.DataFrame(data)


def _r_mcs(loss_wide: pd.DataFrame, tmp_path) -> dict[str, list[str]]:
    csv_path = tmp_path / "loss.csv"
    loss_wide.to_csv(csv_path, index=False)
    script = f'''
library(MCS)
Loss <- as.matrix(read.csv("{csv_path}"))
res <- MCSprocedure(Loss, alpha = {_ALPHA}, B = {_N_BOOT}, statistic = "Tmax",
                     k = {_BLOCK_LENGTH}, verbose = FALSE, seed = {_SEED})
emit_str("included", res@Info$included)
emit_str("excluded", res@Info$excluded)
'''
    result = run_rscript(script, timeout=300)
    return {
        "included": parse_str_list(result.get("included", "")),
        "excluded": parse_str_list(result.get("excluded", "")),
    }


def _to_loss_panel(loss_wide: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for origin, row in loss_wide.iterrows():
        for model_id, value in row.items():
            long_rows.append(
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": origin,
                    "model_id": model_id,
                    "squared_error": float(value),
                }
            )
    return pd.DataFrame(long_rows)


def test_mcs_loss_differences_matches_r_getd_exactly(tmp_path) -> None:
    """Deterministic core of the MCS statistic: no bootstrap on either side."""
    require_r("MCS")
    loss_wide = _make_fixture()
    model_names = list(loss_wide.columns)
    matrix = loss_wide.to_numpy(dtype=float)

    py_dbar, py_dibar = _mcs_loss_differences(matrix)

    csv_path = tmp_path / "loss.csv"
    loss_wide.to_csv(csv_path, index=False)
    script = f'''
library(MCS)
Loss <- as.matrix(read.csv("{csv_path}"))
d <- MCS:::GetD(Loss)
emit("dibar", d$vD_i_bar)
for (i in seq_len(ncol(Loss))) {{
  emit(paste0("dbar_row_", i), d$mD_ij_bar[i, ])
}}
'''
    result = run_rscript(script)
    r_dibar = parse_float_list(result["dibar"])

    assert py_dibar.tolist() == pytest.approx(r_dibar, abs=1e-10), (
        f"dibar mismatch: py={py_dibar.tolist()} vs R(vD_i_bar)={r_dibar}"
    )
    for i, _name in enumerate(model_names):
        r_row = parse_float_list(result[f"dbar_row_{i + 1}"])
        assert py_dbar[i].tolist() == pytest.approx(r_row, abs=1e-10), (
            f"dbar row {i} ({_name}) mismatch: py={py_dbar[i].tolist()} vs R={r_row}"
        )


def test_mcs_survivor_set_matches_r_mcsprocedure(tmp_path) -> None:
    require_r("MCS")
    loss_wide = _make_fixture()
    loss_panel = _to_loss_panel(loss_wide)

    py_result = mf.tests.model_confidence_set(
        loss_panel,
        alpha=_ALPHA,
        n_boot=_N_BOOT,
        block_length=_BLOCK_LENGTH,
        bootstrap_method="mcs_fixed_block",
        statistic="max",
        random_state=_SEED,
    )
    r_result = _r_mcs(loss_wide, tmp_path)

    py_record = next(item for item in py_result["mcs_inclusion"] if item["target"] == "y")
    py_included = set(py_record["models"])
    py_rejected_record = next(item for item in py_result["mcs_rejections"] if item["target"] == "y")
    py_excluded = set(py_rejected_record["models"])

    r_included = set(r_result["included"])
    r_excluded = set(r_result["excluded"])

    assert py_included == r_included, (
        f"MCS survivor set mismatch: py={py_included} vs R={r_included} "
        f"(rejected: py={py_excluded}, R={r_excluded})"
    )
    assert py_excluded == r_excluded, (
        f"MCS rejected-set mismatch: py={py_excluded} vs R={r_excluded}"
    )
    # By construction m3 (lowest base loss, 0.30) should always survive and
    # m1 (highest base loss, 1.15) should always be rejected -- a sanity
    # check on the fixture itself, independent of the R comparison above.
    assert "m3" in py_included and "m3" in r_included
    assert "m1" in py_excluded and "m1" in r_excluded


def test_mcs_elimination_order_is_not_a_meaningful_cross_language_invariant() -> None:
    """Executable demonstration backing the module docstring's claim (b):
    on a clean 3-model fixture (means 0/1/2, shared noise sd), the MIDDLE
    model's deterministic ``D_i_bar`` (its average loss differential
    against the *other active models*, computed by
    ``macroforecast.tests._mcs_loss_differences`` -- the same formula as R's
    ``GetD``) is close to exactly 0, i.e. near the SMALLEST magnitude in
    the group -- not the largest and not the smallest raw mean loss. This
    is why "worst raw mean loss is eliminated first" is not the right
    mental model for the Tmax elimination statistic, independent of any
    bootstrap draw or any R-vs-Python RNG difference.
    """
    n = 200
    rng = np.random.default_rng(1)
    matrix = np.column_stack(
        [
            rng.normal(0.0, 0.05, size=n),
            rng.normal(1.0, 0.05, size=n),
            rng.normal(2.0, 0.05, size=n),
        ]
    )
    _dbar, dibar = _mcs_loss_differences(matrix)
    # dibar order corresponds to columns [low, mid, high] mean loss.
    assert abs(dibar[1]) < abs(dibar[0]) and abs(dibar[1]) < abs(dibar[2]), (
        "expected the middle-mean-loss model to have the smallest |D_i_bar| "
        f"(near-zero by symmetry), got dibar={dibar.tolist()}"
    )
    assert dibar[1] == pytest.approx(0.0, abs=0.05), (
        f"expected the middle model's D_i_bar near 0 by construction, got {dibar[1]!r}"
    )
