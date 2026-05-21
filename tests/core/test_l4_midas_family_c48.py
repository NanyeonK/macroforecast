"""Independent tester validation for C48 MIDAS family promotion.

This file is produced by the StatsClaw tester pipeline (independent of
builder's implementation). Tests are derived exclusively from test-spec.md
(Cycle 48, MIDAS Family). The tester does NOT read spec.md or
implementation.md — results are validated purely against behavioral contracts.

Coverage:
  R  — Registration (families in OPERATIONAL, not FUTURE)
  V  — Validation (YAML recipe accepts/rejects correctly)
  C  — Contract/shape (fit-predict produces correct shape, dtype, finiteness)
  REC — Recovery (synthetic DGP weight recovery within tolerance)
  SD  — Seed-determinism (same seed -> bit-identical predictions)
  BX  — Bit-exact replicate (tiny config, guards RNG state leak)
  EC  — Edge cases (empty, single-obs, collinear, insufficient rows)
  XR  — Cross-reference (U-MIDAS vs manual lstsq, Almon uniform degenerate)

References:
  Ghysels, Santa-Clara & Valkanov (2004) The MIDAS Touch
  Ghysels, Sinko & Valkanov (2007) MIDAS Regressions
  Foroni, Marcellino & Schumacher (2015) U-MIDAS / Step-function lag
  Marcellino & Schumacher (2010) Unrestricted U-MIDAS
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _build_l4_model
from macroforecast.core.ops.l4_ops import (
    OPERATIONAL_MODEL_FAMILIES,
    FUTURE_MODEL_FAMILIES,
    get_family_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def midas_lf_data() -> tuple[pd.DataFrame, pd.Series]:
    """LF synthetic dataset — freq_ratio=1 mode (test-spec.md §2.1).

    DGP: y = 1.0 + 2.0 * x0 + noise, T=100 quarterly observations.
    Seed 42 via np.random.default_rng; DatetimeIndex.
    """
    rng = np.random.default_rng(42)
    T = 100
    N = 3
    X = pd.DataFrame(
        rng.standard_normal((T, N)),
        columns=["x0", "x1", "x2"],
        index=pd.date_range("2000-01-01", periods=T, freq="QS"),
    )
    y = pd.Series(
        1.0 + 2.0 * X["x0"].to_numpy() + 0.1 * rng.standard_normal(T),
        index=X.index,
        name="target",
    )
    return X, y


# ---------------------------------------------------------------------------
# Section R: Registration tests
# ---------------------------------------------------------------------------

class TestRegistration:
    """Verify that all 4 MIDAS families are operational and realized_garch
    remains future (test-spec.md §2.2).
    """

    def test_c48_midas_families_in_operational(self):
        """R-1: MIDAS families must be in OPERATIONAL, not FUTURE.

        Verifies test-spec.md behavioral contract #1:
        midas_almon, midas_beta, midas_step, dfm_unrestricted_midas
        are in OPERATIONAL_MODEL_FAMILIES and NOT in FUTURE_MODEL_FAMILIES.
        """
        for fam in ("midas_almon", "midas_beta", "midas_step", "dfm_unrestricted_midas"):
            assert fam in OPERATIONAL_MODEL_FAMILIES, f"{fam} not in OPERATIONAL"
            assert fam not in FUTURE_MODEL_FAMILIES, f"{fam} still in FUTURE"
            assert get_family_status(fam) == "operational", (
                f"get_family_status('{fam}') != 'operational'"
            )

    def test_c48_realized_garch_remains_future(self):
        """R-2: realized_garch must stay FUTURE (deferred to C49).

        Verifies test-spec.md behavioral contract #2.
        """
        assert "realized_garch" in FUTURE_MODEL_FAMILIES
        assert "realized_garch" not in OPERATIONAL_MODEL_FAMILIES
        assert get_family_status("realized_garch") == "future"

    def test_c48_operational_count_increased(self):
        """R-3: operational count must be >= 39 after C48 (+4 MIDAS families).

        test-spec.md §2.2 R-3 specifies >= 39. Actual is 46 (builder confirms).
        """
        assert len(OPERATIONAL_MODEL_FAMILIES) >= 39, (
            f"Expected >= 39 operational families, got {len(OPERATIONAL_MODEL_FAMILIES)}"
        )


# ---------------------------------------------------------------------------
# Section V: Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Verify that MIDAS recipes pass and realized_garch is rejected
    (test-spec.md §2.3).
    """

    def test_c48_midas_families_recipe_passes_validation(self):
        """V-1: YAML recipe with each MIDAS family passes validate_layer.

        Verifies test-spec.md behavioral contract #3.
        """
        from macroforecast.core.layers.l4 import validate_layer, parse_layer_yaml

        for family in ("midas_almon", "midas_beta", "midas_step", "dfm_unrestricted_midas"):
            yaml_text = f"""
4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}
    - id: src_y
      type: source
      selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}
    - id: fit_1
      type: step
      op: fit_model
      params:
        family: {family}
        freq_ratio: 1
        n_lags_high: 4
      inputs: [src_X, src_y]
    - id: predict_1
      type: step
      op: predict
      inputs: [fit_1]
  sinks:
    l4_forecasts_v1: predict_1
    l4_model_artifacts_v1: fit_1
    l4_training_metadata_v1: auto
"""
            layer = parse_layer_yaml(yaml_text, "l4")
            report = validate_layer(layer)
            assert not report.has_hard_errors, (
                f"family={family} rejected unexpectedly: "
                f"{[e.message for e in report.hard_errors]}"
            )

    def test_c48_realized_garch_recipe_rejected(self):
        """V-2: YAML recipe with realized_garch must be hard-rejected.

        Verifies test-spec.md behavioral contract #4:
        error message must contain 'future' or 'realized_garch'.
        """
        from macroforecast.core.layers.l4 import validate_layer, parse_layer_yaml

        yaml_text = """
4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: f
      type: step
      op: fit_model
      params:
        family: realized_garch
        n_lag: 4
      inputs: [src_X, src_y]
    - id: predict_1
      type: step
      op: predict
      inputs: [f]
  sinks:
    l4_forecasts_v1: predict_1
    l4_model_artifacts_v1: f
    l4_training_metadata_v1: auto
"""
        layer = parse_layer_yaml(yaml_text, "l4")
        report = validate_layer(layer)
        assert report.has_hard_errors, "realized_garch should be hard-rejected"
        messages = [e.message for e in report.hard_errors]
        assert any("future" in m.lower() or "realized_garch" in m for m in messages), (
            f"Error messages do not mention 'future' or 'realized_garch': {messages}"
        )


# ---------------------------------------------------------------------------
# Section C: Contract / shape tests
# ---------------------------------------------------------------------------

# Parametrize all 4 families with same params used in test-spec.md §2.4
_C_PARAMS = [
    ("midas_almon",  {"freq_ratio": 1, "n_lags_high": 4, "polynomial_order": 2, "n_starts": 2, "random_state": 0}),
    ("midas_beta",   {"freq_ratio": 1, "n_lags_high": 4, "n_starts": 2, "random_state": 0}),
    ("midas_step",   {"freq_ratio": 1, "n_lags_high": 4, "n_steps": 2}),
    ("dfm_unrestricted_midas", {"freq_ratio": 1, "n_lags_high": 3, "random_state": 0}),
]


class TestContract:
    """Verify fit-predict contract: correct shape, dtype, finiteness.
    (test-spec.md §2.4)
    """

    @pytest.mark.parametrize("family,params", _C_PARAMS)
    def test_c48_midas_fit_predict_shape(self, family: str, params: dict, midas_lf_data: tuple) -> None:
        """C-1: fit-predict returns ndarray of shape (n_rows,) for all 4 families.

        Verifies test-spec.md behavioral contract #5.
        """
        X, y = midas_lf_data
        model = _build_l4_model(family, {"family": family, **params})
        model.fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray), f"{family}: predict must return np.ndarray"
        assert preds.shape == (len(X),), f"{family}: wrong prediction shape {preds.shape}"
        assert np.all(np.isfinite(preds)), f"{family}: predictions contain non-finite values"

    @pytest.mark.parametrize("family,params", _C_PARAMS)
    def test_c48_midas_single_row_predict(self, family: str, params: dict, midas_lf_data: tuple) -> None:
        """C-2: single-row predict does not crash and returns shape (1,).

        Verifies test-spec.md §2.4 C-2.
        """
        X, y = midas_lf_data
        model = _build_l4_model(family, {"family": family, **params})
        model.fit(X, y)
        preds = model.predict(X.iloc[[0]])
        assert preds.shape == (1,), f"{family}: single-row predict shape wrong: {preds.shape}"
        assert np.isfinite(preds[0]), f"{family}: single-row predict non-finite: {preds[0]}"

    @pytest.mark.parametrize("family,params", _C_PARAMS)
    def test_c48_midas_predict_dtype_float(self, family: str, params: dict, midas_lf_data: tuple) -> None:
        """C-3: predictions must be float-compatible dtype, not object arrays.

        Verifies test-spec.md §2.4 C-3 property-based invariant #1.
        """
        X, y = midas_lf_data
        model = _build_l4_model(family, {"family": family, **params})
        model.fit(X, y)
        preds = model.predict(X)
        assert np.issubdtype(preds.dtype, np.floating) or np.issubdtype(preds.dtype, np.integer), (
            f"{family}: prediction dtype {preds.dtype} not numeric"
        )
        # Must not be object dtype
        assert preds.dtype != object, f"{family}: object dtype returned"


# ---------------------------------------------------------------------------
# Section REC: Recovery tests
# ---------------------------------------------------------------------------

class TestRecovery:
    """Verify that families recover known DGP weights/coefficients within tolerance.
    (test-spec.md §2.5)

    Tolerances used are EXACTLY as specified in test-spec.md (not modified).
    """

    def test_c48_almon_weight_recovery(self) -> None:
        """REC-1: midas_almon recovers Almon weights within atol=0.10.

        DGP: theta=[0.5, -0.1], Q=1, K=3 lags, T=200.
        Expected: np.allclose(w_hat, w_true, atol=0.10).
        Tolerance source: test-spec.md §2.5 REC-1 (atol=0.10).
        """
        rng = np.random.default_rng(99)
        T = 200
        K = 3
        # True Almon weights: Q=1 polynomial theta=[0.5, -0.1]
        theta_true = np.array([0.5, -0.1])
        k_idx = np.arange(K, dtype=float)
        w_raw = theta_true[0] + theta_true[1] * k_idx
        w_raw = np.maximum(w_raw, 0.0)
        w_true = w_raw / w_raw.sum()  # normalized; approx [0.5, 0.3, 0.2]

        x_series = rng.standard_normal(T + K)
        lag_cols = {f"x_lag{k}": x_series[K - k : T + K - k] for k in range(K)}
        X = pd.DataFrame(lag_cols, index=range(T))
        y = pd.Series(
            1.5 + 2.0 * (X.to_numpy() @ w_true) + 0.05 * rng.standard_normal(T),
            index=range(T),
            name="target",
        )
        model = _build_l4_model(
            "midas_almon",
            {"family": "midas_almon", "freq_ratio": 1, "n_lags_high": K,
             "polynomial_order": 1, "n_starts": 5, "random_state": 42},
        )
        model.fit(X, y)
        w_hat = model._w_hat  # shape (K,)
        assert np.allclose(w_hat, w_true, atol=0.10), (
            f"REC-1 Almon weight recovery failed: w_true={w_true}, w_hat={w_hat}"
        )
        # Forecast RMSE on hold-out
        preds = model.predict(X.iloc[-20:])
        y_test = y.iloc[-20:].to_numpy()
        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        assert rmse < 0.50, f"REC-1 Almon RMSE too high: {rmse:.4f}"

    def test_c48_beta_weight_recovery(self) -> None:
        """REC-2: midas_beta recovers Beta weights within atol=0.12.

        DGP: a=1.5, b=3.0, K=6, T=200.
        Tolerance source: test-spec.md §2.5 REC-2 (atol=0.12).
        """
        rng = np.random.default_rng(77)
        T = 200
        K = 6
        kk = (np.arange(K, dtype=float) + 1.0) / (K + 1.0)
        a_true, b_true = 1.5, 3.0
        w_raw = kk ** (a_true - 1.0) * (1.0 - kk) ** (b_true - 1.0)
        w_true = w_raw / w_raw.sum()

        x_series = rng.standard_normal(T + K)
        lag_cols = {f"x_lag{k}": x_series[K - k : T + K - k] for k in range(K)}
        X = pd.DataFrame(lag_cols, index=range(T))
        y = pd.Series(
            0.5 + 1.5 * (X.to_numpy() @ w_true) + 0.05 * rng.standard_normal(T),
            index=range(T),
            name="target",
        )
        model = _build_l4_model(
            "midas_beta",
            {"family": "midas_beta", "freq_ratio": 1, "n_lags_high": K,
             "n_starts": 5, "random_state": 42},
        )
        model.fit(X, y)
        w_hat = model._w_hat
        assert np.allclose(w_hat, w_true, atol=0.12), (
            f"REC-2 Beta weight recovery failed: w_true={w_true}, w_hat={w_hat}"
        )

    def test_c48_step_weight_recovery(self) -> None:
        """REC-3: midas_step OLS recovers piecewise-constant structure.

        DGP: 2 step groups, group0 weight=2.0, group1 weight=0.5, K=6, T=150.
        Expected RMSE < 0.10 (test-spec.md §2.5 REC-3).
        """
        rng = np.random.default_rng(55)
        T = 150
        K = 6
        S = 2
        step_weights = np.array([2.0, 0.5])
        x_series = rng.standard_normal(T + K)
        lag_cols = {f"x_lag{k}": x_series[K - k : T + K - k] for k in range(K)}
        X = pd.DataFrame(lag_cols, index=range(T))
        group0_agg = X[["x_lag0", "x_lag1", "x_lag2"]].mean(axis=1)
        group1_agg = X[["x_lag3", "x_lag4", "x_lag5"]].mean(axis=1)
        y = pd.Series(
            1.0 + step_weights[0] * group0_agg + step_weights[1] * group1_agg
            + 0.05 * rng.standard_normal(T),
            index=range(T),
            name="target",
        )
        model = _build_l4_model(
            "midas_step",
            {"family": "midas_step", "freq_ratio": 1, "n_lags_high": K, "n_steps": S},
        )
        model.fit(X, y)
        preds = model.predict(X)
        rmse = float(np.sqrt(np.mean((preds - y.to_numpy()) ** 2)))
        assert rmse < 0.10, f"REC-3 Step OLS RMSE too high: {rmse:.4f}"

    def test_c48_umidas_ols_recovery(self) -> None:
        """REC-4: dfm_unrestricted_midas OLS recovers free coefficients.

        DGP: psi=[1.0, 0.5, 0.25], K=3, T=200.
        Expected: RMSE < 0.15, coefficient recovery within atol=0.15.
        Tolerance source: test-spec.md §2.5 REC-4.
        """
        rng = np.random.default_rng(33)
        T = 200
        K = 3
        psi_true = np.array([1.0, 0.5, 0.25])

        x_series = rng.standard_normal(T + K)
        lag_cols = {f"x_lag{k}": x_series[K - k : T + K - k] for k in range(K)}
        X = pd.DataFrame(lag_cols, index=range(T))
        y = pd.Series(
            2.0 + X.to_numpy() @ psi_true + 0.05 * rng.standard_normal(T),
            index=range(T),
            name="target",
        )
        model = _build_l4_model(
            "dfm_unrestricted_midas",
            {"family": "dfm_unrestricted_midas", "freq_ratio": 1, "n_lags_high": K},
        )
        model.fit(X, y)
        preds = model.predict(X)
        rmse = float(np.sqrt(np.mean((preds - y.to_numpy()) ** 2)))
        assert rmse < 0.15, f"REC-4 U-MIDAS OLS RMSE too high: {rmse:.4f}"
        # Check coefficient recovery: coef = [intercept, psi_0, psi_1, psi_2]
        coef = model._coef
        psi_hat = coef[1:]
        assert np.allclose(psi_hat, psi_true, atol=0.15), (
            f"REC-4 U-MIDAS coefficient recovery failed: psi_true={psi_true}, psi_hat={psi_hat}"
        )


# ---------------------------------------------------------------------------
# Section SD: Seed-determinism tests
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    """Verify same seed -> bit-identical results; different seeds do not crash.
    (test-spec.md §2.6)
    """

    def test_c48_almon_seed_determinism(self, midas_lf_data: tuple) -> None:
        """SD-1: midas_almon same seed -> bit-identical predictions.

        Verifies test-spec.md behavioral contract #7.
        """
        X, y = midas_lf_data
        params = {"family": "midas_almon", "freq_ratio": 1, "n_lags_high": 4,
                  "polynomial_order": 2, "n_starts": 3, "random_state": 7}
        m1 = _build_l4_model("midas_almon", params)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = _build_l4_model("midas_almon", params)
        m2.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_equal(
            p1, p2,
            err_msg="SD-1: midas_almon same-seed predictions not bit-identical"
        )

    def test_c48_beta_seed_determinism(self, midas_lf_data: tuple) -> None:
        """SD-2: midas_beta same seed -> bit-identical predictions.

        Verifies test-spec.md §2.6 SD-2.
        """
        X, y = midas_lf_data
        params = {"family": "midas_beta", "freq_ratio": 1, "n_lags_high": 4,
                  "n_starts": 3, "random_state": 7}
        m1 = _build_l4_model("midas_beta", params)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = _build_l4_model("midas_beta", params)
        m2.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_equal(
            p1, p2,
            err_msg="SD-2: midas_beta same-seed predictions not bit-identical"
        )

    def test_c48_almon_different_seeds_no_crash(self, midas_lf_data: tuple) -> None:
        """SD-3: different seeds accepted without crash (NLS may converge same).

        test-spec.md §2.6 SD-3: does NOT assert values differ — NLS may converge
        to same optimum. Asserts shapes match.
        """
        X, y = midas_lf_data
        p1 = _build_l4_model("midas_almon",
            {"family": "midas_almon", "random_state": 0, "n_starts": 3, "n_lags_high": 4}
        ).fit(X, y).predict(X)
        p2 = _build_l4_model("midas_almon",
            {"family": "midas_almon", "random_state": 9999, "n_starts": 3, "n_lags_high": 4}
        ).fit(X, y).predict(X)
        assert p1.shape == p2.shape

    @pytest.mark.parametrize("family,params", [
        ("midas_step", {"freq_ratio": 1, "n_lags_high": 4, "n_steps": 2}),
        ("dfm_unrestricted_midas", {"freq_ratio": 1, "n_lags_high": 3}),
    ])
    def test_c48_ols_families_are_deterministic(
        self, family: str, params: dict, midas_lf_data: tuple
    ) -> None:
        """SD-4: OLS families (step, dfm_unrestricted_midas) are deterministic.

        test-spec.md §2.6 SD-4.
        """
        X, y = midas_lf_data
        full_params = {"family": family, **params}
        p1 = _build_l4_model(family, full_params).fit(X, y).predict(X)
        p2 = _build_l4_model(family, full_params).fit(X, y).predict(X)
        np.testing.assert_array_equal(
            p1, p2,
            err_msg=f"SD-4: {family} OLS predictions not deterministic"
        )


# ---------------------------------------------------------------------------
# Section BX: Bit-exact replicate tests
# ---------------------------------------------------------------------------

class TestBitExactReplicate:
    """Bit-exact replicate tests using tiny fixed datasets.
    (test-spec.md §2.7)
    """

    def test_c48_almon_bit_exact_replicate(self) -> None:
        """BX-1: midas_almon bit-exact replicate on tiny dataset (T=20).

        Guards against RNG state leak. Two independent invocations with
        identical seed + data must produce bit-identical predictions.
        test-spec.md §2.7 BX-1.
        """
        rng = np.random.default_rng(0)
        T = 20
        X = pd.DataFrame(
            {"a": rng.standard_normal(T), "b": rng.standard_normal(T)},
            index=range(T),
        )
        y = pd.Series(rng.standard_normal(T), index=range(T), name="y")
        params = {"family": "midas_almon", "freq_ratio": 1, "n_lags_high": 3,
                  "polynomial_order": 2, "n_starts": 2, "random_state": 42}
        p1 = _build_l4_model("midas_almon", params).fit(X, y).predict(X)
        p2 = _build_l4_model("midas_almon", params).fit(X, y).predict(X)
        np.testing.assert_array_equal(
            p1, p2,
            err_msg="BX-1: midas_almon not bit-exactly reproducible"
        )

    def test_c48_umidas_bit_exact_replicate(self) -> None:
        """BX-2: dfm_unrestricted_midas OLS is bit-exact (numpy lstsq deterministic).

        test-spec.md §2.7 BX-2.
        """
        rng = np.random.default_rng(0)
        T = 20
        X = pd.DataFrame(
            {"a": rng.standard_normal(T), "b": rng.standard_normal(T)},
            index=range(T),
        )
        y = pd.Series(rng.standard_normal(T), index=range(T), name="y")
        params = {"family": "dfm_unrestricted_midas", "freq_ratio": 1, "n_lags_high": 2}
        p1 = _build_l4_model("dfm_unrestricted_midas", params).fit(X, y).predict(X)
        p2 = _build_l4_model("dfm_unrestricted_midas", params).fit(X, y).predict(X)
        np.testing.assert_array_equal(
            p1, p2,
            err_msg="BX-2: dfm_unrestricted_midas not bit-exactly reproducible"
        )


# ---------------------------------------------------------------------------
# Section EC: Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty input, single observation, collinear X, insufficient rows.
    (test-spec.md §2.8)
    """

    @pytest.mark.parametrize("family", [
        "midas_almon", "midas_beta", "midas_step", "dfm_unrestricted_midas"
    ])
    def test_c48_midas_empty_input(self, family: str) -> None:
        """EC-1: empty X/y does not crash; predict returns shape (0,).

        test-spec.md §2.8 EC-1.
        """
        X = pd.DataFrame({"a": [], "b": []})
        y = pd.Series([], dtype=float, name="y")
        params = {"family": family, "freq_ratio": 1, "n_lags_high": 2}
        model = _build_l4_model(family, params)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (0,), f"EC-1 {family}: empty predict shape not (0,): {preds.shape}"

    @pytest.mark.parametrize("family", [
        "midas_almon", "midas_beta", "midas_step", "dfm_unrestricted_midas"
    ])
    def test_c48_midas_single_obs(self, family: str) -> None:
        """EC-2: single-observation training does not crash.

        test-spec.md §2.8 EC-2.
        """
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        y = pd.Series([3.0], name="y")
        params = {"family": family, "freq_ratio": 1, "n_lags_high": 2}
        model = _build_l4_model(family, params)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (1,), f"EC-2 {family}: single-obs predict shape wrong: {preds.shape}"
        assert np.isfinite(preds[0]), f"EC-2 {family}: single-obs predict non-finite: {preds[0]}"

    def test_c48_midas_collinear_x(self) -> None:
        """EC-3: perfectly collinear X must not produce non-finite predictions.

        test-spec.md §2.8 EC-3. All 4 families must handle singular X gracefully.
        """
        T = 50
        x = np.arange(T, dtype=float)
        X = pd.DataFrame({"a": x, "b": x * 2.0, "c": x * 3.0})
        y = pd.Series(x + 1.0, name="y")
        for family in ("midas_almon", "midas_beta", "midas_step", "dfm_unrestricted_midas"):
            params = {"family": family, "freq_ratio": 1, "n_lags_high": 2}
            model = _build_l4_model(family, params)
            model.fit(X, y)
            preds = model.predict(X)
            assert np.all(np.isfinite(preds)), (
                f"EC-3 {family}: non-finite preds on collinear X: {preds}"
            )

    def test_c48_midas_insufficient_rows(self) -> None:
        """EC-4: T << n_lags_high must fall back gracefully (no exception, finite preds).

        test-spec.md §2.8 EC-4.
        """
        X = pd.DataFrame({"a": [1.0, 2.0]})
        y = pd.Series([1.5, 2.5], name="y")
        for family in ("midas_almon", "midas_beta", "midas_step", "dfm_unrestricted_midas"):
            params = {"family": family, "freq_ratio": 1, "n_lags_high": 10}
            model = _build_l4_model(family, params)
            model.fit(X, y)
            preds = model.predict(X)
            assert np.all(np.isfinite(preds)), (
                f"EC-4 {family}: non-finite preds on insufficient rows: {preds}"
            )


# ---------------------------------------------------------------------------
# Section XR: Cross-reference tests
# ---------------------------------------------------------------------------

class TestCrossReference:
    """Cross-reference tests against known-good reference implementations.
    (test-spec.md §2.9)
    """

    def test_c48_umidas_matches_manual_ols(self) -> None:
        """XR-1: U-MIDAS OLS result matches manual NumPy lstsq reference.

        test-spec.md §2.9 XR-1. Tolerance: rtol=1e-10.
        This is the primary correctness cross-check for dfm_unrestricted_midas.
        """
        rng = np.random.default_rng(888)
        T, K = 100, 3
        x = rng.standard_normal(T + K)
        lag_cols = {f"x_lag{k}": x[K - k : T + K - k] for k in range(K)}
        X = pd.DataFrame(lag_cols, index=range(T))
        y = pd.Series(rng.standard_normal(T), index=range(T), name="y")

        # Reference: manual OLS via numpy lstsq
        X_arr = X.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        X_aug = np.hstack([np.ones((T, 1)), X_arr])
        coef_ref, *_ = np.linalg.lstsq(X_aug, y_arr, rcond=None)
        preds_ref = X_aug @ coef_ref

        # Model under test
        model = _build_l4_model(
            "dfm_unrestricted_midas",
            {"family": "dfm_unrestricted_midas", "freq_ratio": 1, "n_lags_high": K},
        )
        model.fit(X, y)
        preds_model = model.predict(X)

        np.testing.assert_allclose(
            preds_model, preds_ref, rtol=1e-10,
            err_msg="XR-1: U-MIDAS predict does not match manual lstsq reference"
        )

    def test_c48_almon_linear_order_uniform_degenerate(self) -> None:
        """XR-2: Q=1 Almon with uniform DGP recovers near-uniform weights.

        DGP: equal-weight average of K=4 lags. Almon Q=1 should find near-uniform
        weights (all close to 1/K=0.25). Tolerance: atol=0.15.
        test-spec.md §2.9 XR-2.
        """
        rng = np.random.default_rng(11)
        T, K = 80, 4
        X = pd.DataFrame(
            {f"x_lag{k}": rng.standard_normal(T) for k in range(K)},
            index=range(T),
        )
        # Uniform DGP: y = mean of all lags + tiny noise
        y = pd.Series(X.mean(axis=1) + 0.01 * rng.standard_normal(T), index=range(T), name="y")
        model = _build_l4_model(
            "midas_almon",
            {"family": "midas_almon", "freq_ratio": 1, "n_lags_high": K,
             "polynomial_order": 1, "n_starts": 5, "random_state": 0},
        )
        model.fit(X, y)
        w_hat = model._w_hat
        expected_uniform = np.full(K, 1.0 / K)
        assert np.allclose(w_hat, expected_uniform, atol=0.15), (
            f"XR-2 Almon Q=1 uniform DGP: weights not near-uniform: {w_hat}"
        )

    @pytest.mark.skipif(True, reason="R midasr cross-ref: run manually with rpy2 installed")
    def test_c48_almon_r_midasr_crossref(self) -> None:
        """XR-3: R midasr equivalence — gated, run manually.

        test-spec.md §2.9 XR-3. Expected tolerance: Almon weight difference
        < 1e-4 vs R midasr almonp() for freq_ratio=3, K=9, polynomial_order=2.
        """
        rpy2 = pytest.importorskip("rpy2")
        pass  # rpy2 bridge code to run midasr in R and compare outputs


# ---------------------------------------------------------------------------
# Section PI: Property-based invariants
# ---------------------------------------------------------------------------

class TestPropertyInvariants:
    """Property-based invariants that must hold for all 4 families.
    (test-spec.md §4)
    """

    @pytest.mark.parametrize("family,params", _C_PARAMS)
    def test_c48_non_negative_weights_almon_beta(self, family: str, params: dict, midas_lf_data: tuple) -> None:
        """PI-5: midas_almon and midas_beta weights must be non-negative.

        test-spec.md §4 property-based invariant #5.
        """
        if family not in ("midas_almon", "midas_beta"):
            pytest.skip(f"{family} does not expose _w_hat for non-negativity check")
        X, y = midas_lf_data
        model = _build_l4_model(family, {"family": family, **params})
        model.fit(X, y)
        w_hat = model._w_hat
        assert np.all(w_hat >= -1e-12), (
            f"PI-5 {family}: negative weights found: {w_hat}"
        )

    @pytest.mark.parametrize("family", ["midas_almon", "midas_beta"])
    def test_c48_weight_vector_length(self, family: str, midas_lf_data: tuple) -> None:
        """PI-7: len(model._w_hat) == n_lags_high for Almon and Beta.

        test-spec.md §4 property-based invariant #7.
        """
        X, y = midas_lf_data
        n_lags = 4
        params = {"family": family, "freq_ratio": 1, "n_lags_high": n_lags,
                  "n_starts": 2, "random_state": 0}
        if family == "midas_almon":
            params["polynomial_order"] = 2
        model = _build_l4_model(family, {"family": family, **params})
        model.fit(X, y)
        assert len(model._w_hat) == n_lags, (
            f"PI-7 {family}: len(_w_hat)={len(model._w_hat)}, expected {n_lags}"
        )

    @pytest.mark.parametrize("family", ["midas_step", "dfm_unrestricted_midas"])
    def test_c48_ols_intercept_is_finite(self, family: str, midas_lf_data: tuple) -> None:
        """PI-8: OLS families expose a finite _intercept float.

        test-spec.md §4 property-based invariant #8.
        """
        X, y = midas_lf_data
        if family == "midas_step":
            params = {"family": family, "freq_ratio": 1, "n_lags_high": 3, "n_steps": 2}
        else:
            params = {"family": family, "freq_ratio": 1, "n_lags_high": 3}
        model = _build_l4_model(family, {"family": family, **params})
        model.fit(X, y)
        intercept = model._intercept
        assert isinstance(intercept, float) or np.isscalar(intercept), (
            f"PI-8 {family}: _intercept not scalar: {type(intercept)}"
        )
        assert np.isfinite(float(intercept)), (
            f"PI-8 {family}: _intercept not finite: {intercept}"
        )

    def test_c48_umidas_bic_k_fit_ge1(self) -> None:
        """PI-9: dfm_unrestricted_midas with n_lags_high='bic' sets _K_fit >= 1.

        test-spec.md §4 property-based invariant #9.
        """
        rng = np.random.default_rng(42)
        T = 50
        X = pd.DataFrame(
            rng.standard_normal((T, 4)),
            columns=[f"x{i}" for i in range(4)],
        )
        y = pd.Series(rng.standard_normal(T), name="y")
        model = _build_l4_model(
            "dfm_unrestricted_midas",
            {"family": "dfm_unrestricted_midas", "freq_ratio": 1, "n_lags_high": "bic"},
        )
        model.fit(X, y)
        assert model._K_fit >= 1, f"PI-9: _K_fit={model._K_fit}, expected >= 1"
