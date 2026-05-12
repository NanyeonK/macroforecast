"""test_tester_f15_f16.py — Tester-written independent validation for F-15 and F-16.

Written by tester agent (independent of builder implementation).
Tests are derived from test-spec.md scenarios T-15-01 through T-15-05
and T-16-01 through T-16-07.

Run: pytest tests/core/test_tester_f15_f16.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# F-15 Tests: sparse_macro_factors_risk_premia
# ---------------------------------------------------------------------------


class TestF15SparseRiskPreemia:
    """T-15-01 through T-15-05."""

    def test_t15_01_importable(self):
        """T-15-01: importable from public API."""
        from macroforecast.recipes.paper_methods import sparse_macro_factors_risk_premia  # noqa: F401
        assert callable(sparse_macro_factors_risk_premia)

    def test_t15_02_finite_gamma_hat_on_synthetic(self):
        """T-15-02: Returns finite gamma_hat of shape (J,)."""
        from macroforecast.recipes.paper_methods import sparse_macro_factors_risk_premia

        rng = np.random.default_rng(42)
        T_eff, J, N = 60, 3, 20
        sparse_factors = rng.standard_normal((T_eff, J))
        sparse_factors[0, :] = 0.0
        test_asset_returns = rng.standard_normal((T_eff, N))

        result = sparse_macro_factors_risk_premia(
            sparse_factors, test_asset_returns, q_grid=(0.5, 1.0), cv_folds=2, random_state=42,
        )

        assert isinstance(result, dict)
        gamma_hat = result["gamma_hat"]
        assert gamma_hat.shape == (J,), f"gamma_hat shape {gamma_hat.shape} != ({J},)"
        assert np.all(np.isfinite(gamma_hat)), f"gamma_hat not all finite: {gamma_hat}"
        beta_hat = result["beta_hat"]
        assert beta_hat.shape == (N, J), f"beta_hat shape {beta_hat.shape} != ({N},{J})"
        assert np.all(np.isfinite(beta_hat))
        fmp = result["fmp_returns"]
        assert fmp.shape == (T_eff - 1, J), f"fmp_returns shape {fmp.shape} != ({T_eff-1},{J})"
        assert result["q_selected"] in (0.5, 1.0)
        assert isinstance(result["screened_assets"], list)
        assert len(result["screened_assets"]) == J

    def test_t15_03_shape_mismatch_raises(self):
        """T-15-03: Row mismatch raises ValueError."""
        from macroforecast.recipes.paper_methods import sparse_macro_factors_risk_premia

        rng = np.random.default_rng(0)
        sparse_factors = rng.standard_normal((60, 3))
        test_asset_returns = rng.standard_normal((50, 20))

        with pytest.raises(ValueError):
            sparse_macro_factors_risk_premia(
                sparse_factors, test_asset_returns, q_grid=(1.0,), cv_folds=2,
            )

    def test_t15_04_boundary_row_drop(self):
        """T-15-04: Row 0 dropped; fmp_returns.shape[0] == T - 1."""
        from macroforecast.recipes.paper_methods import sparse_macro_factors_risk_premia

        rng = np.random.default_rng(0)
        T, J, N = 61, 2, 10
        sparse_factors = rng.standard_normal((T, J))
        sparse_factors[0, :] = 0.0
        test_asset_returns = rng.standard_normal((T, N))

        result = sparse_macro_factors_risk_premia(
            sparse_factors, test_asset_returns, q_grid=(1.0,), cv_folds=2,
        )
        assert result["fmp_returns"].shape[0] == T - 1, (
            f"fmp_returns.shape[0] expected {T-1}, got {result['fmp_returns'].shape[0]}"
        )

    def test_t15_05_q_selected_in_grid(self):
        """T-15-05: q_selected is always in provided q_grid."""
        from macroforecast.recipes.paper_methods import sparse_macro_factors_risk_premia

        rng = np.random.default_rng(42)
        T_eff, J, N = 60, 3, 20
        sparse_factors = rng.standard_normal((T_eff, J))
        sparse_factors[0, :] = 0.0
        test_asset_returns = rng.standard_normal((T_eff, N))
        q_grid = (0.25, 0.50, 1.00)

        result = sparse_macro_factors_risk_premia(
            sparse_factors, test_asset_returns, q_grid=q_grid, cv_folds=2, random_state=42,
        )
        assert result["q_selected"] in set(q_grid), (
            f"q_selected {result['q_selected']} not in q_grid {q_grid}"
        )


# ---------------------------------------------------------------------------
# F-16 Tests: maf_per_variable_pca op
# ---------------------------------------------------------------------------


class TestF16MafPerVariablePca:
    """T-16-01 through T-16-07."""

    def test_t16_01_op_registered(self):
        """T-16-01: op is in the registry."""
        from macroforecast.core.ops.registry import _OPS  # type: ignore
        assert "maf_per_variable_pca" in _OPS, "maf_per_variable_pca missing from _OPS registry"

    def test_t16_02_output_shape_t_by_2k(self):
        """T-16-02: Output shape (T, 2K) with n_components_per_var=2."""
        from macroforecast.core.runtime import _maf_per_variable_pca

        rng = np.random.default_rng(7)
        T, K = 50, 4
        frame = pd.DataFrame(rng.standard_normal((T, K)), columns=["a", "b", "c", "d"])
        out = _maf_per_variable_pca(frame, n_lags=3, n_components_per_var=2)

        assert out.shape == (T, 2 * K), f"Expected ({T},{2*K}), got {out.shape}"
        for col in ["a", "b", "c", "d"]:
            assert f"{col}_maf1" in out.columns
            assert f"{col}_maf2" in out.columns
        assert out.iloc[:3].isna().all(axis=1).all(), "First 3 rows must be NaN"
        non_nan = out.iloc[3:]
        assert non_nan.notna().all(axis=1).all(), "Rows >= n_lags must be non-NaN"
        assert np.all(np.isfinite(non_nan.values)), "Non-NaN values must be finite"

    def test_t16_03_n_components_1_gives_shape_t_by_k(self):
        """T-16-03: n_components_per_var=1 → output shape (T, K)."""
        from macroforecast.core.runtime import _maf_per_variable_pca

        rng = np.random.default_rng(7)
        T, K = 50, 4
        frame = pd.DataFrame(rng.standard_normal((T, K)), columns=["a", "b", "c", "d"])
        out = _maf_per_variable_pca(frame, n_lags=3, n_components_per_var=1)

        assert out.shape == (T, K), f"Expected ({T},{K}), got {out.shape}"
        for col in ["a", "b", "c", "d"]:
            assert f"{col}_maf1" in out.columns
            assert f"{col}_maf2" not in out.columns

    def test_t16_04_known_answer_linear_trend_monotone(self):
        """T-16-04: Linear trend → output strictly monotone (PCA captures trend direction)."""
        from macroforecast.core.runtime import _maf_per_variable_pca

        series = np.arange(30, dtype=float)
        frame = pd.DataFrame({"trend": series})
        out = _maf_per_variable_pca(frame, n_lags=2, n_components_per_var=1)

        assert out.shape == (30, 1), f"Expected (30, 1), got {out.shape}"
        non_nan = out.iloc[2:, 0].values
        diffs = np.diff(non_nan)
        assert np.all(diffs > 0) or np.all(diffs < 0), (
            "Linear trend PC output must be strictly monotone (trend preserved); "
            f"diffs min={diffs.min():.4f} max={diffs.max():.4f}"
        )

    def test_t16_05_single_variable_k1_no_fail(self):
        """T-16-05: K=1 does not raise; output shape (T, 2)."""
        from macroforecast.core.runtime import _maf_per_variable_pca

        rng = np.random.default_rng(0)
        T = 40
        frame = pd.DataFrame({"x": rng.standard_normal(T)})
        out = _maf_per_variable_pca(frame, n_lags=5, n_components_per_var=2)

        assert out.shape == (T, 2), f"Expected ({T}, 2), got {out.shape}"

    def test_t16_06_n_lags_1_minimum_boundary(self):
        """T-16-06: n_lags=1 works; first 1 row NaN, rest non-NaN."""
        from macroforecast.core.runtime import _maf_per_variable_pca

        rng = np.random.default_rng(0)
        T, K = 30, 3
        frame = pd.DataFrame(rng.standard_normal((T, K)), columns=["x", "y", "z"])
        out = _maf_per_variable_pca(frame, n_lags=1, n_components_per_var=2)

        assert out.shape == (T, 2 * K), f"Expected ({T},{2*K}), got {out.shape}"
        assert out.iloc[:1].isna().all(axis=1).all(), "First 1 row must be NaN"
        assert out.iloc[1:].notna().all(axis=1).all(), "Rows >= 1 must be non-NaN"

    def test_t16_07_op_dispatch_via_recipe_dag(self, tmp_path):
        """T-16-07: maf_per_variable_pca dispatches via the L3 DAG (full recipe run)."""
        import macroforecast

        rng = np.random.default_rng(0)
        T = 80
        cols = ["x0", "x1", "x2", "x3"]
        dates = pd.date_range("2015-01-01", periods=T, freq="MS").strftime("%Y-%m-%d").tolist()
        y_vals = rng.standard_normal(T).tolist()
        x_vals = {col: rng.standard_normal(T).tolist() for col in cols}

        # Build recipe as a dict (avoids YAML inline formatting issues)
        recipe = {
            "0_meta": {
                "fixed_axes": {
                    "failure_policy": "fail_fast",
                    "reproducibility_mode": "seeded_reproducible",
                },
                "leaf_config": {"random_seed": 42},
            },
            "1_data": {
                "fixed_axes": {
                    "custom_source_policy": "custom_panel_only",
                    "frequency": "monthly",
                    "horizon_set": "custom_list",
                },
                "leaf_config": {
                    "target": "y",
                    "target_horizons": [1],
                    "custom_panel_inline": {
                        "date": dates,
                        "y": y_vals,
                        **x_vals,
                    },
                },
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "transform_policy": "no_transform",
                    "outlier_policy": "none",
                    "imputation_policy": "none_propagate",
                    "frame_edge_policy": "keep_unbalanced",
                },
            },
            "3_feature_engineering": {
                "nodes": [
                    {
                        "id": "src_X",
                        "type": "source",
                        "selector": {
                            "layer_ref": "l2",
                            "sink_name": "l2_clean_panel_v1",
                            "subset": {"role": "predictors"},
                        },
                    },
                    {
                        "id": "src_y",
                        "type": "source",
                        "selector": {
                            "layer_ref": "l2",
                            "sink_name": "l2_clean_panel_v1",
                            "subset": {"role": "target"},
                        },
                    },
                    {
                        "id": "maf_step",
                        "type": "step",
                        "op": "maf_per_variable_pca",
                        "params": {
                            "n_lags": 3,
                            "n_components_per_var": 2,
                            "temporal_rule": "expanding_window_per_origin",
                        },
                        "inputs": ["src_X"],
                    },
                    {
                        "id": "y_h",
                        "type": "step",
                        "op": "target_construction",
                        "params": {"mode": "point_forecast", "method": "direct", "horizon": 1},
                        "inputs": ["src_y"],
                    },
                ],
                "sinks": {
                    "l3_features_v1": {"X_final": "maf_step", "y_final": "y_h"},
                    "l3_metadata_v1": "auto",
                },
            },
            "4_forecasting_model": {
                "nodes": [
                    {
                        "id": "src_X",
                        "type": "source",
                        "selector": {
                            "layer_ref": "l3",
                            "sink_name": "l3_features_v1",
                            "subset": {"component": "X_final"},
                        },
                    },
                    {
                        "id": "src_y",
                        "type": "source",
                        "selector": {
                            "layer_ref": "l3",
                            "sink_name": "l3_features_v1",
                            "subset": {"component": "y_final"},
                        },
                    },
                    {
                        "id": "fit",
                        "type": "step",
                        "op": "fit_model",
                        "params": {
                            "family": "ridge",
                            "alpha": 1.0,
                            "forecast_strategy": "direct",
                            "training_start_rule": "expanding",
                            "refit_policy": "every_origin",
                            "search_algorithm": "none",
                            "min_train_size": 20,
                        },
                        "inputs": ["src_X", "src_y"],
                    },
                    {"id": "predict", "type": "step", "op": "predict", "inputs": ["fit", "src_X"]},
                ],
                "sinks": {
                    "l4_forecasts_v1": "predict",
                    "l4_model_artifacts_v1": "fit",
                    "l4_training_metadata_v1": "auto",
                },
            },
        }

        result = macroforecast.run(recipe, output_directory=str(tmp_path / "f16_dag"))
        assert result.cells, "result.cells must be non-empty after L3 maf_per_variable_pca dispatch"
        assert len(result.cells) >= 1
