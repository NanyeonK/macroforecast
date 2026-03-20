"""Tests for pipeline/features.py — FeatureBuilder."""

import numpy as np
import pytest

from macrocast.pipeline.features import FeatureBuilder


@pytest.fixture()
def synthetic_data():
    """Return (X_panel, y) with T=100 observations and N=20 predictors."""
    rng = np.random.default_rng(0)
    T, N = 100, 20
    X = rng.standard_normal((T, N))
    y = rng.standard_normal(T)
    return X, y


class TestFeatureBuilderFactorsMode:
    def test_fit_transform_shape(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_factors=4, n_lags=3, use_factors=True)
        Z = builder.fit_transform(X, y)
        # Output rows: T - n_lags
        assert Z.shape == (100 - 3, 4 + 3)

    def test_n_features_property(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_factors=4, n_lags=3, use_factors=True)
        builder.fit(X, y)
        assert builder.n_features == 4 + 3

    def test_transform_test_row(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_factors=4, n_lags=3, use_factors=True)
        builder.fit(X, y)
        X_test = X[-1:, :]
        y_lags = y[-3:]
        Z_test = builder.transform(X_test, y_lags)
        assert Z_test.shape == (1 - 3 + 3, 4 + 3)  # 1 row after lag trimming

    def test_no_look_ahead_pca(self, synthetic_data):
        """Fitting on train split must NOT use test rows for PCA."""
        X, y = synthetic_data
        split = 80
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        builder = FeatureBuilder(n_factors=4, n_lags=2, use_factors=True)
        builder.fit(X_tr, y_tr)
        # Transform of test rows should not raise and use train-fitted PCA.
        # Pass full y_tr so there are enough values to build AR lags for all X_te rows.
        Z_te = builder.transform(X_te, y_tr)
        assert Z_te.shape[1] == builder.n_features


class TestFeatureBuilderAROnly:
    def test_ar_only_shape(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_lags=5, use_factors=False)
        Z = builder.fit_transform(X, y)
        assert Z.shape == (100 - 5, 5)

    def test_n_features_no_factors(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_lags=5, use_factors=False)
        builder.fit(X, y)
        assert builder.n_features == 5


class TestFeatureBuilderEdgeCases:
    def test_raises_before_fit(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder()
        with pytest.raises(RuntimeError):
            builder.transform(X, y)

    def test_n_factors_clamped_to_min_dim(self):
        """n_factors should be clamped when panel has fewer columns."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 3))  # only 3 columns
        y = rng.standard_normal(50)
        builder = FeatureBuilder(n_factors=10, n_lags=2, use_factors=True)
        Z = builder.fit_transform(X, y)
        # n_factors clamped to 3 (or less if T is limiting)
        assert Z.shape[1] <= 3 + 2


# ---------------------------------------------------------------------------
# CLSS 2021 additions
# ---------------------------------------------------------------------------


class TestMARXTransform:
    """Unit tests for FeatureBuilder._marx_transform static method."""

    def test_shape(self):
        """(T=10, K=3) with p_marx=4 → (7, 12)."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((10, 3))
        out = FeatureBuilder._marx_transform(X, 4)
        assert out.shape == (10 - 4 + 1, 3 * 4)

    def test_p_marx_1_equals_identity(self):
        """With p_marx=1 the MA_1 is just X itself (trimmed 0 rows)."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((8, 2))
        out = FeatureBuilder._marx_transform(X, 1)
        assert out.shape == (8, 2)
        np.testing.assert_allclose(out, X)

    def test_manual_computation(self):
        """Verify exact values against a 4×2 panel with p_marx=3."""
        # X = [[1,2],[3,4],[5,6],[7,8]]  (T=4, K=2)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        # p_marx=3 → T - p_marx + 1 = 2 rows; K*p_marx = 6 columns
        # Row 0 (t=2, 0-indexed): MA_1=[5,6], MA_2=mean([3,5],[4,6])=[4,5], MA_3=mean([1,3,5],[2,4,6])=[3,4]
        # Row 1 (t=3): MA_1=[7,8], MA_2=mean([5,7],[6,8])=[6,7], MA_3=mean([3,5,7],[4,6,8])=[5,6]
        out = FeatureBuilder._marx_transform(X, 3)
        assert out.shape == (2, 6)
        expected = np.array([
            [5.0, 6.0, 4.0, 5.0, 3.0, 4.0],
            [7.0, 8.0, 6.0, 7.0, 5.0, 6.0],
        ])
        np.testing.assert_allclose(out, expected)


class TestMARXMode:
    """Tests for FeatureBuilder with use_marx=True (without MAF)."""

    def test_ar_only_marx_shape(self):
        """use_marx=True, use_factors=False: MARX columns + AR lags."""
        rng = np.random.default_rng(4)
        T, K, p_marx, n_lags = 50, 5, 4, 2
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)
        builder = FeatureBuilder(
            n_lags=n_lags, use_factors=False, use_marx=True, p_marx=p_marx
        )
        Z = builder.fit_transform(X, y)
        # Effective rows: T - max(p_marx-1, n_lags) = 50 - max(3,2) = 47
        common_start = max(p_marx - 1, n_lags)
        assert Z.shape == (T - common_start, K * p_marx + n_lags)

    def test_factors_marx_shape(self):
        """use_marx=True, use_factors=True: PCA on MARX panel + AR lags."""
        rng = np.random.default_rng(5)
        T, K, n_factors, p_marx, n_lags = 60, 8, 3, 4, 2
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)
        builder = FeatureBuilder(
            n_factors=n_factors, n_lags=n_lags,
            use_factors=True, use_marx=True, p_marx=p_marx
        )
        Z = builder.fit_transform(X, y)
        common_start = max(p_marx - 1, n_lags)
        assert Z.shape == (T - common_start, n_factors + n_lags)

    def test_test_row_produces_one_row(self):
        """Test-row transform returns exactly 1 row for MARX mode."""
        rng = np.random.default_rng(6)
        T, K, p_marx, n_lags = 50, 5, 4, 3
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)
        builder = FeatureBuilder(
            n_lags=n_lags, use_factors=False, use_marx=True, p_marx=p_marx
        )
        builder.fit(X, y)
        X_test = X[-1:, :]
        Z_test = builder.transform(X_test, y[-n_lags:])
        assert Z_test.shape[0] == 1


class TestMAFMode:
    """Tests for FeatureBuilder with use_maf=True."""

    def test_maf_shape_matches_standard_factors(self):
        """MAF output has same (rows, n_factors + n_lags) shape as standard factors."""
        rng = np.random.default_rng(7)
        T, K, n_factors, p_marx, n_lags = 80, 10, 4, 4, 3
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)

        builder_maf = FeatureBuilder(
            n_factors=n_factors, n_lags=n_lags,
            use_maf=True, p_marx=p_marx
        )
        Z_maf = builder_maf.fit_transform(X, y)

        common_start = max(p_marx - 1, n_lags)
        assert Z_maf.shape == (T - common_start, n_factors + n_lags)

    def test_maf_output_differs_from_standard_factors(self):
        """MAF uses MARX-transformed X for PCA, so output values differ from standard."""
        rng = np.random.default_rng(8)
        T, K, n_factors, n_lags = 80, 10, 4, 3
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)

        builder_std = FeatureBuilder(
            n_factors=n_factors, n_lags=n_lags, use_factors=True, p_marx=3
        )
        builder_maf = FeatureBuilder(
            n_factors=n_factors, n_lags=n_lags, use_maf=True, p_marx=3
        )
        Z_std = builder_std.fit_transform(X, y)
        Z_maf = builder_maf.fit_transform(X, y)

        # Same shape (both have p_marx-1=2 < n_lags=3 → common_start=n_lags)
        assert Z_std.shape == Z_maf.shape
        # Values differ because PCA input is different
        assert not np.allclose(Z_std, Z_maf)

    def test_maf_forces_use_marx_and_factors(self):
        """use_maf=True should set use_marx=True and use_factors=True internally."""
        builder = FeatureBuilder(use_maf=True)
        assert builder.use_marx is True
        assert builder.use_factors is True


class TestLevelInclusion:
    """Tests for FeatureBuilder with include_levels=True."""

    def test_levels_increase_column_count(self):
        """Z column count increases by N_levels + 1 when include_levels=True."""
        rng = np.random.default_rng(9)
        T, K, N_levels, n_factors, n_lags = 60, 10, 5, 3, 2
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)
        X_levels = rng.standard_normal((T, N_levels))

        builder_base = FeatureBuilder(
            n_factors=n_factors, n_lags=n_lags, use_factors=True
        )
        Z_base = builder_base.fit_transform(X, y)

        builder_lev = FeatureBuilder(
            n_factors=n_factors, n_lags=n_lags,
            use_factors=True, include_levels=True
        )
        Z_lev = builder_lev.fit_transform(X, y, X_levels=X_levels)

        assert Z_lev.shape[0] == Z_base.shape[0]
        assert Z_lev.shape[1] == Z_base.shape[1] + N_levels + 1

    def test_levels_with_ar_only_mode(self):
        """include_levels also works without PCA (AR-only mode)."""
        rng = np.random.default_rng(10)
        T, K, N_levels, n_lags = 50, 8, 4, 3
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)
        X_levels = rng.standard_normal((T, N_levels))

        builder = FeatureBuilder(
            n_lags=n_lags, use_factors=False, include_levels=True
        )
        Z = builder.fit_transform(X, y, X_levels=X_levels)
        assert Z.shape[1] == n_lags + N_levels + 1

    def test_levels_test_row_shape(self):
        """Level inclusion works correctly for a single test row."""
        rng = np.random.default_rng(11)
        T, K, N_levels, n_lags = 50, 6, 3, 2
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)
        X_levels = rng.standard_normal((T, N_levels))

        builder = FeatureBuilder(
            n_lags=n_lags, use_factors=False, include_levels=True
        )
        builder.fit(X, y, X_levels=X_levels)
        X_test = X[-1:, :]
        X_levels_test = X_levels[-1:, :]
        Z_test = builder.transform(
            X_test, y[-n_lags:], X_levels=X_levels_test
        )
        assert Z_test.shape == (1, n_lags + N_levels + 1)


# ---------------------------------------------------------------------------
# include_raw_x and marx_for_pca additions
# ---------------------------------------------------------------------------


class TestIncludeRawX:
    def test_training_shape_includes_raw_x(self):
        """include_raw_x=True adds N raw X columns to Z."""
        T, N = 50, 10
        rng = np.random.default_rng(0)
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        # AR-only baseline
        builder_base = FeatureBuilder(use_factors=False, n_lags=2, include_raw_x=False)
        Z_base = builder_base.fit_transform(X, y)

        # With raw X appended
        builder_raw = FeatureBuilder(use_factors=False, n_lags=2, include_raw_x=True)
        Z_raw = builder_raw.fit_transform(X, y)

        # raw X adds N columns; row count must be identical
        assert Z_raw.shape[1] == Z_base.shape[1] + N
        assert Z_raw.shape[0] == Z_base.shape[0]

    def test_factors_with_raw_x(self):
        """use_factors=True + include_raw_x=True gives factors + raw_X + ar_lags."""
        T, N = 50, 10
        rng = np.random.default_rng(0)
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        n_factors, n_lags = 3, 2
        builder = FeatureBuilder(
            use_factors=True, n_factors=n_factors, n_lags=n_lags,
            include_raw_x=True,
        )
        Z = builder.fit_transform(X, y)
        # Should have n_factors + N + n_lags columns
        assert Z.shape[1] == n_factors + N + n_lags

    def test_test_row_raw_x_shape(self):
        """include_raw_x=True produces correct shape for a single test row."""
        T, N, n_lags = 50, 8, 3
        rng = np.random.default_rng(12)
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        builder = FeatureBuilder(use_factors=False, n_lags=n_lags, include_raw_x=True)
        builder.fit(X, y)
        X_test = X[-1:, :]
        Z_test = builder.transform(X_test, y[-n_lags:])
        # 1 row; columns = n_lags + N
        assert Z_test.shape == (1, n_lags + N)


class TestFeatureNames:
    """Tests for feature_names_out_ and feature_group_map_ properties."""

    def test_feature_names_empty_before_fit(self):
        """feature_names_out_ is empty before fit() is called."""
        builder = FeatureBuilder(n_factors=2, n_lags=2, use_factors=True)
        assert builder.feature_names_out_ == []
        assert builder.feature_group_map_ == {}

    def test_feature_names_basic(self):
        """use_factors=True, n_factors=2, n_lags=2: names match Z columns."""
        rng = np.random.default_rng(100)
        T, N = 80, 15
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        n_factors, n_lags = 2, 2
        builder = FeatureBuilder(n_factors=n_factors, n_lags=n_lags, use_factors=True)
        Z = builder.fit_transform(X, y)

        names = builder.feature_names_out_
        group_map = builder.feature_group_map_

        # Total count matches Z columns
        assert len(names) == Z.shape[1]

        # All names appear in the group map
        assert set(names) == set(group_map.keys())

        # Groups are drawn from the allowed set
        allowed_groups = {"ar", "factors", "marx", "x", "levels"}
        assert set(group_map.values()).issubset(allowed_groups)

        # Verify prefix structure: n_factors factor columns, then n_lags ar columns
        n_factors_actual = builder._pca.n_components_
        expected_factor_names = [f"factor_{i+1}" for i in range(n_factors_actual)]
        expected_ar_names = [f"y_lag_{i+1}" for i in range(n_lags)]
        assert names == expected_factor_names + expected_ar_names

        # Group tags
        for name in expected_factor_names:
            assert group_map[name] == "factors"
        for name in expected_ar_names:
            assert group_map[name] == "ar"

    def test_feature_names_ar_only(self):
        """use_factors=False: only AR lags, group='ar'."""
        rng = np.random.default_rng(101)
        T, N = 60, 10
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        n_lags = 3
        builder = FeatureBuilder(n_lags=n_lags, use_factors=False)
        Z = builder.fit_transform(X, y)

        names = builder.feature_names_out_
        group_map = builder.feature_group_map_

        assert len(names) == Z.shape[1]
        assert names == [f"y_lag_{i+1}" for i in range(n_lags)]
        assert all(group_map[n] == "ar" for n in names)

    def test_feature_names_marx(self):
        """use_marx=True, use_factors=False: MARX_* names, group='marx'."""
        rng = np.random.default_rng(102)
        T, K = 50, 5
        p_marx, n_lags = 4, 2
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)

        builder = FeatureBuilder(
            n_lags=n_lags, use_factors=False, use_marx=True,
            p_marx=p_marx,
        )
        Z = builder.fit_transform(X, y)

        names = builder.feature_names_out_
        group_map = builder.feature_group_map_

        assert len(names) == Z.shape[1]

        # MARX columns come first, then AR lags
        n_marx_cols = K * p_marx
        marx_names = [f"MARX_{i}" for i in range(n_marx_cols)]
        ar_names = [f"y_lag_{i+1}" for i in range(n_lags)]
        assert names == marx_names + ar_names

        for name in marx_names:
            assert group_map[name] == "marx"
        for name in ar_names:
            assert group_map[name] == "ar"

    def test_feature_names_include_raw_x(self):
        """include_raw_x=True: X_* names present, group='x'."""
        rng = np.random.default_rng(103)
        T, N = 60, 8
        n_lags = 2
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        builder = FeatureBuilder(use_factors=False, n_lags=n_lags, include_raw_x=True)
        Z = builder.fit_transform(X, y)

        names = builder.feature_names_out_
        group_map = builder.feature_group_map_

        assert len(names) == Z.shape[1]

        # AR lags first, then raw X columns
        ar_names = [f"y_lag_{i+1}" for i in range(n_lags)]
        x_names = [f"X_{i}" for i in range(N)]
        assert names == ar_names + x_names

        for name in x_names:
            assert group_map[name] == "x"

    def test_feature_names_include_levels(self):
        """include_levels=True: level_y and level_H columns appended."""
        rng = np.random.default_rng(104)
        T, N, N_levels = 60, 10, 4
        n_lags = 2
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)
        X_levels = rng.standard_normal((T, N_levels))

        builder = FeatureBuilder(use_factors=False, n_lags=n_lags, include_levels=True)
        Z = builder.fit_transform(X, y, X_levels=X_levels)

        names = builder.feature_names_out_
        group_map = builder.feature_group_map_

        assert len(names) == Z.shape[1]

        # Ends with level_y (y level) and level_{i} columns (one per level predictor,
        # 0-indexed)
        assert names[-1] == "level_y"
        level_names = [n for n in names if n.startswith("level_")]
        assert all(group_map[n] == "levels" for n in level_names)

    def test_feature_names_maf_mode(self):
        """MAF mode (use_maf=True): factor columns tagged as 'factors'."""
        rng = np.random.default_rng(105)
        T, K = 80, 10
        n_factors, n_lags, p_marx = 3, 2, 4
        X = rng.standard_normal((T, K))
        y = rng.standard_normal(T)

        builder = FeatureBuilder(
            use_maf=True, n_factors=n_factors, n_lags=n_lags, p_marx=p_marx
        )
        Z = builder.fit_transform(X, y)

        names = builder.feature_names_out_
        group_map = builder.feature_group_map_

        assert len(names) == Z.shape[1]

        n_factors_actual = builder._pca.n_components_
        factor_names = [f"MAF_factor_{i+1}" for i in range(n_factors_actual)]
        ar_names = [f"y_lag_{i+1}" for i in range(n_lags)]
        assert names == factor_names + ar_names

        for name in factor_names:
            assert group_map[name] == "factors"

    def test_feature_names_marx_for_pca_false(self):
        """use_factors=True, use_marx=True, marx_for_pca=False: factors + MARX + AR."""
        rng = np.random.default_rng(106)
        T, N = 80, 8
        n_factors, n_lags, p_marx = 3, 2, 4
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        builder = FeatureBuilder(
            use_factors=True, n_factors=n_factors, n_lags=n_lags,
            use_marx=True, p_marx=p_marx, marx_for_pca=False,
        )
        Z = builder.fit_transform(X, y)

        names = builder.feature_names_out_
        group_map = builder.feature_group_map_

        assert len(names) == Z.shape[1]

        n_factors_actual = builder._pca.n_components_
        n_marx_cols = N * p_marx
        factor_names = [f"factor_{i+1}" for i in range(n_factors_actual)]
        marx_names = [f"MARX_{i}" for i in range(n_marx_cols)]
        ar_names = [f"y_lag_{i+1}" for i in range(n_lags)]
        assert names == factor_names + marx_names + ar_names

        for name in factor_names:
            assert group_map[name] == "factors"
        for name in marx_names:
            assert group_map[name] == "marx"
        for name in ar_names:
            assert group_map[name] == "ar"

    def test_feature_names_returns_copy(self):
        """Mutating returned lists/dicts does not affect internal state."""
        rng = np.random.default_rng(107)
        X = rng.standard_normal((60, 10))
        y = rng.standard_normal(60)
        builder = FeatureBuilder(n_factors=2, n_lags=2, use_factors=True)
        builder.fit(X, y)

        names = builder.feature_names_out_
        names.append("injected")
        assert "injected" not in builder.feature_names_out_

        gmap = builder.feature_group_map_
        gmap["injected"] = "bad"
        assert "injected" not in builder.feature_group_map_


class TestMarxForPca:
    def test_marx_for_pca_false_different_from_maf(self):
        """marx_for_pca=False gives different Z than MAF (marx_for_pca=True)."""
        T, N, p_marx = 80, 8, 4
        rng = np.random.default_rng(1)
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        # MAF: PCA on MARX panel
        builder_maf = FeatureBuilder(
            use_factors=True, n_factors=3, n_lags=2,
            use_marx=True, p_marx=p_marx, marx_for_pca=True,
        )
        Z_maf = builder_maf.fit_transform(X, y)

        # PCA on raw X, MARX columns appended separately
        builder_fx = FeatureBuilder(
            use_factors=True, n_factors=3, n_lags=2,
            use_marx=True, p_marx=p_marx, marx_for_pca=False,
        )
        Z_fx = builder_fx.fit_transform(X, y)

        # MAF has n_factors + n_lags columns; marx_for_pca=False adds N*p_marx extra
        assert Z_maf.shape[1] < Z_fx.shape[1]
        assert Z_fx.shape[1] == 3 + N * p_marx + 2

    def test_marx_for_pca_false_test_row(self):
        """marx_for_pca=False test-row produces correct column count."""
        T, N, p_marx, n_factors, n_lags = 80, 8, 4, 3, 2
        rng = np.random.default_rng(13)
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        builder = FeatureBuilder(
            use_factors=True, n_factors=n_factors, n_lags=n_lags,
            use_marx=True, p_marx=p_marx, marx_for_pca=False,
        )
        builder.fit(X, y)
        X_test = X[-1:, :]
        Z_test = builder.transform(X_test, y[-n_lags:])
        # 1 row; columns = n_factors + N*p_marx + n_lags
        assert Z_test.shape == (1, n_factors + N * p_marx + n_lags)

    def test_marx_for_pca_true_is_default_maf_behavior(self):
        """marx_for_pca=True (default) is identical to explicit use_maf=True."""
        T, N, p_marx, n_factors, n_lags = 80, 8, 4, 3, 2
        rng = np.random.default_rng(14)
        X = rng.standard_normal((T, N))
        y = rng.standard_normal(T)

        builder_maf = FeatureBuilder(
            use_maf=True, n_factors=n_factors, n_lags=n_lags, p_marx=p_marx,
        )
        builder_explicit = FeatureBuilder(
            use_factors=True, n_factors=n_factors, n_lags=n_lags,
            use_marx=True, p_marx=p_marx, marx_for_pca=True,
        )
        # Both builders operate identically: same shapes
        Z_maf = builder_maf.fit_transform(X, y)
        Z_explicit = builder_explicit.fit_transform(X, y)
        assert Z_maf.shape == Z_explicit.shape
