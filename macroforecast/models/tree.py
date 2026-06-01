from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator, optional_import


def decision_tree(
    X: Any,
    y: Any | None = None,
    *,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit a CART regression tree."""

    from sklearn.tree import DecisionTreeRegressor

    # Backend wrapper only: sklearn owns CART fitting; macroforecast owns the
    # callable X/y contract, metadata, diagnostics, and persistence.
    params = {
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "implementation_note": (
            "Thin sklearn.tree.DecisionTreeRegressor wrapper; macroforecast owns "
            "the pandas X/y contract, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    return fit_estimator(
        DecisionTreeRegressor(**estimator_params),
        X,
        y,
        model="decision_tree",
        metadata=params,
    )


def random_forest(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    n_jobs: int | None = 1,
    **kwargs: Any,
) -> ModelFit:
    """Fit a random forest regressor."""

    from sklearn.ensemble import RandomForestRegressor

    # Backend wrapper only: sklearn owns the forest algorithm; macroforecast
    # passes through tree parameters and records the fit metadata.
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "n_jobs": n_jobs,
        "implementation_note": (
            "Thin sklearn.ensemble.RandomForestRegressor wrapper; macroforecast "
            "owns the pandas X/y contract, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    return fit_estimator(
        RandomForestRegressor(**estimator_params),
        X,
        y,
        model="random_forest",
        metadata=params,
    )


def extra_trees(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    n_jobs: int | None = 1,
    **kwargs: Any,
) -> ModelFit:
    """Fit an extremely randomized trees regressor."""

    from sklearn.ensemble import ExtraTreesRegressor

    # Backend wrapper only: sklearn owns ExtraTrees; no package-native tree
    # logic is implemented here.
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "n_jobs": n_jobs,
        "implementation_note": (
            "Thin sklearn.ensemble.ExtraTreesRegressor wrapper; macroforecast owns "
            "the pandas X/y contract, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    return fit_estimator(
        ExtraTreesRegressor(**estimator_params),
        X,
        y,
        model="extra_trees",
        metadata=params,
    )


def gradient_boosting(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit sklearn gradient-boosted regression trees."""

    from sklearn.ensemble import GradientBoostingRegressor

    # Backend wrapper only: sklearn owns stage-wise tree boosting; fit-time
    # boosting ensembles such as Booging live in macroforecast.model_ensemble.
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
        "implementation_note": (
            "Thin sklearn.ensemble.GradientBoostingRegressor wrapper; fit-time "
            "boosting ensembles such as Booging live in macroforecast.model_ensemble."
        ),
        **kwargs,
    }
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    return fit_estimator(
        GradientBoostingRegressor(**estimator_params),
        X,
        y,
        model="gradient_boosting",
        metadata=params,
    )


def xgboost(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 1.0,
    random_state: int = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit an XGBoost regressor. Requires the `xgboost` extra."""

    xgb = optional_import("xgboost", extra="xgboost")
    # Optional backend wrapper: xgboost owns the estimator; macroforecast only
    # normalizes parameter names and lazy dependency errors.
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "random_state": int(random_state),
        "objective": "reg:squarederror",
        "implementation_note": (
            "Thin xgboost.XGBRegressor wrapper; macroforecast owns lazy extra "
            "loading, pandas X/y IO, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    estimator = xgb.XGBRegressor(
        **estimator_params,
    )
    return fit_estimator(estimator, X, y, model="xgboost", metadata=params)


def lightgbm(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    num_leaves: int = 31,
    random_state: int = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit a LightGBM regressor. Requires the `lightgbm` extra."""

    lgb = optional_import("lightgbm", extra="lightgbm")
    # Optional backend wrapper: LightGBM owns the estimator; macroforecast only
    # normalizes parameter names and lazy dependency errors.
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "random_state": int(random_state),
        "implementation_note": (
            "Thin lightgbm.LGBMRegressor wrapper; macroforecast owns lazy extra "
            "loading, pandas X/y IO, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    estimator = lgb.LGBMRegressor(
        **estimator_params,
    )
    return fit_estimator(estimator, X, y, model="lightgbm", metadata=params)


def catboost(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 0,
    verbose: bool = False,
    **kwargs: Any,
) -> ModelFit:
    """Fit a CatBoost regressor. Requires the `catboost` extra."""

    cb = optional_import("catboost", extra="catboost")
    # Optional backend wrapper: CatBoost owns the estimator; macroforecast maps
    # public names such as n_estimators/max_depth onto CatBoost's API.
    params = {
        "iterations": int(n_estimators),
        "learning_rate": float(learning_rate),
        "depth": int(max_depth),
        "random_seed": int(random_state),
        "verbose": verbose,
        "implementation_note": (
            "Thin catboost.CatBoostRegressor wrapper; macroforecast maps public "
            "n_estimators/max_depth names to CatBoost iterations/depth."
        ),
        **kwargs,
    }
    metadata = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
        "verbose": verbose,
        "implementation_note": params["implementation_note"],
        **kwargs,
    }
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    estimator = cb.CatBoostRegressor(
        **estimator_params,
    )
    return fit_estimator(estimator, X, y, model="catboost", metadata=metadata)


def _validate_quantile_levels(levels: Sequence[float]) -> tuple[float, ...]:
    quantiles = tuple(float(level) for level in levels)
    if not quantiles or any(level <= 0.0 or level >= 1.0 for level in quantiles):
        raise ValueError("quantile levels must contain values in (0, 1)")
    return quantiles


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(sorted_weights.sum())
    if total <= 0.0:
        return float(sorted_values[-1])
    cumulative = np.cumsum(sorted_weights) / total
    index = min(
        int(np.searchsorted(cumulative, float(quantile), side="left")),
        len(sorted_values) - 1,
    )
    return float(sorted_values[index])


class QuantileRegressionForestRegressor:
    """Random-forest point forecasts plus empirical leaf quantiles."""

    # Source comparison:
    # - Meinshausen/quantregForest-style QRF estimates a conditional response
    #   distribution by assigning training responses forest weights from shared
    #   terminal leaves. R quantregForest stores in-leaf responses directly;
    #   ranger/grf expose related quantile forest APIs with different forest
    #   builders and inference surfaces.
    # - This class follows the QRF leaf-weight skeleton on top of sklearn's
    #   RandomForestRegressor. For a test row, each tree contributes every
    #   training target in the matching leaf with inverse leaf-size weight.
    #   The common 1 / n_estimators multiplier cancels in weighted quantiles.
    # - This is not a full quantregForest/ranger/grf clone: forest fitting,
    #   splitting, OOB behavior, uncertainty tooling, and optional honesty are
    #   inherited from or absent in sklearn. macroforecast owns the callable
    #   X/y contract, leaf target store, quantile extraction, and metadata.

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 0,
        quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> None:
        self.n_estimators = max(1, int(n_estimators))
        self.max_depth = max_depth
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.random_state = int(random_state)
        self.quantile_levels = _validate_quantile_levels(quantile_levels)
        self._forest: Any = None
        self._leaf_targets: list[dict[int, np.ndarray]] = []
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileRegressionForestRegressor":
        from sklearn.ensemble import RandomForestRegressor

        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        frame = X.fillna(0.0)
        target = np.asarray(y, dtype=float)
        self._fallback = float(np.mean(target)) if target.size else 0.0
        self._forest.fit(frame, target)
        leaves = self._forest.apply(frame)
        self._leaf_targets = []
        for tree_idx in range(leaves.shape[1]):
            tree_leaves = leaves[:, tree_idx]
            self._leaf_targets.append({
                int(leaf): target[tree_leaves == leaf] for leaf in np.unique(tree_leaves)
            })
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._forest is None:
            return np.zeros(len(X), dtype=float)
        return np.asarray(self._forest.predict(X.fillna(0.0)), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        if self._forest is None:
            return None
        return np.asarray(self._forest.feature_importances_, dtype=float)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        levels: tuple[float, ...] | None = None,
    ) -> dict[float, np.ndarray]:
        levels = self.quantile_levels if levels is None else _validate_quantile_levels(levels)
        if self._forest is None or not self._leaf_targets:
            return {q: np.full(len(X), self._fallback, dtype=float) for q in levels}
        leaves = self._forest.apply(X.fillna(0.0))
        out: dict[float, np.ndarray] = {q: np.empty(len(X), dtype=float) for q in levels}
        for i in range(len(X)):
            samples: list[float] = []
            weights: list[float] = []
            for tree_idx in range(leaves.shape[1]):
                values = self._leaf_targets[tree_idx].get(int(leaves[i, tree_idx]), np.array([]))
                leaf_values = np.asarray(values, dtype=float)
                if leaf_values.size == 0:
                    continue
                samples.extend(leaf_values.tolist())
                # QRF forest weights: within each matching terminal leaf, every
                # training response receives equal mass, scaled by leaf size.
                weights.extend([1.0 / float(leaf_values.size)] * int(leaf_values.size))
            arr = np.asarray(samples if samples else [self._fallback], dtype=float)
            weight_arr = np.asarray(weights if weights else [1.0], dtype=float)
            for q in levels:
                out[q][i] = float(_weighted_quantile(arr, weight_arr, q))
        return out


def quantile_regression_forest(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> ModelFit:
    """Fit a quantile regression forest."""

    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "quantile_levels": tuple(float(q) for q in quantile_levels),
        "implementation_note": (
            "Meinshausen/quantregForest-style empirical leaf-weight quantiles "
            "over sklearn RandomForestRegressor terminal leaves."
        ),
    }
    return fit_estimator(
        QuantileRegressionForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            quantile_levels=tuple(float(q) for q in quantile_levels),
        ),
        X,
        y,
        model="quantile_regression_forest",
        metadata=params,
    )


class MacroRandomForestRegressor:
    """Adapter for the vendored MacroRandomForest reference implementation."""

    # Upstream comparison, checked against:
    # - Python MacroRandomForest 1.0.6 (RyanLucas3/MacroRandomForest, MIT).
    # - R prototype philgoucou/macrorf::MRF (GPL-3).
    #
    # Both upstream APIs expect one numeric matrix/data frame that already
    # contains the target, local-linear variables, state/split variables, and
    # out-of-sample rows. R exposes 1-based `y.pos`, `x.pos`, `S.pos`,
    # `oos.pos`; Python exposes `y_pos`, `x_pos`, `S_pos`, `oos_pos` and uses
    # the same data-matrix contract. Both implementations then translate names
    # internally: user `x_pos` becomes `z_pos` for local-linear coefficient
    # variables, and user `S_pos` becomes the split/state variable set.
    #
    # macroforecast intentionally does not reimplement the MRF tree, ridge, or
    # random-walk-regularization math. It vendors the Python reference backend
    # and adapts only the package-facing contract: callable `X, y` inputs,
    # optional column-name selection, metadata, output normalization, cache, and
    # contextual error messages.
    #
    # Main differences from upstream:
    # - `fit(X, y)` stores aligned pandas data; the expensive reference
    #   `_ensemble_loop()` is run lazily from `predict(X_test)`.
    # - `predict()` builds the upstream single matrix by prepending the stored
    #   target and adding dummy zero targets for OOS rows, then derives
    #   `oos_pos` from the train/test boundary.
    # - `x_columns` and `S_columns` are mapped to 1..k feature positions because
    #   the constructed backend matrix always has the target at column 0.
    # - The separated `X, y` API fixes the target location at column 0; a
    #   nonzero upstream-style `y_pos` would point at a feature after matrix
    #   construction, so it is rejected below instead of silently changing the
    #   dependent variable.
    # - R's `pred.given.mrf()` keep-forest path is not called here. The adapter
    #   follows the direct Python/R MRF path by rebuilding the reference object
    #   for each distinct test matrix, while caching repeated predictions.
    def __init__(
        self,
        *,
        x_columns: Sequence[str] | None = None,
        S_columns: Sequence[str] | None = None,
        x_pos: Sequence[int] | None = None,
        S_pos: Sequence[int] | None = None,
        y_pos: int = 0,
        B: int = 50,
        minsize: int = 10,
        mtry_frac: float = 1.0 / 3.0,
        min_leaf_frac_of_x: float = 1.0,
        VI: bool = False,
        ERT: bool = False,
        quantile_rate: float | None = None,
        S_priority_vec: Sequence[float] | None = None,
        random_x: bool = False,
        trend_push: int = 1,
        howmany_random_x: int = 1,
        howmany_keep_best_VI: int = 20,
        cheap_look_at_GTVPs: bool = True,
        prior_var: Sequence[float] | None = None,
        prior_mean: Sequence[float] | None = None,
        subsampling_rate: float = 0.75,
        rw_regul: float = 0.75,
        keep_forest: bool = False,
        block_size: int = 12,
        fast_rw: bool = True,
        ridge_lambda: float = 0.1,
        HRW: int = 0,
        resampling_opt: int = 2,
        print_b: bool = False,
        parallelise: bool = False,
        n_cores: int = 1,
        **kwargs: Any,
    ) -> None:
        if int(y_pos) != 0:
            raise ValueError(
                "macro_random_forest receives target values through y, so y_pos "
                "must be 0. Use x_columns/S_columns or x_pos/S_pos to select "
                "feature roles."
            )
        if x_columns is not None and x_pos is not None:
            raise ValueError("Use either x_columns or x_pos, not both.")
        if S_columns is not None and S_pos is not None:
            raise ValueError("Use either S_columns or S_pos, not both.")
        self.x_columns = None if x_columns is None else tuple(str(column) for column in x_columns)
        self.S_columns = None if S_columns is None else tuple(str(column) for column in S_columns)
        self.x_pos = None if x_pos is None else tuple(int(pos) for pos in x_pos)
        self.S_pos = None if S_pos is None else tuple(int(pos) for pos in S_pos)
        self.y_pos = int(y_pos)
        self.params: dict[str, Any] = {
            "B": int(B),
            "minsize": int(minsize),
            "mtry_frac": float(mtry_frac),
            "min_leaf_frac_of_x": float(min_leaf_frac_of_x),
            "VI": bool(VI),
            "ERT": bool(ERT),
            "quantile_rate": quantile_rate,
            "S_priority_vec": None if S_priority_vec is None else list(S_priority_vec),
            "random_x": bool(random_x),
            "trend_push": int(trend_push),
            "howmany_random_x": int(howmany_random_x),
            "howmany_keep_best_VI": int(howmany_keep_best_VI),
            "cheap_look_at_GTVPs": bool(cheap_look_at_GTVPs),
            "prior_var": [] if prior_var is None else list(prior_var),
            "prior_mean": [] if prior_mean is None else list(prior_mean),
            "subsampling_rate": float(subsampling_rate),
            "rw_regul": float(rw_regul),
            "keep_forest": bool(keep_forest),
            "block_size": int(block_size),
            "fast_rw": bool(fast_rw),
            "ridge_lambda": float(ridge_lambda),
            "HRW": int(HRW),
            "resampling_opt": int(resampling_opt),
            "print_b": bool(print_b),
            "parallelise": bool(parallelise),
            "n_cores": int(n_cores),
            **kwargs,
        }
        self._train_X: pd.DataFrame | None = None
        self._train_y: pd.Series | None = None
        self._feature_names: tuple[str, ...] = ()
        self.output_: dict[str, Any] | None = None
        self.model_: Any = None
        self._prediction_cache_key: tuple[Any, ...] | None = None
        self._prediction_cache_values: np.ndarray | None = None

    @staticmethod
    def _import_external():
        optional_import("joblib", extra="macro_random_forest")
        optional_import("matplotlib", extra="macro_random_forest")
        from macroforecast.models._mrf_reference import MacroRandomForest

        return MacroRandomForest

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MacroRandomForestRegressor":
        self._import_external()
        self._train_X = X.copy()
        self._train_y = y.copy()
        self._feature_names = tuple(str(column) for column in X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._train_X is None or self._train_y is None:
            return np.zeros(len(X), dtype=float)
        cache_key = self._cache_key(X)
        if (
            self._prediction_cache_key == cache_key
            and self._prediction_cache_values is not None
        ):
            return self._prediction_cache_values.copy()
        MacroRandomForest = self._import_external()
        train_X = self._train_X.copy()
        test_X = X.reindex(columns=list(self._feature_names), fill_value=0.0)
        train_y = self._train_y.rename("__target__")
        test_y = pd.Series(0.0, index=test_X.index, name="__target__")
        data = pd.concat(
            [
                pd.concat([train_y, train_X], axis=1),
                pd.concat([test_y, test_X], axis=1),
            ],
            axis=0,
        ).reset_index(drop=True)
        oos_pos = np.arange(len(train_X), len(train_X) + len(test_X))
        x_pos = self._resolve_positions(self.x_columns, self.x_pos, train_X.columns)
        S_pos = self._resolve_positions(self.S_columns, self.S_pos, train_X.columns)
        with self._reference_output_context():
            model = MacroRandomForest(
                data=data,
                y_pos=self.y_pos,
                x_pos=np.asarray(x_pos, dtype=int),
                S_pos=np.asarray(S_pos, dtype=int),
                oos_pos=oos_pos,
                **self.params,
            )
        self.model_ = model
        try:
            with self._reference_output_context(), warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=PendingDeprecationWarning,
                    module=r"macroforecast\.models\._mrf_reference",
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"invalid value encountered in divide",
                    category=RuntimeWarning,
                    module=r"macroforecast\.models\._mrf_reference",
                )
                self.output_ = model._ensemble_loop()
        except Exception as exc:  # noqa: BLE001 - external backend errors need package context.
            raise RuntimeError(
                "MacroRandomForest backend failed while running _ensemble_loop(). "
                "Check x_columns/S_columns and sample size."
            ) from exc
        values = self._prediction_values(self.output_, len(test_X))
        self._prediction_cache_key = cache_key
        self._prediction_cache_values = values.copy()
        return values

    def _reference_output_context(self):
        if self.params.get("print_b"):
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    @staticmethod
    def _resolve_positions(
        columns: Sequence[str] | None,
        positions: Sequence[int] | None,
        feature_index: pd.Index,
    ) -> list[int]:
        if positions is not None:
            resolved = [int(pos) for pos in positions]
            invalid = [
                pos for pos in resolved if pos < 1 or pos > int(len(feature_index))
            ]
            if invalid:
                raise ValueError(
                    "macro_random_forest positions must refer to feature columns "
                    f"1..{len(feature_index)}; got {invalid}"
                )
            return resolved
        if columns is None:
            return list(range(1, len(feature_index) + 1))
        missing = [column for column in columns if column not in feature_index]
        if missing:
            raise ValueError(f"macro_random_forest columns not found in X: {missing}")
        return [int(feature_index.get_loc(column)) + 1 for column in columns]

    @staticmethod
    def _prediction_values(output: dict[str, Any], n: int) -> np.ndarray:
        # Python backend output uses `pred_ensemble`/`S_names`; the R prototype
        # reports analogous fields as `pred.ensemble`/`S.names`. Since this
        # adapter calls the Python backend, only Python keys are accepted here.
        if n == 0:
            return np.array([], dtype=float)
        values = output.get("pred_ensemble")
        if values is None:
            values = output.get("pred")
        if values is None:
            raise RuntimeError(
                "MacroRandomForest backend did not return 'pred_ensemble' or 'pred'."
            )
        if isinstance(values, (pd.Series, pd.DataFrame)):
            arr = values.to_numpy(dtype=float)
        else:
            arr = np.asarray(values, dtype=float)
        if arr.ndim == 2:
            if arr.shape[0] == n:
                arr = arr.reshape(-1) if arr.shape[1] == 1 else arr.mean(axis=1)
            elif arr.shape[1] == n:
                arr = arr.mean(axis=0)
            else:
                arr = arr.reshape(-1)
        out = np.asarray(arr, dtype=float).reshape(-1)
        if len(out) < n:
            raise RuntimeError(
                "MacroRandomForest backend returned fewer predictions than requested."
            )
        return out[-n:]

    @staticmethod
    def _cache_key(X: pd.DataFrame) -> tuple[Any, ...]:
        try:
            hashed = pd.util.hash_pandas_object(X, index=True).to_numpy(dtype=np.uint64)
            value_hash = int(np.bitwise_xor.reduce(hashed)) if len(hashed) else 0
        except Exception:  # noqa: BLE001 - cache keys must never block prediction.
            value_hash = id(X)
        return (tuple(X.index), tuple(str(column) for column in X.columns), X.shape, value_hash)


def macro_random_forest(
    X: Any,
    y: Any | None = None,
    *,
    x_columns: Sequence[str] | None = None,
    S_columns: Sequence[str] | None = None,
    x_pos: Sequence[int] | None = None,
    S_pos: Sequence[int] | None = None,
    y_pos: int = 0,
    B: int = 50,
    minsize: int = 10,
    mtry_frac: float = 1.0 / 3.0,
    min_leaf_frac_of_x: float = 1.0,
    VI: bool = False,
    ERT: bool = False,
    quantile_rate: float | None = None,
    S_priority_vec: Sequence[float] | None = None,
    random_x: bool = False,
    trend_push: int = 1,
    howmany_random_x: int = 1,
    howmany_keep_best_VI: int = 20,
    cheap_look_at_GTVPs: bool = True,
    prior_var: Sequence[float] | None = None,
    prior_mean: Sequence[float] | None = None,
    subsampling_rate: float = 0.75,
    rw_regul: float = 0.75,
    keep_forest: bool = False,
    block_size: int = 12,
    fast_rw: bool = True,
    ridge_lambda: float = 0.1,
    HRW: int = 0,
    resampling_opt: int = 2,
    print_b: bool = False,
    parallelise: bool = False,
    n_cores: int = 1,
    **kwargs: Any,
) -> ModelFit:
    """Fit Macroeconomic Random Forest with the vendored reference backend."""

    params = {
        "x_columns": x_columns,
        "S_columns": S_columns,
        "x_pos": x_pos,
        "S_pos": S_pos,
        "y_pos": int(y_pos),
        "B": int(B),
        "minsize": int(minsize),
        "mtry_frac": float(mtry_frac),
        "min_leaf_frac_of_x": float(min_leaf_frac_of_x),
        "VI": bool(VI),
        "ERT": bool(ERT),
        "quantile_rate": quantile_rate,
        "S_priority_vec": S_priority_vec,
        "random_x": bool(random_x),
        "trend_push": int(trend_push),
        "howmany_random_x": int(howmany_random_x),
        "howmany_keep_best_VI": int(howmany_keep_best_VI),
        "cheap_look_at_GTVPs": bool(cheap_look_at_GTVPs),
        "prior_var": prior_var,
        "prior_mean": prior_mean,
        "subsampling_rate": float(subsampling_rate),
        "rw_regul": float(rw_regul),
        "keep_forest": bool(keep_forest),
        "block_size": int(block_size),
        "fast_rw": bool(fast_rw),
        "ridge_lambda": float(ridge_lambda),
        "HRW": int(HRW),
        "resampling_opt": int(resampling_opt),
        "print_b": bool(print_b),
        "parallelise": bool(parallelise),
        "n_cores": int(n_cores),
        **kwargs,
    }
    estimator = MacroRandomForestRegressor(**params)
    metadata = {
        "x_columns": estimator.x_columns,
        "S_columns": estimator.S_columns,
        "x_pos": estimator.x_pos,
        "S_pos": estimator.S_pos,
        "y_pos": estimator.y_pos,
        **estimator.params,
    }
    return fit_estimator(
        estimator,
        X,
        y,
        model="macro_random_forest",
        metadata=metadata,
        collect_diagnostics=False,
    )


__all__ = [
    "MacroRandomForestRegressor",
    "QuantileRegressionForestRegressor",
    "catboost",
    "decision_tree",
    "extra_trees",
    "gradient_boosting",
    "lightgbm",
    "macro_random_forest",
    "quantile_regression_forest",
    "random_forest",
    "xgboost",
]
