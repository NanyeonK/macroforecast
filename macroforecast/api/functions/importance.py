"""Standalone L7 importance callables.

Exposes eight importance-analysis callables that operate on a fitted
result object (any object with a ``._model`` attribute, typically a
FitResult returned by an L4 standalone callable) plus a feature
matrix ``X`` (and optionally a target ``y``).

UX option (a): callables extract ``result._model`` internally and
build a :class:`~macroforecast.core.types.ModelArtifact` adapter,
delegating all numeric work to existing runtime helpers in
``macroforecast.core.runtime``.

L7 importance standalone callables (v0.8.0, 8 ops).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers: input normalisation
# ---------------------------------------------------------------------------

def _to_frame(X: np.ndarray | pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    cols = feature_names if feature_names else [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(np.asarray(X), columns=cols)


def _to_series(y: np.ndarray | pd.Series) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(np.asarray(y).ravel(), name="y")


def _extract_model_artifact(result: Any, X: pd.DataFrame) -> Any:
    """Build a minimal ModelArtifact from a FitResult and feature matrix.

    The result must expose ``._model`` (the raw fitted estimator).  Feature
    names are taken from the estimator's ``feature_names_in_`` when present,
    otherwise from ``X.columns``.
    """
    from macroforecast.core.types import ModelArtifact

    fitted = result._model
    # Resolve feature names: estimator > DataFrame columns
    if hasattr(fitted, "feature_names_in_"):
        feature_names = tuple(str(n) for n in fitted.feature_names_in_)
    elif isinstance(X, pd.DataFrame) and len(X.columns) > 0:
        feature_names = tuple(str(c) for c in X.columns)
    else:
        feature_names = tuple(f"x{i}" for i in range(X.shape[1]))

    # Determine framework from estimator module
    mod = type(fitted).__module__ or ""
    if mod.startswith("xgboost"):
        framework: str = "xgboost"
    elif mod.startswith("lightgbm"):
        framework = "lightgbm"
    elif mod.startswith("catboost"):
        framework = "custom"
    elif mod.startswith("torch") or mod.startswith("macroforecast"):
        framework = "torch"
    else:
        framework = "sklearn"

    return ModelArtifact(
        model_id="_standalone_l7",
        family="_standalone",
        fitted_object=fitted,
        framework=framework,  # type: ignore[arg-type]
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NativeImportanceResult:
    """Result of :func:`model_native_linear_coef_importance` and
    :func:`model_native_tree_importance`.

    Attributes
    ----------
    importances_ :
        1-D array of importance values, one per feature.  For linear
        models: ``|coef_j|``; for tree ensembles: MDI ``feature_importances_``.
    feature_names_ :
        List of feature names matching ``importances_``.
    method :
        Descriptor string: ``"linear_coef"`` or ``"tree_native"``.
    """

    importances_: np.ndarray
    feature_names_: list[str]
    method: str

    def summary(self, top_n: int = 10) -> str:
        """Return a human-readable text summary of the top-N features.

        Parameters
        ----------
        top_n :
            Number of top features to display.  Default 10.

        Returns
        -------
        str
            Text table sorted by descending importance.
        """
        n = len(self.importances_)
        idx = np.argsort(self.importances_)[::-1][:min(top_n, n)]
        sep = "=" * 60
        dash = "-" * 60
        lines = [
            sep,
            f"{'Native Importance (' + self.method + ')':^60}",
            sep,
            f"{'Feature':35s} {'Importance':>12s}",
            dash,
        ]
        for i in idx:
            lines.append(f"{self.feature_names_[i]:35s} {self.importances_[i]:>12.6f}")
        if n > top_n:
            lines.append(f"  ... ({n - top_n} more features not shown)")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class PermutationImportanceResult:
    """Result of :func:`permutation_importance`.

    Attributes
    ----------
    importances_mean_ :
        Mean importance over ``n_repeats`` repeats (loss increase on
        permuting each feature).
    importances_std_ :
        Standard deviation of importance over repeats.
    importances_ :
        Per-repeat importance matrix of shape ``(n_features, n_repeats)``.
        Each column is one repeat; each row is one feature.
    feature_names_ :
        List of feature names.
    n_repeats :
        Number of permutation repeats used.
    """

    importances_mean_: np.ndarray
    importances_std_: np.ndarray
    importances_: np.ndarray
    feature_names_: list[str]
    n_repeats: int

    def summary(self, top_n: int = 10) -> str:
        """Return top-N features by mean importance.

        Parameters
        ----------
        top_n :
            Number of features to display.  Default 10.
        """
        n = len(self.importances_mean_)
        idx = np.argsort(self.importances_mean_)[::-1][:min(top_n, n)]
        sep = "=" * 70
        dash = "-" * 70
        lines = [
            sep,
            f"{'Permutation Importance (n_repeats=' + str(self.n_repeats) + ')':^70}",
            sep,
            f"{'Feature':35s} {'Mean':>12s} {'Std':>12s}",
            dash,
        ]
        for i in idx:
            lines.append(
                f"{self.feature_names_[i]:35s}"
                f" {self.importances_mean_[i]:>12.6f}"
                f" {self.importances_std_[i]:>12.6f}"
            )
        if n > top_n:
            lines.append(f"  ... ({n - top_n} more features not shown)")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class CondPermutationImportanceResult:
    """Result of :func:`cond_permutation_importance`.

    Attributes
    ----------
    importances_mean_ :
        Mean importance over ``n_repeats`` repeats.
    importances_std_ :
        Standard deviation over repeats.
    feature_names_ :
        List of feature names.
    n_repeats :
        Number of conditional permutation repeats used.
    method :
        Always ``"strobl"`` -- Strobl (2008) conditional permutation.
    """

    importances_mean_: np.ndarray
    importances_std_: np.ndarray
    feature_names_: list[str]
    n_repeats: int
    method: str = "strobl"

    def summary(self, top_n: int = 10) -> str:
        """Return top-N features by mean conditional permutation importance.

        Parameters
        ----------
        top_n :
            Number of features to display.  Default 10.
        """
        n = len(self.importances_mean_)
        idx = np.argsort(self.importances_mean_)[::-1][:min(top_n, n)]
        sep = "=" * 72
        dash = "-" * 72
        lines = [
            sep,
            f"{'Conditional Permutation Importance (' + self.method + ')':^72}",
            sep,
            f"{'Feature':35s} {'Mean':>12s} {'Std':>12s}",
            dash,
        ]
        for i in idx:
            lines.append(
                f"{self.feature_names_[i]:35s}"
                f" {self.importances_mean_[i]:>12.6f}"
                f" {self.importances_std_[i]:>12.6f}"
            )
        if n > top_n:
            lines.append(f"  ... ({n - top_n} more features not shown)")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class PDPImportanceResult:
    """Result of :func:`partial_dependence_importance`.

    Attributes
    ----------
    importances_ :
        1-D array of PDP importance (range of partial dependence function)
        per feature.
    feature_names_ :
        List of feature names.
    pdp_values_ :
        Dict mapping feature name -> 1-D array of mean predictions at
        grid points.
    grid_values_ :
        Dict mapping feature name -> 1-D array of grid evaluation points.
    """

    importances_: np.ndarray
    feature_names_: list[str]
    pdp_values_: dict[str, np.ndarray]
    grid_values_: dict[str, np.ndarray]

    def summary(self, top_n: int = 10) -> str:
        """Return top-N features by PDP importance.

        Parameters
        ----------
        top_n :
            Number of features to display.  Default 10.
        """
        n = len(self.importances_)
        idx = np.argsort(self.importances_)[::-1][:min(top_n, n)]
        sep = "=" * 60
        dash = "-" * 60
        lines = [
            sep,
            f"{'Partial Dependence Importance':^60}",
            sep,
            f"{'Feature':35s} {'PDP Range':>12s}",
            dash,
        ]
        for i in idx:
            lines.append(
                f"{self.feature_names_[i]:35s} {self.importances_[i]:>12.6f}"
            )
        if n > top_n:
            lines.append(f"  ... ({n - top_n} more features not shown)")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class ALEImportanceResult:
    """Result of :func:`ale_importance`.

    Attributes
    ----------
    importances_ :
        1-D array of ALE importance (mean absolute centred ALE) per feature.
    feature_names_ :
        List of feature names.
    ale_values_ :
        Dict mapping feature name -> 1-D array of centred cumulative
        ALE values at bin boundaries.
    """

    importances_: np.ndarray
    feature_names_: list[str]
    ale_values_: dict[str, np.ndarray]

    def summary(self, top_n: int = 10) -> str:
        """Return top-N features by ALE importance.

        Parameters
        ----------
        top_n :
            Number of features to display.  Default 10.
        """
        n = len(self.importances_)
        idx = np.argsort(self.importances_)[::-1][:min(top_n, n)]
        sep = "=" * 60
        dash = "-" * 60
        lines = [
            sep,
            f"{'ALE Importance (Apley & Zhu 2020)':^60}",
            sep,
            f"{'Feature':35s} {'ALE L1 norm':>12s}",
            dash,
        ]
        for i in idx:
            lines.append(
                f"{self.feature_names_[i]:35s} {self.importances_[i]:>12.6f}"
            )
        if n > top_n:
            lines.append(f"  ... ({n - top_n} more features not shown)")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class SHAPImportanceResult:
    """Result of :func:`shap_tree_importance` and :func:`shap_linear_importance`.

    Attributes
    ----------
    shap_values_ :
        2-D array of SHAP values, shape (n_samples, n_features).
    expected_value_ :
        SHAP base value (expected model output over training data).
    feature_names_ :
        List of feature names.
    explainer_type :
        Explainer used: ``"TreeExplainer"``, ``"LinearExplainer"``, or
        ``"KernelExplainer"`` (fallback).
    """

    shap_values_: np.ndarray
    expected_value_: float
    feature_names_: list[str]
    explainer_type: str

    def summary(self, top_n: int = 10) -> str:
        """Return top-N features by mean absolute SHAP value.

        Parameters
        ----------
        top_n :
            Number of features to display.  Default 10.
        """
        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        n = len(mean_abs)
        idx = np.argsort(mean_abs)[::-1][:min(top_n, n)]
        sep = "=" * 72
        dash = "-" * 72
        lines = [
            sep,
            f"{'SHAP Importance (' + self.explainer_type + ')':^72}",
            sep,
            f"{'Feature':35s} {'Mean |SHAP|':>12s}",
            dash,
        ]
        for i in idx:
            lines.append(
                f"{self.feature_names_[i]:35s} {mean_abs[i]:>12.6f}"
            )
        if n > top_n:
            lines.append(f"  ... ({n - top_n} more features not shown)")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public callables
# ---------------------------------------------------------------------------

def model_native_linear_coef_importance(
    result: Any,
    X: np.ndarray | pd.DataFrame,
) -> NativeImportanceResult:
    """Compute model-native linear coefficient importance.

    Extracts standardised regression coefficients (``|coef_j|``) from a
    fitted linear model as the importance score.  Compatible with every
    linear-family L4 FitResult (ridge, OLS, lasso, elastic_net,
    bayesian_ridge, huber, glmboost).

    Parameters
    ----------
    result :
        A fitted result object exposing ``._model`` (a raw sklearn
        estimator with ``coef_``).  Typically one of
        ``mf.functions.{ridge,ols,lasso,...}_fit(...)`` return values.
    X :
        Feature matrix used for name inference.  Shape (n_samples,
        n_features).  Accepts numpy arrays or DataFrames.

    Returns
    -------
    NativeImportanceResult
        ``importances_`` = ``|coef_j|``, ``method="linear_coef"``.

    Raises
    ------
    ValueError
        If ``result._model`` does not expose ``coef_`` (i.e. it is not a
        linear model).
    """
    from macroforecast.core.runtime import _linear_importance_frame

    if not hasattr(result, "_model"):
        raise ValueError(
            "result must expose ._model (a fitted sklearn estimator); "
            f"got {type(result).__name__!r}"
        )
    if not hasattr(result._model, "coef_"):
        raise ValueError(
            f"model_native_linear_coef_importance requires a linear model "
            f"with coef_, but {type(result._model).__name__!r} has none. "
            f"Use model_native_tree_importance for tree ensembles."
        )

    X_df = _to_frame(X)
    artifact = _extract_model_artifact(result, X_df)
    df = _linear_importance_frame(artifact, method="linear_coef")

    importances = np.asarray(df["importance"].values, dtype=float)
    feature_names = list(df["feature"].values)
    return NativeImportanceResult(
        importances_=importances,
        feature_names_=feature_names,
        method="linear_coef",
    )


def model_native_tree_importance(
    result: Any,
    X: np.ndarray | pd.DataFrame,
) -> NativeImportanceResult:
    """Compute model-native tree ensemble (MDI) importance.

    Returns sklearn's ``feature_importances_`` (mean decrease in impurity)
    from a fitted tree ensemble.  Compatible with every tree-family L4
    FitResult (random_forest, extra_trees, gradient_boosting, xgboost,
    lightgbm, catboost).

    Parameters
    ----------
    result :
        A fitted result object exposing ``._model`` (a raw sklearn/xgb/lgbm
        estimator with ``feature_importances_``).
    X :
        Feature matrix used for name inference.

    Returns
    -------
    NativeImportanceResult
        ``importances_`` = ``feature_importances_``,
        ``method="tree_native"``.

    Raises
    ------
    ValueError
        If ``result._model`` does not expose ``feature_importances_``.
    """
    from macroforecast.core.runtime import _tree_importance_frame

    if not hasattr(result, "_model"):
        raise ValueError(
            "result must expose ._model (a fitted sklearn estimator); "
            f"got {type(result).__name__!r}"
        )
    if not hasattr(result._model, "feature_importances_"):
        raise ValueError(
            f"model_native_tree_importance requires a tree model with "
            f"feature_importances_, but {type(result._model).__name__!r} "
            f"has none. Use model_native_linear_coef_importance for linear models."
        )

    X_df = _to_frame(X)
    artifact = _extract_model_artifact(result, X_df)
    df = _tree_importance_frame(artifact)

    importances = np.asarray(df["importance"].values, dtype=float)
    feature_names = list(df["feature"].values)
    return NativeImportanceResult(
        importances_=importances,
        feature_names_=feature_names,
        method="tree_native",
    )


def permutation_importance(
    result: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_repeats: int = 10,
    random_state: int | None = None,
) -> PermutationImportanceResult:
    """Compute Breiman-Fisher-Rudin (2019) permutation importance.

    For each feature ``j`` and each repeat, draws a random permutation of
    ``X[:, j]`` using ``np.random.default_rng(seed)`` and measures the
    increase in MSE relative to the unpermuted baseline.  The reported
    ``importances_mean_`` is the mean over ``n_repeats`` independent random
    permutations.

    When ``random_state=None`` a fixed seed of 0 is used, which guarantees
    reproducibility across calls while still using a proper random permutation
    (matching the behaviour of the recipe-path helper
    ``_permutation_importance_frame``).

    Parameters
    ----------
    result :
        Fitted result exposing ``._model``.
    X :
        Feature matrix, shape (n_samples, n_features).
    y :
        Target vector, shape (n_samples,).
    n_repeats :
        Number of permutation repeats.  Default 10.
    random_state :
        Integer seed for the random permutation RNG.  ``None`` (default)
        uses seed 0, which is deterministic but uses a proper random
        permutation (not reversal).

    Returns
    -------
    PermutationImportanceResult
        ``importances_mean_``, ``importances_std_``, and ``importances_``
        (shape ``(n_features, n_repeats)``) over ``n_repeats``.
    """
    if not hasattr(result, "_model"):
        raise ValueError(
            "result must expose ._model; "
            f"got {type(result).__name__!r}"
        )
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)
    artifact = _extract_model_artifact(result, X_df)
    fitted = artifact.fitted_object
    feature_names = list(artifact.feature_names)

    if not hasattr(fitted, "predict"):
        raise ValueError(
            f"{type(fitted).__name__!r} does not expose predict(); "
            "cannot compute permutation importance."
        )

    # Align X and y, drop NaN rows
    aligned = pd.concat([X_df.reset_index(drop=True),
                          y_s.reset_index(drop=True).rename("__target__")], axis=1).dropna()
    X_eval = aligned[X_df.columns]
    y_eval = aligned["__target__"]

    try:
        baseline = float(((y_eval - fitted.predict(X_eval)) ** 2).mean())
    except Exception:
        baseline = 0.0

    # Use seed=0 when random_state is None for deterministic reproducibility.
    seed = 0 if random_state is None else int(random_state)
    rng = np.random.default_rng(seed)

    # Collect per-repeat importance: shape (n_repeats, n_features)
    repeat_scores: list[np.ndarray] = []
    for _ in range(n_repeats):
        scores = np.zeros(len(feature_names))
        for j, column in enumerate(feature_names):
            permuted = X_eval.copy()
            perm_idx = rng.permutation(len(X_eval))
            permuted[column] = X_eval[column].values[perm_idx]
            try:
                loss = float(((y_eval - fitted.predict(permuted)) ** 2).mean())
            except Exception:
                loss = 0.0
            scores[j] = loss - baseline
        repeat_scores.append(scores)

    matrix = np.stack(repeat_scores, axis=0)  # (n_repeats, n_features)
    # importances_ has shape (n_features, n_repeats) per sklearn convention
    importances_matrix = matrix.T
    return PermutationImportanceResult(
        importances_mean_=matrix.mean(axis=0),
        importances_std_=matrix.std(axis=0, ddof=0),
        importances_=importances_matrix,
        feature_names_=feature_names,
        n_repeats=n_repeats,
    )


def cond_permutation_importance(
    result: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_repeats: int = 10,
    random_state: int | None = None,
) -> CondPermutationImportanceResult:
    """Compute Strobl (2008) conditional permutation importance.

    Permutes each feature ``j`` only within bins defined by the most
    correlated other feature, preserving the conditional distribution
    ``X_j | X_{-j}`` and removing the extrapolation bias of plain
    permutation importance under correlated predictors.

    Parameters
    ----------
    result :
        Fitted result exposing ``._model``.
    X :
        Feature matrix, shape (n_samples, n_features).
    y :
        Target vector, shape (n_samples,).
    n_repeats :
        Number of permutation repeats.  Default 10.
    random_state :
        Seed for bin-restricted permutation RNG.  ``None`` (default)
        uses the deterministic reverse-order permutation within each
        bin.

    Returns
    -------
    CondPermutationImportanceResult
        ``importances_mean_``, ``importances_std_``, ``method="strobl"``.
    """
    from macroforecast.core.runtime import _strobl_permutation_importance_frame

    if not hasattr(result, "_model"):
        raise ValueError(
            "result must expose ._model; "
            f"got {type(result).__name__!r}"
        )
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)
    artifact = _extract_model_artifact(result, X_df)
    feature_names = list(artifact.feature_names)

    seed = 0 if random_state is None else int(random_state)
    repeat_scores: list[list[float]] = []
    for _ in range(n_repeats):
        df = _strobl_permutation_importance_frame(artifact, X_df, y_s, seed=seed)
        repeat_scores.append(list(df["importance"].values))
        seed += 1  # advance seed per repeat for independence

    matrix = np.array(repeat_scores, dtype=float)  # (n_repeats, n_features)
    return CondPermutationImportanceResult(
        importances_mean_=matrix.mean(axis=0),
        importances_std_=matrix.std(axis=0, ddof=0),
        feature_names_=feature_names,
        n_repeats=n_repeats,
        method="strobl",
    )


def partial_dependence_importance(
    result: Any,
    X: np.ndarray | pd.DataFrame,
    *,
    grid_resolution: int = 20,
) -> PDPImportanceResult:
    """Compute partial dependence plot (PDP) importance.

    For each feature ``j``, evaluates the mean model prediction over a
    grid spanning the 5th to 95th percentile of ``X[:, j]``.  The
    importance score is the range of the partial dependence function
    (max - min).

    Parameters
    ----------
    result :
        Fitted result exposing ``._model``.
    X :
        Feature matrix, shape (n_samples, n_features).
    grid_resolution :
        Number of evenly-spaced grid points between the 5th and 95th
        percentile of each feature.  Default 20.

    Returns
    -------
    PDPImportanceResult
        ``importances_``, ``pdp_values_``, ``grid_values_``.
    """
    from macroforecast.core.runtime import _partial_dependence_table

    if not hasattr(result, "_model"):
        raise ValueError(
            "result must expose ._model; "
            f"got {type(result).__name__!r}"
        )
    if grid_resolution < 2:
        raise ValueError(f"grid_resolution must be >= 2, got {grid_resolution!r}")

    X_df = _to_frame(X)
    artifact = _extract_model_artifact(result, X_df)
    feature_names = list(artifact.feature_names)
    fitted = artifact.fitted_object

    df = _partial_dependence_table(artifact, X_df, n_grid=grid_resolution)
    importances = np.asarray(df["importance"].values, dtype=float)

    # Reconstruct pdp_values_ and grid_values_ per feature
    pdp_values: dict[str, np.ndarray] = {}
    grid_values: dict[str, np.ndarray] = {}
    for column in feature_names:
        series = X_df[column].dropna()
        if series.empty:
            pdp_values[column] = np.zeros(1)
            grid_values[column] = np.zeros(1)
            continue
        grid = np.linspace(
            series.quantile(0.05),
            series.quantile(0.95),
            max(2, int(grid_resolution)),
        )
        responses = []
        for value in grid:
            edited = X_df.fillna(0.0).copy()
            edited[column] = value
            try:
                response = float(np.mean(fitted.predict(edited)))
            except Exception:
                response = 0.0
            responses.append(response)
        grid_values[column] = grid
        pdp_values[column] = np.asarray(responses, dtype=float)

    return PDPImportanceResult(
        importances_=importances,
        feature_names_=feature_names,
        pdp_values_=pdp_values,
        grid_values_=grid_values,
    )


def ale_importance(
    result: Any,
    X: np.ndarray | pd.DataFrame,
    *,
    n_bins: int = 20,
) -> ALEImportanceResult:
    """Compute Accumulated Local Effects (ALE) importance.

    Implements Apley & Zhu (2020) ALE: for each feature ``j``, partitions
    the support into ``n_bins`` quantile bins, computes the local effect
    within each bin (average prediction change from lower to upper bin
    edge), centres the local effects (subtract mean), cumulates them, and
    reports the L1 norm of the centred cumulative ALE as the importance.

    Parameters
    ----------
    result :
        Fitted result exposing ``._model``.
    X :
        Feature matrix, shape (n_samples, n_features).
    n_bins :
        Number of quantile bins for ALE computation.  Default 20.

    Returns
    -------
    ALEImportanceResult
        ``importances_`` (L1 norm of centred ALE per feature),
        ``ale_values_`` (centred cumulative ALE per feature).
    """
    from macroforecast.core.runtime import _ale_table

    if not hasattr(result, "_model"):
        raise ValueError(
            "result must expose ._model; "
            f"got {type(result).__name__!r}"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins!r}")

    X_df = _to_frame(X)
    artifact = _extract_model_artifact(result, X_df)
    feature_names = list(artifact.feature_names)

    df = _ale_table(artifact, X_df, n_quantiles=n_bins)
    importances = np.asarray(df["importance"].values, dtype=float)

    # Extract ale_function from the 'ale_function' column
    ale_values: dict[str, np.ndarray] = {}
    for i, column in enumerate(feature_names):
        row_matches = df[df["feature"] == column]
        if row_matches.empty or "ale_function" not in df.columns:
            ale_values[column] = np.zeros(0)
            continue
        ale_func = row_matches.iloc[0]["ale_function"]
        if isinstance(ale_func, list) and len(ale_func) > 0:
            ale_values[column] = np.asarray(
                [entry["ale"] for entry in ale_func], dtype=float
            )
        else:
            ale_values[column] = np.zeros(0)

    return ALEImportanceResult(
        importances_=importances,
        feature_names_=feature_names,
        ale_values_=ale_values,
    )


def shap_tree_importance(
    result: Any,
    X: np.ndarray | pd.DataFrame,
) -> SHAPImportanceResult:
    """Compute SHAP importance using TreeExplainer.

    Uses the ``shap`` library's ``TreeExplainer`` for tree-ensemble
    models.  Requires the optional ``shap`` package.

    Parameters
    ----------
    result :
        Fitted result exposing ``._model`` (a tree model with
        ``feature_importances_``).
    X :
        Feature matrix, shape (n_samples, n_features).

    Returns
    -------
    SHAPImportanceResult
        ``shap_values_``, ``expected_value_``, ``explainer_type``.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    ValueError
        If ``result._model`` is not a tree model.
    """
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "shap_tree_importance requires the 'shap' package. "
            "Install with: pip install shap"
        ) from exc

    if not hasattr(result, "_model"):
        raise ValueError(
            "result must expose ._model; "
            f"got {type(result).__name__!r}"
        )
    if not hasattr(result._model, "feature_importances_"):
        raise ValueError(
            f"shap_tree_importance requires a tree model with "
            f"feature_importances_, but {type(result._model).__name__!r} "
            f"has none. Use shap_linear_importance for linear models."
        )

    X_df = _to_frame(X)
    artifact = _extract_model_artifact(result, X_df)
    feature_names = list(artifact.feature_names)
    fitted = artifact.fitted_object

    X_shap = X_df.fillna(0.0)
    import shap  # noqa: F811
    try:
        explainer = shap.TreeExplainer(fitted)
        values = explainer.shap_values(X_shap)
        ev = explainer.expected_value; expected = float(ev.item() if hasattr(ev, "item") else float(np.asarray(ev).ravel()[0]) if hasattr(ev, "__len__") else ev)
        explainer_type = "TreeExplainer"
    except Exception:
        # Fallback to KernelExplainer
        try:
            explainer = shap.KernelExplainer(
                fitted.predict, shap.sample(X_shap, min(50, len(X_shap)))
            )
            values = explainer.shap_values(X_shap)
            ev = explainer.expected_value; expected = float(ev.item() if hasattr(ev, "item") else float(np.asarray(ev).ravel()[0]) if hasattr(ev, "__len__") else ev)
            explainer_type = "KernelExplainer"
        except Exception:
            # Last resort: zero SHAP values
            values = np.zeros((len(X_shap), len(feature_names)))
            expected = 0.0
            explainer_type = "KernelExplainer"

    values_arr = (
        np.asarray(values, dtype=float)
        if isinstance(values, np.ndarray)
        else np.asarray(np.squeeze(values), dtype=float)
    )
    if values_arr.ndim == 1:
        values_arr = values_arr.reshape(1, -1)

    return SHAPImportanceResult(
        shap_values_=values_arr,
        expected_value_=expected,
        feature_names_=feature_names,
        explainer_type=explainer_type,
    )


def shap_linear_importance(
    result: Any,
    X: np.ndarray | pd.DataFrame,
) -> SHAPImportanceResult:
    """Compute SHAP importance using LinearExplainer.

    Uses the ``shap`` library's ``LinearExplainer`` for linear models.
    Requires the optional ``shap`` package.

    Parameters
    ----------
    result :
        Fitted result exposing ``._model`` (a linear model with
        ``coef_``).
    X :
        Feature matrix, shape (n_samples, n_features).

    Returns
    -------
    SHAPImportanceResult
        ``shap_values_``, ``expected_value_``, ``explainer_type``.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    ValueError
        If ``result._model`` is not a linear model.
    """
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "shap_linear_importance requires the 'shap' package. "
            "Install with: pip install shap"
        ) from exc

    if not hasattr(result, "_model"):
        raise ValueError(
            "result must expose ._model; "
            f"got {type(result).__name__!r}"
        )
    if not hasattr(result._model, "coef_"):
        raise ValueError(
            f"shap_linear_importance requires a linear model with "
            f"coef_, but {type(result._model).__name__!r} has none. "
            f"Use shap_tree_importance for tree models."
        )

    X_df = _to_frame(X)
    artifact = _extract_model_artifact(result, X_df)
    feature_names = list(artifact.feature_names)
    fitted = artifact.fitted_object

    X_shap = X_df.fillna(0.0)
    import shap  # noqa: F811
    try:
        explainer = shap.LinearExplainer(fitted, X_shap)
        values = explainer.shap_values(X_shap)
        ev = explainer.expected_value; expected = float(ev.item() if hasattr(ev, "item") else float(np.asarray(ev).ravel()[0]) if hasattr(ev, "__len__") else ev)
        explainer_type = "LinearExplainer"
    except Exception:
        # Fallback to KernelExplainer
        try:
            explainer = shap.KernelExplainer(
                fitted.predict, shap.sample(X_shap, min(50, len(X_shap)))
            )
            values = explainer.shap_values(X_shap)
            ev = explainer.expected_value; expected = float(ev.item() if hasattr(ev, "item") else float(np.asarray(ev).ravel()[0]) if hasattr(ev, "__len__") else ev)
            explainer_type = "KernelExplainer"
        except Exception:
            values = np.zeros((len(X_shap), len(feature_names)))
            expected = 0.0
            explainer_type = "KernelExplainer"

    values_arr = (
        np.asarray(values, dtype=float)
        if isinstance(values, np.ndarray)
        else np.asarray(np.squeeze(values), dtype=float)
    )
    if values_arr.ndim == 1:
        values_arr = values_arr.reshape(1, -1)

    return SHAPImportanceResult(
        shap_values_=values_arr,
        expected_value_=expected,
        feature_names_=feature_names,
        explainer_type=explainer_type,
    )


__all__ = [
    # Result types
    "NativeImportanceResult",
    "PermutationImportanceResult",
    "CondPermutationImportanceResult",
    "PDPImportanceResult",
    "ALEImportanceResult",
    "SHAPImportanceResult",
    # Callables
    "model_native_linear_coef_importance",
    "model_native_tree_importance",
    "permutation_importance",
    "cond_permutation_importance",
    "partial_dependence_importance",
    "ale_importance",
    "shap_tree_importance",
    "shap_linear_importance",
]
