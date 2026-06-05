from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from importlib import import_module
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.interpretation._anatomy_utils import anatomy_output_transform
from macroforecast.models import ModelFit

_INTERPRETATION_SCHEMA_VERSION = 1

_REFERENCE_CATALOG: dict[str, dict[str, Any]] = {
    "linear_coefficients": {
        "class": "native_attribute_extraction",
        "reference": "scikit-learn-style estimator.coef_ convention",
        "alignment": "direct_attribute_read",
    },
    "tree_importance": {
        "class": "native_attribute_extraction",
        "reference": "scikit-learn-style estimator.feature_importances_ convention",
        "alignment": "direct_attribute_read",
    },
    "permutation_importance": {
        "class": "standard_model_agnostic_diagnostic",
        "reference": "sklearn.inspection.permutation_importance",
        "alignment": "same score-drop logic, expressed as loss degradation for lower-is-better losses",
    },
    "permutation_importance_strobl": {
        "class": "approximation",
        "reference": "Strobl conditional permutation importance idea",
        "alignment": "single most-correlated feature quantile-bin approximation, not exact party/partykit implementation",
    },
    "lofo_importance": {
        "class": "standard_when_refit_else_diagnostic",
        "reference": "lofo-importance Python package leave-one-feature-out refit idea",
        "alignment": "refit mode matches leave-one-feature-out logic; prediction_drop mode is macroforecast diagnostic",
    },
    "partial_dependence": {
        "class": "standard_model_agnostic_diagnostic",
        "reference": "sklearn.inspection.partial_dependence and R pdp::partial brute-force PDP",
        "alignment": "same average-prediction definition with a macroforecast min-max grid",
    },
    "individual_conditional_expectation": {
        "class": "standard_model_agnostic_diagnostic",
        "reference": "sklearn partial dependence kind='individual' and R pdp::partial ice=TRUE",
        "alignment": "same brute-force feature-grid replacement, returned as long-form individual curves",
    },
    "ice_curves": {
        "class": "standard_model_agnostic_diagnostic",
        "reference": "sklearn partial dependence kind='individual' and R pdp::partial ice=TRUE",
        "alignment": "alias for individual_conditional_expectation",
    },
    "accumulated_local_effect": {
        "class": "standard_model_agnostic_diagnostic",
        "reference": "Apley-Zhu first-order ALE / R ALEPlot",
        "alignment": "first-order finite-difference ALE, no second-order ALE or bootstrap intervals",
    },
    "friedman_h_interaction": {
        "class": "standard_formula_approximation",
        "reference": "Friedman-Popescu H-statistic as exposed by R iml/hstats",
        "alignment": "manual PDP-grid pairwise H diagnostic, no exact iml/hstats backend",
    },
    "shap_values": {
        "class": "backend_wrapper",
        "reference": "SHAP Python package",
        "alignment": "direct shap.Explainer/TreeExplainer call, reshaped to long pandas output",
    },
    "shap_importance": {
        "class": "backend_summary",
        "reference": "SHAP Python package",
        "alignment": "mean absolute SHAP value summary from macroforecast shap_values output",
    },
    "anatomy_explain": {
        "class": "backend_wrapper",
        "reference": "anatomy Python package for Borup, Goulet Coulombe, Rapach, Montes Schutte, and Schwenk-Nebbe (2022)",
        "alignment": "wraps precomputed anatomy.Anatomy.explain output and reshapes it to long pandas output",
    },
    "performance_shapley_value": {
        "class": "paper_formula_adapter",
        "reference": "performance-based Shapley value in The Anatomy of Out-of-Sample Forecasting Accuracy",
        "alignment": "exact or sampled Shapley decomposition of point loss over additive forecast contributions",
    },
    "shapley_variable_importance": {
        "class": "table_aggregation",
        "reference": "oShapley-VI aggregation in the anatomy package README; also usable for user-supplied iShapley-VI tables",
        "alignment": "mean absolute Shapley contribution by feature, optionally by model set",
    },
    "ishapley_vi": {
        "class": "table_aggregation",
        "reference": "iShapley-VI_p from in-sample Shapley contribution tables in Borup et al. (2022)",
        "alignment": "same mean-absolute Shapley-VI aggregation as oShapley, applied to in-sample fitted-prediction contributions",
    },
    "oshapley_vi": {
        "class": "backend_wrapper",
        "reference": "anatomy package oShapley-VI from raw forecast explanations",
        "alignment": "calls anatomy.Anatomy.explain on raw forecasts, then averages absolute local contributions",
    },
    "pbsv": {
        "class": "backend_wrapper",
        "reference": "anatomy package PBSV_p loss decomposition",
        "alignment": "calls anatomy.Anatomy.explain with a loss transformer and reshapes backend output",
    },
    "model_accordance_score": {
        "class": "backend_wrapper",
        "reference": "anatomy.MAS model accordance score and hypothesis test",
        "alignment": "direct anatomy.MAS call after converting macroforecast tables to feature-indexed Series",
    },
    "forecast_decomposition": {
        "class": "exact_linear_else_marked_fallback",
        "reference": "linear additive contribution x_j beta_j",
        "alignment": "exact for linear coefficients; nonlinear fallback is explicitly non-additive",
    },
    "cumulative_r2_contribution": {
        "class": "macroforecast_diagnostic",
        "reference": "sequential fixed-model masking diagnostic",
        "alignment": "not a refit sequential R2 package implementation",
    },
    "group_aggregate": {
        "class": "aggregation_plumbing",
        "reference": "feature-importance group aggregation",
        "alignment": "macroforecast table aggregation",
    },
    "lineage_attribution": {
        "class": "aggregation_plumbing",
        "reference": "feature-lineage group aggregation",
        "alignment": "macroforecast metadata aggregation",
    },
    "transformation_attribution": {
        "class": "macroforecast_diagnostic",
        "reference": "Shapley value over user-supplied pipeline utility table",
        "alignment": "cooperative-game summary of mutually exclusive alternatives, not causal component decomposition",
    },
    "attention_weights": {
        "class": "closed_form_linear_algebra",
        "reference": "Ordinary Least Squares as an Attention Mechanism, Goulet Coulombe (2026)",
        "alignment": "closed-form X_test (X_train'X_train + lambda I)^-1 X_train' with unpenalized intercept",
    },
    "ols_attention_weights": {
        "class": "closed_form_linear_algebra",
        "reference": "Ordinary Least Squares as an Attention Mechanism, Goulet Coulombe (2026)",
        "alignment": "exact OLS attention weights X_test (X_train'X_train)^-1 X_train' using a pseudoinverse when needed",
    },
    "ridge_attention_weights": {
        "class": "closed_form_linear_algebra",
        "reference": "Ordinary Least Squares as an Attention Mechanism, Goulet Coulombe (2026), ridge extension",
        "alignment": "ridge-stabilized attention weights X_test (X_train'X_train + alpha I)^-1 X_train' with unpenalized intercept",
    },
    "ols_attention_embedding": {
        "class": "closed_form_linear_algebra",
        "reference": "Ordinary Least Squares as an Attention Mechanism, Goulet Coulombe (2026)",
        "alignment": "whitened train/test embeddings whose inner products reconstruct the OLS/ridge attention matrix",
    },
    "ols_attention_equivalence": {
        "class": "closed_form_linear_algebra",
        "reference": "Ordinary Least Squares as an Attention Mechanism, Goulet Coulombe (2026)",
        "alignment": "audits yhat = attention_weights @ y_train against closed-form or user-supplied reference predictions",
    },
    "dual_decomposition": {
        "class": "closed_form_linear_algebra",
        "reference": "Ordinary Least Squares as an Attention Mechanism, Goulet Coulombe (2026)",
        "alignment": "uses attention_weights to express prediction as weighted training outcomes",
    },
    "observation_weights": {
        "class": "paper_formula_adapter",
        "reference": "Dual Interpretation of Machine Learning Forecasts, Goulet Coulombe, Goebel, and Klieber (2024)",
        "alignment": "ridge, kernel-ridge, and random-forest observation-weight formulas aligned with local DualML Python/R reference code",
    },
    "outcome_contributions": {
        "class": "paper_formula_adapter",
        "reference": "Dual Interpretation of Machine Learning Forecasts, Goulet Coulombe, Goebel, and Klieber (2024)",
        "alignment": "multiplies observation weights by centered or raw in-sample outcomes to form data-portfolio episode contributions",
    },
    "data_portfolio_diagnostics": {
        "class": "paper_formula_adapter",
        "reference": "Dual Interpretation of Machine Learning Forecasts, Goulet Coulombe, Goebel, and Klieber (2024)",
        "alignment": "forecast concentration, short position, leverage, and turnover definitions aligned with DualML metrics",
    },
    "top_episodes": {
        "class": "paper_formula_adapter",
        "reference": "Dual Interpretation of Machine Learning Forecasts, Goulet Coulombe, Goebel, and Klieber (2024)",
        "alignment": "ranks historical observations by signed or absolute data-portfolio weight",
    },
    "episode_group_weights": {
        "class": "paper_formula_adapter",
        "reference": "Dual Interpretation of Machine Learning Forecasts, Goulet Coulombe, Goebel, and Klieber (2024)",
        "alignment": "aggregates data-portfolio weights and outcome contributions over user-defined regimes or historical episode groups",
    },
    "generalized_irf": {
        "class": "econometric_formula",
        "reference": "Pesaran-Shin generalized impulse response",
        "alignment": "formula implementation using statsmodels-compatible VAR outputs",
    },
    "orthogonalised_irf": {
        "class": "backend_or_adapter",
        "reference": "statsmodels VARResults.irf orth_irfs / Cholesky IRF",
        "alignment": "uses statsmodels output when available, otherwise internal statsmodels-like adapter",
    },
    "fevd": {
        "class": "backend_or_adapter",
        "reference": "statsmodels VARResults.fevd",
        "alignment": "uses statsmodels FEVD when available, otherwise computes orthogonalized FEVD from MA representation",
    },
    "historical_decomposition": {
        "class": "reduced_form_diagnostic",
        "reference": "VAR moving-average residual contribution summary",
        "alignment": "reduced-form diagnostic, not structural historical decomposition",
    },
    "lasso_inclusion_frequency": {
        "class": "bootstrap_diagnostic",
        "reference": "bootstrap nonzero-selection frequency",
        "alignment": "macroforecast bootstrap/refit diagnostic, not a randomized-lasso backend",
    },
    "mrf_gtvp": {
        "class": "backend_extraction",
        "reference": "MacroRandomForest GTVP beta output",
        "alignment": "direct extraction of backend betas after prediction",
    },
    "rolling_recompute": {
        "class": "macroforecast_diagnostic",
        "reference": "rolling fixed-model importance recomputation",
        "alignment": "does not refit model per window",
    },
    "bootstrap_jackknife": {
        "class": "bootstrap_diagnostic",
        "reference": "bootstrap uncertainty summary",
        "alignment": "bootstrap with replacement; no jackknife mode yet",
    },
    "gradient_attribution": {
        "class": "manual_torch_attribution",
        "reference": "Captum-style saliency/integrated-gradients/gradient-shap APIs",
        "alignment": "manual torch autograd implementation except deep_lift",
    },
    "saliency_map": {
        "class": "manual_torch_attribution",
        "reference": "Captum Saliency",
        "alignment": "raw input gradient with mean-absolute importance summary",
    },
    "integrated_gradients": {
        "class": "manual_torch_attribution",
        "reference": "Captum IntegratedGradients",
        "alignment": "manual straight-line Riemann approximation, not Captum gausslegendre backend",
    },
    "gradient_shap": {
        "class": "manual_torch_attribution",
        "reference": "Captum GradientShap",
        "alignment": "manual expected-gradient approximation, not Captum backend",
    },
    "deep_lift": {
        "class": "backend_wrapper",
        "reference": "Captum DeepLift",
        "alignment": "direct captum.attr.DeepLift call",
    },
    "lstm_hidden_state": {
        "class": "macroforecast_diagnostic",
        "reference": "recurrent hidden-activation summary",
        "alignment": "forward-hook activation magnitude, no external package equivalence",
    },
    "custom_interpretation": {
        "class": "custom_plumbing",
        "reference": "user-supplied callable",
        "alignment": "macroforecast wrapper only",
    },
}


def linear_coefficients(model: Any, *, sort: bool = True) -> pd.DataFrame:
    """Return native coefficients for linear-style fitted models."""

    fit = _coerce_fit(model)
    estimator = fit.estimator if isinstance(fit, ModelFit) else fit
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        raise ValueError("model does not expose coef_")
    values = np.asarray(coef, dtype=float).reshape(-1)
    names = _feature_names(fit, len(values))
    table = pd.DataFrame(
        {
            "feature": names,
            "coefficient": values,
            "abs_coefficient": np.abs(values),
        }
    )
    if sort:
        table = table.sort_values("abs_coefficient", ascending=False, kind="stable")
    return _attach_schema(
        table.reset_index(drop=True),
        kind="linear_coefficients",
        model=model,
        method="native_coef",
        n_features=len(values),
    )


def tree_importance(model: Any, *, sort: bool = True) -> pd.DataFrame:
    """Return native tree importance for estimators exposing feature_importances_."""

    fit = _coerce_fit(model)
    estimator = fit.estimator if isinstance(fit, ModelFit) else fit
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None:
        raise ValueError("model does not expose feature_importances_")
    values = np.asarray(importance, dtype=float).reshape(-1)
    names = _feature_names(fit, len(values))
    table = pd.DataFrame({"feature": names, "importance": values})
    if sort:
        table = table.sort_values("importance", ascending=False, kind="stable")
    return _attach_schema(
        table.reset_index(drop=True),
        kind="tree_importance",
        model=model,
        method="native_feature_importances",
        n_features=len(values),
    )


def model_native_linear_coef(model: Any, *, sort: bool = True) -> pd.DataFrame:
    """Alias for legacy L7 naming: native linear coefficients."""

    table = linear_coefficients(model, sort=sort)
    table.attrs["macroforecast_metadata_schema"]["method"] = "model_native_linear_coef"
    return table


def model_native_tree_importance(model: Any, *, sort: bool = True) -> pd.DataFrame:
    """Alias for legacy L7 naming: native tree feature importance."""

    table = tree_importance(model, sort=sort)
    table.attrs["macroforecast_metadata_schema"]["method"] = (
        "model_native_tree_importance"
    )
    return table


def permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse",
    n_repeats: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Compute simple model-agnostic permutation importance.

    Importance is the degradation in the loss metric after permuting one
    feature. For score metrics where higher is better, pass a callable that
    already returns a loss-like value if positive degradation is desired.
    """

    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    frame = _as_feature_frame(X)
    target = np.asarray(y, dtype=float).reshape(-1)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    rng = np.random.default_rng(random_state)
    loss = _loss_func(metric)
    # Reference alignment: sklearn.inspection.permutation_importance computes
    # s_reference - mean(s_permuted). With lower-is-better losses, the same
    # diagnostic is expressed as mean(loss_permuted) - loss_reference.
    baseline = loss(target, _predict(model, frame))
    rows: list[dict[str, Any]] = []
    for feature in frame.columns:
        deltas = []
        for _ in range(int(n_repeats)):
            permuted = frame.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            deltas.append(loss(target, _predict(model, permuted)) - baseline)
        values = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "feature": str(feature),
                "importance": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "baseline_loss": float(baseline),
                "n_repeats": int(n_repeats),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return _attach_schema(
        table,
        kind="permutation_importance",
        model=model,
        method="permutation_loss_degradation",
        n_features=frame.shape[1],
        metadata={
            "metric": getattr(loss, "__name__", str(metric)),
            "n_obs": int(len(frame)),
            "n_repeats": int(n_repeats),
        },
    )


def permutation_importance_strobl(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse",
    n_repeats: int = 5,
    n_bins: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Conditional permutation importance following the Strobl idea.

    Each feature is permuted within bins of its most correlated companion
    feature. This keeps the permutation closer to the observed conditional
    distribution than a marginal shuffle, which is the relevant distinction
    when macro predictors are strongly collinear.
    """

    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if n_bins <= 1:
        raise ValueError("n_bins must be greater than 1")
    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    joined = pd.concat([frame, target.rename("__target__")], axis=1).dropna()
    eval_x = joined.loc[:, frame.columns]
    eval_y = joined["__target__"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    loss = _loss_func(metric)
    baseline = loss(eval_y, _predict(model, eval_x))
    rows: list[dict[str, Any]] = []
    for feature in eval_x.columns:
        other = [column for column in eval_x.columns if column != feature]
        conditioning_feature = None
        if other:
            correlations = eval_x[other].corrwith(eval_x[feature]).abs().fillna(0.0)
            conditioning_feature = str(correlations.idxmax())
        deltas: list[float] = []
        for _ in range(int(n_repeats)):
            permuted = eval_x.copy()
            if conditioning_feature is None:
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            else:
                bins = _safe_qcut(eval_x[conditioning_feature], int(n_bins))
                for bin_id in bins.dropna().unique():
                    mask = bins == bin_id
                    if int(mask.sum()) <= 1:
                        continue
                    values = permuted.loc[mask, feature].to_numpy()
                    permuted.loc[mask, feature] = rng.permutation(values)
            deltas.append(loss(eval_y, _predict(model, permuted)) - baseline)
        values = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "feature": str(feature),
                "importance": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "baseline_loss": float(baseline),
                "n_repeats": int(n_repeats),
                "conditioning_feature": conditioning_feature,
                "n_bins": int(n_bins),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return _attach_schema(
        table,
        kind="permutation_importance_strobl",
        model=model,
        method="conditional_permutation_loss_degradation",
        n_features=frame.shape[1],
        metadata={
            "metric": getattr(loss, "__name__", str(metric)),
            "n_obs": int(len(eval_x)),
            "n_repeats": int(n_repeats),
            "n_bins": int(n_bins),
            "reference": "Strobl-style conditional permutation approximation",
            "conditioning_strategy": "single_most_correlated_feature_quantile_bins",
            "exact_reference_implementation": False,
        },
    )


def lofo_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None,
    metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse",
    sort: bool = True,
) -> pd.DataFrame:
    """Leave-one-feature-out importance.

    If ``fit_func`` is supplied, the model is refit without each feature. If it
    is omitted, the already fitted model is evaluated after setting the held-out
    feature to zero. The latter is a prediction-drop diagnostic, not a refit
    LOFO experiment, and the returned metadata records that mode explicitly.
    """

    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    loss = _loss_func(metric)
    baseline_model = fit_func(frame, target) if fit_func is not None else model
    baseline = loss(target.to_numpy(dtype=float), _predict(baseline_model, frame))
    rows: list[dict[str, Any]] = []
    mode = "refit" if fit_func is not None else "prediction_drop"
    for feature in frame.columns:
        if fit_func is not None:
            reduced_x = frame.drop(columns=[feature])
            reduced_model = fit_func(reduced_x, target)
            pred = _predict(reduced_model, reduced_x)
        else:
            reduced_x = frame.copy()
            reduced_x[feature] = 0.0
            pred = _predict(model, reduced_x)
        heldout_loss = loss(target.to_numpy(dtype=float), pred)
        rows.append(
            {
                "feature": str(feature),
                "importance": float(heldout_loss - baseline),
                "baseline_loss": float(baseline),
                "heldout_loss": float(heldout_loss),
                "mode": mode,
            }
        )
    table = pd.DataFrame(rows)
    if sort:
        table = table.sort_values("importance", ascending=False, kind="stable")
    return _attach_schema(
        table.reset_index(drop=True),
        kind="lofo_importance",
        model=baseline_model,
        method=mode,
        n_features=frame.shape[1],
        metadata={"metric": getattr(loss, "__name__", str(metric)), "mode": mode},
    )


def partial_dependence(
    model: Any,
    X: pd.DataFrame,
    *,
    features: Iterable[str] | str,
    grid_size: int = 20,
) -> pd.DataFrame:
    """Compute one-way manual partial-dependence curves."""

    frame = _as_feature_frame(X)
    selected = _resolve_features(frame, features)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1")
    rows: list[dict[str, Any]] = []
    for feature in selected:
        # Reference alignment: sklearn.partial_dependence / R pdp::partial
        # brute-force PDP replaces the target feature with grid values and
        # averages predictions over the empirical complement-feature rows.
        grid = np.linspace(
            float(frame[feature].min()),
            float(frame[feature].max()),
            int(grid_size),
        )
        for value in grid:
            replaced = frame.copy()
            replaced[feature] = value
            pred = _predict(model, replaced)
            rows.append(
                {
                    "feature": str(feature),
                    "value": float(value),
                    "prediction": float(np.mean(pred)),
                }
            )
    return _attach_schema(
        pd.DataFrame(rows),
        kind="partial_dependence",
        model=model,
        method="manual_one_way_pdp",
        n_features=len(selected),
        metadata={
            "grid_size": int(grid_size),
            "features": list(selected),
            "grid_strategy": "linear_min_max",
        },
    )


def individual_conditional_expectation(
    model: Any,
    X: pd.DataFrame,
    *,
    features: Iterable[str] | str,
    grid_size: int = 20,
    center: bool = False,
) -> pd.DataFrame:
    """Compute one-way individual conditional expectation curves."""

    frame = _as_feature_frame(X)
    selected = _resolve_features(frame, features)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1")
    rows: list[dict[str, Any]] = []
    for feature in selected:
        # Reference alignment: sklearn PDP kind="individual" and R pdp ICE
        # keep each observation's complement features fixed while replacing
        # the target feature with each grid value.
        grid = np.linspace(
            float(frame[feature].min()),
            float(frame[feature].max()),
            int(grid_size),
        )
        baseline_prediction: np.ndarray | None = None
        for grid_pos, value in enumerate(grid):
            replaced = frame.copy()
            replaced[feature] = value
            pred = _predict(model, replaced)
            if grid_pos == 0:
                baseline_prediction = pred.copy()
            centered = (
                pred - baseline_prediction
                if center and baseline_prediction is not None
                else np.full_like(pred, np.nan, dtype=float)
            )
            for row_pos, (idx, prediction, centered_value) in enumerate(
                zip(frame.index, pred, centered, strict=False)
            ):
                rows.append(
                    {
                        "feature": str(feature),
                        "row": int(row_pos),
                        "index": idx,
                        "value": float(value),
                        "prediction": float(prediction),
                        "centered_prediction": float(centered_value),
                    }
                )
    return _attach_schema(
        pd.DataFrame(rows),
        kind="individual_conditional_expectation",
        model=model,
        method="manual_one_way_ice",
        n_features=len(selected),
        metadata={
            "grid_size": int(grid_size),
            "features": list(selected),
            "grid_strategy": "linear_min_max",
            "center": bool(center),
        },
    )


def ice_curves(
    model: Any,
    X: pd.DataFrame,
    *,
    features: Iterable[str] | str,
    grid_size: int = 20,
    center: bool = False,
) -> pd.DataFrame:
    """Alias for :func:`individual_conditional_expectation`."""

    table = individual_conditional_expectation(
        model,
        X,
        features=features,
        grid_size=grid_size,
        center=center,
    )
    table.attrs["macroforecast_metadata_schema"]["method"] = "ice_curves_alias"
    return table


def accumulated_local_effect(
    model: Any,
    X: pd.DataFrame,
    *,
    feature: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Compute a first-order accumulated local effect curve."""

    frame = _as_feature_frame(X)
    if feature not in frame.columns:
        raise ValueError(f"feature {feature!r} is not in X")
    if bins <= 1:
        raise ValueError("bins must be greater than 1")
    values = frame[feature].astype(float)
    edges = np.unique(
        np.quantile(values.dropna(), np.linspace(0.0, 1.0, int(bins) + 1))
    )
    if len(edges) < 3:
        raise ValueError("feature needs at least two non-empty ALE bins")
    effects = []
    centers = []
    counts = []
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        # Reference alignment: first-order ALEPlot/Apley-Zhu finite difference
        # within feature intervals, followed by cumulative centering.
        mask = (values >= low) & (
            values <= high if high == edges[-1] else values < high
        )
        if not mask.any():
            effects.append(0.0)
            centers.append(float((low + high) / 2.0))
            counts.append(0)
            continue
        lower = frame.loc[mask].copy()
        upper = lower.copy()
        lower[feature] = low
        upper[feature] = high
        effects.append(float(np.mean(_predict(model, upper) - _predict(model, lower))))
        centers.append(float((low + high) / 2.0))
        counts.append(int(mask.sum()))
    accumulated = np.cumsum(np.asarray(effects, dtype=float))
    # Apley-Zhu (2020) ALE centering: subtract the FREQUENCY-WEIGHTED mean of the
    # accumulated curve (weighted by per-bin observation counts) so it integrates
    # to zero against the empirical feature distribution, not the unweighted mean.
    bin_weights = np.asarray(counts, dtype=float)
    if bin_weights.sum() > 0:
        accumulated = accumulated - np.average(accumulated, weights=bin_weights)
    else:
        accumulated = accumulated - accumulated.mean()
    table = pd.DataFrame(
        {
            "feature": str(feature),
            "bin": np.arange(1, len(accumulated) + 1),
            "center": centers,
            "ale": accumulated,
            "local_effect": effects,
        }
    )
    return _attach_schema(
        table,
        kind="accumulated_local_effect",
        model=model,
        method="first_order_ale",
        n_features=1,
        metadata={
            "feature": str(feature),
            "bins": int(bins),
            "binning": "empirical_quantile",
            "centering": "mean_zero",
        },
    )


def friedman_h_interaction(
    model: Any,
    X: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    grid_size: int = 10,
) -> pd.DataFrame:
    """Compute pairwise Friedman-Popescu H interaction statistics.

    The implementation uses manual one-way and two-way partial dependence on a
    regular grid. Values are bounded to ``[0, inf)`` by construction; larger
    values indicate stronger interaction relative to the pair's joint partial
    dependence variation.
    """

    frame = _as_feature_frame(X)
    selected = (
        tuple(features)
        if features is not None
        else tuple(str(c) for c in frame.columns)
    )
    _resolve_features(frame, selected)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1")
    rows: list[dict[str, Any]] = []
    for left, right in combinations(selected, 2):
        # Reference alignment: Friedman-Popescu H from PDP surfaces. This is
        # computed directly on a regular macroforecast grid rather than by
        # delegating to R iml/hstats.
        left_grid = _grid_values(frame[left], int(grid_size))
        right_grid = _grid_values(frame[right], int(grid_size))
        joint = np.empty((len(left_grid), len(right_grid)), dtype=float)
        for i, left_value in enumerate(left_grid):
            for j, right_value in enumerate(right_grid):
                replaced = frame.copy()
                replaced[left] = left_value
                replaced[right] = right_value
                joint[i, j] = float(np.mean(_predict(model, replaced)))
        left_pd = joint.mean(axis=1, keepdims=True)
        right_pd = joint.mean(axis=0, keepdims=True)
        centered = joint - left_pd - right_pd + joint.mean()
        denom = float(np.var(joint))
        h_value = (
            0.0
            if denom <= 1e-15
            else float(np.sqrt(max(np.var(centered) / denom, 0.0)))
        )
        rows.append(
            {
                "feature_1": str(left),
                "feature_2": str(right),
                "h_statistic": h_value,
                "joint_variance": denom,
                "interaction_variance": float(np.var(centered)),
                "grid_size": int(grid_size),
            }
        )
    return _attach_schema(
        pd.DataFrame(rows)
        .sort_values("h_statistic", ascending=False, kind="stable")
        .reset_index(drop=True),
        kind="friedman_h_interaction",
        model=model,
        method="manual_partial_dependence_h",
        n_features=len(selected),
        metadata={"features": list(selected), "grid_size": int(grid_size)},
    )


def shap_values(
    model: Any,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    explainer: str = "auto",
    check_additivity: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return SHAP values in a long pandas table.

    SHAP is an optional backend. Install ``macroforecast[interpretation]`` to
    use this helper.
    """

    shap = _optional_shap()
    frame = _as_feature_frame(X)
    background_frame = frame if background is None else _as_feature_frame(background)
    background_frame = background_frame.reindex(columns=frame.columns)
    resolved = _normalize_explainer(explainer)

    if resolved == "tree":
        # Backend wrapper: SHAP TreeExplainer is called directly; macroforecast
        # only reshapes SHAP's output into a long pandas table.
        target_model = model.estimator if isinstance(model, ModelFit) else model
        explainer_obj = shap.TreeExplainer(target_model, data=background_frame)
        explanation = explainer_obj.shap_values(
            frame, check_additivity=check_additivity
        )
        values = _coerce_shap_array(explanation, frame)
        base_values = _tree_base_values(explainer_obj, len(frame))
    else:
        # Backend wrapper: generic SHAP Explainer / PermutationExplainer path.
        predict_fn = lambda values: _predict(  # noqa: E731 - SHAP expects callable.
            model,
            _shap_prediction_frame(values, frame),
        )
        explainer_cls = (
            shap.PermutationExplainer if resolved == "permutation" else shap.Explainer
        )
        explainer_obj = explainer_cls(predict_fn, background_frame)
        call_kwargs = dict(kwargs)
        explanation = explainer_obj(frame, **call_kwargs)
        values = _coerce_shap_array(getattr(explanation, "values", explanation), frame)
        base_values = _coerce_base_values(
            getattr(explanation, "base_values", None),
            len(frame),
        )

    records: list[dict[str, Any]] = []
    for row_pos, (idx, row) in enumerate(frame.iterrows()):
        base_value = None if base_values is None else float(base_values[row_pos])
        for feature_pos, feature in enumerate(frame.columns):
            records.append(
                {
                    "row": int(row_pos),
                    "index": idx,
                    "feature": str(feature),
                    "feature_value": float(row.iloc[feature_pos]),
                    "shap_value": float(values[row_pos, feature_pos]),
                    "base_value": base_value,
                }
            )
    return _attach_schema(
        pd.DataFrame(records),
        kind="shap_values",
        model=model,
        method=f"shap_{resolved}",
        n_features=frame.shape[1],
        metadata={
            "explainer": resolved,
            "n_obs": int(len(frame)),
            "background_n_obs": int(len(background_frame)),
        },
    )


def shap_importance(
    model: Any,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    explainer: str = "auto",
    check_additivity: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Summarize SHAP values as global mean absolute feature importance."""

    values = shap_values(
        model,
        X,
        background=background,
        explainer=explainer,
        check_additivity=check_additivity,
        **kwargs,
    )
    grouped = values.groupby("feature", as_index=False).agg(
        importance=("shap_value", lambda item: float(np.mean(np.abs(item)))),
        mean_shap=("shap_value", "mean"),
        std_shap=("shap_value", "std"),
    )
    grouped["std_shap"] = grouped["std_shap"].fillna(0.0)
    grouped = grouped.sort_values(
        "importance", ascending=False, kind="stable"
    ).reset_index(drop=True)
    return _attach_schema(
        grouped,
        kind="shap_importance",
        model=model,
        method=f"shap_{_normalize_explainer(explainer)}_global_importance",
        n_features=_as_feature_frame(X).shape[1],
        metadata=values.attrs.get("macroforecast_metadata_schema", {}).get(
            "metadata", {}
        ),
    )


def shap_tree(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Tree SHAP global importance using the optional ``shap`` backend."""

    return shap_importance(model, X, explainer="tree", **kwargs)


def shap_linear(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Linear SHAP-style global importance using ``shap.Explainer``."""

    return shap_importance(model, X, explainer="auto", **kwargs)


def shap_kernel(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Kernel/permutation SHAP-style global importance."""

    return shap_importance(model, X, explainer="permutation", **kwargs)


def shap_deep(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Deep-model SHAP-style global importance.

    This callable uses the generic SHAP explainer path because deep backends
    vary by installed torch/shap version. Gradient-specific methods are exposed
    separately and require ``captum``.
    """

    return shap_importance(model, X, explainer="auto", **kwargs)


def anatomy_explain(
    anatomy: Any,
    *,
    model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None,
    transformer: Callable[..., Any] | Any | None = None,
    metric: str = "forecast",
    explanation_subset: pd.Index | Sequence[Any] | None = None,
    output: str = "long",
) -> pd.DataFrame:
    """Explain a precomputed ``anatomy.Anatomy`` object.

    This is a thin backend wrapper around the Python ``anatomy`` package from
    Borup et al., *The Anatomy of Out-of-Sample Forecasting Accuracy*.
    """

    anatomy_mod = _optional_anatomy()
    anatomy_obj = (
        anatomy_mod.Anatomy.load(str(anatomy))
        if isinstance(anatomy, (str, Path))
        else anatomy
    )
    if not hasattr(anatomy_obj, "explain"):
        raise TypeError("anatomy must be a precomputed anatomy.Anatomy object or path")

    groups_obj = (
        None
        if model_groups is None
        else anatomy_mod.AnatomyModelCombination(groups=dict(model_groups))
    )
    transformer_obj = _resolve_anatomy_transformer(
        anatomy_mod,
        transformer=transformer,
        metric=metric,
    )
    subset = None if explanation_subset is None else pd.Index(explanation_subset)
    wide = anatomy_obj.explain(
        model_sets=groups_obj,
        transformer=transformer_obj,
        explanation_subset=subset,
    )
    if str(output).lower() == "wide":
        table = wide.copy()
    elif str(output).lower() == "long":
        table = _anatomy_wide_to_long(wide)
    else:
        raise ValueError("output must be 'long' or 'wide'")
    return _attach_schema(
        table,
        kind="anatomy_explain",
        model=None,
        method=f"anatomy_{str(metric).lower()}",
        n_features=max(0, len(wide.columns) - int("base_contribution" in wide.columns)),
        metadata={
            "metric": str(metric),
            "backend": "anatomy",
            "model_groups": None if model_groups is None else dict(model_groups),
            "output": str(output).lower(),
            "paper": "Borup et al. (2022), The Anatomy of Out-of-Sample Forecasting Accuracy",
        },
    )


def shapley_variable_importance(
    contributions: pd.DataFrame | pd.Series,
    *,
    contribution_col: str | None = None,
    feature_col: str = "feature",
    group_col: str | None = None,
    exclude_base: bool = True,
    source: str = "contribution_table",
) -> pd.DataFrame:
    """Aggregate local Shapley contributions into variable importance.

    This matches the anatomy README's oShapley-VI summary rule: average the
    absolute raw-forecast Shapley contribution by predictor. The same table
    adapter can standardize user-supplied in-sample Shapley VI before MAS.
    """

    frame = _coerce_contribution_frame(
        contributions,
        value_name=contribution_col or "contribution",
        feature_col=feature_col,
    )
    value_col = contribution_col or _resolve_contribution_column(frame)
    if feature_col not in frame.columns:
        raise ValueError(f"contributions must contain feature column {feature_col!r}")
    frame = frame.copy()
    frame[feature_col] = frame[feature_col].astype(str)
    if exclude_base:
        frame = frame[~frame[feature_col].isin({"base_contribution", "__base__"})]
    resolved_group = group_col or _first_present(
        frame, ("model_set", "model", "pipeline")
    )
    group_keys = (
        [resolved_group, feature_col] if resolved_group is not None else [feature_col]
    )
    table = (
        frame.groupby(group_keys, dropna=False, as_index=False)
        .agg(
            importance=(value_col, lambda item: float(np.mean(np.abs(item)))),
            mean_contribution=(value_col, "mean"),
            std_contribution=(value_col, "std"),
            n_rows=(value_col, "count"),
        )
        .fillna({"std_contribution": 0.0})
        .sort_values(
            (
                [resolved_group, "importance"]
                if resolved_group is not None
                else ["importance"]
            ),
            ascending=([True, False] if resolved_group is not None else [False]),
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if resolved_group is not None:
        table["rank"] = table.groupby(resolved_group)["importance"].rank(
            ascending=False,
            method="first",
        )
    else:
        table["rank"] = np.arange(1, len(table) + 1)
    return _attach_schema(
        table,
        kind="shapley_variable_importance",
        model=None,
        method="mean_absolute_shapley_contribution",
        n_features=int(table[feature_col].nunique())
        if feature_col in table.columns
        else 0,
        metadata={
            "source": str(source),
            "contribution_column": value_col,
            "feature_column": feature_col,
            "group_column": resolved_group,
            "exclude_base": bool(exclude_base),
        },
    )


def ishapley_vi(
    contributions: pd.DataFrame | pd.Series,
    *,
    contribution_col: str | None = None,
    feature_col: str = "feature",
    group_col: str | None = None,
    exclude_base: bool = True,
) -> pd.DataFrame:
    """Aggregate in-sample Shapley contributions into iShapley-VI_p."""

    table = shapley_variable_importance(
        contributions,
        contribution_col=contribution_col,
        feature_col=feature_col,
        group_col=group_col,
        exclude_base=exclude_base,
        source="in_sample_contribution_table",
    )
    schema = table.attrs["macroforecast_metadata_schema"]
    schema["kind"] = "ishapley_vi"
    schema["method"] = "in_sample_mean_absolute_shapley_contribution"
    schema["reference"] = dict(_REFERENCE_CATALOG["ishapley_vi"])
    schema["metadata"]["vi_scope"] = "in_sample"
    return table


def oshapley_vi(
    anatomy: Any,
    *,
    model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None,
    explanation_subset: pd.Index | Sequence[Any] | None = None,
    exclude_base: bool = True,
) -> pd.DataFrame:
    """Compute anatomy backend oShapley-VI from raw forecast explanations."""

    explained = anatomy_explain(
        anatomy,
        model_groups=model_groups,
        metric="forecast",
        explanation_subset=explanation_subset,
        output="long",
    )
    table = shapley_variable_importance(
        explained,
        contribution_col="contribution",
        feature_col="feature",
        group_col="model_set",
        exclude_base=exclude_base,
        source="anatomy_raw_forecast",
    )
    table.attrs["macroforecast_metadata_schema"]["kind"] = "oshapley_vi"
    table.attrs["macroforecast_metadata_schema"]["method"] = (
        "anatomy_raw_forecast_mean_abs"
    )
    table.attrs["macroforecast_metadata_schema"]["reference"] = dict(
        _REFERENCE_CATALOG["oshapley_vi"]
    )
    return table


def pbsv(
    anatomy: Any,
    *,
    model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None,
    loss: str = "rmse",
    transformer: Callable[..., Any] | Any | None = None,
    explanation_subset: pd.Index | Sequence[Any] | None = None,
    output: str = "long",
) -> pd.DataFrame:
    """Compute backend PBSV_p loss decomposition through ``anatomy``."""

    table = anatomy_explain(
        anatomy,
        model_groups=model_groups,
        transformer=transformer,
        metric=loss,
        explanation_subset=explanation_subset,
        output=output,
    )
    table.attrs["macroforecast_metadata_schema"]["kind"] = "pbsv"
    table.attrs["macroforecast_metadata_schema"]["method"] = (
        f"anatomy_pbsv_{str(loss).lower()}"
    )
    table.attrs["macroforecast_metadata_schema"]["reference"] = dict(
        _REFERENCE_CATALOG["pbsv"]
    )
    table.attrs["macroforecast_metadata_schema"]["metadata"]["loss"] = str(loss)
    table.attrs["macroforecast_metadata_schema"]["metadata"]["backend"] = "anatomy"
    return table


def performance_based_shapley_value(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Alias for :func:`pbsv` when using the anatomy backend."""

    return pbsv(*args, **kwargs)


def model_accordance_score(
    is_vi: pd.DataFrame | pd.Series | Mapping[str, float],
    oos_pbsv: pd.DataFrame | pd.Series | Mapping[str, float],
    *,
    loss_type: str = "lower_is_better",
    mas_type: str = "importance_weighted",
    hypothesis_test: bool = True,
    h0_alpha: float = 0.5,
    n_samples: int = 1_000_000,
    vi_value_col: str | None = None,
    pbsv_value_col: str | None = None,
    feature_col: str = "feature",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Compute the anatomy backend Model Accordance Score."""

    anatomy_mod = _optional_anatomy()
    vi_series = _feature_series(
        is_vi,
        value_col=vi_value_col,
        feature_col=feature_col,
        preferred_columns=("importance", "vi", "contribution", "shap_value"),
    )
    pbsv_series = _feature_series(
        oos_pbsv,
        value_col=pbsv_value_col,
        feature_col=feature_col,
        preferred_columns=("pbsv", "contribution", "importance"),
    )
    vi_series, pbsv_series = _align_vi_pbsv_series(vi_series, pbsv_series)
    loss_enum = _anatomy_loss_type(anatomy_mod, loss_type)
    mas_enum = _anatomy_mas_type(anatomy_mod, mas_type)
    state = np.random.get_state() if random_state is not None else None
    if random_state is not None:
        np.random.seed(int(random_state))
    try:
        result = anatomy_mod.MAS(vi_series, pbsv_series, loss_enum).compute(
            mas_type=mas_enum,
            hypothesis_test=bool(hypothesis_test),
            h0_alpha=float(h0_alpha),
            n_samples=int(n_samples),
        )
    finally:
        if state is not None:
            np.random.set_state(state)
    table = pd.DataFrame(
        [
            {
                "mas": float(result["mas"]),
                "mas_p_value": (
                    float(result["mas_p_value"])
                    if "mas_p_value" in result and result["mas_p_value"] is not None
                    else np.nan
                ),
                "mas_type": str(mas_type),
                "loss_type": str(loss_type),
                "hypothesis_test": bool(hypothesis_test),
                "h0_alpha": float(h0_alpha),
                "n_samples": int(n_samples) if hypothesis_test else 0,
                "n_features": int(len(vi_series)),
            }
        ]
    )
    return _attach_schema(
        table,
        kind="model_accordance_score",
        model=None,
        method="anatomy_mas",
        n_features=int(len(vi_series)),
        metadata={
            "backend": "anatomy",
            "mas_type": str(mas_type),
            "loss_type": str(loss_type),
            "hypothesis_test": bool(hypothesis_test),
            "random_state": random_state,
        },
    )


def performance_shapley_value(
    contributions: pd.DataFrame,
    y: pd.Series | Sequence[float],
    *,
    loss: str = "squared_error",
    row_col: str | None = None,
    feature_col: str = "feature",
    contribution_col: str | None = None,
    base_col: str = "base_value",
    base_value: float = 0.0,
    n_permutations: int | None = None,
    max_exact_features: int = 8,
    random_state: int | None = 0,
    return_local: bool = False,
) -> pd.DataFrame:
    """Compute PBSV-style loss attribution from additive forecast contributions.

    Negative feature contributions reduce the selected point loss; positive
    contributions increase it. Full Borup et al. rolling/expanding refit anatomy
    should use ``anatomy_explain`` with a precomputed ``anatomy`` object.
    """

    frame = contributions.copy()
    if feature_col not in frame.columns:
        raise ValueError(f"contributions must contain feature column {feature_col!r}")
    value_col = contribution_col or _resolve_contribution_column(frame)
    if row_col is None and "row" not in frame.columns and "index" not in frame.columns:
        frame = frame.copy()
        frame["row"] = 0
    resolved_row = row_col or _resolve_contribution_row_column(frame, y)
    targets = _target_by_row(frame, y, row_col=resolved_row)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for row_key, group in frame.groupby(resolved_row, sort=False):
        group = group.copy()
        actual = float(targets[row_key])
        base = _row_base_value(
            group,
            base_col=base_col,
            default=base_value,
        )
        feature_group = (
            group[group[feature_col].astype(str) != "__base__"]
            .groupby(feature_col, sort=False, as_index=False)[value_col]
            .sum()
        )
        feature_names = tuple(str(item) for item in feature_group[feature_col])
        values = feature_group[value_col].to_numpy(dtype=float)
        shapley, meta = _point_loss_shapley(
            actual=actual,
            base_prediction=base,
            contributions=values,
            loss=loss,
            n_permutations=n_permutations,
            max_exact_features=max_exact_features,
            rng=rng,
        )
        full_prediction = float(base + values.sum())
        baseline_loss = _point_loss(actual, base, loss=loss)
        full_loss = _point_loss(actual, full_prediction, loss=loss)
        index_value = group["index"].iloc[0] if "index" in group.columns else row_key
        rows.append(
            {
                "row": row_key,
                "index": index_value,
                "feature": "__base__",
                "pbsv": baseline_loss,
                "actual": actual,
                "base_prediction": base,
                "full_prediction": full_prediction,
                "baseline_loss": baseline_loss,
                "full_loss": full_loss,
                "loss": str(loss),
                "is_base": True,
                **meta,
            }
        )
        for feature, value, contribution in zip(
            feature_names, values, shapley, strict=True
        ):
            rows.append(
                {
                    "row": row_key,
                    "index": index_value,
                    "feature": feature,
                    "forecast_contribution": float(value),
                    "pbsv": float(contribution),
                    "actual": actual,
                    "base_prediction": base,
                    "full_prediction": full_prediction,
                    "baseline_loss": baseline_loss,
                    "full_loss": full_loss,
                    "loss": str(loss),
                    "is_base": False,
                    **meta,
                }
            )
    local = pd.DataFrame(rows)
    error = (
        local.groupby("row", sort=False)["pbsv"].sum()
        - local.groupby("row", sort=False)["full_loss"].first()
    ).abs()
    if return_local:
        table = local.reset_index(drop=True)
        kind = "performance_shapley_value"
        method = "local_point_loss_shapley"
    else:
        table = (
            local.groupby(["feature", "is_base"], as_index=False)
            .agg(
                pbsv=("pbsv", "mean"),
                abs_pbsv=("pbsv", lambda item: float(np.mean(np.abs(item)))),
                n_rows=("row", "nunique"),
                baseline_loss=("baseline_loss", "mean"),
                full_loss=("full_loss", "mean"),
            )
            .sort_values(
                ["is_base", "abs_pbsv"], ascending=[False, False], kind="stable"
            )
            .reset_index(drop=True)
        )
        table["rank"] = np.arange(1, len(table) + 1)
        kind = "performance_shapley_value"
        method = "global_mean_point_loss_shapley"
    return _attach_schema(
        table,
        kind=kind,
        model=None,
        method=method,
        n_features=int(local.loc[~local["is_base"], "feature"].nunique()),
        metadata={
            "loss": str(loss),
            "contribution_column": value_col,
            "row_column": resolved_row,
            "n_rows": int(local["row"].nunique()),
            "n_permutations": n_permutations,
            "max_exact_features": int(max_exact_features),
            "max_efficiency_error": float(error.max()) if len(error) else 0.0,
            "paper": "Borup et al. (2022), The Anatomy of Out-of-Sample Forecasting Accuracy",
            "interpretation": (
                "negative pbsv reduces point loss; positive pbsv increases point loss"
            ),
            "backend": "macroforecast_additive_contribution_adapter",
        },
    )


def forecast_decomposition(
    model: Any,
    X: pd.DataFrame,
    *,
    row: int | str | pd.Timestamp = -1,
    sort: bool = True,
) -> pd.DataFrame:
    """Decompose one prediction into linear feature contributions."""

    frame = _as_feature_frame(X)
    selected = _select_row(frame, row)
    estimator = _estimator(model)
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        native = tree_importance(model, sort=sort)
        fallback = pd.DataFrame(
            {
                "feature": native["feature"].astype(str),
                "importance": native["importance"].astype(float),
                "feature_value": np.nan,
                "coefficient": np.nan,
                "contribution": np.nan,
                "abs_contribution": native["importance"].astype(float),
                "status": "tree_importance_fallback_not_additive",
            }
        )
        return _attach_schema(
            fallback.reset_index(drop=True),
            kind="forecast_decomposition",
            model=model,
            method="tree_importance_fallback_not_additive",
            n_features=len(fallback),
            metadata={
                "row": _jsonish_index(selected.name),
                "prediction_additivity": False,
                "warning": "Nonlinear estimators do not expose coefficient-level additive contributions here.",
            },
        )
    values = np.asarray(coef, dtype=float).reshape(-1)
    names = _feature_names(model, len(values))
    selected = selected.reindex(names, fill_value=0.0).astype(float)
    contribution = selected.to_numpy(dtype=float) * values
    table = pd.DataFrame(
        {
            "feature": names,
            "feature_value": selected.to_numpy(dtype=float),
            "coefficient": values,
            "contribution": contribution,
            "abs_contribution": np.abs(contribution),
        }
    )
    intercept = getattr(estimator, "intercept_", None)
    if intercept is not None:
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [
                        {
                            "feature": "__intercept__",
                            "feature_value": 1.0,
                            "coefficient": float(np.asarray(intercept).reshape(-1)[0]),
                            "contribution": float(np.asarray(intercept).reshape(-1)[0]),
                            "abs_contribution": abs(
                                float(np.asarray(intercept).reshape(-1)[0])
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if sort:
        table = table.sort_values("abs_contribution", ascending=False, kind="stable")
    return _attach_schema(
        table.reset_index(drop=True),
        kind="forecast_decomposition",
        model=model,
        method="linear_contribution",
        n_features=len(names),
        metadata={"row": _jsonish_index(selected.name)},
    )


def cumulative_r2_contribution(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    feature_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Sequential contribution of features to in-sample prediction R-squared."""

    frame = _as_feature_frame(X)
    target = np.asarray(y, dtype=float).reshape(-1)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    if feature_order is None:
        try:
            order = list(linear_coefficients(model)["feature"])
        except Exception:
            order = list(
                permutation_importance(model, frame, target, n_repeats=1)["feature"]
            )
    else:
        order = list(_resolve_features(frame, feature_order))
    active = pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    previous = _r2_score(target, np.repeat(float(np.mean(target)), len(target)))
    rows: list[dict[str, Any]] = []
    for step, feature in enumerate(order, start=1):
        active[feature] = frame[feature]
        current = _r2_score(target, _predict(model, active))
        rows.append(
            {
                "step": int(step),
                "feature": str(feature),
                "r2": float(current),
                "incremental_r2": float(current - previous),
                "cumulative_features": int(step),
            }
        )
        previous = current
    return _attach_schema(
        pd.DataFrame(rows),
        kind="cumulative_r2_contribution",
        model=model,
        method="sequential_zero_fill_prediction",
        n_features=frame.shape[1],
        metadata={
            "feature_order": order,
            "mode": "fixed_model_zero_fill_prediction",
            "refits_model": False,
        },
    )


def group_aggregate(
    table: pd.DataFrame,
    *,
    groups: Mapping[str, str | Sequence[str]] | None = None,
    group_column: str | None = None,
    value_column: str | None = None,
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Aggregate feature-level importance into user or metadata groups."""

    frame = table.copy()
    if "feature" not in frame.columns:
        raise ValueError("table must contain a 'feature' column")
    value = value_column or _infer_importance_column(frame)
    if group_column is not None:
        if group_column not in frame.columns:
            raise ValueError(f"group_column {group_column!r} is not in table")
        frame["group"] = frame[group_column].astype(str)
    else:
        mapping = _normalize_group_mapping(groups)
        frame["group"] = frame["feature"].map(
            lambda item: mapping.get(str(item), str(item).split("_")[0])
        )
    grouped = _aggregate_importance(
        frame, group_by="group", value_column=value, aggregation=aggregation
    )
    return _attach_schema(
        grouped,
        kind="group_aggregate",
        model=None,
        method=aggregation,
        n_features=int(frame["feature"].nunique()),
        metadata={"value_column": value, "aggregation": aggregation},
    )


def lineage_attribution(
    table: pd.DataFrame,
    lineage: Mapping[str, Any],
    *,
    level: str = "pipeline_name",
    value_column: str | None = None,
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Aggregate feature importance using feature-lineage metadata."""

    frame = table.copy()
    if "feature" not in frame.columns:
        raise ValueError("table must contain a 'feature' column")
    value = value_column or _infer_importance_column(frame)

    def resolve(feature: Any) -> str:
        meta = lineage.get(str(feature), {})
        if isinstance(meta, Mapping):
            return str(
                meta.get(level, meta.get("pipeline", meta.get("source", "unknown")))
            )
        return str(meta or "unknown")

    frame["lineage"] = frame["feature"].map(resolve)
    grouped = _aggregate_importance(
        frame, group_by="lineage", value_column=value, aggregation=aggregation
    )
    grouped = grouped.rename(columns={"lineage": level})
    return _attach_schema(
        grouped,
        kind="lineage_attribution",
        model=None,
        method=aggregation,
        n_features=int(frame["feature"].nunique()),
        metadata={"level": level, "value_column": value, "aggregation": aggregation},
    )


def transformation_attribution(
    evaluation: pd.DataFrame,
    *,
    pipeline_column: str | None = None,
    metric: str | None = None,
    method: str = "shapley_over_pipelines",
    target_columns: Sequence[str] = ("target", "horizon"),
    lower_is_better: bool = True,
    baseline: str | float = "worst",
) -> pd.DataFrame:
    """Attribute forecast score differences to preprocessing/feature pipelines.

    This helper works on mutually exclusive pipeline/model rows. It is not a
    component-level causal decomposition unless the input table was designed as
    a component-removal experiment. For loss metrics, the default converts loss
    to improvement relative to the worst observed pipeline in each group.
    """

    frame = evaluation.copy()
    pipeline_col = pipeline_column or _first_present(
        frame, ("pipeline", "pipeline_id", "model", "model_id")
    )
    if pipeline_col is None:
        raise ValueError("evaluation must contain a pipeline/model column")
    metric_col = metric or _first_present(
        frame, ("mse", "rmse", "mae", "loss", "score")
    )
    if metric_col is None:
        raise ValueError("evaluation must contain a metric column")
    if method not in {
        "shapley_over_pipelines",
        "marginal_addition",
        "leave_one_out_pipeline",
    }:
        raise ValueError(
            "method must be 'shapley_over_pipelines', 'marginal_addition', or 'leave_one_out_pipeline'"
        )
    group_cols = [column for column in target_columns if column in frame.columns]
    grouped_iter = (
        frame.groupby(group_cols, dropna=False) if group_cols else [((), frame)]
    )
    rows: list[dict[str, Any]] = []
    for key, group in grouped_iter:
        losses = (
            group.groupby(pipeline_col, as_index=True)[metric_col].mean().astype(float)
        )
        pipelines = list(losses.index.astype(str))
        values = losses.to_numpy(dtype=float)
        if len(pipelines) == 0:
            continue
        resolved_baseline = _resolve_pipeline_baseline(
            values, lower_is_better=lower_is_better, baseline=baseline
        )
        contribution = _pipeline_value_contribution(
            values,
            method=method,
            lower_is_better=lower_is_better,
            baseline=resolved_baseline,
        )
        utility = _pipeline_utility(
            values, lower_is_better=lower_is_better, baseline=resolved_baseline
        )
        base: dict[str, Any] = {}
        if group_cols:
            key_tuple = key if isinstance(key, tuple) else (key,)
            base = dict(zip(group_cols, key_tuple, strict=False))
        for pipeline, loss_value, utility_value, contrib in zip(
            pipelines, values, utility, contribution, strict=False
        ):
            rows.append(
                {
                    **base,
                    "pipeline": pipeline,
                    "loss": float(loss_value),
                    "utility": float(utility_value),
                    "contribution": float(contrib),
                    "baseline": float(resolved_baseline),
                    "method": method,
                    "metric": metric_col,
                    "lower_is_better": bool(lower_is_better),
                }
            )
    return _attach_schema(
        pd.DataFrame(rows),
        kind="transformation_attribution",
        model=None,
        method=method,
        n_features=0,
        metadata={
            "pipeline_column": pipeline_col,
            "metric": metric_col,
            "group_columns": group_cols,
            "lower_is_better": bool(lower_is_better),
            "baseline": baseline,
            "scale": "utility_improvement_from_baseline",
            "component_decomposition": False,
        },
    )


def attention_weights(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    *,
    add_intercept: bool = True,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """OLS attention weights ``Omega = X_test (X_train'X_train)^-1 X_train'``."""

    return _attention_weight_table(
        X_train,
        X_test,
        add_intercept=add_intercept,
        ridge=ridge,
        kind="attention_weights",
        method="ols_closed_form",
        metadata_extra={},
    )


def ols_attention_weights(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    *,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Exact OLS-as-attention weights from Goulet Coulombe (2026)."""

    return _attention_weight_table(
        X_train,
        X_test,
        add_intercept=add_intercept,
        ridge=0.0,
        kind="ols_attention_weights",
        method="ols_attention_exact",
        metadata_extra={
            "paper": "Ordinary Least Squares as an Attention Mechanism",
            "author": "Philippe Goulet Coulombe",
            "year": 2026,
            "ridge_extension": False,
        },
    )


def ridge_attention_weights(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    *,
    alpha: float = 1.0,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Ridge-stabilized OLS attention weights."""

    if float(alpha) < 0.0:
        raise ValueError("alpha must be non-negative")
    return _attention_weight_table(
        X_train,
        X_test,
        add_intercept=add_intercept,
        ridge=float(alpha),
        kind="ridge_attention_weights",
        method="ridge_attention_closed_form",
        metadata_extra={
            "paper": "Ordinary Least Squares as an Attention Mechanism",
            "author": "Philippe Goulet Coulombe",
            "year": 2026,
            "ridge_extension": True,
            "alpha": float(alpha),
        },
    )


def ols_attention_embedding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    *,
    add_intercept: bool = True,
    ridge: float = 0.0,
    tol: float = 1e-12,
) -> pd.DataFrame:
    """Return whitened train/test embeddings behind OLS-as-attention."""

    train = _as_feature_frame(X_train).astype(float)
    test = (
        train
        if X_test is None
        else _as_feature_frame(X_test)
        .reindex(columns=train.columns, fill_value=0.0)
        .astype(float)
    )
    train_matrix = _design_matrix(train, add_intercept=add_intercept)
    test_matrix = _design_matrix(test, add_intercept=add_intercept)
    gram = _attention_penalized_gram(
        train_matrix,
        add_intercept=add_intercept,
        ridge=ridge,
    )
    precision = np.linalg.pinv(gram)
    precision = (precision + precision.T) / 2.0
    values, vectors = np.linalg.eigh(precision)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    keep = values > float(tol)
    if not np.any(keep):
        raise ValueError("attention embedding has no positive precision components")
    values = values[keep]
    vectors = vectors[:, keep]
    transform = vectors * np.sqrt(values).reshape(1, -1)
    train_embedding = train_matrix @ transform
    test_embedding = test_matrix @ transform
    attention = test_embedding @ train_embedding.T
    rows: list[dict[str, Any]] = []
    for sample_name, labels, matrix in (
        ("train", train.index, train_embedding),
        ("test", test.index, test_embedding),
    ):
        for row_pos, row_index in enumerate(labels):
            for component_pos, value in enumerate(matrix[row_pos]):
                rows.append(
                    {
                        "sample": sample_name,
                        "row": int(row_pos),
                        "index": row_index,
                        "component": int(component_pos),
                        "value": float(value),
                        "precision_eigenvalue": float(values[component_pos]),
                    }
                )
    table = pd.DataFrame(rows)
    table.attrs["train_embedding"] = train_embedding
    table.attrs["test_embedding"] = test_embedding
    table.attrs["attention_matrix"] = attention
    table.attrs["precision_matrix"] = precision
    table.attrs["precision_eigenvalues"] = values
    return _attach_schema(
        table,
        kind="ols_attention_embedding",
        model=None,
        method="ols_attention_whitened_embedding",
        n_features=train.shape[1],
        metadata={
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "add_intercept": bool(add_intercept),
            "ridge": float(ridge),
            "intercept_penalized": False if add_intercept else None,
            "n_components": int(len(values)),
            "tol": float(tol),
            "identity": "attention_matrix = test_embedding @ train_embedding.T",
            "paper": "Ordinary Least Squares as an Attention Mechanism",
            "author": "Philippe Goulet Coulombe",
            "year": 2026,
        },
    )


def ols_attention_equivalence(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | None = None,
    *,
    reference_predictions: pd.Series | Sequence[float] | np.ndarray | None = None,
    add_intercept: bool = True,
    ridge: float = 0.0,
) -> pd.DataFrame:
    """Audit that closed-form predictions equal attention-weight predictions."""

    train = _as_feature_frame(X_train).astype(float)
    test = (
        train
        if X_test is None
        else _as_feature_frame(X_test)
        .reindex(columns=train.columns, fill_value=0.0)
        .astype(float)
    )
    target = _align_attention_target(y_train, train.index)
    weights = _attention_weight_table(
        train,
        test,
        add_intercept=add_intercept,
        ridge=ridge,
        kind="ols_attention_weights",
        method="ols_attention_exact" if float(ridge) == 0.0 else "ridge_attention_closed_form",
        metadata_extra={},
    )
    attention_matrix = weights.attrs["attention_matrix"]
    attention_prediction = attention_matrix @ target
    if reference_predictions is None:
        train_matrix = _design_matrix(train, add_intercept=add_intercept)
        test_matrix = _design_matrix(test, add_intercept=add_intercept)
        gram = _attention_penalized_gram(
            train_matrix,
            add_intercept=add_intercept,
            ridge=ridge,
        )
        coef = np.linalg.pinv(gram) @ train_matrix.T @ target
        reference = test_matrix @ coef
        reference_source = "closed_form_normal_equation"
    else:
        reference = np.asarray(reference_predictions, dtype=float).reshape(-1)
        if len(reference) != len(test):
            raise ValueError("reference_predictions must have the same length as X_test")
        reference_source = "user_supplied"
    rows = []
    for test_pos, test_index in enumerate(test.index):
        error = float(attention_prediction[test_pos] - reference[test_pos])
        rows.append(
            {
                "test_row": int(test_pos),
                "test_index": test_index,
                "attention_prediction": float(attention_prediction[test_pos]),
                "reference_prediction": float(reference[test_pos]),
                "equivalence_error": error,
                "abs_equivalence_error": abs(error),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["attention_matrix"] = attention_matrix
    return _attach_schema(
        table,
        kind="ols_attention_equivalence",
        model=None,
        method="ols_attention_prediction_equivalence",
        n_features=train.shape[1],
        metadata={
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "add_intercept": bool(add_intercept),
            "ridge": float(ridge),
            "reference_source": reference_source,
            "max_abs_equivalence_error": float(table["abs_equivalence_error"].max())
            if len(table)
            else 0.0,
            "paper": "Ordinary Least Squares as an Attention Mechanism",
            "author": "Philippe Goulet Coulombe",
            "year": 2026,
        },
    )


def _attention_weight_table(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
    *,
    add_intercept: bool,
    ridge: float,
    kind: str,
    method: str,
    metadata_extra: dict[str, Any],
) -> pd.DataFrame:
    train = _as_feature_frame(X_train).astype(float)
    test = (
        train
        if X_test is None
        else _as_feature_frame(X_test)
        .reindex(columns=train.columns, fill_value=0.0)
        .astype(float)
    )
    train_matrix = _design_matrix(train, add_intercept=add_intercept)
    test_matrix = _design_matrix(test, add_intercept=add_intercept)
    gram = _attention_penalized_gram(
        train_matrix,
        add_intercept=add_intercept,
        ridge=ridge,
    )
    omega = test_matrix @ np.linalg.pinv(gram) @ train_matrix.T
    rows: list[dict[str, Any]] = []
    for test_pos, test_index in enumerate(test.index):
        for train_pos, train_index in enumerate(train.index):
            rows.append(
                {
                    "test_row": int(test_pos),
                    "test_index": test_index,
                    "train_row": int(train_pos),
                    "train_index": train_index,
                    "weight": float(omega[test_pos, train_pos]),
                }
            )
    table = pd.DataFrame(rows)
    table.attrs["attention_matrix"] = omega
    metadata = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "add_intercept": bool(add_intercept),
        "ridge": float(ridge),
        "intercept_penalized": False if add_intercept else None,
    }
    metadata.update(metadata_extra)
    return _attach_schema(
        table,
        kind=kind,
        model=None,
        method=method,
        n_features=train.shape[1],
        metadata=metadata,
    )


def dual_decomposition(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | None = None,
    *,
    add_intercept: bool = True,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """Represent OLS predictions as weighted sums of training outcomes."""

    train = _as_feature_frame(X_train)
    target = np.asarray(y_train, dtype=float).reshape(-1)
    if len(train) != len(target):
        raise ValueError("X_train and y_train must have the same number of rows")
    weights = attention_weights(train, X_test, add_intercept=add_intercept, ridge=ridge)
    table = weights.copy()
    table["train_y"] = table["train_row"].map(lambda row: float(target[int(row)]))
    table["contribution"] = table["weight"] * table["train_y"]
    summary = table.groupby(["test_row", "test_index"], as_index=False).agg(
        prediction=("contribution", "sum"),
        gross_weight=("weight", lambda item: float(np.abs(item).sum())),
    )
    table.attrs["prediction_summary"] = summary
    return _attach_schema(
        table,
        kind="dual_decomposition",
        model=None,
        method="ols_dual_attention",
        n_features=train.shape[1],
        metadata={"add_intercept": bool(add_intercept), "ridge": float(ridge)},
    )


def observation_weights(
    model: Any | None,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    *,
    method: str = "auto",
    lambda_: float = 1e-8,
    kernel: str = "linear",
    sigma: float = 1.0,
    add_intercept: bool = False,
    ridge_penalty_scale: str = "n_train",
    normalize: bool = False,
) -> pd.DataFrame:
    """Compute DualML-style historical-observation weights for forecasts.

    This is the episode-based interpretation from Goulet Coulombe, Goebel, and
    Klieber (2024). It explains a forecast through training observations rather
    than through predictor variables. The implemented paper-aligned routes are:
    ridge/OLS, kernel ridge, and sklearn-style random forests.
    """

    train = _as_feature_frame(X_train).astype(float)
    test = (
        train
        if X_test is None
        else _as_feature_frame(X_test)
        .reindex(columns=train.columns, fill_value=0.0)
        .astype(float)
    )
    resolved = _resolve_dual_method(model, method=method, kernel=kernel)
    if resolved == "ridge":
        matrix = _ridge_observation_weight_matrix(
            train,
            test,
            lambda_=lambda_,
            add_intercept=add_intercept,
            penalty_scale=ridge_penalty_scale,
        )
    elif resolved == "krr":
        k_train = _dual_kernel_matrix(train, train, kernel=kernel, sigma=sigma)
        k_test = _dual_kernel_matrix(test, train, kernel=kernel, sigma=sigma)
        inv = np.linalg.pinv(k_train + float(lambda_) * np.eye(len(train)))
        matrix = k_test @ inv
    elif resolved == "random_forest":
        estimator = _estimator(model)
        if estimator is None:
            raise ValueError("random_forest observation weights require a fitted model")
        matrix = _random_forest_observation_weight_matrix(estimator, train, test)
    else:
        raise ValueError("method must be 'auto', 'ridge', 'ols', 'krr', or 'random_forest'")
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    rows = _long_weight_rows(matrix, train.index, test.index, method=resolved)
    table = pd.DataFrame(rows)
    table.attrs["weight_matrix"] = matrix
    return _attach_schema(
        table,
        kind="observation_weights",
        model=model,
        method=resolved,
        n_features=train.shape[1],
        metadata={
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "lambda": float(lambda_),
            "kernel": str(kernel),
            "sigma": float(sigma),
            "add_intercept": bool(add_intercept),
            "ridge_penalty_scale": str(ridge_penalty_scale),
            "normalize": bool(normalize),
            "paper": "Dual Interpretation of Machine Learning Forecasts",
            "implemented_routes": ["ridge", "krr", "random_forest"],
            "unsupported_routes": ["boosted_tree_axil", "nn_embedding_ridge", "classification_log_odds"],
        },
    )


def outcome_contributions(
    weights: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    center: bool = False,
    include_base: bool = False,
) -> pd.DataFrame:
    """Convert observation weights into historical-outcome contributions."""

    matrix, train_labels, test_labels = _weight_matrix_from_table(weights)
    target = _align_dual_target(y_train, train_labels)
    base = float(np.nanmean(target)) if center else 0.0
    values = target - base if center else target
    contrib = matrix * values.reshape(1, -1)
    rows: list[dict[str, Any]] = []
    for test_pos, test_index in enumerate(test_labels):
        prediction = float(np.nansum(contrib[test_pos]) + (base * matrix[test_pos].sum() if center else 0.0))
        if include_base and center:
            rows.append(
                {
                    "test_row": int(test_pos),
                    "test_index": test_index,
                    "train_row": -1,
                    "train_index": "__base__",
                    "weight": float(matrix[test_pos].sum()),
                    "train_y": float(base),
                    "centered_train_y": 0.0,
                    "contribution": float(base * matrix[test_pos].sum()),
                    "prediction": prediction,
                    "channel": "base",
                }
            )
        for train_pos, train_index in enumerate(train_labels):
            rows.append(
                {
                    "test_row": int(test_pos),
                    "test_index": test_index,
                    "train_row": int(train_pos),
                    "train_index": train_index,
                    "weight": float(matrix[test_pos, train_pos]),
                    "train_y": float(target[train_pos]),
                    "centered_train_y": float(values[train_pos]),
                    "contribution": float(contrib[test_pos, train_pos]),
                    "prediction": prediction,
                    "channel": "episode",
                }
            )
    table = pd.DataFrame(rows)
    return _attach_schema(
        table,
        kind="outcome_contributions",
        model=None,
        method="dual_data_portfolio_contribution",
        n_features=0,
        metadata={
            "n_train": int(len(train_labels)),
            "n_test": int(len(test_labels)),
            "center": bool(center),
            "include_base": bool(include_base),
            "base_value": float(base),
        },
    )


def data_portfolio_diagnostics(
    weights: pd.DataFrame,
    *,
    top_q: float = 0.05,
) -> pd.DataFrame:
    """Summarize DualML data-portfolio concentration, shorts, leverage, turnover."""

    matrix, train_labels, test_labels = _weight_matrix_from_table(weights)
    q_share = _top_q_share(top_q)
    k = max(1, int(np.ceil(matrix.shape[1] * q_share)))
    rows: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    for test_pos, test_index in enumerate(test_labels):
        row = matrix[test_pos]
        abs_sum = float(np.abs(row).sum())
        top_abs = float(np.sort(np.abs(row))[-k:].sum()) if len(row) else 0.0
        turnover = np.nan if previous is None else float(np.abs(row - previous).sum())
        rows.append(
            {
                "test_row": int(test_pos),
                "test_index": test_index,
                "concentration": np.nan if abs_sum == 0 else top_abs / abs_sum,
                "short_position": float(row[row < 0].sum()),
                "short_position_abs": float(np.abs(row[row < 0]).sum()),
                "leverage": float(row.sum()),
                "gross_leverage": abs_sum,
                "turnover": turnover,
                "top_q": float(q_share),
                "top_k": int(k),
                "n_train": int(len(train_labels)),
            }
        )
        previous = row
    return _attach_schema(
        pd.DataFrame(rows),
        kind="data_portfolio_diagnostics",
        model=None,
        method="dual_data_portfolio_metrics",
        n_features=0,
        metadata={
            "top_q": float(q_share),
            "top_k": int(k),
            "turnover": "sum_abs_weight_change_from_previous_forecast",
            "short_position": "signed_sum_of_negative_weights",
            "reference_code": "DualML FC_CR/FSP/FT metrics",
        },
    )


def top_episodes(
    weights: pd.DataFrame,
    *,
    y_train: pd.Series | np.ndarray | None = None,
    n: int = 10,
    sort_by: str = "abs_weight",
) -> pd.DataFrame:
    """Return the largest historical-episode weights for each forecast row."""

    if sort_by not in {"abs_weight", "weight", "contribution", "abs_contribution"}:
        raise ValueError(
            "sort_by must be 'abs_weight', 'weight', 'contribution', or 'abs_contribution'"
        )
    enriched = weights.copy()
    if "abs_weight" not in enriched.columns:
        enriched["abs_weight"] = enriched["weight"].abs()
    if y_train is not None and "contribution" not in enriched.columns:
        enriched = outcome_contributions(enriched, y_train)
        enriched["abs_weight"] = enriched["weight"].abs()
        enriched["abs_contribution"] = enriched["contribution"].abs()
    elif "contribution" in enriched.columns and "abs_contribution" not in enriched.columns:
        enriched["abs_contribution"] = enriched["contribution"].abs()
    rows: list[pd.DataFrame] = []
    for _, group in enriched.groupby(["test_row", "test_index"], dropna=False, sort=False):
        selected = group.sort_values(sort_by, ascending=False).head(int(n)).copy()
        selected["rank"] = np.arange(1, len(selected) + 1)
        rows.append(selected)
    table = pd.concat(rows, ignore_index=True) if rows else enriched.iloc[0:0].copy()
    return _attach_schema(
        table,
        kind="top_episodes",
        model=None,
        method="dual_top_episode_weights",
        n_features=0,
        metadata={"n": int(n), "sort_by": sort_by},
    )


def episode_group_weights(
    weights: pd.DataFrame,
    groups: Mapping[str, Sequence[Any]],
    *,
    y_train: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Aggregate historical-observation weights over named episode groups."""

    enriched = (
        outcome_contributions(weights, y_train)
        if y_train is not None and "contribution" not in weights.columns
        else weights.copy()
    )
    group_map: dict[Any, str] = {}
    for group, members in groups.items():
        for member in members:
            group_map[member] = str(group)
    enriched["episode_group"] = enriched["train_index"].map(group_map).fillna("other")
    agg_spec: dict[str, Any] = {
        "weight": ("weight", "sum"),
        "abs_weight": ("weight", lambda item: float(np.abs(item).sum())),
        "n_episodes": ("train_index", "nunique"),
    }
    if "contribution" in enriched.columns:
        agg_spec["contribution"] = ("contribution", "sum")
        agg_spec["abs_contribution"] = (
            "contribution",
            lambda item: float(np.abs(item).sum()),
        )
    table = (
        enriched.groupby(["test_row", "test_index", "episode_group"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["test_row", "episode_group"])
        .reset_index(drop=True)
    )
    return _attach_schema(
        table,
        kind="episode_group_weights",
        model=None,
        method="dual_episode_group_aggregation",
        n_features=0,
        metadata={"groups": {str(key): list(value) for key, value in groups.items()}},
    )


def accumulated_local_effect_2d(
    model: Any,
    X: pd.DataFrame,
    *,
    features: tuple[str, str],
    bins: int = 10,
) -> pd.DataFrame:
    """Second-order (two-feature) accumulated local effect (Apley-Zhu 2020).

    Computes the pure interaction ALE surface of two features: per 2D quantile
    cell it averages the second-order finite difference of the prediction over
    the cell corners, accumulates it over both axes, and removes the (weighted)
    main effects so the surface is the interaction not explained by the
    first-order ALEs. Returns a tidy table with the cell centres, the interaction
    ALE, and the cell counts. A purely additive model yields ~0 everywhere.
    """

    if len(features) != 2:
        raise ValueError("features must be a (feature_1, feature_2) pair")
    f1, f2 = str(features[0]), str(features[1])
    frame = _as_feature_frame(X)
    for feature in (f1, f2):
        if feature not in frame.columns:
            raise ValueError(f"feature {feature!r} is not in X")
    if bins <= 1:
        raise ValueError("bins must be greater than 1")
    x1 = frame[f1].astype(float).to_numpy()
    x2 = frame[f2].astype(float).to_numpy()
    e1 = np.unique(np.quantile(x1, np.linspace(0.0, 1.0, int(bins) + 1)))
    e2 = np.unique(np.quantile(x2, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(e1) < 3 or len(e2) < 3:
        raise ValueError("each feature needs at least two non-empty ALE bins")
    k1, k2 = len(e1) - 1, len(e2) - 1
    a1 = np.clip(np.searchsorted(e1[1:-1], x1, side="right"), 0, k1 - 1)
    a2 = np.clip(np.searchsorted(e2[1:-1], x2, side="right"), 0, k2 - 1)
    delta = np.zeros((k1, k2), dtype=float)
    counts = np.zeros((k1, k2), dtype=float)
    for i in range(k1):
        for l in range(k2):
            mask = (a1 == i) & (a2 == l)
            n_cell = int(mask.sum())
            counts[i, l] = n_cell
            if n_cell == 0:
                continue
            sub = frame.loc[mask]

            def _corner(v1: float, v2: float) -> np.ndarray:
                grid = sub.copy()
                grid[f1] = v1
                grid[f2] = v2
                return _predict(model, grid)

            diff = (_corner(e1[i + 1], e2[l + 1]) - _corner(e1[i], e2[l + 1])) - (
                _corner(e1[i + 1], e2[l]) - _corner(e1[i], e2[l])
            )
            delta[i, l] = float(np.mean(diff))
    accumulated = np.cumsum(np.cumsum(delta, axis=0), axis=1)
    total = counts.sum()
    weights = counts / total if total > 0 else np.full_like(counts, 1.0 / counts.size)
    row_w = weights.sum(axis=1, keepdims=True)
    col_w = weights.sum(axis=0, keepdims=True)
    grand = float(np.sum(accumulated * weights))
    row_mean = (accumulated * weights).sum(axis=1, keepdims=True) / np.maximum(row_w, 1e-12)
    col_mean = (accumulated * weights).sum(axis=0, keepdims=True) / np.maximum(col_w, 1e-12)
    interaction = accumulated - row_mean - col_mean + grand
    centers1 = 0.5 * (e1[:-1] + e1[1:])
    centers2 = 0.5 * (e2[:-1] + e2[1:])
    rows: list[dict[str, Any]] = []
    for i in range(k1):
        for l in range(k2):
            rows.append({
                "bin_1": int(i),
                "bin_2": int(l),
                "center_1": float(centers1[i]),
                "center_2": float(centers2[l]),
                "ale": float(interaction[i, l]),
                "count": int(counts[i, l]),
            })
    table = pd.DataFrame(rows)
    table.attrs["macroforecast_metadata"] = {
        "kind": "accumulated_local_effect_2d",
        "features": [f1, f2],
        "bins_1": int(k1),
        "bins_2": int(k2),
    }
    return table


def var_impulse_response(
    panel: Any,
    *,
    n_lag: int = 1,
    periods: int = 10,
    orthogonalized: bool = True,
    signif: float = 0.05,
    repl: int = 1000,
    seed: int | None = None,
    trend: str = "c",
) -> pd.DataFrame:
    """Impulse-response functions with Monte-Carlo bootstrap confidence bands.

    Fits a VAR(``n_lag``) on ``panel`` and returns a tidy table with, for each
    horizon/impulse/response triple, the (orthogonalised by default) impulse
    response and its ``1 - signif`` Monte-Carlo error band (statsmodels
    ``IRAnalysis.errband_mc``; R vars::irf with bootstrap CI). Macro IRFs are
    essentially always reported with such bands.
    """

    from statsmodels.tsa.api import VAR as _SMVAR

    frame = pd.DataFrame(panel)
    frame = frame.select_dtypes("number") if hasattr(frame, "select_dtypes") else frame
    names = [str(c) for c in frame.columns]
    k = len(names)
    result = _SMVAR(np.asarray(frame, dtype=float)).fit(maxlags=int(n_lag), trend=trend)
    irf = result.irf(int(periods))
    point = irf.orth_irfs if orthogonalized else irf.irfs
    lower, upper = irf.errband_mc(
        orth=bool(orthogonalized), repl=int(repl), signif=float(signif), seed=seed
    )
    point = np.asarray(point); lower = np.asarray(lower); upper = np.asarray(upper)
    rows: list[dict[str, Any]] = []
    for h in range(point.shape[0]):
        for response in range(k):
            for impulse in range(k):
                rows.append({
                    "horizon": int(h),
                    "impulse": names[impulse],
                    "response": names[response],
                    "irf": float(point[h, response, impulse]),
                    "lower": float(lower[h, response, impulse]),
                    "upper": float(upper[h, response, impulse]),
                })
    table = pd.DataFrame(rows)
    table.attrs["macroforecast_metadata"] = {
        "kind": "var_impulse_response",
        "orthogonalized": bool(orthogonalized),
        "signif": float(signif),
        "repl": int(repl),
        "n_lag": int(n_lag),
        "periods": int(periods),
        "names": names,
    }
    return table


def generalized_irf(
    model: Any, *, n_periods: int = 12, target: str | int | None = None
) -> pd.DataFrame:
    """Pesaran-Shin generalized impulse response importance for VAR models."""

    return _var_irf_table(
        model, n_periods=n_periods, target=target, method="generalized_irf"
    )


def orthogonalised_irf(
    model: Any, *, n_periods: int = 12, target: str | int | None = None
) -> pd.DataFrame:
    """Cholesky orthogonalised impulse response importance for VAR models."""

    return _var_irf_table(
        model, n_periods=n_periods, target=target, method="orthogonalised_irf"
    )


def fevd(
    model: Any, *, n_periods: int = 12, target: str | int | None = None
) -> pd.DataFrame:
    """Forecast error variance decomposition importance for VAR models."""

    results = _var_results(model)
    names = _var_names(results)
    target_pos = _target_position(names, target)
    try:
        decomp = np.asarray(results.fevd(int(n_periods)).decomp, dtype=float)
        # statsmodels shape is typically (equation, horizon, shock).
        values = decomp[target_pos, : int(n_periods), :].sum(axis=0)
        method = "statsmodels_fevd"
    except Exception:
        irf_obj = results.irf(int(n_periods))
        orth = np.asarray(getattr(irf_obj, "orth_irfs", irf_obj.irfs), dtype=float)
        horizons = orth[: int(n_periods)]
        squared = np.square(horizons[:, target_pos, :])
        cumulative = np.cumsum(squared, axis=0)
        denom = cumulative.sum(axis=1)
        denom = np.where(denom > 1e-12, denom, 1.0)
        decomp = cumulative / denom.reshape(-1, 1)
        values = decomp.sum(axis=0)
        method = "manual_orthogonalized_fevd"
    table = pd.DataFrame(
        {
            "feature": names[: len(values)],
            "importance": [float(abs(v)) for v in values],
            "coefficient": [None] * len(values),
            "status": "operational",
        }
    )
    return _attach_schema(
        table.sort_values("importance", ascending=False, kind="stable").reset_index(
            drop=True
        ),
        kind="fevd",
        model=model,
        method=method,
        n_features=len(values),
        metadata={"n_periods": int(n_periods), "target": names[target_pos]},
    )


def historical_decomposition(
    model: Any,
    *,
    max_lag: int = 12,
    target: str | int | None = None,
) -> pd.DataFrame:
    """Reduced-form VAR historical contribution summary."""

    results = _var_results(model)
    names = _var_names(results)
    target_pos = _target_position(names, target)
    ma = np.asarray(results.ma_rep(maxn=int(max_lag)), dtype=float)
    resid = np.asarray(results.resid, dtype=float)
    rows: list[dict[str, Any]] = []
    for shock_pos, name in enumerate(names):
        path = np.zeros(resid.shape[0], dtype=float)
        for t in range(resid.shape[0]):
            total = 0.0
            for lag in range(min(int(max_lag), t) + 1):
                total += float(
                    ma[lag, target_pos, shock_pos] * resid[t - lag, shock_pos]
                )
            path[t] = total
        rows.append(
            {
                "feature": str(name),
                "importance": float(np.mean(np.abs(path))),
                "mean_contribution": float(np.mean(path)),
                "max_abs_contribution": float(np.max(np.abs(path)))
                if len(path)
                else 0.0,
            }
        )
    return _attach_schema(
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True),
        kind="historical_decomposition",
        model=model,
        method="reduced_form_ma_residual_contribution",
        n_features=len(names),
        metadata={"max_lag": int(max_lag), "target": names[target_pos]},
    )


def lasso_inclusion_frequency(
    model: Any,
    X: pd.DataFrame | None = None,
    y: pd.Series | np.ndarray | None = None,
    *,
    fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None,
    n_bootstraps: int = 50,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Estimate coefficient nonzero frequency for lasso-style models."""

    if X is None or y is None or fit_func is None:
        coef = linear_coefficients(model, sort=False)
        coef["inclusion_frequency"] = (coef["coefficient"].abs() > 1e-9).astype(float)
        coef["importance"] = coef["inclusion_frequency"]
        coef.attrs["macroforecast_metadata_schema"]["kind"] = (
            "lasso_inclusion_frequency"
        )
        coef.attrs["macroforecast_metadata_schema"]["method"] = "single_fit_nonzero"
        return coef.sort_values(
            "importance", ascending=False, kind="stable"
        ).reset_index(drop=True)
    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive")
    rng = np.random.default_rng(random_state)
    counts = pd.Series(0.0, index=frame.columns)
    for _ in range(int(n_bootstraps)):
        sample_pos = rng.integers(0, len(frame), size=len(frame))
        sample_x = frame.iloc[sample_pos]
        sample_y = target.iloc[sample_pos]
        fit = fit_func(sample_x, sample_y)
        coef = linear_coefficients(fit, sort=False).set_index("feature")["coefficient"]
        counts = counts.add((coef.abs() > 1e-9).astype(float), fill_value=0.0)
    table = pd.DataFrame(
        {
            "feature": counts.index.astype(str),
            "inclusion_frequency": (counts / float(n_bootstraps)).to_numpy(dtype=float),
        }
    )
    table["importance"] = table["inclusion_frequency"]
    return _attach_schema(
        table.sort_values("importance", ascending=False, kind="stable").reset_index(
            drop=True
        ),
        kind="lasso_inclusion_frequency",
        model=model,
        method="bootstrap_nonzero_frequency",
        n_features=frame.shape[1],
        metadata={"n_bootstraps": int(n_bootstraps)},
    )


def mrf_gtvp(model: Any, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return Macroeconomic Random Forest GTVP coefficient paths.

    The vendored MacroRandomForest backend emits ``betas`` after prediction.
    This callable exposes those paths directly instead of reducing them to a
    static forest importance score.
    """

    estimator = _estimator(model)
    if getattr(estimator, "output_", None) is None and X is not None:
        _predict(model, _as_feature_frame(X))
    output = getattr(estimator, "output_", None)
    if not isinstance(output, Mapping):
        raise ValueError("mrf_gtvp requires a macro_random_forest fit after predict()")
    betas = output.get("betas")
    if betas is None:
        raise ValueError("macro_random_forest output does not contain 'betas'")
    arr = np.asarray(betas, dtype=float)
    if arr.ndim != 2:
        raise ValueError("macro_random_forest 'betas' must be a 2-D array")
    names = _mrf_beta_names(estimator, output, arr.shape[1])
    if X is not None and len(X) == arr.shape[0]:
        index = list(_as_feature_frame(X).index)
    else:
        index = list(range(arr.shape[0]))
    rows: list[dict[str, Any]] = []
    for row_pos, idx in enumerate(index):
        for col_pos, name in enumerate(names):
            coef = float(arr[row_pos, col_pos])
            rows.append(
                {
                    "row": int(row_pos),
                    "index": idx,
                    "feature": str(name),
                    "coefficient": coef,
                    "abs_coefficient": abs(coef),
                    "importance": abs(coef),
                }
            )
    table = pd.DataFrame(rows)
    summary = (
        table.groupby("feature", as_index=False)
        .agg(
            importance=("abs_coefficient", "mean"),
            mean_coefficient=("coefficient", "mean"),
            std_coefficient=("coefficient", "std"),
        )
        .fillna({"std_coefficient": 0.0})
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    table.attrs["summary"] = summary
    return _attach_schema(
        table,
        kind="mrf_gtvp",
        model=model,
        method="macro_random_forest_beta_path",
        n_features=max(0, len(names) - 1),
        metadata={"n_rows": int(arr.shape[0]), "beta_columns": names},
    )


def rolling_recompute(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    window: int | None = None,
    step: int | None = None,
    method: str = "permutation_importance",
    n_repeats: int = 1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Recompute feature importance on rolling evaluation windows."""

    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    if method not in {"permutation_importance", "permutation_importance_strobl"}:
        raise ValueError(
            "method must be 'permutation_importance' or 'permutation_importance_strobl'"
        )
    width = int(window or max(8, len(frame) // 4))
    if width <= 1:
        raise ValueError("window must be greater than 1")
    stride = int(step or max(1, width // 4))
    if stride <= 0:
        raise ValueError("step must be positive")
    rows: list[pd.DataFrame] = []
    for end in range(width, len(frame) + 1, stride):
        sub_x = frame.iloc[end - width : end]
        sub_y = target.iloc[end - width : end]
        if method == "permutation_importance_strobl":
            table = permutation_importance_strobl(
                model,
                sub_x,
                sub_y,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            table = permutation_importance(
                model,
                sub_x,
                sub_y,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        table = table.copy()
        table.insert(0, "window_end", sub_x.index[-1])
        table.insert(0, "window_start", sub_x.index[0])
        table.insert(0, "window_id", len(rows))
        rows.append(table)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return _attach_schema(
        result,
        kind="rolling_recompute",
        model=model,
        method=method,
        n_features=frame.shape[1],
        metadata={
            "window": width,
            "step": stride,
            "n_windows": len(rows),
            "refits_model": False,
            "mode": "fixed_model_window_diagnostic",
        },
    )


def bootstrap_jackknife(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None,
    n_replications: int = 50,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Bootstrap or jackknife-style uncertainty summary for importance."""

    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    if n_replications <= 0:
        raise ValueError("n_replications must be positive")
    rng = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {str(column): [] for column in frame.columns}
    mode = "refit_coefficients" if fit_func is not None else "fixed_model_permutation"
    for _ in range(int(n_replications)):
        sample_pos = rng.integers(0, len(frame), size=len(frame))
        sample_x = frame.iloc[sample_pos]
        sample_y = target.iloc[sample_pos]
        if fit_func is not None:
            fit = fit_func(sample_x, sample_y)
            table = linear_coefficients(fit, sort=False)
            series = table.set_index("feature")["abs_coefficient"]
        else:
            table = permutation_importance(
                model,
                sample_x,
                sample_y,
                n_repeats=1,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            series = table.set_index("feature")["importance"]
        for feature in values:
            values[feature].append(float(series.get(feature, 0.0)))
    rows = []
    for feature, draws in values.items():
        arr = np.asarray(draws, dtype=float)
        rows.append(
            {
                "feature": feature,
                "importance": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "lower": float(np.quantile(arr, 0.05)),
                "upper": float(np.quantile(arr, 0.95)),
                "n_replications": int(n_replications),
            }
        )
    return _attach_schema(
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True),
        kind="bootstrap_jackknife",
        model=model,
        method=mode,
        n_features=frame.shape[1],
        metadata={
            "n_replications": int(n_replications),
            "mode": mode,
            "resampling": "bootstrap_with_replacement",
            "jackknife": False,
        },
    )


def gradient_attribution(
    model: Any,
    X: pd.DataFrame,
    *,
    method: str = "saliency_map",
    baseline: float | pd.DataFrame | np.ndarray | None = None,
    n_steps: int = 50,
    n_samples: int = 20,
    noise_scale: float = 0.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Gradient attribution for torch-backed models."""

    key = str(method).lower().replace("-", "_")
    if key not in {
        "saliency_map",
        "integrated_gradients",
        "deep_lift",
        "gradient_shap",
    }:
        raise ValueError(
            "method must be 'saliency_map', 'integrated_gradients', 'deep_lift', or 'gradient_shap'"
        )
    torch, torch_model, tensor, feature_names = _torch_attribution_context(model, X)
    baseline_tensor, baseline_space = _baseline_tensor(
        torch,
        tensor,
        baseline,
        estimator=_estimator(model),
        feature_names=feature_names,
    )
    if key == "deep_lift":
        try:
            captum = import_module("captum.attr")
        except ImportError as exc:
            raise ImportError(
                "deep_lift requires captum; install macroforecast[deep]"
            ) from exc
        # Backend wrapper: DeepLift is delegated to Captum.
        attr_obj = captum.DeepLift(torch_model)
        attribution = attr_obj.attribute(tensor, baselines=baseline_tensor)
    elif key == "saliency_map":
        # Captum-style manual path: raw input gradients. Captum Saliency
        # defaults to absolute gradients; macroforecast reports signed mean
        # attribution and mean-absolute importance separately.
        attribution = _torch_gradient(torch, torch_model, tensor)
    elif key == "integrated_gradients":
        # Captum-style manual path: straight-line Riemann approximation.
        # Captum's default implementation uses gausslegendre integration.
        attribution = _manual_integrated_gradients(
            torch,
            torch_model,
            tensor,
            baseline_tensor,
            n_steps=max(1, int(n_steps)),
        )
    else:
        # Captum-style manual path: expected-gradient approximation with
        # optional noise; not a direct Captum GradientShap backend call.
        attribution = _manual_gradient_shap(
            torch,
            torch_model,
            tensor,
            baseline_tensor,
            n_samples=max(1, int(n_samples)),
            noise_scale=float(noise_scale),
            random_state=random_state,
        )
    values = attribution.detach().cpu().numpy()
    if values.ndim == 3:
        # Recurrent models: aggregate over sequence positions for each feature.
        feature_values = values.mean(axis=1)
    else:
        feature_values = values
    rows = []
    for pos, feature in enumerate(feature_names):
        column = feature_values[:, pos]
        rows.append(
            {
                "feature": str(feature),
                "importance": float(np.mean(np.abs(column))),
                "mean_attribution": float(np.mean(column)),
                "std_attribution": float(np.std(column, ddof=1))
                if len(column) > 1
                else 0.0,
                "method": key,
            }
        )
    return _attach_schema(
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True),
        kind=key,
        model=model,
        method=key,
        n_features=len(feature_names),
        metadata={
            "n_obs": int(_as_feature_frame(X).shape[0]),
            "n_steps": int(n_steps),
            "n_samples": int(n_samples),
            "noise_scale": float(noise_scale),
            "baseline_space": baseline_space,
        },
    )


def saliency_map(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Vanilla input-gradient attribution for torch-backed models."""

    return gradient_attribution(model, X, method="saliency_map")


def integrated_gradients(
    model: Any,
    X: pd.DataFrame,
    *,
    baseline: float | pd.DataFrame | np.ndarray | None = None,
    n_steps: int = 50,
) -> pd.DataFrame:
    """Integrated gradients for torch-backed models."""

    return gradient_attribution(
        model,
        X,
        method="integrated_gradients",
        baseline=baseline,
        n_steps=n_steps,
    )


def gradient_shap(
    model: Any,
    X: pd.DataFrame,
    *,
    baseline: float | pd.DataFrame | np.ndarray | None = None,
    n_samples: int = 20,
    noise_scale: float = 0.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Expected-gradients approximation to GradientSHAP."""

    return gradient_attribution(
        model,
        X,
        method="gradient_shap",
        baseline=baseline,
        n_samples=n_samples,
        noise_scale=noise_scale,
        random_state=random_state,
    )


def deep_lift(
    model: Any,
    X: pd.DataFrame,
    *,
    baseline: float | pd.DataFrame | np.ndarray | None = None,
) -> pd.DataFrame:
    """DeepLift attribution through Captum for torch-backed models."""

    return gradient_attribution(model, X, method="deep_lift", baseline=baseline)


def lstm_hidden_state(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """LSTM/GRU hidden-unit activation importance for torch-backed models."""

    estimator = _estimator(model)
    if getattr(estimator, "kind", None) == "transformer":
        raise NotImplementedError(
            "lstm_hidden_state is only defined for LSTM/GRU models"
        )
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "lstm_hidden_state requires torch; install macroforecast[deep]"
        ) from exc
    torch_model = getattr(estimator, "model_", None) or getattr(
        estimator, "_model", None
    )
    if torch_model is None:
        raise NotImplementedError("lstm_hidden_state requires a fitted torch model")
    rnn = getattr(torch_model, "rnn", None) or getattr(torch_model, "cell", None)
    if rnn is None:
        raise NotImplementedError(
            "lstm_hidden_state requires a model with an LSTM/GRU recurrent cell"
        )
    frame = _as_feature_frame(X)
    if getattr(estimator, "feature_names_in_", None):
        frame = frame.reindex(columns=list(estimator.feature_names_in_), fill_value=0.0)
    values = frame.astype(float).to_numpy(dtype=float)
    x_mean = getattr(estimator, "x_mean_", None)
    x_scale = getattr(estimator, "x_scale_", None)
    if x_mean is not None and x_scale is not None:
        values = (values - np.asarray(x_mean, dtype=float)) / np.asarray(
            x_scale, dtype=float
        )
    sequence_length = int(getattr(estimator, "sequence_length", 1))
    prefix = getattr(estimator, "train_tail_", None)
    prefix = (
        np.empty((0, values.shape[1]))
        if prefix is None
        else np.asarray(prefix, dtype=float)
    )
    combined = np.vstack([prefix, values])
    seq = np.stack([combined[i : i + sequence_length] for i in range(len(values))])
    captured: list[Any] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output[0].detach().cpu())

    handle = rnn.register_forward_hook(hook)
    try:
        torch_model.eval()
        device = torch.device(getattr(estimator, "device_", "cpu") or "cpu")
        with torch.no_grad():
            torch_model(torch.tensor(seq, dtype=torch.float32, device=device))
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("lstm_hidden_state did not capture recurrent activations")
    out = captured[0]
    importance = out.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    table = pd.DataFrame(
        {
            "feature": [f"hidden_unit_{i}" for i in range(len(importance))],
            "importance": [float(v) for v in importance],
            "coefficient": [None] * len(importance),
        }
    )
    return _attach_schema(
        table,
        kind="lstm_hidden_state",
        model=model,
        method="torch_recurrent_forward_hook",
        n_features=frame.shape[1],
        metadata={"hidden_size": int(len(importance)), "n_obs": int(len(frame))},
    )


def custom_interpretation(
    model: Any,
    X: pd.DataFrame,
    func: Callable[..., Any],
    *,
    y: pd.Series | np.ndarray | None = None,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **params: Any,
) -> pd.DataFrame:
    """Run a user-supplied interpretation callable and attach metadata."""

    frame = _as_feature_frame(X)
    resolved_name = str(name or _callable_name(func) or "custom_interpretation")
    result = func(
        model,
        frame,
        y=y,
        metadata=dict(metadata or {}),
        **params,
    )
    table = _coerce_custom_table(result)
    return _attach_schema(
        table,
        kind="custom_interpretation",
        model=model,
        method=resolved_name,
        n_features=frame.shape[1],
        metadata={
            "name": resolved_name,
            "callable": _callable_name(func),
            "params": dict(params),
            "n_obs": int(len(frame)),
            "has_target": y is not None,
            "user_metadata": dict(metadata or {}),
        },
    )


def _estimator(model: Any) -> Any:
    return model.estimator if isinstance(model, ModelFit) else model


def _resolve_dual_method(model: Any, *, method: str, kernel: str) -> str:
    key = str(method).lower().replace("-", "_")
    if key == "auto":
        estimator = _estimator(model)
        if estimator is not None and hasattr(estimator, "estimators_"):
            return "random_forest"
        return "krr" if str(kernel).lower().replace("-", "_") != "linear" else "ridge"
    aliases = {
        "ols": "ridge",
        "rr": "ridge",
        "linear_ridge": "ridge",
        "kernel_ridge": "krr",
        "kernel": "krr",
        "rf": "random_forest",
        "forest": "random_forest",
        "randomforest": "random_forest",
    }
    return aliases.get(key, key)


def _ridge_observation_weight_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    lambda_: float,
    add_intercept: bool,
    penalty_scale: str,
) -> np.ndarray:
    train_matrix = _design_matrix(train, add_intercept=add_intercept)
    test_matrix = _design_matrix(test, add_intercept=add_intercept)
    gram = train_matrix.T @ train_matrix
    scale_key = str(penalty_scale).lower().replace("-", "_")
    if scale_key in {"n", "n_train", "sample", "samples"}:
        scale = float(len(train_matrix))
    elif scale_key in {"none", "one", "1"}:
        scale = 1.0
    else:
        raise ValueError("ridge_penalty_scale must be 'n_train' or 'none'")
    penalty = float(lambda_) * scale * np.eye(gram.shape[0])
    if add_intercept and penalty.shape[0] > 0:
        # DualML_R/DualML.py use the paper's no-intercept standardized design
        # by default. When macroforecast users add an intercept, keep that
        # intercept unpenalized to match standard ridge regression convention.
        penalty[0, 0] = 0.0
    return test_matrix @ np.linalg.pinv(gram + penalty) @ train_matrix.T


def _dual_kernel_matrix(
    A: pd.DataFrame,
    B: pd.DataFrame,
    *,
    kernel: str,
    sigma: float,
) -> np.ndarray:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    key = str(kernel).lower().replace("-", "_")
    if key in {"linear", "dot"}:
        return left @ right.T
    diff = left[:, None, :] - right[None, :, :]
    if key in {"gaussian", "rbf"}:
        # Matches the local DualML Python/R code: exp(-sigma * squared distance).
        return np.exp(-float(sigma) * np.sum(diff**2, axis=2))
    if key in {"laplace", "laplacian"}:
        # Matches kernlab::laplacedot / local DualML code: exp(-sigma * L1 distance).
        return np.exp(-float(sigma) * np.sum(np.abs(diff), axis=2))
    raise ValueError("kernel must be 'linear', 'gaussian'/'rbf', or 'laplace'/'laplacian'")


def _random_forest_observation_weight_matrix(
    estimator: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    if not hasattr(estimator, "estimators_"):
        raise TypeError("random_forest observation weights require estimators_")
    x_train = train.to_numpy(dtype=float)
    x_test = test.to_numpy(dtype=float)
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    matrix = np.zeros((n_test, n_train), dtype=float)
    bootstrap_samples = _forest_bootstrap_samples(estimator, n_train)
    for tree_pos, tree in enumerate(estimator.estimators_):
        train_leaf = np.asarray(tree.apply(x_train)).reshape(-1)
        test_leaf = np.asarray(tree.apply(x_test)).reshape(-1)
        if bootstrap_samples is None:
            sample_count = np.ones(n_train, dtype=float)
        else:
            sample_count = np.bincount(bootstrap_samples[tree_pos], minlength=n_train).astype(float)
        for test_pos, leaf in enumerate(test_leaf):
            mask = train_leaf == leaf
            counts = sample_count * mask
            total = float(counts.sum())
            if total > 0:
                matrix[test_pos] += counts / total
    matrix /= max(1, len(estimator.estimators_))
    return matrix


def _forest_bootstrap_samples(estimator: Any, n_train: int) -> list[np.ndarray] | None:
    samples = getattr(estimator, "estimators_samples_", None)
    if samples is not None:
        return [np.asarray(sample, dtype=int) for sample in samples]
    if getattr(estimator, "bootstrap", True) is False:
        return None
    try:
        from sklearn.ensemble._forest import (  # type: ignore
            _generate_sample_indices,
            _get_n_samples_bootstrap,
        )
    except Exception:
        return None
    n_bootstrap = _get_n_samples_bootstrap(n_train, getattr(estimator, "max_samples", None))
    return [
        _generate_sample_indices(getattr(tree, "random_state", None), n_train, n_bootstrap)
        for tree in estimator.estimators_
    ]


def _long_weight_rows(
    matrix: np.ndarray,
    train_index: pd.Index,
    test_index: pd.Index,
    *,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_pos, test_label in enumerate(test_index):
        for train_pos, train_label in enumerate(train_index):
            weight = float(matrix[test_pos, train_pos])
            rows.append(
                {
                    "test_row": int(test_pos),
                    "test_index": test_label,
                    "train_row": int(train_pos),
                    "train_index": train_label,
                    "weight": weight,
                    "abs_weight": abs(weight),
                    "channel": method,
                }
            )
    return rows


def _weight_matrix_from_table(
    weights: pd.DataFrame,
) -> tuple[np.ndarray, list[Any], list[Any]]:
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame")
    if {"test_row", "train_row", "weight"}.issubset(weights.columns):
        frame = weights.copy()
        frame = frame[frame["train_row"].astype(int) >= 0]
        train = (
            frame[["train_row", "train_index"]]
            .drop_duplicates("train_row")
            .sort_values("train_row")
        )
        test = (
            frame[["test_row", "test_index"]]
            .drop_duplicates("test_row")
            .sort_values("test_row")
        )
        pivot = frame.pivot_table(
            index="test_row",
            columns="train_row",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(index=test["test_row"], columns=train["train_row"], fill_value=0.0)
        return (
            pivot.to_numpy(dtype=float),
            train["train_index"].tolist(),
            test["test_index"].tolist(),
        )
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("weights must be long-form or a 2D weight matrix")
    train_labels = list(weights.columns)
    test_labels = list(weights.index)
    return matrix, train_labels, test_labels


def _align_dual_target(y_train: pd.Series | np.ndarray, train_labels: Sequence[Any]) -> np.ndarray:
    if isinstance(y_train, pd.Series):
        if set(train_labels).issubset(set(y_train.index)):
            values = y_train.reindex(train_labels).to_numpy(dtype=float)
        else:
            values = y_train.to_numpy(dtype=float)
    else:
        values = np.asarray(y_train, dtype=float).reshape(-1)
    if len(values) != len(train_labels):
        raise ValueError("y_train length must match the number of training observations")
    return values.astype(float, copy=False)


def _top_q_share(top_q: float) -> float:
    q = float(top_q)
    if q <= 0:
        raise ValueError("top_q must be positive")
    if q > 1:
        q = q / 100.0
    return min(q, 1.0)


def _coerce_fit(model: Any) -> Any:
    return model


def _feature_names(model: Any, n_features: int) -> list[str]:
    if isinstance(model, ModelFit) and model.feature_names:
        return list(model.feature_names)
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) == n_features:
        return [str(name) for name in names]
    return [f"x{i}" for i in range(n_features)]


def _as_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    return X.copy()


def _resolve_features(
    frame: pd.DataFrame, features: Iterable[str] | str
) -> tuple[str, ...]:
    selected: tuple[str, ...]
    if isinstance(features, str):
        selected = (features,)
    else:
        selected = tuple(str(feature) for feature in features)
    missing = [feature for feature in selected if feature not in frame.columns]
    if missing:
        raise ValueError(f"features are not in X: {missing}")
    return selected


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, ModelFit):
        return model.predict(X).to_numpy(dtype=float)
    if not hasattr(model, "predict"):
        raise ValueError("model must expose predict() or be a ModelFit")
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def _safe_qcut(values: pd.Series, n_bins: int) -> pd.Series:
    try:
        bins = pd.qcut(values, q=int(n_bins), labels=False, duplicates="drop")
        return pd.Series(bins, index=values.index)
    except Exception:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)


def _grid_values(values: pd.Series, grid_size: int) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("grid feature contains no finite values")
    quantiles = np.linspace(0.0, 1.0, int(grid_size))
    grid = np.unique(np.quantile(clean.to_numpy(dtype=float), quantiles))
    if len(grid) == 1:
        return np.repeat(grid[0], int(grid_size))
    return grid


def _select_row(frame: pd.DataFrame, row: int | str | pd.Timestamp) -> pd.Series:
    if isinstance(row, int):
        selected = frame.iloc[row].copy()
    else:
        selected = frame.loc[row].copy()
    if isinstance(selected, pd.DataFrame):
        selected = selected.iloc[0]
    return selected


def _jsonish_index(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    residual = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-15:
        return 0.0
    return float(1.0 - np.sum(residual**2) / denom)


def _infer_importance_column(frame: pd.DataFrame) -> str:
    for column in (
        "importance",
        "abs_contribution",
        "abs_coefficient",
        "contribution",
        "coefficient",
    ):
        if column in frame.columns:
            return column
    raise ValueError("table must contain an importance-like column")


def _normalize_group_mapping(
    groups: Mapping[str, str | Sequence[str]] | None,
) -> dict[str, str]:
    if groups is None:
        return {}
    mapping: dict[str, str] = {}
    for key, value in groups.items():
        if isinstance(value, str):
            mapping[str(key)] = value
        else:
            for feature in value:
                mapping[str(feature)] = str(key)
    return mapping


def _aggregate_importance(
    frame: pd.DataFrame,
    *,
    group_by: str,
    value_column: str,
    aggregation: str,
) -> pd.DataFrame:
    if aggregation == "sum":
        grouped = frame.groupby(group_by, as_index=False)[value_column].sum()
    elif aggregation == "mean":
        grouped = frame.groupby(group_by, as_index=False)[value_column].mean()
    elif aggregation == "max_abs":
        grouped = (
            frame.assign(__abs__=frame[value_column].abs())
            .groupby(group_by, as_index=False)["__abs__"]
            .max()
            .rename(columns={"__abs__": value_column})
        )
    elif aggregation == "signed_sum":
        grouped = frame.groupby(group_by, as_index=False)[value_column].sum()
    else:
        raise ValueError(
            "aggregation must be 'sum', 'mean', 'max_abs', or 'signed_sum'"
        )
    grouped = grouped.rename(columns={value_column: "importance"})
    return grouped.sort_values(
        "importance", ascending=False, kind="stable"
    ).reset_index(drop=True)


def _first_present(frame: pd.DataFrame, columns: Sequence[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _resolve_pipeline_baseline(
    values: np.ndarray,
    *,
    lower_is_better: bool,
    baseline: str | float,
) -> float:
    arr = np.asarray(values, dtype=float)
    if isinstance(baseline, (int, float)):
        return float(baseline)
    key = str(baseline).lower().replace("-", "_")
    if key == "worst":
        return float(np.max(arr) if lower_is_better else np.min(arr))
    if key == "best":
        return float(np.min(arr) if lower_is_better else np.max(arr))
    if key == "mean":
        return float(np.mean(arr))
    raise ValueError("baseline must be 'worst', 'best', 'mean', or a numeric value")


def _pipeline_utility(
    values: np.ndarray, *, lower_is_better: bool, baseline: float
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return baseline - arr if lower_is_better else arr - baseline


def _pipeline_value_contribution(
    values: np.ndarray,
    *,
    method: str,
    lower_is_better: bool,
    baseline: float,
) -> np.ndarray:
    n_items = len(values)
    utility = _pipeline_utility(
        values, lower_is_better=lower_is_better, baseline=baseline
    )
    if method == "marginal_addition":
        return utility
    if method == "leave_one_out_pipeline":
        full = float(np.mean(utility))
        out = np.zeros(n_items, dtype=float)
        for idx in range(n_items):
            without = np.delete(utility, idx)
            out[idx] = float(full - np.mean(without)) if len(without) else 0.0
        return out
    shapley = np.zeros(n_items, dtype=float)
    indices = list(range(n_items))
    for size in range(n_items):
        for subset in combinations(indices, size):
            subset_set = set(subset)
            subset_value = float(np.mean(utility[list(subset)])) if subset else 0.0
            weight = 1.0 / (n_items * comb(n_items - 1, size))
            for idx in indices:
                if idx in subset_set:
                    continue
                new_subset = list(subset) + [idx]
                new_value = float(np.mean(utility[new_subset]))
                shapley[idx] += weight * (new_value - subset_value)
    return shapley


def _design_matrix(frame: pd.DataFrame, *, add_intercept: bool) -> np.ndarray:
    matrix = frame.to_numpy(dtype=float)
    if add_intercept:
        matrix = np.column_stack([np.ones(len(frame), dtype=float), matrix])
    return matrix


def _attention_penalized_gram(
    train_matrix: np.ndarray,
    *,
    add_intercept: bool,
    ridge: float,
) -> np.ndarray:
    if float(ridge) < 0.0:
        raise ValueError("ridge must be non-negative")
    gram = train_matrix.T @ train_matrix
    if float(ridge) > 0.0:
        penalty = float(ridge) * np.eye(gram.shape[0], dtype=float)
        if add_intercept and penalty.shape[0] > 0:
            # Goulet Coulombe's OLS-attention algebra is no-intercept in its
            # clean matrix form. Macroforecast exposes the common regression
            # intercept convention: include the constant column in the design,
            # but do not shrink that constant when ridge-stabilizing the
            # attention matrix. This matches sklearn/R ridge practice and keeps
            # affine weights summing to one for intercept models.
            penalty[0, 0] = 0.0
        gram = gram + penalty
    return gram


def _align_attention_target(
    y_train: pd.Series | np.ndarray,
    train_index: pd.Index,
) -> np.ndarray:
    if isinstance(y_train, pd.Series):
        target = y_train.reindex(train_index).to_numpy(dtype=float)
    else:
        target = np.asarray(y_train, dtype=float).reshape(-1)
    if len(target) != len(train_index):
        raise ValueError("X_train and y_train must have the same number of rows")
    if not np.all(np.isfinite(target)):
        raise ValueError("y_train must be finite after alignment to X_train")
    return target


def _var_results(model: Any) -> Any:
    estimator = _estimator(model)
    candidates = [
        estimator,
        getattr(estimator, "_results", None),
        getattr(getattr(estimator, "_var", None), "_results", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "irf"):
            return candidate
    if all(hasattr(estimator, attr) for attr in ("coef_", "names_", "n_lag")):
        return _InternalVARResults(estimator)
    raise ValueError("model does not expose fitted VAR results")


class _InternalVARIRF:
    def __init__(self, irfs: np.ndarray, sigma_u: np.ndarray) -> None:
        self.irfs = np.asarray(irfs, dtype=float)
        try:
            chol = np.linalg.cholesky(_positive_definite_covariance(sigma_u))
        except np.linalg.LinAlgError:
            chol = np.eye(self.irfs.shape[1], dtype=float)
        self.orth_irfs = np.asarray([phi @ chol for phi in self.irfs], dtype=float)


class _InternalVARFEVD:
    def __init__(self, ma: np.ndarray, sigma_u: np.ndarray, n_periods: int) -> None:
        irf = _InternalVARIRF(ma[: int(n_periods)], sigma_u)
        orth = irf.orth_irfs
        k = orth.shape[1]
        out = np.zeros((k, int(n_periods), k), dtype=float)
        for horizon in range(int(n_periods)):
            squared = np.square(orth[: horizon + 1]).sum(axis=0)
            denom = squared.sum(axis=1)
            denom = np.where(denom > 1e-12, denom, 1.0)
            out[:, horizon, :] = squared / denom.reshape(-1, 1)
        self.decomp = out


class _InternalVARResults:
    """Minimal statsmodels-like VAR result adapter for macroforecast's internal VAR."""

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator
        self.names = [str(name) for name in getattr(estimator, "names_", ())]
        self.endog_names = self.names
        self.sigma_u = self._sigma_u()
        self.resid = self._residuals()

    def irf(self, periods: int, var_decomp: Any | None = None) -> _InternalVARIRF:
        del var_decomp
        return _InternalVARIRF(self.ma_rep(maxn=int(periods)), self.sigma_u)

    def ma_rep(self, maxn: int) -> np.ndarray:
        coef = np.asarray(getattr(self._estimator, "coef_", None), dtype=float)
        if coef.ndim != 2 or coef.size == 0:
            k = max(1, len(self.names))
            return np.repeat(np.eye(k, dtype=float)[None, :, :], int(maxn) + 1, axis=0)
        k = len(self.names)
        p = int(getattr(self._estimator, "n_lag", 1))
        lag_coef = np.zeros((p, k, k), dtype=float)
        for lag in range(p):
            left = lag * k
            right = left + k
            lag_coef[lag] = coef[:, left:right]
        ma = np.zeros((int(maxn) + 1, k, k), dtype=float)
        ma[0] = np.eye(k, dtype=float)
        for horizon in range(1, int(maxn) + 1):
            total = np.zeros((k, k), dtype=float)
            for lag in range(1, min(p, horizon) + 1):
                total += lag_coef[lag - 1] @ ma[horizon - lag]
            ma[horizon] = total
        return ma

    def fevd(self, periods: int) -> _InternalVARFEVD:
        return _InternalVARFEVD(
            self.ma_rep(maxn=int(periods) - 1), self.sigma_u, int(periods)
        )

    def _residuals(self) -> np.ndarray:
        datamat = getattr(self._estimator, "datamat_", None)
        coef = np.asarray(getattr(self._estimator, "coef_", None), dtype=float)
        if datamat is None or coef.ndim != 2 or coef.size == 0:
            return np.empty((0, max(1, len(self.names))), dtype=float)
        frame = pd.DataFrame(datamat)
        k = len(self.names)
        y = frame.iloc[:, :k].to_numpy(dtype=float)
        rhs = frame.iloc[:, k:].to_numpy(dtype=float)
        return y - rhs @ coef.T

    def _sigma_u(self) -> np.ndarray:
        resid = self._residuals()
        k = max(1, len(self.names))
        if resid.size == 0:
            return np.eye(k, dtype=float)
        # Match statsmodels VARResults.sigma_u: divide by the residual degrees of
        # freedom T_eff - (k*p + n_det), i.e. the per-equation regressor count
        # (coef_ has shape (k, k*p + n_det)). Dividing by T_eff - k under-deflated
        # Sigma_u for p>1 or with deterministic terms, inflating the Cholesky
        # factor and every orthogonalised IRF / GIRF / FEVD built from it.
        coef = np.asarray(getattr(self._estimator, "coef_", None), dtype=float)
        n_reg = int(coef.shape[1]) if coef.ndim == 2 and coef.size else k
        denom = max(1, resid.shape[0] - n_reg)
        sigma = resid.T @ resid / float(denom)
        return _positive_definite_covariance(sigma)


def _positive_definite_covariance(matrix: Any) -> np.ndarray:
    cov = np.asarray(matrix, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    scale = max(float(np.nanmax(np.abs(eigvals))) if eigvals.size else 1.0, 1.0)
    clipped = np.maximum(eigvals, 1e-10 * scale)
    return 0.5 * ((eigvecs * clipped) @ eigvecs.T + ((eigvecs * clipped) @ eigvecs.T).T)


def _var_names(results: Any) -> list[str]:
    names = getattr(results, "names", None)
    if names is not None:
        return [str(name) for name in names]
    endog_names = getattr(results, "endog_names", None)
    if endog_names is not None:
        if isinstance(endog_names, str):
            return [endog_names]
        return [str(name) for name in endog_names]
    k = int(np.asarray(getattr(results, "sigma_u", np.eye(1))).shape[0])
    return [f"var_{idx}" for idx in range(k)]


def _target_position(names: Sequence[str], target: str | int | None) -> int:
    if target is None:
        return 0
    if isinstance(target, int):
        if target < 0 or target >= len(names):
            raise ValueError("target index is out of range")
        return int(target)
    if str(target) not in names:
        raise ValueError(f"target {target!r} is not in VAR names")
    return list(names).index(str(target))


def _var_irf_table(
    model: Any,
    *,
    n_periods: int,
    target: str | int | None,
    method: str,
) -> pd.DataFrame:
    if n_periods < 0:
        raise ValueError("n_periods must be non-negative")
    results = _var_results(model)
    names = _var_names(results)
    target_pos = _target_position(names, target)
    sigma = np.asarray(getattr(results, "sigma_u", np.eye(len(names))), dtype=float)
    if method == "generalized_irf":
        k = sigma.shape[0]
        try:
            irf_obj = results.irf(int(n_periods), var_decomp=np.eye(k))
        except Exception:
            irf_obj = results.irf(int(n_periods))
        irfs = np.asarray(irf_obj.irfs, dtype=float)
        rows: list[dict[str, Any]] = []
        for shock_pos, name in enumerate(names):
            e_j = np.zeros(k, dtype=float)
            e_j[shock_pos] = 1.0
            sigma_jj = float(sigma[shock_pos, shock_pos])
            scale = 1.0 if sigma_jj <= 0 else sigma_jj**-0.5
            response = 0.0
            for horizon in range(irfs.shape[0]):
                girf = scale * irfs[horizon] @ sigma @ e_j
                response += abs(float(girf[target_pos]))
            rows.append(
                {
                    "feature": str(name),
                    "importance": response,
                    "coefficient": None,
                    "status": "operational",
                }
            )
    else:
        irf_obj = results.irf(int(n_periods))
        values = np.asarray(getattr(irf_obj, "orth_irfs", irf_obj.irfs), dtype=float)
        rows = [
            {
                "feature": str(name),
                "importance": float(np.sum(np.abs(values[:, target_pos, shock_pos]))),
                "coefficient": None,
                "status": "operational",
            }
            for shock_pos, name in enumerate(names)
        ]
    return _attach_schema(
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True),
        kind=method,
        model=model,
        method=method,
        n_features=len(names),
        metadata={"n_periods": int(n_periods), "target": names[target_pos]},
    )


def _mrf_beta_names(
    estimator: Any, output: Mapping[str, Any], n_columns: int
) -> list[str]:
    raw_names: list[str] = []
    yandx = output.get("YandX")
    if isinstance(yandx, pd.DataFrame) and yandx.shape[1] >= 2:
        raw_names = [str(column) for column in yandx.columns[1:]]
    elif getattr(estimator, "x_columns", None):
        raw_names = [str(column) for column in estimator.x_columns]
    elif getattr(estimator, "_feature_names", None):
        raw_names = [str(column) for column in estimator._feature_names]
    names = ["__intercept__", *raw_names]
    if len(names) < n_columns:
        names.extend(f"beta_{idx}" for idx in range(len(names), n_columns))
    return names[:n_columns]


def _torch_attribution_context(
    model: Any, X: pd.DataFrame
) -> tuple[Any, Any, Any, list[str]]:
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "gradient attribution requires torch; install macroforecast[deep]"
        ) from exc
    estimator = _estimator(model)
    torch_model = getattr(estimator, "model_", None)
    if torch_model is None:
        raise NotImplementedError(
            "gradient attribution requires a fitted torch-backed model"
        )
    frame = _as_feature_frame(X)
    feature_names = list(getattr(estimator, "feature_names_in_", ())) or [
        str(column) for column in frame.columns
    ]
    frame = frame.reindex(columns=feature_names, fill_value=0.0).astype(float)
    values = frame.to_numpy(dtype=float)
    x_mean = getattr(estimator, "x_mean_", None)
    x_scale = getattr(estimator, "x_scale_", None)
    if x_mean is not None and x_scale is not None:
        values = (values - np.asarray(x_mean, dtype=float)) / np.asarray(
            x_scale, dtype=float
        )
    device = torch.device(getattr(estimator, "device_", "cpu") or "cpu")
    kind = getattr(estimator, "kind", None)
    if kind in {"lstm", "gru", "transformer"}:
        sequence_length = int(getattr(estimator, "sequence_length", 1))
        prefix = getattr(estimator, "train_tail_", None)
        prefix = (
            np.empty((0, values.shape[1]))
            if prefix is None
            else np.asarray(prefix, dtype=float)
        )
        combined = np.vstack([prefix, values])
        tensor_values = np.stack(
            [combined[i : i + sequence_length] for i in range(len(values))]
        )
    else:
        tensor_values = values
    tensor = torch.tensor(
        tensor_values, dtype=torch.float32, device=device, requires_grad=True
    )
    torch_model.eval()
    return torch, torch_model, tensor, feature_names


def _baseline_tensor(
    torch: Any,
    tensor: Any,
    baseline: float | pd.DataFrame | np.ndarray | None,
    *,
    estimator: Any | None = None,
    feature_names: Sequence[str] | None = None,
) -> tuple[Any, str]:
    if baseline is None:
        return torch.zeros_like(tensor), "zero_tensor_model_input_space"
    if isinstance(baseline, (int, float)):
        return torch.full_like(tensor, float(baseline)), "scalar_model_input_space"
    if isinstance(baseline, pd.DataFrame):
        names = list(feature_names or baseline.columns)
        frame = baseline.reindex(columns=names, fill_value=0.0).astype(float)
        arr = frame.to_numpy(dtype=float)
        x_mean = getattr(estimator, "x_mean_", None) if estimator is not None else None
        x_scale = (
            getattr(estimator, "x_scale_", None) if estimator is not None else None
        )
        if x_mean is not None and x_scale is not None:
            arr = (arr - np.asarray(x_mean, dtype=float)) / np.asarray(
                x_scale, dtype=float
            )
            space = "raw_dataframe_scaled_to_model_input_space"
        else:
            space = "dataframe_model_input_space"
    else:
        arr = np.asarray(baseline, dtype=float)
        space = "array_model_input_space"
    if arr.shape != tuple(tensor.shape):
        if (
            arr.ndim == 2
            and tensor.ndim == 3
            and arr.shape == (tensor.shape[0], tensor.shape[2])
        ):
            arr = np.repeat(arr[:, None, :], tensor.shape[1], axis=1)
        else:
            arr = np.broadcast_to(arr, tuple(tensor.shape))
    return torch.tensor(arr, dtype=tensor.dtype, device=tensor.device), space


def _torch_model_output(model: Any, tensor: Any) -> Any:
    output = model(tensor)
    if isinstance(output, tuple):
        output = output[0]
    return output.reshape(-1).sum()


def _torch_gradient(torch: Any, model: Any, tensor: Any) -> Any:
    point = tensor.detach().clone().requires_grad_(True)
    output = _torch_model_output(model, point)
    grad = torch.autograd.grad(output, point, create_graph=False)[0]
    return grad


def _manual_integrated_gradients(
    torch: Any,
    model: Any,
    tensor: Any,
    baseline: Any,
    *,
    n_steps: int,
) -> Any:
    total = torch.zeros_like(tensor)
    for alpha in torch.linspace(0.0, 1.0, steps=n_steps, device=tensor.device):
        point = baseline + alpha * (tensor - baseline)
        total = total + _torch_gradient(torch, model, point)
    return (tensor - baseline) * total / float(n_steps)


def _manual_gradient_shap(
    torch: Any,
    model: Any,
    tensor: Any,
    baseline: Any,
    *,
    n_samples: int,
    noise_scale: float,
    random_state: int | None,
) -> Any:
    if random_state is not None:
        torch.manual_seed(int(random_state))
    total = torch.zeros_like(tensor)
    for _ in range(n_samples):
        alpha = torch.rand((), device=tensor.device)
        point = baseline + alpha * (tensor - baseline)
        if noise_scale > 0:
            point = point + torch.randn_like(point) * float(noise_scale)
        total = total + _torch_gradient(torch, model, point)
    return (tensor - baseline) * total / float(n_samples)


def _shap_prediction_frame(values: Any, template: pd.DataFrame) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    index = template.index if arr.shape[0] == len(template) else None
    return pd.DataFrame(arr, columns=template.columns, index=index)


def _optional_anatomy() -> Any:
    try:
        return import_module("anatomy")
    except ImportError as exc:
        raise ImportError(
            "forecast-accuracy anatomy interpretation requires the optional "
            "anatomy backend. Install with `pip install anatomy` or "
            "`pip install 'macroforecast[interpretation]'`."
        ) from exc


def _resolve_anatomy_transformer(
    anatomy_mod: Any,
    *,
    transformer: Callable[..., Any] | Any | None,
    metric: str,
) -> Any | None:
    if transformer is not None:
        if callable(transformer):
            return anatomy_mod.AnatomyModelOutputTransformer(transform=transformer)
        return transformer
    key = str(metric).lower().replace("-", "_")
    if key in {"forecast", "raw", "prediction"}:
        return None
    transform = anatomy_output_transform(metric)
    return anatomy_mod.AnatomyModelOutputTransformer(transform=transform)


def _anatomy_wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    frame = wide.copy()
    rows: list[dict[str, Any]] = []
    for idx, series in frame.iterrows():
        if isinstance(frame.index, pd.MultiIndex):
            idx_tuple = idx if isinstance(idx, tuple) else (idx,)
            model_set = idx_tuple[0] if idx_tuple else None
            index_value = idx_tuple[1] if len(idx_tuple) > 1 else idx
        else:
            model_set = None
            index_value = idx
        for column, value in series.items():
            feature = str(column)
            rows.append(
                {
                    "model_set": model_set,
                    "index": index_value,
                    "feature": feature,
                    "contribution": float(value),
                    "is_base": feature == "base_contribution",
                }
            )
    return pd.DataFrame(rows)


def _coerce_contribution_frame(
    value: pd.DataFrame | pd.Series,
    *,
    value_name: str,
    feature_col: str,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        series = value.copy()
        if not isinstance(series.index, pd.MultiIndex):
            return pd.DataFrame(
                {
                    feature_col: series.index.astype(str),
                    value_name: series.to_numpy(dtype=float),
                }
            )
        return series.rename(value_name).reset_index()
    raise TypeError("contributions must be a pandas DataFrame or Series")


def _resolve_contribution_column(frame: pd.DataFrame) -> str:
    for column in ("shap_value", "forecast_contribution", "contribution", "value"):
        if column in frame.columns:
            return column
    raise ValueError(
        "contributions must contain a contribution column such as "
        "'shap_value', 'forecast_contribution', 'contribution', or 'value'"
    )


def _feature_series(
    value: pd.DataFrame | pd.Series | Mapping[str, float],
    *,
    value_col: str | None,
    feature_col: str,
    preferred_columns: Sequence[str],
) -> pd.Series:
    if isinstance(value, Mapping):
        series = pd.Series(dict(value), dtype=float)
    elif isinstance(value, pd.Series):
        series = value.astype(float)
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
        if feature_col not in frame.columns:
            raise ValueError(f"table must contain feature column {feature_col!r}")
        resolved_value = value_col or _first_present(frame, preferred_columns)
        if resolved_value is None:
            resolved_value = _resolve_contribution_column(frame)
        series = (
            frame.groupby(feature_col, sort=False)[resolved_value].mean().astype(float)
        )
    else:
        raise TypeError("value must be a Series, mapping, or feature table")
    series.index = series.index.astype(str)
    return series.drop(
        labels=[
            name for name in ("base_contribution", "__base__") if name in series.index
        ]
    )


def _align_vi_pbsv_series(
    vi: pd.Series, pbsv_values: pd.Series
) -> tuple[pd.Series, pd.Series]:
    common = vi.index.intersection(pbsv_values.index)
    if common.empty:
        raise ValueError("is_vi and oos_pbsv have no common features")
    return vi.reindex(common).astype(float), pbsv_values.reindex(common).astype(float)


def _anatomy_loss_type(anatomy_mod: Any, loss_type: str) -> Any:
    key = str(loss_type).lower().replace("-", "_")
    if key in {"lower_is_better", "loss", "minimize", "minimise"}:
        return anatomy_mod.MAS.LossType.LOWER_IS_BETTER
    if key in {
        "larger_is_better",
        "higher_is_better",
        "gain",
        "score",
        "maximize",
        "maximise",
    }:
        return anatomy_mod.MAS.LossType.LARGER_IS_BETTER
    raise ValueError("loss_type must be 'lower_is_better' or 'larger_is_better'")


def _anatomy_mas_type(anatomy_mod: Any, mas_type: str) -> Any:
    key = str(mas_type).lower().replace("-", "_")
    if key in {"importance_weighted", "weighted"}:
        return anatomy_mod.MAS.MASType.IMPORTANCE_WEIGHTED
    if key in {"equal_weighted", "equal"}:
        return anatomy_mod.MAS.MASType.EQUAL_WEIGHTED
    raise ValueError("mas_type must be 'importance_weighted' or 'equal_weighted'")


def _resolve_contribution_row_column(
    frame: pd.DataFrame, y: pd.Series | Sequence[float]
) -> str:
    if isinstance(y, pd.Series) and "index" in frame.columns:
        keys = pd.Index(frame["index"].drop_duplicates())
        if keys.isin(y.index).all():
            return "index"
    if "row" in frame.columns:
        return "row"
    if "index" in frame.columns:
        return "index"
    raise ValueError("contributions must contain a row or index column")


def _target_by_row(
    frame: pd.DataFrame,
    y: pd.Series | Sequence[float],
    *,
    row_col: str,
) -> dict[Any, float]:
    if row_col not in frame.columns:
        raise ValueError(f"row column {row_col!r} is not in contributions")
    keys = list(pd.Index(frame[row_col]).drop_duplicates())
    if isinstance(y, pd.Series):
        if pd.Index(keys).isin(y.index).all():
            return {key: float(y.loc[key]) for key in keys}
        if _keys_are_valid_positions(keys, len(y)):
            return {key: float(y.iloc[int(key)]) for key in keys}
        if len(keys) == len(y):
            return {
                key: float(value)
                for key, value in zip(keys, y.to_numpy(dtype=float), strict=False)
            }
    arr = np.asarray(y, dtype=float).reshape(-1)
    if _keys_are_valid_positions(keys, len(arr)):
        return {key: float(arr[int(key)]) for key in keys}
    if len(keys) == len(arr):
        return {key: float(value) for key, value in zip(keys, arr, strict=False)}
    raise ValueError(
        "y cannot be aligned to contributions; pass row_col explicitly or "
        "provide a Series indexed by the contribution index values"
    )


def _keys_are_valid_positions(keys: Sequence[Any], n_obs: int) -> bool:
    try:
        positions = [int(key) for key in keys]
    except (TypeError, ValueError):
        return False
    return all(
        0 <= pos < n_obs and float(pos) == float(key)
        for pos, key in zip(positions, keys, strict=False)
    )


def _row_base_value(group: pd.DataFrame, *, base_col: str, default: float) -> float:
    if base_col in group.columns:
        values = pd.to_numeric(group[base_col], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return float(default)


def _point_loss(actual: float, prediction: float, *, loss: str) -> float:
    key = str(loss).lower().replace("-", "_")
    error = float(actual) - float(prediction)
    if key in {"squared_error", "mse", "mean_squared_error", "se"}:
        return float(error * error)
    if key in {"absolute_error", "mae", "mean_absolute_error", "ae"}:
        return float(abs(error))
    raise ValueError("loss must be 'squared_error'/'mse' or 'absolute_error'/'mae'")


def _point_loss_shapley(
    *,
    actual: float,
    base_prediction: float,
    contributions: np.ndarray,
    loss: str,
    n_permutations: int | None,
    max_exact_features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(contributions, dtype=float).reshape(-1)
    n_features = int(len(values))
    if n_features == 0:
        return np.empty(0, dtype=float), {
            "shapley_mode": "empty",
            "effective_permutations": 0,
        }
    if max_exact_features < 0:
        raise ValueError("max_exact_features must be non-negative")
    if n_permutations is not None and n_permutations <= 0:
        raise ValueError("n_permutations must be positive when supplied")
    if n_permutations is None and n_features <= int(max_exact_features):
        return _exact_point_loss_shapley(
            actual=actual,
            base_prediction=base_prediction,
            contributions=values,
            loss=loss,
        ), {"shapley_mode": "exact", "effective_permutations": None}
    effective = int(n_permutations or max(128, min(4096, 16 * n_features)))
    out = np.zeros(n_features, dtype=float)
    for _ in range(effective):
        current_prediction = float(base_prediction)
        current_loss = _point_loss(actual, current_prediction, loss=loss)
        for feature_pos in rng.permutation(n_features):
            next_prediction = current_prediction + float(values[feature_pos])
            next_loss = _point_loss(actual, next_prediction, loss=loss)
            out[feature_pos] += next_loss - current_loss
            current_prediction = next_prediction
            current_loss = next_loss
    return out / float(effective), {
        "shapley_mode": "sampled",
        "effective_permutations": effective,
    }


def _exact_point_loss_shapley(
    *,
    actual: float,
    base_prediction: float,
    contributions: np.ndarray,
    loss: str,
) -> np.ndarray:
    values = np.asarray(contributions, dtype=float).reshape(-1)
    n_features = int(len(values))
    out = np.zeros(n_features, dtype=float)
    feature_positions = tuple(range(n_features))
    for feature_pos in feature_positions:
        others = tuple(pos for pos in feature_positions if pos != feature_pos)
        for size in range(n_features):
            weight = 1.0 / (n_features * comb(n_features - 1, size))
            for subset in combinations(others, size):
                subset_prediction = float(base_prediction + values[list(subset)].sum())
                next_prediction = subset_prediction + float(values[feature_pos])
                out[feature_pos] += weight * (
                    _point_loss(actual, next_prediction, loss=loss)
                    - _point_loss(actual, subset_prediction, loss=loss)
                )
    return out


def _loss_func(
    metric: Callable[[np.ndarray, np.ndarray], float] | str,
) -> Callable[[np.ndarray, np.ndarray], float]:
    if callable(metric):
        return metric
    key = str(metric).lower()
    if key == "mse":
        return lambda y, pred: float(np.mean((y - pred) ** 2))
    if key == "mae":
        return lambda y, pred: float(np.mean(np.abs(y - pred)))
    raise ValueError("metric must be 'mse', 'mae', or a callable")


def _attach_schema(
    table: pd.DataFrame,
    *,
    kind: str,
    model: Any,
    method: str,
    n_features: int,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    reference = dict(_REFERENCE_CATALOG.get(kind, {}))
    table.attrs["macroforecast_metadata_schema"] = {
        "kind": kind,
        "version": _INTERPRETATION_SCHEMA_VERSION,
        "method": method,
        "model": _model_label(model),
        "n_features": int(n_features),
        "columns": [str(column) for column in table.columns],
        "reference": reference,
        "metadata": dict(metadata or {}),
    }
    return table


def _coerce_custom_table(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        name = "value" if value.name is None else str(value.name)
        return value.rename(name).to_frame()
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(value)
    raise TypeError(
        "custom interpretation callable must return a DataFrame, Series, mapping, or sequence"
    )


def _callable_name(func: Any) -> str:
    return str(getattr(func, "__name__", func.__class__.__name__))


def _model_label(model: Any) -> str:
    if model is None:
        return "none"
    if isinstance(model, ModelFit):
        return str(model.model)
    return f"{model.__class__.__module__}.{model.__class__.__qualname__}"


def _optional_shap() -> Any:
    try:
        return import_module("shap")
    except ImportError as exc:
        raise ImportError(
            "SHAP interpretation requires the optional shap backend. "
            "Install with `pip install 'macroforecast[interpretation]'`."
        ) from exc


def _normalize_explainer(explainer: str) -> str:
    key = str(explainer).lower().replace("-", "_")
    if key in {"auto", "permutation", "tree"}:
        return key
    raise ValueError("explainer must be 'auto', 'permutation', or 'tree'")


def _coerce_shap_array(values: Any, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("multi-output SHAP values are not supported yet")
        values = values[0]
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.shape != frame.shape:
        raise ValueError(
            "SHAP output shape does not match X; expected "
            f"{frame.shape}, got {arr.shape}"
        )
    return arr


def _coerce_base_values(values: Any, n_obs: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.repeat(float(arr), n_obs)
    arr = arr.reshape(-1)
    if len(arr) == 1:
        return np.repeat(float(arr[0]), n_obs)
    if len(arr) != n_obs:
        return None
    return arr.astype(float, copy=False)


def _tree_base_values(explainer_obj: Any, n_obs: int) -> np.ndarray | None:
    expected = getattr(explainer_obj, "expected_value", None)
    if isinstance(expected, list):
        expected = expected[0] if len(expected) == 1 else None
    return _coerce_base_values(expected, n_obs)
