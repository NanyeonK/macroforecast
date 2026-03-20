"""Treatment effect decomposition (CLSS 2022, Equation 11).

Decomposes OOS R² gains into four orthogonal components via a linear
regression of model-level OOS-R² values on component indicator dummies:

    OOS-R²_m = alpha
               + beta_1 * 1[nonlinearity=m is nonlinear]
               + beta_2 * 1[regularization=m is data-rich]
               + beta_3 * 1[cv=m uses K-fold]
               + beta_4 * 1[loss=m uses L2]
               + epsilon_m

The betas are the average marginal contribution of each component.
Standard errors use HC3 (heteroskedasticity-consistent, small-sample correction)
since the number of models is small.

Reference: Coulombe, Leroux, Stevanovic, Surprenant (2022), Eq. 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DecompositionResult:
    """Result of the four-component treatment effect regression.

    Attributes
    ----------
    coef : dict
        Coefficient estimates {component_name: float}.
    se : dict
        HC3 standard errors {component_name: float}.
    t_stat : dict
        t-statistics {component_name: float}.
    r_squared : float
        R² of the decomposition regression.
    n_models : int
        Number of model cells in the regression.
    summary_df : pd.DataFrame
        Tidy table with coef, se, t_stat for each component.
    """

    coef: dict[str, float]
    se: dict[str, float]
    t_stat: dict[str, float]
    r_squared: float
    n_models: int
    summary_df: pd.DataFrame


def decompose_treatment_effects(
    result_df: pd.DataFrame,
    benchmark_model_id: str = "linear__none__bic__l2",
    horizon: int | None = None,
) -> DecompositionResult:
    """Estimate the four treatment effect components via OLS.

    Parameters
    ----------
    result_df : pd.DataFrame
        Merged result table (from ResultSet.to_dataframe()) containing columns:
        model_id, nonlinearity, regularization, cv_scheme, loss_function,
        horizon, y_hat, y_true.  Must contain the benchmark model.
    benchmark_model_id : str
        model_id of the AR benchmark.  Used to compute relative MSFE.
    horizon : int or None
        If provided, restrict analysis to this horizon.

    Returns
    -------
    DecompositionResult
    """
    df = result_df.copy()
    if horizon is not None:
        df = df[df["horizon"] == horizon]

    # Compute MSFE per model
    df["se"] = (df["y_true"] - df["y_hat"]) ** 2
    model_msfe = df.groupby("model_id")["se"].mean()

    if benchmark_model_id not in model_msfe.index:
        raise ValueError(
            f"Benchmark model '{benchmark_model_id}' not found in result_df. "
            f"Available: {list(model_msfe.index)}"
        )
    msfe_bench = model_msfe[benchmark_model_id]

    # OOS-R² per model
    oos_r2 = 1.0 - model_msfe / msfe_bench

    # Build component indicator matrix
    # One row per model (excluding the benchmark itself)
    model_meta = (
        df[["model_id", "nonlinearity", "regularization", "cv_scheme", "loss_function"]]
        .drop_duplicates("model_id")
        .set_index("model_id")
    )

    # Indicator definitions (binary: 1 = "treated" version of the component)
    # Nonlinearity: 1 if nonlinear (not LINEAR)
    model_meta["d_nonlinear"] = (model_meta["nonlinearity"] != "linear").astype(float)

    # Regularization: 1 if data-rich (FACTORS or any penalized estimator)
    data_rich_regs = {
        "factors",
        "ridge",
        "lasso",
        "adaptive_lasso",
        "group_lasso",
        "elastic_net",
        "tvp_ridge",
        "booging",
    }
    model_meta["d_data_rich"] = (
        model_meta["regularization"].isin(data_rich_regs).astype(float)
    )

    # CV scheme: 1 if K-fold (preferred per CLSS 2022)
    model_meta["d_kfold"] = (
        model_meta["cv_scheme"].str.startswith("_KFoldCV").astype(float)
    )

    # Loss function: 1 if L2
    model_meta["d_l2"] = (model_meta["loss_function"] == "l2").astype(float)

    # Align with OOS-R² vector
    common_idx = model_meta.index.intersection(oos_r2.index)
    y_vec = oos_r2[common_idx].values
    X_cols = ["d_nonlinear", "d_data_rich", "d_kfold", "d_l2"]
    X_mat = np.column_stack(
        [
            np.ones(len(common_idx)),  # intercept
            model_meta.loc[common_idx, X_cols].values,
        ]
    )

    # OLS: beta = (X'X)^{-1} X'y
    XtX = X_mat.T @ X_mat
    Xty = X_mat.T @ y_vec
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    y_hat = X_mat @ beta
    resid = y_vec - y_hat

    # HC3 sandwich standard errors
    # HC3: weight_i = e_i² / (1 - h_ii)²  where h_ii is leverage
    # Use pinv throughout to handle near-singular XtX (few models, collinear dummies)
    XtX_inv = np.linalg.pinv(XtX)
    H_mat = X_mat @ XtX_inv @ X_mat.T
    h_diag = np.diag(H_mat)
    w = resid**2 / np.clip((1 - h_diag) ** 2, 1e-10, None)
    meat = sum(w_i * np.outer(X_mat[i], X_mat[i]) for i, w_i in enumerate(w))
    V_hc3 = XtX_inv @ meat @ XtX_inv
    se_vec = np.sqrt(np.maximum(np.diag(V_hc3), 0))

    t_vec = beta / np.where(se_vec > 0, se_vec, np.nan)

    # R² of the decomposition regression
    ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
    ss_res = np.sum(resid**2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    comp_names = ["intercept"] + X_cols
    coef_dict = dict(zip(comp_names, beta.tolist()))
    se_dict = dict(zip(comp_names, se_vec.tolist()))
    t_dict = dict(zip(comp_names, t_vec.tolist()))

    summary = pd.DataFrame(
        {
            "component": comp_names,
            "coef": beta,
            "se_hc3": se_vec,
            "t_stat": t_vec,
        }
    )

    return DecompositionResult(
        coef=coef_dict,
        se=se_dict,
        t_stat=t_dict,
        r_squared=r_sq,
        n_models=len(common_idx),
        summary_df=summary,
    )
