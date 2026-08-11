"""B1 Medeiros (2021) arm registry — parameters locked to the AUTHOR CODE.

Source of truth = gabrielrvsc/ForecastingInflation (functions/functions.R) and
gabrielrvsc/HDeconometrics, cross-checked against the correct IJF-2021 paper. Where the
design handoff and author code disagreed, author code wins (Fable directive).

Locked constants (author code):
  target   = CPIAUCSL, transform = 100*dlog (monthly inflation), direct h=1..12
  features = all series + 4 PCA factors (princomp on scaled full panel), embed(.,4) =
             lags 0..3 in R embed order, + Nov-2008 dummy (all models except RW/UCSV)
  AR       = BIC order selection (univariate)          -> mf model "ar" + IC(bic) route
  UCSV     = Stock-Watson (2007), gamma=0.2 log-vol RW innovation variance;
             paper Vtau=Vh=0.12 initial priors via new package knobs
  RF       = explicit author config: ntree=500, mtry=p/3, nodesize=5,
             replace=TRUE, sampsize=n, maxnodes=NULL, no CV tuning, seed=42
  RF/OLS   = randomForest(maxnodes=25, ntree=500)  [paper text says 20; code=25 -> use 25]
  adaLASSO = ic.glmnet BIC lambda; penalty (|b|+1/sqrt(n))^-1
  bagging  = R=100, l=5, pre.testing="group-joint"
  CSR      = HDeconometrics defaults K=20, k=4  (paper "n=25,q=4" is an illustrative
             C(25,4)=12650 example, not the config)
  JMA      = plain hat-matrix LOO (gap-LOO absent = [GAP], design D4)
"""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf
from macroforecast.pipeline import Arm, CombinationContender  # noqa: F401

TARGET = "CPIAUCSL"
PANEL_PATH = "qa/medeiros_panel.parquet"
HORIZONS = list(range(1, 13))
UCSV_PARAMS = {
    "gamma": 0.2,  # Stock-Watson log-volatility RW innovation variance.
    "initial_obs_log_vol_variance": 0.12,  # paper Vh via package initial-prior knob.
    "initial_level_log_vol_variance": 0.12,  # paper Vtau via package initial-prior knob.
    "random_state": 42,
}
RF_PARAMS = {
    "n_estimators": 500,    # author randomForest default ntree
    "max_features": 1.0 / 3.0,  # sklearn float -> floor(p/3), matching R mtry
    "min_samples_leaf": 5,  # R randomForest regression default nodesize
    "bootstrap": True,      # R replace=TRUE
    "max_samples": None,    # R sampsize=nrow(Xin) under replace=TRUE
    "max_leaf_nodes": None, # plain RF maxnodes=NULL; maxnodes=25 belongs to RF/OLS
    "random_state": 42,
}

_AUTHOR_PANEL: pd.DataFrame | None = None


def _author_panel() -> pd.DataFrame:
    global _AUTHOR_PANEL
    if _AUTHOR_PANEL is None:
        _AUTHOR_PANEL = pd.read_parquet(PANEL_PATH).sort_index()
    return _AUTHOR_PANEL


def _r_princomp_scores(frame: pd.DataFrame, n_components: int = 4) -> tuple[pd.DataFrame, dict[str, object]]:
    """R `princomp(scale(df))$scores[, 1:k]` with `fix_sign=TRUE`.

    The author code scales with R's `scale()` first, then calls `princomp()` on
    that scaled matrix. `princomp()` uses covariance eigenvectors and flips each
    loading so the first element is nonnegative.
    """

    values = frame.astype(float)
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1).replace(0.0, np.nan).fillna(1.0)
    z = (values - center) / scale
    cov = np.cov(z.to_numpy(dtype=float), rowvar=False, ddof=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    k = min(int(n_components), eigvecs.shape[1])
    loadings = eigvecs[:, :k].copy()
    signs = np.where(loadings[0, :] < 0.0, -1.0, 1.0)
    loadings *= signs
    scores = z.to_numpy(dtype=float) @ loadings
    columns = [f"Comp.{i}" for i in range(1, k + 1)]
    return (
        pd.DataFrame(scores, index=frame.index, columns=columns),
        {
            "columns": tuple(values.columns),
            "center": center,
            "scale": scale,
            "loadings": pd.DataFrame(loadings, index=values.columns, columns=columns),
            "eigenvalues": eigvals[:k],
        },
    )


def author_rf_feature_fit(source: pd.DataFrame, **_: object) -> dict[str, object]:
    panel = _author_panel().reindex(source.index)
    columns = tuple(panel.columns)
    if TARGET not in columns:
        raise ValueError(f"{TARGET} is missing from {PANEL_PATH}")
    if panel.loc[:, columns].isna().any().any():
        raise ValueError("author RF feature fit requires a balanced panel over the fit window")
    _, pca_state = _r_princomp_scores(panel.loc[:, columns], n_components=4)
    return {
        "columns": columns,
        "pca_state": pca_state,
    }


def _author_rf_panel_for_index(index: pd.Index, state: dict[str, object]) -> pd.DataFrame:
    columns = tuple(state["columns"])  # type: ignore[arg-type]
    pca_state = state["pca_state"]  # type: ignore[assignment]
    panel = _author_panel().reindex(index).loc[:, columns].astype(float)
    center = pca_state["center"]  # type: ignore[index]
    scale = pca_state["scale"]  # type: ignore[index]
    loadings = pca_state["loadings"]  # type: ignore[index]
    z = (panel - center) / scale
    scores = z.to_numpy(dtype=float) @ loadings.to_numpy(dtype=float)
    factors = pd.DataFrame(scores, index=panel.index, columns=list(loadings.columns))
    return pd.concat([panel, factors], axis=1)


def author_rf_feature_transform(source: pd.DataFrame, *, state: dict[str, object], **_: object) -> pd.DataFrame:
    x = _author_rf_panel_for_index(source.index, state)
    blocks: list[pd.DataFrame] = []
    for lag in range(4):
        block = x.shift(lag).copy()
        block.columns = [f"{column}_lag{lag}" for column in x.columns]
        blocks.append(block)
    out = pd.concat(blocks, axis=1)
    out["nov2008"] = (out.index == pd.Timestamp("2008-11-01")).astype(float)
    out.index.name = "date"
    return out


author_rf_feature_fit.__mf_digest__ = "medeiros-author-rf-feature-fit-v1"
author_rf_feature_transform.__mf_digest__ = "medeiros-author-rf-feature-transform-v1"

def base_features():
    # Exact author `dataprep()` RF matrix: factors from princomp(scale(df)) on
    # the full in-window panel including CPIAUCSL, then embed(cbind(df, factors), 4).
    return mf.feature_spec(
        predictors="all",
        lags=None,
        target_lags=None,
        feature_steps=[
            {
                "method": "custom",
                "name": "author_rf_embed",
                "fit_func": author_rf_feature_fit,
                "transform_func": author_rf_feature_transform,
                "include": True,
            }
        ],
    )

ML = ("ridge", "lasso", "adaptive_lasso", "elastic_net", "adaptive_elastic_net")

def arms():
    A = [
        Arm("rw",   model="naive", features=None, is_benchmark=True),
        Arm("ar",   model="ar",    features=None),                 # univariate, BIC order
        Arm("ucsv", model="ucsv",  features=None, params=dict(UCSV_PARAMS)),
        *[Arm(k, model=k, features=base_features()) for k in ML],
        Arm("rf",   model="random_forest", features=base_features(), params=dict(RF_PARAMS),
            model_selection={"random_forest": None}),
        Arm("csr",  model="csr", params={"k": 4},
            features=mf.feature_spec(predictors="all", lags=(0, 1, 2, 3),
                feature_steps=[{"method": "predictor_screen",
                                "screen_method": "t_stat", "top_k": 20}])),
        Arm("jma",  model="jma", features=base_features()),        # plain LOO [GAP: gap-LOO]
    ]
    return A

# lane-custom (author 2-stage / algorithm-specific) — deferred past G2:
#   rf_ols (maxnodes=25,ntree=500), ada_rf, bagging(R=100,l=5,group-joint),
#   boosted factor (BN09, v=0.2, BIC stop), JMA gap-LOO
