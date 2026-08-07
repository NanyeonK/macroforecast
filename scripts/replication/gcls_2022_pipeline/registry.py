"""The GCLS (2022, JAE) 46-model registry as native ``macroforecast`` pipeline arms.

Goulet Coulombe, Leroux, Stevanovic, Surprenant (2022), "How is Machine Learning
Useful for Macroeconomic Forecasting?", JAE 37(5), 920-964, Table 1 (46 models) +
the 4-characteristic treatment-effect design (Eq.11-12).

Every arm is a native ``Arm(model=..., features=..., params=..., model_selection=...)``
with NO ``custom_model`` (verified: all point-forecast builders are native). Each arm
carries ``Arm.tags`` for the A3 treatment-effect regression, over the 5 GCLS
characteristics:

  * ``X``  (int 0/1)  -- 0 = data-poor (H_t^-, own-lags only), 1 = data-rich (factors/panel)
  * ``NL`` (int 0/1)  -- 0 = linear, 1 = nonlinear. Per GCLS S0/Fig.1, NL indexes KRR
                         and RF ONLY. SVR's kernel nonlinearity is attributed to the LF
                         (loss) characteristic, not NL (kept in metadata['kernel']).
  * ``SH`` (str)      -- shrinkage/regularization spec: "none" | "ridge" (RR models,
                         zeta=0) | "B{1,2,3}_z0|zhat|z1" (feature-scope block x zeta in
                         {0=ridge, zhat=EN-CV, 1=lasso}).
  * ``CV`` (str)      -- hyperparameter selection: "bic" | "aic" | "poos" | "kfold" |
                         "none" (fixed HP).
  * ``LF`` (str)      -- loss: "quad" (quadratic) | "eps" (eps-insensitive SVR).

--- The ARDI (p_y, p_f, n_f) construction (scout's first blocker) -------------------
GCLS ARDI needs SEPARATED own-lags p_y, factor-lags p_f, and IC/CV-selected factor
count n_f. The native ``far`` builder exposes ``n_factors`` (= n_f) and ``n_lag``
(= p_y) as MODEL params, both IC/CV-selectable via ``model_selection`` -- so the ARDI
arms (BIC/AIC/POOS/KF) are expressed as ``far`` + a search over (n_factors, n_lag).
This gives a genuine per-origin IC/CV-selected ARDI where n_f is the *IC-selected K
factors* (contrast B2/B3 below, which KEEP ALL PCs). The one bounded deviation: ``far``
builds CONTEMPORANEOUS PCA factors of the predictor block (p_f = 1); it does not expose
a separate factor-lag order. GCLS's p_f in {1,3,6,12} grid is therefore NOT native.

FIX-LANE INPUT (do NOT patch the package here): to select (p_y, p_f, n_f) jointly by
IC/CV with an INDEPENDENT p_f, a minor extension is needed -- EITHER (i) model_selection
that can search FEATURE-construction params (target_lags / factor-lags / pca_components),
OR (ii) an ARDI model wrapper that takes (p_y, p_f, n_f) as model params and builds the
lag-then-factor design internally (like ``far`` but with a separate factor-lag order).
Reason it is not native: ``model_selection`` (SearchSpec) searches MODEL-builder params
on a feature matrix that is BUILT ONCE per arm (``pipeline/run.py`` -> ``forecasting.run``
-> ``selection_stage``); p_y/p_f/n_f are feature-construction params, so a fixed feature
matrix cannot vary them. The faithful separated construction (target_lags=p_y + PCA
factors(n_f) each lagged p_f + ``ols``) is expressible for FIXED orders but not
IC/CV-SELECTABLE without the extension.

Distinguishing ARDI vs B2/B3 (design S1b, p.926 S3.2):
  * ARDI  -> ``far``: PCA inside the model, ``n_factors`` = IC-selected K factors.
  * B2/B3 -> feature-pipeline PCA with ``pca_components`` = keep-all (min(N,T)) -> the
    shrinkage estimator then regularizes over the FULL rotated space.
"""
from __future__ import annotations

from typing import Any, Iterable

import os

import macroforecast as mf
from macroforecast.model_selection import SearchSpec, validation_splitter
from macroforecast.pipeline import Arm

# --------------------------------------------------------------------------- #
# Hyperparameter grids
# --------------------------------------------------------------------------- #
# Discrete grids -- CONFIRMED from the paper TeX supplement (b4_scout_findings.md):
P_LAG_GRID: tuple[int, ...] = (1, 3, 6, 12)      # p_y own-lag order (AR / ARDI)
N_FACTOR_GRID: tuple[int, ...] = (3, 6, 10)       # n_f factor count (ARDI, far n_factors)
RF_N_ESTIMATORS = 500                              # CONFIRMED (S6, L1715-1718)
RF_MAX_FEATURES = 1.0 / 3.0                        # CONFIRMED mtry = 1/3
# libsvm defaults to max_iter=-1 (unlimited): on unscaled ARDI features some (C, origin)
# combinations converge pathologically slowly and dominate wall time. Cap it so every
# SVR fit terminates. [ASSUMPTION] -- the faithful G2 should also STANDARDISE the SVR
# inputs (GCLS scales SVR features), which fixes both convergence speed and quality.
SVR_MAX_ITER = 5000                                # [ASSUMPTION] convergence cap

# Continuous penalty grids -- [ASSUMPTION]: never given numerically in the paper (only
# "selected by CV"). Documented log grids; refined at G2 if needed. Treatment-effect
# SIGNS (the primary parity target) are robust to grid choice (design R4). The
# GCLS-2021 R code (tvpridge) uses exp(linspace(-6,20,15)) k-fold=5 for its bespoke
# ridge; we use a standard standardized-feature log grid for the 2022 horse-race
# sklearn/kernlab-style estimators.
RIDGE_ALPHA_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)   # [ASSUMPTION]
EN_L1RATIO_GRID: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)                  # [ASSUMPTION] zeta_hat
# KRR grid matched to GCLS-2022 (source: DualML_R/DualML_run_inflation.R L369
# lambda_grid=seq(1e-3,50,length.out=200); L266/272 rbfdot(sigma~1e-4)=kernlab sigest
# median heuristic; paper eq (KTregression) L364 tau={lambda,sigma,p_y,p_f,n_f}). lambda
# (=sklearn alpha) is CV'd over [1e-3, 50]; sigma (=sklearn gamma; K=exp(-gamma||x-x'||^2),
# same as kernlab rbfdot) is set by the sigest median heuristic -> sklearn KernelRidge
# gamma=None (== 1/n_features, scale-adaptive) on STANDARDISED factor inputs (see the
# KRRARDI arm). The OLD grid (alpha up to 1e3; gamma up to 1.0 on UNSCALED factors, std~5)
# gave gamma*||x-x'||^2 ~ 150 -> a near-diagonal degenerate RBF, i.e. KRR underperformance
# that muted the NL treatment. [ASSUMPTION]: GCLS's exact sigma CV grid is not in the
# data-only archive; sigest (kernlab's documented default) is the faithful substitute.
KRR_ALPHA_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 50.0)          # GCLS lambda in [1e-3, 50]
SVR_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)                          # [ASSUMPTION]
SVR_EPS_GRID: tuple[float, ...] = (0.01, 0.1)                                    # [ASSUMPTION] eps tube
SVR_GAMMA_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)                      # [ASSUMPTION] SVR-RBF (scaled feats)
KRR_GAMMA_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)      # CV grid for sigma (RBF gamma); paper tau={lambda,sigma,p_y,p_f,n_f} CVs sigma. Native SearchSpec param (same as svr_rbf_grid gamma).

# n_f for the factor-BASED non-far arms (RRARDI/RFARDI/KRRARDI/SVR-ARDI). Their factors
# are FEATURE-built (pca_components), which model_selection cannot search, so n_f is
# FIXED here. [ASSUMPTION]: GCLS IC/CV-selects n_f in {3,6,10}; fixed at 6 (mid-grid)
# pending the same feature-param-search fix-lane extension noted in the module header.
ARDI_NF_FIXED = 6                                                               # [ASSUMPTION]

# Keep-all-PC counts (B2/B3). N = 128 series. min(N,T) = N here (T > 240 always at the
# 1960M01 estimation start). A large sentinel is clipped to min(features, train_rows)
# by the PCA step -> "keep all PCs".
B2_KEEPALL = 128            # PCs of X (rotation, keep all)
# PCs of H+ (lag-augmented panel). The true keep-all = min(N_aug, T) varies per origin;
# a static spec needs a fixed int and the pca_step validates n_components <= available
# columns/rows, so we keep the leading 128 PCs (always valid at the smallest fit window;
# matches B2's scale). [ASSUMPTION] -- exact per-origin keep-all is a G2 refinement.
B3_KEEPALL = 128

TARGET_LAGS = range(0, 13)  # provide up to 12 own-lags; the order is selected downstream

# --------------------------------------------------------------------------- #
# model_selection builders
# --------------------------------------------------------------------------- #
def _ic_search(param_grid: dict[str, tuple[Any, ...]], criterion: str) -> SearchSpec:
    """Information-criterion selection (AR/ARDI order) at the retune cadence."""
    return SearchSpec(method="information_criterion", param_grid=param_grid, criterion=criterion)


def _cv_search(param_grid: dict[str, tuple[Any, ...]], cv: str) -> SearchSpec:
    """Grid HP selection by validation loss. cv in {'poos','kfold'} -> POOS-CV
    (last 25% in-sample) or standard k=5 random K-fold."""
    if cv == "poos":
        splitter = validation_splitter("poos", validation_ratio=0.25)
    elif cv == "kfold":
        splitter = validation_splitter("random_kfold", n_splits=5)
    else:  # pragma: no cover
        raise ValueError(f"unknown cv {cv!r}")
    return SearchSpec(method="grid", param_grid=param_grid, validation_splitter=splitter)


# --------------------------------------------------------------------------- #
# Feature builders
# --------------------------------------------------------------------------- #
# FEATURE ROUTE NOTE (empirically verified; root-caused to a package bug, now fixed):
# Under GCLS spec-level preprocessing (transform='official' + em_factor + iqr/
# flag_as_nan) the per-origin panel is dense EXCEPT for series fully absent over the
# fit window (e.g. ACOGNO begins 1992). em_factor cannot fill a zero-observation
# series, so its raw feature columns are all-NaN, and the training slice's row-wise
# dropna then emptied the WHOLE fit sample -> zero forecasts for any raw-wide-predictor
# block (B1 identity, B3 lag-then-PCA). Root cause + fix: macroforecast
# fix/raw-wide-feature-rows @ 4bf4435c (_drop_all_nan_fit_columns drops all-NaN
# fit-window columns before slicing; leak-free; golden gate unchanged). With that fix,
# B1/B3 RUN. Factor arms + B2 use ``pca_step`` (PCA densifies, so they ran even on the
# unpatched package). ARDI is still expressed as the faithful feature-pipeline
# (target_lags=p_y + pca_step factors=n_f + ``ols``) because native ``far`` also
# consumed the raw block; IC/CV selection over (p_y,p_f,n_f) remains fix-lane #1
# (feature-param search / ARDI wrapper), independent of this fix.


def _ar_features(target: str):
    """Data-poor: own y-lags only (no predictors)."""
    return mf.feature_spec(target=target, predictors=[], target_lags=TARGET_LAGS)


#: Factor-lag order ``p_f`` for the ARDI arms.
#:
#: GCLS write ARDI as ``y_{t+h} = c + rho(L) y_t + beta(L) F_t + e`` with
#: ``rho`` and ``beta`` of orders ``p_y`` and ``p_f`` (eq. swardi1), and list the
#: tuning set as ``{lambda, sigma, p_y, p_f, n_f}`` -- so the factor block enters
#: with its OWN lag order.
#:
#: Native ``far`` cannot express that: it regresses on the CONTEMPORANEOUS
#: factors plus the target's own lags, i.e. ``p_f = 0``, and offers no parameter
#: to change it (measured: n_factors=3, n_lag=2 gives exactly 5 coefficients).
#: ``MF_GCLS_PF`` switches ARDI to the composable route, which can.
#:
#: Unset (the default) keeps every artifact produced so far valid.
#:
#: Set it to isolate p_f in a controlled comparison. The two routes differ in
#: TWO ways, not one -- ``far`` IC-selects ``n_f`` from the grid while the
#: composable route has to fix it -- so measuring the cost of ``p_f`` alone needs
#: the composable route at BOTH ``p_f = 0`` and ``p_f > 0``:
#:
#:   MF_GCLS_ARDI_ROUTE=far                       far,        n_f IC-selected, p_f=0
#:   MF_GCLS_ARDI_ROUTE=composable MF_GCLS_PF=0   composable, n_f fixed,       p_f=0
#:   MF_GCLS_ARDI_ROUTE=composable MF_GCLS_PF=2   composable, n_f fixed,       p_f=2
#:
#: The middle row is the control: comparing it with the third isolates p_f, and
#: comparing it with the first prices the loss of IC-selected n_f.
ARDI_ROUTE: str = os.environ.get("MF_GCLS_ARDI_ROUTE", "far").strip().lower()
#: Predictor lags fed to the ARDI design. ``0`` = the paper's X_t only; the
#: package default of ``(0, 1)`` stacks X_{t-1} into the PCA input as well.
X_LAGS = tuple(int(v) for v in os.environ.get("MF_GCLS_XLAGS", "0").split(",") if v != "")
ARDI_PF: int = int(os.environ.get("MF_GCLS_PF", "0"))


def _far_features(target: str, predictors):
    """ARDI via ``far``: the raw predictor panel; ``far`` PCAs it internally
    (n_factors = IC/CV-selected K factors) and uses the target's own lags (n_lag).
    Runs on the raw block once the all-NaN fit-column prune (fix 4bf4435c) is in the
    package -- pre-fix, a fully-absent-over-window series emptied the fit sample.

    With ``MF_GCLS_PF > 0`` this instead builds the paper's ARDI explicitly:
    ``pca_step`` for ``F_t`` then ``lag_step`` for ``beta(L)``. The PCA is pinned
    to ``scale=False`` so the route change does not ALSO silently switch the
    factor-extraction convention -- ``far`` centers without standardizing
    (package issue #495), and ``pca_step`` defaults the other way."""
    cols = "all" if predictors == "all" else list(predictors)
    if ARDI_ROUTE != "composable":
        # ``lags`` MUST be given. Its default is (0, 1), which hands ``far`` a
        # design of every predictor at t AND t-1, so its internal PCA runs on a
        # STACKED [X_t, X_{t-1}] panel -- 264 columns for 132 series. The paper's
        # factor model is X_t = Lambda F_t + u_t (eq. swardi2): factors of the
        # cross-section at ONE time index. MF_GCLS_XLAGS keeps the old behaviour
        # available for comparison; 0 is the paper's spec.
        return mf.feature_spec(
            target=target, predictors=cols, target_lags=TARGET_LAGS, lags=X_LAGS
        )
    return mf.feature_spec(
        target=target,
        target_lags=TARGET_LAGS,
        steps=[
            mf.feature_engineering.pca_step(
                name="fac",
                input="panel",
                columns=None if predictors == "all" else list(predictors),
                n_components=ARDI_NF_FIXED,
                include=False,
                scale=False,
            ),
            mf.feature_engineering.lag_step(
                name="faclag", input="fac", lags=tuple(range(ARDI_PF + 1)), include=True
            ),
        ],
    )


#: Whether the factor PCA standardizes X first. GCLS (2022) does not say.
#: ``True`` (the default, and every artifact produced so far) is the
#: Stock-Watson / McCracken-Ng FRED-MD norm; ``MF_GCLS_PCA_SCALE=0`` selects the
#: covariance convention, so the two can be reported side by side rather than one
#: being chosen to match the published table. See the B3 (ZWW) note for a paper
#: in this literature whose headline turned on exactly this choice.
PCA_SCALE: bool = os.environ.get("MF_GCLS_PCA_SCALE", "1") != "0"


def _factor_features(target: str, n_f: int, predictors):
    """Data-rich factor arm: own y-lags + n_f PCA factors of the predictor panel.

    This is the faithful ARDI/factor construction AND the only route that runs: the
    factors come from ``pca_step(input='panel', columns=predictors)`` (leak-safe --
    the PCA sees only the FRED-MD predictors, never the YOBJ target objects)."""
    cols = None if predictors == "all" else list(predictors)
    step = mf.feature_engineering.pca_step(
        input="panel", columns=cols, n_components=n_f, include=True, scale=PCA_SCALE
    )
    return mf.feature_spec(target=target, target_lags=TARGET_LAGS, steps=[step])


def _svr_ardi_features(target: str, predictors):
    """SVR-ARDI: target own-lags + Z-SCORED PCA factors (StandardScaler, fit_policy=
    expanding = leak-free). GCLS standardises SVR inputs; the PCA factors (std ~5) dwarf
    the growth-object own-lags (std ~0.01), and that scale mismatch is what makes libsvm
    condition/converge badly. Empirically, scaling the FACTORS moves SVR-ARDI,Lin from
    rel-RMSPE 1.11 -> ~1.05; scaling the own-lags too made it worse, so only the factors
    are standardised. target_lags= is kept (re-targetable across targets); the max_iter
    cap stays as the safety net for the large-C CV grid candidates."""
    cols = None if predictors == "all" else list(predictors)
    return mf.feature_spec(
        target=target, target_lags=TARGET_LAGS,
        steps=[
            mf.feature_engineering.pca_step(
                name="fac", input="panel", columns=cols, n_components=ARDI_NF_FIXED, include=False
            ),
            mf.feature_engineering.scale_step(
                name="fac_z", input="fac", method="zscore", fit_policy="expanding", include=True
            ),
        ],
    )


def _keepall_pca_features(target: str, n_components: int, predictors):
    """B2: shrinkage over the FULL rotated space -- keep-all PCs of X (min(N,T))."""
    return _factor_features(target, n_components, predictors)


def _identity_features(target: str, predictors):
    """B1: shrinkage on the raw predictor set (identity), all X (+lags) + y-lags.

    Uses the raw wide predictor block via ``lag_step``. This RUNS once the package
    carries the all-NaN fit-column prune (fix/raw-wide-feature-rows @ 4bf4435c); on an
    unpatched 0.9.5 a fully-absent-over-window series empties the fit sample."""
    cols = None if predictors == "all" else list(predictors)
    step = mf.feature_engineering.lag_step(
        input="panel", columns=cols, lags=(0, 1), include=True
    )
    return mf.feature_spec(target=target, target_lags=TARGET_LAGS, steps=[step])


def _lag_then_keepall_pca_features(target: str, n_components: int, predictors):
    """B3: keep-all PCs of H+ = lag-augmented panel, then PCA (lag -> PCA order).

    The pca_step CHAINS from the lag_step output (``input="lagblk"``) so the PCA is
    taken over the lag-augmented block (not the raw panel again); the intermediate lag
    block is not itself emitted (``include=False``)."""
    cols = None if predictors == "all" else list(predictors)
    steps = [
        mf.feature_engineering.lag_step(
            name="lagblk", input="panel", columns=cols, lags=(0, 1), include=False
        ),
        mf.feature_engineering.pca_step(
            input="lagblk", n_components=n_components, include=True
        ),
    ]
    return mf.feature_spec(target=target, target_lags=TARGET_LAGS, steps=steps)


# --------------------------------------------------------------------------- #
# Arm-spec table: one entry per GCLS model. Feature/model_selection are built lazily
# so they can be re-targeted per target by the pipeline.
# --------------------------------------------------------------------------- #
def _tags(X: int, NL: int, SH: str, CV: str, LF: str) -> dict[str, Any]:
    return {"X": X, "NL": NL, "SH": SH, "CV": CV, "LF": LF}


def build_gcls2022_arms(
    target: str = "YOBJ__INDPRO", predictors: list[str] | None = None
) -> list[Arm]:
    """All 46 GCLS-2022 point-forecast arms for a single ``target`` object column.

    ``target`` is the YOBJ__<col> object column (see data.py); the pipeline re-targets
    each arm's feature spec per target, so any placeholder works. ``predictors`` is the
    FRED-MD information set (original series, excluding the YOBJ object columns). If
    None, defaults to ``predictors='all'`` -- callers that feed the augmented bundle
    MUST pass the explicit list so the YOBJ target objects cannot leak as features.
    """
    if predictors is None:
        predictors = "all"  # type: ignore[assignment]
    arms: list[Arm] = []

    def add(
        name: str, model: str, features, *, params=None, model_selection=None,
        tags: dict[str, Any], is_benchmark: bool = False, metadata: dict[str, Any] | None = None,
    ) -> None:
        arms.append(
            Arm(
                name=name, model=model, features=features, params=params,
                model_selection=model_selection, tags=tags, is_benchmark=is_benchmark,
                metadata=metadata or {},
            )
        )

    # ---- helper grids -----------------------------------------------------
    ar_grid = {"n_lag": P_LAG_GRID}
    ardi_grid = {"n_factors": N_FACTOR_GRID, "n_lag": P_LAG_GRID}
    ridge_grid = {"alpha": RIDGE_ALPHA_GRID}
    en_grid = {"alpha": RIDGE_ALPHA_GRID, "l1_ratio": EN_L1RATIO_GRID}
    krr_grid = {"alpha": KRR_ALPHA_GRID, "gamma": KRR_GAMMA_GRID}  # CV both lambda AND sigma (paper tau); native SearchSpec, same mechanism svr_rbf_grid uses for its gamma
    svr_lin_grid = {"C": SVR_C_GRID, "epsilon": SVR_EPS_GRID}
    svr_rbf_grid = {"C": SVR_C_GRID, "epsilon": SVR_EPS_GRID, "gamma": SVR_GAMMA_GRID}
    rf_params = {"n_estimators": RF_N_ESTIMATORS, "max_features": RF_MAX_FEATURES}

    # ===================================================================== #
    # DATA-POOR (X=0): own y-lags only
    # ===================================================================== #
    ar_feat = _ar_features(target)
    # AR benchmark family (IC + CV). "AR,BIC" is the EvalSpec benchmark.
    add("AR,BIC", "ar", ar_feat, model_selection=_ic_search(ar_grid, "bic"),
        tags=_tags(0, 0, "none", "bic", "quad"), is_benchmark=True)
    add("AR,AIC", "ar", ar_feat, model_selection=_ic_search(ar_grid, "aic"),
        tags=_tags(0, 0, "none", "aic", "quad"))
    add("AR,POOS", "ar", ar_feat, model_selection=_cv_search(ar_grid, "poos"),
        tags=_tags(0, 0, "none", "poos", "quad"))
    add("AR,KF", "ar", ar_feat, model_selection=_cv_search(ar_grid, "kfold"),
        tags=_tags(0, 0, "none", "kfold", "quad"))
    # RRAR (ridge on own lags)
    add("RRAR,POOS", "ridge", ar_feat, model_selection=_cv_search(ridge_grid, "poos"),
        tags=_tags(0, 0, "ridge", "poos", "quad"))
    add("RRAR,KF", "ridge", ar_feat, model_selection=_cv_search(ridge_grid, "kfold"),
        tags=_tags(0, 0, "ridge", "kfold", "quad"))
    # RFAR (random forest on own lags). HPs fixed at GCLS values (500 trees, mtry=1/3);
    # the two variants keep the CV characteristic tag for the treatment design.
    add("RFAR,POOS", "random_forest", ar_feat, params=rf_params,
        tags=_tags(0, 1, "none", "poos", "quad"),
        metadata={"note": "RF HPs fixed (GCLS 500/mtry=1/3); CV tag nominal"})
    add("RFAR,KF", "random_forest", ar_feat, params=rf_params,
        tags=_tags(0, 1, "none", "kfold", "quad"),
        metadata={"note": "RF HPs fixed (GCLS 500/mtry=1/3); CV tag nominal"})
    # KRRAR (kernel ridge, RBF) on own lags
    add("KRRAR,POOS", "kernel_ridge", ar_feat, params={"kernel": "rbf"},
        model_selection=_cv_search(krr_grid, "poos"),
        tags=_tags(0, 1, "ridge", "poos", "quad"), metadata={"kernel": "rbf"})
    add("KRRAR,KF", "kernel_ridge", ar_feat, params={"kernel": "rbf"},
        model_selection=_cv_search(krr_grid, "kfold"),
        tags=_tags(0, 1, "ridge", "kfold", "quad"), metadata={"kernel": "rbf"})
    # SVR-AR: Lin x2, RBF x2 (eps-insensitive loss)
    add("SVR-AR,Lin,POOS", "svr", ar_feat, params={"kernel": "linear", "max_iter": SVR_MAX_ITER},
        model_selection=_cv_search(svr_lin_grid, "poos"),
        tags=_tags(0, 0, "none", "poos", "eps"), metadata={"kernel": "linear"})
    add("SVR-AR,Lin,KF", "svr", ar_feat, params={"kernel": "linear", "max_iter": SVR_MAX_ITER},
        model_selection=_cv_search(svr_lin_grid, "kfold"),
        tags=_tags(0, 0, "none", "kfold", "eps"), metadata={"kernel": "linear"})
    add("SVR-AR,RBF,POOS", "svr", ar_feat, params={"kernel": "rbf", "max_iter": SVR_MAX_ITER},
        model_selection=_cv_search(svr_rbf_grid, "poos"),
        tags=_tags(0, 0, "none", "poos", "eps"), metadata={"kernel": "rbf"})
    add("SVR-AR,RBF,KF", "svr", ar_feat, params={"kernel": "rbf", "max_iter": SVR_MAX_ITER},
        model_selection=_cv_search(svr_rbf_grid, "kfold"),
        tags=_tags(0, 0, "none", "kfold", "eps"), metadata={"kernel": "rbf"})

    # ===================================================================== #
    # DATA-RICH (X=1)
    # ===================================================================== #
    fac_feat = _factor_features(target, ARDI_NF_FIXED, predictors)
    svr_fac_feat = _svr_ardi_features(target, predictors)  # z-scored factors for SVR-ARDI
    far_feat = _far_features(target, predictors)
    # ARDI via native ``far`` (now runs on the raw block after fix 4bf4435c): n_factors
    # (= n_f, the IC-selected K factors) and n_lag (= p_y) are MODEL params, so the 4 CV
    # variants IC/CV-select (n_f, p_y) per origin over the GCLS grid and are genuinely
    # DISTINCT. Residual fix-lane #1: ``far`` uses contemporaneous factors (p_f = 1); a
    # separately IC-selected factor-lag order p_f>1 still needs a feature-param search /
    # ARDI wrapper. This distinguishes ARDI's IC-selected K factors from B2/B3 keep-all.
    _ardi_meta = {
        "construction": "far (internal PCA) + IC/CV over (n_factors=n_f, n_lag=p_y)",
        "factor_count": "IC/CV-selected K (far n_factors)", "p_f": 1,
        "residual_fixlane1": "separate factor-lag order p_f>1 not native",
    }
    ardi_ms = {
        "BIC": _ic_search(ardi_grid, "bic"), "AIC": _ic_search(ardi_grid, "aic"),
        "POOS": _cv_search(ardi_grid, "poos"), "KF": _cv_search(ardi_grid, "kfold"),
    }
    for tag, cvname in (("BIC", "bic"), ("AIC", "aic"), ("POOS", "poos"), ("KF", "kfold")):
        add(f"ARDI,{tag}", "far", far_feat, model_selection=ardi_ms[tag],
            tags=_tags(1, 0, "none", cvname, "quad"), metadata=dict(_ardi_meta))
    # RRARDI / RFARDI / KRRARDI (y-lags + fixed-n_f PCA factors)
    add("RRARDI,POOS", "ridge", fac_feat, model_selection=_cv_search(ridge_grid, "poos"),
        tags=_tags(1, 0, "ridge", "poos", "quad"), metadata={"n_f": ARDI_NF_FIXED})
    add("RRARDI,KF", "ridge", fac_feat, model_selection=_cv_search(ridge_grid, "kfold"),
        tags=_tags(1, 0, "ridge", "kfold", "quad"), metadata={"n_f": ARDI_NF_FIXED})
    add("RFARDI,POOS", "random_forest", fac_feat, params=rf_params,
        tags=_tags(1, 1, "none", "poos", "quad"),
        metadata={"n_f": ARDI_NF_FIXED, "note": "RF HPs fixed; CV tag nominal"})
    add("RFARDI,KF", "random_forest", fac_feat, params=rf_params,
        tags=_tags(1, 1, "none", "kfold", "quad"),
        metadata={"n_f": ARDI_NF_FIXED, "note": "RF HPs fixed; CV tag nominal"})
    # KRR-ARDI: z-scored factors (svr_fac_feat) so the RBF kernel sees STANDARDISED
    # inputs (GCLS-faithful; the unscaled factors were the KRR-underperformance cause).
    add("KRRARDI,POOS", "kernel_ridge", svr_fac_feat, params={"kernel": "rbf"},
        model_selection=_cv_search(krr_grid, "poos"),
        tags=_tags(1, 1, "ridge", "poos", "quad"), metadata={"n_f": ARDI_NF_FIXED, "kernel": "rbf"})
    add("KRRARDI,KF", "kernel_ridge", svr_fac_feat, params={"kernel": "rbf"},
        model_selection=_cv_search(krr_grid, "kfold"),
        tags=_tags(1, 1, "ridge", "kfold", "quad"), metadata={"n_f": ARDI_NF_FIXED, "kernel": "rbf"})

    # ---- B1 / B2 / B3 shrinkage families (zeta in {0=ridge, zhat=EN, 1=lasso}) ----
    def _b_block(prefix: str, feat_builder, keepall: int | None, runtime: str) -> None:
        feat = (
            feat_builder(target, predictors) if keepall is None
            else feat_builder(target, keepall, predictors)
        )
        meta = {"runtime_status": runtime}
        for cv in ("poos", "kfold"):
            tag = cv.upper() if cv == "poos" else "KF"
            add(f"{prefix},ridge,{tag}", "ridge", feat,
                model_selection=_cv_search(ridge_grid, cv),
                tags=_tags(1, 0, f"{prefix}_z0", cv, "quad"), metadata=dict(meta))
            add(f"{prefix},EN,{tag}", "elastic_net", feat,
                model_selection=_cv_search(en_grid, cv),
                tags=_tags(1, 0, f"{prefix}_zhat", cv, "quad"), metadata=dict(meta))
            add(f"{prefix},lasso,{tag}", "lasso", feat,
                model_selection=_cv_search(ridge_grid, cv),
                tags=_tags(1, 0, f"{prefix}_z1", cv, "quad"), metadata=dict(meta))

    # B1/B3 use the raw wide-predictor block; they RUN once the package carries the
    # all-NaN fit-column prune (fix/raw-wide-feature-rows @ 4bf4435c). On an unpatched
    # 0.9.5 they zero-row (a fully-absent-over-window series nukes the fit sample).
    _b_block("B1", _identity_features, None, "runs (needs raw-wide-feature fix 4bf4435c)")
    _b_block("B2", _keepall_pca_features, B2_KEEPALL, "runs (pca_step keep-all)")
    _b_block("B3", _lag_then_keepall_pca_features, B3_KEEPALL,
             "runs (needs raw-wide-feature fix 4bf4435c)")

    # ---- SVR-ARDI: Lin x2, RBF x2 ----
    add("SVR-ARDI,Lin,POOS", "svr", svr_fac_feat, params={"kernel": "linear", "max_iter": SVR_MAX_ITER},
        model_selection=_cv_search(svr_lin_grid, "poos"),
        tags=_tags(1, 0, "none", "poos", "eps"), metadata={"kernel": "linear", "n_f": ARDI_NF_FIXED})
    add("SVR-ARDI,Lin,KF", "svr", svr_fac_feat, params={"kernel": "linear", "max_iter": SVR_MAX_ITER},
        model_selection=_cv_search(svr_lin_grid, "kfold"),
        tags=_tags(1, 0, "none", "kfold", "eps"), metadata={"kernel": "linear", "n_f": ARDI_NF_FIXED})
    add("SVR-ARDI,RBF,POOS", "svr", svr_fac_feat, params={"kernel": "rbf", "max_iter": SVR_MAX_ITER},
        model_selection=_cv_search(svr_rbf_grid, "poos"),
        tags=_tags(1, 0, "none", "poos", "eps"), metadata={"kernel": "rbf", "n_f": ARDI_NF_FIXED})
    add("SVR-ARDI,RBF,KF", "svr", svr_fac_feat, params={"kernel": "rbf", "max_iter": SVR_MAX_ITER},
        model_selection=_cv_search(svr_rbf_grid, "kfold"),
        tags=_tags(1, 0, "none", "kfold", "eps"), metadata={"kernel": "rbf", "n_f": ARDI_NF_FIXED})

    return arms


# Convenience: the canonical arm-name list + characteristic-count self-check.
def arm_names(target: str = "INDPRO") -> list[str]:
    return [a.name for a in build_gcls2022_arms(target)]


if __name__ == "__main__":
    arms = build_gcls2022_arms()
    poor = [a for a in arms if a.tags["X"] == 0]
    rich = [a for a in arms if a.tags["X"] == 1]
    print(f"total arms: {len(arms)}  (data-poor {len(poor)}, data-rich {len(rich)})")
    assert len(arms) == 46, f"expected 46 arms, got {len(arms)}"
    assert len(poor) == 14 and len(rich) == 32, "expected 14 data-poor + 32 data-rich"
    names = [a.name for a in arms]
    assert len(names) == len(set(names)), "duplicate arm names"
    bench = [a for a in arms if a.is_benchmark]
    assert len(bench) == 1 and bench[0].name == "AR,BIC", "benchmark must be AR,BIC"
    for a in arms:
        assert set(a.tags) == {"X", "NL", "SH", "CV", "LF"}, f"{a.name}: bad tag keys"
    print("OK: 46 arms, 14+32, unique names, benchmark=AR,BIC, all 5 tags present")
    for a in arms:
        print(f"  {a.name:22s} model={a.model:14s} tags={a.tags}")
