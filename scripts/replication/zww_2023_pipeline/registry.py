"""Zhang, Wahab & Wang (2023, IJF 39(2):486-502) crude-oil volatility replication.

Faithful 13-arm mapping (paper Table 3, main futures) expressed with the macroforecast
pipeline API.

KEY API RESOLUTION (why not ar/far)
-------------------------------------------
The pipeline builds target lags as RAW lags of the target COLUMN
(feature_engineering/builder.py -> lag(panel, columns=target_values) with no
transform). ZWW's forecast target must be built from the RAW realized variance RV via the
log_average_value transform (y = ln(mean RV_{t+1..t+h})). But ZWW models LV = ln(RV) and
its AR component uses LV lags -- NOT RV lags. The literal ar/far models key on
{target}_lag* = RV lags, so they cannot express the ZWW arms faithfully. We therefore
follow the design doc's fallback ("custom_model = OLS on [LV lags, F/X]"): every arm is an
ols/lasso/elastic_net regression over EXPLICIT LV-lag features (lag_step on the
LV column) plus the model-specific X block.

AUGMENTED-AR VARIABLE SCREENING (ZWW Eq. 3, p.489)
--------------------------------------------------
The t-stat and delta-R^2 screens are AUGMENTED-AR regressions: each candidate predictor is
scored AFTER partialling out the AR(L) lags (keep if |t|>1.65 or incremental R^2>0.01 over the
AR-only model). The lasso/elastic-net screens likewise operate on the AR-residualised system.
We pass controls = the AR(L) benchmark's exact regressors LV_lag0..LV_lag(L-1) =
[LV (=LV_t), LVCTRL1 (=LV_{t-1}), ..., LVCTRL{L-1}], materialised in the panel as
LVCTRL{k} = LV shifted k. predictor_screen always echoes controls in its output, so
we STRIP them with a custom_step before the PCA-VS PCA and the KS-VS feature block (the
forecasting models supply their own AR block via lag_step; controls are for the SCREEN
only -- no double-count).

Arm -> API table (13 arms)
--------------------------
  AR        : ols  on [LV lags]                                    (benchmark)
  PCA-VS x4 : ols  on [LV lags + PC1(screen_vs(X | AR controls))]  vs in {t_stat,delta_r2,lasso,elastic_net}
  Lasso     : lasso        on [LV lags + X], alpha by AICc
  ENet      : elastic_net  on [LV lags + X], l1_ratio=0.5, alpha by AICc
  PCA-all   : ols  on [LV lags + PC1(all X)]
  KS-all    : ols  on [LV lags + all X]
  KS-VS x4  : ols  on [LV lags + screen_vs(X | AR controls)]       vs in {t_stat,delta_r2,lasso,elastic_net}
"""
from __future__ import annotations

from typing import Any, Sequence

import macroforecast as mf
from macroforecast.model_selection import SearchSpec
from macroforecast.pipeline import Arm, TargetSpec

# ----------------------------------------------------------------------------- paper constants
TARGET_COL = "RV"                    # raw realized variance
LV_COL = "LV"                        # = ln(RV); source of the AR lag block AND lag0 control
TARGET_TRANSFORM = "log_average_value"   # y = ln(mean RV_{t+1..t+h})  (Eq. 2, p.488)
TARGET_POLICY = "direct"             # 'direct' preserves log_average_value (direct_average corrupts it)

# AR-control materialisation. LVCTRL{k} = LV_{t-k} (LV shifted by k). Lmax=6 = max AR lag.
LV_CTRL_PREFIX = "LVCTRL"
LMAX = 6
LV_CTRL_COLS: tuple[str, ...] = tuple(f"{LV_CTRL_PREFIX}{k}" for k in range(1, LMAX + 1))

# AR lag order L(h) presets (p.488 fn.3 / p.495 fn.15). Main Table 3 = AIC.
AR_LAG_PRESETS: dict[str, dict[int, int]] = {
    "aic": {1: 2, 3: 6, 6: 6, 12: 5},      # main (Table 3)
    "bic": {1: 2, 3: 2, 6: 1, 12: 1},      # Table 6 alternative
    "adj_r2": {1: 3, 3: 6, 6: 6, 12: 6},   # Table 6 alternative
}

# ZWW screening cuts (Sec. 2.2, p.489). Package defaults (t_stat=1.28, delta_r2=0.0) differ.
TSTAT_THRESHOLD = 1.65      # |t| > 1.65 (10%)
DELTA_R2_THRESHOLD = 0.01   # incremental R^2 > 1%
MIN_K = 5                   # min-5 rule: force top-5 if fewer pass the cut
ENET_L1_RATIO = 0.5         # rho = 0.5

# AICc lambda path (Hurvich-Tsai corrected AIC; p.489 fn.5-6). [ASSUMPTION] 9-point log grid.
ALPHA_GRID: tuple[float, ...] = (
    1e-4, 3.1623e-4, 1e-3, 3.1623e-3, 1e-2, 3.1623e-2, 0.1, 0.31623, 1.0,
)

VS_METHODS: tuple[str, ...] = ("t_stat", "delta_r2", "lasso", "elastic_net")

# PCA scaling. ZWW's "conventional PCA" (Eq. 7-8) extracts PC1 from the tcode'd predictors
# WITHOUT re-standardizing -> COVARIANCE PCA (scale=False). Decisive: VXOCLSx is tcode=1
# (level, std ~8.2, the largest-variance predictor), so covariance PC1 loads on VXO (the
# option-implied vol index ZWW reports as dominant, selected 98.4%), giving the paper's
# positive PCA-VS. Correlation PCA (standardize first) makes every series unit-variance, the
# collinear macro cluster dominates PC1, VXO is drowned, and PCA-VS turns negative. Verified
# full-OOS futures: covariance PCA reproduces ZWW within ~1-2pp at every horizon (PCA-t_stat
# h=3/6/12 = +1.4/+5.9/+8.2%% vs ZWW +1.8/+4.6/~+6%%; PCA-lasso h=6 = +3.9%% vs +4.8%%),
# whereas standardized PCA gave -1.2/-18.7/+3.7%%.
PCA_SCALE = False  # covariance PCA (see note above)

ARM_ORDER: tuple[str, ...] = (
    "AR",
    "PCA-t_stat", "PCA-delta_r2", "PCA-lasso", "PCA-elastic_net",
    "Lasso", "ENet",
    "PCA-all", "KS-all",
    "KS-t_stat", "KS-delta_r2", "KS-lasso", "KS-elastic_net",
)


def _drop_ar_controls(frame, metadata=None, **params):
    """custom_step callable: drop the AR-control columns (LV + LVCTRL*) echoed by the screen,
    so they never double-count as features (the model supplies its own AR block)."""
    drop = [c for c in frame.columns if c == LV_COL or str(c).startswith(LV_CTRL_PREFIX)]
    return frame.drop(columns=drop)


def _ar_controls(lag: int) -> list[str]:
    # Exactly the AR(L) benchmark regressors LV_lag0..LV_lag(L-1) = LV_t..LV_{t-L+1}.
    # lag0 = the LV column; lag k>=1 = LVCTRL{k}. So the screen's null model IS the benchmark.
    return [LV_COL] + [f"{LV_CTRL_PREFIX}{k}" for k in range(1, int(lag))]


def _aicc_search(*, l1_ratio: float | None = None) -> SearchSpec:
    grid: dict[str, tuple] = {"alpha": ALPHA_GRID}
    if l1_ratio is not None:
        grid["l1_ratio"] = (float(l1_ratio),)
    return SearchSpec(method="information_criterion", criterion="aicc", param_grid=grid)


def _ar_lag_step(lag: int):
    # AR component of the FORECASTING model: LV lags 0..L-1 (lag0 = observed LV_t).
    return mf.feature_engineering.lag_step(
        name="ARLV", columns=[LV_COL], lags=range(0, int(lag)), include=True
    )


def _screen_step(vs: str, predictors: Sequence[str], controls: list[str], *, name: str):
    # Augmented-AR screen (include=False -> named intermediate); candidates = 126 FRED only.
    kwargs: dict = dict(
        method=vs, name=name, columns=list(predictors), controls=list(controls),
        min_k=MIN_K, include=False,
    )
    if vs == "t_stat":
        kwargs["threshold"] = TSTAT_THRESHOLD
    elif vs == "delta_r2":
        kwargs["threshold"] = DELTA_R2_THRESHOLD
    elif vs == "lasso":
        kwargs["lambda_search"] = _aicc_search()
    elif vs == "elastic_net":
        kwargs["l1_ratio"] = ENET_L1_RATIO
        kwargs["lambda_search"] = _aicc_search(l1_ratio=ENET_L1_RATIO)
    else:
        raise ValueError(f"unknown vs method {vs!r}")
    return mf.feature_engineering.predictor_screen(**kwargs)


def _drop_controls_step(*, name: str, include: bool):
    return mf.feature_engineering.custom_step(
        name=name, func=_drop_ar_controls, input="screen", include=include
    )


def _feature_spec(predictors: Sequence[str], steps: list):
    # Source frame = X block + LV (AR lag source / lag0 control) + all LVCTRL controls. Every
    # X-block step references the 126 FRED predictors explicitly, so LV/LVCTRL never leak into
    # X/PCA/screen candidates. include_original=False + feature_steps suppress the default block.
    return mf.feature_engineering.feature_spec(
        target=TARGET_COL,
        predictors=list(predictors) + [LV_COL] + list(LV_CTRL_COLS),
        steps=steps,
        target_lags=None,
        target_transform=TARGET_TRANSFORM,
        include_original=False,
    )


def zww_target() -> TargetSpec:
    return TargetSpec(name=TARGET_COL, transform=TARGET_TRANSFORM, policy=TARGET_POLICY)


def zww_arms(
    horizon: int,
    predictors: Sequence[str],
    *,
    ar_lags: str = "aic",
    smoke: bool = False,
) -> list[Arm]:
    """Build the 13 ZWW arms for one horizon (AR lag order L(h) fixed by preset)."""
    L = AR_LAG_PRESETS[ar_lags][int(horizon)]
    P = list(predictors)
    controls = _ar_controls(L)

    def ar_step():
        return _ar_lag_step(L)

    arms: list[Arm] = [
        Arm(
            name="AR", model="ols", features=_feature_spec(P, [ar_step()]),
            is_benchmark=True, metadata={"role": "benchmark", "L": L, "eq": "AR(L)"},
        )
    ]

    # PCA-VS x4: ols on [LV lags + PC1(screened X)]; screen partialled on AR controls; controls
    # stripped before PCA (PCA over selected predictors only); AR block via lag_step.
    for vs in VS_METHODS:
        steps = [
            ar_step(),
            _screen_step(vs, P, controls, name="screen"),
            _drop_controls_step(name="screenP", include=False),
            mf.feature_engineering.pca_step(
                name="F", input="screenP", n_components=1, scale=PCA_SCALE, include=True
            ),
        ]
        arms.append(Arm(
            name=f"PCA-{vs}", model="ols", features=_feature_spec(P, steps),
            nested_in_benchmark=True,
            metadata={"role": "pca-vs", "vs": vs, "L": L, "eq": "Eq.9", "screen": "augmented-AR"},
        ))

    # Lasso / ENet: penalized regression on [LV lags + all X], alpha by AICc.
    # [ASSUMPTION] AR lags included in the penalized design (ZWW does not state penalty-exempt).
    xall = mf.feature_engineering.lag_step(name="X", columns=P, lags=(0,), include=True)
    arms.append(Arm(
        name="Lasso", model="lasso", features=_feature_spec(P, [ar_step(), xall]),
        params={"standardize": True}, model_selection=_aicc_search(),
        nested_in_benchmark=True, metadata={"role": "penalized", "L": L, "lambda": "aicc"},
    ))
    arms.append(Arm(
        name="ENet", model="elastic_net", features=_feature_spec(P, [ar_step(), xall]),
        params={"l1_ratio": ENET_L1_RATIO, "standardize": True},
        model_selection=_aicc_search(l1_ratio=ENET_L1_RATIO),
        nested_in_benchmark=True, metadata={"role": "penalized", "L": L, "lambda": "aicc"},
    ))

    # PCA-all: ols on [LV lags + PC1(all X)]
    arms.append(Arm(
        name="PCA-all", model="ols",
        features=_feature_spec(P, [ar_step(), mf.feature_engineering.pca_step(
            name="F", columns=P, n_components=1, scale=PCA_SCALE, include=True)]),
        nested_in_benchmark=True, metadata={"role": "pca-all", "L": L, "eq": "Eq.8"},
    ))

    # KS-all: ols on [LV lags + all X]
    arms.append(Arm(
        name="KS-all", model="ols", features=_feature_spec(P, [ar_step(), xall]),
        nested_in_benchmark=True, metadata={"role": "ks-all", "L": L, "eq": "Eq.4"},
    ))

    # KS-VS x4: ols on [LV lags + screened X]; screen partialled on AR controls; controls
    # stripped so only the selected predictors are added (AR block via lag_step).
    for vs in VS_METHODS:
        steps = [
            ar_step(),
            _screen_step(vs, P, controls, name="screen"),
            _drop_controls_step(name="screenP", include=True),
        ]
        arms.append(Arm(
            name=f"KS-{vs}", model="ols", features=_feature_spec(P, steps),
            nested_in_benchmark=True,
            metadata={"role": "ks-vs", "vs": vs, "L": L, "eq": "Eq.10", "screen": "augmented-AR"},
        ))

    if smoke:
        keep = {"AR", "PCA-t_stat", "Lasso"}
        arms = [a for a in arms if a.name in keep]
    return arms
