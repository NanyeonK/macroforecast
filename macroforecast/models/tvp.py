from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator


def _default_lambda_candidates() -> np.ndarray:
    return np.exp(np.linspace(-6.0, 20.0, num=15))


@dataclass(frozen=True)
class _GRRResult:
    uhat: np.ndarray
    betas_grr: np.ndarray
    yhat: np.ndarray
    sigmasq: np.ndarray
    lambdas: tuple[float, float]


class TVPRidgeRegressor:
    """Time-varying parameter ridge / 2SRR estimator.

    Source alignment:
    - R package: `TVPRidge`
    - Source file: `R/MV2SRR_v210407.R`
    - Main R callable: `tvp.ridge`

    The implementation keeps the R decomposition:
    `Zfun` basis expansion -> `dualGRR` generalized ridge ->
    `CV.KF.MV` / `cv.univariate` lambda tuning -> 2SRR variance
    reweighting. Indexing changes are explicit because R is 1-based and NumPy
    is 0-based.
    """

    def __init__(
        self,
        *,
        lambda_candidates: Any | None = None,
        oosX: Any | None = None,
        lambda2: float = 0.1,
        kfold: int = 5,
        cv_plot: bool = False,
        cv_2srr: bool = True,
        sig_u_param: float = 0.75,
        sig_eps_param: float = 0.75,
        ols_prior: bool = False,
        random_state: int = 1071,
        use_garch: bool = True,
    ) -> None:
        self.lambda_candidates = lambda_candidates
        self.oosX = oosX
        self.lambda2 = float(lambda2)
        self.kfold = int(kfold)
        self.cv_plot = bool(cv_plot)
        self.cv_2srr = bool(cv_2srr)
        self.sig_u_param = float(sig_u_param)
        self.sig_eps_param = float(sig_eps_param)
        self.ols_prior = bool(ols_prior)
        self.random_state = int(random_state)
        self.use_garch = bool(use_garch)

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> "TVPRidgeRegressor":
        frame = pd.DataFrame(X).astype(float)
        target = _as_target_matrix(y, index=frame.index)
        if len(frame) != len(target):
            raise ValueError("X and y must have the same number of observations")
        if frame.empty:
            raise ValueError("X must contain observations")
        if frame.shape[1] == 0:
            raise ValueError("X must contain at least one predictor")

        x_values = frame.to_numpy(dtype=float)
        y_values = target.to_numpy(dtype=float)
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise ValueError("TVP ridge requires finite X and y values")

        lambdavec = _resolve_lambda_candidates(self.lambda_candidates)
        dim_x = x_values.shape[1] + 1
        n_targets = y_values.shape[1]
        n_obs = x_values.shape[0]
        x_sd = _sample_sd(x_values, axis=0)
        y_sd = _sample_sd(y_values, axis=0)
        scaling_factor = y_sd.reshape(-1, 1) / x_sd.reshape(1, -1)
        x_scaled = x_values / x_sd.reshape(1, -1)
        y_scaled = y_values / y_sd.reshape(1, -1)
        z_basis = _tvp_z_basis(x_scaled)

        betas_rr = np.zeros((n_targets, dim_x, n_obs), dtype=float)
        betas_2srr = np.zeros_like(betas_rr)
        yhat_rr = np.zeros((n_obs, n_targets), dtype=float)
        yhat_2srr = np.zeros_like(yhat_rr)
        lambdas = np.zeros(n_targets, dtype=float)
        lambda_step2 = np.zeros(n_targets, dtype=float)
        eps_weights = np.ones((n_obs, n_targets), dtype=float)
        garch_status: list[str] = []
        cv_records: list[dict[str, Any]] = []

        if len(lambdavec) > 1:
            cv_mv = _cv_kfold_multivariate(
                y_scaled,
                z_basis,
                k=self.kfold,
                lambdavec=lambdavec,
                lambda2=self.lambda2,
                dim_x=dim_x,
                random_state=self.random_state,
            )
            lambda_list = np.asarray(cv_mv["lambdas_het"], dtype=float)
            cv_records.append({"stage": "rr", **cv_mv})
        else:
            lambda_list = np.repeat(float(lambdavec[0]), n_targets)

        for m in range(n_targets):
            lambda1 = float(lambda_list[m])
            grr = _dual_generalized_ridge(
                z_basis,
                y_scaled[:, [m]],
                dim_x=dim_x,
                lambda1=lambda1,
                lambda2=self.lambda2,
                sweights=1.0,
                eweights=1.0,
                ols_prior=self.ols_prior,
            )
            betas_rr[m, :, :] = grr.betas_grr[0, :, :]
            yhat_rr[:, m] = grr.yhat[:, 0]
            lambdas[m] = lambda1

            residual = _glitch_filtered_residual(y_scaled[:, m] - grr.yhat[:, 0])
            ew, status = _estimate_garch_weights(
                residual,
                power=self.sig_eps_param,
                use_garch=self.use_garch,
            )
            eps_weights[:, m] = ew
            garch_status.append(status)

            sigmasq = _coefficient_innovation_weights(
                betas_rr[m, :, :],
                power=self.sig_u_param,
            )
            if self.cv_2srr and len(lambdavec) > 1:
                cv_second = _cv_univariate(
                    y_scaled[:, [m]],
                    z_basis,
                    k=self.kfold,
                    lambdavec=lambdavec,
                    lambda2=self.lambda2,
                    dim_x=dim_x,
                    sweights=sigmasq,
                    eweights=ew,
                    random_state=self.random_state,
                )
                use_lambda = float(cv_second["minimizer"])
                cv_records.append({"stage": f"2srr_target_{m}", **cv_second})
            else:
                use_lambda = lambda1
            lambda_step2[m] = use_lambda

            if self.sig_u_param > 0.0 or self.sig_eps_param > 0.0:
                grrats = _dual_generalized_ridge(
                    z_basis,
                    y_scaled[:, [m]],
                    dim_x=dim_x,
                    lambda1=use_lambda,
                    lambda2=self.lambda2,
                    sweights=sigmasq,
                    eweights=ew,
                    ols_prior=self.ols_prior,
                )
                betas_2srr[m, :, :] = grrats.betas_grr[0, :, :]
                yhat_2srr[:, m] = grrats.yhat[:, 0]
            else:
                betas_2srr[m, :, :] = grr.betas_grr[0, :, :]
                yhat_2srr[:, m] = grr.yhat[:, 0]

        betas_rr, betas_2srr, yhat_rr, yhat_2srr = _rescale_tvp_outputs(
            betas_rr,
            betas_2srr,
            yhat_rr,
            yhat_2srr,
            scaling_factor=scaling_factor,
            y_sd=y_sd,
        )

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.target_names_ = tuple(str(c) for c in target.columns)
        self.training_index_ = frame.index.copy()
        self.betas_rr_ = betas_rr
        self.betas_2srr_ = betas_2srr
        self.lambdas_ = lambdas
        self.lambda_step2_ = lambda_step2
        self.yhat_rr_ = pd.DataFrame(yhat_rr, index=frame.index, columns=target.columns)
        self.yhat_2srr_ = pd.DataFrame(
            yhat_2srr,
            index=frame.index,
            columns=target.columns,
        )
        self.sig_eps_ = pd.DataFrame(
            eps_weights / float(np.mean(eps_weights)),
            index=frame.index,
            columns=target.columns,
        )
        self.forecast_ = self._forecast_from_oosx()
        self.intercept_ = float(betas_2srr[0, 0, -1])
        self.coef_ = betas_2srr[0, 1:, -1].copy()
        self.intercept_path_ = pd.Series(
            betas_2srr[0, 0, :],
            index=frame.index,
            name="intercept",
        )
        self.coef_path_ = pd.DataFrame(
            betas_2srr[0, 1:, :].T,
            index=frame.index,
            columns=[str(c) for c in frame.columns],
        )
        self.coef_path_full_ = _full_coef_path_frame(
            betas_2srr,
            index=frame.index,
            target_names=self.target_names_,
            feature_names=[str(c) for c in frame.columns],
        )
        self.diagnostics_ = {
            "implementation": "Python port of TVPRidge R/MV2SRR_v210407.R",
            "garch_status": tuple(garch_status),
            "cv_records": cv_records,
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "betas_2srr_"):
            raise RuntimeError("TVPRidgeRegressor must be fitted before predict")
        frame = pd.DataFrame(X).reindex(
            columns=list(self.feature_names_in_),
            fill_value=0.0,
        )
        frame = frame.astype(float)
        if (
            hasattr(self, "training_index_")
            and len(frame) == len(self.training_index_)
            and frame.index.equals(self.training_index_)
        ):
            values = self.yhat_2srr_.to_numpy(dtype=float)
            return values[:, 0] if values.shape[1] == 1 else values
        x_aug = np.column_stack(
            [np.ones(len(frame), dtype=float), frame.to_numpy(dtype=float)]
        )
        beta_last = self.betas_2srr_[:, :, -1]
        pred = x_aug @ beta_last.T
        return pred[:, 0] if pred.shape[1] == 1 else pred

    def _forecast_from_oosx(self) -> np.ndarray:
        if self.oosX is None:
            return np.asarray([], dtype=float)
        raw = np.asarray(self.oosX, dtype=float).reshape(-1)
        # R source cue:
        #   if(length(oosX)>1) fcast <- BETAS_VARF[,,T] %*% append(1,oosX)
        # The R condition accidentally omits one-predictor systems. Python
        # treats any supplied vector with the right feature count as a forecast
        # request while keeping the same last-beta forecasting rule.
        if raw.size != len(self.feature_names_in_):
            raise ValueError(
                "oosX must contain exactly one value per fitted predictor"
            )
        x_aug = np.concatenate([[1.0], raw])
        return self.betas_2srr_[:, :, -1] @ x_aug


def tvp_ridge(
    X: Any,
    y: Any | None = None,
    *,
    lambda_candidates: Any | None = None,
    oosX: Any | None = None,
    lambda2: float = 0.1,
    kfold: int = 5,
    cv_plot: bool = False,
    cv_2srr: bool = True,
    sig_u_param: float = 0.75,
    sig_eps_param: float = 0.75,
    ols_prior: bool = False,
    random_state: int = 1071,
    use_garch: bool = True,
) -> ModelFit:
    """Fit Goulet Coulombe TVP ridge / 2SRR as a macroforecast model."""

    params = {
        "lambda_candidates": (
            None
            if lambda_candidates is None
            else tuple(float(v) for v in np.asarray(lambda_candidates).reshape(-1))
        ),
        "oosX": None if oosX is None else tuple(float(v) for v in np.asarray(oosX).reshape(-1)),
        "lambda2": float(lambda2),
        "kfold": int(kfold),
        "cv_plot": bool(cv_plot),
        "cv_2srr": bool(cv_2srr),
        "sig_u_param": float(sig_u_param),
        "sig_eps_param": float(sig_eps_param),
        "ols_prior": bool(ols_prior),
        "random_state": int(random_state),
        "use_garch": bool(use_garch),
    }
    estimator = TVPRidgeRegressor(
        lambda_candidates=lambda_candidates,
        oosX=oosX,
        lambda2=lambda2,
        kfold=kfold,
        cv_plot=cv_plot,
        cv_2srr=cv_2srr,
        sig_u_param=sig_u_param,
        sig_eps_param=sig_eps_param,
        ols_prior=ols_prior,
        random_state=random_state,
        use_garch=use_garch,
    )
    return fit_estimator(estimator, X, y, model="tvp_ridge", metadata=params)


def _as_target_matrix(y: Any, *, index: pd.Index) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        target = y.copy()
    elif isinstance(y, pd.Series):
        target = y.to_frame(name=y.name or "y")
    else:
        arr = np.asarray(y, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"y must be 1-D or 2-D, got shape {arr.shape!r}")
        target = pd.DataFrame(
            arr,
            index=index,
            columns=[f"y{i}" for i in range(arr.shape[1])],
        )
    if len(target) != len(index):
        raise ValueError("y must have the same number of rows as X")
    if not target.index.equals(index):
        target = target.copy()
        target.index = index
    return target.astype(float)


def _resolve_lambda_candidates(values: Any | None) -> np.ndarray:
    if values is None:
        out = _default_lambda_candidates()
    else:
        out = np.asarray(values, dtype=float).reshape(-1)
    if out.size == 0:
        raise ValueError("lambda_candidates must contain at least one value")
    if np.any(out <= 0.0) or not np.isfinite(out).all():
        raise ValueError("lambda_candidates must be positive finite values")
    return out


def _sample_sd(values: np.ndarray, *, axis: int) -> np.ndarray:
    sd = np.std(values, axis=axis, ddof=1)
    return np.where(np.isfinite(sd) & (sd > 0.0), sd, 1.0)


def _tvp_z_basis(data: np.ndarray) -> np.ndarray:
    """Port of R `Zfun(data)`.

    R source cue:
    - `X = cbind(1, data)`
    - `Z[tt, 1:(tt-1), ] = repmat(X[tt,], tt-1, 1)` for `tt=2:T`
    - `Zprime = cbind(Z[,,1], ..., Z[,,K+1], X)`

    Python mapping:
    - row `tt=2` in R is `t=1` in NumPy;
    - each coefficient block has `T-1` innovation columns;
    - the final `K+1` columns are the starting values / static block.
    """

    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must be 2-D")
    n_obs, n_features = values.shape
    x_aug = np.column_stack([np.ones(n_obs, dtype=float), values])
    z = np.zeros((n_obs, max(n_obs - 1, 0), n_features + 1), dtype=float)
    for t in range(1, n_obs):
        z[t, :t, :] = x_aug[t, :]
    blocks = [z[:, :, k] for k in range(n_features + 1)]
    return np.column_stack([*blocks, x_aug])


def _dual_generalized_ridge(
    zprime: np.ndarray,
    y: np.ndarray,
    *,
    dim_x: int,
    lambda1: float,
    lambda2: float = 0.001,
    sweights: Any = 1.0,
    eweights: Any = 1.0,
    ols_prior: bool = False,
    nf: int | None = None,
) -> _GRRResult:
    """Port of R `dualGRR`.

    R source cues:
    - `Kmat_half <- t(Zprime)`
    - innovation rows are scaled by `(1/lambda1) * sweigths[m]`
    - static beta rows are scaled by `1/lambda2`
    - dual branch solves `(Zprime %*% Kmat_half + Lambda_T) alpha = y`
    - beta paths are recovered by cumulatively summing innovations.
    """

    z = np.asarray(zprime, dtype=float)
    yy = np.asarray(y, dtype=float)
    if yy.ndim == 1:
        yy = yy.reshape(-1, 1)
    if z.shape[0] != yy.shape[0]:
        raise ValueError("zprime and y must have the same row count")
    if lambda1 <= 0.0:
        raise ValueError("lambda1 must be positive")
    lambda2 = max(float(lambda2), 1e-7)
    nf = int(dim_x if nf is None else nf)
    ncol_z = z.shape[1]
    n_obs = z.shape[0]
    n_time = int((ncol_z - dim_x) / nf + 1)
    if nf * (n_time - 1) + dim_x != ncol_z:
        raise ValueError("zprime shape is inconsistent with dim_x and nf")

    sw = _as_positive_weights(sweights, nf)
    ew = _as_positive_weights(eweights, n_obs)
    kmat_half = z.T.copy()
    for m in range(nf):
        begin = m * (n_time - 1)
        end = (m + 1) * (n_time - 1)
        kmat_half[begin:end, :] *= (1.0 / float(lambda1)) * sw[m]
        if nf < dim_x and begin < end:
            kmat_half[begin, :] *= 1000.0
    kmat_half[(ncol_z - dim_x) : ncol_z, :] *= 1.0 / lambda2

    param = nf * (n_time - 1) + dim_x
    if param > n_obs:
        kmat_du = z @ kmat_half
        rhs = yy.copy()
        beta_ols = None
        if ols_prior:
            x_static = z[:, (ncol_z - dim_x) : ncol_z]
            beta_ols = _safe_solve(x_static.T @ x_static, x_static.T @ yy)
            rhs = yy - x_static @ beta_ols
        alpha = _safe_solve(kmat_du + np.diag(ew), rhs)
        uhat = kmat_half @ alpha
        if beta_ols is not None:
            uhat[(ncol_z - dim_x) : ncol_z, :] += beta_ols
        yhat = kmat_du @ alpha
    else:
        kmat_half_weighted = kmat_half.copy()
        kmat_half_weighted *= (1.0 / ew.reshape(1, -1))
        kmat_pri = kmat_half_weighted @ z
        if ols_prior:
            x_static = z[:, (ncol_z - dim_x) : ncol_z]
            beta_ols = _safe_solve(x_static.T @ x_static, x_static.T @ yy)
            rhs = kmat_half_weighted @ (yy - x_static @ beta_ols)
            uhat = _safe_solve(kmat_pri + np.eye(param), rhs)
            uhat[(ncol_z - dim_x) : ncol_z, :] += beta_ols
        else:
            rhs = kmat_half_weighted @ yy
            uhat = _safe_solve(kmat_pri + np.eye(param), rhs)
        yhat = z @ uhat

    betas = np.zeros((yy.shape[1], nf, n_time), dtype=float)
    for eq in range(yy.shape[1]):
        betas[eq, :, 0] = uhat[(uhat.shape[0] - dim_x) : (uhat.shape[0] - dim_x + nf), eq]
        for t in range(1, n_time):
            positions = np.asarray(
                [m * (n_time - 1) + (t - 1) for m in range(nf)],
                dtype=int,
            )
            betas[eq, :, t] = betas[eq, :, t - 1] + uhat[positions, eq]
    return _GRRResult(
        uhat=uhat,
        betas_grr=betas,
        yhat=yhat,
        sigmasq=sw,
        lambdas=(float(lambda1), float(lambda2)),
    )


def _cv_univariate(
    y: np.ndarray,
    z_basis: np.ndarray,
    *,
    k: int,
    lambdavec: np.ndarray,
    lambda2: float,
    dim_x: int,
    sweights: Any = 1.0,
    eweights: Any = 1.0,
    random_state: int = 1071,
    nf: int | None = None,
) -> dict[str, Any]:
    return _cv_kfold_core(
        y,
        z_basis,
        k=k,
        lambdavec=lambdavec,
        lambda2=lambda2,
        dim_x=dim_x,
        sweights=sweights,
        eweights=eweights,
        random_state=random_state,
        nf=nf,
        heterogeneous=False,
    )


def _cv_kfold_multivariate(
    y: np.ndarray,
    z_basis: np.ndarray,
    *,
    k: int,
    lambdavec: np.ndarray,
    lambda2: float,
    dim_x: int,
    sweights: Any = 1.0,
    eweights: Any = 1.0,
    random_state: int = 1071,
    nf: int | None = None,
) -> dict[str, Any]:
    return _cv_kfold_core(
        y,
        z_basis,
        k=k,
        lambdavec=lambdavec,
        lambda2=lambda2,
        dim_x=dim_x,
        sweights=sweights,
        eweights=eweights,
        random_state=random_state,
        nf=nf,
        heterogeneous=True,
    )


def _cv_kfold_core(
    y: np.ndarray,
    z_basis: np.ndarray,
    *,
    k: int,
    lambdavec: np.ndarray,
    lambda2: float,
    dim_x: int,
    sweights: Any,
    eweights: Any,
    random_state: int,
    nf: int | None,
    heterogeneous: bool,
) -> dict[str, Any]:
    """Port of R `cv.univariate` and `CV.KF.MV`.

    R source cue:
    - `set.seed(1071)`
    - `index <- sample(1:k, nrow(data), replace = TRUE)`
    - use FWL residualization over the static `X` block;
    - apply `1 + cumul_zeros(bindex)` dropout correction to innovation
      columns before testing each lambda.
    """

    yy = np.asarray(y, dtype=float)
    if yy.ndim == 1:
        yy = yy.reshape(-1, 1)
    z = np.asarray(z_basis, dtype=float)
    n_obs = z.shape[0]
    n_targets = yy.shape[1]
    k = max(2, min(int(k), n_obs))
    lambdas = np.asarray(lambdavec, dtype=float).reshape(-1)
    nf = int(dim_x if nf is None else nf)
    n_time = int((z.shape[1] - dim_x) / nf + 1)
    rng = np.random.default_rng(int(random_state))
    fold_index = rng.integers(1, k + 1, size=n_obs)
    pmse = np.full((k, len(lambdas)), np.nan, dtype=float)
    pmse_het = np.full((k, len(lambdas), n_targets), np.nan, dtype=float)

    sw = _as_positive_weights(sweights, nf)
    ew = _as_positive_weights(eweights, n_obs)

    for fold in range(1, k + 1):
        test_mask = fold_index == fold
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        z_train = z[train_mask, :]
        z_test = z[test_mask, :]
        y_train = yy[train_mask, :]
        y_test = yy[test_mask, :]
        ew_train = ew[train_mask]
        do_factor = 1.0 + _cumul_zeros((~test_mask).astype(int))

        mxz, mx_y_train = _residualize_static(
            z_train,
            y_train,
            dim_x=dim_x,
            lambda2=lambda2,
        )
        mxz = _scale_innovation_blocks(
            mxz,
            n_time=n_time,
            nf=nf,
            sweights=sw,
            dropout_factor=do_factor,
            dim_x=dim_x,
        )
        mxz2 = mxz * (1.0 / ew_train.reshape(-1, 1))
        param = nf * (n_time - 1)

        mxz_test, mx_y_test = _residualize_static(
            z_test,
            y_test,
            dim_x=dim_x,
            lambda2=lambda2,
        )
        mxz_test = _scale_innovation_blocks(
            mxz_test,
            n_time=n_time,
            nf=nf,
            sweights=sw,
            dropout_factor=None,
            dim_x=dim_x,
        )

        if param > n_obs:
            kmat = mxz @ mxz.T
            kmat_test = mxz_test @ mxz.T
            rhs = mx_y_train
            for j, candidate in enumerate(lambdas):
                pred = kmat_test @ _safe_solve(kmat + candidate * np.diag(ew_train), rhs)
                losses = (pred - mx_y_test) ** 2
                pmse[fold - 1, j] = float(np.mean(losses))
                pmse_het[fold - 1, j, :] = np.mean(losses, axis=0)
        else:
            kmat = mxz2.T @ mxz
            rhs = mxz2.T @ mx_y_train
            for j, candidate in enumerate(lambdas):
                pred = mxz_test @ _safe_solve(kmat + candidate * np.eye(kmat.shape[0]), rhs)
                losses = (pred - mx_y_test) ** 2
                pmse[fold - 1, j] = float(np.mean(losses))
                pmse_het[fold - 1, j, :] = np.mean(losses, axis=0)

    score = np.nanmean(pmse, axis=0)
    if not np.isfinite(score).any():
        score = np.repeat(np.inf, len(lambdas))
        score[0] = 0.0
    min_pos = int(np.nanargmin(score))
    out: dict[str, Any] = {
        "minimizer": float(lambdas[min_pos]),
        "minima": float(score[min_pos]),
        "score": tuple(float(v) for v in score),
    }
    if len(lambdas) > 1:
        se = _safe_fold_se(pmse, min_pos)
        out["minimizer1se"] = float(lambdas[_one_se_index(score, min_pos, se, 1.0)])
        out["minimizer2se"] = float(lambdas[_one_se_index(score, min_pos, se, 2.0)])
    if heterogeneous:
        score_het = np.nanmean(pmse_het, axis=0)
        lambdas_het = []
        for target_idx in range(n_targets):
            target_score = score_het[:, target_idx]
            if not np.isfinite(target_score).any():
                lambdas_het.append(float(lambdas[min_pos]))
            else:
                lambdas_het.append(float(lambdas[int(np.nanargmin(target_score))]))
        out["lambdas_het"] = tuple(lambdas_het)
    return out


def _residualize_static(
    z: np.ndarray,
    y: np.ndarray,
    *,
    dim_x: int,
    lambda2: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_static = z[:, -dim_x:]
    ridge = x_static.T @ x_static + float(lambda2) * np.eye(dim_x)
    hat = x_static @ _safe_solve(ridge, x_static.T)
    mx = np.eye(z.shape[0]) - hat
    return mx @ z, mx @ y


def _scale_innovation_blocks(
    z: np.ndarray,
    *,
    n_time: int,
    nf: int,
    sweights: np.ndarray,
    dropout_factor: np.ndarray | None,
    dim_x: int,
) -> np.ndarray:
    out = z.copy()
    for m in range(nf):
        begin = m * (n_time - 1)
        end = (m + 1) * (n_time - 1)
        out[:, begin:end] *= sweights[m]
        if nf < dim_x and begin < end:
            out[:, begin] *= 1000.0
        if dropout_factor is not None:
            out[:, begin:end] *= dropout_factor[1:n_time].reshape(1, -1)
    return out


def _cumul_zeros(x: Any) -> np.ndarray:
    """Literal NumPy port of the R `cumul_zeros` helper."""

    values = ~np.asarray(x, dtype=bool)
    if values.size == 0:
        return np.asarray([], dtype=float)
    lengths: list[int] = []
    states: list[bool] = []
    start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != values[start]:
            lengths.append(idx - start)
            states.append(bool(values[start]))
            start = idx
    cum_len = np.cumsum(lengths)
    z = values.astype(float)
    diffs = np.diff(np.asarray(states, dtype=int))
    i_drops = np.concatenate([[False], diffs < 0])
    shifted = np.concatenate([i_drops[1:], [False]])
    replacement_lengths = np.asarray(lengths, dtype=float)[shifted]
    drop_positions = cum_len[i_drops] - 1
    if len(drop_positions):
        z[drop_positions] = -replacement_lengths
    return values.astype(float) * np.cumsum(z)


def _coefficient_innovation_weights(betas: np.ndarray, *, power: float) -> np.ndarray:
    if power <= 0.0:
        return np.ones(betas.shape[0], dtype=float)
    innovations = np.diff(betas, axis=1)
    raw = np.sum(innovations**2, axis=1) ** float(power)
    mean = float(np.mean(raw)) if raw.size else 1.0
    if mean <= 0.0 or not np.isfinite(mean):
        return np.ones(betas.shape[0], dtype=float)
    return np.where(np.isfinite(raw), raw / mean, 1.0)


def _estimate_garch_weights(
    residual: np.ndarray,
    *,
    power: float,
    use_garch: bool,
) -> tuple[np.ndarray, str]:
    if power <= 0.0:
        return np.ones_like(residual, dtype=float), "homogeneous"
    if not use_garch:
        return np.ones_like(residual, dtype=float), "disabled"
    try:
        from arch import arch_model  # type: ignore[import-not-found]
    except ImportError:
        return np.ones_like(residual, dtype=float), "arch_not_installed"

    try:
        model = arch_model(
            residual,
            mean="Zero",
            vol="Garch",
            p=1,
            q=1,
            rescale=False,
        )
        result = model.fit(disp="off")
        sigma = np.asarray(result.conditional_volatility, dtype=float)
        weights = sigma ** float(power)
        mean = float(np.mean(weights))
        if mean <= 0.0 or not np.isfinite(mean):
            return np.ones_like(residual, dtype=float), "arch_nonfinite"
        return weights / mean, "arch_garch11"
    except Exception as exc:  # noqa: BLE001 - optional volatility stage fallback.
        return np.ones_like(residual, dtype=float), f"arch_failed:{type(exc).__name__}"


def _glitch_filtered_residual(residual: np.ndarray) -> np.ndarray:
    out = np.asarray(residual, dtype=float).copy()
    mad = float(np.median(np.abs(out - np.median(out))))
    if mad > 0.0 and np.isfinite(mad):
        out[np.abs(out) > 50.0 * mad] = 0.0
    return out


def _rescale_tvp_outputs(
    betas_rr: np.ndarray,
    betas_2srr: np.ndarray,
    yhat_rr: np.ndarray,
    yhat_2srr: np.ndarray,
    *,
    scaling_factor: np.ndarray,
    y_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for m in range(betas_rr.shape[0]):
        betas_rr[m, 1:, :] *= scaling_factor[m, :].reshape(-1, 1)
        betas_2srr[m, 1:, :] *= scaling_factor[m, :].reshape(-1, 1)
        betas_rr[m, 0, :] *= y_sd[m]
        betas_2srr[m, 0, :] *= y_sd[m]
        yhat_rr[:, m] *= y_sd[m]
        yhat_2srr[:, m] *= y_sd[m]
    return betas_rr, betas_2srr, yhat_rr, yhat_2srr


def _full_coef_path_frame(
    betas: np.ndarray,
    *,
    index: pd.Index,
    target_names: tuple[str, ...],
    feature_names: list[str],
) -> pd.DataFrame:
    columns: list[tuple[str, str]] = []
    arrays: list[np.ndarray] = []
    coef_names = ["__intercept__", *feature_names]
    for target_idx, target in enumerate(target_names):
        for coef_idx, coef_name in enumerate(coef_names):
            columns.append((target, coef_name))
            arrays.append(betas[target_idx, coef_idx, :])
    return pd.DataFrame(
        np.column_stack(arrays),
        index=index,
        columns=pd.MultiIndex.from_tuples(columns, names=["target", "coefficient"]),
    )


def _as_positive_weights(values: Any, length: int) -> np.ndarray:
    if np.isscalar(values):
        raw = np.repeat(float(values), length)
    else:
        raw = np.asarray(values, dtype=float).reshape(-1)
        if raw.size == 1:
            raw = np.repeat(float(raw[0]), length)
    if raw.size != length:
        raise ValueError(f"weights must have length {length}, got {raw.size}")
    return np.where(np.isfinite(raw) & (raw > 0.0), raw, 1.0)


def _safe_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


def _safe_fold_se(pmse: np.ndarray, min_pos: int) -> float:
    values = pmse[:, min_pos]
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _one_se_index(score: np.ndarray, min_pos: int, se: float, multiplier: float) -> int:
    pos = int(min_pos)
    limit = float(score[min_pos] + multiplier * se)
    while pos + 1 < len(score):
        pos += 1
        if score[pos] > limit:
            break
    return pos


__all__ = [
    "TVPRidgeRegressor",
    "tvp_ridge",
]
