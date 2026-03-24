#' @title Bayesian VAR with Minnesota prior
#'
#' @description
#' Implements a Bayesian linear regression with a Minnesota-style Normal prior.
#' Applied to the Z_train feature matrix passed by FeatureBuilder, this serves
#' as a Bayesian shrinkage benchmark comparable to BVAR usage in macro
#' forecasting horse races.
#'
#' The Minnesota prior places a Normal prior on regression coefficients with
#' diagonal precision:
#'
#' \deqn{\beta \sim \mathcal{N}(0, (\lambda \cdot D)^{-1})}
#'
#' where \eqn{D} is diagonal with \eqn{D_{jj} = 1 / \mathrm{Var}(z_j)} for
#' column-standardized shrinkage (each predictor shrunk proportionally to its
#' variance).  The intercept receives a flat prior (\eqn{D_{11} \approx 0}).
#'
#' The posterior mean (MAP estimate) is analytical:
#'
#' \deqn{\hat{\beta} = (Z^\top Z + \lambda D)^{-1} Z^\top y}
#'
#' The global shrinkage hyperparameter \eqn{\lambda} can be fixed or tuned
#' by minimising leave-one-out cross-validation (LOO-CV) error via the
#' fast hat-matrix formula.
#'
#' @name bvar
NULL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#' Compute LOO-CV error for a given lambda (fast via hat matrix diagonal)
#'
#' @param Z   (T, p) design matrix (with intercept column if desired)
#' @param y   (T,) target vector
#' @param D   (p, p) diagonal prior precision matrix
#' @return Negative mean LOO squared error (maximise to tune lambda)
.bvar_loo <- function(Z, y, D) {
  A    <- crossprod(Z) + D              # (Z'Z + lambda*D)
  beta <- solve(A, crossprod(Z, y))     # MAP estimate
  e    <- y - as.numeric(Z %*% beta)    # in-sample residuals

  # Hat-matrix diagonal: h_ii = z_i' A^{-1} z_i
  A_inv <- solve(A)
  h_diag <- rowSums((Z %*% A_inv) * Z)  # (T,)
  denom <- pmax(1 - h_diag, 1e-10)      # guard against leverage = 1
  loo_mse <- mean((e / denom)^2)

  -loo_mse  # return negative because .bvar_tune maximises this
}


#' Tune lambda over a log-spaced grid by LOO-CV
#'
#' @param Z   (T, p) design matrix
#' @param y   (T,) target
#' @param D0  diagonal of the unscaled prior precision (length p)
#' @param n_grid  number of lambda candidates
#' @return Optimal lambda scalar
.bvar_tune <- function(Z, y, D0, n_grid = 20L) {
  lambdas  <- exp(seq(log(1e-3), log(1e3), length.out = n_grid))
  p        <- length(D0)
  scores   <- vapply(lambdas, function(lam) {
    D <- diag(lam * D0, nrow = p)
    .bvar_loo(Z, y, D)
  }, numeric(1L))
  lambdas[which.max(scores)]
}


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

#' Fit a Bayesian linear regression (Minnesota prior) and return forecast
#'
#' @param Z_train (T_train, N) feature matrix from FeatureBuilder
#' @param y_train (T_train,)  target vector (h-step shifted by caller)
#' @param Z_test  (1, N)      test feature row
#' @param lambda  numeric or NULL. Prior precision scale (global shrinkage).
#'   NULL (default) tunes lambda by LOO-CV.
#' @param intercept logical. Include intercept in model (default TRUE).
#' @param n_grid integer. Number of lambda candidates in the LOO-CV grid
#'   (default 20).  Ignored when lambda is not NULL.
#'
#' @return Named list:
#'   \describe{
#'     \item{y_hat}{Point forecast scalar (posterior mean prediction).}
#'     \item{hp}{List with element \code{lambda} (selected or fixed).}
#'   }
fit_bvar <- function(Z_train, y_train, Z_test,
                     lambda    = NULL,
                     intercept = TRUE,
                     n_grid    = 20L) {
  Z_train <- as.matrix(Z_train)
  Z_test  <- as.matrix(Z_test)
  y_train <- as.numeric(y_train)
  T_train <- nrow(Z_train)
  N       <- ncol(Z_train)

  if (T_train < N + 2L) {
    warning("BVAR: T_train (", T_train, ") < N+2 (", N + 2L, "). ",
            "Falling back to OLS with large shrinkage.")
    lambda <- lambda %||% 1e2
  }

  # Prepend intercept column
  if (intercept) {
    Z_train <- cbind(1, Z_train)
    Z_test  <- cbind(1, Z_test)
  }
  p <- ncol(Z_train)

  # Build unscaled precision diagonal D0:
  #   intercept: near-zero (flat prior)
  #   predictors: 1 / Var(z_j) — shrinks high-variance predictors less
  D0 <- numeric(p)
  if (intercept) {
    D0[1] <- 1e-8  # intercept: flat
    for (j in seq(2L, p)) {
      vj    <- var(Z_train[, j])
      D0[j] <- if (is.na(vj) || vj < 1e-12) 1 else 1 / vj
    }
  } else {
    for (j in seq_len(p)) {
      vj    <- var(Z_train[, j])
      D0[j] <- if (is.na(vj) || vj < 1e-12) 1 else 1 / vj
    }
  }

  # Tune or fix lambda
  if (is.null(lambda)) {
    lambda <- .bvar_tune(Z_train, y_train, D0, n_grid = as.integer(n_grid))
  }

  # Construct precision matrix and compute MAP estimate
  D    <- diag(lambda * D0, nrow = p)
  A    <- crossprod(Z_train) + D
  beta <- solve(A, crossprod(Z_train, y_train))

  y_hat <- as.numeric(Z_test %*% beta)

  list(y_hat = y_hat, hp = list(lambda = lambda))
}
