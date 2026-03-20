#' @title Feature construction for the macrocastR pipeline
#'
#' @description
#' Constructs the predictor matrix Z_t used by all linear models.  Two modes:
#'
#' * **Factors mode** (ARDI): Z_t = [PCA factors f_1..f_{p_f}, AR lags y_{t-1}..y_{t-p_y}]
#' * **AR-only mode**: Z_t = [AR lags y_{t-1}..y_{t-p_y}]
#'
#' PCA is fitted on the training window only (no look-ahead).  The number of
#' factors p_f and lags p_y are tuning parameters exposed to the CV loop.
#'
#' Also implements the MARX transformation (Coulombe 2020 "marx" repo):
#' cross-products of predictors and their lags, expanding the feature space
#' for nonlinear-in-mean patterns while remaining linear in parameters.
#'
#' @name features
NULL


#' Build the predictor matrix Z for a given training window
#'
#' @param X_panel Numeric matrix (T x N).  Stationary-transformed predictor
#'   panel for the training window.
#' @param y Numeric vector (T).  Target series for the training window.  Used
#'   for AR lag columns only; PCA is NOT fitted on y.
#' @param n_factors Integer.  Number of PCA factors.  Ignored when
#'   `use_factors = FALSE`.
#' @param n_lags Integer.  Number of AR lags to append.
#' @param use_factors Logical.  If TRUE, prepend PCA factors (ARDI mode).
#' @param standardize_X Logical.  Standardise X before PCA.  Default TRUE.
#' @param pca_fit List or NULL.  Pre-fitted PCA object (from a prior call with
#'   `return_pca = TRUE`).  If provided, skips fitting and applies the stored
#'   rotation.  Used to transform the test window without look-ahead.
#' @param return_pca Logical.  If TRUE, return the PCA rotation alongside Z.
#'
#' @return If `return_pca = FALSE` (default): numeric matrix (T - n_lags, n_features).
#'   If `return_pca = TRUE`: list with elements `Z` (the feature matrix) and
#'   `pca_fit` (list with `center`, `scale`, `rotation`, `n_factors`).
#'
#' @export
build_features <- function(X_panel, y, n_factors = 8L, n_lags = 4L,
                           use_factors = TRUE, standardize_X = TRUE,
                           pca_fit = NULL, return_pca = FALSE) {
  T_obs <- nrow(X_panel)
  N     <- ncol(X_panel)
  p     <- as.integer(n_lags)

  # In transform mode (pca_fit provided) a single test row is valid
  if (is.null(pca_fit) && T_obs <= p) {
    stop("Training window (", T_obs, " rows) is not larger than n_lags (", p, ").")
  }

  # --- AR lag matrix --------------------------------------------------
  # Training mode: rows p+1..T each get p lags from y.
  # Transform mode with T_obs == 1: y contains the last p values; build
  # one lag row directly (lag 1 = y[p], lag 2 = y[p-1], ..., lag p = y[1]).
  if (T_obs == 1L && !is.null(pca_fit)) {
    y_p     <- tail(y, p)
    ar_lags <- matrix(y_p[seq(p, 1L, by = -1L)], nrow = 1L, ncol = p)
  } else {
    ar_lags <- matrix(NA_real_, nrow = T_obs - p, ncol = p)
    for (lag in seq_len(p)) {
      ar_lags[, lag] <- y[seq(p - lag + 1, T_obs - lag)]
    }
  }
  colnames(ar_lags) <- paste0("y_lag", seq_len(p))

  # --- PCA factors ----------------------------------------------------
  if (use_factors) {
    if (is.null(pca_fit)) {
      # Fit PCA on training panel
      n_factors <- min(as.integer(n_factors), N, T_obs - p - 1L)
      if (standardize_X) {
        col_center <- colMeans(X_panel, na.rm = TRUE)
        col_scale  <- apply(X_panel, 2, sd, na.rm = TRUE)
        col_scale[col_scale == 0] <- 1  # guard zero-variance columns
        X_std <- scale(X_panel, center = col_center, scale = col_scale)
      } else {
        col_center <- rep(0, N)
        col_scale  <- rep(1, N)
        X_std <- X_panel
      }
      # Economy SVD: only need first n_factors right singular vectors
      sv         <- svd(X_std, nu = 0, nv = n_factors)
      rotation   <- sv$v          # (N, n_factors)
      pca_fit    <- list(
        center   = col_center,
        scale    = col_scale,
        rotation = rotation,
        n_factors = n_factors
      )
    } else {
      # Apply pre-fitted rotation (test window)
      n_factors <- pca_fit$n_factors
      X_std <- scale(X_panel,
                     center = pca_fit$center,
                     scale  = pca_fit$scale)
    }

    factors <- X_std %*% pca_fit$rotation   # (T_obs, n_factors)
    # In training mode drop the first p rows (no lags available for them);
    # in single-row transform mode keep the one row as-is.
    if (T_obs > 1L) {
      factors <- factors[seq(p + 1, T_obs), , drop = FALSE]
    }
    colnames(factors) <- paste0("f", seq_len(ncol(factors)))

    Z <- cbind(factors, ar_lags)
  } else {
    Z        <- ar_lags
    pca_fit  <- NULL
  }

  if (return_pca) {
    return(list(Z = Z, pca_fit = pca_fit))
  }
  Z
}


#' MARX feature transformation
#'
#' Implements the Mixed-frequency Autoregressive eXogenous (MARX) transformation
#' from Coulombe (2020, philgoucou/marx).  For each predictor x_j, appends
#' cross-products x_j * y_{t-1} ..  x_j * y_{t-p_y}, expanding the feature space
#' to capture interaction effects between predictors and the target's own history.
#'
#' This keeps the model linear in parameters while introducing
#' nonlinear-in-mean dynamics.  The resulting matrix can be passed to any
#' linear estimator (Ridge, LASSO, etc.).
#'
#' @param Z Numeric matrix (T_obs - n_lags, n_features).  Base feature matrix
#'   from `build_features()`.
#' @param X_panel Numeric matrix (T_obs, N).  Original predictor panel (needed
#'   to extract predictor values for cross-products).
#' @param y Numeric vector (T_obs).  Target series for the training window.
#' @param n_lags Integer.  AR lag order used in `build_features()`.
#' @param p_marx Integer.  Number of target lags to cross with each predictor.
#'   Default 1 (only y_{t-1}); increase for richer interactions.
#'
#' @return Numeric matrix with `ncol(Z) + N * p_marx` columns: original Z
#'   columns followed by cross-product columns.
#'
#' @references
#'   Coulombe, P.G. (2020). "The Macroeconomy as a Random Forest."
#'   philgoucou/marx GitHub repository.
#'
#' @export
marx_transform <- function(Z, X_panel, y, n_lags, p_marx = 1L) {
  T_obs  <- nrow(X_panel)
  N      <- ncol(X_panel)
  T_z    <- nrow(Z)   # T_obs - n_lags
  p      <- as.integer(n_lags)
  p_m    <- as.integer(p_marx)

  # X rows aligned with Z (rows p+1 .. T_obs, 1-indexed)
  X_aligned <- X_panel[seq(p + 1, T_obs), , drop = FALSE]

  # Build cross-product columns: x_j * y_{t-k} for j=1..N, k=1..p_m
  cross_cols <- vector("list", N * p_m)
  idx <- 1L
  for (j in seq_len(N)) {
    for (k in seq_len(p_m)) {
      # y_{t-k} for rows p+1..T_obs: y[p+1-k .. T_obs-k]
      y_lag_k <- y[seq(p + 1 - k, T_obs - k)]
      cross_cols[[idx]] <- X_aligned[, j] * y_lag_k
      idx <- idx + 1L
    }
  }
  cross_mat <- do.call(cbind, cross_cols)
  colnames(cross_mat) <- paste0(
    "marx_x", rep(seq_len(N), each = p_m),
    "_lag", rep(seq_len(p_m), times = N)
  )

  cbind(Z, cross_mat)
}
