#!/usr/bin/env Rscript
# bridge.R — thin CLI entry point for the Python-R model bridge.
#
# Called by Python's RModelEstimator via subprocess:
#   Rscript bridge.R <model_name> <tmpdir>
#
# Input files in tmpdir:
#   Z_train.feather       (T_train, N_feat) feature matrix
#   y_train.feather       (T_train, 1)      target vector
#   Z_test.feather        (1, N_feat)       test feature row
#   config.json           model-specific params (cv_folds, nlambda, etc.)
#
#   For AR model only:
#     y_train_full.feather  (T_full, 1)  un-shifted target for BIC lag selection
#     y_test_lags.feather   (p, 1)       lag values for the AR test forecast
#     (config.json also contains h and max_lag)
#
# Output:
#   output.json in tmpdir:
#     {"y_hat": 1.234, "hp": {"lambda": 0.01}}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript bridge.R <model_name> <tmpdir>", call. = FALSE)
}

model_name <- args[1]
tmpdir     <- args[2]

suppressPackageStartupMessages({
  library(arrow)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Locate and source models.R
# ---------------------------------------------------------------------------

# Resolve this script's own directory via commandArgs (works with Rscript
# /abs/path/bridge.R as well as relative invocations).
.bridge_self <- local({
  args_all <- commandArgs(trailingOnly = FALSE)
  file_arg  <- grep("^--file=", args_all, value = TRUE)
  if (length(file_arg) == 1L) {
    dirname(normalizePath(sub("^--file=", "", file_arg), mustWork = FALSE))
  } else {
    NULL
  }
})

# When called from Python via subprocess, the working directory is the
# project root.  Look for macrocastR as an installed package first, then
# fall back to sourcing models.R directly from the repo tree.
if (requireNamespace("macrocastR", quietly = TRUE)) {
  library(macrocastR)
} else {
  # Resolve path relative to this script's location
  script_dir  <- if (!is.null(.bridge_self)) .bridge_self else tryCatch(
    dirname(normalizePath(sys.frame(1)$ofile, mustWork = FALSE)),
    error = function(e) getwd()
  )
  models_path <- file.path(script_dir, "..", "R", "models.R")
  if (!file.exists(models_path)) {
    # Try relative to working directory (project root)
    models_path <- file.path("macrocastR", "R", "models.R")
  }
  if (!file.exists(models_path)) {
    stop("Cannot locate macrocastR/R/models.R. Install macrocastR or run ",
         "from the project root.", call. = FALSE)
  }
  source(models_path)
}

# Source any model extension files not yet in the installed package.
# bvar.R is sourced here so it works whether macrocastR is installed or not.
.source_r_ext <- function(filename) {
  candidates <- c(
    if (!is.null(.bridge_self)) file.path(.bridge_self, "..", "R", filename),
    file.path("macrocastR", "R", filename)
  )
  for (p in candidates) {
    if (file.exists(p)) { source(normalizePath(p, mustWork = FALSE)); return(invisible(TRUE)) }
  }
  invisible(FALSE)
}

if (!exists("fit_bvar", mode = "function")) .source_r_ext("bvar.R")

# ---------------------------------------------------------------------------
# Read inputs
# ---------------------------------------------------------------------------

Z_train <- as.matrix(read_feather(file.path(tmpdir, "Z_train.feather")))
y_train <- as.matrix(read_feather(file.path(tmpdir, "y_train.feather")))[, 1]
Z_test  <- as.matrix(read_feather(file.path(tmpdir, "Z_test.feather")))
config  <- fromJSON(file.path(tmpdir, "config.json"))

# ---------------------------------------------------------------------------
# Null-coalescing helper (R < 4.4 may not have %||%)
# ---------------------------------------------------------------------------

`%||%` <- function(x, y) if (is.null(x)) y else x

# ---------------------------------------------------------------------------
# Dispatch to model fitter
# ---------------------------------------------------------------------------

result <- switch(
  model_name,

  ar = {
    # AR model has a different interface: receives the un-shifted target series
    # and handles the h-step shift internally.
    y_full_path <- file.path(tmpdir, "y_train_full.feather")
    y_lags_path <- file.path(tmpdir, "y_test_lags.feather")

    if (!file.exists(y_full_path)) {
      stop("AR model requires y_train_full.feather in tmpdir.", call. = FALSE)
    }
    y_full     <- as.matrix(read_feather(y_full_path))[, 1]
    y_test_lags <- if (file.exists(y_lags_path)) {
      as.matrix(read_feather(y_lags_path))[, 1]
    } else {
      tail(y_full, config$max_lag %||% 12L)
    }

    fit_ar(
      Z_train     = NULL,
      y_train     = y_full,
      y_test_lags = y_test_lags,
      h           = as.integer(config$h %||% 1L),
      max_lag     = as.integer(config$max_lag %||% 12L)
    )
  },

  ardi = fit_ardi(
    Z_train   = Z_train,
    y_train   = y_train,
    Z_test    = Z_test,
    intercept = isTRUE(config$intercept %||% TRUE)
  ),

  ridge = fit_ridge(
    Z_train  = Z_train,
    y_train  = y_train,
    Z_test   = Z_test,
    cv_folds = as.integer(config$cv_folds %||% 5L),
    nlambda  = as.integer(config$nlambda  %||% 50L)
  ),

  lasso = fit_lasso(
    Z_train  = Z_train,
    y_train  = y_train,
    Z_test   = Z_test,
    cv_folds = as.integer(config$cv_folds %||% 5L),
    nlambda  = as.integer(config$nlambda  %||% 50L)
  ),

  adaptive_lasso = fit_adaptive_lasso(
    Z_train  = Z_train,
    y_train  = y_train,
    Z_test   = Z_test,
    cv_folds = as.integer(config$cv_folds %||% 5L),
    nlambda  = as.integer(config$nlambda  %||% 50L),
    gamma    = as.numeric(config$gamma    %||% 1)
  ),

  group_lasso = {
    # groups may be NULL (infer from column names) or a vector passed via config
    groups <- config$groups  # NULL or integer vector from JSON
    fit_group_lasso(
      Z_train  = Z_train,
      y_train  = y_train,
      Z_test   = Z_test,
      group    = groups,
      cv_folds = as.integer(config$cv_folds %||% 5L)
    )
  },

  elastic_net = fit_elastic_net(
    Z_train  = Z_train,
    y_train  = y_train,
    Z_test   = Z_test,
    cv_folds = as.integer(config$cv_folds %||% 5L),
    nlambda  = as.integer(config$nlambda  %||% 50L),
    alpha    = as.numeric(config$alpha    %||% 0.5)
  ),

  tvp_ridge = fit_tvp_ridge(
    Z_train     = Z_train,
    y_train     = y_train,
    Z_test      = Z_test,
    poly_degree = as.integer(config$n_poly   %||% 3L),
    cv_folds    = as.integer(config$cv_folds %||% 5L)
  ),

  booging = fit_booging(
    Z_train        = Z_train,
    y_train        = y_train,
    Z_test         = Z_test,
    n_boot         = as.integer(config$n_boot %||% 200L),
    prune_quantile = as.numeric(config$prune_quantile %||% 0.5)
  ),

  bvar = fit_bvar(
    Z_train   = Z_train,
    y_train   = y_train,
    Z_test    = Z_test,
    lambda    = config$lambda,          # NULL → LOO-CV tuning
    intercept = isTRUE(config$intercept %||% TRUE),
    n_grid    = as.integer(config$n_grid %||% 20L)
  ),

  stop(paste("Unknown model:", model_name), call. = FALSE)
)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

# Ensure y_hat is a plain scalar
y_hat_scalar <- as.numeric(result$y_hat)
if (length(y_hat_scalar) != 1L) {
  y_hat_scalar <- mean(y_hat_scalar, na.rm = TRUE)
}

# Flatten hp: keep only length-1 numeric/character values for JSON
hp_flat <- lapply(result$hp, function(v) {
  if (length(v) == 1 && (is.numeric(v) || is.character(v) || is.integer(v))) {
    v
  } else {
    NULL
  }
})
hp_flat <- Filter(Negate(is.null), hp_flat)

output <- list(y_hat = y_hat_scalar, hp = hp_flat)
write_json(output, file.path(tmpdir, "output.json"), auto_unbox = TRUE)
