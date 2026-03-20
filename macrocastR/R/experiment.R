#' @title macrocastR experiment runner
#'
#' @description
#' Outer pseudo-OOS loop for the R-side linear models.  Mirrors the Python
#' `ForecastExperiment` class in structure and output format.
#'
#' For each evaluation date t* in the OOS range:
#'   1. Construct the training window.
#'   2. Build features via `build_features()` (fit on training window only).
#'   3. Align the h-step-ahead target.
#'   4. Fit each model and record the forecast.
#'   5. Accumulate records and write to parquet.
#'
#' Output parquet schema is identical to the Python `ResultSet` parquet so that
#' the Layer 3 Python evaluation layer can merge results transparently.
#'
#' @name experiment
NULL


#' Convert a single forecast result to a one-row data.table
#'
#' @param experiment_id Character.  UUID of the experiment run.
#' @param model_id      Character.  Human-readable model name.
#' @param nonlinearity  Character.  Nonlinearity component value (e.g. "linear").
#' @param regularization Character.  Regularization component value.
#' @param cv_scheme     Character.  CV scheme representation.
#' @param loss_function Character.  Loss function value (e.g. "l2").
#' @param window        Character.  "expanding" or "rolling".
#' @param horizon       Integer.
#' @param train_end     POSIXct or Date.  Last date in training window.
#' @param forecast_date POSIXct or Date.  Date being forecast.
#' @param y_hat         Numeric.  Point forecast.
#' @param y_true        Numeric.  Realised value.
#' @param n_train       Integer.  Training observations.
#' @param n_factors     Integer or NA.  PCA factors used.
#' @param n_lags        Integer.  AR lags used.
#' @param hp            Named list.  Selected hyperparameters.
#'
#' @return data.table with one row.
#'
#' @export
forecast_record_to_df <- function(experiment_id, model_id,
                                  nonlinearity, regularization,
                                  cv_scheme, loss_function,
                                  window, horizon,
                                  train_end, forecast_date,
                                  y_hat, y_true,
                                  n_train, n_factors, n_lags,
                                  hp = list()) {
  # Flatten hp list to scalar columns with prefix "hp_"
  hp_cols <- if (length(hp) > 0) {
    as.data.frame(
      lapply(hp, function(x) if (length(x) == 1) x else paste(x, collapse = ",")),
      check.names = FALSE
    )
  } else {
    data.frame()
  }
  names(hp_cols) <- paste0("hp_", names(hp_cols))

  base <- data.frame(
    experiment_id  = experiment_id,
    model_id       = model_id,
    nonlinearity   = nonlinearity,
    regularization = regularization,
    cv_scheme      = cv_scheme,
    loss_function  = loss_function,
    window         = window,
    horizon        = as.integer(horizon),
    train_end      = as.character(train_end),
    forecast_date  = as.character(forecast_date),
    y_hat          = as.numeric(y_hat),
    y_true         = as.numeric(y_true),
    n_train        = as.integer(n_train),
    n_factors      = if (is.na(n_factors)) NA_integer_ else as.integer(n_factors),
    n_lags         = as.integer(n_lags),
    stringsAsFactors = FALSE
  )
  if (ncol(hp_cols) > 0) {
    base <- cbind(base, hp_cols)
  }
  data.table::as.data.table(base)
}


#' Run the macrocastR experiment (pseudo-OOS loop for linear models)
#'
#' @param panel       data.frame or matrix (T x N).  Stationary-transformed
#'   predictor panel.  Row names or a separate `dates` vector must identify
#'   the time index.
#' @param target      Numeric vector (T).  Target series (same length as panel).
#' @param dates       Date or POSIXct vector (T).  Time index for panel rows.
#' @param horizons    Integer vector.  Forecast horizons.
#' @param models      Character vector.  Names of models to run.  Valid values:
#'   "ar", "ardi", "ridge", "lasso", "adaptive_lasso", "group_lasso",
#'   "elastic_net", "tvp_ridge", "booging".
#' @param n_factors   Integer.  PCA factors for ARDI-type feature construction.
#' @param n_lags      Integer.  AR lags.
#' @param use_marx    Logical.  Apply MARX transformation on top of base features.
#' @param window      Character.  "expanding" (default) or "rolling".
#' @param rolling_size Integer or NULL.  Training window size for rolling window.
#' @param oos_start   Date or character.  Start of OOS evaluation.  Default: 80th
#'   percentile of sample.
#' @param oos_end     Date or character.  End of OOS evaluation.
#' @param cv_folds    Integer.  Folds for glmnet/grpreg cross-validation.
#' @param group_vec   Integer or character vector.  Group membership for Group
#'   LASSO.  NULL for auto-inference from column names.
#' @param experiment_id Character or NULL.  UUID.  Auto-generated if NULL.
#' @param output_path  Character or NULL.  Path for the output parquet file.
#'   If NULL, results are returned as a data.table but not written to disk.
#'
#' @return data.table of forecast records (same schema as Python ResultSet parquet).
#'
#' @export
run_experiment <- function(panel, target, dates,
                           horizons = 1L,
                           models   = c("ar", "ardi", "ridge", "lasso"),
                           n_factors = 8L, n_lags = 4L,
                           use_marx  = FALSE,
                           window    = "expanding",
                           rolling_size = NULL,
                           oos_start = NULL, oos_end = NULL,
                           cv_folds  = 5L,
                           group_vec = NULL,
                           experiment_id = NULL,
                           output_path   = NULL) {
  # --- Argument validation --------------------------------------------------
  if (window == "rolling" && is.null(rolling_size)) {
    stop("rolling_size must be provided when window = 'rolling'.")
  }
  if (is.null(experiment_id)) {
    experiment_id <- paste0(sample(c(letters, 0:9), 8, replace = TRUE), collapse = "")
  }
  panel  <- as.matrix(panel)
  T_full <- nrow(panel)
  dates  <- as.Date(dates)

  # OOS range
  if (is.null(oos_start)) {
    oos_start <- dates[ceiling(0.8 * T_full)]
  } else {
    oos_start <- as.Date(oos_start)
  }
  max_h <- max(horizons)
  if (is.null(oos_end)) {
    oos_end <- dates[T_full - max_h]
  } else {
    oos_end <- as.Date(oos_end)
  }

  oos_idx <- which(dates >= oos_start & dates <= oos_end)
  if (length(oos_idx) == 0) {
    stop("No OOS dates found between oos_start and oos_end.")
  }

  all_records <- vector("list", length(models) * length(horizons) * length(oos_idx))
  rec_idx <- 1L

  for (h in horizons) {
    for (t_star_pos in oos_idx) {
      # Training window end: h periods before t*
      train_end_pos <- t_star_pos - h
      if (train_end_pos < 2L) next

      # Training window start
      if (window == "expanding") {
        train_start_pos <- 1L
      } else {
        train_start_pos <- max(1L, train_end_pos - rolling_size + 1L)
      }

      idx_tr    <- seq(train_start_pos, train_end_pos)
      X_train   <- panel[idx_tr, , drop = FALSE]
      y_tr_full <- target[idx_tr]
      T_tr      <- length(idx_tr)

      # h-step direct target alignment: y_{t+h} for t = 1 .. T_tr - h
      if (T_tr <= h) next
      y_train_aligned <- target[seq(train_start_pos + h, train_end_pos + h)]
      X_train_aligned <- X_train[seq_len(T_tr - h), , drop = FALSE]

      # Fit features on training window (PCA)
      feat_out  <- build_features(
        X_panel    = X_train_aligned,
        y          = y_tr_full[seq_len(T_tr - h)],
        n_factors  = n_factors,
        n_lags     = n_lags,
        use_factors = TRUE,    # always extract PCA; AR-only handled per model
        return_pca  = TRUE
      )
      Z_train_f <- feat_out$Z
      pca_fit   <- feat_out$pca_fit

      # AR-only features (no factors)
      Z_train_ar <- build_features(
        X_panel    = X_train_aligned,
        y          = y_tr_full[seq_len(T_tr - h)],
        n_factors  = n_lags,   # irrelevant
        n_lags     = n_lags,
        use_factors = FALSE
      )

      # Align h-step target with feature rows.
      # X_train_aligned has T_tr - h rows; after build_features drops first
      # n_lags rows, Z has T_tr - h - n_lags rows.  y_fit must match that.
      n_drop <- n_lags
      n_rows_z <- T_tr - h - n_drop
      if (nrow(Z_train_f) == 0 || n_rows_z <= 0L) next
      y_fit_f  <- y_train_aligned[seq(n_drop + 1, T_tr - h)]
      y_fit_ar <- y_train_aligned[seq(n_drop + 1, T_tr - h)]

      # Test features: single row at train_end
      X_test_row <- panel[train_end_pos, , drop = FALSE]
      y_test_lags <- target[seq(max(1, train_end_pos - n_lags + 1), train_end_pos)]

      Z_test_f <- build_features(
        X_panel    = X_test_row,
        y          = y_test_lags,
        n_factors  = n_factors,
        n_lags     = n_lags,
        use_factors = TRUE,
        pca_fit    = pca_fit
      )
      Z_test_ar <- build_features(
        X_panel    = X_test_row,
        y          = y_test_lags,
        n_factors  = n_lags,
        n_lags     = n_lags,
        use_factors = FALSE
      )

      # MARX transformation if requested
      if (use_marx) {
        Z_train_f  <- marx_transform(Z_train_f, X_train_aligned,
                                      y_tr_full[seq_len(T_tr - h)], n_lags)
        Z_test_f   <- marx_transform(Z_test_f,  X_test_row,
                                      y_test_lags, n_lags)
      }

      y_true        <- target[t_star_pos]
      train_end_dt  <- dates[train_end_pos]
      forecast_dt   <- dates[t_star_pos]

      # --- Run each model ---------------------------------------------------
      for (mod_name in models) {
        result <- tryCatch(
          .run_one_model(
            mod_name       = mod_name,
            Z_train_f      = Z_train_f,
            Z_train_ar     = Z_train_ar,
            y_fit_f        = y_fit_f,
            y_fit_ar       = y_fit_ar,
            Z_test_f       = Z_test_f,
            Z_test_ar      = Z_test_ar,
            y_tr_full      = y_tr_full,
            y_test_lags    = y_test_lags,
            h              = h,
            cv_folds       = cv_folds,
            group_vec      = group_vec
          ),
          error = function(e) {
            message("Model '", mod_name, "' failed (h=", h, ", date=",
                    forecast_dt, "): ", conditionMessage(e))
            NULL
          }
        )
        if (is.null(result)) next

        rec <- forecast_record_to_df(
          experiment_id  = experiment_id,
          model_id       = .model_id(mod_name),
          nonlinearity   = "linear",
          regularization = .model_reg(mod_name),
          cv_scheme      = .model_cv(mod_name),
          loss_function  = "l2",
          window         = window,
          horizon        = h,
          train_end      = train_end_dt,
          forecast_date  = forecast_dt,
          y_hat          = result$y_hat,
          y_true         = y_true,
          n_train        = nrow(Z_train_f),
          n_factors      = if (mod_name %in% c("ardi", "ridge", "lasso",
                                                "adaptive_lasso", "group_lasso",
                                                "elastic_net", "tvp_ridge",
                                                "booging")) n_factors else NA_integer_,
          n_lags         = n_lags,
          hp             = result$hp
        )
        all_records[[rec_idx]] <- rec
        rec_idx <- rec_idx + 1L
      }
    }
  }

  # Collect results
  valid_records <- Filter(Negate(is.null), all_records)
  if (length(valid_records) == 0) {
    warning("No forecast records produced.")
    return(data.table::data.table())
  }
  result_dt <- data.table::rbindlist(valid_records, fill = TRUE)

  # Write parquet if requested
  if (!is.null(output_path)) {
    dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
    arrow::write_parquet(result_dt, output_path)
    message("Results written to ", output_path)
  }

  result_dt
}


# ---------------------------------------------------------------------------
# Internal: dispatch to individual model fitters
# ---------------------------------------------------------------------------

.run_one_model <- function(mod_name, Z_train_f, Z_train_ar, y_fit_f, y_fit_ar,
                           Z_test_f, Z_test_ar, y_tr_full, y_test_lags,
                           h, cv_folds, group_vec) {
  switch(mod_name,
    ar = fit_ar(
      y_train    = y_tr_full,
      y_test_lags = y_test_lags,
      h          = h
    ),
    ardi = fit_ardi(
      Z_train = Z_train_f,
      y_train = y_fit_f,
      Z_test  = Z_test_f
    ),
    ridge = fit_ridge(
      Z_train  = Z_train_f,
      y_train  = y_fit_f,
      Z_test   = Z_test_f,
      cv_folds = cv_folds
    ),
    lasso = fit_lasso(
      Z_train  = Z_train_f,
      y_train  = y_fit_f,
      Z_test   = Z_test_f,
      cv_folds = cv_folds
    ),
    adaptive_lasso = fit_adaptive_lasso(
      Z_train  = Z_train_f,
      y_train  = y_fit_f,
      Z_test   = Z_test_f,
      cv_folds = cv_folds
    ),
    group_lasso = fit_group_lasso(
      Z_train  = Z_train_f,
      y_train  = y_fit_f,
      Z_test   = Z_test_f,
      group    = group_vec,
      cv_folds = cv_folds
    ),
    elastic_net = fit_elastic_net(
      Z_train  = Z_train_f,
      y_train  = y_fit_f,
      Z_test   = Z_test_f,
      cv_folds = cv_folds
    ),
    tvp_ridge = fit_tvp_ridge(
      Z_train  = Z_train_f,
      y_train  = y_fit_f,
      Z_test   = Z_test_f,
      cv_folds = cv_folds
    ),
    booging = fit_booging(
      Z_train  = Z_train_f,
      y_train  = y_fit_f,
      Z_test   = Z_test_f
    ),
    stop("Unknown model: '", mod_name, "'. Valid: ar, ardi, ridge, lasso, ",
         "adaptive_lasso, group_lasso, elastic_net, tvp_ridge, booging.")
  )
}


# ---------------------------------------------------------------------------
# Model metadata helpers
# ---------------------------------------------------------------------------

.model_id <- function(mod_name) {
  ids <- c(
    ar             = "linear__none__bic__l2",
    ardi           = "linear__factors__bic__l2",
    ridge          = "linear__ridge__kfold5__l2",
    lasso          = "linear__lasso__kfold5__l2",
    adaptive_lasso = "linear__adaptive_lasso__kfold5__l2",
    group_lasso    = "linear__group_lasso__kfold5__l2",
    elastic_net    = "linear__elastic_net__kfold5__l2",
    tvp_ridge      = "linear__tvp_ridge__kfold5__l2",
    booging        = "linear__booging__none__l2"
  )
  ids[[mod_name]]
}

.model_reg <- function(mod_name) {
  regs <- c(
    ar             = "none",
    ardi           = "factors",
    ridge          = "ridge",
    lasso          = "lasso",
    adaptive_lasso = "adaptive_lasso",
    group_lasso    = "group_lasso",
    elastic_net    = "elastic_net",
    tvp_ridge      = "tvp_ridge",
    booging        = "booging"
  )
  regs[[mod_name]]
}

.model_cv <- function(mod_name) {
  cvs <- c(
    ar             = "_BICScheme()",
    ardi           = "_BICScheme()",
    ridge          = "_KFoldCV(k=5)",
    lasso          = "_KFoldCV(k=5)",
    adaptive_lasso = "_KFoldCV(k=5)",
    group_lasso    = "_KFoldCV(k=5)",
    elastic_net    = "_KFoldCV(k=5)",
    tvp_ridge      = "_KFoldCV(k=5)",
    booging        = "_POOSScheme()"
  )
  cvs[[mod_name]]
}
