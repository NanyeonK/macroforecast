from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import pandas as pd

import macroforecast as mf


TARGET_MAP = {
    "INDPRO": "INDPRO",
    "EMP": "PAYEMS",
    "UNRATE": "UNRATE",
    "INCOME": "RPI",
    "CONS": "DPCERA3M086SBEA",
    "RETAIL": "RETAILx",
    "HOUST": "HOUST",
    "M2": "M2REAL",
    "CPI": "CPIAUCSL",
    "PPI": "WPSFD49207",  # PPI Finished Goods (standard headline PPI); PPICMM=Metals is far more volatile and mismatches the appendix scale
}


def paper_feature_steps(
    case: str,
    target: str,
    transformed_predictors: list[str],
    level_predictors: list[str],
    *,
    factor_lags: range = range(0, 13),
    x_lags: range = range(0, 13),
    level_lags: range = range(0, 1),
) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    parts = case.split("-")
    if "F" in parts:
        steps.append(
            mf.feature_engineering.pca_step(
                name="F_raw",
                columns=transformed_predictors,
                n_components=8,
                scale=True,
                include=False,
            )
        )
        steps.append(
            mf.feature_engineering.lag_step(
                name="F",
                input="F_raw",
                lags=factor_lags,
                include=True,
            )
        )
    if "X" in parts:
        steps.append(
            mf.feature_engineering.lag_step(
                name="X",
                columns=transformed_predictors,
                lags=x_lags,
                include=True,
            )
        )
    if "MARX" in parts:
        steps.append(
            mf.feature_engineering.marx_step(
                name="MARX_X",
                columns=transformed_predictors,
                max_lag=12,
                scale_lags=False,
                include=True,
            )
        )
        steps.append(
            mf.feature_engineering.marx_step(
                name="MARX_y",
                input="target_panel",
                # columns=None -> ALL columns of ``target_panel``. The target_panel
                # is built from ``fitted.targets`` (exactly the active target), so
                # this follows the pipeline's per-target re-targeting (combined
                # multi-target spec) instead of pinning the original target column.
                # For a single-target spec target_panel has one column, so this is
                # numerically identical to the prior columns=[target].
                columns=None,
                max_lag=12,
                scale_lags=False,
                include=True,
            )
        )
    if "MAF" in parts:
        steps.append(
            mf.feature_engineering.maf_step(
                name="MAF_X",
                columns=transformed_predictors,
                max_lag=12,
                n_components=2,
                scale=False,
                include=True,
            )
        )
        steps.append(
            mf.feature_engineering.maf_step(
                name="MAF_y",
                input="target_panel",
                # columns=None -> ALL columns of ``target_panel`` (the active
                # re-targeted target). Same target-agnostic rationale as MARX_y;
                # numerically identical to columns=[target] for a single target.
                columns=None,
                max_lag=12,
                n_components=2,
                scale=False,
                include=True,
            )
        )
    if "Level" in parts:
        steps.append(
            mf.feature_engineering.lag_step(
                name="Level",
                columns=level_predictors,
                lags=level_lags,
                include=True,
            )
        )
    return steps


def paper_model_params(model: str, *, random_state: int, n_estimators: int) -> dict[str, Any]:
    if model == "random_forest":
        return {
            "n_estimators": int(n_estimators),
            "max_features": 1 / 3,
            "min_samples_leaf": 5,
            "bootstrap": True,
            "random_state": int(random_state),
            "n_jobs": 1,
        }
    if model == "gradient_boosting":
        return {
            "n_estimators": int(n_estimators),
            "learning_rate": 0.1,
            "max_depth": 5,
            "max_features": 1 / 3,
            "min_samples_leaf": 5,
            "random_state": int(random_state),
        }
    if model == "adaptive_lasso":
        return {
            "gamma": 1.0,
            "initial": "ridge",
            "max_iter": 20000,
            "random_state": int(random_state),
        }
    if model == "elastic_net":
        return {"max_iter": 20000, "standardize": True}
    if model == "glmboost":
        return {
            "n_iter": 200,
            "learning_rate": 0.1,
            "center": True,
            "candidate_sampling": "random",
            "candidate_fraction": 1 / 3,
            "candidate_cap": 200,
            "candidate_min": 1,
            "candidate_rounding": "floor",
            "random_state": int(random_state),
        }
    if model == "far":
        return {"n_factors": 8, "n_lag": 12, "random_state": int(random_state)}
    return {"random_state": int(random_state)}


def paper_model_search(
    model: str,
    *,
    mode: str,
    random_state: int,
    search_iterations: int,
    ga_population: int,
    ga_generations: int,
) -> Any:
    if mode == "off":
        return None
    small = mode == "paper-small"
    if model in {"random_forest", "far", "ar"}:
        return None
    if model == "elastic_net":
        return mf.model_selection.custom_search(
            "gcls_2021_elastic_net_cv",
            _paper_elastic_net_cv,
            random_state=random_state,
            n_lambda=8 if small else 100,
            n_l1_ratio=8 if small else 100,
            metadata={
                "paper": "Goulet Coulombe, Leroux, Stevanovic, Surprenant (2021)",
                "appendix_rule": (
                    "lambda grid up to lambda_max and alpha/l1-ratio grid in "
                    "[0.01, 1], selected by random 5-fold squared-loss CV"
                ),
                "mode": mode,
            },
        )
    if model == "adaptive_lasso":
        return mf.model_selection.custom_search(
            "gcls_2021_adaptive_lasso_cv",
            _paper_adaptive_lasso_cv,
            random_state=random_state,
            n_lambda=8 if small else 100,
            n_initial_alpha=5 if small else 25,
            metadata={
                "paper": "Goulet Coulombe, Leroux, Stevanovic, Surprenant (2021)",
                "appendix_rule": (
                    "gamma=1 adaptive lasso with ridge first-step weights; "
                    "ridge penalty chosen by GA in the paper, then lasso "
                    "lambda by random 5-fold squared-loss CV"
                ),
                "implementation_note": (
                    "Deterministic ridge-alpha grid approximation to the "
                    "appendix GA because the original MATLAB GA state is not "
                    "available."
                ),
                "mode": mode,
            },
        )
    if model == "gradient_boosting":
        return mf.model_selection.bayesian_search(
            {
                "n_estimators": mf.model_selection.randint(1, 500),
                "learning_rate": mf.model_selection.uniform(1e-4, 1.0),
            },
            n_iter=max(3, int(search_iterations)),
            random_state=random_state,
        )
    if model == "glmboost":
        return mf.model_selection.genetic_search(
            {
                "n_iter": mf.model_selection.randint(1, 500),
                "learning_rate": mf.model_selection.uniform(1e-4, 1.0),
            },
            population_size=max(2, int(ga_population)),
            generations=max(1, int(ga_generations)),
            mutation_rate=0.2,
            random_state=random_state,
        )
    return None


def _paper_elastic_net_cv(
    *,
    model,
    X,
    y,
    splits,
    metric,
    fixed_params,
    search,
    rng,
    maximize,
    evaluate_candidate,
    n_lambda: int,
    n_l1_ratio: int,
) -> tuple[list[Any], dict[str, Any]]:
    lambda_max = _sklearn_lasso_alpha_max(X, y)
    lambdas = _lambda_grid(lambda_max, int(n_lambda))
    l1_ratios = np.linspace(0.01, 1.0, int(n_l1_ratio))
    rows = []
    trial = 0
    for l1_ratio in l1_ratios:
        for alpha in lambdas:
            rows.append(
                evaluate_candidate(
                    model,
                    X,
                    y,
                    splits,
                    metric,
                    fixed_params,
                    {"alpha": float(alpha), "l1_ratio": float(l1_ratio)},
                    trial,
                )
            )
            trial += 1
    return rows, {
        "paper_search": {
            "lambda_max": float(lambda_max),
            "n_lambda": int(len(lambdas)),
            "n_l1_ratio": int(len(l1_ratios)),
            "fold_assignment": "random_kfold from macroforecast.window",
        }
    }


def _paper_adaptive_lasso_cv(
    *,
    model,
    X,
    y,
    splits,
    metric,
    fixed_params,
    search,
    rng,
    maximize,
    evaluate_candidate,
    n_lambda: int,
    n_initial_alpha: int,
) -> tuple[list[Any], dict[str, Any]]:
    lambda_max = _sklearn_lasso_alpha_max(X, y)
    lambdas = _lambda_grid(lambda_max, int(n_lambda))
    initial_alphas = np.geomspace(1e-4, 100.0, int(n_initial_alpha))
    rows = []
    trial = 0
    for initial_alpha in initial_alphas:
        for alpha in lambdas:
            rows.append(
                evaluate_candidate(
                    model,
                    X,
                    y,
                    splits,
                    metric,
                    fixed_params,
                    {
                        "alpha": float(alpha),
                        "initial_alpha": float(initial_alpha),
                    },
                    trial,
                )
            )
            trial += 1
    return rows, {
        "paper_search": {
            "lambda_max": float(lambda_max),
            "n_lambda": int(len(lambdas)),
            "n_initial_alpha": int(len(initial_alphas)),
            "fold_assignment": "random_kfold from macroforecast.window",
            "ridge_penalty_search": "deterministic grid approximation to paper GA",
        }
    }


def _sklearn_lasso_alpha_max(X: pd.DataFrame, y: pd.Series) -> float:
    x = X.astype(float).copy().fillna(X.mean(axis=0)).fillna(0.0).to_numpy(dtype=float)
    target = pd.Series(y, index=X.index).astype(float).to_numpy(dtype=float)
    x_scale = np.std(x, axis=0, ddof=0)
    x_scale[x_scale <= 1e-12] = 1.0
    x = (x - x.mean(axis=0)) / x_scale
    target = target - target.mean()
    alpha_max = float(np.max(np.abs(x.T @ target)) / max(1, len(target)))
    return max(alpha_max, 1e-8)


def _lambda_grid(lambda_max: float, n_lambda: int) -> np.ndarray:
    n = max(1, int(n_lambda))
    high = max(float(lambda_max), 1e-8)
    low = high * 1e-4
    if n == 1:
        return np.asarray([high], dtype=float)
    return np.geomspace(low, high, n)


def _combined_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    valid = frame.dropna(subset=["prediction", "actual"])
    if len(valid) == 0:
        return {"rows": int(len(frame)), "nonmissing_pairs": 0, "rmse": math.nan, "mae": math.nan}
    err = valid["prediction"].astype(float) - valid["actual"].astype(float)
    return {
        "rows": int(len(frame)),
        "nonmissing_pairs": int(len(valid)),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "mae": float(err.abs().mean()),
    }


def _validate_target_dates(frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    if frame.empty:
        return {"invalid_rows": 0, "nan_prediction_rows": 0, "nan_actual_rows": 0}
    dates = pd.to_datetime(frame["date"])
    origins = pd.to_datetime(frame["origin"])
    expected = origins + pd.DateOffset(months=int(horizon))
    invalid = dates != expected
    return {
        "invalid_rows": int(invalid.sum()),
        "nan_prediction_rows": int(frame["prediction"].isna().sum()),
        "nan_actual_rows": int(frame["actual"].isna().sum()),
    }


def _is_empty_horizon_tail_error(error: ValueError) -> bool:
    return "window produced no test origins" in str(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-alias", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--feature-case", required=True)
    parser.add_argument("--target-policy", required=True, choices=("direct_average", "path_average"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--vintage", default="2018-01")
    parser.add_argument("--cache-root", default="/home/nanyeon99/project/macroforecast_replication_cache")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2017)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument(
        "--tuning-mode",
        choices=("off", "paper-small", "paper"),
        default="off",
        help="Use paper-style random 5-fold CV tuning; paper-small is for smoke runs.",
    )
    parser.add_argument("--cv-random-state", type=int, default=123)
    parser.add_argument("--search-iterations", type=int, default=20)
    parser.add_argument("--ga-population", type=int, default=25)
    parser.add_argument("--ga-generations", type=int, default=25)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if args.skip_existing and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if existing.get("status") == "done":
            print("SKIP_DONE", manifest_path, flush=True)
            return

    started = time.time()
    print("macroforecast_version", mf.__version__, flush=True)
    print("args", vars(args), flush=True)

    bundle = mf.data.load_fred_md(vintage=args.vintage, cache_root=args.cache_root)
    processed = mf.preprocessing.reprocess(bundle)
    target = TARGET_MAP[args.target_alias]
    raw_levels = bundle.panel.reindex(processed.panel.index)
    complete_level_columns = [column for column in raw_levels.columns if raw_levels[column].notna().all()]
    replication_panel = processed.panel.join(
        raw_levels[complete_level_columns].add_prefix("LEVEL__")
    )
    panel = replication_panel if "Level" in args.feature_case else processed.panel
    transformed_predictors = [column for column in processed.panel.columns if column != target]
    level_predictors = [f"LEVEL__{column}" for column in complete_level_columns]
    feature_predictors = transformed_predictors + (
        level_predictors if "Level" in args.feature_case else []
    )
    params = paper_model_params(
        args.model,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    search = paper_model_search(
        args.model,
        mode=args.tuning_mode,
        random_state=args.cv_random_state,
        search_iterations=args.search_iterations,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
    )

    manifest: dict[str, Any] = {
        "status": "running",
        "macroforecast_version": mf.__version__,
        "args": vars(args),
        "model_params": params,
        "model_selection": None if search is None else search.to_dict(),
        "raw_shape": list(bundle.panel.shape),
        "processed_shape": list(processed.panel.shape),
        "panel_shape": list(panel.shape),
        "target": target,
        "level_column_policy": "raw FRED-MD level columns with no missing values over the processed panel index",
        "level_column_count": len(level_predictors),
        "feature_column_policy": {
            "transformed_predictors": "all transformed FRED-MD columns except the forecast target",
            "target_derived_blocks": "MARX_y and MAF_y use feature_spec input='target_panel'",
            "level_predictors": "raw FRED-MD level columns, including LEVEL__target for Y_t",
            "factor_lags": "0..12",
            "x_lags": "0..12",
            "level_lags": "0 only",
            "target_lags": "0..12",
        },
        "dropped_level_columns": [
            column for column in bundle.panel.columns if column not in complete_level_columns
        ],
        "chunks": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    frames: list[pd.DataFrame] = []
    for year in range(args.start_year, args.end_year + 1):
        chunk_started = time.time()
        window = mf.window.from_cutoffs(
            estimation_start="1960-01",
            test_start=f"{year}-01",
            test_end=f"{year}-12",
            val_method="random_kfold" if search is not None else "last_block",
            val_size=None if search is not None else 60,
            val_n_splits=5,
            val_random_state=args.cv_random_state,
            horizon=args.horizon,
            step=1,
        )
        features = mf.feature_engineering.feature_spec(
            target=target,
            horizon=args.horizon,
            predictors=feature_predictors,
            steps=paper_feature_steps(
                args.feature_case,
                target,
                transformed_predictors,
                level_predictors,
            ),
            target_lags=range(0, 13),
            target_transform="average_value" if args.target_policy == "direct_average" else "value",
            target_mode="direct" if args.target_policy == "direct_average" else "path",
        )
        try:
            result = mf.forecasting.run(
                panel,
                args.model,
                window=window,
                features=features,
                target=target,
                horizon=args.horizon,
                forecast_policy=args.target_policy,
                target_transform="value",
                model_selection={args.model: search},
                params=params,
                save_models=False,
            )
        except ValueError as error:
            if not _is_empty_horizon_tail_error(error):
                raise
            chunk_meta = {
                "year": year,
                "rows": 0,
                "seconds": round(time.time() - chunk_started, 3),
                "forecast_csv": None,
                "metrics_csv": None,
                "invalid_rows": 0,
                "nan_prediction_rows": 0,
                "nan_actual_rows": 0,
                "skip_reason": "no realized target-date origins remain for this horizon",
            }
            manifest["chunks"].append(chunk_meta)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print("SKIP_YEAR", json.dumps(chunk_meta), flush=True)
            break
        frame = result.to_frame()
        validation = _validate_target_dates(frame, args.horizon)
        metrics = result.evaluate(by=("model", "horizon", "forecast_policy"), metrics=("rmse", "mae"))
        stem = f"{args.target_alias}_h{args.horizon}_{args.feature_case}_{args.target_policy}_{args.model}_{year}"
        frame_path = out_dir / f"{stem}.csv"
        metric_path = out_dir / f"{stem}_metrics.csv"
        frame.to_csv(frame_path, index=False)
        metrics.to_csv(metric_path, index=False)
        frames.append(frame)
        elapsed = round(time.time() - chunk_started, 3)
        chunk_meta = {
            "year": year,
            "rows": int(len(frame)),
            "seconds": elapsed,
            "forecast_csv": str(frame_path),
            "metrics_csv": str(metric_path),
            **validation,
        }
        manifest["chunks"].append(chunk_meta)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("DONE_YEAR", json.dumps(chunk_meta), flush=True)
        print(metrics.to_string(index=False), flush=True)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_stem = (
        f"{args.target_alias}_h{args.horizon}_{args.feature_case}_"
        f"{args.target_policy}_{args.model}_{args.start_year}_{args.end_year}"
    )
    combined_path = out_dir / f"{combined_stem}_all.csv"
    combined_metrics_path = out_dir / f"{combined_stem}_metrics.csv"
    summary = {
        **_combined_metrics(combined),
        **_validate_target_dates(combined, args.horizon),
    }
    combined.to_csv(combined_path, index=False)
    pd.DataFrame([summary]).to_csv(combined_metrics_path, index=False)
    manifest.update(
        {
            "status": "done",
            "total_seconds": round(time.time() - started, 3),
            "combined_csv": str(combined_path),
            "combined_metrics_csv": str(combined_metrics_path),
            "summary": summary,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("DONE_ALL", json.dumps(manifest["summary"]), flush=True)
    print("TOTAL_SECONDS", manifest["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
