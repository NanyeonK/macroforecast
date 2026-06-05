from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time
from pathlib import Path


PAPER_TABLE2 = {
    ("INDPRO", 1): ("direct_average", "random_forest", "F-X-MARX-Level"),
    ("INDPRO", 3): ("direct_average", "random_forest", "MARX"),
    ("INDPRO", 6): ("path_average", "random_forest", "MARX"),
    ("INDPRO", 9): ("path_average", "random_forest", "MARX"),
    ("INDPRO", 12): ("path_average", "random_forest", "MARX"),
    ("INDPRO", 24): ("direct_average", "random_forest", "F-Level"),
    ("EMP", 1): ("direct_average", "random_forest", "F-X-MARX-Level"),
    ("EMP", 3): ("path_average", "random_forest", "F-MARX"),
    ("EMP", 6): ("path_average", "gradient_boosting", "F-MARX"),
    ("EMP", 9): ("path_average", "gradient_boosting", "F-MARX"),
    ("EMP", 12): ("path_average", "gradient_boosting", "F-MARX"),
    ("EMP", 24): ("path_average", "gradient_boosting", "MAF"),
    ("UNRATE", 1): ("direct_average", "gradient_boosting", "F-MARX"),
    ("UNRATE", 3): ("direct_average", "random_forest", "F-X-MARX-Level"),
    ("UNRATE", 6): ("path_average", "random_forest", "F-MARX"),
    ("UNRATE", 9): ("path_average", "glmboost", "F-X-MARX-Level"),
    ("UNRATE", 12): ("path_average", "glmboost", "F-X-MARX-Level"),
    ("UNRATE", 24): ("direct_average", "gradient_boosting", "F-MAF"),
    ("INCOME", 1): ("direct_average", "random_forest", "MARX"),
    ("INCOME", 3): ("direct_average", "random_forest", "F-MARX"),
    ("INCOME", 6): ("path_average", "random_forest", "F-X-MARX"),
    ("INCOME", 9): ("path_average", "random_forest", "F-MARX"),
    ("INCOME", 12): ("path_average", "random_forest", "F-MARX"),
    ("INCOME", 24): ("path_average", "random_forest", "F-X-MARX"),
    ("CONS", 1): ("direct_average", "far", "F"),
    ("CONS", 3): ("direct_average", "random_forest", "F-Level"),
    ("CONS", 6): ("path_average", "random_forest", "F-Level"),
    ("CONS", 9): ("direct_average", "random_forest", "MAF"),
    ("CONS", 12): ("path_average", "random_forest", "F-MAF"),
    ("CONS", 24): ("path_average", "random_forest", "F-MAF"),
    ("RETAIL", 1): ("direct_average", "far", "F"),
    ("RETAIL", 3): ("path_average", "gradient_boosting", "F-X-MARX"),
    ("RETAIL", 6): ("path_average", "adaptive_lasso", "F-MARX"),
    ("RETAIL", 9): ("direct_average", "gradient_boosting", "F-X-MARX-Level"),
    ("RETAIL", 12): ("direct_average", "gradient_boosting", "F-X-Level"),
    ("RETAIL", 24): ("direct_average", "gradient_boosting", "F-X-MAF"),
    ("HOUST", 1): ("direct_average", "elastic_net", "F-Level"),
    ("HOUST", 3): ("path_average", "elastic_net", "F-Level"),
    ("HOUST", 6): ("path_average", "random_forest", "F-X-MARX"),
    ("HOUST", 9): ("direct_average", "random_forest", "F-MAF"),
    ("HOUST", 12): ("direct_average", "random_forest", "F"),
    ("HOUST", 24): ("direct_average", "random_forest", "F"),
    ("M2", 1): ("direct_average", "random_forest", "X-Level"),
    ("M2", 3): ("path_average", "adaptive_lasso", "X-Level"),
    ("M2", 6): ("path_average", "random_forest", "F-Level"),
    ("M2", 9): ("direct_average", "random_forest", "F-Level"),
    ("M2", 12): ("direct_average", "gradient_boosting", "F-Level"),
    ("M2", 24): ("path_average", "random_forest", "F-Level"),
    ("CPI", 1): ("direct_average", "adaptive_lasso", "MARX"),
    ("CPI", 3): ("direct_average", "random_forest", "F"),
    ("CPI", 6): ("direct_average", "random_forest", "F"),
    ("CPI", 9): ("direct_average", "random_forest", "F"),
    ("CPI", 12): ("direct_average", "random_forest", "F"),
    ("CPI", 24): ("path_average", "random_forest", "X"),
    ("PPI", 1): ("direct_average", "elastic_net", "F-MARX"),
    ("PPI", 3): ("direct_average", "elastic_net", "MARX"),
    ("PPI", 6): ("direct_average", "random_forest", "F"),
    ("PPI", 9): ("direct_average", "random_forest", "F"),
    ("PPI", 12): ("direct_average", "random_forest", "F"),
    ("PPI", 24): ("direct_average", "gradient_boosting", "F-Level"),
}


def _slug(target: str, horizon: int, policy: str, model: str, feature: str) -> str:
    return f"{target}_h{horizon}_{policy}_{model}_{feature}".replace("/", "_")


def _csv_values(value: str | None) -> set[str] | None:
    if value is None:
        return None
    values = {part.strip() for part in value.split(",") if part.strip()}
    return values or None


def _csv_ints(value: str | None) -> set[int] | None:
    values = _csv_values(value)
    if values is None:
        return None
    return {int(value) for value in values}


def _filter_tasks(
    tasks: list[tuple[str, int, str, str, str]],
    args: argparse.Namespace,
) -> list[tuple[str, int, str, str, str]]:
    targets = _csv_values(args.targets)
    horizons = _csv_ints(args.horizons)
    policies = _csv_values(args.policies)
    models = _csv_values(args.models)
    features = _csv_values(args.feature_cases)
    contains = _csv_values(args.task_contains)
    out: list[tuple[str, int, str, str, str]] = []
    for task in tasks:
        target, horizon, policy, model, feature = task
        slug = _slug(target, horizon, policy, model, feature)
        if targets is not None and target not in targets:
            continue
        if horizons is not None and horizon not in horizons:
            continue
        if policies is not None and policy not in policies:
            continue
        if models is not None and model not in models:
            continue
        if features is not None and feature not in features:
            continue
        if contains is not None and not any(token in slug for token in contains):
            continue
        out.append(task)
    if args.limit is not None:
        out = out[: max(0, int(args.limit))]
    return out


def _run_task(task: tuple[str, int, str, str, str], args: argparse.Namespace) -> dict[str, object]:
    target, horizon, policy, model, feature = task
    slug = _slug(target, horizon, policy, model, feature)
    out_dir = Path(args.out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    manifest_path = out_dir / "manifest.json"
    if args.skip_existing and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("status") == "done":
            return {"task": slug, "status": "skipped_done", "returncode": 0}
    cmd = [
        "uv",
        "run",
        "python",
        str(Path(args.single_script).resolve()),
        "--target-alias",
        target,
        "--horizon",
        str(horizon),
        "--feature-case",
        feature,
        "--target-policy",
        policy,
        "--model",
        model,
        "--vintage",
        args.vintage,
        "--cache-root",
        args.cache_root,
        "--out-dir",
        str(out_dir),
        "--start-year",
        str(args.start_year),
        "--end-year",
        str(args.end_year),
        "--n-estimators",
        str(args.n_estimators),
        "--random-state",
        str(args.random_state),
        "--tuning-mode",
        args.tuning_mode,
        "--cv-random-state",
        str(args.cv_random_state),
        "--search-iterations",
        str(args.search_iterations),
        "--ga-population",
        str(args.ga_population),
        "--ga-generations",
        str(args.ga_generations),
        "--skip-existing",
    ]
    env = os.environ.copy()
    for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[thread_var] = "1"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("CMD " + " ".join(cmd) + "\n")
        log.write("THREAD_LIMITS OMP=1 MKL=1 OPENBLAS=1 NUMEXPR=1\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    return {
        "task": slug,
        "target": target,
        "horizon": horizon,
        "policy": policy,
        "model": model,
        "feature": feature,
        "status": "done" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 3),
        "out_dir": str(out_dir),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--single-script",
        default=str(Path(__file__).with_name("gcls_2021_table2_single.py")),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--vintage", default="2018-01")
    parser.add_argument("--cache-root", default="/home/nanyeon99/project/macroforecast_replication_cache")
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2017)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument(
        "--tuning-mode",
        choices=("off", "paper-small", "paper"),
        default="off",
    )
    parser.add_argument("--cv-random-state", type=int, default=123)
    parser.add_argument("--search-iterations", type=int, default=20)
    parser.add_argument("--ga-population", type=int, default=25)
    parser.add_argument("--ga-generations", type=int, default=25)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--targets", help="Comma-separated target aliases, e.g. INDPRO,EMP")
    parser.add_argument("--horizons", help="Comma-separated horizons, e.g. 1,3,6")
    parser.add_argument("--policies", help="Comma-separated target policies")
    parser.add_argument("--models", help="Comma-separated model names")
    parser.add_argument("--feature-cases", help="Comma-separated feature cases")
    parser.add_argument(
        "--task-contains",
        help="Comma-separated substrings matched against the task slug",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N filtered tasks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "batch_status.jsonl"
    summary_path = out_root / "batch_summary.json"
    tasks = [
        (target, horizon, policy, model, feature)
        for (target, horizon), (policy, model, feature) in sorted(PAPER_TABLE2.items())
    ]
    tasks = _filter_tasks(tasks, args)
    summary = {
        "status": "running",
        "workers": args.workers,
        "task_count": len(tasks),
        "started_at_epoch": time.time(),
        "args": vars(args),
        "tasks": [
            {
                "target": target,
                "horizon": horizon,
                "policy": policy,
                "model": model,
                "feature": feature,
                "slug": _slug(target, horizon, policy, model, feature),
            }
            for target, horizon, policy, model, feature in tasks
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        for task in summary["tasks"]:
            print("TASK_PLAN", json.dumps(task), flush=True)
        summary.update(
            {
                "status": "dry_run",
                "finished_at_epoch": time.time(),
                "finished_count": 0,
                "failed_count": 0,
                "failures": [],
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return

    results: list[dict[str, object]] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_task, task, args) for task in tasks]
        for future in cf.as_completed(futures):
            result = future.result()
            results.append(result)
            with status_path.open("a", encoding="utf-8") as status_file:
                status_file.write(json.dumps(result) + "\n")
            print("TASK", json.dumps(result), flush=True)

    failures = [result for result in results if result.get("returncode") != 0]
    summary.update(
        {
            "status": "failed" if failures else "done",
            "finished_at_epoch": time.time(),
            "finished_count": len(results),
            "failed_count": len(failures),
            "failures": failures,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
