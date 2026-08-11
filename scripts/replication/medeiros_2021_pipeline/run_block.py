"""B1 G2 pipeline parity runner: two-regime rolling forecasts vs paper Table 5.

Invocation from repo root:
  python3 scripts/replication/medeiros_2021_pipeline/run_block.py

Paper design: two fixed-rolling regimes concatenated for the full 1990-2015 OOS table.
  regime s1 (1990-01..2000-12): R = 360 - h - p - 1
  regime s2 (2001-01..2015-12): R = 492 - h - p - 1     (p = 4 embedding lags)
Table 5 = RMSE ratios of AR, UCSV, RF vs RW, full sample.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import sys
import warnings
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import macroforecast as mf  # noqa: E402
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline  # noqa: E402

try:  # noqa: E402
    from .registry import RF_PARAMS as REGISTRY_RF_PARAMS, UCSV_PARAMS, base_features
except ImportError:  # pragma: no cover - direct script execution
    from registry import RF_PARAMS as REGISTRY_RF_PARAMS, UCSV_PARAMS, base_features


PANEL = pd.read_parquet("qa/medeiros_panel.parquet")
TARGET = "CPIAUCSL"
P = 4
HORIZONS = (1, 3, 6, 12)
REGIMES = (
    ("s1", "1990-01", "2000-12", 360),
    ("s2", "2001-01", "2015-12", 492),
)

RF_PARAMS = {
    "n_estimators": 500,
    "max_features": 1.0 / 3.0,
    "min_samples_leaf": 5,
    "bootstrap": True,
    "max_samples": None,
    "max_leaf_nodes": None,
    "random_state": 42,
}
if dict(REGISTRY_RF_PARAMS) != RF_PARAMS:
    raise RuntimeError(
        "registry.RF_PARAMS must be exactly "
        "{'n_estimators': 500, 'max_features': 1.0/3.0, "
        "'min_samples_leaf': 5, 'bootstrap': True, 'max_samples': None, "
        "'max_leaf_nodes': None, 'random_state': 42}; "
        "do not pass n_jobs here because pipeline_spec(n_jobs='auto') budgets "
        "cell workers and model_threads."
    )

PAPER = {
    "ar": {1: 0.902, 3: 0.790, 6: 0.791, 12: 0.753},
    "ucsv": {1: 0.954, 3: 0.797, 6: 0.777, 12: 0.781},
    "rf": {1: 0.844, 3: 0.706, 6: 0.715, 12: 0.685},
}
PAPER_TOLERANCE = {
    "ar": 0.03,
    "ucsv": 0.05,
    "rf": 0.05,
}

SEQUENTIAL_BEFORE_SECONDS = {
    "rw": 76.3 + 78.5 + 78.7 + 76.5,
    "ar": 96.4 + 95.3 + 94.8 + 92.7,
    "rf": 1637.5 + 1637.3 + 1616.1 + 1574.8,
    "ucsv": 145.0 * 60.0,
}


def medeiros_rolling_size(base: int, horizon: int) -> int:
    size = int(base) - int(horizon) - P - 1
    if size < 1:
        raise ValueError(
            f"invalid Medeiros rolling size {size} for base={base}, horizon={horizon}"
        )
    return size


def regime_window(
    regime: str,
    test_start: str,
    test_end: str,
    base: int,
) -> mf.window.WindowSpec:
    return mf.window.from_cutoffs(
        mode="rolling",
        estimation_size=base,
        estimation_size_rule=medeiros_rolling_size,
        test_start=test_start,
        test_end=test_end,
        horizon=1,
        metadata={
            "medeiros_regime": regime,
            "medeiros_regime_base": int(base),
            "embedding_lags": int(P),
            "estimation_size_formula": "regime_base - horizon - embedding_lags - 1",
        },
    )


def ucsv_level_window(
    regime: str,
    test_start: str,
    test_end: str,
    base: int,
) -> mf.window.WindowSpec:
    return mf.window.from_cutoffs(
        mode="rolling",
        estimation_size=medeiros_rolling_size(base, 1),
        test_start=test_start,
        test_end=test_end,
        horizon=1,
        metadata={
            "medeiros_regime": regime,
            "medeiros_regime_base": int(base),
            "embedding_lags": int(P),
            "ucsv_flat_one_sided": True,
            "estimation_size_formula": "regime_base - 1 - embedding_lags - 1",
            "forecast_rule": (
                "fit unshifted target through origin once; reuse tau_T_given_T "
                "for every horizon"
            ),
        },
    )


def univariate_ar_features():
    return mf.feature_spec(
        predictors=None,
        target_lags=(0, 1, 2, 3),
    )


def g2_arms(selected: set[str] | None = None) -> list[Arm]:
    arms = [
        Arm("rw", model="naive", features=None, is_benchmark=True),
        Arm("ar", model="ar", features=univariate_ar_features()),
        Arm("ucsv", model="ucsv", features=None, params=dict(UCSV_PARAMS)),
        Arm(
            "rf",
            model="random_forest",
            features=base_features(),
            params=dict(RF_PARAMS),
            model_selection={"random_forest": None},
        ),
    ]
    if selected is None:
        return arms
    keep = set(selected)
    keep.add("rw")
    available = {arm.name for arm in arms}
    unknown = sorted(keep - available)
    if unknown:
        raise SystemExit(f"unknown --arms value(s): {unknown}; available: {sorted(available)}")
    return [arm for arm in arms if arm.name in keep]


def regime_arms(
    arms: list[Arm],
    regime: str,
    test_start: str,
    test_end: str,
    base: int,
) -> list[Arm]:
    out: list[Arm] = []
    for arm in arms:
        if arm.name == "ucsv":
            out.append(
                replace(
                    arm,
                    window=ucsv_level_window(regime, test_start, test_end, base),
                )
            )
        else:
            out.append(arm)
    return out


def rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((frame["actual"] - frame["prediction"]) ** 2)))


def aligned_ratio(model_frame: pd.DataFrame, rw_frame: pd.DataFrame) -> tuple[float, int]:
    keys = ["date", "horizon"]
    model = model_frame.set_index(keys).sort_index()
    rw = rw_frame.set_index(keys).sort_index()
    aligned = model.reindex(rw.index).dropna(subset=["prediction", "actual"])
    rw_aligned = rw.loc[aligned.index]
    return rmse(aligned) / rmse(rw_aligned), int(len(aligned))


def verdict(model: str, delta: float) -> str:
    ad = abs(float(delta))
    if ad <= 0.01:
        return "MATCH"
    if ad <= PAPER_TOLERANCE[model]:
        return "CLOSE"
    return "DIVERGENT"


def run_regime(regime: tuple[str, str, str, int], arms: list[Arm]) -> tuple[pd.DataFrame, float]:
    name, test_start, test_end, base = regime
    window = regime_window(name, test_start, test_end, base)
    active_arms = regime_arms(arms, name, test_start, test_end, base)
    target = TargetSpec(TARGET, transform="level", policy="direct")
    spec = pipeline_spec(
        data=PANEL,
        targets=[target],
        horizons=HORIZONS,
        window=window,
        arms=active_arms,
        evaluation=EvalSpec(benchmark="rw"),
        n_jobs="auto",
        seed=42,
        result_store="qa/result_cells",
        preprocessing_cache_dir="qa/prep_cache",
        save_models=False,
        on_unsupported_direct="warn",
        provenance={
            "paper": "Medeiros et al. 2021 IJF",
            "block": "B1 G2",
            "window_formula": "R = regime_base - horizon - 4 - 1",
        },
    )
    if any(arm.name == "ucsv" for arm in active_arms):
        # UCSV is the paper's flat one-sided level-trend benchmark: fit the
        # unshifted target through each origin and reuse tau_{T|T} at every h.
        # Keep RW as the direct-policy benchmark denominator.
        spec = replace(
            spec,
            policy_overrides={
                **dict(spec.policy_overrides),
                ("ucsv", TARGET): "recursive",
            },
        )
    print(
        f"RUN_PIPELINE regime={name} test={test_start}..{test_end} "
        f"base={base} arms={[arm.name for arm in active_arms]} horizons={list(HORIZONS)} "
        f"n_jobs={spec.n_jobs} model_threads={spec.model_threads}",
        flush=True,
    )
    print(
        "WINDOW_R "
        + " ".join(
            f"h={h}:R={medeiros_rolling_size(base, h)}"
            for h in HORIZONS
        ),
        flush=True,
    )
    if any(arm.name == "ucsv" for arm in active_arms):
        print(
            "UCSV_CONFIG "
            f"policy_override=recursive rw_policy=direct "
            f"flat_R={medeiros_rolling_size(base, 1)} "
            "forecast=tau_T_given_T_broadcast",
            flush=True,
        )
    start = perf_counter()
    report = run_pipeline(spec)
    elapsed = perf_counter() - start
    failed = getattr(report, "failed_cells", ())
    if failed is not None and len(failed):
        print(f"FAILED_CELLS regime={name}: {failed}", flush=True)
        raise SystemExit(f"run_pipeline failed {len(failed)} cell(s) in regime {name}")
    forecasts = report.forecasts.copy()
    forecasts["regime"] = name
    print(f"DONE_PIPELINE regime={name} elapsed_sec={elapsed:.1f}", flush=True)
    return forecasts, elapsed


def score(master: pd.DataFrame, arms: list[Arm]) -> dict[tuple[str, int], dict[str, float | int | str]]:
    out: dict[tuple[str, int], dict[str, float | int | str]] = {}
    arm_names = {arm.name for arm in arms}
    print(f"\n{'':6}", *[f"h={h:<7}" for h in HORIZONS])
    rw_frames: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        rw = master.loc[
            (master["arm"] == "rw") & (master["horizon"] == horizon),
            ["date", "horizon", "prediction", "actual"],
        ].dropna(subset=["prediction", "actual"])
        rw_frames[horizon] = rw
    print("OOS n :", *[f"{len(rw_frames[h]):<9}" for h in HORIZONS])
    print("RWrmse:", *[f"{rmse(rw_frames[h]):<9.4f}" for h in HORIZONS])
    print("\n--- RMSE ratio vs RW (paper Table 5, Panel a) ---")
    print("model,h,our_ratio,paper_ratio,d,verdict,aligned_n")
    for model in ("ar", "ucsv", "rf"):
        if model not in arm_names:
            continue
        row = []
        for horizon in HORIZONS:
            frame = master.loc[
                (master["arm"] == model) & (master["horizon"] == horizon),
                ["date", "horizon", "prediction", "actual"],
            ].dropna(subset=["prediction", "actual"])
            ratio, n = aligned_ratio(frame, rw_frames[horizon])
            paper = PAPER[model][horizon]
            delta = ratio - paper
            tag = verdict(model, delta)
            out[(model, horizon)] = {
                "model": model,
                "horizon": horizon,
                "ratio": ratio,
                "paper": paper,
                "delta": delta,
                "verdict": tag,
                "aligned_n": n,
            }
            print(f"{model.upper()},{horizon},{ratio:.6f},{paper:.3f},{delta:+.6f},{tag},{n}")
            row.append(f"{ratio:.3f}(paper {paper:.3f}, d={delta:+.3f}, {tag})")
        print(f"{model.upper():6}", *row)
    return out


def summarize_paper_parity(
    results: dict[tuple[str, int], dict[str, float | int | str]],
) -> None:
    divergent = [
        record
        for record in results.values()
        if str(record["verdict"]) == "DIVERGENT"
    ]
    if divergent:
        print("\nPAPER_T5_PARITY DIVERGENT cells:", flush=True)
        for record in divergent:
            print(
                f"  {str(record['model']).upper()} h={int(record['horizon'])}: "
                f"our={float(record['ratio']):.6f}, paper={float(record['paper']):.3f}, "
                f"d={float(record['delta']):+.6f}, "
                f"tol={PAPER_TOLERANCE[str(record['model'])]:.3f}",
                flush=True,
            )
        return
    print(
        "P4 paper Table 5 parity PASS: no selected AR/UCSV/RF cell exceeds "
        "the corrected paper-oracle tolerance.",
        flush=True,
    )


def summarize_ucsv_flatness(master: pd.DataFrame) -> dict[str, float | int | str]:
    ucsv = master.loc[
        master["arm"] == "ucsv",
        ["origin", "horizon", "forecast_policy", "prediction"],
    ].dropna(subset=["prediction"])
    if ucsv.empty:
        print("UCSV_FLATNESS skipped: UCSV arm not present.", flush=True)
        return {
            "status": "SKIPPED",
            "common_origins": 0,
            "max_abs_range": float("nan"),
        }
    pivot = ucsv.pivot_table(
        index="origin",
        columns="horizon",
        values="prediction",
        aggfunc="first",
    )
    common = pivot.reindex(columns=list(HORIZONS)).dropna()
    max_abs_range = (
        float((common.max(axis=1) - common.min(axis=1)).abs().max())
        if not common.empty
        else float("nan")
    )
    status = (
        "PASS"
        if np.isfinite(max_abs_range) and max_abs_range <= 1e-10
        else "FAIL"
    )
    policies = ",".join(
        sorted(str(value) for value in ucsv["forecast_policy"].dropna().unique())
    )
    print(
        "UCSV_FLATNESS "
        f"status={status} common_origins={len(common)} "
        f"max_abs_range={max_abs_range:.12g} forecast_policy={policies}",
        flush=True,
    )
    if not common.empty:
        sample_origin = common.index[0]
        values = " ".join(
            f"h={h}:{float(common.loc[sample_origin, h]):.12g}"
            for h in HORIZONS
        )
        print(f"UCSV_FLATNESS_SAMPLE origin={sample_origin} {values}", flush=True)
    return {
        "status": status,
        "common_origins": int(len(common)),
        "max_abs_range": max_abs_range,
        "forecast_policy": policies,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        default="rw,ar,ucsv,rf",
        help="Comma-separated subset from rw,ar,ucsv,rf. rw is always included.",
    )
    return parser.parse_args()


def execute_pipeline(arms: list[Arm]) -> tuple[dict[tuple[str, int], dict[str, float | int | str]], float]:
    selected_names = {arm.name for arm in arms}
    print("B1 G2 pipeline parity: target=CPIAUCSL, horizons={1,3,6,12}", flush=True)
    print(
        "RF params explicit from author code: "
        "n_estimators=500, max_features=1.0/3.0, min_samples_leaf=5, "
        "bootstrap=True, max_samples=None, max_leaf_nodes=None, random_state=42; "
        "model_selection disabled; n_jobs omitted.",
        flush=True,
    )
    print("Pipeline config: n_jobs='auto', seed=42, result_store=qa/result_cells, prep_cache=qa/prep_cache, save_models=False", flush=True)
    print("TargetSpec explicit: transform='level', policy='direct' because qa/medeiros_panel.parquet is already transformed and has no t-code metadata.", flush=True)
    total_start = perf_counter()
    frames = []
    elapsed_by_regime: dict[str, float] = {}
    for regime in REGIMES:
        frame, elapsed = run_regime(regime, arms)
        frames.append(frame)
        elapsed_by_regime[regime[0]] = elapsed
    master = pd.concat(frames, ignore_index=True)
    results = score(master, arms)
    summarize_ucsv_flatness(master)
    summarize_paper_parity(results)
    total_elapsed = perf_counter() - total_start
    before = sum(SEQUENTIAL_BEFORE_SECONDS[name] for name in selected_names)
    print("\n--- Runtime seconds ---")
    for regime, elapsed in elapsed_by_regime.items():
        print(f"pipeline {regime}: {elapsed:.1f}")
    print(f"pipeline total: {total_elapsed:.1f}")
    print(f"sequential before comparable arms {sorted(selected_names)}: {before:.1f}")
    print(f"validated speedup: {before / total_elapsed:.2f}x")
    return results, total_elapsed


def main() -> None:
    args = parse_args()
    selected = {part.strip().lower() for part in args.arms.split(",") if part.strip()}
    arms = g2_arms(selected or None)
    execute_pipeline(arms)


if __name__ == "__main__":
    main()
