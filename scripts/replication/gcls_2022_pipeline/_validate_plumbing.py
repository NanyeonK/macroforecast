"""Tiny end-to-end plumbing check for GCLS-2022 STAGE-1 (NOT the G1 gate).

2 arms (AR,BIC + ARDI,BIC) x INDPRO x h=1 over a short 24-origin window. Confirms:
preprocessing runs, the target is built from RAW levels (small Delta-log magnitudes,
not double-transformed), forecasts are finite, relative-MSE is produced.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO))

import macroforecast as mf
from macroforecast.pipeline import EvalSpec, TargetSpec, pipeline_spec, run_pipeline

from scripts.replication.gcls_2022_pipeline.data import augmented_bundle, gcls_targets
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms

bundle, predictors = augmented_bundle()
print("n predictors:", len(predictors))
all_arms = build_gcls2022_arms("YOBJ__INDPRO", predictors)
arms = [a for a in all_arms if a.name in ("AR,BIC", "ARDI,BIC")]
print("arms:", [a.name for a in arms])
indpro_target = [t for t in gcls_targets() if t.name == "YOBJ__INDPRO"]

window = mf.window.from_cutoffs(
    estimation_start="1960-01", test_start="1980-01", test_end="1981-12",
    mode="expanding", retrain_every=1, retune_every=24, retune_on_retrain=False,
)
pp = mf.preprocessing.preprocess_spec(
    transform="official", impute="em_factor", outliers="iqr",
    outlier_action="flag_as_nan", standardize="none",
)
pp_policy = mf.window.stage_policy("origin_available", update=1)

spec = pipeline_spec(
    data=bundle,
    targets=indpro_target,
    horizons=[1],
    window=window,
    arms=arms,
    evaluation=EvalSpec(benchmark="AR,BIC", tests=("dm",)),
    preprocessing=pp,
    preprocessing_policy=pp_policy,
    selection_history=True,
    checkpoint_dir=str(REPO / "runs" / "gcls_b4_stage1" / "_validate_ckpt2"),
    n_jobs=1,
    seed=42,
)
report = run_pipeline(spec)
frame = report.to_frame()
print("frame shape:", frame.shape)
print("frame cols:", list(frame.columns))
import pandas as pd
# actual target values sanity (Delta-log IP average ~ +/-0.01)
for col in ("actual", "y_true", "target_value", "y", "realized"):
    if col in frame.columns:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(s):
            print(f"target[{col}]: n={len(s)} mean={s.mean():.5f} std={s.std():.5f} "
                  f"min={s.min():.5f} max={s.max():.5f}")
        break
# forecasts finite?
for col in ("forecast", "prediction", "yhat", "point_forecast"):
    if col in frame.columns:
        s = pd.to_numeric(frame[col], errors="coerce")
        print(f"forecast[{col}]: n={len(s)} finite={int(s.notna().sum())} "
              f"nan={int(s.isna().sum())}")
        break
print("n rows per arm:", frame.groupby("arm").size().to_dict() if "arm" in frame else "?")
# evaluation table
try:
    ev = report.evaluation if hasattr(report, "evaluation") else None
    print("has evaluation attr:", ev is not None)
except Exception as e:
    print("eval err:", e)
print("OK validate ran")
