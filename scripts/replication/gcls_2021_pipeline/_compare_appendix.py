"""Compare the GCLS full-grid run results against the Appendix B ground truth.

Our run covers AR, FM (benchmark), and Random Forest with four feature sets
(F-Level, X-Level, MARX, F-X-MARX-Level), for 10 targets x 6 horizons x two
policies. The appendix reports relative-RMSE (ratio to FM). We compute
relRMSE = sqrt(relative_mse) from each accuracy CSV and diff it against the
matching appendix cell.

    python scripts/replication/gcls_2021_pipeline/_compare_appendix.py
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
# The appendix ground-truth tables were merged into the single replication page;
# the table format the parser scans is unchanged, only the file moved.
DOC = REPO / "docs/replication/gcls_2021_replication.md"
RUN = Path("/home/nanyeon99/project/macroforecast_runs/gcls_fullgrid_step8")

TARGETS = ["CONS", "CPI", "EMP", "HOUST", "INCOME", "INDPRO", "M2", "PPI", "RETAIL", "UNRATE"]
HORIZONS = [1, 3, 6, 9, 12, 24]
# our contender -> (paper model, paper set)
ARM_MAP = {
    "AR": ("AR", "—"),
    "RF_F-Level": ("Random Forest", "F-Level"),
    "RF_X-Level": ("Random Forest", "X-Level"),
    "RF_MARX": ("Random Forest", "MARX"),
    "RF_F-X-MARX-Level": ("Random Forest", "F-X-MARX-Level"),
}


def parse_appendix() -> dict:
    """Return paper[(policy, horizon, target, model, set)] = relRMSE."""
    text = DOC.read_text().splitlines()
    paper: dict = {}
    policy = horizon = None
    targets_order: list[str] = []
    model = None
    # Match any heading level (the appendix tables were merged into the single
    # replication page and demoted from "### Horizon" to "#### Horizon").
    sec = re.compile(r"^#+ Horizon (\d+) \((direct|path-average)\)")
    for line in text:
        m = sec.match(line)
        if m:
            horizon = int(m.group(1))
            policy = m.group(2)
            targets_order = []
            model = None
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # header row: "Model | Set | INDPRO | EMP | ..."
        if cells[:2] == ["Model", "Set"]:
            targets_order = cells[2:]
            continue
        if cells[0].startswith("---") or not targets_order:
            continue
        if len(cells) != 2 + len(targets_order):
            continue
        if cells[0]:
            model = cells[0]
        set_ = cells[1]
        for tgt, val in zip(targets_order, cells[2:]):
            try:
                paper[(policy, horizon, tgt, model, set_)] = float(val)
            except ValueError:
                pass
    return paper


def _contender_series(df: pd.DataFrame, contender: str) -> pd.DataFrame:
    key = "contender" if "contender" in df.columns else "arm"
    s = df[df[key] == contender].dropna(subset=["prediction", "actual"])
    return s.set_index("origin")[["prediction", "actual"]]


def _pairwise_relmse(con: pd.DataFrame, fm: pd.DataFrame) -> float:
    """relative MSE of a contender vs the FM benchmark on their common origins."""
    co = con.index.intersection(fm.index)
    if len(co) < 8:
        return float("nan")
    mse_c = float(((con.loc[co, "prediction"] - con.loc[co, "actual"]) ** 2).mean())
    mse_f = float(((fm.loc[co, "prediction"] - fm.loc[co, "actual"]) ** 2).mean())
    return mse_c / mse_f if mse_f > 0 else float("nan")


def load_ours() -> pd.DataFrame:
    """Re-score the saved forecasts against the appendix's FM-benchmark convention.

    The appendix prints the SAME FM absolute RMSE above the direct (Tables 3-8)
    and the path-average (Tables 9-14) tables, so the paper uses a single FM
    benchmark -- the DIRECT FM -- as the denominator for both. Our pipeline
    instead scores each policy against its own-policy FM, which for volatile
    real-activity series makes the path-average denominator much larger than the
    paper's and pushes every path relRMSE below the appendix. Here the direct FM
    is the denominator for BOTH tables, matching the paper. Re-computing from the
    per-origin parquets also restores the leak-free 1980-2017 pairwise sample
    that the run's own accuracy CSVs lost to the evaluation-truncation bug.
    """
    rows = []
    for tgt in TARGETS:
        for h in HORIZONS:
            fpd = RUN / tgt / "direct_average" / f"{tgt}_direct_average_h{h}.parquet"
            fpp = RUN / tgt / "path_average" / f"{tgt}_path_average_h{h}.parquet"
            if not fpd.exists():
                continue
            dd = pd.read_parquet(fpd)
            fm_direct = _contender_series(dd, "FM")  # the single benchmark
            sources = [("direct", dd)]
            if fpp.exists():
                sources.append(("path-average", pd.read_parquet(fpp)))
            for pol, frame in sources:
                for arm, (model, set_) in ARM_MAP.items():
                    rel = _pairwise_relmse(_contender_series(frame, arm), fm_direct)
                    rows.append({
                        "policy": pol, "horizon": h, "target": tgt,
                        "model": model, "set": set_, "arm": arm,
                        "ours": math.sqrt(rel) if rel == rel and rel >= 0 else float("nan"),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    paper = parse_appendix()
    ours = load_ours()
    ours["paper"] = ours.apply(
        lambda r: paper.get((r["policy"], r["horizon"], r["target"], r["model"], r["set"])),
        axis=1,
    )
    matched = ours.dropna(subset=["paper"]).copy()
    matched["delta"] = matched["ours"] - matched["paper"]
    matched["abs"] = matched["delta"].abs()

    print(f"matched cells: {len(matched)} / {len(ours)} (unmatched paper lookups: "
          f"{ours['paper'].isna().sum()})")
    print("\n=== overall ===")
    print(f"  mean |delta| = {matched['abs'].mean():.3f}  | median = {matched['abs'].median():.3f}"
          f"  | max = {matched['abs'].max():.3f}")
    print(f"  within 0.02: {(matched['abs']<=0.02).mean()*100:.0f}%   "
          f"within 0.05: {(matched['abs']<=0.05).mean()*100:.0f}%   "
          f"within 0.10: {(matched['abs']<=0.10).mean()*100:.0f}%")

    print("\n=== by arm (mean|delta|, max|delta|, n) ===")
    g = matched.groupby("arm")["abs"].agg(["mean", "max", "count"])
    print(g.round(3).to_string())

    print("\n=== by horizon (mean|delta|) ===")
    print(matched.groupby("horizon")["abs"].mean().round(3).to_string())

    print("\n=== by policy ===")
    print(matched.groupby("policy")["abs"].agg(["mean", "max", "count"]).round(3).to_string())

    print("\n=== largest divergences (|delta| > 0.10) ===")
    big = matched[matched["abs"] > 0.10].sort_values("abs", ascending=False)
    print(f"  count: {len(big)}")
    cols = ["policy", "horizon", "target", "arm", "ours", "paper", "delta"]
    print(big[cols].head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
