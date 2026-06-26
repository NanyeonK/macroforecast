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
    sec = re.compile(r"^### Horizon (\d+) \((direct|path-average)\)")
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


def load_ours() -> pd.DataFrame:
    rows = []
    for tgt in TARGETS:
        for pol_dir, pol in (("direct_average", "direct"), ("path_average", "path-average")):
            for h in HORIZONS:
                f = RUN / tgt / pol_dir / f"{tgt}_{pol_dir}_h{h}_accuracy.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                for _, r in df.iterrows():
                    c = r["contender"]
                    if c not in ARM_MAP:
                        continue
                    model, set_ = ARM_MAP[c]
                    rel = float(r["relative_mse"])
                    rows.append({
                        "policy": pol, "horizon": h, "target": tgt,
                        "model": model, "set": set_, "arm": c,
                        "ours": math.sqrt(rel) if rel >= 0 else float("nan"),
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
