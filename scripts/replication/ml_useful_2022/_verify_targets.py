"""Bounded, deterministic verification of the corrected ML-Useful forecast TARGET.

Builds the SAME augmented bundle as run_full.py, applies the SAME official panel
transform, then constructs the realized forecast target with the SAME builder the
pipeline uses (``direct_target`` with ``average_value`` for the 4 growth/change
targets and ``value`` for the T10YFFM spread level). Reports, restricted to the POOS
sample 1980-2017, per (target, horizon) in {1,3,9,12,24}:

  * count of NON-NaN realized targets -- must be NON-zero for every cell, especially
    INDPRO h24 and CPIAUCSL h24 (previously silently empty under the double-transform);
  * the realized-target scale (mean / std / min / max) so the Delta-log targets show a
    growth scale (|mean| < 0.5), UNRATE a change scale, and T10YFFM a spread level.

This isolates the target construction (no model fitting) so it is fast and exact.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

import macroforecast as mf  # noqa: E402
from macroforecast.feature_engineering.targets import direct_target  # noqa: E402
from registry import YTARGET_PREFIX, ml_useful_targets, ytarget_column  # noqa: E402
from run_full import _augmented_bundle  # noqa: E402

HORIZONS = (1, 3, 9, 12, 24)
POOS_START = "1980-01-01"
POOS_END = "2017-12-01"


def main(data_csv: str) -> None:
    raw = mf.data.load_fred_md(local_source=data_csv)
    bundle = _augmented_bundle(raw)

    # Apply the SAME official panel transform the pipeline uses. The YTARGET__ columns
    # carry identity t-code (1) so they pass through unchanged; the realized forecast
    # target is then built from these one-period objects. (Imputation/outliers do not
    # affect the YTARGET__ columns, which are complete; we keep the official transform.)
    transformed = mf.preprocessing.reprocess(
        bundle, transform="official", impute="none", outliers="none", standardize="none",
    )
    panel = transformed.panel

    print(f"transformed panel: {panel.shape[0]} rows x {panel.shape[1]} cols "
          f"({panel.index.min().date()}..{panel.index.max().date()})")

    rows = []
    for tspec in ml_useful_targets():
        raw_col = tspec.name[len(YTARGET_PREFIX):]
        # transform="value"+policy direct_average -> average_value; policy direct -> value
        target_transform = "value" if tspec.policy == "direct" else "average_value"
        for h in HORIZONS:
            tgt = direct_target(panel, target=tspec.name, horizon=h, transform=target_transform)
            col = tgt.columns[0]
            s = tgt[col]
            # Restrict to the POOS origin sample 1980-2017 (origins t whose target uses
            # Y_{t+h}); we count over origin dates in the POOS window.
            s = s[(s.index >= POOS_START) & (s.index <= POOS_END)]
            a = pd.to_numeric(s, errors="coerce").dropna()
            rows.append((raw_col, h, int(a.shape[0]),
                         float(a.mean()) if len(a) else float("nan"),
                         float(a.std()) if len(a) else float("nan"),
                         float(a.min()) if len(a) else float("nan"),
                         float(a.max()) if len(a) else float("nan")))

    print("\n=== realized target (1980-2017) non-NaN counts + scale, by (target, horizon) ===")
    print(f"{'target':10s} {'h':>3s} {'transform':>14s} {'nonNaN':>7s} "
          f"{'mean':>10s} {'std':>9s} {'min':>10s} {'max':>10s}")
    tmap = {ytarget_column(c): ("value" if t.policy == "direct" else "average_value")
            for c, t in zip([s[len(YTARGET_PREFIX):] for s in
                             [ts.name for ts in ml_useful_targets()]], ml_useful_targets())}
    for raw_col, h, n, mean, std, mn, mx in rows:
        tt = tmap[ytarget_column(raw_col)]
        if n:
            print(f"{raw_col:10s} {h:>3d} {tt:>14s} {n:>7d} "
                  f"{mean:>10.4f} {std:>9.4f} {mn:>10.4f} {mx:>10.4f}")
        else:
            print(f"{raw_col:10s} {h:>3d} {tt:>14s} {n:>7d} "
                  f"{'--':>10s} {'--':>9s} {'--':>10s} {'--':>10s}")

    # Explicit pass/fail on the previously-empty cells.
    by_cell = {(r, h): n for r, h, n, *_ in rows}
    print("\n=== previously-empty cells (must now be NON-zero) ===")
    ok = True
    for raw_col, h in (("INDPRO", 24), ("CPIAUCSL", 9), ("CPIAUCSL", 12), ("CPIAUCSL", 24)):
        n = by_cell.get((raw_col, h), 0)
        flag = "PASS" if n > 0 else "FAIL"
        ok = ok and n > 0
        print(f"  {raw_col} h{h}: nonNaN={n}  [{flag}]")
    print(f"\nVERIFY {'PASS' if ok else 'FAIL'}: all previously-empty cells now non-empty")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ml_useful_data/2018-01.csv")
