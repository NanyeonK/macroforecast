"""HLZ Table 6 -- R2_OS by forecast ensemble.

Table 6 estimates nothing new: it averages the method-level paths that Tables 4
and 5 already produced. So it is assembled from the saved forecast frames and run
through `evaluate(master, spec)` -- same evaluation code as a live run, no refit.

Ensemble definitions, from Section 3.3:
  (a) Linear   = PCR, PLS, S-PCR, LASSO, ENet
  (b) Nonlinear= "neural networks with varying depths". [GAP] The Table 6 note says
      "from one to four" while Table 5 reports NN1-NN5; both readings are produced.
  (c) NN & ENet
  (d) All methods

Inherited disagreement. Our LASSO/ENet columns differ from the paper structurally
(doc section 5.1) and our NN paths depend on a training protocol the paper never
states (section 5.2). Table 6 therefore inherits both; it is reported as what our
components ensemble to, not as an independent check of them.
"""
import sys, warnings, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import macroforecast as mf
from macroforecast.pipeline import (Arm, EvalSpec, CombinationContender,
                                    pipeline_spec, evaluate, TargetSpec)

DESIGNS = {"L1": "Xt", "L6": "Xt + {MA2..MA6}", "L12": "Xt + {MA2..MA12}"}
J = 1                                  # [GAP] number of factors unspecified; J=1 favoured
LINEAR = ["PCR", "PLS", "S-PCR", "LASSO", "ENet"]
PRINTED = {  # (Linear, Nonlinear, NN+ENet, All)
    "L1":  (0.35, -0.18, 0.01, 0.42),
    "L6":  (0.90, 0.99, 1.03, 1.08),
    "L12": (1.20, 1.44, 1.51, 1.45),
}

panel = pd.read_parquet("/tmp/hlz_panel.parquet")

def components(design):
    """Return {method_name: date-indexed prediction Series} plus the HA rows."""
    t4 = pd.read_parquet(f"/tmp/hlz_t4_{design}_J{J}.parquet")
    t4s = pd.read_parquet(f"/tmp/hlz_t4s_{design}.parquet")
    out = {}
    for m in ("PCR", "PLS", "S-PCR"):
        r = t4[t4["contender"] == f"COMB_{m}"]
        out[m] = r.set_index("date")["prediction"].sort_index()
    for m, cand in (("LASSO", ["COMB_LASSO", "LASSO_lag1"]),
                    ("ENet", ["COMB_ENet", "ENet_lag1"])):
        for c in cand:
            r = t4s[t4s["contender"] == c]
            if len(r):
                out[m] = r.set_index("date")["prediction"].sort_index()
                break
        else:
            raise KeyError(f"{m} not found in {design}: {sorted(t4s['contender'].unique())}")
    for k in range(1, 6):
        arr = np.load(f"/tmp/hlz_t5_{design}_NN{k}.npy")
        out[f"NN{k}"] = pd.Series(arr, index=out["PCR"].index[-len(arr):]).sort_index()
    ha = t4[t4["contender"] == "HA"].copy()
    return out, ha

def build_master(comp, ha, members):
    """Clone the HA rows' schema for each member so `evaluate` sees a normal frame."""
    parts = [ha.assign(arm="HA", contender="HA")]
    for name in members:
        s = comp[name].reindex(ha["date"].values)
        row = ha.copy()
        row["prediction"] = s.to_numpy(float)
        row["arm"] = name
        row["contender"] = name
        row["model"] = "ols"
        row["model_spec"] = "ols"
        parts.append(row)
    m = pd.concat(parts, ignore_index=True)
    for c in ("params", "model_selection", "stored_model", "window", "combination"):
        if c in m.columns:
            m[c] = None
    return m

results, sig_out = {}, {}
for design, label in DESIGNS.items():
    comp, ha = components(design)
    NL4, NL5 = [f"NN{k}" for k in range(1, 5)], [f"NN{k}" for k in range(1, 6)]
    ENS = {
        "Linear":        LINEAR,
        "Nonlinear(1-4)": NL4,
        "Nonlinear(1-5)": NL5,
        "NN+ENet(1-4)":  NL4 + ["ENet"],
        "NN+ENet(1-5)":  NL5 + ["ENet"],
        "All(1-4)":      LINEAR + NL4,
        "All(1-5)":      LINEAR + NL5,
    }
    members = sorted({m for v in ENS.values() for m in v})
    master = build_master(comp, ha, members)
    frame = pd.concat([panel["mkt_excess"].rename("y")], axis=1)
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 1})
    w = mf.window.from_cutoffs(test_start=panel.index[456], horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=24)
    feats = mf.feature_engineering.feature_spec(target="y", target_lags=(1,))
    arms = [Arm("HA", model="hist_mean", features=feats, is_benchmark=True)]
    arms += [Arm(m, model="ols", features=feats, nested_in_benchmark=True) for m in members]
    combos = [CombinationContender(name=k, method="mean", over=tuple(v),
                                   nested_in_benchmark=True) for k, v in ENS.items()]
    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec("y", transform="level")], horizons=[1], window=w,
        arms=arms,
        evaluation=EvalSpec(benchmark="HA", metrics=("relative_mse", "r2_oos"),
                            tests=("dm", "cw"), test_options={"dm": {"hac_lags": 4}}),
        combinations=combos, save_models=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evaluate(master, spec)
    acc, sg = res["accuracy"], res["significance"]
    results[design] = {k: float(acc[acc["contender"] == k]["r2_oos"].iloc[0]) * 100 for k in ENS}
    sig_out[design] = {k: (float(sg[sg["contender"] == k]["cw_stat"].iloc[0]),
                           float(sg[sg["contender"] == k]["cw_p"].iloc[0])) for k in ENS}
    results[design]["_members"] = {k: len(v) for k, v in ENS.items()}

COLS = [("Linear", "Linear"), ("Nonlinear", "Nonlinear(1-4)"),
        ("NN+ENet", "NN+ENet(1-4)"), ("All", "All(1-4)")]
print("\n########## TABLE 6 -- NN1..NN4 reading (the Table 6 note) ##########")
print("| design | " + " | ".join(f"{c} mine | printed | Δ" for c, _ in COLS) + " |")
print("|" + "---|" * 13)
for d, label in DESIGNS.items():
    cells = []
    for j, (_, key) in enumerate(COLS):
        m, p = results[d][key], PRINTED[d][j]
        cells.append(f"{m:.3f} | {p:.2f} | {m - p:+.3f}")
    print(f"| {label} | " + " | ".join(cells) + " |")

print("\n########## same, NN1..NN5 reading (Table 5's set) ##########")
COLS5 = [("Linear", "Linear"), ("Nonlinear", "Nonlinear(1-5)"),
         ("NN+ENet", "NN+ENet(1-5)"), ("All", "All(1-5)")]
print("| design | " + " | ".join(f"{c} mine | printed | Δ" for c, _ in COLS5) + " |")
print("|" + "---|" * 13)
for d, label in DESIGNS.items():
    cells = []
    for j, (_, key) in enumerate(COLS5):
        m, p = results[d][key], PRINTED[d][j]
        cells.append(f"{m:.3f} | {p:.2f} | {m - p:+.3f}")
    print(f"| {label} | " + " | ".join(cells) + " |")

print("\n########## Clark-West on the ensembles (NN1..NN4 reading) ##########")
print("| design | ensemble | CW stat | CW p |")
print("|---|---|---|---|")
for d, label in DESIGNS.items():
    for c, key in COLS:
        st, p = sig_out[d][key]
        print(f"| {label} | {c} | {st:.3f} | {p:.3f} |")

json.dump({"results": results, "sig": sig_out}, open("/tmp/hlz_t6.json", "w"), indent=1)
print("\nTABLE6_DONE")
