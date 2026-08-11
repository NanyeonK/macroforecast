"""Purge the 4 CV arms' cells (config digest unchanged, package behavior changed -> must
purge or the stale BIC-collapsed cells are reused). KRR arms changed config -> new digest
-> auto-recompute (not purged here)."""
import glob, json, os

CV_ARMS = {"AR,POOS", "AR,KF", "ARDI,POOS", "ARDI,KF"}
for store in ("_result_store_indpro", "_result_store_g2rest"):
    root = f"runs/gcls_b4_stage1/{store}/cells"
    removed = 0
    for pq in glob.glob(f"{root}/*.parquet"):
        import pandas as pd
        arm = pd.read_parquet(pq, columns=["arm"])["arm"].iloc[0]
        if arm in CV_ARMS:
            os.remove(pq)
            mf = pq[:-8] + ".json"
            if os.path.exists(mf):
                os.remove(mf)
            removed += 1
    print(f"{store}: purged {removed} cells (4 CV arms x horizons x targets)")
print("OK purge")
