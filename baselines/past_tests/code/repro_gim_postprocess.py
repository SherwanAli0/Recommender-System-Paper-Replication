"""
repro_gim_postprocess.py

Hypothesis test for the baselines implementation GIM baseline:

  H_GIM_1: floor / round post-processing on predictions, mirroring the FUS
           replication finding. the baselines implementation's faithful GIM lands at MAE_users
           ~0.817 at k=10, vs paper claim ~0.85, gap of -4 percent. If the
           floor/round trick that closed the FUS gap also applies to GIM,
           it likely closes the GIM gap too.

  H_GIM_2: positive_only=False vs True. The Al-Shamri 2008 paper uses both
           positive and negative ratings; the baselines implementation defaults to positive_only=True
           (matching the upstream GitHub adaptation). Test both.

Output: repro_gim_postprocess.csv
"""
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "code"))
from gim import FuzzyGeneticMethod, load_and_filter

DATA_DIR = os.path.join(_HERE, "..", "..", "..", "ml-100k")
RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RATINGS_PATH = os.path.join(DATA_DIR, "u.data")
ITEMS_PATH = os.path.join(DATA_DIR, "u.item")
USERS_PATH = os.path.join(DATA_DIR, "u.user")

K_VALUES = [1, 2, 4, 6, 8, 10]


def fold_metrics(test_triples, preds):
    actuals = np.array([t[2] for t in test_triples], dtype=np.float64)
    uids = [t[0] for t in test_triples]
    made_mask = ~np.isnan(preds)
    n_total = len(actuals)
    n_made = int(made_mask.sum())
    cr = n_made / n_total if n_total > 0 else 0.0
    if n_made == 0:
        return dict(MAE_data=9.9, MAE_users=9.9, RMSE=9.9, RMSE_users=9.9, CR=0.0)
    errs = np.abs(actuals[made_mask] - preds[made_mask])
    sq = (actuals[made_mask] - preds[made_mask]) ** 2
    mae_d = float(errs.mean())
    rmse = float(np.sqrt(sq.mean()))
    pu_mae = defaultdict(list)
    pu_sq = defaultdict(list)
    for i, m in enumerate(made_mask):
        if m:
            pu_mae[uids[i]].append(abs(actuals[i] - preds[i]))
            pu_sq[uids[i]].append((actuals[i] - preds[i]) ** 2)
    mae_u = float(np.mean([np.mean(v) for v in pu_mae.values()]))
    rmse_u = float(np.mean([math.sqrt(np.mean(v)) for v in pu_sq.values()]))
    return dict(MAE_data=mae_d, MAE_users=mae_u, RMSE=rmse, RMSE_users=rmse_u, CR=cr)


def main():
    t0 = time.time()
    print("=== GIM postprocess + positive-only sweep ===")

    df, item_genres, users_df = load_and_filter(RATINGS_PATH, ITEMS_PATH, USERS_PATH)
    print(f"Dataset: {df['user_id'].nunique()} users x "
          f"{df['item_id'].nunique()} items x {len(df)} ratings")

    triples = [(int(a), int(b), float(c)) for a, b, c in
               df[["user_id", "item_id", "rating"]].values.tolist()]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_rows = []
    configs = [("positive_only_True", True), ("positive_only_False", False)]

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(triples)):
        train_list = [triples[i] for i in train_idx]
        test_list  = [triples[i] for i in test_idx]
        train_df = pd.DataFrame(train_list, columns=["user_id", "item_id", "rating"])
        test_triples = test_list

        for variant_label, pos_only in configs:
            gim = FuzzyGeneticMethod(run_ga=False, positive_only=pos_only)
            gim.fit(train_df, item_genres, users_df)
            uids = [t[0] for t in test_triples]
            iids = [t[1] for t in test_triples]
            for k in K_VALUES:
                preds = gim.predict_batch(uids, iids, k)
                # raw
                m = fold_metrics(test_triples, preds)
                all_rows.append(dict(variant=variant_label,
                                     postproc="raw", fold=fold_idx, k=k, **m))
                # floor for data
                preds_floor = np.where(np.isnan(preds), preds,
                                        np.clip(np.floor(preds), 1.0, 5.0))
                m_floor = fold_metrics(test_triples, preds_floor)
                all_rows.append(dict(variant=variant_label,
                                     postproc="floor", fold=fold_idx, k=k, **m_floor))
                # round
                preds_round = np.where(np.isnan(preds), preds,
                                        np.clip(np.rint(preds), 1.0, 5.0))
                m_round = fold_metrics(test_triples, preds_round)
                all_rows.append(dict(variant=variant_label,
                                     postproc="round", fold=fold_idx, k=k, **m_round))

        print(f"  fold {fold_idx+1}/10 done ({(time.time()-t0)/60:.1f} min)")

    out = os.path.join(RESULTS_DIR, "repro_gim_postprocess.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nWrote {out} ({len(all_rows)} rows)")

    df_r = pd.DataFrame(all_rows)
    print("\n=== Summary at k=10 (mean over 10 folds) ===")
    print("Paper GIM target: MAE_users ~0.85, CR ~0.92")
    grp = df_r[df_r.k == 10].groupby(["variant", "postproc"]).mean(numeric_only=True)
    print(grp[["MAE_users", "RMSE", "CR"]].round(4))


if __name__ == "__main__":
    main()
