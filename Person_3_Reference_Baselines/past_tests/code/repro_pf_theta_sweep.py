"""
repro_pf_theta_sweep.py

Hypothesis test for Person 3 PF baseline. Tests:

  H_PF_1: theta sweep theta in {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60}.
          Hao 2016 sec 4.3 explicitly sweeps theta in [0.3, 0.9] and reports
          good CR around 0.4. Person 3 uses 0.4 as default. We confirm.
  H_PF_2: positive_scores = (4, 5) instead of (3, 4, 5). Hao text says
          "high score" maps to {3,4,5} but their figures use a strict-positive
          variant in some cases.
  H_PF_3: floor/round post-processing on predictions, paralleling the FUS
          replication finding.

Output: repro_pf_theta_sweep.csv

Sources:
  Hao 2016 sec 4.3 + Eqs 4-7
  D'Aniello 2026 Fig 6/7 (PF reference curves)
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
from pf import ProbabilisticFiltering, load_and_filter

DATA_DIR = os.path.join(_HERE, "..", "..", "..", "ml-100k")
RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RATINGS_PATH = os.path.join(DATA_DIR, "u.data")
ITEMS_PATH = os.path.join(DATA_DIR, "u.item")

K_VALUES = [1, 2, 4, 6, 8, 10]


def fold_metrics(test_triples, preds):
    """Compute MAE_data, MAE_users, RMSE, RMSE_users, CR for one fold-k."""
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
    print("=== PF theta sweep + floor/round ===")

    df, item_topics = load_and_filter(RATINGS_PATH, ITEMS_PATH)
    print(f"Dataset: {df['user_id'].nunique()} users x "
          f"{df['item_id'].nunique()} items x {len(df)} ratings")

    triples = df[["user_id", "item_id", "rating"]].values.tolist()
    triples = [(int(a), int(b), float(c)) for a, b, c in triples]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_rows = []
    configs = []
    # theta sweep with default positive_scores
    for theta in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        configs.append(("default_pos_scores", theta, (3, 4, 5)))
    # positive_scores variant
    for theta in [0.30, 0.40, 0.50]:
        configs.append(("strict_pos_scores", theta, (4, 5)))

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(triples)):
        train_list = [triples[i] for i in train_idx]
        test_list  = [triples[i] for i in test_idx]
        train_df = pd.DataFrame(train_list, columns=["user_id", "item_id", "rating"])
        test_triples = test_list

        for variant_label, theta, pos_scores in configs:
            pf = ProbabilisticFiltering(theta=theta, positive_scores=pos_scores)
            pf.fit(train_df, item_topics)
            uids = [t[0] for t in test_triples]
            iids = [t[1] for t in test_triples]
            for k in K_VALUES:
                preds = pf.predict_batch(uids, iids, k)
                # raw
                m = fold_metrics(test_triples, preds)
                row = dict(variant=f"{variant_label}_theta{theta}",
                           postproc="raw", fold=fold_idx, k=k, **m)
                all_rows.append(row)
                # floor for data
                preds_floor = np.where(np.isnan(preds), preds, np.clip(np.floor(preds), 1.0, 5.0))
                m_floor = fold_metrics(test_triples, preds_floor)
                row = dict(variant=f"{variant_label}_theta{theta}",
                           postproc="floor", fold=fold_idx, k=k, **m_floor)
                all_rows.append(row)
                # round
                preds_round = np.where(np.isnan(preds), preds, np.clip(np.rint(preds), 1.0, 5.0))
                m_round = fold_metrics(test_triples, preds_round)
                row = dict(variant=f"{variant_label}_theta{theta}",
                           postproc="round", fold=fold_idx, k=k, **m_round)
                all_rows.append(row)

        print(f"  fold {fold_idx+1}/10 done ({(time.time()-t0)/60:.1f} min)")

    out = os.path.join(RESULTS_DIR, "repro_pf_theta_sweep.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nWrote {out} ({len(all_rows)} rows)")

    df_r = pd.DataFrame(all_rows)
    print("\n=== Summary at k=10 (mean over 10 folds) ===")
    print("Paper PF target: MAE_users ~0.88, CR ~0.92")
    grp = df_r[df_r.k == 10].groupby(["variant", "postproc"]).mean(numeric_only=True)
    print(grp[["MAE_users", "RMSE", "CR"]].round(4))


if __name__ == "__main__":
    main()
