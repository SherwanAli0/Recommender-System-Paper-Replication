"""
eval_warm_paper_curve_approx.py - Person 2 / Stream P2

Protocol A paper-curve approximation for pure FUS only.

This script intentionally does NOT replace eval_warm.py.  It keeps the
paper-faithful top-k-neighbor prediction availability, then applies the
best reverse-engineered post-processing found in REPRO_DEBUG_NOTES.md:

  - dataset-level MAE/RMSE are computed from floor(prediction)
  - user-level MAE/RMSE are computed from round(prediction)
  - CR is computed from the original prediction availability

This is a diagnostic/replication script for explaining the article figures,
not the clean novelty baseline.
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
import fus


_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "ml-100k")
RESULTS_DIR = os.path.join(_HERE, "..", "results")
RATINGS_PATH = os.path.join(DATA_DIR, "u.data")

os.makedirs(RESULTS_DIR, exist_ok=True)

K_VALUES = [1, 2, 4, 6, 8, 10, 15, 20, 30, 50]
ALPHA_VALUES = [0.0, 0.7, 0.8, 0.9]


def _precompute_sorted_nbrs(sim_mat):
    """Source: numpy.argsort https://numpy.org/doc/stable/reference/generated/numpy.argsort.html"""
    sorted_nbrs = []
    for i in range(sim_mat.shape[0]):
        row = sim_mat[i].copy()
        row[i] = -1.0
        sorted_nbrs.append(np.argsort(row)[::-1])
    return sorted_nbrs


def _build_fus_walks(test_triples, user_to_idx, sorted_nbrs, item_raters_dict, sim_mat, k_max):
    """Source: paper Section IV.B Eq. 19, top-k neighborhood definition."""
    walks = {}
    for uid, item_id, _ in test_triples:
        u_idx = user_to_idx[uid]
        key = (u_idx, item_id)
        if key in walks:
            continue

        raters = item_raters_dict.get(item_id, {})
        walk = []
        for rank, j in enumerate(sorted_nbrs[u_idx][:k_max]):
            j_idx = int(j)
            if j_idx in raters:
                walk.append((rank, j_idx, float(sim_mat[u_idx, j_idx]), raters[j_idx]))
        walks[key] = walk
    return walks


def _resnick_walk(walk, k, u_mean_i, user_means):
    """Source: paper Section IV.B Eq. 19."""
    top = [(j, s, r) for rank, j, s, r in walk if rank < k]
    if not top:
        return np.nan
    denom = sum(s for _, s, _ in top)
    if denom == 0.0:
        return np.nan
    numer = sum((r - user_means[j]) * s for j, s, r in top)
    return float(np.clip(u_mean_i + numer / denom, 1.0, 5.0))


def compute_paper_curve_metrics(data_errors, user_errors, n_predicted, n_total):
    """
    Mixed metric convention discovered by reverse engineering:
      data_errors: errors from floor(prediction)
      user_errors: errors from round(prediction)

    Source formulas: paper Section V.B Eqs. 20-24.
    """
    if not data_errors:
        return 9.9, 9.9, 9.9, 9.9, 0.0

    data_abs = [abs(e) for e in data_errors]
    data_sq = [e * e for e in data_errors]
    mae_data = float(np.mean(data_abs))
    rmse = float(np.sqrt(np.mean(data_sq)))

    per_user_mae = []
    per_user_rmse = []
    for errs in user_errors.values():
        if errs:
            per_user_mae.append(float(np.mean([abs(e) for e in errs])))
            per_user_rmse.append(float(np.sqrt(np.mean([e * e for e in errs]))))

    mae_users = float(np.mean(per_user_mae)) if per_user_mae else 9.9
    rmse_users = float(np.mean(per_user_rmse)) if per_user_rmse else 9.9
    cr = float(n_predicted / n_total) if n_total > 0 else 0.0
    return mae_data, mae_users, rmse, rmse_users, cr


def evaluate_fold_alpha(fold_idx, alpha, test_triples, sim_mat, item_raters_dict,
                        user_means, user_to_idx, k_values):
    k_max = max(k_values)
    print("      precomputing sorted_nbrs ...", end=" ", flush=True)
    sorted_nbrs = _precompute_sorted_nbrs(sim_mat)
    print("done")

    print("      building walks ...", end=" ", flush=True)
    walks = _build_fus_walks(test_triples, user_to_idx, sorted_nbrs,
                             item_raters_dict, sim_mat, k_max)
    print("done")

    rows = []
    for k in k_values:
        data_errors = []
        user_errors = defaultdict(list)
        n_pred = 0
        n_total = 0

        for uid, item_id, r_actual in test_triples:
            u_idx = user_to_idx[uid]
            pred = _resnick_walk(walks[(u_idx, item_id)], k, float(user_means[u_idx]), user_means)

            n_total += 1
            if pred is None or (isinstance(pred, float) and math.isnan(pred)):
                continue

            pred_floor = float(np.clip(np.floor(pred), 1.0, 5.0))
            pred_round = float(np.clip(np.rint(pred), 1.0, 5.0))

            data_errors.append(pred_floor - r_actual)
            user_errors[uid].append(pred_round - r_actual)
            n_pred += 1

        mae_d, mae_u, rmse, rmse_u, cr = compute_paper_curve_metrics(
            data_errors, user_errors, n_pred, n_total
        )
        rows.append({
            "system": "FUS_paper_curve_approx",
            "protocol": "A_warm_paper_curve_approx",
            "fold": fold_idx,
            "k": k,
            "alpha": alpha,
            "MAE_data": round(mae_d, 4),
            "MAE_users": round(mae_u, 4),
            "RMSE": round(rmse, 4),
            "RMSE_users": round(rmse_u, 4),
            "CR": round(cr, 4),
        })

    return rows


def main():
    t0 = time.time()
    print("=== Protocol A - FUS Paper-Curve Approximation ===")

    ratings = fus.filter_dataset(fus.load_ratings(RATINGS_PATH))
    print(f"Filtered: {ratings['user_id'].nunique()} users x "
          f"{ratings['item_id'].nunique()} items x {len(ratings)} ratings")

    users = sorted(ratings["user_id"].unique().tolist())
    items = sorted(ratings["item_id"].unique().tolist())
    user_to_idx = {uid: i for i, uid in enumerate(users)}
    item_to_idx = {iid: i for i, iid in enumerate(items)}
    n_items = len(items)

    triples = ratings[["user_id", "item_id", "rating"]].values.tolist()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(triples)):
        print(f"\n-- Fold {fold_idx + 1}/10 --")
        train_list = [triples[i] for i in train_idx]
        test_list = [triples[i] for i in test_idx]
        train_df = pd.DataFrame(train_list, columns=["user_id", "item_id", "rating"])

        print("  Building FUS signatures ...")
        user_sigs = []
        for uid in users:
            u_train = train_df[train_df["user_id"] == uid]
            mat = fus.build_user_matrix(u_train, item_to_idx, n_items)
            user_sigs.append(fus.user_signature(mat))

        item_raters_dict = fus.build_item_raters_dict(train_df, user_to_idx)
        user_means = fus.compute_user_means(train_df, users, user_to_idx)
        test_triples = [(int(r[0]), int(r[1]), float(r[2])) for r in test_list]

        for alpha in ALPHA_VALUES:
            print(f"    alpha={alpha} - computing FUS sim matrix ...", end=" ", flush=True)
            sim_mat = fus.compute_sim_matrix(user_sigs, alpha)
            print("done")
            all_rows.extend(evaluate_fold_alpha(
                fold_idx, alpha, test_triples, sim_mat, item_raters_dict,
                user_means, user_to_idx, K_VALUES,
            ))

        print(f"  Fold {fold_idx + 1} done. Elapsed: {(time.time() - t0) / 60:.1f} min")

    results_df = pd.DataFrame(all_rows)
    col_order = ["system", "protocol", "fold", "k", "alpha",
                 "MAE_data", "MAE_users", "RMSE", "RMSE_users", "CR"]
    results_df = results_df[col_order]

    out_path = os.path.join(RESULTS_DIR, "results_FUS_A_warm_paper_curve_approx.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(results_df)} rows)")

    rows = results_df[(results_df["alpha"] == 0.0) & (results_df["k"].isin([1, 10, 50]))]
    print("\n=== Paper-Curve Approx alpha=0 summary ===")
    print(rows.groupby("k")[["MAE_data", "MAE_users", "RMSE", "RMSE_users", "CR"]].mean())


if __name__ == "__main__":
    main()
