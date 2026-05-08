"""
repro_den_all_floor.py
Test hypothesis: paper Eq.19 denominator sums ALL k neighbors (not just raters)
combined with floor/round post-processing that was found as the best approximation.

Paper text says: "if a user uj has not voted for item m, the whole term of the
summation is set to zero." The denominator Sigma sim over N^k_ui is NOT similarly
restricted, so den_all is more faithful to the literal formula.

Compares den_raters vs den_all, each with raw, floor-data, and round-both
post-processing, at alpha=0.

Sources:
  paper Eq. 19: D'Aniello et al. IEEE Access 2026, p.9980
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


def _precompute_sorted_nbrs(sim_mat):
    n = sim_mat.shape[0]
    sorted_nbrs = []
    for i in range(n):
        row = sim_mat[i].copy()
        row[i] = -1.0
        sorted_nbrs.append(np.argsort(row)[::-1])
    return sorted_nbrs


def _build_full_walks(test_triples, user_to_idx, sorted_nbrs, item_raters_dict,
                      sim_mat, k_max):
    """
    Walk stores ALL k_max neighbors (raters AND non-raters).
    Each entry: (rank, j_idx, sim, rating, is_rater).
    Non-raters have rating=0.0, is_rater=False.
    """
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
            sim = float(sim_mat[u_idx, j_idx])
            if j_idx in raters:
                walk.append((rank, j_idx, sim, raters[j_idx], True))
            else:
                walk.append((rank, j_idx, sim, 0.0, False))
        walks[key] = walk
    return walks


def _pred_den_raters(walk, k, u_mean_i, user_means):
    """Resnick with denominator over RATERS only (our current faithful impl)."""
    top_raters = [
        (j, s, r)
        for rank, j, s, r, is_r in walk
        if rank < k and is_r
    ]
    if not top_raters:
        return np.nan
    denom = sum(s for _, s, _ in top_raters)
    if denom == 0.0:
        return np.nan
    numer = sum((r - user_means[j]) * s for j, s, r in top_raters)
    return float(np.clip(u_mean_i + numer / denom, 1.0, 5.0))


def _pred_den_all(walk, k, u_mean_i, user_means):
    """
    Resnick with denominator over ALL k neighbors (paper Eq.19 literal reading).
    Numerator: zero for non-raters (paper: 'the whole term is set to zero').
    Denominator: all k neighbors' sim values (no exception stated in paper).
    """
    top = [(j, s, r, is_r) for rank, j, s, r, is_r in walk if rank < k]
    if not top:
        return np.nan
    denom = sum(s for _, s, _, _ in top)
    if denom == 0.0:
        return np.nan
    numer = sum((r - user_means[j]) * s for j, s, r, is_r in top if is_r)
    return float(np.clip(u_mean_i + numer / denom, 1.0, 5.0))


def _metrics(errors_data, errors_by_user, n_pred, n_total):
    if not errors_data:
        return 9.9, 9.9, 9.9, 0.0
    mae_data = float(np.mean([abs(e) for e in errors_data]))
    rmse = float(np.sqrt(np.mean([e * e for e in errors_data])))
    per_u = [np.mean([abs(e) for e in v]) for v in errors_by_user.values() if v]
    mae_users = float(np.mean(per_u)) if per_u else 9.9
    cr = float(n_pred / n_total)
    return mae_data, mae_users, rmse, cr


def evaluate_fold(fold_idx, test_triples, sim_mat, item_raters_dict,
                  user_means, user_to_idx, k_values):
    k_max = max(k_values)
    sorted_nbrs = _precompute_sorted_nbrs(sim_mat)
    walks = _build_full_walks(
        test_triples, user_to_idx, sorted_nbrs,
        item_raters_dict, sim_mat, k_max
    )

    rows = []
    configs = [
        ("den_raters_raw",   _pred_den_raters, False, False),
        ("den_raters_floor", _pred_den_raters, True,  False),
        ("den_raters_round", _pred_den_raters, True,  True),
        ("den_all_raw",      _pred_den_all,    False, False),
        ("den_all_floor",    _pred_den_all,    True,  False),
        ("den_all_round",    _pred_den_all,    True,  True),
    ]

    for variant, pred_fn, use_floor_data, use_round_user in configs:
        for k in k_values:
            data_errs = []
            user_errs = defaultdict(list)
            n_pred = 0
            n_total = 0

            for uid, item_id, r_actual in test_triples:
                u_idx = user_to_idx[uid]
                pred = pred_fn(walks[(u_idx, item_id)], k,
                               float(user_means[u_idx]), user_means)
                n_total += 1
                if pred is None or (isinstance(pred, float) and math.isnan(pred)):
                    continue

                if use_floor_data:
                    pred_d = float(np.clip(np.floor(pred), 1.0, 5.0))
                else:
                    pred_d = pred

                if use_round_user:
                    pred_u = float(np.clip(np.rint(pred), 1.0, 5.0))
                else:
                    pred_u = pred

                data_errs.append(pred_d - r_actual)
                user_errs[uid].append(pred_u - r_actual)
                n_pred += 1

            mae_d, mae_u, rmse, cr = _metrics(data_errs, user_errs, n_pred, n_total)
            rows.append({
                "variant": variant, "fold": fold_idx, "k": k,
                "MAE_data": round(mae_d, 4),
                "MAE_users": round(mae_u, 4),
                "RMSE": round(rmse, 4),
                "CR": round(cr, 4),
            })
    return rows


def main():
    t0 = time.time()
    print("=== repro_den_all_floor: testing den_all + floor/round ===")

    ratings = fus.filter_dataset(fus.load_ratings(RATINGS_PATH))
    print(f"Dataset: {ratings['user_id'].nunique()} users x "
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
        print(f"Fold {fold_idx + 1}/10 ...", end=" ", flush=True)
        train_list = [triples[i] for i in train_idx]
        test_list = [triples[i] for i in test_idx]
        train_df = pd.DataFrame(train_list, columns=["user_id", "item_id", "rating"])

        user_sigs = []
        for uid in users:
            u_train = train_df[train_df["user_id"] == uid]
            mat = fus.build_user_matrix(u_train, item_to_idx, n_items)
            user_sigs.append(fus.user_signature(mat))

        item_raters_dict = fus.build_item_raters_dict(train_df, user_to_idx)
        user_means = fus.compute_user_means(train_df, users, user_to_idx)

        sim_mat = fus.compute_sim_matrix(user_sigs, 0.0)
        test_triples = [(int(r[0]), int(r[1]), float(r[2])) for r in test_list]

        all_rows.extend(evaluate_fold(
            fold_idx, test_triples, sim_mat,
            item_raters_dict, user_means, user_to_idx, K_VALUES
        ))
        print(f"done ({(time.time()-t0)/60:.1f} min elapsed)")

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "repro_den_all_floor.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows)")

    print("\n=== Summary (averaged over 10 folds, alpha=0) ===")
    print(f"Paper targets: k=1  -> MAEdata~0.985, MAEusers~0.857, RMSE~1.316")
    print(f"               k=10 -> MAEdata~0.870, MAEusers~0.730, RMSE~1.200")
    print(f"               k=50 -> MAEdata~0.830, MAEusers~0.703, RMSE~1.107")
    print()
    for v in ["den_raters_raw", "den_raters_floor", "den_raters_round",
              "den_all_raw", "den_all_floor", "den_all_round"]:
        sub = df[df["variant"] == v]
        grp = sub.groupby("k")[["MAE_data", "MAE_users", "RMSE", "CR"]].mean()
        print(f"--- {v} ---")
        for k in [1, 10, 50]:
            if k in grp.index:
                row = grp.loc[k]
                print(f"  k={k:2d}: MAEdata={row['MAE_data']:.4f}  "
                      f"MAEusers={row['MAE_users']:.4f}  "
                      f"RMSE={row['RMSE']:.4f}  CR={row['CR']:.4f}")
        print()


if __name__ == "__main__":
    main()
