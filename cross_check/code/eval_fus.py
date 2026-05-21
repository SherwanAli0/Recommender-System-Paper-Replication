"""
eval_fus.py: Stream cross_check (faithful FUS only, no novelty)

This is the paper-only Protocol A evaluation runner. Runs the faithful FUS
implementation across the standard k sweep at alpha = 0 with 10-fold CV
on the same MovieLens-100k filtered dataset described in shared_contract.md.

Novelty extensions (cold-start variants and alternative protocols) are not
included in this implementation release.

Sources:
  paper Eqs 1-19, 20-24
  fus.py for the algorithm
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import fus

DATA_DIR = os.path.join(_HERE, "..", "..", "..", "ml-100k")
RESULTS_DIR = os.path.join(_HERE, "..", "results", "protocol_A")
os.makedirs(RESULTS_DIR, exist_ok=True)
RATINGS_PATH = os.path.join(DATA_DIR, "u.data")

# Paper §V.C.2: "k = {1, 2, 4, 6, 8, 10, ..., 48, 50}" -- 26 values.
K_VALUES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
            30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
ALPHAS = [0.0, 0.7, 0.8, 0.9]                    # paper §V.C.2


def precompute_sorted_nbrs(sim_mat):
    n = sim_mat.shape[0]
    out = []
    for i in range(n):
        row = sim_mat[i].copy()
        row[i] = -1.0
        out.append(np.argsort(row)[::-1])
    return out


def build_walks(test_triples, user_to_idx, sorted_nbrs, item_raters_dict,
                sim_mat, k_max):
    walks = {}
    for uid, item_id, _ in test_triples:
        if uid not in user_to_idx:
            continue
        u_idx = user_to_idx[uid]
        key = (u_idx, item_id)
        if key in walks:
            continue
        raters = item_raters_dict.get(item_id, {})
        w = []
        for rank, j in enumerate(sorted_nbrs[u_idx][:k_max]):
            jj = int(j)
            if jj in raters:
                w.append((rank, jj, float(sim_mat[u_idx, jj]), raters[jj]))
        walks[key] = w
    return walks


def resnick(walk, k, u_mean_i, user_means):
    # Audit (May 2026): no clip. Paper Eq 19 is real-valued.
    # shared_contract.md §4.1 was updated to remove the clip mandate.
    # This is the production prediction function called by the
    # cross_check evaluator (eval_fus.main below). The other
    # resnick_predict in fus.py is the standalone reference version.
    top = [(j, s, r) for rank, j, s, r in walk if rank < k]
    if not top:
        return None
    denom = sum(s for _, s, _ in top)
    if denom == 0.0:
        return None
    numer = sum((r - user_means[j]) * s for j, s, r in top)
    return float(u_mean_i + numer / denom)


def metrics(test_triples, preds_dict, user_to_idx):
    """preds_dict: dict {(u_idx, item_id): pred_or_None}, indexed test pairs."""
    data_errs = []
    user_errs = defaultdict(list)
    user_sq = defaultdict(list)
    n_pred = n_total = 0
    for uid, item_id, r_actual in test_triples:
        if uid not in user_to_idx:
            continue
        u_idx = user_to_idx[uid]
        n_total += 1
        pred = preds_dict.get((u_idx, item_id))
        if pred is None:
            continue
        n_pred += 1
        e = pred - r_actual
        data_errs.append(e)
        user_errs[uid].append(abs(e))
        user_sq[uid].append(e * e)
    cr = n_pred / n_total if n_total > 0 else 0.0
    if not data_errs:
        return dict(MAE_data=9.9, MAE_users=9.9, RMSE=9.9, RMSE_users=9.9, CR=cr)
    mae_d = float(np.mean([abs(e) for e in data_errs]))
    rmse = float(np.sqrt(np.mean([e * e for e in data_errs])))
    mae_u = float(np.mean([np.mean(v) for v in user_errs.values()]))
    rmse_u = float(np.mean([np.sqrt(np.mean(v)) for v in user_sq.values()]))
    return dict(MAE_data=mae_d, MAE_users=mae_u, RMSE=rmse, RMSE_users=rmse_u, CR=cr)


def main():
    t0 = time.time()
    print("=== Faithful FUS Protocol A evaluation ===")

    ratings = fus.filter_dataset(fus.load_ratings(RATINGS_PATH))
    print(f"Dataset: {ratings['user_id'].nunique()} users x "
          f"{ratings['item_id'].nunique()} items x {len(ratings)} ratings")

    users = sorted(ratings["user_id"].unique().tolist())
    items = sorted(ratings["item_id"].unique().tolist())
    user_to_idx = {uid: i for i, uid in enumerate(users)}
    item_to_idx = {iid: i for i, iid in enumerate(items)}
    n_items = len(items)

    triples = [(int(a), int(b), float(c)) for a, b, c in
               ratings[["user_id", "item_id", "rating"]].values.tolist()]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(triples)):
        train_list = [triples[i] for i in train_idx]
        test_list  = [triples[i] for i in test_idx]
        train_df = pd.DataFrame(train_list, columns=["user_id", "item_id", "rating"])
        test_triples = test_list

        # Per-user matrices (built once, reused across alphas)
        per_user_mat = []
        for uid in users:
            u_train = train_df[train_df["user_id"] == uid]
            mat = fus.build_user_matrix(u_train, item_to_idx, n_items)
            per_user_mat.append(mat)

        item_raters_dict = fus.build_item_raters_dict(train_df, user_to_idx)
        user_means = fus.compute_user_means(train_df, users, user_to_idx)

        for alpha in ALPHAS:
            user_sigs = [fus.user_signature(m) for m in per_user_mat]
            sim_mat = fus.compute_sim_matrix(user_sigs, alpha)
            sorted_nbrs = precompute_sorted_nbrs(sim_mat)
            walks = build_walks(test_triples, user_to_idx, sorted_nbrs,
                                item_raters_dict, sim_mat, max(K_VALUES))

            for k in K_VALUES:
                preds = {}
                for uid, item_id, _ in test_triples:
                    if uid not in user_to_idx:
                        continue
                    u_idx = user_to_idx[uid]
                    walk = walks.get((u_idx, item_id), [])
                    pred = resnick(walk, k, float(user_means[u_idx]), user_means)
                    preds[(u_idx, item_id)] = pred
                m = metrics(test_triples, preds, user_to_idx)
                rows.append(dict(system="FUS", protocol="A_warm",
                                 fold=fold_idx, k=k, alpha=alpha, **m))

        print(f"  fold {fold_idx + 1}/10 done ({(time.time() - t0) / 60:.1f} min)")

    out = os.path.join(RESULTS_DIR, "results_FUS_A_warm.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
