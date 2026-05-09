"""
eval.py - the core implementation / Stream core (Faithful Baseline)
Protocol A: warm-user 10-fold CV for FUS and CF.

FUS results reused from the cross-check implementation (identical algorithm, same dataset, same folds).
This script runs only the CF sweep (unique to the core implementation) then generates the
4 paper-style plots (Figs 2-5).

Sources (NO-ORIGINAL-CODE RULE):
  paper §V       : evaluation methodology, metrics Eqs 20-24
  shared_contract: §3 (KFold), §4 (metrics), §5 (CSV), §6 (sweeps)
  sklearn KFold  : https://scikit-learn.org/stable/modules/generated/
                     sklearn.model_selection.KFold.html
  pandas to_csv  : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
  shutil.copy2   : https://docs.python.org/3/library/shutil.html#shutil.copy2
  matplotlib bar : https://matplotlib.org/stable/gallery/statistics/errorbar_features.html
  matplotlib line: https://matplotlib.org/stable/tutorials/introductory/pyplot.html
"""

import os
import sys
import math
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
import fus
import cf as cf_mod

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
# ml-100k lives at the repo root, three levels up from this file
# (Implementation/core/code/ → ../../../ml-100k)
DATA_DIR     = os.path.join(_HERE, "..", "..", "..", "ml-100k")
RESULTS_DIR  = os.path.join(_HERE, "..", "results")
FIGS_DIR     = os.path.join(RESULTS_DIR, "figs")
RATINGS_PATH = os.path.join(DATA_DIR, "u.data")

# the cross-check implementation FUS CSV - reused instead of rerunning the identical FUS algorithm
# (the cross-check implementation writes to cross_check/results/protocol_A/results_FUS_A_warm.csv
# under the same repo)
P2_FUS_CSV = os.path.join(_HERE, "..", "..", "cross_check",
                           "results", "protocol_A", "results_FUS_A_warm.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# SWEEP PARAMETERS - shared_contract.md §6
# -----------------------------------------------------------------------------
K_VALUES     = [1, 2, 4, 6, 8, 10, 15, 20, 30, 50]
ALPHA_VALUES = [0.0, 0.7, 0.8, 0.9]
ALPHA_COLORS = {0.0: "#1f77b4", 0.7: "#ff7f0e", 0.8: "#2ca02c", 0.9: "#d62728"}


# -----------------------------------------------------------------------------
# METRICS - paper §V.B Eqs 20-24
# Source: paper Eqs 20-24; shared_contract.md §4
# -----------------------------------------------------------------------------
def compute_metrics(errors_all, user_errors, n_predicted, n_total):
    if len(errors_all) == 0:
        return 9.9, 9.9, 9.9, 9.9, 0.0
    abs_e = [abs(e) for e in errors_all]
    sq_e  = [e * e  for e in errors_all]
    mae_data = float(np.mean(abs_e))
    rmse     = float(np.sqrt(np.mean(sq_e)))
    pu_mae, pu_rmse = [], []
    for uid, errs in user_errors.items():
        if errs:
            pu_mae.append(float(np.mean([abs(e) for e in errs])))
            pu_rmse.append(float(np.sqrt(np.mean([e*e for e in errs]))))
    mae_users  = float(np.mean(pu_mae))  if pu_mae  else 9.9
    rmse_users = float(np.mean(pu_rmse)) if pu_rmse else 9.9
    cr         = float(n_predicted / n_total) if n_total > 0 else 0.0
    return mae_data, mae_users, rmse, rmse_users, cr


# -----------------------------------------------------------------------------
# EVALUATION CORE - one sim_mat x all k values
#
# Rank-aware walk approach (correct top-k-from-all-users):
#   Select top-k neighbors by sim from ALL users, then filter to those who
#   rated the target item. This gives CR~60% at k=1 (matches paper).
#   Selecting from raters-only gives CR=100% (wrong approach).
# Source: fus.precompute_sorted_nbrs, fus.build_walks, fus.resnick_from_walk
#         paper §IV.B Eq 19.
# -----------------------------------------------------------------------------
def evaluate_sim_matrix(system_name, fold_idx, alpha, sim_mat, test_triples,
                        item_raters_dict, user_means, user_to_idx, k_values,
                        protocol="A_warm"):
    k_max = max(k_values)
    sorted_nbrs = fus.precompute_sorted_nbrs(sim_mat)
    walks = fus.build_walks(test_triples, user_to_idx, sorted_nbrs,
                            item_raters_dict, sim_mat, k_max)
    rows = []
    for k in k_values:
        acc = {"errs": [], "user_errs": defaultdict(list),
               "n_pred": 0, "n_total": 0}
        for uid, item_id, r_actual in test_triples:
            u_idx  = user_to_idx[uid]
            u_mean = float(user_means[u_idx])
            pred   = fus.resnick_from_walk(
                walks[(u_idx, item_id)], k, u_mean, user_means)
            acc["n_total"] += 1
            if pred is not None and not (isinstance(pred, float)
                                         and math.isnan(pred)):
                err = pred - r_actual
                acc["errs"].append(err)
                acc["user_errs"][uid].append(err)
                acc["n_pred"] += 1
        mae_d, mae_u, rmse, rmse_u, cr = compute_metrics(
            acc["errs"], acc["user_errs"], acc["n_pred"], acc["n_total"])
        rows.append({
            "system": system_name, "protocol": protocol,
            "fold": fold_idx, "k": k, "alpha": alpha,
            "MAE_data":   round(mae_d,  4),
            "MAE_users":  round(mae_u,  4),
            "RMSE":       round(rmse,   4),
            "RMSE_users": round(rmse_u, 4),
            "CR":         round(cr,     4),
        })
    return rows


# -----------------------------------------------------------------------------
# PLOTS - paper Figs 2-5
# Source: matplotlib errorbar https://matplotlib.org/stable/gallery/statistics/errorbar_features.html
#         matplotlib pyplot   https://matplotlib.org/stable/tutorials/introductory/pyplot.html
# -----------------------------------------------------------------------------
def plot_fig2(fus_df, out_path):
    """Fig 2: MAE_data +/- std across folds for k in {2,4,6,8,10}, alpha=0."""
    sub = fus_df[(fus_df["alpha"] == 0.0) & fus_df["k"].isin([2, 4, 6, 8, 10])]
    grp = (sub.groupby("k")["MAE_data"]
           .agg(["mean", "std"]).reset_index().sort_values("k"))
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len(grp))
    ax.bar(xs, grp["mean"], yerr=grp["std"], capsize=5,
           color="#1f77b4", alpha=0.8, error_kw={"elinewidth": 1.5})
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["k=" + str(k) for k in grp["k"]])
    ax.set_ylabel("MAE_data")
    ax.set_title("Fig 2 - MAE per fold (FUS, alpha=0)")
    ax.set_ylim(0, (grp["mean"] + grp["std"]).max() * 1.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved " + out_path)


def plot_fig3(fus_df, out_path):
    """Fig 3: MAE_data and MAE_users vs k, all alpha, averaged over folds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha in ALPHA_VALUES:
        sub = (fus_df[fus_df["alpha"] == alpha]
               .groupby("k")[["MAE_data", "MAE_users"]].mean()
               .reset_index().sort_values("k"))
        c = ALPHA_COLORS[alpha]
        lbl = "alpha=" + str(alpha)
        axes[0].plot(sub["k"], sub["MAE_data"],  marker="o", color=c, label=lbl)
        axes[1].plot(sub["k"], sub["MAE_users"], marker="s", color=c, label=lbl)
    for ax, title in zip(axes, ["MAE_data", "MAE_users"]):
        ax.set_xlabel("k")
        ax.set_ylabel(title)
        ax.set_title("Fig 3 - " + title + " vs k (FUS)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved " + out_path)


def plot_fig4(fus_df, out_path):
    """Fig 4: RMSE and RMSE_users vs k, all alpha, averaged over folds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha in ALPHA_VALUES:
        sub = (fus_df[fus_df["alpha"] == alpha]
               .groupby("k")[["RMSE", "RMSE_users"]].mean()
               .reset_index().sort_values("k"))
        c = ALPHA_COLORS[alpha]
        lbl = "alpha=" + str(alpha)
        axes[0].plot(sub["k"], sub["RMSE"],        marker="o", color=c, label=lbl)
        axes[1].plot(sub["k"], sub["RMSE_users"],  marker="s", color=c, label=lbl)
    for ax, title in zip(axes, ["RMSE", "RMSE_users"]):
        ax.set_xlabel("k")
        ax.set_ylabel(title)
        ax.set_title("Fig 4 - " + title + " vs k (FUS)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved " + out_path)


def plot_fig5(fus_df, out_path):
    """Fig 5: CR vs k, all alpha, averaged over folds."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for alpha in ALPHA_VALUES:
        sub = (fus_df[fus_df["alpha"] == alpha]
               .groupby("k")["CR"].mean()
               .reset_index().sort_values("k"))
        ax.plot(sub["k"], sub["CR"], marker="o",
                color=ALPHA_COLORS[alpha], label="alpha=" + str(alpha))
    ax.set_xlabel("k")
    ax.set_ylabel("CR (Coverage Rate)")
    ax.set_title("Fig 5 - CR vs k (FUS)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved " + out_path)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=== Protocol A -- the core implementation Faithful Baseline ===")

    # ------------------------------------------------------------------
    # Step 1: Reuse the cross-check implementation FUS CSV (same algorithm, no need to rerun)
    # Source: shutil.copy2 https://docs.python.org/3/library/shutil.html
    # ------------------------------------------------------------------
    p1_fus_csv = os.path.join(RESULTS_DIR, "results_FUS_A_warm.csv")
    if not os.path.exists(P2_FUS_CSV):
        print("ERROR: the cross-check implementation FUS CSV not found at:")
        print("  " + P2_FUS_CSV)
        print("Run cross_check/code/eval_fus.py first.")
        return
    shutil.copy2(P2_FUS_CSV, p1_fus_csv)
    print("Copied FUS CSV from the cross-check implementation ->")
    print("  " + p1_fus_csv)

    fus_df = pd.read_csv(p1_fus_csv)
    print("FUS: " + str(len(fus_df)) + " rows loaded")

    # Quick sanity check on the copied data
    cr_k1  = fus_df[(fus_df["alpha"] == 0.0) & (fus_df["k"] == 1)]["CR"].mean()
    row50  = fus_df[(fus_df["alpha"] == 0.0) & (fus_df["k"] == 50)]
    mae_u  = row50["MAE_users"].mean()
    cr_k50 = row50["CR"].mean()
    print("  CR at k=1:   " + str(round(cr_k1,  4)) + "  (expect ~0.60)")
    print("  MAE_users k=50: " + str(round(mae_u,  4)))
    print("  CR at k=50:  " + str(round(cr_k50, 4)) + "  (expect ~0.9997)")

    # ------------------------------------------------------------------
    # Step 2: Load data for CF sweep
    # ------------------------------------------------------------------
    print("\nLoading ratings ...")
    ratings = fus.load_ratings(RATINGS_PATH)
    ratings = fus.filter_dataset(ratings)
    n_u = ratings["user_id"].nunique()
    n_i = ratings["item_id"].nunique()
    n_r = len(ratings)
    print("  " + str(n_u) + " users x " + str(n_i) + " items x " + str(n_r) + " ratings  OK")

    users       = sorted(ratings["user_id"].unique().tolist())
    items       = sorted(ratings["item_id"].unique().tolist())
    user_to_idx = {uid: i for i, uid in enumerate(users)}
    item_to_idx = {iid: i for i, iid in enumerate(items)}
    n_items     = len(items)

    # ------------------------------------------------------------------
    # Step 3: CF (Pearson KNN) sweep -- unique the core implementation contribution
    # Source: cf.compute_pearson_sim_matrix (Resnick 1994 CF formula)
    # ------------------------------------------------------------------
    print("\nRunning CF (Pearson KNN) sweep ...")
    triples = ratings[["user_id", "item_id", "rating"]].values.tolist()
    kf      = KFold(n_splits=10, shuffle=True, random_state=42)
    cf_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(triples)):
        print("\n-- Fold " + str(fold_idx + 1) + "/10 --")
        train_list   = [triples[i] for i in train_idx]
        test_list    = [triples[i] for i in test_idx]
        train_df     = pd.DataFrame(train_list,
                                    columns=["user_id", "item_id", "rating"])
        test_triples = [(int(r[0]), int(r[1]), float(r[2])) for r in test_list]

        user_means       = fus.compute_user_means(train_df, users, user_to_idx)
        item_raters_dict = fus.build_item_raters_dict(train_df, user_to_idx)

        print("  CF Pearson sim ...", end=" ", flush=True)
        cf_sim = cf_mod.compute_pearson_sim_matrix(
            train_df, users, user_to_idx,
            item_to_idx, n_items, user_means,
        )
        print("done -- evaluating ...", end=" ", flush=True)

        rows = evaluate_sim_matrix(
            "CF", fold_idx, 0.0, cf_sim, test_triples,
            item_raters_dict, user_means, user_to_idx, K_VALUES,
        )
        cf_rows.extend(rows)
        print("done  (" + str(round(time.time() - t0)) + "s elapsed)")

    # ------------------------------------------------------------------
    # Step 4: Write CF CSV - shared_contract.md §5
    # ------------------------------------------------------------------
    col_order = ["system", "protocol", "fold", "k", "alpha",
                 "MAE_data", "MAE_users", "RMSE", "RMSE_users", "CR"]
    cf_df   = pd.DataFrame(cf_rows)[col_order]
    cf_path = os.path.join(RESULTS_DIR, "results_CF_A_warm.csv")
    cf_df.to_csv(cf_path, index=False)
    print("\nWrote " + cf_path + "  (" + str(len(cf_df)) + " rows)")

    row50_cf = cf_df[cf_df["k"] == 50]
    print("CF at k=50: MAE_users=" + str(round(row50_cf["MAE_users"].mean(), 4)) +
          "  CR=" + str(round(row50_cf["CR"].mean(), 4)))

    # ------------------------------------------------------------------
    # Step 5: Generate 4 plots from FUS results
    # Source: matplotlib
    # ------------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_fig2(fus_df, os.path.join(FIGS_DIR, "fig2_mae_data_per_fold.png"))
    plot_fig3(fus_df, os.path.join(FIGS_DIR, "fig3_mae_vs_k_alpha.png"))
    plot_fig4(fus_df, os.path.join(FIGS_DIR, "fig4_rmse_vs_k_alpha.png"))
    plot_fig5(fus_df, os.path.join(FIGS_DIR, "fig5_cr_vs_k_alpha.png"))

    print("\nDone. Total time: " + str(round((time.time() - t0) / 60, 1)) + " min")
    print("Results in: " + RESULTS_DIR)


if __name__ == "__main__":
    main()
