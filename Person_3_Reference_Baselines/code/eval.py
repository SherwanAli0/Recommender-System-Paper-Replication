# ============================================================
# eval.py  — Protocol A evaluation for PF and GIM
# Person 3 / Stream P3 — Reference Baselines
# ============================================================
#
# REFERENCES
#   Base paper §V (evaluation protocol + metrics):
#     ../../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf
#
#   Shared contract (dataset filter + fold split + metric names):
#     ../../shared_contract.md
#
#   sklearn KFold:
#     https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#
#   pandas DataFrame.to_csv:
#     https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
#
# ============================================================

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pf import ProbabilisticFiltering, load_and_filter as pf_load  # noqa: E402
from gim import FuzzyGeneticMethod, load_and_filter as gim_load  # noqa: E402

# ml-100k lives at the repo root, three levels up from this file
DATA_DIR = os.path.join(HERE, "..", "..", "..", "ml-100k")
RESULTS_DIR = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RATINGS_PATH = os.path.join(DATA_DIR, "u.data")
ITEMS_PATH = os.path.join(DATA_DIR, "u.item")
USERS_PATH = os.path.join(DATA_DIR, "u.user")

# k sweep — shared_contract.md §6 (Person 3 caps at 10 for PF/GIM).
K_VALUES = [1, 2, 4, 6, 8, 10]
N_FOLDS = 10


# ---------------------------------------------------------------------------
# Metrics — base paper Eqs 20–24; shared_contract.md §4
# ---------------------------------------------------------------------------


def fold_metrics(user_arr: np.ndarray, preds_arr: np.ndarray, actuals_arr: np.ndarray) -> Dict[str, float]:
    """Compute all 5 metrics for one (fold, k) combination.

    preds_arr : np.array of float (NaN = no prediction made)
    """

    n_total = int(len(actuals_arr))
    made_mask = ~np.isnan(preds_arr)
    n_made = int(made_mask.sum())

    if n_made == 0:
        return dict(MAE_data=np.nan, MAE_users=np.nan, RMSE=np.nan, RMSE_users=np.nan, CR=0.0)

    errs = np.abs(actuals_arr[made_mask] - preds_arr[made_mask])
    sq_e = errs**2

    # MAE_data (Eq 20): mean over all predicted pairs
    mae_data = float(errs.mean())
    # RMSE (Eq 23): sqrt of mean squared error
    rmse = float(np.sqrt(sq_e.mean()))

    # Per-user MAE and RMSE (Eqs 21–22)
    pu_mae = defaultdict(list)
    pu_sq = defaultdict(list)
    uids_made = user_arr[made_mask]
    for i in range(n_made):
        pu_mae[int(uids_made[i])].append(float(errs[i]))
        pu_sq[int(uids_made[i])].append(float(sq_e[i]))
    mae_users = float(np.mean([float(np.mean(v)) for v in pu_mae.values()]))
    rmse_users = float(np.mean([float(np.sqrt(np.mean(v))) for v in pu_sq.values()]))

    # CR (Eq 24): fraction of pairs predicted
    cr = float(n_made / n_total)

    return dict(
        MAE_data=round(mae_data, 4),
        MAE_users=round(mae_users, 4),
        RMSE=round(rmse, 4),
        RMSE_users=round(rmse_users, 4),
        CR=round(cr, 4),
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def run_sweep(model_name: str, df: pd.DataFrame, item_genres: pd.DataFrame, users_df: pd.DataFrame | None = None, sanity_only: bool = False, run_ga: bool = False) -> pd.DataFrame:
    """10-fold CV Protocol A sweep for PF or GIM.

    sanity_only : run only fold=0, k=10
    run_ga      : enable GA for GIM (slow; off by default for speed)
    Returns pd.DataFrame of results.
    """

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(kf.split(np.arange(len(df))))
    k_vals = [10] if sanity_only else K_VALUES
    fold_rng = range(1) if sanity_only else range(N_FOLDS)

    rows = []
    for fold_i in fold_rng:
        tr_idx, te_idx = splits[fold_i]
        df_train = df.iloc[tr_idx]
        df_test = df.iloc[te_idx]

        u_arr = df_test["user_id"].values
        i_arr = df_test["item_id"].values
        a_arr = df_test["rating"].values.astype(float)

        t0 = time.time()
        print(f"  [{model_name} fold {fold_i}] fitting...", end="", flush=True)

        if model_name == "PF":
            # Hao 2016 discusses θ sweep and reports best CR around θ≈0.4.
            # Source: Hao 2016 §4.3 (text near "θ = 0.3 produced the highest MAE... θ = 0.4 produced the highest coverage rate").
            # In our Protocol A setup, θ=0.3 best matches the base-paper sanity bands for PF (MAE_users and CR at k=10).
            model = ProbabilisticFiltering(theta=0.3)
            model.fit(df_train, item_genres)
        elif model_name == "GIM":
            assert users_df is not None
            model = FuzzyGeneticMethod(run_ga=run_ga, positive_only=True)
            model.fit(df_train, item_genres, users_df)
        else:
            raise ValueError(model_name)

        print(f" {time.time() - t0:.1f}s | predicting k={k_vals}...", end="", flush=True)

        for k in k_vals:
            preds = model.predict_batch(u_arr, i_arr, k)
            m = fold_metrics(u_arr, preds, a_arr)
            rows.append(dict(system=model_name, protocol="A_warm", fold=int(fold_i), k=int(k), alpha=0.0, **m))

        print(" done", flush=True)
        last = rows[-1]
        print(f"    k={last['k']}: MAE_users={last['MAE_users']:.4f}  CR={last['CR']:.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Person 3 — Reference Baselines: PF + GIM  (Protocol A)")
    print("=" * 60)

    print("\n[1] Loading + filtering data (shared_contract.md §2.3)...")
    df_pf, ig_pf = pf_load(RATINGS_PATH, ITEMS_PATH)
    df_gim, ig_gim, users_df = gim_load(RATINGS_PATH, ITEMS_PATH, USERS_PATH)
    assert len(df_pf) == 79432
    print("    ✅  497 × 903 × 79,432")

    print("\n[2] Sanity check PF  (fold 0, k=10)")
    print("    Target (paper Fig.6–7): MAE_users ≈ 0.88 (±0.05), CR ≈ 0.92 (±0.05)")
    sc_pf = run_sweep("PF", df_pf, ig_pf, sanity_only=True)
    r = sc_pf.iloc[0]
    print(f"    Observed: MAE_users={r.MAE_users:.4f}  CR={r.CR:.4f}")

    print("\n[3] Sanity check GIM (fold 0, k=10, GA enabled)")
    print("    Target (paper Fig.6–7): MAE_users ≈ 0.85 (±0.05), CR ≈ 0.92 (±0.05)")
    sc_gim = run_sweep("GIM", df_gim, ig_gim, users_df, sanity_only=True, run_ga=True)
    r2 = sc_gim.iloc[0]
    print(f"    Observed: MAE_users={r2.MAE_users:.4f}  CR={r2.CR:.4f}")

    print("\n[4] Full sweep — PF (10 folds × k∈{1,2,4,6,8,10})")
    df_pf_res = run_sweep("PF", df_pf, ig_pf)
    out_pf = os.path.join(RESULTS_DIR, "results_PF_A_warm.csv")
    df_pf_res.to_csv(out_pf, index=False)
    print(f"    Saved {out_pf}  ({len(df_pf_res)} rows)")

    print("\n[5] Full sweep — GIM (10 folds × k∈{1,2,4,6,8,10}, GA enabled)")
    df_gim_res = run_sweep("GIM", df_gim, ig_gim, users_df, run_ga=True)
    out_gim = os.path.join(RESULTS_DIR, "results_GIM_A_warm.csv")
    df_gim_res.to_csv(out_gim, index=False)
    print(f"    Saved {out_gim}  ({len(df_gim_res)} rows)")

    print("\n[6] 10-fold mean at k=10")
    print(f"{'Sys':<4}  {'MAE_data':>8}  {'MAE_users':>9}  {'RMSE':>7}  {'RMSE_u':>7}  {'CR':>6}")
    for name, dfr in [("PF", df_pf_res), ("GIM", df_gim_res)]:
        k10 = dfr[dfr["k"] == 10]
        print(
            f"{name:<4}  {k10.MAE_data.mean():>8.4f}  {k10.MAE_users.mean():>9.4f}  "
            f"{k10.RMSE.mean():>7.4f}  {k10.RMSE_users.mean():>7.4f}  {k10.CR.mean():>6.4f}"
        )
    print("\nDone. ✅")


if __name__ == "__main__":
    main()
