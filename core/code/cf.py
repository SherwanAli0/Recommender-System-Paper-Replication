"""
cf.py - the core implementation / Stream core (Faithful Baseline)
Standard memory-based collaborative filtering baseline.

Implements Pearson-correlation KNN CF using the same Resnick prediction
formula as FUS (paper Eq 19), with Pearson similarity replacing kindredness.
This is equivalent to scikit-surprise's KNNWithMeans with pearson similarity,
implemented in pure NumPy to avoid the scikit-surprise C-extension install
dependency on Python 3.14 / Windows without MSVC Build Tools.

Sources (NO-ORIGINAL-CODE RULE):
  Resnick 1994 GroupLens : https://dl.acm.org/doi/10.1145/192844.192905
  Pearson formula        : https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
  KNNWithMeans concept   : https://surprise.readthedocs.io/en/stable/knn_inspired.html
  numpy matrix multiply  : https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
  numpy.outer            : https://numpy.org/doc/stable/reference/generated/numpy.outer.html
  numpy.clip             : https://numpy.org/doc/stable/reference/generated/numpy.clip.html
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# PEARSON SIMILARITY MATRIX
#
# Source: Pearson correlation formula adapted from
#   https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
#   and the GroupLens CF implementation rationale from Resnick 1994:
#   https://dl.acm.org/doi/10.1145/192844.192905
#
# Vectorised via mean-centered rating matrix R_c:
#   pearson(i, j) = Σ_{m ∈ I_i ∩ I_j} (r_im - r̄_i)(r_jm - r̄_j)
#                  / (norm_i × norm_j)
# where norm_i = sqrt( Σ_{m ∈ I_i} (r_im - r̄_i)² ).
# Unrated cells are 0 in R_c, so the dot-product naturally sums only over
# co-rated items - no explicit co-rated-item mask needed.
# ─────────────────────────────────────────────────────────────────────────────
def compute_pearson_sim_matrix(train_df, users, user_to_idx,
                                item_to_idx, n_items, user_means):
    """
    Compute (n_users, n_users) Pearson similarity matrix from training data.

    train_df   : DataFrame [user_id, item_id, rating, ...]
    users      : ordered list of user IDs (length n_users)
    user_to_idx: dict uid → 0-based index
    item_to_idx: dict iid → 0-based index
    n_items    : total number of items
    user_means : ndarray shape (n_users,) - training mean per user

    Returns sim_mat where sim_mat[i, i] = -1.0 (self excluded).
    Values in [-1, 1].

    Source: numpy.zeros https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
            numpy broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html
    """
    n_users = len(users)

    # Build dense rating matrix R: shape (n_users, n_items), 0 = unrated
    # Source: numpy.zeros
    R = np.zeros((n_users, n_items), dtype=np.float64)
    for _, row in train_df.iterrows():
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        if uid in user_to_idx and iid in item_to_idx:
            R[user_to_idx[uid], item_to_idx[iid]] = float(row["rating"])

    # Mean-center: subtract user mean from rated cells; unrated stay 0
    # Source: numpy broadcasting
    rated_mask = (R != 0.0)
    R_c = R.copy()
    for i in range(n_users):
        R_c[i, rated_mask[i]] -= user_means[i]
    # R_c[i, m] = r_im - r̄_i  if user i rated item m, else 0

    # Norms: sqrt( Σ_m R_c[i,m]² ) = sqrt( Σ_{m ∈ I_i} (r_im - r̄_i)² )
    # Source: numpy.sqrt https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
    norms = np.sqrt((R_c ** 2).sum(axis=1))    # shape (n_users,)
    norms[norms == 0.0] = 1.0                  # avoid division by zero

    # Pearson sim matrix via matrix multiply
    # Source: numpy.matmul https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    #         numpy.outer  https://numpy.org/doc/stable/reference/generated/numpy.outer.html
    sim_mat = (R_c @ R_c.T) / np.outer(norms, norms)  # shape (n_users, n_users)

    # Clip to valid range and exclude self (set diagonal to -1)
    # Source: numpy.clip https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    np.clip(sim_mat, -1.0, 1.0, out=sim_mat)
    np.fill_diagonal(sim_mat, -1.0)

    return sim_mat
