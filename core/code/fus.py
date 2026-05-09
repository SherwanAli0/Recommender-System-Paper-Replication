"""
fus.py - the core implementation / Stream core (Faithful Baseline)
Independent faithful re-implementation of the paper's FUS recommender.
Implements paper Equations 1-19 exactly. No deviations, no improvements.

Sources (NO-ORIGINAL-CODE RULE - every algorithmic block is cited):
  paper  : D'Aniello et al., "A Recommendation System Based on Fuzzy
           Signature," IEEE Access vol.14 pp.9975-9985, Jan 2026.
  pandas : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
  numpy  : https://numpy.org/doc/stable/reference/
  sklearn: https://scikit-learn.org/stable/modules/generated/
             sklearn.model_selection.KFold.html
"""

import numpy as np
import pandas as pd

# Rating scale S = {1, 2, 3, 4, 5} - paper §IV.A
N_RATINGS  = 5
RATING_MIN = 1
RATING_MAX = 5


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# Source: pandas.read_csv docs
#   https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# ─────────────────────────────────────────────────────────────────────────────
def load_ratings(path):
    """Load ml-100k/u.data. Returns DataFrame [user_id, item_id, rating, timestamp]."""
    return pd.read_csv(
        path, sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET FILTERING - paper §V.A; shared_contract.md §2.3
# Source: pandas Series.value_counts
#   https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
#   pandas Series.nlargest
#   https://pandas.pydata.org/docs/reference/api/pandas.Series.nlargest.html
#
# Empirically verified order that produces 497 × 903 × 79,432:
#   1. Keep top-497 users by rating count in the FULL table.
#   2. Drop movies with < 20 ratings within that user-filtered subset.
# (Filtering items from the full table first yields 939 items - wrong.)
# ─────────────────────────────────────────────────────────────────────────────
def filter_dataset(ratings_df):
    """
    paper §V.A: "removing the movies with fewer than 20 votes and selecting
    the top 497 users based on the number of votes."
    Returns filtered DataFrame with exactly 497 users × 903 items × 79,432 ratings.
    """
    # Step 1: top-497 users from the full raw table
    user_counts = ratings_df["user_id"].value_counts()
    top_users   = user_counts.nlargest(497).index
    r = ratings_df[ratings_df["user_id"].isin(top_users)].copy()

    # Step 2: items with ≥ 20 ratings within this user subset
    item_counts = r["item_id"].value_counts()
    valid_items = item_counts[item_counts >= 20].index
    r = r[r["item_id"].isin(valid_items)].reset_index(drop=True)

    assert r["user_id"].nunique() == 497, \
        f"Expected 497 users, got {r['user_id'].nunique()}"
    assert r["item_id"].nunique() == 903, \
        f"Expected 903 items, got {r['item_id'].nunique()}"
    assert len(r) == 79_432, \
        f"Expected 79,432 ratings, got {len(r)}"
    return r


# ─────────────────────────────────────────────────────────────────────────────
# 3. PER-USER 2-D MATRIX Mat_ui - paper §IV.A
# Source: paper §IV.A Eq 8-9; numpy.zeros:
#   https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
# ─────────────────────────────────────────────────────────────────────────────
def build_user_matrix(user_ratings_df, item_to_idx, n_items):
    """
    Build Mat_ui of shape (N_RATINGS=5, n_items).
    Mat_ui[j, k] = 1.0 iff user gave rating (j+1) to item at column k.
    paper: "Matui(j, k) represents that the user rated item mk with rating sj."
    """
    mat = np.zeros((N_RATINGS, n_items), dtype=np.float64)
    for _, row in user_ratings_df.iterrows():
        s_idx = int(row["rating"]) - 1           # rating 1→0 … 5→4
        m_idx = item_to_idx[int(row["item_id"])]
        mat[s_idx, m_idx] = 1.0
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# 4. RATING FREQUENCY and RATING PER ITEM - paper §IV.A Eqs 8-9
# Source: paper §IV.A Eqs 8-9
#   numpy.sum: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
# ─────────────────────────────────────────────────────────────────────────────
def rating_frequency(mat_ui):
    """Eq 8: SF(sj) = Σ_k Mat_ui[j, k]. Returns shape (5,)."""
    return mat_ui.sum(axis=1)


def rating_per_item(mat_ui):
    """Eq 9: SI(mk) = Σ_j Mat_ui[j, k]. Returns shape (n_items,)."""
    return mat_ui.sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RATING POPULARITY and ITEM ATTRACTIVENESS - paper §IV.A Eqs 10-13
# Source: paper §IV.A Eqs 10-13
# ─────────────────────────────────────────────────────────────────────────────
def rat_pop(mat_ui):
    """
    paper Eqs 10-11: RatPopui(sj) = SF(sj) / max_s SF(s).
    Denominator is the per-user maximum (NOT global).
    Returns shape (5,), values in [0, 1].
    """
    sf     = rating_frequency(mat_ui)
    max_sf = sf.max()
    if max_sf == 0.0:
        return np.zeros(N_RATINGS, dtype=np.float64)
    return sf / max_sf


def item_attr(mat_ui):
    """
    paper Eqs 12-13: ItemAttrui(mk) = SI(mk) / max_m SI(m).
    In MovieLens each user rates each item at most once → ItemAttr is binary.
    Returns shape (n_items,), values in {0, 1}.
    """
    si     = rating_per_item(mat_ui)
    max_si = si.max()
    if max_si == 0.0:
        return np.zeros_like(si, dtype=np.float64)
    return si / max_si


# ─────────────────────────────────────────────────────────────────────────────
# 6. USER SIGNATURE - paper §IV.A Eqs 14-15
# Source: paper §IV.A Eqs 14-15
#   "USui(sj, mk) = min{RatPopui(sj), ItemAttrui(mk)}"
#   Applied only to cells where Matui(j, k) = 1 (the "marked cells").
#   numpy.minimum: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
# ─────────────────────────────────────────────────────────────────────────────
def user_signature(mat_ui):
    """
    paper Eqs 14-15: USui(sj, mk) = min(RatPop(sj), ItemAttr(mk)).
    Applied only to rated cells (mat_ui > 0); unrated cells remain 0.
    Returns shape (5, n_items).
    """
    rp = rat_pop(mat_ui)    # shape (5,)
    ia = item_attr(mat_ui)  # shape (n_items,)

    # Outer min over full grid; mask to marked cells only
    # Source: numpy broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html
    us = np.minimum(rp[:, np.newaxis], ia[np.newaxis, :])  # (5, n_items)
    us = us * (mat_ui > 0)                                  # zero unrated cells
    return us


# ─────────────────────────────────────────────────────────────────────────────
# 7. USER KINDREDNESS - paper §IV Eqs 16-18, ASYMMETRIC
# Source: paper §IV Eqs 16-18
#   "sim(ui, uj) = sum(Tnorm(USui, USuj)) / sum(USui)"
#   α-cut: zero out elements ≤ α before sums.
#   ASYMMETRIC - denominator uses only USi (not USj).
#   numpy.where: https://numpy.org/doc/stable/reference/generated/numpy.where.html
# ─────────────────────────────────────────────────────────────────────────────
def kindredness(us_i, us_j, alpha=0.0):
    """
    paper Eq 18: sim(ui, uj) = Σ min(USi_α, USj_α) / Σ USi_α
    α-cut applied before sums; denominator uses only USi.
    """
    us_i_cut = np.where(us_i > alpha, us_i, 0.0)
    us_j_cut = np.where(us_j > alpha, us_j, 0.0)
    denom = us_i_cut.sum()
    if denom == 0.0:
        return 0.0
    numer = np.minimum(us_i_cut, us_j_cut).sum()
    return float(numer / denom)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FULL PAIRWISE SIMILARITY MATRIX (vectorised)
# Source: paper §IV Eq 18 applied to all (i, j) pairs.
#   numpy broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
#   numpy.minimum: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
# ─────────────────────────────────────────────────────────────────────────────
def compute_sim_matrix(all_us, alpha=0.0):
    """
    Compute (n_users, n_users) kindredness matrix.
    all_us : list of (5, n_items) arrays indexed 0 … n_users-1.
    Returns sim[i, j] = sim(user_i, user_j) using paper Eq 18.
    """
    n = len(all_us)
    # Source: numpy.stack https://numpy.org/doc/stable/reference/generated/numpy.stack.html
    sigs_flat = np.stack([u.ravel() for u in all_us], axis=0)  # (n, D)
    sigs_cut  = np.where(sigs_flat > alpha, sigs_flat, 0.0)    # α-cut
    denoms    = sigs_cut.sum(axis=1)                            # (n,)

    sim_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if denoms[i] == 0.0:
            continue
        mins       = np.minimum(sigs_cut[i], sigs_cut)  # (n, D)
        numer      = mins.sum(axis=1)                   # (n,)
        sim_mat[i] = numer / denoms[i]
    return sim_mat


# ─────────────────────────────────────────────────────────────────────────────
# 9. RESNICK PREDICTION - paper §IV.B Eq 19
# Source: paper §IV.B Eq 19 (exclusion correction applied via item_raters_dict)
#   Resnick (1994): https://dl.acm.org/doi/10.1145/192844.192905
#   numpy.clip: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
# ─────────────────────────────────────────────────────────────────────────────
def resnick_predict(u_idx, item_id, sim_row, item_raters_lookup, user_means, k):
    """
    paper Eq 19: p^m_ui = r_ui + Σ_{uj∈Nk} (r^m_uj − r_uj)·sim / Σ sim
    Exclusion correction: only neighbors who rated item_id in training are included.
    Returns np.nan when no qualifying neighbor exists.
    """
    if item_id not in item_raters_lookup:
        return np.nan
    rater_idxs, rater_ratings = item_raters_lookup[item_id]
    if len(rater_idxs) == 0:
        return np.nan

    sims   = sim_row[rater_idxs]
    top_k  = min(k, len(sims))
    order  = np.argsort(sims)[::-1][:top_k]
    t_sims = sims[order]
    t_rats = rater_ratings[order]
    t_uj   = rater_idxs[order]

    denom = t_sims.sum()
    if denom == 0.0:
        return np.nan

    r_ui  = user_means[u_idx]
    numer = ((t_rats - user_means[t_uj]) * t_sims).sum()
    pred  = r_ui + numer / denom
    return float(np.clip(pred, RATING_MIN, RATING_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# 10. HELPERS - item rater lists, user means
# Source: pandas.DataFrame.groupby
#   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# ─────────────────────────────────────────────────────────────────────────────
def build_item_raters(train_df, user_to_idx):
    """
    Returns dict: item_id → (rater_idxs ndarray, ratings ndarray).
    Only users in user_to_idx are included.
    """
    item_raters = {}
    for item_id, grp in train_df.groupby("item_id"):
        idxs, rats = [], []
        for _, row in grp.iterrows():
            uid = int(row["user_id"])
            if uid in user_to_idx:
                idxs.append(user_to_idx[uid])
                rats.append(float(row["rating"]))
        item_raters[item_id] = (
            np.array(idxs, dtype=np.int32),
            np.array(rats, dtype=np.float64),
        )
    return item_raters


def build_item_raters_dict(train_df, user_to_idx):
    """
    Returns dict: item_id → {u_idx: rating (float)}.
    Supports the correct top-k-from-all-users prediction (rank-aware walks).
    Source: same exclusion correction as build_item_raters (paper §IV.B).
    """
    item_raters_dict = {}
    for item_id, grp in train_df.groupby("item_id"):
        d = {}
        for _, row in grp.iterrows():
            uid = int(row["user_id"])
            if uid in user_to_idx:
                d[user_to_idx[uid]] = float(row["rating"])
        item_raters_dict[item_id] = d
    return item_raters_dict


def compute_user_means(train_df, users, user_to_idx):
    """
    Returns ndarray shape (n_users,) with each user's mean training rating.
    Fallback 3.0 for users with no training ratings.
    Source: pandas groupby mean
      https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.mean.html
    """
    n = len(users)
    means    = np.full(n, 3.0, dtype=np.float64)
    m_series = train_df.groupby("user_id")["rating"].mean()
    for uid, idx in user_to_idx.items():
        if uid in m_series.index:
            means[idx] = m_series[uid]
    return means


# ─────────────────────────────────────────────────────────────────────────────
# 11. RANK-AWARE WALK HELPERS - correct top-k-from-all-users prediction
#
# KEY: top-k neighbors selected from ALL users by sim, THEN filter to raters.
# Selecting from raters-only gives CR=100% (wrong). From all users gives
# CR≈60% at k=1 (matches paper exactly).
#
# Source: numpy.argsort https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
#         paper §IV.B Eq 19.
# ─────────────────────────────────────────────────────────────────────────────
def precompute_sorted_nbrs(sim_mat):
    """
    For every user i, sort all other users by sim_mat[i] descending.
    Self excluded by setting sim_mat[i, i] = -1.0 before argsort.
    Returns list of n_users arrays, each shape (n_users,).
    Source: numpy.argsort
    """
    n = sim_mat.shape[0]
    sorted_nbrs = []
    for i in range(n):
        row    = sim_mat[i].copy()
        row[i] = -1.0
        sorted_nbrs.append(np.argsort(row)[::-1])
    return sorted_nbrs


def build_walks(test_triples, user_to_idx,
                sorted_nbrs, item_raters_dict, sim_mat, k_max):
    """
    For each unique (u_idx, item_id) test pair, walk the top-k_max NEIGHBORS
    (in descending sim order) and record those who rated item_id, along with
    their global rank position.

    Returns dict (u_idx, item_id) → list of (rank, j, sim, rating).

    For a given k, use entries with rank < k → correct top-k Resnick.
    Source: paper §IV.B Eq 19 (exclusion correction).
    """
    walks = {}
    for uid, item_id, _ in test_triples:
        u_idx = user_to_idx[uid]
        key   = (u_idx, item_id)
        if key in walks:
            continue
        raters = item_raters_dict.get(item_id, {})
        w = []
        for rank, j in enumerate(sorted_nbrs[u_idx][:k_max]):
            if j in raters:
                w.append((rank, int(j), float(sim_mat[u_idx, j]), raters[j]))
        walks[key] = w
    return walks


def resnick_from_walk(walk, k, u_mean_i, user_means):
    """
    Resnick prediction from a rank-aware walk sliced to entries with rank < k.
    Returns np.nan if no qualifying neighbor or sim-sum is zero.
    Source: paper §IV.B Eq 19.
    """
    top = [(j, s, r) for rank, j, s, r in walk if rank < k]
    if not top:
        return np.nan
    denom = sum(s for _, s, _ in top)
    if denom == 0.0:
        return np.nan
    numer = sum((r - user_means[j]) * s for j, s, r in top)
    return float(np.clip(u_mean_i + numer / denom, RATING_MIN, RATING_MAX))
