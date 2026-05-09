"""
pf.py  - Probabilistic Filtering (PF baseline, Hao et al. 2016)

This file implements the PF baseline *from the Hao 2016 paper equations*,
then evaluates it under the base paper's Protocol A and metric definitions.

Primary sources:
  - Hao et al. 2016 (PF): see ../papers/hao2016_9535808_pf.pdf
    Key equations used here:
      * Eq. (4): UM probability matrix definition p(s|t)
      * Eq. (5): Bayes p(s|t) = p(s ∩ t) / p(t)
      * Eq. (6): similarity sim(ux,uy) using overlap of positive-topic sets
      * Eq. (7): positive-topic set pos(u) thresholded by θ on positive probability
    Also: text states "only the positive probability is considered" and that
    high scores are 3-5 when ratings are 1-5; and discusses θ in §4.3 with
    θ swept 0.3-0.9 and good CR around θ≈0.4.

  - Base paper (evaluation protocol + prediction formula Eq. (19)):
    ../../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf

  - Shared contract (filter shape, folds, metrics, CSV layout):
    ../../shared_contract.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# MovieLens-100k u.item genre columns (19, including 'unknown').
# Source: GroupLens MovieLens 100k dataset documentation (u.item schema).
GENRE_COLS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

# Hao 2016 uses MovieLens genres as "topics" and reports 18 topics (Table 1),
# i.e., drop the "unknown" column. Source: Hao 2016 dataset description and Table 1.
TOPIC_COLS = GENRE_COLS[1:]


def load_ratings(ratings_path: str) -> pd.DataFrame:
    """Load MovieLens ratings triples.

    Source: MovieLens 100k schema (u.data columns).
    """

    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=cols,
        usecols=["user_id", "item_id", "rating"],
    )
    return df


def load_item_genres(items_path: str) -> pd.DataFrame:
    """Load MovieLens item genres as a binary matrix indexed by item_id.

    Source: MovieLens 100k schema (u.item).
    """

    cols = ["item_id", "movie_title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
    items = pd.read_csv(
        items_path,
        sep="|",
        names=cols,
        encoding="latin-1",
        usecols=["item_id"] + GENRE_COLS,
    ).set_index("item_id")
    return items


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the base paper's dataset filtering protocol (shared_contract §2.3).

    NOTE:
      The shared_contract.md lists an item-first ordering, but the contract also
      locks the *final* filtered shape to 497 × 903 × 79,432. On the provided
      MovieLens-100k `u.data`, the only ordering that reproduces that exact
      shape is:

      1) Keep the top-497 users by rating count in the full table.
      2) Drop items with <20 ratings in the user-restricted table.

    Must yield: 497 users × 903 items × 79,432 ratings.
    """

    # Step 1: top users in the full table
    user_counts = df.groupby("user_id").size().sort_values(ascending=False)
    top_users = user_counts.head(497).index
    df2 = df[df["user_id"].isin(top_users)].copy()

    # Step 2: item frequency threshold in the user-restricted table
    item_counts = df2.groupby("item_id").size()
    keep_items = item_counts[item_counts >= 20].index
    df2 = df2[df2["item_id"].isin(keep_items)].reset_index(drop=True)

    # Contract-shape guardrail
    assert df2["user_id"].nunique() == 497
    assert df2["item_id"].nunique() == 903
    assert len(df2) == 79432
    return df2


def load_and_filter(ratings_path: str, items_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience loader used by eval.py."""

    df = load_ratings(ratings_path)
    df = filter_dataset(df)
    item_genres = load_item_genres(items_path).loc[sorted(df["item_id"].unique())]
    # PF topics are the 18 non-"unknown" genres. (Hao 2016 Table 1: 18 topics.)
    item_topics = item_genres[TOPIC_COLS].copy()
    return df, item_topics


@dataclass
class _IndexMaps:
    users: List[int]
    items: List[int]
    user_idx: Dict[int, int]
    item_idx: Dict[int, int]


class ProbabilisticFiltering:
    """PF recommender (Hao et al. 2016).

    Model:
      - Construct a probability-based user model UM with entries p(s|t)
        (Hao 2016 Eq. (4) and Eq. (5)).
      - Define each user's positive-topic set pos(u) by thresholding the
        positive probability pos_p(u,t) = Σ_{s∈{3,4,5}} p(s|t) against θ
        (Hao 2016 Eq. (7) + accompanying text).
      - Define similarity as overlap of positive-topic sets
        (Hao 2016 Eq. (6)).

    Prediction:
      - Fixed top-k neighbourhood by similarity (Hao 2016 §3.4).
      - Resnick-style prediction formula (base paper Eq. (19)), using only
        neighbours that rated the target item.
    """

    def __init__(self, theta: float = 0.4, positive_scores: Tuple[int, ...] = (3, 4, 5)):
        # θ: Hao 2016 Eq. (7) threshold; paper discusses θ sweep and shows good CR around θ≈0.4.
        self.theta = float(theta)
        # Hao 2016: "high score ranges from 3 to 5 when the score ranges from 1 to 5."
        self.positive_scores = tuple(int(s) for s in positive_scores)

        self._maps: Optional[_IndexMaps] = None
        self._R: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self._sim: Optional[np.ndarray] = None
        self._nbr_cache: Dict[int, List[np.ndarray]] = {}

    def fit(self, df_train: pd.DataFrame, item_genres: pd.DataFrame) -> None:
        """Fit PF: build UM-derived similarity matrix.

        Paper mapping:
          - UM definition: Hao 2016 Eq. (4)
          - p(s|t): Hao 2016 Eq. (5)
          - pos(u) via θ and positive probability: Hao 2016 Eq. (7)
          - sim(u,v): Hao 2016 Eq. (6)
        """

        # Accept either 18-topic (no unknown) or 19-genre input; PF uses 18 topics.
        if "unknown" in item_genres.columns:
            item_topics = item_genres[TOPIC_COLS]
        else:
            item_topics = item_genres

        users = sorted(df_train["user_id"].unique())
        items = sorted(df_train["item_id"].unique())
        maps = _IndexMaps(
            users=users,
            items=items,
            user_idx={u: i for i, u in enumerate(users)},
            item_idx={it: i for i, it in enumerate(items)},
        )
        self._maps = maps

        n_u = len(users)
        n_i = len(items)

        # Build ratings matrix R[u, i] where 0 = unrated.
        # Source: standard MovieLens matrix construction; numpy.zeros docs.
        R = np.zeros((n_u, n_i), dtype=np.float32)
        for row in df_train.itertuples(index=False):
            R[maps.user_idx[row.user_id], maps.item_idx[row.item_id]] = float(row.rating)
        self._R = R

        rated_mask = R > 0
        user_cnt = rated_mask.sum(axis=1).astype(np.float32)
        user_cnt[user_cnt == 0] = 1.0
        self._mu = (R.sum(axis=1) / user_cnt).astype(np.float32)

        # -------------------------------------------------------------------
        # Build UM probabilities p(s|t) from ratings and topic membership.
        #
        # Hao 2016 Eq. (4): UM is |S|×|T| matrix of p(s|t).
        # Hao 2016 Eq. (5): p(s|t) = p(s ∩ t) / p(t).
        #
        # In frequency form (equivalent to Eq. (5) with empirical probabilities):
        #   p(s|t) = count_u,t,s / count_u,t
        # where count_u,t is number of rated items by u that belong to topic t,
        # and count_u,t,s are those with rating s.
        # -------------------------------------------------------------------

        topic_mat = item_topics.reindex(items, fill_value=0).values.astype(np.int8)  # (n_i, n_t=18)
        n_t = topic_mat.shape[1]

        # counts_topic[u,t] and counts_score[u,t,s] for s in 1..5
        counts_topic = np.zeros((n_u, n_t), dtype=np.int32)
        counts_score = np.zeros((n_u, n_t, 6), dtype=np.int32)  # index by score value

        for row in df_train.itertuples(index=False):
            ui = maps.user_idx[row.user_id]
            ii = maps.item_idx[row.item_id]
            s = int(row.rating)
            if s < 1 or s > 5:
                continue
            # Multi-topic items contribute to each topic they belong to (MovieLens genres are multi-hot).
            t_mask = topic_mat[ii] > 0
            if not np.any(t_mask):
                continue
            counts_topic[ui, t_mask] += 1
            counts_score[ui, t_mask, s] += 1

        # p(s|t) per user/topic/score (Hao 2016 Eq. (5))
        denom = counts_topic.astype(np.float64)
        denom[denom == 0] = np.nan
        p_s_given_t = counts_score.astype(np.float64) / denom[:, :, None]
        p_s_given_t = np.nan_to_num(p_s_given_t, nan=0.0, posinf=0.0, neginf=0.0)

        # Positive probability pos_p(u,t) = Σ_{s∈{3,4,5}} p(s|t)
        # Source: Hao 2016 text below Eq. (7).
        pos_p = np.zeros((n_u, n_t), dtype=np.float64)
        for s in self.positive_scores:
            pos_p += p_s_given_t[:, :, s]

        # pos(u) (Hao 2016 Eq. (7))
        pos_mask = (pos_p >= self.theta).astype(np.int8)  # (n_u, n_t)

        # sim(u,v) (Hao 2016 Eq. (6))
        #
        # The equation is printed as:
        #   sim(ux,uy) = |pos(ux) ∩ pos(uy)| / |T|
        # where T is the topic set (Hao 2016 Table 1 reports |T| = 18).
        #
        # Note: the surrounding prose is slightly ambiguous about whether |T|
        # means the full topic set or the combined interest-topic set of the
        # two users. For convergence with the PF baseline curves in the base
        # paper (Figs 6-7), we follow the literal Eq. (6) denominator |T| = 18.
        inter = (pos_mask @ pos_mask.T).astype(np.int32)
        sim_uv = (inter / float(n_t)).astype(np.float32)
        np.fill_diagonal(sim_uv, 0.0)
        self._sim = sim_uv

        # Clear any neighbourhood caches (fit invalidates them).
        self._nbr_cache.clear()

    def _neighbourhoods_for_k(self, k: int) -> List[np.ndarray]:
        """Return list of neighbour-index arrays (len<=k) for each user index.

        Base paper semantics: N^k_u is top-k users with highest similarity to u.
        """

        if k in self._nbr_cache:
            return self._nbr_cache[k]
        assert self._sim is not None

        n_u = self._sim.shape[0]
        neighs: List[np.ndarray] = []

        for ui in range(n_u):
            sims = self._sim[ui].copy()
            sims[ui] = -np.inf  # exclude self
            sims[sims <= 0.0] = -np.inf  # similarity is non-negative; drop zeros for neighbourhood ranking.

            # Handle the case where fewer than k candidates remain.
            finite_idx = np.where(np.isfinite(sims))[0]
            if finite_idx.size == 0:
                neighs.append(np.array([], dtype=np.int32))
                continue
            kk = int(min(k, finite_idx.size))
            top = np.argpartition(sims, -kk)[-kk:]
            top = top[np.argsort(sims[top])[::-1]]
            top = top[np.isfinite(sims[top])]
            neighs.append(top.astype(np.int32))

        self._nbr_cache[k] = neighs
        return neighs

    def predict_batch(self, user_ids: Iterable[int], item_ids: Iterable[int], k: int) -> np.ndarray:
        """Batch prediction for (user,item) pairs.

        Returns np.array of float with NaN when no prediction is possible.
        Prediction formula: base paper Eq. 19 (Resnick), clipped to [1,5]
        per shared_contract.md §4.1.
        """

        assert self._maps is not None and self._R is not None and self._mu is not None and self._sim is not None
        maps = self._maps
        R = self._R
        mu = self._mu
        sim = self._sim

        user_ids_l = list(user_ids)
        item_ids_l = list(item_ids)
        preds = np.full(len(user_ids_l), np.nan, dtype=np.float64)

        neighs = self._neighbourhoods_for_k(int(k))

        # Group by item to reuse raters list.
        from collections import defaultdict

        item_to_pairs = defaultdict(list)
        for idx, (uid, iid) in enumerate(zip(user_ids_l, item_ids_l)):
            item_to_pairs[iid].append((idx, uid))

        for iid, pairs in item_to_pairs.items():
            ii = maps.item_idx.get(iid)
            if ii is None:
                continue
            for idx, uid in pairs:
                ui = maps.user_idx.get(uid)
                if ui is None:
                    continue
                nbr = neighs[ui]
                if nbr.size == 0:
                    continue

                # Prediction: Resnick-style (base paper Eq. (19)).
                # We sum only over neighbours that rated the target item.
                r_all = R[nbr, ii].astype(np.float64)
                rated_mask = r_all > 0
                if not np.any(rated_mask):
                    continue
                v = nbr[rated_mask]
                s_uv = sim[ui, v].astype(np.float64)
                denom = float(np.sum(s_uv))
                if denom == 0.0 or not np.isfinite(denom):
                    continue
                numer = float(np.sum((r_all[rated_mask] - mu[v]) * s_uv))
                pred = float(mu[ui] + numer / denom)
                preds[idx] = float(np.clip(pred, 1.0, 5.0))

        return preds

    def predict(self, user_id: int, item_id: int, k: int) -> Optional[float]:
        """Convenience wrapper for a single prediction."""

        v = self.predict_batch([user_id], [item_id], k)[0]
        return None if np.isnan(v) else float(v)
