# ============================================================
# gim.py  - Fuzzy-Genetic Method (GIM baseline)
# the baselines implementation / Stream baselines - Reference Baselines
# ============================================================
#
# REFERENCES
#   GIM source paper:
#     M. Y. H. Al-Shamri, K. K. Bharadwaj,
#     "Fuzzy-genetic approach to recommender systems based on a novel hybrid
#      user model," Expert Systems with Applications, vol. 35, no. 3,
#     pp. 1386-1399, Oct. 2008.
#     DOI: 10.1016/j.eswa.2007.08.016
#
#   Public implementation used for feature computation + membership functions:
#     https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System
#     (We adapt its GIM feature extraction + fuzzy membership definitions.)
#
#   Base paper (uses GIM as baseline; defines evaluation protocol + neighbourhood):
#     ../../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf
#     - Neighbourhood definition (top-k by similarity): §IV (text near "neighborhood N^k_ui").
#     - Prediction formula: Resnick-style (base paper Eq. 19).
#
#   Shared contract (dataset filter + metrics):
#     ../../shared_contract.md
#
# NOTE ON IMPLEMENTATION ORIGIN
#   This implementation is adapted from the upstream GitHub repository
#   (theanilbajar/Fuzzy-Genetic-Recommender-System) with structural fixes
#   to match the base paper's fixed-N^k neighbourhood semantics.
#
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from pf import GENRE_COLS, filter_dataset, load_item_genres, load_ratings

N_GENRES = len(GENRE_COLS)  # 19 (includes 'unknown')
N_DEMO = 3  # age fuzzy: young, middle, old
N_GIM_LEVELS = 6  # very_bad…excellent
FEATURE_LENGTH = N_DEMO + N_GENRES * N_GIM_LEVELS  # 117


def load_users(users_path: str) -> pd.DataFrame:
    """Load MovieLens user ages.

    Source: MovieLens 100k schema (u.user).
    """

    cols = ["user_id", "age", "sex", "occupation", "zip_code"]
    users = pd.read_csv(users_path, sep="|", names=cols, encoding="latin-1", usecols=["user_id", "age"]).set_index(
        "user_id"
    )
    return users


def load_and_filter(ratings_path: str, items_path: str, users_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader used by eval.py."""

    df = load_ratings(ratings_path)
    df = filter_dataset(df)
    item_genres = load_item_genres(items_path).loc[sorted(df["item_id"].unique())]
    users_df = load_users(users_path)
    return df, item_genres, users_df


def _age_fuzzy_batch(ages: np.ndarray) -> np.ndarray:
    """Age membership [young, middle, old].

    Adapted from:
      https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System
      (see fuzzy_sets.py Age membership functions)
    """

    a = np.asarray(ages, dtype=np.float32)
    # Young
    y = np.where(a < 20, 1.0, np.where(a < 35, (35.0 - a) / 15.0, 0.0))
    # Middle
    m = np.where(
        (a > 20) & (a <= 35),
        (a - 20.0) / 15.0,
        np.where((a > 35) & (a <= 45), 1.0, np.where((a > 45) & (a <= 60), (60.0 - a) / 15.0, 0.0)),
    )
    # Old
    o = np.where(a <= 45, 0.0, np.where(a <= 60, (a - 45.0) / 15.0, 1.0))
    return np.stack([y, m, o], axis=1).astype(np.float32)


def _gim_fuzzy_batch(X: np.ndarray) -> np.ndarray:
    """GIM membership for scalar GIM values.

    Adapted from:
      https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System
      (see fuzzy_sets.py GIM membership functions)
    """

    X = np.asarray(X, dtype=np.float32)
    out = np.zeros((len(X), N_GIM_LEVELS), dtype=np.float32)

    # very_bad
    out[:, 0] = np.where(X <= 1.0, 1.0, 0.0)

    # bad, average, good, very_good (triangular-ish levels)
    # (This follows the draft's mapping; the upstream repo uses comparable piecewise definitions.)
    for j, c in enumerate([2.0, 3.0, 4.0, 5.0], start=1):
        lo = c - 2.0
        out[:, j] = np.where((X > lo) & (X <= c - 1.0), X - lo, np.where((X > c - 1.0) & (X <= c), c - X, 0.0))

    # excellent
    out[:, 5] = np.where(X > 4.0, X - 4.0, 0.0)
    return out


@dataclass
class _IndexMaps:
    users: List[int]
    items: List[int]
    user_idx: Dict[int, int]
    item_idx: Dict[int, int]


def _build_feature_matrix(df_train: pd.DataFrame, item_genres: pd.DataFrame, users_df: pd.DataFrame):
    """Build (n_users, 117) feature matrix + ratings matrix.

    Feature extraction is adapted from:
      https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System
      (see gim.py / fuzzy_sets.py for the GIM computation and fuzzification).
    """

    users = sorted(df_train["user_id"].unique())
    items = sorted(item_genres.index.tolist())
    maps = _IndexMaps(users=users, items=items, user_idx={u: i for i, u in enumerate(users)}, item_idx={it: i for i, it in enumerate(items)})

    nu = len(users)
    ni = len(items)

    # Rating matrix R (0 = unrated)
    R = np.zeros((nu, ni), dtype=np.float32)
    for row in df_train.itertuples(index=False):
        R[maps.user_idx[row.user_id], maps.item_idx[row.item_id]] = float(row.rating)

    # Use float64 for the big matmuls to avoid float32 overflow/NaN propagation on some BLAS builds.
    genre_mat = item_genres.reindex(items, fill_value=0).values.astype(np.float64)  # (ni, 19)
    pos_mask = (R >= 3.0).astype(np.float32)
    rated_mask = (R > 0).astype(np.float32)

    tr = R.sum(axis=1)  # total rating sum per user
    tf = rated_mask.sum(axis=1)  # total items rated per user

    # genre_rating[u,g] = Σ r_ui for positive items in genre g
    R_pos = R * pos_mask
    with np.errstate(all="ignore"):
        gr = R_pos.astype(np.float64) @ genre_mat

    # rgr[u,g] = gr[u,g] / Σ(all ratings by user)
    tr_s = np.where(tr > 0, tr, 1.0)[:, None]
    rgr = gr / tr_s

    # mrgf_add[u,g] = Σ(r-2) for positive items in genre g
    with np.errstate(all="ignore"):
        mrgf_add = (((R - 2.0) * pos_mask).astype(np.float64)) @ genre_mat
    tf_s = np.where(tf > 0, tf, 1.0)[:, None]
    mrgf = mrgf_add / (3.0 * tf_s)

    # GIM harmonic mean (nf=5 in the upstream implementation).
    nf = 5.0
    denom = mrgf + rgr
    gim_vals = np.where(denom > 0, 2.0 * nf * mrgf * rgr / np.where(denom > 0, denom, 1.0), 0.0)

    # Fuzzify GIM values into 6 levels per genre.
    flat = gim_vals.reshape(-1)
    gim_fuzz = _gim_fuzzy_batch(flat).reshape(nu, N_GENRES * N_GIM_LEVELS)

    # Demographic fuzzy features: age.
    ages = np.array([float(users_df.loc[u, "age"]) if u in users_df.index else 35.0 for u in users], dtype=np.float32)
    demo = _age_fuzzy_batch(ages)

    feat = np.concatenate([demo, gim_fuzz], axis=1).astype(np.float32)
    return maps, feat, R


class FuzzyGeneticMethod:
    """GIM recommender baseline.

    The full genetic optimisation is expensive; we keep a lightweight option but
    default to uniform weights (as in the earlier project draft).

    IMPORTANT (base paper semantics):
      We build a fixed top-k neighbourhood N^k_u for each user u based on the
      similarity matrix, then predict using ONLY neighbours in N^k_u that rated
      the target item. If no neighbour in N^k_u rated the item, we return NaN.
    """

    def __init__(self, run_ga: bool = False, positive_only: bool = True):
        self.run_ga = bool(run_ga)
        self.positive_only = bool(positive_only)

        self._maps: Optional[_IndexMaps] = None
        self._R: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self._sim: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._nbr_cache: Dict[int, List[np.ndarray]] = {}

    def fit(self, df_train: pd.DataFrame, item_genres: pd.DataFrame, users_df: pd.DataFrame) -> None:
        maps, feat_mat, R = _build_feature_matrix(df_train, item_genres, users_df)
        self._maps = maps
        self._R = R

        rated_mask = R > 0
        user_cnt = rated_mask.sum(axis=1).astype(np.float32)
        user_cnt[user_cnt == 0] = 1.0
        self._mu = (R.sum(axis=1) / user_cnt).astype(np.float32)

        # Feature weights: GA or uniform.
        # Adapted from the earlier draft: /Users/yaprak/Downloads/baselines/code/gim.py
        if self.run_ga:
            weights = self._run_ga(feat_mat, R)
        else:
            weights = np.ones(FEATURE_LENGTH, dtype=np.float32)
        self._weights = weights

        # Weighted cosine similarity.
        w_feat = feat_mat * weights
        norms = np.linalg.norm(w_feat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normed = w_feat / norms
        with np.errstate(all="ignore"):
            self._sim = (normed.astype(np.float64) @ normed.astype(np.float64).T).astype(np.float32)
        self._sim[~np.isfinite(self._sim)] = -np.inf

        self._nbr_cache.clear()

    def _run_ga(self, feat_mat: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Genetic algorithm to learn feature weights.

        Sources:
          - Al-Shamri & Bharadwaj 2008 (GIM) describes GA-based weight learning (paper §4, per abstract/overview).
            (Full text may require access; this code follows the public implementation structure below.)
          - Public reference implementation:
              https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System
              (see genetic.py for GA-style optimisation of weights)
          - DEAP library (allowed by shared_contract.md §7):
              https://deap.readthedocs.io/en/master/

        Practical note:
          Running a full GA with an exact MAE fitness over all ratings is expensive.
          We therefore evaluate fitness on a fixed subsample of user-item ratings from
          the fold's training set (deterministic seed), which preserves comparability
          across k in the same fold while keeping runtime reasonable.
        """

        from deap import base, creator, tools  # DEAP docs: https://deap.readthedocs.io/en/master/

        rng = np.random.default_rng(42)
        nu = feat_mat.shape[0]

        rated_mask = R > 0
        mu = R.sum(axis=1) / np.where(rated_mask.sum(axis=1) > 0, rated_mask.sum(axis=1), 1)

        # Build a deterministic evaluation set of (u,i,actual) from the training matrix.
        rated_pairs = np.argwhere(R > 0)
        if rated_pairs.shape[0] == 0:
            return np.ones(FEATURE_LENGTH, dtype=np.float32)
        sample_n = int(min(2000, rated_pairs.shape[0]))
        sample_idx = rng.choice(rated_pairs.shape[0], size=sample_n, replace=False)
        eval_pairs = rated_pairs[sample_idx]

        def fitness(individual):
            w = np.asarray(individual, dtype=np.float32)
            wf = feat_mat * w
            nrm = np.linalg.norm(wf, axis=1, keepdims=True)
            nrm[nrm == 0] = 1e-9
            normed = wf / nrm
            with np.errstate(all="ignore"):
                sim = (normed.astype(np.float64) @ normed.astype(np.float64).T).astype(np.float32)
            sim[~np.isfinite(sim)] = 0.0

            errs = []
            # Use a small neighbourhood size for fitness evaluation (k_fit=10)
            # so weights optimise similarity usefulness in the regime we evaluate.
            k_fit = 10
            for ui, ii in eval_pairs:
                ui = int(ui)
                ii = int(ii)
                # top-k neighbours by similarity
                s_row = sim[ui].copy()
                s_row[ui] = -np.inf
                # Keep only positive similarity neighbours for stability.
                s_row[s_row <= 0] = -np.inf
                finite = np.where(np.isfinite(s_row))[0]
                if finite.size == 0:
                    continue
                kk = int(min(k_fit, finite.size))
                top = np.argpartition(s_row, -kk)[-kk:]
                top = top[np.argsort(s_row[top])[::-1]]
                top = top[np.isfinite(s_row[top])]

                r_all = R[top, ii].astype(np.float64)
                rated_n = r_all > 0
                if not np.any(rated_n):
                    continue
                v = top[rated_n]
                s_uv = sim[ui, v].astype(np.float64)
                denom = float(np.sum(s_uv))
                if denom == 0.0 or not np.isfinite(denom):
                    continue
                numer = float(np.sum((r_all[rated_n] - mu[v]) * s_uv))
                pred = float(np.clip(mu[ui] + numer / denom, 1.0, 5.0))
                errs.append(abs(pred - float(R[ui, ii])))

            # Minimise MAE on the subsample; DEAP expects a tuple.
            return (float(np.mean(errs)) if errs else 1.0,)

        # DEAP requires creator classes; define them once.
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", rng.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=FEATURE_LENGTH)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.15, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=30)
        # Evaluate initial population
        fits = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        ngen = 20
        cxpb = 0.5
        mutpb = 0.3

        for _ in range(ngen):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < cxpb:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mut in offspring:
                if rng.random() < mutpb:
                    toolbox.mutate(mut)
                    del mut.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            pop[:] = offspring

        best = tools.selBest(pop, 1)[0]
        w = np.clip(np.asarray(best, dtype=np.float32), 0.0, 1.0)
        return w

    def _neighbourhoods_for_k(self, k: int) -> List[np.ndarray]:
        if k in self._nbr_cache:
            return self._nbr_cache[k]
        assert self._sim is not None

        n_u = self._sim.shape[0]
        neighs: List[np.ndarray] = []
        for ui in range(n_u):
            sims = self._sim[ui].copy()
            sims[ui] = -np.inf
            if self.positive_only:
                sims[sims <= 0.0] = -np.inf
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
        """Batch Resnick prediction (base paper Eq. 19), clipped to [1,5]."""

        assert self._maps is not None and self._R is not None and self._mu is not None and self._sim is not None
        maps = self._maps
        R = self._R
        mu = self._mu
        sim = self._sim

        user_ids_l = list(user_ids)
        item_ids_l = list(item_ids)
        preds = np.full(len(user_ids_l), np.nan, dtype=np.float64)
        neighs = self._neighbourhoods_for_k(int(k))

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
                # Base paper Eq. (19) + explanatory text (see pf.py for details):
                # sum numerator/denominator only over neighbours that rated the item.
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
                preds[idx] = float(np.clip(float(mu[ui] + numer / denom), 1.0, 5.0))

        return preds

    def predict(self, user_id: int, item_id: int, k: int) -> Optional[float]:
        v = self.predict_batch([user_id], [item_id], k)[0]
        return None if np.isnan(v) else float(v)
