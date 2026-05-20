# Implementation Contract

**Status:** v1, locked on 2026-04-26.
**Audience:** Every contributor.
**Why this exists:** Three implementation streams run in parallel on different machines. They never see each other's code mid-stream. The only thing that makes the outputs merge cleanly at the end is everyone following this contract identically.

---

## 1. Project context

This is the paper-faithful replication track of:

> G. D'Aniello, M. Della Corte, M. Gaeta, *"A Recommendation System Based on Fuzzy Signature,"* IEEE Access, vol. 14, pp. 9975 to 9985, Jan 2026.

The PDF is at the project root: `A_Recommendation_System_Based_on_Fuzzy_Signature.pdf`. Read Sections IV (the FUS method), V (evaluation methodology, dataset filtering, metrics), and the equations cited below. Treat the PDF as the only source of truth for FUS-related questions.

Three independently-implemented systems plus one classical baseline are produced under this contract: FUS (the `core` and `cross_check` streams cross-verify), CF (in `core`), PF (in `baselines`), GIM (in `baselines`).

---

## 2. Dataset: paths, schema, filtering

### 2.1 Files
| File | Path | Format |
|---|---|---|
| Ratings | `ml-100k/u.data` | TSV: `user_id\titem_id\trating\ttimestamp` |
| Movies + genres | `ml-100k/u.item` | Pipe-separated; last 19 cols are binary genre flags |

All IDs in `u.data` and `u.item` are **1-indexed**.

### 2.2 Filtering: produces 497 x 903 x 79,432

The paper text in Section V.A says "removing the movies with fewer than 20 votes and selecting the top 497 users." A literal items-first reading does NOT yield the paper's stated 497 x 903 x 79,432 shape on the provided `u.data`. The order that DOES produce the contract shape is:

1. Read the full ratings table from `u.data` (100,000 rows).
2. Keep the top 497 users by total rating count in the full table.
3. Drop movies that have fewer than 20 ratings within the user-restricted table.
4. Final result must be exactly: **497 unique users, 903 unique items, 79,432 ratings.**

If your filtered shape is not 497 x 903 x 79,432, your filter is wrong. Fix it before going further. No exceptions: the whole comparison story breaks if any contributor uses a different sample.

---

## 3. Cross-validation: identical fold split for everyone

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
```

Split on the list of (user, item, rating) triples (the long-format ratings, after the filtering above). Indexes returned by `kf.split(...)` define train and test for each fold. Same `random_state=42` everywhere.

Reference (sklearn KFold doc): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

---

## 4. Metrics: paper's Eqs 20 to 24, exactly

| Metric | Equation | Definition |
|---|---|---|
| `MAE_data` | Eq 20 | mean of absolute errors over all (user, item) pairs in the test fold |
| `MAE_users` | Eq 22 (via Eq 21) | mean of per-user MAE over users that have at least 1 prediction in the fold |
| `RMSE` | Eq 23 | root-mean-square of errors over all (user, item) pairs |
| `RMSE_users` | analogous to Eq 22 | mean of per-user RMSE over users with at least 1 prediction |
| `CR` | Eq 24 | coverage rate: predictions made divided by predictions attempted |

### 4.1 Prediction clipping (revised May 2026 audit)
Predicted ratings are **NOT** clipped to [1, 5]. Paper Eq 19 is a real-valued residual-shrinkage formula with no clip; the literal-paper reading is to return the raw value. Previous versions of this contract mandated a clip (May 2026 audit revision: removed). Effect: MAE_data and RMSE rise slightly because some predictions land outside [1, 5]; the per-user metrics (MAE_users, RMSE_users) are nearly unaffected. CR matches the paper's published Fig. 5 to three decimals (0.601 vs 0.60 at k=1) either way.

### 4.2 Coverage Rate definition (do not confuse with accuracy)
A prediction "counts as made" when the recommender returns a number for the (user, item) pair, even if that number is wrong. CR is therefore the *attempt* rate, not the *correctness* rate. The paper makes this point in Section V.D. Keep it in mind when interpreting results.

---

## 5. Output CSV format: exact column order, exact names

Each system writes one CSV per system. Filename pattern:

```
results_<SYSTEM>_A_warm.csv
```

with `<SYSTEM>` in `{FUS, CF, PF, GIM}`.

### Columns (exact order, exact names)

```
system, protocol, fold, k, alpha, MAE_data, MAE_users, RMSE, RMSE_users, CR
```

| Column | Type | Notes |
|---|---|---|
| `system` | string | one of FUS, CF, PF, GIM |
| `protocol` | string | always `A_warm` for the paper-faithful release |
| `fold` | int | 0 to 9 |
| `k` | int | neighborhood size |
| `alpha` | float | alpha-cut threshold (FUS only; use 0.0 for non-FUS systems) |
| `MAE_data` | float | rounded to 4 decimals is fine |
| `MAE_users` | float | |
| `RMSE` | float | |
| `RMSE_users` | float | |
| `CR` | float | in [0, 1] |

### Rows
One row per `(fold, k, alpha)` combination. So if you sweep k in {1, 2, 4, 6, 8, 10, 15, 20, 30, 50} (10 values) x alpha in {0, 0.7, 0.8, 0.9} (4 values) x 10 folds, you get 400 rows for FUS.

---

## 6. Hyperparameter sweeps

| Sweep | Protocol A |
|---|---|
| `k` (neighborhood size) | {1, 2, 4, 6, 8, 10, 15, 20, 30, 50} |
| `alpha` (FUS only) | {0, 0.7, 0.8, 0.9} |

The reference baselines (PF, GIM) only need k up to 10. The paper's Figs 6 and 7 cap there because PF and GIM degrade beyond k=10.

---

## 7. Environment

- Python 3.11 (3.10 also fine).
- Allowed libraries: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `scikit-surprise` (optional for CF baseline; pure-numpy is preferred), `deap` (optional for GIM genetic algorithm).
- Not used: PyTorch, TensorFlow, JAX, Keras. The whole project is classical ML.

Suggested install:

```
pip install numpy pandas scikit-learn scipy matplotlib
```

---

## 8. Code provenance policy

Every algorithmic block in this repository must be adapted from an explicit public source. Acceptable sources are:

- A GitHub repository (cite the full URL in a comment above the block).
- Official library documentation (cite URL).
- A Stack Overflow answer (cite the question URL).
- The paper's own pseudo-code or numbered equations (cite "paper Section IV.B Eq 19" or similar).

Trivial glue (imports, variable assignments, calling a documented library function as documented) does not need a citation. Anything algorithmic does. If no public source can be found for a snippet, the contributor stops and consults the team before inventing the implementation.

This rule keeps the replication auditable: a reviewer can trace every algorithmic line to a citable origin.

---

## 9. Stream layout

| Stream | Code folder | Results folder |
|---|---|---|
| Core implementation (FUS + CF) | `core/code/` | `core/results/` |
| Independent FUS cross-check | `cross_check/code/` | `cross_check/results/` |
| Reference baselines (PF + GIM) | `baselines/code/` | `baselines/results/` |

Put `.py` files in `code/`. Put output `.csv` files (one per system) in `results/`. Put plots in `results/figs/`.

---

## 10. Sanity-check numbers

Pure FUS at `k = 50, alpha = 0` on Protocol A should match the paper's reported values within plus or minus 0.02:

| Metric | Paper value | Pass band |
|---|---|---|
| `MAE_data` | 0.83 | 0.81 to 0.85 |
| `MAE_users` | 0.703 | 0.683 to 0.723 |
| `RMSE` | 1.107 | 1.087 to 1.127 |
| `CR` | ~0.99 | 0.97 to 1.00 |

If you are outside the pass band, the FUS implementation is wrong somewhere. Common bugs: forgetting the "exclude unrated neighbors" rule (paper Section IV.B), wrong axis when normalising `RatPop` / `ItemAttr`, applying alpha-cut after the kindredness sum instead of before.

For the reference baselines (PF, GIM), the paper reports approximate visual values from Figs 6 and 7 at k=10:
- PF: `MAE_users` approximately 0.88, `CR` approximately 0.92, pass band plus or minus 0.05.
- GIM: `MAE_users` approximately 0.85, `CR` approximately 0.92, pass band plus or minus 0.05.

These are older papers, so small variation is expected.
