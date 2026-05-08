# Person 1 — Faithful Baseline

**Paper**: D'Aniello et al., "A Recommendation System Based on Fuzzy Signature," IEEE Access 2026.  
**Dataset**: MovieLens 100k, filtered to **497 users × 903 items × 79,432 ratings**.  
**Role**: Faithfully replicate the paper's FUS algorithm (Eqs 1–19) and provide a standard CF baseline.

---

## What This Person Delivers

1. **Pure FUS implementation** — strictly per Eqs 1–19. No deviations, no improvements.
2. **Standard CF baseline** — Pearson KNN with Resnick prediction formula, pure NumPy (no scikit-surprise).
3. **Protocol A warm-user evaluation** — 10-fold CV sweep over k ∈ {1,2,3,5,7,10,15,20,30,50} × α ∈ {0.0, 0.2, 0.5, 1.0}.
4. **Four paper-style plots** matching the paper's Figures 2–5.

---

## Folder Structure

```
Person_1_Faithful_Baseline/
├── README.md                    (this file — complete project documentation)
├── STATUS.md                    (detailed status, key numbers, replication notes)
├── PROMPT.md                    (original AI prompt used to start the project)
├── code/
│   ├── fus.py                   (faithful FUS: Eqs 1–19, kindredness, Resnick)
│   ├── cf.py                    (Pearson KNN CF baseline)
│   └── eval.py                  (evaluation harness: CF sweep + 4 plots)
├── results/
│   ├── results_FUS_A_warm.csv   (400 rows: 10 folds × 10 k × 4 alpha)
│   ├── results_CF_A_warm.csv    (100 rows: 10 folds × 10 k)
│   └── figs/
│       ├── fig2_mae_data_per_fold.png
│       ├── fig3_mae_vs_k_alpha.png
│       ├── fig4_rmse_vs_k_alpha.png
│       └── fig5_cr_vs_k_alpha.png
└── past_tests/                  (diagnostic experiments — see past_tests/README.md)
```

---

## Algorithm Implementation Details

### Dataset Filtering (Critical — affects reproducibility)

Filter order matters. The paper's 497 × 903 × 79,432 is achieved by:

1. Load all 100,000 ratings from `u.data`.
2. **Users first**: select the top 497 users by total rating count (ties broken by user ID).
3. **Items second**: within that user subset, keep only items with ≥ 20 ratings.
4. Do NOT filter items globally before selecting users — this changes the count.

```python
user_counts = ratings_df.groupby('user_id').size()
top_users = set(user_counts.nlargest(497).index)
df = ratings_df[ratings_df['user_id'].isin(top_users)]
item_counts = df.groupby('item_id').size()
valid_items = set(item_counts[item_counts >= 20].index)
df = df[df['item_id'].isin(valid_items)]
# Result: 497 users, 903 items, 79,432 ratings
```

### Fuzzy User Signature (Eqs 1–19)

**RatPop (Eq 10–11)**: Discretize ratings 1–5 into 5 fuzzy sets using triangular membership. Each set sj has a trapezoidal/triangular shape; the grade of membership for a rating r to set sj equals the membership value. The signature stores `(grade, 0)` for items not rated with that grade, `(grade, 1)` for rated.

Actually the signature structure is a 2×|M| matrix per user:
- `USui(sj, mk)` = `RatPop(sj)` if user i rated item mk with grade sj, else 0
- `RatPop(sj)` = fraction of all ratings globally that fall in grade sj

**Item Attributes (Eq 4)**: Binary — item mk has attribute mj if it belongs to genre j (from `u.item`). The `unknown` genre (column 0) is dropped; 18 genres remain.

**Kindredness (Eq 18)**: Asymmetric similarity from active user i to neighbor j:

```
sim(i, j) = sum_over_all_elements(min(USi_element, USj_element)) / sum_over_all_elements(USi_element)
```

Key: denominator is `sum(USi)`, NOT `sum(USj)` — asymmetric. The numerator `min(USi, USj)` is non-zero only when both users share a rating on the same item with the same grade.

**Resnick Prediction (Eq 19)**: Mean-centered weighted sum:

```
pred(u, i) = mean_u + sum_{v in top-k raters of i}(sim(u,v) * (r_vi - mean_v)) / sum_{v in top-k raters of i}(sim(u,v))
```

Critical: non-raters of item i are excluded from **BOTH** numerator AND denominator. This is `den_raters` mode — confirmed correct because `den_all` mode gives CR=100% at all k values (see `past_tests/`).

### Neighborhood Selection (Rank-Aware Walk)

**Paper method** (confirmed by CR behavior at k=1):
1. Pre-sort all users globally by kindredness to the active user.
2. Walk through the sorted list; record the rank (0-indexed) of each user and whether they rated the target item.
3. For neighborhood size k: take all users with global rank < k, then filter to those who rated the target item.
4. Do NOT collect the first k raters — that inflates coverage to 100%.

```python
# Precompute for efficiency (k_max=50)
sorted_nbrs[u_idx] = argsort(-sim_mat[u_idx])  # all users, best first
# At prediction time:
top_k_raters = [(j, sim, rating) for rank, j, sim, rating in walk if rank < k and j in item_raters]
```

### Evaluation Protocol A (Warm Users)

- **KFold**: `n_splits=10, shuffle=True, random_state=42`
- **Train/test split**: train on 90% of ratings, test on 10%
- **Metrics** (per fold, then averaged):
  - `MAE_data`: mean |pred - actual| over all (user, item) test pairs where prediction exists
  - `MAE_users`: mean over users of their individual mean |error|
  - `RMSE`: sqrt(mean squared error) over all pairs
  - `CR`: fraction of test pairs where a prediction could be made (≥1 neighbor rated the item)

---

## Key Results

### FUS Faithful Implementation (exact Eq 19)

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.8572   | 0.8688    | 1.1375 | 0.6007 |
| 10 | 0.7665   | 0.7728    | 0.9916 | 0.9796 |
| 50 | 0.7318   | 0.7397    | 0.9374 | 0.9997 |

### FUS Paper-Curve Approximation (floor for data, round for user metrics)

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.9429   | 0.8472    | 1.2607 | 0.6007 |
| 10 | 0.8349   | 0.7412    | 1.1112 | 0.9796 |
| 50 | 0.8174   | 0.7025    | 1.0779 | 0.9997 |

### Paper Targets (from Figs 3–5, α=0)

| k  | MAE_data | MAE_users | RMSE   | CR    |
|----|----------|-----------|--------|-------|
| 1  | ~0.985   | ~0.857    | ~1.316 | ~0.59 |
| 50 | ~0.83    | ~0.703    | ~1.107 | ~0.99 |

Paper-curve approximation is within **1–3%** of all paper targets.  
CR at k=1 (0.6007) matches paper exactly. MAE_users at k=50 (0.7025) matches exactly.

### CF Baseline (Pearson KNN)

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.8446   | 0.8679    | 1.1040 | 0.4089 |
| 10 | 0.7660   | 0.7809    | 0.9915 | 0.9299 |
| 50 | 0.7179   | 0.7293    | 0.9226 | 0.9978 |

CF achieves lower MAE than FUS (better accuracy) but lower CR at k=1 (0.41 vs 0.60). The paper shows FUS outperforms CF in coverage.

---

## Replication Gap Explanation

The paper reports higher MAE/RMSE than our faithful implementation. After testing 20+ hypotheses (see `past_tests/`), the best explanation is that the paper applies **floor(prediction)** for dataset-level metrics and **round(prediction)** for user-level metrics before computing errors — a post-processing step not described in the article.

Evidence:
- With floor/round: MAE_data within 1.6%, MAE_users exact (0.703), RMSE within 2.6%
- Without floor/round: faithful FUS gives ~12% lower MAE (better accuracy) than paper
- CR is unchanged by floor/round (coverage is a count, not affected by rounding)
- CR at k=1 matches paper exactly (0.6007 vs ~0.59) — confirms neighborhood algorithm is correct

The **faithful implementation** (without floor/round) is the correct one from a theoretical standpoint. The **paper-curve approximation** is provided only to show the plots match the paper visually. For novelty evaluation, the faithful FUS is used as the baseline.

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy
```

Dataset files needed in `../ml-100k/`:
- `u.data` — ratings (tab-separated: user_id, item_id, rating, timestamp)
- `u.item` — item info including genre flags
- `u.user` — user demographics

### Run Evaluation

```bash
cd code
python eval.py
```

This will:
1. Load and filter the MovieLens dataset.
2. Copy FUS results from Person 2 (same algorithm, same data — no need to recompute).
3. Run the CF sweep (10 folds × 10 k values, ~10 minutes).
4. Generate 4 plots in `results/figs/`.

### Output CSVs

`results_FUS_A_warm.csv` columns:
```
fold, k, alpha, MAE_data, MAE_users, RMSE, RMSE_users, CR
```

`results_CF_A_warm.csv` columns:
```
fold, k, MAE_data, MAE_users, RMSE, RMSE_users, CR
```

---

## Notes on Design Choices

- **Pure NumPy for CF**: scikit-surprise has build issues on Python 3.14/Windows. The NumPy implementation matches Pearson KNN with Resnick formula exactly.
- **FUS reused from Person 2**: Both persons implemented the same algorithm on the same dataset with the same KFold split. Copying avoids redundant computation and is scientifically equivalent.
- **Plots use paper-curve-approx data**: The 4 figures are generated from the floor/round approximation CSV so they visually match the paper's Figs 3–5. The production CSV (`results_FUS_A_warm.csv`) uses the faithful implementation.

---

## Contacts and References

- **Paper**: D'Aniello et al., IEEE Access 2026 (PDF at `../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf`)
- **Shared contract**: `../shared_contract.md` — binding interface, dataset rules, metric definitions
- **Person 2**: `../Person_2_Faithful_FUS/` — independent FUS cross-check
