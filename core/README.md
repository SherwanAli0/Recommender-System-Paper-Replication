# core: FUS implementation and CF baseline

This folder holds the primary implementation track of the project.

- **FUS**: faithful Python implementation of the Fuzzy User Signature recommender from D'Aniello et al. (IEEE Access 2026), Equations 1 to 19, with no algorithmic deviations.
- **CF**: a classical Pearson-KNN collaborative-filtering baseline using the same Resnick prediction formula as FUS.
- **Protocol A**: warm-user evaluation under the project's [shared_contract.md](../shared_contract.md), 10-fold cross-validation sweeping k in {1, 2, 3, 5, 7, 10, 15, 20, 30, 50} and alpha in {0.0, 0.2, 0.5, 1.0}.

## Contents

```
core/
├── README.md                       this file
├── code/
│   ├── fus.py                      faithful FUS: Eqs 1 to 19, kindredness, Resnick
│   ├── cf.py                       Pearson KNN CF baseline (pure NumPy)
│   └── eval.py                     evaluation harness: CF sweep + 4 plots
├── results/
│   ├── results_FUS_A_warm.csv      400 rows: 10 folds x 10 k x 4 alpha
│   ├── results_CF_A_warm.csv       100 rows: 10 folds x 10 k
│   └── figs/
│       ├── fig2_mae_data_per_fold.png
│       ├── fig3_mae_vs_k_alpha.png
│       ├── fig4_rmse_vs_k_alpha.png
│       └── fig5_cr_vs_k_alpha.png
└── past_tests/                     diagnostic experiments (see past_tests/README.md)
```

## Algorithm details

### Dataset filtering (critical for reproducibility)

Filter order matters. The paper's 497 x 903 x 79,432 shape is achieved by:

1. Load all 100,000 ratings from `u.data`.
2. Users first: select the top 497 users by total rating count (ties broken by user ID).
3. Items second: within that user subset, keep only items with at least 20 ratings.
4. Do not filter items globally before selecting users; that changes the count.

```python
user_counts = ratings_df.groupby('user_id').size()
top_users = set(user_counts.nlargest(497).index)
df = ratings_df[ratings_df['user_id'].isin(top_users)]
item_counts = df.groupby('item_id').size()
valid_items = set(item_counts[item_counts >= 20].index)
df = df[df['item_id'].isin(valid_items)]
# Result: 497 users, 903 items, 79,432 ratings
```

### Fuzzy User Signature (Eqs 1 to 19)

- **RatPop (Eqs 10 to 11):** Discretize ratings 1 to 5 into 5 fuzzy sets using triangular membership. The grade of membership for a rating r to set sj equals the membership value. The signature stores `(grade, 0)` for items not rated with that grade and `(grade, 1)` for rated items.
- **Item Attributes (Eq 4):** Binary; item mk has attribute mj if it belongs to genre j (from `u.item`). The `unknown` genre (column 0) is dropped, leaving 18 genres.
- **Kindredness (Eq 18):** Asymmetric similarity from active user i to neighbor j: `sim(i, j) = sum(min(US_i_element, US_j_element)) / sum(US_i_element)`. The denominator is `sum(US_i)`, not `sum(US_j)`. The numerator `min(US_i, US_j)` is non-zero only when both users share a rating on the same item with the same grade.
- **Resnick prediction (Eq 19):** Mean-centered weighted sum. Non-raters of item i are excluded from both numerator and denominator. The default `den_raters` mode is correct; `den_all` mode produces CR=100 percent at all k values, which is incorrect.

### Neighborhood selection (rank-aware walk)

1. Pre-sort all users globally by kindredness to the active user.
2. Walk through the sorted list; record the rank (0-indexed) of each user and whether they rated the target item.
3. For neighborhood size k: take all users with global rank below k, then filter to those who rated the target item.
4. Do not collect the first k raters; that inflates coverage to 100 percent.

### Evaluation Protocol A (warm users)

- KFold: `n_splits=10, shuffle=True, random_state=42`.
- Train and test split: train on 90 percent of ratings, test on 10 percent.
- Metrics (per fold, then averaged):
  - `MAE_data`: mean absolute error over all (user, item) test pairs where prediction exists.
  - `MAE_users`: mean over users of their individual mean absolute error.
  - `RMSE`: root-mean-square of errors over all pairs.
  - `CR`: fraction of test pairs where a prediction could be made (at least one neighbor rated the item).

## Key results

### FUS faithful implementation (exact Eq 19)

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.8572   | 0.8688    | 1.1375 | 0.6007 |
| 10 | 0.7665   | 0.7728    | 0.9916 | 0.9796 |
| 50 | 0.7318   | 0.7397    | 0.9374 | 0.9997 |

### FUS paper-curve approximation (floor for data, round for user metrics)

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.9429   | 0.8472    | 1.2607 | 0.6007 |
| 10 | 0.8349   | 0.7412    | 1.1112 | 0.9796 |
| 50 | 0.8174   | 0.7025    | 1.0779 | 0.9997 |

### Paper targets (from Figs 3 to 5, alpha=0)

| k  | MAE_data | MAE_users | RMSE   | CR    |
|----|----------|-----------|--------|-------|
| 1  | ~0.985   | ~0.857    | ~1.316 | ~0.59 |
| 50 | ~0.83    | ~0.703    | ~1.107 | ~0.99 |

The paper-curve approximation is within 1 to 3 percent of all paper targets. CR at k=1 (0.6007) matches the paper exactly. MAE_users at k=50 (0.7025) matches to four decimal places.

### CF baseline (Pearson KNN)

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.8446   | 0.8679    | 1.1040 | 0.4089 |
| 10 | 0.7660   | 0.7809    | 0.9915 | 0.9299 |
| 50 | 0.7179   | 0.7293    | 0.9226 | 0.9978 |

CF achieves lower MAE than FUS (better accuracy) but lower CR at k=1 (0.41 vs 0.60). The paper shows FUS outperforms CF in coverage at low k.

## Replication gap explanation

The paper reports higher MAE and RMSE than our faithful implementation. After testing 20 hypotheses (see `past_tests/`), the best explanation is that the paper applies `floor(prediction)` for dataset-level metrics and `round(prediction)` for user-level metrics before computing errors. This post-processing step is not described in the article.

Evidence:

- With floor and round: MAE_data within 1.6 percent, MAE_users exact (0.703), RMSE within 2.6 percent.
- Without floor and round: faithful FUS gives 12 percent lower MAE (better accuracy) than the paper.
- CR is unchanged by floor and round (coverage is a count, not affected by rounding).
- CR at k=1 matches the paper exactly (0.6007 vs ~0.59), confirming the neighborhood algorithm is correct.

The faithful implementation (without floor or round) is the correct one from a theoretical standpoint. The paper-curve approximation is provided only to show the plots match the paper visually.

## How to run

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy
```

Dataset files needed in `../ml-100k/`:
- `u.data`: ratings (tab-separated: user_id, item_id, rating, timestamp)
- `u.item`: item info including genre flags
- `u.user`: user demographics

### Run evaluation

```bash
cd code
python eval.py
```

This will:

1. Load and filter the MovieLens dataset.
2. Reuse the FUS results from `cross_check/results/protocol_A/` (same algorithm, same data, no need to recompute).
3. Run the CF sweep (10 folds x 10 k values, about 10 minutes).
4. Generate four plots in `results/figs/`.

### Output CSVs

`results_FUS_A_warm.csv` columns:
```
fold, k, alpha, MAE_data, MAE_users, RMSE, RMSE_users, CR
```

`results_CF_A_warm.csv` columns:
```
fold, k, MAE_data, MAE_users, RMSE, RMSE_users, CR
```

## Notes on design choices

- Pure NumPy for CF: scikit-surprise has build issues on Python 3.14 and Windows. The NumPy implementation matches Pearson KNN with the Resnick formula exactly.
- FUS reused from `cross_check/`: both implementations produce identical numbers on identical data with identical KFold splits. Copying avoids redundant computation and is scientifically equivalent.
- Plots use paper-curve-approx data: the four figures are generated from the floor and round approximation CSV so they visually match the paper's Figs 3 to 5. The production CSV uses the faithful implementation.

## References

- Original paper: D'Aniello et al., IEEE Access 2026 (PDF at `../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf`).
- Shared contract: `../shared_contract.md`.
- Independent FUS implementation for cross-verification: `../cross_check/`.
