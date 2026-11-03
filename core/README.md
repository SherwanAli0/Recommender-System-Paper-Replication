# core: FUS implementation and CF baseline

This folder holds the primary implementation track of the project (Software roles in [CONTRIBUTORS.md](../CONTRIBUTORS.md)).

- **FUS**: Python implementation of the Fuzzy User Signature recommender from D'Aniello et al. (IEEE Access 2026), Equations 1 to 19. The FUS results CSV in `results/` is a copy of the one produced by [`../cross_check/`](../cross_check/) (see "Notes on design choices").
- **CF**: a classical Pearson-KNN collaborative-filtering baseline using the same Resnick prediction formula as FUS.
- **Protocol A**: warm-user evaluation under the project's [shared_contract.md](../shared_contract.md), 10-fold cross-validation. FUS sweeps k in {1, 2, 4, ..., 48, 50} (26 values, paper Section V.C.2) and alpha in {0.0, 0.7, 0.8, 0.9}; CF sweeps the baseline set k in {1, 2, 4, 6, 8, 10, 15, 20, 30, 50}.

## Contents

```
core/
├── README.md                       this file
├── code/
│   ├── fus.py                      FUS: Eqs 1 to 19, kindredness, Resnick
│   ├── cf.py                       Pearson KNN CF baseline (pure NumPy)
│   └── eval.py                     evaluation harness: copies the FUS CSV, runs the CF sweep, draws 4 figures
├── results/
│   ├── results_FUS_A_warm.csv      1040 rows: 10 folds x 26 k x 4 alpha (copy of the cross_check CSV)
│   ├── results_CF_A_warm.csv       100 rows: 10 folds x 10 k
│   └── figs/                       fig2 to fig5, drawn from the faithful FUS CSV
└── past_tests/                     diagnostic experiments (see past_tests/README.md)
```

## Algorithm details

### Dataset filtering (critical for reproducibility)

Filter order matters. The paper's 497 x 903 x 79,432 shape is achieved by:

1. Load all 100,000 ratings from `u.data`.
2. Users first: keep the top 497 users by total rating count in the full table.
3. Items second: within that user subset, keep only items with at least 20 ratings.
4. Do not filter items globally before selecting users; that changes the count (the paper's wording lists the item filter first, but that order does not give the paper's shape).

```python
user_counts = ratings_df["user_id"].value_counts()
top_users = user_counts.nlargest(497).index
df = ratings_df[ratings_df["user_id"].isin(top_users)]
item_counts = df["item_id"].value_counts()
valid_items = item_counts[item_counts >= 20].index
df = df[df["item_id"].isin(valid_items)]
# asserted in the loader: 497 users, 903 items, 79,432 ratings
```

How ties at the 497th user are broken is whatever pandas does for `value_counts().nlargest`; it is not specified by user ID.

### Fuzzy User Signature (Eqs 1 to 19)

- **Mat_ui (Eq 8 to 9 context):** for each user, a 5 x n_items 0/1 matrix with a 1 where the user gave rating s_j to item m_k.
- **RatPop (Eqs 10 to 11):** `RatPop(s_j) = SF(s_j) / max_s SF(s)`, where `SF(s_j)` is the number of items the user rated with `s_j`. The denominator is the per-user maximum.
- **ItemAttr (Eqs 12 to 13):** `SI(m_k) / max_m SI(m)`. In MovieLens a user rates an item at most once, so ItemAttr is 0 or 1. Genres from `u.item` are not used by FUS.
- **User signature (Eqs 14 to 15):** `US(s_j, m_k) = min(RatPop(s_j), ItemAttr(m_k))`, kept only on the marked (rated) cells.
- **Kindredness (Eq 18):** asymmetric similarity from active user i to neighbor j: `sim(i, j) = sum(min(US_i, US_j)) / sum(US_i)` after the alpha-cut. The denominator is `sum(US_i)`, not `sum(US_j)`.
- **Resnick prediction (Eq 19):** mean-centred weighted sum over the neighbors in the top-k that rated the item; non-raters are excluded from numerator and denominator. Predictions are not clipped to [1, 5] (changed in v1.1.0). The "all-neighbors" denominator reading gives CR = 100 percent at every k, which contradicts the paper's CR of about 0.60 at k=1, so it is not used.

### Neighborhood selection (rank-aware walk)

1. Pre-sort all users globally by kindredness to the active user (self excluded).
2. Walk through the sorted list; record the rank (0-indexed) of each user and whether they rated the target item.
3. For neighborhood size k: take all users with global rank below k, then filter to those who rated the target item.
4. Do not collect the first k raters; that inflates coverage to 100 percent.

### Evaluation Protocol A (warm users)

- KFold: `n_splits=10, shuffle=True, random_state=42`.
- Train and test split: train on 90 percent of ratings, test on 10 percent.
- Metrics (per fold, then averaged over folds):
  - `MAE_data`: mean absolute error over the test pairs for which a prediction was made.
  - `MAE_users`: mean over users of their individual mean absolute error.
  - `RMSE`: root-mean-square error over the same pairs; `RMSE_users` is the per-user analogue.
  - `CR`: fraction of test pairs for which a prediction could be made (at least one top-k neighbor rated the item). It is a fraction in [0, 1].
  - MAE and RMSE are in rating points, lower is better; CR is higher-is-better. Because MAE is computed only over predicted pairs, systems with different CR are compared on different sets of pairs.

## Key results (means over folds, alpha = 0)

### FUS faithful implementation (exact Eq 19, committed run)

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.8843   | 0.9008    | 1.1590 | 0.6007 |
| 10 | 0.7718   | 0.7788    | 0.9957 | 0.9796 |
| 50 | 0.7333   | 0.7415    | 0.9383 | 0.9997 |

### FUS floor/round approximation (historical diagnostic run)

Floor of the prediction for MAE_data and RMSE, round for MAE_users, on predictions clipped to [1, 5]. Produced 2026-05-09, before the v1.1.0 audit (10 k values), and not regenerated.

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.9429   | 0.8472    | 1.2607 | 0.6007 |
| 10 | 0.8349   | 0.7412    | 1.1112 | 0.9796 |
| 50 | 0.8174   | 0.7025    | 1.0779 | 0.9997 |

### Paper targets (from the paper's text, alpha=0)

| k  | MAE_data | MAE_users | RMSE   | CR    |
|----|----------|-----------|--------|-------|
| 1  | 0.985    | 0.857     | 1.316  | slightly below 0.60 |
| 50 | 0.83     | 0.703     | 1.107  | around 0.99 |

The floor/round run is within 3 percent of the paper's k=50 values and about 4 percent low on MAE_data and RMSE at k=1. The k=50 MAE_users value (0.7025, mean 0.70254) agrees with the paper's 0.703 at the three decimals the paper reports; it is not a four-decimal match. The faithful run is 10 to 15 percent below the paper on MAE_data and RMSE and about 5 percent above it on MAE_users. CR is close to the paper's curve in both.

### CF baseline (Pearson KNN, committed run)

This run predates the removal of prediction clipping (v1.1.0), so its predictions were clipped to [1, 5] and a re-run will differ slightly.

| k  | MAE_data | MAE_users | RMSE   | CR     |
|----|----------|-----------|--------|--------|
| 1  | 0.8446   | 0.8679    | 1.1040 | 0.4089 |
| 10 | 0.7659   | 0.7809    | 0.9915 | 0.9299 |
| 50 | 0.7179   | 0.7293    | 0.9226 | 0.9978 |

CF has lower MAE_users than FUS at k=1 and at k=50 and is within about 0.004 of FUS for k from 4 to 10, but its CR is much lower at small k (0.41 versus 0.60 at k=1). The paper's own CF is much worse than FUS (MAE_users about 0.93 at k=10), so this CF is stronger than the paper's.

## Replication gap explanation

The faithful implementation differs from the paper's reported error values (see above). One post-processing that brings the k=50 numbers close is `floor(prediction)` for dataset-level metrics and `round(prediction)` for user-level metrics. The paper does not describe such a step, so this is a hypothesis found by trial, not an established explanation, and with it PF and GIM overshoot the paper. CR is unaffected by it. See [`../README.md`](../README.md) "Replication notes" and `../cross_check/REPRO_DEBUG_NOTES.md`.

## How to run

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

Dataset files needed in `../ml-100k/` (the repository root, one level above `core/`):
- `u.data`: ratings (tab-separated: user_id, item_id, rating, timestamp)

### Run evaluation

Run `cross_check/code/eval_fus.py` first (or keep its committed CSV), then:

```bash
cd code
python eval.py
```

This will:

1. Check that `../../cross_check/results/protocol_A/results_FUS_A_warm.csv` exists (exit code 1 if it does not) and copy it to `results/results_FUS_A_warm.csv`.
2. Load and filter the MovieLens dataset.
3. Run the CF sweep (10 folds x 10 k values; the original authors reported about 10 minutes, not re-measured).
4. Overwrite the four figures in `results/figs/` from the faithful FUS CSV.

### Output CSVs

Both CSVs use the columns of `../shared_contract.md` Section 5:
```
system, protocol, fold, k, alpha, MAE_data, MAE_users, RMSE, RMSE_users, CR
```

## Notes on design choices

- Pure NumPy for CF: scikit-surprise has build issues on Python 3.14 and Windows. The NumPy implementation follows Pearson KNN with the Resnick formula (mean-centred weighted sum).
- FUS reused from `cross_check/`: `eval.py` copies the cross_check FUS CSV instead of recomputing it, so the two committed FUS CSVs are identical by construction. Agreement of the core and cross_check FUS code is checked separately on a synthetic fixture in `../tests/test_sanity.py`.
- Figures: `results/figs/` holds figures drawn from the faithful CSV (what `eval.py` produces). The figures drawn from the floor/round run, annotated with the paper's values, are in `past_tests/results/figs/` (made by `past_tests/code/regenerate_plots_approx.py`).

## References

- Original paper: D'Aniello et al., IEEE Access 2026 (PDF at `../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf`).
- Shared contract: `../shared_contract.md`.
- Second FUS implementation: `../cross_check/`.
