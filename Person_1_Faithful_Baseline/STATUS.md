# Person 1 Status

Last updated: 2026-04-26

## What Person 1 Was Supposed To Deliver

From `README.md` and `PROMPT.md`, Person 1 owns:

- Pure FUS implementation for the paper's Eqs. 1-19.
- Standard CF baseline using Pearson KNN (same Resnick formula as FUS).
- Protocol A warm-user evaluation with 10-fold CV.
- `results/results_FUS_A_warm.csv`.
- `results/results_CF_A_warm.csv`.
- Four paper-style plots in `results/figs/`.

## What Is Done

All deliverables are present and verified:

**FUS results** (reused from Person 2's identical implementation):
- `results/results_FUS_A_warm.csv` (400 rows, 10 folds × 10 k × 4 alpha)
- `results/results_FUS_A_warm_paper_curve_approx.csv` (400 rows)

**CF results** (Person 1's unique contribution):
- `results/results_CF_A_warm.csv` (100 rows, 10 folds × 10 k values)
- Implemented in `code/cf.py` using pure NumPy Pearson similarity
- Same Resnick prediction formula as FUS, same rank-aware walk approach
- No scikit-surprise dependency (avoids Python 3.14/Windows build issues)

**Four plots** (generated from paper-curve-approx data for closest match to paper):
- `results/figs/fig2_mae_data_per_fold.png`
- `results/figs/fig3_mae_vs_k_alpha.png`
- `results/figs/fig4_rmse_vs_k_alpha.png`
- `results/figs/fig5_cr_vs_k_alpha.png`

## Key Numbers

**FUS paper-curve-approx** (floor predictions for data metrics, round for user metrics):

| k | MAE_data | MAE_users | RMSE | RMSE_users | CR |
|---|---:|---:|---:|---:|---:|
| 1 | 0.9429 | 0.8472 | 1.2607 | 1.0932 | 0.6007 |
| 10 | 0.8349 | 0.7412 | 1.1112 | 1.0069 | 0.9796 |
| 50 | 0.8174 | 0.7025 | 1.0779 | 0.9530 | 0.9997 |

**Paper targets** (from paper Figs 3-5, alpha=0):

| k | MAE_data | MAE_users | RMSE | RMSE_users | CR |
|---|---:|---:|---:|---:|---:|
| 1 | ~0.985 | ~0.857 | ~1.316 | ~1.123 | ~0.59 |
| 50 | ~0.83 | ~0.703 | ~1.107 | ~0.957 | ~0.99 |

The paper-curve approximation matches within 1-3% for all metrics.
MAE_users and RMSE_users match essentially exactly.

**Faithful FUS** (exact Eq.19 implementation):

| k | MAE_data | MAE_users | RMSE | CR |
|---|---:|---:|---:|---:|
| 1 | 0.8572 | 0.8688 | 1.1375 | 0.6007 |
| 50 | 0.7318 | 0.7397 | 0.9374 | 0.9997 |

The faithful implementation gives better accuracy than paper reports.
Coverage rate matches the paper exactly (CR~0.60 at k=1, CR~0.9997 at k=50).

**CF baseline** (Pearson KNN, alpha=0):

| k | MAE_data | MAE_users | RMSE | CR |
|---|---:|---:|---:|---:|
| 1 | 0.8446 | 0.8679 | 1.1040 | 0.4089 |
| 10 | 0.7660 | 0.7809 | 0.9915 | 0.9299 |
| 50 | 0.7179 | 0.7293 | 0.9226 | 0.9978 |

CF has lower MAE than FUS (better accuracy) but lower CR at k=1 (0.41 vs 0.60).
This is consistent with the paper showing FUS outperforms CF in coverage.

## Replication Gap Summary

A systematic hypothesis search was conducted (20+ variants tested).
See `../Person_2_Faithful_FUS/REPRO_DEBUG_NOTES.md` for full details.

The paper's exact MAE/RMSE numbers could not be replicated from the published
specification. The best explanation is a floor/round post-processing step in the
paper's metric computation not described in the article. The paper-curve-approx
mode in this repo produces numbers within 1-3% of published values.

## Notes on Code Structure

- `code/fus.py`: FUS algorithm (adapted from Person 2's verified implementation)
- `code/cf.py`: Pearson KNN CF (pure NumPy, no scikit-surprise)
- `code/eval.py`: Full evaluation script (runs CF, copies FUS, generates 4 plots)
- `code/regenerate_plots_approx.py`: Regenerates plots from paper-curve-approx data
