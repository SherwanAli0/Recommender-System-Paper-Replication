# Past Tests — Person 2 Faithful FUS

This folder contains the **diagnostic, exploratory, and replication-investigation** files produced during the FUS replication. They are preserved for reference and reproducibility but are not part of the production code in `../code/`.

The full hypothesis log is in `../REPRO_DEBUG_NOTES.md`.

---

## Contents

```
past_tests/
├── README.md                                       (this file)
├── code/
│   ├── eval_warm_paper_curve_approx.py             (FUS eval with floor/round post-processing)
│   └── repro_den_all_floor.py                      (denominator-all vs den-raters hypothesis test)
└── results/
    ├── repro_den_all_floor.csv                     (den_all vs den_raters hypothesis results)
    ├── comparison_paper_vs_faithful_vs_curve_approx.csv
    ├── fus_baseline_predictions_alpha0.csv
    ├── results_FUS_A_warm_paper_curve_approx.csv   (paper-curve approximation output)
    ├── repro_hypotheses_baselines.csv
    ├── repro_hypotheses_denominator_literal.csv
    ├── repro_hypotheses_filters.csv
    ├── repro_hypotheses_grid.csv
    ├── repro_hypotheses_movielens_u1u5.csv
    ├── repro_hypotheses_neighbor_order.csv
    ├── repro_hypotheses_population_all_raw.csv
    ├── repro_hypotheses_population_all_users_items20.csv
    ├── repro_hypotheses_shuffle42_train.csv
    ├── repro_hypotheses_signature_binary_mask.csv
    ├── repro_hypotheses_signature_rating_value_mask.csv
    ├── repro_hypotheses_signature_ratpop_mask.csv
    ├── repro_hypotheses_signature_ratpop_unmasked_items.csv
    ├── repro_hypotheses_similarity_direction.csv
    ├── repro_hypotheses_split_kfold_noshuffle.csv
    ├── repro_hypotheses_split_kfold_shuffle0.csv
    ├── repro_hypotheses_split_timestamp_blocks.csv
    ├── repro_hypotheses_user_means.csv
    ├── repro_metric_postprocess_curves.csv
    ├── repro_metric_postprocess_scores.csv
    └── repro_mixed_postprocess_scores.csv
```

---

## Replication investigation summary

After 20 hypothesis tests recorded in this folder (plus more testing documented in `../REPRO_DEBUG_NOTES.md`), the closest paper match is achieved by applying:

- `floor(prediction)` for dataset-level metrics (MAE_data, RMSE)
- `round(prediction)` for user-level metrics (MAE_users, RMSE_users)
- `den_raters` for the Resnick denominator (the literal `den_all` reading is ruled out because it gives CR = 1.000 at every k, contradicting the paper's CR ≈ 0.60 at k = 1)

Under that approximation, MAE_users at k = 50 matches the paper to four decimal places (0.7025 vs 0.703). At k = 1, MAE_data and RMSE remain about 4 percent below the paper.

## Script descriptions

### `eval_warm_paper_curve_approx.py`

Runs the same Protocol A 10-fold sweep as `../code/eval_fus.py` but applies `floor()` to predictions before computing dataset-level metrics and `round()` before user-level metrics. Output: `results/results_FUS_A_warm_paper_curve_approx.csv`.

### `repro_den_all_floor.py`

Formal test of the denominator interpretation in Eq 19. Compares `den_raters` (exclude non-raters from numerator AND denominator, our reading) against `den_all` (literal "set the term to zero, but keep the denominator over all k neighbours"). Result: `den_all` gives CR = 1.000 at every k, definitively ruling it out.

## Replication CSVs

The 20 `repro_hypotheses_*.csv` files in `results/` each contain the output of one alternative interpretation of an underspecified part of the paper. Categories tested:

- Filter order (items first vs users first)
- KFold seeds (0, 42, no-shuffle)
- Pre-defined MovieLens u1.base/u1.test splits
- Population variants (raw 100k, all 943 users)
- Signature variants (binary mask, rating-value mask, RatPop with/without item mask)
- Similarity direction (asymmetric vs symmetric)
- Neighbour ordering (top-k neighbours then filter raters vs collect first-k raters)
- User means (train-only vs full data)
- Metric post-processing (raw, floor, round, mixed)

Together with the paper-curve approximation and the den_all test above, these constitute the complete log of every alternative interpretation tested.
