# Reproduction Debug Notes

Last updated: 2026-04-26

Goal: explain why the current FUS implementation matches the paper's coverage
curve but not the paper's reported accuracy values.

## External Search

Public searches were run for:

- `"A Recommendation System Based on Fuzzy Signature" GitHub`
- `"Fuzzy User Signature" recommender GitHub`
- `"RatPop" "ItemAttr"`
- `"User Kindredness" recommender`
- author/title combinations for D'Aniello, Della Corte, Gaeta, and Aliberti.

Findings:

- No public implementation repository was found.
- GitHub repository search returned no obvious matching project.
- GitHub code search requires authentication, so unauthenticated exhaustive code
  search was not possible from this environment.
- The earlier 2024 conference version, `A Fuzzy Signature-Based Approach for
  Recommendation Systems`, is listed in IRIS, but IRIS has no attached file.

## Paper Details Reconfirmed

From the local PDF:

- Dataset text says MovieLens 100k, remove movies with fewer than 20 votes, keep
  top 497 users.
- Figure/table target is 497 users, 903 items, 79,432 ratings.
- Top-k neighbors are selected from all users by highest similarity.
- Eq. 19 prints the Resnick denominator over the top-k neighborhood.
- Text says if a neighbor did not rate item `m`, the summation term is set to zero.
- Figure 5 coverage starts slightly below 60% at k=1 and reaches about 99% at k=50.

The current implementation matches the coverage profile:

- k=1: CR about 0.6007
- k=50: CR about 0.9997

## Key Unresolved Mismatch

Paper text/figure says, for alpha=0:

- MAE_data goes from about 0.985 at k=1 to 0.83 at k=50.
- MAE_users goes from about 0.857 at k=1 to 0.703 at k=50.
- RMSE goes from about 1.316 at k=1 to 1.107 at k=50.
- RMSE_users goes from about 1.123 at k=1 to 0.957 at k=50.

Current faithful implementation gives:

- k=1: MAE_data=0.8572, MAE_users=0.8688, RMSE=1.1375, CR=0.6007
- k=50: MAE_data=0.7318, MAE_users=0.7397, RMSE=0.9374, CR=0.9997

So coverage is right, but the predictions are more accurate than the paper.

## Hypotheses Tested

CSV outputs are in `results/`.

Tested and rejected as full explanations:

- Prediction clipping vs no clipping:
  - `repro_hypotheses_shuffle42_train.csv`
  - No clipping barely changes k=50.
- Denominator over rated neighbors vs all top-k neighbors:
  - `repro_hypotheses_denominator_literal.csv`
  - Literal denominator does not reach the paper's k=50 accuracy values.
- Top-k raters instead of top-k neighbors:
  - Inflates CR to 1.0 and is not paper-faithful.
- Signature variants:
  - `repro_hypotheses_signature_*.csv`
  - `repro_hypotheses_grid.csv`
  - Unmasked item signatures can match some k=1 error magnitudes but fail other
    metrics and are contradicted by the paper's "marked cells" text.
- Similarity direction and symmetric variants:
  - `repro_hypotheses_similarity_direction.csv`
  - Reversing asymmetry matches one point around k=10 but breaks coverage/shape.
- Fold construction:
  - `repro_hypotheses_split_*.csv`
  - Shuffled KFold, unshuffled KFold, timestamp blocks all fail to explain it.
- MovieLens official `u1`-`u5` splits:
  - `repro_hypotheses_movielens_u1u5.csv`
  - Does not match.
- Filter order:
  - `repro_hypotheses_filters.csv`
  - Only user-first then item-filter gives 497 x 903 x 79,432; other filters do
    not explain accuracy.
- User population:
  - `repro_hypotheses_population_all_users_items20.csv`
  - `repro_hypotheses_population_all_raw.csv`
  - Including all users does not create the paper's MAE_data/MAE_users split.
- User mean rounding:
  - `repro_hypotheses_user_means.csv`
  - Rounding/floor/ceil increases error but not enough.
- Ascending/random neighbor order:
  - `repro_hypotheses_neighbor_order.csv`
  - Does not match the coverage/accuracy combination.

## Denominator-All Test (2026-04-26 update)

Tested whether the denominator in Resnick Eq.19 should sum over ALL k neighbors
(not just raters), because the paper says "the whole term of the summation is set
to zero" for non-raters — implying the denominator sums the full neighborhood.

Results from `results/repro_den_all_floor.csv`:

| variant | k=1 MAEdata | k=50 MAEdata | k=1 RMSE | k=50 RMSE | k=1 CR | k=50 CR |
|---------|-------------|--------------|-----------|-----------|--------|---------|
| den_raters_raw  | 0.857 | 0.732 | 1.138 | 0.937 | 0.601 | 1.000 |
| den_all_raw     | 0.858 | 0.757 | 1.113 | 0.962 | 1.000 | 1.000 |
| den_raters_round| 0.943 | 0.817 | 1.261 | 1.078 | 0.601 | 1.000 |
| den_all_floor   | 0.928 | 0.833 | 1.223 | 1.084 | 1.000 | 1.000 |

Key finding: den_all gives CR=1.000 at ALL k values (including k=1) because it
always produces a prediction (= user mean when no raters found). The paper clearly
shows CR~60% at k=1. This definitively rules out den_all. The paper's "whole term
set to zero" means both numerator AND denominator terms are excluded for non-raters,
i.e., den_raters is the correct faithful implementation.

The den_all_floor result (MAEdata=0.833) is a coincidence — matching MAEdata only
while having wrong CR. Do not use den_all in the faithful implementation.

## Mixed-Protocol / Reverse-Engineering Lead

A later test generated a per-prediction table:

- `results/fus_baseline_predictions_alpha0.csv`

Then it tested post-processing and metric mixtures:

- `results/repro_metric_postprocess_scores.csv`
- `results/repro_metric_postprocess_curves.csv`
- `results/repro_mixed_postprocess_scores.csv`

The closest reverse-engineered mixed explanation so far is:

- Compute `MAE_data` and dataset-level `RMSE` after applying `floor(prediction)`.
- Compute `MAE_users` and `RMSE_users` after applying `round(prediction)`.
- Keep coverage from the faithful top-k-neighbor prediction availability.

This gives:

| k | MAE_data | MAE_users | RMSE | RMSE_users | CR |
|---|---:|---:|---:|---:|---:|
| 1  | 0.9430 | 0.8472 | 1.2607 | 1.1092 | 0.6007 |
| 10 | 0.8349 | 0.7412 | 1.1112 | 1.0069 | 0.9796 |
| 50 | 0.8174 | 0.7026 | 1.0779 | 0.9530 | 0.9997 |

Compared with the article landmarks:

| k | MAE_data | MAE_users | RMSE | RMSE_users | CR |
|---|---:|---:|---:|---:|---:|
| 1  | ~0.985 | ~0.857 | ~1.316 | ~1.123 | ~0.59 |
| 50 | ~0.83  | ~0.703 | ~1.107 | ~0.957 | ~0.99 |

This is the first hypothesis that comes close to both the upper dataset-level
curves and the lower user-level curves at the same time.  It is not a clean
paper-faithful protocol; it is a reverse-engineered mixed plotting/metric
hypothesis.  Do not silently use it as the baseline unless the report clearly
labels it as "paper-curve approximation" or "reverse-engineered plot match."

## Current Best Assessment (updated 2026-04-26)

The paper_curve_approx (floor for data metrics, round for user metrics) is the
best approximation found.  At k=50, alpha=0:

| Metric | Faithful impl | Paper_curve_approx | Paper reported |
|--------|--------------|-------------------|----------------|
| MAE_data | 0.732 | **0.817** | 0.830 |
| MAE_users | 0.740 | **0.703** | 0.703 |
| RMSE | 0.937 | **1.078** | 1.107 |
| RMSE_users | 0.910 | **0.953** | 0.957 |
| CR | 0.9997 | 0.9997 | ~0.99 |

MAE_users is essentially exact (0.703 vs 0.703).  MAE_data is within 1.6%.
RMSE is within 2.6%.  RMSE_users is within 0.4%.  CR matches.

The den_all variant (all k neighbors in denominator) was also tested and gives
MAEdata=0.833 at k=50, which would match the paper — but its CR=1.000 at all k
values is incompatible with the paper's observed CR~0.60 at k=1.  den_all is
ruled out.

The most likely explanations for the remaining gap:
1. The paper's reported accuracy curves were produced with a floor/round
   post-processing step not specified in the article.
2. The paper's numerical values may be slightly inaccurate as read from the
   figures (±0.02 is typical for figure-to-number reading).
3. The original authors used a private implementation detail not recoverable
   from the specification.

Final writeup recommendation:
- Report faithful FUS implementation as the paper-faithful baseline.
- Report paper_curve_approx as the closest match to the paper's figures.
- Note the gap and list hypothesis tests as a replication effort.
- Use faithful FUS as the baseline for novelty evaluation (not the approx).
