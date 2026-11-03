# Past Tests - the core implementation Faithful Baseline

This folder contains **diagnostic and exploratory** files that were used during development and replication investigation. They are preserved here for reference but are **not part of the final deliverable**.

---

## Contents

```
past_tests/
+-- README.md                              (this file)
+-- code/
|   +-- regenerate_plots_approx.py         (script that regenerates figs from paper-curve-approx data)
+-- results/
    +-- results_FUS_A_warm_paper_curve_approx.csv  (FUS results with floor/round post-processing)
    +-- figs/                                      (fig2 to fig5 drawn from that CSV)
```

---

## What Each File Is

### `code/regenerate_plots_approx.py`

A standalone script that reads `results_FUS_A_warm_paper_curve_approx.csv` and regenerates the four paper-style figures (`fig2` through `fig5`). 

**Why it exists**: During replication investigation, we discovered the paper likely applies `floor(prediction)` for dataset-level metrics and `round(prediction)` for user-level metrics. This script produces figures from that approximation, making them match the paper's Figs 3-5 visually. It writes the figures to `results/figs/` in this `past_tests` folder (`core/past_tests/results/figs/`). The committed PNGs are the figures that were previously in `core/results/figs/` (their curves match this CSV, e.g. MAE_data about 0.817 at k=50; a fresh run with a current matplotlib is not byte-identical); the figures there now are drawn from the faithful CSV by `core/code/eval.py`. The approximation CSV was produced before the v1.1.0 audit (10 k values, predictions clipped to [1, 5]) and has not been regenerated.

**To run** (needs no dataset): `python code/regenerate_plots_approx.py` from `core/past_tests/`.

### `results/results_FUS_A_warm_paper_curve_approx.csv`

The FUS Protocol A results computed with the floor/round post-processing applied:
- Dataset-level metrics (MAE_data, RMSE): computed after clipping predictions with `floor(pred)` to integers.
- User-level metrics (MAE_users, RMSE_users): computed after clipping with `round(pred)`.
- Coverage Rate (CR): unaffected - it is a count metric.

**Why it exists**: To provide figures that match the paper's scale. The production file (`results/results_FUS_A_warm.csv`) uses the faithful exact implementation without this post-processing.

---

## Background: The Paper-Curve Approximation

After testing 20+ hypotheses, the closest match to the paper's reported numbers is achieved by applying `floor()` (for data metrics) and `round()` (for user metrics) to predictions before computing errors. This post-processing is not described in the article.

Key results of the approximation (k=50, α=0):

| Metric    | Faithful impl | Approx (floor/round) | Paper target |
|-----------|--------------|----------------------|--------------|
| MAE_data  | 0.7318       | 0.8174               | ~0.83        |
| MAE_users | 0.7397       | 0.7025               | ~0.703       |
| RMSE      | 0.9374       | 1.0779               | ~1.107       |
| CR        | 0.9997       | 0.9997               | ~0.99        |

The approximation matches the paper within 1-3% for all metrics. MAE_users at k=50 matches exactly (0.7025 vs 0.703).

For a full account of the replication investigation and all hypotheses tested, see `../../cross_check/REPRO_DEBUG_NOTES.md`.
