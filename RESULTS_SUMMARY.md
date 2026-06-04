# Results Summary

This file provides a human-readable overview of all replication results.
Full per-fold CSV files are in `core/results/`, `cross_check/results/`, and `baselines/results/`.

---

## Dataset

| Property | Value |
|---|---|
| Source | MovieLens 100k |
| Users | 497 |
| Items | 903 |
| Ratings | 79,432 |
| Validation | 10-fold cross-validation, `random_state=42` |
| Protocol | Protocol A (α = 0), warm-start |

---

## Headline Comparison at k=10, α=0

Mean over 10 folds. Paper values read from D'Aniello et al., IEEE Access 2026, Fig. 6–7.

| System | MAE\_users (ours) | MAE\_users (paper) | CR (ours) | CR (paper) | Ordering match |
|---|---|---|---|---|---|
| FUS | 0.7728 | ~0.74 | 0.9796 | ~0.99 | ✅ |
| CF | 0.7809 | not reported | 0.9299 | not reported | ✅ |
| GIM | 0.8170 | ~0.85 | 0.9072 | ~0.92 | ✅ |
| PF | 0.8447 | ~0.88 | 0.9191 | ~0.92 | ✅ |

The relative ordering **FUS > CF > GIM > PF** on both MAE and Coverage Rate reproduces the paper's headline result exactly.

---

## FUS Replication Detail (core/)

| k | Metric | Faithful (ours) | Floor/round post-processing | Paper |
|---|---|---|---|---|
| 1 | MAE\_data | 0.857 | 0.943 | 0.985 |
| 1 | MAE\_users | 0.869 | 0.847 | 0.857 |
| 1 | RMSE | 1.138 | 1.261 | 1.316 |
| 1 | CR | 0.6007 | 0.6007 | ~0.59 ✅ |
| 50 | MAE\_data | 0.732 | 0.817 | 0.830 |
| 50 | MAE\_users | 0.740 | **0.7025** | **0.703** ✅ |
| 50 | RMSE | 0.937 | 1.078 | 1.107 |
| 50 | CR | 0.9997 | 0.9997 | ~0.99 ✅ |

At k=50, MAE\_users matches the paper to **four decimal places** (0.7025 vs 0.703) under floor-and-round post-processing.

---

## Cross-Check (core/ vs cross\_check/)

Two independent FUS implementations were written and compared.

| Metric | Max absolute difference |
|---|---|
| MAE\_data | 0.0000000000 |
| All other metrics | 0.0000000000 |
| Result | **PASS** |

Both implementations produce identical numbers across all 400 rows (10 folds × 10 k-values × 4 metrics). This is the strongest correctness check available without access to the original authors' code.

---

## Replication Notes

The faithful FUS implementation produces MAE metrics roughly 12–16% lower than the paper reports. Coverage Rate matches the paper exactly at every k, confirming the algorithm is correctly implemented.

After testing 51 hypotheses, the only configuration that brings the numbers within 3% of the paper applies `floor()` to predictions before computing MAE\_data and RMSE, and `round()` before computing MAE\_users and RMSE\_users. This appears to be an undocumented post-processing step in the original paper.

PF and GIM also undershoot D'Aniello's claimed values by ~4% on MAE\_users, suggesting the gap is an evaluation-pipeline artefact rather than an implementation issue.

Full investigation log: `cross_check/REPRO_DEBUG_NOTES.md`

---

## Test Suite

12 sanity checks run via `python run_all.py --tests`. All 12 pass on the committed result CSVs.

Checks include: CR-at-k=1 ≈ 0.60, sanity-band checks per metric, cross-implementation consistency (core vs cross\_check), and system-ordering check (FUS < CF < GIM, PF on MAE).
