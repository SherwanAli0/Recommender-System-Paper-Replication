# Results Summary

This file provides a human-readable overview of the committed replication results.
Full per-fold CSV files are in `core/results/`, `cross_check/results/`, and `baselines/results/`.
The README has the same tables with more context. Numbers here are recomputed from those CSVs.

---

## Dataset

| Property | Value |
|---|---|
| Source | MovieLens 100k |
| Users | 497 |
| Items | 903 |
| Ratings | 79,432 |
| Validation | 10-fold cross-validation, `random_state=42` |
| Protocol | Protocol A (alpha = 0 for the headline table), warm-start |

The counts are enforced by `assert` statements in the loaders and were confirmed on the real data in the latest rerun (see the README section "Latest full rerun", which also lists where a fresh run differs from the committed CSVs). The dataset is not part of the repository.

---

## Headline Comparison at k=10, alpha=0

Mean over 10 folds. MAE\_users is in rating points (lower is better); CR is a fraction (higher is better; the paper's figures show percent). Paper values are read by eye from D'Aniello et al., IEEE Access 2026, Figs. 6 and 7 (about +/-0.01 for MAE\_users, about 1 point for CR), except FUS CR (97.9%) and the PF/GIM CR maximum (91.8%), which the paper's text states.

| System | MAE\_users (ours) | MAE\_users (paper) | CR (ours) | CR (paper) |
|---|---|---|---|---|
| FUS | 0.7788 | ~0.74 | 0.9796 | 0.979 |
| CF | 0.7809 | ~0.93 | 0.9299 | ~0.88 |
| GIM | 0.8170 | ~0.83 | 0.9072 | ~0.92 |
| PF | 0.8447 | ~0.91 | 0.9191 | ~0.92 |

Ordering by MAE\_users (best first): paper FUS, GIM, PF, CF; here FUS, CF, GIM, PF. Ordering by CR: paper FUS, PF and GIM about equal, CF; here FUS, CF, PF, GIM.

FUS has the lowest MAE\_users and the highest CR in both the paper and this replication, and GIM beats PF on MAE\_users in both. The paper's CF is the worst system; the CF here is much stronger and is roughly tied with FUS at k=10 (difference 0.002; paired fold difference 0.0020, standard deviation 0.0078, FUS lower in 6 of 10 folds) and better than FUS at k=50 (0.7293 vs 0.7415, CF lower in 10 of 10 folds).

The FUS row is the audit-fixed run (no prediction clipping). The CF row was produced before the clip was removed, and PF and GIM still clip predictions to [1, 5], so the table mixes clipped and unclipped predictions.

---

## FUS Replication Detail (alpha=0)

"Faithful" is the committed run in `cross_check/results/protocol_A/` (an identical copy is in `core/results/`): predictions unclipped, no post-processing. "Floor/round" is the historical diagnostic run in `core/past_tests/results/`: floor of the prediction for MAE\_data and RMSE, round for MAE\_users, applied to predictions clipped to [1, 5], 10 k values, produced 2026-05-09 before the audit and not regenerated. Paper values are stated in the paper's text.

| k | Metric | Faithful (ours) | Floor/round (ours) | Paper |
|---|---|---|---|---|
| 1 | MAE\_data | 0.8843 | 0.9429 | 0.985 |
| 1 | MAE\_users | 0.9008 | 0.8472 | 0.857 |
| 1 | RMSE | 1.1590 | 1.2607 | 1.316 |
| 1 | CR | 0.6007 | 0.6007 | slightly below 0.60 |
| 50 | MAE\_data | 0.7333 | 0.8174 | 0.83 |
| 50 | MAE\_users | 0.7415 | 0.7025 | 0.703 |
| 50 | RMSE | 0.9383 | 1.0779 | 1.107 |
| 50 | CR | 0.9997 | 0.9997 | around 0.99 |

At k=50 the floor/round MAE\_users (0.7025, mean 0.70254) agrees with the paper's 0.703 at the three decimals the paper reports. This is not a four-decimal match, and it is the floor/round run, not the faithful one. At k=1 the floor/round MAE\_data and RMSE are still about 4 percent below the paper.

---

## Cross-Check (core/ vs cross\_check/)

The core and cross\_check folders contain two FUS code paths. The committed FUS CSVs in the two folders are byte-identical (1040 rows each: 10 folds x 26 k values x 4 alpha values), because `core/code/eval.py` copies the cross\_check CSV instead of recomputing FUS. Their equality therefore says nothing about agreement of the code. That is tested on a small synthetic fixture in `tests/test_sanity.py::test_core_and_cross_check_fus_code_agree`. The two modules are structurally close, so this is a consistency check, not an independent derivation.

---

## Replication Notes

The faithful FUS run gives MAE\_data and RMSE about 10 to 15 percent below the paper, and MAE\_users about 5 percent above it. CR is close to the paper's curve. A floor/round post-processing, found by trial and not described in the paper, brings the k=50 values within 3 percent of the paper; it is a fit, not a demonstrated explanation. The gaps for PF, GIM and CF point in different directions (MAE\_users at k=10 against Fig. 6: GIM -1.4%, PF -7.2%, CF -16%, FUS +5.4%), so a single evaluation-artefact explanation is not established.

Full investigation log: `cross_check/REPRO_DEBUG_NOTES.md`

---

## Test Suite

`python run_all.py --tests` runs `pytest tests/`. It reads the committed CSVs, needs no dataset and regenerates nothing. Checks: required CSV columns, FUS CR at k=1 about 0.60, CR non-decreasing in k, FUS k=50 sanity band, PF and GIM k=10 sanity bands, FUS lowest MAE\_users at k=10, each script's data path pointing to `<repo>/ml-100k`, and core/cross\_check FUS code agreement on a synthetic fixture.
