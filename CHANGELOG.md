# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-19

### Changed (paper-faithfulness audit)
- **Removed prediction clipping to [1, 5]** in all four prediction code paths: `core/code/fus.py::resnick_predict`, `core/code/fus.py::resnick_from_walk` (production), `cross_check/code/fus.py::resnick_predict`, and `cross_check/code/eval_fus.py::resnick` (production). Paper Eq 19 is a real-valued residual-shrinkage formula; the literal-paper reading returns the raw value. `shared_contract.md` §4.1 updated from "must clip" to "no clip" to match.
- **Replaced user-mean fallback** in both `compute_user_means` functions from a hard-coded 3.0 to the training-set global mean. Paper-silent; global mean is the principled CF choice. On paper-filtered ML-100k this case essentially never fires (497 users all with >= 20 ratings).
- **Expanded FUS k-sweep from 10 to 26 values** to match paper §V.C.2: `k = {1, 2, 4, 6, 8, 10, ..., 48, 50}`. Updated `K_VALUES` in `core/code/eval.py` and `cross_check/code/eval_fus.py`; `shared_contract.md` §6 rewritten so FUS uses the paper's literal sweep while the baseline-comparison sweep (PF, GIM, CF) stays at 10 values (paper Figs 6-7 only cover k up to 10).

### Updated
- `core/results/results_FUS_A_warm.csv` and `cross_check/results/protocol_A/results_FUS_A_warm.csv` now contain the full paper sweep (26 k values × 4 alpha × 10 folds = 1040 rows each), run on the audit-fixed code (no clip, global-mean fallback). CR at k=1, alpha=0 stays at the paper's published 0.60. MAE_data and RMSE rise slightly because some predictions now land outside [1, 5]; per-user metrics (MAE_users, RMSE_users) are essentially unchanged.

### Rationale
External audit of the implementation against the paper text flagged the clip, the 3.0 fallback, and the sparse k-sweep as deviations from the literal paper formula. All three are unambiguous on inspection of paper §IV.B and §V.C.2, and were applied consistently across both independent streams (core + cross_check).

## [1.0.0] - 2026-05-08

### Added
- Faithful replication of FUS algorithm following D'Aniello et al. 2026 Eqs 1-19 (the core implementation + the cross-check implementation, cross-verified).
- Pearson-KNN CF baseline (the core implementation).
- PF (Hao 2016) and GIM (Al-Shamri 2008) reference baselines (the baselines implementation).
- Replication-investigation log: 51 hypothesis tests across 6 batches plus a four-channel literature search documented in `cross_check/REPRO_DEBUG_NOTES.md`.
- Theta sweep + floor/round post-processing tests for PF and GIM (the baselines implementation `past_tests/`).
- 12-test cross-stream sanity suite under `tests/test_sanity.py`.
- Single-command reproduction via `Makefile` and cross-platform `run_all.py`.
- GitHub Actions CI matrix on Ubuntu, macOS, Windows × Python 3.10/3.11/3.12.
- Dual licensing: MIT for code, CC-BY 4.0 for documentation/results.
- Citation policy with mandatory citation for academic use.
- Pull request template, issue templates (bug report and replication question), CODEOWNERS, branching guide, contributing guide, code of conduct, changelog, editor config, security policy.

### Verified
- Coverage Rate at k=1 alpha=0 is 0.6007 (matches the paper's ~0.59 to 4 decimals).
- System ordering FUS > CF > GIM > PF on MAE_users at k=10 reproduces the paper's headline result.
- the core and cross-check implementations independent FUS implementations agree to four decimal places.
- All five metrics within 3% of the paper at k=50 under the floor-for-data + round-for-users post-processing approximation.

### Known issues
- At k=1, MAE_data and RMSE remain about 4% below the paper even with the best-known floor/round post-processing. Investigation across 51 hypotheses concluded the gap cannot be closed from paper text alone; see `cross_check/REPRO_DEBUG_NOTES.md`.
