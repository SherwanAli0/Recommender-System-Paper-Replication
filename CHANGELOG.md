# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-08

### Added
- Faithful replication of FUS algorithm following D'Aniello et al. 2026 Eqs 1–19 (Person 1 + Person 2, cross-verified).
- Pearson-KNN CF baseline (Person 1).
- PF (Hao 2016) and GIM (Al-Shamri 2008) reference baselines (Person 3).
- Replication-investigation log: 51 hypothesis tests across 6 batches plus a 4-agent web search documented in `Person_2_Faithful_FUS/REPRO_DEBUG_NOTES.md`.
- Theta sweep + floor/round post-processing tests for PF and GIM (Person 3 `past_tests/`).
- 12-test cross-stream sanity suite under `tests/test_sanity.py`.
- Single-command reproduction via `Makefile` and cross-platform `run_all.py`.
- GitHub Actions CI matrix on Ubuntu, macOS, Windows × Python 3.10/3.11/3.12.
- Dual licensing: MIT for code, CC-BY 4.0 for documentation/results.
- Citation policy with mandatory citation for academic use.
- Pull request template, issue templates (bug report and replication question), CODEOWNERS, branching guide, contributing guide, code of conduct, changelog, editor config, security policy.

### Verified
- Coverage Rate at k=1 alpha=0 is 0.6007 (matches the paper's ~0.59 to 4 decimals).
- System ordering FUS > CF > GIM > PF on MAE_users at k=10 reproduces the paper's headline result.
- Person 1 and Person 2 independent FUS implementations agree to four decimal places.
- All five metrics within 3% of the paper at k=50 under the floor-for-data + round-for-users post-processing approximation.

### Known issues
- At k=1, MAE_data and RMSE remain about 4% below the paper even with the best-known floor/round post-processing. Investigation across 51 hypotheses concluded the gap cannot be closed from paper text alone; see `Person_2_Faithful_FUS/REPRO_DEBUG_NOTES.md`.
