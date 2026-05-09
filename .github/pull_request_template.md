<!--
  Thanks for contributing! Please fill in this template so reviewers
  understand the change quickly.
-->

## What

<!-- One sentence describing the change. -->

## Why

<!-- The motivation. Link to an issue if one exists: "Closes #12" -->

## How

<!-- The approach. Mention any trade-offs or alternatives considered. -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature / capability
- [ ] Documentation update
- [ ] New replication hypothesis added to `past_tests/`
- [ ] Test addition or improvement
- [ ] Refactor (no functional change)
- [ ] Build / CI / tooling change

## Faithfulness check (paper-faithful releases only)

- [ ] No production code under `core/code/`, `cross_check/code/`, or `baselines/code/` was modified to change algorithm semantics, OR the change is justified by a specific paper section or equation cited in the PR.
- [ ] The CR-at-k=1 sanity check still produces ~0.6007 (`python -m pytest tests/test_sanity.py::test_fus_cr_at_k1_matches_paper`).
- [ ] No novelty (genre-hybrid, cold-start, lambda schedules) was introduced into the production tree.

## Tested

- [ ] All 12 tests pass (`python run_all.py --tests` or `make tests`)
- [ ] Manually ran the affected stream end-to-end (`python run_all.py --only <stream>`)
- [ ] Verified output CSVs match expected pass-bands

## Reviewer checklist

- [ ] Branch name follows convention (`<initials>-<type>/<short-desc>`)
- [ ] No `__pycache__/` or other generated files committed
- [ ] No hardcoded local paths
- [ ] Source citations present in any new algorithmic code (no-original-code rule, see CONTRIBUTING.md)
