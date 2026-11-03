# cross_check: Independent FUS implementation

This folder contains a second, independent Python implementation of the FUS algorithm (D'Aniello et al., IEEE Access 2026, Equations 1 to 19). It exists to cross-check the implementation in [`../core/`](../core/), and it is also the stream that produces the FUS results CSV: `core/code/eval.py` copies this stream's CSV rather than recomputing FUS.

The two FUS modules are written as separate files (see [CONTRIBUTORS.md](../CONTRIBUTORS.md)) but are structurally close, and the two committed FUS CSVs are byte-identical because of the copy. Agreement of the code is therefore a consistency check, not an independent derivation; it is tested on a synthetic fixture in `tests/test_sanity.py`.

## Contents

```
cross_check/
├── README.md                                 this file
├── REPRO_DEBUG_NOTES.md                      full replication investigation log
├── code/
│   ├── fus.py                                Eqs 1 to 19, paper-faithful
│   └── eval_fus.py                           Protocol A runner
├── past_tests/                               hypothesis tests + literature search
└── results/
    └── protocol_A/
        └── results_FUS_A_warm.csv            10 folds x 26 k x 4 alpha = 1040 rows
```

## What this implementation produces

The `fus.py` module implements the paper's Eqs 1 to 19 directly: 

- Per-user fuzzy signature matrix `USui(s, m) = min(RatPop(s), ItemAttr(m))` for rated cells.
- Asymmetric kindredness similarity `sim(i, j) = sum(min(USi, USj)) / sum(USi)`.
- Resnick prediction with mean centring and rater-only denominator.
- Top-k neighbourhood selected from globally sorted similarity rank.

`eval_fus.py` runs the standard 10-fold cross-validation (`KFold(10, shuffle=True, random_state=42)`) at alpha in {0.0, 0.7, 0.8, 0.9} across the paper's 26 k values, k in {1, 2, 4, 6, 8, 10, 12, ..., 48, 50}. Predictions are not clipped to [1, 5].

## Cross-verification against the core implementation

[`../core/code/fus.py`](../core/code/fus.py) and `code/fus.py` here are separate files, and `tests/test_sanity.py::test_core_and_cross_check_fus_code_agree` checks that they give the same similarity matrices and predictions on a small synthetic fixture. It does not use MovieLens, and it does not check the two modules against the paper. The core FUS CSV is a copy of this stream's CSV, so comparing the two CSVs shows nothing about the code.

## past_tests folder

Contains the replication investigation: the hypothesis tests run to explain the gap to the paper's error values, organised in batches A through F according to the authors' notes (the total number of tests is not enumerated in `REPRO_DEBUG_NOTES.md`). Highlights:

- 20 original hypotheses (filter order, KFold seeds, signature variants, similarity direction, denominator interpretations, post-processing).
- Default substitution variants and RatPop denominator alternatives.
- Mathematical diagnostic showing missing-pair MAE = 1.18 is unreachable.
- Final 16 exhaustive variants (per-user fold split, similarity threshold, T-norms, sigma-count cardinality).
- 9 remaining hypotheses (bias correction, symmetric kindredness, alpha sweep).
- A four-channel literature search across Yager-Reformat 2013, Hao 2016, the 2024 FUZZ-IEEE precursor, and code repositories.

The most valuable artefact is `REPRO_DEBUG_NOTES.md`, which documents every hypothesis tried and ruled out.

## Replication summary

| k  | Faithful MAE_data | Faithful MAE_users | Floor/round MAE_data | Floor/round MAE_users | Paper MAE_data | Paper MAE_users |
|----|-------------------|--------------------|----------------------|-----------------------|----------------|-----------------|
| 1  | 0.8843            | 0.9008             | 0.9429               | 0.8472                | 0.985          | 0.857           |
| 50 | 0.7333            | 0.7415             | 0.8174               | 0.7025                | 0.830          | 0.703           |

"Faithful" is the committed CSV in `results/protocol_A/` (alpha=0, mean over folds, unclipped predictions). "Floor/round" is the historical diagnostic CSV in `past_tests/results/` (produced before the v1.1.0 audit, 10 k values, predictions clipped to [1, 5] before floor/round). CR is close to the paper's curve (0.6007 at k=1, 0.9796 at k=10, 0.9997 at k=50 against about 0.60, 0.979 and around 0.99 in the paper). The k=50 floor/round values are within 3 percent of the paper; the faithful values are 10 to 15 percent below on MAE_data and RMSE and about 5 percent above on MAE_users. See the main README "Replication notes".

## How to run

```bash
cd code
python eval_fus.py
```

The dataset must be in `ml-100k/` in the repository root. It took about 18 minutes when measured (Windows 11, 12 logical cores, shared with other processes); the authors' earlier 4 to 6 minutes predates the growth of the sweep from 10 to 26 k values. Output appears in `results/protocol_A/results_FUS_A_warm.csv` and overwrites the committed file.
