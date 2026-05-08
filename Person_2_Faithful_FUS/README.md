# Person 2 - Independent FUS implementation (faithful, no novelty)

This folder contains the second independent implementation of the FUS algorithm, used for cross-verification against Person 1. The novelty extension (Adaptive Genre-Hybrid for cold-start) developed alongside this work is intentionally NOT in this Implementation release. See `../README.md` and the original `Person_2_Novelty/` folder for the novelty.

## What is in this folder

```
Person_2_Faithful_FUS/
├── README.md                        (this file)
├── REPRO_DEBUG_NOTES.md             (full replication investigation log)
├── CONTINUE_NOTES.md                (state summary)
├── code/
│   ├── fus.py                       (Eqs 1-19, paper-faithful)
│   └── eval_fus.py                  (Protocol A runner, no novelty)
├── past_tests/                      (51 hypothesis tests + 4-agent web search)
└── results/
    └── protocol_A/
        └── results_FUS_A_warm.csv   (10 folds * 10 k * 4 alpha = 400 rows)
```

## What this implementation produces

The `fus.py` module implements the paper's Eqs 1 to 19 directly:

- Per-user fuzzy signature matrix `USui(s, m) = min(RatPop(s), ItemAttr(m))` for rated cells.
- Asymmetric kindredness similarity `sim(i, j) = sum(min(USi, USj)) / sum(USi)`.
- Resnick prediction with mean centring and rater-only denominator.
- Top-k neighbourhood selected from globally sorted similarity rank.

`eval_fus.py` runs the standard 10-fold cross-validation at alpha in {0.0, 0.2, 0.5, 1.0} across k in {1, 2, 3, 5, 7, 10, 15, 20, 30, 50}.

## Cross-verification against Person 1

Person 1 implements the same algorithm independently. The two implementations produce identical numbers to four decimal places. This is the cross-check the project plan explicitly requires.

## past_tests folder

Contains the most thorough piece of the project: 51 hypothesis tests run during the replication investigation, organised in batches A through F. Highlights:

- 20 original hypotheses (filter order, KFold seeds, signature variants, similarity direction, denominator interpretations, post-processing).
- The friend's Claude minimum-co-rated-items test (no-op on this dataset).
- Default substitution variants and RatPop denominator alternatives.
- Mathematical diagnostic showing missing-pair MAE = 1.18 is unreachable.
- Final 16 exhaustive variants (per-user fold split, similarity threshold, T-norms, sigma-count cardinality, etc.).
- 9 remaining hypotheses (bias correction, symmetric kindredness, alpha sweep).

Plus the four-agent deep web search across Yager-Reformat 2013, Hao 2016, the 2024 FUZZ-IEEE precursor, and code repositories.

The most valuable artefact is `REPRO_DEBUG_NOTES.md` which documents every hypothesis tried and ruled out.

## Replication summary

| k  | Faithful MAE_data | Faithful MAE_users | Floor/round MAE_data | Floor/round MAE_users | Paper MAE_data | Paper MAE_users |
|----|-------------------|--------------------|----------------------|-----------------------|----------------|-----------------|
| 1  | 0.857             | 0.869              | 0.943                | 0.847                 | 0.985          | 0.857           |
| 50 | 0.732             | 0.740              | 0.817                | 0.703                 | 0.830          | 0.703           |

CR matches the paper exactly at every k. At k = 50 with floor/round post-processing all five metrics are within 3 percent of the paper.

## How to run

```bash
cd code
python eval_fus.py
```

Expect about 4 to 6 minutes runtime on a modern laptop. Output appears in `results/protocol_A/results_FUS_A_warm.csv`.
