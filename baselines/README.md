# baselines: PF and GIM reference recommenders

This folder contains the two literature baseline recommenders that the FUS paper compares against:

- **PF (Probabilistic Filtering)**, the closest competitor to FUS in the paper's comparison. Uses fuzzy probability instead of fuzzy logic.
  - Source: J. Hao, Y. Yan, G. Wang, L. Gong, B. Zhao, "A probability-based hybrid user model for recommendation system," *Mathematical Problems in Engineering*, 2016.
- **GIM (Fuzzy-Genetic Method)**, a fuzzy-genetic approach from the same family of recommenders.
  - Source: M. Y. H. Al-Shamri, K. K. Bharadwaj, "Fuzzy-genetic approach to recommender systems based on a novel hybrid user model," *Expert Systems with Applications*, vol. 35, no. 3, pp. 1386 to 1399, 2008.

Both are evaluated under the same 10-fold protocol as the FUS implementation in [`../core/`](../core/) and the cross-check FUS in [`../cross_check/`](../cross_check/), so the final comparison table covers all four systems on identical data.

## Contents

```
baselines/
├── README.md                              this file
├── code/
│   ├── pf.py                              PF, Hao 2016
│   ├── gim.py                             GIM, Al-Shamri & Bharadwaj 2008
│   ├── eval.py                            10-fold CV, sweep, metrics, CSV writers
│   └── test_baselines.py                  unit tests for filter shape and PF neighbourhood
├── papers/
│   └── hao2016_9535808_pf.pdf             source paper for PF
├── past_tests/                            theta sweep + post-processing hypothesis tests
└── results/
    ├── results_PF_A_warm.csv
    └── results_GIM_A_warm.csv
```

## Success criteria

- Filter produces 497 x 903 x 79,432.
- PF at k=10: MAE_users in [0.83, 0.93], CR in [0.87, 0.97].
- GIM at k=10: MAE_users in [0.80, 0.90], CR in [0.87, 0.97].
- Both CSVs written with exact column names from `../shared_contract.md` Section 5.
- Every `.py` file carries source citations in comments. PF and GIM cite the source paper and any GitHub or Stack Overflow references the implementation was adapted from.

## How to run

### Prerequisites

```bash
pip install numpy pandas scikit-learn
```

Dataset files needed in `../ml-100k/`. See [`../shared_contract.md`](../shared_contract.md) Section 2.

### Run evaluation

```bash
cd code
python eval.py
```

This sweeps k in {1, 2, 4, 6, 8, 10} for both PF and GIM (capped at 10 because both methods degrade beyond this in the paper, see Figs 6 and 7).

### Run unit tests

```bash
cd code
python test_baselines.py
```

## References

- Hao et al., 2016. PF source paper.
- Al-Shamri and Bharadwaj, 2008. GIM source paper.
- The GIM implementation adapts feature-extraction logic from [theanilbajar/Fuzzy-Genetic-Recommender-System](https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System).
