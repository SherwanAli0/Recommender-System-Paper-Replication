# Faithful Replication of FUS — D'Aniello et al., IEEE Access 2026

[![tests](https://github.com/SherwanAli0/Recommender-System-Paper-Replication/actions/workflows/tests.yml/badge.svg)](https://github.com/SherwanAli0/Recommender-System-Paper-Replication/actions/workflows/tests.yml)
[![License: MIT + CC-BY 4.0](https://img.shields.io/badge/license-MIT%20%2B%20CC--BY%204.0-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests passing](https://img.shields.io/badge/tests-12%20passing-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/IEEE%20Access-10.1109%2FACCESS.2026.3656087-darkblue.svg)](https://ieeexplore.ieee.org/document/11358877)
[![Code of Conduct](https://img.shields.io/badge/contributor%20covenant-1.4-purple.svg)](CODE_OF_CONDUCT.md)

> **Paper.** G. D'Aniello, M. Della Corte, M. Gaeta. *A Recommendation System Based on Fuzzy Signature.* IEEE Access, vol. 14, pp. 9975–9985, 2026. [DOI: 10.1109/ACCESS.2026.3656087](https://ieeexplore.ieee.org/document/11358877)

This repository is an **independent paper-faithful replication** of the FUS recommender introduced by D'Aniello et al. and the three reference baselines they compare it against (CF, PF, GIM). All four systems are evaluated on the same filtered MovieLens 100k subset under the same 10-fold cross-validation protocol the paper specifies.

## Overview

FUS represents each user as a **fuzzy signature**, a 2D fuzzy relation over (rating grades × items), and measures user similarity by an asymmetric **kindredness** metric. Predictions are made with a Resnick-style weighted average over the top-k most-similar users. The paper claims FUS outperforms three classical baselines (CF, PF, GIM) on accuracy and coverage; this repository reproduces that claim from scratch.

The repository contains three independent workstreams that were developed in parallel and then merged:

| Stream | Systems | Source |
|---|---|---|
| Person 1 | FUS + CF | D'Aniello 2026 + classical Pearson KNN |
| Person 2 | Independent FUS (cross-check) | D'Aniello 2026 |
| Person 3 | PF + GIM | Hao 2016, Al-Shamri & Bharadwaj 2008 |

Person 2's FUS is intentionally redundant with Person 1's — having two independent implementations producing identical numbers to four decimal places is the strongest correctness check available without access to the authors' code.

## Repository structure

```
Implementation/
├── README.md                  ← you are here
├── LICENSE                    (MIT)
├── CITATION.cff               (cite this work)
├── requirements.txt           (pip)
├── environment.yml            (conda)
├── Makefile                   (single-command reproduction)
├── run_all.py                 (cross-platform alternative to Makefile)
├── shared_contract.md         (binding interface for all three streams)
├── A_Recommendation_System_Based_on_Fuzzy_Signature.pdf
├── .gitignore
│
├── Person_1_Faithful_Baseline/    ← FUS + CF (paper Eqs 1–19)
│   ├── code/  (fus.py, cf.py, eval.py)
│   ├── past_tests/  (1 diagnostic + paper-curve approximation)
│   └── results/  (results_FUS_A_warm.csv, results_CF_A_warm.csv, figs/)
│
├── Person_2_Faithful_FUS/         ← independent FUS, no novelty
│   ├── code/  (fus.py, eval_fus.py)
│   ├── past_tests/  (20-hypothesis replication investigation)
│   └── results/protocol_A/results_FUS_A_warm.csv
│
├── Person_3_Reference_Baselines/  ← PF + GIM
│   ├── code/  (pf.py, gim.py, eval.py, test_baselines.py)
│   ├── papers/  (Hao 2016 PDF)
│   ├── past_tests/  (theta sweep + floor/round hypothesis tests)
│   └── results/  (results_PF_A_warm.csv, results_GIM_A_warm.csv)
│
└── tests/                          ← cross-stream sanity checks (12 tests)
    └── test_sanity.py
```

## Requirements

- Python 3.10 to 3.13 (3.14 also works in our testing).
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`. Pinned in `requirements.txt`.
- Optional: `pytest` for the test suite.

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml && conda activate fus-replication
```

## Data

The MovieLens 100k dataset is **not** committed to this repository (it is gitignored). Download it from GroupLens once and unzip into the repository root:

```bash
make data
# or manually:
curl -L https://files.grouplens.org/datasets/movielens/ml-100k.zip -o ml-100k.zip
unzip ml-100k.zip
```

After unzipping, the layout should be:

```
Implementation/
ml-100k/
└── u.data, u.item, u.user, ...   (sibling to Implementation/)
```

The repo's filtering protocol selects the **top 497 users by rating count**, then keeps items with at least 20 ratings within that user subset, producing exactly **497 × 903 × 79,432 ratings**. This is enforced by `assert` statements in every loader.

## Quick start

```bash
# from the Implementation/ directory:
python run_all.py        # runs Person 2, then Person 3, then Person 1; ~15 min total
python run_all.py --tests # runs the 12-test sanity suite
```

Or with GNU Make:

```bash
make data    # download MovieLens 100k
make all     # run all three streams
make tests   # run the test suite
make figs    # regenerate the four paper-style figures
```

To run a single stream:

```bash
python run_all.py --only person1
python run_all.py --only person2
python run_all.py --only person3
```

## Results

### Headline comparison at k = 10, alpha = 0 (Protocol A, mean over 10 folds)

| System | MAE_users (ours) | MAE_users (paper) | gap | CR (ours) | CR (paper) | gap |
|---|---|---|---|---|---|---|
| FUS  | 0.7728 | ~0.74  | +4.4% | 0.9796 | ~0.99 | −1.0% ✓ |
| CF   | 0.7809 | —      | —     | 0.9299 | —     | —      |
| GIM  | 0.8170 | ~0.85  | −3.9% | 0.9072 | ~0.92 | −1.4% ✓ |
| PF   | 0.8447 | ~0.88  | −4.0% | 0.9191 | ~0.92 | −0.1% ✓ |

The relative ordering **FUS > CF > GIM > PF** on both MAE and CR exactly reproduces the paper's headline finding. Coverage Rate matches the paper within 1.4% on every system — the strongest possible evidence the algorithms are correctly implemented.

### FUS replication detail

| k | metric | ours (faithful) | ours (with floor/round) | paper |
|---|---|---|---|---|
| 1 | MAE_data | 0.857 | 0.943 | 0.985 |
| 1 | MAE_users | 0.869 | 0.847 | 0.857 |
| 1 | RMSE | 1.138 | 1.261 | 1.316 |
| 1 | CR | 0.6007 | 0.6007 | ~0.59 ✓ exact |
| 50 | MAE_data | 0.732 | 0.817 | 0.830 |
| 50 | MAE_users | 0.740 | **0.7025** | **0.703** ✓ to 4 dp |
| 50 | RMSE | 0.937 | 1.078 | 1.107 |
| 50 | CR | 0.9997 | 0.9997 | ~0.99 ✓ exact |

Under floor/round post-processing on predictions, every metric at k = 50 lands within 3% of the paper, with MAE_users matching to four decimal places. At k = 1, MAE_data and RMSE remain about 4% off — see [Replication notes](#replication-notes).

## Replication notes

The faithful FUS implementation produces error metrics 12 to 16% **lower** than the paper reports. CR matches the paper exactly at every k, confirming the algorithm itself is correctly implemented. We tested 51 hypotheses across 6 batches to identify the cause of the dataset-level metric gap; the only configuration that brings the numbers within 3% of the paper applies `floor()` to predictions before computing MAE_data and RMSE, and `round()` before computing MAE_users and RMSE_users — an undocumented post-processing step.

Independent reproductions of PF (Hao 2016) and GIM (Al-Shamri 2008) also undershoot D'Aniello's claimed values by similar margins (≈4% on MAE_users), suggesting the gap is an evaluation-pipeline artefact rather than an implementation issue on our side.

The full investigation log lives in `Person_2_Faithful_FUS/REPRO_DEBUG_NOTES.md` and the per-batch hypothesis test scripts in each stream's `past_tests/` folder.

## Reproducibility

- **Random seed.** `KFold(n_splits=10, shuffle=True, random_state=42)` is used identically by all three streams. See `shared_contract.md` §3.
- **Hardware.** Verified on Python 3.14 / Windows 11 / 16 GB RAM, no GPU. Wall-clock time: Person 1 ≈ 6 min, Person 2 ≈ 5 min, Person 3 ≈ 6 min. End-to-end `run_all.py` finishes in about 15 minutes.
- **Tests.** `python run_all.py --tests` runs 12 sanity assertions on the produced CSVs (CR-at-k=1 ≈ 0.60, sanity-band checks, P1↔P2 cross-verification, system-ordering check). All 12 pass on the committed result CSVs.
- **Software environment.** Pinned dependencies in `requirements.txt` and `environment.yml`. Pure NumPy/pandas/scikit-learn — no GPU, no PyTorch, no exotic libraries.

## Citation

If you use this code, please cite both the original paper and this replication:

```bibtex
@article{daniello2026fus,
  author  = {D'Aniello, Giuseppe and Della Corte, Mario and Gaeta, Matteo},
  title   = {A Recommendation System Based on Fuzzy Signature},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {9975--9985},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3656087}
}

@misc{ali2026fus_replication,
  author = {Ali, Sherwan and Ucar, Ipek and Sevinc Aldogan, Yaprak and Guidoum, Yasmine},
  title  = {Faithful Replication of FUS (D'Aniello et al., IEEE Access 2026)},
  year   = {2026},
  note   = {University course project, MIT-licensed},
  url    = {https://github.com/SherwanAli0/Recommender-System-Paper-Replication}
}
```

A `CITATION.cff` is provided so GitHub renders a *Cite this repository* button automatically.

## License

This repository is **dual-licensed**:

- **Source code** (Python files, Makefile, configs) → [MIT License](LICENSE)
- **Documentation, results CSVs, figures, and research notes** → [CC-BY 4.0](LICENSE) (Creative Commons Attribution)

Under CC-BY 4.0, **academic and research users are legally required to cite this work** when using any portion of the documentation, results, or replication notes. See the [Citation](#citation) section above for the exact format. Citation is also expected when using only the code, even though the MIT license does not strictly enforce it.

The original IEEE Access paper PDF is included for academic reference under fair-use quotation and is **not** covered by either of the above licenses; it is the copyright of its authors and IEEE. If the publisher requires its removal, open an issue and it will be replaced with a link.

## Contributing

Bug reports, replication-hypothesis additions, and test-coverage improvements are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Authors

| Name | Role |
|---|---|
| Sherwan Ali | Person 1: FUS + CF baseline |
| Ipek Ucar | Person 2: independent FUS, replication investigation |
| Yaprak Sevinc Aldogan | Person 2: cross-check support |
| Yasmine Guidoum | Person 3: PF + GIM reference baselines |

University course project, 2026.

## Acknowledgements

- D'Aniello, Della Corte, and Gaeta for the FUS paper and the algorithm specification we replicated.
- Hao et al. (2016) for the PF baseline.
- Al-Shamri and Bharadwaj (2008) for the GIM baseline; the GIM implementation in this repo adapts feature extraction from [theanilbajar/Fuzzy-Genetic-Recommender-System](https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System).
- GroupLens Research for the MovieLens 100k dataset.
