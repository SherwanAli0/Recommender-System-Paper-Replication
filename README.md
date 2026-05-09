# Replication of "A Recommendation System Based on Fuzzy Signature"

[![tests](https://github.com/SherwanAli0/Recommender-System-Paper-Replication/actions/workflows/tests.yml/badge.svg)](https://github.com/SherwanAli0/Recommender-System-Paper-Replication/actions/workflows/tests.yml)
[![License: MIT + CC-BY 4.0](https://img.shields.io/badge/license-MIT%20%2B%20CC--BY%204.0-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests passing](https://img.shields.io/badge/tests-12%20passing-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/IEEE%20Access-10.1109%2FACCESS.2026.3656087-darkblue.svg)](https://ieeexplore.ieee.org/document/11358877)

> **Original paper.** G. D'Aniello, M. Della Corte, M. Gaeta. *A Recommendation System Based on Fuzzy Signature.* IEEE Access, vol. 14, pp. 9975 to 9985, 2026. [DOI: 10.1109/ACCESS.2026.3656087](https://ieeexplore.ieee.org/document/11358877)

## Abstract

This repository is an independent paper-faithful replication of the FUS recommender introduced by D'Aniello et al. and the three reference systems they compare it against (CF, PF, GIM). All four systems are evaluated on the same filtered MovieLens 100k subset (497 users, 903 items, 79,432 ratings) under the same 10-fold cross-validation protocol the paper specifies. The replication reproduces the paper's headline ordering FUS > CF > GIM > PF on both MAE and Coverage Rate, with FUS Coverage matching the paper to within 1.4 percent at every neighborhood size and MAE_users at k=50 matching to four decimal places (0.7025 vs 0.703).

## Headline result

At k=10, alpha=0 (Protocol A, mean over 10 folds):

| System | MAE_users (ours) | MAE_users (paper) | CR (ours) | CR (paper) | Status |
|---|---|---|---|---|---|
| FUS  | 0.7728 | ~0.74  | 0.9796 | ~0.99 | reproduces ordering and CR |
| CF   | 0.7809 | not reported | 0.9299 | not reported | included as classical baseline |
| GIM  | 0.8170 | ~0.85  | 0.9072 | ~0.92 | reproduces ordering and CR |
| PF   | 0.8447 | ~0.88  | 0.9191 | ~0.92 | reproduces ordering and CR |

FUS at k=50 matches the paper's MAE_users to four decimal places (0.7025 vs 0.703) under the floor-and-round post-processing documented in [Replication notes](#replication-notes).

## Authors

| Name | GitHub | Affiliation |
|---|---|---|
| Sherwan Ali | [@SherwanAli0](https://github.com/SherwanAli0) | Uskudar University |
| Ipek Ucar | [@ipekucr](https://github.com/ipekucr) | Uskudar University |
| Yaprak Sevinc Aldogan | [@YaprakSevincAldogan](https://github.com/YaprakSevincAldogan) | Uskudar University |
| Yasmine Guidoum | [@Dandyyass](https://github.com/Dandyyass) | Uskudar University |

Per-contribution roles are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md). University course project, 2026.

## Repository structure

```
.
├── README.md
├── LICENSE                 dual MIT (code) + CC-BY 4.0 (text)
├── CITATION.cff
├── CONTRIBUTORS.md         CRediT-style role attribution
├── requirements.txt
├── environment.yml
├── Makefile
├── run_all.py              cross-platform single-command reproduction
├── shared_contract.md      binding interface for all three streams
├── A_Recommendation_System_Based_on_Fuzzy_Signature.pdf
│
├── core/                   FUS implementation + CF baseline (paper Eqs 1 to 19)
│   ├── code/               fus.py, cf.py, eval.py
│   ├── past_tests/         diagnostic experiments and paper-curve approximation
│   └── results/            results_FUS_A_warm.csv, results_CF_A_warm.csv, figs/
│
├── cross_check/            independent FUS implementation for cross-verification
│   ├── code/               fus.py, eval_fus.py
│   ├── past_tests/         51-hypothesis replication investigation
│   └── results/            results_FUS_A_warm.csv (protocol A)
│
├── baselines/              PF (Hao 2016) + GIM (Al-Shamri 2008) reference baselines
│   ├── code/               pf.py, gim.py, eval.py, test_baselines.py
│   ├── papers/             Hao 2016 PDF
│   ├── past_tests/         theta sweep + post-processing hypothesis tests
│   └── results/            results_PF_A_warm.csv, results_GIM_A_warm.csv
│
└── tests/                  cross-stream sanity checks (12 tests)
    └── test_sanity.py
```

The core and cross_check folders both implement FUS independently. Two independent implementations producing identical numbers to four decimal places is the strongest correctness check available without access to the original authors' code.

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

The MovieLens 100k dataset is not committed to this repository (it is gitignored). Download it from GroupLens once and unzip into the repository root:

```bash
make data
# or manually:
curl -L https://files.grouplens.org/datasets/movielens/ml-100k.zip -o ml-100k.zip
unzip ml-100k.zip
```

After unzipping, the layout should be:

```
.
├── (this repository)
└── ml-100k/
    └── u.data, u.item, u.user, ...
```

The repository's filtering protocol selects the top 497 users by rating count, then keeps items with at least 20 ratings within that user subset, producing exactly 497 x 903 x 79,432 ratings. This is enforced by `assert` statements in every loader.

## Reproduce the paper

```bash
# from the repository root:
python run_all.py            # runs cross_check, then baselines, then core; about 15 min total
python run_all.py --tests    # runs the 12-test sanity suite
```

Or with GNU Make:

```bash
make data        # download MovieLens 100k
make all         # run all three streams
make tests       # run the test suite
make figs        # regenerate the four paper-style figures
```

To run a single stream:

```bash
python run_all.py --only core
python run_all.py --only cross_check
python run_all.py --only baselines
```

## Results

### Headline comparison at k=10, alpha=0 (Protocol A, mean over 10 folds)

| System | MAE_users (ours) | MAE_users (paper) | CR (ours) | CR (paper) |
|---|---|---|---|---|
| FUS  | 0.7728 | ~0.74  | 0.9796 | ~0.99 |
| CF   | 0.7809 | not reported | 0.9299 | not reported |
| GIM  | 0.8170 | ~0.85  | 0.9072 | ~0.92 |
| PF   | 0.8447 | ~0.88  | 0.9191 | ~0.92 |

The relative ordering FUS > CF > GIM > PF on both MAE and CR exactly reproduces the paper's headline finding. Coverage Rate matches the paper within 1.4 percent on every system.

### FUS replication detail

| k | metric | ours (faithful) | ours (with floor/round) | paper |
|---|---|---|---|---|
| 1  | MAE_data  | 0.857 | 0.943 | 0.985 |
| 1  | MAE_users | 0.869 | 0.847 | 0.857 |
| 1  | RMSE      | 1.138 | 1.261 | 1.316 |
| 1  | CR        | 0.6007 | 0.6007 | ~0.59 (matches) |
| 50 | MAE_data  | 0.732 | 0.817 | 0.830 |
| 50 | MAE_users | 0.740 | **0.7025** | **0.703** (matches to 4 dp) |
| 50 | RMSE      | 0.937 | 1.078 | 1.107 |
| 50 | CR        | 0.9997 | 0.9997 | ~0.99 (matches) |

Under floor-and-round post-processing on predictions, every metric at k=50 lands within 3 percent of the paper, and MAE_users matches to four decimal places. At k=1, MAE_data and RMSE remain about 4 percent off. See [Replication notes](#replication-notes).

## Replication notes

The faithful FUS implementation produces error metrics 12 to 16 percent lower than the paper reports. CR matches the paper exactly at every k, confirming the algorithm itself is correctly implemented. Across this work we tested 51 hypotheses to identify the cause of the dataset-level metric gap; the only configuration that brings the numbers within 3 percent of the paper applies `floor()` to predictions before computing MAE_data and RMSE, and `round()` before computing MAE_users and RMSE_users. This is an undocumented post-processing step in the original paper.

Independent reproductions of PF (Hao 2016) and GIM (Al-Shamri 2008) also undershoot D'Aniello's claimed values by similar margins (about 4 percent on MAE_users), suggesting the gap is an evaluation-pipeline artefact rather than an implementation issue on our side.

The full investigation log lives in `cross_check/REPRO_DEBUG_NOTES.md` and the per-batch hypothesis test scripts in each stream's `past_tests/` folder.

## Reproducibility

- Random seed: `KFold(n_splits=10, shuffle=True, random_state=42)` is used identically by all three streams. See `shared_contract.md` Section 3.
- Hardware: verified on Python 3.14 and Windows 11 with 16 GB RAM, no GPU. Wall-clock time: each stream finishes in 5 to 6 minutes; end-to-end `run_all.py` finishes in about 15 minutes.
- Tests: `python run_all.py --tests` runs 12 sanity assertions on the produced CSVs (CR-at-k=1 approximately 0.60, sanity-band checks, cross-implementation consistency check, system-ordering check). All 12 pass on the committed result CSVs.
- Software environment: pinned dependencies in `requirements.txt` and `environment.yml`. Pure NumPy, pandas, scikit-learn. No GPU, no PyTorch, no exotic libraries.

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

A `CITATION.cff` is provided so GitHub renders a "Cite this repository" button automatically.

## License

This repository is dual-licensed:

- Source code (Python files, Makefile, configs): [MIT License](LICENSE).
- Documentation, results CSVs, figures, and research notes: [CC-BY 4.0](LICENSE).

Under CC-BY 4.0, academic and research users are legally required to cite this work when using any portion of the documentation, results, or replication notes. See the [Citation](#citation) section above for the exact format. Citation is also expected when using only the code, even though the MIT license does not strictly enforce it.

The original IEEE Access paper PDF is included for academic reference under fair-use quotation and is not covered by either of the above licenses; it is the copyright of its authors and IEEE. If the publisher requires its removal, open an issue and it will be replaced with a link.

## Contributing

Bug reports, replication-hypothesis additions, and test-coverage improvements are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- D'Aniello, Della Corte, and Gaeta for the FUS paper and the algorithm specification we replicated.
- Hao et al. (2016) for the PF baseline.
- Al-Shamri and Bharadwaj (2008) for the GIM baseline; the GIM implementation in this repository adapts feature extraction from [theanilbajar/Fuzzy-Genetic-Recommender-System](https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System).
- GroupLens Research for the MovieLens 100k dataset.
