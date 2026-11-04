# Replication of "A Recommendation System Based on Fuzzy Signature"

[![tests](https://github.com/SherwanAli0/Recommender-System-Paper-Replication/actions/workflows/tests.yml/badge.svg)](https://github.com/SherwanAli0/Recommender-System-Paper-Replication/actions/workflows/tests.yml)
[![License: MIT + CC-BY 4.0](https://img.shields.io/badge/license-MIT%20%2B%20CC--BY%204.0-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/IEEE%20Access-10.1109%2FACCESS.2026.3656087-darkblue.svg)](https://ieeexplore.ieee.org/document/11358877)

> **Original paper.** G. D'Aniello, M. Della Corte, M. Gaeta. *A Recommendation System Based on Fuzzy Signature.* IEEE Access, vol. 14, pp. 9975 to 9985, 2026. [DOI: 10.1109/ACCESS.2026.3656087](https://ieeexplore.ieee.org/document/11358877)

## Abstract

This repository is a four-person university course replication of the FUS recommender introduced by D'Aniello et al. and of the three reference systems they compare it against (CF, PF, GIM). All four systems are evaluated on the same filtered MovieLens 100k subset (497 users, 903 items, 79,432 ratings) under the 10-fold cross-validation protocol the paper describes. What the committed results do and do not show:

- **Coverage Rate (CR) of FUS follows the paper closely.** About 0.60 at k=1, 0.98 at k=10 and 0.9997 at k=50, against the paper's Fig. 5/7 curve (read by eye, roughly one percentage point of precision), its text value of 97.9% at k=10 and "around 99%" at k=50.
- **FUS error metrics are not reproduced exactly.** The faithful implementation gives MAE_data and RMSE about 10 to 15 percent below the paper and MAE_users about 5 percent above it (k=1 and k=50). A floor/round post-processing, found by trial and not described in the paper, brings the k=50 numbers within 3 percent of the paper (see [Replication notes](#replication-notes)).
- **The paper's ranking of the four systems at k=10 is only partly reproduced.** FUS has the lowest MAE_users and the highest CR, and GIM beats PF on MAE_users, as in the paper. The CF baseline here is much stronger than the paper's CF, so FUS and CF are roughly tied on MAE_users at k=10 and CF is better at k=50.

## Headline result

At k=10, alpha=0 (Protocol A, mean over 10 folds). MAE_users is in rating points (1 to 5 scale), lower is better. CR is a fraction in [0, 1], higher is better (the paper's figures show it in percent). Paper values are read by eye from the paper's Figs. 6 and 7 (about +/-0.01 for MAE_users, about 1 point for CR), except the FUS and PF/GIM coverage values, which the paper's text states (97.9% and a 91.8% maximum).

| System | MAE_users (ours) | MAE_users (paper, Fig. 6) | CR (ours) | CR (paper, Fig. 7 and text) |
|---|---|---|---|---|
| FUS  | 0.7788 | ~0.74 | 0.9796 | 0.979 |
| CF   | 0.7809 | ~0.93 | 0.9299 | ~0.88 |
| GIM  | 0.8170 | ~0.83 | 0.9072 | ~0.92 |
| PF   | 0.8447 | ~0.91 | 0.9191 | ~0.92 |

Ordering by MAE_users at k=10 (best first): paper FUS, GIM, PF, CF; this replication FUS, CF, GIM, PF. Ordering by CR: paper FUS, then PF and GIM about equal, then CF; this replication FUS, CF, PF, GIM. The FUS-over-CF margin on MAE_users at k=10 is 0.002, smaller than the fold-to-fold spread (paired fold difference 0.0020 with standard deviation 0.0078, FUS lower in 6 of 10 folds).

Provenance: the FUS row is the committed audit-fixed run (no prediction clipping). The CF row comes from a run that predates the clip removal, and PF and GIM still clip predictions to [1, 5] (see [Replication notes](#replication-notes)), so the table mixes clipped and unclipped predictions. The paper does report CF in its Figs. 6 and 7.

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
+-- README.md
+-- LICENSE                 dual MIT (code) + CC-BY 4.0 (text)
+-- LICENSE-CC-BY-4.0.txt   full legal code of CC-BY 4.0 (verbatim official text)
+-- CITATION.cff
+-- CONTRIBUTORS.md         CRediT-style role attribution
+-- requirements.txt
+-- environment.yml
+-- Makefile
+-- run_all.py              cross-platform single-command reproduction
+-- shared_contract.md      binding interface for all three streams
+-- A_Recommendation_System_Based_on_Fuzzy_Signature.pdf
|
+-- core/                   FUS implementation + CF baseline (paper Eqs 1 to 19)
|   +-- code/               fus.py, cf.py, eval.py
|   +-- past_tests/         diagnostic experiments, floor/round approximation, its figures
|   +-- results/            results_FUS_A_warm.csv, results_CF_A_warm.csv, figs/
|
+-- cross_check/            second FUS implementation (also produces the FUS CSV)
|   +-- code/               fus.py, eval_fus.py
|   +-- past_tests/         replication investigation (hypothesis scripts and CSVs)
|   +-- results/            results_FUS_A_warm.csv (protocol A)
|
+-- baselines/              PF (Hao 2016) + GIM (Al-Shamri 2008) reference baselines
|   +-- code/               pf.py, gim.py, eval.py, test_baselines.py
|   +-- papers/             Hao 2016 PDF
|   +-- past_tests/         theta sweep + post-processing hypothesis tests
|   +-- results/            results_PF_A_warm.csv, results_GIM_A_warm.csv
|
+-- tests/                  sanity checks on committed CSVs plus data-free regression checks
    +-- test_sanity.py
```

The `core` and `cross_check` folders hold two FUS implementations (written by different contributors, see [CONTRIBUTORS.md](CONTRIBUTORS.md)). They are structurally close, and the core stream reuses the cross_check FUS results instead of recomputing them, so agreement between them is a consistency check, not an independent derivation. See [Reproducibility](#reproducibility).

## Requirements

- Python 3.10 to 3.13 is the documented range. CI runs the tests on 3.10, 3.11 and 3.12; the authors report running the evaluation on 3.14.
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` (version ranges in `requirements.txt`), and `deap` for GIM's genetic algorithm in the baselines stream (it was missing from the requirements before and made `run_all.py` fail at the GIM step).
- Optional: `pytest` for the test suite.
- Last checked: Python 3.11.9 with numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, matplotlib 3.11.2, deap 1.4.4 and pytest 9.2.0 on Windows 11. Both the sanity tests and the full reproduction (all three streams) were run with these versions; see [Latest full rerun](#latest-full-rerun).

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml && conda activate fus-replication
```

## Data

The MovieLens 100k dataset (about 5 MB zipped, from GroupLens) is not committed to this repository (it is gitignored). Download it once and unzip it **inside the repository root**, so that `ml-100k/u.data` sits next to `run_all.py`:

```bash
make data
# or manually (Linux/macOS/Git Bash):
curl -L https://files.grouplens.org/datasets/movielens/ml-100k.zip -o ml-100k.zip
unzip ml-100k.zip
```

```powershell
# Windows PowerShell
Invoke-WebRequest https://files.grouplens.org/datasets/movielens/ml-100k.zip -OutFile ml-100k.zip
Expand-Archive ml-100k.zip -DestinationPath .
```

After unzipping, the layout should be:

```
.                         (this repository)
+-- run_all.py
+-- core/ cross_check/ baselines/ ...
+-- ml-100k/
    +-- u.data, u.item, u.user, ...
```

`u.data` is used by every stream, `u.item` (genres) and `u.user` (ages) by PF and GIM. The three eval scripts and `run_all.py` all look for `ml-100k/` in the repository root.

The filtering protocol selects the top 497 users by rating count in the full table, then keeps items with at least 20 ratings within that user subset. The loaders assert 497 users, 903 items and 79,432 ratings (`core/code/fus.py`, `cross_check/code/fus.py`, `baselines/code/pf.py`, which `gim.py` reuses). Note that this order (users first, items second) is the one that produces those counts; the paper's wording lists the item filter first. The latest check ran all of these loaders on the real data and they give exactly 497 users, 903 items and 79,432 ratings (see [Latest full rerun](#latest-full-rerun)).

## Reproduce the paper

```bash
# from the repository root, after downloading the data:
python run_all.py            # runs cross_check, then baselines, then core
python run_all.py --tests    # tests only, see below
```

- `python run_all.py` **regenerates and overwrites** the committed result CSVs (and, for the core stream, the four faithful-FUS figures in `core/results/figs/`). The core stream does not recompute FUS: it copies `cross_check/results/protocol_A/results_FUS_A_warm.csv` to `core/results/results_FUS_A_warm.csv` and recomputes only CF. Because of the audit changes recorded in the CHANGELOG, re-running will change the CF numbers slightly relative to the committed CF CSV, which predates the removal of prediction clipping.
- `python run_all.py --tests` runs `pytest tests/` and nothing else. It reads the committed result CSVs and runs two small data-free checks; it needs no dataset, regenerates no output, and ignores `--only` and `--skip`.
- Run time: measured on Windows 11 (12 logical cores, other processes running): cross_check about 18 minutes, baselines about 8 minutes, core under 1 minute, about 27 minutes in total. The authors' earlier figure of 15 minutes end to end predates the expansion of the FUS k sweep from 10 to 26 values.

Or with GNU Make:

```bash
make data        # download MovieLens 100k
make all         # run all three streams
make tests       # run the test suite
make figs        # rerun core/code/eval.py (CF sweep plus the four figures)
```

To run a single stream:

```bash
python run_all.py --only core
python run_all.py --only cross_check
python run_all.py --only baselines
```

## Results

### Headline comparison at k=10, alpha=0

See [Headline result](#headline-result) above.

### FUS replication detail (alpha=0, mean over 10 folds)

MAE and RMSE are in rating points, lower is better; CR is higher-is-better. "Faithful" is the committed run in `cross_check/results/protocol_A/` (identical copy in `core/results/`), predictions not clipped and no post-processing. "Floor/round" is the historical diagnostic run in `core/past_tests/results/` described under [Replication notes](#replication-notes). Paper values are the numbers the paper's text states.

| k | metric | ours (faithful) | ours (floor/round) | paper |
|---|---|---|---|---|
| 1  | MAE_data  | 0.8843 | 0.9429 | 0.985 |
| 1  | MAE_users | 0.9008 | 0.8472 | 0.857 |
| 1  | RMSE      | 1.1590 | 1.2607 | 1.316 |
| 1  | CR        | 0.6007 | 0.6007 | slightly below 0.60 |
| 50 | MAE_data  | 0.7333 | 0.8174 | 0.83 |
| 50 | MAE_users | 0.7415 | 0.7025 | 0.703 |
| 50 | RMSE      | 0.9383 | 1.0779 | 1.107 |
| 50 | CR        | 0.9997 | 0.9997 | around 0.99 |

Difference from the paper: faithful MAE_data -10.2% and -11.6%, RMSE -11.9% and -15.2%, MAE_users +5.1% and +5.5% (k=1 and k=50). Floor/round MAE_data -4.3% and -1.5%, MAE_users -1.1% and -0.1%, RMSE -4.2% and -2.6%. The k=50 MAE_users value is 0.7025 (mean 0.70254), which agrees with the paper's 0.703 at the three decimals the paper reports; it is not a match to four decimals, and it comes from the floor/round run, not the faithful one.

The committed figures in `core/results/figs/` (Figs. 2 to 5 style) were regenerated from the faithful CSV with the plotting functions in `core/code/eval.py` and are titled "FUS faithful". The earlier set, drawn from the floor/round run (10 k values, annotated with the paper's values), is kept in `core/past_tests/results/figs/`.

## Replication notes

There are two separate post-processing protocols and they should not be mixed:

1. **Faithful (the repository's primary result).** Prediction is paper Eq. 19 with the exclusion rule, no clipping to [1, 5] (changed in v1.1.0, see CHANGELOG) and no rounding. Errors are computed on the real-valued predictions. MAE_data and RMSE average over the predictions actually made, so missing predictions are not counted as errors; they are reported through CR.
2. **Floor/round (a diagnostic hypothesis).** Dataset-level metrics (MAE_data, RMSE) use `floor(prediction)` and user-level metrics (MAE_users, RMSE_users) use `round(prediction)`; CR is unchanged. This is not described in the paper; it is the best fit found by trial, in `cross_check/past_tests/` and `core/past_tests/`. The committed floor/round CSV was produced on 2026-05-09, before the v1.1.0 audit: it covers 10 k values (400 rows, not the audit's 26), and the script that made it (`cross_check/past_tests/code/eval_warm_paper_curve_approx.py`) clips predictions to [1, 5] before and after the floor/round step. It has not been regenerated.

Against the paper, the faithful implementation gives lower MAE_data and RMSE (10 to 15 percent) but higher MAE_users (about 5 percent), while coverage is close. Under floor/round all k=50 metrics are within 3 percent of the paper, but at k=1 MAE_data and RMSE are still about 4 percent low. That is an empirical fit to figures and text, not an exact match and not evidence that the paper used this procedure. With the same post-processing PF and GIM overshoot the paper (see `baselines/past_tests/README.md`).

The error-metric gaps do not all point the same way, so a single "evaluation artefact" explanation is not established. Using the paper's Fig. 6 values at k=10 (read by eye), this replication's MAE_users is higher than the paper's for FUS (+5.4%) and lower for GIM (-1.4%), PF (-7.2%) and CF (-16%).

Convention differences between streams, relevant when comparing systems:

- `core/` FUS and CF: no clipping in the code now. The committed CF CSV was produced earlier, when this function clipped to [1, 5].
- `baselines/` PF and GIM: predictions are still clipped to [1, 5] (`baselines/code/pf.py`, `baselines/code/gim.py`). Contract Section 4.1 says no clipping; the baselines and the committed CF results were not re-run after that change.
- The sanity test for FUS at k=50 uses an upper limit of 0.793 for MAE_users, wider than the 0.723 in contract Section 10; the faithful value 0.7415 is outside the Section 10 band.

The investigation log lives in `cross_check/REPRO_DEBUG_NOTES.md` and the per-batch hypothesis scripts in each stream's `past_tests/` folder.

## Reproducibility

- Splits: `KFold(n_splits=10, shuffle=True, random_state=42)` is used by all three streams (`core/code/eval.py`, `cross_check/code/eval_fus.py`, `baselines/code/eval.py`), applied to the filtered ratings in `u.data` order, so the fold indices are the same given the same filtered rows. See `shared_contract.md` Section 3. The core, cross_check and baselines (PF and GIM) loaders were checked to return identical rows in identical order, and the folds from `KFold` on a list of triples and on `np.arange(n)` are identical. Core and cross_check use `value_counts().nlargest(497)` and baselines use `groupby().size().sort_values().head(497)`, which could differ only on a tie at the 497th rating count; there is none (the 497th user has 60 ratings and the 498th has 59).
- Hardware: no GPU is needed. The authors report Python 3.14 on Windows 11 with 16 GB RAM; Python 3.14 was not re-checked.
- Tests: `python run_all.py --tests` (pytest) checks the committed CSVs (columns, CR at k=1 about 0.60, CR monotone in k, sanity bands, system ordering at k=10) and two data-free checks: that each script's data path points to `<repo>/ml-100k`, and that the core and cross_check FUS code give the same predictions on a small synthetic fixture. These tests do not recompute any committed result. The core and cross_check FUS CSVs are byte-identical because the core stream copies the cross_check file, so agreement between those two CSVs is by construction. The two FUS modules are structurally close, so the synthetic check is a consistency check rather than an independent derivation.
- Software: no GPU, no PyTorch. The scientific stack is NumPy, pandas, SciPy, scikit-learn and matplotlib.

### Latest full rerun

All three streams were run with `python run_all.py` (baselines with `deap` installed and UTF-8 output) in a scratch copy of the repository, so the committed CSVs and figures were not overwritten, on the MovieLens 100k files from GroupLens (zip checksum checked against the published MD5). Versions are listed under [Requirements](#requirements). Summary of rerun versus committed results:

| Result | Rerun versus committed |
|---|---|
| Loaders (core, cross_check, baselines PF and GIM) | 497 users, 903 items, 79,432 ratings in all four; identical rows and order |
| FUS (cross_check, 1040 rows; core is a copy) | alpha=0 per-fold values within 0.0006 (the committed file is rounded to 4 decimals, the script writes full precision). alpha 0.7 to 0.9: per-fold values differ by up to 0.010 and CR by up to 0.006, fold means over all (alpha, k) by at most 0.0012. The neighbor sort is sensitive to tie order (on fold 0, alpha=0.8, k=1 MAE_users was 0.8972 with the default sort, 0.8933 with a stable sort and 0.8896 committed). Headline alpha=0 means at k=1, 10, 50 agree to 4 decimals except k=1 MAE_users 0.9007 versus 0.9008. |
| CF (core) | CR identical. MAE and RMSE higher in the rerun (k=10 MAE_users 0.7933 versus 0.7809, per-fold differences up to 0.047). The committed CF file is reproduced exactly (difference 0.0 on all 10 folds at k=1, 10, 50) when predictions are clipped to [1, 5], so it is the pre-audit clipped run. |
| PF (baselines) | Not reproduced: k=10 MAE_users 0.8563 versus 0.8447 and CR 0.8976 versus 0.9191 committed. The rerun equals the committed theta=0.3 row of `baselines/past_tests/results/repro_pf_theta_sweep.csv` exactly, so the committed `results_PF_A_warm.csv` does not come from the current code, nor from the theta 0.3, 0.4 or 0.5 raw variants recorded there (the only ones compared). |
| GIM (baselines) | k=10 MAE_users 0.8229 versus 0.8170, CR 0.9035 versus 0.9072. The genetic algorithm is not deterministic: two fits of fold 0 gave MAE_users 0.8243 and 0.8242 and CR 0.9047 and 0.9031 (DEAP's tournament selection uses an unseeded random generator). |

Committed results were not replaced. With the rerun values the k=10 MAE_users order is FUS 0.7788, CF 0.7933, GIM 0.8229, PF 0.8563 and the CR order is FUS 0.9796, CF 0.9299, GIM 0.9035, PF 0.8976, so the committed headline table (Headline result above) differs from a fresh run for CF, PF and GIM. The open questions about re-running with or without clipping and about seeding the GA are with the repository owner.

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
- Documentation, results CSVs, figures, and research notes: [CC-BY 4.0](LICENSE). The full legal code is included in [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt).

Under CC-BY 4.0, academic and research users are legally required to cite this work when using any portion of the documentation, results, or replication notes. See the [Citation](#citation) section above for the exact format. Citation is also expected when using only the code, even though the MIT license does not strictly enforce it.

The original IEEE Access paper PDF is included for academic reference under fair-use quotation and is not covered by either of the above licenses; it is the copyright of its authors and IEEE. If the publisher requires its removal, open an issue and it will be replaced with a link.

## Contributing

Bug reports, replication-hypothesis additions, and test-coverage improvements are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- D'Aniello, Della Corte, and Gaeta for the FUS paper and the algorithm specification we replicated.
- Hao et al. (2016) for the PF baseline.
- Al-Shamri and Bharadwaj (2008) for the GIM baseline; the GIM implementation in this repository adapts feature extraction from [theanilbajar/Fuzzy-Genetic-Recommender-System](https://github.com/theanilbajar/Fuzzy-Genetic-Recommender-System).
- GroupLens Research for the MovieLens 100k dataset.
