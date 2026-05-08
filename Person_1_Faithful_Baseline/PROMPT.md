# Prompt for AI — Person 1 (Faithful Baseline)

Paste everything below the divider into your AI tool as the first message in a new conversation.

---

You are helping me implement the **faithful baseline** for a 3-person university course project on recommender systems. You will be working from inside the project folder; relative paths in this prompt are correct as long as the folder structure is preserved.

## Your role

**Stream P1 — Faithful Baseline.** Implement the paper's original Fuzzy User Signature (FUS) recommender exactly per its equations, plus a standard collaborative-filtering baseline, plus the paper's "warm" evaluation protocol. Two other people are independently working on the novelty (Person 2) and on two other reference baselines (Person 3); you do not interact with their work mid-stream.

## Files to read FIRST, in this order

1. `../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf` — the base paper. Read Sections II (Eqs 1–7, Yager-Reformat origins), IV (Eqs 8–19, the FUS recommendation method), and V (evaluation methodology).
2. `../shared_contract.md` — the binding interface document. Dataset paths, filtering rules, fold split, metric formulas, output CSV format, environment, and the no-original-code rule. **You must follow it identically.**

## What to build (3 files in `code/`)

The empty stubs are already in place. Each contains structured comments describing the contents. Fill them in:

### `code/fus.py` — faithful FUS implementation
Functions to implement (paper sections in parentheses):
- Data loading & filtering (paper §V.A, exact protocol in `../shared_contract.md` §2.3).
- `Mat_ui` per-user 2-D matrix construction (paper §IV.A, Eqs 8–9).
- `RatPop_ui` and `ItemAttr_ui` (paper Eqs 10–13).
- `US_ui` user signature via `min` (paper Eqs 14–15).
- `UK(u_i, u_j)` user kindredness (paper Eqs 16–18). **Asymmetric.** Apply α-cut threshold to signatures *before* the similarity sum.
- `predict(u_i, m, k)` Resnick formula (paper Eq 19) with the **exclusion correction**: if neighbor `u_j` did not rate item `m`, that term is excluded from BOTH numerator AND denominator (paper §IV.B). Clip to [1, 5].

### `code/cf.py` — standard CF baseline
Wrap Surprise's `KNNWithMeans` (memory-based CF, Resnick-style):
- Library doc: https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans
- Same sklearn `KFold(n_splits=10, shuffle=True, random_state=42)` split as FUS.
- Same metrics, same output CSV format.

### `code/eval.py` — runs Protocol A
- Load filtered dataset (verify shape: 497 × 903 × 79,432 — fail loudly if wrong).
- 10-fold CV per `../shared_contract.md` §3.
- Sweep `k ∈ {1, 2, 4, 6, 8, 10, 15, 20, 30, 50}` and `α ∈ {0, 0.7, 0.8, 0.9}`.
- For each (system, fold, k, α), compute the 5 metrics from `../shared_contract.md` §4.
- Write `results/results_FUS_A_warm.csv` and `results/results_CF_A_warm.csv` with exact column names from `../shared_contract.md` §5.
- Generate 4 plots (paper's Figs 2–5) into `results/figs/`.

## Sanity-check numbers (do this before the full sweep)

After implementing FUS, run a single (k=50, α=0) configuration on one fold. Pure FUS should hit:

| Metric | Target | Pass band |
|---|---|---|
| `MAE_data` | 0.83 | [0.81, 0.85] |
| `MAE_users` | 0.703 | [0.683, 0.723] |
| `RMSE` | 1.107 | [1.087, 1.127] |
| `CR` | ~0.99 | [0.97, 1.00] |

If outside the pass band, the implementation is wrong. Common bugs:
- Forgot the "exclude unrated neighbors" rule in Resnick (paper §IV.B).
- Wrong axis when normalising `RatPop` / `ItemAttr` (denominator should be the per-user max, not global max).
- Applied α-cut after the similarity sum instead of filtering signatures first.
- Symmetric similarity instead of asymmetric (must normalise by `|US_ui|` only, not both).

## NO-ORIGINAL-CODE RULE — STRICT

You must **NOT** author original code from scratch. Every code block you produce must be adapted from a real public source:
- GitHub repository — cite full URL in a comment above the block.
- Official library documentation — cite URL.
- Stack Overflow answer — cite question URL.
- The paper's own pseudo-code or numbered equations — cite "paper §IV.B Eq 19" etc.

If you cannot find a public source for a snippet you think is needed, **STOP and tell the human** — do not silently invent code. Trivial glue (imports, basic pandas / numpy operations called as documented) does not need a citation; anything algorithmic does.

The human will inspect every `.py` file and expect to see source URLs in comments.

## Working order

1. Confirm you have read the paper, the shared contract, and this prompt.
2. Confirm the dataset filter produces 497 × 903 × 79,432 (write a tiny script for this first).
3. Implement FUS, sanity-check on one fold, then run the full sweep.
4. Implement CF, run its sweep.
5. Generate plots.
6. Final check: every code block has a source citation. If any doesn't, you violated the rule — fix it.
