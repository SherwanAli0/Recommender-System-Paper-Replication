# Prompt for AI — Person 3 (Reference Baselines)

Paste everything below the divider into your AI tool as the first message in a new conversation.

---

You are helping me implement two **literature baseline recommenders** for a 3-person university course project on recommender systems. You will be working from inside the project folder; relative paths in this prompt are correct as long as the folder structure is preserved.

## Your role

**Stream P3 — Reference Baselines.** Implement two recommender systems from prior literature that the FUS paper (the project's base paper) uses as comparison baselines. Two other people (Person 1, Person 2) are independently building the FUS reproduction and the novelty; you do not interact with their work mid-stream. Your output is what makes the final comparison table complete.

## Files to read FIRST

1. `../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf` — the project's base paper. Read **only Section V** (Evaluation) — this tells you how PF and GIM were used. The base paper does **not** describe how PF and GIM work internally; for that you need the source papers below.
2. `../shared_contract.md` — the binding interface document. **Identical filtering, fold split, metric formulas, and CSV format to the other two streams. Do not deviate.**

## The two baseline papers you must implement

### PF — Probabilistic Filtering (the base paper's reference [12])
> J. Hao, Y. Yan, G. Wang, L. Gong, B. Zhao, *"A probability-based hybrid user model for recommendation system,"* Mathematical Problems in Engineering, 2016.
- Open-access on Hindawi. Search for the PDF; download.
- The PF method computes hybrid user probabilities for predicting ratings.
- The base paper notes PF uses fuzzy probability (not fuzzy logic), and is structurally close to FUS but uses probability-based similarity instead of kindredness.

### GIM — Fuzzy-Genetic Method (the base paper's reference [13])
> M. Y. H. Al-Shamri, K. K. Bharadwaj, *"Fuzzy-genetic approach to recommender systems based on a novel hybrid user model,"* Expert Systems with Applications, vol. 35, no. 3, pp. 1386-1399, Oct 2008.
- Elsevier — may need university access.
- Hybrid user model + genetic algorithm for tuning user-similarity weights.

## STEP 1 — HUNT FOR EXISTING IMPLEMENTATIONS BEFORE WRITING ANYTHING

Before implementing from scratch, do a thorough search:
- GitHub: query for `"probability-based hybrid user model" recommendation`, `Hao 2016 fuzzy probability recommender`, `fuzzy genetic recommender Al-Shamri`, `fuzzy genetic MovieLens`.
- Papers With Code: search for both papers by title.
- Authors' personal pages, ResearchGate, university lab pages.
- The base paper's references [12] and [13] for any code-release citations.

**If you find usable code, base your implementation on it and cite the URL.** Per the no-original-code rule, this is the preferred path. If no public implementation exists, implement from the source papers' equations — citing "Hao 2016 §X Eq Y" or "Al-Shamri 2008 §X Eq Y" in comments above each block.

## What to build (3 files in `code/`)

The empty stubs are already in place. Each contains structured comments describing the contents.

### `code/pf.py` — Hao 2016 PF
- Read Hao 2016. Identify the probabilistic user-model formulas and the prediction equation.
- Implement the data loading & filtering per `../shared_contract.md` §2.3 (must produce 497 × 903 × 79,432).
- Implement the PF predictor.
- Cite Hao 2016 paper sections, plus any GitHub source you adapted from.

### `code/gim.py` — Al-Shamri & Bharadwaj 2008 GIM
- Read Al-Shamri 2008. Identify the fuzzy user model and the genetic-algorithm fitness/selection.
- For the genetic-algorithm part, use the `deap` library if implementing from scratch:
  https://deap.readthedocs.io/en/master/
- Implement data loading & filtering per the shared contract (same 497 × 903 × 79,432).
- Implement the GIM predictor.
- Cite Al-Shamri 2008 paper sections plus library docs.

### `code/eval.py` — runs Protocol A for both
- Load filtered dataset (verify shape: 497 × 903 × 79,432 — fail loudly if wrong).
- Same fold split as everyone: `KFold(n_splits=10, shuffle=True, random_state=42)` per `../shared_contract.md` §3.
- Sweep `k ∈ {1, 2, 4, 6, 8, 10}`. **Do NOT sweep beyond k=10** — both PF and GIM degrade at higher k (this is why the base paper's Figs 6–7 cap at k=10).
- For each (system, fold, k), compute the 5 metrics from `../shared_contract.md` §4. Use `alpha = 0.0` in the CSV (PF and GIM have no α-cut).
- Write `results/results_PF_A_warm.csv` and `results/results_GIM_A_warm.csv`.

## Sanity-check numbers (do this before the full sweep)

The base paper reports these approximate values from its Figs 6–7 at k=10:

| System | `MAE_users` target | `CR` target |
|---|---|---|
| PF | ~0.88 (pass band [0.83, 0.93]) | ~0.92 (pass band [0.87, 0.97]) |
| GIM | ~0.85 (pass band [0.80, 0.90]) | ~0.92 (pass band [0.87, 0.97]) |

These are older papers; small variation is expected. If you are wildly outside the pass band, your implementation has diverged from the source paper — pause and ask the human.

## NO-ORIGINAL-CODE RULE — STRICT

Same as the other streams: every code block must be adapted from a real public source — for PF and GIM, the most natural sources are the source papers' equations themselves (cited as "Hao 2016 §X Eq Y") and any GitHub implementations you find. If you cannot find a source for a snippet, **STOP and tell the human** — do not silently invent code.

## Working order

1. Confirm you have read the base paper §V, the shared contract, and this prompt.
2. **Search for existing public implementations of PF and GIM. Spend at least an hour on this before writing anything.**
3. Confirm the dataset filter produces 497 × 903 × 79,432.
4. Implement PF — sanity-check at k=10, then run the full sweep.
5. Implement GIM — sanity-check at k=10, then run the full sweep.
6. Final check: every code block has a source citation.

## If something is going wrong

If after a few days it becomes clear that one of PF / GIM cannot be faithfully implemented within the time budget, **tell the human immediately**. The fallback is to cite the base paper's reported numbers in the comparison table with a footnote rather than ship a buggy implementation. Do not invent numbers.
