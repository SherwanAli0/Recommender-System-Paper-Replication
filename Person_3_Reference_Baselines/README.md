# Person 3 — Reference Baselines (PF + GIM)

**Role:** Implement the two literature baselines that the FUS paper compares against — PF (Hao et al. 2016) and GIM (Al-Shamri & Bharadwaj 2008). Run both under the paper's "warm" evaluation protocol so that the final comparison table contains all four systems (FUS, CF, PF, GIM) on identical data.

**Time budget:** ~11 days. PF and GIM are independent older papers and your work is the wild card of the project — start by hunting for existing public implementations before writing anything.

---

## What you build

1. **PF (Probabilistic Filtering)** — Hao et al. 2016. The closest competitor to FUS in the paper's comparison; uses fuzzy probability instead of fuzzy logic.
2. **GIM (Fuzzy-Genetic Method)** — Al-Shamri & Bharadwaj 2008. A fuzzy genetic approach, also from the same family of recommenders.
3. **Protocol A evaluation** for both, on the same filtered MovieLens-100k dataset as Person 1 and Person 2.

---

## Step-by-step for the human (you)

1. **Read the base paper** (`../A_Recommendation_System_Based_on_Fuzzy_Signature.pdf`). You only need Section V (Evaluation) for context — the paper does not specify how PF / GIM work internally; for that you need the source papers.
2. **Read `../shared_contract.md`** — dataset paths, filter rules, fold split, metric formulas, output CSV format. **Binding interface.**
3. **HUNT FOR EXISTING IMPLEMENTATIONS FIRST** (this is your most important step):
   - Search GitHub: `"probability-based hybrid user model" recommendation`, `Hao 2016 fuzzy probability recommender`, `fuzzy genetic recommender Al-Shamri`.
   - Check Papers With Code, the authors' personal pages, ResearchGate.
   - The Hao 2016 paper is in *Mathematical Problems in Engineering* (Hindawi, open-access — PDF should be free).
   - The Al-Shamri 2008 paper is in *Expert Systems with Applications* (Elsevier — may need university access).
   - **If you find any usable code, base your implementation on it and cite the URL.** Per the no-original-code rule, this is preferred over re-implementing from scratch.
4. **Open your AI tool** and paste `PROMPT.md` (next to this file).
5. Let the AI fill in the `code/*.py` files. The empty stubs document what each file should contain.
6. **Verify before sweeping:**
   - Filter must produce 497 users × 903 items × 79,432 ratings.
   - PF at k=10 should give MAE_users ≈ 0.88, CR ≈ 0.92 (pass band ±0.05).
   - GIM at k=10 should give MAE_users ≈ 0.85, CR ≈ 0.92 (pass band ±0.05).
7. **Run the sweep** for both systems with k ∈ {1, 2, 4, 6, 8, 10} (capped at 10 because both methods degrade beyond — the FUS paper only compares them up to k=10 for the same reason).
8. **Save CSVs to `results/`** following the naming pattern in `../shared_contract.md` §5.
9. **When done, zip your entire `Person_3_Reference_Baselines/` folder** and send to Ipek.

---

## Files in your folder

```
Person_3_Reference_Baselines/
├── README.md                      (this file)
├── PROMPT.md                      (paste into your AI)
├── code/
│   ├── pf.py                      (PF — Hao 2016)
│   ├── gim.py                     (GIM — Al-Shamri & Bharadwaj 2008)
│   ├── eval.py                    (10-fold CV, sweep, metrics, CSV writers)
│   └── test_baselines.py          (unit tests for filter shape and PF neighbourhood)
├── papers/
│   └── hao2016_9535808_pf.pdf     (source paper for PF)
├── past_tests/                    (theta sweep + floor/round hypothesis tests; see past_tests/README.md)
└── results/
    ├── results_PF_A_warm.csv
    └── results_GIM_A_warm.csv
```

---

## Success criteria

- [ ] Filter produces 497 × 903 × 79,432.
- [ ] PF at k=10: MAE_users in [0.83, 0.93], CR in [0.87, 0.97].
- [ ] GIM at k=10: MAE_users in [0.80, 0.90], CR in [0.87, 0.97].
- [ ] Both CSVs written with exact column names from `../shared_contract.md` §5.
- [ ] Every `.py` file has source citations in comments — for PF and GIM, cite both the source paper AND any GitHub/SO references you adapted from.

---

## Realistic fallback plan

If, after honest searching, no usable public implementation of PF or GIM exists AND re-implementing from the source papers is taking too long (more than ~6 days), tell Ipek immediately. The fallback is to cite the paper's own reported numbers for PF / GIM in the final comparison table with a footnote, rather than risking a buggy reimplementation. **Do not invent numbers. Do not fudge. Tell Ipek.**

---

## Anything unclear

Ask Ipek before guessing. PF and GIM are the trickiest parts of this project — better to pause and check than to ship a wrong implementation.
