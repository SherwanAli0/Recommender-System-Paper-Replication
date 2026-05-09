# Contributing

Thanks for your interest in this paper-replication repository. This is a course-project artefact, but contributions that improve reproducibility, documentation, or test coverage are welcome.

## Ground rules

1. **Faithfulness to the paper is paramount.** Production code under `the per-stream/code/` must implement only what the paper specifies. Hypotheses, tweaks, and "what if" experiments belong in `the per-stream/past_tests/code/`.
2. **Coverage Rate is the algorithm-correctness check.** Any change that breaks `CR ≈ 0.6007 at k=1` for FUS is a regression and will be rejected.
3. **No-original-code rule.** Every algorithmic block must cite a public source (a GitHub URL, a Stack Overflow URL, the paper's equation number, or library docs). See `shared_contract.md` §8.

## How to propose a change

1. **Open an issue first.** Describe the problem, paste the smallest reproducible example, and reference the paper section / equation that motivates the change.
2. **Branch + PR.** Branch off `main`, keep one logical change per PR. Reference the issue.
3. **Run the tests.** Before opening a PR, run `python run_all.py --tests` (or `make tests`). All assertions must pass.
4. **Update README and docstrings** if the change affects user-facing behaviour.

## What we accept

- Documentation fixes (typos, broken links, clarifications).
- New replication hypotheses added to `past_tests/`.
- Test cases that strengthen the sanity-check coverage.
- Cross-platform fixes (Linux, macOS, Windows compatibility).
- Performance improvements that produce identical numbers (verified by tests).

## What we do NOT accept

- Algorithmic tweaks to FUS, CF, PF, or GIM that change the production numbers without a paper-text justification.
- Novelty extensions (Adaptive Genre-Hybrid for cold-start, etc.). This repository is the paper-faithful replication; novelty work belongs in a separate repository.
- Changes to `shared_contract.md` without consensus from all four authors.

## Code style

- Python 3.10+ syntax. Type hints where useful, not mandatory.
- Docstrings cite paper equation numbers in the format `paper §IV.B Eq 19` or external sources as URLs.
- One algorithmic concept per function; no "kitchen-sink" helpers.
- 4-space indent, no tabs. UTF-8 encoded files. LF line endings.

## Questions?

Open an issue or contact the authors listed in `CITATION.cff`.
