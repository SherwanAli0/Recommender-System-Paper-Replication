---
name: Bug report
about: Report something that's broken or producing wrong numbers
title: "[BUG] "
labels: bug
assignees: ''
---

## What happened

<!-- A clear description of what went wrong. -->

## What you expected

<!-- What should have happened instead. Reference paper section / contract value if relevant. -->

## To reproduce

Steps to reproduce the behaviour:
1. Run `...`
2. Observe output `...`
3. Compare to expected `...`

## Error output / log

```
<paste any traceback or unexpected numbers here>
```

## Environment

- OS: <Windows / macOS / Linux>
- Python version: <output of `python --version`>
- Repo commit: <output of `git rev-parse HEAD`>
- Pip freeze for the relevant packages:
  ```
  numpy==
  pandas==
  scipy==
  scikit-learn==
  ```

## Has a sanity test failed?

- [ ] `python -m pytest tests/` produces failures (paste below if yes)
- [ ] CR at k=1 has changed
- [ ] Numbers are outside the pass-band defined in `shared_contract.md` §10
- [ ] None of the above; this is a new failure mode
