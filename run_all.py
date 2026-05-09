"""
run_all.py - cross-platform single-command reproduction.

For users without GNU Make (typically Windows). Runs the cross-check FUS first,
then the reference baselines, then the core FUS + CF implementation (which
reuses the cross-check FUS CSV for figures).

Usage:
    python run_all.py
    python run_all.py --skip core
    python run_all.py --only cross_check
    python run_all.py --tests
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ML100K = ROOT.parent / "ml-100k"


STREAMS = [
    ("cross_check", "cross_check", "code/eval_fus.py"),
    ("baselines",   "baselines",   "code/eval.py"),
    ("core",        "core",        "code/eval.py"),
]


def check_dataset() -> None:
    if not ML100K.is_dir() or not (ML100K / "u.data").is_file():
        print(f"ERROR: dataset not found at {ML100K}")
        print("Download from https://files.grouplens.org/datasets/movielens/ml-100k.zip")
        print("Unzip into the repository root so that ml-100k/u.data exists.")
        sys.exit(1)


def run_stream(label: str, folder: str, script: str) -> int:
    cwd = ROOT / folder / Path(script).parent
    cmd = [sys.executable, Path(script).name]
    print(f"\n=== Running {label}: {folder}/{script} ===")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=cwd).returncode
    dt = time.time() - t0
    print(f"=== {label} finished in {dt:.1f}s (exit code {rc}) ===")
    return rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=[lbl for lbl, *_ in STREAMS],
                   help="Run only this stream")
    p.add_argument("--skip", choices=[lbl for lbl, *_ in STREAMS],
                   help="Skip this stream")
    p.add_argument("--tests", action="store_true",
                   help="Run the test suite instead of the eval streams")
    args = p.parse_args()

    if args.tests:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"], cwd=ROOT
        ).returncode

    check_dataset()

    streams = STREAMS
    if args.only:
        streams = [s for s in streams if s[0] == args.only]
    if args.skip:
        streams = [s for s in streams if s[0] != args.skip]

    failures = []
    for lbl, folder, script in streams:
        rc = run_stream(lbl, folder, script)
        if rc != 0:
            failures.append(lbl)

    print("\n" + ("=" * 60))
    if failures:
        print(f"FAILED streams: {failures}")
        return 1
    print("All streams completed successfully.")
    print("Results CSVs are in core/results/, cross_check/results/, baselines/results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
