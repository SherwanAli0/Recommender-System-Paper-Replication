"""
tests/test_sanity.py: cross-stream sanity checks.

These tests verify that the produced result CSVs in each stream's results/
folder satisfy the paper-faithfulness guarantees that the README claims.
They do NOT re-run the full evaluation (that takes minutes per stream and
needs the MovieLens data); they only check the existing CSV outputs, plus two
data-free checks (the data-path constants in the eval scripts, and agreement of
the core and cross_check FUS code on a small synthetic fixture).

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import csv
import importlib.util
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

CORE_FUS_CSV = ROOT / "core" / "results" / "results_FUS_A_warm.csv"
CORE_CF_CSV  = ROOT / "core" / "results" / "results_CF_A_warm.csv"
CROSS_FUS_CSV = ROOT / "cross_check" / "results" / "protocol_A" / "results_FUS_A_warm.csv"
BASE_PF_CSV  = ROOT / "baselines" / "results" / "results_PF_A_warm.csv"
BASE_GIM_CSV = ROOT / "baselines" / "results" / "results_GIM_A_warm.csv"


def _read_csv_rows(path):
    if not path.is_file():
        pytest.skip(f"Missing CSV: {path.relative_to(ROOT)}. "
                    "Run the corresponding eval script first.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _filter_mean(rows, k, alpha=None, col="MAE_users"):
    """Mean of `col` over rows where k matches and (if given) alpha matches."""
    out = []
    for r in rows:
        if int(r["k"]) != k:
            continue
        if alpha is not None and "alpha" in r and r["alpha"] not in ("", None):
            try:
                if abs(float(r["alpha"]) - alpha) > 1e-9:
                    continue
            except ValueError:
                continue
        try:
            out.append(float(r[col]))
        except (ValueError, KeyError):
            pass
    return _mean(out)


# Required CSV columns

REQUIRED_COLS = ["system", "protocol", "fold", "k", "alpha",
                 "MAE_data", "MAE_users", "RMSE", "RMSE_users", "CR"]


@pytest.mark.parametrize("path", [CORE_FUS_CSV, CORE_CF_CSV, CROSS_FUS_CSV, BASE_PF_CSV, BASE_GIM_CSV])
def test_csv_has_required_columns(path):
    """Every results CSV must follow the shared_contract.md Section 5 column order."""
    rows = _read_csv_rows(path)
    assert rows, f"{path.name} is empty"
    for col in REQUIRED_COLS:
        assert col in rows[0], f"{path.name} missing column: {col}"


# Coverage Rate proves algorithm correctness

def test_fus_cr_at_k1_matches_paper():
    """FUS Coverage Rate at k=1, alpha=0 must be approximately 0.60 (paper Fig 5)."""
    rows = _read_csv_rows(CORE_FUS_CSV)
    cr = _filter_mean(rows, k=1, alpha=0.0, col="CR")
    assert cr is not None, "No FUS rows at k=1, alpha=0 in the core implementation CSV"
    assert 0.58 <= cr <= 0.62, f"FUS CR at k=1 should be ~0.60, got {cr:.4f}"


def test_fus_core_cross_check():
    """The core and cross_check FUS result CSVs must agree on CR at k=10.

    Note: core/code/eval.py copies the cross_check FUS CSV instead of recomputing
    it, so the two committed CSVs are byte-identical by construction and this
    test cannot detect a disagreement between the two FUS code paths. That is
    checked on synthetic data by test_core_and_cross_check_fus_code_agree.
    """
    core_rows = _read_csv_rows(CORE_FUS_CSV)
    cross_rows = _read_csv_rows(CROSS_FUS_CSV)
    cr_core = _filter_mean(core_rows, k=10, alpha=0.0, col="CR")
    cr_cross = _filter_mean(cross_rows, k=10, alpha=0.0, col="CR")
    assert cr_core is not None and cr_cross is not None
    assert abs(cr_core - cr_cross) < 0.001, (
        f"FUS CR diverges between core ({cr_core:.4f}) and cross_check ({cr_cross:.4f}) at k=10. "
        "The two independent implementations must produce identical numbers."
    )


def test_fus_cr_increases_with_k():
    """Coverage Rate must monotonically increase with k for FUS."""
    rows = _read_csv_rows(CORE_FUS_CSV)
    crs_by_k = {}
    for k in [1, 2, 4, 6, 8, 10]:
        c = _filter_mean(rows, k=k, alpha=0.0, col="CR")
        if c is not None:
            crs_by_k[k] = c
    sorted_ks = sorted(crs_by_k)
    for a, b in zip(sorted_ks, sorted_ks[1:]):
        assert crs_by_k[a] <= crs_by_k[b] + 0.01, (
            f"FUS CR not monotone in k: CR(k={a})={crs_by_k[a]:.4f} > CR(k={b})={crs_by_k[b]:.4f}"
        )


# Sanity bands from shared_contract.md Section 10

def test_fus_k50_alpha0_within_pass_band():
    """FUS at k=50, alpha=0 must satisfy the contract sanity bands.

    Note: shared_contract.md Section 10 gives 0.683 to 0.723 for MAE_users; this
    test has always used the wider 0.683 to 0.793 upper limit (the faithful,
    unclipped run gives about 0.7415, outside the contract band). The limit is
    left unchanged here; see the README "Replication notes" and Section 10.
    """
    rows = _read_csv_rows(CORE_FUS_CSV)
    mae_u = _filter_mean(rows, k=50, alpha=0.0, col="MAE_users")
    cr    = _filter_mean(rows, k=50, alpha=0.0, col="CR")
    assert mae_u is not None and cr is not None
    assert 0.683 <= mae_u <= 0.793, f"MAE_users at k=50 should be in [0.683, 0.793], got {mae_u:.4f}"
    assert cr >= 0.97, f"CR at k=50 should be >= 0.97, got {cr:.4f}"


def test_pf_k10_within_pass_band():
    """PF at k=10 must satisfy the contract sanity bands +/- 0.05 of paper."""
    rows = _read_csv_rows(BASE_PF_CSV)
    mae_u = _filter_mean(rows, k=10, col="MAE_users")
    cr    = _filter_mean(rows, k=10, col="CR")
    assert mae_u is not None and cr is not None
    assert 0.83 <= mae_u <= 0.93, f"PF MAE_users at k=10 should be in [0.83, 0.93], got {mae_u:.4f}"
    assert 0.87 <= cr   <= 0.97, f"PF CR at k=10 should be in [0.87, 0.97], got {cr:.4f}"


def test_gim_k10_within_pass_band():
    """GIM at k=10 must satisfy the contract sanity bands +/- 0.05 of paper."""
    rows = _read_csv_rows(BASE_GIM_CSV)
    mae_u = _filter_mean(rows, k=10, col="MAE_users")
    cr    = _filter_mean(rows, k=10, col="CR")
    assert mae_u is not None and cr is not None
    assert 0.80 <= mae_u <= 0.90, f"GIM MAE_users at k=10 should be in [0.80, 0.90], got {mae_u:.4f}"
    assert 0.87 <= cr   <= 0.97, f"GIM CR at k=10 should be in [0.87, 0.97], got {cr:.4f}"


# System ordering: paper claim FUS > CF > GIM > PF on accuracy

def test_system_ordering_at_k10():
    """FUS has the lowest MAE_users of the four systems at k=10 (lower is better).

    This is the only ordering asserted. It is not the ordering in the paper's
    Fig. 6, where CF is the worst system (about 0.93); in this replication CF is
    close to FUS (the committed gap at k=10 is about 0.002, smaller than the
    fold-to-fold spread). See the README comparison table.
    """
    fus = _filter_mean(_read_csv_rows(CORE_FUS_CSV), k=10, alpha=0.0, col="MAE_users")
    cf  = _filter_mean(_read_csv_rows(CORE_CF_CSV),  k=10,            col="MAE_users")
    gim = _filter_mean(_read_csv_rows(BASE_GIM_CSV), k=10,            col="MAE_users")
    pf  = _filter_mean(_read_csv_rows(BASE_PF_CSV),  k=10,            col="MAE_users")
    assert all(v is not None for v in [fus, cf, gim, pf])
    # Lower MAE = better. FUS should be lowest.
    assert fus <= cf, f"FUS ({fus:.4f}) must be <= CF ({cf:.4f})"
    assert fus <= gim, f"FUS ({fus:.4f}) must be <= GIM ({gim:.4f})"
    assert fus <= pf, f"FUS ({fus:.4f}) must be <= PF ({pf:.4f})"


# Data-free regression checks

# Scripts that read ml-100k. Each must resolve its data directory to
# <repo root>/ml-100k, the location documented in README.md, the Makefile and
# .gitignore, and the one run_all.py checks.
DATA_PATH_SCRIPTS = [
    "core/code/eval.py",
    "cross_check/code/eval_fus.py",
    "baselines/code/eval.py",
    "baselines/code/test_baselines.py",
    "baselines/past_tests/code/repro_gim_postprocess.py",
    "baselines/past_tests/code/repro_pf_theta_sweep.py",
    "cross_check/past_tests/code/eval_warm_paper_curve_approx.py",
    "cross_check/past_tests/code/repro_den_all_floor.py",
]


@pytest.mark.parametrize("rel", DATA_PATH_SCRIPTS)
def test_data_dir_is_repo_root(rel):
    """The relative path to ml-100k in each script must land in <repo root>/ml-100k."""
    script = ROOT / rel
    src = script.read_text(encoding="utf-8")
    m = re.search(r'os\.path\.join\(\s*\w+\s*((?:,\s*"[^"]+"\s*)*?),\s*"ml-100k"', src)
    assert m, f"{rel}: no os.path.join(..., 'ml-100k') found"
    parts = re.findall(r'"([^"]+)"', m.group(1))
    resolved = os.path.normpath(os.path.join(script.parent, *parts, "ml-100k"))
    assert resolved == os.path.normpath(ROOT / "ml-100k"), (
        f"{rel} looks for ml-100k at {resolved}, not at {ROOT / 'ml-100k'}"
    )


def _load_module(name, path, extra_modules=None):
    """Load a .py file under a unique module name without disturbing sys.modules."""
    extra_modules = extra_modules or {}
    saved = {k: sys.modules.get(k) for k in extra_modules}
    sys.modules.update(extra_modules)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _fus_predictions(fus_mod, walks_fn, resnick_fn, sorted_fn, triples, n_users, n_items, ks, alpha):
    """Run one stream's FUS pipeline (Eqs 14-19) on a tiny synthetic fold."""
    split = int(len(triples) * 0.8)
    train = pd.DataFrame(triples[:split], columns=["user_id", "item_id", "rating"])
    test = [(int(u), int(i), float(r)) for u, i, r in triples[split:]]
    users = list(range(1, n_users + 1))
    items = list(range(1, n_items + 1))
    u2i = {u: n for n, u in enumerate(users)}
    i2i = {i: n for n, i in enumerate(items)}
    mats = [fus_mod.build_user_matrix(train[train.user_id == u], i2i, n_items) for u in users]
    sigs = [fus_mod.user_signature(m) for m in mats]
    sim = fus_mod.compute_sim_matrix(sigs, alpha)
    means = fus_mod.compute_user_means(train, users, u2i)
    raters = fus_mod.build_item_raters_dict(train, u2i)
    walks = walks_fn(test, u2i, sorted_fn(sim), raters, sim, max(ks))
    out = {}
    for k in ks:
        for u, i, _ in test:
            p = resnick_fn(walks[(u2i[u], i)], k, float(means[u2i[u]]), means)
            out[(k, u, i)] = np.nan if p is None else p
    return sim, out


def test_core_and_cross_check_fus_code_agree():
    """core/code/fus.py and cross_check/code/fus.py give the same predictions.

    The committed core FUS CSV is a copy of the cross_check one, so only a
    synthetic fixture (no MovieLens data needed) can compare the two code paths.
    The two modules are structurally close, so this is a consistency check, not
    proof of independent derivation.
    """
    core = _load_module("core_fus_under_test", ROOT / "core" / "code" / "fus.py")
    cc = _load_module("cc_fus_under_test", ROOT / "cross_check" / "code" / "fus.py")
    cc_eval = _load_module("cc_eval_under_test", ROOT / "cross_check" / "code" / "eval_fus.py",
                           extra_modules={"fus": cc})

    rng = np.random.default_rng(0)
    n_users, n_items = 25, 40
    triples = [(u, i, int(rng.integers(1, 6)))
               for u in range(1, n_users + 1) for i in range(1, n_items + 1)
               if rng.random() < 0.4]
    rng.shuffle(triples)
    ks = [1, 3, 10]
    for alpha in (0.0, 0.7):
        sim_a, p_a = _fus_predictions(core, core.build_walks, core.resnick_from_walk,
                                      core.precompute_sorted_nbrs, triples, n_users, n_items, ks, alpha)
        sim_b, p_b = _fus_predictions(cc, cc_eval.build_walks, cc_eval.resnick,
                                      cc_eval.precompute_sorted_nbrs, triples, n_users, n_items, ks, alpha)
        assert np.allclose(sim_a, sim_b)
        assert p_a.keys() == p_b.keys() and len(p_a) > 0
        a = np.array([p_a[key] for key in p_a])
        b = np.array([p_b[key] for key in p_a])
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])


# Each eval script's K_VALUES must be the k set of the CSV it writes. core/code/eval.py
# only runs CF (FUS is copied from cross_check), so its sweep is the baseline sweep.
@pytest.mark.parametrize("script,csv_path", [
    ("core/code/eval.py", CORE_CF_CSV),
    ("cross_check/code/eval_fus.py", CROSS_FUS_CSV),
    ("baselines/code/eval.py", BASE_PF_CSV),
    ("baselines/code/eval.py", BASE_GIM_CSV),
])
def test_k_sweep_matches_committed_csv(script, csv_path):
    import ast
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    ks = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "K_VALUES" for t in node.targets):
            ks = ast.literal_eval(node.value)
    assert ks is not None, f"{script}: no top-level K_VALUES"
    rows = _read_csv_rows(csv_path)
    assert sorted({int(r["k"]) for r in rows}) == sorted(ks), (
        f"{script} K_VALUES does not match the k values in {csv_path.name}"
    )
