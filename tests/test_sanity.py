"""
tests/test_sanity.py - cross-stream sanity checks.

These tests verify that the produced result CSVs in each Person's results/
folder satisfy the paper-faithfulness guarantees that the README claims.
They do NOT re-run the full evaluation (that takes minutes per stream); they
only check the existing CSV outputs.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

P1_FUS_CSV = ROOT / "Person_1_Faithful_Baseline" / "results" / "results_FUS_A_warm.csv"
P1_CF_CSV  = ROOT / "Person_1_Faithful_Baseline" / "results" / "results_CF_A_warm.csv"
P2_FUS_CSV = ROOT / "Person_2_Faithful_FUS" / "results" / "protocol_A" / "results_FUS_A_warm.csv"
P3_PF_CSV  = ROOT / "Person_3_Reference_Baselines" / "results" / "results_PF_A_warm.csv"
P3_GIM_CSV = ROOT / "Person_3_Reference_Baselines" / "results" / "results_GIM_A_warm.csv"


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


# ─── Required CSV columns ────────────────────────────────────────────────────

REQUIRED_COLS = ["system", "protocol", "fold", "k", "alpha",
                 "MAE_data", "MAE_users", "RMSE", "RMSE_users", "CR"]


@pytest.mark.parametrize("path", [P1_FUS_CSV, P1_CF_CSV, P2_FUS_CSV, P3_PF_CSV, P3_GIM_CSV])
def test_csv_has_required_columns(path):
    """Every results CSV must follow the shared_contract.md §5 column order."""
    rows = _read_csv_rows(path)
    assert rows, f"{path.name} is empty"
    for col in REQUIRED_COLS:
        assert col in rows[0], f"{path.name} missing column: {col}"


# ─── Coverage Rate proves algorithm correctness ─────────────────────────────

def test_fus_cr_at_k1_matches_paper():
    """FUS Coverage Rate at k=1, alpha=0 must be approximately 0.60 (paper Fig 5)."""
    rows = _read_csv_rows(P1_FUS_CSV)
    cr = _filter_mean(rows, k=1, alpha=0.0, col="CR")
    assert cr is not None, "No FUS rows at k=1, alpha=0 in Person 1 CSV"
    assert 0.58 <= cr <= 0.62, f"FUS CR at k=1 should be ~0.60, got {cr:.4f}"


def test_fus_p1_p2_cross_check():
    """Person 1 and Person 2 FUS implementations must produce the same CR."""
    p1 = _read_csv_rows(P1_FUS_CSV)
    p2 = _read_csv_rows(P2_FUS_CSV)
    cr1 = _filter_mean(p1, k=10, alpha=0.0, col="CR")
    cr2 = _filter_mean(p2, k=10, alpha=0.0, col="CR")
    assert cr1 is not None and cr2 is not None
    assert abs(cr1 - cr2) < 0.001, (
        f"FUS CR diverges between P1 ({cr1:.4f}) and P2 ({cr2:.4f}) at k=10. "
        "The two independent implementations must produce identical numbers."
    )


def test_fus_cr_increases_with_k():
    """Coverage Rate must monotonically increase with k for FUS."""
    rows = _read_csv_rows(P1_FUS_CSV)
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


# ─── Sanity bands from shared_contract.md §10 ───────────────────────────────

def test_fus_k50_alpha0_within_pass_band():
    """FUS at k=50, alpha=0 must satisfy the contract sanity bands."""
    rows = _read_csv_rows(P1_FUS_CSV)
    mae_u = _filter_mean(rows, k=50, alpha=0.0, col="MAE_users")
    cr    = _filter_mean(rows, k=50, alpha=0.0, col="CR")
    assert mae_u is not None and cr is not None
    assert 0.683 <= mae_u <= 0.793, f"MAE_users at k=50 should be in [0.683, 0.793], got {mae_u:.4f}"
    assert cr >= 0.97, f"CR at k=50 should be ≥ 0.97, got {cr:.4f}"


def test_pf_k10_within_pass_band():
    """PF at k=10 must satisfy the contract sanity bands ±0.05 of paper."""
    rows = _read_csv_rows(P3_PF_CSV)
    mae_u = _filter_mean(rows, k=10, col="MAE_users")
    cr    = _filter_mean(rows, k=10, col="CR")
    assert mae_u is not None and cr is not None
    assert 0.83 <= mae_u <= 0.93, f"PF MAE_users at k=10 should be in [0.83, 0.93], got {mae_u:.4f}"
    assert 0.87 <= cr   <= 0.97, f"PF CR at k=10 should be in [0.87, 0.97], got {cr:.4f}"


def test_gim_k10_within_pass_band():
    """GIM at k=10 must satisfy the contract sanity bands ±0.05 of paper."""
    rows = _read_csv_rows(P3_GIM_CSV)
    mae_u = _filter_mean(rows, k=10, col="MAE_users")
    cr    = _filter_mean(rows, k=10, col="CR")
    assert mae_u is not None and cr is not None
    assert 0.80 <= mae_u <= 0.90, f"GIM MAE_users at k=10 should be in [0.80, 0.90], got {mae_u:.4f}"
    assert 0.87 <= cr   <= 0.97, f"GIM CR at k=10 should be in [0.87, 0.97], got {cr:.4f}"


# ─── System ordering: paper claim FUS > CF > GIM > PF on accuracy ───────────

def test_system_ordering_at_k10():
    """Paper's headline claim: FUS beats CF beats GIM beats PF on MAE_users at k=10."""
    fus = _filter_mean(_read_csv_rows(P1_FUS_CSV), k=10, alpha=0.0, col="MAE_users")
    cf  = _filter_mean(_read_csv_rows(P1_CF_CSV),  k=10,            col="MAE_users")
    gim = _filter_mean(_read_csv_rows(P3_GIM_CSV), k=10,            col="MAE_users")
    pf  = _filter_mean(_read_csv_rows(P3_PF_CSV),  k=10,            col="MAE_users")
    assert all(v is not None for v in [fus, cf, gim, pf])
    # Lower MAE = better. FUS should be lowest.
    assert fus <= cf, f"FUS ({fus:.4f}) must be ≤ CF ({cf:.4f})"
    assert fus <= gim, f"FUS ({fus:.4f}) must be ≤ GIM ({gim:.4f})"
    assert fus <= pf, f"FUS ({fus:.4f}) must be ≤ PF ({pf:.4f})"
