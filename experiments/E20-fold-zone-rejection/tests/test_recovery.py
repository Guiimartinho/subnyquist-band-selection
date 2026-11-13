"""Pre-registered E20 recovery criterion (recovery.py)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "e20_recovery", Path(__file__).resolve().parents[1] / "recovery.py"
)
rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rec)


def test_verdict_follows_the_preregistered_thresholds():
    assert rec.verdict(20, 0.009) == "recupera"
    assert rec.verdict(20, 0.01) == "inconclusivo"  # p < 0.01, strict
    assert rec.verdict(25, 0.02) == "inconclusivo"
    assert rec.verdict(15, 1e-6) == "inconclusivo"
    assert rec.verdict(10, 1e-6) == "nao recupera"


def test_mcnemar_matches_the_binomial_tail():
    assert rec.mcnemar_exact_p(0, 0) == 1.0
    assert abs(rec.mcnemar_exact_p(0, 6) - 2 / 64) < 1e-12
    assert rec.mcnemar_exact_p(3, 3) == 1.0


def test_compare_counts_discordant_pairs_and_records():
    ctrl = {("O-A-1", "0"): 0, ("O-A-1", "1"): 0, ("O-B-1", "0"): 1, ("O-B-1", "1"): 1}
    arm = {("O-A-1", "0"): 1, ("O-A-1", "1"): 1, ("O-B-1", "0"): 1, ("O-B-1", "1"): 0}
    r = rec.compare(arm, ctrl)
    assert (r["hits"], r["gain"], r["loss"]) == (3, 2, 1)
    assert (r["records_up"], r["records_down"]) == (1, 1)
