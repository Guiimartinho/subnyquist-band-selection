"""E24: remoção de picos, guarda, teste por rolamento e Holm."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "e24_run", Path(__file__).resolve().parents[1] / "run.py"
)
e24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e24)


def test_despike_removes_isolated_spikes_and_keeps_the_rest():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20_000)
    idx = np.arange(500, 19_500, 997)
    x[idx] += 25.0
    y = e24.despike(x)
    assert np.all(np.abs(y[idx]) < 6.0)
    changed = np.flatnonzero(y != x)
    assert set(idx.tolist()) <= set(changed.tolist())
    assert changed.size <= idx.size + 5  # quase nada além dos picos


def test_cluster_test_detects_a_consistent_gain_and_not_a_null():
    a = {("OR", f"A{i}"): 0.8 for i in range(6)} | {("IR", f"B{i}"): 0.7 for i in range(6)}
    b = {k: v - 0.2 for k, v in a.items()}
    res = e24.cluster_test(a, b, 500, 0)
    assert res["diff_points"] == pytest.approx(20.0)
    assert res["p_one_sided"] < 0.01
    null = e24.cluster_test(a, a, 500, 0)
    assert null["p_one_sided"] == 1.0  # Δ* = 0 conta contra, como no E15


def test_holm():
    assert e24.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
