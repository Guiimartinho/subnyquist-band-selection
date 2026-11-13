"""E26: janelas sem ordens inteiras, proeminência e mediana do fluxo decimado (guided.py)."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "e26guided", Path(__file__).resolve().parents[1] / "guided.py"
)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


def test_windows_exclude_integer_orders():
    # BPFO do Paderborn a ±2 %: [2,993; 3,115]; a exclusão 3,0 ± 0,015 come o começo da janela
    assert g.windows(3.0543, 1, 0.02, 0.015) == [(3.015, 3.0543 * 1.02)]
    # BPFI a ±2 %: [4,847; 5,045]; a exclusão 5,0 ± 0,015 a corta em dois
    segs = g.windows(4.9457, 1, 0.02, 0.015)
    assert len(segs) == 2
    assert abs(segs[0][1] - 4.985) < 1e-9 and abs(segs[1][0] - 5.015) < 1e-9
    # sem ordem inteira dentro, a janela fica inteira
    assert g.windows(3.5, 1, 0.02, 0.015) == [(3.5 * 0.98, 3.5 * 1.02)]


def test_prominence_finds_a_line_and_ignores_excluded_bins():
    o = np.linspace(2.9, 3.2, 301)
    mag = np.ones_like(o)
    mag[np.argmin(np.abs(o - 3.06))] = 10.0  # raia de falha
    mag[np.argmin(np.abs(o - 3.0))] = 50.0  # harmônica do eixo, excluída
    segs = g.windows(3.0543, 1, 0.02, 0.015)
    assert abs(g.prominence(o, mag, segs) - 10.0) < 1e-9


def test_median3_removes_an_isolated_spike():
    x = np.zeros(100)
    x[50] = 30.0
    assert np.max(np.abs(g.median3(x))) == 0.0
