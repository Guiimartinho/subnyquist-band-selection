"""E26: windows without integer orders, prominence, median of the decimated stream (guided.py)."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "e26guided", Path(__file__).resolve().parents[1] / "guided.py"
)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


def test_windows_exclude_integer_orders():
    # Paderborn BPFO at ±2 %: [2.993, 3.115]; the exclusion 3.0 ± 0.015 eats the window start
    assert g.windows(3.0543, 1, 0.02, 0.015) == [(3.015, 3.0543 * 1.02)]
    # BPFI at ±2 %: [4.847, 5.045]; the exclusion 5.0 ± 0.015 cuts it in two
    segs = g.windows(4.9457, 1, 0.02, 0.015)
    assert len(segs) == 2
    assert abs(segs[0][1] - 4.985) < 1e-9 and abs(segs[1][0] - 5.015) < 1e-9
    # with no integer order inside, the window stays whole
    assert g.windows(3.5, 1, 0.02, 0.015) == [(3.5 * 0.98, 3.5 * 1.02)]


def test_prominence_finds_a_line_and_ignores_excluded_bins():
    o = np.linspace(2.9, 3.2, 301)
    mag = np.ones_like(o)
    mag[np.argmin(np.abs(o - 3.06))] = 10.0  # fault line
    mag[np.argmin(np.abs(o - 3.0))] = 50.0  # shaft harmonic, excluded
    segs = g.windows(3.0543, 1, 0.02, 0.015)
    assert abs(g.prominence(o, mag, segs) - 10.0) < 1e-9


def test_median3_removes_an_isolated_spike():
    x = np.zeros(100)
    x[50] = 30.0
    assert np.max(np.abs(g.median3(x))) == 0.0
