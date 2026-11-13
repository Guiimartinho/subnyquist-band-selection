"""E26: windows without integer orders, prominence, median of the decimated stream (guided.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

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


def _relabel_tokens() -> dict:
    """TOKENS of tools/relabel_map.py (old label -> English label), or skip if it is absent."""
    path = Path(__file__).resolve().parents[3] / "tools" / "relabel_map.py"
    if not path.exists():
        pytest.skip("tools/relabel_map.py is not part of this repository")
    spec = importlib.util.spec_from_file_location("relabel_map", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.TOKENS


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_random_band_seed_comes_from_the_legacy_cleaning_label():
    """Values computed before the labels were renamed, from "bruto" and "mediana3"."""
    rule = _load("e26rule", "run_paderborn_rule.py")
    assert rule.segment_seed("KA01", "N15_M07_F10", 2, 4, "raw") == 3882564165
    assert rule.segment_seed("KI21", "N15_M01_F10", 20, 32, "median3") == 1367934109
    tokens = _relabel_tokens()
    assert all(tokens[old] == new for new, old in rule.LEGACY_CLEAN.items())


def test_analyses_sort_by_the_legacy_labels():
    an = _load("e26an", "analyze_paderborn.py")
    assert an.legacy_label("zone1|wideband_rule") == "zona1|banda_larga_regra"
    assert an.legacy_label("antialias|guided") == "antialias|guiado"
    tokens = _relabel_tokens()
    assert all(tokens[old] == new for new, old in an.LEGACY_LABELS.items())
