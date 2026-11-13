"""E22 helpers: Nyquist zone with a closed edge, and the TAT."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "e22_run", Path(__file__).resolve().parents[1] / "run.py"
)
e22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e22)


def test_band_ending_exactly_at_nyquist_is_in_one_zone():
    assert e22.in_one_zone(12_750.0, 25_000.0, 50_000.0)


def test_band_crossing_a_zone_edge_is_rejected():
    assert not e22.in_one_zone(20_000.0, 30_000.0, 50_000.0)


def test_band_inside_a_higher_zone_is_accepted():
    # zone 3 of 50 kSPS: [75; 100] kHz
    assert e22.in_one_zone(76_000.0, 84_000.0, 50_000.0)


def test_ceiling_at_respects_effective_nyquist_and_fraction():
    assert e22.ceiling_at(25_000.0, 200_000.0) == 25_000.0
    assert e22.ceiling_at(25_000.0, 12_500.0) == 6_250.0
    assert e22.ceiling_at(None, 51_200.0, 0.5) == 12_800.0


CFG = {
    "min_samples": 64,
    "samples_per_rev": 8,
    "order_min": 2.0,
    "kurtogram": {"max_level": 3, "f_min_hz": 500.0},
    "tat": {"folded_share_threshold": 0.5},
}


def test_classic_skips_when_ceiling_is_below_kurtogram_floor():
    """The case that brought down the 1st complete run: ceiling 400 Hz < floor 500 Hz."""
    import numpy as np

    x = np.random.default_rng(0).standard_normal(8192)
    v = np.full(x.size, 10.0)
    out = e22.classic(x, v, 51_200.0, 16, 400.0, CFG)
    assert out == {"skipped": "ceiling below the kurtogram floor"}


def test_tat_is_inadmissible_with_empty_search_range():
    import math

    import numpy as np

    x = np.random.default_rng(1).standard_normal(8192)
    ok, share = e22.tat_admissible(x, 51_200.0, 16, 400.0, CFG)
    assert ok is False and math.isnan(share)


def test_run_record_safe_turns_exceptions_into_error_records():
    out = e22.run_record_safe({"cfg": {}, "dataset": "ottawa", "id": "X"})
    assert out["id"] == "X" and "error" in out


def _relabel_tokens() -> dict:
    """TOKENS of tools/relabel_map.py (old label -> English label), or skip if it is absent."""
    path = Path(__file__).resolve().parents[3] / "tools" / "relabel_map.py"
    if not path.exists():
        pytest.skip("tools/relabel_map.py is not part of this repository")
    spec = importlib.util.spec_from_file_location("relabel_map", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.TOKENS


def test_summary_keeps_the_row_order_of_the_versioned_files():
    """summary.csv is sorted by the labels the runs wrote, before they were renamed to English."""
    arms = ("uniform", "tat_rate", "bandpass_lu", "random_cs", "uniform_ceiling", "despike")
    rows = [
        {"dataset": "ottawa", "group": "outer_race", "budget": 4, "arm": a, "hit": 1} for a in arms
    ]
    order = [r["arm"] for r in e22.summarize(rows)]
    assert order == [
        "random_cs",
        "despike",
        "bandpass_lu",
        "tat_rate",
        "uniform",
        "uniform_ceiling",
    ]
    tokens = _relabel_tokens()
    assert all(tokens[old] == new for new, old in e22.LEGACY_LABELS.items())
