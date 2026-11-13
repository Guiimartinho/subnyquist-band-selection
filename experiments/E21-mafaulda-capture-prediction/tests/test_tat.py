"""The TAT measures what it claims: a high folded share when the chosen band comes from the fold."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "e21_run", Path(__file__).resolve().parents[1] / "run.py"
)
e21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e21)

FS = 51_200.0
CFG = {"kurtogram": {"max_level": 4, "f_min_hz": 500.0}, "tat": {"folded_share_threshold": 0.5}}


def bursts(fc_hz: float, n: int, rate_hz: float, rng) -> np.ndarray:
    """Impulsive bursts on a carrier fc, repeated at rate_hz: impulsive on purpose."""
    t = np.arange(n) / FS
    env = np.zeros(n)
    for t0 in np.arange(0.01, n / FS, 1.0 / rate_hz):
        i = int(t0 * FS)
        env[i : i + 200] += np.exp(-np.arange(min(200, n - i)) / 40.0)
    return env * np.sin(2 * np.pi * fc_hz * t) + 0.05 * rng.standard_normal(n)


def test_folded_share_high_when_impulsive_content_folds_into_search():
    rng = np.random.default_rng(1)
    n = 1 << 16
    # bursts at 20 kHz: at 4x (fs_eff = 12.8 kHz) they fold to 20000 - 12800 = 7.2 kHz
    x = bursts(20_000.0, n, 37.0, rng) + 0.2 * np.sin(2 * np.pi * 3_000.0 * np.arange(n) / FS)
    cell = e21.tat(x, FS, 4, e21.search_ceiling(None, FS / 4), CFG)
    assert cell["tat_folded_share"] > 0.9
    assert cell["tat_inadmissible"] == 1


def test_folded_share_low_when_impulsive_content_is_native():
    rng = np.random.default_rng(2)
    n = 1 << 16
    x = bursts(4_000.0, n, 37.0, rng)  # native, below the effective Nyquist at 4x (6.4 kHz)
    cell = e21.tat(x, FS, 4, e21.search_ceiling(None, FS / 4), CFG)
    assert cell["tat_folded_share"] < 0.1
    assert cell["tat_inadmissible"] == 0


def test_aaf_ideal_is_identity_at_full_rate():
    x = np.random.default_rng(3).standard_normal(1024)
    assert e21.aaf_ideal(x, FS, 1) is x
