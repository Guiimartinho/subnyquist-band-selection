"""As intervencoes do E20 removem o que dizem remover, e so isso."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "e20_run", Path(__file__).resolve().parents[1] / "run.py"
)
e20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e20)

FS = 200_000.0
N = 200_000  # 1 s: resolucao de 1 Hz, tons no centro do bin


def tone(f_hz: float) -> np.ndarray:
    t = np.arange(N) / FS
    return np.sin(2 * np.pi * f_hz * t)


def amp(x: np.ndarray, f_hz: float) -> float:
    xf = np.abs(np.fft.rfft(x)) * 2 / x.size
    return float(xf[int(round(f_hz))])


def test_none_returns_input_unchanged():
    x = tone(80_000.0)
    assert e20.apply_intervention(x, FS, {"kind": "none"}, 4) is x


def test_stop_removes_only_its_zone():
    x = tone(80_000.0) + tone(3_000.0) + tone(30_000.0)
    y = e20.apply_intervention(x, FS, {"kind": "stop", "bands_hz": [[75_000.0, 87_250.0]]}, 4)
    assert amp(y, 80_000.0) < 1e-9
    assert abs(amp(y, 3_000.0) - 1.0) < 1e-9
    assert abs(amp(y, 30_000.0) - 1.0) < 1e-9


def test_aaf_ideal_kills_everything_above_effective_nyquist():
    x = tone(80_000.0) + tone(30_000.0) + tone(10_000.0)
    y = e20.apply_intervention(x, FS, {"kind": "aaf_ideal"}, 4)  # Nyquist efetivo = 25 kHz
    assert amp(y, 80_000.0) < 1e-9
    assert amp(y, 30_000.0) < 1e-9
    assert abs(amp(y, 10_000.0) - 1.0) < 1e-9


def test_zone_z3_folds_onto_search_band_at_4x_and_2x():
    """A aritmetica que motivou o E20: 80 kHz aparece em 20 kHz tanto em 4x quanto em 2x."""
    x = tone(80_000.0)
    for factor in (2, 4):
        d = x[::factor]
        xf = np.abs(np.fft.rfft(d))
        f = np.fft.rfftfreq(d.size, d=factor / FS)
        assert abs(f[int(np.argmax(xf))] - 20_000.0) < 1.0


def test_mcnemar_exact_p_matches_known_values():
    assert e20.mcnemar_exact_p(0, 0) == 1.0
    assert abs(e20.mcnemar_exact_p(19, 1) - 2 * 21 / 2**20) < 1e-12
    assert e20.mcnemar_exact_p(5, 5) == 1.0


def test_despike_and_median3_remove_an_isolated_spike_without_a_band_mask():
    import numpy as np

    fs = 200_000.0
    t = np.arange(20_000) / fs
    x = np.sin(2 * np.pi * 3000.0 * t)
    y = x.copy()
    y[5_000] -= 5.0  # pico isolado de uma amostra
    for kind in ("despike", "median3"):
        z = e20.apply_intervention(y, fs, {"kind": kind}, factor=4)
        # despike interpola; a mediana de 3 devolve um dos vizinhos (nao interpola)
        lo, hi = sorted((x[4_999], x[5_001]))
        assert lo - 1e-9 <= z[5_000] <= hi + 1e-9, kind
        assert np.max(np.abs(np.delete(z - x, [4_999, 5_000, 5_001]))) < 0.05, kind
