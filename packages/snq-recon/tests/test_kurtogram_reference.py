"""Kurtogram against the source: Antoni, MSSP 21 (2007) 108-124.

Checked against the PDF (`literature/pdf/antoni2007kurtogram.pdf`):

- Eq. (5)-(6): at level ``k`` band ``i`` has center ``(i + 1/2)·2^(-k-1)`` and width
  ``2^(-k-1)``, in frequency normalized by ``fs``. Fig. 5 adds the
  1/3-binary levels with widths 1/6, 1/12, 1/24.
- Eq. (7): ``K = <|c|^4> / <|c|^2>^2 - 2`` over the complex envelope.
- Section 1.1, property (ii): the SK of a stationary Gaussian process is zero.
- Note 3: the first coefficients, affected by the filters' transient, are left out of
  the kurtosis.

What this module locks in as a **documented deviation** (not Antoni): with ``f_min_hz`` or
``f_max_hz`` the partition is that of the search range, and not that of ``[0, fs/2]``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from snq_core import alias_frequency_hz, fold_band
from snq_recon import fast_kurtogram, spectral_kurtosis
from snq_sim import impulse_train


def _structural_background(n: int, fs_hz: float, seed: int) -> np.ndarray:
    """Stationary Gaussian noise with almost all of its energy below ~200 Hz.

    It is cut from the middle of a realization three times longer, so that the record is
    **not** periodic: the periodic extension has a jump at the seam, as in a real record.
    Synthesizing with ``irfft`` would give an exactly periodic signal and hide the defect.
    """
    rng = np.random.default_rng(seed)
    b, a = signal.butter(4, 200.0, fs=fs_hz)
    y = signal.lfilter(b, a, rng.normal(size=3 * n))[n : 2 * n]
    return y / np.std(y)


# ----------------------------------------------------------------------
# geometry: Eq. (5)-(6) and Fig. 5
# ----------------------------------------------------------------------
def test_pavimentacao_sem_faixa_de_busca_e_a_de_antoni():
    """Without ``f_min_hz``/``f_max_hz``, level ``L`` has ``2^L`` bands of width ``2^(-L-1)``.

    Level 0 (the whole band) is not evaluated: a documented deviation, not an error.
    """
    rng = np.random.default_rng(0)
    k = fast_kurtogram(rng.normal(size=4096), 1.0, max_level=4)
    levels = sorted({cell[0] for cell in k.grid})
    assert levels == pytest.approx(
        [1.0, np.log2(3.0), 2.0, 1.0 + np.log2(3.0), 3.0, 2.0 + np.log2(3.0), 4.0]
    )
    for level in levels:
        cells = sorted(cell for cell in k.grid if cell[0] == level)
        n_bands = int(round(2.0**level))
        assert len(cells) == n_bands
        width = 2.0 ** (-level - 1.0)
        for i, (_, lo, hi, _) in enumerate(cells):
            assert lo == pytest.approx(i * width)
            assert hi == pytest.approx((i + 1) * width)
            assert 0.5 * (lo + hi) == pytest.approx((i + 0.5) * width)  # Eq. (5)


def test_faixa_de_busca_reancora_a_particao_e_a_torna_independente_de_fs():
    """Deviation from Antoni, locked in here because E20 depends on it.

    The partition is that of ``[f_min, min(f_max, fs/2)]``. With E07's 25 kHz ceiling, factors
    1, 2 and 4 over 200 kSPS see **the same** candidate bands; only at 8x (Nyquist of
    12.5 kHz) does the partition change. In Antoni, anchored at ``[0, fs/2]``, it would change
    at every factor.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=8192)

    def cells(fs_hz: float) -> list[tuple[float, float, float]]:
        k = fast_kurtogram(x, fs_hz, max_level=3, f_min_hz=500.0, f_max_hz=min(25e3, fs_hz / 2))
        return sorted((c[0], c[1], c[2]) for c in k.grid)

    full, half, quarter, eighth = (cells(fs) for fs in (200e3, 100e3, 50e3, 25e3))
    assert full == half == quarter
    assert eighth != quarter
    level1 = [(lo, hi) for lev, lo, hi in full if lev == 1.0]
    assert level1 == [(500.0, 12750.0), (12750.0, 25000.0)]


# ----------------------------------------------------------------------
# estimator: Eq. (7) and property (ii)
# ----------------------------------------------------------------------
def test_curtose_espectral_de_tom_puro_e_menos_um():
    """Constant envelope: ``<|c|^4>/<|c|^2>^2 = 1``, hence ``K = -1`` by Eq. (7)."""
    z = np.exp(2j * np.pi * 0.1 * np.arange(4096))
    assert spectral_kurtosis(z) == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sk_de_gaussiano_estacionario_colorido_e_nula_em_toda_celula(seed):
    """Property (ii) of Antoni (2007): a stationary Gaussian process gives zero SK.

    The background has the spectrum of a real machine, almost all of its energy at low
    frequency. Regression: with circular filtering and no edge handling, the jump at the seam
    of the periodic extension became a single "transient", and the SK exceeded 100 in bands
    with no transient at all. Measured before the fix: up to ~4000 without the energy guard.
    """
    fs = 48_000.0
    x = _structural_background(int(fs), fs, seed) + 0.03 * np.random.default_rng(seed + 10).normal(
        size=int(fs)
    )
    k = fast_kurtogram(x, fs, max_level=4, min_energy_frac=0.0)
    worst = max(abs(cell[3]) for cell in k.grid)
    assert worst < 0.5, f"SK of Gaussian noise reached {worst:.2f}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kurtograma_acha_a_ressonancia_sob_fundo_estrutural(seed):
    """Impulsive burst at a known resonance (6 kHz), under a realistic background.

    Before the fix, on these seeds the winning band was [500, 4417] or [500, 3438] Hz,
    with K between 130 and 430: it was the record seam, not the fault.
    """
    fs = 48_000.0
    n = int(fs)
    rng = np.random.default_rng(seed)
    fault = impulse_train(1.0, fs, rate_hz=104.0, resonance_hz=6000.0, damping=0.02, seed=seed)
    x = _structural_background(n, fs, seed) + 0.02 * rng.normal(size=n)
    x = x + 0.01 * fault / np.std(fault)
    k = fast_kurtogram(x, fs, max_level=4, f_min_hz=500.0)
    assert k.f_lo_hz <= 6000.0 <= k.f_hi_hz, k.summary()
    assert 0.5 < k.kurtosis < 10.0, k.summary()


def test_transiente_colado_na_borda_nao_entra_na_curtose():
    """Cost of the fix, documented as a negative test.

    As in note 3 of Antoni, the samples closer than ``4·fs/Δf`` to the edges are left out of
    the kurtosis. A short transient lying entirely there is invisible; the same transient in
    the middle is not. The smallest guard of this search is ``4·48000/2000 = 96`` samples
    (2 kHz band); the transient decays in 5 samples. A transient with a tail longer than the
    guard would remain partially visible.
    """
    fs = 48_000.0
    n = 48_000
    rng = np.random.default_rng(5)
    ring = np.exp(-np.arange(60) / 5.0) * np.sin(2 * np.pi * 6000.0 * np.arange(60) / fs)

    def one_burst(at: int) -> float:
        x = 0.01 * rng.normal(size=n)
        x[at : at + ring.size] += ring
        return fast_kurtogram(x, fs, max_level=3, f_min_hz=4000.0, f_max_hz=8000.0).kurtosis

    assert one_burst(n // 2) > 20.0
    assert one_burst(0) < 1.0


# ----------------------------------------------------------------------
# the ADR 0006 mechanism, on a controlled synthetic signal
# ----------------------------------------------------------------------
FS_ORIG = 200_000.0
SEARCH = {"max_level": 4, "f_min_hz": 500.0, "f_max_hz": 25e3}


def _fault_and_interferer(kind: str, seed: int = 11) -> np.ndarray:
    """Fault at 5 kHz (inside the search) and, optionally, an interferer at 76-84 kHz.

    The interferer is band-limited by a zero-phase bandpass, so that at full
    rate **none** of it falls in the search range: only folding can bring it inside.
    ``kind``: ``"none"``, ``"impulsive"`` (bursts at 9 Hz) or ``"gaussian"`` (same power,
    stationary).
    """
    duration = 0.5
    n = int(FS_ORIG * duration)
    rng = np.random.default_rng(seed)
    fault = impulse_train(
        duration, FS_ORIG, rate_hz=104.0, resonance_hz=5000.0, damping=0.02, seed=1
    )
    x = fault / np.std(fault) + 0.3 * rng.normal(size=n)
    if kind == "none":
        return x
    sos = signal.butter(8, [76e3, 84e3], btype="bandpass", fs=FS_ORIG, output="sos")
    raw = (
        impulse_train(duration, FS_ORIG, rate_hz=9.0, resonance_hz=80e3, damping=0.01, seed=2)
        if kind == "impulsive"
        else rng.normal(size=n)
    )
    interferer = signal.sosfiltfilt(sos, raw)
    return x + 2.0 * interferer / np.std(interferer)


def _band(x: np.ndarray, factor: int) -> tuple[float, float]:
    return fast_kurtogram(x[::factor], FS_ORIG / factor, **SEARCH).band_hz  # plain decimation


def test_rajada_fora_da_busca_captura_o_kurtograma_onde_o_mapa_de_alias_preve():
    """The ADR 0006 mechanism, isolated: folding of **impulsive** content outside the search.

    At 1x the interferer does not change the choice. At 4x (``x[::4]``, 50 kSPS) it lands at
    ``|80 - 2·50| = 20`` kHz, in zone 3, inverted, and the chosen band becomes the one that
    contains the fold, [12.75, 25] kHz, the same as E07's valley.
    This proves that the code reproduces the mechanism. It does not prove that this is the
    mechanism at work on Ottawa: that is E20.
    """
    clean, hit = _fault_and_interferer("none"), _fault_and_interferer("impulsive")
    assert _band(hit, 1) == _band(clean, 1)

    fs_eff = FS_ORIG / 4
    alias = float(alias_frequency_hz(80e3, fs_eff))
    assert alias == pytest.approx(20e3)
    assert fold_band(76e3, 84e3, fs_eff).inverted  # odd zone: the spectrum arrives inverted

    lo, hi = _band(hit, 4)
    assert lo <= alias <= hi
    assert (lo, hi) != _band(clean, 4)
    assert (lo, hi) == (12750.0, 25000.0)


def test_interferente_potente_mas_gaussiano_dobra_sem_capturar():
    """Counter-test: the same fold, with the same power, of stationary noise does not capture.

    It is the distinction that ADR 0006 claims: what brings the selector down is folded
    content that is **impulsive**, not powerful.
    """
    gauss, clean = _fault_and_interferer("gaussian"), _fault_and_interferer("none")
    assert _band(gauss, 4) == _band(clean, 4)
