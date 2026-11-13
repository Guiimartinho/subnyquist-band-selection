"""Envelope and order spectrum against the identities they have to satisfy.

- Analytic signal: ``X_a(f) = 2·X(f)`` for ``f > 0``, ``X_a(0) = X(0)``, zero for ``f < 0``.
  The whole band has to reproduce ``scipy.signal.hilbert``.
- AM envelope: ``|(1 + m·cos 2π f_m t)·e^{j2π f_c t}| = 1 + m·cos 2π f_m t`` for ``m < 1``.
- Angular resampling is decimation in angle: without a low-pass, order ``o`` above
  ``spr/2`` reappears at ``|o - spr·round(o/spr)|``. The test without the filter checks the
  position of the fold; the test with the filter checks that it is gone.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from snq_recon import (
    angular_resample,
    bandpass_analytic,
    envelope_order_spectrum,
    envelope_spectrum,
    order_spectrum,
    squared_envelope_spectrum,
)


# ----------------------------------------------------------------------
# band analytic signal
# ----------------------------------------------------------------------
def _with_dc_and_no_nyquist(n: int, seed: int, dc: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.normal(size=n))
    spec[-1] = 0.0  # no component at fs/2, where the analytic signal is ambiguous
    return np.fft.irfft(spec, n=n) + dc


def test_banda_inteira_reproduz_hilbert_inclusive_o_dc():
    """Regression: the DC bin used to be multiplied by 2, and ``Re(z)`` gained the mean again."""
    x = _with_dc_and_no_nyquist(1024, seed=0, dc=3.0)
    z = bandpass_analytic(x, 1.0, 0.0, 0.5)
    assert np.allclose(z, signal.hilbert(x), atol=1e-12)
    assert np.mean(z.real) == pytest.approx(np.mean(x), abs=1e-12)  # before: 2x the mean


def test_banda_que_exclui_o_dc_tem_media_nula_e_parte_real_filtrada():
    """Outside the whole-band case, ``Re(z)`` is the ideal filtering of the band."""
    x = _with_dc_and_no_nyquist(1024, seed=1, dc=3.0)
    z = bandpass_analytic(x, 1.0, 0.1, 0.3)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size)
    spec[(f < 0.1) | (f >= 0.3)] = 0.0
    assert abs(np.mean(z)) < 1e-12
    assert np.allclose(z.real, np.fft.irfft(spec, n=x.size), atol=1e-12)


# ----------------------------------------------------------------------
# envelope
# ----------------------------------------------------------------------
def test_envelope_de_am_recupera_a_moduladora_com_escala_de_meia_amplitude():
    """Line at ``f_m``; the magnitude is ``m/2`` because the coherent gain of the Hann window
    is not compensated. The scale is only good for ratios and argmax, which is how the package
    uses it."""
    fs, n = 10_000.0, 10_000
    t = np.arange(n) / fs
    m, fm, fc = 0.5, 37.0, 2500.0
    x = (1.0 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2 * np.pi * fc * t)
    spec = envelope_spectrum(x, fs, (2000.0, 3000.0))
    f_pk, mag = spec.peak_near(fm, 1.0)
    assert f_pk == pytest.approx(fm)
    assert mag == pytest.approx(m / 2.0, rel=1e-3)
    # the magnitude envelope of pure AM has no second harmonic; the squared one has m²/4
    assert spec.peak_near(2 * fm, 1.0)[1] < 1e-3
    sq = squared_envelope_spectrum(x, fs, (2000.0, 3000.0))
    assert sq.peak_near(fm, 1.0)[1] == pytest.approx(m, rel=1e-3)
    assert sq.peak_near(2 * fm, 1.0)[1] == pytest.approx(m * m / 4.0, rel=1e-3)


# ----------------------------------------------------------------------
# angular resampling: folding in the angle domain
# ----------------------------------------------------------------------
FS = 20_000.0
N = 200_000


def _locked(order: float) -> tuple[np.ndarray, np.ndarray]:
    """Component locked to an order, with the shaft speed rising 30% over the record."""
    t = np.arange(N) / FS
    rot = 20.0 * (1.0 + 0.3 * t / t[-1])
    revs = np.concatenate([[0.0], np.cumsum(0.5 * (rot[:-1] + rot[1:]) / FS)])
    return np.sin(2 * np.pi * order * revs), rot


def _peak(ang: np.ndarray, spr: float, near: float) -> float:
    orders, mag = order_spectrum(ang, spr)
    sel = np.abs(orders - near) <= 0.2
    return float(mag[sel].max())


@pytest.mark.parametrize("order", [3.5, 25.0, 31.0])
def test_ordem_dentro_da_banda_angular_preserva_amplitude(order):
    x, rot = _locked(order)
    ang, spr = angular_resample(x, FS, rot, samples_per_rev=64)
    assert _peak(ang, spr, order) == pytest.approx(0.5, rel=0.01)


@pytest.mark.parametrize("order", [40.0, 70.0, 130.5])
def test_sem_filtro_a_ordem_dobra_onde_o_mapa_de_alias_preve(order):
    """Outside the validity region of the resampling: the folding identity has to hold."""
    x, rot = _locked(order)
    ang, spr = angular_resample(x, FS, rot, samples_per_rev=64, anti_alias=False)
    folded = abs(order - 64.0 * round(order / 64.0))
    orders, mag = order_spectrum(ang, spr)
    pico = float(orders[1:][np.argmax(mag[1:])])
    assert pico == pytest.approx(folded, abs=0.01)
    # linear interpolation attenuates high orders (sinc^2), hence the loose floor
    assert _peak(ang, spr, folded) > 0.4


@pytest.mark.parametrize("order", [40.0, 70.0, 130.5])
def test_ordem_acima_do_nyquist_angular_e_removida_e_nao_dobrada(order):
    """Regression: without a low-pass, order 70 showed up at order 6 with full amplitude."""
    x, rot = _locked(order)
    ang, spr = angular_resample(x, FS, rot, samples_per_rev=64)
    folded = abs(order - 64.0 * round(order / 64.0))
    assert _peak(ang, spr, folded) < 0.005


def test_espectro_de_ordem_do_envelope_nao_dobra_modulacao_de_ordem_alta():
    """Envelope with modulation at order 70 and at 5.43: only 5.43 may show up.

    With folding, order 70 landed at 6, 0.57 order away from the fault, and won the argmax.
    """
    t = np.arange(N) / FS
    rot = 20.0 * (1.0 + 0.3 * t / t[-1])
    revs = np.concatenate([[0.0], np.cumsum(0.5 * (rot[:-1] + rot[1:]) / FS)])
    env = 1.0 + 0.2 * np.cos(2 * np.pi * 5.43 * revs) + 0.6 * np.cos(2 * np.pi * 70.0 * revs)
    x = env * np.cos(2 * np.pi * 6000.0 * t)
    spec = envelope_order_spectrum(x, FS, rot, (4000.0, 8000.0), samples_per_rev=64)
    sel = spec.freqs_hz >= 2.0
    pico = float(spec.freqs_hz[sel][np.argmax(spec.magnitude[sel])])
    assert pico == pytest.approx(5.43, abs=0.05)
    assert spec.peak_near(6.0, 0.1)[1] < 0.01 * spec.peak_near(5.43, 0.1)[1]
    assert spec.folds_above_nyquist is False


def test_espectro_de_ordem_sem_filtro_declara_que_dobra():
    x, rot = _locked(5.0)
    spec = envelope_order_spectrum(x, FS, rot, None, samples_per_rev=64, anti_alias=False)
    assert spec.folds_above_nyquist is True


def test_reamostragem_mantem_a_grade_angular():
    """Same grid and same length with and without the filter: the fix does not change the API."""
    x, rot = _locked(5.0)
    a, spr_a = angular_resample(x, FS, rot, samples_per_rev=64)
    b, spr_b = angular_resample(x, FS, rot, samples_per_rev=64, anti_alias=False)
    assert spr_a == spr_b == 64.0
    assert a.shape == b.shape
