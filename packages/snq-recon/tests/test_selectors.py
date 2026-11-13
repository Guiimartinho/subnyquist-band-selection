"""Robust selectors: reference values from the paper and Antoni's impulsive-noise example."""

from __future__ import annotations

import numpy as np
import pytest
from snq_recon.envelope import _band_analytic_from_spectrum
from snq_recon.kurtogram import fast_kurtogram
from snq_recon.selectors import band_statistic, fast_band_selector


def _band(x, fs, lo, up):
    spec = np.fft.fft(x)
    freqs = np.fft.fftfreq(x.size, d=1.0 / fs)
    return _band_analytic_from_spectrum(spec, freqs, lo, up)


def test_negentropies_of_stationary_gaussian_noise_match_antoni_2016():
    """Section 4.3: 1.1525 (SE) and 0.4228 (SES) under stationary Gaussian noise."""
    rng = np.random.default_rng(0)
    fs, n = 16_000.0, 1 << 20
    x = rng.normal(size=n)
    z = _band(x, fs, 3000.0, 4000.0)  # L = n * 1000 / 16000 = 65536 critical samples
    assert band_statistic(z, fs, 1000.0, "se_negentropy") == pytest.approx(1.1525, abs=0.08)
    assert band_statistic(z, fs, 1000.0, "ses_negentropy") == pytest.approx(0.4228, abs=0.02)


def test_gini_limits():
    fs = 1000.0
    tone = np.exp(2j * np.pi * 100.0 * np.arange(4096) / fs)  # constant envelope
    assert band_statistic(tone, fs, fs, "gini") == pytest.approx(0.0, abs=1e-9)
    spike = np.zeros(4096, dtype=complex)
    spike[100] = 1.0
    assert band_statistic(spike, fs, fs, "gini") == pytest.approx(1.0, abs=1e-3)


def _ringing(n, fs, f0_hz, times_s, amps, decay_s):
    t = np.arange(n) / fs
    y = np.zeros(n)
    for t0, a in zip(times_s, amps, strict=True):
        m = t >= t0
        y[m] += a * np.exp(-(t[m] - t0) / decay_s) * np.sin(2 * np.pi * f0_hz * (t[m] - t0))
    return y


def test_ses_infogram_ignores_isolated_impulses_that_capture_the_kurtogram():
    """Section 5.1.1: the kurtogram goes to the isolated impulses; the SES infogram, to the
    repetitive ones."""
    rng = np.random.default_rng(1)
    fs, n = 20_000.0, 40_000
    rep = _ringing(n, fs, 2000.0, np.arange(0.01, 2.0, 1 / 37.0), np.ones(74), 0.002)
    imp = _ringing(n, fs, 7000.0, [0.3, 1.1, 1.7], [25.0, 25.0, 25.0], 0.002)
    x = rep + imp + 0.3 * rng.normal(size=n)
    k = fast_kurtogram(x, fs, max_level=4, f_min_hz=500.0)
    s = fast_band_selector(x, fs, "ses_negentropy", max_level=4, f_min_hz=500.0)
    assert k.f_lo_hz <= 7000.0 < k.f_hi_hz
    assert s.f_lo_hz <= 2000.0 < s.f_hi_hz


def test_sk_statistic_reproduces_fast_kurtogram():
    rng = np.random.default_rng(2)
    x = rng.normal(size=8192) + _ringing(8192, 8000.0, 1500.0, [0.2, 0.5], [5.0, 5.0], 0.003)
    a = fast_kurtogram(x, 8000.0, max_level=3, f_min_hz=200.0)
    b = fast_band_selector(x, 8000.0, "sk", max_level=3, f_min_hz=200.0)
    assert (a.f_lo_hz, a.f_hi_hz) == (b.f_lo_hz, b.f_hi_hz)
    assert a.kurtosis == pytest.approx(b.kurtosis)
