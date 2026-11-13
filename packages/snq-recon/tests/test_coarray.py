"""Difference coarray: cyclic correlation of coprime samples (snq_recon.coarray)."""

import numpy as np
import pytest
from snq_recon.coarray import (
    band_ses,
    cyclic_autocorrelation,
    sample_mask,
)

FS = 64000.0


def am_band_noise(n, fs, f_lo, f_hi, alpha, depth, rng):
    """Band noise [f_lo, f_hi] amplitude-modulated at ``alpha`` Hz (cyclostationary)."""
    xf = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / fs)
    xf[(f < f_lo) | (f > f_hi)] = 0.0
    carrier = np.fft.irfft(xf, n=n)
    t = np.arange(n) / fs
    return carrier * (1.0 + depth * np.cos(2 * np.pi * alpha * t))


def test_coprime_mask_covers_every_lag_and_uniform_only_multiples():
    n = 5000
    cop = sample_mask(n, "coprime", 8, 9)
    uni = sample_mask(n, "uniform", 4)
    for tau in range(1, 40):
        assert (cop[tau:] & cop[: n - tau]).any()
        assert (uni[tau:] & uni[: n - tau]).any() == (tau % 4 == 0)
    assert abs(cop.mean() - (1 / 8 + 1 / 9 - 1 / 72)) < 1e-3
    with pytest.raises(ValueError):
        sample_mask(n, "coprime", 4, 6)


def test_stationary_correlation_from_coprime_matches_full_rate():
    rng = np.random.default_rng(1)
    x = am_band_noise(2**17, FS, 2000, 6000, 0.0, 0.0, rng)
    full = cyclic_autocorrelation(x, sample_mask(x.size, "full"), 24, FS, 1.0)
    cop = cyclic_autocorrelation(x, sample_mask(x.size, "coprime", 8, 9), 24, FS, 1.0)
    r_full = np.real(full.r[:, 0]) / np.real(full.r[0, 0])
    r_cop = np.real(cop.r[:, 0]) / np.real(cop.r[0, 0])
    assert np.max(np.abs(r_full - r_cop)) < 0.08


def test_coprime_ses_sees_a_band_above_the_decimated_nyquist_without_folding():
    """Modulated band at 20-24 kHz and its image under decimation by 4 (fs_eff = 16 kHz).

    With uniform decimation by 4, 20-24 kHz folds onto 4-8 kHz: the envelope spectrum of the
    4-8 kHz band shows the modulation, which is not there. With coprime samples (8, 9), it is the
    20-24 kHz band that shows the modulation; the 4-8 kHz and 0.5-4 kHz bands do not.
    """
    rng = np.random.default_rng(2)
    n, alpha = 2**18, 75.0
    x = am_band_noise(n, FS, 20000, 24000, alpha, 0.9, rng)
    x += 0.5 * am_band_noise(n, FS, 600, 3800, 0.0, 0.0, rng)  # low band without modulation
    lag = 64
    cop = cyclic_autocorrelation(x, sample_mask(n, "coprime", 8, 9), lag, FS, 200.0)
    k = int(np.argmin(np.abs(cop.alphas_hz - alpha)))

    def cyc(cc, lo, hi):  # cyclic content of the band at alpha
        return band_ses(cc, lo, hi)[k]

    hi_c, mid_c, lo_c = cyc(cop, 20000, 24000), cyc(cop, 4000, 8000), cyc(cop, 500, 4000)
    assert hi_c > 10 * mid_c and hi_c > 10 * lo_c  # coprime: the modulation is at 20-24 kHz
    xd = x[::4]  # uniform decimation: the high band folds onto 4-8 kHz
    uni = cyclic_autocorrelation(xd, sample_mask(xd.size, "full"), lag // 4, FS / 4, 200.0)
    assert cyc(uni, 4000, 8000) > 10 * cyc(uni, 500, 4000)  # uniform: it shows up at 4-8 kHz
