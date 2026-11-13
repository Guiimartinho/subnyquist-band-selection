"""Impulse-robust band selectors, on the same partition as this package's fast kurtogram.

- **Infogram** (Antoni, MSSP 74:73-94, 2016, sections 3 and 4, checked against the PDF on
  2026-11-12).
  - The energy flow of the band is ε(n) = |c(n)|², sampled at the critical rate.
  - Negentropy in time (eq. 10-11): ΔI_ε = <(ε²/<ε²>)·ln(ε²/<ε²>)>.
  - Negentropy in the squared envelope spectrum (eq. 12): the same expression over |E(α)|²,
    where E is the DFT of ε, without the α = 0 component, as section 4.2 recommends.
  - Average: ΔI_1/2 = (ΔI_ε + ΔI_E)/2 (eq. 13).
  - Per section 5.1.1, the SES infogram is insensitive to impulsive noise; the SE infogram and
    the kurtogram are not.
  - Under stationary Gaussian noise, the expected values are 1.1525 and 0.4228 (section 4.3).
    The tests check both.
- **Spectral Gini** (Hebda-Sobkowicz, Zimroz and Wyłomańska, Appl. Sci. 10:2657, 2020, eq. 3):
  Gini index of the band envelope |c(n)|, in place of the kurtosis.

Declared deviations.
- The filters are ideal FFT filters, on the partition of `fast_kurtogram`, and not the STFT of
  Antoni's section 4.2.
- For the ideal analytic signal of width Δf, the critical rate is Δf.
- The conditional-variance selector (CVB) is not here: the full definition is in another
  paper (`hebdasobkowicz2020condvar`), which has not been read.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snq_recon.envelope import _band_analytic_from_spectrum
from snq_recon.kurtogram import KurtogramResult, _edge_guard, _levels, spectral_kurtosis

STATISTICS = ("sk", "se_negentropy", "ses_negentropy", "avg_negentropy", "gini")


def _plogp(p: NDArray[np.float64]) -> NDArray[np.float64]:
    safe = np.where(p > 0.0, p, 1.0)
    return np.where(p > 0.0, p * np.log(safe), 0.0)


def band_statistic(
    z: NDArray[np.complex128], fs_hz: float, bandwidth_hz: float, statistic: str
) -> float:
    """Statistic of a band from its analytic signal ``z`` (with the edges already removed)."""
    if statistic == "sk":
        return float(spectral_kurtosis(z))
    if statistic not in STATISTICS:
        raise ValueError(f"unknown statistic: {statistic}")
    c = z[:: max(1, int(round(fs_hz / bandwidth_hz)))]  # critical rate of the ideal filter
    if c.size < 16:
        return -np.inf
    if statistic == "gini":
        a = np.sort(np.abs(c))
        s = float(a.sum())
        if s <= 0.0:
            return -np.inf
        n = a.size
        i = np.arange(1, n + 1)
        return float(1.0 - 2.0 * np.sum((a / s) * (n - i + 0.5) / n))
    e = np.abs(c) ** 2
    m2 = float(np.mean(e**2))
    if m2 <= 0.0:
        return -np.inf
    d_se = float(np.mean(_plogp(e**2 / m2)))
    spec = np.abs(np.fft.fft(e)[1:]) ** 2  # without alpha = 0 (section 4.2)
    m_spec = float(np.mean(spec))
    d_ses = float(np.mean(_plogp(spec / m_spec))) if m_spec > 0.0 else -np.inf
    if statistic == "se_negentropy":
        return d_se
    if statistic == "ses_negentropy":
        return d_ses
    return 0.5 * (d_se + d_ses)


def fast_band_selector(
    x: ArrayLike,
    fs_hz: float,
    statistic: str = "ses_negentropy",
    max_level: int = 6,
    f_min_hz: float = 0.0,
    f_max_hz: float | None = None,
    min_energy_frac: float = 1e-4,
) -> KurtogramResult:
    """The same sweep as `fast_kurtogram`, with a different statistic per band.

    With ``statistic="sk"`` it reproduces `fast_kurtogram`. The ``kurtosis`` field of the result
    holds the value of the chosen statistic.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    nyq = fs_hz / 2.0
    hi = min(f_max_hz, nyq) if f_max_hz is not None else nyq
    if hi <= f_min_hz:
        raise ValueError("empty search range")
    grid: list[tuple[float, float, float, float]] = []
    best = (f_min_hz, hi, -np.inf, 0.0)
    total_energy = float(np.sum(x**2))
    mirrored = np.concatenate([x, x[::-1]])
    spec = np.fft.fft(mirrored)
    freqs = np.fft.fftfreq(mirrored.size, d=1.0 / fs_hz)
    for level in _levels(max_level):
        n_bands = int(round(2.0**level))
        edges = np.linspace(f_min_hz, hi, n_bands + 1)
        for lo, up in zip(edges[:-1], edges[1:], strict=True):
            if up - lo <= 0.0:
                continue
            z = _band_analytic_from_spectrum(spec, freqs, float(lo), float(up))[:n]
            if (
                total_energy > 0.0
                and float(np.sum(np.abs(z) ** 2)) < min_energy_frac * total_energy
            ):
                continue
            guard = _edge_guard(n, fs_hz, float(up - lo))
            k = band_statistic(z[guard : n - guard], fs_hz, float(up - lo), statistic)
            grid.append((level, float(lo), float(up), k))
            if k > best[2]:
                best = (float(lo), float(up), k, level)
    return KurtogramResult(
        f_lo_hz=best[0], f_hi_hz=best[1], kurtosis=best[2], level=best[3], grid=grid
    )
