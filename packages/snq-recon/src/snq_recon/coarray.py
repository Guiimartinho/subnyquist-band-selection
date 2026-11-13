"""Second-order cyclic statistics of non-uniform samples (difference coarray).

Bearing diagnosis does not need the waveform. It is enough to have the cyclic correlation at
the cyclic frequencies of the fault and, through it, the squared envelope spectrum of a band.

    R(τ, α) = (1/c_τ) Σ_n x[n] x[n−τ] m[n] m[n−τ] e^{−j2π α n/f_s}

- ``m`` is the mask of the sampled instants on the full-rate grid.
- ``c_τ`` is the number of pairs (n, n−τ) present in the mask.
- **Uniform decimation by M:** only lags that are multiples of M exist, and the carrier axis
  folds.
- **Coprime samplers** (steps M and N, with gcd(M, N) = 1, running continuously):
  - every integer lag appears, by Bézout's theorem, with about T/(MN) pairs;
  - the carrier axis stays unfolded up to f_s/2, at the cost of more variance per lag.

The squared envelope spectrum of a band b follows from the cyclic correlation:

    SES_b(α) ≈ Σ_τ r_b(τ) R(τ, α)

- ``r_b`` is the autocorrelation of an ideal band.
- The approximation holds for α much smaller than the bandwidth, and for bands wider than
  f_s/L, where L is the largest estimated lag.
- For real x, R(−τ, α) = e^{j2π α τ/f_s} R(τ, α).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_mask(n: int, pattern: str, m: int = 1, n2: int | None = None) -> np.ndarray:
    """Boolean mask of the sampled instants on the full-rate grid.

    ``full``: all; ``uniform``: step ``m``; ``coprime``: union of steps ``m`` and ``n2``.
    """
    mask = np.zeros(n, dtype=bool)
    if pattern == "full":
        mask[:] = True
    elif pattern == "uniform":
        mask[::m] = True
    elif pattern == "coprime":
        if n2 is None or np.gcd(m, n2) != 1:
            raise ValueError("coprime requires two coprime steps")
        mask[::m] = True
        mask[::n2] = True
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return mask


@dataclass(frozen=True)
class CyclicCorrelation:
    """R(τ, α) for τ = 0..L and α on the FFT grid of the record, up to ``alpha_max_hz``."""

    r: np.ndarray  # (L+1, K) complex
    lags: np.ndarray  # (L+1,)
    alphas_hz: np.ndarray  # (K,)
    counts: np.ndarray  # (L+1,) pairs per lag
    fs_hz: float


def cyclic_autocorrelation(
    x: np.ndarray, mask: np.ndarray, max_lag: int, fs_hz: float, alpha_max_hz: float
) -> CyclicCorrelation:
    """Estimates R(τ, α) using only the pairs of sampled instants (see the module docstring).

    A lag with no pair at all gets R = NaN.
    """
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if x.shape != mask.shape:
        raise ValueError("x and mask must have the same size")
    n = x.size
    xm = np.where(mask, x, 0.0)
    alphas = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    keep = alphas <= alpha_max_hz
    out = np.full((max_lag + 1, int(keep.sum())), np.nan, dtype=complex)
    counts = np.zeros(max_lag + 1, dtype=int)
    for tau in range(max_lag + 1):
        both = mask[tau:] & mask[: n - tau]
        c = int(both.sum())
        counts[tau] = c
        if c == 0:
            continue
        p = np.zeros(n)
        p[tau:] = xm[tau:] * xm[: n - tau]  # zero where either of the two was not sampled
        out[tau] = np.fft.rfft(p)[keep] / c
    return CyclicCorrelation(out, np.arange(max_lag + 1), alphas[keep], counts, fs_hz)


def band_lag_weights(lags: np.ndarray, fs_hz: float, f_lo_hz: float, f_hi_hz: float) -> np.ndarray:
    """Autocorrelation of a real ideal band [f_lo, f_hi] at the lags, equal to 1 at τ = 0."""
    tau = np.asarray(lags, dtype=float) / fs_hz
    bw = f_hi_hz - f_lo_hz
    fc = 0.5 * (f_lo_hz + f_hi_hz)
    return np.sinc(bw * tau) * np.cos(2.0 * np.pi * fc * tau)


def band_ses(cc: CyclicCorrelation, f_lo_hz: float, f_hi_hz: float) -> np.ndarray:
    """Squared envelope spectrum of the band, in magnitude, from the cyclic correlation (α ≥ 0).

    Uses the lags from −L to L; the negative ones come from R(−τ, α) = e^{j2π α τ/f_s} R(τ, α).
    Lags without pairs enter with zero weight.
    """
    w = band_lag_weights(cc.lags, cc.fs_hz, f_lo_hz, f_hi_hz)
    r = np.nan_to_num(cc.r, nan=0.0)
    phase = np.exp(2j * np.pi * np.outer(cc.lags, cc.alphas_hz) / cc.fs_hz)
    # τ = 0 once; τ > 0 twice (the lag itself and its negative mirror)
    total = w[0] * r[0] + np.sum(w[1:, None] * (r[1:] + phase[1:] * r[1:]), axis=0)
    return np.abs(total)
