"""Mathematics of structured sampling.

The project's theoretical floor: **Landau (1967)** — the minimum sampling rate is the
measure of the occupied spectral support, not twice the highest frequency present.
The relation ``f_max = f_s/2`` is the pessimistic corollary, valid only for an
arbitrary signal that occupies the whole baseband.

References implemented here:

- H. J. Landau, *Necessary density conditions for sampling and interpolation of
  certain entire functions*, Acta Mathematica 117:37-52, 1967.
- R. G. Vaughan, N. L. Scott, D. R. White, *The theory of bandpass sampling*,
  IEEE Trans. Signal Processing 39(9):1973-1984, 1991.

Convention: all frequencies in Hz, all times in seconds, and the parameter name
carries the unit (``fs_hz``, ``sigma_t_s``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Band = tuple[float, float]


# --------------------------------------------------------------------------
# Minimum rate
# --------------------------------------------------------------------------
def landau_rate_hz(bands_hz: list[Band]) -> float:
    """Landau rate for a **real** signal occupying ``bands_hz``.

    For a real signal each positive band has a corresponding negative image, so the
    total measure of the support is twice the sum of the widths.

    >>> landau_rate_hz([(140e3, 160e3)])
    40000.0
    """
    if any(hi <= lo for lo, hi in bands_hz):
        raise ValueError("each band must satisfy f_lo < f_hi")
    return 2.0 * float(sum(hi - lo for lo, hi in bands_hz))


def nyquist_rate_hz(bands_hz: list[Band]) -> float:
    """Classical Nyquist rate: ``2 * f_max``. It is here for comparison."""
    return 2.0 * float(max(hi for _, hi in bands_hz))


# --------------------------------------------------------------------------
# Nyquist zones and spectral folding
# --------------------------------------------------------------------------
def nyquist_zone(f_hz: ArrayLike, fs_hz: float) -> NDArray[np.int_]:
    """Index of the Nyquist zone that contains ``f_hz``. Zone ``k`` = ``[k·fs/2, (k+1)·fs/2]``.

    Zone 0 is the baseband. **An odd zone inverts the spectrum** when it folds.
    """
    return np.floor(2.0 * np.asarray(f_hz, dtype=float) / fs_hz).astype(int)


def alias_frequency_hz(f_hz: ArrayLike, fs_hz: float) -> NDArray[np.float64]:
    """Apparent frequency of ``f_hz`` after uniform sampling at ``fs_hz`` (real signal).

    >>> float(alias_frequency_hz(150e3, 40e3))
    10000.0
    """
    r = np.mod(np.asarray(f_hz, dtype=float), fs_hz)
    out: NDArray[np.float64] = np.where(r > fs_hz / 2.0, fs_hz - r, r)
    return out


def alias_order(f_hz: ArrayLike, fs_hz: float) -> NDArray[np.int_]:
    """Fold order ``k``: how many integer multiples of ``fs`` were removed.

    It is the unknown the project needs to identify. It is **not** observable in a
    single uniform acquisition — only the aliased frequency is. It can be recovered
    from the relative phase between shifted observations (see
    `multicoset_resolvable_orders`) or from multiple coprime rates.
    """
    return np.rint(np.asarray(f_hz, dtype=float) / fs_hz).astype(int)


@dataclass(frozen=True)
class FoldedBand:
    """Result of folding a band down to baseband."""

    lo_hz: float
    hi_hz: float
    zone: int
    inverted: bool
    self_overlap: bool
    """``True`` if the band crosses a zone boundary and folds onto itself —
    in which case the information is irrecoverably destroyed."""


def fold_band(f_lo_hz: float, f_hi_hz: float, fs_hz: float) -> FoldedBand:
    """Folds ``[f_lo, f_hi]`` down to baseband under sampling at ``fs_hz``.

    The project's canonical case — a resonant acoustic-emission transducer:

    >>> fb = fold_band(140e3, 160e3, 40e3)
    >>> fb.lo_hz, fb.hi_hz, fb.inverted, fb.self_overlap
    (0.0, 20000.0, True, False)
    """
    if f_hi_hz <= f_lo_hz:
        raise ValueError("f_lo_hz must be less than f_hi_hz")

    k_lo = int(np.floor(2.0 * f_lo_hz / fs_hz))
    # The upper edge touches the boundary when it is an exact multiple of fs/2;
    # in that case it belongs to the zone below.
    k_hi_raw = 2.0 * f_hi_hz / fs_hz
    k_hi = int(np.ceil(k_hi_raw)) - 1 if float(k_hi_raw).is_integer() else int(np.floor(k_hi_raw))

    overlap = k_lo != k_hi
    inverted = (k_lo % 2) == 1

    a = float(alias_frequency_hz(f_lo_hz, fs_hz))
    b = float(alias_frequency_hz(f_hi_hz, fs_hz))
    lo, hi = (b, a) if inverted else (a, b)
    return FoldedBand(lo_hz=lo, hi_hz=hi, zone=k_lo, inverted=inverted, self_overlap=overlap)


# --------------------------------------------------------------------------
# Bandpass sampling (Vaughan, Scott & White, 1991)
# --------------------------------------------------------------------------
def valid_bandpass_rates_hz(
    f_lo_hz: float, f_hi_hz: float, fs_max_hz: float | None = None
) -> list[tuple[float, float, int]]:
    """Ranges of ``fs`` that sample ``[f_lo, f_hi]`` without overlap.

    Classical condition, for ``n = 1 .. floor(f_hi / B)``::

        2·f_hi/n  <=  fs  <=  2·f_lo/(n-1)

    ``n = 1`` is the classical case (``fs >= 2·f_hi``, no upper bound).
    The largest ``n`` gives the lowest possible rate.

    Returns ``[(fs_min, fs_max, n)]`` in decreasing order of ``fs_min``.
    ``fs_max`` is ``inf`` for ``n = 1``. If ``fs_max_hz`` is given, ranges lying
    entirely above it are discarded — that is how one filters by the converter's
    ceiling.

    >>> [(round(a), round(b), n) for a, b, n in
    ...  valid_bandpass_rates_hz(140e3, 160e3, fs_max_hz=250e3)][-1]
    (40000, 40000, 8)
    """
    if f_hi_hz <= f_lo_hz or f_lo_hz < 0:
        raise ValueError("invalid band")
    bandwidth = f_hi_hz - f_lo_hz
    n_max = int(np.floor(f_hi_hz / bandwidth))

    out: list[tuple[float, float, int]] = []
    for n in range(1, n_max + 1):
        fs_min = 2.0 * f_hi_hz / n
        fs_max = float("inf") if n == 1 else 2.0 * f_lo_hz / (n - 1)
        if fs_max < fs_min:
            continue
        if fs_max_hz is not None and fs_min > fs_max_hz:
            continue
        out.append((fs_min, fs_max, n))
    return out


def min_bandpass_rate_hz(f_lo_hz: float, f_hi_hz: float) -> float:
    """Lowest ``fs`` that samples the band without overlap.

    It is never lower than the Landau rate; the difference is the price of uniform
    sampling in a single phase.

    >>> min_bandpass_rate_hz(140e3, 160e3)
    40000.0
    """
    rates = valid_bandpass_rates_hz(f_lo_hz, f_hi_hz)
    return min(fs_min for fs_min, _, _ in rates)


# --------------------------------------------------------------------------
# Multi-coset: how many phases are needed
# --------------------------------------------------------------------------
def multicoset_resolvable_orders(n_cosets: int) -> int:
    """How many distinct alias orders ``M`` uniform cosets resolve.

    With offsets ``delta_m = m·Ts/M``, the relative phase between channels
    distinguishes ``k mod M`` — a bandwidth extension of exactly ``M×``.

    ``delta = Ts/2`` (M=2) distinguishes only the parity of ``k``.
    Irrational offsets distinguish every order in the absence of noise, but the
    conditioning degrades exponentially — see `coset_matrix`.
    """
    if n_cosets < 1:
        raise ValueError("n_cosets >= 1")
    return n_cosets


def coset_matrix(offsets_s: ArrayLike, fs_hz: float, orders: ArrayLike) -> NDArray[np.complex128]:
    """Matrix that relates the amplitudes per alias order to the shifted observations.

    Element ``(m, i) = exp(-j·2π·(f_a + k_i·fs)·delta_m)``, up to the common factor
    ``exp(-j·2π·f_a·delta_m)``, which is omitted because it does not depend on ``k``.
    What remains is the Vandermonde matrix in ``exp(-j·2π·fs·delta_m)``.

    The condition number of this matrix is what decides practical feasibility:
    being invertible is not enough, it has to be well conditioned under 12 bits.
    """
    d = np.asarray(offsets_s, dtype=float)[:, None]
    k = np.asarray(orders, dtype=float)[None, :]
    out: NDArray[np.complex128] = np.exp(-2j * np.pi * fs_hz * k * d)
    return out


def uniform_coset_offsets_s(n_cosets: int, fs_hz: float) -> NDArray[np.float64]:
    """Offsets ``m·Ts/M`` — the choice that makes `coset_matrix` a DFT.

    It is the 1D analogue of microscanning: optimal conditioning, condition number 1.
    """
    ts = 1.0 / fs_hz
    return np.arange(n_cosets, dtype=float) * ts / n_cosets
