"""Kurtogram — picks the resonant band where the transient is most impulsive.

It is the piece that decides **where** bandpass sampling should be centered. Without it the
choice of band is arbitrary, and it is precisely this link — the kurtogram picks the
band, bandpass theory says at what rate to sample it — that supports the project's system
contribution.

Antoni (MSSP 21(1):108-124, 2007) shows that the cost is **O(N log N)**, the same order
as an FFT. This was checked against the source and has an architectural consequence: the
sweep **fits on the node**, and there is no need to transmit a full-rate record to the gateway
just to pick the band — which would be exactly the radio expenditure to avoid.

Implementation: 1/3-binary grid, ideal filtering in the FFT domain, and kurtosis of the
complex envelope by Eq. (7). It is not the filter-bank recursion of the paper. What
agrees with the source and what does not, checked against the PDF on 2026-11-11:

- **Agrees:** without ``f_min_hz``/``f_max_hz``, level ``L`` has ``2^L`` bands of width
  ``fs·2^(-L-1)``, with centers from Eq. (5) and widths from Eq. (6); the intermediate levels
  ``k + log2(3) - 1`` give the widths 1/6, 1/12, 1/24 of Fig. 5. Estimator from Eq. (7).
- **Deviation 1, partition of the search range.** With ``f_min_hz``/``f_max_hz`` the ``2^L``
  bands divide ``[f_min, min(f_max, fs/2)]``, and not ``[0, fs/2]``. The limits **change the
  partition**, they do not only mask candidates. Consequence: the partition depends on ``fs``
  only when ``fs/2 < f_max``. In E07 (500 Hz to 25 kHz over 200 kSPS) factors 1, 2 and 4
  see the same bands and at 8x they change; in Antoni they would change at every factor.
- **Deviation 2, level 0 absent.** The whole band is not a candidate.
- **Deviation 3, filters.** Ideal masks, with no overlap between neighboring bands and no
  decimation; the paper's quasi-analytic FIR filters overlap (Figs. 3 and 4) and the
  kurtosis is taken at the critically decimated output.
- **Edges, as in note 3 of the paper.** FFT filtering is circular, and the jump at the
  seam of the record becomes a false transient at the edges. The record is extended by
  mirroring (the periodic extension becomes continuous) and the ``EDGE_GUARD_PERIODS·fs/Δf``
  samples at each edge are left out of the kurtosis. Before the fix of 2026-11-11, stationary
  Gaussian noise with a machine spectrum gave SK above 100, contrary to
  property (ii) of the paper, which requires it to be zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snq_recon.envelope import _band_analytic_from_spectrum

EDGE_GUARD_PERIODS = 4.0
"""Samples excluded at each edge, in units of ``fs/Δf`` (the inverse of the bandwidth).

It is the order of the transient of the 8-coefficient FIR prototype that Antoni uses in all
the examples (level ``k``: ``L·2^k = L·fs/(2·Δf)`` samples, ``L = 8``) — that calculation is
this project's, not the paper's. Measured on colored Gaussian noise: with mirroring, 4 is
enough for the SK to drop to the estimator's floor; without mirroring not even 32 was enough.
"""


def _edge_guard(n_samples: int, fs_hz: float, bandwidth_hz: float) -> int:
    """Samples to discard at each edge; never more than a quarter of the record."""
    return min(int(np.ceil(EDGE_GUARD_PERIODS * fs_hz / bandwidth_hz)), n_samples // 4)


def spectral_kurtosis(z: NDArray[np.complex128]) -> float:
    """Kurtosis of the envelope of a band analytic signal.

    Normalized so that circular complex Gaussian noise gives ~0 and a train of
    rare impulses gives a high value. It is the usual definition in machine diagnostics.
    """
    e2 = np.abs(z) ** 2
    m2 = float(np.mean(e2))
    if m2 <= 0.0:
        return 0.0
    return float(np.mean(e2**2) / (m2 * m2) - 2.0)


@dataclass(frozen=True)
class KurtogramResult:
    """Best (band, resolution) pair and the full map."""

    f_lo_hz: float
    f_hi_hz: float
    kurtosis: float
    level: float
    grid: list[tuple[float, float, float, float]]
    """All evaluated cells: ``(level, f_lo, f_hi, kurtosis)``."""

    @property
    def band_hz(self) -> tuple[float, float]:
        return (self.f_lo_hz, self.f_hi_hz)

    @property
    def center_hz(self) -> float:
        return 0.5 * (self.f_lo_hz + self.f_hi_hz)

    @property
    def bandwidth_hz(self) -> float:
        return self.f_hi_hz - self.f_lo_hz

    def summary(self) -> str:
        return (
            f"band [{self.f_lo_hz / 1e3:.2f}, {self.f_hi_hz / 1e3:.2f}] kHz "
            f"(center {self.center_hz / 1e3:.2f} kHz, width {self.bandwidth_hz / 1e3:.2f} kHz) "
            f"| kurtosis {self.kurtosis:.2f} | level {self.level:g}"
        )


def _levels(max_level: int) -> list[float]:
    """1/3-binary levels of Antoni (2007), Fig. 5: ``k`` and ``k + log2(3) - 1``.

    The intermediate level splits the band into three, and it is not a 1/3-octave step: it lies
    0.585 octave from level ``k``. Level 0 is not included (deviation 2 in the module
    docstring).
    """
    out: list[float] = []
    for k in range(1, max_level + 1):
        out += [float(k), k + np.log2(3.0) - 1.0]
    return sorted(v for v in out if v <= max_level)


def fast_kurtogram(
    x: ArrayLike,
    fs_hz: float,
    max_level: int = 6,
    f_min_hz: float = 0.0,
    f_max_hz: float | None = None,
    min_energy_frac: float = 1e-4,
) -> KurtogramResult:
    """Sweeps the (center frequency, width) plane and returns the most impulsive band.

    ``min_energy_frac`` discards practically empty bands: in a noiseless synthetic
    signal, the kurtosis of a band with no energy measures floating-point rounding.
    Not to be confused with the false transient from the record seam, which is handled
    separately (mirroring and edge guard, see the module docstring) and which affected bands
    with real energy, above any reasonable threshold.

    ``max_level`` controls the finest resolution: at level ``k`` the search range is
    divided into ``2**k`` bands. ``f_min_hz`` and ``f_max_hz`` **define the range that is
    partitioned**, they do not only mask candidates: the partition of
    ``[f_min, min(f_max, fs/2)]`` does not depend on ``fs`` as long as ``fs/2 >= f_max``. With
    both at their defaults, the partition is that of Antoni (2007).

    The result feeds directly into `snq_core.valid_bandpass_rates_hz`: the band
    picked here is the one that bandpass sampling will center on.
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

    # Even extension: the periodic extension of [x, x reversed] has no jump at the seam,
    # only a corner, whose transient decays as 1/t^2 and fits within the edge guard.
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
                continue  # empty band: the kurtosis would be numerically degenerate
            guard = _edge_guard(n, fs_hz, float(up - lo))
            k = spectral_kurtosis(z[guard : n - guard])  # note 3 of Antoni (2007)
            grid.append((level, float(lo), float(up), k))
            if k > best[2]:
                best = (float(lo), float(up), k, level)

    return KurtogramResult(
        f_lo_hz=best[0], f_hi_hz=best[1], kurtosis=best[2], level=best[3], grid=grid
    )


def band_from_kurtogram(x: ArrayLike, fs_hz: float, **kw: object) -> tuple[float, float]:
    """Shortcut: only the chosen band, for chaining with the envelope analysis."""
    return fast_kurtogram(x, fs_hz, **kw).band_hz  # type: ignore[arg-type]
