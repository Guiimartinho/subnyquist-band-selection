"""High-resolution spectral estimation, and order analysis.

Two pieces that the literature reading added to the scope of the baseline.

**ESPRIT** is here as an honest control. Before claiming that a learned estimator
improves on E01 in the −10 to −20 dB region, one has to know whether a classical
high-resolution method already closes the gap. If it does, the marginal gain of
learning is small — and that is a finding, not a setback.

**Order analysis** comes from `sensors2025csvibration` (Kato & Otaka, Sensors 2025,
doi:10.3390/s25103167). Under variable shaft speed, Fourier no longer represents the vibration
sparsely and compression collapses; resampling at constant angular increments
restores sparsity, because the fault frequencies become **fixed orders** of the shaft
rotation. It is a prerequisite for XJTU-SY and FEMTO, which have shaft speed variation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snq_recon.envelope import EnvelopeSpectrum, envelope


def esprit(x: ArrayLike, n_sources: int, fs_hz: float, subspace: int | None = None) -> NDArray:
    """Frequencies by ESPRIT — resolution beyond the FFT limit.

    Exploits the rotational invariance of the signal subspace: two submatrices
    shifted by one sample are related by a rotation whose eigenvalues are
    ``exp(j·2π·f/fs)``. Unlike the FFT peak, it is not limited by ``fs/N``.

    ``n_sources`` is the number of sinusoids to estimate. Overestimating it produces spurious
    frequencies; underestimating it mixes components. That is the weakness of the method
    compared with an estimator that detects the model order by itself.

    >>> fs = 1000.0
    >>> t = np.arange(512) / fs
    >>> x = np.sin(2 * np.pi * 123.4 * t)
    >>> float(esprit(x, 1, fs)[0])  # doctest: +ELLIPSIS
    123.4...
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    m = subspace or max(2 * n_sources + 1, n // 3)
    if m >= n:
        raise ValueError(f"subspace ({m}) must be smaller than the signal ({n})")

    # Hankel matrix: each column is a sliding window of the signal.
    rows = n - m + 1
    h = np.lib.stride_tricks.sliding_window_view(x, m)[:rows].T

    u, _, _ = np.linalg.svd(h, full_matrices=False)
    signal_space = u[:, : 2 * n_sources]  # real signal: 2 exponentials per sinusoid

    # Rotational invariance between the submatrices shifted by one row.
    phi = np.linalg.pinv(signal_space[:-1]) @ signal_space[1:]
    eig = np.linalg.eigvals(phi)

    freqs = np.abs(np.angle(eig)) * fs_hz / (2.0 * np.pi)
    freqs = np.unique(np.round(freqs[freqs > 0], 9))
    return np.sort(freqs)[:n_sources] if freqs.size else np.array([np.nan])


def estimate_tone_esprit(x: ArrayLike, fs_hz: float) -> float:
    """Frequency of the dominant tone by ESPRIT. Same signature as the FFT estimator."""
    f = esprit(x, n_sources=1, fs_hz=fs_hz)
    return float(f[0]) if f.size else float("nan")


# ----------------------------------------------------------------------
# order analysis
# ----------------------------------------------------------------------
def phase_revolutions(
    shaft_hz: ArrayLike | float, fs_hz: float, n_samples: int
) -> NDArray[np.float64]:
    """Accumulated phase in **revolutions**, sample by sample, by integrating the shaft speed.

    ``shaft_hz`` is a scalar (constant speed) or a profile with ``n_samples`` points.
    The integration is trapezoidal, and the first point is zero by construction.

    It exists as a function of its own because two things need the same phase and must not
    diverge: the angular resampling of the classical arm and the order basis of the
    sparse reconstruction arm (`snq_recon.sparse`).

    >>> phase_revolutions(10.0, 1000.0, 5).round(3).tolist()
    [0.0, 0.01, 0.02, 0.03, 0.04]
    """
    n = int(n_samples)
    speed = (
        np.full(n, float(shaft_hz))
        if np.isscalar(shaft_hz)
        else np.asarray(shaft_hz, dtype=float).ravel()
    )
    if speed.size != n:
        raise ValueError(f"shaft speed profile has {speed.size} points, signal has {n}")
    if np.any(speed <= 0):
        raise ValueError("shaft speed must be positive over the whole record")
    dt = 1.0 / float(fs_hz)
    return np.concatenate([[0.0], np.cumsum(0.5 * (speed[:-1] + speed[1:]) * dt)])


_MAX_ANGULAR_OVERSAMPLING = 64
"""Ceiling of the fine angular grid, in multiples of the record length (see below)."""


def angular_resample(
    x: ArrayLike,
    fs_hz: float,
    shaft_hz: ArrayLike | float,
    samples_per_rev: int = 64,
    *,
    anti_alias: bool = True,
) -> tuple[NDArray[np.float64], float]:
    """Resamples the signal at constant angular increments.

    ``shaft_hz`` may be a scalar (constant speed) or a profile sampled at the
    same rate as ``x`` (variable speed, from a tachometer).

    Returns ``(signal_in_angle, samples_per_rev)``. After that, a fault frequency
    that varied with the shaft speed becomes a **fixed order**: BPFO at 3.58× the shaft speed
    always appears at the same place in the order spectrum, regardless of the operating regime.

    **Resampling in angle is decimating in angle.** When the record has more than
    ``samples_per_rev`` samples per revolution, orders above ``samples_per_rev/2``
    fold: order ``o`` reappears at ``|o - spr·round(o/spr)|``. With ``anti_alias=True``
    (the default since 2026-11-11) the signal is interpolated onto a fine angular grid, with at
    least as many samples per revolution as the record has at the minimum speed, goes through an
    ideal low-pass at order ``spr/2`` over the even extension, and only then is decimated. The
    output grid is the same in both modes. ``anti_alias=False`` reproduces the old behavior,
    direct linear interpolation, which folds; it exists for reproduction and for studying the
    folding.

    The fine grid has about ``n·(mean speed / minimum speed)`` points; above
    ``64·n`` (speed varying by more than ~64:1 within the record) the function refuses, instead
    of silently blowing up memory.

    It is the mechanism that `sensors2025csvibration` uses to restore sparsity under
    variable speed — and without it E02 does not apply to XJTU-SY or to FEMTO.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size

    # Accumulated angle in revolutions: integral of the shaft speed.
    revs = phase_revolutions(shaft_hz, fs_hz, n)
    total = float(revs[-1])
    if total < 1.0:
        raise ValueError(f"record covers only {total:.2f} revolutions; insufficient")

    grid = np.arange(0.0, total, 1.0 / samples_per_rev)
    # Source samples per revolution in the worst case (minimum speed). Up to spr,
    # interpolation only raises the rate and there is nothing to fold.
    source_per_rev = 1.0 / float(np.min(np.diff(revs))) if n > 1 else 0.0
    ratio = int(np.ceil(source_per_rev / samples_per_rev))
    if not anti_alias or ratio <= 1:
        return np.interp(grid, revs, x), float(samples_per_rev)

    m = grid.size
    if m * ratio > _MAX_ANGULAR_OVERSAMPLING * n:
        raise ValueError(
            f"fine angular grid of {m * ratio} points for a record of {n}: the shaft speed "
            "varies too much within the record; segment it or use anti_alias=False"
        )
    fine_rate = float(samples_per_rev) * ratio
    fine = np.interp(np.arange(m * ratio) / fine_rate, revs, x)
    # Ideal low-pass at order spr/2. The even extension avoids the jump at the seam, which
    # in a circular filter would turn into Gibbs oscillation at the edges.
    ext = np.concatenate([fine, fine[::-1]])
    spec = np.fft.rfft(ext)
    spec[np.fft.rfftfreq(ext.size, d=1.0 / fine_rate) > samples_per_rev / 2.0] = 0.0
    return np.fft.irfft(spec, n=ext.size)[: m * ratio : ratio], float(samples_per_rev)


def order_spectrum(
    x_angular: ArrayLike, samples_per_rev: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Spectrum in **orders** of the shaft rotation, from the signal already resampled in angle.

    Returns ``(orders, magnitude)``. Order 1 is the shaft rotation; BPFO sits at a
    fixed order given by the geometry, and no longer at a frequency that changes with the
    operating regime.
    """
    x = np.asarray(x_angular, dtype=float).ravel()
    n = x.size
    mag = np.abs(np.fft.rfft(x * np.hanning(n))) * (2.0 / n)
    orders = np.fft.rfftfreq(n, d=1.0 / samples_per_rev)
    return orders, mag


def fault_orders(geom_ratio: float, n_elements: int) -> dict[str, float]:
    """Fault frequencies expressed in **orders of the shaft rotation**, not in hertz.

    They are constants of the geometry: they do not depend on the operating regime. It is the
    correct way to state the prior when the shaft speed varies.
    """
    return {
        "BPFO": (n_elements / 2.0) * (1.0 - geom_ratio),
        "BPFI": (n_elements / 2.0) * (1.0 + geom_ratio),
        "BSF": (1.0 / (2.0 * geom_ratio)) * (1.0 - geom_ratio**2) if geom_ratio > 0 else np.inf,
        "FTF": 0.5 * (1.0 - geom_ratio),
    }


def envelope_order_spectrum(
    x: ArrayLike,
    fs_hz: float,
    shaft_hz: ArrayLike | float,
    band_hz: tuple[float, float] | None = None,
    samples_per_rev: int = 64,
    *,
    anti_alias: bool = True,
) -> EnvelopeSpectrum:
    """Envelope resampled in angle, returned as a spectrum **in orders**.

    Order of operations, and it cannot be permuted: filter in the resonance band and
    take the envelope **in time**, and only then resample in angle. The structural
    resonance is fixed in hertz — it does not follow the shaft speed — so resampling first
    would smear the very band that carries the modulation.

    The returned object is an `EnvelopeSpectrum` whose fields are in **orders of the shaft
    rotation**, not in hertz: ``freqs_hz`` are orders and ``fs_hz`` is ``samples_per_rev``.
    The reuse is deliberate — `score_fault` and `estimate_from_candidates` are
    indifferent to the unit, and under variable speed the geometric prior is a set
    of orders, not of frequencies.

    The envelope has content up to the bandwidth, far above order ``spr/2`` in
    time; without angular anti-aliasing it folded onto ``[0, spr/2]`` (see `angular_resample`),
    and a modulation at order 70 showed up at order 6. With ``anti_alias=True`` the returned
    spectrum declares ``folds_above_nyquist=False``, and `score_fault` does not look for a
    folded harmonic.
    """
    env = envelope(x, fs_hz, band_hz)
    env = env - float(np.mean(env))
    ang, spr = angular_resample(
        env, fs_hz, shaft_hz, samples_per_rev=samples_per_rev, anti_alias=anti_alias
    )
    orders, mag = order_spectrum(ang, spr)
    return EnvelopeSpectrum(
        freqs_hz=orders,
        magnitude=mag,
        fs_hz=spr,
        band_hz=band_hz,
        folds_above_nyquist=not anti_alias,
    )
