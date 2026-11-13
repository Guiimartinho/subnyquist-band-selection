"""Envelope analysis — the project's official baseline.

It is the method that industry already uses, and the number that anything learned has to
beat. `CLAUDE.md`: the baseline to beat is classical bandpass sampling + envelope,
not a CNN.

A pitfall found on a simulation bench and recorded in the lab notebook of
2026-09-02: **taking the magnitude of a real signal doubles the bandwidth**. If the
doubled band occupies the whole of `[0, fs/2]`, `|y|` aliases onto itself and the envelope
spectrum turns into garbage. That is why the envelope here is always computed on the
**analytic signal** (Hilbert), and `envelope_headroom_ok` exists to check the headroom before
trusting the result.

References: Randall & Antoni, *Rolling element bearing diagnostics — a tutorial*,
MSSP 2011; Antoni, *Fast computation of the kurtogram*, MSSP 2007.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal


def _band_analytic_from_spectrum(
    spec: NDArray[np.complex128], freqs: NDArray[np.float64], f_lo_hz: float, f_hi_hz: float
) -> NDArray[np.complex128]:
    """Band analytic signal from an already computed FFT: ``2·X`` at positive bins, ``X`` at DC.

    DC enters with weight 1, not 2: the analytic signal is ``x + j·H{x}`` and ``H`` cancels DC,
    so ``X_a(0) = X(0)``. With weight 2 the real part gained the mean again (fixed on
    2026-11-11; only affects bands with ``f_lo_hz <= 0``). The ``fs/2`` bin falls among the
    negative frequencies of ``fftfreq`` and stays out, as before.
    """
    keep = (freqs >= f_lo_hz) & (freqs < f_hi_hz)  # positive frequencies only => analytic
    out = np.zeros(spec.size, dtype=complex)
    out[keep] = 2.0 * spec[keep]
    if keep[0]:
        out[0] = spec[0]
    return np.fft.ifft(out)


def bandpass_analytic(x: ArrayLike, fs_hz: float, f_lo_hz: float, f_hi_hz: float) -> NDArray:
    """Analytic signal of the band ``[f_lo, f_hi)``, by filtering in the FFT domain.

    Zeroes everything outside the band and the negative frequencies in one go — which yields
    the analytic signal of the band without having to filter and then apply Hilbert. With the
    whole band ``[0, fs/2)`` it coincides with ``scipy.signal.hilbert`` (except for the
    ``fs/2`` bin).

    **The filtering is circular.** It is exact for the periodic extension of the record, and
    that extension has a jump at the seam ``x[-1] -> x[0]``: the jump shows up as a
    transient at both edges of ``z``. In the envelope spectra of this module the Hann window
    erases it. In a statistic without a window, such as the kurtosis, it becomes a false
    transient — which is why `fast_kurtogram` does not use this function directly.
    """
    x = np.asarray(x, dtype=float).ravel()
    spec = np.fft.fft(x)
    freqs = np.fft.fftfreq(x.size, d=1.0 / fs_hz)
    return _band_analytic_from_spectrum(spec, freqs, f_lo_hz, f_hi_hz)


def envelope(x: ArrayLike, fs_hz: float, band_hz: tuple[float, float] | None = None) -> NDArray:
    """Envelope of the signal, optionally restricted to a band.

    Without a band, it uses the Hilbert transform of the whole signal.
    """
    if band_hz is None:
        return np.abs(signal.hilbert(np.asarray(x, dtype=float).ravel()))
    return np.abs(bandpass_analytic(x, fs_hz, *band_hz))


@dataclass(frozen=True)
class EnvelopeSpectrum:
    """Envelope spectrum in hertz, or in orders (see `envelope_order_spectrum`).

    ``magnitude`` is ``|rfft(env·hann)|·2/n``: a line of amplitude ``A`` comes out as ``A/2``,
    because the coherent gain of the Hann window is not compensated. It is meant for argmax and
    ratios, which is how the package uses it, and not as a calibrated amplitude.
    """

    freqs_hz: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    fs_hz: float
    band_hz: tuple[float, float] | None
    folds_above_nyquist: bool = True
    """Does the content above ``fs_hz/2`` reappear folded in the spectrum?

    True for an envelope computed on uniform samples (`envelope_spectrum`):
    the harmonics above Nyquist are there, at the position given by the alias map. False when
    that content was removed beforehand, as in the order spectrum with angular anti-aliasing.
    `score_fault` only looks for a folded harmonic when this field is true.
    """

    def peak_near(self, f_hz: float, tol_hz: float) -> tuple[float, float]:
        """Largest line within ``f_hz ± tol_hz``. Returns ``(frequency, magnitude)``."""
        sel = np.abs(self.freqs_hz - f_hz) <= tol_hz
        if not np.any(sel):
            return (float("nan"), 0.0)
        idx = np.flatnonzero(sel)
        best = idx[int(np.argmax(self.magnitude[idx]))]
        return float(self.freqs_hz[best]), float(self.magnitude[best])


def envelope_spectrum(
    x: ArrayLike,
    fs_hz: float,
    band_hz: tuple[float, float] | None = None,
    *,
    detrend: bool = True,
) -> EnvelopeSpectrum:
    """Envelope spectrum — where the fault frequencies show up.

    The mean value of the envelope is removed by default: it is a large DC component
    that dominates the spectrum and carries no fault information.
    """
    env = envelope(x, fs_hz, band_hz)
    if detrend:
        env = env - float(np.mean(env))
    n = env.size
    mag = np.abs(np.fft.rfft(env * np.hanning(n))) * (2.0 / n)
    return EnvelopeSpectrum(
        freqs_hz=np.fft.rfftfreq(n, d=1.0 / fs_hz),
        magnitude=mag,
        fs_hz=fs_hz,
        band_hz=band_hz,
    )


def envelope_headroom_ok(band_hz: tuple[float, float], fs_hz: float, factor: float = 2.5) -> bool:
    """Does the rate leave headroom for the envelope, or will it alias?

    The bandwidth of the envelope is ~2× that of the signal band. Sampling at exactly
    ``2·B`` accommodates the folded carrier but **not** the envelope. The project's rule of
    thumb is ``fs >= 2.5·B``.

    >>> envelope_headroom_ok((140e3, 160e3), fs_hz=40e3)
    False
    >>> envelope_headroom_ok((140e3, 160e3), fs_hz=60e3)
    True
    """
    bandwidth = band_hz[1] - band_hz[0]
    return fs_hz >= factor * bandwidth


def squared_envelope_spectrum(
    x: ArrayLike, fs_hz: float, band_hz: tuple[float, float] | None = None
) -> EnvelopeSpectrum:
    """**Squared** envelope spectrum.

    Randall & Antoni argue that the squared envelope is preferable to the magnitude
    envelope: it is what follows naturally from cyclostationary theory, and it avoids the
    distortion introduced by the square root implicit in ``|·|``.
    """
    env = (
        np.abs(bandpass_analytic(x, fs_hz, *band_hz)) ** 2
        if band_hz
        else np.abs(signal.hilbert(np.asarray(x, dtype=float).ravel())) ** 2
    )
    env = env - float(np.mean(env))
    n = env.size
    mag = np.abs(np.fft.rfft(env * np.hanning(n))) * (2.0 / n)
    return EnvelopeSpectrum(np.fft.rfftfreq(n, d=1.0 / fs_hz), mag, fs_hz, band_hz)
