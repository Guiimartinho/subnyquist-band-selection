"""Sampling-clock jitter — the real enemy of this project.

Under undersampling, the phase error is proportional to the **input** frequency, not
to the conversion frequency. Undersampling by ``N×`` amplifies the effect of jitter
``N×``. This module exists so that no undersampling-factor promise is made without
going through here first.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def phase_error_rad(f_in_hz: ArrayLike, sigma_t_s: float) -> NDArray[np.float64]:
    """Standard deviation of the phase error induced by jitter ``sigma_t_s``.

    ``sigma_phi = 2·pi·f_in·sigma_t``. Depends on the **input** frequency.
    """
    return 2.0 * np.pi * np.asarray(f_in_hz, dtype=float) * sigma_t_s


def snr_limit_db(f_in_hz: ArrayLike, sigma_t_s: float) -> NDArray[np.float64]:
    """SNR ceiling imposed by jitter alone: ``-20·log10(2·pi·f_in·sigma_t)``.

    Compare with the quantization SNR of a ``B``-bit ADC (~``6.02·B + 1.76``).
    When this ceiling falls below that one, **the converter is no longer the limiting
    factor** and the oscillator is — the point at which the project needs an external
    reference.
    """
    return -20.0 * np.log10(phase_error_rad(f_in_hz, sigma_t_s))


def quantization_snr_db(n_bits: int) -> float:
    """Ideal quantization SNR for a full-scale sine wave, in dB."""
    return 6.02 * n_bits + 1.76


def max_input_frequency_hz(n_bits: int, sigma_t_s: float) -> float:
    """Input frequency at which the jitter equals the quantization floor.

    It is the practical undersampling ceiling of the hardware. Above it, increasing
    the ADC resolution buys nothing.
    """
    target = quantization_snr_db(n_bits)
    return float(10.0 ** (-target / 20.0) / (2.0 * np.pi * sigma_t_s))
