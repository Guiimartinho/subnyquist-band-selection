"""Tone estimation and resolution of the alias order with multiple rates.

The project's central idea, in operational form: the aliased frequency alone
does not determine ``f0``, but the **folding pattern under distinct rates** does.
Two rates ``fs1`` and ``fs2`` resolve without ambiguity up to
``lcm(fs1, fs2) / 2`` — which is the real-signal analogue of the Chinese Remainder
Theorem used in radar with staggered PRF.

The formulation here is a **hypothesis test over a discrete set**, and not
an inversion. This is deliberate: it is what allows injecting the geometric prior
(the lines that the machine can produce) without changing the algorithm.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray
from snq_core import alias_frequency_hz


def estimate_tone_hz(x: ArrayLike, fs_hz: float) -> float:
    """Frequency of the dominant tone, by Hann-windowed FFT and parabolic interpolation.

    The three-point interpolation on the magnitude takes the estimator off the FFT
    grid; without it the error is stuck at ``fs/(2N)`` and dominates everything else that this
    experiment wants to measure.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    sp = np.abs(np.fft.rfft(x * np.hanning(n)))
    sp[0] = 0.0  # ignore DC
    k = int(np.argmax(sp))
    if k <= 0 or k >= sp.size - 1:
        return float(k * fs_hz / n)
    a, b, c = float(sp[k - 1]), float(sp[k]), float(sp[k + 1])
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    return float((k + delta) * fs_hz / n)


def unambiguous_range_hz(fs_hz: Sequence[float], quantum_hz: float = 1.0) -> float:
    """Largest ``f0`` resolvable without ambiguity by a set of rates.

    Equals ``lcm(fs) / 2``. The rates are reduced to integers in units of
    ``quantum_hz`` before the lcm — choose ``quantum_hz`` as the intended greatest
    common divisor of the rate design.

    >>> unambiguous_range_hz([40e3, 42e3], quantum_hz=2e3)
    420000.0
    """
    units = [int(round(f / quantum_hz)) for f in fs_hz]
    lcm = units[0]
    for u in units[1:]:
        lcm = lcm * u // gcd(lcm, u)
    return float(lcm * quantum_hz / 2.0)


def alias_candidates_hz(f_alias_hz: float, fs_hz: float, f_max_hz: float) -> NDArray[np.float64]:
    """All ``f0`` in ``[0, f_max]`` that would fold onto ``f_alias_hz`` at ``fs_hz``.

    They are ``f0 = k·fs ± f_alias``. The set is small — about
    ``2·f_max/fs`` elements — and it is this set that the second rate disambiguates.

    >>> len(alias_candidates_hz(10e3, 40e3, 100e3))
    5
    """
    ks = np.arange(0, int(np.ceil((f_max_hz + f_alias_hz) / fs_hz)) + 1)
    cand = np.concatenate([ks * fs_hz - f_alias_hz, ks * fs_hz + f_alias_hz])
    cand = cand[(cand >= 0.0) & (cand <= f_max_hz)]
    return np.unique(np.round(cand, 6))


def resolve_multirate_hz(
    alias_measured_hz: Sequence[float],
    fs_hz: Sequence[float],
    candidates_hz: ArrayLike,
) -> tuple[float, float]:
    """Picks the candidate whose folding pattern best explains the measured aliases.

    For each candidate ``c``, ``alias(c, fs_i)`` is predicted and compared with the
    measurement; the cost is the **worst** deviation across the rates (infinity norm), which
    is more selective than the mean when one of the rates is corrupted.

    Returns ``(f0_estimate, residual_hz)``. The residual is the confidence metric:
    a large residual means that no candidate explains the observations, and
    in that case the answer should be rejected rather than accepted.
    """
    cand = np.asarray(candidates_hz, dtype=float).ravel()
    if cand.size == 0:
        raise ValueError("empty candidate set")

    # (n_candidates, n_rates)
    pred = np.stack([alias_frequency_hz(cand, fs) for fs in fs_hz], axis=1)
    meas = np.asarray(alias_measured_hz, dtype=float)[None, :]
    cost = np.max(np.abs(pred - meas), axis=1)
    i = int(np.argmin(cost))
    return float(cand[i]), float(cost[i])


def prior_grid_hz(f_lo_hz: float, f_hi_hz: float, spacing_hz: float) -> NDArray[np.float64]:
    """Discrete set of admissible frequencies — the physical prior, in 1D.

    In the real case this set comes from the geometry (known resonances,
    harmonics of the shaft speed); see ``snq_core.bearing.candidate_lines_hz``. Here it
    is a regular grid, so that the gain from the prior is measured as a function of its
    **cardinality** only, without confounding it with the particular structure of the
    set.
    """
    return np.arange(f_lo_hz, f_hi_hz + spacing_hz, spacing_hz, dtype=float)
