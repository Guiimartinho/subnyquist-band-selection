"""Sparse reconstruction over an **order basis**, and sampling at random instants.

It is the implementation of the project's most direct competitor: Kato & Otaka,
*Sensors* 25(10):3167, 2025 (`sensors2025csvibration`, doi:10.3390/s25103167).

**The prevailing reading of their method was wrong and was corrected against the source.**
The internal report described a "random Gaussian measurement matrix". The source says
something else, in the text of Section 2 and of Figure 1a: *"both the rotational speed and
vibration signals are first randomly sampled. The rotational speed data are then
interpolated and integrated to obtain the rotational phase. Using this rotational
phase information, the order basis is constructed"*. The mention of a Gaussian matrix
appears as a generic justification of RIP, not as the acquisition operator used.
What they do is **random selection of sampling instants**. For the bearing vibration
case the source reports ``α = 0.0391``.

The distinction matters for this project and is not cosmetic: multiplying the signal by a
dense Gaussian matrix requires acquiring the whole signal first — it saves no
conversions at all. Selecting instants saves conversions, and it is exactly what
DPPI + TIMER → SAADC does in hardware on the nRF54L15, with the CPU asleep.

The pieces, and nothing beyond them:

``random_time_indices``   which instants to convert;
``interpolated_speed_hz`` shaft speed interpolated onto the original grid;
``order_basis``           cos/sin dictionary at the shaft orders, evaluated at those instants;
``omp``                   Orthogonal Matching Pursuit over that dictionary;
``reconstruct_from_orders`` the waveform back in time, which the source **demodulates**.

The last step is not optional if the goal is to reproduce the source. In the experimental
bearing case it reconstructs and **then** demodulates: *"the envelope
processing results of the reconstructed waveform in the time domain and its Fourier
transform are shown in Figure 17b […] envelope processing of the CS results using the
order basis can detect outer ring failures"*. Stopping at the order spectrum of the
reconstructed carrier is an amputated version of their method, not their method.

What is **not** implemented here, and why: FISTA (l1) and robust CRT. Neither of them
is needed for the controlled comparison of E13, and advertising a nonexistent function
was a real defect of this package — see the docstring of ``snq_recon/__init__.py``.

Declared limitation: no minimum-interval constraint between conversions is imposed by
default. The ``min_gap_samples`` parameter exists because the SAR core has a finite
conversion time and a schedule with two back-to-back triggers is not realizable — but the
realizable value on the nRF54L15 is a bench measurement that has not been made yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snq_recon.envelope import EnvelopeSpectrum
from snq_recon.highres import phase_revolutions

__all__ = [
    "SparseSolution",
    "interpolated_phase_rev",
    "interpolated_speed_hz",
    "omp",
    "order_basis",
    "order_of_column",
    "random_time_indices",
    "reconstruct_from_orders",
    "sparse_order_spectrum",
    "uniform_time_indices",
]


# ----------------------------------------------------------------------
# sampling schedule
# ----------------------------------------------------------------------
def uniform_time_indices(n_samples: int, factor: int) -> NDArray[np.int64]:
    """Instants of uniform decimation: ``0, factor, 2·factor, …``.

    It is the control of E13. Kept separate from ``snq_data.undersample`` on purpose: here
    what matters is the **set of instants**, so that the three arms can be
    compared by the number of conversions and not by the resulting signal.

    >>> uniform_time_indices(10, 3).tolist()
    [0, 3, 6, 9]
    """
    if n_samples < 1:
        raise ValueError("n_samples >= 1")
    if factor < 1:
        raise ValueError("factor >= 1")
    return np.arange(0, int(n_samples), int(factor), dtype=np.int64)


def random_time_indices(
    n_samples: int,
    rng: np.random.Generator,
    *,
    alpha: float | None = None,
    n_selected: int | None = None,
    min_gap_samples: int = 1,
) -> NDArray[np.int64]:
    """Instants chosen at random, without repetition, in increasing order.

    Exactly one of ``alpha`` (fraction of the original grid) or ``n_selected`` (absolute
    number of conversions). The latter exists to match the conversion budget to that of
    the uniform arm **exactly**, and not up to a rounding — the
    E13 comparison is in absolute samples per second.

    ``min_gap_samples`` guarantees a minimum spacing between consecutive conversions. The
    construction draws on a shrunken interval and re-expands, which is uniform over
    the valid configurations; ``min_gap_samples=1`` reduces to the simple draw.
    """
    n = int(n_samples)
    if n < 1:
        raise ValueError("n_samples >= 1")
    if (alpha is None) == (n_selected is None):
        raise ValueError("give exactly one of alpha or n_selected")

    m = int(round(float(alpha) * n)) if alpha is not None else int(n_selected)  # type: ignore[arg-type]
    if not 1 <= m <= n:
        raise ValueError(f"number of conversions ({m}) must be in [1, {n}]")

    gap = int(min_gap_samples)
    if gap < 1:
        raise ValueError("min_gap_samples >= 1")

    available = n - (m - 1) * (gap - 1)
    if available < m:
        raise ValueError(
            f"min_gap_samples={gap} does not fit: {m} conversions in {n} samples require "
            f"at least {(m - 1) * gap + 1} grid samples"
        )
    base = np.sort(rng.choice(available, size=m, replace=False))
    return (base + np.arange(m, dtype=np.int64) * (gap - 1)).astype(np.int64)


def interpolated_speed_hz(
    indices: ArrayLike,
    speed_hz_at_indices: ArrayLike,
    n_samples: int,
) -> NDArray[np.float64]:
    """Shaft speed in hertz interpolated onto the **original grid**, from the instants.

    It is the first step of the source's method — *"the rotational speed data are then
    interpolated and integrated to obtain the rotational phase"*.

    It exists separately from `interpolated_phase_rev` because demodulating the reconstructed
    signal (the step of Figure 17b of the source) requires the **dense** speed profile, and
    not only the phase at the converted instants. Without it, anyone who needed the dense
    profile would have to reimplement the interpolation — and two copies diverge.
    """
    idx = np.asarray(indices, dtype=np.int64).ravel()
    speed = np.asarray(speed_hz_at_indices, dtype=float).ravel()
    if idx.size != speed.size:
        raise ValueError(f"{idx.size} instants against {speed.size} shaft speed values")
    if idx.size < 2:
        raise ValueError("shaft speed interpolation requires at least 2 instants")
    grid = np.arange(int(n_samples), dtype=float)
    return np.interp(grid, idx.astype(float), speed)


def interpolated_phase_rev(
    indices: ArrayLike,
    speed_hz_at_indices: ArrayLike,
    fs_hz: float,
    n_samples: int,
) -> NDArray[np.float64]:
    """Rotation phase, in **revolutions**, from the shaft speed measured only at the instants.

    Reproduces the source's step: interpolates the sparsely sampled shaft speed onto the
    original grid and integrates it to obtain the phase. Returns the phase **at the instants
    of ``indices``**, which is what the order basis needs.

    In this project the shaft speed does not go through the ADC: it comes from pulse counting
    on a digital input (TIMER/GPIOTE). The conversion budget accounted for is that of the
    vibration. Even so, it is undersampled here, so that the arm reproduces the
    source's method and not an easier version of it.
    """
    idx = np.asarray(indices, dtype=np.int64).ravel()
    dense = interpolated_speed_hz(idx, speed_hz_at_indices, int(n_samples))
    return phase_revolutions(dense, fs_hz, int(n_samples))[idx]


# ----------------------------------------------------------------------
# order dictionary
# ----------------------------------------------------------------------
def order_basis(phase_rev: ArrayLike, orders: ArrayLike) -> NDArray[np.float64]:
    """Dictionary ``(n_obs, 2·n_orders)`` of cosines/sines at the shaft orders.

    Column ``2k`` is ``cos(2π·order_k·θ)`` and column ``2k+1`` is ``sin(2π·order_k·θ)``,
    with ``θ`` the phase in revolutions. It is the source's order basis: under variable speed
    it is sparse where the Fourier basis in hertz is not, because the fault component
    stays at a fixed order while the frequency in hertz moves.

    **It is not normalized.** It is the honest acquisition operator, and it is the one that goes
    to ``snq_eval.attribute``. The column normalization that OMP needs is done
    inside ``omp`` and undone in the returned coefficients.
    """
    theta = np.asarray(phase_rev, dtype=float).ravel()
    ords = np.asarray(orders, dtype=float).ravel()
    if ords.size == 0:
        raise ValueError("empty order grid")

    arg = 2.0 * np.pi * np.outer(theta, ords)
    a = np.empty((theta.size, 2 * ords.size), dtype=float)
    a[:, 0::2] = np.cos(arg)
    a[:, 1::2] = np.sin(arg)
    return a


def order_of_column(orders: ArrayLike) -> NDArray[np.float64]:
    """Order associated with each column of ``order_basis`` — each order takes up two."""
    return np.repeat(np.asarray(orders, dtype=float).ravel(), 2)


# ----------------------------------------------------------------------
# OMP
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SparseSolution:
    """Sparse solution, always carrying the residual — without it there is no way to audit."""

    coefficients: NDArray[np.float64]
    """Vector of length ``n_columns``, zero outside the support, on the scale of the
    **original** (unnormalized) dictionary."""

    support: NDArray[np.int64]
    """Chosen column indices, in the order in which OMP chose them."""

    residual_norm: float
    """Absolute ``||y - A·x||``."""

    observation_norm: float
    """``||y||``, to form the relative residual without recomputing."""

    n_iterations: int

    @property
    def relative_residual(self) -> float:
        return (
            self.residual_norm / self.observation_norm
            if self.observation_norm > 0.0
            else self.residual_norm
        )


def omp(
    a: ArrayLike,
    y: ArrayLike,
    *,
    n_nonzero: int,
    tol: float = 0.0,
    normalize: bool = True,
) -> SparseSolution:
    """Orthogonal Matching Pursuit.

    At each step it picks the column most correlated with the residual, solves least
    squares **over the whole accumulated support**, and starts over. The "orthogonal" is that
    refit: without it the residual does not become orthogonal to the support and the same
    column is picked again.

    ``n_nonzero`` is the maximum number of columns. ``tol`` stops earlier if the relative
    residual drops below it; ``tol=0`` always runs all ``n_nonzero`` iterations.

    ``normalize`` scales the columns to unit norm **only internally**, so that
    the correlation compares directions and not amplitudes. The returned coefficients are
    on the scale of ``a``, so that ``a @ solution.coefficients`` reproduces the fit.

    **Cost, and it matters for anyone who is going to quote run times.** The fit is
    re-solved from scratch at every iteration, without incremental QR or Cholesky
    updates. It is the simple and correct formulation, and it grows fast with ``n_nonzero``:
    measured on ``(7813, 2632)`` by E13's `sensitivity_atoms.py`, 40 atoms took
    0.5 s, 120 took 2.9 s and 400 took 48 s — about ``k^2``, not the theoretical cube,
    because the LAPACK behind ``lstsq`` is not the bottleneck that a naive flop count
    suggests.

    These measurements are fragile, and the fragility has been established: two consecutive
    runs of the same sweep, on the same machine, gave 28.1 s and 47.9 s at the 400-atom point,
    with residual and estimate identical to the last digit. An older measurement of the same
    point recorded 743 s; it **was not reproduced** and must not be cited. Wall-clock time on
    a shared machine is the least reliable number in this repository — the
    counts (atoms, columns, conversions) do not have that problem.

    If the bottleneck of some experiment ever becomes this loop, the fix is an incremental
    QR update — that is not the case for E13, which runs with 40 atoms.
    """
    op = np.atleast_2d(np.asarray(a, dtype=float))
    obs = np.asarray(y, dtype=float).ravel()
    if op.shape[0] != obs.size:
        raise ValueError(f"a has {op.shape[0]} rows but y has {obs.size} elements")

    k_max = int(n_nonzero)
    if not 1 <= k_max <= min(op.shape):
        raise ValueError(f"n_nonzero must be in [1, {min(op.shape)}], got {k_max}")

    norms = np.linalg.norm(op, axis=0)
    scale = np.where(norms > 0.0, norms, 1.0) if normalize else np.ones(op.shape[1])
    dic = op / scale if normalize else op

    obs_norm = float(np.linalg.norm(obs))
    resid = obs.copy()
    support: list[int] = []
    coef_support = np.zeros(0)

    for _ in range(k_max):
        corr = dic.T @ resid
        if support:
            corr[support] = 0.0
        col = int(np.argmax(np.abs(corr)))
        if corr[col] == 0.0:
            break  # nothing correlated is left: dictionary exhausted
        support.append(col)
        sub = dic[:, support]
        coef_support = np.linalg.lstsq(sub, obs, rcond=None)[0]
        resid = obs - sub @ coef_support
        if tol > 0.0 and obs_norm > 0.0 and float(np.linalg.norm(resid)) <= tol * obs_norm:
            break

    coefficients = np.zeros(op.shape[1], dtype=float)
    if support:
        coefficients[support] = coef_support / scale[support]
    return SparseSolution(
        coefficients=coefficients,
        support=np.asarray(support, dtype=np.int64),
        residual_norm=float(np.linalg.norm(resid)),
        observation_norm=obs_norm,
        n_iterations=len(support),
    )


# ----------------------------------------------------------------------
# output in a form comparable with the classical arm
# ----------------------------------------------------------------------
def sparse_order_spectrum(coefficients: ArrayLike, orders: ArrayLike) -> EnvelopeSpectrum:
    """Amplitude per order, packaged as an `EnvelopeSpectrum` **in orders**.

    ``magnitude[k] = hypot(c_cos, c_sin)`` of order ``k``. The reuse of the type is the
    same one that `envelope_order_spectrum` makes: `score_fault`, `estimate_from_candidates`
    and `estimate_without_prior` are indifferent to the axis unit, and so the CS
    arm is judged by the **same estimator** as the classical arm. Without this the comparison
    would measure a difference in estimator, not a difference in acquisition.

    ``fs_hz`` is filled with ``2·max_order``, the Nyquist analogue on the order
    axis; the field is informative, and anyone reading the axis ceiling reads ``freqs_hz[-1]``.
    """
    coef = np.asarray(coefficients, dtype=float).ravel()
    ords = np.asarray(orders, dtype=float).ravel()
    if coef.size != 2 * ords.size:
        raise ValueError(f"{coef.size} coefficients do not form pairs over {ords.size} orders")
    mag = np.hypot(coef[0::2], coef[1::2])
    # The dictionary only contains orders up to order_max: nothing above "folds" into the axis,
    # so score_fault must not look for a folded harmonic (audit of 2026-11-11).
    return EnvelopeSpectrum(
        freqs_hz=ords,
        magnitude=mag,
        fs_hz=2.0 * float(ords[-1]),
        band_hz=None,
        folds_above_nyquist=False,
    )


def reconstruct_from_orders(
    phase_rev: ArrayLike, orders: ArrayLike, coefficients: ArrayLike
) -> NDArray[np.float64]:
    """Reconstructed signal at an arbitrary phase, from the sparse coefficients.

    Walks only the support, one atom at a time: reconstructing on a dense grid of
    hundreds of thousands of points with the whole dictionary in memory does not fit, and the
    solution is sparse precisely so as not to need that.
    """
    theta = np.asarray(phase_rev, dtype=float).ravel()
    ords = np.asarray(orders, dtype=float).ravel()
    coef = np.asarray(coefficients, dtype=float).ravel()
    if coef.size != 2 * ords.size:
        raise ValueError(f"{coef.size} coefficients do not form pairs over {ords.size} orders")

    out = np.zeros(theta.size, dtype=float)
    col_order = order_of_column(ords)
    for col in np.flatnonzero(coef):
        arg = 2.0 * np.pi * col_order[col] * theta
        out += coef[col] * (np.sin(arg) if col % 2 else np.cos(arg))
    return out
