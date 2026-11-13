"""Measurement vs. prior attribution — the metric that separates recovery from hallucination.

Project rule: **every reconstruction reports this decomposition**. A pretty result
without it is good-looking hallucination.

Given a linear acquisition operator ``A`` and an estimate ``x_hat``, it decomposes::

    x_hat = A⁺A·x_hat        (range space: determined by the data)
          + (I - A⁺A)·x_hat  (null space: chosen by the prior, not measured)

The null-space component is invisible to the observations: any value there produces
exactly the same data. It can be plausible, useful and correct — but it is **not
measured**, and reporting it as recovery is a methodological error.

Conceptual basis: range/null-space decomposition, used in reconstruction with a
learned prior (deep null-space learning; DDNM). Here it is used as an *audit*, and
not as a reconstruction method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Attribution:
    """Result of the audit of a reconstruction."""

    measured_energy: float
    """Energy of ``x_hat`` in the range space of ``Aᴴ`` — comes from the data."""

    prior_energy: float
    """Energy of ``x_hat`` in the null space of ``A`` — comes from the prior."""

    data_consistency: float
    """``||A·x_hat - y|| / ||y||``. Large = the reconstruction does not even reproduce
    what was measured."""

    rank: int
    n_unknowns: int

    @property
    def measured_fraction(self) -> float:
        """Fraction of the estimate's energy that is actually measured, in [0, 1].

        It is the number that goes into the paper. Close to 1: recovery. Close to 0:
        the prior wrote almost everything.
        """
        total = self.measured_energy + self.prior_energy
        return float(self.measured_energy / total) if total > 0.0 else 0.0

    @property
    def null_fraction(self) -> float:
        return 1.0 - self.measured_fraction

    def summary(self) -> str:
        return (
            f"measured {100 * self.measured_fraction:.1f}% | "
            f"prior {100 * self.null_fraction:.1f}% | "
            f"consistency {self.data_consistency:.2e} | "
            f"rank {self.rank}/{self.n_unknowns}"
        )


def attribute(
    a: NDArray[np.complex128] | NDArray[np.float64],
    x_hat: NDArray[np.complex128] | NDArray[np.float64],
    y: NDArray[np.complex128] | NDArray[np.float64],
    rcond: float = 1e-10,
) -> Attribution:
    """Decomposes ``x_hat`` into a measured component and a component coming from the prior.

    ``a``     acquisition operator, shape ``(n_obs, n_unknowns)``
    ``x_hat`` estimate, shape ``(n_unknowns,)``
    ``y``     observations, shape ``(n_obs,)``

    The projector onto the range space is built from the SVD, and not from
    ``pinv(a) @ a``, so that ``rcond`` explicitly controls what counts as a measured
    direction. Directions with a singular value below the cutoff are treated as null
    space: they are formally measured, but with a noise amplification that makes them
    useless under 12 bits.
    """
    op = np.atleast_2d(np.asarray(a))
    xh = np.asarray(x_hat).ravel()
    obs = np.asarray(y).ravel()

    if op.shape[1] != xh.size:
        raise ValueError(f"a has {op.shape[1]} columns but x_hat has {xh.size} elements")
    if op.shape[0] != obs.size:
        raise ValueError(f"a has {op.shape[0]} rows but y has {obs.size} elements")

    _, s, vh = np.linalg.svd(op, full_matrices=False)
    cutoff = rcond * (s[0] if s.size else 0.0)
    keep = s > cutoff
    rank = int(np.count_nonzero(keep))

    v_meas = vh[keep].conj().T  # orthonormal basis of the range space of A^H
    coeffs = v_meas.conj().T @ xh
    measured = float(np.real(np.vdot(coeffs, coeffs)))
    total = float(np.real(np.vdot(xh, xh)))
    prior = max(total - measured, 0.0)

    resid = float(np.linalg.norm(op @ xh - obs))
    denom = float(np.linalg.norm(obs))
    consistency = resid / denom if denom > 0.0 else resid

    return Attribution(
        measured_energy=measured,
        prior_energy=prior,
        data_consistency=consistency,
        rank=rank,
        n_unknowns=int(op.shape[1]),
    )


def hf_recovery_ratio(
    x_hat: NDArray[np.float64],
    x_true: NDArray[np.float64],
    fs_hz: float,
    f_cut_hz: float,
) -> float:
    """Energy recovered above ``f_cut_hz`` over the true energy there.

    Spectral metric requested in the original scope. Use it **together** with
    `attribute`: on its own it does not distinguish recovered energy from invented
    energy that happened to fall in the right band.
    """
    x_hat = np.asarray(x_hat, dtype=float).ravel()
    x_true = np.asarray(x_true, dtype=float).ravel()
    if x_hat.size != x_true.size:
        raise ValueError("x_hat and x_true must have the same size")

    freqs = np.fft.rfftfreq(x_true.size, d=1.0 / fs_hz)
    band = freqs >= f_cut_hz
    e_true = float(np.sum(np.abs(np.fft.rfft(x_true)[band]) ** 2))
    if e_true <= 0.0:
        raise ValueError("there is no energy above f_cut_hz in the true signal")
    e_hat = float(np.sum(np.abs(np.fft.rfft(x_hat)[band]) ** 2))
    return e_hat / e_true
