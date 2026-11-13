"""The CS order spectrum does not fold, and score_fault must not look for folded harmonics in it."""

from __future__ import annotations

import numpy as np
from snq_recon import score_fault, sparse_order_spectrum


def test_sparse_order_spectrum_declares_no_folding():
    orders = np.arange(0.5, 24.0 + 1e-9, 0.5)
    coef = np.zeros(2 * orders.size)
    spec = sparse_order_spectrum(coef, orders)
    assert spec.folds_above_nyquist is False


def test_harmonic_above_axis_is_not_folded_back():
    """Line only at order 6: a candidate at 18 (its h=1 falls outside the axis) must not score
    with 6."""
    orders = np.arange(0.5, 12.0 + 1e-9, 0.5)
    coef = np.zeros(2 * orders.size)
    coef[2 * int(np.flatnonzero(np.isclose(orders, 6.0))[0])] = 1.0
    spec = sparse_order_spectrum(coef, orders)
    s = score_fault(spec, 18.0, n_harmonics=1, tol_hz=0.1)
    assert s.harmonic_energy == 0.0
