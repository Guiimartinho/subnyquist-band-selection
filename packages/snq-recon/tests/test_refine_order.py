"""refine_order returns the center of the score plateau, not the first point (review 2026-11-11)."""

from __future__ import annotations

import numpy as np
import pytest
from snq_recon import EnvelopeSpectrum, refine_order, score_fault


def line_spectrum(lines: dict[float, float], step: float = 0.0261, top: float = 32.0):
    """Synthetic order spectrum: exact lines on the grid with step ``step`` (1/revolutions)."""
    f = np.arange(0.0, top, step)
    mag = np.full(f.size, 1e-3)
    for order, amp in lines.items():
        mag[int(round(order / step))] = amp
    return EnvelopeSpectrum(
        freqs_hz=f, magnitude=mag, fs_hz=2 * top, band_hz=None, folds_above_nyquist=False
    )


def test_center_of_plateau_not_first_point():
    step = 0.0261
    true_bin = round(5.43 / step)
    order = true_bin * step
    spec = line_spectrum({order: 1.0, 2 * order: 0.5, 3 * order: 0.3}, step=step)
    nominal, span, tol = 5.427, 0.12, 0.06
    got = refine_order(spec, nominal, span=span, n_harmonics=3, tol=tol)
    grid = np.arange(nominal * (1 - span), nominal * (1 + span), 0.005)
    scores = np.array(
        [score_fault(spec, float(c), n_harmonics=3, tol_hz=tol).harmonic_energy for c in grid]
    )
    first = float(grid[int(np.argmax(scores))])
    assert first < got  # the first point lies below the center
    assert abs(got - order) < abs(first - order)  # and the center is closer to the line
    assert abs(got - order) <= 0.005  # within one grid step


def test_unique_maximum_is_returned_unchanged():
    # score with an isolated maximum (no plateau): center == first point
    spec = line_spectrum({3.40: 1.0}, step=0.001)
    got = refine_order(spec, 3.40, span=0.05, n_harmonics=1, tol=0.0004)
    assert abs(got - 3.40) <= 0.0026


def test_empty_grid_raises():
    spec = line_spectrum({3.4: 1.0})
    with pytest.raises(ValueError):
        refine_order(spec, 3.4, span=0.0, n_harmonics=1, tol=0.06)


def test_detail_flags_a_plateau_touching_the_grid_edge():
    from snq_recon import refine_order_detail

    # the only line is outside the grid (below nominal*(1-span)): the maximum falls on the edge
    spec = line_spectrum({3.0: 1.0}, step=0.0261)
    d = refine_order_detail(spec, 3.573, span=0.12, n_harmonics=1, tol=0.06)
    assert d.at_edge


def test_detail_center_matches_refine_order_and_reports_width():
    from snq_recon import refine_order, refine_order_detail

    step = 0.0261
    order = round(5.43 / step) * step
    spec = line_spectrum({order: 1.0, 2 * order: 0.5, 3 * order: 0.3}, step=step)
    d = refine_order_detail(spec, 5.427, span=0.12, n_harmonics=3, tol=0.06)
    assert not d.at_edge
    assert d.plateau_width > 0.0
    assert d.center == refine_order(spec, 5.427, span=0.12, n_harmonics=3, tol=0.06)
