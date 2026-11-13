"""Estimators on the envelope spectrum: search range and handling of harmonics.

Hand-built synthetic spectra, so that each assertion depends on a single rule.
"""

from __future__ import annotations

import numpy as np
import pytest
from snq_core import alias_frequency_hz
from snq_recon import (
    EnvelopeSpectrum,
    estimate_from_candidates,
    estimate_without_prior,
    score_fault,
)


def _lines(
    lines: dict[float, float], nyquist: float = 32.0, step: float = 0.01, **kw
) -> EnvelopeSpectrum:
    """Spectrum with a unit floor and lines of given height, on the grid ``[0, nyquist]``."""
    freqs = np.arange(0.0, nyquist + 0.5 * step, step)
    mag = np.ones_like(freqs)
    for f, a in lines.items():
        mag[int(round(f / step))] = a
    return EnvelopeSpectrum(freqs_hz=freqs, magnitude=mag, fs_hz=2.0 * nyquist, band_hz=None, **kw)


def test_sem_prior_e_o_maximo_global_entre_f_min_e_nyquist():
    """Without the prior there is no ceiling: a shaft harmonic stronger than the fault wins.

    It is the declared behavior of E07's ``sem_prior`` (without prior) mode, and the reason it
    loses hits on records where the 2nd shaft harmonic dominates the envelope.
    """
    spec = _lines({1.0: 90.0, 2.0: 50.0, 3.4: 40.0})
    assert estimate_without_prior(spec, f_min_hz=2.0) == pytest.approx(2.0)
    assert estimate_without_prior(spec, f_min_hz=2.5) == pytest.approx(3.4)
    assert np.isnan(estimate_without_prior(spec, f_min_hz=40.0))


def test_ponderacao_1_sobre_h_rejeita_o_subharmonico():
    """The series of ``f/3`` contains ``f``; with 1/h it cannot beat the series of ``f``."""
    f = 3.4
    spec = _lines({f: 40.0, 2 * f: 20.0, 3 * f: 10.0})
    real = score_fault(spec, f, n_harmonics=3, tol_hz=0.03)
    sub = score_fault(spec, f / 3.0, n_harmonics=3, tol_hz=0.03)
    assert real.harmonic_energy > sub.harmonic_energy
    assert real.n_harmonics_found == 3
    est, _ = estimate_from_candidates(
        spec, np.array([f / 3.0, f, 2.0 * f]), n_harmonics=3, tol_hz=0.03
    )
    assert est == pytest.approx(f)


def test_harmonico_acima_de_nyquist_e_procurado_onde_o_mapa_de_alias_preve():
    """Spectrum that folds (envelope of uniform samples): the 3rd harmonic of 12.5 at
    ``fs = 64`` is at ``alias(37.5, 64) = 26.5``, and it is counted there."""
    f = 12.5
    folded = float(alias_frequency_hz(3 * f, 64.0))
    assert folded == pytest.approx(26.5)
    spec = _lines({f: 40.0, 2 * f: 20.0, folded: 10.0})
    assert score_fault(spec, f, n_harmonics=3, tol_hz=0.03).n_harmonics_found == 3
    assert (
        score_fault(spec, f, n_harmonics=3, tol_hz=0.03, fold_harmonics=False).n_harmonics_found
        == 2
    )


def test_espectro_sem_dobra_nao_tem_harmonico_dobrado_a_procurar():
    """Regression: after angular anti-aliasing, the order spectrum does not fold.

    Looking for the harmonic at the folded position would add whatever is there. With
    ``folds_above_nyquist=False`` the search stops at Nyquist, even with ``fold_harmonics``.
    """
    f = 12.5
    spec = _lines({f: 40.0, 2 * f: 20.0, 26.5: 10.0}, folds_above_nyquist=False)
    assert score_fault(spec, f, n_harmonics=3, tol_hz=0.03).n_harmonics_found == 2


def test_candidata_acima_de_nyquist_e_descartada_e_conjunto_vazio_e_erro():
    spec = _lines({3.4: 40.0})
    est, _ = estimate_from_candidates(spec, np.array([3.4, 40.0]), n_harmonics=3, tol_hz=0.03)
    assert est == pytest.approx(3.4)
    with pytest.raises(ValueError, match="Nyquist"):
        estimate_from_candidates(spec, np.array([40.0, 50.0]))
