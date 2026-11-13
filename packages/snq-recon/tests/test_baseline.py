"""Official baseline: the kurtogram picks the band, the envelope finds the fault.

The test signal is the canonical bearing fault model — a train of damped impacts
exciting a structural resonance, with ~1% slip, which is what
makes the signal cyclostationary rather than periodic.
"""

import numpy as np
import pytest
from snq_core import BearingGeometry, fault_frequencies_hz
from snq_recon import (
    diagnose,
    envelope_headroom_ok,
    envelope_spectrum,
    estimate_from_candidates,
    estimate_without_prior,
    fast_kurtogram,
    spectral_kurtosis,
)
from snq_sim import impulse_train

FS = 48_000.0
RESONANCE = 6_000.0
BPFO = 104.0
DURATION = 2.0


@pytest.fixture(scope="module")
def bearing_signal() -> np.ndarray:
    """Impacts at 104 Hz exciting a resonance at 6 kHz, plus noise at ~0 dB."""
    x = impulse_train(DURATION, FS, rate_hz=BPFO, resonance_hz=RESONANCE, damping=0.02, seed=7)
    rng = np.random.default_rng(7)
    return x + rng.normal(0.0, float(np.std(x)), size=x.size)


# ----------------------------------------------------------------------
# spectral kurtosis
# ----------------------------------------------------------------------
def test_curtose_separa_impulsivo_de_gaussiano():
    """Complex Gaussian noise gives ~0; a train of rare impulses gives much more."""
    rng = np.random.default_rng(0)
    ruido = rng.normal(size=8192) + 1j * rng.normal(size=8192)
    impulsivo = np.zeros(8192, dtype=complex)
    impulsivo[::400] = 10.0

    assert abs(spectral_kurtosis(ruido)) < 0.2
    assert spectral_kurtosis(impulsivo) > 10.0


def test_curtose_de_sinal_nulo_nao_explode():
    """Negative test: division by zero has to be handled, not propagated."""
    assert spectral_kurtosis(np.zeros(128, dtype=complex)) == 0.0


# ----------------------------------------------------------------------
# kurtogram
# ----------------------------------------------------------------------
def test_kurtograma_encontra_a_banda_ressonante(bearing_signal):
    """The chosen band has to contain the resonance that generated the impacts."""
    k = fast_kurtogram(bearing_signal, FS, max_level=5)
    assert k.f_lo_hz <= RESONANCE <= k.f_hi_hz, k.summary()
    assert k.kurtosis > 0.5


def test_kurtograma_em_ruido_puro_nao_produz_curtose_alta():
    """Negative test: without a transient, no band should stand out."""
    rng = np.random.default_rng(1)
    k = fast_kurtogram(rng.normal(size=int(FS)), FS, max_level=4)
    assert k.kurtosis < 2.0


def test_kurtograma_respeita_a_faixa_de_busca(bearing_signal):
    k = fast_kurtogram(bearing_signal, FS, max_level=4, f_min_hz=1e3, f_max_hz=10e3)
    assert k.f_lo_hz >= 1e3
    assert k.f_hi_hz <= 10e3


# ----------------------------------------------------------------------
# envelope
# ----------------------------------------------------------------------
def test_envelope_recupera_a_frequencia_de_impacto(bearing_signal):
    """The whole baseline: kurtogram -> band -> envelope -> BPFO."""
    band = fast_kurtogram(bearing_signal, FS, max_level=5).band_hz
    spec = envelope_spectrum(bearing_signal, FS, band)
    f_est = estimate_without_prior(spec, f_min_hz=20.0)
    assert abs(f_est - BPFO) < 2.0, f"estimated {f_est:.1f} Hz, expected {BPFO}"


def test_envelope_em_banda_errada_nao_encontra_a_falha(bearing_signal):
    """Negative test: a band far from the resonance loses the signal."""
    spec = envelope_spectrum(bearing_signal, FS, (20e3, 23e3))
    f_est = estimate_without_prior(spec, f_min_hz=20.0)
    assert abs(f_est - BPFO) > 2.0


def test_folga_de_envelope():
    """Project rule: fs >= 2.5*B, because |y| doubles the bandwidth."""
    assert envelope_headroom_ok((140e3, 160e3), fs_hz=40e3) is False
    assert envelope_headroom_ok((140e3, 160e3), fs_hz=60e3) is True


# ----------------------------------------------------------------------
# diagnosis against the geometric prior
# ----------------------------------------------------------------------
def test_diagnostico_escolhe_a_falha_certa():
    """Impacts generated at the BPFO of the geometry have to be diagnosed as BPFO."""
    geom = BearingGeometry(n_elements=9, d_ball_m=7.94e-3, d_pitch_m=39.04e-3, name="test")
    shaft = 29.0
    bpfo = fault_frequencies_hz(geom, shaft)["BPFO"]

    x = impulse_train(DURATION, FS, rate_hz=bpfo, resonance_hz=RESONANCE, damping=0.02, seed=3)
    rng = np.random.default_rng(3)
    x = x + rng.normal(0.0, 0.5 * float(np.std(x)), size=x.size)

    band = fast_kurtogram(x, FS, max_level=5).band_hz
    d = diagnose(envelope_spectrum(x, FS, band), geom, shaft)

    assert d.best.fault == "BPFO", "\n" + d.summary()
    assert d.error_hz("BPFO") < 2.0


def test_prior_restringe_a_um_conjunto_discreto(bearing_signal):
    """With the prior, the estimate is forced to land on an admissible candidate."""
    band = fast_kurtogram(bearing_signal, FS, max_level=5).band_hz
    spec = envelope_spectrum(bearing_signal, FS, band)
    candidatas = np.array([61.0, BPFO, 157.0, 203.0])
    f_est, energia = estimate_from_candidates(spec, candidatas)
    assert f_est == pytest.approx(BPFO)
    assert energia > 0.0
