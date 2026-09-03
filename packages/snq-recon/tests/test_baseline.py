"""Baseline oficial: kurtograma escolhe a banda, envelope encontra a falha.

O sinal de teste é o modelo canônico de falha de rolamento — trem de impactos
amortecidos excitando uma ressonância estrutural, com escorregamento de ~1%, que é o
que torna o sinal ciclo-estacionário em vez de periódico.
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
    """Impactos a 104 Hz excitando ressonância em 6 kHz, mais ruído a ~0 dB."""
    x = impulse_train(DURATION, FS, rate_hz=BPFO, resonance_hz=RESONANCE, damping=0.02, seed=7)
    rng = np.random.default_rng(7)
    return x + rng.normal(0.0, float(np.std(x)), size=x.size)


# ----------------------------------------------------------------------
# curtose espectral
# ----------------------------------------------------------------------
def test_curtose_separa_impulsivo_de_gaussiano():
    """Ruído gaussiano complexo dá ~0; trem de impulsos raros dá muito mais."""
    rng = np.random.default_rng(0)
    ruido = rng.normal(size=8192) + 1j * rng.normal(size=8192)
    impulsivo = np.zeros(8192, dtype=complex)
    impulsivo[::400] = 10.0

    assert abs(spectral_kurtosis(ruido)) < 0.2
    assert spectral_kurtosis(impulsivo) > 10.0


def test_curtose_de_sinal_nulo_nao_explode():
    """Teste negativo: divisão por zero tem de ser tratada, não propagada."""
    assert spectral_kurtosis(np.zeros(128, dtype=complex)) == 0.0


# ----------------------------------------------------------------------
# kurtograma
# ----------------------------------------------------------------------
def test_kurtograma_encontra_a_banda_ressonante(bearing_signal):
    """A banda escolhida tem de conter a ressonância que gerou os impactos."""
    k = fast_kurtogram(bearing_signal, FS, max_level=5)
    assert k.f_lo_hz <= RESONANCE <= k.f_hi_hz, k.summary()
    assert k.kurtosis > 0.5


def test_kurtograma_em_ruido_puro_nao_produz_curtose_alta():
    """Teste negativo: sem transiente, nenhuma banda deve se destacar."""
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
    """O baseline inteiro: kurtograma -> banda -> envelope -> BPFO."""
    band = fast_kurtogram(bearing_signal, FS, max_level=5).band_hz
    spec = envelope_spectrum(bearing_signal, FS, band)
    f_est = estimate_without_prior(spec, f_min_hz=20.0)
    assert abs(f_est - BPFO) < 2.0, f"estimou {f_est:.1f} Hz, esperado {BPFO}"


def test_envelope_em_banda_errada_nao_encontra_a_falha(bearing_signal):
    """Teste negativo: banda longe da ressonância perde o sinal."""
    spec = envelope_spectrum(bearing_signal, FS, (20e3, 23e3))
    f_est = estimate_without_prior(spec, f_min_hz=20.0)
    assert abs(f_est - BPFO) > 2.0


def test_folga_de_envelope():
    """Regra do projeto: fs >= 2.5*B, porque |y| dobra a largura de banda."""
    assert envelope_headroom_ok((140e3, 160e3), fs_hz=40e3) is False
    assert envelope_headroom_ok((140e3, 160e3), fs_hz=60e3) is True


# ----------------------------------------------------------------------
# diagnóstico contra o prior geométrico
# ----------------------------------------------------------------------
def test_diagnostico_escolhe_a_falha_certa():
    """Impactos gerados na BPFO da geometria têm de ser diagnosticados como BPFO."""
    geom = BearingGeometry(n_elements=9, d_ball_m=7.94e-3, d_pitch_m=39.04e-3, name="teste")
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
    """Com prior, a estimativa é forçada a cair numa candidata admissível."""
    band = fast_kurtogram(bearing_signal, FS, max_level=5).band_hz
    spec = envelope_spectrum(bearing_signal, FS, band)
    candidatas = np.array([61.0, BPFO, 157.0, 203.0])
    f_est, energia = estimate_from_candidates(spec, candidatas)
    assert f_est == pytest.approx(BPFO)
    assert energia > 0.0
