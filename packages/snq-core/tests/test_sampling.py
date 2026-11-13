"""Numerical verification of the project's reference results.

Project rule (skill `derive-dsp`): every symbolic derivation ends in a numerical
check, and every positive test has a negative test that checks the boundary of the
region of validity.
"""

import numpy as np
import pytest
from snq_core import sampling as S

# --------------------------------------------------------------------------
# Landau vs Nyquist
# --------------------------------------------------------------------------


def test_landau_muito_menor_que_nyquist_para_banda_estreita():
    band = [(140e3, 160e3)]
    assert S.landau_rate_hz(band) == pytest.approx(40e3)
    assert S.nyquist_rate_hz(band) == pytest.approx(320e3)
    # exactly 8x lower: this is the number quoted in the SOTA review
    assert S.nyquist_rate_hz(band) / S.landau_rate_hz(band) == pytest.approx(8.0)


def test_landau_iguala_nyquist_para_sinal_de_banda_base():
    """Boundary: for a band that starts at zero, Landau degenerates into Nyquist."""
    band = [(0.0, 20e3)]
    assert S.landau_rate_hz(band) == pytest.approx(S.nyquist_rate_hz(band))


# --------------------------------------------------------------------------
# Spectral folding, the project's canonical case
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("f_hz", "expected"),
    [(140e3, 20e3), (150e3, 10e3), (160e3, 0.0), (10e3, 10e3)],
)
def test_alias_frequency_caso_canonico(f_hz, expected):
    assert float(S.alias_frequency_hz(f_hz, 40e3)) == pytest.approx(expected, abs=1e-6)


def test_fold_band_zona_impar_inverte_e_nao_sobrepoe():
    fb = S.fold_band(140e3, 160e3, 40e3)
    assert fb.lo_hz == pytest.approx(0.0, abs=1e-6)
    assert fb.hi_hz == pytest.approx(20e3)
    assert fb.zone == 7
    assert fb.inverted is True
    assert fb.self_overlap is False


def test_fold_band_detecta_sobreposicao_quando_banda_cruza_zona():
    """Negative test: with the wrong fs the band folds onto itself and the
    information is irrecoverably destroyed."""
    fb = S.fold_band(140e3, 160e3, 50e3)
    assert fb.self_overlap is True


def test_alias_e_periodico_em_fs():
    f = np.array([150e3, 190e3, 230e3])
    a = S.alias_frequency_hz(f, 40e3)
    assert np.allclose(a, a[0])


# --------------------------------------------------------------------------
# Bandpass sampling (Vaughan, Scott & White, 1991)
# --------------------------------------------------------------------------


def test_menor_taxa_passa_faixa_do_caso_canonico():
    assert S.min_bandpass_rate_hz(140e3, 160e3) == pytest.approx(40e3)


def test_taxas_validas_cabem_no_conversor_do_nrf54l15():
    rates = S.valid_bandpass_rates_hz(140e3, 160e3, fs_max_hz=250e3)
    assert rates
    assert min(fs_min for fs_min, _, _ in rates) == pytest.approx(40e3)


def test_toda_taxa_valida_realmente_evita_sobreposicao():
    """Vaughan's condition has to agree with the computed fold."""
    for fs_min, _, _ in S.valid_bandpass_rates_hz(140e3, 160e3, fs_max_hz=250e3):
        assert S.fold_band(140e3, 160e3, fs_min).self_overlap is False


def test_taxa_invalida_produz_sobreposicao():
    """Negative test: 5% below the lowest valid rate already fails."""
    fs_bad = 0.95 * S.min_bandpass_rate_hz(140e3, 160e3)
    assert S.fold_band(140e3, 160e3, fs_bad).self_overlap is True


# --------------------------------------------------------------------------
# Multi-coset: the phase carries the true frequency
# --------------------------------------------------------------------------


def test_fase_relativa_codifica_a_ordem_do_alias():
    """Central result of the project.

    A complex exponential sampled with an offset delta applied before sampling
    produces a factor exp(-j2*pi*f0*delta). The observed frequency is the aliased
    one, but the complex amplitude carries the true f0.
    """
    fs = 40e3
    ts = 1.0 / fs
    n = np.arange(64)
    f_alias = 10e3
    delta = ts / 4.0
    for order in (0, 1, 2, 3):
        f0 = f_alias + order * fs
        x0 = np.exp(2j * np.pi * f0 * (n * ts))
        x1 = np.exp(2j * np.pi * f0 * (n * ts - delta))
        assert float(S.alias_frequency_hz(f0, fs)) == pytest.approx(f_alias)
        ratio = np.mean(x1 / x0)
        assert ratio == pytest.approx(np.exp(-2j * np.pi * f0 * delta), abs=1e-9)


def test_coset_uniforme_tem_condicionamento_otimo():
    """Offsets m*Ts/M give a DFT: condition number equal to 1."""
    fs = 40e3
    m = 4
    offsets = S.uniform_coset_offsets_s(m, fs)
    a = S.coset_matrix(offsets, fs, orders=np.arange(m))
    assert np.linalg.cond(a) == pytest.approx(1.0, rel=1e-9)


def test_coset_quase_degenerado_e_mal_condicionado():
    """Negative test: clustered offsets destroy the conditioning."""
    fs = 40e3
    m = 4
    ts = 1.0 / fs
    offsets = np.array([0.0, 1e-3, 2e-3, 3e-3]) * ts
    a = S.coset_matrix(offsets, fs, orders=np.arange(m))
    assert np.linalg.cond(a) > 1e6


def test_dois_cosets_resolvem_apenas_paridade():
    """delta = Ts/2 distinguishes only k mod 2, i.e. a bandwidth extension of 2x."""
    fs = 40e3
    ts = 1.0 / fs
    delta = ts / 2.0
    phases = {k: np.exp(-2j * np.pi * (k * fs) * delta) for k in range(4)}
    assert phases[0] == pytest.approx(phases[2])
    assert phases[1] == pytest.approx(phases[3])
    assert abs(phases[0] - phases[1]) > 1.0
