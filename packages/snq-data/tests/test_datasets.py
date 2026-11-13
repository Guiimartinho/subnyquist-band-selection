"""Dataset ingestion and artificial undersampling."""

import numpy as np
import pytest
from snq_data import (
    AntiAliasFilterUsedError,
    Manifest,
    load_all_manifests,
    undersample,
    undersample_cosets,
    undersampling_report,
)

MANIFESTS = "datasets/manifests"


def test_todos_os_manifests_carregam():
    ms = load_all_manifests(MANIFESTS)
    assert set(ms) >= {"cwru", "paderborn", "mfpt", "xjtu-sy", "mafaulda"}
    for m in ms.values():
        assert m.fs_hz and all(f > 0 for f in m.fs_hz)
        assert m.license


def test_decimacao_pura():
    assert undersample(np.arange(10.0), 3).tolist() == [0.0, 3.0, 6.0, 9.0]
    assert undersample(np.arange(10.0), 3, offset=1).tolist() == [1.0, 4.0, 7.0]


def test_decimar_com_aaf_e_erro_explicito():
    """The error exists to prevent the project's most expensive silent failure mode."""
    with pytest.raises(AntiAliasFilterUsedError, match="super-Nyquist"):
        undersample(np.arange(10.0), 2, anti_alias=True)


def test_cosets_cobrem_o_sinal_sem_repetir():
    x = np.arange(12.0)
    cs = undersample_cosets(x, 4, offsets=[0, 1, 2, 3])
    assert len(cs) == 4
    assert sorted(np.concatenate(cs).tolist()) == x.tolist()


def test_geometria_e_verdade_fundamental_do_cwru():
    m = Manifest.load(f"{MANIFESTS}/cwru.yaml")
    g = m.geometry("drive_end")
    assert g.n_elements == 9
    truth = m.fault_truth_hz(shaft_hz=29.95)
    # exact identity, independent of the geometry
    assert truth["BPFO"] + truth["BPFI"] == pytest.approx(9 * 29.95)


def test_relatorio_de_subamostragem_detecta_fator_inutil():
    """A factor that folds the band onto itself has to be rejected before the split."""
    bom = undersampling_report(320e3, factor=8, band_hz=(140e3, 160e3))  # fs = 40 kSPS
    assert bom["fs_effective_hz"] == pytest.approx(40e3)
    assert bom["usable"] is True
    assert bom["inverted"] is True  # odd zone

    # Shifted band, with no harmonic relation to fs_original: the upper edge no longer
    # falls on a zone boundary and the fold starts to overlap itself.
    ruim = undersampling_report(320e3, factor=5, band_hz=(142e3, 163e3))  # fs = 64 kSPS
    assert ruim["usable"] is False


def test_fator_inteiro_restringe_as_taxas_alcancaveis():
    """Real finding: with a fixed fs_original, not every band has a valid integer factor.

    For [142, 163] kHz from 320 kSPS, no integer factor greater than 1
    avoids the overlap. In a public dataset the source rate is given, so this
    restriction belongs to the experiment, not to the method - and it must be reported.
    """
    band = (142e3, 163e3)
    usaveis = [f for f in range(2, 17) if undersampling_report(320e3, f, band)["usable"]]
    assert usaveis == []


# ----------------------------------------------------------------------
# plain decimation: grid, effective rate, alias phase and zone inversion
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("f0_hz", "inverted"),
    [(1100.0, False), (4100.0, True), (7100.0, False), (10100.0, True)],
)
def test_fase_do_alias_carrega_f0_verdadeiro_e_inverte_em_zona_impar(f0_hz, inverted):
    """``undersample(x, M, offset=o)`` samples at ``t = (o + kM)/fs``.

    For a real tone, the line appears at ``alias(f0, fs/M)`` with phase ``theta_o``, or
    ``-theta_o`` if the zone is odd (inverted spectrum), where
    ``theta_o = 2*pi*f0*o/fs + phi``. The difference between offsets 1 and 0 is
    ``+-2*pi*f0/fs``: it carries f0 modulo fs, and not only the alias modulo fs/M.
    """
    from snq_core import alias_frequency_hz, fold_band
    from snq_data import effective_rate_hz

    fs, factor, phi = 48_000.0, 8, 0.7
    fs_eff = effective_rate_hz(fs, factor)
    assert fs_eff == pytest.approx(6000.0)
    assert fold_band(f0_hz - 10.0, f0_hz + 10.0, fs_eff).inverted is inverted

    n = np.arange(600 * factor)
    x = np.cos(2 * np.pi * f0_hz * n / fs + phi)
    y0, y1 = undersample(x, factor, offset=0), undersample(x, factor, offset=1)
    assert np.allclose(y1, np.cos(2 * np.pi * f0_hz * (1 + factor * np.arange(y1.size)) / fs + phi))

    alias = float(alias_frequency_hz(f0_hz, fs_eff))
    k = int(round(alias / (fs_eff / y0.size)))
    s0, s1 = np.fft.rfft(y0)[k], np.fft.rfft(y1)[k]
    sign = -1.0 if inverted else 1.0
    assert np.angle(s0) == pytest.approx(sign * phi, abs=1e-9)
    expected = np.exp(1j * sign * 2 * np.pi * f0_hz / fs)
    assert s1 / s0 == pytest.approx(expected, abs=1e-9)


def test_fator_e_offset_invalidos_sao_erro():
    with pytest.raises(ValueError, match="factor"):
        undersample(np.arange(10.0), 0)
    with pytest.raises(ValueError, match="offset"):
        undersample(np.arange(10.0), 3, offset=3)
