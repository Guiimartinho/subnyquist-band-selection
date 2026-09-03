"""Ingestao de dataset e subamostragem artificial."""

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
    """O erro existe para impedir o modo de falha silencioso mais caro do projeto."""
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
    # identidade exata, independe da geometria
    assert truth["BPFO"] + truth["BPFI"] == pytest.approx(9 * 29.95)


def test_relatorio_de_subamostragem_detecta_fator_inutil():
    """Fator que dobra a banda sobre si mesma tem de ser rejeitado antes do split."""
    bom = undersampling_report(320e3, factor=8, band_hz=(140e3, 160e3))  # fs = 40 kSPS
    assert bom["fs_effective_hz"] == pytest.approx(40e3)
    assert bom["usable"] is True
    assert bom["inverted"] is True  # zona impar

    # Banda deslocada, sem relacao harmonica com fs_original: a borda superior
    # deixa de cair numa fronteira de zona e a dobra passa a se sobrepor.
    ruim = undersampling_report(320e3, factor=5, band_hz=(142e3, 163e3))  # fs = 64 kSPS
    assert ruim["usable"] is False


def test_fator_inteiro_restringe_as_taxas_alcancaveis():
    """Achado real: com fs_original fixa, nem toda banda tem fator inteiro valido.

    Para [142, 163] kHz a partir de 320 kSPS, nenhum fator inteiro maior que 1
    evita a sobreposicao. Em dataset publico a taxa de origem e dada, entao esta
    restricao e do experimento, nao do metodo - e precisa ser reportada.
    """
    band = (142e3, 163e3)
    usaveis = [f for f in range(2, 17) if undersampling_report(320e3, f, band)["usable"]]
    assert usaveis == []
