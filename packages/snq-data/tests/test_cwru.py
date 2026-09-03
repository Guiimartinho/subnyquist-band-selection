"""Triagem do CWRU segundo Smith & Randall.

Não baixa dados: testa só a lógica de catálogo. O teste de ponta a ponta com sinal
real vive no E02, que precisa dos `.mat` em `datasets/raw/`.
"""

import pytest
from snq_data import CwruCatalog, parse_record_id

CATALOG = "datasets/catalogs/cwru-smith2015-categories.yaml"


@pytest.fixture(scope="module")
def cat() -> CwruCatalog:
    return CwruCatalog.load(CATALOG)


def test_identificador_com_e_sem_ponto_de_medicao():
    assert parse_record_id("120DE") == (120, "DE")
    assert parse_record_id("130") == (130, None)
    assert parse_record_id(3001) == (3001, None)
    with pytest.raises(ValueError, match="inválido"):
        parse_record_id("120XY")


def test_registros_classicos_e_nao_diagnosticaveis(cat):
    assert cat.screen(130, "DE") == "classical"
    assert cat.screen(209, "DE") == "classical"
    assert cat.screen(118, "DE") == "undiagnosable"
    assert cat.screen(3001) == "undiagnosable"
    assert cat.screen(174, "DE") == "undiagnosable"


def test_registro_desconhecido_nao_e_classificado(cat):
    assert cat.screen(999, "DE") == "other"


def test_subconjunto_nao_conferido_e_tratado_como_desconhecido(cat):
    """Conservador de propósito: melhor deixar de fora que classificar errado.

    A linha de fan end da Tabela 5 saiu do PDF com atribuição de coluna ambígua.
    Enquanto `human_checked` for falso, esses registros voltam como `other` em vez
    de receberem um rótulo que pode estar errado.
    """
    assert cat.human_checked is False
    assert "fan_end_12k" not in cat.reliable_subsets
    assert cat.screen(282, "FE") == "other"
    # ... mas o rótulo existe no arquivo, e sai quando se dispensa a exigência
    assert cat.screen(282, "FE", require_reliable=False) == "undiagnosable"


def test_ponto_de_medicao_discrimina(cat):
    """120DE é não diagnosticável; o registro 120 em outro ponto não está listado."""
    assert cat.screen(120, "DE") == "undiagnosable"
    assert cat.screen(120, "FE") == "other"


def test_registros_ceifados_sao_sinalizados(cat):
    """Ceifamento é não-linearidade e contamina qualquer análise espectral."""
    assert cat.is_clipped(214, "DE") is True
    assert cat.is_clipped(130, "DE") is False


def test_as_duas_tabelas_nao_se_sobrepoem(cat):
    """Um registro não pode ser simultaneamente clássico e não diagnosticável."""
    classicos = set(cat.records("classical"))
    ruins = set(cat.records("undiagnosable"))
    assert not (classicos & ruins), sorted(classicos & ruins)


def test_catalogo_tem_volume_esperado(cat):
    assert len(cat.records("undiagnosable")) == 41
    assert len(cat.records("classical")) == 73
