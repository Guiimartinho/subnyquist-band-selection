"""CWRU screening according to Smith & Randall.

Does not download data: it tests only the catalog logic. The end-to-end test with a real
signal lives in E02, which needs the `.mat` files in `datasets/raw/`.
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
    with pytest.raises(ValueError, match="invalid"):
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
    """Conservative on purpose: better to leave out than to classify wrongly.

    The fan-end row of Table 5 came out of the PDF with an ambiguous column assignment.
    While `human_checked` is false, these records come back as `other` instead
    of receiving a label that may be wrong.
    """
    assert cat.human_checked is False
    assert "fan_end_12k" not in cat.reliable_subsets
    assert cat.screen(282, "FE") == "other"
    # ... but the label exists in the file, and comes out when the requirement is waived
    assert cat.screen(282, "FE", require_reliable=False) == "undiagnosable"


def test_ponto_de_medicao_discrimina(cat):
    """120DE is undiagnosable; record 120 at another point is not listed."""
    assert cat.screen(120, "DE") == "undiagnosable"
    assert cat.screen(120, "FE") == "other"


def test_registros_ceifados_sao_sinalizados(cat):
    """Clipping is a nonlinearity and contaminates any spectral analysis."""
    assert cat.is_clipped(214, "DE") is True
    assert cat.is_clipped(130, "DE") is False


def test_as_duas_tabelas_nao_se_sobrepoem(cat):
    """A record cannot be both classical and undiagnosable."""
    classicos = set(cat.records("classical"))
    ruins = set(cat.records("undiagnosable"))
    assert not (classicos & ruins), sorted(classicos & ruins)


def test_catalogo_tem_volume_esperado(cat):
    assert len(cat.records("undiagnosable")) == 41
    assert len(cat.records("classical")) == 73
