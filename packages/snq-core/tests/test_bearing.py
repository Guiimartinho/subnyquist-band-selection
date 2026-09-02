"""Prior físico: as frequências de falha saem da geometria, não de um modelo."""

import numpy as np
import pytest
from snq_core import bearing as B


def test_ordem_das_frequencias_caracteristicas():
    """BPFI > BPFO > FTF sempre, e BPFO/FTF é o número de elementos."""
    f = B.fault_frequencies_hz(B.CWRU_6205_2RS_JEM_SKF, shaft_hz=29.95)
    assert f["BPFI"] > f["BPFO"] > f["FTF"] > 0.0
    assert f["BPFO"] / f["FTF"] == pytest.approx(B.CWRU_6205_2RS_JEM_SKF.n_elements)


def test_soma_de_bpfo_e_bpfi_e_proporcional_a_rotacao():
    """BPFO + BPFI = n * f_r, identidade exata que independe da geometria."""
    geom = B.CWRU_6205_2RS_JEM_SKF
    fr = 29.95
    f = B.fault_frequencies_hz(geom, fr)
    assert f["BPFO"] + f["BPFI"] == pytest.approx(geom.n_elements * fr)


def test_frequencias_escalam_linearmente_com_rotacao():
    """É essa proporcionalidade que permite desambiguar a ordem do alias
    variando o regime de rotação."""
    geom = B.CWRU_6205_2RS_JEM_SKF
    a = B.fault_frequencies_hz(geom, 20.0)
    b = B.fault_frequencies_hz(geom, 40.0)
    for k in a:
        assert b[k] == pytest.approx(2.0 * a[k])


def test_conjunto_de_candidatas_e_finito_e_ordenado():
    lines = B.candidate_lines_hz(B.CWRU_6205_2RS_JEM_SKF, shaft_hz=29.95)
    assert lines.ndim == 1
    assert lines.size > 0
    assert np.all(np.diff(lines) > 0)
    assert np.all(lines > 0)
