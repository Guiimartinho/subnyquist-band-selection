"""A auditoria que separa recuperação de alucinação."""

import numpy as np
import pytest
from snq_eval import attribute, hf_recovery_ratio


def test_operador_de_posto_completo_atribui_tudo_a_medicao():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8))
    x = rng.normal(size=8)
    y = a @ x
    r = attribute(a, x, y)
    assert r.measured_fraction == pytest.approx(1.0)
    assert r.data_consistency < 1e-10
    assert r.rank == 8


def test_operador_deficiente_expoe_a_componente_inventada():
    """Metade das direções não é observável: o que estiver lá veio do prior."""
    a = np.zeros((4, 8))
    a[np.arange(4), np.arange(4)] = 1.0  # observa só as 4 primeiras coordenadas
    x_hat = np.ones(8)
    y = a @ x_hat
    r = attribute(a, x_hat, y)
    assert r.rank == 4
    assert r.measured_fraction == pytest.approx(0.5)
    assert r.null_fraction == pytest.approx(0.5)


def test_prior_que_preenche_o_nucleo_nao_muda_a_consistencia_de_dados():
    """O ponto central: inventar no núcleo é invisível às observações.

    Duas estimativas radicalmente diferentes reproduzem os mesmos dados. Só a
    atribuição distingue as duas — nenhuma métrica de consistência consegue.
    """
    a = np.zeros((4, 8))
    a[np.arange(4), np.arange(4)] = 1.0
    y = np.ones(4)

    honesta = np.concatenate([np.ones(4), np.zeros(4)])
    alucinada = np.concatenate([np.ones(4), 10.0 * np.ones(4)])

    r_h = attribute(a, honesta, y)
    r_a = attribute(a, alucinada, y)

    assert r_h.data_consistency == pytest.approx(r_a.data_consistency, abs=1e-12)
    assert r_h.measured_fraction == pytest.approx(1.0)
    assert r_a.measured_fraction < 0.02


def test_valores_singulares_minusculos_contam_como_nucleo():
    """Direção formalmente medida, mas com amplificação de ruído inútil sob 12 bits."""
    a = np.diag([1.0, 1.0, 1e-14, 1e-14])
    x = np.ones(4)
    y = a @ x
    r = attribute(a, x, y, rcond=1e-10)
    assert r.rank == 2
    assert r.measured_fraction == pytest.approx(0.5)


def test_consistencia_de_dados_detecta_reconstrucao_que_nao_reproduz_o_medido():
    a = np.eye(4)
    y = np.ones(4)
    r = attribute(a, np.zeros(4), y)
    assert r.data_consistency == pytest.approx(1.0)


def test_hf_recovery_ratio_reconhece_recuperacao_perfeita():
    fs = 1000.0
    t = np.arange(1024) / fs
    x = np.sin(2 * np.pi * 300 * t)
    assert hf_recovery_ratio(x, x, fs, f_cut_hz=200.0) == pytest.approx(1.0)


def test_hf_recovery_ratio_nao_distingue_energia_inventada():
    """Por isso a métrica espectral nunca vai sozinha no paper.

    Um sinal na frequência errada, mas dentro da banda, marca 1.0 — e está errado.
    """
    fs = 1000.0
    t = np.arange(1024) / fs
    verdadeiro = np.sin(2 * np.pi * 300 * t)
    inventado = np.sin(2 * np.pi * 400 * t)
    ratio = hf_recovery_ratio(inventado, verdadeiro, fs, f_cut_hz=200.0)
    assert ratio == pytest.approx(1.0, rel=0.05)
