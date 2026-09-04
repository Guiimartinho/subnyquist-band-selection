"""Teste do envelope no domínio de ordem.

A propriedade que interessa é a que o E07 explora: sob rotação **variável**, a
frequência de falha em hertz se move e a ordem não. Um teste que use rotação
constante não distingue o código certo do errado, então aqui a rotação varre.
"""

from __future__ import annotations

import numpy as np
import pytest
from snq_recon import envelope_order_spectrum


def _impactos_modulados(fs_hz=50_000.0, seconds=2.0, order=5.43, f_res=8000.0):
    """Trem de impactos travado numa ordem, sob rotação que varre 15→30 Hz."""
    n = int(fs_hz * seconds)
    rot = np.linspace(15.0, 30.0, n)  # rotacao variavel: e o ponto do teste
    revs = np.cumsum(rot) / fs_hz

    # Um impacto a cada 1/order de revolucao, nao a cada intervalo fixo de tempo.
    fase = revs * order
    impacto = np.zeros(n)
    impacto[np.flatnonzero(np.diff(np.floor(fase)) > 0)] = 1.0

    # Ressonancia fixa em HERTZ - nao acompanha a rotacao. E o motivo de o envelope
    # ter de ser tomado em tempo, antes da reamostragem angular.
    decay = np.exp(-np.arange(int(0.002 * fs_hz)) * 900.0 / fs_hz)
    ring = decay * np.sin(2 * np.pi * f_res * np.arange(decay.size) / fs_hz)
    return np.convolve(impacto, ring)[:n], rot


def test_ordem_de_falha_aparece_apesar_da_rotacao_variar():
    x, rot = _impactos_modulados(order=5.43)
    spec = envelope_order_spectrum(x, 50_000.0, rot, (6000.0, 11000.0), samples_per_rev=64)
    sel = spec.freqs_hz >= 2.0
    pico = float(spec.freqs_hz[sel][np.argmax(spec.magnitude[sel])])
    assert pico == pytest.approx(5.43, abs=0.05)


def test_a_mesma_falha_em_hertz_se_espalha():
    """Controle: em hertz a raia borra porque a rotação varre 15→30 Hz.

    Sem este contraste o teste acima não prova que a reamostragem angular fez alguma
    coisa. A métrica é concentração em janela **relativa** ao redor do pico, e não
    max/média, porque os dois espectros têm número de raias muito diferente e
    max/média premia quem tem menos raias.
    """
    from snq_recon import envelope_spectrum

    x, rot = _impactos_modulados(order=5.43)
    ordem = envelope_order_spectrum(x, 50_000.0, rot, (6000.0, 11000.0), samples_per_rev=64)
    hertz = envelope_spectrum(x, 50_000.0, (6000.0, 11000.0))

    def concentracao(eixo, mag, lo):
        sel = eixo >= lo
        eixo, mag = eixo[sel], mag[sel]
        pico = float(eixo[np.argmax(mag)])
        perto = np.abs(eixo - pico) <= 0.02 * pico
        largo = (eixo >= 0.5 * pico) & (eixo <= 2.0 * pico)
        return float(mag[perto].sum() / mag[largo].sum())

    # Em hertz a falha ocupa 81-163 Hz (5,43 x 15 a 5,43 x 30); em ordens, uma raia.
    assert concentracao(ordem.freqs_hz, ordem.magnitude, 2.0) > 3.0 * concentracao(
        hertz.freqs_hz, hertz.magnitude, 20.0
    )


def test_unidades_do_objeto_devolvido_sao_ordens():
    x, rot = _impactos_modulados()
    spec = envelope_order_spectrum(x, 50_000.0, rot, (6000.0, 11000.0), samples_per_rev=64)
    # fs_hz carrega samples_per_rev e a maior "frequencia" e a ordem de Nyquist.
    assert spec.fs_hz == pytest.approx(64.0)
    assert spec.freqs_hz[-1] == pytest.approx(32.0, rel=0.01)
