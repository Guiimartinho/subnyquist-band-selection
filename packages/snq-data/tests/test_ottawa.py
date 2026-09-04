"""Testes do loader de Ottawa 2018 — sem rede.

O que precisa ser garantido aqui é a extração do perfil de rotação a partir do
encoder, porque é ela que toda a análise de ordem consome e é o único ponto onde um
erro silencioso escalaria todas as ordens por um fator constante.
"""

from __future__ import annotations

import numpy as np
import pytest
from snq_data.ottawa import FAULT_ORDERS, FS_HZ, PULSES_PER_REV, OttawaRecord, record_names


def _synthetic(rot_hz: float = 20.0, seconds: float = 1.0, ppr: int = PULSES_PER_REV):
    """Encoder ideal a rotação constante, e vibração qualquer."""
    n = int(FS_HZ * seconds)
    t = np.arange(n) / FS_HZ
    encoder = np.sign(np.sin(2 * np.pi * rot_hz * ppr * t))
    return OttawaRecord(
        name="X-A-1",
        condition="inner_race",
        scenario="A",
        trial=1,
        vibration=np.sin(2 * np.pi * 5000.0 * t),
        encoder=encoder,
    )


def test_perfil_de_rotacao_recupera_a_velocidade_constante():
    """A 20 Hz e 1024 ppr os pulsos saem a 20,5 kHz, contra 200 kSPS de amostragem.

    Sobram ~9,8 amostras por pulso, então o instante de cada borda é quantizado no
    relógio de amostragem e a rotação por borda erra até ~5%. A mediana é exata a
    2,3%, e esse viés é a própria quantização — não um erro de código. No nó real o
    TIMER captura bordas a 128 MHz e o efeito some; aqui ele fica registrado porque
    define o piso de resolução de qualquer análise de ordem feita fora do MCU.
    """
    rec = _synthetic(rot_hz=20.0)
    speed = rec.speed_profile_hz()
    assert float(np.median(speed)) == pytest.approx(20.0, rel=0.05)
    assert float(np.std(speed[1000:-1000])) / 20.0 < 0.10


def test_ppr_errado_escala_todas_as_ordens():
    """Metade dos pulsos por revolução dobra a rotação estimada.

    É o modo de falha que faria uma ordem de falha aparecer no lugar errado sem que
    nada mais desse sinal de erro — daí o valor de 1024 estar registrado como
    determinado empiricamente, e não suposto.
    """
    rec = _synthetic(rot_hz=20.0)
    assert np.median(rec.speed_profile_hz(PULSES_PER_REV // 2)) == pytest.approx(40.0, rel=0.05)


def test_encoder_sem_pulsos_recusa_em_vez_de_inventar():
    rec = _synthetic()
    mudo = OttawaRecord(
        name=rec.name,
        condition=rec.condition,
        scenario=rec.scenario,
        trial=rec.trial,
        vibration=rec.vibration,
        encoder=np.zeros_like(rec.encoder),
    )
    with pytest.raises(ValueError, match="pulsos"):
        mudo.speed_profile_hz()


def test_ordens_de_falha_somam_o_numero_de_elementos():
    """Identidade geométrica BPFO + BPFI = n. Vale para o prior NOMINAL.

    O E07 mede 8,82 em vez de 9 nos dados reais; a diferença é o desvio do catálogo
    que aquele experimento quantifica, não um erro desta constante.
    """
    total = FAULT_ORDERS["inner_race"] + FAULT_ORDERS["outer_race"]
    assert total == pytest.approx(9.0)


def test_nomes_de_registro_sao_deterministicos_e_completos():
    nomes = record_names()
    assert len(nomes) == 36
    assert nomes == sorted(nomes, key=lambda s: ("HIO".index(s[0]), s))
    assert "I-A-1.mat" in nomes and "O-D-3.mat" in nomes
