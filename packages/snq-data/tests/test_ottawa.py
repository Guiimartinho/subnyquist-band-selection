"""Tests of the Ottawa 2018 loader — no network.

What has to be guaranteed here is the extraction of the rotation profile from the
encoder, because it is what all the order analysis consumes and it is the only point where a
silent error would scale all orders by a constant factor.
"""

from __future__ import annotations

import numpy as np
import pytest
from snq_data.ottawa import FAULT_ORDERS, FS_HZ, PULSES_PER_REV, OttawaRecord, record_names


def _synthetic(rot_hz: float = 20.0, seconds: float = 1.0, ppr: int = PULSES_PER_REV):
    """Ideal encoder at constant rotation, and an arbitrary vibration."""
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
    """At 20 Hz and 1024 ppr the pulses come out at 20.5 kHz, against 200 kSPS of sampling.

    That leaves ~9.8 samples per pulse, so the instant of each edge is quantized on the
    sampling clock and the per-edge rotation is off by up to ~5%. The median is exact to
    within 2.3%, and that bias is the quantization itself — not a code error. On the real node
    the TIMER captures edges at 128 MHz and the effect disappears; here it is recorded because
    it sets the resolution floor of any order analysis done outside the MCU.
    """
    rec = _synthetic(rot_hz=20.0)
    speed = rec.speed_profile_hz()
    assert float(np.median(speed)) == pytest.approx(20.0, rel=0.05)
    assert float(np.std(speed[1000:-1000])) / 20.0 < 0.10


def test_ppr_errado_escala_todas_as_ordens():
    """Half the pulses per revolution doubles the estimated rotation.

    It is the failure mode that would make a fault order appear in the wrong place without
    anything else signalling an error — hence the value of 1024 being recorded as
    determined empirically, and not assumed.
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
    with pytest.raises(ValueError, match="pulses"):
        mudo.speed_profile_hz()


def test_ordens_de_falha_somam_o_numero_de_elementos():
    """Geometric identity BPFO + BPFI = n. It holds for the NOMINAL prior.

    E07 measures 8.82 instead of 9 on the real data; the difference is the catalog deviation
    that experiment quantifies, not an error in this constant.
    """
    total = FAULT_ORDERS["inner_race"] + FAULT_ORDERS["outer_race"]
    assert total == pytest.approx(9.0)


def test_nomes_de_registro_sao_deterministicos_e_completos():
    nomes = record_names()
    assert len(nomes) == 36
    assert nomes == sorted(nomes, key=lambda s: ("HIO".index(s[0]), s))
    assert "I-A-1.mat" in nomes and "O-D-3.mat" in nomes
