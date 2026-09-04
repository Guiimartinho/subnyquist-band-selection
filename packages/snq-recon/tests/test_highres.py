"""Alta resolução e análise de ordem."""

import numpy as np
import pytest
from snq_recon import (
    angular_resample,
    esprit,
    estimate_tone_esprit,
    estimate_tone_hz,
    fault_orders,
    order_spectrum,
)

FS = 1000.0


def _tone(f, n=512, amp=1.0, phase=0.0):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / FS + phase)


def test_esprit_supera_a_grade_da_fft():
    """Frequência entre bins: a FFT trava na grade, o ESPRIT não.

    Com N=512 e fs=1000 a resolução da FFT é 1,95 Hz. Uma raia em 123,4 Hz cai entre
    bins, e é exatamente onde métodos de subespaço ganham.
    """
    x = _tone(123.4)
    assert abs(estimate_tone_esprit(x, FS) - 123.4) < 0.05


def test_esprit_separa_duas_raias_proximas():
    """Duas senoides a menos de um bin de distância — a FFT não separa."""
    x = _tone(100.0) + _tone(101.0)
    f = esprit(x, n_sources=2, fs_hz=FS)
    assert f.size == 2
    assert abs(f[0] - 100.0) < 0.3
    assert abs(f[1] - 101.0) < 0.3


def test_esprit_com_subespaco_grande_demais_e_erro():
    """Teste negativo: subespaço maior que o sinal não tem sentido."""
    with pytest.raises(ValueError, match="subespaço"):
        esprit(_tone(100.0, n=64), n_sources=1, fs_hz=FS, subspace=64)


def test_ordens_de_falha_nao_dependem_do_regime():
    """É a razão de existir da análise de ordem: a ordem é constante da geometria."""
    o = fault_orders(geom_ratio=0.2034, n_elements=9)
    assert o["BPFO"] + o["BPFI"] == pytest.approx(9.0)
    assert o["BPFO"] < o["BPFI"]
    assert o["FTF"] == pytest.approx(o["BPFO"] / 9.0)


def test_reamostragem_angular_fixa_a_ordem_sob_rotacao_variavel():
    """O resultado que justifica o módulo.

    Um sinal cuja frequência acompanha a rotação espalha energia no espectro de
    frequência quando a rotação varia. No espectro de **ordem** ele volta a ser uma
    raia única.
    """
    n, fs = 20_000, 5_000.0
    t = np.arange(n) / fs
    # rotacao subindo 30% ao longo do registro
    rot = 20.0 * (1.0 + 0.3 * t / t[-1])
    fase = 2 * np.pi * np.cumsum(rot) / fs
    x = np.sin(3.5 * fase)  # componente na ordem 3.5

    ang, spr = angular_resample(x, fs, rot, samples_per_rev=64)
    ordens, mag = order_spectrum(ang, spr)
    pico = ordens[np.argmax(mag[1:]) + 1]
    assert abs(pico - 3.5) < 0.05, f"pico em {pico:.3f}, esperado 3.5"


def test_reamostragem_recusa_registro_curto_demais():
    """Teste negativo: menos de uma revolução não define ordem alguma."""
    with pytest.raises(ValueError, match="revolução"):
        angular_resample(np.zeros(100), 1000.0, shaft_hz=1.0)


def test_reamostragem_recusa_rotacao_nula():
    with pytest.raises(ValueError, match="positiva"):
        angular_resample(np.zeros(1000), 1000.0, shaft_hz=np.zeros(1000))


def test_esprit_e_fft_concordam_em_alto_snr():
    """Sanidade: com SNR alto e raia sobre um bin, os dois têm de dar o mesmo."""
    x = _tone(125.0)  # 125 Hz cai exatamente num bin com N=512, fs=1000
    assert abs(estimate_tone_esprit(x, FS) - estimate_tone_hz(x, FS)) < 0.5
