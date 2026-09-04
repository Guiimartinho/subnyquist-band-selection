"""Codificação temporal — a realização A2 do ADR 0004.

O que se testa aqui é a propriedade que justifica a rota: **energia proporcional à
atividade do sinal, não ao tempo decorrido**.
"""

import numpy as np
import pytest
from snq_sim import IntegrateFireTEM, crossing_times, multichannel_tem

FS = 200_000.0
DUR = 0.05


def _tone(f_hz: float, amp: float = 1.0, n: int | None = None) -> np.ndarray:
    t = np.arange(n or int(DUR * FS)) / FS
    return amp * np.sin(2.0 * np.pi * f_hz * t)


# ----------------------------------------------------------------------
# integrate-and-fire
# ----------------------------------------------------------------------
def test_disparos_sao_irregulares_mas_monotonos():
    t = IntegrateFireTEM().encode(_tone(500.0), FS)
    assert t.size > 10
    assert np.all(np.diff(t) > 0), "instantes têm de ser estritamente crescentes"
    # irregular de verdade: os intervalos variam com o sinal
    intervalos = np.diff(t)
    assert intervalos.std() / intervalos.mean() > 0.01


def test_bias_insuficiente_e_erro_explicito():
    """Se a amplitude excede o viés, o integrador pode nunca disparar."""
    tem = IntegrateFireTEM(bias=0.5)
    with pytest.raises(ValueError, match="bias"):
        tem.encode(_tone(500.0, amp=1.0), FS)


def test_taxa_nominal_bate_com_o_parametro():
    """A taxa de projeto é `bias / (2*delta*kappa)`; `for_rate` inverte a relação."""
    tem = IntegrateFireTEM.for_rate(50e3)
    assert tem.nominal_rate_hz() == pytest.approx(50e3)
    # e a taxa medida com entrada de média zero fica perto da nominal
    assert tem.firing_rate_hz(_tone(500.0, amp=0.5), FS) == pytest.approx(50e3, rel=0.05)


def test_taxa_do_iftem_nao_depende_da_amplitude():
    """Correção de uma suposição minha, medida aqui.

    Para entrada de média zero, a integral de ``x + bias`` num período vale
    ``bias * T`` **independentemente da amplitude** — a amplitude se cancela. Logo o
    IF-TEM **não** gasta menos quando a máquina está parada.

    Isso desfaz metade do argumento de energia que o ADR 0004 atribuía a esta rota:
    o ganho do IF-TEM é dispensar o relógio global, não disparar menos.
    """
    tem = IntegrateFireTEM(bias=2.0)
    taxas = [tem.firing_rate_hz(_tone(500.0, amp=a), FS) for a in (0.01, 0.5, 1.9)]
    assert taxas[0] == pytest.approx(taxas[1]) == pytest.approx(taxas[2])


def test_taxa_do_iftem_depende_da_media():
    """O que de fato governa a taxa: ``(media(x) + bias) / (2*delta*kappa)``."""
    tem = IntegrateFireTEM(bias=2.0)
    for dc in (-0.5, 0.0, 0.5):
        x = _tone(500.0, amp=0.3) + dc
        previsto = (dc + tem.bias) / (2.0 * tem.delta * tem.kappa)
        assert tem.firing_rate_hz(x, FS) == pytest.approx(previsto, rel=0.01)


def test_cruzamento_de_nivel_e_que_e_proporcional_a_atividade():
    """O argumento de energia da rota A2 pertence a esta variante, não ao IF-TEM.

    Sinal mais ativo atravessa mais níveis e gera mais eventos; sinal parado não
    gera nenhum. É aqui que "o que não acontece não custa" vale de verdade.
    """
    niveis, h = [-0.5, 0.0, 0.5], 0.005
    fraco = crossing_times(_tone(500.0, amp=0.01), FS, niveis, hysteresis=h).size
    forte = crossing_times(_tone(500.0, amp=1.0), FS, niveis, hysteresis=h).size
    assert forte > 2 * fraco


def test_quantizacao_do_instante_respeita_o_timer():
    """No nRF54L15 o instante vem de captura de TIMER a 128 MHz, não é exato."""
    tem = IntegrateFireTEM(timer_hz=1e6)
    t = tem.encode(_tone(500.0), FS)
    resto = np.abs(t * 1e6 - np.round(t * 1e6))
    assert np.all(resto < 1e-6)


def test_sem_timer_o_instante_nao_e_quantizado():
    """Teste negativo: com `timer_hz=None` os instantes não caem numa grade."""
    t = IntegrateFireTEM(timer_hz=None).encode(_tone(500.0), FS)
    assert not np.allclose(t * 1e6, np.round(t * 1e6))


def test_multicanal_produz_um_fluxo_por_canal():
    canais = multichannel_tem(_tone(500.0), FS, n_channels=4)
    assert len(canais) == 4
    assert all(c.size > 0 for c in canais)
    # canais distintos: e a diversidade entre eles que da a banda extra
    assert not np.allclose(canais[0][: min(map(len, canais))], canais[1][: min(map(len, canais))])


# ----------------------------------------------------------------------
# cruzamento de nível
# ----------------------------------------------------------------------
def test_cruzamento_de_nivel_conta_o_esperado():
    """Uma senoide cruza o nível zero duas vezes por período."""
    f, n_per = 500.0, 2
    t = crossing_times(_tone(f), FS, levels=[0.0], timer_hz=None)
    esperado = n_per * f * DUR
    assert abs(t.size - esperado) <= 2


def test_cruzamento_ignora_nivel_fora_da_faixa():
    """Teste negativo: nível acima da amplitude não gera evento algum."""
    assert crossing_times(_tone(500.0, amp=1.0), FS, levels=[5.0]).size == 0


def test_sem_histerese_o_ruido_domina_os_eventos():
    """Modo de falha real de ADC por cruzamento de nível, e a razão da histerese.

    Ruído em torno de um nível faz o comparador oscilar sem parar. Aqui o "silêncio"
    de amplitude 0,001 gera **mais** eventos que uma senoide de amplitude 1 — o que
    destruiria o argumento de energia da rota A2.
    """
    n = int(DUR * FS)
    rng = np.random.default_rng(1)
    silencio = 0.001 * rng.normal(size=n)
    ativo = _tone(500.0, amp=1.0, n=n)
    niveis = [-0.5, 0.0, 0.5]
    assert crossing_times(silencio, FS, niveis).size > crossing_times(ativo, FS, niveis).size


def test_com_histerese_o_silencio_volta_a_custar_pouco():
    """A promessa da rota A2, restaurada: o que não acontece não custa.

    A histerese precisa ser maior que a amplitude do ruído para suprimir o chattering.
    """
    n = int(DUR * FS)
    rng = np.random.default_rng(1)
    silencio = 0.001 * rng.normal(size=n)
    ativo = _tone(500.0, amp=1.0, n=n)
    niveis = [-0.5, 0.0, 0.5]
    h = 0.01  # dez vezes a amplitude do ruido
    n_silencio = crossing_times(silencio, FS, niveis, hysteresis=h).size
    n_ativo = crossing_times(ativo, FS, niveis, hysteresis=h).size
    assert n_silencio == 0
    assert n_ativo > 100
