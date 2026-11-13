"""Time encoding — the A2 realization of ADR 0004.

What is tested here is the property that justifies the route: **energy proportional
to the activity of the signal, not to the elapsed time**.
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
    assert np.all(np.diff(t) > 0), "instants must be strictly increasing"
    # truly irregular: the intervals vary with the signal
    intervalos = np.diff(t)
    assert intervalos.std() / intervalos.mean() > 0.01


def test_bias_insuficiente_e_erro_explicito():
    """If the amplitude exceeds the bias, the integrator may never fire."""
    tem = IntegrateFireTEM(bias=0.5)
    with pytest.raises(ValueError, match="bias"):
        tem.encode(_tone(500.0, amp=1.0), FS)


def test_taxa_nominal_bate_com_o_parametro():
    """The design rate is `bias / (2*delta*kappa)`; `for_rate` inverts the relation."""
    tem = IntegrateFireTEM.for_rate(50e3)
    assert tem.nominal_rate_hz() == pytest.approx(50e3)
    # and the rate measured with a zero-mean input stays close to the nominal one
    assert tem.firing_rate_hz(_tone(500.0, amp=0.5), FS) == pytest.approx(50e3, rel=0.05)


def test_taxa_do_iftem_nao_depende_da_amplitude():
    """Correction of an assumption of mine, measured here.

    For a zero-mean input, the integral of ``x + bias`` over one period equals
    ``bias * T`` **regardless of the amplitude** — the amplitude cancels out. So the
    IF-TEM does **not** spend less when the machine is stopped.

    This undoes half of the energy argument that ADR 0004 attributed to this route:
    the gain of the IF-TEM is doing without the global clock, not firing less.
    """
    tem = IntegrateFireTEM(bias=2.0)
    taxas = [tem.firing_rate_hz(_tone(500.0, amp=a), FS) for a in (0.01, 0.5, 1.9)]
    assert taxas[0] == pytest.approx(taxas[1]) == pytest.approx(taxas[2])


def test_taxa_do_iftem_depende_da_media():
    """What actually governs the rate: ``(mean(x) + bias) / (2*delta*kappa)``."""
    tem = IntegrateFireTEM(bias=2.0)
    for dc in (-0.5, 0.0, 0.5):
        x = _tone(500.0, amp=0.3) + dc
        previsto = (dc + tem.bias) / (2.0 * tem.delta * tem.kappa)
        assert tem.firing_rate_hz(x, FS) == pytest.approx(previsto, rel=0.01)


def test_cruzamento_de_nivel_e_que_e_proporcional_a_atividade():
    """The energy argument of route A2 belongs to this variant, not to the IF-TEM.

    A more active signal crosses more levels and generates more events; a still
    signal generates none. This is where "what does not happen does not cost" really
    holds.
    """
    niveis, h = [-0.5, 0.0, 0.5], 0.005
    fraco = crossing_times(_tone(500.0, amp=0.01), FS, niveis, hysteresis=h).size
    forte = crossing_times(_tone(500.0, amp=1.0), FS, niveis, hysteresis=h).size
    assert forte > 2 * fraco


def test_quantizacao_do_instante_respeita_o_timer():
    """On the nRF54L15 the instant comes from a TIMER capture at 128 MHz; it is not exact."""
    tem = IntegrateFireTEM(timer_hz=1e6)
    t = tem.encode(_tone(500.0), FS)
    resto = np.abs(t * 1e6 - np.round(t * 1e6))
    assert np.all(resto < 1e-6)


def test_sem_timer_o_instante_nao_e_quantizado():
    """Negative test: with `timer_hz=None` the instants do not fall on a grid."""
    t = IntegrateFireTEM(timer_hz=None).encode(_tone(500.0), FS)
    assert not np.allclose(t * 1e6, np.round(t * 1e6))


def test_multicanal_produz_um_fluxo_por_canal():
    canais = multichannel_tem(_tone(500.0), FS, n_channels=4)
    assert len(canais) == 4
    assert all(c.size > 0 for c in canais)
    # distinct channels: it is the diversity among them that gives the extra bandwidth
    assert not np.allclose(canais[0][: min(map(len, canais))], canais[1][: min(map(len, canais))])


# ----------------------------------------------------------------------
# level crossing
# ----------------------------------------------------------------------
def test_cruzamento_de_nivel_conta_o_esperado():
    """A sine wave crosses the zero level twice per period."""
    f, n_per = 500.0, 2
    t = crossing_times(_tone(f), FS, levels=[0.0], timer_hz=None)
    esperado = n_per * f * DUR
    assert abs(t.size - esperado) <= 2


def test_cruzamento_ignora_nivel_fora_da_faixa():
    """Negative test: a level above the amplitude generates no event at all."""
    assert crossing_times(_tone(500.0, amp=1.0), FS, levels=[5.0]).size == 0


def test_sem_histerese_o_ruido_domina_os_eventos():
    """A real failure mode of level-crossing ADCs, and the reason for hysteresis.

    Noise around a level makes the comparator toggle endlessly. Here the "silence" of
    amplitude 0.001 generates **more** events than a sine wave of amplitude 1 — which
    would destroy the energy argument of route A2.
    """
    n = int(DUR * FS)
    rng = np.random.default_rng(1)
    silencio = 0.001 * rng.normal(size=n)
    ativo = _tone(500.0, amp=1.0, n=n)
    niveis = [-0.5, 0.0, 0.5]
    assert crossing_times(silencio, FS, niveis).size > crossing_times(ativo, FS, niveis).size


def test_com_histerese_o_silencio_volta_a_custar_pouco():
    """The promise of route A2, restored: what does not happen does not cost.

    The hysteresis must be larger than the noise amplitude to suppress chattering.
    """
    n = int(DUR * FS)
    rng = np.random.default_rng(1)
    silencio = 0.001 * rng.normal(size=n)
    ativo = _tone(500.0, amp=1.0, n=n)
    niveis = [-0.5, 0.0, 0.5]
    h = 0.01  # ten times the noise amplitude
    n_silencio = crossing_times(silencio, FS, niveis, hysteresis=h).size
    n_ativo = crossing_times(ativo, FS, niveis, hysteresis=h).size
    assert n_silencio == 0
    assert n_ativo > 100
