"""Energy model: the breakdown adds up, scales as it should, and can say 'I don't know'.

The third group is the one that matters most. A missing coefficient treated as zero
produces a presentable and false µJ per diagnosis, and it is the easiest way for the
project's system metric to turn into fiction.
"""

import math

import pytest
from snq_embed import energy as E

# Fictitious coefficients, used ONLY to exercise the algebra of the model. None of
# them is from the nRF54L15 - the real platform is in datasheet_overview_parameters()
# and has all of its energy fields unknown.
FICTICIOUS_SOURCE = "fictitious test value; not from the nRF54L15"


def complete_params(**over) -> E.PlatformParameters:
    base = dict(
        energy_per_conversion_uj=1.0,
        current_active_ma=4.0,
        current_sleep_ua=2.0,
        current_radio_tx_ma=5.0,
        energy_per_ble_byte_uj=3.0,
        supply_voltage_v=3.0,
        cycles_per_butterfly=10.0,
        cpu_clock_hz=128e6,
        supply_voltage_min_v=1.7,
        supply_voltage_max_v=3.5,
        ram_bytes=256 * 1024,
    )
    base.update(over)
    return E.PlatformParameters(sources=dict.fromkeys(base, FICTICIOUS_SOURCE), **base)


# ---------------------------------------------------------------------------
# the breakdown adds up to the total
# ---------------------------------------------------------------------------
def test_o_total_e_exatamente_a_soma_das_tres_parcelas():
    w = E.workload(fs_hz=25_000.0, window_s=2.5, payload_bytes=32)
    p = complete_params()
    b = E.diagnosis_energy_uj(w, p)
    assert b.total_uj == pytest.approx(b.acquisition_uj + b.processing_uj + b.radio_uj)
    assert b.unknown_parts() == ()


def test_as_fracoes_das_tres_parcelas_somam_um():
    w = E.workload(fs_hz=25_000.0, window_s=2.5, payload_bytes=32)
    b = E.diagnosis_energy_uj(w, complete_params())
    fracs = [b.fraction_of(part) for part in ("acquisition_uj", "processing_uj", "radio_uj")]
    assert sum(fracs) == pytest.approx(1.0)


def test_a_parcela_dominante_e_a_de_maior_valor():
    w = E.workload(fs_hz=25_000.0, window_s=2.5, payload_bytes=32)
    b = E.diagnosis_energy_uj(w, complete_params())
    values = {p: getattr(b, p) for p in ("acquisition_uj", "processing_uj", "radio_uj")}
    assert b.dominant_part() == max(values, key=values.get)


# ---------------------------------------------------------------------------
# monotonicity in the number of samples
# ---------------------------------------------------------------------------
def test_aquisicao_cresce_com_o_numero_de_conversoes():
    p = complete_params()
    prev = -math.inf
    for factor in (64, 48, 32, 24, 16, 8, 4, 2, 1):
        w = E.workload(fs_hz=200_000.0 / factor, window_s=2.5, payload_bytes=32)
        value = E.acquisition_energy_uj(w, p)
        assert value > prev
        prev = value


def test_processamento_cresce_com_o_tamanho_da_fft():
    p = complete_params()
    prev = -math.inf
    for size in (256, 512, 1024, 2048, 4096):
        w = E.workload(fs_hz=200_000.0, window_s=2.5, payload_bytes=32, fft_size=size)
        value = E.processing_energy_uj(w, p)
        assert value > prev
        prev = value


def test_radio_com_payload_fixo_nao_depende_da_taxa():
    """This is the asymmetry the E06 hypothesis comes from.

    With a features payload, the radio part is constant while the acquisition part
    falls with the rate. Without this asymmetry there is no crossover at all.
    """
    p = complete_params()
    values = {
        factor: E.radio_energy_uj(
            E.workload(fs_hz=200_000.0 / factor, window_s=2.5, payload_bytes=32), p
        )
        for factor in (1, 8, 64)
    }
    assert len(set(values.values())) == 1


def test_a_razao_de_troca_e_conhecida_por_contagem():
    """The crossover point can be stated without any platform coefficient."""
    w = E.workload(fs_hz=200_000.0, window_s=2.5, payload_bytes=32)
    assert w.n_conversions == 500_000
    assert w.conversions_per_payload_byte == pytest.approx(500_000 / 32)
    assert w.butterflies_per_payload_byte == pytest.approx(w.n_butterflies / 32)


# ---------------------------------------------------------------------------
# the published threshold has to be what the model computes
# ---------------------------------------------------------------------------
def test_o_limiar_de_r_incorpora_o_termo_de_repouso_que_a_aquisicao_soma():
    """``R > A`` is not the criterion: acquisition includes idle, and the threshold rises with it.

    With ``R`` set exactly at ``A``, the radio part ties with the conversions and
    **loses** to the whole acquisition, because the idle energy is left over. It is
    the divergence between the published criterion and the model, exercised
    numerically.
    """
    w = E.workload(fs_hz=200_000.0, window_s=2.5, payload_bytes=32)
    a = w.conversions_per_payload_byte
    p = complete_params(energy_per_conversion_uj=1.0, energy_per_ble_byte_uj=a)
    b = E.diagnosis_energy_uj(w, p)
    assert b.radio_uj == pytest.approx(b.acquisition_uj - E.idle_energy_uj(w, p))
    assert b.radio_uj < b.acquisition_uj

    t = E.dominance_thresholds(w, p)
    assert t.radio_over_acquisition_min == pytest.approx(a)
    assert t.radio_over_acquisition_exact > a
    # with R exactly at the exact threshold, the tie happens
    no_limiar = complete_params(
        energy_per_conversion_uj=1.0,
        energy_per_ble_byte_uj=t.radio_over_acquisition_exact,
    )
    b2 = E.diagnosis_energy_uj(w, no_limiar)
    assert b2.radio_uj == pytest.approx(b2.acquisition_uj)


def test_s_e_o_deslocamento_do_limiar_e_sai_desconhecido_na_plataforma_real():
    w = E.workload(fs_hz=200_000.0, window_s=2.5, payload_bytes=32)
    p = complete_params()
    s = E.idle_conversions_equivalent(w, p)
    assert s == pytest.approx(
        p.current_sleep_ua * p.supply_voltage_v * w.window_s / p.energy_per_conversion_uj
    )
    t = E.dominance_thresholds(w, p)
    assert t.radio_over_acquisition_exact == pytest.approx(
        w.conversions_per_payload_byte + s / w.payload_bytes
    )

    real = E.dominance_thresholds(w, E.datasheet_overview_parameters())
    assert real.idle_conversions_equivalent is None
    assert real.radio_over_acquisition_exact is None
    # counting survives: the two counted thresholds remain known
    assert real.radio_over_acquisition_min == pytest.approx(500_000 / 32)
    assert real.radio_over_processing == pytest.approx(w.n_butterflies / 32)


def test_dominar_e_vencer_as_duas_outras_parcelas_nao_so_a_aquisicao():
    """With ``R`` above the acquisition threshold the radio can still lose to the FFT.

    This is the flaw of comparing two parts out of three: ``cycles_per_butterfly`` is
    unknown, and a high enough value puts processing on top at any point of the
    sweep.
    """
    w = E.workload(fs_hz=200_000.0, window_s=2.5, payload_bytes=32)
    t = E.dominance_thresholds(w, complete_params())
    p = complete_params(
        energy_per_conversion_uj=1.0,
        energy_per_ble_byte_uj=2.0 * t.radio_over_acquisition_exact,
        cycles_per_butterfly=1.0e5,
    )
    b = E.diagnosis_energy_uj(w, p)
    assert b.radio_uj > b.acquisition_uj
    assert b.dominant_part() == "processing_uj"

    # Q below B is exactly what describes this defeat.
    q = p.energy_per_ble_byte_uj / E.energy_per_butterfly_uj(p)
    assert q < t.radio_over_processing


def test_no_braco_raw_o_limiar_de_processamento_nao_e_constante():
    """Refutes the claim that nothing can cross over in the ``raw`` arm.

    ``A`` is 0.5 at every factor because ``payload = 2·n_conversions``. ``B`` is not:
    ``n_butterflies`` follows the FFT, which is the power of two below the count, and
    the ratio moves across the factors — even non-monotonically, because the FFT size
    jumps in steps. So the dominant part can change within the arm.
    """
    b_por_fator = {}
    for factor in (1, 2, 4, 8, 16, 24, 32, 48, 64):
        n = E.workload(200_000.0 / factor, 2.5, payload_bytes=1).n_conversions
        w = E.workload(200_000.0 / factor, 2.5, payload_bytes=2 * n)
        assert w.conversions_per_payload_byte == pytest.approx(0.5)
        b_por_fator[factor] = w.butterflies_per_payload_byte

    assert len(set(round(v, 6) for v in b_por_fator.values())) > 1
    assert b_por_fator[24] > b_por_fator[16]  # not monotonic: the FFT jumps in steps
    assert min(b_por_fator.values()) < 0.7 * max(b_por_fator.values())


def test_no_braco_raw_o_termo_de_repouso_tambem_move_o_limiar_de_r():
    """Even the radio × acquisition comparison is not constant in the ``raw`` arm.

    ``S/payload_bytes`` grows when the payload shrinks, and in the ``raw`` arm the
    payload shrinks along with the rate.
    """
    p = complete_params()
    exato = {}
    for factor in (1, 64):
        n = E.workload(200_000.0 / factor, 2.5, payload_bytes=1).n_conversions
        w = E.workload(200_000.0 / factor, 2.5, payload_bytes=2 * n)
        exato[factor] = E.dominance_thresholds(w, p).radio_over_acquisition_exact
    assert exato[64] > exato[1]


# ---------------------------------------------------------------------------
# unknown propagates as unknown, never as zero
# ---------------------------------------------------------------------------
def test_coeficiente_ausente_propaga_para_a_parcela_e_para_o_total():
    w = E.workload(fs_hz=25_000.0, window_s=2.5, payload_bytes=32)
    b = E.diagnosis_energy_uj(w, complete_params(energy_per_ble_byte_uj=None))
    assert b.radio_uj is None
    assert b.total_uj is None
    assert b.unknown_parts() == ("radio_uj",)
    assert b.dominant_part() is None


def test_parcela_desconhecida_nao_vira_zero_silencioso():
    """The total with one part missing cannot be the sum of the other two.

    The assertion that used to be here was ``b.total_uj is not piso``, which is
    vacuously true: ``None`` is never the same object as a float, so it passed
    without exercising anything. What needs to be checked is that **no public path**
    of the model returns the silent floor in place of the total, and that the true
    total is strictly greater than it.
    """
    w = E.workload(fs_hz=25_000.0, window_s=2.5, payload_bytes=32)
    b = E.diagnosis_energy_uj(w, complete_params(energy_per_ble_byte_uj=None))
    piso_silencioso = b.acquisition_uj + b.processing_uj
    assert piso_silencioso > 0.0

    assert b.total_uj is None
    assert b.dominant_part() is None
    assert b.fraction_of("acquisition_uj") is None
    assert b.fraction_of("processing_uj") is None

    completo = E.diagnosis_energy_uj(w, complete_params())
    assert completo.total_uj > piso_silencioso


def test_tensao_ausente_derruba_processamento_e_aquisicao_juntos():
    """Voltage enters both parts; its absence cannot affect only one of them."""
    w = E.workload(fs_hz=25_000.0, window_s=2.5, payload_bytes=32)
    b = E.diagnosis_energy_uj(w, complete_params(supply_voltage_v=None))
    assert b.acquisition_uj is None
    assert b.processing_uj is None
    assert b.radio_uj is not None
    assert E.energy_per_cycle_uj(complete_params(supply_voltage_v=None)) is None


def test_a_plataforma_real_declara_todos_os_campos_de_energia_como_desconhecidos():
    """Actual state of the evidence in the repository, and the model has to run like this."""
    p = E.datasheet_overview_parameters()
    unknown = p.unknown_parameters()
    for name in (
        "energy_per_conversion_uj",
        "current_active_ma",
        "current_sleep_ua",
        "current_radio_tx_ma",
        "energy_per_ble_byte_uj",
        "supply_voltage_v",
        "cycles_per_butterfly",
    ):
        assert name in unknown, name
        assert p.source_of(name)

    b = E.diagnosis_energy_uj(E.workload(25_000.0, 2.5, 32), p)
    assert b.total_uj is None
    assert b.unknown_parts() == ("acquisition_uj", "processing_uj", "radio_uj")


def test_procedencia_e_obrigatoria_em_todo_campo():
    with pytest.raises(ValueError, match="missing provenance"):
        E.PlatformParameters(
            energy_per_conversion_uj=1.0,
            current_active_ma=None,
            current_sleep_ua=None,
            current_radio_tx_ma=None,
            energy_per_ble_byte_uj=None,
            supply_voltage_v=None,
            cycles_per_butterfly=None,
            cpu_clock_hz=128e6,
            supply_voltage_min_v=None,
            supply_voltage_max_v=None,
            ram_bytes=None,
            sources={"energy_per_conversion_uj": "source"},
        )


# ---------------------------------------------------------------------------
# bits x rate axis
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("fs_hz", "n_bits"),
    [(200_000.0, 12), (250_000.0, 12), (250_001.0, 10), (31_250.0, 14), (3_125.0, 14)],
)
def test_baixar_a_taxa_sobe_a_resolucao_disponivel(fs_hz, n_bits):
    point = E.operating_point_for(fs_hz)
    assert point is not None
    assert point.n_bits == n_bits


def test_acima_do_maior_ponto_declarado_nao_ha_ponto_de_operacao():
    assert E.operating_point_for(2.1e6) is None


def test_todo_ponto_de_operacao_carrega_procedencia():
    for point in E.DATASHEET_OVERVIEW_ADC_POINTS:
        assert "nordic2024nrf54l15" in point.source


# ---------------------------------------------------------------------------
# counting guards
# ---------------------------------------------------------------------------
def test_fft_maior_que_a_janela_e_recusada():
    with pytest.raises(ValueError, match="larger than"):
        E.workload(fs_hz=1_000.0, window_s=1.0, payload_bytes=8, fft_size=2048)


def test_taxa_ou_janela_nao_positiva_e_recusada():
    with pytest.raises(ValueError, match="positive"):
        E.workload(fs_hz=0.0, window_s=1.0, payload_bytes=8)
