"""Modelo de energia: a decomposição fecha, escala como deve, e sabe dizer 'não sei'.

O terceiro grupo é o que mais importa. Um coeficiente ausente tratado como zero
produz um µJ por diagnóstico apresentável e falso, e é a forma mais fácil de a
métrica de sistema do projeto virar ficção.
"""

import math

import pytest
from snq_embed import energy as E

# Coeficientes fictícios, usados SÓ para exercitar a álgebra do modelo. Nenhum
# deles é do nRF54L15 - a plataforma real está em datasheet_overview_parameters()
# e tem todos os campos de energia desconhecidos.
FICTICIOUS_SOURCE = "valor ficticio de teste; nao e do nRF54L15"


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
# a decomposicao soma o total
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
# monotonicidade em numero de amostras
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
    """É a assimetria de onde sai a hipótese do E06.

    Com payload de features, a parcela de rádio é constante enquanto a de aquisição
    cai com a taxa. Sem essa assimetria não existe cruzamento nenhum.
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
    """O ponto de cruzamento se enuncia sem nenhum coeficiente de plataforma."""
    w = E.workload(fs_hz=200_000.0, window_s=2.5, payload_bytes=32)
    assert w.n_conversions == 500_000
    assert w.conversions_per_payload_byte == pytest.approx(500_000 / 32)
    assert w.butterflies_per_payload_byte == pytest.approx(w.n_butterflies / 32)


# ---------------------------------------------------------------------------
# o limiar publicado tem de ser o que o modelo calcula
# ---------------------------------------------------------------------------
def test_o_limiar_de_r_incorpora_o_termo_de_repouso_que_a_aquisicao_soma():
    """``R > A`` não é o critério: a aquisição soma repouso, e o limiar sobe com ele.

    Com ``R`` posto exatamente em ``A``, a parcela de rádio empata com as conversões
    e **perde** para a aquisição inteira, porque sobra o repouso. É a divergência
    entre o critério publicado e o modelo, exercitada numericamente.
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
    # com R exatamente no limiar exato, o empate acontece
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
    # a contagem sobrevive: os dois limiares contados continuam conhecidos
    assert real.radio_over_acquisition_min == pytest.approx(500_000 / 32)
    assert real.radio_over_processing == pytest.approx(w.n_butterflies / 32)


def test_dominar_e_vencer_as_duas_outras_parcelas_nao_so_a_aquisicao():
    """Com ``R`` acima do limiar de aquisição o rádio ainda pode perder para a FFT.

    É o defeito de comparar duas parcelas de três: ``cycles_per_butterfly`` é
    desconhecido, e um valor alto o bastante põe o processamento no topo em
    qualquer ponto da varredura.
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

    # Q abaixo de B e exatamente o que descreve essa derrota.
    q = p.energy_per_ble_byte_uj / E.energy_per_butterfly_uj(p)
    assert q < t.radio_over_processing


def test_no_braco_raw_o_limiar_de_processamento_nao_e_constante():
    """Refuta a afirmação de que no braço ``raw`` nada pode cruzar.

    ``A`` é 0,5 em todos os fatores porque ``payload = 2·n_conversions``. ``B`` não:
    ``n_butterflies`` acompanha a FFT, que é a potência de dois abaixo da contagem, e
    a razão anda entre os fatores — inclusive não monotonicamente, porque o tamanho
    da FFT salta em degraus. Logo a parcela dominante pode mudar dentro do braço.
    """
    b_por_fator = {}
    for factor in (1, 2, 4, 8, 16, 24, 32, 48, 64):
        n = E.workload(200_000.0 / factor, 2.5, payload_bytes=1).n_conversions
        w = E.workload(200_000.0 / factor, 2.5, payload_bytes=2 * n)
        assert w.conversions_per_payload_byte == pytest.approx(0.5)
        b_por_fator[factor] = w.butterflies_per_payload_byte

    assert len(set(round(v, 6) for v in b_por_fator.values())) > 1
    assert b_por_fator[24] > b_por_fator[16]  # nao monotono: a FFT salta em degraus
    assert min(b_por_fator.values()) < 0.7 * max(b_por_fator.values())


def test_no_braco_raw_o_termo_de_repouso_tambem_move_o_limiar_de_r():
    """Mesmo a comparação rádio × aquisição não é constante no braço ``raw``.

    ``S/payload_bytes`` cresce quando o payload encolhe, e no braço ``raw`` ele
    encolhe junto com a taxa.
    """
    p = complete_params()
    exato = {}
    for factor in (1, 64):
        n = E.workload(200_000.0 / factor, 2.5, payload_bytes=1).n_conversions
        w = E.workload(200_000.0 / factor, 2.5, payload_bytes=2 * n)
        exato[factor] = E.dominance_thresholds(w, p).radio_over_acquisition_exact
    assert exato[64] > exato[1]


# ---------------------------------------------------------------------------
# desconhecido propaga como desconhecido, nunca como zero
# ---------------------------------------------------------------------------
def test_coeficiente_ausente_propaga_para_a_parcela_e_para_o_total():
    w = E.workload(fs_hz=25_000.0, window_s=2.5, payload_bytes=32)
    b = E.diagnosis_energy_uj(w, complete_params(energy_per_ble_byte_uj=None))
    assert b.radio_uj is None
    assert b.total_uj is None
    assert b.unknown_parts() == ("radio_uj",)
    assert b.dominant_part() is None


def test_parcela_desconhecida_nao_vira_zero_silencioso():
    """O total com uma parcela ausente não pode ser a soma das outras duas.

    A asserção que existia aqui era ``b.total_uj is not piso``, que é vacuamente
    verdadeira: ``None`` nunca é o mesmo objeto que um float, então ela passava sem
    exercitar nada. O que precisa ser verificado é que **nenhuma via pública** do
    modelo devolve o piso silencioso no lugar do total, e que o total verdadeiro é
    estritamente maior que ele.
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
    """A tensão entra nas duas parcelas; ausência dela não pode sobrar em uma só."""
    w = E.workload(fs_hz=25_000.0, window_s=2.5, payload_bytes=32)
    b = E.diagnosis_energy_uj(w, complete_params(supply_voltage_v=None))
    assert b.acquisition_uj is None
    assert b.processing_uj is None
    assert b.radio_uj is not None
    assert E.energy_per_cycle_uj(complete_params(supply_voltage_v=None)) is None


def test_a_plataforma_real_declara_todos_os_campos_de_energia_como_desconhecidos():
    """Estado real da evidência no repositório, e o modelo tem de rodar assim."""
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
    with pytest.raises(ValueError, match="procedencia ausente"):
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
            sources={"energy_per_conversion_uj": "fonte"},
        )


# ---------------------------------------------------------------------------
# eixo bits x taxa
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
# guardas de contagem
# ---------------------------------------------------------------------------
def test_fft_maior_que_a_janela_e_recusada():
    with pytest.raises(ValueError, match="maior que"):
        E.workload(fs_hz=1_000.0, window_s=1.0, payload_bytes=8, fft_size=2048)


def test_taxa_ou_janela_nao_positiva_e_recusada():
    with pytest.raises(ValueError, match="positivos"):
        E.workload(fs_hz=0.0, window_s=1.0, payload_bytes=8)
