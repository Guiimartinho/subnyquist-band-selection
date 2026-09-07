"""Modelo de energia do nó: µJ por diagnóstico, decomposto em três parcelas.

É a métrica de sistema declarada do projeto — µJ por diagnóstico e bytes de BLE por
diagnóstico, nunca latência. Este módulo é **modelo**, não medição: a Medida 2 de
bancada (`bench/protocols/`, Power Profiler Kit II ou shunt + osciloscópio) é quem
valida ou refuta cada coeficiente.

Duas coisas são deliberadamente separadas aqui, e a separação é o ponto do módulo:

``DiagnosisWorkload``
    **Contagem pura.** Quantas conversões, quantas borboletas de FFT, quantos bytes
    de payload um diagnóstico exige. Nada disso depende de datasheet — sai da taxa
    de amostragem, da janela e da política de payload. É conhecido sempre.

``PlatformParameters``
    **Coeficientes de plataforma.** µJ por conversão, µJ por ciclo, µJ por byte de
    rádio. Cada campo carrega procedência obrigatória, e o valor pode ser ``None``.

``None`` significa *desconhecido*, e desconhecido **propaga**: qualquer parcela que
dependa de um coeficiente ausente sai ``None`` e o total sai ``None``. Um zero
silencioso nesse lugar produziria um número de energia falso, que é exatamente o
resultado que este projeto não pode publicar.

Procedência do que se sabe da plataforma
----------------------------------------
O único documento de plataforma no repositório é ``literature/pdf/nordic2024nrf54l15.pdf``.
Ele tem 6 páginas e é a **página de visão geral** do nRF54L15/L10/L05, não o datasheet
v0.9 completo que ela referencia. Verificado nesta sessão com::

    uv run --with pypdf python literature/read_pdf.py nordic2024nrf54l15 \\
        --grep "SAADC" "current" "consumption" "mA" "uA" "BLE"

Resultado: ``SAADC`` e ``current`` têm **zero** ocorrências; ``consumption`` aparece
só em prosa de marketing ("ultra-low power consumption"); ``mA`` só casa dentro de
palavras. Não há corrente, energia por conversão, energia por byte de rádio, banda de
track-and-hold nem jitter nesse arquivo. Todos esses campos ficam ``None``.

O eixo bits × taxa
------------------
A página declara três pontos de operação do ADC — 14 bit a 31,25 kSPS, 12 bit a
250 kSPS e até 10 bit a 2 MSPS. É o eixo que separa esta tese de ``vol2024poweraware``,
que aprende as regiões de quantização (**bits**) e não mexe na **taxa**. Aqui a taxa é
a variável, e reduzi-la move o conversor para um ponto de operação com mais bits: a
partir de 31,25 kSPS o mesmo periférico entrega 14 bits em vez de 12. Ver
``operating_point_for``.

Dominância é comparação de três parcelas, não de duas
------------------------------------------------------
``EnergyBreakdown.dominant_part`` escolhe entre aquisição, processamento e rádio.
Logo "o rádio domina" são **duas** desigualdades, e ``DominanceThresholds`` publica
as duas com a correção do termo de repouso incluída::

    rádio > aquisição     ⟺  R > A + S / payload_bytes
    rádio > processamento ⟺  Q > B

``A = n_conversions / payload_bytes`` e ``B = n_butterflies / payload_bytes`` são
contagem. ``R``, ``Q`` e ``S`` são razões de coeficientes de plataforma, hoje todas
desconhecidas. Escrever o critério como ``R > A`` seria comparar duas parcelas de
três e, além disso, apagar o termo de repouso que ``acquisition_energy_uj`` soma.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TypeAlias

__all__ = [
    "BLE_PHY_RATES_BPS",
    "DATASHEET_OVERVIEW_ADC_POINTS",
    "UNKNOWN_SOURCE",
    "AdcOperatingPoint",
    "DiagnosisWorkload",
    "DominanceThresholds",
    "EnergyBreakdown",
    "PlatformParameters",
    "acquisition_energy_uj",
    "datasheet_overview_parameters",
    "diagnosis_energy_uj",
    "dominance_thresholds",
    "energy_per_butterfly_uj",
    "energy_per_cycle_uj",
    "idle_conversions_equivalent",
    "idle_energy_uj",
    "operating_point_for",
    "processing_energy_uj",
    "radio_energy_uj",
    "workload",
]

#: Um coeficiente pode faltar. ``None`` é a única marca de desconhecido aceita.
Maybe: TypeAlias = float | None

#: Texto de procedência obrigatório para todo campo cujo valor é desconhecido.
UNKNOWN_SOURCE = "nao localizado - pendente do datasheet v0.9 completo ou da Medida 2"

_OVERVIEW = "nordic2024nrf54l15.pdf (pagina de visao geral, 6 pag.)"


# ----------------------------------------------------------------------------
# pontos de operacao do conversor - o eixo bits x taxa
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class AdcOperatingPoint:
    """Um par (resolução, taxa máxima) declarado pelo fabricante."""

    n_bits: int
    max_rate_sps: float
    source: str


#: Literal da página lida: "ADC with up to eight programmable gain channels. 14-bit at
#: 31.25 ksps, 12-bit at 250 ksps, and up to 10-bit at 2 Msps." (p. 5).
DATASHEET_OVERVIEW_ADC_POINTS: tuple[AdcOperatingPoint, ...] = (
    AdcOperatingPoint(14, 31_250.0, f"{_OVERVIEW} p.5, '14-bit at 31.25 ksps'"),
    AdcOperatingPoint(12, 250_000.0, f"{_OVERVIEW} p.5, '12-bit at 250 ksps'"),
    AdcOperatingPoint(10, 2_000_000.0, f"{_OVERVIEW} p.5, 'up to 10-bit at 2 Msps'"),
)

#: Taxas de PHY do BLE 6.0 declaradas na mesma página (p. 4). Entram no modelo só
#: como contexto: sem µJ por byte, taxa de PHY não vira energia.
BLE_PHY_RATES_BPS: tuple[float, ...] = (2e6, 1e6, 500e3, 125e3)


def operating_point_for(fs_hz: float) -> AdcOperatingPoint | None:
    """Ponto de operação de **maior resolução** que sustenta ``fs_hz``.

    Devolve ``None`` se nenhum ponto declarado alcança a taxa pedida. É a leitura
    útil do eixo bits × taxa: baixar a taxa de amostragem não é só economizar
    conversões, é subir de resolução no mesmo periférico.
    """
    viable = [p for p in DATASHEET_OVERVIEW_ADC_POINTS if fs_hz <= p.max_rate_sps]
    return max(viable, key=lambda p: p.n_bits) if viable else None


# ----------------------------------------------------------------------------
# coeficientes de plataforma
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PlatformParameters:
    """Coeficientes de energia do nó, cada um com procedência declarada.

    ``sources`` precisa cobrir **todos** os campos numéricos; o construtor recusa a
    instância se faltar procedência de algum. Isso é intencional: um coeficiente sem
    origem declarada é indistinguível de um chute depois que entra numa planilha.
    """

    energy_per_conversion_uj: Maybe
    """Energia de uma conversão do SAADC, porta a porta (amostragem + SAR)."""

    current_active_ma: Maybe
    """Corrente do Cortex-M33 processando a 128 MHz."""

    current_sleep_ua: Maybe
    """Corrente entre conversões, com DPPI/TIMER acordando o SAADC e a CPU dormindo."""

    current_radio_tx_ma: Maybe
    """Corrente do rádio transmitindo. Contexto: o modelo usa µJ por byte."""

    energy_per_ble_byte_uj: Maybe
    """Energia por byte de payload BLE entregue, já amortizando conexão e overhead."""

    supply_voltage_v: Maybe
    """Tensão do ponto de operação escolhido. A faixa é conhecida, o ponto não."""

    cycles_per_butterfly: Maybe
    """Ciclos por borboleta radix-2 na FFT real, com instruções DSP do M33."""

    cpu_clock_hz: Maybe
    """Relógio do núcleo."""

    supply_voltage_min_v: Maybe
    supply_voltage_max_v: Maybe
    ram_bytes: Maybe
    """Contexto de orçamento: o buffer da janela precisa caber aqui."""

    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        missing = [f.name for f in self._numeric_fields() if f.name not in self.sources]
        if missing:
            raise ValueError(f"procedencia ausente para: {', '.join(missing)}")

    @classmethod
    def _numeric_fields(cls) -> tuple:
        return tuple(f for f in fields(cls) if f.name != "sources")

    def unknown_parameters(self) -> tuple[str, ...]:
        """Nomes dos coeficientes desconhecidos, em ordem de declaração."""
        return tuple(f.name for f in self._numeric_fields() if getattr(self, f.name) is None)

    def source_of(self, name: str) -> str:
        return self.sources[name]

    def summary(self) -> str:
        unknown = self.unknown_parameters()
        return (
            f"{len(self._numeric_fields()) - len(unknown)} conhecidos, "
            f"{len(unknown)} desconhecidos: {', '.join(unknown) if unknown else '-'}"
        )


def datasheet_overview_parameters() -> PlatformParameters:
    """O que a página de visão geral do nRF54L15 sustenta — e só isso.

    Todo campo de energia sai ``None``. Não é omissão: é o estado real da evidência
    disponível no repositório, e o modelo tem de rodar assim.
    """
    unknown = UNKNOWN_SOURCE
    return PlatformParameters(
        energy_per_conversion_uj=None,
        current_active_ma=None,
        current_sleep_ua=None,
        current_radio_tx_ma=None,
        energy_per_ble_byte_uj=None,
        supply_voltage_v=None,
        cycles_per_butterfly=None,
        cpu_clock_hz=128e6,
        supply_voltage_min_v=1.7,
        supply_voltage_max_v=3.5,
        ram_bytes=256 * 1024,
        sources={
            "energy_per_conversion_uj": unknown,
            "current_active_ma": unknown,
            "current_sleep_ua": unknown,
            "current_radio_tx_ma": unknown,
            "energy_per_ble_byte_uj": unknown,
            "supply_voltage_v": (
                f"{_OVERVIEW} p.5 declara a faixa '1.7 V to 3.5 V supply'; "
                "o ponto de operacao nao esta declarado - nao localizado"
            ),
            "cycles_per_butterfly": (
                "depende da implementacao de FFT escolhida e do M33; nao localizado"
            ),
            "cpu_clock_hz": f"{_OVERVIEW} p.4, '128 MHz Arm Cortex-M33'",
            "supply_voltage_min_v": f"{_OVERVIEW} p.5, '1.7 V to 3.5 V supply and I/O voltage'",
            "supply_voltage_max_v": f"{_OVERVIEW} p.5, '1.7 V to 3.5 V supply and I/O voltage'",
            "ram_bytes": f"{_OVERVIEW} p.5, 'nRF54L15 - ... and 256 KB RAM'",
        },
    )


def energy_per_cycle_uj(params: PlatformParameters) -> Maybe:
    """µJ por ciclo de CPU, de ``I·V/f``. ``None`` se faltar qualquer um dos três."""
    if (
        params.current_active_ma is None
        or params.supply_voltage_v is None
        or params.cpu_clock_hz is None
        or params.cpu_clock_hz <= 0.0
    ):
        return None
    watts = params.current_active_ma * 1e-3 * params.supply_voltage_v
    return watts / params.cpu_clock_hz * 1e6


# ----------------------------------------------------------------------------
# contagem: o que um diagnostico exige, sem nenhum numero de plataforma
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DiagnosisWorkload:
    """Contagem pura de um diagnóstico. Independe de datasheet e nunca é desconhecida."""

    fs_hz: float
    window_s: float
    n_conversions: int
    fft_size: int
    n_butterflies: float
    payload_bytes: int

    @property
    def adc_operating_point(self) -> AdcOperatingPoint | None:
        return operating_point_for(self.fs_hz)

    @property
    def conversions_per_payload_byte(self) -> float:
        """``A`` — contagem que dá o **piso** do limiar de ``R``, não o limiar.

        A parcela de rádio ultrapassa a de aquisição quando
        ``R := energy_per_ble_byte_uj / energy_per_conversion_uj`` excede
        ``A + S / payload_bytes``, onde ``S`` é o termo de repouso convertido em
        conversões equivalentes (ver :func:`idle_conversions_equivalent`). Como
        ``S >= 0``, este número sozinho só sustenta a metade contável do enunciado:
        ``R <= A`` **garante** que o rádio não ultrapassa a aquisição; ``R > A`` não
        garante o contrário enquanto ``S`` for desconhecido.
        """
        return math.inf if self.payload_bytes == 0 else self.n_conversions / self.payload_bytes

    @property
    def butterflies_per_payload_byte(self) -> float:
        """``B`` — limiar **exato** da comparação rádio × processamento.

        A parcela de rádio ultrapassa a de processamento quando
        ``Q := energy_per_ble_byte_uj / energia_por_borboleta_uj`` excede este
        número. Também é contagem pura, e é a segunda desigualdade sem a qual
        "o rádio domina" não fica dito: dominar é ser a maior das **três** parcelas.
        """
        return math.inf if self.payload_bytes == 0 else self.n_butterflies / self.payload_bytes


def _fft_size_for(n_samples: int) -> int:
    """Maior potência de dois que cabe em ``n_samples``."""
    if n_samples < 2:
        return 0
    return 1 << (n_samples.bit_length() - 1)


def workload(
    fs_hz: float,
    window_s: float,
    payload_bytes: int,
    fft_size: int | None = None,
) -> DiagnosisWorkload:
    """Monta a contagem de um diagnóstico a partir da taxa, da janela e do payload.

    ``fft_size`` omitido usa a maior potência de dois que cabe na janela. A contagem
    de borboletas é a de uma FFT radix-2, ``(N/2)·log2 N``, que é o custo dominante
    da cadeia envelope → espectro de ordem usada no E07.
    """
    if fs_hz <= 0.0 or window_s <= 0.0:
        raise ValueError("fs_hz e window_s precisam ser positivos")
    if payload_bytes < 0:
        raise ValueError("payload_bytes nao pode ser negativo")

    n_conversions = int(round(fs_hz * window_s))
    size = _fft_size_for(n_conversions) if fft_size is None else int(fft_size)
    if size > n_conversions:
        raise ValueError(f"fft_size {size} maior que as {n_conversions} conversoes da janela")
    butterflies = 0.0 if size < 2 else (size / 2.0) * math.log2(size)
    return DiagnosisWorkload(
        fs_hz=float(fs_hz),
        window_s=float(window_s),
        n_conversions=n_conversions,
        fft_size=size,
        n_butterflies=butterflies,
        payload_bytes=int(payload_bytes),
    )


# ----------------------------------------------------------------------------
# composicao das tres parcelas
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyBreakdown:
    """As três parcelas de um diagnóstico. ``None`` em qualquer uma é desconhecido."""

    acquisition_uj: Maybe
    processing_uj: Maybe
    radio_uj: Maybe

    _PARTS = ("acquisition_uj", "processing_uj", "radio_uj")

    @property
    def total_uj(self) -> Maybe:
        """Soma das três, ou ``None`` se alguma for desconhecida.

        Somar tratando desconhecido como zero devolveria um piso apresentável e
        errado. O total só existe quando as três parcelas existem.
        """
        parts = [getattr(self, p) for p in self._PARTS]
        return None if any(v is None for v in parts) else float(sum(parts))  # type: ignore[arg-type]

    def unknown_parts(self) -> tuple[str, ...]:
        return tuple(p for p in self._PARTS if getattr(self, p) is None)

    def dominant_part(self) -> str | None:
        """Nome da parcela dominante, ou ``None`` se alguma parcela for desconhecida.

        A hipótese do E06 é sobre *qual* parcela domina. Responder isso com uma
        parcela ausente seria responder sobre um subconjunto e chamar de conclusão.
        """
        if self.unknown_parts():
            return None
        return max(self._PARTS, key=lambda p: getattr(self, p))

    def fraction_of(self, part: str) -> Maybe:
        total = self.total_uj
        value = getattr(self, part)
        if total is None or value is None or total <= 0.0:
            return None
        return float(value) / total


def idle_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """Energia de repouso da janela: ``current_sleep_ua · supply_voltage_v · window_s``.

    Existe porque a arquitetura do nó é DPPI + TIMER → SAADC com a CPU dormindo:
    baixar a taxa reduz as conversões mas **não** encurta a janela, então esta
    parcela não escala com o fator. É por isso que ela precisa ter nome próprio —
    todo limiar que a ignore está comparando um modelo diferente do que roda.
    """
    if params.current_sleep_ua is None or params.supply_voltage_v is None:
        return None
    # uA * V * s = uJ
    return params.current_sleep_ua * params.supply_voltage_v * w.window_s


def acquisition_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """Conversões × energia por conversão, mais a energia de repouso na janela."""
    if params.energy_per_conversion_uj is None:
        return None
    idle = idle_energy_uj(w, params)
    if idle is None:
        return None
    return w.n_conversions * params.energy_per_conversion_uj + idle


def energy_per_butterfly_uj(params: PlatformParameters) -> Maybe:
    """µJ de uma borboleta radix-2: ciclos por borboleta × µJ por ciclo."""
    per_cycle = energy_per_cycle_uj(params)
    if per_cycle is None or params.cycles_per_butterfly is None:
        return None
    return params.cycles_per_butterfly * per_cycle


def processing_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """Borboletas × energia por borboleta."""
    per_butterfly = energy_per_butterfly_uj(params)
    if per_butterfly is None:
        return None
    return w.n_butterflies * per_butterfly


def radio_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """Bytes de payload × energia por byte entregue."""
    if params.energy_per_ble_byte_uj is None:
        return None
    return w.payload_bytes * params.energy_per_ble_byte_uj


def diagnosis_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> EnergyBreakdown:
    """As três parcelas de um diagnóstico, cada uma podendo sair desconhecida."""
    return EnergyBreakdown(
        acquisition_uj=acquisition_energy_uj(w, params),
        processing_uj=processing_energy_uj(w, params),
        radio_uj=radio_energy_uj(w, params),
    )


# ----------------------------------------------------------------------------
# a regiao de dominancia - duas desigualdades, tres razoes de coeficientes
# ----------------------------------------------------------------------------
def idle_conversions_equivalent(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """``S`` — a energia de repouso medida em conversões equivalentes.

    ``S := idle_energy_uj / energy_per_conversion_uj``. É adimensional e é a
    terceira razão de coeficientes do modelo: ela desloca o limiar de ``R`` para
    cima, e é exatamente o termo que um limiar escrito como ``R > A`` omite.
    """
    idle = idle_energy_uj(w, params)
    if idle is None or params.energy_per_conversion_uj is None:
        return None
    if params.energy_per_conversion_uj <= 0.0:
        return None
    return idle / params.energy_per_conversion_uj


@dataclass(frozen=True)
class DominanceThresholds:
    """Onde, no espaço de coeficientes, a parcela de rádio domina as outras **duas**.

    ``dominant_part`` decide entre três parcelas, então "o rádio domina" são duas
    desigualdades simultâneas, não uma:

    * rádio > aquisição   ⟺  ``R > A + S / payload_bytes``
    * rádio > processamento ⟺  ``Q > B``

    com ``R := energy_per_ble_byte_uj / energy_per_conversion_uj``,
    ``Q := energy_per_ble_byte_uj / energy_per_butterfly_uj`` e ``S`` dado por
    :func:`idle_conversions_equivalent`. ``A`` e ``B`` são **contagem pura**;
    ``R``, ``Q`` e ``S`` são razões de coeficientes de plataforma.

    O que a contagem sozinha decide, sem nenhum coeficiente: ``R <= A`` implica que
    o rádio **não** ultrapassa a aquisição (porque ``S >= 0``), e ``Q <= B`` implica
    que não ultrapassa o processamento. O recíproco depende de ``S``, e ``S`` está
    desconhecido — por isso ``radio_over_acquisition_exact`` sai ``None``.
    """

    payload_bytes: int
    conversions_per_payload_byte: float
    """``A``."""

    butterflies_per_payload_byte: float
    """``B`` — limiar exato de ``Q``, sem correção nenhuma pendente."""

    idle_conversions_equivalent: Maybe
    """``S``, ou ``None`` enquanto faltar coeficiente de repouso ou de conversão."""

    @property
    def radio_over_acquisition_min(self) -> float:
        """Piso do limiar de ``R``: vale ``A``, e é contagem."""
        return self.conversions_per_payload_byte

    @property
    def radio_over_acquisition_exact(self) -> Maybe:
        """Limiar exato de ``R``, ``A + S/payload_bytes``. ``None`` se ``S`` faltar."""
        if self.idle_conversions_equivalent is None or self.payload_bytes <= 0:
            return None
        return self.conversions_per_payload_byte + self.idle_conversions_equivalent / (
            self.payload_bytes
        )

    @property
    def radio_over_processing(self) -> float:
        """Limiar exato de ``Q``: vale ``B``, e é contagem."""
        return self.butterflies_per_payload_byte


def dominance_thresholds(w: DiagnosisWorkload, params: PlatformParameters) -> DominanceThresholds:
    """Os limiares das duas desigualdades que definem a dominância do rádio."""
    return DominanceThresholds(
        payload_bytes=w.payload_bytes,
        conversions_per_payload_byte=w.conversions_per_payload_byte,
        butterflies_per_payload_byte=w.butterflies_per_payload_byte,
        idle_conversions_equivalent=idle_conversions_equivalent(w, params),
    )
