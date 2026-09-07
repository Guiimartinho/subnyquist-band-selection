"""snq-embed - quantizacao, geracao de C e orcamento para o MCU.

Pacote folha: nada do projeto importa daqui.

Implementado: modelo de energia (`energy`) - microjoules por diagnostico decomposto
em aquisicao, processamento e radio, com procedencia obrigatoria por coeficiente e
propagacao explicita de valor desconhecido. E modelo, nao medicao: a Medida 2 de
bancada valida.

A ser implementado: quantize_int8, export_c_kernel, ram_budget, cycle_estimate.
Alvo: nRF54L15, Cortex-M33 @128 MHz, 256 kB RAM, 1.5 MB NVM.
Metrica reportada: microjoules por diagnostico e bytes de BLE por diagnostico.
"""

from snq_embed.energy import (
    BLE_PHY_RATES_BPS,
    DATASHEET_OVERVIEW_ADC_POINTS,
    UNKNOWN_SOURCE,
    AdcOperatingPoint,
    DiagnosisWorkload,
    DominanceThresholds,
    EnergyBreakdown,
    PlatformParameters,
    acquisition_energy_uj,
    datasheet_overview_parameters,
    diagnosis_energy_uj,
    dominance_thresholds,
    energy_per_butterfly_uj,
    energy_per_cycle_uj,
    idle_conversions_equivalent,
    idle_energy_uj,
    operating_point_for,
    processing_energy_uj,
    radio_energy_uj,
    workload,
)

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
