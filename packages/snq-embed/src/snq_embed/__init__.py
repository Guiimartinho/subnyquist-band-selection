"""snq-embed - quantization, C generation and budgeting for the MCU.

Leaf package: nothing in the project imports from here.

Implemented: energy model (`energy`) - microjoules per diagnosis broken down into
acquisition, processing and radio, with mandatory provenance per coefficient and
explicit propagation of unknown values. It is a model, not a measurement: bench
Measurement 2 validates it.

To be implemented: quantize_int8, export_c_kernel, ram_budget, cycle_estimate.
Target: nRF54L15, Cortex-M33 @128 MHz, 256 kB RAM, 1.5 MB NVM.
Reported metric: microjoules per diagnosis and BLE bytes per diagnosis.
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
