"""Node energy model: µJ per diagnosis, broken down into three parts.

It is the project's declared system metric — µJ per diagnosis and BLE bytes per
diagnosis, never latency. This module is a **model**, not a measurement: bench
Measurement 2 (`bench/protocols/`, Power Profiler Kit II or shunt + oscilloscope) is
what validates or refutes each coefficient.

Two things are deliberately kept separate here, and the separation is the point of
the module:

``DiagnosisWorkload``
    **Pure counting.** How many conversions, how many FFT butterflies, how many
    payload bytes a diagnosis requires. None of this depends on a datasheet — it
    follows from the sampling rate, the window and the payload policy. It is always
    known.

``PlatformParameters``
    **Platform coefficients.** µJ per conversion, µJ per cycle, µJ per radio byte.
    Each field carries mandatory provenance, and the value may be ``None``.

``None`` means *unknown*, and unknown **propagates**: any part that depends on a
missing coefficient comes out ``None`` and the total comes out ``None``. A silent
zero in that place would produce a false energy number, which is exactly the result
this project cannot publish.

Provenance of what is known about the platform
----------------------------------------------
The only platform document in the repository is ``literature/pdf/nordic2024nrf54l15.pdf``.
It has 6 pages and is the **overview page** of the nRF54L15/L10/L05, not the full v0.9
datasheet it references. Verified in this session with::

    uv run --with pypdf python literature/read_pdf.py nordic2024nrf54l15 \\
        --grep "SAADC" "current" "consumption" "mA" "uA" "BLE"

Result: ``SAADC`` and ``current`` have **zero** occurrences; ``consumption`` appears
only in marketing prose ("ultra-low power consumption"); ``mA`` only matches inside
words. There is no current, energy per conversion, energy per radio byte,
track-and-hold bandwidth or jitter in that file. All those fields stay ``None``.

The bits × rate axis
--------------------
The page declares three ADC operating points — 14 bit at 31.25 kSPS, 12 bit at
250 kSPS and up to 10 bit at 2 MSPS. This is the axis that separates this thesis from
``vol2024poweraware``, which learns the quantization regions (**bits**) and does not
touch the **rate**. Here the rate is the variable, and lowering it moves the converter
to an operating point with more bits: at 31.25 kSPS and below, the same peripheral
delivers 14 bits instead of 12. See ``operating_point_for``.

Dominance is a comparison of three parts, not of two
----------------------------------------------------
``EnergyBreakdown.dominant_part`` chooses among acquisition, processing and radio.
So "the radio dominates" is **two** inequalities, and ``DominanceThresholds``
publishes both, with the idle-term correction included::

    radio > acquisition  ⟺  R > A + S / payload_bytes
    radio > processing   ⟺  Q > B

``A = n_conversions / payload_bytes`` and ``B = n_butterflies / payload_bytes`` are
counts. ``R``, ``Q`` and ``S`` are ratios of platform coefficients, all of them
currently unknown. Writing the criterion as ``R > A`` would compare two parts out of
three and, on top of that, erase the idle term that ``acquisition_energy_uj`` adds.
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

#: A coefficient may be missing. ``None`` is the only accepted marker of unknown.
Maybe: TypeAlias = float | None

#: Mandatory provenance text for every field whose value is unknown. The literal stays
#: in Portuguese ("not located - pending the full v0.9 datasheet or Measurement 2"):
#: E06 repeats it verbatim in its config and results, and flags unfilled provenance by
#: the substring ``nao localizado``.
UNKNOWN_SOURCE = "nao localizado - pendente do datasheet v0.9 completo ou da Medida 2"

_OVERVIEW = "nordic2024nrf54l15.pdf (overview page, 6 pp.)"


# ----------------------------------------------------------------------------
# converter operating points - the bits x rate axis
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class AdcOperatingPoint:
    """A (resolution, maximum rate) pair declared by the manufacturer."""

    n_bits: int
    max_rate_sps: float
    source: str


#: Verbatim from the page that was read: "ADC with up to eight programmable gain
#: channels. 14-bit at 31.25 ksps, 12-bit at 250 ksps, and up to 10-bit at 2 Msps." (p. 5).
DATASHEET_OVERVIEW_ADC_POINTS: tuple[AdcOperatingPoint, ...] = (
    AdcOperatingPoint(14, 31_250.0, f"{_OVERVIEW} p.5, '14-bit at 31.25 ksps'"),
    AdcOperatingPoint(12, 250_000.0, f"{_OVERVIEW} p.5, '12-bit at 250 ksps'"),
    AdcOperatingPoint(10, 2_000_000.0, f"{_OVERVIEW} p.5, 'up to 10-bit at 2 Msps'"),
)

#: BLE 6.0 PHY rates declared on the same page (p. 4). They enter the model only as
#: context: without µJ per byte, a PHY rate does not turn into energy.
BLE_PHY_RATES_BPS: tuple[float, ...] = (2e6, 1e6, 500e3, 125e3)


def operating_point_for(fs_hz: float) -> AdcOperatingPoint | None:
    """Operating point of **highest resolution** that sustains ``fs_hz``.

    Returns ``None`` if no declared point reaches the requested rate. This is the
    useful reading of the bits × rate axis: lowering the sampling rate is not just
    saving conversions, it is moving up in resolution on the same peripheral.
    """
    viable = [p for p in DATASHEET_OVERVIEW_ADC_POINTS if fs_hz <= p.max_rate_sps]
    return max(viable, key=lambda p: p.n_bits) if viable else None


# ----------------------------------------------------------------------------
# platform coefficients
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PlatformParameters:
    """Energy coefficients of the node, each with declared provenance.

    ``sources`` must cover **all** numeric fields; the constructor rejects the
    instance if provenance is missing for any of them. This is intentional: a
    coefficient without a declared origin is indistinguishable from a guess once it
    enters a spreadsheet.
    """

    energy_per_conversion_uj: Maybe
    """Energy of one SAADC conversion, end to end (sampling + SAR)."""

    current_active_ma: Maybe
    """Current of the Cortex-M33 while processing at 128 MHz."""

    current_sleep_ua: Maybe
    """Current between conversions, with DPPI/TIMER waking the SAADC and the CPU asleep."""

    current_radio_tx_ma: Maybe
    """Current of the radio while transmitting. Context: the model uses µJ per byte."""

    energy_per_ble_byte_uj: Maybe
    """Energy per delivered BLE payload byte, already amortizing connection and overhead."""

    supply_voltage_v: Maybe
    """Voltage of the chosen operating point. The range is known, the point is not."""

    cycles_per_butterfly: Maybe
    """Cycles per radix-2 butterfly in the real FFT, with the M33 DSP instructions."""

    cpu_clock_hz: Maybe
    """Core clock."""

    supply_voltage_min_v: Maybe
    supply_voltage_max_v: Maybe
    ram_bytes: Maybe
    """Budget context: the window buffer has to fit here."""

    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        missing = [f.name for f in self._numeric_fields() if f.name not in self.sources]
        if missing:
            raise ValueError(f"missing provenance for: {', '.join(missing)}")

    @classmethod
    def _numeric_fields(cls) -> tuple:
        return tuple(f for f in fields(cls) if f.name != "sources")

    def unknown_parameters(self) -> tuple[str, ...]:
        """Names of the unknown coefficients, in declaration order."""
        return tuple(f.name for f in self._numeric_fields() if getattr(self, f.name) is None)

    def source_of(self, name: str) -> str:
        return self.sources[name]

    def summary(self) -> str:
        unknown = self.unknown_parameters()
        return (
            f"{len(self._numeric_fields()) - len(unknown)} known, "
            f"{len(unknown)} unknown: {', '.join(unknown) if unknown else '-'}"
        )


def datasheet_overview_parameters() -> PlatformParameters:
    """What the nRF54L15 overview page supports — and only that.

    Every energy field comes out ``None``. It is not an omission: it is the actual
    state of the evidence available in the repository, and the model has to run
    like this.
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
                f"{_OVERVIEW} p.5 declares the range '1.7 V to 3.5 V supply'; "
                "the operating point is not declared - nao localizado"
            ),
            "cycles_per_butterfly": (
                "depends on the chosen FFT implementation and on the M33; nao localizado"
            ),
            "cpu_clock_hz": f"{_OVERVIEW} p.4, '128 MHz Arm Cortex-M33'",
            "supply_voltage_min_v": f"{_OVERVIEW} p.5, '1.7 V to 3.5 V supply and I/O voltage'",
            "supply_voltage_max_v": f"{_OVERVIEW} p.5, '1.7 V to 3.5 V supply and I/O voltage'",
            "ram_bytes": f"{_OVERVIEW} p.5, 'nRF54L15 - ... and 256 KB RAM'",
        },
    )


def energy_per_cycle_uj(params: PlatformParameters) -> Maybe:
    """µJ per CPU cycle, from ``I·V/f``. ``None`` if any of the three is missing."""
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
# counting: what a diagnosis requires, without any platform number
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DiagnosisWorkload:
    """Pure counting for one diagnosis. Independent of any datasheet and never unknown."""

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
        """``A`` — the count that gives the **floor** of the ``R`` threshold, not the threshold.

        The radio part exceeds the acquisition part when
        ``R := energy_per_ble_byte_uj / energy_per_conversion_uj`` exceeds
        ``A + S / payload_bytes``, where ``S`` is the idle term converted into
        equivalent conversions (see :func:`idle_conversions_equivalent`). Since
        ``S >= 0``, this number alone only supports the countable half of the
        statement: ``R <= A`` **guarantees** that the radio does not exceed the
        acquisition; ``R > A`` does not guarantee the opposite while ``S`` is unknown.
        """
        return math.inf if self.payload_bytes == 0 else self.n_conversions / self.payload_bytes

    @property
    def butterflies_per_payload_byte(self) -> float:
        """``B`` — the **exact** threshold of the radio × processing comparison.

        The radio part exceeds the processing part when
        ``Q := energy_per_ble_byte_uj / energy_per_butterfly_uj`` exceeds this
        number. It is also pure counting, and it is the second inequality without which
        "the radio dominates" is not established: dominating means being the largest
        of the **three** parts.
        """
        return math.inf if self.payload_bytes == 0 else self.n_butterflies / self.payload_bytes


def _fft_size_for(n_samples: int) -> int:
    """Largest power of two that fits in ``n_samples``."""
    if n_samples < 2:
        return 0
    return 1 << (n_samples.bit_length() - 1)


def workload(
    fs_hz: float,
    window_s: float,
    payload_bytes: int,
    fft_size: int | None = None,
) -> DiagnosisWorkload:
    """Builds the count for one diagnosis from the rate, the window and the payload.

    If ``fft_size`` is omitted, the largest power of two that fits in the window is
    used. The butterfly count is that of a radix-2 FFT, ``(N/2)·log2 N``, which is
    the dominant cost of the envelope → order spectrum chain used in E07.
    """
    if fs_hz <= 0.0 or window_s <= 0.0:
        raise ValueError("fs_hz and window_s must be positive")
    if payload_bytes < 0:
        raise ValueError("payload_bytes cannot be negative")

    n_conversions = int(round(fs_hz * window_s))
    size = _fft_size_for(n_conversions) if fft_size is None else int(fft_size)
    if size > n_conversions:
        raise ValueError(f"fft_size {size} larger than the window's {n_conversions} conversions")
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
# composition of the three parts
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyBreakdown:
    """The three parts of one diagnosis. ``None`` in any of them means unknown."""

    acquisition_uj: Maybe
    processing_uj: Maybe
    radio_uj: Maybe

    _PARTS = ("acquisition_uj", "processing_uj", "radio_uj")

    @property
    def total_uj(self) -> Maybe:
        """Sum of the three, or ``None`` if any of them is unknown.

        Summing while treating unknown as zero would return a presentable and wrong
        floor. The total only exists when all three parts exist.
        """
        parts = [getattr(self, p) for p in self._PARTS]
        return None if any(v is None for v in parts) else float(sum(parts))  # type: ignore[arg-type]

    def unknown_parts(self) -> tuple[str, ...]:
        return tuple(p for p in self._PARTS if getattr(self, p) is None)

    def dominant_part(self) -> str | None:
        """Name of the dominant part, or ``None`` if any part is unknown.

        The E06 hypothesis is about *which* part dominates. Answering that with a part
        missing would be answering about a subset and calling it a conclusion.
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
    """Idle energy of the window: ``current_sleep_ua · supply_voltage_v · window_s``.

    It exists because the node architecture is DPPI + TIMER → SAADC with the CPU
    asleep: lowering the rate reduces the conversions but does **not** shorten the
    window, so this part does not scale with the factor. That is why it needs a name
    of its own — any threshold that ignores it is comparing a model different from
    the one that runs.
    """
    if params.current_sleep_ua is None or params.supply_voltage_v is None:
        return None
    # uA * V * s = uJ
    return params.current_sleep_ua * params.supply_voltage_v * w.window_s


def acquisition_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """Conversions × energy per conversion, plus the idle energy over the window."""
    if params.energy_per_conversion_uj is None:
        return None
    idle = idle_energy_uj(w, params)
    if idle is None:
        return None
    return w.n_conversions * params.energy_per_conversion_uj + idle


def energy_per_butterfly_uj(params: PlatformParameters) -> Maybe:
    """µJ of one radix-2 butterfly: cycles per butterfly × µJ per cycle."""
    per_cycle = energy_per_cycle_uj(params)
    if per_cycle is None or params.cycles_per_butterfly is None:
        return None
    return params.cycles_per_butterfly * per_cycle


def processing_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """Butterflies × energy per butterfly."""
    per_butterfly = energy_per_butterfly_uj(params)
    if per_butterfly is None:
        return None
    return w.n_butterflies * per_butterfly


def radio_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """Payload bytes × energy per delivered byte."""
    if params.energy_per_ble_byte_uj is None:
        return None
    return w.payload_bytes * params.energy_per_ble_byte_uj


def diagnosis_energy_uj(w: DiagnosisWorkload, params: PlatformParameters) -> EnergyBreakdown:
    """The three parts of one diagnosis, each of which may come out unknown."""
    return EnergyBreakdown(
        acquisition_uj=acquisition_energy_uj(w, params),
        processing_uj=processing_energy_uj(w, params),
        radio_uj=radio_energy_uj(w, params),
    )


# ----------------------------------------------------------------------------
# the dominance region - two inequalities, three coefficient ratios
# ----------------------------------------------------------------------------
def idle_conversions_equivalent(w: DiagnosisWorkload, params: PlatformParameters) -> Maybe:
    """``S`` — the idle energy measured in equivalent conversions.

    ``S := idle_energy_uj / energy_per_conversion_uj``. It is dimensionless and it is
    the model's third coefficient ratio: it shifts the ``R`` threshold upward, and it
    is exactly the term that a threshold written as ``R > A`` omits.
    """
    idle = idle_energy_uj(w, params)
    if idle is None or params.energy_per_conversion_uj is None:
        return None
    if params.energy_per_conversion_uj <= 0.0:
        return None
    return idle / params.energy_per_conversion_uj


@dataclass(frozen=True)
class DominanceThresholds:
    """Where, in coefficient space, the radio part dominates the other **two**.

    ``dominant_part`` decides among three parts, so "the radio dominates" is two
    simultaneous inequalities, not one:

    * radio > acquisition  ⟺  ``R > A + S / payload_bytes``
    * radio > processing   ⟺  ``Q > B``

    with ``R := energy_per_ble_byte_uj / energy_per_conversion_uj``,
    ``Q := energy_per_ble_byte_uj / energy_per_butterfly_uj`` and ``S`` given by
    :func:`idle_conversions_equivalent`. ``A`` and ``B`` are **pure counting**;
    ``R``, ``Q`` and ``S`` are ratios of platform coefficients.

    What counting alone decides, without any coefficient: ``R <= A`` implies that the
    radio does **not** exceed the acquisition (because ``S >= 0``), and ``Q <= B``
    implies that it does not exceed the processing. The converse depends on ``S``,
    and ``S`` is unknown — that is why ``radio_over_acquisition_exact`` comes out
    ``None``.
    """

    payload_bytes: int
    conversions_per_payload_byte: float
    """``A``."""

    butterflies_per_payload_byte: float
    """``B`` — exact threshold of ``Q``, with no correction pending."""

    idle_conversions_equivalent: Maybe
    """``S``, or ``None`` while an idle or conversion coefficient is missing."""

    @property
    def radio_over_acquisition_min(self) -> float:
        """Floor of the ``R`` threshold: it equals ``A``, and it is a count."""
        return self.conversions_per_payload_byte

    @property
    def radio_over_acquisition_exact(self) -> Maybe:
        """Exact threshold of ``R``, ``A + S/payload_bytes``. ``None`` if ``S`` is missing."""
        if self.idle_conversions_equivalent is None or self.payload_bytes <= 0:
            return None
        return self.conversions_per_payload_byte + self.idle_conversions_equivalent / (
            self.payload_bytes
        )

    @property
    def radio_over_processing(self) -> float:
        """Exact threshold of ``Q``: it equals ``B``, and it is a count."""
        return self.butterflies_per_payload_byte


def dominance_thresholds(w: DiagnosisWorkload, params: PlatformParameters) -> DominanceThresholds:
    """The thresholds of the two inequalities that define radio dominance."""
    return DominanceThresholds(
        payload_bytes=w.payload_bytes,
        conversions_per_payload_byte=w.conversions_per_payload_byte,
        butterflies_per_payload_byte=w.butterflies_per_payload_byte,
        idle_conversions_equivalent=idle_conversions_equivalent(w, params),
    )
