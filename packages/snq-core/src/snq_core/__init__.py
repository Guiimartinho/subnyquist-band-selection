"""snq-core — mathematics of structured sampling and physical priors.

Imports nothing from the project's other packages. The dependency arrows never
reverse.
"""

from snq_core.bearing import (
    CWRU_6205_2RS_JEM_SKF,
    BearingGeometry,
    candidate_lines_hz,
    fault_frequencies_hz,
)
from snq_core.jitter import (
    max_input_frequency_hz,
    phase_error_rad,
    quantization_snr_db,
    snr_limit_db,
)
from snq_core.sampling import (
    FoldedBand,
    alias_frequency_hz,
    alias_order,
    coset_matrix,
    fold_band,
    landau_rate_hz,
    min_bandpass_rate_hz,
    nyquist_rate_hz,
    nyquist_zone,
    uniform_coset_offsets_s,
    valid_bandpass_rates_hz,
)

__all__ = [
    "CWRU_6205_2RS_JEM_SKF",
    "BearingGeometry",
    "FoldedBand",
    "alias_frequency_hz",
    "alias_order",
    "candidate_lines_hz",
    "coset_matrix",
    "fault_frequencies_hz",
    "fold_band",
    "landau_rate_hz",
    "max_input_frequency_hz",
    "min_bandpass_rate_hz",
    "nyquist_rate_hz",
    "nyquist_zone",
    "phase_error_rad",
    "quantization_snr_db",
    "snr_limit_db",
    "uniform_coset_offsets_s",
    "valid_bandpass_rates_hz",
]
