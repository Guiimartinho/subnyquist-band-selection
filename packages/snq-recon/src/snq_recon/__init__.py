"""snq-recon - algoritmos classicos de reconstrucao e estimacao.

Baseline oficial do projeto: amostragem passa-faixa + analise de envelope.
Se um metodo aprendido nao bate isso, ele nao existe.

A ser implementado, em ordem de prioridade:
  bandpass_envelope   baseline; demodulacao apos dobra espectral
  kurtogram           selecao da banda ressonante a subamostrar (Antoni 2007)
  prony / esprit      estimacao de linhas espectrais, decodificador FRI
  robust_crt          desambiguacao da ordem do alias por taxas coprimas
  omp / fista         reconstrucao esparsa
"""

from snq_recon.diagnosis import (
    Diagnosis,
    FaultScore,
    diagnose,
    estimate_from_candidates,
    estimate_without_prior,
    score_fault,
)
from snq_recon.envelope import (
    EnvelopeSpectrum,
    bandpass_analytic,
    envelope,
    envelope_headroom_ok,
    envelope_spectrum,
    squared_envelope_spectrum,
)
from snq_recon.estimation import (
    alias_candidates_hz,
    estimate_tone_hz,
    prior_grid_hz,
    resolve_multirate_hz,
    unambiguous_range_hz,
)
from snq_recon.highres import (
    angular_resample,
    envelope_order_spectrum,
    esprit,
    estimate_tone_esprit,
    fault_orders,
    order_spectrum,
)
from snq_recon.kurtogram import (
    KurtogramResult,
    band_from_kurtogram,
    fast_kurtogram,
    spectral_kurtosis,
)

__all__ = [
    "Diagnosis",
    "EnvelopeSpectrum",
    "FaultScore",
    "KurtogramResult",
    "alias_candidates_hz",
    "angular_resample",
    "envelope_order_spectrum",
    "esprit",
    "estimate_tone_esprit",
    "fault_orders",
    "order_spectrum",
    "band_from_kurtogram",
    "bandpass_analytic",
    "diagnose",
    "envelope",
    "envelope_headroom_ok",
    "envelope_spectrum",
    "estimate_from_candidates",
    "estimate_without_prior",
    "fast_kurtogram",
    "score_fault",
    "spectral_kurtosis",
    "squared_envelope_spectrum",
    "estimate_tone_hz",
    "prior_grid_hz",
    "resolve_multirate_hz",
    "unambiguous_range_hz",
]
