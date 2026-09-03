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

from snq_recon.estimation import (
    alias_candidates_hz,
    estimate_tone_hz,
    prior_grid_hz,
    resolve_multirate_hz,
    unambiguous_range_hz,
)

__all__ = [
    "alias_candidates_hz",
    "estimate_tone_hz",
    "prior_grid_hz",
    "resolve_multirate_hz",
    "unambiguous_range_hz",
]
