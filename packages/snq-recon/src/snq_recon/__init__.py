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

__all__: list[str] = []
