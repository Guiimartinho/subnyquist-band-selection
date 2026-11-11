# Rodada com mask.lr_multiplier = 10 (2026-11-11, 14:35–21:05 UTC): SELADA pela regra B5

A regra B5, pré-registrada no README, foi aplicada sobre o `summary.csv`. A fração de divisões em
que a máscara de `dec_aprendida` saiu do ruído de Gumbel (`mask_converged_frac`), na variante bruta,
foi:

| orçamento | 1/4 | 1/8 | 1/16 | 1/32 |
|---|---|---|---|---|
| `mask_converged_frac` | 0,6 | **0,4** | **0,4** | **0,2** |

A fração ficou abaixo de 0,5 em três orçamentos, e a regra manda refazer a rodada com
`mask.lr_multiplier = 30` antes de abrir as hipóteses. Por isso, **`hypotheses.csv` e
`comparisons.csv` desta rodada não foram abertos**. Os arquivos ficam aqui como registro do que
rodou, e não entram em nenhuma análise.

Lidos para aplicar a regra: só `summary.csv` (coluna `mask_converged_frac`) e os diagnósticos
`converged` de `masks.json`.
