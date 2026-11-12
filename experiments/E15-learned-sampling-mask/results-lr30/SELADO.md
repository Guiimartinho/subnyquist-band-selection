# Rodada com `mask.lr_multiplier = 30`: SELADA, sem análise

- **Período:** de 2026-11-11, 21:07 UTC, até a interrupção em 2026-11-12, 01:23 UTC.
- **Regra B5 na variante bruta:** a `mask_converged_frac` de `dec_aprendida` foi 1,0 / 1,0 /
  **0,4** / 0,6 de 1/4 a 1/32. 1/16 ficou abaixo de 0,5.
- **Decisão do usuário, antes de abrir qualquer número de teste:** a rodada corrigida
  (`stop_after_anneal`) decide H1 e H2. Esta foi interrompida na metade `sem_picos`, que não
  decide nada.
- **Arquivos:** como não chegou ao fim, a rodada não gerou `metrics.csv`, `summary.csv`,
  `hypotheses.csv` nem `comparisons.csv`. O cache por execução (225 arquivos em `runs/`)
  ficou no servidor do grupo, na árvore do E15, sem análise.
- **Aqui há só `convergencia.txt`:** diagnósticos de treino da máscara aprendida
  (convergência, margem, melhor época, épocas, lr e posições). Nenhum número de teste.
- **Outro diagnóstico de treino, que motivou a regra de avaliabilidade da H2:** a máscara
  livre (`dec_livre`, `dec_livre_projetada`) saiu do ruído em 1/5, 1/5, 1/5 e 0/5 das divisões.
- **Número de teste visto, por descuido, durante o acompanhamento.** As linhas de progresso do
  log trazem acertos de teste por divisão. Às 22:33 UTC, duas delas foram exibidas: em 1/32,
  na divisão de índice 1, `dec_multicoset_vb` com 8/12 e `dec_uniforme_treino` com 9/12 pela
  régua local.
  - Eram de baselines, e nenhuma da máscara aprendida.
  - Todas as regras da rodada corrigida foram fixadas em código e commitadas antes de
    qualquer número dela.
  - Desde então, o acompanhamento extrai do log só a posição.
