# E15 - Máscara de amostragem aprendida sob conjunto viável

Número reservado no mapa de experimentos como "política de amostragem aprendida, realizável em
hardware". Aberto em 2026-11-11 pelo [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md),
como a Parte 2 (método) do artigo único. Roda na GPU do servidor.

## Pergunta

No front-end A (sem filtro analógico sintonizável) e com o mesmo orçamento de amostras/s, uma
máscara de amostragem **aprendida contra a tarefa** supera os baselines publicados do E22? A
máscara é restrita a um conjunto viável: padrão periódico sobre a grade de ticks, com
espaçamento mínimo entre conversões.

## Conjunto viável

- **Grade:** a taxa de aquisição do conjunto (Ottawa 200 kSPS; MAFAULDA 51,2 kSPS) é a grade de
  ticks.
- **Padrão periódico** de período L ticks, com K posições ativas por período. A taxa média é
  K/L × fs, e com K/L = 1/orçamento o custo bate com o do E22. Isso é amostragem periódica
  não-uniforme (multi-coset), realizável com TIMER e DPPI.
- **Espaçamento mínimo** de g ticks entre posições ativas, que representa o tempo de conversão
  de um único SAR.

## Modelo

**O decodificador não é uma CNN genérica.** É o pipeline clássico desdobrado e diferenciável:
1. o sinal mascarado e preenchido com zeros;
2. um banco de filtros complexos aprendíveis, que pode representar reconstrução multi-coset
   linear de uma banda;
3. o envelope ao quadrado;
4. a reamostragem angular pelo tacômetro ou encoder, por interpolação diferenciável;
5. o espectro de ordem.

A estimativa é o mesmo `estimate_without_prior` do E22 (argmax da ordem), então os números se
comparam com o E22.

**Perda:** entropia cruzada sobre bins de ordem, com alvo suave na ordem verdadeira (largura da
tolerância de 0,06). É orientada à tarefa, como em `shlezinger2022taskadc`.

**Máscara:** logits por posição do período, top-K de Gumbel com straight-through. A construção
vem de `huijben2020dps` (DPS), e o ADPS (`vangorp2021adps`) a estende ao caso adaptativo. O
espaçamento mínimo entra por penalidade durante o treino e por projeção exata no fim. A pergunta
do gap de projeção (treinar sob a restrição contra projetar depois) tem precedente em MRI,
`weiss2021pilot` (PILOT).

## Baselines

Todos são reimplementados, no mesmo protocolo e no mesmo orçamento. Tabela revista em 2026-11-11
pela revisão adversarial; os nomes são os da coluna `arm` do `metrics.csv`.

| braço | referência | o que isola |
|---|---|---|
| `uniforme_treino`: melhor uniforme escolhida fora do teste | E22 | o melhor clássico realizável |
| `dec_uniforme_treino`: mesmo decodificador, uniforme nesse fator | E07, E22 | decodificador contra clássico, na mesma taxa |
| `dec_uniforme`: mesmo decodificador, uniforme no orçamento | E07, E22 | ganho de aprender o padrão |
| `dec_aleatoria`: máscara aleatória fixa, semeada | `sensors2025csvibration` | aprendido contra aleatório fixo |
| `dec_multicoset`: multi-coset de menor alias de pico | `zhao2018coprime`, `sun2013multirate` | aprendido contra projetado sem dados |
| `dec_multicoset_vb`: multi-coset para suporte conhecido | `venkataramani2001optimal` | aprendido contra projetado com o suporte |
| `dec_livre_projetada`: top-K sem restrição, depois projetado | `huijben2020dps`, `weiss2021pilot` | o **gap de projeção** (H2) |
| `dec_loupe_*` (opcional): LOUPE, depois projetado | `bahadir2020loupe` | o mesmo, com outro amostrador |
| `dec_adaptativa_blocos`: adaptativo por blocos inspirado no ADPS | `vangorp2021adps` | adaptativo contra fixo |
| `wdcnn`: rede genérica sobre o sinal uniformemente decimado | `zhang2017wdcnn` | rede genérica de diagnóstico |
| `uniforme_orcamento`, `oraculo_uniforme` | E22 | referência; o oráculo usa o rótulo de teste |

## Protocolo de treino

- **Épocas:** teto de **500**, com **early stopping por paciência** (40 épocas) na perda de
  validação. O melhor checkpoint vale.
- **Validação separada por registro**, sem vazamento de segmentos. Divisões repetidas pelo
  arcabouço do E05.
- **Repetições:** 5 divisões por registro, com uma semente de treino cada (ver o pré-registro).
- **Ajuste:** o mesmo orçamento de hiperparâmetros para todos os métodos aprendidos (uma taxa de
  aprendizado, sem busca).
- **Conjuntos:** Ottawa decide; ver as alterações e correções abaixo.
- **Validação das reimplementações:** antes de comparar, cada baseline precisa reproduzir algo
  conhecido. Por exemplo, a WDCNN em taxa cheia no CWRU tem número conhecido na literatura, e o
  LOUPE tem de recuperar a máscara de baixa frequência num exemplo de brinquedo.

## Hipóteses

Texto corrigido em 2026-11-11. A versão anterior comparava com `taxa_tat` e com o oráculo.
`taxa_tat` não existe neste experimento, e o oráculo escolhe com o rótulo de teste. O teste exato
está no **pré-registro**, mais abaixo.

- **H1.** Em pelo menos dois orçamentos, a máscara aprendida (`dec_aprendida_sel`) tem acurácia
  balanceada de ordem maior que a de cada um destes três braços:
  - `uniforme_treino`, a melhor uniforme escolhida fora do teste;
  - `dec_uniforme_treino`, o mesmo decodificador com a máscara uniforme nesse fator;
  - `dec_aleatoria`, a máscara aleatória fixa.

  Vale só se as duas réguas concordarem.
- **H2.** O gap de projeção é maior que zero: treinar já sob a restrição (`dec_aprendida`) é
  melhor que treinar o top-K de Gumbel sem a restrição e projetar depois (`dec_livre_projetada`).

## O que refutaria

- **H1 é refutada** se o critério do pré-registro falhar em qualquer uma das réguas. Então a
  Parte 2 é resultado negativo, e o artigo a reporta como tal, como manda o ADR 0006.
- **H2 é refutada** se o critério do pré-registro falhar. Nesse caso o conjunto viável não é
  contribuição.

## Alterações de protocolo (2026-11-11)

Pedidas pela coordenação depois de o E21 e o E22 rodarem. O desenho acima fica como registro.
Dois itens foram superados pela revisão adversarial da seção seguinte, e estão marcados.

- **Conjuntos.** Ottawa (pistas interna e externa, com encoder) e CWRU 48k DE (rotação constante
  pelo rpm do registro, triagem `smith2015cwru`). **Superado:** o CWRU saiu da decisão (A2).
  - O MAFAULDA sai e fica no config como opção desligada. O E21 mostrou que, nesse rolamento,
    BPFO = 2,998 e BPFI = 5,002 coincidem com o 3º e o 5º harmônicos do eixo, e aí o "acerto"
    passa a medir o harmônico do eixo.
  - Não há teste fora do conjunto de treino: as grades de ticks dos conjuntos não são a mesma, e
    a máscara não se transfere sem reamostrar.
- **Orçamentos.** Ottawa: 1/4, 1/8, 1/16 e 1/32; 1/48 e 1/64 são opcionais. De 96× a 160× não
  servem de alvo, porque ali a parede é do pipeline, não da taxa.
- **Baseline que decide:** `uniforme_treino`, a melhor uniforme escolhida nos registros fora do
  teste da mesma divisão. É um fator por conjunto, porque o nó não sabe a família, e o empate vai
  para a maior taxa.
- **Verdade.** Centro do platô de escore (`snq_recon.refine_order`), e não o primeiro ponto, que
  fica cerca de 0,0175 ordem abaixo. O primeiro ponto vai para o manifest. **Superado:** a régua
  única por registro deu lugar às duas réguas (A1).
- **Picos isolados de uma amostra no Ottawa** (cerca de 1300/s, em todos os registros). Os
  resultados saem nas variantes `bruto` e `sem_picos`, pelo critério do E20/E22.
- **Piso do seletor.** No decodificador aprendido, o piso de 500 Hz do kurtograma só inicializa o
  banco. Nos braços clássicos, nas grades usadas, a guarda do piso não dispara. Em fator alto,
  porém, o piso estreita a faixa de busca; isso é parede do pipeline.

## Correções da revisão adversarial (2026-11-11)

A primeira rodada completa foi interrompida antes de terminar. Não houve vazamento do teste, e o
`uniforme_treino` estava correto, mas a rodada não valia por duas falhas (A1 e A2). A chave de
retomada mudou, e a rodada recomeça do zero.

- **A1. A raia da pista externa muda com a rotação dentro do registro.** A coordenação mediu, com
  `refine_order` em taxa cheia por segmento: O-A-1 vai de 3,519 a 16,7 Hz até 3,322 a 25,8 Hz, e
  a pista interna fica estável (variação ≤ 0,045). Um rótulo constante por registro errava a raia
  em boa parte das janelas da pista externa. Correção:
  - **régua local**: `refine_order` em taxa cheia numa grade deslizante de janelas de 2,5 s, com
    passo de 0,25 s, sobre os registros de treino e de validação. Cada janela de treino recebe o
    rótulo do próprio trecho, e o trecho com platô na borda sai do treino;
  - a mesma régua escolhe o `uniforme_treino`, nos segmentos dos registros fora do teste;
  - na avaliação saem **duas réguas lado a lado**: a raia em taxa cheia do próprio segmento de
    teste, só para pontuar e nunca para escolher, e a mediana aninhada do segmento 0.
- **A2. O CWRU 48k não decide nada como estava:** 12 registros de pista externa contra 1 de
  interna, que nunca vai a teste; classe única no teste; 3 unidades físicas. Escolha: **(b), sair
  da decisão.** O adaptador fica, desligado no config. A opção (a), detecção de BPFO de classe
  única, responderia outra pergunta e não teria intervalo por registro.
- **B1. A comparação que decide misturava decodificador, máscara e taxa.**
  - Entra `dec_uniforme_treino`: o mesmo decodificador com a máscara uniforme no fator escolhido
    fora do teste.
  - O braço aprendido decisivo, `dec_aprendida_sel`, escolhe pela validação entre os orçamentos
    treinados ≥ b. O critério é a acurácia balanceada pela régua local, com desempate pela menor
    perda e depois pela maior taxa.
- **B2. Mesmo orçamento de estágios.** Toda máscara buscada vira um `FixedMask`, e o
  decodificador é retreinado do zero com ela, no mesmo protocolo dos braços fixos. A versão do
  estágio de busca sai como `_sem_ajuste`.
- **B3. LOUPE.**
  - No forward relaxado, a taxa vista no treino era 1,5 a 2,4 vezes a do orçamento. O gap de
    projeção passa a usar o top-K de Gumbel sem a restrição (`dec_livre`): horizonte de 4
    períodos, contagem só no horizonte e nenhum espaçamento. Ele é projetado exatamente com
    `fold_project`.
  - `masks.json` grava a fração de máscaras duras que violam g e a projeção do mapa inicial
    contra a final.
  - O LOUPE fica opcional, agora com forward duro e straight-through.
- **B4. Estatística.** Está pré-registrada na seção seguinte, antes de qualquer número.
- **B5. A máscara podia não sair do ruído.**
  - Logits iniciais de 0,01·randn e taxa de 0,001 a 0,003 com 8 passos por época, contra ruído de
    Gumbel de desvio próximo de 1,28. Agora a máscara tem taxa própria: 10 vezes a do
    decodificador (`mask.lr_multiplier`).
  - `masks.json` grava os logits finais, a margem entre o K-ésimo e o (K+1)-ésimo logit,
    `last_hard_violations`, o `best_epoch` e as posições finais contra a projeção dos logits
    iniciais, que a semente reproduz.
- **B6.** O braço adaptativo se chama "adaptativo por blocos inspirado no ADPS"
  (`dec_adaptativa_blocos`). Como ele tem um estágio só, é comparado com
  `dec_aprendida_sem_ajuste`.
- **B7.** Entra `dec_multicoset_vb`, o multi-coset desenhado para o suporte conhecido
  (`venkataramani2001optimal`; DOI conferido no Crossref).
  - O suporte é a banda do kurtograma em taxa cheia mais frequente entre os registros fora do
    teste, encolhida até caber em K células.
  - O critério é a amplificação de ruído da reconstrução por mínimos quadrados. É uma leitura no
    espírito do artigo, cuja cota exata **não foi conferida**, porque o resumo não estava
    disponível nas bases consultadas.

## Pré-registro (2026-11-11)

Escrito antes de qualquer número da rodada completa. O config espelha o texto no bloco
`preregistration`, e o manifest grava o sha256 deste README (`readme_sha256`). Mudar o texto ou o
bloco depois de ver um número invalida o pré-registro.

- **Conjunto e variante que decidem:** Ottawa, variante `bruto`. A `sem_picos` é secundária,
  porque remover picos exige a taxa cheia e não é realizável no nó.
- **Orçamentos:** 1/4, 1/8, 1/16 e 1/32.
- **Métrica:** acurácia balanceada de ordem, isto é, a média, sobre as duas pistas, da fração de
  segmentos de teste com |estimativa − régua| < 0,06 ordem.
- **Réguas.** Cada hipótese é testada nas duas, e só vale se as duas concordarem.
  - **Local:** a raia em taxa cheia do próprio segmento de teste.
  - **Aninhada:** a mediana das calibrações do segmento 0 dos registros fora do teste da mesma
    pista.
- **Unidade de reamostragem:** o aglomerado condição × cenário, 4 por pista (I-A a I-D e O-A a
  O-D). Também reamostra as 5 divisões.
  - Antes de reamostrar, cada (divisão, registro) é resumido pela média sobre os segmentos e as
    sementes.
  - Cada reamostra sorteia as divisões e, dentro de cada pista, os aglomerados, com reposição. O
    peso de um par é o produto das contagens.
  - São 4000 reamostras. O p unilateral é (1 + #{Δ* ≤ 0})/(1 + B).
  - **Elegibilidade:** o teste só vale com pelo menos 3 aglomerados por pista e 2 divisões entre
    os pares. Abaixo disso, todas as reamostras saem quase iguais, e o p viraria 1/(1 + B) sem
    medir variabilidade nenhuma; o teste recebe p = 1.
- **Semente e divisão.** Há uma semente de treino por divisão, então a variabilidade de semente
  fica confundida com a de divisão. O bootstrap reamostra as divisões, e o intervalo carrega as
  duas juntas. `protocol.seeds_per_split` separa as duas, ao custo de GPU.
- **Comparações decisivas e multiplicidade.**
  - H1: `dec_aprendida_sel` contra `uniforme_treino`, `dec_uniforme_treino` e `dec_aleatoria`, nos
    4 orçamentos. São 12 testes por régua, com correção de Holm sobre os 12.
  - H2: `dec_aprendida` contra `dec_livre_projetada`, nos 4 orçamentos. São 4 testes por régua,
    com Holm sobre os 4.
  - α = 0,05.
- **Regra de decisão.**
  - Um orçamento vence na H1 quando os três testes dele são significativos depois de Holm e com
    diferença positiva.
  - A H1 vale numa régua quando vencem pelo menos dois orçamentos; vale no experimento quando
    vale nas duas réguas.
  - A H2 segue a mesma regra, com um teste por orçamento.
  - `results/hypotheses.csv` traz cada teste, o p ajustado e o veredito.
- **Não decide nada:** as comparações de `results/comparisons.csv`, que saem sem correção, o
  oráculo, a variante `sem_picos`, a família de falha, a WDCNN e o controle saudável.
- **Convergência da máscara (B5), decidida só por diagnóstico de treino.** A máscara "saiu do
  ruído" quando a margem entre o K-ésimo e o (K+1)-ésimo logit passa do desvio do ruído de
  Gumbel, π/√6 ≈ 1,28 (`mask_converged`). O `summary.csv` traz a fração das divisões em que isso
  aconteceu (`mask_converged_frac`).
  - Se a fração ficar abaixo de 0,5 em `dec_aprendida`, na variante `bruto`, em qualquer
    orçamento, a rodada é refeita com `mask.lr_multiplier` = 30 **antes** de abrir
    `hypotheses.csv`.
  - A troca é registrada aqui. É uma decisão de otimização tomada pelo treino, não pelo teste.
  - **Aplicada em 2026-11-11.** Na rodada com `lr_multiplier = 10` (14:35–21:05 UTC), a
    `mask_converged_frac` de `dec_aprendida`, bruto, foi 0,6, 0,4, 0,4 e 0,2 de 1/4 a 1/32.
    A rodada ficou selada em `results-lr10/`, sem abrir `hypotheses.csv` nem `comparisons.csv`,
    e o config passou a `lr_multiplier = 30`.
  - **Rodada com `lr_multiplier = 30`** (iniciada às 21:07 UTC de 2026-11-11). Na variante bruta,
    a `mask_converged_frac` de `dec_aprendida` foi 1,0 / 1,0 / **0,4** / 0,6 de 1/4 a 1/32.
    - O pré-registro previa uma única refação, e ela não resolveu 1/16.
    - **Desvio declarado antes de abrir `hypotheses.csv`:** nesta rodada, a H1 de 1/16 fica
      **não avaliável** (a máscara não saiu do ruído). 1/4, 1/8 e 1/32 são avaliados como
      pré-registrado.
    - A pedido do usuário, a rodada vai até o fim; a variante sem picos é descritiva.
  - **Diagnóstico de treino** (só treino, nunca teste): as 8 divisões da variante bruta que não
    convergiram têm a melhor época em 22–84 e pararam por paciência antes do fim do recozimento
    da temperatura (`tau_anneal_epochs = 100`). As que convergiram, com uma exceção, treinaram
    além dele.
  - **Pré-registro da rodada seguinte (B5, 2ª aplicação):** `training.stop_after_anneal: true`.
    - Nos braços com máscara aprendida e recozimento, a parada antecipada só conta e o melhor
      checkpoint só é escolhido a partir de `tau_anneal_epochs` (`FitConfig.min_epochs`).
    - O critério de convergência (margem > π/√6), `lr_multiplier = 30` e todo o resto ficam
      iguais.
    - Se mesmo assim a fração ficar abaixo de 0,5 em algum orçamento da variante bruta, esse
      orçamento é declarado não avaliável, sem nova refação.
  - **Decisão do usuário (2026-11-12, 01:23 UTC), antes de abrir qualquer número de teste da
    rodada 30×:** a rodada corrigida decide H1 e H2 em todos os orçamentos, com um único
    procedimento de treino.
    - A decisão veio antes dos números porque a rodada corrigida só pode melhorar o resultado.
      Decidir rodá-la só depois de ver a 30× seria escolher a rodada pelo resultado.
    - A rodada 30× foi interrompida na metade `sem_picos`, que não decide nada, e selada sem
      análise em `results-lr30/` (`SELADO.md`). Como não chegou ao fim, não gerou
      `hypotheses.csv`. Só os diagnósticos de treino saíram dela (`convergencia.txt`).
    - A rodada corrigida roda só a variante `bruto` (`despike_variants: [false]`). A `sem_picos`
      fica para uma rodada secundária posterior.
  - **Avaliabilidade por orçamento, em código** (`evaluable_budgets`; `min_mask_converged_frac`
    e `convergence_arms` no bloco `preregistration`):
    - Um orçamento só é avaliável numa hipótese se a máscara de cada braço de
      `convergence_arms` saiu do ruído em pelo menos 0,5 das divisões.
    - Os testes de um orçamento não avaliável entram no Holm com p = 1, o que é conservador, e
      ele não pode vencer. `hypotheses.csv` marca `evaluable` e a convergência de cada braço.
    - **H1:** `dec_aprendida` (o B5 como pré-registrado) e `dec_aprendida_sel` (a máscara
      efetivamente usada em cada divisão, que pode vir de um orçamento treinado mais esparso).
    - **H2:** `dec_aprendida` e `dec_livre_projetada`. Na rodada 30×, a máscara livre saiu do
      ruído em só 1/5, 1/5, 1/5 e 0/5 das divisões (diagnóstico de treino). Sem essa exigência,
      a H2 mediria o gap de projeção contra um baseline preso no ruído.

  Verificação no recorte da fumaça (1/32, 120 passos, sem parada antecipada; não é resultado):
  - com multiplicador 1, o desvio dos logits foi de 0,01 a 0,047, dentro do ruído;
  - com 10, foi a 0,475, mas a margem do top-K ainda era de 0,008.

  Nos dois casos a melhor época foi a última. A rodada completa tem muito mais passos; a regra
  acima decide.

## Ressalvas para o texto do artigo

- **O escopo é um rolamento por classe.** No Ottawa, cada condição tem o seu rolamento, então
  "família de falha" equivale a "qual rolamento". Isso vale para a WDCNN, que classifica família.
- **Faltam baselines publicados:** `han2024undersampledcnn`, `ahmed2018compressedbearing` e
  `zhao2026adaptive`.
- **`sem_picos` exige taxa cheia** e não é realizável no nó.
- **Não há T/H, jitter nem conversão modelados.** As amostras são as do conjunto, na grade de
  ticks. A cadeia do `snq-sim` não entra.
- **Origem da máscara e da pergunta do gap:** DPS (`huijben2020dps`) e PILOT (`weiss2021pilot`)
  precisam ser citados como tal.
- **O nRF54L15 não foi conferido.** O número de registradores CC por TIMER não foi verificado. A
  grade de 48 kSPS não é agendável a 16 MHz, porque dá 333,3 contagens por tick.

## Implementação

| peça | onde |
|---|---|
| conjunto viável, projeção exata, `fold_project`, uniforme, aleatória, projetadas | `snq_learn/masks.py` |
| máscara aprendida (DPS) com diagnósticos, LOUPE, adaptativa por blocos | `snq_learn/masks.py` |
| decodificador desdobrado (banco LPTV, envelope ao quadrado, espectro de ordem) | `snq_learn/order_decoder.py` |
| WDCNN da Tabela 2 de `zhang2017wdcnn`, conferida no PMC5336047 | `snq_learn/wdcnn.py` |
| laço com teto de épocas, paciência, melhor checkpoint e taxas por grupo | `snq_learn/training.py` |
| réguas, braços, seleção entre orçamentos, pré-registro e bootstrap por aglomerado | `run.py` |

Decisões que o desenho não fixava, ou das quais ele se afasta:

1. **K = 4 e L = 4 × orçamento.** Com K pequeno, um TIMER com K registradores de comparação
   basta (`timer_schedule`). O espaçamento é g = 2 ticks no Ottawa (10 µs). O valor realizável é
   medida de bancada, que não foi feita. A 16 MHz, a grade de 200 kSPS dá 80 contagens por tick.
2. **Banco periódico no tempo**, `Y_c(f) = Σ_r G_{c,r}(f)·Z(f − r·fs/L)`, com R = 5 deslocamentos
   (R ≥ K). Com só r = 0 ele é LTI.
   - Na máscara uniforme, iniciado nas bandas do kurtograma, ele reproduz o braço uniforme do E22
     (há teste para isso).
   - São 92 mil parâmetros, contra 54 mil da WDCNN. São respostas em frequência numa grade de
     1024 pontos, e não camadas genéricas, mas um revisor pode perguntar.
3. **A seleção de canal é o kurtograma desdobrado:** `softmax(τ·SK + b)`. Não há termo aprendido
   por bin de ordem, então o modelo não pinta pico onde o espectro não tem.
4. **Família de falha** dos métodos de ordem: a família de ordem central (mediana aninhada) mais
   próxima da estimativa, dentro de ±12 %; fora disso, "nenhuma". A WDCNN classifica direto, por
   voto suave das janelas do segmento.
5. **Controle saudável** (Ottawa H, nunca no treino): a fração de estimativas a menos de 0,06 de
   uma ordem de falha. A decomposição espaço-imagem/núcleo não se define, porque não há
   reconstrução de forma de onda.
6. **Ficou de fora:** a validação da WDCNN no CWRU 12k de 10 classes contra o número do artigo.
   Ela pede registros que não estão na máquina local, e o mapeamento de registro para classe ainda
   precisa ser conferido. A validação do LOUPE num brinquedo está nos testes.

## Estado

Implementado, testado e com fumaça em CPU (`config/smoke.yaml`, `results-smoke/`). **A fumaça
não é resultado.** A rodada completa usa `config/base.yaml` no servidor de GPU.

Cada execução treinada fica gravada em `results/runs/`. Um reinício retoma do ponto em que parou,
enquanto o config, o código e a CLI não mudarem. O `masks.json` junta só as execuções da
invocação corrente.

Fumaça do protocolo corrigido (2026-11-11). O recorte é o cenário A do Ottawa, 1/16 e 1/32, nas
duas variantes, com uma divisão e no máximo 6 épocas. O que ela prova é mecânica:
- **Laço:** 40 execuções e 480 linhas. A parada antecipada disparou em 21 de 60 treinos.
- **Retomada:** as 40 execuções foram retomadas, e `metrics.csv` e `hypotheses.csv` saíram
  idênticos byte a byte.
- **Régua local:** 14 a 15 das 15 janelas por registro saíram válidas. Um segmento de teste com
  platô na borda ficou fora da pontuação local.
- **LOUPE:** com o forward duro, a taxa no treino foi de 0,0315 contra 1/32.
- **Bootstrap:** com uma divisão e um aglomerado por pista, todos os testes saem inelegíveis
  (p = 1). O veredito sai 0, como deve.

## Rodada completa no servidor (lançada em 2026-11-11, 11:58 UTC)

> Interrompida pela revisão adversarial (ver "Correções da revisão adversarial"). O ambiente
> abaixo continua valendo para o relançamento.

- **Árvore separada.** A rodada usa `~/snq/e15/sub_nyquist`, com o dado bruto ligado por link
  simbólico à árvore principal. A árvore principal está rodando a fila de re-execuções (E23 picos,
  E07, E20, E22), e sincronizar o `snq-learn` nela mudaria, no meio das rodadas, o hash de fonte
  que elas gravam. O manifest do E15 registra o hash da árvore dele.
- **Desvio de ambiente.** O lock instala `torch 2.14.0+cu130`, mas o driver NVIDIA do servidor só
  vai até CUDA 12.6, e o torch não via a GPU. Só nessa árvore foi instalado
  `torch 2.14.0+cu126`, a mesma versão com outro build de CUDA (índice
  `download.pytorch.org/whl/cu126`). A rodada usa o Python do `.venv` direto, porque `uv run`
  re-sincronizaria com o lock.
- **GPU.** RTX 4090 de 25,3 GB, selecionada pelo UUID com `CUDA_VISIBLE_DEVICES`. A outra placa do
  servidor está com defeito: o `nvidia-smi` falha em `0000:0C:00.0`.
- **Comando.** `run.py --config config/base.yaml --workers 6`, com os 7 métodos, os 2 conjuntos e
  até 500 épocas com paciência de 40. Na GPU, a reprodutibilidade bit a bit não é garantida (ver a
  fumaça em CPU).

## Resultado da rodada decisiva (2026-11-12)

Rodada de 2026-11-12, de 01:35 a 05:48 UTC, no commit `314b0c3`.
- Os quatro hashes do manifest (fonte, config, `run.py` e README) batem com o commit.
- O `git_sha` sai nulo porque a árvore do servidor não é um checkout do git. Os hashes cobrem
  isso.
- Este README ganhou esta seção depois da rodada. O `readme_sha256` do manifest é o do commit
  `314b0c3`.

### Avaliabilidade (B5), conferida antes de abrir as hipóteses

| orçamento | `dec_aprendida` | `dec_aprendida_sel` | `dec_livre_projetada` |
|---|---|---|---|
| 1/4 | 1,0 | 1,0 | 0,6 |
| 1/8 | 1,0 | 1,0 | 0,2 |
| 1/16 | 1,0 | 1,0 | 0,2 |
| 1/32 | 0,8 | 0,8 | 0,0 |

A H1 é avaliável nos quatro orçamentos; a H2, só em 1/4.

### Veredito pré-registrado

- **A H1 não passou em nenhuma das réguas.** Nenhum dos 24 testes é significativo, nem antes da
  correção: o menor p unilateral é 0,20.
- **A H2 não se sustenta.** Só pôde ser avaliada em 1/4, onde a diferença foi de 0,0 ponto:
  [−14,3; +11,4] pela régua local e [−6,25; +6,25] pela aninhada.

Acurácia balanceada de ordem (%), variante `bruto` (`summary.csv`):

| braço | régua | 1/4 | 1/8 | 1/16 | 1/32 |
|---|---|---|---|---|---|
| `uniforme_treino` (clássico, fator escolhido no treino) | local | 94,8 | 94,8 | 89,7 | 89,7 |
| `dec_aprendida_sel` | local | 93,1 | 89,7 | 79,3 | 74,2 |
| `dec_uniforme_treino` | local | 91,4 | 86,3 | 79,3 | 75,9 |
| `dec_aleatoria` | local | 86,2 | 93,1 | 75,9 | 69,1 |
| `oraculo_uniforme` (escolhe com o teste) | local | 98,3 | 98,3 | 98,3 | 98,3 |
| `uniforme_treino` | aninhada | 88,3 | 88,3 | 86,7 | 85,0 |
| `dec_aprendida_sel` | aninhada | 86,7 | 85,0 | 76,7 | 70,0 |
| `dec_uniforme_treino` | aninhada | 90,0 | 86,7 | 78,3 | 75,0 |
| `dec_aleatoria` | aninhada | 81,7 | 90,0 | 73,3 | 68,3 |
| `oraculo_uniforme` | aninhada | 90,0 | 90,0 | 90,0 | 90,0 |

### Revisão adversarial (`reviewer-2`, 2026-11-12)

O veredito pré-registrado fica, mas o enunciado muda. O resultado é **ausência de evidência de
ganho num desenho sem poder**. Não é evidência de que aprender a máscara não ajude.

1. **O teste não podia passar.** Com um braço perfeito no lugar de `dec_aprendida_sel`, a mesma
   regra dá veredito 0 nas duas réguas.

   | contra `uniforme_treino`, braço perfeito | 1/4 | 1/8 | 1/16 | 1/32 |
   |---|---|---|---|---|
   | p sem correção, régua local | 0,17 | 0,16 | 0,05 | 0,07 |
   | p sem correção, régua aninhada | 0,025 | 0,022 | 0,018 | 0,024 |

   Depois do Holm, os p ficam em ≥ 0,43 (local) e ≥ 0,14 (aninhada). Os motivos:
   - O clássico já acerta 56 de 59 segmentos em 1/4 e 1/8 pela régua local, e o oráculo, 58.
   - A pista interna fica no teto em quase todos os braços. Só os 29 segmentos da pista externa
     discriminam.
   - Há 4 aglomerados por pista, de tamanhos desiguais: na externa, 1, 5, 6 e 3 pares.
   - A diferença detectável contra `uniforme_treino` fica entre 18 e 40 pontos pela régua local e
     entre 11 e 39 pela aninhada, acima da folga até o oráculo.
2. **O déficit contra o clássico vem em boa parte de um registro, O-C-2.**
   - O-C-2 está no teste em 4 das 5 divisões e soma 8 dos 29 segmentos da pista externa.
   - Nele, todos os braços com o decodificador aprendido devolvem a ordem 2,0, que é a borda
     inferior da busca (`order_min`).
   - Sem O-C-2 (exploratório), `dec_aprendida_sel − uniforme_treino` passa a +4,6 / +2,3 / 0,0 /
     −3,9 pontos.
   - O mecanismo provável, não provado: com o envelope |y|² do decodificador, a raia de ordem 2
     compete com a BPFO nesse registro.
   - As divisões são de Monte Carlo, e 6 dos 24 registros nunca vão a teste.
3. **Com o mesmo decodificador, a máscara quase não muda a decisão.**
   - Entre `dec_aprendida_sel` e `dec_uniforme_treino`, 12 de 236 segmentos discordam (régua
     local).
   - Fora de 1/4, as máscaras aprendidas não concordam entre divisões mais do que máscaras viáveis
     sorteadas.
   - Duas réplicas que só diferem pela semente (`dec_uniforme_treino` em 1/4 e em 1/8, com o mesmo
     fator) diferem em cerca de 5 pontos.
4. **Não há vazamento, mas o baseline tem um grau de liberdade que a máscara não tem.**
   - `uniforme_treino` escolhe o fator em [b, 48] só com treino e validação. Saiu 13 ou 14 em 1/4
     e 1/8, e de 30 a 46 em 1/16 e 1/32, isto é, de 2 a 3,5 vezes menos amostras que o
     orçamento.
   - A máscara tem L = 4b e K = 4 fixos.
   - Por isso a comparação é "até o orçamento", e não "na mesma taxa".
5. **`stop_after_anneal` não piorou o braço decisivo.**
   - O estágio B para cedo, e a validação tem 8 janelas.
   - Os braços `_sem_ajuste` ficam iguais aos ajustados ou acima deles, e nenhum passa de
     `uniforme_treino`.
6. **A seleção entre orçamentos não explica o negativo.**
7. **H2:** o termo certo é "não sustentada", e não "refutada". Com 3 de 4 orçamentos não
   avaliáveis, "o conjunto viável não é contribuição" é forte demais.
8. **`dec_adaptativa_blocos` é ruído até prova em contrário.**
   - Tem os maiores números pontuais em 1/4 a 1/16.
   - Mas é o máximo entre 12 braços, está fora do pré-registro e treinou mais que os outros.
9. **Não há bug** em `per_record`, `cluster_bootstrap`, `holm` nem `hypotheses`. O
   `hypotheses.csv` se reproduz exatamente a partir do `metrics.csv`.
10. **Anterioridade e escopo.**
    - Em MRI, `bakker2022adaptive` acha políticas aprendidas empatadas com a equidistante a 4×.
    - O resultado aqui vale para decimação de registros a 200 kSPS, com amostrador ideal, sem
      track-and-hold e sem jitter.

### Enunciado para o artigo

Parte (b):

> No conjunto Ottawa (variante bruta), em simulação por decimação de registros a 200 kSPS, sem
> modelo de track-and-hold nem de jitter, a máscara aprendida sob o conjunto viável não superou a
> melhor taxa uniforme escolhida nos registros de treino (que ficou abaixo do orçamento, entre
> 1/13 e 1/46) em nenhum dos quatro orçamentos, de 1/4 a 1/32. As diferenças de acurácia
> balanceada de ordem foram de −1,7 a −15 pontos, com intervalos de 95 % por bootstrap de
> aglomerado entre −46 e +14 pontos, e nenhum dos 24 testes pré-registrados foi significativo. O
> desenho não tinha poder: em simulação, nem um método que acertasse todos os segmentos de teste
> passaria no critério.

H2:

> O gap de projeção só pôde ser avaliado em 1/4, onde foi de 0,0 ponto [−14,3; +11,4]. Nos demais
> orçamentos, a regra pré-registrada o declarou não avaliável.

### Exploratório: estabilidade da máscara usada

Medida com `explore_mask_stability.py`, só sobre diagnósticos de treino
(`results/exploratory/mask_stability.csv`). É a fração das posições usadas que sobrevive ao ruído
de Gumbel do treino, em mediana entre as divisões.

| orçamento | aprendida | livre projetada |
|---|---|---|
| 1/4 | 0,999 | 0,986 |
| 1/8 | 0,998 | 0,974 |
| 1/16 | 0,995 | 0,909 |
| 1/32 | 0,969 | 0,832 |

As máscaras estão firmes, então o resultado não vem de máscara presa no ruído. A livre projetada é
mais estável do que o critério de margem sugeria. Isso não muda a avaliabilidade, que foi
pré-registrada.

### O que destrava o quê

1. **Porta de poder.** Roda em CPU, sobre o `metrics.csv` atual: simular a regra com o braço
   perfeito e com efeitos plantados (+5 e +10 pontos) em desenhos candidatos. Se nada passar no
   Ottawa, a H1 não é testável nele.
2. **Divisões:** cada registro testado uma vez e `seeds_per_split` ≥ 3. Depende de 1.
3. **Depois de 2, em paralelo:**
   - a ablação da falha em O-C-2 (`envelope_power` 1 contra 2, e a borda `order_min`);
   - o decodificador clássico sobre a máscara aprendida;
   - um braço que parte da melhor uniforme (`init: uniform`, com L = 4 × o fator escolhido);
   - o mesmo orçamento de treino para todos os braços.
4. **Braço adaptativo como primário pré-registrado,** com controle pareado. Depende de 2 e 3.

## Porta de poder (pré-registro, 2026-11-12)

Escrito antes de rodar. A porta responde a uma pergunta: o Ottawa consegue, em algum desenho,
detectar pela regra da H1 um braço que seja de fato melhor que o baseline? Ela roda só em CPU, com
`power_gate.py`, sobre o `metrics.csv` da rodada decisiva.

- **Braço plantado.**
  - Parte das linhas de `uniforme_treino`. Cada segmento em que ele erra vira acerto com
    probabilidade q, com o mesmo sorteio nas duas réguas. Onde ele acerta, o plantado também
    acerta.
  - O q de cada (orçamento, régua) produz o efeito pedido em acurácia balanceada. Quando o teto não
    comporta o efeito, o q para no braço perfeito (q = 1).
- **Efeitos:** +5 pontos (o que decide), +10 pontos e o braço perfeito.
- **Desenhos:**
  - **D0:** como foi rodado, com 5 divisões de Monte Carlo e 1 semente.
  - **D1 e D2:** com 3 e 5 sementes, emuladas como sorteios independentes do braço plantado e
    tiradas em média por (divisão, registro).
  - **DK:** cada registro testado uma vez, em 4 dobras estratificadas por pista, com 5 sementes.
    - A grade clássica não fica em cache, então o baseline é sintético.
    - Em cada pista, ele erra cada segmento com a taxa média de erro observada, o que espalha os
      erros por todos os aglomerados.
  - **D3, só como referência anticonservadora:** D2 com bootstrap por registro em vez de por
    aglomerado.
    - Não é um desenho válido para o artigo, porque registros da mesma condição × cenário não são
      independentes.
    - Serve para medir quanto o aglomerado limita o poder.
- **Regras de decisão:**
  - **Pré-registrada:** a da H1, com os três baselines e Holm sobre os 12 testes de cada régua. Vale
    em D0 a D3.
  - **Otimista:** só o teste contra `uniforme_treino`. Os 8 testes contra os baselines aprendidos
    entram no Holm como significativos (p = 0). Vale em todos os desenhos e, no DK, é a única.
- **Otimismo deliberado.** Os baselines aprendidos ficam fixos, sem ruído de semente, e o plantado
  nunca piora. No DK, os erros do clássico ficam espalhados do jeito mais favorável ao teste. Por
  isso um "não" é conclusivo, e um "sim" não prova que um desenho real funcione.
- **Critério:** a H1 é testável no Ottawa se, em algum dos desenhos D0, D1, D2 ou DK, o poder pela
  regra otimista para +5 pontos for de pelo menos 80 %. Se não for, a H1 não é testável no Ottawa,
  e a parte (b) precisa de outro conjunto ou sai do artigo.
- **Simulação:**
  - São 200 repetições por desenho e efeito. O braço perfeito, que fora do DK só varia pelo
    bootstrap, tem 20.
  - Cada teste usa 2000 reamostras, com semente fixa.
  - Antes, o bootstrap vetorizado é conferido contra o `cluster_bootstrap` do `run.py` nos dados
    reais.

### Resultado da porta de poder (2026-11-12)

- **Execução:** `power_gate.py`, cerca de 1 min de CPU. As saídas estão em `results/power_gate/`
  (`power.csv`, `validation.json` e `baseline.json`).
- **Validação do bootstrap vetorizado** contra o `hypotheses.csv` real: as diferenças pontuais são
  idênticas, e os p ficam dentro do erro de Monte Carlo (maior desvio de 0,026, com 4000
  reamostras de cada lado).

Poder da H1, isto é, a fração das repetições em que a regra passa nas duas réguas:

| desenho | regra | +5 pontos | +10 pontos | braço perfeito |
|---|---|---|---|---|
| D0 (como rodado) | pré-registrada / otimista | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |
| D1 (3 sementes) | pré-registrada / otimista | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |
| D2 (5 sementes) | pré-registrada / otimista | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |
| DK (cada registro testado uma vez) | otimista | 0,03 | 0,06 | 0,07 |
| D3 (bootstrap por registro, inválido) | pré-registrada / otimista | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |

**Veredito pelo critério pré-registrado: a H1 não é testável no Ottawa.** Entre os desenhos
válidos, o maior poder pela regra otimista com +5 pontos é de 0,03. O critério exigia 0,80.

- **O limite é o número de erros do clássico.** `uniforme_treino` erra só de 3 a 9 dos 59 ou 60
  segmentos de teste, conforme o orçamento e a régua, e todos na pista externa. A pista interna
  não tem erro em nenhum orçamento.
  - Pela régua local, em 1/4 e 1/8, os 3 erros estão em O-A (o único segmento válido do
    aglomerado) e em O-B.
- **Por que nem o braço perfeito passa em D0 a D2.**
  - Na regra otimista, os 8 testes tomados como significativos ocupam os primeiros postos do Holm.
  - Assim, o menor p contra `uniforme_treino` precisaria ficar abaixo de 0,05/4 = 0,0125.
  - O braço perfeito chega, no máximo, a cerca de 0,02, pela régua aninhada.
- **Mais sementes não mudam isso,** porque o baseline é determinístico. Testar cada registro uma
  vez também não, porque continuam sendo poucos erros para corrigir.
- **Nem a referência anticonservadora passa a regra completa.** É o D3, em que cada registro é um
  aglomerado.

**Consequência.**
- A parte (b) do artigo não pode ser decidida no Ottawa.
- A pergunta precisa de um conjunto em que o baseline erre o bastante, espalhado por aglomerados
  independentes o bastante, para que um ganho de +5 pontos seja detectável.
- Essa conta pode ser feita antes de treinar, porque a porta só precisa do estágio clássico, que
  roda em CPU.

## Porta de poder no Paderborn (pré-registro, 2026-11-12)

Escrito antes de rodar, depois da ingestão e da checagem de adequação
(`datasets/manifests/paderborn.yaml`). A pergunta é a mesma da porta do Ottawa: o conjunto
consegue detectar, pela regra da H1, um braço de fato melhor que o baseline? Roda só em CPU, com
`paderborn_gate.py` e `config/paderborn_gate.yaml`.

- **Registros:** rolamento × ajuste, nos três ajustes de 1500 rpm. A medição 1 calibra a régua
  aninhada, e as medições 2 a 20 são os segmentos de teste, de 4 s cada (100 voltas, 0,010 ordem
  de resolução).
- **Grupos e aglomerados:** a pista é o grupo, e o rolamento é o aglomerado do bootstrap.
  - Pista externa: 12 rolamentos (7 artificiais e 5 reais).
  - Pista interna: 11 rolamentos (5 artificiais e 6 reais).
  - Ficam fora os combinados (KB), cuja família de ordem é ambígua, e os saudáveis.
- **Divisões:** K-fold por rolamento, com 4 dobras estratificadas por pista. Cada rolamento é
  testado uma vez, e nenhum aparece no treino e no teste da mesma dobra.
- **Pipeline clássico:** o do E15, com quatro mudanças.
  - **Tolerância de 0,025 ordem**, em vez de 0,06. A BPFO e a BPFI ficam a 0,054 das harmônicas 3
    e 5 do eixo.
  - **Busca da régua em ±1 % da nominal**, em vez de ±12 %, para excluir 3,0 e 5,0.
  - **Teto do kurtograma no Nyquist efetivo.**
  - **Guarda de 1,0 × 64 pontos por volta**, a mesma do CWRU no E15. Com 2,0, o fator 32 não
    rodaria.
- **Régua local:** a raia em taxa cheia do próprio segmento.
  - Só vale se não encostar na borda da busca e se o pico, em ±0,015 ordem do centro, tiver pelo
    menos 5 vezes a mediana da janela de ±7 % da nominal.
  - Na checagem de adequação, as raias visíveis deram de 14 a 141, e as ausentes de 0,3 a 2,5.
- **Régua aninhada:** a mediana das calibrações válidas da medição 1 dos registros fora do teste,
  na mesma pista. Ela vale para todos os segmentos de teste, inclusive os de rolamento sem raia
  visível.
- **Baseline:** `uniforme_treino`, a melhor taxa uniforme até o orçamento, escolhida pela régua
  local nos registros fora do teste.
- **Porta:** a do Ottawa, com braços plantados a partir do baseline (+5 pontos, +10 pontos e
  perfeito) e 1, 3 ou 5 sementes (D0, D1 e D2).
  - Só a regra otimista: não há braço aprendido, então os 8 testes contra os baselines
    aprendidos entram no Holm como significativos.
  - São 200 repetições por desenho (20 para o perfeito) e 2000 reamostras por teste.
- **Critério:** a H1 é testável no Paderborn se o poder para +5 pontos, nas duas réguas, for de
  pelo menos 80 % em algum dos desenhos D0, D1 ou D2.
  - Se for, o E15 roda no Paderborn, com os braços aprendidos e as correções da revisão
    adversarial, depois de um pré-registro próprio.
  - Se não for, a parte (b) sai do artigo.
- **Correção antes de qualquer número (2026-11-12, achada na fumaça).**
  - O número de amostras de 64 kHz por arquivo varia (de 255999 a 262062), mas o eixo de tempo
    vai sempre de 0 a 4 s.
  - A corrente do PMSM, que gira a 4× a rotação, mostra que a taxa é de 64 kHz fixos: razão
    mediana de 0,99998, com 1 arquivo de 2559 acima de 0,5 %.
  - O carregador passa a cortar a vibração em 4 s, alinhados com o canal de rotação.
  - Um arquivo não abre (`KA08/N15_M01_F10_KA08_2.mat`). A medição fica fora e vai registrada no
    manifest.
  - Nada do protocolo acima mudou.

### Resultado da porta no Paderborn (2026-11-12)

O estágio clássico levou 3246 s em 28 processos de CPU. As saídas estão em
`results-paderborn-gate/` (`power.csv`, `metrics.csv`, `truths.json` e `manifest.json`).

| desenho (regra otimista) | +5 pontos | +10 pontos | perfeito |
|---|---|---|---|
| D0 (1 semente) | 0,74 | 0,99 | 1,00 |
| D1 (3 sementes) | **1,00** | 1,00 | 1,00 |
| D2 (5 sementes) | **1,00** | 1,00 | 1,00 |

**Veredito pelo critério pré-registrado: a H1 é testável no Paderborn, com pelo menos 3 sementes
por dobra.** No Ottawa, nem o braço perfeito passava.

- **Há espaço para ganho.** `uniforme_treino` acerta 41,5 / 44,8 / 49,7 / 26,7 % pela régua local
  e 16,8 / 17,4 / 18,6 / 13,2 % pela aninhada, de 1/4 a 1/32.
- **Régua local:** só 6 dos 23 rolamentos têm raia visível em taxa cheia na medição 1 (KA01,
  KA06, KA16, KI01, KI16 e KI18). São 180 segmentos de 1291 com régua local.
- **Régua aninhada:** vale para todos os segmentos.
  - Inclui rolamentos sem assinatura visível, nos quais nenhum método talvez acerte.
  - A porta mede se o teste consegue detectar o ganho, e não se ele é alcançável.
- **Medições fora: 21.**
  - 1 ilegível; 2 com ~3,9 s; 3 com o canal de rotação de duração errada.
  - 15 só têm 3 a 5 amostras a menos que 4 s. Esse critério foi rígido demais, e a rodada do E15
    no Paderborn o relaxa: aceita até 1 ms a menos.
  - Isso não muda o veredito, com poder de 1,00.
- **Consequência:** a parte (b) volta ao artigo, condicionada ao E15 no Paderborn. Ele terá
  pré-registro próprio, as correções da revisão adversarial e 3 sementes por dobra.

## E15 no Paderborn (pré-registro, 2026-11-12)

Escrito antes de rodar, depois de a porta de poder passar. O config é `config/paderborn.yaml`,
derivado de `config/base.yaml`. Valem as hipóteses H1 e H2, a regra de decisão, o Holm, a regra de
avaliabilidade (B5) e o bootstrap por aglomerado da seção "Pré-registro", com as mudanças abaixo.

- **Dados:** os da porta.
  - 23 rolamentos (12 de pista externa e 11 de pista interna) nos três ajustes de 1500 rpm.
  - A medição 1 calibra; as medições 2 a 20 são os segmentos, cortados em 255936 amostras.
  - A medição que não abre ou que é mais curta que isso fica de fora.
- **Divisões:** K-fold por rolamento, com as 4 dobras da porta (há teste que confere).
  - A validação sai dos rolamentos fora do teste, com 25 % por pista e pelo menos 1.
  - O aglomerado do bootstrap é o rolamento.
- **Sementes:** 3 por dobra. A porta deu poder de 1,00 para +5 pontos com 3 sementes, e de 0,74
  com 1.
- **Réguas:** as da porta.
  - A local só vale onde a raia é visível em taxa cheia (pico de pelo menos 5× a mediana).
  - A aninhada vale em todos os segmentos.
  - Tolerância de 0,025 ordem, busca da régua em ±1 % da nominal.
- **Rótulos de treino e validação:** a régua aninhada, porque a local só existe em 6 dos 23
  rolamentos.
  - O decodificador não tem termo aprendido por ordem, então não consegue devolver a ordem da
    pista sem uma raia no espectro.
- **Métodos:** os do Ottawa, menos `dec_adaptativa` e `wdcnn`, que não decidem nada.
- **Guarda:** 1,0 × 64 pontos por volta, como na porta.
- **Correções da revisão adversarial:**
  - adotadas: K-fold por rolamento e 3 sementes;
  - não adotadas nesta rodada: o decodificador clássico sobre a máscara aprendida, o braço que
    parte da melhor uniforme e o orçamento de treino igual. Ficam como limitação declarada.
- **O que refuta:** o mesmo da seção "O que refutaria". Com a porta passando, um negativo aqui é
  evidência de ausência de ganho de pelo menos 5 pontos nesta tarefa, e não falta de poder.

### Execução paralela no Paderborn (2026-11-12)

O desenho, as dobras, as sementes, os métodos e os critérios ficam os do pré-registro acima. Muda só
quantos processos calculam as células.

- **Por que.** A rodada principal começou o estágio de GPU às 15:27 UTC.
  - Pelo NVML, a GPU ficou em cerca de 6 % de uso, com 3,2 GiB e 76 W.
  - O processo principal ocupa um núcleo de CPU (87 %), então o treino é limitado pelo Python, e
    não pela GPU.
  - A primeira execução no Paderborn (`dec_uniforme`, 1/4, dobra 1, semente 1) levou 181 s. No
    Ottawa, a média desse método foi de 55 s. A rodada tem 48 células (4 dobras × 3 sementes × 4
    orçamentos) de 7 métodos.
- **Como.** `shard.py` roda as células que recebe em outros processos, na mesma GPU.
  - Ele usa o mesmo `run.py`, sem alteração, e grava cada execução em `runs/` com o mesmo `run_key`.
  - O `run_key` cobre o config, o `run.py`, o código dos pacotes e a CLI que muda o resultado.
    `--budgets` e `--workers` não entram nele.
  - O processo principal, ao chegar numa célula já gravada, a retoma, e o manifest dele a marca
    como `"retomado"`.
- **O que garante que o resultado é o mesmo.**
  - Os pedaços chamam `run.run_method` com o mesmo contexto. A semente de dados sai de
    `derive_seed` por célula, e o cudnn é determinístico.
  - Os pedaços saem antes de `summarize`. Por isso não escrevem `metrics.csv`, `summary.csv`,
    `hypotheses.csv`, `masks.json` nem o manifest. Quem os escreve é o processo principal, que
    passa por todas as células.
  - Cada execução de pedaço é gravada de forma atômica (temporário + `os.replace`), e o processo
    principal nunca lê um JSON pela metade.
  - O estágio clássico dos pedaços vem de um cache. A chave dele é o hash do config, do `run.py` e
    dos pacotes, e um cache de outra chave é recusado.
- **Validação antes de usar** (local, CPU, config de fumaça do Ottawa, 4 métodos, 3 épocas):
  - `metrics.csv`, `summary.csv`, `hypotheses.csv` e `comparisons.csv` saíram **idênticos** aos de
    uma rodada sem pedaços.
  - As execuções gravadas diferem só no tempo de relógio (`info/.../seconds`). Isso vale também
    para as que o próprio processo principal calculou, então não vem do pedaço.
  - O manifest do principal marca como `"retomado"` as 4 execuções calculadas pelo pedaço.
  - Nenhum arquivo temporário ficou para trás.
- **Proveniência.**
  - O processo principal roda no checkout do servidor criado no commit `4e3bf37`. O
    `readme_sha256` do manifest será o desse commit, porque este README só muda no repositório
    local.
  - `shard.py` entrou no checkout do servidor como arquivo novo, e não muda nenhum hash do
    `run_key`.

## Resultado no Paderborn (2026-11-13)

- **Rodada:** `results-paderborn/`, com 336 execuções, 4 dobras por rolamento × 3 sementes × 4
  orçamentos × 7 métodos.
  - O estágio clássico rodou uma vez e serviu de cache para os pedaços (`shard.py`).
  - O `metrics.csv` vai comprimido (`metrics.csv.gz`); os `runs/` ficam no servidor.
- **Avaliabilidade (B5), conferida antes das hipóteses:**
  - A máscara restrita convergiu em 92, 83, 75 e 92 % das 12 execuções por orçamento, de 1/4 a
    1/32. A selecionada convergiu em 83, 75, 83 e 92 %. **A H1 é avaliável nos quatro.**
  - A livre projetada convergiu em 33, 0, 17 e 42 %. **A H2 não é avaliável em nenhum.**
- **Veredito pré-registrado: a H1 não passa em nenhum orçamento, em nenhuma régua.** São 0 de 24
  testes significativos.
  - **Régua aninhada:**
    - contra `dec_uniforme_treino`: −1,8, −2,2, −1,0 e −5,2 pontos;
    - contra `dec_aleatoria`: −3,1, −0,5, +4,4 e −5,2;
    - contra `uniforme_treino`: +8,3, +6,2, +8,8 e +12,6.
    - O menor p unilateral é 0,037 (1/32, contra `uniforme_treino`, IC de −1,1 a +28,2). O Holm
      o leva a 0,44.
  - **Régua local** (17 registros, 7 aglomerados): de −16,9 a +18,9 pontos, sem nada
    significativo.
- **Acurácia balanceada, régua aninhada (%):**

  | braço | 1/4 | 1/8 | 1/16 | 1/32 |
  |---|---|---|---|---|
  | `dec_aprendida_sel` | 25,7 | 24,4 | 27,5 | 25,4 |
  | `dec_uniforme_treino` | 27,5 | 26,7 | 28,5 | 30,5 |
  | `dec_aleatoria` | 28,7 | 25,0 | 23,1 | 30,6 |
  | `dec_multicoset_vb` | 26,2 | 23,6 | 26,5 | 34,8 |
  | `uniforme_treino` (pipeline clássico) | 17,4 | 18,3 | 18,8 | 12,7 |

- **Leitura:**
  - **A margem sobre o pipeline clássico é do decodificador, não da máscara.** A uniforme
    treinada e a aleatória usam o mesmo decodificador e ficam tão à frente quanto a aprendida.
  - **Com poder de 1,00 para +5 pontos** (porta no Paderborn), é evidência contra um ganho dessa
    ordem da máscara aprendida sobre a decimação uniforme treinada e sobre a aleatória.
  - **O limite é o pipeline clássico,** já capturado em taxa cheia no Paderborn (E24), e a régua
    aninhada que rotula o treino.
- **Números do artigo:** `papers/shared/figures/num_partB_paderborn.py` →
  `papers/mssp-2026/numbers/partB_paderborn.json`.


## De onde vem a margem do decodificador no Paderborn (exploratório, 2026-11-13)

Pedido pela revisão adversarial do manuscrito. Não é pré-registrado. Script:
`explore_line_split.py` → `results-paderborn/exploratory/line_split.csv` (mesma agregação do
`summary.csv`, régua aninhada).

- **16 rolamentos sem régua local** (sem raia visível em taxa cheia): o clássico fica em 7,7–8,0 %,
  e os três braços com decodificador em 16,8–24,0 %. A margem sobre o clássico está aqui.
- **7 rolamentos com régua local** (KA01, KA06, KA16, KI01, KI16, KI17, KI18): o clássico dá 42,4,
  45,1 e 47,2 % de 1/4 a 1/16, e a aprendida 34,2, 31,5 e 43,8 %. Em 1/32 o clássico cai a 26,8 %, e
  a aprendida fica em 45,2 %.
- **Na régua local,** a aprendida fica 11,7, 15,6 e 6,4 pontos abaixo do clássico de 1/4 a 1/16, e
  18,9 acima em 1/32. Nada disso é significativo.
- **Correção do texto anterior:** "evidência contra um ganho de 5 pontos" não se sustenta. Os
  limites superiores do IC 95 % contra `dec_uniforme_treino` e `dec_aleatoria` vão de +3,4 a +18,2
  pontos. A porta de poder mediu poder só contra o clássico, que é determinístico.
- **Leitura, não testada:** os decodificadores ganham onde o kurtograma está capturado, e são
  treinados com rótulos que dizem que ordem procurar. Isso é compatível com escolha de banda guiada
  pelos rótulos, o efeito da regra do E26 sem aprendizado. Não houve controle com rolamentos sãos.

## Figura do Paderborn (2026-11-13)

![Máscaras aprendidas nas duas réguas](results-paderborn/figures/e15_masks_paderborn.png)

Gerada por `papers/shared/figures/fig_masks_paderborn.py`, com a estatística dos testes
pré-registrados: média por (divisão, registro), depois média por pista, depois média das pistas.
O script confere as diferenças entre a máscara aprendida e cada referência contra o `diff_points`
do `hypotheses.csv` e para se algo divergir.

Na régua local, essa média por registro difere bastante do acerto agrupado por segmento do
`summary.csv`, porque os registros têm números muito diferentes de segmentos com raia. Os números
do artigo seguem a estatística dos testes.
