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
