# E20 - Rejeição por zona antes de decimar: o que captura o kurtograma em 4×

Aberto em 2026-11-10 pelo [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md). É a
Parte 1 (mecanismo) do artigo único.

## Contexto

No E07 (Ottawa 2018, coluna `sem_prior`), a pista externa cai de 23/36 acertos em 8× para 6/36
em 4×, enquanto a pista interna fica em 31–32/36. Em 4× o kurtograma escolhe a banda
[12,75; 25] kHz em 26/36 segmentos da pista externa e erra todos eles.

A hipótese do relatório era que conteúdo de 75–87,2 kHz dobra sobre essa banda. Mas essa causa
não está isolada:
- a mesma faixa também dobra em 2×, onde não há vale;
- em 4× passam a dobrar também 25–37,25 kHz e 62,75–75 kHz;
- em 8× a banda [12,75; 25] kHz fica acima do Nyquist efetivo e sai da busca.

Há portanto três explicações rivais:
1. **dobra de uma zona específica**;
2. **dobra do conjunto das zonas**;
3. **geometria da busca**: o teto da busca em relação ao Nyquist efetivo, ou a partição diádica
   do kurtograma, que muda com `fs_eff`, e não a dobra.

## Hipótese

O vale de 4× na pista externa é causado por **conteúdo dobrado**. Remover antes da decimação as
zonas de frequência original que dobram sobre [12,75; 25] kHz em 4× (Z1 = 25–37,25 kHz,
Z2 = 62,75–75 kHz, Z3 = 75–87,25 kHz) faz a pista externa em 4× voltar a **pelo menos 20/36**.
O filtro anti-alias ideal tem o mesmo efeito.

Hipóteses secundárias, reportadas qualquer que seja o resultado:
- **H2 (atribuição):** uma única zona responde pela maior parte da recuperação.
- **H3 (teto):** baixar o teto da busca para 12,5 kHz também recupera a pista externa em 4× e
  piora a pista interna em 1× e 2×, cuja banda informativa está em 16,8–25 kHz.

## O que refutaria esta hipótese

- **Refuta a dobra:** em 4× na pista externa, `aaf_ideal` **e** `stop_z123` ficam em
  **no máximo 10/36**. Nesse caso o vale não é dobra: é geometria da busca ou da partição
  (explicação 3), e a Parte 1 do artigo muda de tese.
- **Refuta H2:** nenhuma zona isolada recupera mais da metade do que `stop_z123` recupera.
- **Refuta H3:** com teto de 12,5 kHz, a pista externa em 4× continua em no máximo 10/36.

**Critério de recuperação:** pelo menos 20/36 (o nível de 8× é 23/36) **e** McNemar exato
pareado contra `none`, no mesmo fator e no mesmo teto, com p < 0,01.

## Baseline

O pipeline do E07 sem intervenção (`none`, teto de 25 kHz): kurtograma restrito, envelope,
espectro em ordens e estimativa sem prior. É o "passa-faixa + envelope" do projeto.

**Controle de reprodução:** `none` com teto de 25 kHz precisa reproduzir exatamente o E07 em 1×,
2×, 4× e 8× (56, 53, 37 e 55 acertos de 72). Se não reproduzir, nada abaixo vale.

## Verdade fundamental

A mesma do E07:
- **Calibração:** ordem de falha calibrada por registro em taxa cheia, refinada em torno da
  ordem nominal (BPFO ou BPFI do ER16K).
- **Verdade:** a mediana dos **outros** registros da mesma condição (deixa-um-de-fora).
- **Acerto:** |estimativa − verdade| < 0,06 ordem.

O rótulo de falha vem do conjunto de dados. A rotação vem do encoder (1024 ppr), que é
independente do canal de vibração. Dados: Ottawa 2018, Mendeley `v43hmbwxpm`, CC BY 4.0.

## Procedimento

1. **Passe 1**, idêntico ao E07: calibrar a ordem de cada registro de falha (I e O, 24
   registros) em taxa cheia.
2. **Passe 2.** Para cada registro, segmento (3 × 2,5 s), intervenção, fator
   (1, 2, 3, 4, 5, 6, 8) e teto (25 kHz, 12,5 kHz):
   - aplicar a intervenção ao segmento **em taxa cheia**, como máscara ideal de FFT de fase zero;
   - decimar sem filtro;
   - rodar o kurtograma com teto `min(teto, fs_eff/2)`;
   - calcular o espectro de envelope em ordens e a estimativa sem prior.

   Os fatores 3, 5 e 6 mostram por onde cada zona dobra fora das potências de 2.
3. **Execução** em paralelo por (registro, segmento), no servidor.
4. **Análise:** acertos por condição × fator × intervenção × teto, McNemar exato pareado
   contra `none`, e a banda escolhida.

| intervenção | o que remove antes de decimar |
|---|---|
| `none` | nada (controle) |
| `stop_z1` | 25–37,25 kHz |
| `stop_z2` | 62,75–75 kHz |
| `stop_z3` | 75–87,25 kHz |
| `stop_z123` | as três zonas |
| `aaf_ideal` | tudo acima de `fs/(2·fator)`, ou seja, nenhuma dobra |

## Métricas

- acerto de ordem (`sem_prior`) por condição, fator, intervenção e teto, com `n`;
- McNemar exato pareado contra `none`;
- banda escolhida pelo kurtograma e curtose;
- atribuição medição/prior: **não se aplica**, porque não há reconstrução nem operador linear
  (ver `bhadra2021hallucinations` e o ADR 0006).

## Resultado

Rodado em 2026-11-11 no servidor remoto (CPython 3.14.10), com 6048 linhas em
`results/metrics.csv`, tabela em `results/summary.csv` e manifest com `source_sha256` e
`entrypoint_sha256`.

**Re-executado com o núcleo de DSP corrigido** pela auditoria independente de 2026-11-11
(kurtograma sem emenda circular, reamostragem angular com anti-alias; ver o caderno do dia).
Em relação à rodada provisória, nenhuma célula mudou mais de 2 acertos, e todas as conclusões
abaixo se mantêm.

**Controle:** `none` com teto de 25 kHz reproduz exatamente o E07 re-executado com o DSP
corrigido: 57, 53, 38 e 57 de 72 em 1×, 2×, 4× e 8×.

**Pista externa, teto de 25 kHz**, acertos de 36:

| intervenção | 1× | 2× | 3× | 4× | 5× | 6× | 8× |
|---|---|---|---|---|---|---|---|
| `none` | 25 | 21 | 4 | 6 | 2 | 11 | 24 |
| `stop_z1` (25–37,25 kHz) | 25 | 21 | 4 | 7 | 3 | 12 | 24 |
| `stop_z2` (62,75–75 kHz) | 25 | 21 | 4 | **22** | 12 | 12 | 25 |
| `stop_z3` (75–87,25 kHz) | 25 | 26 | **20** | **22** | 2 | **24** | 25 |
| `stop_z123` | 25 | 26 | 22 | **25** | 15 | 23 | 27 |
| `aaf_ideal` | 25 | 25 | 25 | **25** | **26** | 24 | 26 |

Em 4×, contra `none`: `stop_z2` e `stop_z3` dão p = 3,1·10⁻⁵, e `stop_z123` e `aaf_ideal` dão
p = 4·10⁻⁶ (McNemar exato pareado).

**Teto de 12,5 kHz:** a pista externa fica em 26–27/36 em todos os fatores de 1× a 6×, e em
24–27 em 8×, com qualquer intervenção. Pareado contra o teto de 25 kHz (`none`): 26 contra 4 em
3×, 27 contra 6 em 4×, 27 contra 2 em 5× e 27 contra 11 em 6×, sempre **sem nenhuma perda**
(p de 6,0·10⁻⁸ a 3,1·10⁻⁵).

**Pista interna:** fica entre 31 e 33 de 36 em todas as células, com os dois tetos.

**A banda escolhida em 4× na pista externa confirma o mecanismo:**

| condição | banda escolhida |
|---|---|
| `none`, teto 25 kHz | [12,75; 25] kHz em 26/36 |
| `stop_z123` | 2,0–4,6 kHz em 32/36 |
| `aaf_ideal` | 2,0–4,6 kHz em 32/36 |
| teto 12,5 kHz | 2,0–4,5 kHz em 32/36 |

**Leitura original (2026-11-11, manhã), corrigida abaixo.**
1. **Hipótese principal.** O vale vem de conteúdo acima do Nyquist efetivo que sobrevive à
   decimação pura. O filtro anti-alias ideal o elimina em todos os fatores. **Isso se sustenta.**
2. **H2 (atribuição).** "Em 4× a captura exige Z2 e Z3 juntas", e "em 3× e 6× é a Z3". **Retirado**;
   ver a revisão abaixo.
3. **H3 (teto).** Baixar o teto para 12,5 kHz elimina o vale, e a previsão de que isso pioraria a
   pista interna foi refutada. Isso se sustenta, mas o teto não distingue dobra de geometria da
   busca.

### Revisão adversarial (`reviewer-2`, 2026-11-11), conferida

**O captor é um trem de picos isolados de uma amostra no registro bruto, não conteúdo de banda de
62,75–87,25 kHz.**

- **Remoção de picos (*despike*) em taxa cheia, sem zerar nenhuma faixa.** Toda amostra que
  desvia mais de 10 MAD da média dos vizinhos, com os vizinhos desviando em sentido oposto, é
  trocada por essa média. Isso afeta cerca de 0,65 % das amostras. A pista externa fica em
  **26/26/26/26/25/23/27** de 36 de 1× a 8×, contra 25/21/4/6/2/11/24 em `none`. Empata com o
  `aaf_ideal` e supera o `stop_z123` em 5× (25 contra 15). A mediana de 3 amostras dá o mesmo.
  Na pista interna nada piora.
- **Conferência independente:** re-executei a tabela e quatro segmentos em 4×. Em O-B-2 e O-D-1,
  `none` escolhe 12,75–25 kHz e erra, e o *despike* volta a 2,54–4,58 kHz e acerta.
- **Os picos estão em todos os registros, inclusive nos saudáveis.** H-A-1 tem cerca de 2.460
  picos/s e O-A-1 cerca de 1.330/s, com desvio negativo em 96 % dos casos, a 16–18 MAD. Pelo
  revisor:
  - de 40 a 100 kHz o espectro é plano em ±1 dB;
  - o RMS em 62,75–87,25 kHz é o mesmo nas três condições.

  Um evento de uma amostra (5 µs) não é resposta de acelerômetro. Vem da cadeia de aquisição:
  interferência amostrada sem anti-alias ou artefato do sistema de aquisição.
- **"Z2 e Z3 juntas" não identifica duas fontes.** Em 4×, `stop_z2` e `stop_z3` dão o mesmo
  resultado em 36/36 segmentos. Mas isso é o esperado se as duas remoções apenas desfazem a captura
  e o seletor volta para 2–4,6 kHz, banda sobre a qual nenhuma das duas zonas dobra. Em 3×, 5× e
  6× elas não coincidem (11, 20 e 13 de 36).
- **Ponto em aberto.** Remover a Z1 (25–37,25 kHz), que em 4× também dobra sobre 12,75–25 kHz, não
  ajuda (7/36). Isso é compatível com a Z1 ter conteúdo vibratório genuíno forte (há uma
  ressonância perto de 34 kHz em taxa cheia) além dos picos. Não está isolado.

**Estatística.**
- **Nível de segmento.** O McNemar trata os 36 segmentos como independentes, mas são 3 por registro.
- **Nível de registro.** O efeito do anti-alias sobrevive: 11 registros melhoram, 1 empata e 0
  pioram, teste do sinal p = 9,8·10⁻⁴. É de 30 a 250 vezes maior que o p por segmento.
- **Nível de cenário.** Com 4 cenários de rotação, nenhum teste chega a p < 0,01.
- **Nível de rolamento.** Há um rolamento por tipo de falha.

**Parede física.** Decimar um arquivo de 200 kSPS não é adquirir a 50 kSPS.
- Se os picos são artefato digital do sistema de aquisição original, um ADC real abaixo do Nyquist
  não os reproduz.
- Se são interferência na cadeia analógica, um conversor abaixo do Nyquist sem passa-baixas, o
  caso da amostragem passa-faixa, os dobraria do mesmo jeito.

O conjunto não permite distinguir os dois casos.

**O que a Parte 1 pode afirmar.** Num conjunto público, picos esparsos de uma amostra no registro
bruto tornam a decimação pura não monótona no acerto. De 3× a 6×, o seletor de banda por curtose é
capturado por uma banda onde a energia dobrada dos picos domina. Anti-alias ou remoção de picos
antes de decimar desfazem o efeito.

A captura da curtose espectral por interferência elétrica impulsiva em taxa cheia é o objeto de
`smith2016emi` e `vaerenberg2025cyclicemi`. O que resta de próprio é estreito: a volta, por dobra,
do conteúdo que o teto da busca excluía em taxa cheia, e a dependência disso em relação ao fator.

**O que fica pendente.**
- *Despike* e mediana de 3 como braços pré-registrados aqui e no E22.
- Caracterização temporal do conteúdo acima de 40 kHz.
- Testes no nível de registro e de cenário.
- No E23, um braço de trem de picos de uma amostra contra rajada de banda limitada de mesma energia.

**O que surpreendeu.**
- O vale vai de **3× a 6×**. A grade do E07 (1, 2, 4, 8, ...) só mostrava o 4×.
- O teto de 25 kHz do E07 foi posto para tirar da busca o conteúdo de 50–100 kHz, que em taxa
  cheia capturava o seletor. **Sob decimação pura, esse mesmo conteúdo volta por dobra para
  dentro da faixa protegida.** Uma restrição de busca que protege o seletor em taxa cheia é
  desfeita pela subamostragem.
- O eixo do banco é acionado por inversor. A revisão mostrou que o captor é um trem de picos de
  uma amostra presente também nos registros saudáveis (ver acima). Se a origem é interferência
  eletromagnética ou artefato do sistema de aquisição, o conjunto não diz.

**O que isso muda para o artigo** (texto da manhã; o mecanismo foi corrigido pela revisão acima).
O efeito do anti-alias está estabelecido por intervenção no Ottawa, e o agente causal são os
picos de uma amostra. As duas mitigações simples (filtro anti-alias, teto da busca abaixo da faixa onde cai o
conteúdo dobrado) passam a ser **baselines obrigatórios** do E22: o método proposto tem de
empatar ou ganhar delas. O E21 testa se o fenômeno, e um preditor calculado a partir de uma
captura em taxa cheia, generalizam para o MAFAULDA.

## Re-execução com a régua corrigida (2026-11-11, tarde)

Rodada às 12:56 UTC no servidor, com três mudanças:
- verdade pelo **centro do platô** (`snq_recon.refine_order`);
- **régua local** gravada ao lado (`truth_local_order`, `hit_local`: a raia em taxa cheia do próprio
  segmento, porque a ordem de BPFO da pista externa deriva com a rotação);
- as intervenções pré-registradas `despike` e `median3`.

O controle bate exatamente com o E07 re-executado com a mesma régua: 62, 55, 41 e 61 de 72 em 1×,
2×, 4× e 8×. A saída com código 2 da rodada foi a trava de controle comparando com os valores da
régua antiga, e o config foi atualizado.

Pista externa, teto de 25 kHz, acertos de 1× a 8× (1, 2, 3, 4, 5, 6, 8):

| intervenção | régua antiga (de 36) | régua local (de 35) |
|---|---|---|
| `none` | 27 · 20 · **4 · 6 · 2 · 11** · 25 | 31 · 22 · **4 · 6 · 2 · 11** · 28 |
| `stop_z2` | 27 · 20 · 4 · 21 · 12 · 12 · 26 | 31 · 22 · 4 · 24 · 13 · 12 · 29 |
| `stop_z3` | 27 · 28 · 19 · 21 · 2 · 24 · 26 | 31 · 32 · 21 · 24 · 2 · 27 · 29 |
| `stop_z123` | 27 · 28 · 21 · 27 · 15 · 23 · 29 | 31 · 32 · 23 · 31 · 16 · 26 · 32 |
| `aaf_ideal` | 27 · 27 · 27 · 27 · 27 · 25 · 27 | 31 · 31 · 31 · 31 · 29 · 28 · 30 |
| **`despike`** | **28 · 28 · 28 · 28 · 27 · 24 · 29** | **32 · 32 · 32 · 32 · 30 · 27 · 32** |
| `median3` | 28 · 28 · 28 · 28 · 25 · 25 · 28 | 32 · 32 · 32 · 32 · 28 · 28 · 31 |

A pista interna fica em 35–36/36 em todas as células, nas duas réguas. Antes da correção da régua
eram 31–33.

**Leitura.**
- O vale de 3× a 6× é o mesmo nas duas réguas.
- A remoção de picos de uma amostra (cerca de 0,65 % das amostras, sem zerar faixa nenhuma) o
  desfaz por inteiro e empata com o anti-alias ideal ou o supera.
- Isso confirma, em rodada oficial e pré-registrada, o achado da revisão adversarial: o captor são
  os picos.
- A régua local sobe o patamar da pista externa em taxa cheia (27/36 → 31/35), o que mostra que
  parte dos erros de antes era da régua, não do método.

## Seletores robustos (pré-registro, 2026-11-12)

Escrito antes de rodar. A pergunta é a primeira que um revisor faz: um seletor de banda robusto a
impulsos evita o vale de 4× **sem** remover os picos?

- **Seletores:** de `snq_recon.selectors`, testados contra os valores do próprio artigo.
  - `infograma_ses`: a negentropia do espectro do envelope ao quadrado (Antoni 2016, eq. 12). Pela
    seção 5.1.1 do artigo, ele é insensível a ruído impulsivo.
  - `gini`: o índice de Gini do envelope da banda (Hebda-Sobkowicz et al. 2020, eq. 3).
  - Os dois rodam na mesma partição do kurtograma.
- **Braços** (`config/selectors.yaml`): `none` e `despike` repetem os do E20; os novos são
  `infograma_ses`, `infograma_ses_despike` e `gini`. Fatores, tetos, régua e controle do E07 ficam
  iguais.
- **Critério:** o mesmo da recuperação do E20, na pista externa, em 4× e com teto de 25 kHz.
  - Recupera: pelo menos 20 de 36 acertos e McNemar exato pareado contra `none` com p < 0,01.
  - Não recupera: até 10 de 36.
  - Vale para as duas réguas (a do E07 e a local).
- **Leitura:**
  - Se `infograma_ses` recuperar, um seletor robusto basta, e a correção não exige a taxa cheia.
  - Se não recuperar, a remoção dos picos antes de decimar continua necessária, e os seletores
    robustos da literatura não resolvem a captura sob decimação pura.

## Remoção depois de decimar (pré-registro, 2026-11-12)

Escrito antes de rodar. O braço `despike` remove os picos em taxa cheia, o que o nó não pode
fazer. A pergunta aqui é se a mesma remoção, aplicada ao fluxo **já decimado**, desfaz o vale.

- **O que muda:** só o momento da remoção.
  - Um pico de uma amostra em taxa cheia só sobrevive à decimação se cair numa amostra mantida.
  - Se cair, continua isolado no fluxo decimado.
  - O critério é o mesmo do E20 (10 MAD, vizinhos opostos e menores que 0,8 vez), calculado na
    taxa efetiva.
- **Braços** (`config/post_despike.yaml`): `none` e `despike` repetem os do E20, e o novo é
  `despike_decimado`.
- **Critério:** o da recuperação do E20, na pista externa, em 4× e com teto de 25 kHz.
  - Recupera: pelo menos 20 de 36 acertos e McNemar exato pareado contra `none` com p < 0,01.
  - Não recupera: até 10 de 36.
  - Vale para as duas réguas.
- **Leitura:** se recuperar, a correção é realizável no nó, porque não precisa da taxa cheia.

**Análise dos dois pré-registros acima** (escrita em 2026-11-12, antes de qualquer número das duas
rodadas):
- O script é `recovery.py <diretório>/metrics.csv`, que grava `recovery.csv` ao lado.
- Ele aplica o critério nas duas réguas: a aninhada, que é a coluna `hit`, e a local, `hit_local`.
- Também reporta o teste do sinal exato sobre os 12 registros da pista externa. Esse teste é
  sensibilidade, fora do pré-registro, e responde ao limite de "Nível de segmento" acima.
- Aplicado ao `results/` atual, o script reproduz a tabela da re-execução. `despike` em 4× fica em:
  - 28/36 contra 6/36, com 22 a 0 nos pares discordantes, na régua aninhada (p = 4,8·10⁻⁷);
  - 32/35 contra 6/35 na régua local (p = 3·10⁻⁸);
  - 11 registros a 0 no teste do sinal (p = 0,001).

