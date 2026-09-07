# E07 — degradação por subamostragem no domínio de ordem

## Hipótese

A ordem de falha sobrevive à subamostragem muito além do que a curva do E02 sugeria,
porque o E02 mediu a fronteira aritmética do CWRU e não a do método. Sob rotação
variável, com referência angular, a informação de diagnóstico está na **ordem** — que
é constante da geometria — e não na frequência em hertz, que se move com o regime.

## O que refutaria

O acerto cair monotonicamente com o fator de subamostragem, como caía no CWRU 12k.
**Não caiu.**

## Dados

University of Ottawa 2018 (Mendeley `v43hmbwxpm`, CC BY 4.0). 24 registros de falha
— 12 de pista interna, 12 de pista externa — em 4 cenários de rotação variável,
200 kSPS, acelerômetro no canal 1 e encoder no canal 2. Três janelas disjuntas de
2,5 s por registro: **72 medições por fator por modo**, contra 15 no E02.

O encoder tem **1024 pulsos por revolução**, valor não documentado pelos autores e
determinado empiricamente.

## O encoder não passa pelo ADC

Contagem de pulsos numa entrada digital — TIMER/GPIOTE no nRF54L15 — não consome
orçamento de amostragem. O que é decimado neste experimento é só a vibração; o perfil
de rotação acompanha porque varia devagar, e o CSV registra em cada linha quantos
pontos por revolução sobram. Segmentos com menos de dois pontos por incremento
angular são descartados em vez de interpolados às cegas.

## Resultado (2026-09-04) — a curva não desce

Acerto da ordem de falha, tolerância 0,06 ordens, 72 medições por ponto:

| fator | taxa efetiva | sem prior | prior nominal | prior calibrado |
|---|---|---|---|---|
| 1× | 200 kSPS | 78% | 50% | **83%** |
| 2× | 100 kSPS | 74% | 50% | 78% |
| 4× | 50 kSPS | 51% | 49% | 56% |
| 8× | 25 kSPS | 76% | 50% | 82% |
| 16× | 12,5 kSPS | 68% | 50% | 82% |
| 24× | 8,3 kSPS | 74% | 50% | 86% |
| 32× | 6,25 kSPS | 78% | 50% | **88%** |
| 48× | 4,17 kSPS | 74% | 50% | 81% |
| 64× | 3,12 kSPS | 64% | 51% | 73% |

**De 200 kSPS a 3,12 kSPS — fator 64 — o acerto não degrada.** Em 32× ele é maior do
que na taxa cheia, e a diferença está dentro do intervalo de Wilson: o que se mede é
ausência de degradação, não melhora.

A figura `e07_spectra` mostra por quê: a paisagem espectral é irreconhecível entre
200 kSPS e 3,1 kSPS, e a raia de ordem 5,38 com seu segundo harmônico em 10,76 fica
exatamente onde estava. A falha vive na modulação do envelope, não na portadora, e a
dobra move a portadora sem mover a modulação.

## O prior de catálogo erra, e erra em silêncio

A geometria do ER16K (9 esferas, esfera 0,3125", primitivo 1,516") prevê
BPFO = 3,573 e BPFI = 5,427. Medido em taxa cheia nos 24 registros:

| | nominal | calibrado | desvio |
|---|---|---|---|
| BPFI (pista interna) | 5,427 | 5,401 | **−0,5%** |
| BPFO (pista externa) | 3,573 | 3,414 | **−4,4%** |

O desvio é consistente nos 12 registros de cada condição e depende do cenário: os
cenários que começam acelerando (A, C) dão ordens sistematicamente mais altas do que
os que começam desacelerando (B, D) — 5,42–5,43 contra 5,38 na pista interna. O
mecanismo não foi determinado aqui. A soma medida BPFO + BPFI = 8,82 viola a
identidade geométrica BPFO + BPFI = n em 2%, o que descarta escorregamento puro de
gaiola como explicação única.

**O modo de falha importa mais do que o número.** Com o prior nominal, os registros
de pista externa não produzem estimativa ruim da BPFO: eles produzem **1,985** em 215
de 310 casos (69 %) — o quinto harmônico da FTF nominal, 5 × 0,397 = 1,985, perto da
ordem 2 do eixo. (O denominador publicado antes, 263, era sobra de uma rodada anterior
à truncagem em 64×; recontado em `results/metrics.csv` são 310.) A candidata nominal
3,573 não tem energia, então o escore escolhe outra família e **reporta a falha
errada com confiança**. Um prior desalinhado em 4,4% não degrada suavemente; ele
troca o diagnóstico. É exatamente a alucinação com boa aparência que a regra de
atribuição do `CLAUDE.md` existe para pegar.

Ressalva que não pode ficar implícita: o prior nominal "acerta" a pista interna
apenas porque |5,427 − 5,401| = 0,026 cabe na tolerância de 0,06. É acerto por folga
de tolerância, não por prior correto. Daí os 50% cravados em todos os fatores — 12
registros internos que passam raspando, 12 externos que erram sempre.

## O mergulho em 4× é a escolha de banda, não a taxa

Em 4× o acerto cai para 56% e volta a 82% em 8×. A causa está medida:

- Em 4× o kurtograma escolhe uma banda acima de 12 kHz em **91%** dos segmentos.
  Em 8× e além, essa banda deixa de existir (Nyquist efetivo abaixo de 12,5 kHz) e o
  acerto se recupera.
- Dos 32 erros em 4×, **22 vêm da banda 12,8–25 kHz**. Os acertos têm curtose mediana
  122; os erros, 26,7.
- A 50 kSPS o conteúdo de 75–87 kHz dobra exatamente sobre 12,8–25 kHz. Medindo a
  curtose dessa banda fixa no mesmo segmento, ela sobe **2,4× a 4,8×** de 1× para 4×
  nos registros de pista externa — justamente os que têm 1,8% a 3,6% da energia acima
  de 50 kHz e os que mais perdem acerto. Num registro de pista interna com 0,42% de
  energia lá em cima, a curtose cai em vez de subir.

Não é a taxa que falha: é a interação entre a taxa escolhida e o conteúdo fora da
banda de busca. É a preocupação do ADR 0002 medida em dado real, e é o argumento
empírico de que a taxa precisa ser **escolhida**, não apenas reduzida — que é
exatamente o problema em aberto que o projeto reivindica.

## O teto de 25 kHz no kurtograma inverteu uma conclusão

Sem teto, 12 de 12 registros de pista externa escolhiam a banda 50–100 kHz, com
curtose ~55, e a falha não aparecia em ordem nenhuma. A leitura fácil seria "a pista
externa deste dataset não é diagnosticável", e ela estaria errada: com o teto em
25 kHz, a mesma falha aparece na ordem 3,4 nos 12 registros.

**Correção de 2026-09-07 à justificativa deste teto.** Esta seção afirmava que a faixa
acima de 20 kHz carrega **0,1 %** da energia e que a curtose ali mede vazamento
numérico. Esse 0,1 % foi medido em **um** registro do dry-run e generalizado para 24.
Recomputado sobre os **n = 24** registros de falha (canal de vibração, FFT do registro
inteiro): **mediana 5,02 %** acima de 20 kHz, mín 2,97 %, máx 6,83 % — cerca de 50× o
valor publicado (`docs/lab-notebook/2026-09-07.md`). O **efeito** do teto continua
medido e não muda; o que cai é a explicação. A faixa não é numericamente vazia, então
o teto de 25 kHz não é consequência de energia nula: é um **prior de banda**, e é
assim que deve ser declarado.

O guarda `min_energy_frac` do kurtograma existe para essa degeneração e o limiar de
1e-4 é frouxo demais para este dataset.

## Verdade fundamental, e por que não é circular

A ordem de referência de um registro é a **mediana das ordens medidas em taxa cheia
nos outros registros da mesma condição** — nunca a do próprio. Sem isso o fator 1
seria acerto por construção no modo sem prior. É também o que um comissionamento real
teria: calibração em unidades da mesma família, não no equipamento sob teste.

A dispersão entre registros da mesma condição é ~0,035 em ordens, e a tolerância de
0,06 a cobre. Essa dispersão é um piso de erro irredutível deste desenho.

## Controle negativo: o estimador inventa falha onde não há?

Acerto medido só sobre registros que **têm** falha não diz nada sobre falso positivo,
e um estimador restrito a um conjunto de candidatas sempre devolve uma candidata — ele
não tem como responder "nenhuma". Sem este controle, os 88% em 32× seriam compatíveis
com um estimador que aponta sempre a mesma ordem.

Os 12 registros saudáveis (`H-*`) entram como negativo, e a separação é medida por AUC
— a probabilidade de um segmento com falha pontuar acima de um segmento saudável na
ordem alvo. 0,5 é ausência de separação.

| fator | taxa efetiva | AUC BPFI | AUC BPFO |
|---|---|---|---|
| 1× | 200 kSPS | 1,000 | 0,958 |
| 4× | 50 kSPS | 1,000 | 0,743 |
| 8× | 25 kSPS | 0,999 | 0,902 |
| 16× | 12,5 kSPS | 1,000 | 0,827 |
| 32× | 6,25 kSPS | 0,988 | 0,923 |
| 64× | 3,12 kSPS | 1,000 | 0,995 |

**A separação não degrada.** Em 3,12 kSPS os registros com falha ficam com a raia da
ordem alvo no topo da faixa em praticamente todos os segmentos, e os saudáveis não.
O mergulho da pista externa em 4× é o mesmo sequestro de banda descrito acima.

### A primeira métrica dizia o contrário, e estava errada

A versão inicial normalizava a energia harmônica pela **mediana do espectro**. Com ela,
a separação parecia colapsar a partir de 8× e chegava a **inverter** — AUC de 0,15,
com os saudáveis pontuando acima dos defeituosos. Era artefato: sob subamostragem forte
o espectro de um rolamento saudável fica pobre em raias e a mediana desaba, inflando o
quociente sem que nada físico tenha mudado. O piso se movia junto com a métrica.

A estatística usada agora é a **posição relativa da raia na faixa de ordens** — a
fração de raias abaixo do pico na ordem alvo. Livre de escala e de piso. A lição vale
além deste experimento: uma métrica normalizada por estatística do próprio sinal
precisa ser verificada contra a subamostragem, porque é o denominador que muda
primeiro.

## O que este experimento não sustenta

- **Não é uma comparação de classificação.** Mede erro de estimação de ordem e
  separação falha/saudável, não discriminação entre tipos de falha. O E16 é quem faz
  classificação multiclasse com poder estatístico.
- **Não mede consumo.** A taxa efetiva está em toda linha do CSV, mas µJ por
  diagnóstico ainda não existe no código (E06).
- **Não testa aquisição não-uniforme.** Decimação uniforme apenas. Multi-coset e
  agendamento aprendido são o Paper 2.
- **O piso de 500 Hz na busca do kurtograma consome fração crescente da banda
  disponível** conforme o fator sobe: em 64× ele corta um terço do que resta. Os dois
  últimos pontos da curva carregam essa restrição junto com a subamostragem.

## Reprodução

```bash
uv run python experiments/E07-order-domain-ottawa/run.py \
    --config experiments/E07-order-domain-ottawa/config/base.yaml
uv run python papers/shared/figures/fig_e07_order_domain.py
uv run python papers/shared/figures/fig_e07_spectra.py --record I-B-1.mat
uv run python experiments/E07-order-domain-ottawa/control_healthy.py     --config experiments/E07-order-domain-ottawa/config/base.yaml
uv run python papers/shared/figures/fig_e07_control.py
```

O download é programático pela API pública do Mendeley, sem cadastro.
`results/manifest.json`, como está gravado, tem as ordens calibradas por registro, o
SHA do config e a versão de cada pacote — ele é anterior aos campos de procedência
acrescentados em 2026-09-07 e não foi reescrito, porque regravar um manifest sem
repetir a medição é inventar procedência. Quem rodar de novo terá os campos novos, que
a seção "O que o manifest passou a gravar" lista.

---

# Extensão — `config/extended.yaml`

Seção acrescentada em 2026-09-07. Não substitui nada acima: a tabela de resultados
daquela seção continua sendo a da rodada `base.yaml`, gravada em `results/metrics.csv`.

## Hipótese

Depois de subtraído, **fator a fator**, o degrau de guarda medido nos nove fatores que
as duas rodadas têm em comum, o acerto com prior calibrado nos fatores **96, 128 e
160** permanece **dentro de 13 pontos percentuais** do valor corrigido em 64×. Ou
seja: a curva do E07 não desce entre 3,12 kSPS e 1,25 kSPS, e a parada em 64× era
mesmo um artefato de guarda, não um joelho.

**De onde vêm os 13 pontos.** Com `n = 72` medições por ponto e acerto em torno de
0,80, o intervalo de Wilson a 95 % de um único ponto tem meia-largura de ±9,1 pontos, e
a diferença entre dois pontos independentes de `n = 72` tem meia-largura normal de
1,96·√(2·0,8·0,2/72) = **13,1 pontos**. Treze pontos é, portanto, a menor queda que
este desenho distingue de ruído de amostragem: abaixo disso não há afirmação a fazer, e
acima disso há. O limiar é uma consequência do `n`, não uma preferência — dobrar
`segments_per_record` o apertaria, ao custo de janelas menores e de um confundidor novo.

## O que refutaria esta hipótese

Quatro rotas, cada uma com o número que a dispara. Nenhuma delas é "o resultado ficou
feio":

1. **Queda além de 13 pontos.** Se o acerto corrigido com prior calibrado em 96×, 128×
   ou 160× cair mais de 13 pontos abaixo do corrigido em 64×, a hipótese está refutada:
   o joelho existe e fica entre 64× e 160×, e o fator publicável do E07 continua sendo
   64. Esta é a rota que decide se a extensão vale um ponto de curva ou uma nota de
   rodapé.
2. **Captura por uma ordem fixa.** Se, em qualquer um dos três fatores novos, a
   estimativa do modo com prior calibrado se concentrar em **mais de metade** dos
   segmentos numa única ordem que não é a verdade do registro — em especial a **1,985**,
   o harmônico de FTF perto da ordem 2 (FTF nominal 0,397) que já capturou 215 de 310
   casos de pista externa sob o prior nominal —, então o número não é medição de ordem
   de falha: é ponto fixo do estimador. Nesse caso a curva estendida é nula
   **independentemente do acerto**, inclusive se o acerto for alto. O CSV já tem a
   coluna `estimate_order`; o teste é um histograma.
3. **Degrau de guarda não aditivo.** A subtração do item 1 só é legítima se o degrau
   medido nos nove fatores repetidos for um deslocamento, e não uma função do fator. Se
   ele tiver tendência com o fator nesses nove pontos, a extrapolação para 96–160 não
   se sustenta, o desenho **não separa** guarda de taxa, e a rodada deve ser declarada
   inconclusiva em vez de publicada. A medição parcial já feita (tabela abaixo) mostra o
   degrau variando entre −1,4 e −11,1 pontos em seis fatores: esta rota está longe de
   ser hipotética.
4. **`n` desigual entre fatores.** Se algum dos doze fatores entrar com menos de 72
   medições, a comparação vira troca de denominador — o defeito já encontrado na tabela
   do E02 48k. Esta rota está **executada e fechada**: `tests/test_guards.py` verifica,
   sobre os 24 registros × 3 janelas reais, que os doze fatores admitem 72 de 72.

Estas quatro rotas testam a **extensão**. A refutação registrada no topo deste
README — "o acerto cair monotonicamente com o fator" — pertence ao E07 base, já foi
respondida com a curva de `results/metrics.csv`, e não cobre nada do que esta seção
afirma.

## Por que existe

A varredura de `base.yaml` parou por **critério de análise, não por queda de
desempenho**. Contado em `results/metrics.csv`: 216 linhas por fator (72 medições × 3
modos) até o fator 48, **135 no fator 64** — 45 das 72 — e **zero nos fatores 96 e
128**, que estavam na lista e nunca produziram medição.

O último ponto publicável da curva é portanto 64×, e ele já entra com 62% da amostra
dos demais. É o ataque mais previsível de um revisor, e é a intervenção que mais move
o número defensável: decide se o fator continua em 6,4× ou vai substancialmente além.

## Os três guardas, e qual deles realmente bloqueia

`run.py` descarta um segmento em três testes, todos em `guard_verdict`, na ordem em
que são aplicados (o primeiro a disparar é o que conta):

| ordem | guarda | parâmetro |
|---|---|---|
| 1 | `n_samples < cfg["min_samples"]` | `min_samples` |
| 2 | `fs_eff/2 <= cfg["kurtogram"]["f_min_hz"]` | `f_min_hz` |
| 3 | `fs_eff/max(v) < 2·cfg["samples_per_rev"]` | `samples_per_rev` |

Com `FS_HZ = 200000` e `segment_samples = 500000`, e usando as duas rotações de pior
caso — 1953,1 rpm, o máximo do registro I-A-3 lido na coluna `rpm_hi` de
`results/metrics.csv`, e 1674,1 rpm, o máximo medido dentro de um segmento
efetivamente analisado (o pico de I-A-3 cai nas últimas 500000 amostras, que
`segments_per_record: 3` nunca alcança):

| fator | `fs_eff` | `n = ceil(500000/f)` | `fs_eff/2` | pts/rev @1953,1 | @1674,1 |
|---|---|---|---|---|---|
| 64 | 3125,00 Hz | 7813 | 1562,50 Hz | 96,00 | 112,00 |
| 96 | 2083,33 Hz | 5209 | 1041,67 Hz | 64,00 | 74,67 |
| 128 | 1562,50 Hz | 3907 | 781,25 Hz | 48,00 | 56,00 |
| 160 | 1250,00 Hz | 3125 | 625,00 Hz | 38,40 | 44,80 |
| 192 | 1041,67 Hz | 2605 | 520,83 Hz | 32,00 | 37,33 |
| 256 | 781,25 Hz | 1954 | 390,62 Hz | 24,00 | 28,00 |

Lendo guarda a guarda:

- **`samples_per_rev: 64`** exige pts/rev ≥ 128. É ele que descarta 27 das 72
  medições do fator 64 (pior segmento: 112,00) e que zera o 96 (74,67) e o 128
  (56,00). É o guarda dominante.
- **`min_samples: 4096`** zera o 128 por conta própria: `ceil(500000/128) = 3907`.
  Não toca o 96 nem o 64.
- **`f_min_hz: 500` não bloqueia nem o 96 nem o 128.** O guarda pede
  `fs_eff/2 > 500`, e no fator 128 isso é `781,25 > 500`. Ele só passa a vincular a
  partir do fator 200 (`fs_eff/2 = 500,0`). A leitura de que o piso de 500 Hz zerava
  os dois fatores — registrada em `docs/plano-de-experimentos.md`, T1.1 — **não se
  sustenta no código**.

O `extended.yaml` relaxa portanto só dois guardas: `min_samples: 4096 → 2048` e
`samples_per_rev: 64 → 16`. **`f_min_hz` fica em 500 Hz**, deliberadamente.

Isto está **executado**, não previsto, e o que foi executado é a **admissão de
segmento** — quantas das 72 janelas cada fator aceita — não o acerto.
`tests/test_guards.py` aplica `run.guard_verdict` (a mesma função que `run.py` usa no
laço) aos 24 registros × 3 janelas reais e verifica duas coisas:

- sob `base.yaml`, o resultado é 72 medições por fator até 48, **45** no 64, **zero**
  em 96 e 128 — que é linha a linha o que `results/metrics.csv` contém (216 linhas por
  fator até 48, 135 no 64), e é isso que autoriza usar a mesma função para falar de
  fatores ainda não rodados;
- sob `extended.yaml`, **72 de 72 nos doze fatores**, inclusive 96, 128 e 160.

Nenhum ponto da curva estendida entra com `n` reduzido — o que fecha a rota 4 de
refutação e evita a troca de denominador que já contaminou a tabela do E02 48k. O
acerto nesses fatores continua **não medido**: é o que a rodada existe para descobrir.

## A advertência que não pode ficar implícita

**A rodada estendida mede subamostragem MAIS relaxamento de guarda.** Os dois efeitos
se confundem, e o mecanismo é concreto:

1. **Baixar `samples_per_rev` degrada a resolução angular.** Com `spr = 16` o Nyquist
   do espectro de **ordem** cai de 32 para 8. Isso muda a estimativa por dois
   caminhos independentes: `score_fault` passa a **dobrar** harmônicos que antes
   cabiam abaixo do Nyquist (`diagnosis.py:114-119`), mudando a composição do escore;
   e `estimate_from_candidates` **descarta** candidatas acima do Nyquist do espectro
   (`diagnosis.py:185`), encolhendo o conjunto de candidatas. Medido: das 24
   candidatas do prior nominal sobrevivem **23 sob `spr = 64`** e apenas **12 sob
   `spr = 16`**.
   O que *não* muda é o passo em ordens: `order_spectrum` devolve
   `rfftfreq(n, 1/spr)` com `n = revoluções·spr`, logo o passo é `1/revoluções`,
   independente de `spr` — conferido em 0,0261 nos fatores 64 a 160.
2. **Baixar `samples_per_rev` move a PRÓPRIA VERDADE.** A ordem de referência não é
   externa: sai da mesma reamostragem angular, no passe 1 de `run.py`, que usa o `spr`
   do config. A rodada estendida, portanto, não compara estimativas contra a mesma
   régua da rodada base — ela recalibra a régua. Medido, repetindo o passe 1 nos 24
   registros sob os dois `spr` (taxa cheia, mesma banda de kurtograma):

   | | `spr = 64` | `spr = 16` |
   |---|---|---|
   | mediana da calibração, pista interna | 5,4008 | 5,4008 |
   | mediana da calibração, pista externa | 3,4142 | 3,4167 |

   Por registro o deslocamento tem mediana **0,005 ordens** e máximo **0,095**
   (O-A-2), e **1 dos 24** se move além da tolerância de 0,06. A verdade que entra na
   comparação é a mediana deixa-um-de-fora de 11 registros, e por isso ela se move
   pouco — no máximo 0,0025 ordens, contra tolerância 0,06 e dispersão entre registros
   de 0,035. O confundidor existe, é de segunda ordem **nesta magnitude**, e precisa
   ser recontado se `spr` cair mais.
3. **Baixar `f_min_hz` deixaria o kurtograma buscar perto de DC**, onde está o
   conteúdo estrutural. Este config **não** baixa, e a advertência fica registrada
   para quem for tentado a baixar: recomputado sobre os **n = 24** registros de
   falha, canal de vibração, FFT do registro inteiro, a energia abaixo de 500 Hz tem
   **mediana 14,86 %** (mín 1,14 %, máx 22,06 %) —
   `docs/lab-notebook/2026-09-07.md`. **Não são os 97 %** que o comentário de
   `config/base.yaml` (97 % abaixo de 500 Hz) e o docstring de
   `packages/snq-data/src/snq_data/ottawa.py` (97 % abaixo de 50 Hz) afirmam; esse
   número saiu de um registro e foi generalizado para 24. Como a faixa abaixo de 50 Hz
   está contida na de 500 Hz, a mediana de 14,86 % limita as duas afirmações de cima.

**O sinal do viés é medido, e é o contrário do que esta seção afirmava.** A versão
anterior dizia que o conjunto menor de candidatas "aponta na direção de melhorar o
acerto" e que o efeito é um deslocamento constante. As duas coisas foram medidas
diretamente — reprocessando os mesmos segmentos sob os dois `spr`, com a calibração da
rodada base — e as duas estão erradas. Acerto do modo com prior calibrado, comparação
**pareada** (só os segmentos que passam nos dois guardas):

| fator | `spr = 64` | `spr = 16` | degrau | n |
|---|---|---|---|---|
| 1× | 83,3 % | 76,4 % | **−6,9** | 72 |
| 8× | 81,9 % | 77,8 % | **−4,2** | 72 |
| 16× | 81,9 % | 73,6 % | **−8,3** | 72 |
| 32× | 87,5 % | 76,4 % | **−11,1** | 72 |
| 48× | 80,6 % | 79,2 % | **−1,4** | 72 |
| 64× | 73,3 % | 71,1 % | **−2,2** | 45 |

(A coluna `spr = 64` reproduz a tabela publicada da rodada base — 83, 82, 82, 88, 81,
73 % — o que confere o procedimento.) O relaxamento **piora** o acerto em todos os seis
fatores medidos, mediana −5,5 pontos, e o modo sem prior piora mais, de −4,4 a −22,2
pontos. O degrau também **não é constante**: varia de −1,4 a −11,1 entre fatores, o que
é a rota 3 de refutação acima e a razão de a subtração ter de ser feita fator a fator,
nunca com um valor único.

**Como separar os dois efeitos — é parte do experimento, não uma sugestão.**
`extended.yaml` re-roda a lista **inteira** de fatores, não só os novos, e os nove
fatores já medidos (1 a 64) são o braço de controle desta rodada. Eles aparecem duas
vezes:

- em `results/metrics.csv`, sob `spr = 64` e `min_samples = 4096`;
- em `results/extended/metrics.csv`, sob `spr = 16` e `min_samples = 2048`.

A diferença linha a linha nesses nove fatores é **efeito de guarda puro** — a taxa é
idêntica, os segmentos são os mesmos. Só depois de subtraí-la, fator a fator, é que a
queda entre 64×, 96×, 128× e 160× pode ser lida como efeito de taxa. Uma curva
estendida publicada sem esse controle não distingue as duas causas e não deve ser
usada; se o braço de controle não for rodado junto, não há experimento, só uma segunda
curva incomparável com a primeira.

Confundidores deliberadamente evitados: `segment_samples`, `segments_per_record`,
`tolerance_order`, `n_harmonics`, `score_harmonics`, `calibration_span`, `order_min`,
`f_min_hz`, `f_max_hz`, `max_level` e a geometria nominal ficam **idênticos** ao base,
e `seed` também (`run.py` não usa RNG; manter o valor mantém a comparação exata).

## Por que a lista para em 160×

Não é o guarda de pontos por revolução — com `spr = 16` o fator 192 ainda passaria
(37,33 contra o limiar de 32). É a **janela de busca do kurtograma**, `[f_min_hz,
fs_eff/2]`, que colapsa contra o piso de 500 Hz:

| fator | janela | largura | sub-banda (nível 4) | bins de FFT por sub-banda |
|---|---|---|---|---|
| 64 | 500–1562,50 Hz | 1062,50 Hz | 66,406 Hz | 166,0 |
| 96 | 500–1041,67 Hz | 541,67 Hz | 33,854 Hz | 84,7 |
| 128 | 500–781,25 Hz | 281,25 Hz | 17,578 Hz | 44,0 |
| 160 | 500–625,00 Hz | 125,00 Hz | 7,812 Hz | 19,5 |
| 192 | 500–520,83 Hz | 20,83 Hz | 1,302 Hz | 3,3 |

No 192 cada sub-banda do nível mais fino tem 3 bins: a seleção de banda deixa de ser
medição. A descida já é visível antes disso — a curtose da banda escolhida em
I-A-3 segmento 0 vai de 3,6 (64×) a 1,9 (96×), 2,1 (128×) e **0,5** (160×). E no 256
dois guardas zeram o fator: `min_samples` dispara primeiro (`ceil(500000/256) = 1954`,
abaixo dos 2048 do `extended.yaml`) e, se não disparasse, `fs_eff/2 = 390,62 ≤ 500`
zeraria igual. É o que `tests/test_guards.py` fixa para o fator 256.

Isto reformula a ressalva da seção "O que este experimento não sustenta": o piso de
500 Hz não é só uma restrição que acompanha os últimos pontos — **é o guarda que
define onde esta cadeia de análise termina**.

## Correção correlata, feita em `extended.yaml`

`config/base.yaml` afirma, sob o rótulo "Teto medido, nao arbitrado", que acima de
20 kHz há **0,1 %** da energia. Recomputado sobre os **n = 24** registros de falha:
**mediana 5,02 %** (mín 2,97 %, máx 6,83 %) — cerca de 50× o valor citado
(`docs/lab-notebook/2026-09-07.md`). A mesma afirmação estava repetida acima neste
README, na seção "O teto de 25 kHz no kurtograma inverteu uma conclusão", e **foi
corrigida lá também** — corrigir só o comentário do config deixaria o número errado
vivo no documento que as pessoas leem.

Os dois números errados desta família, e onde cada um ainda está:

| afirmação | onde | medido sobre n = 24 |
|---|---|---|
| 0,1 % da energia acima de 20 kHz | `config/base.yaml`, e antes neste README | mediana **5,02 %** (2,97–6,83) |
| 97 % da energia abaixo de 500 Hz | `config/base.yaml` | mediana **14,86 %** abaixo de 500 Hz (1,14–22,06) |
| 97 % da energia abaixo de 50 Hz | docstring de `snq_data/ottawa.py` | contido na linha acima, logo ≤ 14,86 % |

A terceira linha está fora do escopo desta correção — `packages/` não é editado a
partir de um experimento — e fica registrada como pendência.

**O teto de 25 kHz continua em 25 kHz e o efeito relatado continua medido**: sem ele,
12 de 12 registros de pista externa escolhiam 50–100 kHz e a falha não aparecia. O
que cai é a justificativa. A faixa acima de 20 kHz não é numericamente vazia, então
"ali a curtose mede vazamento numérico" fica sem base, e o teto passa a ser o que
sempre foi na prática: um **prior de banda**, declarado como prior e não como
consequência de energia nula. `extended.yaml` carrega o comentário corrigido.

`config/base.yaml` **não** foi editado de propósito: seu SHA-256 está gravado em
`results/manifest.json` como procedência da rodada base, e alterá-lo invalidaria esse
registro sem alterar nada do que foi medido.

## Reprodução da extensão

```bash
# antes da varredura: os guardas, sobre os dados reais (pula sem os .mat)
uv run pytest experiments/E07-order-domain-ottawa/tests -q

uv run python experiments/E07-order-domain-ottawa/run.py \
    --config experiments/E07-order-domain-ottawa/config/extended.yaml
```

Todo parâmetro que a extensão toca já vivia no config. A saída vai para
`results/extended/`, no padrão de `E02-bandpass-envelope-cwru/config/cwru48k.yaml`, que
grava em `results/48k/`; `results/metrics.csv` fica intacto.

O caminho do teste precisa ser explícito porque `testpaths` cobre `packages/`, e isso é
deliberado: nada em `packages/` importa `experiments/`. O teste vive dentro do
experimento e só importa o `run.py` dele.

### O que o manifest passou a gravar

`write_manifest` gravava `git_sha`, `config_sha256`, versões de pacote e
`platform.python_version()`. Isso **não identifica o código que rodou**: o commit não
descreve uma árvore suja, e a versão de pacote (`0.1.0`) não muda quando o fonte muda.
Acrescentados:

| campo | o que resolve |
|---|---|
| `platform` | `platform.platform()` completo, exigido por `experiments/_template/run.py:45-55` |
| `input_sha256` | SHA-256 de cada `.mat` efetivamente carregado, por nome |
| `git_dirty` / `git_dirty_paths` | se verdadeiro, o `git_sha` **não** descreve o que rodou, e os caminhos sujos ficam nominados |
| `source_sha256` (+ `_files`, `_scheme`) | hash do conteúdo dos fontes importados: SHA-256 sobre `"<caminho>\n<sha256 do arquivo>\n"` de cada `packages/*/src/**/*.py`, em ordem de caminho POSIX. Muda quando o fonte muda, com ou sem commit |
| `runner_sha256` | SHA-256 do próprio `run.py`, que também não é descrito pelo commit quando a árvore está suja |
| `samples_per_rev` / `min_samples` | os dois guardas relaxados, para que uma comparação entre `results/` e `results/extended/` não dependa de reabrir os YAML |

`results/manifest.json` da rodada base é anterior a esses campos e **não** foi
reescrito: ele é a procedência do que foi medido, e regravá-lo sem repetir a medição
inventaria procedência. O mesmo vale para `config/base.yaml`, cujo SHA-256 está
gravado ali.
