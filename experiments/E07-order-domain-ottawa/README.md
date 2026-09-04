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
de 263 casos — um harmônico de FTF perto da ordem 2 do eixo. A candidata nominal
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
externa deste dataset não é diagnosticável", e ela estaria errada: essa faixa carrega
0,1% da energia, e a curtose ali mede vazamento numérico, não transiente. Com o teto
em 25 kHz — número que vem da medição de energia do dry-run, não de arbítrio — a
mesma falha aparece na ordem 3,4 nos 12 registros.

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

O download é programático pela API pública do Mendeley, sem cadastro. `results/manifest.json`
grava as ordens calibradas por registro, o SHA do config e a versão de cada pacote.
