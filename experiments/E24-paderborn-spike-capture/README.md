# E24 - Captura do seletor por picos sob decimação pura, no Paderborn

Réplica da Parte 1 do ADR 0006 num segundo conjunto real. No Ottawa, picos de uma amostra no
registro bruto dobram sob decimação pura e capturam o kurtograma, o que produz o vale de 4×. A
remoção dos picos antes de decimar desfaz o vale (E07, E20).

Na checagem exploratória de 2026-11-12 (`docs/lab-notebook/2026-11-12.md`), o Paderborn mostrou o
mesmo tipo de pico:
- de 2 a 730 por segundo, de ~12 MAD;
- eles respondem pela curtose do sinal bruto, de 13 a 16;
- levam o kurtograma para 11–21,5 kHz, a faixa que contém o chaveamento de 16 kHz do inversor.

## Hipóteses (pré-registro, 2026-11-12)

Escrito e commitado antes de rodar. Mudar depois de ver um número invalida o pré-registro.

- **H1 (réplica).** Remover os picos antes de decimar (critério do E20) aumenta a acurácia
  balanceada de ordem do pipeline clássico em pelo menos um dos orçamentos 1/4, 1/8, 1/16 e 1/32.
- **H2 (seletor robusto).** Trocar o kurtograma pelo infograma SES (Antoni 2016), sem remover os
  picos, aumenta a mesma acurácia em pelo menos um desses orçamentos.
- **H3 (remoção realizável; pré-registrada em 2026-11-12, antes de qualquer número do E24).**
  Remover os picos do fluxo **já decimado**, pelo mesmo critério, aumenta a acurácia do
  pipeline clássico em pelo menos um dos orçamentos. É o `config/post_despike.yaml`, que tem o
  mesmo bootstrap e o mesmo Holm. Se a H3 valer, a correção não exige a taxa cheia.

## O que refutaria

- **H1:** nenhum dos quatro testes significativo depois do Holm, ou diferença pontual negativa. A
  captura por picos não se reproduziria fora do Ottawa.
- **H2:** o mesmo critério. O seletor robusto não bastaria sob decimação pura.

## Baseline

O pipeline clássico do E07 e do E22: decimação pura, kurtograma (nível 4, piso de 500 Hz, teto no
Nyquist efetivo), espectro de envelope em ordens e argmax sem prior acima da ordem 2. É o braço
`kurtograma`.

## Verdade fundamental

As ordens de falha vêm da geometria do FAG 6203 (PHME16, tabela 1): BPFO 3,054 e BPFI 4,946. Elas
são calibradas pelos dados de duas formas.

- **Régua aninhada (principal).** É a mediana, na mesma pista, das calibrações válidas da medição
  1 dos **outros** rolamentos (deixa-um-rolamento-de-fora).
  - A calibração usa taxa cheia, sinal bruto, refino em ±1 % da nominal e raia visível (pico de
    pelo menos 5× a mediana).
  - Ela não depende de nenhum braço e vale para todos os segmentos.
- **Régua local (secundária).** É a raia em taxa cheia do próprio segmento, pelo mesmo critério.
  Só vale onde a raia é visível.

A tolerância é de 0,025 ordem: a BPFO e a BPFI ficam a 0,054 das harmônicas 3 e 5 do eixo.

## Procedimento

- **Dados:** os do E15 no Paderborn.
  - 23 rolamentos com dano de uma pista (12 de pista externa e 11 de pista interna), nos três
    ajustes de 1500 rpm.
  - As medições 2 a 20 são os segmentos, de 255936 amostras. A medição mais curta que isso ou
    ilegível fica de fora.
- **Braços** (`config/base.yaml`): `kurtograma`, `kurtograma_sem_picos`, `infograma_ses`,
  `infograma_ses_sem_picos` e `gini`.
  - A remoção de picos vem antes de decimar, em taxa cheia.
- **Fatores:** 1, 4, 8, 16 e 32. O fator 1 é descritivo. A guarda é de 1,0 × 64 pontos por volta.
- **Inferência:** bootstrap por aglomerado, com aglomerado = rolamento e reamostragem dentro de
  cada pista.
  - São 4000 reamostras, e o p unilateral é (1 + #{Δ* ≤ 0})/(1 + B).
  - Holm sobre os 4 testes de cada hipótese, pela régua aninhada; α = 0,05.
  - A régua local é reportada com o mesmo cálculo, como secundária.

## Métricas

- Acurácia balanceada de ordem: a média, sobre as pistas, do acerto médio por rolamento.
- A banda escolhida por braço e fator, para mostrar para onde os picos levam o seletor.
- Não há reconstrução de forma de onda, então a decomposição espaço-imagem/núcleo não se aplica.

## Resultado (2026-11-12)

**Proveniência.**
- Rodada base: `config/base.yaml`, commit `fb091b8`, das 14:13 às 20:52 UTC.
- H3: `config/post_despike.yaml`, commit `6a95a72`, das 15:24 às 20:52 UTC, num checkout separado.
- Os hashes de `run.py`, config, README e fonte (`8491841f…`) no manifest batem com esses commits.
- São 69 registros e 1305 segmentos. A régua local cobre 7 rolamentos.

**Vereditos pré-registrados** (a régua aninhada decide; Holm sobre os 4 orçamentos; α = 0,05):

| hipótese | resultado (régua aninhada) | veredicto |
|---|---|---|
| H1, remover os picos antes de decimar | +2,1 pontos em 8× [0,15; 4,49], p de Holm = 0,04999 (4 × 50/4001); nos outros orçamentos, +0,8, +1,6 e +2,2, sem significância | **apoiada só em 8×, no limite** |
| H2, infograma SES sem remover os picos | +3,1 pontos em 32× [0,95; 5,78], p de Holm = 0,001; em 4×, +8,8 [−0,1; 19,1], p de Holm = 0,08 | **apoiada em 32×** |
| H3, remover os picos depois de decimar | de −1,0 a +0,7 ponto, nenhum significativo | **não apoiada** |

Pela régua local (7 rolamentos), nenhuma das três passa.

**Descritivo:**
- **Não há vale.** A acurácia balanceada do kurtograma, pela régua aninhada, é de 7,7, 16,1, 18,6,
  19,2 e 13,0 % em 1×, 4×, 8×, 16× e 32×. A pior é a da taxa cheia.
- **A captura já acontece em taxa cheia.** O kurtograma escolhe 11,0–21,5 kHz em 909 segmentos,
  justamente a faixa que contém o chaveamento de 16 kHz do inversor.
  - Remover os picos antes de decimar não o tira dali: 350 segmentos continuam em 11,0–21,5 kHz,
    e 735 dos 955 restantes (77 %) ficam acima de 16 kHz.
  - Sob decimação, a escolha vai para a banda mais baixa da busca, 0,5–1,1 kHz em 4× e 8×.
- **Aritmética da dobra (hipótese, não testada).** A 64 kS/s, com os fatores 4, 8, 16 e 32, todo
  múltiplo de 16 kHz cai exatamente em 0 Hz. As bandas laterais do chaveamento cairiam então no
  fundo da faixa de busca.
- **Visibilidade da raia.** Só 7 dos 23 rolamentos têm raia visível em algum segmento: KA01, KA06,
  KA16, KI01, KI16, KI17 e KI18.
  - A visibilidade é medida na banda que o kurtograma escolhe em taxa cheia, e essa banda está
    capturada (`line_calibration`).
  - Nos outros 16 rolamentos, o kurtograma não acerta nenhum segmento em taxa cheia pela régua
    aninhada, o que vale por construção, porque a visibilidade é medida no mesmo espectro. O
    infograma SES chega a 17,0 % de acurácia balanceada neles: 34,0 % na pista externa e 0 %
    na interna.
  - Por isso, "sem raia visível" é em parte uma falha do seletor, e não só do sinal.

**Leitura:**
- O vale do Ottawa não se reproduz no Paderborn. Ali o seletor já é capturado em taxa cheia pela
  faixa do inversor, que é o cenário de `smith2016emi`.
- A remoção de picos de uma amostra muda pouco. A H1 passa no limite, com efeito pequeno, e não
  sustenta que "a captura por picos se reproduz fora do Ottawa".
- O infograma SES ajuda no Paderborn, ao contrário do Ottawa.
- **Antes de entrar no artigo:** revisão adversarial.

### Revisão adversarial (`reviewer-2`, 2026-11-12): revisão maior

O revisor reproduziu o `hypotheses.csv` a partir do `metrics.csv` com as funções do `run.py`. Os
números e a proveniência conferem, e os commits são anteriores às rodadas. O problema está na
inferência e na leitura.

**Conferido por mim, com código no repositório** (`exploratory/robustness.py`, que grava
`results/exploratory/robustness.json`):
- **A H1 não se sustenta.** Em 8×, o teste exato de troca de sinal, estratificado por pista, dá
  p = 0,0625 e p de Holm = 0,25. O bootstrap pré-registrado com B = 200 000 dá p de Holm = 0,055.
  - Só 6 rolamentos diferem, 5 deles a favor.
  - Com 6 rolamentos que respondem, a troca de sinal não poderia passar no Holm nem com os 6 a
    favor: o menor p seria 1/64.
  - O "apoio" pré-registrado (p de Holm = 0,04999, com B = 4000) é erro de Monte Carlo.
  - Pelo critério de refutação deste README, a H1 está refutada.
- **A H2 se sustenta só em 1/32.** A troca de sinal dá p = 0,0039 e p de Holm = 0,016, com os 8
  rolamentos que diferem a favor. Em 4× ela não passa: p de Holm de 0,20 pela troca de sinal e de
  0,07 pelo bootstrap.
- **A H3 tem resultado nulo** nos dois testes.

**Conferido por mim no sinal bruto** (`exploratory/inverter_lines.py`, checagem C4, medição 2 de
cada registro, 68 legíveis):
- **Não há raia de PWM na vibração.** Em nenhum registro aparece raia 10 dB acima da mediana local
  em 16 kHz ± 2k·f_e ou em 32 kHz − (2k+1)·f_e. A f_e é medida na corrente de fase, com mediana de
  101,6 Hz.
- **A faixa de 15–17 kHz é mais impulsiva que a de 11–15 kHz** em 68 de 68 registros, com curtose
  espectral mediana de 10,9 contra 5,8.
- A captura em taxa cheia fica, portanto, associada a conteúdo impulsivo perto de 16 kHz, mas a
  fonte não está estabelecida. A hipótese do inversor não é confirmada nem refutada.

**Do revisor, sem script no repositório:**
- **A régua não é neutra.** Ela vem de 12 calibrações em 6 rolamentos, todas medidas na banda
  capturada.
  - A variação das frequências cinemáticas entre fabricantes (1 a 2 %, PHME16) passa da tolerância
    de 0,025 ordem.
  - Há quase-acertos fora da janela: o SES no KA08 e o kurtograma sem picos no KI04.
  - A régua tende a favorecer o kurtograma, o que torna a H2 conservadora.
- **A réplica não testa o mecanismo do Ottawa.** O mecanismo pressupõe o seletor correto em taxa
  cheia, e isso só vale em 2 dos 23 rolamentos: KA01, com 63 %, e KI01, com 98 %.
  - No KI01 há uma parede, e não um vale: 98, 81, 75, 46 e 2 %.
  - Em KA01, KA04 e KA16, a decimação ajuda, o que sugere efeito de teto de busca (não isolado).
  - A grade não tem 2× nem 3×.
- **A banda de 11,0–21,5 kHz é larga demais para apontar o inversor.** A dobra para 0 Hz não é
  testável com fatores múltiplos de 4. A aritmética de amostrar num submúltiplo da portadora é
  conhecida: `blasko1998pwmsampling`.
- Correção: "de 2 a 730 picos por segundo" vem da checagem exploratória. No E24 são de 0 a 5553
  picos por segmento de 4 s, até cerca de 1390/s no KI08.

**O que o artigo pode afirmar:**
- No Paderborn, a decimação pura não produziu vale de 1/4 a 1/32.
- Remover os picos de uma amostra, antes ou depois de decimar, muda pouco e sem significância
  robusta.
- Em taxa cheia, o kurtograma escolhe bandas acima de 11 kHz em 88 % dos segmentos, sem acerto
  nelas. Isso é compatível com interferência impulsiva, e a fonte não foi verificada.
- O infograma SES sem pré-branqueamento é melhor em 1/32 (+3,1 pontos, p de Holm = 0,016), quase só
  na pista externa.
- O Paderborn não testa o mecanismo do Ottawa.

**O que não pode:**
- "H1 apoiada" ou "a captura por picos se reproduz";
- "o inversor captura" ou "a dobra para 0 Hz explica", como fato;
- "seletores robustos resolvem";
- "a decimação melhora o diagnóstico";
- "o mecanismo generaliza", ou que é refutado.

**Checagens sugeridas e não feitas:**
- C1: teto sem dobra em 1×.
- C2: fatores 5, 7 e 9.
- C3: *notch* em 15–17 kHz e 30,5–32 kHz, com controle.
- C5: régua independente do braço.

## E24b: teto sem dobra e notch (pré-registro, 2026-11-12)

Escrito e commitado antes de rodar, junto com a análise (`exploratory/teto_notch_analysis.py`). São
as checagens C1 e C3 da revisão adversarial.

A pergunta: no Paderborn, a decimação ajuda porque estreita a faixa de busca (efeito de teto) ou
por causa do que dobra? E o conteúdo perto de 16 kHz é o que captura o kurtograma em taxa cheia?

- **Braços** (`config/teto_notch.yaml`), todos sem remover os picos:
  - **C1:** `kurtograma_teto8k`, `teto4k`, `teto2k` e `teto1k`, mais os mesmos quatro para o
    infograma SES. Todos rodam em 1×, com o teto da busca em 8, 4, 2 e 1 kHz.
  - A partição da busca é idêntica à de 4×, 8×, 16× e 32×, porque só depende de
    [f_min, min(teto, fs/2)]. A diferença é o conteúdo dobrado e a taxa do envelope.
  - **C3:** `kurtograma_notch` zera 15–17 kHz e 30,5–32 kHz em taxa cheia, antes de decimar, e roda
    em 1× e 4×. `kurtograma_notch_controle` zera faixas de mesma largura em 12–14 kHz e
    26,5–28 kHz.
  - Os braços de referência sem teto e sem notch são lidos de `results/`. O código é o mesmo e é
    determinístico.
- **Critérios** (régua aninhada; troca de sinal exata por rolamento, estratificada por pista):
  - **C1a, dobra.** É a fração de segmentos na banda do piso (início ≤ 501 Hz e fim ≤ 1200 Hz) com
    teto de 8 kHz em 1×.
    - Com ≥ 20 %, a escolha do piso em 4× (23 %) é conteúdo de banda base, e a dobra fica
      refutada como causa.
    - Com ≤ 10 %, a dobra está implicada. Entre os dois, inconclusivo.
  - **C1b, teto.** É o kurtograma com teto contra o kurtograma sem teto, os dois em 1×, com Holm
    sobre os 4 tetos. O teto alivia a captura se algum passar (Holm p < 0,05, diferença > 0).
  - **C1c (descritivo).** É o teto em 1× contra a decimação de mesma partição. Diz se o ganho da
    decimação vem do teto ou da dobra.
  - **C3a, notch.** Entre os segmentos em que o kurtograma em 1× escolhe 11,0–21,5 kHz, é a fração
    que muda de banda com o notch.
    - Com ≥ 50 %, o conteúdo das faixas zeradas sustenta a escolha. Com < 20 %, não sustenta.
    - Entre os dois, inconclusivo.
  - **C3b, validade.** O notch de controle precisa mudar a banda em menos de 10 % dos segmentos em
    1×. Se mudar mais, a C3a não é interpretável.
- **Fora do escopo:** a C2 (fatores 5, 7 e 9) e a C5 (régua independente do braço).

