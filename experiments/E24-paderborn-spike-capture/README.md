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
    e quase todos os outros ficam acima de 16 kHz.
  - Sob decimação, a escolha vai para a banda mais baixa da busca, 0,5–1,1 kHz em 4× e 8×.
- **Aritmética da dobra (hipótese, não testada).** A 64 kS/s, com os fatores 4, 8, 16 e 32, todo
  múltiplo de 16 kHz cai exatamente em 0 Hz. As bandas laterais do chaveamento cairiam então no
  fundo da faixa de busca.
- **Visibilidade da raia.** Só 7 dos 23 rolamentos têm raia visível em algum segmento: KA01, KA06,
  KA16, KI01, KI16, KI17 e KI18.
  - A visibilidade é medida na banda que o kurtograma escolhe em taxa cheia, e essa banda está
    capturada (`line_calibration`).
  - Nos outros 16 rolamentos, o kurtograma não acerta nenhum segmento em taxa cheia pela régua
    aninhada. O infograma SES acerta 19 % deles.
  - Por isso, "sem raia visível" é em parte uma falha do seletor, e não só do sinal.

**Leitura:**
- O vale do Ottawa não se reproduz no Paderborn. Ali o seletor já é capturado em taxa cheia pela
  faixa do inversor, que é o cenário de `smith2016emi`.
- A remoção de picos de uma amostra muda pouco. A H1 passa no limite, com efeito pequeno, e não
  sustenta que "a captura por picos se reproduz fora do Ottawa".
- O infograma SES ajuda no Paderborn, ao contrário do Ottawa.
- **Antes de entrar no artigo:** revisão adversarial.
