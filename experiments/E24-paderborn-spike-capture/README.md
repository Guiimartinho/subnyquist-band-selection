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

## Resultado

Preenchido depois de rodar.
