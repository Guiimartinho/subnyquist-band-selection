# E22 - Baselines publicados na mesma taxa de amostras/s

Aberto em 2026-11-11 pelo [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md). Fixa o
que a máscara aprendida (E15) precisa vencer. Toda comparação é feita na **mesma taxa média de
amostras convertidas por segundo**. Todos os braços são métodos publicados reimplementados, ou as
mitigações estabelecidas pelo E20. Nenhum espantalho.

## Pergunta

Dado um orçamento de amostras/s abaixo da taxa de aquisição, qual método publicado entrega o
melhor acerto de ordem de falha? A amostragem aleatória com CS (o concorrente direto) supera a
uniforme bem escolhida?

## Cenários de front-end

Eles são declarados porque mudam qual método é ótimo.

- **A. Sem filtro analógico sintonizável.** Por exemplo, um acelerômetro MEMS ligado direto ao
  ADC. É onde a dobra do E20 acontece, e o cenário principal do artigo.
- **B. Passa-faixa analógico fixo na ressonância**, escolhido numa calibração em taxa cheia
  (`lu2018undersampled`). É o ótimo clássico quando a banda é conhecida de antemão.

## Braços

| braço | front-end | método | referência |
|---|---|---|---|
| `uniforme` | A | decimação na taxa do orçamento, kurtograma até o Nyquist efetivo | pipeline do E07 (`antoni2007kurtogram`) |
| `uniforme_teto` | A | idem, com teto da busca escolhido pelo TAT na calibração | mitigação do E20 |
| `taxa_tat` | A | o **menor** fator admissível pelo TAT dentro do orçamento, ou seja, a maior taxa admissível | E21 |
| `oraculo_uniforme` | A | o melhor fator uniforme dentro do orçamento, escolhido **com o rótulo de teste** | limite superior; não é método |
| `aleatorio_cs` | A | instantes aleatórios + OMP em base de ordem + envelope | `sensors2025csvibration` (braço do E13) |
| `multitaxa` | A | duas correntes uniformes coprimas somando o orçamento, com consistência de ordem entre elas | `zhao2018coprime`, `sun2013multirate` |
| `passa_faixa_lu` | B | passa-faixa ideal na banda da calibração + amostragem passa-faixa na menor taxa válida do orçamento | `lu2018undersampled`, `vaughan1991` |
| `zhao_asns` | A | amostragem sub-Nyquist adaptativa | `zhao2026adaptive`, **só se o PDF permitir reproduzir** |

Seletores robustos (`antoni2016infogram`, `smith2016emi`, `mauricio2020iesfogram`) entram como
variante do braço `uniforme` quando implementados.

## O que cada método pode usar

Declarado porque muda o que é justo comparar:
- `taxa_tat`, `uniforme_teto` e `passa_faixa_lu` usam **uma captura curta em taxa cheia da
  própria máquina** (calibração).
- A máscara do E15 usa **registros de treino** de outras máquinas e condições, e nenhuma captura
  da máquina de teste.
- O artigo reporta os dois regimes, sem misturar.

## Hipóteses

- **H1.** Em todos os orçamentos, `aleatorio_cs` não supera `uniforme_teto`. Estimar sem
  reconstruir ganha de reconstruir, como no E13 a 64×.
- **H2.** `taxa_tat` fica a no máximo 5 pontos do `oraculo_uniforme` em todos os orçamentos.

## O que refutaria

- **H1 é refutada** se `aleatorio_cs` superar `uniforme_teto` em algum orçamento, com intervalo
  bootstrap por registro acima de zero. Isso é resultado publicável contra a rota atual, e o
  artigo o reporta como tal.
- **H2 é refutada** se `taxa_tat` ficar mais de 5 pontos abaixo do oráculo em algum orçamento.

## Protocolo

- **Conjuntos:** Ottawa (pistas interna e externa) e MAFAULDA (pista externa, fora da triagem do
  tacômetro).
- **Orçamentos**, como fração da taxa de aquisição: 1/4, 1/8, 1/16 e 1/32 no Ottawa; 1/4, 1/8 e
  1/16 no MAFAULDA.
- **Divisão:** calibração (1ª metade) e teste (2ª metade) dentro de cada registro, para os braços
  que calibram. Divisões repetidas **por registro** (E05) para tudo o que é escolhido em treino.
- **Métricas:** acerto de ordem com o `n`, intervalo bootstrap por registro, e McNemar pareado
  entre braços na mesma taxa.

## Estado

**Fase 1 implementada** (`run.py`). Braços: `uniforme`, `uniforme_teto`, `taxa_tat`,
`oraculo_uniforme` (na análise), `aleatorio_cs` e `passa_faixa_lu`. Ficam para a fase 2 a
`multitaxa`, que precisa de uma regra de fusão própria, e o `zhao_asns`, que depende do PDF. Os
testes de auxiliares estão em `tests/`.

**Teste de fumaça** (2026-11-11; `config/smoke.yaml`: Ottawa, 2 registros de pista interna,
orçamentos 1/16 e 1/32):
- O pipeline roda de ponta a ponta nos seis braços.
- `aleatorio_cs` erra 8 de 8 células, coerente com o E13.
- `passa_faixa_lu` só ficou fiel depois de **escolher, entre as bandas do kurtograma de
  calibração, a mais impulsiva que o orçamento consegue amostrar**. Na primeira versão, que usava
  a melhor banda sem restrição, ela era larga demais para qualquer taxa válida. Depois da
  correção, acerta 8 de 8.
- `uniforme_teto` com teto de 1/4 perdeu a banda informativa num registro e caiu no 2º
  harmônico. É comportamento da regra, e vai reportado como tal.

A fumaça não é resultado; a rodada completa usa `config/base.yaml` no servidor.

## Resultado (2026-11-11, fase 1, Ottawa)

Rodada completa no servidor: 24 registros (12 da pista interna e 12 da externa), segmento 0 para
calibração e segmentos 1 e 2 para teste, ou 48 segmentos por orçamento. Análise em `analyze.py`,
gravada em `results/analysis.json`, com bootstrap por registro estratificado por grupo (4000
reamostras) e McNemar exato pareado por segmento.

O **MAFAULDA saiu desta rodada**. O E21 mostrou que a ordem de BPFO dele coincide com o 3º
harmônico do eixo, e por isso o acerto lá não mede a falha.

Acerto (%) nos dois grupos somados:

| braço | 1/4 | 1/8 | 1/16 | 1/32 |
|---|---|---|---|---|
| `uniforme` (fator = orçamento) | 56,2 | 89,6 | 85,4 | 87,5 |
| `uniforme_teto` | **87,5** | 89,6 | 54,2 | 29,2 |
| `taxa_tat` | 79,2 | 89,6 | 85,4 | 87,5 |
| `passa_faixa_lu` (Lu 2018) | 79,2 | 79,2 | 79,2 | 72,9 |
| `aleatorio_cs` (Kato & Otaka) | 2,9 | 4,2 | 6,2 | 4,2 |
| `oraculo_uniforme` (escolhido no teste) | 97,9 | 97,9 | 97,9 | 97,9 |
| melhor uniforme escolhida em outros registros (*post hoc*) | 95,8 | 95,8 | 93,8 | 95,8 |

**H1, números de pé mas comparação inválida (revisão de 2026-11-11).** O aleatório com CS fica
de 25 a 85 pontos abaixo do `uniforme_teto` (IC 95 % abaixo de zero) e perde para o `uniforme` em
todos os segmentos discordantes. O braço, porém, é um **espantalho por construção**:
- O dicionário cobre as ordens 0,5 a 24 (`config/base.yaml`, seção `cs`), até cerca de 520 Hz nos
  segmentos do Ottawa. Fica abaixo do piso de 500 Hz do kurtograma e longe das ressonâncias que
  carregam a falha (2–4,6 kHz e 16,8–25 kHz). O envelope do sinal reconstruído não pode conter a
  modulação da falha.
- O sistema é sobredeterminado em todos os orçamentos: cerca de 2.316 colunas, com razão
  linhas/colunas de 54 a 6,7. Não é *compressed sensing*.
- Kato & Otaka (`sensors2025csvibration`) reconstroem 0–2.500 Hz no caso de rolamento.

**H1 sai do artigo** até existir um braço fiel: dicionário que cubra a ressonância, regime
subdeterminado e operador rápido. O mesmo teto vale para o braço de CS do E13 (ver o relatório, §5).

**H2, refutada nos quatro orçamentos.** O `taxa_tat` fica abaixo do oráculo em −18,7
[−29,2; −8,3], −8,3 [−14,6; −2,1], −12,5 [−18,8; −6,2] e −10,4 [−18,8; −4,2] pontos.
- **O que a regra faz bem:** resgata o vale de 1/4 contra o `uniforme` (11 × 0, p = 0,001).
- **O que ela não faz:** achar a melhor taxa. Ela escolhe a maior taxa admissível, e a melhor
  uniforme da pista externa é o fator 46 (4,35 kSPS), abaixo de qualquer orçamento.

**O "fator 46 generaliza entre registros" foi retirado** (análise *post hoc*; revisão de
2026-11-11).
- **Vazamento pela verdade.** A verdade de cada registro, a mediana deixa-um-de-fora das
  calibrações do grupo, usa também os registros da dobra de teste. Com a verdade **aninhada**
  (só registros de treino), a melhor uniforme escolhida em outros registros dá 87,5 / 87,5 / 89,6
  / 91,7 % nas duas pistas somadas, e **75,0 / 75,0 / 79,2 / 83,3 %** na pista externa, em vez de
  95,8 / 95,8 / 93,8 / 95,8 %. Os dois conjuntos de números foram recalculados de forma independente
  (`check_nested_cv.py`, notebook de 2026-11-11).
- **O 46 é o topo de um platô, não um ótimo.** Pareado, ele não se distingue dos vizinhos
  (contra 45: 3 × 0, p = 0,25; contra 47: 4 × 0, p = 0,125). Pelas contas do revisor, ainda não
  refeitas, num nulo por permutação do fator dentro do segmento P(max ≥ 23) = 0,019, e na
  validação por cenário (LOSO) com verdade aninhada o 46 volta em 2 de 4 dobras. O que generaliza é um platô de fatores ≥ 7, fora do vale de 3× a 6×.
- **Os 12 registros de cada grupo vêm de um rolamento só.** "Entre registros" quer dizer entre
  repetições do mesmo ensaio.
- **Contra a uniforme escolhida com verdade aninhada,** o `taxa_tat` fica −12,5 / −4,2 / −8,4 /
  −4,2 pontos abaixo (−16,6 / −6,2 / −8,4 / −4,2 por cenário). Com bootstrap por cenário, todos os
  IC incluem 0. A única diferença robusta é a da pista externa em 1/4.

**Os outros braços.**
- `uniforme_teto` é o melhor braço em 1/4, mas desaba em 1/16 e 1/32 (0 × 15 e 2 × 30 contra o
  `uniforme`). Reduzir o teto corta a banda informativa da pista interna, que fica em 16,8–25 kHz.
- `passa_faixa_lu` fica estável em torno de 79 % e perde do `uniforme` em 1/32 (0 × 7, p = 0,016).

**Consequência para o E15 (corrigida em 2026-11-11).** A primeira versão desta seção mandava o
E15 mirar 96–128×. Isso foi retirado. Com a verdade aninhada, a melhor uniforme escolhida no treino
dá 75–83 % na pista externa de 1/4 a 1/32, e há espaço para ganhar ali. Já em 96–128× o baseline
uniforme cai por uma parede do pipeline (bandas mais estreitas que a frequência de falha; ver o
E07, extensão), e um ganho ali seria contra um espantalho.
- **Orçamentos:** 1/4 a 1/32.
- **Baseline que decide:** a melhor uniforme escolhida nos registros de treino, com verdade
  aninhada.

A correção foi comunicada ao E15 no mesmo dia.

## Re-execução com a régua corrigida (2026-11-11, tarde)

Rodada às 13:57 UTC, com a verdade pelo centro do platô, a régua local gravada ao lado e o braço
`despike` pré-registrado. A análise (`analyze.py --ruler global|local`) está em
`results/analysis.json` e `results/analysis-local.json`, com procedência gravada.
- **Régua global:** verdade deixa-um-de-fora, com validação cruzada aninhada.
- **Régua local:** a raia em taxa cheia do próprio segmento, só para pontuar. A escolha fora do teste
  não tem vazamento por construção.

Acerto (%) nas duas pistas somadas:

| braço | global 1/4 · 1/8 · 1/16 · 1/32 | local 1/4 · 1/8 · 1/16 · 1/32 |
|---|---|---|
| `uniforme` | 56,2 · 87,5 · 83,3 · 87,5 | 57,4 · 91,5 · 87,2 · 91,5 |
| **`despike`** | **91,7 · 93,8 · 89,6 · 89,6** | **97,9 · 97,9 · 93,6 · 93,6** |
| `uniforme_teto` | 85,4 · 87,5 · 54,2 · 29,2 | 89,4 · 91,5 · 55,3 · 29,8 |
| `taxa_tat` | 77,1 · 87,5 · 83,3 · 87,5 | 78,7 · 89,4 · 87,2 · 91,5 |
| `passa_faixa_lu` | 79,2 · 79,2 · 79,2 · 77,1 | 83,0 · 83,0 · 83,0 · 80,9 |
| `aleatorio_cs` (espantalho, ver acima) | 2,9 · 4,2 · 6,2 · 4,2 | 2,9 · 4,3 · 6,4 · 4,3 |
| melhor uniforme escolhida em outros registros | 89,6 · 89,6 · 87,5 · 89,6 | **97,9 · 97,9 · 95,7 · 97,9** |
| oráculo (escolhido no teste) | 95,8 nos quatro | 97,9 nos quatro |

**Leitura.**
- **Remover os picos de uma amostra resolve o vale de 1/4**, como no E20: 56 → 92 % na régua global,
  57 → 98 % na local.
- **Pela régua local, a melhor uniforme escolhida fora do teste fica a 0–2 pontos do oráculo.** Na
  pista externa o fator escolhido é o mesmo nas 3 dobras: 12 em 1/4 e 1/8, 46 em 1/32. A
  retirada anterior do "fator 46 generaliza" vinha da régua: a verdade aninhada também carrega a
  deriva da raia com a rotação.
- **O H2 continua refutado,** agora nas duas réguas. O `taxa_tat` fica −19,2 [−29,2; −4,3] pontos
  abaixo da uniforme escolhida fora do teste em 1/4 (régua local), e os outros intervalos incluem 0.
- **Consequência para o E15.** Com a régua local, o baseline que decide, a melhor uniforme
  escolhida no treino, está em 96–98 %. Sobram de 2 a 4 pontos para a máscara aprendida no Ottawa.
  Um resultado negativo é plausível e será reportado como tal.
