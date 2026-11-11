# E21 - MAFAULDA: o seletor também é capturado, e um teste de calibração prevê onde?

Aberto em 2026-11-11 pelo [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md). É a
generalização da Parte 1, e depende do mecanismo estabelecido no
[E20](../E20-fold-zone-rejection/README.md).

## Contexto

O E20 mostrou, no Ottawa (pista externa), que o vale de acerto de 3× a 6× vem de conteúdo
impulsivo fora da banda que dobra sobre a faixa de busca do kurtograma e o captura. Duas
mitigações resolvem: filtro anti-alias, ou teto da busca abaixo da faixa onde cai a dobra. Isso
vale para um banco só, com um rolamento por falha e eixo acionado por inversor.

Para a Parte 1 do artigo valer, o fenômeno precisa aparecer em outro conjunto. Também é preciso
haver uma forma de **prever, sem conhecer a falha**, quais taxas são perigosas. É o resíduo
defensável da Contribuição 2 do ADR 0005.

## Hipótese

**H1.** No MAFAULDA (pista externa, 372 sequências), sob decimação pura, o acerto de ordem não é
monótono no fator: há pelo menos um fator com acerto **pelo menos 15 pontos abaixo** de um fator
maior, e o filtro anti-alias ideal elimina esse vale.

**H2.** O teste de admissibilidade da taxa (TAT) prevê esses erros. O TAT é calculado só na
**primeira metade** de cada sequência, em taxa cheia. Ele marca um fator como inadmissível quando
a banda escolhida pelo kurtograma após a decimação tem mais da metade da energia vinda de
conteúdo dobrado.

## O que refutaria

- **H1 é refutada** se o acerto for monótono, a menos de 5 pontos, em todos os tetos. O fenômeno
  passa a ser do banco do Ottawa, e a Parte 1 fica restrita a ele (ADR 0006).
- **H2 é refutada** se, nas células onde o TAT diz "inadmissível", o acerto na segunda metade
  **não** for menor que nas admissíveis do mesmo fator (diferença ≤ 10 pontos, ou intervalo
  bootstrap por sequência cruzando zero).
- **H2 também é refutada** se a acurácia balanceada do TAT, prevendo erro contra acerto na
  segunda metade, ficar abaixo de 0,65.

## Baseline

O mesmo pipeline do E07 e do E20 (kurtograma restrito, envelope, espectro em ordens, estimativa
sem prior), com o filtro anti-alias ideal como teto do que a dobra custa. Para a previsão, o
baseline é o **acaso estratificado por fator**: prever erro com a taxa de erro marginal do fator.

## Verdade fundamental

- **Ordem de falha:** BPFO = 2,998, da geometria documentada e conferida na ingestão. Ela é
  calibrada por sequência em taxa cheia, na primeira metade, refinando ±12 % em torno do nominal.
- **Verdade de cada sequência:** a mediana das **outras** sequências do mesmo mancal
  (deixa-um-de-fora).
- **Rotação:** vem do tacômetro, com 1 pulso por revolução, histerese e bloqueio
  (`snq_data.mafaulda`).
- **Acerto:** |estimativa − verdade| < 0,06 ordem.

A taxa de 51,2 kHz é a inferida na ingestão. A análise de ordem é imune a ela.

## Procedimento

1. Para cada sequência de pista externa, usar o acelerômetro **radial do mancal com falha**.
   Dividir em calibração (1ª metade) e teste (2ª metade).
2. **Calibração**, em taxa cheia: calcular a ordem calibrada e, para cada fator e teto, o TAT.
   - Decimar a 1ª metade.
   - Rodar o kurtograma e obter a banda B.
   - Calcular a fração dobrada `1 − E_aaf(B)/E(B)`, onde `E_aaf` é a energia de B na versão
     filtrada antes de decimar.
   - O fator é inadmissível quando essa fração passa de 0,5.
3. **Teste:** na 2ª metade, para cada intervenção (`none`, `aaf_ideal`), fator e teto, rodar o
   pipeline e registrar acerto, banda e curtose.
4. **Análise:**
   - curva de acerto por fator e teto;
   - acerto por TAT (admissível ou inadmissível) dentro do fator;
   - acurácia balanceada do TAT, com bootstrap por sequência.

## Métricas

- acerto de ordem (`sem_prior`) por fator × teto × intervenção, com `n`;
- TAT: fração dobrada, previsão, e acurácia balanceada contra o erro observado;
- atribuição medição/prior: não se aplica (sem operador linear).

## Resultado

Rodado em 2026-11-11 no servidor (372 sequências, 22.320 linhas, DSP corrigido). **Inconclusivo
por construção: a métrica não é válida neste conjunto.**

**O que se mediu.**
- **Acerto em taxa cheia, sem intervenção:** 35 % no mancal *underhang* e 3 % no *overhang*. O
  filtro anti-alias ideal não muda o quadro, e nenhum fator mostra o vale de H1 (*underhang*: 30
  a 53 % ao longo dos fatores; *overhang*: 1 a 10 %).
- **TAT:** acurácia balanceada de 0,47 (IC 95 % de 0,46 a 0,49). As células inadmissíveis acertam
  10,8 pontos **a mais** que as admissíveis (IC de −18,5 a −4,2).

**Por que isso não testa H1 nem H2.** Neste rolamento, BPFO = 2,998 e BPFI = 5,002 são,
na prática, **ordens inteiras**: coincidem com o 3º e o 5º harmônicos do eixo.
- Em taxa cheia, 59 % das estimativas ficam a ±0,03 de uma ordem inteira. As mais frequentes
  são 3, 32, 2, 5 e 4.
- As ordens "calibradas" se espalham pela janela de ±12 % (2,64 a 3,35; só 11 % a ±0,01 de 3,000).

O "acerto" mede, portanto, se o pipeline achou o 3º harmônico do eixo, e não a falha. Como o
espectro é dominado pelos harmônicos inteiros do eixo, o próprio baseline não diagnostica.
Sem uma métrica válida, a pergunta sobre a subamostragem não se coloca aqui.

**O que surpreendeu.** O conjunto mais usado para multi-falha com tacômetro tem a ordem de BPFO
sobre um harmônico do eixo. Isso se soma à taxa real de 51,2 kHz, e não 50 kHz, achada na
ingestão. Os dois fatos valem uma nota no artigo.

**O que fica da ingestão:** a taxa de 51,2 kHz, a triagem do tacômetro e o loader. O MAFAULDA
pode servir a outra tarefa (detecção contra o normal, ou falha de esfera com BSF = 1,871), mas
**não** à estimação de ordem de BPFO.

**Consequência (ADR 0006, atualização de 2026-11-11):** a generalização da Parte 1 passa ao
**E23**, que injeta um interferente impulsivo controlado num conjunto com ordens de falha não
inteiras (CWRU 48k, BPFO 3,585) e, se possível, usa um segundo conjunto real com essa
propriedade (Paderborn).
