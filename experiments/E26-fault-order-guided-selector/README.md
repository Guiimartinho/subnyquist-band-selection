# E26 - Seleção de banda guiada pelas ordens de falha, no fluxo já decimado

Aberto em 2026-11-12, depois do piloto negativo do E25.

## Por que este experimento

O E20, o E23 e o E25 mostram que a captura sob decimação pura é um problema de **escolha da
banda**. O seletor cego (curtose) vai para onde a energia impulsiva dobrada cai. No piloto do E25,
uma banda fixa acertou 92 % com o interferente, contra 23–38 % do kurtograma cego, mas essa banda
foi calibrada com a falha presente.

Aqui o seletor escolhe a banda pelo que interessa ao diagnóstico: a raia nas ordens de falha que a
geometria do rolamento dá (BPFO e BPFI). Um interferente que não é cíclico nessas ordens não ganha
a escolha. Tudo é feito no fluxo já decimado, sem taxa cheia e sem comissionamento. A ideia é a dos
seletores dirigidos da literatura (`mauricio2020iesfogram`). O que se testa aqui é se ela evita a
captura sob decimação pura.

## O cuidado que muda a métrica

Com o prior das ordens, "estimar a ordem" perde o sentido. Uma busca a ±2 % da ordem nominal cai
dentro da tolerância por construção (no Ottawa, ±0,07 contra a tolerância de 0,06), e isso seria o
prior acertando, não a medição. Por isso a métrica é de **identificação**:
- **O que o método decide:** qual falha está presente (pista externa, pista interna ou nenhuma),
  pela proeminência da raia em cada ordem candidata.
- **Limiar:** o percentil 95 do escore nos registros **sãos**, por fator. Dá 5 % de alarme falso.
  É a separação entre medição e prior: sem falha, o método tem de dizer "nenhuma".
- **Comparação:** o acerto do kurtograma cego do E20 nos mesmos segmentos também é uma
  identificação. Uma estimativa dentro da tolerância da falha verdadeira identifica a falha certa.

## Piloto exploratório (2026-11-12, NÃO pré-registrado)

O piloto decide se vale um pré-registro. Os números dele não vão ao artigo como confirmatórios.

- **Dados.** Ottawa, com os 36 registros: 12 sãos, 12 de pista interna e 12 de pista externa. São
  3 segmentos de 2,5 s cada, os mesmos do E20.
- **Fatores e busca.** Fatores 1, 2, 3, 4, 5, 6 e 8. Busca de 500 Hz até min(25 kHz, f_s/2M), com
  a partição do kurtograma até o nível 3 (24 bandas).
- **Escore.** Para cada banda e falha candidata c, é a média, nas harmônicas k = 1, 2, 3, da
  proeminência da raia: o pico a ±2 % de k·o_c dividido pela mediana a ±7 %. As ordens o_c vêm da
  geometria (BPFO = 3,573 e BPFI = 5,427). O escore da falha é o máximo sobre as bandas.
- **Decisão.** A falha de maior escore, se ele passar do limiar dos sãos naquele fator. Senão,
  "nenhuma".
- **Métricas.**
  - acerto de identificação por pista e balanceado;
  - alarme falso, com o limiar calibrado nos próprios sãos, ou seja, dentro da amostra;
  - o kurtograma cego do E20 (`results/metrics.csv`, braço `none`, teto de 25 kHz, régua
    aninhada) nos mesmos segmentos.

## Resultado do piloto, primeira rodada (janela de ±2 %, `results-pilot/`)

| fator | guiado: externa | guiado: interna | guiado: sãos corretos | cego (E20): externa | cego (E20): interna |
|---|---|---|---|---|---|
| 1× | 0 % | 100 % | 94 % | 75 % | 97 % |
| 4× | 0 % | 100 % | 94 % | 17 % | 97 % |
| 8× | 0 % | 100 % | 94 % | 69 % | 100 % |

**Por que a pista externa fica em 0 %, até em 1×.**
- A BPFO real do Ottawa fica 4 % abaixo da nominal: a mediana da régua aninhada é 3,432, contra
  3,573.
- Na régua local, ela vai de 3,33 a 3,79, porque a raia deriva com a rotação.
- A janela de ±2 % nunca a alcança, e o escore da pista externa fica no nível dos sãos (mediana de
  3,2, contra 3,4).
- Na pista interna, a BPFI real fica a 0,2 % da nominal, e o acerto é de 100 %.
- **Foi erro do desenho, e não um efeito da decimação.** O E07 e o E20 calibram o Ottawa em ±12 %
  justamente por essa incerteza.

## Revisão do piloto, segunda rodada (2026-11-12, depois da primeira; `results-pilot-w12/`)

- **Janela:** ±12 % em torno de cada ordem candidata, com o pico e a mediana na mesma janela. É o
  span de calibração do Ottawa no E07 e no E20.
- **Harmônicas:** só 1 e 2. Com a janela larga, 3·BPFO sobreporia 2·BPFI.
- **O resto não muda,** inclusive o limiar nos sãos.
- A revisão foi feita depois de ver a primeira rodada. Ela corrige um erro de desenho conhecido de
  antemão, mas é um piloto exploratório e assim deve ser lido.

## Resultado da segunda rodada (janela de ±12 %, `results-pilot-w12/`)

| fator | guiado: externa | guiado: interna | limiar (sãos) | cego (E20): externa |
|---|---|---|---|---|
| 1× | 100 % | 100 % | 7,0 | 75 % |
| 2× | 72 % | 100 % | 11,2 | 56 % |
| 3× | **100 %** | 100 % | 6,2 | 11 % |
| 4× | 11 % | 100 % | **14,3** | 17 % |
| 5× | 6 % | 100 % | **14,9** | 6 % |
| 6× | **97 %** | 100 % | 8,7 | 31 % |
| 8× | 97 % | 100 % | 8,4 | 69 % |

Os sãos ficam corretos em 94 % dos segmentos, em todos os fatores.

**Por que 4× e 5× falham.**
- Nos registros **sãos**, aparecem raias fortes dentro da janela da BPFI, em ordens de 5,63 a 5,84,
  com a 2ª harmônica em 11,26–11,67.
- Elas surgem nas bandas altas (12,7–25 kHz), justamente onde os picos dobrados caem, e não em
  ordens inteiras. Mudam de registro para registro.
- Isso é compatível com um trem de picos de taxa fixa em Hz, que, sob rotação variável, aparece em
  ordens diferentes.
- O limiar calibrado nos sãos sobe, e a pista externa deixa de passar.
- É o modo de falha esperado do seletor guiado: um interferente cíclico perto de uma ordem
  candidata.

## Terceira variante (2026-11-12): mediana de 3 no fluxo decimado antes do seletor guiado

É a mesma janela de ±12 %, com a correção do E20 aplicada antes. Testa se os picos produzem as
raias falsas dos sãos (`pilot.py --clean median3`, `results-pilot-w12-median3/`). Também é
exploratória e foi decidida depois de ver a segunda rodada.

### Resultado da terceira variante (`results-pilot-w12-median3/`)

| fator | mediana + guiado: externa | interna | limiar (sãos) | sem mediana: externa | cego (E20): externa |
|---|---|---|---|---|---|
| 1× | 19 % | 100 % | **13,9** | 100 % | 75 % |
| 2× | **89 %** | 100 % | 10,4 | 72 % | 56 % |
| 3× | **100 %** | 100 % | 5,9 | 100 % | 11 % |
| 4× | **97 %** | 100 % | 8,7 | 11 % | 17 % |
| 5× | **100 %** | 100 % | 7,1 | 6 % | 6 % |
| 6× | **100 %** | 100 % | 5,8 | 97 % | 31 % |
| 8× | **100 %** | 100 % | 7,1 | 97 % | 69 % |

Os sãos ficam corretos em 94 % dos segmentos, em todos os fatores. É dentro da amostra.

**Leitura** (exploratória):
- **De 2× a 8×, a combinação identifica a pista externa em 89–100 % dos segmentos,** contra 6–69 %
  do kurtograma cego, com a pista interna em 100 %.
  - Em 4× e 5×, o limiar dos sãos volta a 7–9. Os picos dobrados eram a fonte das raias falsas.
  - Como referência indireta, a mediana seguida do kurtograma (E20, `results-post-median`) acerta
    a **ordem** em 75–81 % de 4× a 8×. A métrica aqui é de identificação, então a comparação não é
    exata.
- **Em 1× a combinação cai para 19 %:** o limiar dos sãos sobe para 13,9, sem explicação por ora.
  Não é o regime sub-Nyquist, mas precisa ser entendido.

**Limites:**
- O desenho foi revisado duas vezes depois de ver os dados.
- O limiar é calibrado nos mesmos sãos em que se mede o alarme falso.
- O kurtograma cego não tem limiar de alarme falso nesta comparação. O controle de sãos do E07
  (`healthy_control.csv`) é a referência para isso.

**Antes do artigo:** é preciso uma rodada **pré-registrada** em dados que não serviram para ajustar
o desenho (o Paderborn), com:
- janela de busca fixada a priori pela incerteza de ordem do conjunto;
- limiar em rolamentos sãos com validação cruzada;
- os mesmos baselines: kurtograma cego, mediana + kurtograma e teto baixo.

