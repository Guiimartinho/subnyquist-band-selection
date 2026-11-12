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

## Resultado do piloto

Preenchido depois de rodar.
