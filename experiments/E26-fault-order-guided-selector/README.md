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

## Confirmação no Paderborn (pré-registro, 2026-11-13)

Escrito e commitado antes de rodar, junto com `run_paderborn.py`, `analyze_paderborn.py` e
`config/paderborn.yaml`. O desenho fica fixo daqui em diante. O piloto do Ottawa **não** faz parte
do teste, porque foi revisado duas vezes depois de ver os dados. O Paderborn não serviu para
ajustá-lo.

- **Dados.**
  - 23 rolamentos danificados (12 de pista externa e 11 de pista interna, os do E24) e os 6 sãos
    (K001–K006).
  - Os três ajustes de 1500 rpm, com as medições 2 a 20 de 255 936 amostras.
  - A medição 1 dos danificados serve à régua aninhada (`line_calibration` do E24).
- **Fatores:** 1, 4, 8, 16 e 32. Decidem 4, 8, 16 e 32; o 1 é descritivo.
- **Braços,** todos no fluxo já decimado:
  - `kurtograma`, o pipeline cego do E24;
  - `mediana_kurtograma`, a mediana de 3 seguida do kurtograma;
  - `guiado`, o seletor guiado sem limpeza;
  - `mediana_guiado`, a mediana de 3 seguida do seletor guiado.
- **Seletor guiado,** com parâmetros fixados a priori:
  - **Ordens:** as da geometria, BPFO = 3,0543 e BPFI = 4,9457, e só as harmônicas 1 e 2.
  - **Janela de ±2 %:** é a dispersão de 1 a 2 % entre fabricantes (PHME16, seção 3).
  - **Ordens inteiras excluídas, ±0,015.** No Paderborn, a BPFO e a BPFI ficam 0,054 da 3ª e da
    5ª harmônica do eixo, e a rotação é constante. A resolução em 4 s é de 0,01 ordem.
  - **Partição:** a do kurtograma, níveis 1 a 3, com 23 bandas até o Nyquist efetivo.
  - **Escore:** a proeminência média, isto é, o pico dividido pela mediana na própria janela.
- **Limiar:** o percentil 95 dos escores dos sãos, por fator e limpeza. Para avaliar cada são, o
  limiar sai dos outros 5.
- **Identificação.**
  - **Cego:** pista externa ou interna, se a ordem estimada ficar a ≤ 0,025 da verdadeira; senão,
    "nenhuma". A verdadeira é a régua aninhada do E24 para a própria pista, e a mediana de todas as
    calibrações para a outra pista e para os sãos.
  - **Guiado:** a falha de maior escore, se ele passar do limiar; senão, "nenhuma".
- **Métricas.**
  - Acurácia balanceada nos danificados: a média, sobre as pistas, do acerto por rolamento.
  - Especificidade nos sãos: a fração de "nenhuma".
- **Hipóteses.**
  - **H1:** `mediana_guiado` > `kurtograma`.
  - **H2:** `mediana_guiado` > `mediana_kurtograma`. É o ganho do seletor guiado além da limpeza.
  - **O teste:** troca de sinal por rolamento, estratificada por pista, unilateral, nas diferenças
    de acerto por rolamento. É exata até 2^18 combinações e Monte Carlo com 10^6 acima disso. A
    revisão do E24 mostrou que o bootstrap é anticonservador com tão poucos aglomerados.
  - Holm sobre os 4 fatores, com α = 0,05.
  - **Um fator passa** se o p de Holm ficar abaixo de 0,05, a diferença for positiva e a
    especificidade do `mediana_guiado` ali for de pelo menos 90 %.
- **O que refutaria.** A H1 cai se nenhum fator passar. O mesmo vale para a H2.
- **Limites declarados.**
  - O cego não tem limiar de alarme falso; a especificidade dele nos sãos é reportada.
  - A régua aninhada vem de poucas calibrações, medidas na banda que o kurtograma escolhe (revisão
    do E24). Ela só entra na identificação do cego.
  - A janela do guiado (±2 %) é mais larga que a tolerância do cego (0,025). O preço disso é o
    alarme falso, controlado pelo limiar dos sãos.

**Correção antes da rodada completa (2026-11-13).** A fumaça, com 6 rolamentos, 1 ajuste, 2
medições e os fatores 1 e 4, rodou, mas a análise quebrou ao procurar o fator 8, que ela não tinha.
A análise passou a testar só os fatores decisivos presentes nos dados. Nenhum critério mudou, e os
números da fumaça não foram olhados.

## Resultado da confirmação no Paderborn (2026-11-13; sob revisão adversarial)

- **Rodada:** `results-paderborn/`, de 00:25 a 05:21 UTC, 16 470 linhas.
  - Os hashes do config, do `run_paderborn.py`, do `guided.py`, do README e do `run.py` do E24
    batem com o commit `57a6a89`. Não houve recorte de fumaça.
  - 6 medições ficaram de fora por arquivo curto, rotação fora de 4 s ou `.mat` ilegível. A
    lista está no manifest.
- **Veredito pré-registrado** (`analysis.json`): a H1 e a H2 passam nos quatro fatores decisivos.
- **Acurácia balanceada (%), régua aninhada no cego e limiar dos sãos no guiado:**

  | braço | 1× | 4× | 8× | 16× | 32× |
  |---|---|---|---|---|---|
  | `kurtograma` | 7,7 | 16,1 | 18,6 | 19,2 | 13,0 |
  | `mediana_kurtograma` | 15,6 | 18,8 | 27,8 | 20,8 | 17,4 |
  | `guiado` | 36,2 | 52,2 | 53,0 | 45,4 | 32,2 |
  | `mediana_guiado` | 38,3 | 49,0 | 50,9 | 45,2 | 25,2 |

  - A especificidade nos sãos é de 93,9 a 95,0 % nos braços guiados e de 99,4 a 100 % nos cegos.
- **Testes** (troca de sinal por rolamento, estratificada por pista, Holm sobre 4, 4× a 32×):
  - **H1** (`mediana_guiado` contra `kurtograma`): +32,9, +32,3, +26,0 e +12,2 pontos, com 16 de
    17, 17 de 18, 18 de 19 e 21 de 22 rolamentos discordantes a favor. O p de Holm fica abaixo de
    0,001 nos quatro.
  - **H2** (`mediana_guiado` contra `mediana_kurtograma`): +30,2, +23,1, +24,3 e +7,8 pontos, com
    p de Holm de 0,0004, 0,0014, 0,0006 e 0,031.
- **O que o veredito não diz, e que a revisão precisa atacar:**
  - **A mediana não contribui no Paderborn.** O `guiado`, sem limpeza, empata com o
    `mediana_guiado` ou fica acima dele. Em 32× fica 7 pontos acima.
  - **O ganho já existe em 1×,** com 36,2 % contra 7,7 %. Então não é específico da decimação:
    no Paderborn o kurtograma já é capturado na taxa cheia (E24).
  - **Os braços mudam duas coisas ao mesmo tempo:** a banda e a regra de identificação.
    - O guiado escolhe entre as duas ordens da geometria, com limiar calibrado nos sãos.
    - O cego estima a ordem sem prior e precisa acertar a ±0,025.
    - O ganho pode vir da regra de identificação, não da banda. Isso foi declarado no
      pré-registro como limite, mas não foi separado.

