# E23 - Interferente impulsivo injetado no CWRU 48k: a captura é previsível e evitável?

Aberto em 2026-11-11 pela atualização do [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md)
depois do E21. É a generalização da Parte 1 por intervenção controlada.

## Por que este experimento

- **E20:** no Ottawa, a dobra de conteúdo impulsivo fora da banda (provavelmente interferência
  eletromagnética do inversor) captura o kurtograma, de 3× a 6×.
- **E21:** o MAFAULDA não serve para testar a generalização, porque nele BPFO coincide com o 3º
  harmônico do eixo.
- **E02:** o CWRU 48k, cujas ordens de falha não são inteiras (BPFO 3,585, BPFI 5,415), cai de
  forma monótona sob subamostragem, porque não tem interferente de banda alta.

Injetar um interferente controlado no CWRU separa três perguntas do artigo:
1. dado um interferente impulsivo, a captura acontece **onde a aritmética da dobra prevê**, e só
   ali?
2. é a **impulsividade** que captura, e não a potência?
3. uma calibração em taxa cheia (TAT) ou as mitigações a evitam?

## Desenho

- **Sinal:** os 13 registros de falha de 48 kHz do CWRU (pista DE, catálogo `smith2015cwru`; 12
  BPFO e 1 BPFI), com rotação constante pelo rpm do registro. A 1ª metade de cada registro é
  calibração e a 2ª é teste. A verdade é a frequência de falha pela geometria do 6205, com
  tolerância de 2 Hz, como no E02.
- **Pipeline:** o do E02 em hertz (kurtograma + envelope + estimativa sem prior), com a faixa de
  busca do kurtograma restrita a [500 Hz, teto].
- **Interferente:** rajadas `exp(-t/τ)` numa portadora fc, com τ = 0,5 ms, a 97,7 Hz. Essa taxa
  não é ordem do eixo nem da falha.
  - fc ∈ {13, 16, 19, 22} kHz, todas acima do teto de 10 kHz;
  - potência relativa ∈ {−10, 0, +10, +20} dB, medida contra a potência da banda que o
    kurtograma escolhe em taxa cheia, na calibração sem injeção;
  - a realização das rajadas é a mesma em todas as potências (só a escala muda) e é fixa por
    registro e portadora.
- **Controle de impulsividade:** o **substituto de fase aleatória** da mesma realização. Tem o
  mesmo espectro de amplitude, bin a bin, e curtose média próxima de 3. A única diferença entre
  os dois braços é a impulsividade.
- **Fatores:** 1 a 8, 10 e 12.
- **Tetos cruzados no braço uniforme:** 10, 5 e 2,5 kHz, limitados a fs_eff/2. No mesmo fator e
  com o mesmo sinal, a imagem do interferente cai dentro da busca com um teto e acima dele com
  outro. É o contraste que testa a especificidade da previsão.
- **Previsão aritmética:** a fração da potência do interferente cuja imagem, sob `x[::M]`, cai na
  faixa de busca, calculada do espectro da realização injetada. Uma fração de pelo menos 0,5 é
  captura prevista; abaixo de 0,05, sem captura prevista; entre os dois, borda, que fica fora do
  critério binário. A conta inclui a cauda lorentziana das rajadas, que vaza para a busca mesmo
  sem dobra; o teste compara com a referência analítica.
- **Indicador direto de captura:** a estimativa cai sobre 97,7 Hz ou um harmônico até o 3º, isto
  é, o seletor passou a olhar o interferente.
- **Braços:** `uniforme` (três tetos), `aaf_ideal`, `uniforme_teto` (teto pelo TAT) e `taxa_tat`
  (fator pelo TAT). O oráculo uniforme sai na análise.

### Revisões do desenho antes da rodada completa

Nenhuma destas mudanças usou o resultado da rodada completa:

1. **Controle.** O ruído gaussiano retangular de mesma potência virou o substituto de fase. O
   ruído retangular não tinha o mesmo espectro das rajadas, que têm espectro lorentziano.
2. **Previsão.** Em vez da imagem da portadora, a fração dobrada. A cauda lorentziana torna a
   portadora uma previsão ruim: com a imagem em 0 Hz, cerca de 60 % da potência ainda cai na busca.
3. **Tetos cruzados e τ.** A tabela de previsão, calculada a priori só com a aritmética, tinha 30 a
   35 de 40 células com captura prevista e quase nenhuma célula dobrada sem captura. Sem isso não
   haveria teste de especificidade.
4. **Potência estendida a +20 dB.** Na fumaça (2 registros, 16 kHz, 0 dB), as células previstas
   não capturaram: o kurtograma passou a incluir a imagem, mas a linha de BPFO continuou
   dominante. As falhas do CWRU 48k são fortes.

### Braço de picos e braço de remoção (acrescentados em 2026-11-11, antes de qualquer resultado)

**Motivação.** A revisão adversarial do E20 mostrou que o captor real do Ottawa é um trem de picos
isolados de uma amostra (1.300–2.500/s, desvio negativo em 96 % dos casos, a 16–18 MAD), não uma
rajada de banda limitada. As rajadas do desenho original testam outro interferente.

**Desenho** (`config/spikes.yaml`, rodada separada em `results-spikes/`):
- **Interferente:** picos isolados de uma amostra, por processo de Poisson, com sinal negativo.
  - taxas ∈ {500, 1.300, 2.500}/s;
  - amplitudes ∈ {8, 16, 32} × o MAD do desvio de cada amostra em relação à média dos vizinhos,
    medido na calibração.
- **Braço `despike`:** o critério da revisão, aplicado em taxa cheia antes de decimar. Uma amostra
  que desvia mais de 10 MAD, com os dois vizinhos desviando em sentido oposto, vira a média deles.
  - **Limitação:** dois picos a duas amostras um do outro escapam, cerca de 3 % a 1.300/s e 48 kHz.

**H-picos (pré-registrada).** Com picos de parâmetros do Ottawa (1.300/s, 16 MAD):
- o braço uniforme, no teto de referência, cai pelo menos 20 pontos contra o mesmo fator sem
  injeção em pelo menos um fator ≥ 2;
- nos braços `aaf_ideal` e `despike`, a queda é de no máximo 5 pontos em todos os fatores.

A dose-resposta em amplitude e taxa é descritiva.

**O que a refutaria.** Nenhum fator ≥ 2 cai 20 pontos em nenhuma amplitude. Nesse caso os picos,
sozinhos, não reproduzem a captura no CWRU, e o fenômeno do Ottawa depende de outra propriedade
daquele conjunto. Resultado entre os dois limiares é inconclusivo.

## Hipóteses e critérios pré-registrados

Os critérios estão em `config/base.yaml`, na seção `analysis`, e em `analyze.py`.

- **Célula avaliável:** um par (portadora, fator, teto) do braço uniforme em que, sem injeção, o
  acerto é de pelo menos 50 %. A **queda** é o acerto sem injeção menos o acerto com injeção, em
  pontos, pareado por registro.
- **H1 (a aritmética da dobra prevê a captura).**
  - (a) Sensibilidade: na potência máxima, pelo menos 80 % das células previstas avaliáveis caem
    20 pontos ou mais.
  - (b) Especificidade: em todas as potências, no máximo 20 % das células dobradas (fator ≥ 2)
    sem captura prevista caem 20 pontos ou mais.
- **H1' (é a impulsividade).** P\* é a menor potência em que a queda média do impulsivo, nas
  células previstas, chega a 20 pontos. Em P\*, a queda média do substituto nas mesmas células é
  menor que metade da do impulsivo.
- **H2 (o TAT evita a captura).** O TAT, como preditor de acerto, tem acurácia balanceada de pelo
  menos 0,8 sobre todas as células (registro, condição, fator) do braço uniforme no teto de
  referência. E `taxa_tat` fica, em média, a no máximo 5 pontos do oráculo uniforme.

## O que refutaria

- **H1** é refutada se:
  - (a) menos de 50 % das células previstas caem na potência máxima; ou
  - (b) mais de 50 % das células dobradas sem captura prevista caem, em alguma potência.
- **H1'** é refutada se, em P\*, a queda do substituto for pelo menos 0,8 vez a do impulsivo.
- **H2** é refutada se a acurácia balanceada do TAT for menor que 0,65, ou se `taxa_tat` ficar mais
  de 5 pontos abaixo do oráculo em média.
- **Entre os limiares de apoio e de refutação, o resultado é reportado como inconclusivo.**

## Baseline

O pipeline do E02 sem injeção, no mesmo fator e no mesmo teto.

## Verdade fundamental

A frequência de falha pela geometria do 6205 (DE) e o rpm do registro. O interferente e a sua
imagem são conhecidos por construção.

## Limitações declaradas

- São 13 registros, com uma falha de pista interna, e o intervalo de confiança é por registro.
- O interferente é sintético. A generalização a um segundo conjunto real (Paderborn) fica para
  depois.

## Resultado da rodada base (2026-11-11; rajadas e substituto)

Feita no servidor de 06:20 a 10:58 UTC: 13 registros × 33 condições, 429 tarefas, nenhuma falha,
22.308 linhas. A análise (`analyze.py`), escrita antes da rodada, está em `results/analysis.json`,
com IC por bootstrap por registro.

As células do braço uniforme se dividem em 48 com captura prevista, 19 sem captura prevista e 21 de
borda.

| critério pré-registrado | medido | veredicto |
|---|---|---|
| H1a: na potência máxima (+20 dB), ≥ 80 % das células previstas caem ≥ 20 pontos | **85,4 %**, IC [47,9; 89,6] % | apoiada pelo ponto; o IC desce abaixo do limiar de refutação (50 %) |
| H1b: ≤ 20 % das células dobradas sem captura prevista caem, em todas as potências | **0 %** em todas | apoiada, com só **7** células avaliáveis |
| H1': em P\*, substituto/impulsivo < 0,5 | P\* = +20 dB; razão **0,14**, IC [−0,01; 0,23] | apoiada |
| H2: acurácia balanceada do TAT ≥ 0,8 | **0,69**, IC [0,63; 0,79] | inconclusiva (entre 0,65 e 0,8) |
| H2: `taxa_tat` no máximo 5 pontos abaixo do oráculo | **−5,2 pontos**, IC [−8,4; −2,7] | refutada, por pouco; pior célula −69 |

**Dose-resposta.** A queda média nas células previstas é de cerca de 0 ponto a −10 dB e a 0 dB, 9
pontos a +10 dB e 41 pontos a +20 dB. As falhas do CWRU 48k são fortes, e o interferente só captura
o seletor com potência cerca de 100 vezes a da banda da falha.

**Descritivo** (teto de referência, fatores ≥ 2):
- **Estimativa sobre a taxa das rajadas, o sinal direto de captura,** com o impulsivo a +20 dB:
  44 % nas células previstas, 28 % nas de borda, 6 % nas sem captura prevista. Com o substituto,
  ≤ 0,3 %.
- **Acerto do braço uniforme:** 94 % sem injeção, 54 % com o impulsivo a +20 dB, 93 % com o
  substituto a +20 dB.
- **`aaf_ideal`:** 85 % sem injeção e 83 % com o impulsivo a +20 dB. Fica imune ao interferente,
  mas o acerto sem injeção cai 9 pontos, porque corta a ressonância da falha que a dobra traria para
  a busca. É o lado oposto do argumento a favor da amostragem passa-faixa: o anti-alias protege, e
  cobra por isso.
- **Regras do TAT:** o `uniforme_teto` cai a 78 % mesmo sem injeção, e o `taxa_tat` só sobe de 54 %
  para 63 % a +20 dB.

**Leitura.**
- Num conjunto sem interferente nativo, uma rajada impulsiva de banda limitada, dobrada sobre a
  faixa de busca, captura o seletor por curtose **onde a fração dobrada prevê**.
- A captura vem da **impulsividade e não da potência**: o substituto de fase, com o mesmo espectro
  bin a bin, não captura.
- A calibração TAT não prevê nem evita a captura de modo confiável.

**Limites.**
- São 13 registros (12 BPFO e 1 BPFI); o IC da sensibilidade é largo.
- Só 7 células dobradas sem captura prevista sustentam a especificidade.
- A potência que captura é alta.
- O interferente é uma rajada de banda limitada, não o trem de picos de uma amostra do Ottawa. Esse
  é o braço de picos, rodando em `results-spikes/`.

## Resultado do braço de picos (2026-11-11; `results-spikes/`)

São 13 registros × 10 condições (130 tarefas, nenhuma falha, 8.060 linhas). Veredicto em
`results-spikes/analysis.json`, calculado por `analyze.py --config config/spikes.yaml`. Cada registro
vale 7,7 pontos.

**H-picos: inconclusiva.** Com picos de parâmetros do Ottawa (1.300/s, 16 MAD):
- o braço uniforme cai **31 pontos em 2×** e **23 em 4×**, o que atende a primeira metade;
- as mitigações **não** ficam em ≤ 5 pontos: `aaf_ideal` cai até 15 (4× e 12×), e `despike` até 15
  (6× e 12×).

**Dose-resposta** (queda máxima do uniforme em fator ≥ 2): cerca de 8 pontos a 8 MAD, 31 a 16 MAD
e 62 a 32 MAD, nas três taxas. O padrão em fator também é não monótono. Em 1.300/s a 16 MAD, 3×
e 5× não caem, enquanto 2×, 4× e 12× caem.

**Por que as mitigações não bastam aqui:**
- **Anti-alias.** Um pico de uma amostra é branco. O anti-alias tira só a parte acima do Nyquist
  efetivo, e a parte dentro da banda continua impulsiva.
- **`despike`.** Os pares de picos a duas amostras escapam do critério. São cerca de 3 % a 1.300/s
  e 48 kHz, e a 32 MAD ainda capturam o seletor: o `despike` cai até 38 pontos (12×).
- **Diferença para o Ottawa.** No Ottawa a remoção recupera tudo (26/36 em 4×). Ou os picos de
  lá são mais isolados que um processo de Poisson, ou o sinal de falha do CWRU compete de outro
  jeito. Não está isolado.
- **TAT.** O `taxa_tat` não protege, e o `uniforme_teto` piora.

**Leitura conjunta das duas rodadas.**
- Num conjunto sem interferente nativo, tanto uma rajada impulsiva de banda limitada quanto um
  trem de picos de uma amostra capturam o seletor por curtose sob decimação pura.
- A captura depende da impulsividade (substituto de fase), da amplitude e de onde a energia
  dobrada cai.
- O anti-alias ideal desfaz quase toda a captura pela rajada, mas não a dos picos, e custa acerto
  sem interferente.
- A remoção de picos ajuda contra picos e não contra rajadas.

Nenhuma mitigação simples resolve as duas.

**Antes de entrar no artigo:** revisão adversarial (`reviewer-2`) das duas rodadas juntas.

## Estado

Implementado. Testes em `tests/test_e23.py`, com referência analítica para a fração dobrada e
controle do substituto. Fumaça com 2 registros. A rodada completa foi lançada no servidor às
06:20 UTC de 2026-11-11.

A análise (`analyze.py`) foi escrita **antes** da rodada completa e implementa exatamente os
critérios acima. `tests/test_analyze.py` verifica o veredicto em casos sintéticos de resposta
conhecida: captura só onde prevista, e substituto que derruba tanto quanto o impulsivo.

## Revisão adversarial (`reviewer-2`, 2026-11-11): revisão maior, conferida

Os números das duas rodadas batem com o CSV. Recalculei de forma independente os três pontos que
mais pesam: a dose, as unidades independentes e o poder preditivo da fração dobrada.

**Correções de fato:**
- **A dose está rotulada 3 dB acima.** O `band_power` (`run.py`) devolve metade da potência: um
  seno de potência 0,5 dá 0,25. O "+20 dB" é **+17 dB** (cerca de 50×, não 100×), e o mesmo vale para
  as outras potências.
  - A dose é normalizada pela banda que o próprio kurtograma escolhe, e a largura dela vai de 594 a
    4.750 Hz entre registros. Em densidade espectral, o mesmo rótulo vai de 16,7 a 25,7 dB.
  - Pelo revisor, as perdas por registro seguem essa largura, com ρ = 0,96.
- **Unidades independentes.** Os 12 registros BPFO são **uma só falha (0,007") em 3 posições**
  (centrada 135–138, ortogonal 148–151, oposta 161–164) × 4 cargas. Com o 213 (BPFI), são 4
  aglomerados.
  - Pelo revisor, o bootstrap por aglomerado dá a sensibilidade em [0,42; 0,90], com
    P(< 0,5) = 0,21. Só a posição ortogonal dá 0,43.
- **A fração dobrada não prevê o tamanho da queda.** Dentro das células previstas, Spearman ≈ 0,00
  (conferido). Nenhuma das 6 células previstas de 10× e 12× cai 20 pontos.
- **O H1b é tautológico.** O kurtograma particiona só [f_min, teto] com máscaras ideais, então uma
  imagem acima do teto não entra em banda nenhuma por construção. O 0/7 não podia falhar.
- **O ponto final da H1a mudou depois da fumaça.** A H1a passou a ser lida na potência máxima depois
  que a fumaça a refutou a 0 dB. O README declarou a extensão de potência, mas não a troca de ponto
  final.
  - Sem os dois registros da fumaça (135DE e 136DE), a H1a cai a 0,60: inconclusiva.
  - Sem commit, o pré-registro não é verificável: o diretório está fora do git, e `git_sha` é nulo.
- **Os picos não têm os "parâmetros do Ottawa".**
  - A taxa foi casada por segundo. Por amostra, a densidade é 4,1× a do Ottawa: 2,7 % a 48 kSPS contra
    0,67 % a 200 kSPS.
  - Medida no sinal contaminado, a amplitude é de 13,7 a 16,2 MAD.
  - O `despike` deixa escapar de 6 a 14 % dos picos, não "cerca de 3 %". Na densidade do Ottawa, os
    pares cairiam a cerca de 1,3 %, o que explica por que lá a remoção recupera tudo.
- **O "inconclusivo" da H-picos estava determinado pelo desenho.** O limiar de 5 pontos fica abaixo de
  1 registro (7,7 pontos) e inclui 1×, onde o `aaf_ideal` é a identidade.
- **"31 a 16 MAD nas três taxas" está errado:** são 23, 31 e 23. O padrão em fator a 16 MAD é ruído
  de ±1 registro.

**O que o artigo pode afirmar a partir do E23:**
- sob decimação pura, um interferente impulsivo acima do teto volta à busca e, com dose suficiente,
  captura o kurtograma; 97 % das perdas caem sobre a taxa das rajadas;
- para seletores por curtose, a captura exige impulsividade: um substituto gaussiano de mesmo
  espectro dá razão 0,14 (IC por aglomerado [0,02; 0,22]);
- picos de uma amostra também capturam, com dose-resposta em amplitude;
- um teto de busca que protege o seletor em taxa cheia é desfeito pela subamostragem; é a
  contribuição própria, estreita;
- o passa-baixas evita a captura e custa 8,5 pontos em 10× e 12×, o compromisso clássico.

**O que NÃO pode afirmar:**
- "onde a aritmética da dobra prevê, e só ali";
- "a impulsividade, e não a potência" em geral, para qualquer seletor; o substituto preserva a
  cicloestacionariedade;
- "picos com os parâmetros do Ottawa";
- "nenhuma mitigação simples resolve as duas";
- a H1a como confirmatória;
- novidade sobre EMI contra o kurtograma em taxa cheia (`smith2016emi`, `vaerenberg2025cyclicemi`,
  Mauricio et al. 2020).

**O que faltaria para uma versão confirmatória (E23b, não lançado):**
- **Desenho:**
  - desenho e análise congelados em commit antes da rodada;
  - dose independente do seletor, com `band_power` corrigido;
  - preditor pré-registrado que compare o interferente dobrado com a falha dobrada;
  - estatística por aglomerado;
  - picos com densidade por amostra casada (≈ 320–590/s a 48 kHz) e o `despike` corrigido para os
    pares.
- **Baselines que faltam:**
  - anti-alias **passa-faixa** com taxa válida (`vaughan1991`);
  - seletores robustos: infograma, protrugrama, autograma, coerência espectral e filtros FRESH;
  - `despike` e mediana aplicados ao fluxo **já decimado**, o que é realizável.
- **Confirmação:**
  - cadeia física do `snq-sim`;
  - um segundo conjunto (Paderborn).
