# E25 - Co-arranjo coprimo: estatística cíclica da falha sem dobra do eixo da portadora

Aberto em 2026-11-12, depois do E20, do E23 e do E24.

## Por que este experimento

A Parte (a) do artigo mostra o mecanismo: sob decimação uniforme pura, conteúdo de fora da banda
dobra sobre a faixa de busca e captura o seletor de banda. A proposta aqui é amostrar de modo que
a dobra não aconteça no domínio que o diagnóstico usa.

- **O diagnóstico não precisa da forma de onda.** Basta a correlação cíclica R(τ, α) nas
  frequências cíclicas da falha, e a geometria do rolamento dá quais são.
- **Dois amostradores coprimos.**
  - Com passos M e N (mdc = 1) rodando sem parar, todo atraso inteiro aparece (Bézout), com cerca
    de T/(MN) pares por atraso.
  - Por isso o eixo da portadora fica sem dobra até f_s/2, com taxa média de 1/M + 1/N − 1/(MN)
    da taxa cheia.
  - Com M = 8 e N = 9, a taxa média é 0,222, abaixo de 1/4.
- **O espectro do envelope ao quadrado de qualquer banda** sai de R(τ, α)
  (`snq_recon.coarray.band_ses`). O pipeline continua o clássico; só a estimação de segunda ordem
  muda.
- **Hardware:** no nRF54L15 são dois temporizadores disparando o SAADC pelo DPPI, sem a CPU.

## Novidade (checagem de 2026-11-12, `lit-scout`)

**Termos buscados** (citações para frente no Semantic Scholar e palavras-chave no Crossref e no
OpenAlex): amostragem coprima, régua esparsa ou multicoset, combinados com espectro cíclico,
coerência espectral ou cicloestacionaridade, e com rolamento, vibração ou diagnóstico.

Não foi encontrado trabalho que reúna as quatro peças:
- amostragem coprima ou régua esparsa;
- estimação da correlação cíclica pelo co-arranjo;
- frequências cíclicas da falha dadas pela geometria;
- diagnóstico de rolamento.

**Os mais próximos são parciais:**
- `xu2026covpreserving`: régua esparsa em máquina, sem estatística cíclica; preprint não revisado.
- `cao2024dcs`: co-arranjo coprimo para senoides estacionárias.
- `tian2012cyclic`, `cohen2017subnyquistcyc` e `zhang2026fccse`: espectro cíclico sub-Nyquist em
  comunicações.
- `gkoktsi2019psbs`: multicoset de baixa potência para espectro de potência estrutural.

**Citações para frente de 2022 para cá**, filtradas pelo título, sem nenhum título de rolamento:
- 661 de `vaidyanathan2011coprime`;
- 26 de `tian2012cyclic`;
- 50 de `romero2016ccs`.

**Teoria que não se aplica diretamente.** O limite de 4/5 da taxa de Nyquist de
`cohen2017subnyquistcyc` (Seção IV-B, Teorema 1, Eq. 52) é uma condição suficiente para recuperar o
espectro cíclico **inteiro**, todas as frequências cíclicas, sem conhecimento prévio. Aqui só
importam poucas frequências cíclicas baixas e conhecidas.

A condição para o nosso caso precisa ser derivada. Ela tem duas partes:
- **Cobertura de atrasos:** todos os inteiros, pelo teorema de Bézout.
- **Taxa cíclica:** α_max < f_s/(2MN), porque o produto de cada atraso só existe uma vez a cada MN
  amostras.

**Riscos apontados pela literatura:**
- a variância por atraso cresce com MN/T;
- o comprimento de registro necessário;
- a rotação variável (`abboud2017orderfreq`);
- a hipótese de estacionaridade no sentido amplo.

## Piloto exploratório (2026-11-12, NÃO pré-registrado)

O piloto decide se vale um pré-registro. Os números dele não vão ao artigo como confirmatórios.

**Pergunta.** Na mesma taxa média, ou abaixo dela, o co-arranjo coprimo preserva a assinatura
cíclica da falha melhor que a decimação uniforme quando um interferente impulsivo acima do Nyquist
efetivo dobra sobre a banda da falha?

**Protocolo** (`pilot.py`), o de um nó com comissionamento:
- **Dados:** os 13 registros de falha do CWRU 48k do E23, com rotação constante.
- **Calibração:** a banda da falha é calibrada uma vez em taxa cheia, na primeira metade do
  registro e sem injeção, pelo kurtograma do E23 com teto de 10 kHz.
- **Teste:** na segunda metade.
  - Sem injeção, ou com as rajadas impulsivas do E23 (97,7 Hz, portadoras de 13, 16, 19 e
    22 kHz) na dose máxima. É a mesma semente do E23, com rótulo de +20 dB e +17 dB reais.
- **Padrões, todos com o mesmo estimador:** espectro do envelope ao quadrado da banda calibrada,
  pela correlação cíclica, até o atraso 192.
  - taxa cheia;
  - uniforme por 4 (0,25);
  - coprimo (8, 9) (0,222).
- **Estimativa:** o argmax entre 20 e 330 Hz, sem prior. O teto de 330 Hz fica abaixo de
  f_s/(2·72) = 333 Hz.
- **Acerto:** |estimativa − frequência da geometria| ≤ 2 Hz, como no E02 e no E23.
- **Também registrado:** a proeminência da raia de falha, isto é, o pico a ±2 Hz da verdade
  dividido pela mediana em ±20 %.
- **Referência:** o resultado do kurtograma cego do E23 em 4×, nas mesmas condições.

## Resultado do piloto

Preenchido depois de rodar.
