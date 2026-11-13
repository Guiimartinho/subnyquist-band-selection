# Artigo único do ADR 0006 (MSSP)

**Estado em 2026-11-13:** o rascunho está completo, em inglês, porque a revista é a MSSP.
- A revisão adversarial do manuscrito inteiro (`reviewer-2`) pediu revisão maior, e ela foi
  aplicada; o que mudou está no caderno `docs/lab-notebook/2026-11-13.md`.
- O apêndice (`sections/appendix_ledger.tex`) lista as 35 hipóteses pré-registradas, com veredito
  e com a informação de que cada uma foi commitada antes de rodar ou não.
- Os `\todo` que restam aparecem no topo do PDF: autores, link do código e declarações.

**Estrutura:**

| Caminho | O que é |
|---|---|
| `main.tex` | título, resumo (até 250 palavras), palavras-chave, ordem das seções |
| `sections/` | introdução, trabalhos relacionados, métodos, resultados, discussão, conclusão, apêndice |
| `tables/` | tabelas geradas por `papers/shared/figures/tab_*.py` |
| `figures/` | PDFs vetoriais gerados por `papers/shared/figures/fig_*.py` |
| `numbers/` | JSONs com os números citados, gerados por `papers/shared/figures/num_*.py` |
| `highlights.txt` | os cinco destaques da Elsevier, até 85 caracteres cada |

**Figuras:**

| Figura | Script | Conteúdo |
|---|---|---|
| `fig_valley_nested.pdf`, `fig_valley_local.pdf` | `fig_partA_valley.py --ruler nested\|local` | vale de decimação no Ottawa (E20) |
| `fig_paderborn.pdf` | `fig_partA_paderborn.py` | réplica no Paderborn (E24) |
| `fig_rule_paderborn.pdf` | `fig_rule_paderborn.py` | regra das ordens de falha, aquisição e perfil por zona (E26, E26b, E26c) |
| `fig_masks_paderborn.pdf` | `fig_masks_paderborn.py` | máscaras aprendidas nas duas réguas (E15) |

Toda curva tem banda de 95 % por bootstrap das unidades independentes. Cada script também grava um
PNG no diretório `figures/` do experimento, para o relatório em Markdown. Os dois scripts novos
conferem os pontos contra os arquivos de análise da rodada e param se algo divergir.

**Como compilar:**

```
latexmk -pdf main.tex
```

Não há toolchain de LaTeX nesta máquina nem no servidor, então o PDF ainda não foi compilado. A
checagem estática (rótulos, `\input`, figuras e ambientes) passa. A bibliografia é a fonte única,
`../../literature/refs.bib`, e o texto só cita entradas com campo `verified`.

**Antes de submeter:**
- [x] nenhuma referência `unverified` no texto (`ci_checks.py refs`);
- [x] toda figura regenerável a partir de `metrics.csv`;
- [x] caso de falha documentado (seção "Where the method fails" e apêndice);
- [x] `reviewer-2` no manuscrito inteiro;
- [ ] `literature/scout.py --anchors` rodado de novo, com as âncoras novas sugeridas pela revisão;
- [ ] autores, CRediT, conflito de interesses, financiamento e disponibilidade de dados;
- [ ] declaração de uso de IA generativa na redação (política da Elsevier);
- [ ] link do código e dos dados, com DOI arquivado;
- [ ] compilação, e conferência visual do PDF;
- [ ] escolha do periódico.
