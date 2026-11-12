# Artigo único do ADR 0006 (MSSP)

Esqueleto de 2026-11-12, em inglês, porque a revista é a MSSP.

**Estado das seções:**
- Introdução, trabalhos relacionados e métodos estão escritos.
- Resultados, discussão e conclusão esperam três rodadas:
  - o E20 com os seletores robustos;
  - o E24 (réplica no Paderborn);
  - o E15 no Paderborn.
- Cada lacuna está marcada com `\todo`, e a lista aparece no topo do PDF.

**Como compilar:**

```
latexmk -pdf main.tex
```

A bibliografia é a fonte única, `../../literature/refs.bib`, e só cita entradas com campo
`verified`.

**Antes de submeter** (skill `paper-draft`):
- nenhuma referência `unverified` no texto;
- toda figura regenerável a partir de `metrics.csv`;
- o caso de falha documentado;
- `literature/scout.py --anchors` rodado de novo;
- `reviewer-2` no manuscrito inteiro.
