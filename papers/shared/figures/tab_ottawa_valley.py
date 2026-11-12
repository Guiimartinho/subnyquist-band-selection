"""Tabela da Parte (a) no Ottawa (E20): acertos da pista externa por intervenção e fator.

Regra do projeto: a tabela nasce de metrics.csv, nunca de número transcrito.

- **Célula:** "aninhada (local)". A régua aninhada tem 36 segmentos (12 registros x 3) e a local
  tem 35, porque um segmento não tem raia válida em taxa cheia.
- **Linhas:** as intervenções com teto de 25 kHz. A última é `none` com o teto de 12,5 kHz.

    uv run python papers/shared/figures/tab_ottawa_valley.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METRICS = ROOT / "experiments/E20-fold-zone-rejection/results/metrics.csv"
OUT = ROOT / "papers/mssp-2026/tables/tab_ottawa_valley.tex"
ROWS = [  # (intervenção, teto, rótulo LaTeX)
    ("none", 25000.0, r"raw, no intervention"),
    ("stop_z1", 25000.0, r"reject $Z_1$ (25--37.25~kHz)"),
    ("stop_z2", 25000.0, r"reject $Z_2$ (62.75--75~kHz)"),
    ("stop_z3", 25000.0, r"reject $Z_3$ (75--87.25~kHz)"),
    ("stop_z123", 25000.0, r"reject $Z_1 \cup Z_2 \cup Z_3$"),
    ("aaf_ideal", 25000.0, r"ideal anti-alias filter"),
    ("despike", 25000.0, r"single-sample spikes removed"),
    ("median3", 25000.0, r"3-point median"),
    ("none", 12500.0, r"raw, search ceiling 12.5~kHz"),
]


def main() -> None:
    hits: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    with METRICS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["condition"] != "outer_race":
                continue
            k = (r["intervention"], float(r["ceiling_hz"]), int(r["factor"]))
            if r["hit"] != "":
                hits[k][0] += int(r["hit"])
                hits[k][1] += 1
            if r["hit_local"] != "":
                hits[k][2] += int(r["hit_local"])
                hits[k][3] += 1
    factors = sorted({k[2] for k in hits})
    n_nested = {v[1] for v in hits.values()}
    n_local = {v[3] for v in hits.values()}
    assert len(n_nested) == 1 and len(n_local) == 1, (n_nested, n_local)
    lines = [
        r"\begin{tabular}{l" + "c" * len(factors) + "}",
        r"\toprule",
        r" & \multicolumn{" + str(len(factors)) + r"}{c}{decimation factor $M$} \\",
        r"\cmidrule(lr){2-" + str(len(factors) + 1) + "}",
        "intervention before decimation & " + " & ".join(f"{m}$\\times$" for m in factors) + r" \\",
        r"\midrule",
    ]
    for i, (iname, ceiling, label) in enumerate(ROWS):
        if i == len(ROWS) - 1:
            lines.append(r"\midrule")
        cells = []
        for m in factors:
            hn, _nn, hl, _nl = hits[(iname, ceiling, m)]
            cells.append(f"{hn} ({hl})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = (
        f"% gerado por papers/shared/figures/tab_ottawa_valley.py a partir de\n"
        f"% experiments/E20-fold-zone-rejection/results/metrics.csv; nao editar a mao.\n"
        f"% celula: acertos na regua aninhada (de {n_nested.pop()}) e, entre parenteses, na local "
        f"(de {n_local.pop()})\n"
    )
    OUT.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
