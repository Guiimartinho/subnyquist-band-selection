"""Tabela do E26 no Paderborn: o seletor guiado contra o kurtograma cego, no fluxo decimado.

Regra do projeto: a tabela nasce do ``analysis.json`` do experimento, nunca de número transcrito.

- **Fonte:** `E26/results-paderborn/analysis.json`, escrito por `analyze_paderborn.py`.
- **Conteúdo:**
  - a acurácia balanceada nos rolamentos danificados e a especificidade nos sãos, por braço e
    fator;
  - as diferenças pré-registradas (H1 e H2), com o p de Holm.

    uv run python papers/shared/figures/tab_guided_paderborn.py [--analysis PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E26 = ROOT / "experiments/E26-fault-order-guided-selector"
ANALYSIS = E26 / "results-paderborn/analysis.json"
OUT = ROOT / "papers/mssp-2026/tables/tab_guided_paderborn.tex"
ARMS = [
    ("kurtograma", "blind kurtogram"),
    ("mediana_kurtograma", "median + blind kurtogram"),
    ("guiado", "guided"),
    ("mediana_guiado", "median + guided"),
]
TESTS = [("H1", "vs blind kurtogram (H1)"), ("H2", "vs median + blind kurtogram (H2)")]


def fmt(v) -> str:
    return "--" if v is None else f"{v:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=ANALYSIS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rep = json.loads(args.analysis.read_text(encoding="utf-8"))
    summary, tests = rep["summary"], rep["tests"]
    factors = sorted({int(k.split("|")[1].rstrip("x")) for k in summary})
    cols = " & ".join(f"{f}$\\times$" for f in factors)
    n = len(factors) + 1
    lines = [
        rf"\begin{{tabular}}{{l{'c' * len(factors)}}}",
        r"\toprule",
        rf"decimation factor & {cols} \\",
    ]
    for title, field in (
        ("balanced accuracy on damaged bearings [\\%]", "balanced"),
        ("specificity on healthy bearings [\\%]", "specificity"),
    ):
        lines += [r"\midrule", rf"\multicolumn{{{n}}}{{l}}{{\emph{{{title}}}}} \\"]
        for arm, label in ARMS:
            cells = " & ".join(fmt(summary[f"{arm}|{f}x"][field]) for f in factors)
            lines.append(f"{label} & {cells} \\\\")
    diff_title = "median + guided minus baseline [points] (Holm $p$)"
    lines += [r"\midrule", rf"\multicolumn{{{n}}}{{l}}{{\emph{{{diff_title}}}}} \\"]
    for hyp, label in TESTS:
        by_f = {t["factor"]: t for t in tests if t["hypothesis"] == hyp}
        cells = []
        for f in factors:
            t = by_f.get(f)
            if t is None:
                cells.append("--")
                continue
            mark = r"$^{\ast}$" if t["passes"] else ""
            cells.append(f"{t['diff_points']:+.1f} ({t['p_holm']:.3f}){mark}")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    head = (
        "% gerado por papers/shared/figures/tab_guided_paderborn.py a partir do analysis.json do\n"
        "% E26 no Paderborn; nao editar a mao. Asterisco: passa o criterio pre-registrado.\n"
    )
    args.out.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
