"""Tabela do E26 no Paderborn: regra das ordens de falha, escolha de banda e aquisição.

Regra do projeto: a tabela nasce dos ``analysis.json`` do experimento, nunca de número transcrito.

- **Fontes,** todas do fluxo bruto:
  - `E26/results-paderborn/analysis.json`: o kurtograma cego;
  - `E26/results-paderborn-regra/analysis.json`: o E26b, com a regra fixa e a banda mudando;
  - `E26/results-paderborn-aquisicao/analysis.json`: o E26c, com anti-alias antes de decimar.
- **Linhas:** a acurácia balanceada nos danificados, por fator.

    uv run python papers/shared/figures/tab_guided_paderborn.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E26 = ROOT / "experiments/E26-fault-order-guided-selector"
OUT = ROOT / "papers/mssp-2026/tables/tab_guided_paderborn.tex"
FACTORS = (1, 4, 8, 16, 32)


def load(name: str) -> dict:
    return json.loads((E26 / name / "analysis.json").read_text(encoding="utf-8"))["summary"]


def main() -> None:
    e26, e26b, e26c = (
        load("results-paderborn"),
        load("results-paderborn-regra"),
        load("results-paderborn-aquisicao"),
    )
    rows = [
        ("pure decimation", "kurtogram band, blind order estimate", e26, "kurtograma|{f}x"),
        ("", "kurtogram band, fault-order rule", e26b, "kurtograma_regra|bruto|{f}x"),
        ("", "wide band, fault-order rule", e26b, "banda_larga_regra|bruto|{f}x"),
        ("", "guided band, fault-order rule", e26b, "guiado|bruto|{f}x"),
        (
            "anti-alias filter",
            "kurtogram band, fault-order rule",
            e26c,
            "antialias|kurtograma_regra|{f}x",
        ),
        ("", "wide band, fault-order rule", e26c, "antialias|banda_larga_regra|{f}x"),
        ("", "guided band, fault-order rule", e26c, "antialias|guiado|{f}x"),
    ]
    cols = " & ".join(f"{f}$\\times$" for f in FACTORS)
    lines = [
        r"\begin{tabular}{ll" + "c" * len(FACTORS) + "}",
        r"\toprule",
        rf"acquisition & band and decision & {cols} \\",
        r"\midrule",
    ]
    for i, (acq, label, src, key) in enumerate(rows):
        if i == 4:
            lines.append(r"\midrule")
        cells = []
        for f in FACTORS:
            v = src.get(key.format(f=f))
            cells.append("--" if v is None or v.get("balanced") is None else f"{v['balanced']:.1f}")
        lines.append(f"{acq} & {label} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = (
        "% gerado por papers/shared/figures/tab_guided_paderborn.py a partir dos analysis.json do\n"
        "% E26, E26b e E26c no Paderborn; nao editar a mao. Fluxo bruto, acuracia balanceada (%).\n"
    )
    OUT.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
