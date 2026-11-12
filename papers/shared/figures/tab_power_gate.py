"""Tabela da porta de poder (E15): poder da regra da H1 contra um braço plantado.

Regra do projeto: a tabela nasce dos CSVs dos experimentos, nunca de número transcrito.

- **Fontes:** `E15/results/power_gate/power.csv` (Ottawa) e `E15/results-paderborn-gate/power.csv`.
- **Recorte:** só a regra otimista, e só os desenhos válidos. O D3 do Ottawa, que faz bootstrap
  por registro, fica de fora por ser anticonservador.

    uv run python papers/shared/figures/tab_power_gate.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E15 = ROOT / "experiments/E15-learned-sampling-mask"
OUT = ROOT / "papers/mssp-2026/tables/tab_power_gate.tex"
SOURCES = [  # (rótulo, csv, [(desenho, rótulo do desenho)])
    (
        "Ottawa",
        E15 / "results/power_gate/power.csv",
        [
            ("D0", "as run (1 seed)"),
            ("D1", "3 seeds"),
            ("D2", "5 seeds"),
            ("DK", "each record once"),
        ],
    ),
    (
        "Paderborn",
        E15 / "results-paderborn-gate/power.csv",
        [("D0", "1 seed"), ("D1", "3 seeds"), ("D2", "5 seeds")],
    ),
]
EFFECTS = ["5", "10", "perfeito"]


def main() -> None:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"benchmark & design & +5 points & +10 points & perfect arm \\",
    ]
    for bench, path, designs in SOURCES:
        with path.open(encoding="utf-8") as f:
            power = {
                (r["design"], r["effect_points"]): float(r["power_h1"])
                for r in csv.DictReader(f)
                if r["rule"] == "otimista"
            }
        lines.append(r"\midrule")
        for i, (code, label) in enumerate(designs):
            cells = " & ".join(f"{power[(code, e)]:.2f}" for e in EFFECTS)
            lines.append(f"{bench if i == 0 else ''} & {label} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = (
        "% gerado por papers/shared/figures/tab_power_gate.py a partir dos power.csv do E15;\n"
        "% nao editar a mao. Regra otimista, desenhos validos.\n"
    )
    OUT.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
