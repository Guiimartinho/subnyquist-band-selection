"""Exploratorio (revisao adversarial de 2026-11-13): de onde vem a margem do decodificador?

Separa a acuracia balanceada pela regua aninhada do E15 no Paderborn em dois estratos de
rolamentos: os que tem regua local valida (raia de falha visivel em taxa cheia em algum segmento)
e os que nao tem. A agregacao e a do ``summary.csv``: acerto agrupado por pista, media das pistas.

Nao e pre-registrado. Uso:
    uv run python experiments/E15-learned-sampling-mask/explore_line_split.py
Grava ``results-paderborn/exploratory/line_split.csv``.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "results-paderborn" / "metrics.csv.gz"
OUT = HERE / "results-paderborn" / "exploratory" / "line_split.csv"
ARMS = ("uniforme_treino", "dec_uniforme_treino", "dec_aleatoria", "dec_aprendida_sel")
BUDGETS = ("4", "8", "16", "32")


def bearing(record: str) -> str:
    return record.split("_")[0]


def balanced(rows: list[dict]) -> tuple[float, int]:
    hits: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["hit_nested"] in ("", "nan") or r.get("skipped") == "1":
            continue
        hits[r["family_true"]] += float(r["hit_nested"])
        n[r["family_true"]] += 1
    if not n:
        return float("nan"), 0
    return 100.0 * sum(hits[k] / n[k] for k in n) / len(n), sum(n.values())


def main() -> None:
    with gzip.open(SRC, "rt", encoding="utf-8") as fh:
        rows = [
            r for r in csv.DictReader(fh) if r["dataset"] == "paderborn" and r["variant"] == "bruto"
        ]
    with_line = {bearing(r["record"]) for r in rows if r["truth_local"] not in ("", "nan")}
    out = []
    for b in BUDGETS:
        for arm in ARMS:
            sub = [r for r in rows if r["budget"] == b and r["arm"] == arm]
            strata = {
                "todos": sub,
                "com_raia": [r for r in sub if bearing(r["record"]) in with_line],
                "sem_raia": [r for r in sub if bearing(r["record"]) not in with_line],
            }
            for name, part in strata.items():
                acc, n = balanced(part)
                bearings = len({bearing(r["record"]) for r in part})
                out.append(
                    {
                        "budget": b,
                        "arm": arm,
                        "stratum": name,
                        "bearings": bearings,
                        "segments": n,
                        "bal_acc_nested_pct": round(acc, 1),
                    }
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("rolamentos com regua local:", sorted(with_line))
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
