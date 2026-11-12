"""Critério de recuperação do E20 aplicado a um metrics.csv, nas duas réguas.

O critério é o pré-registrado no README (seções "Seletores robustos" e "Remoção depois de
decimar"), na pista externa, em 4× e com teto de 25 kHz:
- **recupera:** pelo menos 20 de 36 acertos e McNemar exato pareado contra ``none`` com p < 0,01;
- **não recupera:** até 10 de 36;
- **inconclusivo:** o resto.

Vale para as duas réguas: a aninhada (coluna ``hit``, a mediana deixa-um-registro-de-fora do E07)
e a local (``hit_local``).

**Sensibilidade, fora do pré-registro.** O McNemar trata os 36 segmentos como independentes, mas
são 3 por registro (README, "Nível de segmento"). Por isso o script também reporta um teste do
sinal exato sobre os 12 registros, com a diferença de acertos por registro.

    uv run python experiments/E20-fold-zone-rejection/recovery.py results-selectors/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
RULERS = {"nested": "hit", "local": "hit_local"}
CELL = {"condition": "outer_race", "factor": 4, "ceiling_hz": 25000.0}
RECOVER_HITS, NO_RECOVERY_HITS, ALPHA = 20, 10, 0.01


def mcnemar_exact_p(b: int, c: int) -> float:
    """p bicaudal exato de McNemar (igual ao de ``run.py``)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def sign_test_p(diffs: list[int]) -> float:
    """p bicaudal exato do teste do sinal; zeros saem."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return mcnemar_exact_p(pos, neg)


def verdict(hits: int, p: float) -> str:
    if hits >= RECOVER_HITS and p < ALPHA:
        return "recupera"
    if hits <= NO_RECOVERY_HITS:
        return "nao recupera"
    return "inconclusivo"


def load(path: Path) -> dict[tuple, dict[tuple, int]]:
    """cell[(intervenção, régua, fator)] -> {(registro, segmento): acerto}.

    Só a pista externa e o teto de 25 kHz.
    """
    cell: dict[tuple, dict[tuple, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["condition"] != CELL["condition"] or float(r["ceiling_hz"]) != CELL["ceiling_hz"]:
                continue
            for ruler, col in RULERS.items():
                if r[col] != "":
                    key = (r["intervention"], ruler, int(r["factor"]))
                    cell[key][(r["record"], r["segment"])] = int(r[col])
    return cell


def compare(arm: dict[tuple, int], ctrl: dict[tuple, int]) -> dict:
    common = sorted(set(arm) & set(ctrl))
    b = sum(1 for k in common if arm[k] == 1 and ctrl[k] == 0)
    c = sum(1 for k in common if arm[k] == 0 and ctrl[k] == 1)
    per_rec: dict[str, int] = defaultdict(int)
    for rec, seg in common:
        per_rec[rec] += arm[(rec, seg)] - ctrl[(rec, seg)]
    p = mcnemar_exact_p(b, c)
    hits = sum(arm.values())
    return {
        "n": len(arm),
        "hits": hits,
        "hits_none": sum(ctrl.values()),
        "gain": b,
        "loss": c,
        "mcnemar_p": float(f"{p:.3g}"),  # 3 algarismos: p pequeno não vira 0
        "verdict": verdict(hits, p),
        "records_up": sum(1 for d in per_rec.values() if d > 0),
        "records_down": sum(1 for d in per_rec.values() if d < 0),
        "sign_test_p_records": float(f"{sign_test_p(list(per_rec.values())):.3g}"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path, help="relativo ao diretório do E20 ou absoluto")
    args = ap.parse_args()
    path = args.metrics if args.metrics.is_absolute() else EXP / args.metrics
    cell = load(path)
    arms = sorted({k[0] for k in cell if k[0] != "none"})
    factors = sorted({k[2] for k in cell})
    rows = []
    for arm in arms:
        for ruler in RULERS:
            ctrl = cell.get(("none", ruler, CELL["factor"]), {})
            res = compare(cell.get((arm, ruler, CELL["factor"]), {}), ctrl)
            sweep = " ".join(f"{f}x:{sum(cell.get((arm, ruler, f), {}).values())}" for f in factors)
            rows.append({"intervention": arm, "ruler": ruler, **res, "hits_by_factor": sweep})
    out = path.parent / "recovery.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    base = {
        r: " ".join(f"{f}x:{sum(cell.get(('none', r, f), {}).values())}" for f in factors)
        for r in RULERS
    }
    print(f"pista externa, teto 25 kHz; none: aninhada {base['nested']} | local {base['local']}")
    for r in rows:
        print(
            f"{r['intervention']:<24} {r['ruler']:<6} 4x: {r['hits']}/{r['n']} "
            f"(none {r['hits_none']}) +{r['gain']}/-{r['loss']} McNemar p={r['mcnemar_p']:.2g} "
            f"-> {r['verdict']} | registros +{r['records_up']}/-{r['records_down']} "
            f"sinal p={r['sign_test_p_records']:.2g} | {r['hits_by_factor']}"
        )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
