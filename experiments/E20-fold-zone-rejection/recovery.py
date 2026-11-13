"""E20 recovery criterion applied to a metrics.csv, under both rulers.

The criterion is the one pre-registered in the README (sections "Robust selectors" and "Removal
after decimating"), on the outer race, at 4× and with a 25 kHz ceiling:
- **recovers:** at least 20 of 36 hits and exact paired McNemar against ``none`` with p < 0.01;
- **does not recover:** at most 10 of 36;
- **inconclusive:** the rest.

It holds for both rulers: the nested one (column ``hit``, the E07 leave-one-record-out median)
and the local one (``hit_local``).

**Sensitivity, outside the pre-registration.** McNemar treats the 36 segments as independent, but
there are 3 per record (README, "Segment level"). So the script also reports an exact sign test
over the 12 records, on the per-record difference in hits.

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
    """Exact two-sided McNemar p (same as in ``run.py``)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def sign_test_p(diffs: list[int]) -> float:
    """Exact two-sided sign-test p; zeros are dropped."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return mcnemar_exact_p(pos, neg)


def verdict(hits: int, p: float) -> str:
    # The labels are data (recovery.csv, tests) and stay as they are: "recupera" = recovers,
    # "nao recupera" = does not recover, "inconclusivo" = inconclusive.
    if hits >= RECOVER_HITS and p < ALPHA:
        return "recupera"
    if hits <= NO_RECOVERY_HITS:
        return "nao recupera"
    return "inconclusivo"


def load(path: Path) -> dict[tuple, dict[tuple, int]]:
    """cell[(intervention, ruler, factor)] -> {(record, segment): hit}.

    Only the outer race and the 25 kHz ceiling.
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
        "mcnemar_p": float(f"{p:.3g}"),  # 3 significant digits: a small p does not become 0
        "verdict": verdict(hits, p),
        "records_up": sum(1 for d in per_rec.values() if d > 0),
        "records_down": sum(1 for d in per_rec.values() if d < 0),
        "sign_test_p_records": float(f"{sign_test_p(list(per_rec.values())):.3g}"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path, help="relative to the E20 directory, or absolute")
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
    print(f"outer race, 25 kHz ceiling; none: nested {base['nested']} | local {base['local']}")
    for r in rows:
        print(
            f"{r['intervention']:<24} {r['ruler']:<6} 4x: {r['hits']}/{r['n']} "
            f"(none {r['hits_none']}) +{r['gain']}/-{r['loss']} McNemar p={r['mcnemar_p']:.2g} "
            f"-> {r['verdict']} | records +{r['records_up']}/-{r['records_down']} "
            f"sign p={r['sign_test_p_records']:.2g} | {r['hits_by_factor']}"
        )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
