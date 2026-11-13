"""Exploratory (adversarial review of 2026-11-13): where does the decoder's margin come from?

Splits the nested-ruler balanced accuracy of E15 on Paderborn into two strata of bearings: those
with a valid local ruler (fault line visible at full rate in some segment) and those without. The
aggregation is the one in ``summary.csv``: hits grouped by race, mean over the races.

Not preregistered. Usage:
    uv run python experiments/E15-learned-sampling-mask/explore_line_split.py
Writes ``results-paderborn/exploratory/line_split.csv``.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "results-paderborn" / "metrics.csv.gz"
OUT = HERE / "results-paderborn" / "exploratory" / "line_split.csv"
ARMS = ("uniform_trained", "dec_uniform_trained", "dec_random", "dec_learned_sel")
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
            r for r in csv.DictReader(fh) if r["dataset"] == "paderborn" and r["variant"] == "raw"
        ]
    with_line = {bearing(r["record"]) for r in rows if r["truth_local"] not in ("", "nan")}
    out = []
    for b in BUDGETS:
        for arm in ARMS:
            sub = [r for r in rows if r["budget"] == b and r["arm"] == arm]
            strata = {
                "all": sub,
                "with_line": [r for r in sub if bearing(r["record"]) in with_line],
                "without_line": [r for r in sub if bearing(r["record"]) not in with_line],
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
    print("bearings with a local ruler:", sorted(with_line))
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
