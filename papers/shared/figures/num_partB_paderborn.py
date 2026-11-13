"""Numbers of Part (b) on Paderborn (E15), read from the results, never typed by hand.

uv run python papers/shared/figures/num_partB_paderborn.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "experiments/E15-learned-sampling-mask/results-paderborn"
OUT = ROOT / "papers/mssp-2026/numbers/partB_paderborn.json"
BUDGETS = ("4", "8", "16", "32")


def main() -> None:
    with (RES / "hypotheses.csv").open(encoding="utf-8") as f:
        allrows = list(csv.DictReader(f))
    hyp = [r for r in allrows if r["budget"]]
    with (RES / "summary.csv").open(encoding="utf-8") as f:
        summ = {(r["arm"], r["budget"]): r for r in csv.DictReader(f)}
    h1 = [r for r in hyp if r["hypothesis"] == "H1"]
    diffs = {
        ruler: {
            base: {
                r["budget"]: float(r["diff_points"])
                for r in h1
                if r["ruler"] == ruler and r["baseline"] == base
            }
            for base in sorted({r["baseline"] for r in h1})
        }
        for ruler in ("local", "nested")
    }
    best = min(h1, key=lambda r: float(r["p_one_sided"]))
    conv = {
        arm: {b: float(summ[(arm, b)]["mask_converged_frac"]) for b in BUDGETS}
        for arm in ("dec_aprendida", "dec_aprendida_sel", "dec_livre_projetada")
    }
    rep = {
        "h1_evaluable_budgets": sorted({r["budget"] for r in h1 if r["evaluable"] == "1"}, key=int),
        "h1_significant": sum(int(r["significant"] or 0) for r in h1),
        "h1_tests": len(h1),
        "h1_diff_points": diffs,
        "h1_smallest_p": {
            k: best[k]
            for k in (
                "ruler",
                "budget",
                "baseline",
                "diff_points",
                "ci_lo",
                "ci_hi",
                "p_one_sided",
                "p_holm",
            )
        },
        "h2_not_evaluable_budgets": sorted(
            {
                b
                for r in allrows
                if r["hypothesis"] == "H2" and r["budgets_not_evaluable"]
                for b in r["budgets_not_evaluable"].split()
            },
            key=int,
        ),
        "mask_converged_frac": conv,
        "bal_acc_nested_pct": {
            arm: {b: round(100 * float(summ[(arm, b)]["bal_acc_nested"]), 1) for b in BUDGETS}
            for arm in (
                "dec_aprendida_sel",
                "dec_uniforme_treino",
                "dec_aleatoria",
                "dec_multicoset_vb",
                "uniforme_treino",
            )
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=1)[:2500])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
