"""E26 numbers in the paper (fault-order rule and acquisition), read from ``analysis.json``.

The difference in differences and the per-bearing count come from the scripts of the adversarial
review, versioned in ``experiments/E26-fault-order-guided-selector/exploratory/review_2026-11-13/``
(``a2_did.py`` and ``a10_bearinglevel.py``).

    uv run python papers/shared/figures/num_partA_rule.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E26 = ROOT / "experiments/E26-fault-order-guided-selector"
OUT = ROOT / "papers/mssp-2026/numbers/partA_rule.json"


def load(name: str) -> dict:
    return json.loads((E26 / name / "analysis.json").read_text(encoding="utf-8"))


def tests(rep: dict) -> list[dict]:
    keep = ("hypothesis", "clean", "factor", "diff_points", "bearings_up", "bearings_differing")
    return [{**{k: t[k] for k in keep if k in t}, "p_holm": t["p_holm"]} for t in rep["tests"]]


def main() -> None:
    e26, e26b, e26c = (
        load("results-paderborn"),
        load("results-paderborn-rule"),
        load("results-paderborn-acquisition"),
    )
    s26, sb, sc = e26["summary"], e26b["summary"], e26c["summary"]
    zones = {
        f"{f}x": {
            k.split("|")[0]: sc[k]["balanced"]
            for k in sc
            if k.endswith(f"|wideband_rule|{f}x")
            and (k.startswith("antialias") or k.split("|")[0][4:].isdigit())
        }
        for f in (4, 8)
    }
    spec = [
        v["specificity"] for v in [*sb.values(), *sc.values()] if v.get("specificity") is not None
    ]
    rep = {
        "confirmation": {
            "blind_kurtogram": {f: s26[f"kurtogram|{f}x"]["balanced"] for f in (1, 4, 8, 16, 32)},
            "median_guided": {f: s26[f"median_guided|{f}x"]["balanced"] for f in (1, 4, 8, 16, 32)},
            "median_guided_specificity": {
                f: s26[f"median_guided|{f}x"]["specificity"] for f in (1, 4, 8, 16, 32)
            },
            "tests": tests(e26),
        },
        "rule_fixed": {"tests": tests(e26b)},
        "acquisition": {
            "tests": tests(e26c),
            "zone_profile_wide_band": zones,
            "leakage_and_decoys": e26c["leakage_and_decoys"],
            "integrity": e26c["integrity"],
        },
        "specificity_range_rule_arms": [min(spec), max(spec)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "confirmation"}, indent=1)[:3000])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
