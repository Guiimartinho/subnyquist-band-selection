# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

tau = thresholds(95)
A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guided" else pred_blind(r))
VIS = {"KA01", "KA06", "KA16", "KI01", "KI16", "KI17", "KI18"}  # visible line (E24)
bearings = sorted({(r["race"], r["bearing"]) for r in ROWS if r["race"] in ("OR", "IR")})
print("== Hit rate per bearing (%), blind=raw kurtogram, mg=median_guided, g=guided ==")
print("bearing orig vis |" + "|".join(f"   {f:2d}x blind/mg/g " for f in FACTORS))
for race, b in bearings:
    s = []
    for f in FACTORS:
        k = (race, b)
        s.append(
            f" {100 * A[('kurtogram', f)][k]:5.1f}/{100 * A[('median_guided', f)][k]:5.1f}/{100 * A[('guided', f)][k]:5.1f}"
        )
    print(f"{b} {race} {'art' if b in ART else 'real'} {'V' if b in VIS else '-'}  |" + "|".join(s))
print("\n== Specificity per healthy bearing (threshold without itself), median_guided / guided ==")
for b in HEALTHY:
    print(
        b,
        " ".join(
            f"{f}x {100 * A[('median_guided', f)][('H', b)]:5.1f}/{100 * A[('guided', f)][('H', b)]:5.1f}"
            for f in FACTORS
        ),
    )
print(
    "\n== Strata: artificial x real, visible line x not (balanced accuracy of median_guided and kurtogram) =="
)


def bal_subset(arm, f, pred):
    return balanced({k: v for k, v in A[(arm, f)].items() if k[0] in ("OR", "IR") and pred(k[1])})


for name, pred in (
    ("artificial", lambda b: b in ART),
    ("real", lambda b: b not in ART),
    ("with line (E24)", lambda b: b in VIS),
    ("without line (E24)", lambda b: b not in VIS),
):
    print(
        f"  {name:15s} "
        + " | ".join(
            f"{f}x mg={bal_subset('median_guided', f, pred):5.1f} g={bal_subset('guided', f, pred):5.1f} k={bal_subset('kurtogram', f, pred):5.1f}"
            for f in FACTORS
        )
    )
print("\n== Bearings with guided hit rate < 10% and > 80%, per factor (median_guided) ==")
for f in FACTORS:
    v = {k: A[("median_guided", f)][k] for k in bearings}
    lo = sorted(k[1] for k, x in v.items() if x < 0.10)
    hi = sorted(k[1] for k, x in v.items() if x > 0.80)
    print(f"  {f:2d}x <10%: {len(lo):2d} {lo}\n       >80%: {len(hi):2d} {hi}")
print(
    "\n== Leaving one bearing out: smallest H1 difference and largest p (one-sided sign-flip test, exact/MC) =="
)
for f in (4, 8, 16, 32):
    a, b_ = (
        {k: v for k, v in A[("median_guided", f)].items() if k[0] in ("OR", "IR")},
        {k: v for k, v in A[("kurtogram", f)].items() if k[0] in ("OR", "IR")},
    )
    worst = (1e9, None, None)
    for k in a:
        aa = {kk: vv for kk, vv in a.items() if kk != k}
        bb = {kk: vv for kk, vv in b_.items() if kk != k}
        o, p, n, u = signflip(aa, bb, mc=100000)
        if o < worst[0]:
            worst = (o, k[1], p)
    print(f"  {f:2d}x: smallest difference without {worst[1]}: {worst[0]:+.1f} (p={worst[2]:.2g})")
