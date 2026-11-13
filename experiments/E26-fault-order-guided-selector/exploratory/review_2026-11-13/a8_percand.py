# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *


# Per-candidate threshold (post hoc): percentile q of each score on the healthy; decision: candidates above their own threshold,
# the one with the largest score/threshold ratio. q chosen for a global FA of ~5% (checked below, LOO on the healthy).
def tau_pc(q):
    t = {}
    for f in FACTORS:
        for cl in ("raw", "median3"):
            for c in ("BPFO", "BPFI"):
                hs = defaultdict(list)
                for r in ROWS:
                    if r["race"] == "H" and int(r["factor"]) == f and r["clean"] == cl:
                        hs[r["bearing"]].append(sc(r, c))
                t[(f, cl, c, None)] = float(np.percentile([s for v in hs.values() for s in v], q))
                for h in HEALTHY:
                    t[(f, cl, c, h)] = float(
                        np.percentile([s for b, v in hs.items() if b != h for s in v], q)
                    )
    return t


def pred_pc(r, t):
    f, cl, race, b = int(r["factor"]), r["clean"], r["race"], r["bearing"]
    key = b if race == "H" else None
    z = {c: sc(r, c) / t[(f, cl, c, key)] for c in ("BPFO", "BPFI")}
    top = max(z, key=z.get)
    return ("OR" if top == "BPFO" else "IR") if z[top] >= 1 else "none"


for q in (97.5,):
    t = tau_pc(q)
    A = acc_table(lambda r, sel: pred_pc(r, t) if sel == "guided" else pred_blind(r))
    print(f"== Per-candidate threshold, percentile {q} of each score (post hoc) ==")
    for arm in ("guided", "median_guided"):
        s = []
        for f in FACTORS:
            fa = A[(arm, f)]
            byr = {
                rc: 100 * np.mean([v for k, v in fa.items() if k[0] == rc])
                for rc in ("OR", "IR", "H")
            }
            s.append(
                f"{f}x bal={balanced(fa):4.1f} (OR {byr['OR']:4.1f} IR {byr['IR']:4.1f}) spec={byr['H']:4.1f}"
            )
        print(f"  {arm:15s} " + " | ".join(s))
    for arm in ("guided", "median_guided"):
        dm = lambda f: {k: v for k, v in A[(arm, f)].items() if k[0] in ("OR", "IR")}
        s = []
        for f in (4, 8, 16, 32):
            o, p, n, u = signflip(dm(f), dm(1), two_sided=True)
            s.append(f"{f}x-1x {o:+5.1f} (p={p:.3f})")
        print(f"  {arm:15s} decimation against 1x: " + " | ".join(s))
tau = thresholds(95)
A0 = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guided" else pred_blind(r))
print(
    "\n  (pre-registered, single p95 threshold on the max) guided: "
    + " | ".join(
        f"{f}x OR {100 * np.mean([v for k, v in A0[('guided', f)].items() if k[0] == 'OR']):4.1f} IR {100 * np.mean([v for k, v in A0[('guided', f)].items() if k[0] == 'IR']):4.1f}"
        for f in FACTORS
    )
)
