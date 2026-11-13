# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

tau = thresholds(95)


def cat(est, race, b):
    if not np.isfinite(est):
        return "nan"
    if race in ("OR", "IR"):
        own = truth(b, race, race)
        o = own
        if abs(est - o) <= TOL:
            return "fault_1x"
        if abs(est - 2 * o) <= 2 * TOL:
            return "fault_2x"
        if abs(est - 3 * o) <= 3 * TOL:
            return "fault_3x"
    if abs(est - round(est)) <= 0.015:
        return f"integer"
    if est < 2.2:
        return "floor(<2.2)"
    for c, o in ORD.items():
        for k in (1, 2, 3):
            if abs(est - k * o) <= 0.02 * k * o:
                return f"window_{c}x{k}"
    return "other"


print("== Where the blind argmax goes (fraction of segments) ==")
for clean in ("raw", "median3"):
    for race in ("OR", "IR", "H"):
        for f in FACTORS:
            rr = [
                r
                for r in ROWS
                if int(r["factor"]) == f and r["clean"] == clean and r["race"] == race
            ]
            cs = defaultdict(int)
            for r in rr:
                est = float(r["blind_order"]) if r["blind_order"] != "" else np.nan
                cs[cat(est, race, r["bearing"])] += 1
            n = len(rr)
            top = sorted(cs.items(), key=lambda kv: -kv[1])
            print(
                f"{clean:8s} {race} {f:2d}x n={n:4d} | "
                + "  ".join(f"{k}={100 * v / n:.1f}" for k, v in top[:7])
            )
print("\n== Most frequent integer orders in the blind argmax (raw, all factors, damaged) ==")
from collections import Counter

c = Counter()
for r in ROWS:
    if r["clean"] == "raw" and r["race"] in ("OR", "IR") and r["blind_order"] != "":
        e = float(r["blind_order"])
        if abs(e - round(e)) <= 0.015:
            c[int(round(e))] += 1
print(c.most_common(10))
print(
    "\n== Blind with the SAME acceptance as the guided arm: |est - k*o| <= 2% of k*o, k=1,2, with integers +-0.015 excluded =="
)


def in_guided_window(est, o):
    for k in (1, 2):
        if abs(est - k * o) <= 0.02 * k * o and abs(est - round(est)) > 0.015:
            return True
    return False


def pred_blind_wide(r, rule):
    est = float(r["blind_order"]) if r["blind_order"] != "" else np.nan
    if not np.isfinite(est):
        return "none"
    hits = []
    for rc, c in (("OR", "BPFO"), ("IR", "BPFI")):
        if rule == "geo":
            o = ORD[c]
            if in_guided_window(est, o):
                hits.append(rc)
        elif (
            rule == "nested12"
        ):  # tolerance 0.025 at the 1st and 0.05 at the 2nd harmonic around the ruler
            o = truth(r["bearing"], r["race"], rc)
            if abs(est - o) <= TOL or abs(est - 2 * o) <= 2 * TOL:
                hits.append(rc)
    return hits[0] if len(hits) == 1 else ("none" if not hits else hits[0])


for rule in ("nested12", "geo"):
    A = acc_table(
        lambda r, sel: pred_guided(r, tau) if sel == "guided" else pred_blind_wide(r, rule)
    )
    for arm in ("kurtogram", "median_kurtogram"):
        s = []
        for f in FACTORS:
            bal = balanced(A[(arm, f)])
            spec = balanced(A[(arm, f)], races=("H",))
            s.append(f"{f}x {bal:5.1f}/{spec:5.1f}")
        print(f"  rule={rule:8s} {arm:20s} bal/spec: " + " | ".join(s))
print("\n== Tolerance of the blind arm (nested ruler, 1st harmonic only): bal/spec ==")
for tol in (0.01, 0.025, 0.05, 0.10):
    A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guided" else pred_blind(r, tol=tol))
    s = []
    for f in FACTORS:
        s.append(
            f"{f}x {balanced(A[('kurtogram', f)]):5.1f}/{balanced(A[('kurtogram', f)], races=('H',)):5.1f}"
        )
    print(f"  tol={tol:.3f} kurtogram: " + " | ".join(s))
print("\n== Nominal ruler (geometry) instead of the nested one, tol 0.025 ==")
A = acc_table(
    lambda r, sel: pred_guided(r, tau) if sel == "guided" else pred_blind(r, rule="nominal")
)
for arm in ("kurtogram", "median_kurtogram"):
    print(f"  {arm:20s} " + " | ".join(f"{f}x {balanced(A[(arm, f)]):5.1f}" for f in FACTORS))
