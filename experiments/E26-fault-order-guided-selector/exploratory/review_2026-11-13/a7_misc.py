# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

tau = thresholds(95)
print("== Per setting: balanced accuracy median_guided / kurtogram (common p95 threshold) ==")
for st in ("N15_M07_F10", "N15_M01_F10", "N15_M07_F04"):
    s = []
    for f in FACTORS:
        acc = defaultdict(lambda: defaultdict(list))
        for r in ROWS:
            if r["setting"] != st or int(r["factor"]) != f or r["race"] not in ("OR", "IR"):
                continue
            if r["clean"] == "median3":
                acc["mg"][(r["race"], r["bearing"])].append(pred_guided(r, tau) == r["race"])
            if r["clean"] == "raw":
                acc["k"][(r["race"], r["bearing"])].append(pred_blind(r) == r["race"])
        mg = balanced({k: np.mean(v) for k, v in acc["mg"].items()})
        k = balanced({k: np.mean(v) for k, v in acc["k"].items()})
        s.append(f"{f}x {mg:4.1f}/{k:4.1f}")
    print(f"  {st}: " + " | ".join(s))
print(
    "\n== On the healthy: which window gives the maximum (BPFO x BPFI) and median of each score =="
)
for cl in ("raw", "median3"):
    for f in FACTORS:
        hh = [r for r in ROWS if r["race"] == "H" and int(r["factor"]) == f and r["clean"] == cl]
        wO = np.mean([sc(r, "BPFO") > sc(r, "BPFI") for r in hh])
        print(
            f"  {cl:8s} {f:2d}x: max at BPFO in {100 * wO:4.1f}% | median BPFO={np.median([sc(r, 'BPFO') for r in hh]):5.2f} BPFI={np.median([sc(r, 'BPFI') for r in hh]):5.2f} | p95 BPFO={np.percentile([sc(r, 'BPFO') for r in hh], 95):5.2f} BPFI={np.percentile([sc(r, 'BPFI') for r in hh], 95):5.2f}"
        )
print("\n== Number of bins per window (grid of 0.01 order, ~100 rev in 4 s) ==")
import math


def windows(order, k, frac, exclusion):
    lo, hi = k * order * (1 - frac), k * order * (1 + frac)
    segs = [(lo, hi)]
    for n in range(int(math.floor(lo)), int(math.ceil(hi)) + 1):
        a, b = n - exclusion, n + exclusion
        new = []
        for s, e in segs:
            if b <= s or a >= e:
                new.append((s, e))
                continue
            if s < a:
                new.append((s, a))
            if b < e:
                new.append((b, e))
        segs = new
    return segs


grid = np.arange(0, 32, 0.01)
for c, o in ORD.items():
    for k in (1, 2):
        segs = windows(o, k, 0.02, 0.015)
        nb = sum(((grid >= s) & (grid <= e)).sum() for s, e in segs)
        edge = [round(x, 3) for s, e in segs for x in (s, e)]
        print(
            f"  {c} k={k}: segments={[(round(s, 3), round(e, 3)) for s, e in segs]}  bins={nb}  distance from the true line to the nearest integer = {abs(k * ALLMED['OR' if c == 'BPFO' else 'IR'] - round(k * ALLMED['OR' if c == 'BPFO' else 'IR'])):.3f}"
        )


# Response of the Hann window to an integer line off the bin centre
def hann_resp(delta):
    if abs(delta) < 1e-9:
        return 1.0
    if abs(abs(delta) - 1) < 1e-9:
        return 0.5
    return abs(np.sinc(delta) / (1 - delta**2))


print(
    "\n== Hann leakage of an integer line into the 1st/2nd bin outside the exclusion (relative amplitude) =="
)
for off in (0.0, 0.25, 0.5):
    print(
        f"  line at {off:.2f} bin from the centre: bin +1.5={hann_resp(1.5 - off):.3f}  +2={hann_resp(2 - off):.3f}  +2.5={hann_resp(2.5 - off):.3f}  +3={hann_resp(3 - off):.4f}"
    )
