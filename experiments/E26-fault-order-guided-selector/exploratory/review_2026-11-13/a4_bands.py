# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *
from collections import Counter

tau = thresholds(95)
print("== Band agreement: blind band == guided band of the TRUE fault (damaged) ==")
print(
    "   (where the bands coincide, the guided score in the blind band is the guided score itself: the difference is only the rule)"
)
for clean, blind_arm, g_arm in (
    ("raw", "kurtogram", "guided"),
    ("median3", "median_kurtogram", "median_guided"),
):
    for f in FACTORS:
        same = {"ok_g": 0, "ok_b": 0, "n": 0}
        diff = {"ok_g": 0, "ok_b": 0, "n": 0}
        for r in ROWS:
            if int(r["factor"]) != f or r["clean"] != clean or r["race"] not in ("OR", "IR"):
                continue
            c = RACE = "BPFO" if r["race"] == "OR" else "BPFI"
            g = pred_guided(r, tau) == r["race"]
            b = pred_blind(r) == r["race"]
            d = same if r["blind_band"] == r[f"band_{c}"] else diff
            d["n"] += 1
            d["ok_g"] += g
            d["ok_b"] += b
        print(
            f"  {clean:8s} {f:2d}x | same band: {100 * same['n'] / (same['n'] + diff['n']):4.1f}% of segm.; hit rate guided={100 * same['ok_g'] / max(1, same['n']):4.1f} blind={100 * same['ok_b'] / max(1, same['n']):4.1f} | different band: guided={100 * diff['ok_g'] / max(1, diff['n']):4.1f} blind={100 * diff['ok_b'] / max(1, diff['n']):4.1f}"
        )
print("\n== Blind orders at the 'floor' (2.015 to 2.2): most frequent values, raw ==")
c = Counter()
h = Counter()
for r in ROWS:
    if r["clean"] == "raw" and r["blind_order"] != "":
        e = float(r["blind_order"])
        if 2.0 <= e < 2.2:
            (c if r["race"] != "H" else h)[round(e, 2)] += 1
print("  damaged:", c.most_common(8))
print("  healthy:", h.most_common(8))
print("\n== Chosen bands: blind (raw kurtogram) and guided (true fault), most frequent ==")
for f in FACTORS:
    rr = [
        r
        for r in ROWS
        if int(r["factor"]) == f and r["clean"] == "raw" and r["race"] in ("OR", "IR")
    ]
    cb = Counter(r["blind_band"] for r in rr)
    cg = Counter(r["band_BPFO" if r["race"] == "OR" else "band_BPFI"] for r in rr)
    hh = [r for r in ROWS if int(r["factor"]) == f and r["clean"] == "raw" and r["race"] == "H"]
    ch = Counter(r["blind_band"] for r in hh)
    print(
        f"  {f:2d}x blind: {cb.most_common(3)}\n      guided: {cg.most_common(3)}\n      blind on healthy: {ch.most_common(2)}"
    )
