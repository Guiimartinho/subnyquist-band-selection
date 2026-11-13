# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

tau = thresholds(95)
A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guided" else pred_blind(r))
dmg = lambda arm, f: {k: v for k, v in A[(arm, f)].items() if k[0] in ("OR", "IR")}
print(
    "== Guided - blind gain per factor (balanced accuracy points) and interaction with decimation =="
)
for arm, base in (
    ("median_guided", "kurtogram"),
    ("guided", "kurtogram"),
    ("median_guided", "median_kurtogram"),
    ("guided", "median_kurtogram"),
):
    g1 = {k: dmg(arm, 1)[k] - dmg(base, 1)[k] for k in dmg(arm, 1)}
    o1, p1, n1, u1 = signflip(dmg(arm, 1), dmg(base, 1))
    print(f"\n{arm} - {base}: 1x gain={o1:+.1f} (p1={p1:.2g}, {u1}/{n1})")
    for f in (4, 8, 16, 32):
        gf = {k: dmg(arm, f)[k] - dmg(base, f)[k] for k in dmg(arm, f)}
        keys = sorted(set(g1) & set(gf))
        # DiD: (gain at f) - (gain at 1x), per bearing
        a = {k: gf[k] for k in keys}
        b = {k: g1[k] for k in keys}
        o, p2, n, u = signflip(a, b, two_sided=True)
        _, p1s, _, _ = signflip(a, b, two_sided=False)
        lo, hi = ci_signflip({k: a[k] - b[k] for k in keys})
        og, _, _, _ = signflip(dmg(arm, f), dmg(base, f))
        print(
            f"   {f:2d}x: gain={og:+5.1f}  DiD(gain_f - gain_1x)={o:+5.1f}  CI95=[{lo:+.1f},{hi:+.1f}]  two-sided p={p2:.3f}  p(DiD>0)={p1s:.3f}  ({u}/{n} in favour)"
        )
print("\n== Each arm against itself at 1x (does decimation make it worse or better?) ==")
for arm in ARMS:
    s = []
    for f in (4, 8, 16, 32):
        o, p, n, u = signflip(dmg(arm, f), dmg(arm, 1), two_sided=True)
        s.append(f"{f}x {o:+5.1f} (p={p:.3f}, {u}/{n})")
    print(f"{arm:20s} " + " | ".join(s))
print(
    "\n== Contribution of the median within the guided arm: median_guided - guided (two-sided) =="
)
for f in FACTORS:
    o, p, n, u = signflip(dmg("median_guided", f), dmg("guided", f), two_sided=True)
    lo, hi = ci_signflip(
        {k: dmg("median_guided", f)[k] - dmg("guided", f)[k] for k in dmg("guided", f)}
    )
    print(
        f"  {f:2d}x: {o:+5.1f} CI95=[{lo:+.1f},{hi:+.1f}] p={p:.3f} ({u}/{n} in favour of the median)"
    )
print("\n== CI95 by inversion of the sign-flip test for H1/H2 ==")
for name, arm, base in (
    ("H1", "median_guided", "kurtogram"),
    ("H2", "median_guided", "median_kurtogram"),
    ("H1'", "guided", "kurtogram"),
):
    for f in FACTORS:
        d = {k: dmg(arm, f)[k] - dmg(base, f)[k] for k in dmg(arm, f)}
        lo, hi = ci_signflip(d)
        o, _, _, _ = signflip(dmg(arm, f), dmg(base, f))
        print(f"  {name:3s} {f:2d}x: {o:+5.1f}  CI95=[{lo:+.1f},{hi:+.1f}]")
