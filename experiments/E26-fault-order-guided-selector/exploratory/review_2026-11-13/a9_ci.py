# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

exec(
    open(sys.argv[1] + "/a8_percand.py", encoding="utf-8")
    .read()
    .split("for q in (97.5,):")[0]
    .split("from common import *")[1]
)
t = tau_pc(97.5)
A = acc_table(lambda r, sel: pred_pc(r, t) if sel == "guided" else pred_blind(r))
tau = thresholds(95)
A0 = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guided" else pred_blind(r))
for lab, AA in (
    ("single p95 threshold (pre-registered)", A0),
    ("per-candidate threshold p97.5 (post hoc)", A),
):
    print(f"== {lab}: CI95 (inverted sign-flip test) of [factor] - [1x], per arm ==")
    for arm in ("guided", "median_guided"):
        dm = lambda f: {k: v for k, v in AA[(arm, f)].items() if k[0] in ("OR", "IR")}
        s = []
        for f in (4, 8, 16, 32):
            d = {k: dm(f)[k] - dm(1)[k] for k in dm(1)}
            lo, hi = ci_signflip(d)
            s.append(
                f"{f}x-1x {100 * np.mean([np.mean([v for k, v in d.items() if k[0] == rc]) for rc in ('OR', 'IR')]):+5.1f} [{lo:+.1f},{hi:+.1f}]"
            )
        print(f"  {arm:15s} " + " | ".join(s))
print(
    "\n== Per-candidate threshold: H1 (median_guided - kurtogram) and guided at 1x against kurtogram at 4x =="
)
for f in FACTORS:
    a = {k: v for k, v in A[("median_guided", f)].items() if k[0] in ("OR", "IR")}
    b = {k: v for k, v in A[("kurtogram", f)].items() if k[0] in ("OR", "IR")}
    o, p, n, u = signflip(a, b, mc=100000)
    print(f"  {f:2d}x H1={o:+5.1f} p={p:.2g} ({u}/{n})")
