# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *
from math import comb

tau = thresholds(95)
A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r))
print(
    "== Bearing level: identified if > 50% of the segments are right; healthy with alarm if < 50% of 'none' =="
)
for f in FACTORS:
    s = []
    for arm in ARMS:
        d = A[(arm, f)]
        idd = sum(1 for k, v in d.items() if k[0] in ("OR", "IR") and v > 0.5)
        fa = sum(1 for k, v in d.items() if k[0] == "H" and v < 0.5)
        s.append(f"{arm} {idd}/23 (FA {fa}/6)")
    # exact McNemar: mediana_guiado x kurtograma
    g = A[("mediana_guiado", f)]
    k = A[("kurtograma", f)]
    b10 = sum(1 for kk in g if kk[0] in ("OR", "IR") and g[kk] > 0.5 and k[kk] <= 0.5)
    b01 = sum(1 for kk in g if kk[0] in ("OR", "IR") and g[kk] <= 0.5 and k[kk] > 0.5)
    n = b10 + b01
    p = sum(comb(n, i) for i in range(b10, n + 1)) / 2**n if n else 1.0
    print(
        f"  {f:2d}x: "
        + " | ".join(s)
        + f" || McNemar mg x k: {b10} guided only, {b01} blind only, one-sided p={p:.4f}"
    )
