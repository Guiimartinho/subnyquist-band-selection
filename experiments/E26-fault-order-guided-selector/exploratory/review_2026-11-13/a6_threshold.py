# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

print(
    "== Sensitivity to the percentile of the healthy threshold (LOO on the healthy): bal / spec ; H1 = mediana_guiado - kurtograma =="
)
base = acc_table(lambda r, sel: pred_blind(r))
for q in (90, 95, 97.5, 99, 100):
    tau = thresholds(q)
    A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r))
    s = []
    for f in FACTORS:
        a = {k: v for k, v in A[("mediana_guiado", f)].items() if k[0] in ("OR", "IR")}
        b = {k: v for k, v in A[("kurtograma", f)].items() if k[0] in ("OR", "IR")}
        o, p, n, u = signflip(a, b, mc=100000)
        s.append(
            f"{f}x {balanced(A[('mediana_guiado', f)]):4.1f}/{balanced(A[('mediana_guiado', f)], races=('H',)):5.1f} H1={o:+5.1f}"
        )
    print(f"  p{q:<5} " + " | ".join(s))
print("\n== Pooled specificity with CI (t, 5 df, over the 6 healthy), p95 ==")
tau = thresholds(95)
A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r))
t5 = 2.5706
for arm in ("guiado", "mediana_guiado"):
    s = []
    for f in FACTORS:
        v = np.array([A[(arm, f)][("H", h)] for h in HEALTHY]) * 100
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        s.append(f"{f}x {m:4.1f} [{m - t5 * se:4.1f}, {m + t5 * se:5.1f}] min={v.min():4.1f}")
    print(f"  {arm:15s} " + " | ".join(s))
print(
    "\n== Healthy thresholds per factor (p95) and median score of the damaged on the right candidate =="
)
for cl in ("bruto", "mediana3"):
    s = []
    for f in FACTORS:
        rr = [r for r in ROWS if int(r["factor"]) == f and r["clean"] == cl]
        o = np.median([sc(r, "BPFO") for r in rr if r["race"] == "OR"])
        i = np.median([sc(r, "BPFI") for r in rr if r["race"] == "IR"])
        hh = np.median([max(sc(r, "BPFO"), sc(r, "BPFI")) for r in rr if r["race"] == "H"])
        s.append(
            f"{f}x tau={tau[(f, cl, None)]:5.2f} medH={hh:4.2f} medOR(BPFO)={o:5.2f} medIR(BPFI)={i:4.2f}"
        )
    print(f"  {cl:8s} " + " | ".join(s))
