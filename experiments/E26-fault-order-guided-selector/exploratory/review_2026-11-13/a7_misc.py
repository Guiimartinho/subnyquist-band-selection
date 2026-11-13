# ruff: noqa
# Escrito pela revisão adversarial (reviewer-2) de 2026-11-13; ver o README do E26.
# Uso: uv run --no-project --quiet --with numpy python <script> <este diretório>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

tau = thresholds(95)
print("== Por ajuste: acuracia balanceada mediana_guiado / kurtograma (limiar p95 comum) ==")
for st in ("N15_M07_F10", "N15_M01_F10", "N15_M07_F04"):
    s = []
    for f in FACTORS:
        acc = defaultdict(lambda: defaultdict(list))
        for r in ROWS:
            if r["setting"] != st or int(r["factor"]) != f or r["race"] not in ("OR", "IR"):
                continue
            if r["clean"] == "mediana3":
                acc["mg"][(r["race"], r["bearing"])].append(pred_guided(r, tau) == r["race"])
            if r["clean"] == "bruto":
                acc["k"][(r["race"], r["bearing"])].append(pred_blind(r) == r["race"])
        mg = balanced({k: np.mean(v) for k, v in acc["mg"].items()})
        k = balanced({k: np.mean(v) for k, v in acc["k"].items()})
        s.append(f"{f}x {mg:4.1f}/{k:4.1f}")
    print(f"  {st}: " + " | ".join(s))
print("\n== Nos saos: qual janela da o maximo (BPFO x BPFI) e mediana de cada escore ==")
for cl in ("bruto", "mediana3"):
    for f in FACTORS:
        hh = [r for r in ROWS if r["race"] == "H" and int(r["factor"]) == f and r["clean"] == cl]
        wO = np.mean([sc(r, "BPFO") > sc(r, "BPFI") for r in hh])
        print(
            f"  {cl:8s} {f:2d}x: max na BPFO em {100 * wO:4.1f}% | mediana BPFO={np.median([sc(r, 'BPFO') for r in hh]):5.2f} BPFI={np.median([sc(r, 'BPFI') for r in hh]):5.2f} | p95 BPFO={np.percentile([sc(r, 'BPFO') for r in hh], 95):5.2f} BPFI={np.percentile([sc(r, 'BPFI') for r in hh], 95):5.2f}"
        )
print("\n== Numero de bins por janela (grade de 0,01 ordem, ~100 rev em 4 s) ==")
sys.path.insert(
    0, r"<repository root>/experiments/E26-fault-order-guided-selector"
)
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
            f"  {c} k={k}: segmentos={[(round(s, 3), round(e, 3)) for s, e in segs]}  bins={nb}  distancia da linha verdadeira ao inteiro mais proximo = {abs(k * ALLMED['OR' if c == 'BPFO' else 'IR'] - round(k * ALLMED['OR' if c == 'BPFO' else 'IR'])):.3f}"
        )


# Resposta da janela de Hann a uma raia inteira fora do centro do bin
def hann_resp(delta):
    if abs(delta) < 1e-9:
        return 1.0
    if abs(abs(delta) - 1) < 1e-9:
        return 0.5
    return abs(np.sinc(delta) / (1 - delta**2))


print(
    "\n== Vazamento de Hann de uma raia inteira na 1a/2a bin fora da exclusao (amplitude relativa) =="
)
for off in (0.0, 0.25, 0.5):
    print(
        f"  raia a {off:.2f} bin do centro: bin +1,5={hann_resp(1.5 - off):.3f}  +2={hann_resp(2 - off):.3f}  +2,5={hann_resp(2.5 - off):.3f}  +3={hann_resp(3 - off):.4f}"
    )
