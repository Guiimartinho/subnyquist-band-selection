# ruff: noqa
# Escrito pela revisão adversarial (reviewer-2) de 2026-11-13; ver o README do E26.
# Uso: uv run --no-project --quiet --with numpy python <script> <este diretório>
import sys

sys.path.insert(0, sys.argv[1])
from common import *
from math import comb

tau = thresholds(95)
A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r))
print(
    "== Nivel de rolamento: identificado se > 50% dos segmentos certos; sao com alarme se < 50% de 'nenhuma' =="
)
for f in FACTORS:
    s = []
    for arm in ARMS:
        d = A[(arm, f)]
        idd = sum(1 for k, v in d.items() if k[0] in ("OR", "IR") and v > 0.5)
        fa = sum(1 for k, v in d.items() if k[0] == "H" and v < 0.5)
        s.append(f"{arm} {idd}/23 (FA {fa}/6)")
    # McNemar exato: mediana_guiado x kurtograma
    g = A[("mediana_guiado", f)]
    k = A[("kurtograma", f)]
    b10 = sum(1 for kk in g if kk[0] in ("OR", "IR") and g[kk] > 0.5 and k[kk] <= 0.5)
    b01 = sum(1 for kk in g if kk[0] in ("OR", "IR") and g[kk] <= 0.5 and k[kk] > 0.5)
    n = b10 + b01
    p = sum(comb(n, i) for i in range(b10, n + 1)) / 2**n if n else 1.0
    print(
        f"  {f:2d}x: "
        + " | ".join(s)
        + f" || McNemar mg x k: {b10} so guiado, {b01} so cego, p unilateral={p:.4f}"
    )
