# ruff: noqa
# Escrito pela revisão adversarial (reviewer-2) de 2026-11-13; ver o README do E26.
# Uso: uv run --no-project --quiet --with numpy python <script> <este diretório>
import sys

sys.path.insert(0, sys.argv[1])
from common import *


# Limiar por candidato (post hoc): percentil q de cada escore nos saos; decisao: candidatos acima do proprio limiar,
# o de maior razao escore/limiar. q escolhido para FA global ~5% (verificado abaixo, LOO nos saos).
def tau_pc(q):
    t = {}
    for f in FACTORS:
        for cl in ("bruto", "mediana3"):
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
    A = acc_table(lambda r, sel: pred_pc(r, t) if sel == "guiado" else pred_blind(r))
    print(f"== Limiar por candidato, percentil {q} de cada escore (post hoc) ==")
    for arm in ("guiado", "mediana_guiado"):
        s = []
        for f in FACTORS:
            fa = A[(arm, f)]
            byr = {
                rc: 100 * np.mean([v for k, v in fa.items() if k[0] == rc])
                for rc in ("OR", "IR", "H")
            }
            s.append(
                f"{f}x bal={balanced(fa):4.1f} (OR {byr['OR']:4.1f} IR {byr['IR']:4.1f}) espec={byr['H']:4.1f}"
            )
        print(f"  {arm:15s} " + " | ".join(s))
    for arm in ("guiado", "mediana_guiado"):
        dm = lambda f: {k: v for k, v in A[(arm, f)].items() if k[0] in ("OR", "IR")}
        s = []
        for f in (4, 8, 16, 32):
            o, p, n, u = signflip(dm(f), dm(1), two_sided=True)
            s.append(f"{f}x-1x {o:+5.1f} (p={p:.3f})")
        print(f"  {arm:15s} decimacao contra 1x: " + " | ".join(s))
tau = thresholds(95)
A0 = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r))
print(
    "\n  (pre-registrado, limiar unico p95 do max) guiado: "
    + " | ".join(
        f"{f}x OR {100 * np.mean([v for k, v in A0[('guiado', f)].items() if k[0] == 'OR']):4.1f} IR {100 * np.mean([v for k, v in A0[('guiado', f)].items() if k[0] == 'IR']):4.1f}"
        for f in FACTORS
    )
)
