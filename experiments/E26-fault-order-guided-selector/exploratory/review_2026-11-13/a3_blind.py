# ruff: noqa
# Escrito pela revisão adversarial (reviewer-2) de 2026-11-13; ver o README do E26.
# Uso: uv run --no-project --quiet --with numpy python <script> <este diretório>
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
            return "falha_1x"
        if abs(est - 2 * o) <= 2 * TOL:
            return "falha_2x"
        if abs(est - 3 * o) <= 3 * TOL:
            return "falha_3x"
    if abs(est - round(est)) <= 0.015:
        return f"inteira"
    if est < 2.2:
        return "piso(<2,2)"
    for c, o in ORD.items():
        for k in (1, 2, 3):
            if abs(est - k * o) <= 0.02 * k * o:
                return f"janela_{c}x{k}"
    return "outra"


print("== Para onde vai o argmax do cego (fracao de segmentos) ==")
for clean in ("bruto", "mediana3"):
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
print(
    "\n== Ordens inteiras mais frequentes no argmax do cego (bruto, todos os fatores, danificados) =="
)
from collections import Counter

c = Counter()
for r in ROWS:
    if r["clean"] == "bruto" and r["race"] in ("OR", "IR") and r["blind_order"] != "":
        e = float(r["blind_order"])
        if abs(e - round(e)) <= 0.015:
            c[int(round(e))] += 1
print(c.most_common(10))
print(
    "\n== Cego com a MESMA aceitacao do guiado: |est - k*o| <= 2% de k*o, k=1,2, com inteiras +-0,015 excluidas =="
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
        elif rule == "nested12":  # tolerancia 0,025 no 1o e 0,05 no 2o harmonico em torno da regua
            o = truth(r["bearing"], r["race"], rc)
            if abs(est - o) <= TOL or abs(est - 2 * o) <= 2 * TOL:
                hits.append(rc)
    return hits[0] if len(hits) == 1 else ("none" if not hits else hits[0])


for rule in ("nested12", "geo"):
    A = acc_table(
        lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind_wide(r, rule)
    )
    for arm in ("kurtograma", "mediana_kurtograma"):
        s = []
        for f in FACTORS:
            bal = balanced(A[(arm, f)])
            spec = balanced(A[(arm, f)], races=("H",))
            s.append(f"{f}x {bal:5.1f}/{spec:5.1f}")
        print(f"  regra={rule:8s} {arm:20s} bal/espec: " + " | ".join(s))
print("\n== Tolerancia do cego (regua aninhada, so 1o harmonico): bal/espec ==")
for tol in (0.01, 0.025, 0.05, 0.10):
    A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r, tol=tol))
    s = []
    for f in FACTORS:
        s.append(
            f"{f}x {balanced(A[('kurtograma', f)]):5.1f}/{balanced(A[('kurtograma', f)], races=('H',)):5.1f}"
        )
    print(f"  tol={tol:.3f} kurtograma: " + " | ".join(s))
print("\n== Regua nominal (geometria) em vez da aninhada, tol 0,025 ==")
A = acc_table(
    lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r, rule="nominal")
)
for arm in ("kurtograma", "mediana_kurtograma"):
    print(f"  {arm:20s} " + " | ".join(f"{f}x {balanced(A[(arm, f)]):5.1f}" for f in FACTORS))
