# ruff: noqa
# Escrito pela revisão adversarial (reviewer-2) de 2026-11-13; ver o README do E26.
# Uso: uv run --no-project --quiet --with numpy python <script> <este diretório>
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
A = acc_table(lambda r, sel: pred_pc(r, t) if sel == "guiado" else pred_blind(r))
tau = thresholds(95)
A0 = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r))
for lab, AA in (
    ("limiar unico p95 (pre-registrado)", A0),
    ("limiar por candidato p97,5 (post hoc)", A),
):
    print(f"== {lab}: IC95 (troca de sinal invertida) de [fator] - [1x], por braco ==")
    for arm in ("guiado", "mediana_guiado"):
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
    "\n== Limiar por candidato: H1 (mediana_guiado - kurtograma) e guiado em 1x contra kurtograma em 4x =="
)
for f in FACTORS:
    a = {k: v for k, v in A[("mediana_guiado", f)].items() if k[0] in ("OR", "IR")}
    b = {k: v for k, v in A[("kurtograma", f)].items() if k[0] in ("OR", "IR")}
    o, p, n, u = signflip(a, b, mc=100000)
    print(f"  {f:2d}x H1={o:+5.1f} p={p:.2g} ({u}/{n})")
