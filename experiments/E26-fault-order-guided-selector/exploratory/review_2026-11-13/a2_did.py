# ruff: noqa
# Escrito pela revisão adversarial (reviewer-2) de 2026-11-13; ver o README do E26.
# Uso: uv run --no-project --quiet --with numpy python <script> <este diretório>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

tau = thresholds(95)
A = acc_table(lambda r, sel: pred_guided(r, tau) if sel == "guiado" else pred_blind(r))
dmg = lambda arm, f: {k: v for k, v in A[(arm, f)].items() if k[0] in ("OR", "IR")}
print(
    "== Ganho guiado - cego por fator (pontos de acuracia balanceada) e interacao com a decimacao =="
)
for arm, base in (
    ("mediana_guiado", "kurtograma"),
    ("guiado", "kurtograma"),
    ("mediana_guiado", "mediana_kurtograma"),
    ("guiado", "mediana_kurtograma"),
):
    g1 = {k: dmg(arm, 1)[k] - dmg(base, 1)[k] for k in dmg(arm, 1)}
    o1, p1, n1, u1 = signflip(dmg(arm, 1), dmg(base, 1))
    print(f"\n{arm} - {base}: 1x ganho={o1:+.1f} (p1={p1:.2g}, {u1}/{n1})")
    for f in (4, 8, 16, 32):
        gf = {k: dmg(arm, f)[k] - dmg(base, f)[k] for k in dmg(arm, f)}
        keys = sorted(set(g1) & set(gf))
        # DiD: (ganho em f) - (ganho em 1x), por rolamento
        a = {k: gf[k] for k in keys}
        b = {k: g1[k] for k in keys}
        o, p2, n, u = signflip(a, b, two_sided=True)
        _, p1s, _, _ = signflip(a, b, two_sided=False)
        lo, hi = ci_signflip({k: a[k] - b[k] for k in keys})
        og, _, _, _ = signflip(dmg(arm, f), dmg(base, f))
        print(
            f"   {f:2d}x: ganho={og:+5.1f}  DiD(ganho_f - ganho_1x)={o:+5.1f}  IC95=[{lo:+.1f},{hi:+.1f}]  p bilateral={p2:.3f}  p(DiD>0)={p1s:.3f}  ({u}/{n} a favor)"
        )
print("\n== Cada braco contra ele mesmo em 1x (a decimacao piora ou melhora?) ==")
for arm in ARMS:
    s = []
    for f in (4, 8, 16, 32):
        o, p, n, u = signflip(dmg(arm, f), dmg(arm, 1), two_sided=True)
        s.append(f"{f}x {o:+5.1f} (p={p:.3f}, {u}/{n})")
    print(f"{arm:20s} " + " | ".join(s))
print("\n== Contribuicao da mediana dentro do guiado: mediana_guiado - guiado (bilateral) ==")
for f in FACTORS:
    o, p, n, u = signflip(dmg("mediana_guiado", f), dmg("guiado", f), two_sided=True)
    lo, hi = ci_signflip(
        {k: dmg("mediana_guiado", f)[k] - dmg("guiado", f)[k] for k in dmg("guiado", f)}
    )
    print(f"  {f:2d}x: {o:+5.1f} IC95=[{lo:+.1f},{hi:+.1f}] p={p:.3f} ({u}/{n} a favor da mediana)")
print("\n== IC95 por inversao da troca de sinal para H1/H2 ==")
for name, arm, base in (
    ("H1", "mediana_guiado", "kurtograma"),
    ("H2", "mediana_guiado", "mediana_kurtograma"),
    ("H1'", "guiado", "kurtograma"),
):
    for f in FACTORS:
        d = {k: dmg(arm, f)[k] - dmg(base, f)[k] for k in dmg(arm, f)}
        lo, hi = ci_signflip(d)
        o, _, _, _ = signflip(dmg(arm, f), dmg(base, f))
        print(f"  {name:3s} {f:2d}x: {o:+5.1f}  IC95=[{lo:+.1f},{hi:+.1f}]")
