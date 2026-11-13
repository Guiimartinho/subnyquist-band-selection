# ruff: noqa
# Escrito pela revisão adversarial (reviewer-2) de 2026-11-13; ver o README do E26.
# Uso: uv run --no-project --quiet --with numpy python <script> <este diretório>
import sys

sys.path.insert(0, sys.argv[1])
from common import *
from collections import Counter

tau = thresholds(95)
print(
    "== Concordancia de banda: banda do cego == banda guiada da falha VERDADEIRA (danificados) =="
)
print(
    "   (onde as bandas coincidem, o escore guiado na banda do cego e o proprio escore guiado: a diferenca e so de regra)"
)
for clean, blind_arm, g_arm in (
    ("bruto", "kurtograma", "guiado"),
    ("mediana3", "mediana_kurtograma", "mediana_guiado"),
):
    for f in FACTORS:
        same = {"ok_g": 0, "ok_b": 0, "n": 0}
        diff = {"ok_g": 0, "ok_b": 0, "n": 0}
        for r in ROWS:
            if int(r["factor"]) != f or r["clean"] != clean or r["race"] not in ("OR", "IR"):
                continue
            c = RACE = "BPFO" if r["race"] == "OR" else "BPFI"
            g = pred_guided(r, tau) == r["race"]
            b = pred_blind(r) == r["race"]
            d = same if r["blind_band"] == r[f"band_{c}"] else diff
            d["n"] += 1
            d["ok_g"] += g
            d["ok_b"] += b
        print(
            f"  {clean:8s} {f:2d}x | mesma banda: {100 * same['n'] / (same['n'] + diff['n']):4.1f}% dos segm.; acerto guiado={100 * same['ok_g'] / max(1, same['n']):4.1f} cego={100 * same['ok_b'] / max(1, same['n']):4.1f} | banda diferente: guiado={100 * diff['ok_g'] / max(1, diff['n']):4.1f} cego={100 * diff['ok_b'] / max(1, diff['n']):4.1f}"
        )
print("\n== Ordens do cego em 'piso' (2,015 a 2,2): valores mais frequentes, bruto ==")
c = Counter()
h = Counter()
for r in ROWS:
    if r["clean"] == "bruto" and r["blind_order"] != "":
        e = float(r["blind_order"])
        if 2.0 <= e < 2.2:
            (c if r["race"] != "H" else h)[round(e, 2)] += 1
print("  danificados:", c.most_common(8))
print("  saos:", h.most_common(8))
print(
    "\n== Bandas escolhidas: cego (kurtograma bruto) e guiado (falha verdadeira), mais frequentes =="
)
for f in FACTORS:
    rr = [
        r
        for r in ROWS
        if int(r["factor"]) == f and r["clean"] == "bruto" and r["race"] in ("OR", "IR")
    ]
    cb = Counter(r["blind_band"] for r in rr)
    cg = Counter(r["band_BPFO" if r["race"] == "OR" else "band_BPFI"] for r in rr)
    hh = [r for r in ROWS if int(r["factor"]) == f and r["clean"] == "bruto" and r["race"] == "H"]
    ch = Counter(r["blind_band"] for r in hh)
    print(
        f"  {f:2d}x cego: {cb.most_common(3)}\n      guiado: {cg.most_common(3)}\n      cego nos saos: {ch.most_common(2)}"
    )
