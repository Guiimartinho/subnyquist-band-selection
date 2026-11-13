# ruff: noqa
# Escrito pela revisão adversarial (reviewer-2) de 2026-11-13; ver o README do E26.
# Uso: uv run --no-project --quiet --with numpy python <script> <este diretório>
import sys

sys.path.insert(0, sys.argv[1])
from common import *

tau = thresholds(95)
print(
    "BPFO,BPFI geom:",
    round(BPFO, 4),
    round(BPFI, 4),
    "| allmed:",
    {k: round(v, 4) for k, v in ALLMED.items()},
)
print(
    "n valid calibs:",
    {r: sum(1 for c in VALID if c["race"] == r) for r in ("OR", "IR")},
    "bearings w/ valid:",
    sorted({c["bearing"] for c in VALID}),
)
print("\n== Matriz de confusao (fracao de segmentos), linhas = verdade ==")
for arm, (cl, sel) in ARMS.items():
    for f in FACTORS:
        cnt = defaultdict(lambda: defaultdict(int))
        for r in ROWS:
            if int(r["factor"]) != f or r["clean"] != cl:
                continue
            p = pred_guided(r, tau) if sel == "guiado" else pred_blind(r)
            cnt[r["race"]][p] += 1
        s = []
        for race in ("OR", "IR", "H"):
            n = sum(cnt[race].values())
            s.append(
                f"{race}: "
                + " ".join(f"{p}={100 * cnt[race][p] / n:4.1f}" for p in ("OR", "IR", "none"))
            )
        print(f"{arm:20s} {f:2d}x | " + " | ".join(s))
print("\n== Decomposicao guiado: P(deteccao) e P(pista certa | detectou), por pista (segmentos) ==")
for arm in ("guiado", "mediana_guiado"):
    cl = ARMS[arm][0]
    for f in FACTORS:
        out = []
        for race in ("OR", "IR"):
            rr = [
                r for r in ROWS if int(r["factor"]) == f and r["clean"] == cl and r["race"] == race
            ]
            p = [pred_guided(r, tau) for r in rr]
            det = [x for x in p if x != "none"]
            pc = sum(1 for x in det if x == race) / len(det) if det else float("nan")
            out.append(f"{race}: det={100 * len(det) / len(p):4.1f} certo|det={100 * pc:4.1f}")
        print(f"{arm:15s} {f:2d}x | " + " | ".join(out))
print("\n== Excedencia do limiar pelo candidato ERRADO (sem argmax) e pelo certo ==")
for arm in ("guiado", "mediana_guiado"):
    cl = ARMS[arm][0]
    for f in FACTORS:
        out = []
        for race, right, wrong in (("OR", "BPFO", "BPFI"), ("IR", "BPFI", "BPFO")):
            rr = [
                r for r in ROWS if int(r["factor"]) == f and r["clean"] == cl and r["race"] == race
            ]
            th = tau[(f, cl, None)]
            out.append(
                f"{race}: certo>={100 * np.mean([sc(r, right) >= th for r in rr]):4.1f} errado>={100 * np.mean([sc(r, wrong) >= th for r in rr]):4.1f}"
            )
        hh = [r for r in ROWS if int(r["factor"]) == f and r["clean"] == cl and r["race"] == "H"]
        hO = np.mean([sc(r, "BPFO") >= tau[(f, cl, r["bearing"])] for r in hh])
        hI = np.mean([sc(r, "BPFI") >= tau[(f, cl, r["bearing"])] for r in hh])
        print(
            f"{arm:15s} {f:2d}x | "
            + " | ".join(out)
            + f" | H: BPFO>={100 * hO:4.1f} BPFI>={100 * hI:4.1f}"
        )


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    rk = allv.argsort().argsort().astype(float) + 1
    # ranks com empates medios
    from collections import Counter

    order = np.argsort(allv)
    ranks = np.empty_like(allv, dtype=float)
    i = 0
    sv = allv[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


print(
    "\n== AUC (segmentos): deteccao = max(escores) danificado x sao; pista = log(BPFO/BPFI) OR x IR =="
)
for cl in ("bruto", "mediana3"):
    for f in FACTORS:
        rr = [r for r in ROWS if int(r["factor"]) == f and r["clean"] == cl]
        dmg = [max(sc(r, "BPFO"), sc(r, "BPFI")) for r in rr if r["race"] in ("OR", "IR")]
        hea = [max(sc(r, "BPFO"), sc(r, "BPFI")) for r in rr if r["race"] == "H"]
        lo = [
            np.log(sc(r, "BPFO") / sc(r, "BPFI"))
            for r in rr
            if r["race"] == "OR" and np.isfinite(sc(r, "BPFO")) and np.isfinite(sc(r, "BPFI"))
        ]
        li = [
            np.log(sc(r, "BPFO") / sc(r, "BPFI"))
            for r in rr
            if r["race"] == "IR" and np.isfinite(sc(r, "BPFO")) and np.isfinite(sc(r, "BPFI"))
        ]
        # detecao do candidato certo: OR->BPFO vs H->BPFO ; IR->BPFI vs H->BPFI
        oO = [sc(r, "BPFO") for r in rr if r["race"] == "OR"]
        hO = [sc(r, "BPFO") for r in rr if r["race"] == "H"]
        iI = [sc(r, "BPFI") for r in rr if r["race"] == "IR"]
        hI = [sc(r, "BPFI") for r in rr if r["race"] == "H"]
        # candidato errado vs sao (deveria ser ~0,5 se o escore e especifico da falha)
        oI = [sc(r, "BPFI") for r in rr if r["race"] == "OR"]
        iO = [sc(r, "BPFO") for r in rr if r["race"] == "IR"]
        print(
            f"{cl:8s} {f:2d}x | det AUC={auc(dmg, hea):.3f} | pista AUC={auc(lo, li):.3f} | BPFO: OR x H={auc(oO, hO):.3f}, IR x H={auc(iO, hO):.3f} | BPFI: IR x H={auc(iI, hI):.3f}, OR x H={auc(oI, hI):.3f}"
        )
