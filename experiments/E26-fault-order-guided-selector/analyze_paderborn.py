"""Análise pré-registrada do E26 no Paderborn (README, "Confirmação no Paderborn").

- **Braços** (limpeza × seletor):
  - ``kurtograma`` (bruto, cego) e ``mediana_kurtograma`` (mediana, cego);
  - ``guiado`` (bruto, guiado) e ``mediana_guiado`` (mediana, guiado).
- **Identificação.**
  - **Cego:** é pista externa se a ordem estimada fica a ≤ ``tolerance_order`` da BPFO
    verdadeira, pista interna se fica assim da BPFI, e "nenhuma" no resto.
    - A ordem verdadeira da própria pista é a régua aninhada do E24: a mediana das calibrações
      válidas dos **outros** rolamentos da mesma pista.
    - Para a outra pista e para os sãos, é a mediana de todas as calibrações válidas dela.
  - **Guiado:** é a falha de maior escore, se ele passar do limiar; senão, "nenhuma".
    - O limiar é o percentil 95 dos escores dos sãos, por fator e limpeza.
    - Para avaliar um são, o limiar sai dos outros 5, deixando um rolamento são de fora.
- **Por rolamento.**
  - Nos danificados, é a fração de segmentos com a falha certa. A acurácia balanceada é a média,
    sobre as pistas, da média dos rolamentos.
  - Nos sãos, é a fração com "nenhuma" (especificidade).
- **Testes.**
  - H1: ``mediana_guiado`` contra ``kurtograma``. H2: ``mediana_guiado`` contra
    ``mediana_kurtograma``.
  - São feitos nos fatores decisivos, por troca de sinal estratificada por pista sobre as
    diferenças por rolamento: exata até 2^``signflip_exact_max``, senão Monte Carlo.
  - Holm sobre os fatores.
  - Passa se o p de Holm for menor que α, a diferença for positiva e a especificidade do
    ``mediana_guiado`` naquele fator for de pelo menos ``min_specificity``.

    uv run python experiments/E26-fault-order-guided-selector/analyze_paderborn.py \
        --config experiments/E26-fault-order-guided-selector/config/paderborn.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parent
ARMS = {
    "kurtograma": ("bruto", "cego"),
    "mediana_kurtograma": ("mediana3", "cego"),
    "guiado": ("bruto", "guiado"),
    "mediana_guiado": ("mediana3", "guiado"),
}
RACE_FAULT = {"OR": "BPFO", "IR": "BPFI"}


def holm(p: list[float]) -> list[float]:
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def signflip(a: dict, b: dict, exact_max: int, mc: int, seed: int) -> dict:
    keys = sorted(set(a) & set(b))
    races = sorted({k[0] for k in keys})
    d = {r: np.array([a[k] - b[k] for k in keys if k[0] == r]) for r in races}
    obs = float(np.mean([v.mean() for v in d.values()]))
    nz = [(r, i) for r in races for i in np.flatnonzero(d[r] != 0)]
    up = sum(1 for r, i in nz if d[r][i] > 0)
    if not nz:
        return {"diff_points": 0.0, "bearings_differing": 0, "bearings_up": 0, "p_one_sided": 1.0}
    w = np.array([abs(d[r][i]) / (d[r].size * len(races)) for r, i in nz])
    if len(nz) <= exact_max:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(nz))))
        p = float(np.mean(signs @ w >= obs - 1e-12))
    else:
        signs = np.random.default_rng(seed).choice((1.0, -1.0), size=(mc, len(nz)))
        p = float((1 + np.count_nonzero(signs @ w >= obs - 1e-12)) / (1 + mc))
    return {
        "diff_points": round(100 * obs, 2),
        "bearings_differing": len(nz),
        "bearings_up": up,
        "p_one_sided": p,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--results", type=str, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    an, tol = cfg["analysis"], float(cfg["tolerance_order"])
    res = EXP / (args.results or cfg["output"]["results_dir"])
    with (res / "metrics.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["skipped"]]
    with (res / "calib.csv").open(encoding="utf-8") as f:
        calib = list(csv.DictReader(f))
    valid = [c for c in calib if c["ok"] == "1"]
    all_med = {
        race: float(np.median([float(c["center"]) for c in valid if c["race"] == race]))
        for race in ("OR", "IR")
    }

    def truth(bearing: str, race_of_bearing: str, race: str) -> float:
        if race == race_of_bearing:  # régua aninhada: sem o próprio rolamento
            peers = [
                float(c["center"]) for c in valid if c["race"] == race and c["bearing"] != bearing
            ]
            return float(np.median(peers))
        return all_med[race]

    def score(r, c):
        return float(r[f"score_{c}"]) if r[f"score_{c}"] != "" else -np.inf

    factors = sorted({int(r["factor"]) for r in rows})
    healthy = sorted({r["bearing"] for r in rows if r["race"] == "H"})
    tau: dict = {}
    for f in factors:
        for clean in ("bruto", "mediana3"):
            hs = defaultdict(list)
            for r in rows:
                if r["race"] == "H" and int(r["factor"]) == f and r["clean"] == clean:
                    hs[r["bearing"]].append(max(score(r, "BPFO"), score(r, "BPFI")))
            q = 100 * (1 - float(cfg["guided"]["false_alarm"]))
            tau[(f, clean, None)] = float(np.percentile([s for v in hs.values() for s in v], q))
            for h in healthy:
                rest = [s for b, v in hs.items() if b != h for s in v]
                tau[(f, clean, h)] = float(np.percentile(rest, q))

    acc: dict = defaultdict(lambda: defaultdict(list))  # (arm, f) -> (race, bearing) -> [ok]
    for r in rows:
        f, clean, race, b = int(r["factor"]), r["clean"], r["race"], r["bearing"]
        for arm, (cl, sel) in ARMS.items():
            if cl != clean:
                continue
            if sel == "cego":
                est = float(r["blind_order"]) if r["blind_order"] != "" else np.nan
                pred = "none"
                for rc in ("OR", "IR"):
                    if np.isfinite(est) and abs(est - truth(b, race, rc)) <= tol:
                        pred = rc
            else:
                th = tau[(f, clean, b if race == "H" else None)]
                top = max(("BPFO", "BPFI"), key=lambda c: score(r, c))
                pred = ("OR" if top == "BPFO" else "IR") if score(r, top) >= th else "none"
            target = "none" if race == "H" else race
            acc[(arm, f)][(race, b)].append(int(pred == target))

    def per_bearing(arm, f, races):
        return {k: float(np.mean(v)) for k, v in acc[(arm, f)].items() if k[0] in races}

    def balanced(pb_):
        by = defaultdict(list)
        for (race, _b), v in pb_.items():
            by[race].append(v)
        return {race: round(100 * float(np.mean(v)), 1) for race, v in sorted(by.items())}

    summary = {}
    for arm in ARMS:
        for f in factors:
            fault = balanced(per_bearing(arm, f, ("OR", "IR")))
            spec = balanced(per_bearing(arm, f, ("H",))).get("H")
            summary[f"{arm}|{f}x"] = {
                **fault,
                "balanced": round(float(np.mean(list(fault.values()))), 1),
                "specificity": spec,
            }
    tests = []
    for name, (arm, base) in {
        "H1": ("mediana_guiado", "kurtograma"),
        "H2": ("mediana_guiado", "mediana_kurtograma"),
    }.items():
        cell = []
        for f in (int(x) for x in an["decisive_factors"]):
            t = signflip(
                per_bearing(arm, f, ("OR", "IR")),
                per_bearing(base, f, ("OR", "IR")),
                int(an["signflip_exact_max"]),
                int(an["signflip_mc"]),
                int(an["seed"]) + f,
            )
            cell.append(
                {
                    "hypothesis": name,
                    "arm": arm,
                    "baseline": base,
                    "factor": f,
                    **t,
                    "specificity_arm": summary[f"{arm}|{f}x"]["specificity"],
                }
            )
        for t, pa in zip(cell, holm([t["p_one_sided"] for t in cell]), strict=True):
            t["p_holm"] = round(pa, 4)
            t["passes"] = int(
                pa < float(an["alpha"])
                and t["diff_points"] > 0
                and (t["specificity_arm"] or 0) >= 100 * float(an["min_specificity"])
            )
        tests += cell
    verdict = {
        h: [t["factor"] for t in tests if t["hypothesis"] == h and t["passes"]]
        for h in ("H1", "H2")
    }
    rep = {
        "thresholds": {
            f"{f}x|{c}": round(tau[(f, c, None)], 3) for f in factors for c in ("bruto", "mediana3")
        },
        "nested_all_median": all_med,
        "summary": summary,
        "tests": tests,
        "verdict_factors_passed": verdict,
    }
    (res / "analysis.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), "utf-8")
    for k, v in summary.items():
        print(k, v)
    for t in tests:
        print(t)
    print("veredito:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
