"""Análise pré-registrada do E26c (README, "E26c: decimação pura contra anti-alias e passa-faixa").

- **Regra,** a do E26 em todos os braços:
  - o limiar é o percentil 95 dos escores máximos (BPFO e BPFI) dos sãos, por braço e fator;
  - para avaliar um são, o limiar sai dos outros 5;
  - a decisão é a falha de maior escore, se ele passar do limiar; senão, "nenhuma".
- **Braços,** do ``metrics.csv``: ``aquisição|estratégia``. A esses somam-se dois passa-faixa com a
  zona escolhida no comissionamento:
  - ``zona_kurtograma|banda_larga_regra``: a zona que contém o centro da banda do kurtograma em
    taxa cheia na medição 1 do próprio rolamento e ajuste;
  - ``zona_frota|banda_larga_regra``: a zona de maior acurácia balanceada nos outros rolamentos.
    Para um rolamento danificado, a zona sai dos demais danificados. Para um são, sai de todos os
    danificados, com o limiar deixando o próprio são de fora.
- **H5:** ``pura|guiado`` > ``antialias|guiado`` em 4×, 8×, 16× e 32×, com Holm sobre 4.
- **H6:** ``pura|guiado`` > ``zona_frota|banda_larga_regra`` em 4× e 8×, com Holm sobre 2.
- **Os testes:** troca de sinal por rolamento, estratificada por pista, unilateral. Um teste passa
  se o p de Holm ficar abaixo de α e a diferença for positiva.
- **Descritivos:**
  - o kurtograma e a banda larga, ``pura`` contra ``antialias``;
  - o perfil por zona;
  - o vazamento: entre as detecções do ``pura|guiado``, a fração com o pico da 1ª harmônica a até
    0,03 de uma ordem inteira, e o acerto com exclusão de ±0,03;
  - as iscas: a fração de segmentos em que a maior isca passa do limiar, nos danificados e nos
    sãos.
- **Integridade.** Os escores do ``pura|guiado`` têm de ser idênticos aos do guiado bruto do E26.

    uv run python experiments/E26-fault-order-guided-selector/analyze_paderborn_acq.py \
        --config experiments/E26-fault-order-guided-selector/config/paderborn_acq.yaml
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e26an", EXP / "analyze_paderborn.py")
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)
FAULTS = ("BPFO", "BPFI")
DECOYS = ("D_BPFO_09", "D_BPFO_11", "D_BPFI_09", "D_BPFI_11")
FS = 64000.0


def val(r: dict, k: str) -> float:
    return float(r[k]) if r.get(k, "") != "" else -np.inf


def balanced(pb_: dict) -> dict:
    by = defaultdict(list)
    for (race, _b), v in pb_.items():
        by[race].append(v)
    return {race: round(100 * float(np.mean(v)), 1) for race, v in sorted(by.items())}


def evaluate(rows: list[dict], q: float, tag: str = "") -> tuple[dict, dict]:
    """Acerto por (rolamento) e limiares, para um braço e um fator."""
    hs = defaultdict(list)
    for r in rows:
        if r["race"] == "H":
            hs[r["bearing"]].append(max(val(r, f"s{tag}_{c}") for c in FAULTS))
    if not hs:
        return {}, {}
    tau_all = float(np.percentile([s for v in hs.values() for s in v], q))
    tau = {h: float(np.percentile([s for b, v in hs.items() if b != h for s in v], q)) for h in hs}
    acc = defaultdict(list)
    for r in rows:
        th = tau[r["bearing"]] if r["race"] == "H" else tau_all
        top = max(FAULTS, key=lambda c: val(r, f"s{tag}_{c}"))
        ok = val(r, f"s{tag}_{top}") >= th
        pred = ("OR" if top == "BPFO" else "IR") if ok else "none"
        acc[(r["race"], r["bearing"])].append(
            int(pred == ("none" if r["race"] == "H" else r["race"]))
        )
    return {k: float(np.mean(v)) for k, v in acc.items()}, {"all": tau_all, **tau}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--results", type=str, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    a = cfg["analysis_acq"]
    q = 100 * (1 - float(cfg["guided"]["false_alarm"]))
    res = EXP / (args.results or cfg["output"]["results_dir"])
    with (res / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (res / "commission.csv").open(encoding="utf-8") as f:
        comm = {(c["bearing"], c["setting"]): c for c in csv.DictReader(f)}
    factors = sorted({int(r["factor"]) for r in rows})
    arms = defaultdict(list)
    for r in rows:
        arms[(f"{r['acquisition']}|{r['strategy']}", int(r["factor"]))].append(r)

    # passa-faixa com a zona do comissionamento (kurtograma em taxa cheia na medição 1)
    for dec in (int(z) for z in cfg["zone_factors"] if int(z) in factors):
        width = FS / (2 * dec)
        for r in rows:
            if int(r["factor"]) != dec or r["strategy"] != "banda_larga_regra":
                continue
            c = comm.get((r["bearing"], r["setting"]))
            if c is None:
                continue
            k = min(dec - 1, int(0.5 * (float(c["lo_hz"]) + float(c["hi_hz"])) // width))
            if r["acquisition"] == ("antialias" if k == 0 else f"zona{k}"):
                arms[("zona_kurtograma|banda_larga_regra", dec)].append(r)

    per: dict = {}
    for (arm, f), rs in arms.items():
        per[(arm, f)], _ = evaluate(rs, q)

    # passa-faixa com a zona da frota, deixando o rolamento de fora
    for dec in (int(z) for z in cfg["zone_factors"] if int(z) in factors):
        zones = ["antialias", *[f"zona{k}" for k in range(1, dec)]]
        zacc = {z: per.get((f"{z}|banda_larga_regra", dec), {}) for z in zones}
        frota = {}
        for key in {k for z in zones for k in zacc[z]}:
            race, b = key

            def score_zone(z, b=b, zacc=zacc):
                dmg = {k: v for k, v in zacc[z].items() if k[0] in ("OR", "IR") and k[1] != b}
                bal = balanced(dmg)
                return float(np.mean(list(bal.values()))) if bal else -1.0

            zbest = max(zones, key=score_zone)
            if key in zacc[zbest]:
                frota[key] = zacc[zbest][key]
        per[("zona_frota|banda_larga_regra", dec)] = frota

    summary = {}
    for (arm, f), pbr in sorted(per.items()):
        fault = balanced({k: v for k, v in pbr.items() if k[0] in ("OR", "IR")})
        spec = balanced({k: v for k, v in pbr.items() if k[0] == "H"}).get("H")
        summary[f"{arm}|{f}x"] = {
            **fault,
            "balanced": round(float(np.mean(list(fault.values()))), 1) if fault else None,
            "specificity": spec,
        }

    def dmg(arm, f):
        return {k: v for k, v in per.get((arm, f), {}).items() if k[0] in ("OR", "IR")}

    tests = []
    for name, arm, basearm, fs_ in (
        ("H5", "pura|guiado", "antialias|guiado", a["h5_factors"]),
        ("H6", "pura|guiado", "zona_frota|banda_larga_regra", a["h6_factors"]),
    ):
        cell = []
        for f in (int(x) for x in fs_ if int(x) in factors):
            t = an.signflip(
                dmg(arm, f),
                dmg(basearm, f),
                int(a["signflip_exact_max"]),
                int(a["signflip_mc"]),
                int(a["seed"]) + f,
            )
            cell.append({"hypothesis": name, "arm": arm, "baseline": basearm, "factor": f, **t})
        for t, pa in zip(cell, an.holm([t["p_one_sided"] for t in cell]), strict=True):
            t["p_holm"] = round(pa, 4)
            t["passes"] = int(pa < float(a["alpha"]) and t["diff_points"] > 0)
        tests += cell

    # vazamento e iscas no pura|guiado
    leak = {}
    for f in factors:
        rs = arms[("pura|guiado", f)]
        _, tau = evaluate(rs, q)
        dets, near_int, decoy_dmg, decoy_h, true_dmg = 0, 0, [], [], []
        for r in rs:
            th = tau.get(r["bearing"], tau["all"]) if r["race"] == "H" else tau["all"]
            top = max(FAULTS, key=lambda c: val(r, f"s_{c}"))
            dmax = max(val(r, f"s_{d}") for d in DECOYS)
            (decoy_h if r["race"] == "H" else decoy_dmg).append(int(dmax >= th))
            if r["race"] != "H":
                true = "BPFO" if r["race"] == "OR" else "BPFI"
                true_dmg.append(int(val(r, f"s_{true}") >= th))
            if r["race"] != "H" and val(r, f"s_{top}") >= th:
                dets += 1
                pk = val(r, f"peak_{top}")
                near_int += int(np.isfinite(pk) and abs(pk - round(pk)) <= 0.03)
        p03, _ = evaluate(rs, q, tag="03")
        leak[f"{f}x"] = {
            "detections_damaged": dets,
            "share_peak_within_003_of_integer": round(near_int / dets, 3) if dets else None,
            "balanced_with_exclusion_003": round(
                float(
                    np.mean(
                        list(
                            balanced(
                                {k: v for k, v in p03.items() if k[0] in ("OR", "IR")}
                            ).values()
                        )
                    )
                ),
                1,
            ),
            "decoy_over_threshold_damaged": round(float(np.mean(decoy_dmg)), 3)
            if decoy_dmg
            else None,
            "decoy_over_threshold_healthy": round(float(np.mean(decoy_h)), 3) if decoy_h else None,
            "true_fault_over_threshold_damaged": round(float(np.mean(true_dmg)), 3)
            if true_dmg
            else None,
        }

    # integridade: pura|guiado contra o guiado bruto do E26
    with (EXP / a["guided_results"] / "metrics.csv").open(encoding="utf-8") as fh:
        g = {
            (r["bearing"], r["setting"], r["measurement"], r["factor"]): r
            for r in csv.DictReader(fh)
            if not r["skipped"] and r["clean"] == "bruto"
        }
    diffs = []
    for r in rows:
        if r["acquisition"] == "pura" and r["strategy"] == "guiado":
            o = g.get((r["bearing"], r["setting"], r["measurement"], r["factor"]))
            if o:
                for c in FAULTS:
                    if r[f"s_{c}"] and o[f"score_{c}"]:
                        diffs.append(abs(float(r[f"s_{c}"]) - float(o[f"score_{c}"])))
    integrity = {"matches": len(diffs), "max_abs_diff": max(diffs) if diffs else None}

    verdict = {
        h: [t["factor"] for t in tests if t["hypothesis"] == h and t["passes"]]
        for h in ("H5", "H6")
    }
    rep = {
        "integrity": integrity,
        "summary": summary,
        "tests": tests,
        "leakage_and_decoys": leak,
        "verdict_factors_passed": verdict,
    }
    (res / "analysis.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), "utf-8")
    print("integridade:", integrity)
    for k, v in summary.items():
        print(k, v)
    for t in tests:
        print(t)
    for k, v in leak.items():
        print("vazamento", k, v)
    print("veredito:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
