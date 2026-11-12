"""Números do E23 que o artigo cita, recalculados do metrics.csv.

Três números da lista "O que o artigo pode afirmar" (README) vieram do recálculo do `reviewer-2` e
não tinham script: os 97 % das perdas sobre a taxa das rajadas, os 8,5 pontos do anti-alias e o IC
por aglomerado da razão do substituto. Este script os recalcula com as definições abaixo, escritas
antes de rodá-lo. Se o valor diferir do README, vale o daqui, e o README é corrigido.

1. **Perdas sobre a linha do interferente.** Vale para o braço uniforme, no teto de referência,
   em fatores ≥ 2 e com o impulsivo na maior dose. Entre os casos (registro, portadora, fator) em
   que o braço acerta sem injeção e erra com ela, é a fração cuja estimativa cai na taxa das rajadas
   ou num harmônico até o 3º (`interferer_line`).
2. **Custo do anti-alias ideal sem injeção.** É o acerto do `uniforme` no teto de referência menos
   o do `aaf_ideal`, os dois sem injeção, em pontos: em 10× e 12× (o que o README atribui ao
   revisor) e em todos os fatores ≥ 2 (o que a rodada base reporta).
3. **Razão substituto/impulsivo em P\*.** É a estatística pré-registrada da H1', de `analyze.py`.
   O IC sai do bootstrap dos 4 aglomerados de falha independentes (`CLUSTERS`), no lugar do
   bootstrap por registro.
4. **Fração dobrada contra o tamanho da queda.** É o Spearman, entre as células de captura
   prevista avaliáveis, na maior dose, entre a fração dobrada média da célula e a queda do
   impulsivo (`Data.drop`).
5. **Dose-resposta dos picos.** É a queda máxima do uniforme em fatores ≥ 2, por taxa e amplitude,
   de `results-spikes/metrics.csv`.

    uv run python experiments/E23-injected-interferer-cwru/paper_numbers.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

E = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e23_analyze", E / "analyze.py")
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)

# Os 12 registros BPFO são uma falha de 0,007" em 3 posições x 4 cargas; o 213 é o BPFI (README,
# revisão adversarial, "Unidades independentes").
CLUSTERS = {
    "centrada": range(135, 139),
    "ortogonal": range(148, 152),
    "oposta": range(161, 165),
    "bpfi_213": range(213, 214),
}


def cluster_of(record: str) -> str:
    n = int(record[:3])
    for name, span in CLUSTERS.items():
        if n in span:
            return name
    raise KeyError(record)


def losses_on_line(data, cfg: dict) -> dict:
    db = max(data.powers)
    n_loss = n_line = 0
    for (kind, _fc, d, rec, f, c), r in data.inj.items():
        if kind != "impulsive" or d != db or f < 2 or c != an.ceiling0(cfg, f):
            continue
        if data.base.get((rec, f, c)) == 1 and r["hit"] == 0:
            n_loss += 1
            n_line += int(r["line"] == 1)
    return {
        "dose_db_label": db,
        "losses": n_loss,
        "on_line": n_line,
        "share": round(n_line / n_loss, 4) if n_loss else None,
    }


def aaf_cost(rows: list[dict], cfg: dict, factors=(10, 12)) -> dict:
    uni = [
        r["hit"]
        for r in rows
        if r["arm"] == "uniforme"
        and r["kind"] == "none"
        and r["factor"] in factors
        and r["search_ceiling_hz"] == an.ceiling0(cfg, r["factor"])
        and r["hit"] is not None
    ]
    aaf = [
        r["hit"]
        for r in rows
        if r["arm"] == "aaf_ideal"
        and r["kind"] == "none"
        and r["factor"] in factors
        and r["hit"] is not None
    ]
    return {
        "factors": list(factors),
        "n_uniform": len(uni),
        "n_aaf": len(aaf),
        "acc_uniform": round(100 * float(np.mean(uni)), 2),
        "acc_aaf_ideal": round(100 * float(np.mean(aaf)), 2),
        "cost_points": round(100 * (float(np.mean(uni)) - float(np.mean(aaf))), 2),
    }


def surrogate_ratio_cluster_ci(data, cfg: dict) -> dict:
    by: dict[str, list[str]] = defaultdict(list)
    for rec in data.records:
        by[cluster_of(rec)].append(rec)
    names = sorted(by)
    point = data.h1(data.records)["surrogate_ratio_at_p_star"]
    rng = np.random.default_rng(int(cfg["seed"]) + 7)
    boots = []
    for _ in range(int(cfg["analysis"]["bootstrap"])):
        sample = [rec for k in rng.choice(names, size=len(names)) for rec in by[k]]
        v = data.h1(sample)["surrogate_ratio_at_p_star"]
        if v is not None and np.isfinite(v):
            boots.append(v)
    return {
        "point": round(point, 4),
        "n_boot_valid": len(boots),
        "ci95_cluster": [round(float(np.percentile(boots, q)), 3) for q in (2.5, 97.5)],
        "clusters": {k: by[k] for k in names},
    }


def folded_share_vs_drop(data) -> dict:
    from scipy.stats import spearmanr

    db = max(data.powers)
    share: dict = defaultdict(list)
    for (kind, fc, d, _rec, f, c), r in data.inj.items():
        if kind == "impulsive" and d == db and r["folded_share"] not in ("", None):
            share[(fc, f, c)].append(float(r["folded_share"]))
    xs, ys = [], []
    for cell in data.cells:
        if data.pred.get(cell) != 1 or not data.evaluable(cell, data.records) or not share[cell]:
            continue
        dr = data.drop("impulsive", cell[0], db, cell[1], cell[2], data.records)
        if dr is not None:
            xs.append(float(np.mean(share[cell])))
            ys.append(dr)
    rho, p = spearmanr(xs, ys)
    return {"n_cells": len(xs), "spearman_rho": round(float(rho), 3), "p": round(float(p), 3)}


def spike_dose_response(path: Path, cfg: dict) -> dict:
    hit: dict = defaultdict(dict)
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["hit"] == "" or r["arm"] != "uniforme":
            continue
        f = int(r["factor"])
        if float(r["search_ceiling_hz"]) != an.ceiling0(cfg, f):
            continue
        hit[(r["kind"], r["spike_rate_hz"], r["spike_amp_mad"], f)][r["record"]] = int(r["hit"])
    out = {}
    for (kind, rate, amp, f), v in hit.items():
        if kind == "none" or f < 2:
            continue
        base = hit.get(("none", "", "", f), {})
        common = sorted(set(v) & set(base))
        if not common:
            continue
        drop = 100.0 * (np.mean([base[k] for k in common]) - np.mean([v[k] for k in common]))
        key = f"rate={rate}|amp={amp}"
        out[key] = max(out.get(key, -1e9), round(float(drop), 1))
    return dict(sorted(out.items()))


def main() -> int:
    cfg = yaml.safe_load((E / "config" / "base.yaml").read_text(encoding="utf-8"))
    rows = an.load(E / "results" / "metrics.csv")
    data = an.Data(rows, cfg)
    unmapped = [r for r in data.records if r[:3].isdigit() is False]
    assert not unmapped, unmapped
    spikes_cfg = yaml.safe_load((E / "config" / "spikes.yaml").read_text(encoding="utf-8"))
    rep = {
        "losses_on_interferer_line": losses_on_line(data, cfg),
        "aaf_ideal_cost_no_injection": aaf_cost(rows, cfg),
        "aaf_ideal_cost_no_injection_factors_ge2": aaf_cost(
            rows, cfg, tuple(sorted({r["factor"] for r in rows if r["factor"] >= 2}))
        ),
        "surrogate_ratio_at_p_star": surrogate_ratio_cluster_ci(data, cfg),
        "folded_share_vs_drop": folded_share_vs_drop(data),
        "spike_max_drop_points_factor_ge2": spike_dose_response(
            E / "results-spikes" / "metrics.csv", spikes_cfg
        ),
    }
    out = E / "results" / "paper_numbers.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
