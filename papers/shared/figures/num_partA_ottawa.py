"""Números da Parte (a) no Ottawa citados no texto do artigo, recalculados dos metrics.csv.

Regra do projeto: número do artigo sai de metrics.csv, nunca de transcrição.

- **Banda escolhida (E20).** Pista externa, teto de 25 kHz, em 1× e 4×, por intervenção: quantos
  segmentos ficam com banda acima de 12,75 kHz, exatamente em [12,75; 25] kHz, e entre 2,0 e
  4,6 kHz.
- **Parede de taxa (E07 estendido, `sem_prior`).** Por fator: acerto nas duas réguas, largura
  mediana da banda escolhida, frequência de falha mediana (ordem verdadeira x rotação) e fração das
  estimativas em [2,0; 2,2) ordem, junto do limite inferior da busca.

    uv run python papers/shared/figures/num_partA_ottawa.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E20 = ROOT / "experiments/E20-fold-zone-rejection/results/metrics.csv"
E07 = ROOT / "experiments/E07-order-domain-ottawa/results/extended/metrics.csv"
OUT = ROOT / "papers/mssp-2026/numbers/partA_ottawa.json"


def band_selection() -> dict:
    out = {}
    with E20.open(encoding="utf-8") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["condition"] == "outer_race" and float(r["ceiling_hz"]) == 25000.0
        ]
    for iname in ("none", "despike", "aaf_ideal", "stop_z123"):
        for m in (1, 4):
            sel = [r for r in rows if r["intervention"] == iname and int(r["factor"]) == m]
            lo = [float(r["band_lo_hz"]) for r in sel]
            hi = [float(r["band_hi_hz"]) for r in sel]
            out[f"{iname}|{m}x"] = {
                "n": len(sel),
                "above_12750": sum(1 for a in lo if a >= 12750.0 - 1e-6),
                "exactly_12750_25000": sum(
                    1
                    for a, b in zip(lo, hi, strict=True)
                    if abs(a - 12750) < 1 and abs(b - 25000) < 1
                ),
                "within_2000_4600": sum(
                    1 for a, b in zip(lo, hi, strict=True) if a >= 2000.0 and b <= 4600.0
                ),
            }
    return out


def rate_wall() -> dict:
    hits: dict = defaultdict(list)
    bw: dict = defaultdict(list)
    est: dict = defaultdict(list)
    ff: dict = defaultdict(list)
    with E07.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mode"] != "sem_prior":
                continue
            m = int(r["factor"])
            if r["hit_local"] != "":
                hits[(m, "local")].append(int(r["hit_local"]))
            if r["hit"] != "":
                hits[(m, "nested")].append(int(r["hit"]))
            if r["band_lo_hz"] != "":
                bw[m].append(float(r["band_hi_hz"]) - float(r["band_lo_hz"]))
            if r["estimate_order"] != "":
                est[m].append(float(r["estimate_order"]))
            if r["truth_order"] != "" and r["speed_seg_hz"] != "":
                ff[m].append(float(r["truth_order"]) * float(r["speed_seg_hz"]))
    out = {}
    for m in sorted({k[0] for k in hits}):
        loc, nes = hits[(m, "local")], hits[(m, "nested")]
        out[f"{m}x"] = {
            "rate_ksps": round(200.0 / m, 3),
            "hits_local_pct": round(100.0 * sum(loc) / len(loc), 1),
            "n_local": len(loc),
            "hits_nested_pct": round(100.0 * sum(nes) / len(nes), 1),
            "median_band_hz": round(st.median(bw[m]), 1),
            "median_fault_hz": round(st.median(ff[m]), 1),
            "estimates_in_2_to_2p2_pct": round(
                100.0 * sum(1 for e in est[m] if 2.0 <= e < 2.2) / len(est[m]), 1
            ),
        }
    return out


def main() -> None:
    rep = {"band_selection_e20": band_selection(), "rate_wall_e07_extended": rate_wall()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["band_selection_e20"], indent=0))
    for k, v in rep["rate_wall_e07_extended"].items():
        print(k, v)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
