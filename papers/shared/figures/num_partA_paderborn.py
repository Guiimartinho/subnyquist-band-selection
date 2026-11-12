"""Números da réplica no Paderborn (E24) citados no texto, recalculados dos CSVs.

Regra do projeto: número do artigo sai dos resultados, nunca de transcrição.

- **Captura em taxa cheia.** Quantos segmentos o kurtograma manda para bandas a partir de 11 kHz, e
  quantos acertos há neles, pela régua aninhada.
- **Pré-condição do mecanismo do Ottawa.** O acerto do kurtograma em taxa cheia por rolamento, pela
  régua aninhada. O mecanismo pressupõe um seletor que acerta em taxa cheia.
- **Testes de robustez e checagem de raias.** Lidos de `results/exploratory/robustness.json` e de
  `inverter_lines.json`, gerados pelos scripts em `exploratory/`.

    uv run python papers/shared/figures/num_partA_paderborn.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E24 = ROOT / "experiments/E24-paderborn-spike-capture"
OUT = ROOT / "papers/mssp-2026/numbers/partA_paderborn.json"


def ses_without_line(rows: list[dict]) -> dict:
    """Infograma SES em taxa cheia nos rolamentos sem raia visível (régua aninhada, balanceada)."""
    visible = {r["bearing"] for r in rows if r["truth_local"] != ""}
    per: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        if (
            r["arm"] == "infograma_ses"
            and int(r["factor"]) == 1
            and r["hit_nested"] != ""
            and r["bearing"] not in visible
        ):
            per[(r["race"], r["bearing"])].append(int(r["hit_nested"]))
    by_race: dict[str, list[float]] = defaultdict(list)
    for (race, _b), v in per.items():
        by_race[race].append(sum(v) / len(v))
    race_acc = {race: round(100.0 * sum(v) / len(v), 1) for race, v in sorted(by_race.items())}
    return {
        "bearings": len(per),
        "by_race": race_acc,
        "balanced": round(sum(race_acc.values()) / len(race_acc), 1),
    }


def main() -> None:
    with (E24 / "results/metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    full = [r for r in rows if r["arm"] == "kurtograma" and int(r["factor"]) == 1]
    high = [r for r in full if r["band_lo_hz"] != "" and float(r["band_lo_hz"]) >= 11000.0 - 1e-6]
    per: dict[str, list[int]] = defaultdict(list)
    for r in full:
        if r["hit_nested"] != "":
            per[r["bearing"]].append(int(r["hit_nested"]))
    acc = {b: round(100.0 * sum(v) / len(v), 1) for b, v in sorted(per.items())}
    rep = {
        "segments": len(full),
        "bearings": len(per),
        "fullrate_bands_from_11khz": len(high),
        "fullrate_bands_from_11khz_pct": round(100.0 * len(high) / len(full), 1),
        "hits_in_those_bands": sum(int(r["hit_nested"] or 0) for r in high),
        "fullrate_accuracy_by_bearing": acc,
        "bearings_above_50pct_fullrate": [b for b, a in acc.items() if a > 50.0],
        "ses_on_bearings_without_visible_line": ses_without_line(rows),
        "robustness": json.loads((E24 / "results/exploratory/robustness.json").read_text()),
        "inverter_lines": json.loads((E24 / "results/exploratory/inverter_lines.json").read_text()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print({k: v for k, v in rep.items() if k not in ("robustness", "inverter_lines")})
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
