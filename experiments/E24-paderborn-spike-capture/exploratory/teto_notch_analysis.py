"""Análise pré-registrada do E24b (README, "E24b: teto sem dobra e notch"), escrita antes de rodar.

Os braços de referência (``kurtograma`` e ``infograma_ses`` sem teto e sem notch) são lidos de
``results/metrics.csv``. O código é o mesmo e é determinístico. Os novos vêm de
``results-teto-notch/metrics.csv``. As acurácias usam a régua aninhada, e as comparações usam a
troca de sinal exata por rolamento, estratificada por pista (``robustness.sign_flip_p``).

- **C1a (dobra).** É a fração de segmentos na banda do piso (início ≤ 501 Hz e fim ≤ 1200 Hz) com
  teto de 8 kHz em 1×, contra o kurtograma em 4×.
  - Com ≥ 20 %, a escolha do piso é conteúdo de banda base, e a dobra fica refutada como causa.
  - Com ≤ 10 %, a dobra está implicada.
  - Entre os dois, inconclusivo.
- **C1b (teto).** É o kurtograma com teto em 1× contra o kurtograma sem teto em 1×, com Holm sobre
  os 4 tetos. O teto alivia a captura se algum passar (Holm p < 0,05 e diferença > 0).
- **C1c (descritivo).** É o teto em 1× contra a decimação no fator de mesma partição (8 kHz e 4×,
  4 kHz e 8×, 2 kHz e 16×, 1 kHz e 32×), com p bilateral. O mesmo vale para o infograma SES.
- **C3a (notch).** Entre os segmentos em que o kurtograma sem notch escolhe 11,0–21,5 kHz em 1×, é
  a fração que muda de banda com o notch de 15–17 kHz e 30,5–32 kHz.
  - Com ≥ 50 %, o conteúdo dessas faixas sustenta a escolha.
  - Com < 20 %, não sustenta.
  - Entre os dois, inconclusivo.
- **C3b (validade).** O notch de controle (12–14 kHz e 26,5–28 kHz) precisa mudar a banda em menos
  de 10 % de todos os segmentos em 1×. Se mudar mais, a C3a não é interpretável.
- **C3c (descritivo).** É a mudança de acurácia com o notch, em 1× e em 4×.

    uv run python experiments/E24-paderborn-spike-capture/exploratory/teto_notch_analysis.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e24robust", EXP / "exploratory" / "robustness.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
e24 = rb.e24
PAIRS = {8: 4, 4: 8, 2: 16, 1: 32}  # teto em kHz -> fator de mesma partição


def rows_of(name: str) -> list[dict]:
    with (EXP / name / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["factor"] = int(r["factor"])
    return rows


def bands(rows, arm, factor) -> dict:
    return {
        (r["record"], r["segment"]): (round(float(r["band_lo_hz"])), round(float(r["band_hi_hz"])))
        for r in rows
        if r["arm"] == arm and r["factor"] == factor and r["band_lo_hz"] != ""
    }


def floor_share(b: dict) -> float:
    return sum(1 for lo, hi in b.values() if lo <= 501 and hi <= 1200) / max(1, len(b))


def compare(rows_a, arm_a, f_a, rows_b, arm_b, f_b) -> dict:
    a = e24.per_bearing(rows_a, arm_a, f_a, "nested")
    b = e24.per_bearing(rows_b, arm_b, f_b, "nested")
    p_up, n_nz, n_up = rb.sign_flip_p(a, b)
    p_down, _, _ = rb.sign_flip_p(b, a)
    return {
        "diff_points": round(e24.balanced(a) - e24.balanced(b), 2),
        "bearings_differing": n_nz,
        "bearings_up": n_up,
        "p_one_sided": p_up,
        "p_two_sided": min(1.0, 2 * min(p_up, p_down)),
    }


def three(x: float, hi: float, lo: float, yes: str, no: str) -> str:
    return yes if x >= hi else (no if x <= lo else "inconclusivo")


def main() -> int:
    base, new = rows_of("results"), rows_of("results-teto-notch")
    rep: dict = {"C1a": {}, "C1b": {}, "C1c": {}, "C3": {}}
    for khz, m in PAIRS.items():
        share_t = floor_share(bands(new, f"kurtograma_teto{khz}k", 1))
        share_d = floor_share(bands(base, "kurtograma", m))
        rep["C1a"][f"{khz}k_vs_{m}x"] = {
            "floor_share_ceiling": round(share_t, 3),
            "floor_share_decimation": round(share_d, 3),
        }
    rep["C1a"]["verdict"] = three(
        rep["C1a"]["8k_vs_4x"]["floor_share_ceiling"],
        0.20,
        0.10,
        "dobra refutada como causa do piso",
        "dobra implicada",
    )
    for sel in ("kurtograma", "infograma_ses"):
        tests = [compare(new, f"{sel}_teto{khz}k", 1, base, sel, 1) for khz in PAIRS]
        adj = e24.holm([t["p_one_sided"] for t in tests])
        for khz, t, pa in zip(PAIRS, tests, adj, strict=True):
            t["p_holm"] = round(pa, 4)
            rep["C1b"][f"{sel}|teto{khz}k_vs_sem_teto"] = t
            rep["C1c"][f"{sel}|teto{khz}k_vs_{PAIRS[khz]}x"] = compare(
                new, f"{sel}_teto{khz}k", 1, base, sel, PAIRS[khz]
            )
    won = [
        k
        for k, t in rep["C1b"].items()
        if k.startswith("kurtograma|") and t["p_holm"] < 0.05 and t["diff_points"] > 0
    ]
    rep["C1b"]["verdict"] = "teto alivia a captura" if won else "teto nao alivia"
    rep["C1b"]["ceilings_won"] = won
    b0 = bands(base, "kurtograma", 1)
    captured = [k for k, b in b0.items() if b == (11000, 21500)]
    bn = bands(new, "kurtograma_notch", 1)
    bc = bands(new, "kurtograma_notch_controle", 1)
    left = sum(1 for k in captured if k in bn and bn[k] != b0[k]) / max(1, len(captured))
    ctrl = sum(1 for k in b0 if k in bc and bc[k] != b0[k]) / max(1, len(b0))
    rep["C3"] = {
        "captured_segments": len(captured),
        "left_band_with_notch": round(left, 3),
        "control_changed_any_band": round(ctrl, 3),
        "control_valid": ctrl < 0.10,
        "verdict": (
            three(
                left,
                0.50,
                0.20,
                "o conteudo notcheado sustenta a escolha",
                "o conteudo notcheado nao sustenta a escolha",
            )
            if ctrl < 0.10
            else "controle invalido"
        ),
        "accuracy_1x": compare(new, "kurtograma_notch", 1, base, "kurtograma", 1),
        "accuracy_4x": compare(new, "kurtograma_notch", 4, base, "kurtograma", 4),
        "control_accuracy_1x": compare(new, "kurtograma_notch_controle", 1, base, "kurtograma", 1),
    }
    out = EXP / "results-teto-notch" / "analysis.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
