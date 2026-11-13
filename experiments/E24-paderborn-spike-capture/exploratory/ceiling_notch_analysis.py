"""Pre-registered analysis of E24b (README, "E24b: ceiling without folding and notch"), written
before running.

The reference arms (``kurtogram`` and ``ses_infogram``, with no ceiling and no notch) are read from
``results/metrics.csv``. The code is the same and is deterministic. The new ones come from
``results-ceiling-notch/metrics.csv``. The accuracies use the nested ruler, and the comparisons use
the exact sign-flip test by bearing, stratified by race (``robustness.sign_flip_p``).

- **C1a (folding).** It is the fraction of segments in the floor band (start ≤ 501 Hz and
  end ≤ 1200 Hz) with an 8 kHz ceiling at 1×, against the kurtogram at 4×.
  - With ≥ 20 %, the floor choice is baseband content, and folding is refuted as the cause.
  - With ≤ 10 %, folding is implicated.
  - In between, inconclusive.
- **C1b (ceiling).** It is the kurtogram with a ceiling at 1× against the kurtogram without a
  ceiling at 1×, with Holm over the 4 ceilings. The ceiling relieves the capture if any of them
  passes (Holm p < 0.05 and difference > 0).
- **C1c (descriptive).** It is the ceiling at 1× against decimation at the factor with the same
  partition (8 kHz and 4×, 4 kHz and 8×, 2 kHz and 16×, 1 kHz and 32×), with a two-sided p. The
  same holds for the SES infogram.
- **C3a (notch).** Among the segments in which the kurtogram without notch picks 11.0–21.5 kHz at
  1×, it is the fraction that changes band with the 15–17 kHz and 30.5–32 kHz notch.
  - With ≥ 50 %, the content of those bands sustains the choice.
  - With < 20 %, it does not.
  - In between, inconclusive.
- **C3b (validity).** The control notch (12–14 kHz and 26.5–28 kHz) must change the band in less
  than 10 % of all segments at 1×. If it changes more, C3a is not interpretable.
- **C3c (descriptive).** It is the change in accuracy with the notch, at 1× and at 4×.

    uv run python experiments/E24-paderborn-spike-capture/exploratory/ceiling_notch_analysis.py
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
PAIRS = {8: 4, 4: 8, 2: 16, 1: 32}  # ceiling in kHz -> factor with the same partition


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
    return yes if x >= hi else (no if x <= lo else "inconclusive")


def main() -> int:
    base, new = rows_of("results"), rows_of("results-ceiling-notch")
    rep: dict = {"C1a": {}, "C1b": {}, "C1c": {}, "C3": {}}
    for khz, m in PAIRS.items():
        share_t = floor_share(bands(new, f"kurtogram_ceiling{khz}k", 1))
        share_d = floor_share(bands(base, "kurtogram", m))
        rep["C1a"][f"{khz}k_vs_{m}x"] = {
            "floor_share_ceiling": round(share_t, 3),
            "floor_share_decimation": round(share_d, 3),
        }
    rep["C1a"]["verdict"] = three(
        rep["C1a"]["8k_vs_4x"]["floor_share_ceiling"],
        0.20,
        0.10,
        "folding refuted as the cause of the floor choice",
        "folding implicated",
    )
    for sel in ("kurtogram", "ses_infogram"):
        tests = [compare(new, f"{sel}_ceiling{khz}k", 1, base, sel, 1) for khz in PAIRS]
        adj = e24.holm([t["p_one_sided"] for t in tests])
        for khz, t, pa in zip(PAIRS, tests, adj, strict=True):
            t["p_holm"] = round(pa, 4)
            rep["C1b"][f"{sel}|ceiling{khz}k_vs_no_ceiling"] = t
            rep["C1c"][f"{sel}|ceiling{khz}k_vs_{PAIRS[khz]}x"] = compare(
                new, f"{sel}_ceiling{khz}k", 1, base, sel, PAIRS[khz]
            )
    won = [
        k
        for k, t in rep["C1b"].items()
        if k.startswith("kurtogram|") and t["p_holm"] < 0.05 and t["diff_points"] > 0
    ]
    rep["C1b"]["verdict"] = (
        "the ceiling relieves the capture" if won else "the ceiling does not relieve the capture"
    )
    rep["C1b"]["ceilings_won"] = won
    b0 = bands(base, "kurtogram", 1)
    captured = [k for k, b in b0.items() if b == (11000, 21500)]
    bn = bands(new, "kurtogram_notch", 1)
    bc = bands(new, "kurtogram_notch_control", 1)
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
                "the notched content sustains the choice",
                "the notched content does not sustain the choice",
            )
            if ctrl < 0.10
            else "invalid control"
        ),
        "accuracy_1x": compare(new, "kurtogram_notch", 1, base, "kurtogram", 1),
        "accuracy_4x": compare(new, "kurtogram_notch", 4, base, "kurtogram", 4),
        "control_accuracy_1x": compare(new, "kurtogram_notch_control", 1, base, "kurtogram", 1),
    }
    out = EXP / "results-ceiling-notch" / "analysis.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
