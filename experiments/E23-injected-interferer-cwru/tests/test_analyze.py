"""A analise pre-registrada do E23 acerta o veredicto num caso sintetico de resposta conhecida."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "e23_analyze", Path(__file__).resolve().parents[1] / "analyze.py"
)
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)

CFG = {
    "fs_hz": 48000.0,
    "ceiling0_hz": 10000.0,
    "analysis": {
        "evaluable_baseline": 0.5,
        "drop_points": 20,
        "predicted_fraction": 0.8,
        "refute_fraction": 0.5,
        "clear_false_alarm": 0.2,
        "surrogate_ratio": 0.5,
        "surrogate_refute_ratio": 0.8,
        "tat_min_balanced_accuracy": 0.8,
        "tat_refute_balanced_accuracy": 0.65,
        "oracle_gap_points": 5,
    },
}
COLS = [
    "record", "kind", "fc_hz", "rel_db", "factor", "arm", "search_ceiling_hz", "hit",
    "predicted", "interferer_line", "tat_admissible_here",
]  # fmt: skip


def row(rec, kind, fc, db, f, arm, ceil, hit, pred, tat):
    return dict(
        zip(
            COLS,
            [rec, kind, fc, db, f, arm, ceil, hit, pred, "" if kind == "none" else 1 - hit, tat],
            strict=True,
        )
    )


def synthetic(tmp_path: Path, surrogate_drops: bool) -> list[dict]:
    rows = []
    for rec in ("A", "B", "C", "D"):
        for f, ceil in ((2, 10000.0), (2, 5000.0)):
            rows.append(row(rec, "none", "", "", f, "uniforme", ceil, 1, "", 1))
            pred = 1 if ceil == 10000.0 else 0  # imagem em 8 kHz: dentro com 10k, acima com 5k
            hit_imp = 0 if pred == 1 else 1
            rows.append(
                row(
                    rec,
                    "impulsive",
                    16000.0,
                    20.0,
                    f,
                    "uniforme",
                    ceil,
                    hit_imp,
                    pred,
                    hit_imp,  # o TAT acompanha o acerto: acuracia balanceada 1
                )
            )
            hit_sur = 0 if (surrogate_drops and pred == 1) else 1
            rows.append(
                row(rec, "surrogate", 16000.0, 20.0, f, "uniforme", ceil, hit_sur, pred, hit_sur)
            )
    p = tmp_path / "metrics.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return an.load(p)


def test_capture_only_where_predicted_and_not_by_the_surrogate(tmp_path):
    data = an.Data(synthetic(tmp_path, surrogate_drops=False), CFG)
    h1 = data.h1(data.records)
    assert h1["sensitivity_at_pmax"] == 1.0
    assert h1["max_false_alarm"] == 0.0
    assert h1["p_star_db"] == 20.0
    assert h1["surrogate_ratio_at_p_star"] == 0.0
    v = an.verdict(h1, data.h2(data.records), CFG["analysis"])
    assert v["H1a_sensibilidade"] == "apoiada"
    assert v["H1b_especificidade"] == "apoiada"
    assert v["H1_linha_impulsividade"] == "apoiada"


def test_surrogate_dropping_as_much_refutes_the_impulsiveness_line(tmp_path):
    data = an.Data(synthetic(tmp_path, surrogate_drops=True), CFG)
    h1 = data.h1(data.records)
    assert h1["surrogate_ratio_at_p_star"] == 1.0
    v = an.verdict(h1, data.h2(data.records), CFG["analysis"])
    assert v["H1_linha_impulsividade"] == "refutada"


def test_tat_balanced_accuracy_is_perfect_when_tat_tracks_the_hits(tmp_path):
    data = an.Data(synthetic(tmp_path, surrogate_drops=False), CFG)
    h2 = data.h2(data.records)
    assert h2["tat_balanced_accuracy"] == 1.0
