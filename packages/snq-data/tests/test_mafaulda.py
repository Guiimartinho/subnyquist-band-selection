"""MAFAULDA: geometria contra a documentacao, esquema de caminhos e, com dado, taxa e tacometro."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from snq_data import mafaulda as M

ROOT = Path(__file__).resolve().parents[3] / "datasets" / "raw" / "mafaulda"


def test_computed_orders_match_documented_values():
    """Checklist de ingestao: as ordens da geometria batem com as que o dataset publica."""
    got = M.computed_orders()
    for family, documented in M.DOCUMENTED_ORDERS.items():
        assert abs(got[family] - documented) < 1e-3, family


def test_parse_path_scheme():
    assert M.parse_path("normal/12.288.csv") == (None, "normal", None, 12.288)
    assert M.parse_path("underhang/outer_race/6g/30.1056.csv") == (
        "underhang",
        "outer_race",
        6,
        30.1056,
    )
    with pytest.raises(ValueError):
        M.parse_path("underhang/inner_race/0g/12.0.csv")


def _one(fault_dir: str) -> str | None:
    if not ROOT.exists():
        return None
    recs = [r for r in M.list_records(ROOT) if r.startswith(fault_dir)]
    return recs[0] if recs else None


def _record_with_tach(tach: np.ndarray) -> M.MafauldaRecord:
    data = np.zeros((tach.size, len(M.COLUMNS)), dtype=np.float32)
    data[:, 0] = tach
    return M.MafauldaRecord("normal/10.0.csv", None, "normal", None, 10.0, data)


def test_tachometer_hysteresis_ignores_noise_and_glitches():
    """10 Hz por 2 s, com ruido e um glitch curto dentro de cada pulso: 20 flancos, nao 40+."""
    rng = np.random.default_rng(0)
    n = int(2 * M.FS_HZ)
    t = np.arange(n) / M.FS_HZ
    # pulsos de 10 ms comecando em t = 0,05 + k/10 (o sinal comeca em nivel baixo)
    tach = np.where((t * 10.0 + 0.5) % 1.0 < 0.1, 5.0, 0.0) + 0.3 * rng.standard_normal(n)
    for k in range(20):  # glitch: queda de 5 amostras 5 ms dentro de cada pulso
        i = int((0.05 + k / 10.0 + 0.005) * M.FS_HZ)
        tach[i : i + 5] = 0.0
    rec = _record_with_tach(tach)
    assert rec.tachometer_edges_s().size == 20
    assert abs(rec.tachometer_rotation_hz() - 10.0) < 0.01


@pytest.mark.skipif(_one("underhang/outer_race") is None, reason="MAFAULDA nao baixado")
def test_real_file_rate_is_51k2():
    """A taxa inferida (51,2 kHz) faz o tacometro bater com a rotacao do nome do arquivo."""
    rec = M.load(_one("underhang/outer_race"), ROOT)
    assert rec.data.shape == (250_000, 8)
    rel = abs(rec.tachometer_rotation_hz() - rec.nominal_rotation_hz) / rec.nominal_rotation_hz
    # Com a taxa declarada (50 kHz) o desvio seria de ~2,3 %.
    assert rel < 0.01, f"rotacao pelo tacometro difere {100 * rel:.2f}% da nominal"


@pytest.mark.skipif(
    os.environ.get("SNQ_FULL_DATA_CHECK") != "1", reason="checagem completa opcional"
)
def test_tachometer_screen_catalog_is_exact():
    """O catalogo de triagem lista exatamente as sequencias com tacometro fora de 1 %."""
    import yaml

    catalog = ROOT.parents[1] / "catalogs" / "mafaulda-tachometer-screen.yaml"
    listed = set(yaml.safe_load(catalog.read_text(encoding="utf-8"))["excluded"])
    bad = set()
    for rel in M.list_records(ROOT):
        rec = M.load(rel, ROOT, cache_dir=ROOT.parents[1] / "processed" / "mafaulda")
        dev = abs(rec.tachometer_rotation_hz() - rec.nominal_rotation_hz) / rec.nominal_rotation_hz
        if dev > 0.01:
            bad.add(rel)
    assert bad == listed, (sorted(bad - listed)[:5], sorted(listed - bad)[:5])
