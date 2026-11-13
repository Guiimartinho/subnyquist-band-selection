"""Executable check of the three E07 guards and of the extension.

The extension (`config/extended.yaml`) relaxes two guards so that factors 96, 128 and
160 start producing measurements. How many measurements each factor admits is the **premise** of
the extension, and it used to be a calculation done outside the repository. Here it becomes
execution:

- `test_primeiro_guarda_por_fator` (first guard per factor) runs arithmetic only, without data,
  and pins the README table (which guard fires first at each factor, under each config);
- `test_extended_relaxa_exatamente_dois_guardas` (extended relaxes exactly two guards) prevents a
  third confounder from entering the extended config unnoticed;
- `test_admissao_base_reproduz_o_csv_medido` (base admission reproduces the measured CSV)
  compares the guard model with the run that is in `results/metrics.csv` — this is what
  authorizes using the model to talk about factors not yet run;
- `test_admissao_extended_e_n_cheio` (extended admission is full n) executes the claim "72 of 72
  at the twelve factors".

The last two read the Ottawa `.mat` files, which are not versioned; without them the test is
skipped, it does not fail.

    uv run pytest experiments/E07-order-domain-ottawa/tests -q

The path must be explicit because `testpaths` covers `packages/`, and that is
deliberate: nothing in `packages/` imports `experiments/`. This test lives inside the
experiment itself and only imports its `run.py`.
"""

from __future__ import annotations

import csv
import importlib.util
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

EXP = Path(__file__).resolve().parents[1]
RAW = EXP / "datasets/raw/ottawa"
FS_HZ = 200_000.0

_spec = importlib.util.spec_from_file_location("e07_run", EXP / "run.py")
assert _spec is not None and _spec.loader is not None
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def load_cfg(name: str) -> dict:
    return yaml.safe_load((EXP / "config" / f"{name}.yaml").read_text(encoding="utf-8"))


BASE = load_cfg("base")
EXTENDED = load_cfg("extended")

# Worst-case rotation speed WITHIN an analyzed segment: 1674.1 rpm = 27.9017 Hz, the
# maximum of fs_eff/max(v) over the 72 windows. It is the value that the guard sees —
# `run.py` applies `np.max` to the decimated profile of the segment, not to the whole record.
WORST_SPEED_HZ = 1674.107142878 / 60.0


def geometry(factor: int, cfg: dict) -> tuple[int, float, float]:
    """`(n_samples, fs_eff_hz, points_per_rev)` in the worst segment."""
    n = math.ceil(int(cfg["segment_samples"]) / factor)
    fs_eff = FS_HZ / factor
    return n, fs_eff, fs_eff / WORST_SPEED_HZ


@pytest.mark.parametrize(
    ("factor", "sob_base", "sob_extended"),
    [
        (48, None, None),
        # 64: `samples_per_rev` knocks out part of the windows in base (112.00 < 128) and
        # none in extended (112.00 >= 32).
        (64, "samples_per_rev", None),
        (96, "samples_per_rev", None),
        # 128: the two relaxed guards zero it in base; `min_samples` fires first.
        (128, "min_samples", None),
        (160, "min_samples", None),
        # 192 passes in extended; the reason the sweep stops at 160 is the width of the
        # kurtogram search window, not a guard — see the README.
        (192, "min_samples", None),
        # 256 is the first factor that extended also rejects: ceil(500000/256)=1954.
        (256, "min_samples", "min_samples"),
    ],
)
def test_primeiro_guarda_por_fator(factor: int, sob_base: str | None, sob_extended: str | None):
    assert run.guard_verdict(*geometry(factor, BASE), BASE) == sob_base
    assert run.guard_verdict(*geometry(factor, EXTENDED), EXTENDED) == sob_extended


def test_f_min_hz_nao_e_o_guarda_que_zera_96_e_128():
    """The 500 Hz floor only starts to bind from factor 200 on."""
    for factor in (96, 128, 160, 192):
        _, fs_eff, _ = geometry(factor, BASE)
        assert fs_eff / 2.0 > float(BASE["kurtogram"]["f_min_hz"])
    assert float(BASE["kurtogram"]["f_min_hz"]) == FS_HZ / 200 / 2.0


def test_extended_relaxa_exatamente_dois_guardas():
    """A third confounder entering the extended config breaks this test."""
    esperado = {"min_samples", "samples_per_rev", "sweep", "output"}
    difere = {k for k in BASE.keys() | EXTENDED.keys() if BASE.get(k) != EXTENDED.get(k)}
    assert difere == esperado
    assert EXTENDED["min_samples"] == 2048 and BASE["min_samples"] == 4096
    assert EXTENDED["samples_per_rev"] == 16 and BASE["samples_per_rev"] == 64
    # The extended list contains the whole base list: these are the nine control factors that
    # measure the guard step at an identical rate.
    assert set(BASE["sweep"]["undersampling_factor"]) - {96, 128} <= set(
        EXTENDED["sweep"]["undersampling_factor"]
    )
    assert EXTENDED["output"]["results_dir"] != BASE["output"]["results_dir"]


# ----------------------------------------------------------------------------
# Check with the real data: skipped when the .mat files are not present.
# ----------------------------------------------------------------------------
needs_data = pytest.mark.skipif(
    not (RAW / "I-A-1.mat").exists(), reason=f"Ottawa data missing in {RAW}"
)


@pytest.fixture(scope="module")
def segment_geometry() -> dict[int, list[tuple[int, float, float]]]:
    """For each factor, the real geometry of the 72 windows (24 records × 3 segments)."""
    from snq_data import effective_rate_hz, undersample
    from snq_data.ottawa import load, record_names

    factors = sorted(
        set(BASE["sweep"]["undersampling_factor"]) | set(EXTENDED["sweep"]["undersampling_factor"])
    )
    seg_n = int(BASE["segment_samples"])
    out: dict[int, list[tuple[int, float, float]]] = {f: [] for f in factors}
    names = record_names(
        tuple(BASE["conditions"]), tuple(BASE["scenarios"]), tuple(int(t) for t in BASE["trials"])
    )
    for name in names:
        rec = load(name, RAW)
        speed = rec.speed_profile_hz(int(BASE["pulses_per_rev"]))
        for seg in range(int(BASE["segments_per_record"])):
            a, b = seg * seg_n, (seg + 1) * seg_n
            if b > rec.vibration.size:
                break
            spd = speed[a:b]
            for factor in factors:
                v = undersample(spd, factor)
                fs_eff = effective_rate_hz(FS_HZ, factor)
                n = undersample(rec.vibration[a:b], factor).size
                out[factor].append((n, fs_eff, fs_eff / float(np.max(v))))
    return out


def admitidas(geom: dict[int, list[tuple[int, float, float]]], cfg: dict) -> dict[int, int]:
    c = Counter()
    for factor in cfg["sweep"]["undersampling_factor"]:
        c[int(factor)] = sum(run.guard_verdict(*g, cfg) is None for g in geom[int(factor)])
    return dict(c)


@needs_data
def test_geometria_das_janelas(segment_geometry):
    assert all(len(v) == 72 for v in segment_geometry.values())
    # The worst-case rotation speed used in the arithmetic above is the measured one, not a
    # choice.
    pior = min(pts for _, _, pts in segment_geometry[64])
    assert pior == pytest.approx(112.0, abs=0.01)


@needs_data
def test_admissao_base_reproduz_o_csv_medido(segment_geometry):
    """The guard model must reproduce the run that is in the CSV."""
    csv_path = EXP / "results/metrics.csv"
    if not csv_path.exists():
        pytest.skip("results/metrics.csv missing")
    medido = Counter()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            medido[int(row["factor"])] += 1
    modelo = admitidas(segment_geometry, BASE)
    # Three modes per admitted measurement.
    assert {f: n * 3 for f, n in modelo.items() if n} == dict(medido)
    assert modelo == {
        1: 72,
        2: 72,
        4: 72,
        8: 72,
        16: 72,
        24: 72,
        32: 72,
        48: 72,
        64: 45,
        96: 0,
        128: 0,
    }


@needs_data
def test_admissao_extended_e_n_cheio(segment_geometry):
    """ "72 of 72 at the twelve factors" — executed, and only on segment admission.

    This test says nothing about the hit rate: it says that no point of the extended curve
    enters with a reduced `n`, and therefore that a difference between factors cannot be
    a change of denominator.
    """
    modelo = admitidas(segment_geometry, EXTENDED)
    assert modelo == dict.fromkeys(EXTENDED["sweep"]["undersampling_factor"], 72)
    assert len(modelo) == 12
