"""E15 - learned sampling mask under a feasible set (ADR 0006, Part 2).

Question (README): on front-end A and at the same budget of samples/s, does a mask learned
against the task, restricted to a feasible set (periodic, K positions per period of L = K x
budget ticks, circular spacing >= g), outperform the best uniform one? The test is the one
PREREGISTERED in the README (section "Preregistration") and in the `preregistration` block of
the config.

Arms (`arm` column of metrics.csv)
----------------------------------
Classical (E07/E22 pipeline: kurtogram -> envelope -> order -> argmax), CPU only:
  uniform_budget              decimation at the budget factor (the `uniform` arm of E22)
  uniform_trained             factor >= budget with the most hits on the records outside the
                              test set, by the local ruler; one factor per dataset; tie ->
                              higher rate
  uniform_trained_per_race    same, one factor per group (uses the family of the test record)
  uniform_oracle[_per_race]   chosen WITH the test label: upper bound, not a method
Same unrolled decoder (`snq_learn.order_decoder`), fixed mask, one stage:
  dec_uniform, dec_random, dec_multicoset (minimum peak alias),
  dec_multicoset_vb (known support, venkataramani2001optimal),
  dec_uniform_trained         uniform mask at the `uniform_trained` factor
Searched mask (stage A, jointly with the decoder) and decoder RETRAINED FROM SCRATCH with the
final mask fixed (stage B), the same protocol as the fixed arms:
  dec_learned                 Gumbel top-K on the feasible set (B); _norefit: A projected
  dec_learned_sel             the dec_learned of the trained budget >= b with the best
                              validation
  dec_free                    Gumbel top-K WITHOUT the constraint (horizon M.L, no g): A, not
                              realizable; _projected (B) and _projected_norefit (A projected)
  dec_loupe_*                 LOUPE (optional), same structure
  dec_adaptive_blocks         block-adaptive, inspired by ADPS (one stage)
  wdcnn                       generic network (zhang2017wdcnn), family only, decimated signal

Rulers
------
- **local**: the line of the excerpt itself, at full rate, from the center of the score plateau
  (`snq_recon.refine_order`). Training and validation use the sliding grid of the training and
  validation records; the choice of the uniform factor uses the segments of the records outside
  the test set; scoring uses the test segment itself (only to score, never to choose). An
  excerpt whose plateau is at the edge of the refinement grid is left out.
- **nested**: median of the segment-0 calibrations of the group's records outside the test set.

    uv run --frozen python experiments/E15-learned-sampling-mask/run.py \
        --config experiments/E15-learned-sampling-mask/config/base.yaml

Classical-pipeline helpers copied from E20/E22, because nothing imports experiments/.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from snq_core import CWRU_6205_2RS_JEM_SKF, fault_frequencies_hz
from snq_data import CwruCatalog, download_record, ottawa, undersample
from snq_data import load_record as cwru_load
from snq_eval import classification_scores, record_splits
from snq_learn import (
    WDCNN,
    BlockAdaptiveMask,
    FeasibleSet,
    FitConfig,
    FitResult,
    FixedMask,
    GumbelTopKMask,
    LoupeMask,
    OrderAxis,
    OrderDecoder,
    SampledOrderModel,
    alias_weights,
    angular_plan,
    band_support_cells,
    designed_positions,
    envelope_sample_positions,
    fit,
    fold_project,
    kurtogram_bands_hz,
    noise_amplification,
    order_cross_entropy,
    random_feasible_positions,
    shrink_band_to_support,
    soft_order_target,
    stack_plans,
    support_designed_positions,
    timer_schedule,
    uniform_positions,
)
from snq_recon import (
    envelope_order_spectrum,
    estimate_without_prior,
    fast_kurtogram,
    fault_orders,
    phase_revolutions,
    refine_order,
    score_fault,
)

EXPERIMENT = "E15-learned-sampling-mask"
PACKAGES = ("snq-core", "snq-sim", "snq-data", "snq-recon", "snq-eval", "snq-learn")
REPO_ROOT = Path(__file__).resolve().parents[2]
README = Path(__file__).resolve().parent / "README.md"
FAULT_OF = {"inner_race": "BPFI", "outer_race": "BPFO"}
GROUP_OF_FAMILY = {"BPFI": "inner_race", "BPFO": "outer_race", "BSF": "ball", "FTF": "cage"}
HEALTHY = "healthy_bearing"
NO_FAMILY = "no_fault"
RULERS = ("local", "nested")
FIXED_METHODS = (
    "dec_uniform",
    "dec_random",
    "dec_multicoset",
    "dec_multicoset_vb",
    "dec_uniform_trained",
)
ARMS_OF_METHOD = {
    **{m: (m,) for m in FIXED_METHODS},
    "dec_learned": ("dec_learned", "dec_learned_norefit"),
    "dec_free": ("dec_free", "dec_free_projected", "dec_free_projected_norefit"),
    "dec_loupe": ("dec_loupe_free", "dec_loupe_projected", "dec_loupe_projected_norefit"),
    "dec_adaptive": ("dec_adaptive_blocks",),
    "wdcnn": ("wdcnn",),
}
SECONDARY = (  # descriptive comparisons, outside the preregistration: no correction
    ("dec_learned_sel", "uniform_oracle"),
    ("dec_learned_sel", "uniform_budget"),
    ("dec_learned", "dec_uniform"),
    ("dec_learned", "dec_multicoset"),
    ("dec_learned", "dec_multicoset_vb"),
    ("dec_learned", "dec_learned_norefit"),
    ("dec_free", "dec_free_projected"),
    ("dec_adaptive_blocks", "dec_learned_norefit"),
    ("dec_uniform_trained", "uniform_trained"),
    ("dec_uniform", "uniform_budget"),
)


def at_repo(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def derive_seed(*parts: object) -> int:
    """Stable integer seed from heterogeneous parts (sha256 of each part)."""
    words = [int(hashlib.sha256(str(p).encode()).hexdigest()[:8], 16) for p in parts]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


# The labels as the runs wrote them, before they were renamed to English
# (tools/relabel_map.py). Seeds are derived from these and the output files are sorted by them,
# so a rerun draws the same random numbers and writes its rows in the order of the versioned files.
LEGACY_LABELS = {
    "raw": "bruto",
    "despiked": "sem_picos",
    "uniform_budget": "uniforme_orcamento",
    "uniform_trained": "uniforme_treino",
    "uniform_trained_per_race": "uniforme_treino_por_grupo",
    "uniform_oracle": "oraculo_uniforme",
    "uniform_oracle_per_race": "oraculo_uniforme_por_grupo",
    "dec_uniform": "dec_uniforme",
    "dec_uniform_trained": "dec_uniforme_treino",
    "dec_random": "dec_aleatoria",
    "dec_learned": "dec_aprendida",
    "dec_learned_A": "dec_aprendida_A",
    "dec_learned_sel": "dec_aprendida_sel",
    "dec_learned_norefit": "dec_aprendida_sem_ajuste",
    "dec_free": "dec_livre",
    "dec_free_A": "dec_livre_A",
    "dec_free_projected": "dec_livre_projetada",
    "dec_free_projected_norefit": "dec_livre_projetada_sem_ajuste",
    "dec_loupe_free": "dec_loupe_livre",
    "dec_loupe_projected": "dec_loupe_projetada",
    "dec_loupe_projected_norefit": "dec_loupe_projetada_sem_ajuste",
    "dec_adaptive": "dec_adaptativa",
    "dec_adaptive_A": "dec_adaptativa_A",
    "dec_adaptive_blocks": "dec_adaptativa_blocos",
    "learned": "aprendida",  # mask column of exploratory/mask_stability.csv
    "free_projected": "livre_projetada",
}


def legacy_label(part: object) -> object:
    """A label as the runs wrote it; any other part is returned unchanged."""
    return LEGACY_LABELS.get(part, part) if isinstance(part, str) else part


def legacy_path(rel: str) -> str:
    """A ``runs/`` path (or ``masks.json`` key) with the labels the runs wrote."""
    *dirs, name = rel.split("/")
    stem, dot, ext = name.partition(".")
    return "/".join([*(str(legacy_label(d)) for d in dirs), f"{legacy_label(stem)}{dot}{ext}"])


def variant_name(despiked: bool) -> str:
    return "despiked" if despiked else "raw"


def finite(v: Any) -> bool:
    return v is not None and v != "" and bool(np.isfinite(float(v)))


# ----------------------------------------------------------------------
# classical pipeline (copied from E20/E22)
# ----------------------------------------------------------------------
def ceiling_at(ceiling0_hz: float | None, fs_eff_hz: float, frac: float = 1.0) -> float:
    base = fs_eff_hz / 2.0 if ceiling0_hz is None else min(float(ceiling0_hz), fs_eff_hz / 2.0)
    return base * frac


def guard_verdict(
    n_samples: int, fs_eff_hz: float, points_per_rev: float, cfg: dict, guard_factor: float
) -> str | None:
    """E22 guard, with a configurable points-per-revolution factor (E22: 2)."""
    if n_samples < int(cfg["min_samples"]):
        return "min_samples"
    if fs_eff_hz / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
        return "f_min_hz"
    if points_per_rev < guard_factor * int(cfg["samples_per_rev"]):
        return "samples_per_rev"
    return None


def kurtogram(x, fs_hz: float, f_max_hz: float, cfg: dict):
    kg = cfg["kurtogram"]
    return fast_kurtogram(
        x, fs_hz, max_level=int(kg["max_level"]), f_min_hz=float(kg["f_min_hz"]), f_max_hz=f_max_hz
    )


def despike(x: np.ndarray, k_mad: float = 10.0) -> tuple[np.ndarray, int]:
    """Replace isolated one-sample spikes with the mean of the neighbors (E20/E22 criterion).

    A deviation above ``k_mad`` MADs from the mean of the neighbors, with both neighbors
    deviating in the opposite direction and by less than 0.8 times as much. Returns the signal
    and how many samples were replaced. Requires the full rate: it is not realizable on the
    node, and that is why the deciding variant is `raw`.
    """
    d = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    mad = 1.4826 * np.median(np.abs(d))
    big = np.abs(d) > k_mad * mad
    dl = np.r_[0.0, d[:-1]]
    dr = np.r_[d[1:], 0.0]
    iso = (
        big
        & (np.sign(dl) == -np.sign(d))
        & (np.sign(dr) == -np.sign(d))
        & (np.abs(dl) < 0.8 * np.abs(d))
        & (np.abs(dr) < 0.8 * np.abs(d))
    )
    idx = np.flatnonzero(iso) + 1
    y = x.copy()
    y[idx] = 0.5 * (x[idx - 1] + x[idx + 1])
    return y, int(idx.size)


def classic(x, v, fs_hz: float, factor: int, f_max_hz: float, cfg: dict, guard: float) -> dict:
    """E07/E22 pipeline at one factor: estimate without prior and reason for discarding."""
    xd, vd = undersample(x, factor), undersample(v, factor)
    fs_eff = fs_hz / factor
    verdict = guard_verdict(xd.size, fs_eff, fs_eff / float(np.max(vd)), cfg, guard)
    if verdict:
        return {"skipped": verdict}
    if f_max_hz <= float(cfg["kurtogram"]["f_min_hz"]):
        return {"skipped": "ceiling below the kurtogram floor"}
    try:
        k = kurtogram(xd, fs_eff, f_max_hz, cfg)
        spec = envelope_order_spectrum(
            xd, fs_eff, vd, k.band_hz, samples_per_rev=int(cfg["samples_per_rev"])
        )
        est = estimate_without_prior(spec, float(cfg["order_min"]))
    except ValueError as exc:
        return {"skipped": f"ValueError: {str(exc)[:60]}"}
    return {"estimate": float(est)}


def calibrate(spec, nominal: float, cfg: dict) -> dict:
    """Order at the plateau center (`refine_order`) and, for the record, the 1st plateau point."""
    span = float(cfg["prior"]["calibration_span"])
    nh = int(cfg["prior"]["score_harmonics"])
    tol = float(cfg["tolerance_order"])
    step = 0.005
    center = refine_order(spec, nominal, span=span, n_harmonics=nh, tol=tol, step=step)
    grid = np.arange(nominal * (1 - span), nominal * (1 + span), step)
    scores = np.array(
        [score_fault(spec, float(f), n_harmonics=nh, tol_hz=tol).harmonic_energy for f in grid]
    )
    top = float(scores.max())
    at_max = np.flatnonzero(scores >= top - 1e-12 * max(abs(top), 1.0))
    # plateau touching the grid edge: refine_order says to treat it as a calibration failure
    at_edge = bool(at_max[0] == 0 or at_max[-1] == grid.size - 1 or top <= 0.0)
    return {
        "center": float(center),
        "first_point": float(grid[int(np.argmax(scores))]),
        "at_edge": at_edge,
    }


def dataset_cfg(cfg: dict, d: dict) -> dict:
    """Effective config of a dataset: the global one with the dataset's ``overrides`` on top.

    Without ``overrides`` (Ottawa, CWRU, MAFAULDA) it returns a copy equal to the global one.
    """
    out = copy.deepcopy(cfg)

    def merge(a: dict, b: dict) -> None:
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = copy.deepcopy(v)

    merge(out, d.get("overrides") or {})
    return out


def line_prominence(spec, nominal: float, center: float, cfg: dict) -> float:
    """Peak within ±0.015 order of the center over the median within ±``prominence_window``."""
    o, a = np.asarray(spec.freqs_hz), np.asarray(spec.magnitude)
    w = float(cfg.get("prominence_window", 0.07))
    win = (o >= nominal * (1 - w)) & (o <= nominal * (1 + w))
    near = np.abs(o - center) <= 0.015
    if not win.any() or not near.any():
        return math.nan
    med = float(np.median(a[win]))
    return float(a[near].max() / med) if med > 0 else math.nan


def full_rate_calibration(x, v, fs_hz: float, nominal: float, cfg: dict, c0) -> dict:
    """Full-rate line of an excerpt: kurtogram (ceiling ``c0``), envelope, order and refinement."""
    try:
        k = kurtogram(x, fs_hz, ceiling_at(c0, fs_hz), cfg)
        spr = int(cfg["samples_per_rev"])
        spec = envelope_order_spectrum(x, fs_hz, v, k.band_hz, samples_per_rev=spr)
        out = {**calibrate(spec, nominal, cfg), "band_hz": [k.f_lo_hz, k.f_hi_hz]}
        thr = cfg.get("min_line_prominence")
        if thr is not None:  # a line that does not stand out is not valid as a ruler (Paderborn)
            prom = line_prominence(spec, nominal, out["center"], cfg)
            out["prominence"] = prom
            out["at_edge"] = bool(out["at_edge"] or not (np.isfinite(prom) and prom >= float(thr)))
        return out
    except ValueError as exc:
        return {"center": math.nan, "first_point": math.nan, "at_edge": True, "error": str(exc)}


# ----------------------------------------------------------------------
# datasets (adapters)
# ----------------------------------------------------------------------
@dataclass
class Record:
    dataset: str
    rid: str
    group: str
    cluster: str
    fs_hz: float
    x: np.ndarray
    speed_hz: np.ndarray
    cal: tuple[int, int]
    tests: list[tuple[int, int, int]]
    nominal: float | None
    path: Path
    files: tuple[Path, ...] = ()


def _cwru_number(rid: str) -> tuple[int, str]:
    stem = rid.rstrip("DEFBA")
    return int(stem), rid[len(stem) :]


def list_ids(d: dict, healthy: bool = False) -> list[str]:
    """Identifiers of the dataset's fault records (or of its healthy control)."""
    kind = d["kind"]
    if kind == "ottawa":
        if healthy:
            h = d.get("healthy_control") or {}
            if not h.get("enabled", False):
                return []
            return ottawa.record_names(
                tuple(h.get("conditions", ["H"])),
                tuple(h.get("scenarios", d["scenarios"])),
                tuple(int(t) for t in h.get("trials", d["trials"])),
            )
        return ottawa.record_names(
            tuple(d["conditions"]), tuple(d["scenarios"]), tuple(int(t) for t in d["trials"])
        )
    if kind == "paderborn":
        if healthy:
            h = d.get("healthy_control") or {}
            if not h.get("enabled", False):
                return []
            return [f"{c}_{s}" for c in h["bearings"] for s in d["settings"]]
        return [f"{c}_{s}" for c in [*d["outer"], *d["inner"]] for s in d["settings"]]
    if healthy:
        return []
    if kind == "cwru":
        cat = CwruCatalog.load(at_repo(d["catalog"]))
        ids = [f"{n}{p}" for n, p, _ in cat.selection(d["table"], d["subset"], d["point"])]
        only = {str(r) for r in d.get("records") or []}
        return [i for i in ids if not only or i in only or str(_cwru_number(i)[0]) in only]
    if kind == "mafaulda":
        from snq_data import mafaulda as maf

        screen = set(
            yaml.safe_load(at_repo(d["screen_catalog"]).read_text(encoding="utf-8"))["excluded"]
        )
        return [
            r
            for r in maf.list_records(at_repo(d["data_dir"]), subsets=tuple(d["bearings"]))
            if Path(r).parts[1] == d["fault"] and r not in screen
        ]
    raise ValueError(f"unknown dataset kind: {kind}")


def load_record(ds: str, d: dict, rid: str) -> Record:
    kind = d["kind"]
    if kind == "ottawa":
        raw = at_repo(d["data_dir"])
        path = ottawa.download(rid, raw)
        rec = ottawa.load(rid, raw)
        n = int(d["segment_samples"])
        c = int(d["calibration_segment"])
        nominal = None
        if rec.condition in FAULT_OF:
            nom = d["nominal"]
            orders = fault_orders(float(nom["geom_ratio"]), int(nom["n_elements"]))
            nominal = orders[FAULT_OF[rec.condition]]
        return Record(
            dataset=ds,
            rid=rid,
            group=rec.condition,
            cluster=rid.rsplit("-", 1)[0],  # condition x scenario: the bootstrap cluster
            fs_hz=ottawa.FS_HZ,
            x=rec.vibration,
            speed_hz=rec.speed_profile_hz(int(d["pulses_per_rev"])),
            cal=(c * n, (c + 1) * n),
            tests=[(int(s), int(s) * n, (int(s) + 1) * n) for s in d["test_segments"]],
            nominal=nominal,
            path=Path(path),
        )
    if kind == "cwru":
        raw = at_repo(d["data_dir"])
        cat = CwruCatalog.load(at_repo(d["catalog"]))
        number, point = _cwru_number(rid)
        family = cat.fault_of(number, point)
        if family is None:
            raise ValueError(f"{rid}: not in the catalog")
        path = download_record(number, raw)
        rec = cwru_load(number, point, raw, cat, fs_hz=float(d["fs_hz"]))
        if rec.shaft_hz is None:
            raise ValueError(f"{rid}: no RPM in the file")
        h = rec.signal.size // 2
        orders = fault_frequencies_hz(CWRU_6205_2RS_JEM_SKF, 1.0)  # Hz per shaft Hz = orders
        return Record(
            dataset=ds,
            rid=rid,
            group=GROUP_OF_FAMILY[family[1]],
            cluster=family[0],
            fs_hz=rec.fs_hz,
            x=rec.signal,
            speed_hz=np.full(rec.signal.size, rec.shaft_hz),
            cal=(0, h),
            tests=[(1, h, rec.signal.size)],
            nominal=float(orders[family[1]]),
            path=Path(path),
        )
    if kind == "mafaulda":
        from snq_data import mafaulda as maf

        root = at_repo(d["data_dir"])
        mrec = maf.load(rid, root, cache_dir=at_repo(d["cache_dir"]))
        x = mrec.vibration(axis=d["axis"])
        h = x.size // 2
        return Record(
            dataset=ds,
            rid=rid,
            group=mrec.fault,
            cluster=rid,
            fs_hz=mrec.fs_hz,
            x=x,
            speed_hz=mrec.speed_profile_hz(),
            cal=(0, h),
            tests=[(1, h, x.size)],
            nominal=float(maf.DOCUMENTED_ORDERS[d["fault_family"]]),
            path=root / rid,
        )
    if kind == "paderborn":
        return paderborn_record(ds, d, rid)
    raise ValueError(f"unknown dataset kind: {kind}")


def paderborn_record(ds: str, d: dict, rid: str) -> Record:
    """Bearing x setting: the measurements concatenated, each one a segment of ``segment_samples``.

    An unreadable or short measurement (`snq_data.paderborn`) is left out. The training window and
    the local ruler use ``window_samples = segment_samples`` and an equal hop, aligned with the
    measurements.
    """
    from snq_data import paderborn as pb

    root = at_repo(d["data_dir"])
    code, setting = rid.split("_", 1)
    n_seg = int(d["segment_samples"])
    xs, vs, spans, files, start = [], [], {}, [], 0
    for n in [int(d["calibration_measurement"]), *[int(t) for t in d["test_measurements"]]]:
        try:
            m = pb.load_measurement(root, code, setting, n, n_samples=n_seg, min_samples=n_seg)
        except ValueError:
            continue
        xs.append(m.vibration)
        vs.append(m.speed_profile_hz())
        spans[n] = (start, start + n_seg)
        files.append(pb.measurement_path(root, code, setting, n))
        start += n_seg
    c = int(d["calibration_measurement"])
    if c not in spans:
        raise ValueError(f"{rid}: unreadable calibration measurement")
    race = pb.BEARINGS[code][1]
    orders = pb.computed_orders()
    nominal = None if race == "none" else float(orders["BPFO" if race == "OR" else "BPFI"])
    return Record(
        dataset=ds,
        rid=rid,
        group="H" if race == "none" else race,
        cluster=code,  # the bearing: the bootstrap cluster
        fs_hz=pb.FS_HZ,
        x=np.concatenate(xs),
        speed_hz=np.concatenate(vs),
        cal=spans[c],
        tests=[(n, a, b) for n, (a, b) in spans.items() if n != c],
        nominal=nominal,
        path=root / code,
        files=tuple(files),
    )


def record_sha256(rec: Record) -> str:
    """sha256 of the record's file, or of the files that make it up (Paderborn)."""
    if not rec.files:
        return hashlib.sha256(rec.path.read_bytes()).hexdigest()
    h = hashlib.sha256()
    for p in rec.files:
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


# ----------------------------------------------------------------------
# classical stage (CPU, in parallel): full-rate rulers and uniform grid
# ----------------------------------------------------------------------
def classical_task(task: dict) -> dict:
    cfg, ds, rid = task["cfg"], task["dataset"], task["rid"]
    d = cfg["datasets"][ds]
    rec = load_record(ds, d, rid)
    fs = rec.fs_hz
    c0 = d.get("ceiling0_hz")
    out: dict[str, Any] = {
        "dataset": ds,
        "rid": rid,
        "group": rec.group,
        "cluster": rec.cluster,
        "sha256": record_sha256(rec),
        "grid": {},
        "despiked_samples": {},
    }
    if rec.nominal is not None:  # rulers always on the raw signal: truth does not vary by variant
        a, b = rec.cal
        out["calib"] = full_rate_calibration(
            rec.x[a:b], rec.speed_hz[a:b], fs, rec.nominal, cfg, c0
        )
        w = int(d["window_samples"])
        hop = int(d["local_calibration"]["hop_samples"])
        local = []
        for start in range(0, rec.x.size - w + 1, hop):
            r = full_rate_calibration(
                rec.x[start : start + w], rec.speed_hz[start : start + w], fs, rec.nominal, cfg, c0
            )
            local.append([start, r["center"], r["at_edge"]])
        out["local_grid"] = local
        out["segment_truth"] = {}
        for seg, a, b in rec.tests:
            r = full_rate_calibration(rec.x[a:b], rec.speed_hz[a:b], fs, rec.nominal, cfg, c0)
            out["segment_truth"][seg] = [r["center"], r["at_edge"]]
    grid = range(min(int(b) for b in d["budgets"]), int(d["factor_grid_max"]) + 1)
    guard = float(d.get("guard_points_per_rev_factor", 2.0))
    cfg_ds = {**cfg, "min_samples": int(d.get("min_samples", cfg["min_samples"]))}
    for despiked in task["variants"]:
        x, n_rep = despike(rec.x, float(cfg["despike"]["k_mad"])) if despiked else (rec.x, 0)
        res = {}
        for seg, a, b in rec.tests:
            for m in grid:
                res[(seg, m)] = classic(
                    x[a:b], rec.speed_hz[a:b], fs, m, ceiling_at(c0, fs / m), cfg_ds, guard
                )
        out["grid"][variant_name(despiked)] = res
        out["despiked_samples"][variant_name(despiked)] = n_rep
    return out


# ----------------------------------------------------------------------
# rulers, family and choice of the best uniform factor
# ----------------------------------------------------------------------
@dataclass
class RowCtx:
    """What a metrics.csv row needs to know about its cell."""

    ds: str
    variant: str
    split: dict
    budget: int
    seed: int | str
    nested: dict[str, dict[str, Any]]
    seg_truth: dict[tuple[str, int], float]
    centers: dict[str, float]
    tol: float
    span: float


def nested_truths(
    recs: list[str],
    group_of: dict[str, str],
    calib: dict[str, float],
    ok: dict[str, bool],
    test: set[str],
) -> dict[str, dict[str, Any]]:
    """Nested ruler: median of the segment-0 calibrations outside the test set, per group.

    Outside the test set: leave-one-out among the records outside the test set; with no peers,
    the record's own calibration, flagged. In the test set: median of the group's records
    outside the test set.
    """
    pool = [r for r in recs if r not in test and ok[r]]
    out: dict[str, dict[str, Any]] = {}
    for r in recs:
        peers = [calib[q] for q in pool if q != r and group_of[q] == group_of[r]]
        if peers:
            out[r] = {"truth": float(np.median(peers)), "source": "nested"}
        elif r not in test and ok[r]:
            out[r] = {"truth": calib[r], "source": "own_calibration"}
        else:
            out[r] = {"truth": math.nan, "source": "undefined"}
    return out


def family_centers(
    recs: list[str],
    group_of: dict[str, str],
    calib: dict[str, float],
    ok: dict[str, bool],
    test: set[str],
) -> dict[str, float]:
    """Order of each family (group), from the median of the calibrations outside the test set."""
    out: dict[str, float] = {}
    for g in sorted({group_of[r] for r in recs}):
        vals = [calib[r] for r in recs if r not in test and ok[r] and group_of[r] == g]
        if vals:
            out[g] = float(np.median(vals))
    return out


def family_of_estimate(est: float | None, centers: dict[str, float], span: float) -> str:
    """Family with the nearest central order, within a relative ``±span``; otherwise 'no_fault'."""
    if est is None or not np.isfinite(est):
        return NO_FAMILY
    best, dist = NO_FAMILY, math.inf
    for g, c in centers.items():
        dd = abs(est - c)
        if dd <= span * c and dd < dist:
            best, dist = g, dd
    return best


def hits_at(
    factor: int,
    recs: list[str],
    truth: dict[tuple[str, int], float],
    est: dict,
    segs: dict[str, list[int]],
    tol: float,
) -> tuple[int, int]:
    """Hits and total at one factor, by the per-segment ruler ``truth``.

    A segment without a ruler (plateau at the edge) is left out; a segment discarded by the
    pipeline's guard counts as a miss.
    """
    h = n = 0
    for r in recs:
        for s in segs[r]:
            t = truth.get((r, s))
            if t is None or not np.isfinite(t):
                continue
            n += 1
            e = est.get((r, s, factor))
            h += int(e is not None and abs(e - t) < tol)
    return h, n


def choose_factor(
    budget: int,
    factors: list[int],
    recs: list[str],
    truth: dict[tuple[str, int], float],
    est: dict,
    segs: dict[str, list[int]],
    tol: float,
) -> int | None:
    """Factor >= budget with the most hits on ``recs``; a tie goes to the higher rate."""
    best: tuple[float, int | None] = (-1.0, None)
    for f in sorted(f for f in factors if f >= budget):  # ascending: ties go to the higher rate
        h, n = hits_at(f, recs, truth, est, segs, tol)
        if n and h / n > best[0]:
            best = (h / n, f)
    return best[1]


def row_base(rc: RowCtx, arm: str, rid: str, group: str, seg: int) -> dict:
    return {
        "dataset": rc.ds,
        "variant": rc.variant,
        "budget": rc.budget,
        "arm": arm,
        "split": rc.split["index"],
        "split_seed": rc.split["seed"],
        "seed": rc.seed,
        "record": rid,
        "group": group,
        "segment": seg,
    }


def score_row(row: dict, est: float | None, rc: RowCtx, rid: str, seg: int) -> dict:
    """Hit by both rulers, family and, if healthy, whether the estimate looks like a fault."""
    ok = est is not None and bool(np.isfinite(est))
    row.update(
        estimate=round(float(est), 4) if ok else "",
        family_pred=family_of_estimate(est, rc.centers, rc.span),
    )
    if rid in rc.nested:  # faulty record
        tl = rc.seg_truth.get((rid, seg), math.nan)
        tn = rc.nested[rid]["truth"]
        row.update(
            truth_local=round(tl, 4) if np.isfinite(tl) else "",
            hit_local=int(ok and abs(est - tl) < rc.tol) if np.isfinite(tl) else "",
            truth_nested=round(tn, 4) if np.isfinite(tn) else "",
            truth_nested_source=rc.nested[rid]["source"],
            hit_nested=int(ok and abs(est - tn) < rc.tol) if np.isfinite(tn) else "",
            family_true=row["group"],
        )
    else:  # healthy control: there is no truth; measures whether the method "finds" a fault order
        row.update(
            fault_like=int(ok and any(abs(est - c) < rc.tol for c in rc.centers.values())),
            family_true=HEALTHY,
        )
    return row


def classical_rows(
    rc: RowCtx,
    grid_est: dict,
    budgets: list[int],
    factors: list[int],
    group_of: dict[str, str],
    segs: dict[str, list[int]],
    healthy: list[str],
) -> tuple[list[dict], dict[int, int | None]]:
    """Classical arms of one (dataset, variant, split), and the `uniform_trained` factor.

    The choice outside the test set uses the local ruler of the segments of the records outside
    the test set; the oracle uses that of the test records.
    """
    split = rc.split
    nontest, test = split["train"] + split["val"], split["test"]
    groups = sorted({group_of[r] for r in nontest + test})
    rows: list[dict] = []
    chosen: dict[int, int | None] = {}
    for b in budgets:
        rcb = replace(rc, budget=b)

        def pick(recs: list[str], b: int = b) -> int | None:
            return choose_factor(b, factors, recs, rc.seg_truth, grid_est, segs, rc.tol)

        chosen[b] = m_train = pick(nontest)
        arms: dict[str, tuple[Any, bool]] = {  # arm -> (factor or per-group factor, healthy?)
            "uniform_budget": (b, True),
            "uniform_trained": (m_train, True),
            "uniform_trained_per_race": (
                {g: pick([r for r in nontest if group_of[r] == g]) for g in groups},
                False,
            ),
            "uniform_oracle": (pick(test), False),
            "uniform_oracle_per_race": (
                {g: pick([r for r in test if group_of[r] == g]) for g in groups},
                False,
            ),
        }
        for arm, (choice, with_healthy) in arms.items():
            for rid in test + (healthy if with_healthy else []):
                g = group_of[rid]
                f = choice.get(g) if isinstance(choice, dict) else choice
                for seg in segs[rid]:
                    row = row_base(rcb, arm, rid, g, seg)
                    row["chosen_factor"] = f if f is not None else ""
                    est = grid_est.get((rid, seg, f)) if f is not None else None
                    if est is None:
                        row["skipped"] = grid_est.get(("skip", rid, seg, f), "no factor chosen")
                    rows.append(score_row(row, est, rcb, rid, seg))
    return rows, chosen


# ----------------------------------------------------------------------
# per-record splits (E05): test set, then validation within the rest
# ----------------------------------------------------------------------
def make_splits(recs: list[str], group_of: dict[str, str], pcfg: dict, seed: int) -> list[dict]:
    groups = sorted({group_of[r] for r in recs})
    label = {g: i for i, g in enumerate(groups)}
    outer = record_splits(
        recs,
        [label[group_of[r]] for r in recs],
        int(pcfg["n_splits"]),
        float(pcfg["test_fraction"]),
        seed,
        n_classes=len(groups),
        stratify=True,
        min_test_per_class=1,
    )
    out = []
    for s in outer:
        rest = list(s.train_records)
        inner = record_splits(
            rest,
            [label[group_of[r]] for r in rest],
            1,
            float(pcfg["val_fraction"]),
            s.seed,
            n_classes=len(groups),
            stratify=True,
            min_test_per_class=1,
        )[0]
        out.append(
            {
                "index": s.index,
                "seed": s.seed,
                "train": list(inner.train_records),
                "val": list(inner.test_records),
                "test": list(s.test_records),
                "never_tested": [groups[c] for c in s.classes_never_tested],
            }
        )
    return out


# ----------------------------------------------------------------------
# windows and batches
# ----------------------------------------------------------------------
class GuardError(ValueError):
    """Window outside the decoder's conditions (rotation speed too high or too low)."""


@dataclass
class Bank:
    """Records in memory for one variant (raw or despiked), signal on the device."""

    recs: dict[str, Record]
    x: dict[str, torch.Tensor]
    phase: dict[str, np.ndarray]
    device: torch.device


def kfold_by_cluster(
    d: dict, recs: list[str], cluster_of: dict[str, str], pcfg: dict, seed: int
) -> list[dict]:
    """K-fold by cluster (bearing), stratified by race: each one goes to the test set once.

    The folds repeat those of the Paderborn power gate (`paderborn_gate.kfold_by_bearing`), with
    the same seed and the same order of ``outer`` and ``inner``. Validation comes from the
    clusters outside the test set: ``val_fraction`` per race, at least 1.
    """
    sp = d["splits"]
    k = int(sp["n_folds"])
    rng = np.random.default_rng(derive_seed(int(sp["split_seed"]), "paderborn", "splits"))
    folds: list[list[str]] = [[] for _ in range(k)]
    for codes in (d["outer"], d["inner"]):
        for i, c in enumerate(rng.permutation(codes)):
            folds[i % k].append(str(c))
    out = []
    for f in range(k):
        test_c = set(folds[f])
        rng_v = np.random.default_rng(derive_seed(seed, "val", f))
        val_c: set[str] = set()
        for codes in (d["outer"], d["inner"]):
            rest = [c for c in codes if c not in test_c]
            n_val = max(1, int(round(float(pcfg["val_fraction"]) * len(rest))))
            val_c.update(str(c) for c in rng_v.choice(rest, size=n_val, replace=False))
        out.append(
            {
                "index": f,
                "seed": derive_seed(seed, "fold", f),
                "train": [r for r in recs if cluster_of[r] not in test_c | val_c],
                "val": [r for r in recs if cluster_of[r] in val_c],
                "test": [r for r in recs if cluster_of[r] in test_c],
                "test_clusters": sorted(test_c),
                "val_clusters": sorted(val_c),
                "never_tested": [],
            }
        )
    return out


def make_bank(recs: dict[str, Record], despiked: bool, k_mad: float, device: torch.device) -> Bank:
    xs, ph = {}, {}
    for rid, r in recs.items():
        x = despike(r.x, k_mad)[0] if despiked else r.x
        xs[rid] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        ph[rid] = phase_revolutions(r.speed_hz, r.fs_hz, r.x.size)
    return Bank(recs, xs, ph, device)


def window_plan(bank: Bank, rid: str, start: int, n: int, dec: OrderDecoder) -> dict:
    """Angular plan of a window, with the guard on the decoder's conditions.

    The window phase is the phase of the whole record (trapezoidal integral of the rotation
    speed) minus the phase at the window start, the same as `phase_revolutions` applied to the
    window alone.
    """
    r = bank.recs[rid]
    v = r.speed_hz[start : start + n]
    vmin, vmax = float(np.min(v)), float(np.max(v))
    n_env = dec.n_env(n)
    fs_env = n_env * r.fs_hz / n
    if fs_env / 2.0 < dec.axis.order_max * vmax:
        raise GuardError(
            f"reduced envelope does not cover order {dec.axis.order_max} at {vmax:.1f} Hz"
        )
    if dec.axis.samples_per_rev < fs_env / vmin:
        raise GuardError(f"angular spr {dec.axis.samples_per_rev} < {fs_env / vmin:.0f} points/rev")
    pos = start + envelope_sample_positions(n, n_env)
    ph = bank.phase[rid]
    base = np.floor(pos).astype(np.int64)
    nxt = np.minimum(base + 1, ph.size - 1)
    th = ph[base] + (pos - base) * (ph[nxt] - ph[base]) - ph[start]
    try:
        return angular_plan(th, dec.axis.samples_per_rev, dec.axis.n_fft)
    except ValueError as exc:
        raise GuardError(str(exc)) from exc


def fixed_windows(bank: Bank, rids: list[str], period: int) -> list[tuple[str, int, int, int]]:
    """Evaluation windows: the test segments, trimmed to a multiple of ``period``."""
    out = []
    for rid in rids:
        for seg, a, b in bank.recs[rid].tests:
            out.append((rid, seg, a, (b - a) - (b - a) % period))
    return out


def train_pick(
    rng: np.random.Generator, pool: dict[str, dict[str, list[tuple[int, float]]]]
) -> tuple[str, int, float]:
    """Training window: group, record within the group and grid excerpt, each drawn uniformly."""
    groups = sorted(pool)
    by_rec = pool[groups[int(rng.integers(len(groups)))]]
    rids = sorted(by_rec)
    rid = rids[int(rng.integers(len(rids)))]
    start, label = by_rec[rid][int(rng.integers(len(by_rec[rid])))]
    return rid, int(start), float(label)


def wdcnn_pick(
    rng: np.random.Generator, by_group: dict[str, list[str]], bank: Bank, n: int
) -> tuple[str, int]:
    """WDCNN window: group, record and start each drawn uniformly (label = family)."""
    groups = sorted(by_group)
    pool = by_group[groups[int(rng.integers(len(groups)))]]
    rid = pool[int(rng.integers(len(pool)))]
    return rid, int(rng.integers(0, bank.recs[rid].x.size - n + 1))


# ----------------------------------------------------------------------
# learned methods
# ----------------------------------------------------------------------
@dataclass
class Ctx:
    cfg: dict
    d: dict
    rc: RowCtx
    fs_set: FeasibleSet
    bank: Bank
    pool: dict[str, dict[str, list[tuple[int, float]]]]
    val_labels: dict[tuple[str, int], float]
    groups: list[str]
    healthy: list[str]
    device: torch.device
    data_seed: int
    max_epochs: int | None
    uniform_factor: int | None
    known_band: tuple[float, float] | None


def make_decoder(ctx: Ctx, fs_hz: float, factor: int, period: int) -> OrderDecoder:
    """Decoder initialized on the kurtogram bands at the rate ``fs/factor``."""
    dc, dd = ctx.cfg["decoder"], ctx.d["decoder"]
    f_hi = ceiling_at(ctx.d.get("ceiling0_hz"), fs_hz / factor)
    bands = kurtogram_bands_hz(float(ctx.cfg["kurtogram"]["f_min_hz"]), f_hi, dc["init_levels"])
    axis = OrderAxis(
        int(dd["spr_ang"]),
        int(dd["n_fft"]),
        float(ctx.cfg["order_min"]),
        float(ctx.cfg["order_max"]),
    )
    return OrderDecoder(
        fs_hz=fs_hz,
        fs_env_hz=float(dd["fs_env_hz"]),
        axis=axis,
        bands_hz=bands,
        shifts=tuple(int(s) for s in dc["shifts"]),
        period_ticks=period,
        n_control=int(dd.get("n_control", dc["n_control"])),
        transition_frac=float(dc["transition_frac"]),
        envelope_power=int(dc["envelope_power"]),
        selection=str(dc["selection"]),
        sk_guard_frac=float(dc["sk_guard_frac"]),
        beta_init=float(dc["beta_init"]),
    ).to(ctx.device)


def fit_cfg(ctx: Ctx, lr: float) -> FitConfig:
    t = ctx.cfg["training"]
    return FitConfig(
        max_epochs=int(ctx.max_epochs or t["max_epochs"]),
        patience=int(t["patience"]),
        lr=float(lr),
        weight_decay=float(t["weight_decay"]),
        grad_clip=float(t["grad_clip"]) if t.get("grad_clip") is not None else None,
    )


def _scrub(model: torch.nn.Module) -> torch.nn.Module:
    """Drops diagnostic tensors attached to the graph, so that ``deepcopy`` works."""
    for m in model.modules():
        if isinstance(getattr(m, "last_penalty", None), torch.Tensor):
            m.last_penalty = torch.zeros(())
    return model


class OrderTask:
    """Batches, loss, validation and evaluation of an order model in one cell.

    ``trim`` is the period the windows are trimmed to: the mask's (``M·L`` for the free mask) is a
    multiple of the filter bank's period ``L``.
    """

    def __init__(self, ctx: Ctx, decoder: OrderDecoder, trim: int) -> None:
        self.ctx = ctx
        t = ctx.cfg["training"]
        w = int(ctx.d["window_samples"])
        self.n_train = w - w % trim
        self.batch = int(t["batch_size"])
        self.steps = max(1, int(t["windows_per_epoch"]) // self.batch)
        self.sigma = float(ctx.cfg["tolerance_order"]) / 2.0
        self.lam = float(t["spacing_penalty"])
        if not ctx.pool:
            raise ValueError("no training window with a valid local ruler")
        self.decoder_ref = decoder
        val = sorted({rid for rid, _seg in ctx.val_labels})
        self.val_windows = [
            w_ for w_ in fixed_windows(ctx.bank, val, trim) if (w_[0], w_[1]) in ctx.val_labels
        ]
        self.test_windows = fixed_windows(ctx.bank, ctx.rc.split["test"], trim)
        self.healthy_windows = fixed_windows(ctx.bank, ctx.healthy, trim)
        self._plans: dict[tuple[str, int, int], dict] = {}

    def plan(self, rid: str, start: int, n: int) -> dict:
        key = (rid, start, n)
        if key not in self._plans:
            self._plans[key] = window_plan(self.ctx.bank, rid, start, n, self.decoder_ref)
        return self._plans[key]

    def batches(self, epoch: int) -> Iterator[tuple]:
        rng = np.random.default_rng([self.ctx.data_seed, epoch])
        bank = self.ctx.bank
        for _ in range(self.steps):
            xs, plans, truth = [], [], []
            tries = 0
            while len(xs) < self.batch:
                tries += 1
                if tries > 100 * self.batch:
                    raise RuntimeError("too many training windows rejected by the guard")
                rid, start, label = train_pick(rng, self.ctx.pool)
                try:
                    p = self.plan(rid, start, self.n_train)
                except GuardError:
                    continue
                xs.append(bank.x[rid][start : start + self.n_train])
                plans.append(p)
                truth.append(label)
            yield (
                torch.stack(xs),
                stack_plans(plans).to(bank.device),
                torch.tensor(truth, dtype=torch.float32, device=bank.device),
            )

    def loss(self, model: SampledOrderModel, batch: tuple) -> torch.Tensor:
        x, plan, truth = batch
        logits, _ = model(x, plan)
        target = soft_order_target(model.decoder.orders, truth, self.sigma)
        return order_cross_entropy(logits, target) + self.lam * model.penalty().to(logits.device)

    def single(self, rid: str, start: int, n: int) -> tuple[torch.Tensor, Any]:
        bank = self.ctx.bank
        plan = stack_plans([self.plan(rid, start, n)]).to(bank.device)
        return bank.x[rid][start : start + n].unsqueeze(0), plan

    def val_loss(self, model: SampledOrderModel) -> float:
        losses = []
        for rid, seg, a, n in self.val_windows:
            try:
                x, plan = self.single(rid, a, n)
            except GuardError:
                continue
            truth = torch.tensor([self.ctx.val_labels[(rid, seg)]], device=x.device)
            losses.append(float(self.loss(model, (x, plan, truth)).detach()))
        return float(np.mean(losses)) if losses else math.inf

    def val_metrics(self, model: SampledOrderModel) -> dict[str, float]:
        """Validation loss and balanced order accuracy, by the local ruler."""
        model.eval()
        hits: dict[str, list[int]] = defaultdict(list)
        with torch.no_grad():
            for rid, seg, a, n in self.val_windows:
                try:
                    x, plan = self.single(rid, a, n)
                except GuardError:
                    continue
                logits, _ = model(x, plan)
                est = float(model.decoder.estimate(logits)[0])
                hit = abs(est - self.ctx.val_labels[(rid, seg)]) < self.ctx.rc.tol
                hits[self.ctx.bank.recs[rid].group].append(int(hit))
            v = self.val_loss(model)
        bal = float(np.mean([np.mean(h) for h in hits.values()])) if hits else math.nan
        return {"val_loss": v, "val_bal_acc": bal}

    def evaluate(self, model: SampledOrderModel, arm: str, extra: dict) -> list[dict]:
        """metrics.csv rows on the test records and on the healthy control."""
        rc = self.ctx.rc
        model.eval()
        rows = []
        for rid, seg, a, n in self.test_windows + self.healthy_windows:
            row = {**row_base(rc, arm, rid, self.ctx.bank.recs[rid].group, seg), **extra}
            try:
                x, plan = self.single(rid, a, n)
            except GuardError as exc:
                row["skipped"] = f"guard: {str(exc)[:60]}"
                rows.append(score_row(row, None, rc, rid, seg))
                continue
            with torch.no_grad():
                logits, aux = model(x, plan)
            row.update(
                realized_rate=round(float(aux["mask_rate"][0]), 6),
                n_samples=n,
                selection_max_weight=round(float(aux["weights"].max()), 4),
            )
            dropped = getattr(model.sampler, "last_dropped", None)
            if dropped is not None:
                row["adps_dropped"] = int(dropped)
            rows.append(score_row(row, float(model.decoder.estimate(logits)[0]), rc, rid, seg))
        return rows


def mask_info(positions: np.ndarray, period: int, ctx: Ctx, fs_hz: float) -> dict:
    clock = float(ctx.cfg["feasible"]["timer_clock_hz"])
    return {
        "positions": " ".join(str(int(p)) for p in positions),
        "period_ticks": period,
        "peak_alias": round(float(alias_weights(positions, period)[1:].max()), 4),
        "timer": timer_schedule(positions, period, fs_hz, clock),
    }


_DESIGNED: dict[tuple, np.ndarray] = {}


def designed_mask(fs_set: FeasibleSet, cfg: dict) -> np.ndarray:
    mc = cfg["multicoset"]
    key = (fs_set.period_ticks, fs_set.active_per_period, fs_set.min_gap_ticks, mc["criterion"])
    if key not in _DESIGNED:
        _DESIGNED[key] = designed_positions(
            fs_set, str(mc["criterion"]), max_enumeration=int(mc["max_enumeration"])
        )
    return _DESIGNED[key]


def vb_mask(ctx: Ctx, fs_hz: float) -> tuple[np.ndarray | None, dict]:
    """Multi-coset for the known support (`venkataramani2001optimal`), see `masks`.

    The support is the most frequent full-rate kurtogram band among the records outside the
    test set (training data only), shrunk around its center until it fits in K cells.
    """
    fs_set = ctx.fs_set
    if ctx.known_band is None:
        return None, {"vb_error": "no known band"}
    try:
        band = shrink_band_to_support(
            ctx.known_band, fs_hz, fs_set.period_ticks, fs_set.active_per_period
        )
    except ValueError as exc:
        return None, {"vb_error": str(exc)}
    sup = band_support_cells(band, fs_hz, fs_set.period_ticks)
    key = (fs_set.period_ticks, fs_set.active_per_period, fs_set.min_gap_ticks, band)
    if key not in _DESIGNED:
        _DESIGNED[key] = support_designed_positions(
            fs_set, sup, max_enumeration=int(ctx.cfg["multicoset"]["max_enumeration"])
        )
    pos = _DESIGNED[key]
    uni = uniform_positions(fs_set)
    return pos, {
        "known_band_hz": list(ctx.known_band),
        "designed_band_hz": [round(v, 1) for v in band],
        "support_cells": [[w, list(s)] for w, s in sup],
        "noise_amplification": noise_amplification(pos, fs_set.period_ticks, sup),
        "noise_amplification_uniform": noise_amplification(uni, fs_set.period_ticks, sup),
    }


def fixed_positions(method: str, ctx: Ctx, fs_hz: float) -> tuple[Any, ...]:
    """Pattern, feasible set, rate factor and information of a fixed-mask arm."""
    fs_set, b = ctx.fs_set, ctx.rc.budget
    if method == "dec_uniform":
        return uniform_positions(fs_set), fs_set, b, {}
    if method == "dec_random":  # the same mask for all seeds and learning rates
        # "aleatoria" (like "dados" below) is a seed tag of the runs and keeps its name
        rng = np.random.default_rng(
            derive_seed(ctx.cfg["seed"], ctx.rc.ds, ctx.rc.split["index"], b, "aleatoria")
        )
        return random_feasible_positions(fs_set, rng), fs_set, b, {}
    if method == "dec_multicoset":
        return designed_mask(fs_set, ctx.cfg), fs_set, b, {}
    if method == "dec_multicoset_vb":
        pos, info = vb_mask(ctx, fs_hz)
        return pos, fs_set, b, info
    if method == "dec_uniform_trained":
        m = ctx.uniform_factor
        if m is None:
            return None, fs_set, b, {"skipped": "no uniform_trained factor"}
        fs_m = FeasibleSet.for_budget(m, fs_set.active_per_period, fs_set.min_gap_ticks)
        return uniform_positions(fs_m), fs_m, m, {"chosen_factor": m}
    raise ValueError(f"unknown fixed method: {method}")


def search_sampler(method: str, ctx: Ctx, seed: int) -> tuple[torch.nn.Module, int]:
    """Stage-A sampler and the period the windows are trimmed to."""
    cfg, t, fs_set = ctx.cfg, ctx.cfg["training"], ctx.fs_set
    period, k, g = fs_set.period_ticks, fs_set.active_per_period, fs_set.min_gap_ticks
    if method == "dec_learned":
        m = cfg["mask"]
        return GumbelTopKMask(
            fs_set,
            tau=float(t["tau_start"]),
            init=str(m["init"]),
            init_scale=float(m["init_scale"]),
            straight_through=m["relaxation"] == "topk_st",
            seed=seed,
        ), period
    if method == "dec_free":
        mult = int(cfg["free"]["period_multiple"])
        free = FeasibleSet(period * mult, k * mult, 1)
        m = cfg["mask"]
        return GumbelTopKMask(
            free,
            tau=float(t["tau_start"]),
            init=str(m["init"]),
            init_scale=float(m["init_scale"]),
            straight_through=m["relaxation"] == "topk_st",
            seed=seed,
            audit_gap=g,
        ), period * mult
    if method == "dec_loupe":
        lp = cfg["loupe"]
        return LoupeMask(
            period * int(lp["period_multiple"]),
            fs_set.rate_fraction,
            slope=float(lp["slope"]),
            sample_slope=float(lp["sample_slope"]),
            seed=seed,
            eval_seed=derive_seed(seed, "loupe_eval"),
        ), period
    if method == "dec_adaptive":
        ap = cfg["adps"]
        return BlockAdaptiveMask(
            fs_set,
            n_blocks=int(ap["n_blocks"]),
            n_features=int(ap["n_features"]),
            hidden=int(ap["hidden"]),
            tau=float(t["tau_start"]),
            seed=seed,
        ), period
    raise ValueError(f"unknown search method: {method}")


def train_one(
    ctx: Ctx, model: SampledOrderModel, task: OrderTask, lr: float, anneal: bool
) -> FitResult:
    """One training run with early stopping; the mask has its own learning rate."""
    t = ctx.cfg["training"]
    mult = float(ctx.cfg["mask"]["lr_multiplier"])
    groups = [
        {"params": list(model.decoder.parameters()), "lr": float(lr)},
        {"params": list(model.sampler.parameters()), "lr": float(lr) * mult},
    ]

    def on_epoch(epoch: int) -> None:
        if anneal and hasattr(model.sampler, "set_temperature"):
            frac = min(1.0, epoch / max(1, int(t["tau_anneal_epochs"])))
            tau = float(t["tau_start"]) * (float(t["tau_end"]) / float(t["tau_start"])) ** frac
            model.sampler.set_temperature(tau)

    fc = fit_cfg(ctx, lr)
    if anneal and bool(t.get("stop_after_anneal", False)):
        # rule B5, 2nd application (2026-11-11): stopping and checkpoint only after annealing
        fc = dataclasses.replace(fc, min_epochs=int(t["tau_anneal_epochs"]))
    return fit(
        model,
        train_batches=task.batches,
        batch_loss=task.loss,
        val_loss=task.val_loss,
        cfg=fc,
        on_epoch_start=on_epoch,
        param_groups=groups,
    )


def _fit_info(res: FitResult, lr: float, seconds: float) -> dict:
    return {
        "lr": lr,
        "best_epoch": res.best_epoch,
        "epochs_run": res.epochs_run,
        "stopped_early": res.stopped_early,
        "diverged": res.diverged,
        "seconds": round(seconds, 1),
    }


def best_over_lr(
    ctx: Ctx, label: str, build: Callable[[int], tuple[SampledOrderModel, OrderTask]], anneal: bool
) -> tuple[SampledOrderModel, OrderTask, dict, list]:
    """Trains at each rate of the grid and keeps the lowest validation error (validation only)."""
    rc = ctx.rc
    best: tuple[float, SampledOrderModel, OrderTask, dict] | None = None
    history = []
    for lr_i, lr in enumerate(ctx.cfg["training"]["lr_grid"]):
        seed = derive_seed(
            ctx.cfg["seed"],
            rc.ds,
            legacy_label(rc.variant),
            rc.split["index"],
            rc.seed,
            rc.budget,
            legacy_label(label),
            lr_i,
        )
        torch.manual_seed(seed)
        model, task = build(seed)
        t0 = time.time()
        res = train_one(ctx, model, task, float(lr), anneal)
        history.append({"label": label, "lr": lr, "history": res.history})
        info = _fit_info(res, float(lr), time.time() - t0)
        if best is None or res.best_val_loss < best[0]:
            best = (res.best_val_loss, copy.deepcopy(_scrub(model)).cpu(), task, info)
    assert best is not None
    _val, model_cpu, task, info = best
    model = model_cpu.to(ctx.device)
    assert isinstance(model, SampledOrderModel)
    return model, task, info, history


def fit_fixed(
    ctx: Ctx, positions: np.ndarray, fs_set: FeasibleSet, factor: int, label: str, fs_hz: float
) -> tuple[SampledOrderModel, OrderTask, dict, list]:
    """Fixed mask + decoder trained from scratch: the protocol shared by all arms."""
    period = fs_set.period_ticks

    def build(seed: int) -> tuple[SampledOrderModel, OrderTask]:
        decoder = make_decoder(ctx, fs_hz, factor, period)
        model = SampledOrderModel(FixedMask(positions, period).to(ctx.device), decoder)
        return model.to(ctx.device), OrderTask(ctx, decoder, period)

    model, task, info, hist = best_over_lr(ctx, label, build, anneal=False)
    info.update(mask_info(positions, period, ctx, fs_hz))
    info.update(task.val_metrics(model))
    return model, task, info, hist


def search(ctx: Ctx, method: str, fs_hz: float) -> tuple[SampledOrderModel, OrderTask, dict, list]:
    """Stage A: mask and decoder together, with temperature annealing."""
    period = ctx.fs_set.period_ticks

    def build(seed: int) -> tuple[SampledOrderModel, OrderTask]:
        sampler, trim = search_sampler(method, ctx, seed)
        decoder = make_decoder(ctx, fs_hz, ctx.rc.budget, period)
        model = SampledOrderModel(sampler.to(ctx.device), decoder)
        return model.to(ctx.device), OrderTask(ctx, decoder, trim)

    model, task, info, hist = best_over_lr(ctx, f"{method}_A", build, anneal=True)
    if isinstance(model.sampler, GumbelTopKMask):
        info["diagnostics"] = model.sampler.diagnostics()
    if isinstance(model.sampler, LoupeMask):
        info["loupe_last_train_rate"] = model.sampler.last_train_rate
    info.update(task.val_metrics(model))
    return model, task, info, hist


def _extra(info: dict) -> dict:
    diag = info.get("diagnostics") or {}
    return {
        "lr": info.get("lr", ""),
        "best_epoch": info.get("best_epoch", ""),
        "epochs_run": info.get("epochs_run", ""),
        "positions": info.get("positions", ""),
        "chosen_factor": info.get("chosen_factor", ""),
        # B5: did the searched mask get out of the Gumbel noise? (top-K margin / noise std)
        "mask_margin_over_gumbel": round(float(diag["margin_over_gumbel_std"]), 4)
        if "margin_over_gumbel_std" in diag
        else "",
        "mask_converged": int(diag["converged"]) if "converged" in diag else "",
    }


def run_method(method: str, ctx: Ctx) -> dict:
    """One method in one cell: test rows, per-arm information and loss history."""
    fs_hz = next(iter(ctx.bank.recs.values())).fs_hz
    rows: list[dict] = []
    infos: dict[str, dict] = {}
    hist: list = []
    if method in FIXED_METHODS:
        pos, fs_set, factor, extra_info = fixed_positions(method, ctx, fs_hz)
        if pos is None:
            return {"rows": [], "info": {method: extra_info}, "history": []}
        model, task, info, hist = fit_fixed(ctx, pos, fs_set, factor, method, fs_hz)
        info.update(extra_info)
        rows += task.evaluate(model, method, _extra(info))
        return {"rows": rows, "info": {method: info}, "history": hist}

    model_a, task_a, info_a, hist = search(ctx, method, fs_hz)
    if method == "dec_adaptive":
        info_a["block_positions_last_window"] = getattr(model_a.sampler, "last_block_positions", [])
        rows += task_a.evaluate(model_a, "dec_adaptive_blocks", _extra(info_a))
        return {"rows": rows, "info": {"dec_adaptive_blocks": info_a}, "history": hist}

    fs_set, period = ctx.fs_set, ctx.fs_set.period_ticks
    if method == "dec_learned":
        pos = model_a.sampler.positions()
        arm_a, arm_b = "dec_learned_norefit", "dec_learned"
    elif method == "dec_free":
        free = model_a.sampler
        assert isinstance(free, GumbelTopKMask)
        k_h = free.fs.active_per_period
        pos = fold_project(free.logits.detach().cpu().numpy(), k_h, fs_set)
        pos0 = fold_project(free.initial_logits.cpu().numpy(), k_h, fs_set)
        info_a.update(
            realizable=free.diagnostics()["final_mask_audit_violations"] == 0,
            projected_positions_initial=pos0.tolist(),
            projected_positions_final=pos.tolist(),
            projected_moved=pos.tolist() != pos0.tolist(),
        )
        rows += task_a.evaluate(model_a, "dec_free", _extra(info_a))
        infos["dec_free"] = info_a
        arm_a, arm_b = "dec_free_projected_norefit", "dec_free_projected"
    elif method == "dec_loupe":
        loupe = model_a.sampler
        assert isinstance(loupe, LoupeMask)
        rows += task_a.evaluate(model_a, "dec_loupe_free", {**_extra(info_a)})
        infos["dec_loupe_free"] = {**info_a, "realizable": False}
        pos = loupe.project(fs_set)
        arm_a, arm_b = "dec_loupe_projected_norefit", "dec_loupe_projected"
    else:
        raise ValueError(f"unknown method: {method}")
    # without fine-tuning: the stage-A decoder with the projected mask, no training at all
    model_p = SampledOrderModel(FixedMask(pos, period).to(ctx.device), model_a.decoder)
    task_p = OrderTask(ctx, model_a.decoder, period)
    info_p = {**info_a, **mask_info(pos, period, ctx, fs_hz), **task_p.val_metrics(model_p)}
    rows += task_p.evaluate(model_p, arm_a, _extra(info_p))
    infos[arm_a] = info_p
    # stage B: the final mask, fixed, with a decoder trained from scratch
    model_b, task_b, info_b, hist_b = fit_fixed(ctx, pos, fs_set, ctx.rc.budget, arm_b, fs_hz)
    info_b["search"] = {k: v for k, v in info_a.items() if k != "diagnostics"}
    info_b["diagnostics"] = info_a.get("diagnostics")
    rows += task_b.evaluate(model_b, arm_b, _extra(info_b))
    infos[arm_b] = info_b
    return {"rows": rows, "info": infos, "history": hist + hist_b}


def select_across_budgets(outs: dict[int, dict], budgets: list[int]) -> list[dict]:
    """`dec_learned_sel`: at each b, the `dec_learned` of trained budget >= b with the best
    validation (balanced order accuracy by the local ruler; tie: lowest loss; then the highest
    rate). No test label is used."""
    rows = []
    for b in budgets:
        cands = []
        for bb, out in outs.items():
            if bb < b or "dec_learned" not in out["info"]:
                continue
            info = out["info"]["dec_learned"]
            bal = info.get("val_bal_acc", math.nan)
            key = (-(bal if np.isfinite(bal) else -1.0), info.get("val_loss", math.inf), bb)
            cands.append((key, bb))
        if not cands:
            continue
        chosen = min(cands)[1]
        for r in outs[chosen]["rows"]:
            if r["arm"] == "dec_learned":
                rows.append({**r, "arm": "dec_learned_sel", "budget": b, "chosen_budget": chosen})
    return rows


# ----------------------------------------------------------------------
# WDCNN (fault-family baseline only, on the uniformly decimated signal)
# ----------------------------------------------------------------------
def run_wdcnn(ctx: Ctx) -> dict:
    cfg, t, w, rc = ctx.cfg, ctx.cfg["training"], ctx.cfg["wdcnn"], ctx.rc
    b, length = rc.budget, int(w["input_length"])
    span_full = length * b
    bank = ctx.bank
    label = {g: i for i, g in enumerate(ctx.groups)}
    by_group: dict[str, list[str]] = defaultdict(list)
    for rid in rc.split["train"]:
        by_group[bank.recs[rid].group].append(rid)
    batch = int(w["batch_size"])
    steps = max(1, int(w["windows_per_epoch"]) // batch)

    def cut(rid: str, start: int) -> torch.Tensor:
        z = bank.x[rid][start : start + span_full : b][:length]
        return (z - z.mean()) / z.std().clamp_min(1e-12)

    def starts_of(a: int, stop: int, cap: int | None) -> list[int]:
        starts = list(range(a, stop - span_full + 1, span_full))
        return starts[:cap] if cap else starts

    cap = int(w["max_val_windows_per_segment"])
    val = [
        (rid, s)
        for rid in rc.split["val"]
        for _seg, a, e in bank.recs[rid].tests
        for s in starts_of(a, e, cap)
    ]

    def batches(epoch: int) -> Iterator[tuple]:
        rng = np.random.default_rng([ctx.data_seed, epoch, 7])
        for _ in range(steps):
            picks = [wdcnn_pick(rng, by_group, bank, span_full) for _ in range(batch)]
            x = torch.stack([cut(rid, s) for rid, s in picks])
            y = torch.tensor([label[bank.recs[rid].group] for rid, _ in picks], device=bank.device)
            yield x, y

    def loss(model: torch.nn.Module, batch_: tuple) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(batch_[0]), batch_[1])

    def val_loss(model: torch.nn.Module) -> float:
        if not val:
            return math.inf
        x = torch.stack([cut(rid, s) for rid, s in val])
        y = torch.tensor([label[bank.recs[rid].group] for rid, _ in val], device=bank.device)
        return float(loss(model, (x, y)))

    best: tuple[float, torch.nn.Module, dict] | None = None
    history = []
    for lr_i, lr in enumerate(t["lr_grid"]):
        seed = derive_seed(
            cfg["seed"],
            rc.ds,
            legacy_label(rc.variant),
            rc.split["index"],
            rc.seed,
            b,
            "wdcnn",
            lr_i,
        )
        torch.manual_seed(seed)
        model = WDCNN(len(ctx.groups), length).to(ctx.device)
        t0 = time.time()
        res = fit(
            model, train_batches=batches, batch_loss=loss, val_loss=val_loss, cfg=fit_cfg(ctx, lr)
        )
        history.append({"label": "wdcnn", "lr": lr, "history": res.history})
        info = {**_fit_info(res, float(lr), time.time() - t0), "n_parameters": model.n_parameters()}
        if best is None or res.best_val_loss < best[0]:
            best = (res.best_val_loss, copy.deepcopy(model).cpu(), info)
    assert best is not None
    val_l, model_cpu, info = best
    model = model_cpu.to(ctx.device).eval()
    rows = []
    for rid in rc.split["test"] + ctx.healthy:
        r = bank.recs[rid]
        fam_true = r.group if rid in rc.nested else HEALTHY
        for seg, a, e in r.tests:
            row = row_base(rc, "wdcnn", rid, r.group, seg)
            starts = starts_of(a, e, None)
            if not starts:  # counts as a miss: the segment enters the denominator
                row.update(skipped="segment shorter than one window", family_pred=NO_FAMILY)
                rows.append({**row, "family_true": fam_true})
                continue
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.stack([cut(rid, s) for s in starts])), -1)
            votes = lp.argmax(-1).cpu().numpy()
            row.update(
                family_pred=ctx.groups[int(lp.mean(0).argmax())],
                family_true=fam_true,
                n_windows=len(starts),
                lr=info["lr"],
                best_epoch=info["best_epoch"],
                epochs_run=info["epochs_run"],
            )
            if fam_true != HEALTHY:
                row["window_acc"] = round(float(np.mean(votes == label[r.group])), 4)
            rows.append(row)
    return {"rows": rows, "info": {"wdcnn": {**info, "val_loss": val_l}}, "history": history}


# ----------------------------------------------------------------------
# summary, preregistered statistics and comparisons
# ----------------------------------------------------------------------
def summarize(rows: list[dict], groups_of_ds: dict[str, list[str]]) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["dataset"], r["variant"], int(r["budget"]), r["arm"])].append(r)
    out = []
    for (ds, var, b, arm), rs in sorted(
        cells.items(),
        key=lambda kv: (kv[0][0], legacy_label(kv[0][1]), kv[0][2], legacy_label(kv[0][3])),
    ):
        fault = [r for r in rs if r.get("family_true") not in (HEALTHY, None, "")]
        healthy = [r for r in rs if r.get("family_true") == HEALTHY]
        row: dict[str, Any] = {"dataset": ds, "variant": var, "budget": b, "arm": arm}
        for ruler in RULERS:
            scored = [r for r in fault if r.get(f"hit_{ruler}") not in ("", None)]
            by_g: dict[str, list[int]] = defaultdict(list)
            for r in scored:
                by_g[r["group"]].append(int(r[f"hit_{ruler}"]))
            n = len(scored)
            hits = sum(sum(v) for v in by_g.values())
            row[f"n_{ruler}"] = n if arm != "wdcnn" else ""
            row[f"hits_{ruler}"] = hits if arm != "wdcnn" else ""
            row[f"acc_{ruler}_pct"] = round(100.0 * hits / n, 1) if n and arm != "wdcnn" else ""
            bal = float(np.mean([np.mean(v) for v in by_g.values()])) if by_g else math.nan
            row[f"bal_acc_{ruler}"] = round(bal, 4) if np.isfinite(bal) and arm != "wdcnn" else ""
        classes = groups_of_ds[ds] + [NO_FAMILY]
        idx = {g: i for i, g in enumerate(classes)}
        per_cell: dict[tuple, list[dict]] = defaultdict(list)
        for r in fault:
            per_cell[(r["split"], r.get("seed", ""))].append(r)
        fam = []
        for srs in per_cell.values():
            yt = [idx[r["family_true"]] for r in srs]
            yp = [idx.get(r.get("family_pred") or NO_FAMILY, idx[NO_FAMILY]) for r in srs]
            fam.append(classification_scores(yt, yp, len(classes)).balanced_accuracy)
        factors = {
            str(r["chosen_factor"]) for r in fault if r.get("chosen_factor") not in (None, "")
        }
        chosen_b = {
            str(r["chosen_budget"]) for r in fault if r.get("chosen_budget") not in (None, "")
        }
        row.update(
            skipped=sum(1 for r in fault if r.get("skipped")),
            family_bal_acc_mean=round(float(np.mean(fam)), 4) if fam else "",
            family_bal_acc_min=round(float(np.min(fam)), 4) if fam else "",
            family_bal_acc_max=round(float(np.max(fam)), 4) if fam else "",
            n_cells=len(per_cell),
            healthy_n=len(healthy),
            healthy_fault_like=sum(int(r.get("fault_like") or 0) for r in healthy)
            if arm != "wdcnn"
            else "",
            chosen_factors=" ".join(sorted(factors, key=int)),
            chosen_budgets=" ".join(sorted(chosen_b, key=int)),
        )
        # B5: fraction of the (split, seed) cells in which the searched mask got out of the noise
        conv = {}
        for r in fault:
            if r.get("mask_converged") not in (None, ""):
                conv[(r["split"], r.get("seed", ""))] = int(r["mask_converged"])
        row["mask_converged_frac"] = round(float(np.mean(list(conv.values()))), 3) if conv else ""
        out.append(row)
    return out


def per_record(rows: list[dict], ruler: str) -> dict[tuple[Any, str], float]:
    """Mean hit per (split, record), over segments and seeds: the per-record mean that comes
    BEFORE resampling. A row discarded by the guard counts as a miss."""
    acc: dict[tuple[Any, str], list[int]] = defaultdict(list)
    for r in rows:
        if r.get("family_true") in (HEALTHY, None, "") or r.get(f"hit_{ruler}") in ("", None):
            continue
        acc[(r["split"], r["record"])].append(int(r[f"hit_{ruler}"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cluster_bootstrap(
    a: dict[tuple[Any, str], float],
    b: dict[tuple[Any, str], float],
    group_of: dict[str, str],
    cluster_of: dict[str, str],
    n_boot: int,
    seed: int,
    min_clusters: int = 3,
    min_splits: int = 2,
) -> dict[str, Any] | None:
    """Balanced-accuracy difference a - b, in points, with a cluster bootstrap.

    Paired by (split, record). The statistic is the mean, over the groups (races), of the mean
    difference within the group: the balanced accuracy. Each resample draws, with replacement,
    the splits and, within each group, the clusters (condition x scenario on Ottawa); the weight
    of a pair is the number of times its split and its cluster were drawn. One-sided p for
    a > b: ``(1 + #{Δ* <= 0})/(1 + B)``. A test without ``min_clusters`` clusters in some group,
    or with fewer than ``min_splits`` splits, is not eligible and gets p = 1.
    """
    keys = sorted(set(a) & set(b), key=lambda k: (str(k[0]), k[1]))
    if not keys:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    groups = sorted({group_of[k[1]] for k in keys})
    splits = sorted({k[0] for k in keys}, key=str)
    s_idx = np.array([splits.index(k[0]) for k in keys])
    g_idx = np.array([groups.index(group_of[k[1]]) for k in keys])
    clusters = {g: sorted({cluster_of[k[1]] for k in keys if group_of[k[1]] == g}) for g in groups}
    c_idx = np.array([clusters[group_of[k[1]]].index(cluster_of[k[1]]) for k in keys])
    point = float(np.mean([d[g_idx == gi].mean() for gi in range(len(groups))]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        sw = np.bincount(rng.integers(0, len(splits), len(splits)), minlength=len(splits))
        vals = []
        for gi, g in enumerate(groups):
            nc = len(clusters[g])
            cw = np.bincount(rng.integers(0, nc, nc), minlength=nc)
            sel = g_idx == gi
            w = sw[s_idx[sel]] * cw[c_idx[sel]]
            if w.sum() == 0:
                break
            vals.append(float((w * d[sel]).sum() / w.sum()))
        else:
            boots.append(float(np.mean(vals)))
    arr = np.asarray(boots)
    min_c = min(len(c) for c in clusters.values())
    # preregistered eligibility: with fewer than 3 clusters per group or 1 split, all resamples
    # come out almost the same and p becomes 1/(1+B) without measuring any variability
    eligible = min_c >= min_clusters and len(splits) >= min_splits and arr.size > 0
    p = float((1 + np.count_nonzero(arr <= 0.0)) / (1 + arr.size)) if eligible else 1.0
    return {
        "n_pairs": len(keys),
        "n_records": len({k[1] for k in keys}),
        "n_clusters": sum(len(c) for c in clusters.values()),
        "min_clusters_per_group": min_c,
        "n_splits": len(splits),
        "diff_points": round(100.0 * point, 2),
        "ci_lo": round(100.0 * float(np.percentile(arr, 2.5)), 2) if arr.size else "",
        "ci_hi": round(100.0 * float(np.percentile(arr, 97.5)), 2) if arr.size else "",
        "p_one_sided": p,
        "eligible": int(eligible),
        "n_boot_valid": int(arr.size),
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-adjusted p-values (step-down), in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def arm_rows(rows: list[dict], ds: str, var: str, b: int, arm: str) -> list[dict]:
    return [
        r
        for r in rows
        if r["dataset"] == ds and r["variant"] == var and int(r["budget"]) == b and r["arm"] == arm
    ]


def mask_converged_frac(rows: list[dict], ds: str, var: str, b: int, arm: str) -> float | None:
    """B5: fraction of the (split, seed) cells in which the arm's mask got out of the noise."""
    conv: dict[tuple, int] = {}
    for r in arm_rows(rows, ds, var, b, arm):
        if r.get("mask_converged") not in (None, ""):
            conv[(r["split"], r.get("seed", ""))] = int(r["mask_converged"])
    return float(np.mean(list(conv.values()))) if conv else None


def evaluable_budgets(
    rows: list[dict], ds: str, var: str, budgets: list[int], arms: list[str], min_frac: float
) -> dict[int, tuple[bool, str]]:
    """Budget evaluable in a hypothesis: every arm in `arms` with its mask out of the noise in at
    least `min_frac` of the splits. A listed arm without diagnostics counts as not evaluable, so
    that it does not pass silently; with no arms listed, every budget is evaluable."""
    out = {}
    for b in budgets:
        fr = {a: mask_converged_frac(rows, ds, var, b, a) for a in arms}
        ok = all(f is not None and f >= min_frac for f in fr.values())
        txt = " ".join(f"{a}={'' if f is None else round(f, 3)}" for a, f in fr.items())
        out[b] = (ok, txt)
    return out


def hypotheses(
    rows: list[dict], cfg: dict, group_of: dict[str, str], cluster_of: dict[str, str]
) -> list[dict]:
    """Preregistered tests (README, section "Preregistration"; `preregistration` block)."""
    pr = cfg["preregistration"]
    ds, var, alpha = pr["dataset"], pr["variant"], float(pr["alpha"])
    n_boot, seed = int(cfg["analysis"]["bootstrap_resamples"]), int(cfg["seed"])
    budgets = sorted({int(r["budget"]) for r in rows if r["dataset"] == ds and r["variant"] == var})
    min_conv = float(pr.get("min_mask_converged_frac", 0.0))
    out: list[dict] = []
    for name, key in (("H1", "h1"), ("H2", "h2")):
        h = pr[key]
        arm = h["arm"]
        baselines = list(h["baselines"]) if "baselines" in h else [h["baseline"]]
        conv_arms = list(h.get("convergence_arms", []))
        evals = evaluable_budgets(rows, ds, var, budgets, conv_arms, min_conv)
        verdict_rulers = {}
        for ruler in pr["rulers"]:
            tests = []
            for b in budgets:
                ok, conv_txt = evals[b]
                if not ok:
                    # B5: mask stuck in noise -> p = 1 in Holm (conservative); it does not win
                    for base in baselines:
                        nt = {"budget": b, "baseline": base, "p_one_sided": 1.0}
                        tests.append({**nt, "evaluable": 0, "convergence": conv_txt})
                    continue
                a_rec = per_record(arm_rows(rows, ds, var, b, arm), ruler)
                for base in baselines:
                    b_rec = per_record(arm_rows(rows, ds, var, b, base), ruler)
                    s = derive_seed(seed, name, ruler, b, legacy_label(arm), legacy_label(base))
                    res = cluster_bootstrap(
                        a_rec,
                        b_rec,
                        group_of,
                        cluster_of,
                        n_boot,
                        s,
                        min_clusters=int(pr["min_clusters_per_group"]),
                        min_splits=int(pr["min_splits"]),
                    )
                    t = {"budget": b, "baseline": base, **(res or {"p_one_sided": 1.0})}
                    tests.append({**t, "evaluable": 1, "convergence": conv_txt})
            adj = holm([float(t["p_one_sided"]) for t in tests])
            passed = defaultdict(list)
            for t, p_adj in zip(tests, adj, strict=True):
                sig = bool(p_adj < alpha and float(t.get("diff_points", 0.0) or 0.0) > 0.0)
                passed[t["budget"]].append(sig)
                out.append(
                    {
                        "hypothesis": name,
                        "ruler": ruler,
                        "arm": arm,
                        **t,
                        "p_holm": round(p_adj, 4),
                        "significant": int(sig),
                    }
                )
            wins = [bb for bb, v in passed.items() if v and all(v)]
            verdict_rulers[ruler] = len(wins) >= int(pr["min_budgets"])
            out.append(
                {
                    "hypothesis": name,
                    "ruler": ruler,
                    "arm": arm,
                    "verdict_ruler": int(verdict_rulers[ruler]),
                    "budgets_won": " ".join(str(x) for x in sorted(wins)),
                    "budgets_not_evaluable": " ".join(str(x) for x in budgets if not evals[x][0]),
                }
            )
        out.append(
            {
                "hypothesis": name,
                "ruler": "+".join(pr["rulers"]),
                "arm": arm,
                "verdict": int(all(verdict_rulers.values())),
            }
        )
    return out


def comparisons(
    rows: list[dict], cfg: dict, group_of: dict[str, str], cluster_of: dict[str, str]
) -> list[dict]:
    """Secondary comparisons (outside the preregistration), same bootstrap and no correction."""
    n_boot, seed = int(cfg["analysis"]["bootstrap_resamples"]), int(cfg["seed"])
    cells = sorted(
        {(r["dataset"], r["variant"], int(r["budget"])) for r in rows},
        key=lambda c: (c[0], legacy_label(c[1]), c[2]),
    )
    out = []
    for ds, var, b in cells:
        present = {
            r["arm"] for r in rows if (r["dataset"], r["variant"], int(r["budget"])) == (ds, var, b)
        }
        for arm_a, arm_b in SECONDARY:
            if arm_a not in present or arm_b not in present:
                continue
            for ruler in RULERS:
                res = cluster_bootstrap(
                    per_record(arm_rows(rows, ds, var, b, arm_a), ruler),
                    per_record(arm_rows(rows, ds, var, b, arm_b), ruler),
                    group_of,
                    cluster_of,
                    n_boot,
                    derive_seed(
                        seed,
                        ds,
                        legacy_label(var),
                        b,
                        legacy_label(arm_a),
                        legacy_label(arm_b),
                        ruler,
                    ),
                )
                if res:
                    cell = {"dataset": ds, "variant": var, "budget": b, "ruler": ruler}
                    out.append({**cell, "arm_a": arm_a, "arm_b": arm_b, **res})
    return out


# ----------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def source_sha256() -> tuple[str | None, int]:
    files = sorted((REPO_ROOT / "packages").glob("*/src/**/*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return (digest.hexdigest() if files else None), len(files)


def write_manifest(config_path: Path, results: Path, cfg: dict, extra: dict) -> None:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    src_sha, src_n = source_sha256()
    dirty = _git("status", "--porcelain")
    manifest = {
        "experiment": EXPERIMENT,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": int(cfg["seed"]),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "readme_sha256": sha256_file(README) if README.exists() else None,
        "preregistration": cfg.get("preregistration"),
        "packages": versions,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else [ln[3:] for ln in dirty.splitlines() if ln.strip()],
        "source_sha256": src_sha,
        "source_n_files": src_n,
        **extra,
    }
    results.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, default=str)
    (results / "manifest.json").write_text(text, encoding="utf-8")


def write_csv(path: Path, data: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(data)


# ----------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def execute(
    method: str, ctx: Ctx, runs_dir: Path, run_key: dict, no_resume: bool, manifest: dict
) -> dict:
    """Runs one method in one cell, or resumes the saved run with the same key."""
    rc = ctx.rc
    rel = Path(rc.ds) / rc.variant / f"b{rc.budget}" / f"split{rc.split['index']}"
    rel = rel / f"seed{rc.seed}" / f"{method}.json"
    run_file = runs_dir / rel
    if run_file.exists() and not no_resume:
        saved = json.loads(run_file.read_text(encoding="utf-8"))
        if saved.get("run_key") == run_key:
            manifest["runs"][rel.as_posix()] = "resumed"
            return saved
    t0 = time.time()
    out = run_wdcnn(ctx) if method == "wdcnn" else run_method(method, ctx)
    out["run_key"] = run_key
    run_file.parent.mkdir(parents=True, exist_ok=True)
    run_file.write_text(json.dumps(out, default=float), encoding="utf-8")
    manifest["runs"][rel.as_posix()] = round(time.time() - t0, 1)
    fault = [r for r in out["rows"] if r.get("family_true") not in (HEALTHY, None)]
    hits = sum(int(r.get("hit_local") or 0) for r in fault)
    fam = sum(int(r.get("family_pred") == r.get("family_true")) for r in fault)
    tag = f"[{rc.ds} {rc.variant} 1/{rc.budget} split {rc.split['index']} seed {rc.seed}] {method}"
    print(
        f"  {tag}: order (local ruler) {hits}/{len(fault)}, family {fam}/{len(fault)}, "
        f"{time.time() - t0:.0f} s",
        flush=True,
    )
    return out


def mode_band(bands: list[list[float] | None]) -> tuple[float, float] | None:
    """Most frequent band (tie: the one with the lowest lower edge)."""
    vals = [tuple(round(float(v), 1) for v in bnd) for bnd in bands if bnd]
    if not vals:
        return None
    counts = Counter(vals)
    top = max(counts.values())
    best = min(v for v, c in counts.items() if c == top)
    return (float(best[0]), float(best[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of the datasets")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records")
    ap.add_argument("--methods", nargs="*", default=None, help="subset of the methods")
    ap.add_argument("--budgets", nargs="*", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None, help="overrides the epoch cap")
    ap.add_argument("--device", default=None, help="auto, cpu or cuda")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true", help="redoes runs already saved")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg_sha = sha256_file(args.config)
    # what changes the result of a saved run: config, code and CLI
    run_key = {
        "config_sha256": cfg_sha,
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "source_sha256": source_sha256()[0],
        "max_epochs": args.max_epochs,
        "limit": args.limit,
    }
    results = args.config.parent.parent / cfg["output"]["results_dir"]
    runs_dir = results / "runs"
    device = resolve_device(args.device or cfg["compute"].get("device", "auto"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    workers = int(args.workers or cfg["compute"]["workers"])
    datasets = args.datasets or [ds for ds, d in cfg["datasets"].items() if d.get("enabled")]
    for ds in datasets:
        if ds not in cfg["datasets"]:
            raise SystemExit(f"dataset {ds!r} is not in the config")
    methods = args.methods or list(cfg["methods"])
    unknown = [m for m in methods if m not in ARMS_OF_METHOD]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")
    print(f"device {device}; datasets {datasets}; methods {methods}", flush=True)
    tol = float(cfg["tolerance_order"])
    span = float(cfg["prior"]["calibration_span"])
    seeds_per_split = int(cfg["protocol"].get("seeds_per_split", 1))

    all_rows: list[dict] = []
    manifest: dict[str, Any] = {
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        "rulers": {
            "local": "full-rate line of the segment itself (snq_recon.refine_order, centre of the "
            "plateau); sliding grid for training and validation, segment for selection and scoring",
            "nested": "median of the segment-0 calibrations of the records outside the test set "
            "of the group",
        },
        "input_sha256": {},
        "calibrated_order": {},
        "local_grid": {},
        "segment_truth": {},
        "despiked_samples": {},
        "splits": {},
        "feasible_sets": {},
        "known_band": {},
        "chosen_uniform_factor": {},
        "groups": {},
        "runs": {},
    }
    groups_of_ds: dict[str, list[str]] = {}
    group_all: dict[str, str] = {}
    cluster_all: dict[str, str] = {}
    for ds in datasets:
        d = cfg["datasets"][ds]
        cfg_ds = dataset_cfg(cfg, d)  # equal to the global one without ``overrides``
        tol = float(cfg_ds["tolerance_order"])
        span = float(cfg_ds["prior"]["calibration_span"])
        ids = list_ids(d)
        ids = ids[: args.limit] if args.limit else ids
        healthy_ids = list_ids(d, healthy=True)
        healthy_ids = healthy_ids[: args.limit] if args.limit else healthy_ids
        variants = [bool(v) for v in d.get("despike_variants", [False])]
        allowed = [int(x) for x in d["budgets"]]
        budgets = [int(b) for b in (args.budgets or allowed) if int(b) in allowed]
        names = [variant_name(v) for v in variants]
        print(
            f"[{ds}] {len(ids)} faulty records, {len(healthy_ids)} healthy, "
            f"budgets {budgets}, variants {names}",
            flush=True,
        )
        t0 = time.time()
        tasks = [
            {"cfg": cfg_ds, "dataset": ds, "rid": r, "variants": variants}
            for r in ids + healthy_ids
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cls = list(pool.map(classical_task, tasks))
        print(f"[{ds}] classical stage and full-rate rulers: {time.time() - t0:.0f} s", flush=True)
        by_id = {c["rid"]: c for c in cls}
        group_of = {c["rid"]: c["group"] for c in cls}
        cluster_of = {c["rid"]: c["cluster"] for c in cls}
        group_all.update({f"{ds}:{k}": v for k, v in group_of.items()})
        cluster_all.update({f"{ds}:{k}": v for k, v in cluster_of.items()})
        calib0 = {r: by_id[r]["calib"]["center"] for r in ids}
        ok0 = {r: not by_id[r]["calib"]["at_edge"] for r in ids}
        band0 = {r: by_id[r]["calib"].get("band_hz") for r in ids}
        seg_truth = {
            (r, int(s)): (float(c) if not edge else math.nan)
            for r in ids
            for s, (c, edge) in by_id[r]["segment_truth"].items()
        }
        grid_labels = {
            r: [(int(st), float(c)) for st, c, edge in by_id[r]["local_grid"] if not edge]
            for r in ids
        }
        for c in cls:
            key = f"{ds}:{c['rid']}"
            manifest["input_sha256"][key] = c["sha256"]
            manifest["despiked_samples"][key] = c["despiked_samples"]
            if "calib" in c:
                manifest["calibrated_order"][key] = c["calib"]
                manifest["local_grid"][key] = c["local_grid"]
                manifest["segment_truth"][key] = c["segment_truth"]
        groups = sorted({group_of[r] for r in ids})
        groups_of_ds[ds] = groups
        manifest["groups"][ds] = groups
        if (d.get("splits") or {}).get("kind") == "kfold_cluster":
            splits = kfold_by_cluster(
                d, ids, cluster_of, cfg["protocol"], derive_seed(cfg["seed"], ds, "splits")
            )
        else:
            splits = make_splits(
                ids, group_of, cfg["protocol"], derive_seed(cfg["seed"], ds, "splits")
            )
        manifest["splits"][ds] = splits
        factors = list(range(min(allowed), int(d["factor_grid_max"]) + 1))
        recs = {r: load_record(ds, d, r) for r in ids + healthy_ids}
        segs = {r: [s for s, _a, _b in recs[r].tests] for r in recs}
        k_per = int(cfg["feasible"]["active_per_period"])
        fs_sets = {b: FeasibleSet.for_budget(b, k_per, int(d["min_gap_ticks"])) for b in budgets}
        for b, fsb in fs_sets.items():
            manifest["feasible_sets"][f"{ds}:1/{b}"] = {
                "period_ticks": fsb.period_ticks,
                "active_per_period": fsb.active_per_period,
                "min_gap_ticks": fsb.min_gap_ticks,
                "rate_fraction": fsb.rate_fraction,
            }
        for despiked in variants:
            var = variant_name(despiked)
            grid_est: dict = {}
            for c in cls:
                for (seg, m), res in c["grid"][var].items():
                    if "estimate" in res:
                        grid_est[(c["rid"], seg, m)] = res["estimate"]
                    else:
                        grid_est[("skip", c["rid"], seg, m)] = res.get("skipped", "")
            bank = make_bank(recs, despiked, float(cfg["despike"]["k_mad"]), device)
            for split in splits:
                test = set(split["test"])
                nontest = split["train"] + split["val"]
                nested = nested_truths(ids, group_of, calib0, ok0, test)
                centers = family_centers(ids, group_of, calib0, ok0, test)
                rc0 = RowCtx(ds, var, split, 0, "", nested, seg_truth, centers, tol, span)
                rows_c, chosen = classical_rows(
                    rc0, grid_est, budgets, factors, group_of, segs, healthy_ids
                )
                all_rows += rows_c
                manifest["chosen_uniform_factor"][f"{ds}:{var}:split{split['index']}"] = chosen
                known = mode_band([band0[r] for r in nontest if ok0[r]])
                manifest["known_band"][f"{ds}:split{split['index']}"] = known
                pool: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(dict)
                if d.get("train_labels") == "nested":
                    # Paderborn: the local ruler only exists on 6 of the 23 bearings; the training
                    # and validation labels are the nested ruler, defined on all of them
                    for r in split["train"]:
                        t_r = nested[r]["truth"]
                        if np.isfinite(t_r):
                            pool[group_of[r]][r] = [
                                (int(st), float(t_r)) for st, _c, _e in by_id[r]["local_grid"]
                            ]
                    val_labels = {
                        (r, s): float(nested[r]["truth"])
                        for r in split["val"]
                        for s in segs[r]
                        if np.isfinite(nested[r]["truth"])
                    }
                else:
                    for r in split["train"]:
                        if grid_labels[r]:
                            pool[group_of[r]][r] = grid_labels[r]
                    val_labels = {
                        (r, s): seg_truth[(r, s)]
                        for r in split["val"]
                        for s in segs[r]
                        if np.isfinite(seg_truth[(r, s)])
                    }
                for seed_i in range(seeds_per_split):
                    learned: dict[int, dict] = {}
                    for b in budgets:
                        rc = RowCtx(
                            ds, var, split, b, seed_i, nested, seg_truth, centers, tol, span
                        )
                        ctx = Ctx(
                            cfg=cfg_ds,
                            d=d,
                            rc=rc,
                            fs_set=fs_sets[b],
                            bank=bank,
                            pool=dict(pool),
                            val_labels=val_labels,
                            groups=groups,
                            healthy=healthy_ids,
                            device=device,
                            data_seed=derive_seed(  # "dados": the seed tag of the runs
                                cfg["seed"], ds, split["index"], seed_i, b, "dados"
                            ),
                            max_epochs=args.max_epochs,
                            uniform_factor=chosen.get(b),
                            known_band=known,
                        )
                        for method in methods:
                            out = execute(method, ctx, runs_dir, run_key, args.no_resume, manifest)
                            all_rows += out["rows"]
                            if method == "dec_learned":
                                learned[b] = out
                    all_rows += select_across_budgets(learned, budgets)
            del bank
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = summarize(all_rows, groups_of_ds)
    keyed_group = {k.split(":", 1)[1]: v for k, v in group_all.items()}
    keyed_cluster = {k.split(":", 1)[1]: v for k, v in cluster_all.items()}
    hyp = hypotheses(all_rows, cfg, keyed_group, keyed_cluster) if "preregistration" in cfg else []
    comps = comparisons(all_rows, cfg, keyed_group, keyed_cluster)
    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "metrics.csv", all_rows)
    write_csv(results / "summary.csv", summary)
    write_csv(results / "hypotheses.csv", hyp)
    write_csv(results / "comparisons.csv", comps)
    masks = {}
    # only the runs of this invocation: old files in runs/ (another key) are not included
    for rel in sorted(manifest["runs"], key=legacy_path):
        saved = json.loads((runs_dir / rel).read_text(encoding="utf-8"))
        masks[Path(rel).with_suffix("").as_posix()] = saved.get("info", {})
    text = json.dumps(masks, indent=2, default=str)
    (results / "masks.json").write_text(text, encoding="utf-8")
    manifest["n_rows"] = len(all_rows)
    # only at the end: querying the GPU initializes the CUDA context, and the classical stage
    # forks processes
    manifest["device_name"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    )
    write_manifest(args.config, results, cfg, manifest)
    print(f"\n{len(all_rows)} rows -> {results / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
