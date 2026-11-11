"""E21 - captura do seletor no MAFAULDA e teste de admissibilidade da taxa (ADR 0006).

Cada sequencia de pista externa e dividida em calibracao (1a metade) e teste (2a metade).
Na calibracao, em taxa cheia, calcula-se a ordem calibrada e o TAT: para cada fator e
teto, a fracao da energia da banda escolhida pelo kurtograma que vem de conteudo dobrado.
No teste roda o pipeline do E07/E20 sob decimacao pura e sob filtro anti-alias ideal.

    uv run python experiments/E21-mafaulda-capture-prediction/run.py \
        --config experiments/E21-mafaulda-capture-prediction/config/base.yaml

Os auxiliares de analise sao os do E07/E20, copiados porque nada importa experiments/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml
from snq_data import mafaulda as maf
from snq_data import undersample
from snq_recon import (
    envelope_order_spectrum,
    estimate_without_prior,
    fast_kurtogram,
    refine_order,
)

PACKAGES = ("snq-core", "snq-sim", "snq-data", "snq-recon", "snq-eval")
REPO_ROOT = Path(__file__).resolve().parents[2]


def at_repo(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


# ----------------------------------------------------------------------
# sinal
# ----------------------------------------------------------------------
def aaf_ideal(x: np.ndarray, fs_hz: float, factor: int) -> np.ndarray:
    """Mascara ideal de FFT: zera tudo acima de ``fs/(2*factor)`` (nenhuma dobra sobrevive)."""
    if factor == 1:
        return x
    xf = np.fft.rfft(x)
    xf[np.fft.rfftfreq(x.size, d=1.0 / fs_hz) > fs_hz / (2.0 * factor)] = 0.0
    return np.fft.irfft(xf, n=x.size)


def band_energy(x: np.ndarray, fs_hz: float, lo_hz: float, hi_hz: float) -> float:
    xf = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    return float(np.sum(xf[(f >= lo_hz) & (f <= hi_hz)]))


def search_ceiling(ceiling_hz: float | None, fs_eff_hz: float) -> float:
    return fs_eff_hz / 2.0 if ceiling_hz is None else min(float(ceiling_hz), fs_eff_hz / 2.0)


# ----------------------------------------------------------------------
# analise (mesma logica do E07/E20)
# ----------------------------------------------------------------------
def guard_verdict(n_samples: int, fs_eff_hz: float, points_per_rev: float, cfg: dict) -> str | None:
    if n_samples < int(cfg["min_samples"]):
        return "min_samples"
    if fs_eff_hz / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
        return "f_min_hz"
    if points_per_rev < 2.0 * int(cfg["samples_per_rev"]):
        return "samples_per_rev"
    return None


def refine(spec, nominal: float, span: float, n_harmonics: int, tol: float) -> float:
    # Centro do platô de escore, não o primeiro ponto: o max() antigo enviesava a verdade
    # em meia largura do platô (0,0175 ordem com spr = 64; revisão de 2026-11-11).
    return refine_order(spec, nominal, span=span, n_harmonics=n_harmonics, tol=tol)


def kurtogram(x, fs_hz: float, f_max_hz: float, cfg: dict):
    kg = cfg["kurtogram"]
    return fast_kurtogram(
        x, fs_hz, max_level=int(kg["max_level"]), f_min_hz=float(kg["f_min_hz"]), f_max_hz=f_max_hz
    )


def analyse(x, fs_hz: float, speed, cfg: dict, f_max_hz: float):
    k = kurtogram(x, fs_hz, f_max_hz, cfg)
    spec = envelope_order_spectrum(
        x, fs_hz, speed, k.band_hz, samples_per_rev=int(cfg["samples_per_rev"])
    )
    return k, spec


def tat(x_full: np.ndarray, fs_hz: float, factor: int, f_max_hz: float, cfg: dict) -> dict:
    """Teste de admissibilidade da taxa, so com a captura de calibracao em taxa cheia."""
    fs_eff = fs_hz / factor
    xd = undersample(x_full, factor)
    xa = undersample(aaf_ideal(x_full, fs_hz, factor), factor)
    k = kurtogram(xd, fs_eff, f_max_hz, cfg)
    e = band_energy(xd, fs_eff, k.f_lo_hz, k.f_hi_hz)
    share = 1.0 - band_energy(xa, fs_eff, k.f_lo_hz, k.f_hi_hz) / e if e > 0 else math.nan
    thr = float(cfg["tat"]["folded_share_threshold"])
    return {
        "tat_folded_share": round(share, 4),
        "tat_inadmissible": int(share > thr),
        "tat_band_lo_hz": round(k.f_lo_hz, 1),
        "tat_band_hi_hz": round(k.f_hi_hz, 1),
    }


def run_sequence(task: dict) -> dict:
    cfg, rel = task["cfg"], task["rel"]
    root, cache = at_repo(cfg["data_dir"]), at_repo(cfg["cache_dir"])
    rec = maf.load(rel, root, cache_dir=cache)
    raw_sha = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    x = rec.vibration(axis=cfg["axis"])
    v = rec.speed_profile_hz()
    fs = rec.fs_hz
    half = x.size // 2
    xc, vc, xt, vt = x[:half], v[:half], x[half:], v[half:]
    tol = float(cfg["tolerance_order"])

    nominal = maf.DOCUMENTED_ORDERS[cfg["fault_family"]]
    _, spec = analyse(xc, fs, vc, cfg, fs / 2.0)
    calib = refine(
        spec,
        nominal,
        float(cfg["prior"]["calibration_span"]),
        int(cfg["prior"]["score_harmonics"]),
        tol,
    )

    factors = [int(f) for f in cfg["sweep"]["undersampling_factor"]]
    ceilings = cfg["sweep"]["search_ceiling_hz"]
    tat_cells = {}
    for factor in factors:
        if fs / factor / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
            continue
        for ceil in ceilings:
            tat_cells[(factor, ceil)] = tat(xc, fs, factor, search_ceiling(ceil, fs / factor), cfg)

    meta = {
        "path": rel,
        "bearing": rec.bearing,
        "load_g": rec.load_g,
        "rotation_hz": round(rec.nominal_rotation_hz, 4),
    }
    rows = []
    for iv in cfg["sweep"]["interventions"]:
        for factor in factors:
            xi = aaf_ideal(xt, fs, factor) if iv == "aaf_ideal" else xt
            xd, vd = undersample(xi, factor), undersample(vt, factor)
            fs_eff = fs / factor
            guard = guard_verdict(xd.size, fs_eff, fs_eff / float(np.max(vd)), cfg)
            for ceil in ceilings:
                row = {
                    **meta,
                    "intervention": iv,
                    "factor": factor,
                    "fs_eff_hz": round(fs_eff, 2),
                    "ceiling_hz": "" if ceil is None else float(ceil),
                    **tat_cells.get((factor, ceil), {}),
                    "estimate_order": "",
                    "band_lo_hz": "",
                    "band_hi_hz": "",
                    "kurtosis": "",
                    "skipped": guard or "",
                }
                if guard is None:
                    try:
                        k, sp = analyse(xd, fs_eff, vd, cfg, search_ceiling(ceil, fs_eff))
                        est = estimate_without_prior(sp, float(cfg["order_min"]))
                        row.update(
                            estimate_order=round(float(est), 4) if np.isfinite(est) else "",
                            band_lo_hz=round(k.f_lo_hz, 1),
                            band_hi_hz=round(k.f_hi_hz, 1),
                            kurtosis=round(k.kurtosis, 3),
                        )
                    except ValueError as exc:
                        row["skipped"] = f"ValueError: {str(exc)[:60]}"
                rows.append(row)
    return {"rel": rel, "bearing": rec.bearing, "calib": calib, "sha256": raw_sha, "rows": rows}


# ----------------------------------------------------------------------
# estatistica
# ----------------------------------------------------------------------
def balanced_accuracy(pred: np.ndarray, miss: np.ndarray) -> float:
    pos, neg = miss == 1, miss == 0
    if not pos.any() or not neg.any():
        return math.nan
    return 0.5 * (float(np.mean(pred[pos] == 1)) + float(np.mean(pred[neg] == 0)))


def tat_report(rows: list[dict], cfg: dict) -> dict:
    """Desempenho do TAT em `none`, fatores > 1, com bootstrap por sequencia."""
    rng = np.random.default_rng(int(cfg["seed"]))
    sel = [
        r
        for r in rows
        if r["intervention"] == "none"
        and r["factor"] > 1
        and r["hit"] != ""
        and r.get("tat_inadmissible", "") != ""
    ]
    if not sel:
        return {}
    seqs = sorted({r["path"] for r in sel})
    by_seq = defaultdict(list)
    for r in sel:
        by_seq[r["path"]].append(r)

    def stats(sample: list[dict]) -> tuple[float, float]:
        pred = np.array([r["tat_inadmissible"] for r in sample])
        miss = np.array([1 - r["hit"] for r in sample])
        gap = (
            float(np.mean(1 - miss[pred == 0]) - np.mean(1 - miss[pred == 1]))
            if (pred == 1).any() and (pred == 0).any()
            else math.nan
        )
        return balanced_accuracy(pred, miss), 100.0 * gap

    ba, gap = stats(sel)
    boots = []
    for _ in range(int(cfg["analysis"]["bootstrap_resamples"])):
        pick = rng.choice(len(seqs), size=len(seqs), replace=True)
        boots.append(stats([r for i in pick for r in by_seq[seqs[i]]]))
    b = np.array(boots, dtype=float)

    def q(a: np.ndarray) -> list[float]:
        return [
            round(float(np.nanpercentile(a, 2.5)), 4),
            round(float(np.nanpercentile(a, 97.5)), 4),
        ]

    return {
        "n_cells": len(sel),
        "n_inadmissible": int(sum(r["tat_inadmissible"] for r in sel)),
        "balanced_accuracy": round(ba, 4),
        "balanced_accuracy_ci95": q(b[:, 0]),
        "hit_gap_points_admissible_minus_inadmissible": round(gap, 2),
        "hit_gap_ci95": q(b[:, 1]),
    }


def summarize(rows: list[dict]) -> list[dict]:
    cells = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # n, hits, n_inad, hits_inad, n_adm, hits_adm
    for r in rows:
        if r["hit"] == "":
            continue
        c = cells[(r["bearing"], r["intervention"], r["factor"], r["ceiling_hz"])]
        c[0] += 1
        c[1] += r["hit"]
        if r.get("tat_inadmissible", "") == 1:
            c[2] += 1
            c[3] += r["hit"]
        elif r.get("tat_inadmissible", "") == 0:
            c[4] += 1
            c[5] += r["hit"]
    out = []
    for (bearing, iv, factor, ceil), c in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], str(kv[0][3]))
    ):
        out.append(
            {
                "bearing": bearing,
                "intervention": iv,
                "factor": factor,
                "ceiling_hz": ceil,
                "n": c[0],
                "hits": c[1],
                "n_tat_inadmissible": c[2],
                "hits_tat_inadmissible": c[3],
                "n_tat_admissible": c[4],
                "hits_tat_admissible": c[5],
            }
        )
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
            versions[pkg] = "nao instalado"
    src_sha, src_n = source_sha256()
    dirty = _git("status", "--porcelain")
    manifest = {
        "experiment": "E21-mafaulda-capture-prediction",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": int(cfg["seed"]),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "dataset": (
            "MAFAULDA (UFRJ), licenca nao declarada; fs inferida 51,2 kHz (ver manifest do dataset)"
        ),
        "fs_hz": maf.FS_HZ,
        "packages": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else [ln[3:] for ln in dirty.splitlines() if ln.strip()],
        "source_sha256": src_sha,
        "source_n_files": src_n,
        **extra,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="so as primeiras N sequencias (fumaca)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = args.config.parent.parent / cfg["output"]["results_dir"]

    screen = set(
        yaml.safe_load(at_repo(cfg["screen_catalog"]).read_text(encoding="utf-8"))["excluded"]
    )
    recs = [
        r
        for r in maf.list_records(at_repo(cfg["data_dir"]), subsets=tuple(cfg["bearings"]))
        if Path(r).parts[1] == cfg["fault"] and r not in screen
    ]
    if args.limit:
        recs = recs[: args.limit]
    workers = int(args.workers or cfg["compute"]["workers"])
    print(f"{len(recs)} sequencias ({cfg['fault']}), {workers} processos", flush=True)

    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(run_sequence, [{"cfg": cfg, "rel": r} for r in recs]), 1):
            out.append(res)
            if i % 30 == 0 or i == len(recs):
                print(f"  {i}/{len(recs)} sequencias", flush=True)

    calib = {o["rel"]: o["calib"] for o in out}
    bearing_of = {o["rel"]: o["bearing"] for o in out}
    tol = float(cfg["tolerance_order"])
    rows = []
    for o in out:
        others = [v for k, v in calib.items() if k != o["rel"] and bearing_of[k] == o["bearing"]]
        truth = float(np.median(others))
        for r in o["rows"]:
            est = r["estimate_order"]
            err = abs(est - truth) if est != "" else math.inf
            r.update(
                truth_order=round(truth, 4),
                error_order=round(err, 4) if np.isfinite(err) else "",
                hit="" if r["skipped"] else int(err < tol),
            )
            rows.append(r)

    rows.sort(key=lambda r: (r["path"], r["intervention"], r["factor"], str(r["ceiling_hz"])))
    summary = summarize(rows)
    report = tat_report(rows, cfg)
    write_manifest(
        args.config,
        results,
        cfg,
        {
            "n_sequences": len(out),
            "calibrated_order_per_sequence": {k: round(v, 4) for k, v in calib.items()},
            "input_sha256": {o["rel"]: o["sha256"] for o in out},
            "tat_report": report,
            "limit": args.limit,
        },
    )
    for name, data in (("metrics.csv", rows), ("summary.csv", summary)):
        with (results / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print(f"\n{len(rows)} linhas -> {results / 'metrics.csv'}")
    print("TAT:", json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
