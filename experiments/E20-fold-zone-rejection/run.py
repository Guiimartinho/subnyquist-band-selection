"""E20 - zone rejection before decimating (ADR 0006).

Question: which zone of original frequency, folded onto the kurtogram's search band,
causes the 4x valley on the Ottawa outer race (E07)? And what is the role of the
search ceiling relative to the effective Nyquist?

The pipeline is the E07 one (restricted kurtogram, envelope order spectrum,
estimate without prior), with two new dimensions: an intervention applied to the
full-rate segment, BEFORE pure decimation, and the kurtogram's search ceiling.

    uv run python experiments/E20-fold-zone-rejection/run.py \
        --config experiments/E20-fold-zone-rejection/config/base.yaml

The analysis helpers are those of E07, copied because nothing imports experiments/.
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
from snq_data import effective_rate_hz, undersample
from snq_data.ottawa import FS_HZ, download, load, record_names
from snq_recon import (
    envelope_order_spectrum,
    estimate_without_prior,
    fast_kurtogram,
    fault_orders,
    refine_order,
    refine_order_detail,
)
from snq_recon.selectors import fast_band_selector

PACKAGES = ("snq-core", "snq-sim", "snq-data", "snq-recon", "snq-eval")
FAULT_OF = {"inner_race": "BPFI", "outer_race": "BPFO"}
REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------
# intervention
# ----------------------------------------------------------------------
def despike(x: np.ndarray, k_mad: float = 10.0) -> np.ndarray:
    """Replaces isolated single-sample spikes with the neighbours' mean (the review's criterion)."""
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
    return y


def apply_intervention(x: np.ndarray, fs_hz: float, spec: dict, factor: int) -> np.ndarray:
    """Applies the intervention to the full-rate segment, as an ideal FFT mask.

    - ``none`` returns the signal untouched;
    - ``stop`` zeroes the ORIGINAL-frequency bands in ``bands_hz``;
    - ``aaf_ideal`` zeroes everything above the effective Nyquist ``fs/(2*factor)``, so that
      no fold survives the decimation;
    - ``despike`` replaces isolated single-sample spikes with the neighbours' mean (deviation
      above ``k_mad`` MAD, neighbours in the opposite direction and smaller than 0.8 times),
      without zeroing any band;
    - ``median3`` applies a 3-sample moving median.

    ``despike`` and ``median3`` came in on 2026-11-11, pre-registered after the adversarial
    review that found the valley's captor (a train of single-sample spikes in the raw record).

    The mask is zero-phase and not realizable: the goal is causal attribution, not
    realizability.
    """
    kind = spec["kind"]
    if kind == "none":
        return x
    if kind == "despike":
        return despike(x, float(spec.get("k_mad", 10.0)))
    if kind == "median3":
        y = x.copy()
        y[1:-1] = np.median(np.vstack([x[:-2], x[1:-1], x[2:]]), axis=0)
        return y
    xf = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    if kind == "stop":
        for lo, hi in spec["bands_hz"]:
            xf[(f >= float(lo)) & (f <= float(hi))] = 0.0
    elif kind == "aaf_ideal":
        xf[f > fs_hz / (2.0 * factor)] = 0.0
    else:
        raise ValueError(f"unknown intervention: {kind}")
    return np.fft.irfft(xf, n=x.size)


# ----------------------------------------------------------------------
# analysis (same logic as E07)
# ----------------------------------------------------------------------
def guard_verdict(n_samples: int, fs_eff_hz: float, points_per_rev: float, cfg: dict) -> str | None:
    """The three E07 guards, in the same order: min_samples, f_min_hz, samples_per_rev."""
    if n_samples < int(cfg["min_samples"]):
        return "min_samples"
    if fs_eff_hz / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
        return "f_min_hz"
    if points_per_rev < 2.0 * int(cfg["samples_per_rev"]):
        return "samples_per_rev"
    return None


def refine(spec, nominal: float, span: float, n_harmonics: int, tol: float) -> float:
    """Order of highest harmonic energy in a relative neighbourhood of ``nominal`` (E07)."""
    # Centre of the score plateau, not its first point: the old max() biased the truth
    # by half the plateau width (0.0175 order with spr = 64; review of 2026-11-11).
    return refine_order(spec, nominal, span=span, n_harmonics=n_harmonics, tol=tol)


def analyse(x, fs_hz: float, speed, cfg: dict, ceiling_hz: float, selector: str = "sk"):
    """Band selector with ceiling ``min(ceiling_hz, fs/2)`` and envelope order spectrum.

    ``selector="sk"`` is the usual kurtogram. The others (``ses_negentropy``, ``gini``, ...) are
    the robust selectors of `snq_recon.selectors`, on the same partition, pre-registered on
    2026-11-12 (README, "Robust selectors").
    """
    kg = cfg["kurtogram"]
    common = {
        "max_level": int(kg["max_level"]),
        "f_min_hz": float(kg["f_min_hz"]),
        "f_max_hz": min(float(ceiling_hz), fs_hz / 2.0),
    }
    if selector == "sk":
        k = fast_kurtogram(x, fs_hz, **common)
    else:
        k = fast_band_selector(x, fs_hz, statistic=selector, **common)
    spec = envelope_order_spectrum(
        x, fs_hz, speed, k.band_hz, samples_per_rev=int(cfg["samples_per_rev"])
    )
    return k, spec


def run_segment(task: dict) -> list[dict]:
    """All intervention x factor x ceiling combinations of one segment."""
    cfg, meta = task["cfg"], task["meta"]
    vib, spd = task["vib"], task["spd"]
    tol = float(cfg["tolerance_order"])
    # local ruler: the full-rate line of this segment (for scoring only)
    _, spec_full = analyse(vib, FS_HZ, spd, cfg, float(cfg["calibration_ceiling_hz"]))
    loc = refine_order_detail(
        spec_full,
        meta["nominal_order"],
        span=float(cfg["prior"]["calibration_span"]),
        n_harmonics=int(cfg["prior"]["score_harmonics"]),
        tol=tol,
    )
    truth_local = math.nan if loc.at_edge else loc.center
    meta = {
        **meta,
        "truth_local_order": round(truth_local, 4) if math.isfinite(truth_local) else "",
        "speed_seg_hz": round(float(np.mean(spd)), 3),
    }
    rows: list[dict] = []
    for iname, ispec in cfg["interventions"].items():
        filtered = (
            None if ispec["kind"] == "aaf_ideal" else apply_intervention(vib, FS_HZ, ispec, 1)
        )
        for factor in (int(f) for f in cfg["sweep"]["undersampling_factor"]):
            xi = filtered if filtered is not None else apply_intervention(vib, FS_HZ, ispec, factor)
            x, v = undersample(xi, factor), undersample(spd, factor)
            if ispec.get("post") == "despike":  # on the decimated stream: full rate not needed
                x = despike(x, float(ispec.get("k_mad", 10.0)))
            elif ispec.get("post") == "median3":  # median of 3 on the already-decimated stream
                x = apply_intervention(x, FS_HZ / factor, {"kind": "median3"}, 1)
            fs_eff = effective_rate_hz(FS_HZ, factor)
            pts_per_rev = fs_eff / float(np.max(v))
            guard = guard_verdict(x.size, fs_eff, pts_per_rev, cfg)
            for ceiling in (float(c) for c in cfg["sweep"]["search_ceiling_hz"]):
                row = {
                    **meta,
                    "intervention": iname,
                    "factor": factor,
                    "fs_eff_hz": round(fs_eff, 2),
                    "ceiling_hz": ceiling,
                    "estimate_order": "",
                    "error_order": "",
                    "hit": "",
                    "hit_local": "",
                    "band_lo_hz": "",
                    "band_hi_hz": "",
                    "kurtosis": "",
                    "skipped": guard or "",
                }
                if guard is None:
                    try:
                        k, spec = analyse(x, fs_eff, v, cfg, ceiling, ispec.get("selector", "sk"))
                        value = estimate_without_prior(spec, float(cfg["order_min"]))
                        err = abs(value - meta["truth_order"]) if np.isfinite(value) else math.inf
                        row.update(
                            estimate_order=round(float(value), 4) if np.isfinite(value) else "",
                            error_order=round(err, 4) if np.isfinite(err) else "",
                            hit=int(err < tol),
                            hit_local=int(abs(value - truth_local) < tol)
                            if math.isfinite(truth_local) and np.isfinite(value)
                            else "",
                            band_lo_hz=round(k.f_lo_hz, 1),
                            band_hi_hz=round(k.f_hi_hz, 1),
                            kurtosis=round(k.kurtosis, 3),
                        )
                    except ValueError as exc:
                        row["skipped"] = f"ValueError: {str(exc)[:60]}"
                rows.append(row)
    return rows


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p; b and c are the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict]) -> list[dict]:
    """Hits per cell and paired McNemar against ``none`` at the same factor and ceiling."""
    hits: dict[tuple, dict[tuple, int]] = defaultdict(dict)
    for r in rows:
        if r["hit"] == "":
            continue
        cell = (r["condition"], r["factor"], r["ceiling_hz"], r["intervention"])
        hits[cell][(r["record"], r["segment"])] = int(r["hit"])
    out = []
    for (cond, factor, ceiling, iname), pairs in sorted(hits.items()):
        ctrl = hits.get((cond, factor, ceiling, "none"), {})
        common = sorted(set(pairs) & set(ctrl))
        b = sum(1 for k in common if pairs[k] == 1 and ctrl[k] == 0)
        c = sum(1 for k in common if pairs[k] == 0 and ctrl[k] == 1)
        out.append(
            {
                "condition": cond,
                "factor": factor,
                "ceiling_hz": ceiling,
                "intervention": iname,
                "n": len(pairs),
                "hits": sum(pairs.values()),
                "gain_vs_none": b,
                "loss_vs_none": c,
                "mcnemar_p": round(mcnemar_exact_p(b, c), 6),
            }
        )
    return out


# ----------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def source_sha256() -> tuple[str | None, int]:
    """Hash of the contents of packages/*/src/**/*.py (the template's scheme)."""
    files = sorted((REPO_ROOT / "packages").glob("*/src/**/*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return (digest.hexdigest() if files else None), len(files)


def write_manifest(
    config_path: Path, results: Path, cfg: dict, extra: dict, inputs: list[Path]
) -> None:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "nao instalado"
    src_sha, src_n = source_sha256()
    dirty = _git("status", "--porcelain")
    manifest = {
        "experiment": "E20-fold-zone-rejection",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": int(cfg["seed"]),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "input_sha256": {p.name: sha256_file(p) for p in inputs},
        "dataset": "University of Ottawa 2018 (Mendeley v43hmbwxpm), CC BY 4.0",
        "packages": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        # Without a git repository (e.g. a server that received a package), git_* is None and
        # it is source_sha256 + entrypoint_sha256 that identify the code that ran.
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
    ap.add_argument("--workers", type=int, default=None, help="overrides compute.workers")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    results = args.config.parent.parent / cfg["output"]["results_dir"]
    raw = Path(cfg["data_dir"])
    raw = raw if raw.is_absolute() else REPO_ROOT / raw
    seg_n, n_seg = int(cfg["segment_samples"]), int(cfg["segments_per_record"])
    tol = float(cfg["tolerance_order"])
    workers = int(args.workers or cfg["compute"]["workers"])

    nominal = fault_orders(float(cfg["nominal"]["geom_ratio"]), int(cfg["nominal"]["n_elements"]))
    names = record_names(
        tuple(cfg["conditions"]), tuple(cfg["scenarios"]), tuple(int(t) for t in cfg["trials"])
    )

    # pass 1 - calibrate the order of each record at full rate (identical to E07)
    records: dict[str, tuple] = {}
    calib: dict[str, float] = {}
    inputs: list[Path] = []
    for name in names:
        path = download(name, raw)
        rec = load(name, raw)
        speed = rec.speed_profile_hz(int(cfg["pulses_per_rev"]))
        _, spec = analyse(
            rec.vibration[:seg_n], FS_HZ, speed[:seg_n], cfg, float(cfg["calibration_ceiling_hz"])
        )
        fault = FAULT_OF[rec.condition]
        calib[rec.name] = refine(
            spec,
            nominal[fault],
            float(cfg["prior"]["calibration_span"]),
            int(cfg["prior"]["score_harmonics"]),
            tol,
        )
        records[rec.name] = (rec, speed)
        inputs.append(path)
    print(f"{len(records)} records calibrated")

    def truth_loo(name: str, condition: str) -> float:
        others = [v for k, v in calib.items() if k != name and records[k][0].condition == condition]
        return float(np.median(others))

    # pass 2 - one task per (record, segment)
    tasks = []
    for name, (rec, speed) in records.items():
        meta_rec = {
            "record": rec.name,
            "condition": rec.condition,
            "scenario": rec.scenario,
            "trial": rec.trial,
            "fault": FAULT_OF[rec.condition],
            "truth_order": round(truth_loo(name, rec.condition), 4),
            "nominal_order": float(nominal[FAULT_OF[rec.condition]]),
            "rpm_lo": round(float(speed.min() * 60), 1),
            "rpm_hi": round(float(speed.max() * 60), 1),
        }
        for seg in range(n_seg):
            a, b = seg * seg_n, (seg + 1) * seg_n
            if b > rec.vibration.size:
                break
            tasks.append(
                {
                    "cfg": cfg,
                    "meta": {**meta_rec, "segment": seg},
                    "vib": np.ascontiguousarray(rec.vibration[a:b]),
                    "spd": np.ascontiguousarray(speed[a:b]),
                }
            )
    print(f"{len(tasks)} segments, {workers} processes")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, seg_rows in enumerate(pool.map(run_segment, tasks), 1):
            rows.extend(seg_rows)
            if i % 12 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} segments", flush=True)
    order = {k: i for i, k in enumerate(cfg["interventions"])}
    rows.sort(
        key=lambda r: (
            r["record"],
            r["segment"],
            order[r["intervention"]],
            r["factor"],
            -r["ceiling_hz"],
        )
    )

    write_manifest(
        args.config,
        results,
        cfg,
        {
            "nominal_orders": {k: round(v, 4) for k, v in nominal.items()},
            "calibrated_per_record": {k: round(v, 4) for k, v in calib.items()},
            "records": sorted(records),
            "workers": workers,
        },
        inputs=sorted(inputs),
    )
    with (results / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with (results / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # E07 reproduction control: none, 25 kHz ceiling, both races
    want = {int(k): int(v) for k, v in cfg["analysis"]["e07_control_hits"].items()}
    ok = True
    for factor, expected in want.items():
        got = sum(
            s["hits"]
            for s in summary
            if s["intervention"] == "none" and s["factor"] == factor and s["ceiling_hz"] == 25000.0
        )
        ok &= got == expected
        verdict = "OK" if got == expected else "DIFFERENT"
        print(f"E07 control {factor}x: {got}/72 (expected {expected}) {verdict}")
    verdict = "reproduces" if ok else "does NOT reproduce"
    print(f"\n{len(rows)} rows -> {results / 'metrics.csv'}; control {verdict} E07")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
