"""E07 — undersampling degradation in the order domain, with real headroom.

Curve of **undersampling factor × fault-order hit rate** on Ottawa 2018, under variable
rotation, with three distinct priors. It is E02 redone where the dataset is not the bottleneck
— and where the quality of the prior becomes an independent variable instead of an assumption.

Method decisions, all forced by the data and not chosen in advance:

- **Variable rotation, prior in orders.** In hertz the fault frequency moves with
  the operating regime. In orders it is a constant of the geometry. Among the candidates, only
  this dataset has the angular reference that this requires.
- **25 kHz ceiling on the kurtogram search.** Without it, 12 of 12 outer-race
  records chose 50–100 kHz — a band with 0.1% of the energy, where kurtosis measures
  numerical leakage — and the fault did not show up. With the ceiling, the same fault shows up at
  order 3.4 in all of them. The finding inverted the conclusion: the outer race looked
  non-diagnosable, and it was band choice.
- **Three priors, not one.** The nominal prior comes from the ER16K catalog and misses the outer
  race by ~4%; the calibrated one comes from full-rate measurement on the *other* records.
  The distance between the two curves is the cost of prior misalignment — which is
  exactly the measurement/prior boundary that the project commits to reporting.
- **Leave-one-out truth.** The reference order of a record is the median
  measured at full rate on the other records of the same condition. Never on the record itself,
  or factor 1 would be a hit by construction.
- **The encoder does not go through the ADC.** Pulse counting on a digital input
  (TIMER/GPIOTE on the nRF54L15) does not consume sampling budget. What is decimated
  is the vibration; the rotation profile follows along because it varies slowly, and the CSV
  records how many points per revolution remain at each factor.
- **Pure decimation.** `snq_data.undersample` refuses to apply an anti-aliasing filter.

    uv run python experiments/E07-order-domain-ottawa/run.py \
        --config experiments/E07-order-domain-ottawa/config/base.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml
from snq_data import effective_rate_hz, undersample
from snq_data.ottawa import FS_HZ, download, load, record_names
from snq_recon import (
    envelope_order_spectrum,
    estimate_from_candidates,
    estimate_without_prior,
    fast_kurtogram,
    fault_orders,
    refine_order,
    refine_order_detail,
)

PACKAGES = ("snq-core", "snq-sim", "snq-data", "snq-recon", "snq-eval")
FAULT_OF = {"inner_race": "BPFI", "outer_race": "BPFO"}
REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_SHA256_SCHEME = (
    "sha256 over the concatenation, in POSIX path order, of "
    "'<path relative to the repository>\\n<sha256 of the file content>\\n' "
    "for each packages/*/src/**/*.py"
)


def guard_verdict(n_samples: int, fs_eff_hz: float, points_per_rev: float, cfg: dict) -> str | None:
    """Name of the guard that discards this segment, or ``None`` if it passes all three.

    The three guards in one place, so that the count of admitted measurements per
    factor can be verified without running the sweep (``tests/test_guards.py``). The order
    is the order of application, and it matters: the first guard to fire is the one reported.

    - ``min_samples``: length of the decimated vector;
    - ``f_min_hz``: the kurtogram search window ``[f_min_hz, fs_eff/2]`` must
      exist;
    - ``samples_per_rev``: the angular resampling requires at least two points per
      angular increment, at the worst-case rotation speed of the segment.
    """
    if n_samples < int(cfg["min_samples"]):
        return "min_samples"
    if fs_eff_hz / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
        return "f_min_hz"
    if points_per_rev < 2.0 * int(cfg["samples_per_rev"]):
        return "samples_per_rev"
    return None


def harmonics(fundamentals: dict[str, float], n: int) -> np.ndarray:
    """Discrete set of admitted orders: harmonics of each fault family."""
    h = np.arange(1, n + 1, dtype=float)
    return np.unique(np.concatenate([o * h for o in fundamentals.values() if np.isfinite(o)]))


def refine(spec, nominal: float, span: float, n_harmonics: int, tol: float) -> float:
    """Order with the highest harmonic energy in a relative neighborhood of ``nominal``.

    Refining within a range instead of taking the global peak avoids landing on the 2nd harmonic
    of the shaft, which dominates the envelope of several outer-race records and is not a fault.
    """
    # Center of the score plateau, not the first point: the old max() biased the truth
    # by half the plateau width (0.0175 order with spr = 64; review of 2026-11-11).
    return refine_order(spec, nominal, span=span, n_harmonics=n_harmonics, tol=tol)


def analyse(x, fs_hz, speed, cfg) -> tuple[object, object]:
    """Restricted kurtogram + envelope spectrum in orders."""
    kg = cfg["kurtogram"]
    k = fast_kurtogram(
        x,
        fs_hz,
        max_level=int(kg["max_level"]),
        f_min_hz=float(kg["f_min_hz"]),
        f_max_hz=min(float(kg["f_max_hz"]), fs_hz / 2.0),
    )
    spec = envelope_order_spectrum(
        x, fs_hz, speed, k.band_hz, samples_per_rev=int(cfg["samples_per_rev"])
    )
    return k, spec


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def source_sha256(root: Path = REPO_ROOT) -> tuple[str | None, int]:
    """Hash of the content of the sources that this experiment imports.

    ``git_sha`` alone does not identify the code that ran: the tree may be dirty,
    and the package version (``0.1.0``) does not change when the source changes. This hash does.
    The scheme is declared in ``SOURCE_SHA256_SCHEME`` and recorded in the manifest together
    with the value, so that the hash can be recomputed without reading this file.
    """
    files = sorted((root / "packages").glob("*/src/**/*.py"))
    if not files:
        return None, 0
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(root).as_posix()}\n{sha256_file(p)}\n".encode())
    return h.hexdigest(), len(files)


def write_manifest(
    config_path: Path, results: Path, cfg: dict, extra: dict, inputs: list[Path]
) -> None:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    head = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    dirty = [line[3:] for line in porcelain.splitlines()] if porcelain else []
    src_sha, src_n = source_sha256()
    results.mkdir(parents=True, exist_ok=True)
    (results / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "E07-order-domain-ottawa",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "seed": cfg["seed"],
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "input_sha256": {p.name: sha256_file(p) for p in inputs if p.exists()},
                "dataset": "University of Ottawa 2018 (Mendeley v43hmbwxpm), CC BY 4.0",
                "pulses_per_rev": cfg["pulses_per_rev"],
                "ppr_provenance": "determined empirically; not documented by the authors",
                "packages": versions,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_sha": head.strip() if head else None,
                # git_sha points to the commit, not to the tree that ran. The three
                # fields below cover the difference: if git_dirty is true, the
                # commit does NOT describe the executed code, and source_sha256 is the only
                # identifier of what actually ran.
                "git_dirty": bool(dirty),
                "git_dirty_paths": dirty,
                "source_sha256": src_sha,
                "source_sha256_files": src_n,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "source_sha256_scheme": SOURCE_SHA256_SCHEME,
                **extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    root = args.config.parent.parent
    results = root / cfg["output"]["results_dir"]
    # T0.1: a relative data_dir is resolved from the repository root, not from the cwd.
    raw = Path(cfg["data_dir"])
    raw = raw if raw.is_absolute() else REPO_ROOT / raw
    tol = float(cfg["tolerance_order"])
    seg_n = int(cfg["segment_samples"])
    n_seg = int(cfg["segments_per_record"])
    ppr = int(cfg["pulses_per_rev"])
    nh_prior = int(cfg["prior"]["n_harmonics"])
    nh_score = int(cfg["prior"]["score_harmonics"])
    span = float(cfg["prior"]["calibration_span"])

    nominal = fault_orders(float(cfg["nominal"]["geom_ratio"]), int(cfg["nominal"]["n_elements"]))
    prior_nominal = harmonics(nominal, nh_prior)
    names = record_names(
        tuple(cfg["conditions"]), tuple(cfg["scenarios"]), tuple(int(t) for t in cfg["trials"])
    )
    print(f"{len(names)} records; nominal " + ", ".join(f"{k}={v:.3f}" for k, v in nominal.items()))

    # ------------------------------------------------------------------
    # pass 1 — calibrate the orders at full rate, one value per record
    # ------------------------------------------------------------------
    records: dict[str, object] = {}
    calib: dict[str, float] = {}
    inputs: list[Path] = []
    for name in names:
        try:
            path = download(name, raw)
            rec = load(name, raw)
        except (OSError, KeyError, ValueError) as exc:
            print(f"  {name}: skipped ({type(exc).__name__}: {str(exc)[:70]})")
            continue
        speed = rec.speed_profile_hz(ppr)
        _, spec = analyse(rec.vibration[:seg_n], FS_HZ, speed[:seg_n], cfg)
        fault = FAULT_OF[rec.condition]
        calib[rec.name] = refine(spec, nominal[fault], span, nh_score, tol)
        records[rec.name] = (rec, speed)
        inputs.append(path)
        print(
            f"  {rec.name}: {fault} calibrated = {calib[rec.name]:.3f}"
            f" (nominal {nominal[fault]:.3f})"
        )

    if not records:
        print("\nno record loaded")
        return 1

    # Leave-one-out truth: median of the other records of the same condition.
    def truth_loo(name: str, condition: str) -> float:
        others = [v for k, v in calib.items() if k != name and records[k][0].condition == condition]
        return float(np.median(others)) if others else float("nan")

    by_cond = {
        c: [calib[n] for n in calib if records[n][0].condition == c]
        for c in {r[0].condition for r in records.values()}
    }
    for c, v in by_cond.items():
        nom = nominal[FAULT_OF[c]]
        print(
            f"  {c}: calibrated {np.median(v):.3f} (min {min(v):.3f}, max {max(v):.3f}), "
            f"nominal {nom:.3f}, deviation {100 * (np.median(v) / nom - 1):+.1f}%"
        )

    # ------------------------------------------------------------------
    # pass 2 — undersampling sweep
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for name, (rec, speed) in records.items():
        fault = FAULT_OF[rec.condition]
        truth = truth_loo(name, rec.condition)
        loo = {f: (truth if f == fault else nominal[f] * truth / nominal[fault]) for f in nominal}
        prior_calib = harmonics(loo, nh_prior)

        for seg in range(n_seg):
            a, b = seg * seg_n, (seg + 1) * seg_n
            if b > rec.vibration.size:
                break
            vib_seg, spd_seg = rec.vibration[a:b], speed[a:b]
            # local ruler: the full-rate spectral line of this segment (for scoring only)
            _, spec_full = analyse(vib_seg, FS_HZ, spd_seg, cfg)
            loc = refine_order_detail(
                spec_full, nominal[fault], span=span, n_harmonics=nh_score, tol=tol
            )
            truth_local = float("nan") if loc.at_edge else loc.center
            speed_seg = float(np.mean(spd_seg))

            for factor in cfg["sweep"]["undersampling_factor"]:
                factor = int(factor)
                x, v = undersample(vib_seg, factor), undersample(spd_seg, factor)
                fs_eff = effective_rate_hz(FS_HZ, factor)
                pts_per_rev = fs_eff / float(np.max(v))
                if guard_verdict(x.size, fs_eff, pts_per_rev, cfg) is not None:
                    continue

                try:
                    k, spec = analyse(x, fs_eff, v, cfg)
                    est = {
                        "sem_prior": estimate_without_prior(spec, float(cfg["order_min"])),
                        "prior_nominal": estimate_from_candidates(
                            spec, prior_nominal, n_harmonics=nh_score, tol_hz=tol
                        )[0],
                        "prior_calibrado": estimate_from_candidates(
                            spec, prior_calib, n_harmonics=nh_score, tol_hz=tol
                        )[0],
                    }
                except ValueError as exc:
                    print(f"    {name} f={factor}: skipped ({str(exc)[:55]})")
                    continue

                for mode, value in est.items():
                    err = abs(value - truth) if np.isfinite(value) else float("inf")
                    rows.append(
                        {
                            "record": rec.name,
                            "condition": rec.condition,
                            "scenario": rec.scenario,
                            "trial": rec.trial,
                            "segment": seg,
                            "fault": fault,
                            "rpm_lo": round(float(speed.min() * 60), 1),
                            "rpm_hi": round(float(speed.max() * 60), 1),
                            "factor": factor,
                            "fs_eff_hz": round(fs_eff, 2),
                            "mode": mode,
                            "truth_order": round(truth, 4),
                            "nominal_order": round(nominal[fault], 4),
                            "estimate_order": round(float(value), 4) if np.isfinite(value) else "",
                            "error_order": round(err, 4) if np.isfinite(err) else "",
                            "hit": int(err < tol),
                            "truth_local_order": round(truth_local, 4)
                            if np.isfinite(truth_local)
                            else "",
                            "hit_local": int(abs(value - truth_local) < tol)
                            if np.isfinite(truth_local) and np.isfinite(value)
                            else "",
                            "speed_seg_hz": round(speed_seg, 3),
                            "band_lo_hz": round(k.f_lo_hz, 1),
                            "band_hi_hz": round(k.f_hi_hz, 1),
                            "kurtosis": round(k.kurtosis, 3),
                            "points_per_rev": round(pts_per_rev, 1),
                            "n_samples": x.size,
                        }
                    )
        print(f"    {name}: {len(rows)} rows accumulated")

    if not rows:
        print("\nno rows produced")
        return 1

    write_manifest(
        args.config,
        results,
        cfg,
        {
            "nominal_orders": {k: round(v, 4) for k, v in nominal.items()},
            "calibrated_per_record": {k: round(v, 4) for k, v in calib.items()},
            "records": sorted(records),
            # The ground truth depends on samples_per_rev: the orders above are
            # measured in the angular resampling of this config. Comparing two runs with
            # different spr is comparing two distinct references.
            "samples_per_rev": int(cfg["samples_per_rev"]),
            "min_samples": int(cfg["min_samples"]),
        },
        inputs=sorted(inputs),
    )
    out = results / "metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
