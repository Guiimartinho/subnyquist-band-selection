"""E22 - published baselines at the same samples/s rate (ADR 0006), phase 1.

Arms (see README): uniforme, uniforme_teto (ceiling by the TAT), taxa_tat (factor by the TAT),
oraculo_uniforme (upper bound, computed in the analysis from `uniforme_grid`),
aleatorio_cs (sensors2025csvibration, via E13) and passa_faixa_lu (lu2018undersampled).

Each record has a full-rate calibration part (Ottawa: segment 0; MAFAULDA: 1st
half) and test parts. Only the arms that declare that they use calibration use it.

    uv run python experiments/E22-equal-rate-baselines/run.py \
        --config experiments/E22-equal-rate-baselines/config/base.yaml

Helpers copied from E13/E20/E21, because nothing imports experiments/.
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
from snq_core.sampling import fold_band
from snq_data import mafaulda as maf
from snq_data import ottawa, undersample
from snq_recon import (
    envelope_order_spectrum,
    estimate_without_prior,
    fast_kurtogram,
    fault_orders,
    interpolated_speed_hz,
    omp,
    order_basis,
    phase_revolutions,
    random_time_indices,
    reconstruct_from_orders,
    refine_order,
    refine_order_detail,
)

PACKAGES = ("snq-core", "snq-sim", "snq-data", "snq-recon", "snq-eval")
REPO_ROOT = Path(__file__).resolve().parents[2]
FAULT_OF = {"inner_race": "BPFI", "outer_race": "BPFO"}


def at_repo(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


# ----------------------------------------------------------------------
# signal and classical pipeline (E07/E20/E21)
# ----------------------------------------------------------------------
def fft_mask(x: np.ndarray, fs_hz: float, keep) -> np.ndarray:
    xf = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    xf[~keep(f)] = 0.0
    return np.fft.irfft(xf, n=x.size)


def aaf_ideal(x: np.ndarray, fs_hz: float, factor: int) -> np.ndarray:
    return x if factor == 1 else fft_mask(x, fs_hz, lambda f: f <= fs_hz / (2.0 * factor))


def band_energy(x: np.ndarray, fs_hz: float, lo_hz: float, hi_hz: float) -> float:
    xf = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    return float(np.sum(xf[(f >= lo_hz) & (f <= hi_hz)]))


def ceiling_at(ceiling0_hz: float | None, fs_eff_hz: float, frac: float = 1.0) -> float:
    base = fs_eff_hz / 2.0 if ceiling0_hz is None else min(float(ceiling0_hz), fs_eff_hz / 2.0)
    return base * frac


def guard_verdict(n_samples: int, fs_eff_hz: float, points_per_rev: float, cfg: dict) -> str | None:
    if n_samples < int(cfg["min_samples"]):
        return "min_samples"
    if fs_eff_hz / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
        return "f_min_hz"
    if points_per_rev < 2.0 * int(cfg["samples_per_rev"]):
        return "samples_per_rev"
    return None


def refine(spec, nominal: float, span: float, n_harmonics: int, tol: float) -> float:
    # Centre of the score plateau, not its first point: the old max() biased the truth by
    # half the plateau width (0.0175 order with spr = 64; review of 2026-11-11).
    return refine_order(spec, nominal, span=span, n_harmonics=n_harmonics, tol=tol)


def kurtogram(x, fs_hz: float, f_max_hz: float, cfg: dict):
    kg = cfg["kurtogram"]
    return fast_kurtogram(
        x, fs_hz, max_level=int(kg["max_level"]), f_min_hz=float(kg["f_min_hz"]), f_max_hz=f_max_hz
    )


def despike(x: np.ndarray, k_mad: float = 10.0) -> np.ndarray:
    """Replaces isolated one-sample spikes by the neighbours' mean (criterion of the E20 review).

    Deviation above ``k_mad`` MADs from the mean of the neighbours, with both neighbours
    deviating in the opposite direction and by less than 0.8 times as much. It zeroes no band.
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
    return y


def classic(x, v, fs_hz: float, factor: int, f_max_hz: float, cfg: dict) -> dict:
    """E07 pipeline at one factor: estimate without prior, band and reason for skipping."""
    xd, vd = undersample(x, factor), undersample(v, factor)
    fs_eff = fs_hz / factor
    guard = guard_verdict(xd.size, fs_eff, fs_eff / float(np.max(vd)), cfg)
    if guard:
        return {"skipped": guard}
    if f_max_hz <= float(cfg["kurtogram"]["f_min_hz"]):
        return {"skipped": "teto abaixo do piso do kurtograma"}
    try:
        k = kurtogram(xd, fs_eff, f_max_hz, cfg)
        spec = envelope_order_spectrum(
            xd, fs_eff, vd, k.band_hz, samples_per_rev=int(cfg["samples_per_rev"])
        )
        est = estimate_without_prior(spec, float(cfg["order_min"]))
    except ValueError as exc:
        return {"skipped": f"ValueError: {str(exc)[:60]}"}
    return {
        "estimate": float(est),
        "band_lo_hz": round(k.f_lo_hz, 1),
        "band_hi_hz": round(k.f_hi_hz, 1),
    }


def tat_admissible(
    x_cal, fs_hz: float, factor: int, f_max_hz: float, cfg: dict
) -> tuple[bool, float]:
    """E21 TAT: share of the energy of the chosen band that comes from folded content."""
    fs_eff = fs_hz / factor
    if f_max_hz <= float(cfg["kurtogram"]["f_min_hz"]) or fs_eff / 2.0 <= float(
        cfg["kurtogram"]["f_min_hz"]
    ):
        return False, math.nan  # empty search range: nothing to evaluate, not admissible
    xd = undersample(x_cal, factor)
    xa = undersample(aaf_ideal(x_cal, fs_hz, factor), factor)
    try:
        k = kurtogram(xd, fs_eff, f_max_hz, cfg)
    except ValueError:
        return False, math.nan
    e = band_energy(xd, fs_eff, k.f_lo_hz, k.f_hi_hz)
    share = 1.0 - band_energy(xa, fs_eff, k.f_lo_hz, k.f_hi_hz) / e if e > 0 else math.nan
    return bool(share <= float(cfg["tat"]["folded_share_threshold"])), share


def arm_cs(x, v, fs_hz: float, indices, cfg: dict) -> dict:
    """sensors2025csvibration as in E13: order basis + OMP, reconstructs and demodulates."""
    c = cfg["cs"]
    n_total = x.size
    dense_speed = interpolated_speed_hz(indices, v[indices], n_total)
    phase_dense = phase_revolutions(dense_speed, fs_hz, n_total)
    phase = phase_dense[indices]
    revs = float(phase[-1] - phase[0])
    step = float(c["order_step_per_rev"]) / revs
    orders = np.arange(float(c["order_min"]), float(c["order_max"]) + 0.5 * step, step)
    if indices.size * 2.0 * orders.size > float(c["max_dictionary_elements"]):
        return {"skipped": f"dicionario {indices.size}x{2 * orders.size} acima do teto"}
    sol = omp(
        order_basis(phase, orders),
        x[indices],
        n_nonzero=int(c["n_atoms"]),
        tol=float(c["residual_tol"]),
    )
    x_hat = reconstruct_from_orders(phase_dense, orders, sol.coefficients)
    spec = envelope_order_spectrum(
        x_hat, fs_hz, dense_speed, None, samples_per_rev=int(c["cs_samples_per_rev"])
    )
    return {"estimate": float(estimate_without_prior(spec, float(cfg["order_min"])))}


def in_one_zone(lo_hz: float, hi_hz: float, fs_eff_hz: float) -> bool:
    """The band [lo, hi] fits in a single Nyquist zone of fs_eff (upper edge closed)."""
    half = fs_eff_hz / 2.0
    return math.floor(lo_hz / half) == math.ceil(hi_hz / half) - 1


def arm_lu(
    x,
    v,
    fs_hz: float,
    bands: list[tuple[float, float, float]],
    budget: int,
    grid: list[int],
    cfg: dict,
) -> dict:
    """lu2018undersampled: analog band-pass on the resonance + band-pass sampling.

    As in Lu et al., the band is chosen in the calibration and has to be samplable: among the
    bands of the calibration kurtogram, in decreasing order of kurtosis, the winner is the first
    one that has an integer factor within the budget with the whole band in one Nyquist zone.
    """
    for lo, hi, kurt in bands:
        for factor in (m for m in grid if m >= budget):
            fs_eff = fs_hz / factor
            if not in_one_zone(lo, hi, fs_eff):
                continue
            xb = fft_mask(x, fs_hz, lambda f, lo=lo, hi=hi: (f >= lo) & (f <= hi))
            xd, vd = undersample(xb, factor), undersample(v, factor)
            if guard_verdict(xd.size, fs_eff, fs_eff / float(np.max(vd)), cfg):
                continue
            fb = fold_band(lo, hi, fs_eff)
            spec = envelope_order_spectrum(
                xd, fs_eff, vd, (fb.lo_hz, fb.hi_hz), samples_per_rev=int(cfg["samples_per_rev"])
            )
            return {
                "estimate": float(estimate_without_prior(spec, float(cfg["order_min"]))),
                "chosen_factor": factor,
                "band_lo_hz": round(lo, 1),
                "band_hi_hz": round(hi, 1),
                "lu_band_kurtosis": round(kurt, 3),
            }
    return {"skipped": "nenhuma banda da calibracao e amostravel no orcamento"}


# ----------------------------------------------------------------------
# one record
# ----------------------------------------------------------------------
def load_record(ds: str, rid: str, cfg: dict):
    d = cfg["datasets"][ds]
    if ds == "ottawa":
        raw = at_repo(d["data_dir"])
        path = ottawa.download(rid, raw)
        rec = ottawa.load(rid, raw)
        x, v, fs = rec.vibration, rec.speed_profile_hz(int(d["pulses_per_rev"])), ottawa.FS_HZ
        n = int(d["segment_samples"])
        seg = lambda s: (x[s * n : (s + 1) * n], v[s * n : (s + 1) * n])  # noqa: E731
        cal = seg(int(d["calibration_segment"]))
        tests = [(s, *seg(s)) for s in d["test_segments"]]
        nominal = fault_orders(float(d["nominal"]["geom_ratio"]), int(d["nominal"]["n_elements"]))[
            FAULT_OF[rec.condition]
        ]
        return rec.condition, cal, tests, fs, nominal, path
    root = at_repo(d["data_dir"])
    rec = maf.load(rid, root, cache_dir=at_repo(d["cache_dir"]))
    x, v = rec.vibration(axis=d["axis"]), rec.speed_profile_hz()
    h = x.size // 2
    return (
        rec.bearing,
        (x[:h], v[:h]),
        [(1, x[h:], v[h:])],
        rec.fs_hz,
        maf.DOCUMENTED_ORDERS[d["fault_family"]],
        root / rid,
    )


def run_record(task: dict) -> dict:
    cfg, ds, rid = task["cfg"], task["dataset"], task["id"]
    d = cfg["datasets"][ds]
    group, (xc, vc), tests, fs, nominal, path = load_record(ds, rid, cfg)
    tol = float(cfg["tolerance_order"])
    c0 = d["ceiling0_hz"]
    budgets = [int(b) for b in d["budgets"]]
    grid = list(range(min(budgets), int(d["factor_grid_max"]) + 1))

    # full-rate calibration: calibrated order and band for the Lu arm
    k_full = kurtogram(xc, fs, ceiling_at(c0, fs), cfg)
    spec = envelope_order_spectrum(
        xc, fs, vc, k_full.band_hz, samples_per_rev=int(cfg["samples_per_rev"])
    )
    calib = refine(
        spec,
        nominal,
        float(cfg["prior"]["calibration_span"]),
        int(cfg["prior"]["score_harmonics"]),
        tol,
    )
    tat_cache: dict[tuple[int, float], bool] = {}

    def admissible(factor: int, f_max: float) -> bool:
        key = (factor, round(f_max, 3))
        if key not in tat_cache:
            tat_cache[key] = tat_admissible(xc, fs, factor, f_max, cfg)[0]
        return tat_cache[key]

    rows = []
    base0 = {"dataset": ds, "record": rid, "group": group}
    for seg, xt, vt in tests:
        # local ruler (review of 2026-11-11): the full-rate line of THIS segment, only for scoring
        k_seg = kurtogram(xt, fs, ceiling_at(c0, fs), cfg)
        spec_seg = envelope_order_spectrum(
            xt, fs, vt, k_seg.band_hz, samples_per_rev=int(cfg["samples_per_rev"])
        )
        loc = refine_order_detail(
            spec_seg,
            nominal,
            span=float(cfg["prior"]["calibration_span"]),
            n_harmonics=int(cfg["prior"]["score_harmonics"]),
            tol=tol,
        )
        base = {
            **base0,
            "truth_local_order": "" if loc.at_edge else round(loc.center, 4),
            "speed_seg_hz": round(float(np.mean(vt)), 3),
        }
        uni = {m: classic(xt, vt, fs, m, ceiling_at(c0, fs / m), cfg) for m in grid}
        for m in grid:
            rows.append(
                {
                    **base,
                    "segment": seg,
                    "budget": m,
                    "arm": "uniforme_grid",
                    "chosen_factor": m,
                    **uni[m],
                }
            )
        for b in budgets:
            arms = {}
            arms["uniforme"] = {**uni[b], "chosen_factor": b}
            if "despike" in cfg["arms"]:
                # removal of one-sample spikes before decimating (review of 2026-11-11)
                arms["despike"] = {
                    **classic(despike(xt), vt, fs, b, ceiling_at(c0, fs / b), cfg),
                    "chosen_factor": b,
                }
            fracs = [float(f) for f in cfg["tat"]["ceiling_fractions"]]
            f_min = float(cfg["kurtogram"]["f_min_hz"])
            valid = [f for f in fracs if ceiling_at(c0, fs / b, f) > f_min] or [fracs[0]]
            # the highest ceiling that the TAT admits; if none, the most protective valid one
            frac = next((f for f in valid if admissible(b, ceiling_at(c0, fs / b, f))), valid[-1])
            arms["uniforme_teto"] = {
                **classic(xt, vt, fs, b, ceiling_at(c0, fs / b, frac), cfg),
                "chosen_factor": b,
                "ceiling_frac": frac,
            }
            m_tat = next(
                (
                    m
                    for m in grid
                    if m >= b and "skipped" not in uni[m] and admissible(m, ceiling_at(c0, fs / m))
                ),
                b,
            )
            arms["taxa_tat"] = {**uni[m_tat], "chosen_factor": m_tat}
            if "aleatorio_cs" in cfg["arms"]:
                rng = np.random.default_rng(
                    [
                        int(cfg["seed"]),
                        int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16),
                        seg,
                        b,
                    ]
                )
                idx = random_time_indices(
                    xt.size,
                    rng,
                    n_selected=xt.size // b,
                    min_gap_samples=int(cfg["cs"]["min_gap_samples"]),
                )
                try:
                    arms["aleatorio_cs"] = {**arm_cs(xt, vt, fs, idx, cfg), "chosen_factor": b}
                except (ValueError, MemoryError, np.linalg.LinAlgError) as exc:
                    arms["aleatorio_cs"] = {
                        "skipped": f"{type(exc).__name__}: {str(exc)[:50]}",
                        "chosen_factor": b,
                    }
            if "passa_faixa_lu" in cfg["arms"]:
                lu_bands = sorted(
                    ((lo, hi, k) for _lvl, lo, hi, k in k_full.grid), key=lambda t: -t[2]
                )
                try:
                    arms["passa_faixa_lu"] = arm_lu(xt, vt, fs, lu_bands, b, grid, cfg)
                except ValueError as exc:
                    arms["passa_faixa_lu"] = {"skipped": f"ValueError: {str(exc)[:60]}"}
            for arm, res in arms.items():
                rows.append({**base, "segment": seg, "budget": b, "arm": arm, **res})
    return {
        "dataset": ds,
        "id": rid,
        "group": group,
        "calib": calib,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "rows": rows,
    }


def run_record_safe(task: dict) -> dict:
    """A record with an error becomes a record with an error, and does not bring down the run.

    In the first complete run (2026-11-11), a ValueError in one record aborted the pool and
    lost everything that had already been processed.
    """
    try:
        return run_record(task)
    except Exception as exc:  # noqa: BLE001
        return {
            "dataset": task["dataset"],
            "id": task["id"],
            "error": f"{type(exc).__name__}: {exc}",
        }


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------
def summarize(rows: list[dict]) -> list[dict]:
    cell = defaultdict(lambda: [0, 0, 0])  # n, hits, skipped
    for r in rows:
        k = (r["dataset"], r["group"], r["budget"], r["arm"])
        if r.get("hit", "") == "":
            cell[k][2] += 1
        else:
            cell[k][0] += 1
            cell[k][1] += int(r["hit"])
    out = []
    for (ds, g, b, arm), (n, h, sk) in sorted(
        cell.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])
    ):
        if arm == "uniforme_grid":
            continue
        out.append(
            {"dataset": ds, "group": g, "budget": b, "arm": arm, "n": n, "hits": h, "skipped": sk}
        )
    # oracle: the best uniform factor >= budget, chosen WITH the test label
    grid = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["arm"] == "uniforme_grid" and r.get("hit", "") != "":
            g = grid[(r["dataset"], r["group"], r["chosen_factor"])]
            g[0] += 1
            g[1] += int(r["hit"])
    for ds, g, b in sorted({(o["dataset"], o["group"], o["budget"]) for o in out}):
        cands = [
            (h / n, f, n, h)
            for (d2, g2, f), (n, h) in grid.items()
            if d2 == ds and g2 == g and f >= b and n
        ]
        if cands:
            _, f, n, h = max(cands)
            out.append(
                {
                    "dataset": ds,
                    "group": g,
                    "budget": b,
                    "arm": "oraculo_uniforme",
                    "n": n,
                    "hits": h,
                    "skipped": 0,
                    "oracle_factor": f,
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
            versions[pkg] = "not installed"
    src_sha, src_n = source_sha256()
    dirty = _git("status", "--porcelain")
    manifest = {
        "experiment": "E22-equal-rate-baselines",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": int(cfg["seed"]),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "entrypoint_sha256": sha256_file(Path(__file__)),
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
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of datasets")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records per dataset")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = args.config.parent.parent / cfg["output"]["results_dir"]

    tasks = []
    for ds in args.datasets or list(cfg["datasets"]):
        d = cfg["datasets"][ds]
        if ds == "ottawa":
            ids = ottawa.record_names(
                tuple(d["conditions"]), tuple(d["scenarios"]), tuple(int(t) for t in d["trials"])
            )
        else:
            screen = set(
                yaml.safe_load(at_repo(d["screen_catalog"]).read_text(encoding="utf-8"))["excluded"]
            )
            ids = [
                r
                for r in maf.list_records(at_repo(d["data_dir"]), subsets=tuple(d["bearings"]))
                if Path(r).parts[1] == d["fault"] and r not in screen
            ]
        ids = ids[: args.limit] if args.limit else ids
        tasks += [{"cfg": cfg, "dataset": ds, "id": i} for i in ids]
    workers = int(args.workers or cfg["compute"]["workers"])
    print(f"{len(tasks)} records, {workers} processes", flush=True)

    out, failed = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(run_record_safe, tasks), 1):
            (failed if "error" in res else out).append(res)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} records ({len(failed)} with errors)", flush=True)
    for f in failed:
        print(f"  ERROR {f['dataset']}:{f['id']}: {f['error']}", flush=True)

    tol = float(cfg["tolerance_order"])
    rows = []
    for o in out:
        peers = [
            p["calib"]
            for p in out
            if p["dataset"] == o["dataset"] and p["group"] == o["group"] and p["id"] != o["id"]
        ]
        truth = float(np.median(peers))
        for r in o["rows"]:
            est = r.get("estimate")
            ok = est is not None and np.isfinite(est) and "skipped" not in r
            r.update(
                truth_order=round(truth, 4),
                estimate=round(est, 4) if ok else "",
                hit=int(abs(est - truth) < tol) if ok else "",
                hit_local=int(abs(est - float(r["truth_local_order"])) < tol)
                if ok and r.get("truth_local_order", "") != ""
                else "",
                skipped=r.get("skipped", ""),
            )
            rows.append(r)
    summary = summarize(rows)
    write_manifest(
        args.config,
        results,
        cfg,
        {
            "n_records": len(out),
            "failed_records": {f"{f['dataset']}:{f['id']}": f["error"] for f in failed},
            "input_sha256": {f"{o['dataset']}:{o['id']}": o["sha256"] for o in out},
            "calibrated_order": {f"{o['dataset']}:{o['id']}": round(o["calib"], 4) for o in out},
            "limit": args.limit,
        },
    )
    for name, data in (("metrics.csv", rows), ("summary.csv", summary)):
        cols = list(dict.fromkeys(k for r in data for k in r))  # union of the keys, in order
        with (results / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(data)
    print(f"\n{len(rows)} rows -> {results / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
