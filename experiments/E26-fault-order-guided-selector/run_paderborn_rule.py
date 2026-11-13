"""E26b on Paderborn: the guided identification rule with other band choices.

The E26 confirmation changed two things at the same time, the band and the identification rule.
Here the rule stays fixed: the score of each fault is the mean prominence in the windows of the
harmonics of the geometric order, and the decision uses the healthy threshold. Only the band
changes:

- ``kurtogram``: the band of the fast kurtogram, up to the effective Nyquist (as in E24);
- ``ses_infogram``: the band of the SES infogram, on the same partition;
- ``wideband``: [f_min, fs/2], without selection (the classical envelope without a kurtogram);
- ``random_band``: a band drawn at random from the partition of the guided selector, with a seed per
  segment.

The guided selector of E26 is the maximum of the same score over the 23 bands of the partition. It
comes from ``results-paderborn/metrics.csv`` and is not recomputed here.

    uv run python experiments/E26-fault-order-guided-selector/run_paderborn_rule.py \
        --config experiments/E26-fault-order-guided-selector/config/paderborn_rule.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import platform
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from snq_data import paderborn as pb
from snq_data import undersample
from snq_recon.highres import envelope_order_spectrum

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guided = _load("e26guided", EXP / "guided.py")
base = _load("e26run", EXP / "run_paderborn.py")
e24 = base.e24
FS = pb.FS_HZ
ORDERS = base.ORDERS
CLEAN = base.CLEAN
# the cleaning labels as the run wrote them, before they were renamed to English
# (tools/relabel_map.py): the seed of each segment is derived from them, so a rerun draws the
# same random bands
LEGACY_CLEAN = {"raw": "bruto", "median3": "mediana3"}


def segment_seed(code: str, setting: str, n: int, factor: int, clean: str) -> int:
    """Seed of the random band of one segment, from the legacy label of its cleaning."""
    return zlib.crc32(f"{code}|{setting}|{n}|{factor}|{LEGACY_CLEAN[clean]}".encode())


def band_scores(x, v, fs: float, band, cfg: dict) -> dict[str, float]:
    """Guided score of each fault in a single band: mean prominence in the harmonic windows."""
    g = cfg["guided"]
    try:
        spec = envelope_order_spectrum(x, fs, v, band, samples_per_rev=int(cfg["samples_per_rev"]))
    except ValueError:
        return {c: math.nan for c in ORDERS}
    o, a = np.asarray(spec.freqs_hz), np.asarray(spec.magnitude)
    out = {}
    for c, order in ORDERS.items():
        vals = [
            guided.prominence(
                o,
                a,
                guided.windows(
                    order, int(k), float(g["window_frac"]), float(g["integer_exclusion_order"])
                ),
            )
            for k in g["harmonics"]
        ]
        out[c] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
    return out


def bands_for(x, fs: float, cfg: dict, seed: int) -> dict[str, tuple[float, float]]:
    f_min = float(cfg["kurtogram"]["f_min_hz"])
    out = {}
    for name, sel in (("kurtogram", "sk"), ("ses_infogram", "ses_negentropy")):
        with contextlib.suppress(ValueError):  # empty search: the strategy gets no row
            out[name] = tuple(float(b) for b in e24.select_band(x, fs, sel, cfg).band_hz)
    out["wideband"] = (f_min, fs / 2.0)
    part = guided.bands(fs, f_min, int(cfg["guided"]["max_level"]))
    out["random_band"] = part[int(np.random.default_rng(seed).integers(len(part)))]
    return out


def task(job) -> dict:
    cfg, root, code, setting = job
    race = base.race_of(code)
    n_seg = int(cfg["segment_samples"])
    out: dict = {"rows": [], "sha256": {}, "unreadable": {}}
    for n in (int(t) for t in cfg["test_measurements"]):
        path = pb.measurement_path(root, code, setting, n)
        if not path.exists():
            out["unreadable"][n] = "missing"
            continue
        out["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            m = pb.load_measurement(root, code, setting, n, n_samples=n_seg, min_samples=n_seg)
        except ValueError as exc:
            out["unreadable"][n] = str(exc)[:120]
            continue
        x, v = m.vibration, m.speed_profile_hz()
        for f in (int(f) for f in cfg["factors"]):
            xd, vd, fs = undersample(x, f), undersample(v, f), FS / f
            if not base.guard_ok(xd, vd, fs, cfg):
                continue
            for clean, fn in CLEAN.items():
                xc = fn(xd)
                seed = segment_seed(code, setting, n, f, clean)
                for strat, band in bands_for(xc, fs, cfg, seed).items():
                    sc = band_scores(xc, vd, fs, band, cfg)
                    out["rows"].append(
                        {
                            "bearing": code,
                            "race": race,
                            "setting": setting,
                            "measurement": n,
                            "factor": f,
                            "clean": clean,
                            "strategy": strat,
                            "band": f"{band[0]:.0f}-{band[1]:.0f}",
                            **{
                                f"score_{c}": round(s, 4) if np.isfinite(s) else ""
                                for c, s in sc.items()
                            },
                        }
                    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bearings", nargs="*", default=None, help="smoke test only")
    ap.add_argument("--settings", nargs="*", default=None, help="smoke test only")
    ap.add_argument("--measurements", nargs="*", type=int, default=None, help="smoke test only")
    ap.add_argument("--factors", nargs="*", type=int, default=None, help="smoke test only")
    ap.add_argument("--results", type=str, default=None, help="smoke test only")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.measurements:
        cfg["test_measurements"] = args.measurements
    if args.factors:
        cfg["factors"] = args.factors
    root = ROOT / cfg["data_dir"]
    codes = args.bearings or [*cfg["outer"], *cfg["inner"], *cfg["healthy"]]
    jobs = [(cfg, str(root), c, s) for c in codes for s in (args.settings or cfg["settings"])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers or int(cfg["compute"]["workers"])) as pool:
        res = list(pool.map(task, jobs))
    print(f"{len(res)} records in {time.time() - t0:.0f} s", flush=True)
    out = EXP / (args.results or cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for o in res for r in o["rows"]]
    base.write_csv(out / "metrics.csv", rows)
    src = hashlib.sha256()
    for p in sorted((ROOT / "packages").glob("*/src/**/*.py")):
        src.update(p.relative_to(ROOT).as_posix().encode())
        src.update(p.read_bytes())
    manifest = {
        "experiment": "E26b-fault-order-guided-selector (Paderborn, fixed rule)",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": args.config.as_posix(),
        "config_sha256": base.sha256_file(args.config),
        "entrypoint_sha256": base.sha256_file(Path(__file__)),
        "run_paderborn_sha256": base.sha256_file(EXP / "run_paderborn.py"),
        "guided_sha256": base.sha256_file(EXP / "guided.py"),
        "readme_sha256": base.sha256_file(EXP / "README.md"),
        "e24_run_sha256": base.sha256_file(base.E24_RUN),
        "source_sha256": src.hexdigest(),
        "overrides": {
            "bearings": args.bearings,
            "settings": args.settings,
            "measurements": args.measurements,
            "factors": args.factors,
        },
        "input_sha256": {k: v for o in res for k, v in o["sha256"].items()},
        "unreadable": {str(i): o["unreadable"] for i, o in enumerate(res) if o["unreadable"]},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n_rows": len(rows),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
