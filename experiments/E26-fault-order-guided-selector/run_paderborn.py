"""E26 confirmatório no Paderborn: kurtograma cego e seletor guiado, com e sem mediana no fluxo
decimado. Pré-registro no README ("Confirmação no Paderborn"); a análise é ``analyze_paderborn.py``.

    uv run python experiments/E26-fault-order-guided-selector/run_paderborn.py \
        --config experiments/E26-fault-order-guided-selector/config/paderborn.yaml [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml
from snq_data import paderborn as pb
from snq_data import undersample
from snq_recon import estimate_without_prior, fast_kurtogram
from snq_recon.highres import envelope_order_spectrum

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
E24_RUN = ROOT / "experiments/E24-paderborn-spike-capture/run.py"
PACKAGES = ("snq-core", "snq-sim", "snq-data", "snq-recon", "snq-eval")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guided = _load("e26guided", EXP / "guided.py")
e24 = _load("e24run", E24_RUN)
FS = pb.FS_HZ
ORDERS = {c: pb.computed_orders()[c] for c in ("BPFO", "BPFI")}
CLEAN = {"bruto": lambda x: x, "mediana3": guided.median3}


def race_of(code: str) -> str:
    origin, race, _level = pb.BEARINGS[code]
    return "H" if origin == "healthy" else race


def guard_ok(xd: np.ndarray, vd: np.ndarray, fs: float, cfg: dict) -> bool:
    g = float(cfg["guard_points_per_rev_factor"]) * int(cfg["samples_per_rev"])
    return (
        xd.size >= int(cfg["min_samples"])
        and fs / 2.0 > float(cfg["kurtogram"]["f_min_hz"])
        and fs / float(np.max(vd)) >= g
    )


def blind(x: np.ndarray, v: np.ndarray, fs: float, cfg: dict):
    """O pipeline clássico do E24: kurtograma até o Nyquist efetivo e argmax sem prior."""
    kg = cfg["kurtogram"]
    k = fast_kurtogram(
        x, fs, max_level=int(kg["max_level"]), f_min_hz=float(kg["f_min_hz"]), f_max_hz=fs / 2.0
    )
    spec = envelope_order_spectrum(x, fs, v, k.band_hz, samples_per_rev=int(cfg["samples_per_rev"]))
    return float(estimate_without_prior(spec, float(cfg["order_min"]))), k.band_hz


def task(job) -> dict:
    cfg, root, code, setting = job
    race = race_of(code)
    g = cfg["guided"]
    n_seg = int(cfg["segment_samples"])
    cal = int(cfg["calibration_measurement"])
    out: dict = {
        "code": code,
        "setting": setting,
        "race": race,
        "calib": None,
        "rows": [],
        "sha256": {},
        "unreadable": {},
    }
    for n in [cal, *[int(t) for t in cfg["test_measurements"]]]:
        path = pb.measurement_path(root, code, setting, n)
        if not path.exists():
            out["unreadable"][n] = "ausente"
            continue
        out["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            m = pb.load_measurement(root, code, setting, n, n_samples=n_seg, min_samples=n_seg)
        except ValueError as exc:
            out["unreadable"][n] = str(exc)[:120]
            continue
        x, v = m.vibration, m.speed_profile_hz()
        if n == cal:
            if race in ("OR", "IR"):
                nominal = ORDERS["BPFO" if race == "OR" else "BPFI"]
                out["calib"] = e24.line_calibration(x, v, nominal, cfg)
            continue
        for f in (int(f) for f in cfg["factors"]):
            xd, vd, fs = undersample(x, f), undersample(v, f), FS / f
            base = {
                "bearing": code,
                "race": race,
                "setting": setting,
                "measurement": n,
                "factor": f,
            }
            if not guard_ok(xd, vd, fs, cfg):
                for clean in CLEAN:
                    out["rows"].append({**base, "clean": clean, "skipped": "guarda"})
                continue
            for clean, fn in CLEAN.items():
                xc = fn(xd)
                try:
                    est, band = blind(xc, vd, fs, cfg)
                except ValueError:
                    est, band = math.nan, (math.nan, math.nan)
                sc = guided.guided_scores(
                    xc,
                    vd,
                    fs,
                    ORDERS,
                    float(cfg["kurtogram"]["f_min_hz"]),
                    int(g["max_level"]),
                    float(g["window_frac"]),
                    float(g["integer_exclusion_order"]),
                    tuple(int(h) for h in g["harmonics"]),
                    int(cfg["samples_per_rev"]),
                )
                out["rows"].append(
                    {
                        **base,
                        "clean": clean,
                        "blind_order": round(est, 4) if np.isfinite(est) else "",
                        "blind_band": f"{band[0]:.0f}-{band[1]:.0f}"
                        if np.isfinite(band[0])
                        else "",
                        **{
                            f"score_{c}": round(s, 4) if np.isfinite(s) else ""
                            for c, (s, _b) in sc.items()
                        },
                        **{
                            f"band_{c}": f"{b[0]:.0f}-{b[1]:.0f}" if b else ""
                            for c, (_s, b) in sc.items()
                        },
                        "skipped": "",
                    }
                )
    if out["calib"] is None and race in ("OR", "IR"):
        out["calib"] = {"center": math.nan, "ok": False, "prominence": math.nan}
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bearings", nargs="*", default=None, help="só para fumaça")
    ap.add_argument("--settings", nargs="*", default=None, help="só para fumaça")
    ap.add_argument("--measurements", nargs="*", type=int, default=None, help="só para fumaça")
    ap.add_argument("--factors", nargs="*", type=int, default=None, help="só para fumaça")
    ap.add_argument("--results", type=str, default=None, help="só para fumaça")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.measurements:
        cfg["test_measurements"] = args.measurements
    if args.factors:
        cfg["factors"] = args.factors
    root = ROOT / cfg["data_dir"]
    codes = args.bearings or [*cfg["outer"], *cfg["inner"], *cfg["healthy"]]
    settings = args.settings or cfg["settings"]
    jobs = [(cfg, str(root), c, s) for c in codes for s in settings]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers or int(cfg["compute"]["workers"])) as pool:
        res = list(pool.map(task, jobs))
    print(f"{len(res)} registros em {time.time() - t0:.0f} s", flush=True)
    out = EXP / (args.results or cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for o in res for r in o["rows"]]
    write_csv(out / "metrics.csv", rows)
    calib = [
        {
            "bearing": o["code"],
            "setting": o["setting"],
            "race": o["race"],
            "center": o["calib"]["center"],
            "ok": int(bool(o["calib"]["ok"])),
            "prominence": o["calib"]["prominence"],
        }
        for o in res
        if o["calib"] is not None
    ]
    write_csv(out / "calib.csv", calib)
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = version(p)
        except PackageNotFoundError:
            versions[p] = "nao instalado"
    src = hashlib.sha256()
    for p in sorted((ROOT / "packages").glob("*/src/**/*.py")):
        src.update(p.relative_to(ROOT).as_posix().encode())
        src.update(p.read_bytes())
    manifest = {
        "experiment": "E26-fault-order-guided-selector (Paderborn)",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": args.config.as_posix(),
        "config_sha256": sha256_file(args.config),
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "guided_sha256": sha256_file(EXP / "guided.py"),
        "readme_sha256": sha256_file(EXP / "README.md"),
        "e24_run_sha256": sha256_file(E24_RUN),
        "source_sha256": src.hexdigest(),
        "overrides": {
            "bearings": args.bearings,
            "settings": args.settings,
            "measurements": args.measurements,
            "factors": args.factors,
        },
        "input_sha256": {k: v for o in res for k, v in o["sha256"].items()},
        "unreadable": {
            f"{o['code']}/{o['setting']}": o["unreadable"] for o in res if o["unreadable"]
        },
        "packages": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n_rows": len(rows),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
