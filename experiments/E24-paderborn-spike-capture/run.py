"""E24: captura do seletor por picos sob decimação pura, no Paderborn (pré-registro no README).

Só CPU. Para cada registro (rolamento × ajuste de 1500 rpm), faz três coisas.

1. **Calibração:** a raia em taxa cheia da medição 1, para a régua aninhada.
2. **Régua local:** a raia em taxa cheia de cada segmento (medições 2 a 20).
3. **Estimativa:** para cada braço (seletor × remoção de picos) e fator, decimação pura, banda,
   espectro de envelope em ordens e argmax sem prior.

Depois: régua aninhada deixando o próprio rolamento de fora, acurácia balanceada, bootstrap por
rolamento e Holm.

    OMP_NUM_THREADS=1 python experiments/E24-paderborn-spike-capture/run.py \
        --config experiments/E24-paderborn-spike-capture/config/base.yaml --workers 6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml
from snq_data import paderborn as pb
from snq_data import undersample
from snq_recon import (
    envelope_order_spectrum,
    estimate_without_prior,
    fast_kurtogram,
    refine_order_detail,
)
from snq_recon.selectors import fast_band_selector

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
FS = pb.FS_HZ
ORDERS = pb.computed_orders()
RULERS = ("nested", "local")
PACKAGES = ("snq-core", "snq-data", "snq-recon")


def at_repo(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def despike(x: np.ndarray, k_mad: float = 10.0) -> np.ndarray:
    """Critério do E20: desvio acima de ``k_mad`` MAD em relação à média dos vizinhos, com os
    dois vizinhos desviando em sentido oposto e menos de 0,8 vez; troca pela média dos vizinhos."""
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
    y = x.copy()
    idx = np.flatnonzero(iso) + 1
    y[idx] = 0.5 * (x[idx - 1] + x[idx + 1])
    return y


def select_band(x, fs_hz: float, selector: str, cfg: dict):
    kg = cfg["kurtogram"]
    kw = {
        "max_level": int(kg["max_level"]),
        "f_min_hz": float(kg["f_min_hz"]),
        "f_max_hz": fs_hz / 2.0,
    }
    if selector == "sk":
        return fast_kurtogram(x, fs_hz, **kw)
    return fast_band_selector(x, fs_hz, statistic=selector, **kw)


def line_calibration(x, v, nominal: float, cfg: dict) -> dict:
    """Raia em taxa cheia (sinal bruto, kurtograma): refino em ±span e proeminência do pico."""
    try:
        k = select_band(x, FS, "sk", cfg)
        spec = envelope_order_spectrum(x, FS, v, k.band_hz, samples_per_rev=cfg["samples_per_rev"])
        r = refine_order_detail(
            spec,
            nominal,
            span=float(cfg["prior"]["calibration_span"]),
            n_harmonics=int(cfg["prior"]["score_harmonics"]),
            tol=float(cfg["tolerance_order"]),
        )
    except ValueError:
        return {"center": math.nan, "ok": False, "prominence": math.nan}
    o, a = np.asarray(spec.freqs_hz), np.asarray(spec.magnitude)
    w = float(cfg["prominence_window"])
    win = (o >= nominal * (1 - w)) & (o <= nominal * (1 + w))
    near = np.abs(o - r.center) <= 0.015
    med = float(np.median(a[win])) if win.any() else math.nan
    prom = float(a[near].max() / med) if near.any() and med > 0 else math.nan
    ok = (not r.at_edge) and np.isfinite(prom) and prom >= float(cfg["min_line_prominence"])
    return {"center": float(r.center), "ok": bool(ok), "prominence": round(prom, 3)}


def estimate(x, v, factor: int, selector: str, cfg: dict):
    """Pipeline do E07/E22 num fator, com o seletor do braço. Devolve (ordem, banda) ou None."""
    xd, vd = undersample(x, factor), undersample(v, factor)
    fs_eff = FS / factor
    guard = float(cfg["guard_points_per_rev_factor"]) * int(cfg["samples_per_rev"])
    if (
        xd.size < int(cfg["min_samples"])
        or fs_eff / 2.0 <= float(cfg["kurtogram"]["f_min_hz"])
        or fs_eff / float(np.max(vd)) < guard
    ):
        return None
    try:
        k = select_band(xd, fs_eff, selector, cfg)
        spec = envelope_order_spectrum(
            xd, fs_eff, vd, k.band_hz, samples_per_rev=cfg["samples_per_rev"]
        )
        return float(estimate_without_prior(spec, float(cfg["order_min"]))), k.band_hz
    except ValueError:
        return None


def record_task(task: tuple[dict, str, str]) -> dict:
    cfg, root, rid = task
    code, setting = rid.split("_", 1)
    race = pb.BEARINGS[code][1]
    nominal = ORDERS["BPFO" if race == "OR" else "BPFI"]
    n_seg = int(cfg["segment_samples"])
    cal = int(cfg["calibration_measurement"])
    out = {
        "rid": rid,
        "code": code,
        "race": race,
        "sha256": {},
        "calib": None,
        "truth_local": {},
        "est": {},
        "unreadable": {},
        "despiked_samples": {},
    }
    for n in [cal, *[int(t) for t in cfg["test_measurements"]]]:
        path = pb.measurement_path(root, code, setting, n)
        out["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            m = pb.load_measurement(root, code, setting, n, n_samples=n_seg, min_samples=n_seg)
        except ValueError as exc:
            out["unreadable"][n] = str(exc)[:120]
            continue
        x, v = m.vibration, m.speed_profile_hz()
        lc = line_calibration(x, v, nominal, cfg)
        if n == cal:
            out["calib"] = lc
            continue
        out["truth_local"][n] = lc["center"] if lc["ok"] else math.nan
        xs = {False: x, True: despike(x, float(cfg["despike"]["k_mad"]))}
        out["despiked_samples"][n] = int(np.count_nonzero(xs[True] != x))
        for arm, spec in cfg["arms"].items():
            xa = xs[bool(spec["despike"])]
            for f in (int(f) for f in cfg["factors"]):
                out["est"][(n, arm, f)] = estimate(xa, v, f, spec["selector"], cfg)
    if out["calib"] is None:
        out["calib"] = {"center": math.nan, "ok": False, "prominence": math.nan}
    return out


def build_rows(cfg: dict, cls: list[dict]) -> list[dict]:
    tol = float(cfg["tolerance_order"])
    calib = {c["rid"]: c["calib"] for c in cls}
    rows = []
    for c in cls:
        peers = [
            calib[q["rid"]]["center"]
            for q in cls
            if q["race"] == c["race"] and q["code"] != c["code"] and calib[q["rid"]]["ok"]
        ]
        nested = float(np.median(peers)) if peers else math.nan
        for (n, arm, f), res in c["est"].items():
            est = res[0] if res is not None else None
            tl = c["truth_local"].get(n, math.nan)
            ok = est is not None and np.isfinite(est)
            rows.append(
                {
                    "record": c["rid"],
                    "bearing": c["code"],
                    "race": c["race"],
                    "segment": n,
                    "arm": arm,
                    "factor": f,
                    "estimate": round(est, 4) if ok else "",
                    "band_lo_hz": round(res[1][0], 1) if res is not None else "",
                    "band_hi_hz": round(res[1][1], 1) if res is not None else "",
                    "truth_nested": round(nested, 4) if np.isfinite(nested) else "",
                    "truth_local": round(tl, 4) if np.isfinite(tl) else "",
                    "hit_nested": int(ok and abs(est - nested) < tol)
                    if np.isfinite(nested)
                    else "",
                    "hit_local": int(ok and abs(est - tl) < tol) if np.isfinite(tl) else "",
                    "skipped": "" if res is not None else "guarda ou ValueError",
                }
            )
    return rows


def per_bearing(rows, arm: str, factor: int, ruler: str) -> dict[tuple[str, str], float]:
    acc = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and r["factor"] == factor and r[f"hit_{ruler}"] != "":
            acc[(r["race"], r["bearing"])].append(int(r[f"hit_{ruler}"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def balanced(acc: dict[tuple[str, str], float]) -> float:
    by = defaultdict(list)
    for (race, _b), v in acc.items():
        by[race].append(v)
    return 100.0 * float(np.mean([np.mean(v) for v in by.values()])) if by else math.nan


def cluster_test(a: dict, b: dict, n_boot: int, seed: int) -> dict:
    """a - b em acurácia balanceada; bootstrap dos rolamentos dentro de cada pista."""
    keys = sorted(set(a) & set(b))
    races = sorted({k[0] for k in keys})
    d = {r: np.array([a[k] - b[k] for k in keys if k[0] == r]) for r in races}
    point = 100.0 * float(np.mean([v.mean() for v in d.values()]))
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        boots[i] = np.mean([v[rng.integers(0, v.size, v.size)].mean() for v in d.values()])
    eligible = all(v.size >= 3 for v in d.values())
    p = float((1 + np.count_nonzero(boots <= 0.0)) / (1 + n_boot)) if eligible else 1.0
    return {
        "n_bearings": len(keys),
        "diff_points": round(point, 2),
        "ci_lo": round(100.0 * float(np.percentile(boots, 2.5)), 2),
        "ci_hi": round(100.0 * float(np.percentile(boots, 97.5)), 2),
        "p_one_sided": p,
    }


def holm(p: list[float]) -> list[float]:
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, data: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = EXP / cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    root = at_repo(cfg["data_dir"])
    ids = [f"{c}_{s}" for c in cfg["outer"] + cfg["inner"] for s in cfg["settings"]]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers or int(cfg["compute"]["workers"])) as pool:
        cls = list(pool.map(record_task, [(cfg, str(root), r) for r in ids]))
    print(f"{len(cls)} registros em {time.time() - t0:.0f} s", flush=True)
    rows = build_rows(cfg, cls)
    write_csv(out_dir / "metrics.csv", rows)

    an = cfg["analysis"]
    summary = []
    for arm in cfg["arms"]:
        for f in (int(f) for f in cfg["factors"]):
            row = {"arm": arm, "factor": f}
            for ru in RULERS:
                acc = per_bearing(rows, arm, f, ru)
                row[f"bal_acc_{ru}"] = round(balanced(acc), 2)
                row[f"n_bearings_{ru}"] = len(acc)
            summary.append(row)
    write_csv(out_dir / "summary.csv", summary)

    hyp = []
    for name, t in cfg["tests"].items():
        for ru in RULERS:
            tests = []
            for f in (int(f) for f in an["decisive_factors"]):
                res = cluster_test(
                    per_bearing(rows, t["arm"], f, ru),
                    per_bearing(rows, t["baseline"], f, ru),
                    int(an["bootstrap_resamples"]),
                    int(cfg["seed"]) + 1000 * f + (0 if ru == "nested" else 1),
                )
                tests.append(
                    {
                        "hypothesis": name,
                        "ruler": ru,
                        "arm": t["arm"],
                        "baseline": t["baseline"],
                        "factor": f,
                        **res,
                    }
                )
            adj = holm([x["p_one_sided"] for x in tests])
            for x, pa in zip(tests, adj, strict=True):
                x["p_holm"] = round(pa, 4)
                x["significant"] = int(pa < float(an["alpha"]) and x["diff_points"] > 0)
            hyp += tests
            won = [x["factor"] for x in tests if x["significant"]]
            hyp.append(
                {
                    "hypothesis": name,
                    "ruler": ru,
                    "verdict": int(bool(won)),
                    "factors_won": " ".join(map(str, won)),
                    "decisive": int(ru == an["primary_ruler"]),
                }
            )
    write_csv(out_dir / "hypotheses.csv", hyp)

    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "nao instalado"
    files = sorted((REPO_ROOT / "packages").glob("*/src/**/*.py"))
    src = hashlib.sha256()
    for p in files:
        src.update(p.relative_to(REPO_ROOT).as_posix().encode())
        src.update(p.read_bytes())
    manifest = {
        "experiment": "E24-paderborn-spike-capture",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": int(cfg["seed"]),
        "config_sha256": sha256_file(args.config),
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "readme_sha256": sha256_file(EXP / "README.md"),
        "source_sha256": src.hexdigest(),
        "packages": versions,
        "python": platform.python_version(),
        "dataset": "Paderborn KAt-DataCenter, CC BY-NC 4.0 (datasets/manifests/paderborn.yaml)",
        "input_sha256": {k: v for c in cls for k, v in c["sha256"].items()},
        "unreadable": {c["rid"]: c["unreadable"] for c in cls if c["unreadable"]},
        "despiked_samples": {c["rid"]: c["despiked_samples"] for c in cls},
        "calibration": {c["rid"]: c["calib"] for c in cls},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for x in hyp:
        if "verdict" in x:
            print(
                f"{x['hypothesis']} regua {x['ruler']}: veredito {x['verdict']} "
                f"(fatores {x['factors_won'] or '-'})",
                flush=True,
            )
    print(f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
