"""E15: porta de poder no Paderborn (README, "Porta de poder no Paderborn (pré-registro)").

Roda só em CPU e sem treinar nada. Faz três coisas em sequência.

1. **Estágio clássico do E15 no Paderborn:** réguas em taxa cheia e a grade uniforme de
   decimação pura, com ``classic``, ``calibrate`` e ``choose_factor`` do ``run.py``.
2. **Baseline:** ``uniforme_treino``, com o fator escolhido nos rolamentos fora do teste, em
   K-fold por rolamento.
3. **Porta de poder:** a do Ottawa (``power_gate.py``), com aglomerado = rolamento e só a regra
   otimista, porque não há braço aprendido.

    OMP_NUM_THREADS=1 python experiments/E15-learned-sampling-mask/paderborn_gate.py \
        --config experiments/E15-learned-sampling-mask/config/paderborn_gate.yaml --workers 16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from snq_data import paderborn as pb
from snq_recon.highres import envelope_order_spectrum

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import power_gate as pg  # noqa: E402
import run as e15  # noqa: E402

FS = pb.FS_HZ
ORDERS = pb.computed_orders()
RULERS = ("local", "nested")


def nominal_of(code: str) -> float:
    return ORDERS["BPFO"] if pb.BEARINGS[code][1] == "OR" else ORDERS["BPFI"]


def line_calibration(x: np.ndarray, v: np.ndarray, nominal: float, cfg: dict) -> dict:
    """Raia em taxa cheia: kurtograma, envelope em ordens, refino em ±span e proeminência."""
    try:
        k = e15.kurtogram(x, FS, e15.ceiling_at(cfg["ceiling0_hz"], FS), cfg)
        spec = envelope_order_spectrum(x, FS, v, k.band_hz, samples_per_rev=cfg["samples_per_rev"])
        cal = e15.calibrate(spec, nominal, cfg)
    except ValueError as exc:
        return {
            "center": math.nan,
            "at_edge": True,
            "prominence": math.nan,
            "ok": False,
            "error": str(exc)[:80],
        }
    o, a = np.asarray(spec.freqs_hz), np.asarray(spec.magnitude)
    w = float(cfg["prominence_window"])
    win = (o >= nominal * (1 - w)) & (o <= nominal * (1 + w))
    near = np.abs(o - cal["center"]) <= 0.015
    med = float(np.median(a[win])) if win.any() else math.nan
    prom = float(a[near].max() / med) if near.any() and med > 0 else math.nan
    ok = (not cal["at_edge"]) and np.isfinite(prom) and prom >= float(cfg["min_line_prominence"])
    return {
        "center": float(cal["center"]),
        "at_edge": bool(cal["at_edge"]),
        "prominence": round(prom, 3),
        "ok": bool(ok),
        "band_hz": [float(b) for b in k.band_hz],
    }


def classical_task(task: tuple[dict, str, str]) -> dict:
    """Um registro (rolamento x ajuste): calibração da medição 1, réguas e grade dos testes."""
    cfg, root, rid = task
    code, setting = rid.split("_", 1)
    nominal = nominal_of(code)
    factors = range(min(cfg["budgets"]), int(cfg["factor_grid_max"]) + 1)
    guard = float(cfg["guard_points_per_rev_factor"])
    out = {
        "rid": rid,
        "code": code,
        "race": pb.BEARINGS[code][1],
        "sha256": {},
        "truth": {},
        "grid": {},
        "unreadable": {},
    }
    for n in [int(cfg["calibration_measurement"]), *[int(t) for t in cfg["test_measurements"]]]:
        path = pb.measurement_path(root, code, setting, n)
        out["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            m = pb.load_measurement(root, code, setting, n)
        except ValueError as exc:  # medição ilegível: fica fora (ver snq_data.paderborn)
            out["unreadable"][n] = str(exc)[:120]
            if n == int(cfg["calibration_measurement"]):
                out["calib"] = {
                    "center": math.nan,
                    "at_edge": True,
                    "ok": False,
                    "error": "ilegivel",
                }
            continue
        x, v = m.vibration, m.speed_profile_hz()
        lc = line_calibration(x, v, nominal, cfg)
        if n == int(cfg["calibration_measurement"]):
            out["calib"] = lc
            continue
        out["truth"][n] = lc
        for f in factors:
            res = e15.classic(x, v, FS, f, e15.ceiling_at(cfg["ceiling0_hz"], FS / f), cfg, guard)
            out["grid"][(n, f)] = res.get("estimate")
    return out


def kfold_by_bearing(cfg: dict, ids: list[str], seed: int) -> list[dict]:
    rng = np.random.default_rng(e15.derive_seed(seed, "paderborn", "splits"))
    k = int(cfg["splits"]["n_folds"])
    folds: list[list[str]] = [[] for _ in range(k)]
    for codes in (cfg["outer"], cfg["inner"]):
        for i, c in enumerate(rng.permutation(codes)):
            folds[i % k].append(str(c))
    out = []
    for f in range(k):
        test = [r for r in ids if r.split("_", 1)[0] in folds[f]]
        out.append(
            {
                "index": f,
                "test_bearings": sorted(folds[f]),
                "test": test,
                "train": [r for r in ids if r not in set(test)],
            }
        )
    return out


def build_rows(cfg, ids, cls, splits) -> list[dict]:
    tol = float(cfg["tolerance_order"])
    by_id = {c["rid"]: c for c in cls}
    group_of = {r: by_id[r]["race"] for r in ids}
    calib0 = {r: by_id[r]["calib"]["center"] for r in ids}
    ok0 = {r: by_id[r]["calib"]["ok"] for r in ids}
    seg_truth = {
        (r, int(n)): (t["center"] if t["ok"] else math.nan)
        for r in ids
        for n, t in by_id[r]["truth"].items()
    }
    grid_est = {
        (r, int(n), int(f)): e
        for r in ids
        for (n, f), e in by_id[r]["grid"].items()
        if e is not None and np.isfinite(e)
    }
    segs = {
        r: [int(n) for n in cfg["test_measurements"] if int(n) not in by_id[r]["unreadable"]]
        for r in ids
    }
    factors = list(range(min(cfg["budgets"]), int(cfg["factor_grid_max"]) + 1))
    rows = []
    for sp in splits:
        test, nontest = set(sp["test"]), sp["train"]
        nested = e15.nested_truths(ids, group_of, calib0, ok0, test)
        for b in cfg["budgets"]:
            arms = {
                "uniforme_treino": e15.choose_factor(
                    b, factors, nontest, seg_truth, grid_est, segs, tol
                ),
                "uniforme_orcamento": b,
                "oraculo_uniforme": e15.choose_factor(
                    b, factors, sorted(test), seg_truth, grid_est, segs, tol
                ),
            }
            for arm, f in arms.items():
                for r in sorted(test):
                    tn = nested[r]["truth"]
                    for n in segs[r]:
                        est = grid_est.get((r, n, f)) if f is not None else None
                        tl = seg_truth.get((r, n), math.nan)
                        ok = est is not None
                        rows.append(
                            {
                                "dataset": "paderborn",
                                "variant": "bruto",
                                "budget": b,
                                "arm": arm,
                                "split": sp["index"],
                                "seed": 0,
                                "record": r,
                                "group": group_of[r],
                                "segment": n,
                                "chosen_factor": f if f is not None else "",
                                "estimate": round(est, 4) if ok else "",
                                "truth_local": round(tl, 4) if np.isfinite(tl) else "",
                                "truth_nested": round(tn, 4) if np.isfinite(tn) else "",
                                "hit_local": int(ok and abs(est - tl) < tol)
                                if np.isfinite(tl)
                                else "",
                                "hit_nested": int(ok and abs(est - tn) < tol)
                                if np.isfinite(tn)
                                else "",
                                "family_true": group_of[r],
                            }
                        )
    return rows


def gate(cfg, rows, seed) -> list[dict]:
    """Porta de poder com a regra otimista: só o teste contra ``uniforme_treino``."""
    pgc = cfg["power_gate"]
    group_of = {r["record"]: r["group"] for r in rows}
    cluster_of = {r["record"]: r["record"].split("_", 1)[0] for r in rows}
    cells, n_rows = {}, {}
    for b in cfg["budgets"]:
        base = [r for r in rows if r["arm"] == "uniforme_treino" and r["budget"] == b]
        n_rows[b] = len(base)
        for ru in RULERS:
            per_b = e15.per_record(base, ru)
            keys = sorted(per_b, key=lambda k: (str(k[0]), k[1]))
            kidx = {k: i for i, k in enumerate(keys)}
            sk, sh, sr = [], [], []
            for i, r in enumerate(base):
                if r[f"hit_{ru}"] in ("", None):
                    continue
                sk.append(kidx[(r["split"], r["record"])])
                sh.append(int(r[f"hit_{ru}"]))
                sr.append(i)
            gacc = defaultdict(list)
            for (_, rid), v in per_b.items():
                gacc[group_of[rid]].append(v)
            bal = 100.0 * float(np.mean([np.mean(v) for v in gacc.values()]))
            pr = pg.make_pairs(keys, group_of, cluster_of)
            tests = {
                "uniforme_treino": (
                    pr,
                    np.array([kidx[k] for k in pr.keys]),
                    np.array([per_b[k] for k in pr.keys]),
                )
            }
            cells[(b, ru)] = pg.Cell(np.array(sk), np.array(sh), np.array(sr), keys, bal, tests)
    out = []
    for effect in pgc["effects"]:
        for name, n_seeds in pgc["designs"].items():
            n_sim = int(pgc["n_sim_perfect"] if effect == "perfeito" else pgc["n_sim"])
            rng = np.random.default_rng(e15.derive_seed(seed, "paderborn_gate", name, effect))
            qs = {
                k: (1.0 if effect == "perfeito" else min(1.0, float(effect) / (100.0 - c.bal)))
                if c.bal < 100.0
                else 0.0
                for k, c in cells.items()
            }
            h1, ruler_ok, wins = 0, defaultdict(int), defaultdict(int)
            for _ in range(n_sim):
                u = {b: rng.random((int(n_seeds), n_rows[b])) for b in cfg["budgets"]}
                passed = {}
                for ru in RULERS:
                    tests = []
                    for b in cfg["budgets"]:
                        c = cells[(b, ru)]
                        a = pg.planted_means(c, u[b], qs[(b, ru)])
                        pr, ai, bv = c.tests["uniforme_treino"]
                        diff, p = pg.boot_p(a[ai] - bv, pr, int(pgc["n_boot"]), rng)
                        tests += [(b, diff, p), (b, 1.0, 0.0), (b, 1.0, 0.0)]
                    ok, w = pg.decide(tests)
                    passed[ru] = ok
                    ruler_ok[ru] += int(ok)
                    for b in w:
                        wins[(ru, b)] += 1
                h1 += int(all(passed.values()))
            row = {
                "design": name,
                "effect_points": effect,
                "rule": "otimista",
                "n_sim": n_sim,
                "power_h1": round(h1 / n_sim, 3),
                "power_local": round(ruler_ok["local"] / n_sim, 3),
                "power_nested": round(ruler_ok["nested"] / n_sim, 3),
            }
            for ru in RULERS:
                for b in cfg["budgets"]:
                    row[f"win_{ru}_b{b}"] = round(wins[(ru, b)] / n_sim, 3)
                    row[f"bal_baseline_{ru}_b{b}"] = round(cells[(b, ru)].bal, 2)
                    row[f"q_{ru}_b{b}"] = round(qs[(b, ru)], 3)
            out.append(row)
            print(
                f"{name} {effect!s:>9}: H1 {row['power_h1']:.2f} (local {row['power_local']:.2f},"
                f" aninhada {row['power_nested']:.2f})",
                flush=True,
            )
    return out


def write_csv(path: Path, data: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=EXP / "results-paderborn-gate")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    root = e15.at_repo(cfg["data_dir"])
    ids = [f"{c}_{s}" for c in cfg["outer"] + cfg["inner"] for s in cfg["settings"]]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cls = list(pool.map(classical_task, [(cfg, str(root), r) for r in ids]))
    print(f"estagio classico: {len(cls)} registros em {time.time() - t0:.0f} s", flush=True)
    splits = kfold_by_bearing(cfg, ids, seed)
    rows = build_rows(cfg, ids, cls, splits)
    write_csv(args.out / "metrics.csv", rows)
    truths = {
        c["rid"]: {"calib": c["calib"], "truth": {str(k): v for k, v in c["truth"].items()}}
        for c in cls
    }
    (args.out / "truths.json").write_text(json.dumps(truths, indent=1), encoding="utf-8")
    power = gate(cfg, rows, seed)
    write_csv(args.out / "power.csv", power)
    crit = cfg["power_gate"]["criterion"]
    best = max(r["power_h1"] for r in power if r["effect_points"] == crit["effect"])
    manifest = {
        "experiment": "E15 paderborn_gate",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "config_sha256": e15.sha256_file(args.config),
        "entrypoint_sha256": {
            p: e15.sha256_file(EXP / p) for p in ("paderborn_gate.py", "run.py", "power_gate.py")
        },
        "source_sha256": e15.source_sha256()[0],
        "splits": splits,
        "input_sha256": {k: v for c in cls for k, v in c["sha256"].items()},
        "unreadable": {c["rid"]: c["unreadable"] for c in cls if c["unreadable"]},
        "criterion": crit,
        "best_power_at_criterion": best,
        "verdict": "testavel" if best >= float(crit["power"]) else "nao_testavel",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(
        f"VEREDITO {manifest['verdict']} (melhor poder em +{crit['effect']}: {best:.2f}) -> "
        f"{args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
