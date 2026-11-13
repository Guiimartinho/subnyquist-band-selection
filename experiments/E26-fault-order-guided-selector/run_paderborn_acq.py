"""E26c no Paderborn: decimação pura contra anti-alias e passa-faixa, com a regra guiada fixa.

A revisão do E26 mostrou que o ganho do guiado já existe na taxa cheia. A pergunta que sobra para
a tese: na mesma taxa, deixar tudo dobrar (``x[::M]``) perde ou ganha informação frente a filtrar
antes de decimar? Só no fluxo bruto (a mediana não contribuiu no E26).

- **Aquisições,** todas na taxa ``FS/M``:
  - ``pura``: ``x[::M]``;
  - ``antialias``: passa-baixas ideal em ``FS/(2M)``, depois ``[::M]``;
  - ``zona{k}`` (só ``M`` em ``zone_factors``): passa-faixa ideal na zona
    ``[k, k+1]·FS/(2M)``, depois ``[::M]``. É a amostragem passa-faixa clássica; a ``zona0`` é o
    anti-alias.
- **Bandas,** com a regra guiada do E26 (proeminência média nas janelas das harmônicas):
  - ``guiado``: o máximo sobre a partição de 23 bandas, só em ``pura`` e ``antialias``;
  - ``kurtograma_regra``: a banda do kurtograma rápido, só em ``pura`` e ``antialias``;
  - ``banda_larga_regra``: [f_min, fs/2], em todas as aquisições.
- **Colunas extras para o vazamento** (revisão, ataque 6):
  - a ordem do pico na janela da 1ª harmônica, na banda escolhida;
  - o escore com exclusão de ±0,03 em torno das ordens inteiras;
  - os escores de 4 janelas-isca, em 0,9 e 1,1 vez a BPFO e a BPFI.
- **Comissionamento:** a banda do kurtograma em taxa cheia na medição 1 de cada rolamento e ajuste,
  gravada em ``commission.csv``.

    uv run python experiments/E26-fault-order-guided-selector/run_paderborn_acq.py \
        --config experiments/E26-fault-order-guided-selector/config/paderborn_acq.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from snq_data import paderborn as pb
from snq_data import undersample
from snq_recon import fast_kurtogram
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
FS = pb.FS_HZ
ORDERS = base.ORDERS
DECOYS = {
    "D_BPFO_09": 0.9 * ORDERS["BPFO"],
    "D_BPFO_11": 1.1 * ORDERS["BPFO"],
    "D_BPFI_09": 0.9 * ORDERS["BPFI"],
    "D_BPFI_11": 1.1 * ORDERS["BPFI"],
}
TARGETS = {**ORDERS, **DECOYS}


def ideal_band(x: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Filtro ideal por máscara na FFT, em taxa cheia: mantém [lo, hi)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1.0 / FS)
    spec[(f < lo_hz) | (f >= hi_hz)] = 0.0
    return np.fft.irfft(spec, n=x.size)


def score_spectrum(orders: np.ndarray, mag: np.ndarray, cfg: dict) -> dict:
    """Escores de todos os alvos, com as duas exclusões, e a ordem do pico na 1ª harmônica."""
    g = cfg["guided"]
    out = {}
    for c, order in TARGETS.items():
        for tag, exc in (
            ("", float(g["integer_exclusion_order"])),
            ("03", float(cfg["exclusion_wide"])),
        ):
            vals = [
                guided.prominence(
                    orders, mag, guided.windows(order, int(k), float(g["window_frac"]), exc)
                )
                for k in g["harmonics"]
            ]
            out[f"s{tag}_{c}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
        segs = guided.windows(
            order, 1, float(g["window_frac"]), float(g["integer_exclusion_order"])
        )
        m = np.zeros(orders.size, dtype=bool)
        for s, e in segs:
            m |= (orders >= s) & (orders <= e)
        out[f"peak_{c}"] = float(orders[m][np.argmax(mag[m])]) if m.any() else math.nan
    return out


def scores_in(x, v, fs, band, cfg) -> dict | None:
    try:
        spec = envelope_order_spectrum(x, fs, v, band, samples_per_rev=int(cfg["samples_per_rev"]))
    except ValueError:
        return None
    return score_spectrum(np.asarray(spec.freqs_hz), np.asarray(spec.magnitude), cfg)


def guided_best(x, v, fs, cfg) -> dict:
    """Por alvo, o máximo do escore (exclusão de ±0,015) sobre a partição.

    O escore com exclusão de ±0,03 e a ordem do pico vêm da mesma banda.
    """
    f_min = float(cfg["kurtogram"]["f_min_hz"])
    best: dict = {}
    for band in guided.bands(fs, f_min, int(cfg["guided"]["max_level"])):
        sc = scores_in(x, v, fs, band, cfg)
        if sc is None:
            continue
        for c in TARGETS:
            s = sc[f"s_{c}"]
            if np.isfinite(s) and s > best.get(c, (-math.inf,))[0]:
                best[c] = (s, sc[f"s03_{c}"], sc[f"peak_{c}"], band)
    out = {}
    for c in TARGETS:
        s, s03, pk, band = best.get(c, (math.nan, math.nan, math.nan, None))
        out[f"s_{c}"], out[f"s03_{c}"], out[f"peak_{c}"] = s, s03, pk
        out[f"band_{c}"] = f"{band[0]:.0f}-{band[1]:.0f}" if band else ""
    return out


def fmt(d: dict) -> dict:
    return {
        k: (
            round(v, 4)
            if isinstance(v, float) and np.isfinite(v)
            else ("" if isinstance(v, float) else v)
        )
        for k, v in d.items()
    }


def task(job) -> dict:
    cfg, root, code, setting = job
    race = base.race_of(code)
    n_seg = int(cfg["segment_samples"])
    f_min = float(cfg["kurtogram"]["f_min_hz"])
    out: dict = {"rows": [], "commission": None, "sha256": {}, "unreadable": {}}
    cal = int(cfg["calibration_measurement"])
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
        if n == cal:  # comissionamento: a banda do kurtograma em taxa cheia
            kg = cfg["kurtogram"]
            try:
                b = fast_kurtogram(
                    x, FS, max_level=int(kg["max_level"]), f_min_hz=f_min, f_max_hz=FS / 2
                ).band_hz
                out["commission"] = {
                    "bearing": code,
                    "setting": setting,
                    "race": race,
                    "lo_hz": round(float(b[0]), 1),
                    "hi_hz": round(float(b[1]), 1),
                }
            except ValueError:
                pass
            continue
        for dec in (int(f) for f in cfg["factors"]):
            fs = FS / dec
            vd = undersample(v, dec)
            acqs = {
                "pura": undersample(x, dec),
                "antialias": undersample(ideal_band(x, 0.0, FS / (2 * dec)), dec),
            }
            if dec in [int(z) for z in cfg["zone_factors"]]:
                for k in range(1, dec):
                    lo, hi = k * FS / (2 * dec), (k + 1) * FS / (2 * dec)
                    acqs[f"zona{k}"] = undersample(ideal_band(x, lo, hi), dec)
            for acq, xd in acqs.items():
                if not base.guard_ok(xd, vd, fs, cfg):
                    continue
                key = {
                    "bearing": code,
                    "race": race,
                    "setting": setting,
                    "measurement": n,
                    "factor": dec,
                    "acquisition": acq,
                }
                wide = scores_in(xd, vd, fs, (f_min, fs / 2.0), cfg)
                if wide is not None:
                    out["rows"].append(
                        {
                            **key,
                            "strategy": "banda_larga_regra",
                            "band": f"{f_min:.0f}-{fs / 2:.0f}",
                            **fmt(wide),
                        }
                    )
                if acq in ("pura", "antialias"):
                    out["rows"].append(
                        {
                            **key,
                            "strategy": "guiado",
                            "band": "",
                            **fmt(guided_best(xd, vd, fs, cfg)),
                        }
                    )
                    try:
                        kb = base.e24.select_band(xd, fs, "sk", cfg).band_hz
                        kb = (float(kb[0]), float(kb[1]))
                        ks = scores_in(xd, vd, fs, kb, cfg)
                        if ks is not None:
                            out["rows"].append(
                                {
                                    **key,
                                    "strategy": "kurtograma_regra",
                                    "band": f"{kb[0]:.0f}-{kb[1]:.0f}",
                                    **fmt(ks),
                                }
                            )
                    except ValueError:
                        pass
    return out


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
    jobs = [(cfg, str(root), c, s) for c in codes for s in (args.settings or cfg["settings"])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers or int(cfg["compute"]["workers"])) as pool:
        res = list(pool.map(task, jobs))
    print(f"{len(res)} registros em {time.time() - t0:.0f} s", flush=True)
    out = EXP / (args.results or cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for o in res for r in o["rows"]]
    base.write_csv(out / "metrics.csv", rows)
    base.write_csv(out / "commission.csv", [o["commission"] for o in res if o["commission"]])
    src = hashlib.sha256()
    for p in sorted((ROOT / "packages").glob("*/src/**/*.py")):
        src.update(p.relative_to(ROOT).as_posix().encode())
        src.update(p.read_bytes())
    manifest = {
        "experiment": "E26c-fault-order-guided-selector (Paderborn, aquisicao)",
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
