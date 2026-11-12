"""E20 - rejeicao por zona antes de decimar (ADR 0006).

Pergunta: qual zona de frequencia original, dobrada sobre a faixa de busca do
kurtograma, causa o vale de 4x na pista externa do Ottawa (E07)? E qual o papel do
teto da busca em relacao ao Nyquist efetivo?

O pipeline e o do E07 (kurtograma restrito, espectro de envelope em ordens,
estimativa sem prior), com duas dimensoes novas: uma intervencao aplicada ao
segmento em taxa cheia, ANTES da decimacao pura, e o teto da busca do kurtograma.

    uv run python experiments/E20-fold-zone-rejection/run.py \
        --config experiments/E20-fold-zone-rejection/config/base.yaml

Os auxiliares de analise sao os do E07, copiados porque nada importa experiments/.
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
# intervencao
# ----------------------------------------------------------------------
def despike(x: np.ndarray, k_mad: float = 10.0) -> np.ndarray:
    """Troca picos isolados de uma amostra pela media dos vizinhos (criterio da revisao)."""
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
    """Aplica a intervencao ao segmento em taxa cheia, como mascara ideal de FFT.

    - ``none`` devolve o sinal intacto;
    - ``stop`` zera as faixas de frequencia ORIGINAL de ``bands_hz``;
    - ``aaf_ideal`` zera tudo acima do Nyquist efetivo ``fs/(2*factor)``, e com isso
      nenhuma dobra sobrevive a decimacao;
    - ``despike`` troca picos isolados de uma amostra pela media dos vizinhos (desvio acima de
      ``k_mad`` MAD, vizinhos em sentido oposto e menores que 0,8 vez), sem zerar faixa nenhuma;
    - ``median3`` aplica mediana movel de 3 amostras.

    ``despike`` e ``median3`` entraram em 2026-11-11, pre-registrados depois da revisao
    adversarial que achou o captor do vale (um trem de picos de uma amostra no registro bruto).

    A mascara e de fase zero e nao e realizavel: o objetivo e atribuicao causal, nao
    realizabilidade.
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
        raise ValueError(f"intervencao desconhecida: {kind}")
    return np.fft.irfft(xf, n=x.size)


# ----------------------------------------------------------------------
# analise (mesma logica do E07)
# ----------------------------------------------------------------------
def guard_verdict(n_samples: int, fs_eff_hz: float, points_per_rev: float, cfg: dict) -> str | None:
    """Os tres guardas do E07, na mesma ordem: min_samples, f_min_hz, samples_per_rev."""
    if n_samples < int(cfg["min_samples"]):
        return "min_samples"
    if fs_eff_hz / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
        return "f_min_hz"
    if points_per_rev < 2.0 * int(cfg["samples_per_rev"]):
        return "samples_per_rev"
    return None


def refine(spec, nominal: float, span: float, n_harmonics: int, tol: float) -> float:
    """Ordem de maior energia harmonica numa vizinhanca relativa de ``nominal`` (E07)."""
    # Centro do platô de escore, não o primeiro ponto: o max() antigo enviesava a verdade
    # em meia largura do platô (0,0175 ordem com spr = 64; revisão de 2026-11-11).
    return refine_order(spec, nominal, span=span, n_harmonics=n_harmonics, tol=tol)


def analyse(x, fs_hz: float, speed, cfg: dict, ceiling_hz: float, selector: str = "sk"):
    """Seletor de banda com teto ``min(ceiling_hz, fs/2)`` e espectro de envelope em ordens.

    ``selector="sk"`` é o kurtograma de sempre. Os outros (``ses_negentropy``, ``gini``, ...) são
    os seletores robustos de `snq_recon.selectors`, na mesma partição, pré-registrados em
    2026-11-12 (README, "Seletores robustos").
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
    """Todas as combinacoes intervencao x fator x teto de um segmento."""
    cfg, meta = task["cfg"], task["meta"]
    vib, spd = task["vib"], task["spd"]
    tol = float(cfg["tolerance_order"])
    # regua local: a raia em taxa cheia deste segmento (so para pontuar)
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
            if ispec.get("post") == "despike":  # no fluxo já decimado: realizável no nó
                x = despike(x, float(ispec.get("k_mad", 10.0)))
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
# estatistica
# ----------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> float:
    """p bicaudal exato de McNemar; b e c sao os pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict]) -> list[dict]:
    """Acertos por celula e McNemar pareado contra ``none`` no mesmo fator e teto."""
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
    """Hash do conteudo de packages/*/src/**/*.py (esquema do template)."""
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
        # Sem repositorio git (ex.: servidor que recebeu um pacote), git_* fica None e
        # sao source_sha256 + entrypoint_sha256 que identificam o codigo que rodou.
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
    ap.add_argument("--workers", type=int, default=None, help="sobrepoe compute.workers")
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

    # passe 1 - calibrar a ordem de cada registro em taxa cheia (identico ao E07)
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
    print(f"{len(records)} registros calibrados")

    def truth_loo(name: str, condition: str) -> float:
        others = [v for k, v in calib.items() if k != name and records[k][0].condition == condition]
        return float(np.median(others))

    # passe 2 - uma tarefa por (registro, segmento)
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
    print(f"{len(tasks)} segmentos, {workers} processos")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, seg_rows in enumerate(pool.map(run_segment, tasks), 1):
            rows.extend(seg_rows)
            if i % 12 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} segmentos", flush=True)
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

    # controle de reproducao do E07: none, teto de 25 kHz, ambas as pistas
    want = {int(k): int(v) for k, v in cfg["analysis"]["e07_control_hits"].items()}
    ok = True
    for factor, expected in want.items():
        got = sum(
            s["hits"]
            for s in summary
            if s["intervention"] == "none" and s["factor"] == factor and s["ceiling_hz"] == 25000.0
        )
        ok &= got == expected
        verdict = "OK" if got == expected else "DIFERENTE"
        print(f"controle E07 {factor}x: {got}/72 (esperado {expected}) {verdict}")
    verdict = "reproduz" if ok else "NAO reproduz"
    print(f"\n{len(rows)} linhas -> {results / 'metrics.csv'}; controle {verdict} o E07")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
