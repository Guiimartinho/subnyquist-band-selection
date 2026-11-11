"""E23 - interferente impulsivo injetado no CWRU 48k (ADR 0006, atualizacao pos-E21).

Para cada registro de falha (48 kHz, DE) a 1a metade e calibracao e a 2a e teste. No
teste injeta-se um interferente numa portadora acima do teto da busca: rajadas impulsivas,
ou o substituto de fase aleatoria da mesma realizacao (mesmo espectro de potencia bin a bin,
curtose ~3), que isola a impulsividade da potencia. O pipeline e o do E02 em hertz, com a
faixa de busca restrita a [f_min, teto]. Bracos: uniforme, aaf_ideal, uniforme_teto (teto
pelo TAT) e taxa_tat (fator pelo TAT); o oraculo uniforme sai na analise.

    uv run python experiments/E23-injected-interferer-cwru/run.py \
        --config experiments/E23-injected-interferer-cwru/config/base.yaml

Auxiliares copiados do E20/E21/E22, porque nada importa experiments/.
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
from snq_core import CWRU_6205_2RS_JEM_SKF as GEOM
from snq_core import fault_frequencies_hz
from snq_core.sampling import alias_frequency_hz
from snq_data import CwruCatalog, load_record, undersample
from snq_recon import envelope_spectrum, estimate_without_prior, fast_kurtogram

PACKAGES = ("snq-core", "snq-sim", "snq-data", "snq-recon", "snq-eval")
REPO_ROOT = Path(__file__).resolve().parents[2]


def at_repo(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


# ----------------------------------------------------------------------
# sinal
# ----------------------------------------------------------------------
def fft_mask(x: np.ndarray, fs_hz: float, keep) -> np.ndarray:
    xf = np.fft.rfft(x)
    xf[~keep(np.fft.rfftfreq(x.size, d=1.0 / fs_hz))] = 0.0
    return np.fft.irfft(xf, n=x.size)


def aaf_ideal(x: np.ndarray, fs_hz: float, factor: int) -> np.ndarray:
    return x if factor == 1 else fft_mask(x, fs_hz, lambda f: f <= fs_hz / (2.0 * factor))


def band_power(x: np.ndarray, fs_hz: float, lo_hz: float, hi_hz: float) -> float:
    xf = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    return float(np.sum(xf[(f >= lo_hz) & (f <= hi_hz)])) / x.size**2


def burst_bandwidth_hz(tau_s: float) -> float:
    """Largura a meia potencia (FWHM) do espectro lorentziano de uma rajada exp(-t/tau)."""
    return 1.0 / (math.pi * tau_s)


def impulsive_bursts(n: int, fs_hz: float, fc_hz: float, cfg: dict, rng) -> np.ndarray:
    """Rajadas exp(-t/tau) numa portadora fc, a taxa fixa, com atraso e fase iniciais aleatorios."""
    inj = cfg["injection"]
    tau = float(inj["burst_tau_s"])
    env = np.zeros(n)
    period = 1.0 / float(inj["burst_rate_hz"])
    span = int(6 * tau * fs_hz)
    decay = np.exp(-np.arange(span) / (tau * fs_hz))
    for t0 in np.arange(rng.uniform(0.0, period), n / fs_hz, period):
        i = int(t0 * fs_hz)
        m = min(span, n - i)
        env[i : i + m] += decay[:m]
    t = np.arange(n) / fs_hz
    return env * np.sin(2 * np.pi * fc_hz * t + rng.uniform(0.0, 2 * np.pi))


def phase_surrogate(s: np.ndarray, rng) -> np.ndarray:
    """Mesmo espectro de amplitude de s, bin a bin, com fases aleatorias.

    A estatistica de segunda ordem fica identica e a curtose vai a ~3: a unica diferenca para s
    e a impulsividade.
    """
    sf = np.fft.rfft(s)
    ph = rng.uniform(0.0, 2 * np.pi, sf.size)
    ph[0] = 0.0
    if s.size % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(sf) * np.exp(1j * ph), n=s.size)


def interferer(n: int, fs_hz: float, kind: str, fc_hz: float, cfg: dict, seed_parts) -> np.ndarray:
    """Interferente de potencia unitaria.

    ``impulsive`` e ``surrogate`` partem da MESMA realizacao de rajadas (mesma semente), e o
    substituto so troca as fases. A semente nao depende da potencia, entao a varredura de
    potencia escala a mesma realizacao.
    """
    s = impulsive_bursts(n, fs_hz, fc_hz, cfg, np.random.default_rng([*seed_parts, 0]))
    if kind == "surrogate":
        s = phase_surrogate(s, np.random.default_rng([*seed_parts, 1]))
    elif kind != "impulsive":
        raise ValueError(f"tipo de interferente desconhecido: {kind}")
    return s / math.sqrt(float(np.mean(s**2)))


def spike_train(n: int, fs_hz: float, rate_hz: float, sign: float, rng) -> np.ndarray:
    """Picos isolados de uma amostra (amplitude 1, sinal ``sign``) num processo de Poisson.

    E o captor que a revisao achou no Ottawa: cerca de 1.300 a 2.500 picos/s, de uma amostra,
    com desvio negativo em 96 % dos casos. Picos colados sao separados para continuarem isolados.
    """
    hit = rng.random(n) < rate_hz / fs_hz
    hit[0] = hit[-1] = False
    hit[1:] &= ~hit[:-1]  # nunca dois picos em amostras vizinhas
    return np.where(hit, float(sign), 0.0)


def residual_mad(x: np.ndarray) -> float:
    """MAD (escala gaussiana) do desvio de cada amostra em relacao a media dos vizinhos."""
    d = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    return float(1.4826 * np.median(np.abs(d)))


def despike(x: np.ndarray, k_mad: float = 10.0) -> np.ndarray:
    """Troca picos isolados de uma amostra pela media dos vizinhos (criterio da revisao do E20).

    Pico: desvio acima de ``k_mad`` MAD, com os dois vizinhos desviando em sentido oposto e
    menos de 0,8 vez. Nao zera faixa nenhuma do espectro.
    Limitacao: dois picos a duas amostras um do outro escapam, porque o vizinho do meio desvia
    tanto quanto eles.
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


def fold_to_baseband_hz(f_hz, fs_eff_hz: float):
    """Imagem de f em [0, fs_eff/2] sob decimacao pura (mesma convencao do snq_core)."""
    return np.abs((np.asarray(f_hz, dtype=float) + fs_eff_hz / 2.0) % fs_eff_hz - fs_eff_hz / 2.0)


def folded_share_in_search(
    s: np.ndarray, fs_hz: float, factor: int, f_min_hz: float, f_max_hz: float
) -> float:
    """Fracao da potencia de s cuja imagem, sob x[::factor], cai na faixa de busca [f_min, f_max].

    E a previsao aritmetica de H1. Conta a cauda lorentziana da rajada, que vaza para a faixa
    de busca ate sem dobra (em 1x).
    """
    p = np.abs(np.fft.rfft(s)) ** 2
    img = fold_to_baseband_hz(np.fft.rfftfreq(s.size, d=1.0 / fs_hz), fs_hz / factor)
    inside = (img >= f_min_hz) & (img <= f_max_hz)
    return float(p[inside].sum() / p.sum())


def on_interferer_line(est_hz: float, cfg: dict, tol_hz: float) -> bool:
    """A estimativa caiu sobre a taxa das rajadas (ou um harmonico ate o 3o)?

    E o sinal direto de captura: o seletor passou a olhar o interferente.
    """
    rate = float(cfg["injection"]["burst_rate_hz"])
    return any(abs(est_hz - k * rate) < tol_hz for k in (1, 2, 3))


def prediction_class(share: float, cfg: dict) -> int:
    """1: captura prevista; 0: sem captura prevista; -1: borda (fora do criterio binario)."""
    pr = cfg["prediction"]
    if share >= float(pr["capture_share"]):
        return 1
    if share < float(pr["clear_share"]):
        return 0
    return -1


# ----------------------------------------------------------------------
# pipeline (E02 em hertz, com faixa de busca restrita)
# ----------------------------------------------------------------------
def kurtogram(x, fs_hz: float, f_max_hz: float, cfg: dict):
    kg = cfg["kurtogram"]
    return fast_kurtogram(
        x, fs_hz, max_level=int(kg["max_level"]), f_min_hz=float(kg["f_min_hz"]), f_max_hz=f_max_hz
    )


def ceiling_at(fs_eff_hz: float, cfg: dict, frac: float = 1.0) -> float:
    return min(float(cfg["ceiling0_hz"]), fs_eff_hz / 2.0) * frac


def classic_hz(x, fs_hz: float, factor: int, f_max_hz: float, truth_hz: float, cfg: dict) -> dict:
    xd = undersample(x, factor)
    fs_eff = fs_hz / factor
    f_min = float(cfg["kurtogram"]["f_min_hz"])
    if xd.size < int(cfg["min_samples"]):
        return {"skipped": "min_samples"}
    if fs_eff / 2.0 <= 2.0 * truth_hz:
        return {"skipped": "envelope_nyquist"}  # o guarda do E02
    if f_max_hz <= f_min:
        return {"skipped": "teto abaixo do piso do kurtograma"}
    try:
        k = kurtogram(xd, fs_eff, f_max_hz, cfg)
        est = estimate_without_prior(
            envelope_spectrum(xd, fs_eff, k.band_hz), float(cfg["estimate_f_min_hz"])
        )
    except ValueError as exc:
        return {"skipped": f"ValueError: {str(exc)[:60]}"}
    return {
        "estimate_hz": float(est),
        "band_lo_hz": round(k.f_lo_hz, 1),
        "band_hi_hz": round(k.f_hi_hz, 1),
    }


def tat_admissible(x_cal, fs_hz: float, factor: int, f_max_hz: float, cfg: dict) -> bool:
    fs_eff = fs_hz / factor
    f_min = float(cfg["kurtogram"]["f_min_hz"])
    if f_max_hz <= f_min or fs_eff / 2.0 <= f_min:
        return False
    xd = undersample(x_cal, factor)
    xa = undersample(aaf_ideal(x_cal, fs_hz, factor), factor)
    try:
        k = kurtogram(xd, fs_eff, f_max_hz, cfg)
    except ValueError:
        return False
    e = band_power(xd, fs_eff, k.f_lo_hz, k.f_hi_hz)
    share = 1.0 - band_power(xa, fs_eff, k.f_lo_hz, k.f_hi_hz) / e if e > 0 else math.nan
    return bool(share <= float(cfg["tat"]["folded_share_threshold"]))


class TatCache:
    """TAT por (fator, teto) sobre uma captura de calibracao, avaliado uma vez."""

    def __init__(self, x_cal: np.ndarray, fs_hz: float, cfg: dict) -> None:
        self.x_cal, self.fs_hz, self.cfg = x_cal, fs_hz, cfg
        self._memo: dict[tuple[int, float], bool] = {}

    def __call__(self, factor: int, f_max_hz: float) -> bool:
        key = (factor, round(f_max_hz, 3))
        if key not in self._memo:
            self._memo[key] = tat_admissible(self.x_cal, self.fs_hz, factor, f_max_hz, self.cfg)
        return self._memo[key]


# ----------------------------------------------------------------------
# um registro
# ----------------------------------------------------------------------
Condition = tuple[str, float | None, float | None, float | None, float | None]


def conditions(cfg: dict) -> list[Condition]:
    """(tipo, portadora, potencia relativa, taxa de picos, amplitude dos picos em MAD).

    Rajadas e substituto variam portadora e potencia; o trem de picos varia taxa e amplitude.
    """
    inj = cfg["injection"]
    out: list[Condition] = [("none", None, None, None, None)]
    for k in inj["kinds"]:
        if k == "spikes":
            sp = cfg["spikes"]
            out += [
                ("spikes", None, None, float(r), float(a))
                for r in sp["rate_hz"]
                for a in sp["amplitude_mad"]
            ]
        else:
            out += [
                (k, float(fc), float(db), None, None)
                for fc in inj["carriers_hz"]
                for db in inj["relative_power_db"]
            ]
    return out


def run_record(task: dict) -> dict:
    cfg, number, point, fault = task["cfg"], task["number"], task["point"], task["fault"]
    catalog = CwruCatalog.load(at_repo(cfg["catalog"]))
    rec = load_record(number, point, at_repo(cfg["data_dir"]), catalog, fs_hz=float(cfg["fs_hz"]))
    if rec.shaft_hz is None:
        return {"number": number, "error": "sem RPM"}
    x, fs = rec.signal.astype(float), float(cfg["fs_hz"])
    h = x.size // 2
    xc, xt = x[:h], x[h:]
    truth = float(fault_frequencies_hz(GEOM, rec.shaft_hz)[fault])
    factors = [int(f) for f in cfg["sweep"]["undersampling_factor"]]
    f_min = float(cfg["kurtogram"]["f_min_hz"])
    tol = float(cfg["tolerance_hz"])
    search_ceilings = [float(c) for c in cfg["search_ceilings_hz"]]

    # potencia de referencia: a do registro na banda que o kurtograma escolhe em taxa cheia
    k_cal = kurtogram(xc, fs, ceiling_at(fs, cfg), cfg)
    p_ref = band_power(xc, fs, k_cal.f_lo_hz, k_cal.f_hi_hz)

    seed_rec = int(hashlib.sha256(f"{number}{point}".encode()).hexdigest()[:8], 16)
    rows = []
    # uma tarefa = um registro x uma condicao (o pool paraleliza sobre as duas)
    mad_ref = residual_mad(xc)
    for kind, fc, rel_db, spike_rate, spike_amp in task.get("conditions") or conditions(cfg):
        if kind == "none":
            xcal, xtest, s_test = xc, xt, None
        elif kind == "spikes":
            sign = float(cfg["spikes"].get("sign", -1.0))
            parts = [int(cfg["seed"]), seed_rec, int(spike_rate), int(round(10 * spike_amp))]
            a = spike_amp * mad_ref
            xcal = xc + a * spike_train(
                xc.size, fs, spike_rate, sign, np.random.default_rng([*parts, 0])
            )
            s_test = spike_train(xt.size, fs, spike_rate, sign, np.random.default_rng([*parts, 1]))
            xtest = xt + a * s_test
        else:
            amp = math.sqrt(p_ref * 10.0 ** (rel_db / 10.0))
            parts = [int(cfg["seed"]), seed_rec, int(round(fc))]
            xcal = xc + amp * interferer(xc.size, fs, kind, fc, cfg, [*parts, 0])
            s_test = interferer(xt.size, fs, kind, fc, cfg, [*parts, 1])
            xtest = xt + amp * s_test
        admissible = TatCache(xcal, fs, cfg)

        base = {
            "record": f"{number}{point}",
            "fault": fault,
            "truth_hz": round(truth, 3),
            "kind": kind,
            "fc_hz": fc if fc is not None else "",
            "rel_db": rel_db if rel_db is not None else "",
            "spike_rate_hz": spike_rate if spike_rate is not None else "",
            "spike_amp_mad": spike_amp if spike_amp is not None else "",
        }
        uni = {m: classic_hz(xtest, fs, m, ceiling_at(fs / m, cfg), truth, cfg) for m in factors}
        for m in factors:
            fs_eff = fs / m
            c0 = ceiling_at(fs_eff, cfg)
            image = "" if fc is None else round(float(alias_frequency_hz(fc, fs_eff)), 1)
            # braco uniforme em cada teto da busca (tetos efetivos distintos, sem duplicar celula)
            arms: list[tuple[str, float, dict]] = []
            for c_eff in sorted(
                {round(min(c, fs_eff / 2.0), 3) for c in search_ceilings}, reverse=True
            ):
                res = (
                    uni[m] if c_eff == round(c0, 3) else classic_hz(xtest, fs, m, c_eff, truth, cfg)
                )
                arms.append(("uniforme", c_eff, res))
            arms.append(
                ("aaf_ideal", c0, classic_hz(aaf_ideal(xtest, fs, m), fs, m, c0, truth, cfg))
            )
            if cfg.get("despike_arm", False):
                arms.append(("despike", c0, classic_hz(despike(xtest), fs, m, c0, truth, cfg)))
            fracs = [float(f) for f in cfg["tat"]["ceiling_fractions"]]
            valid = [f for f in fracs if ceiling_at(fs_eff, cfg, f) > f_min] or [fracs[0]]
            frac = next((f for f in valid if admissible(m, ceiling_at(fs_eff, cfg, f))), valid[-1])
            c_teto = ceiling_at(fs_eff, cfg, frac)
            arms.append(
                (
                    "uniforme_teto",
                    c_teto,
                    {**classic_hz(xtest, fs, m, c_teto, truth, cfg), "ceiling_frac": frac},
                )
            )
            m_tat = next(
                (
                    q
                    for q in factors
                    if q >= m and "skipped" not in uni[q] and admissible(q, ceiling_at(fs / q, cfg))
                ),
                m,
            )
            arms.append(
                ("taxa_tat", ceiling_at(fs / m_tat, cfg), {**uni[m_tat], "chosen_factor": m_tat})
            )
            tat_here = int(admissible(m, c0))
            for arm, c_eff, res in arms:
                if s_test is None or arm != "uniforme":
                    share, pred = "", ""
                else:
                    sh = folded_share_in_search(s_test, fs, m, f_min, c_eff)
                    share, pred = round(sh, 4), prediction_class(sh, cfg)
                est = res.get("estimate_hz")
                ok = est is not None and np.isfinite(est) and "skipped" not in res
                rows.append(
                    {
                        **base,
                        "factor": m,
                        "arm": arm,
                        "search_ceiling_hz": round(c_eff, 1),
                        "folded_share": share,
                        "predicted": pred,
                        "carrier_image_hz": image,
                        "tat_admissible_here": tat_here,
                        **res,
                        "estimate_hz": round(est, 3) if ok else "",
                        "hit": int(abs(est - truth) < tol) if ok else "",
                        "interferer_line": int(on_interferer_line(est, cfg, tol)) if ok else "",
                        "skipped": res.get("skipped", ""),
                    }
                )
    return {
        "number": number,
        "point": point,
        "sha256": hashlib.sha256(rec_path(cfg, number).read_bytes()).hexdigest(),
        "rows": rows,
    }


def rec_path(cfg: dict, number: int) -> Path:
    return at_repo(cfg["data_dir"]) / f"{number}.mat"


def run_record_safe(task: dict) -> dict:
    try:
        return run_record(task)
    except Exception as exc:  # noqa: BLE001
        cond = task.get("conditions")
        return {
            "number": task["number"],
            "conditions": cond,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ----------------------------------------------------------------------
# analise
# ----------------------------------------------------------------------
def summarize(rows: list[dict]) -> list[dict]:
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["hit"] == "":
            continue
        k = (
            r["kind"],
            r["fc_hz"],
            r["rel_db"],
            r["factor"],
            r["arm"],
            r["search_ceiling_hz"],
            r["predicted"],
        )
        cell[k][0] += 1
        cell[k][1] += int(r["hit"])
    return [
        {
            "kind": k[0],
            "fc_hz": k[1],
            "rel_db": k[2],
            "factor": k[3],
            "arm": k[4],
            "search_ceiling_hz": k[5],
            "predicted": k[6],
            "n": n,
            "hits": h,
        }
        for k, (n, h) in sorted(cell.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
    ]


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
        "experiment": "E23-injected-interferer-cwru",
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = args.config.parent.parent / cfg["output"]["results_dir"]
    catalog = CwruCatalog.load(at_repo(cfg["catalog"]))
    sel = catalog.selection("classical", cfg["subset"], cfg["point"])
    sel = sel[: args.limit] if args.limit else sel
    tasks = [
        {"cfg": cfg, "number": n, "point": p, "fault": f, "conditions": [c]}
        for n, p, f in sel
        for c in conditions(cfg)
    ]
    workers = int(args.workers or cfg["compute"]["workers"])
    print(
        f"{len(tasks)} tarefas ({len(sel)} registros x {len(conditions(cfg))} condicoes), "
        f"{workers} processos",
        flush=True,
    )
    out, failed = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(run_record_safe, tasks):
            (failed if "error" in res else out).append(res)
            done = len(out) + len(failed)
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} tarefas ({len(failed)} com erro)", flush=True)
    rows = [r for o in out for r in o["rows"]]
    summary = summarize(rows)
    write_manifest(
        args.config,
        results,
        cfg,
        {
            "n_records": len({(o["number"], o["point"]) for o in out}),
            "n_tasks": len(tasks),
            "failed_tasks": [
                {"record": f["number"], "conditions": f.get("conditions"), "error": f["error"]}
                for f in failed
            ],
            "input_sha256": {f"{o['number']}{o['point']}": o["sha256"] for o in out},
            "limit": args.limit,
        },
    )
    for name, data in (("metrics.csv", rows), ("summary.csv", summary)):
        cols = list(dict.fromkeys(k for r in data for k in r))
        with (results / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(data)
    print(f"\n{len(rows)} linhas -> {results / 'metrics.csv'}; {len(failed)} registros com erro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
