"""E07 — degradação por subamostragem no domínio de ordem, com folga real.

Curva **fator de subamostragem × acerto da ordem de falha** sobre Ottawa 2018, sob
rotação variável, com três priors distintos. É o E02 refeito onde o dataset não é o
gargalo — e onde a qualidade do prior vira variável independente em vez de suposição.

Decisões de método, todas forçadas pelos dados e não escolhidas de antemão:

- **Rotação variável, prior em ordens.** Em hertz a frequência de falha se move com
  o regime. Em ordens é constante da geometria. Só este dataset entre os candidatos
  tem a referência angular que isso exige.
- **Teto de 25 kHz na busca do kurtograma.** Sem ele, 12 de 12 registros de pista
  externa escolhiam 50–100 kHz — banda com 0,1% da energia, onde a curtose mede
  vazamento numérico — e a falha não aparecia. Com o teto, a mesma falha aparece na
  ordem 3,4 em todos eles. O achado inverteu a conclusão: a pista externa parecia
  não-diagnosticável e era escolha de banda.
- **Três priors, não um.** O prior nominal vem do catálogo do ER16K e erra a pista
  externa em ~4%; o calibrado vem de medição em taxa cheia nos *outros* registros.
  A distância entre as duas curvas é o custo do desalinhamento de prior — que é
  exatamente a fronteira medição/prior que o projeto se obriga a reportar.
- **Verdade por deixa-um-de-fora.** A ordem de referência de um registro é a mediana
  medida em taxa cheia nos demais registros da mesma condição. Nunca no próprio, ou
  o fator 1 seria acerto por construção.
- **O encoder não passa pelo ADC.** Contagem de pulsos numa entrada digital
  (TIMER/GPIOTE no nRF54L15) não consome orçamento de amostragem. O que é decimado
  é a vibração; o perfil de rotação acompanha porque varia devagar, e o CSV registra
  quantos pontos por revolução sobram em cada fator.
- **Decimação pura.** `snq_data.undersample` recusa aplicar filtro anti-aliasing.

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
    score_fault,
)

PACKAGES = ("snq-core", "snq-sim", "snq-data", "snq-recon", "snq-eval")
FAULT_OF = {"inner_race": "BPFI", "outer_race": "BPFO"}
REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_SHA256_SCHEME = (
    "sha256 sobre a concatenacao, em ordem de caminho POSIX, de "
    "'<caminho relativo ao repositorio>\\n<sha256 do conteudo do arquivo>\\n' "
    "para cada packages/*/src/**/*.py"
)


def guard_verdict(n_samples: int, fs_eff_hz: float, points_per_rev: float, cfg: dict) -> str | None:
    """Nome do guarda que descarta este segmento, ou ``None`` se ele passa nos três.

    Os três guardas em um só lugar, para que a contagem de medições admitidas por
    fator seja verificável sem rodar a varredura (``tests/test_guards.py``). A ordem
    é a de aplicação e importa: o primeiro guarda a disparar é o reportado.

    - ``min_samples``: comprimento do vetor decimado;
    - ``f_min_hz``: a janela de busca do kurtograma ``[f_min_hz, fs_eff/2]`` precisa
      existir;
    - ``samples_per_rev``: a reamostragem angular exige pelo menos dois pontos por
      incremento angular, na rotação de pior caso do segmento.
    """
    if n_samples < int(cfg["min_samples"]):
        return "min_samples"
    if fs_eff_hz / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
        return "f_min_hz"
    if points_per_rev < 2.0 * int(cfg["samples_per_rev"]):
        return "samples_per_rev"
    return None


def harmonics(fundamentals: dict[str, float], n: int) -> np.ndarray:
    """Conjunto discreto de ordens admitidas: harmônicos de cada família de falha."""
    h = np.arange(1, n + 1, dtype=float)
    return np.unique(np.concatenate([o * h for o in fundamentals.values() if np.isfinite(o)]))


def refine(spec, nominal: float, span: float, n_harmonics: int, tol: float) -> float:
    """Ordem de maior energia harmônica numa vizinhança relativa de ``nominal``.

    Refinar numa faixa em vez de pegar o pico global evita cair no 2º harmônico do
    eixo, que domina o envelope de vários registros de pista externa e não é falha.
    """
    grid = np.arange(nominal * (1 - span), nominal * (1 + span), 0.005)
    scores = [score_fault(spec, float(f), n_harmonics=n_harmonics, tol_hz=tol) for f in grid]
    return float(max(scores, key=lambda s: s.harmonic_energy).expected_hz)


def analyse(x, fs_hz, speed, cfg) -> tuple[object, object]:
    """Kurtograma restrito + espectro de envelope em ordens."""
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
    """Hash do conteúdo dos fontes que este experimento importa.

    O ``git_sha`` sozinho não identifica o código que rodou: a árvore pode estar suja,
    e a versão de pacote (``0.1.0``) não muda quando o fonte muda. Este hash muda.
    O esquema está declarado em ``SOURCE_SHA256_SCHEME`` e gravado no manifest junto
    com o valor, para que o hash seja recomputável sem ler este arquivo.
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
            versions[pkg] = "nao instalado"
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
                "ppr_provenance": "determinado empiricamente; nao documentado pelos autores",
                "packages": versions,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_sha": head.strip() if head else None,
                # git_sha aponta para o commit, nao para a arvore que rodou. Os tres
                # campos abaixo cobrem a diferenca: se git_dirty e verdadeiro, o
                # commit NAO descreve o codigo executado, e source_sha256 e o unico
                # identificador do que de fato rodou.
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
    raw = Path(cfg["data_dir"])
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
    print(
        f"{len(names)} registros; nominal " + ", ".join(f"{k}={v:.3f}" for k, v in nominal.items())
    )

    # ------------------------------------------------------------------
    # passe 1 — calibrar as ordens em taxa cheia, um valor por registro
    # ------------------------------------------------------------------
    records: dict[str, object] = {}
    calib: dict[str, float] = {}
    inputs: list[Path] = []
    for name in names:
        try:
            path = download(name, raw)
            rec = load(name, raw)
        except (OSError, KeyError, ValueError) as exc:
            print(f"  {name}: pulado ({type(exc).__name__}: {str(exc)[:70]})")
            continue
        speed = rec.speed_profile_hz(ppr)
        _, spec = analyse(rec.vibration[:seg_n], FS_HZ, speed[:seg_n], cfg)
        fault = FAULT_OF[rec.condition]
        calib[rec.name] = refine(spec, nominal[fault], span, nh_score, tol)
        records[rec.name] = (rec, speed)
        inputs.append(path)
        print(
            f"  {rec.name}: {fault} calibrado = {calib[rec.name]:.3f}"
            f" (nominal {nominal[fault]:.3f})"
        )

    if not records:
        print("\nnenhum registro carregado")
        return 1

    # Verdade por deixa-um-de-fora: mediana dos demais registros da mesma condicao.
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
            f"  {c}: calibrado {np.median(v):.3f} (min {min(v):.3f}, max {max(v):.3f}), "
            f"nominal {nom:.3f}, desvio {100 * (np.median(v) / nom - 1):+.1f}%"
        )

    # ------------------------------------------------------------------
    # passe 2 — varredura de subamostragem
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
                    print(f"    {name} f={factor}: pulado ({str(exc)[:55]})")
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
                            "band_lo_hz": round(k.f_lo_hz, 1),
                            "band_hi_hz": round(k.f_hi_hz, 1),
                            "kurtosis": round(k.kurtosis, 3),
                            "points_per_rev": round(pts_per_rev, 1),
                            "n_samples": x.size,
                        }
                    )
        print(f"    {name}: acumulado {len(rows)} linhas")

    if not rows:
        print("\nnenhuma linha produzida")
        return 1

    write_manifest(
        args.config,
        results,
        cfg,
        {
            "nominal_orders": {k: round(v, 4) for k, v in nominal.items()},
            "calibrated_per_record": {k: round(v, 4) for k, v in calib.items()},
            "records": sorted(records),
            # A verdade fundamental depende de samples_per_rev: as ordens acima sao
            # medidas na reamostragem angular deste config. Comparar duas rodadas com
            # spr diferente e comparar duas referencias distintas.
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
    print(f"\n{len(rows)} linhas -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
