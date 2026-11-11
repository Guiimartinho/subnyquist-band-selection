"""E07, controle negativo — o estimador com prior inventa falha onde não há?

O E07 mede acerto sobre registros que **têm** falha. Isso não diz nada sobre falso
positivo, e um estimador restrito a um conjunto de candidatas sempre devolve uma
candidata: ele não tem como responder "nenhuma". A pergunta que este controle
responde é se a candidata escolhida separa rolamento com falha de rolamento
saudável, e se essa separação sobrevive à subamostragem.

Sem isto, os 88% em 32× do experimento principal são compatíveis com um estimador
que aponta a mesma ordem sempre.

**Duas estatísticas, porque a primeira se mostrou enganosa.** `snr` normaliza pela
mediana do espectro, e sob subamostragem forte esse piso desaba nos registros
saudáveis — o que infla a razão sem que nada físico tenha acontecido. `rank` é a
fração de raias da faixa de ordens que ficam **abaixo** do pico na ordem alvo: é
livre de escala e de piso, e é a que deve ser lida.

    uv run python experiments/E07-order-domain-ottawa/control_healthy.py \
        --config experiments/E07-order-domain-ottawa/config/base.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from run import FAULT_OF, analyse  # noqa: F401  (mesmo diretório)
from snq_data import effective_rate_hz, undersample
from snq_data.ottawa import FS_HZ, download, load, record_names
from snq_recon import score_fault


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--segments", type=int, default=None, help="sobrescreve segments_per_record")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    root = args.config.parent.parent
    results = root / cfg["output"]["results_dir"]
    # T0.1: data_dir relativo vale a partir da raiz do repositorio, nao do cwd.
    raw = Path(cfg["data_dir"])
    raw = raw if raw.is_absolute() else Path(__file__).resolve().parents[2] / raw
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    calib: dict[str, float] = manifest["calibrated_per_record"]

    # As ordens de falha calibradas no experimento principal. Sao elas que o
    # controle procura nos registros saudaveis - onde nao deveria haver energia.
    targets = {
        "BPFI": float(np.median([v for k, v in calib.items() if k.startswith("I-")])),
        "BPFO": float(np.median([v for k, v in calib.items() if k.startswith("O-")])),
    }
    print("ordens alvo:", {k: round(v, 3) for k, v in targets.items()})

    seg_n = int(cfg["segment_samples"])
    n_seg = int(args.segments or cfg["segments_per_record"])
    ppr = int(cfg["pulses_per_rev"])
    nh = int(cfg["prior"]["score_harmonics"])
    tol = float(cfg["tolerance_order"])

    names = record_names(("H", "I", "O"), tuple(cfg["scenarios"]), tuple(cfg["trials"]))
    rows: list[dict] = []
    for name in names:
        try:
            download(name, raw)
            rec = load(name, raw)
        except (OSError, KeyError, ValueError) as exc:
            print(f"  {name}: pulado ({type(exc).__name__}: {str(exc)[:60]})")
            continue
        speed = rec.speed_profile_hz(ppr)

        for seg in range(n_seg):
            a, b = seg * seg_n, (seg + 1) * seg_n
            if b > rec.vibration.size:
                break
            for factor in cfg["sweep"]["undersampling_factor"]:
                factor = int(factor)
                x = undersample(rec.vibration[a:b], factor)
                v = undersample(speed[a:b], factor)
                fs_eff = effective_rate_hz(FS_HZ, factor)
                if fs_eff / 2.0 <= float(cfg["kurtogram"]["f_min_hz"]):
                    continue
                if fs_eff / float(np.max(v)) < 2.0 * int(cfg["samples_per_rev"]):
                    continue
                try:
                    _, spec = analyse(x, fs_eff, v, cfg)
                except ValueError:
                    continue

                floor = float(np.median(spec.magnitude))
                banda = (spec.freqs_hz >= 2.0) & (spec.freqs_hz <= 14.0)
                for fault, order in targets.items():
                    s = score_fault(spec, order, n_harmonics=nh, tol_hz=tol)
                    perto = np.abs(spec.freqs_hz - order) <= tol
                    pico = float(spec.magnitude[perto].max()) if perto.any() else 0.0
                    rank = float(np.mean(spec.magnitude[banda] < pico))
                    rows.append(
                        {
                            "record": rec.name,
                            "condition": rec.condition,
                            "factor": factor,
                            "fs_eff_hz": round(fs_eff, 2),
                            "target": fault,
                            "target_order": round(order, 4),
                            "harmonic_energy": float(s.harmonic_energy),
                            "snr": round(float(s.harmonic_energy) / (nh * floor), 4),
                            "rank": round(rank, 5),
                        }
                    )
        print(f"  {rec.name} ({rec.condition}): {len(rows)} linhas")

    out = results / "healthy_control.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} linhas -> {out}")

    # Resumo em AUC: probabilidade de um segmento com falha pontuar acima de um
    # segmento saudavel na ordem alvo. 0,5 e ausencia total de separacao; abaixo de
    # 0,5 a separacao esta invertida. Nao depende de escala nem de piso.
    def auc(f: int, fault: str, cond: str, key: str) -> float:
        a = [
            r[key]
            for r in rows
            if r["factor"] == f and r["target"] == fault and r["condition"] == cond
        ]
        b = [
            r[key]
            for r in rows
            if r["factor"] == f and r["target"] == fault and r["condition"] == "healthy"
        ]
        if not a or not b:
            return float("nan")
        return sum((x > y) + 0.5 * (x == y) for x in a for y in b) / (len(a) * len(b))

    print(f"\n{'fator':>6} {'AUC BPFI':>10} {'AUC BPFO':>10}   (estatística rank)")
    for f in sorted({r["factor"] for r in rows}):
        print(
            f"{f:6d} {auc(f, 'BPFI', 'inner_race', 'rank'):10.3f}"
            f" {auc(f, 'BPFO', 'outer_race', 'rank'):10.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
