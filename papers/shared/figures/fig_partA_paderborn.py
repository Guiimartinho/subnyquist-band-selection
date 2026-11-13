"""Figura da réplica da Parte (a) no Paderborn (E24): acurácia balanceada contra o fator.

Regra do projeto: a figura nasce de metrics.csv, nunca de número transcrito.

- **Fontes:** o E24 (`results/`) e, se existir, a rodada da H3 (`results-post-despike/`), que
  repete `kurtograma` e `kurtograma_sem_picos`. Braço repetido é lido de `results/`.
- **Métrica:** a do E24, ou seja, a média, sobre as pistas, do acerto médio por rolamento.
- **Incerteza:** banda de 95 % por bootstrap dos rolamentos dentro de cada pista, como no
  `cluster_test` do E24.
- **Painéis:** (a) a régua aninhada, que é a decisiva, e (b) a régua local, que só existe nos
  segmentos com raia visível em taxa cheia.

    uv run python papers/shared/figures/fig_partA_paderborn.py [--results DIR] [--out ARQ]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 3 reprova no PDF eXpress (T0.8)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 7
# a fonte do corpo do elsarticle (preprint, 12pt) e a Computer Modern
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["CMU Serif", "Computer Modern Roman", "cmr10"]
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.formatter.use_mathtext"] = True
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
E24 = ROOT / "experiments/E24-paderborn-spike-capture"
OUT = ROOT / "papers/mssp-2026/figures/fig_paderborn.pdf"
PNG = E24 / "results/figures"
FS_FULL_KSPS = 64.0
# braço -> (rótulo, cor, marcador, traço, ordem de desenho)
ARMS = {
    "kurtograma": ("kurtogram, raw (baseline)", "0.55", "o", "--", 1),
    "kurtograma_sem_picos": ("kurtogram, spikes removed before decimation", "black", "s", "-", 4),
    "kurtograma_sem_picos_decimado": (
        "kurtogram, spikes removed after decimation",
        "#B4502B",
        "x",
        "-.",
        5,
    ),
    "infograma_ses": ("SES infogram, raw", "#0F6E78", "^", ":", 2),
    "infograma_ses_sem_picos": ("SES infogram, spikes removed", "#0F6E78", "v", "-", 3),
    "gini": ("spectral Gini, raw", "#7A5C99", "D", ":", 2),
}
RULERS = {"nested": "(a) nested ruler", "local": "(b) local ruler"}


def load(results: Path) -> dict[tuple, dict[tuple, list[int]]]:
    """hits[(braço, régua, fator)] -> {(pista, rolamento): [acertos]}."""
    hits: dict[tuple, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    for path in (results / "metrics.csv", results.parent / "results-post-despike/metrics.csv"):
        if not path.exists():
            continue
        new: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] in seen:  # repetido na rodada da H3
                    continue
                new.add(r["arm"])
                for ru in RULERS:
                    if r[f"hit_{ru}"] != "":
                        key = (r["arm"], ru, int(r["factor"]))
                        hits[key][(r["race"], r["bearing"])].append(int(r[f"hit_{ru}"]))
        seen |= new
    return hits


def balanced_ci(cell: dict[tuple, list[int]], n_boot: int = 4000, seed: int = 0):
    by: dict[str, list[float]] = defaultdict(list)
    for (race, _b), v in sorted(cell.items()):
        by[race].append(float(np.mean(v)))
    arr = {r: np.array(v) for r, v in by.items()}
    point = 100.0 * float(np.mean([v.mean() for v in arr.values()]))
    rng = np.random.default_rng(seed)
    boots = [
        np.mean([v[rng.integers(0, v.size, v.size)].mean() for v in arr.values()])
        for _ in range(n_boot)
    ]
    return point, 100.0 * np.percentile(boots, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=E24 / "results")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    hits = load(args.results)
    present = sorted({k[0] for k in hits}, key=lambda a: ARMS[a][4])
    factors = sorted({k[2] for k in hits})
    fig, axes = plt.subplots(
        1, 2, figsize=(5.4, 3.7), sharey=False
    )  # largura do texto do elsarticle
    for ax, (ru, label) in zip(axes, RULERS.items(), strict=True):
        for arm in present:
            name, color, marker, ls, z = ARMS[arm]
            fs = [f for f in factors if (arm, ru, f) in hits]
            if not fs:
                continue
            pts = [balanced_ci(hits[(arm, ru, f)]) for f in fs]
            y = np.array([p[0] for p in pts])
            lo = np.array([p[1][0] for p in pts])
            hi = np.array([p[1][1] for p in pts])
            ax.fill_between(fs, lo, hi, color=color, alpha=0.10, lw=0, zorder=z)
            ax.plot(
                fs, y, ls=ls, marker=marker, color=color, ms=4, lw=1.1, label=name, zorder=z + 5
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(factors)
        ax.set_xticklabels([str(f) for f in factors])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("decimation factor $M$")
        ax.set_ylabel("balanced order accuracy [%]")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=8, va="top")
        top = ax.secondary_xaxis(
            "top", functions=(lambda m: FS_FULL_KSPS / m, lambda r: FS_FULL_KSPS / r)
        )
        top.set_xticks([FS_FULL_KSPS / f for f in factors])
        top.set_xticklabels([f"{FS_FULL_KSPS / f:g}" for f in factors], fontsize=7)
        top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        top.set_xlabel("effective rate $f_s/M$ [kS/s]", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.22, 1, 1))  # legenda embaixo, fora dos paineis
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    # copia PNG para o relatorio em Markdown (docs/resultados-subnyquist.md)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG / "e24_paderborn.png", dpi=200)
    print(f"braços: {present}; fatores {factors} -> {args.out}")


if __name__ == "__main__":
    main()
