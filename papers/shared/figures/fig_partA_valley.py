"""Figura principal da Parte (a): acerto de ordem contra o fator de decimação no Ottawa (E20).

Regra do projeto: a figura nasce de metrics.csv, nunca de número transcrito.

- **Fontes:** o E20 (`results/`) para `none` e `despike`. Os braços que existirem em
  `results-selectors/` (`infograma_ses`, `gini`) e em `results-post-despike/` (`despike_decimado`)
  entram sozinhos.
- **Recorte:** teto de 25 kHz e régua local, ou a aninhada (mediana deixa-um-registro-de-fora,
  a do E07) com ``--ruler nested``.
- **Incerteza:** banda de 95 % por bootstrap dos aglomerados condição × cenário (4 por pista), os
  mesmos da análise. Com 4 aglomerados, a banda é larga, e isso é parte do resultado.

    uv run python papers/shared/figures/fig_partA_valley.py [--ruler local|nested]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

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
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
E20 = ROOT / "experiments/E20-fold-zone-rejection"
SOURCES = [
    (E20 / "results/metrics.csv", ("none", "despike")),
    (E20 / "results-selectors/metrics.csv", ("infograma_ses", "gini")),
    (E20 / "results-post-despike/metrics.csv", ("despike_decimado",)),
]
OUT = ROOT / "papers/mssp-2026/figures"
PNG = E20 / "results/figures"
FS_FULL_KSPS = 200.0
CEILING_HZ = 25000.0
# braço -> (rótulo, cor, marcador, traço, ordem de desenho)
ARMS = {
    "none": ("raw (baseline)", "0.55", "o", "--", 1),
    "despike": ("spikes removed before decimation", "black", "s", "-", 3),
    "despike_decimado": ("spikes removed after decimation", "#B4502B", "x", "-.", 4),
    "infograma_ses": ("SES infogram, raw", "#0F6E78", "^", ":", 2),
    "gini": ("spectral Gini, raw", "#7A5C99", "D", ":", 2),
}
RACES = {"outer_race": "(a) outer race", "inner_race": "(b) inner race"}


def load(ruler: str):
    """hits[(race, arm, factor)] -> {(cluster, record, segment): hit}."""
    col = "hit_local" if ruler == "local" else "hit"
    hits: dict[tuple, dict[tuple, int]] = defaultdict(dict)
    for path, arms in SOURCES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["intervention"] not in arms or float(r["ceiling_hz"]) != CEILING_HZ:
                    continue
                if r[col] == "":
                    continue
                cluster = r["record"].rsplit("-", 1)[0]  # condição x cenário
                key = (r["condition"], r["intervention"], int(r["factor"]))
                hits[key][(cluster, r["record"], r["segment"])] = int(r[col])
    return hits


def cluster_ci(cell: dict[tuple, int], n_boot: int = 2000, seed: int = 0):
    by = defaultdict(list)
    for (cl, _rec, _seg), h in cell.items():
        by[cl].append(h)
    cls = sorted(by)
    rng = np.random.default_rng(seed)
    tot = np.array([len(by[c]) for c in cls], dtype=float)
    hit = np.array([sum(by[c]) for c in cls], dtype=float)
    idx = rng.integers(0, len(cls), size=(n_boot, len(cls)))
    boots = hit[idx].sum(axis=1) / tot[idx].sum(axis=1)
    return 100.0 * hit.sum() / tot.sum(), 100.0 * np.percentile(boots, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler", choices=("local", "nested"), default="local")
    args = ap.parse_args()
    hits = load(args.ruler)
    fig, axes = plt.subplots(
        1, 2, figsize=(5.4, 2.6), sharey=True
    )  # largura do texto do elsarticle
    present = sorted({k[1] for k in hits}, key=lambda a: ARMS[a][4])
    for ax, (race, label) in zip(axes, RACES.items(), strict=True):
        for arm in present:
            name, color, marker, ls, z = ARMS[arm]
            fs = sorted(f for (rc, a, f) in hits if rc == race and a == arm)
            if not fs:
                continue
            pts = [cluster_ci(hits[(race, arm, f)]) for f in fs]
            y = np.array([p[0] for p in pts])
            lo = np.array([p[1][0] for p in pts])
            hi = np.array([p[1][1] for p in pts])
            ax.fill_between(fs, lo, hi, color=color, alpha=0.12, lw=0, zorder=z)
            ax.plot(
                fs, y, ls=ls, marker=marker, color=color, ms=4, lw=1.1, label=name, zorder=z + 5
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
        ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "8"])
        ax.set_xlabel("decimation factor $M$")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.02, 0.04, label, transform=ax.transAxes, fontsize=8, va="bottom")
        top = ax.secondary_xaxis(
            "top", functions=(lambda m: FS_FULL_KSPS / m, lambda r: FS_FULL_KSPS / r)
        )
        top.set_xticks([200, 100, 66.7, 50, 40, 33.3, 25])
        top.set_xticklabels(["200", "100", "66.7", "50", "40", "33.3", "25"], fontsize=7)
        top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())  # sem rótulos 6×10¹
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        top.set_xlabel("effective rate $f_s/M$ [kS/s]", fontsize=7)
    ruler = f"{args.ruler} ruler"
    axes[0].set_ylabel(f"order hits [%] ({ruler})")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"fig_valley_{args.ruler}.pdf"
    fig.savefig(out)
    # copia PNG para o relatorio em Markdown (docs/resultados-subnyquist.md)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG / f"e20_valley_{args.ruler}.png", dpi=200)
    print(f"braços: {present} -> {out}")


if __name__ == "__main__":
    main()
