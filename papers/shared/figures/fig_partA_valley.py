"""Main figure of Part (a): order accuracy against the decimation factor on Ottawa (E20).

Project rule: the figure comes from metrics.csv, never from a transcribed number.

- **Sources:** E20 (`results/`) for `none` and `despike`. The arms present in
  `results-selectors/` (`infograma_ses`, `gini`) and in `results-post-despike/`
  (`despike_decimado`) come in on their own.
- **Selection:** 25 kHz ceiling and the local ruler, or the nested one (leave-one-record-out
  median, the one of E07) with ``--ruler nested``.
- **Uncertainty:** 95 % band by bootstrap over the condition × scenario clusters (4 per race),
  the same ones as in the analysis. With 4 clusters the band is wide, and that is part of the
  result.

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
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 3 fonts fail PDF eXpress (T0.8)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 7
# the body font of elsarticle (preprint, 12pt) is Computer Modern
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
# arm -> (label, colour, marker, line style, drawing order)
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
                cluster = r["record"].rsplit("-", 1)[0]  # condition x scenario
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
    # elsarticle text width; legend below, outside the panels
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 3.3), sharey=True)
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
        # panel label above the rate axis, so that it covers neither data nor the legend
        ax.set_title(label, loc="left", fontsize=8, pad=24)
        top = ax.secondary_xaxis(
            "top", functions=(lambda m: FS_FULL_KSPS / m, lambda r: FS_FULL_KSPS / r)
        )
        top.set_xticks([200, 100, 66.7, 50, 40, 33.3, 25])
        top.set_xticklabels(["200", "100", "66.7", "50", "40", "33.3", "25"], fontsize=7)
        top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())  # no 6×10¹ labels
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        top.set_xlabel("effective rate $f_s/M$ [kS/s]", fontsize=7)
    ruler = f"{args.ruler} ruler"
    axes[0].set_ylabel(f"order hits [%] ({ruler})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"fig_valley_{args.ruler}.pdf"
    fig.savefig(out)
    # PNG copy for the Markdown report (docs/resultados-subnyquist.md)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG / f"e20_valley_{args.ruler}.png", dpi=200)
    print(f"arms: {present} -> {out}")


if __name__ == "__main__":
    main()
