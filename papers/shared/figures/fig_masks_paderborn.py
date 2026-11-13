"""Figure of the learned masks on Paderborn (E15): balanced accuracy against the budget.

Project rule: the figure comes from metrics.csv, never from a transcribed number.

- **Source:** `experiments/E15-learned-sampling-mask/results-paderborn/metrics.csv.gz`, raw
  stream, with all folds and seeds.
- **Arms:** the best uniform rate of the classical pipeline chosen in training
  (`uniform_trained`), and the same decoder with the trained uniform decimation
  (`dec_uniform_trained`), with the random mask (`dec_random`) and with the learned mask
  (`dec_learned_sel`).
- **Metric:** that of the pre-registered tests (`per_record` and `cluster_bootstrap` of `run.py`):
  mean hit per (split, record), over segments and seeds; mean per race; mean over the races.
  Under the local ruler, the records have very different numbers of segments with a line, and the
  per-segment pooled hit rate of `summary.csv` would weight them differently.
- **Uncertainty:** 95 % band by bootstrap over the bearings within each race (4000).
- **Panels:** (a) nested ruler, 23 bearings; (b) local ruler, the 7 bearings with a line visible
  at full rate.
- **Check:** the difference between the learned mask and each reference has to match the
  `diff_points` of `hypotheses.csv` (±0.06 point); otherwise the script stops.

    uv run python papers/shared/figures/fig_masks_paderborn.py
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 3 fonts fail PDF eXpress (T0.8)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 7
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["CMU Serif", "Computer Modern Roman", "cmr10"]
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.formatter.use_mathtext"] = True
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
E15 = ROOT / "experiments/E15-learned-sampling-mask/results-paderborn"
OUT = ROOT / "papers/mssp-2026/figures/fig_masks_paderborn.pdf"
PNG = E15 / "figures/e15_masks_paderborn.png"
FS_FULL_KSPS = 64.0
BUDGETS = (4, 8, 16, 32)
LEARNED = "dec_learned_sel"
# palette checked with the validator of the dataviz skill (chroma, colour blindness, contrast)
GRAY, TEAL, PURPLE = "0.55", "#0089A3", "#7B4CC0"
ARMS = {
    "uniform_trained": ("classical, best uniform rate", GRAY, "o", "--", 1),
    "dec_uniform_trained": ("decoder, trained uniform decimation", TEAL, "s", "-.", 2),
    "dec_random": ("decoder, random mask", PURPLE, "^", ":", 3),
    LEARNED: ("decoder, learned mask", "black", "D", "-", 4),
}
RULERS = {"nested": "(a) nested ruler, 23 bearings", "local": "(b) local ruler, 7 bearings"}

Cell = dict[tuple[str, str], list[int]]


def bearing(record: str) -> str:
    return record.split("_")[0]


def load() -> tuple[dict[tuple[str, str, int], Cell], dict[str, str]]:
    """cells[(arm, ruler, budget)] -> {(split, record): [hits]}; race per record."""
    cells: dict[tuple[str, str, int], Cell] = defaultdict(lambda: defaultdict(list))
    race: dict[str, str] = {}
    with gzip.open(E15 / "metrics.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "paderborn" or r["variant"] != "raw" or r["arm"] not in ARMS:
                continue
            race[r["record"]] = r["family_true"]
            for ru in RULERS:
                h = r[f"hit_{ru}"]
                if h in ("", "nan"):
                    continue
                cells[(r["arm"], ru, int(r["budget"]))][(r["split"], r["record"])].append(int(h))
    return cells, race


def per_record(cell: Cell) -> dict[tuple[str, str], float]:
    return {k: float(np.mean(v)) for k, v in cell.items()}


def balanced(pr: dict[tuple[str, str], float], race: dict[str, str]) -> float:
    by: dict[str, list[float]] = defaultdict(list)
    for (_s, rec), v in pr.items():
        by[race[rec]].append(v)
    return 100.0 * float(np.mean([np.mean(v) for v in by.values()]))


def balanced_ci(pr: dict[tuple[str, str], float], race: dict[str, str], n_boot: int = 4000):
    """95 % CI by bootstrap over the bearings within each race."""
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_s, rec), v in pr.items():
        groups[race[rec]][bearing(rec)].append(v)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        means = []
        for by_b in groups.values():
            names = list(by_b)
            pick = rng.integers(0, len(names), len(names))
            vals = [v for i in pick for v in by_b[names[i]]]
            means.append(np.mean(vals))
        boots.append(np.mean(means))
    lo, hi = 100.0 * np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def reference_diffs() -> dict[tuple[str, str, int], float]:
    out = {}
    with (E15 / "hypotheses.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["hypothesis"] == "H1" and r["arm"] == LEARNED and r["budget"]:
                out[(r["baseline"], r["ruler"], int(r["budget"]))] = float(r["diff_points"])
    return out


def main() -> None:
    cells, race = load()
    ref = reference_diffs()
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 3.4))  # elsarticle text width
    for ax, (ru, title) in zip(axes, RULERS.items(), strict=True):
        for b in BUDGETS:
            learned = per_record(cells[(LEARNED, ru, b)])
            for base in ("uniform_trained", "dec_uniform_trained", "dec_random"):
                other = per_record(cells[(base, ru, b)])
                keys = set(learned) & set(other)
                d = balanced({k: learned[k] for k in keys}, race) - balanced(
                    {k: other[k] for k in keys}, race
                )
                if abs(d - ref[(base, ru, b)]) > 0.06:
                    raise SystemExit(
                        f"{ru} 1/{b} {LEARNED}-{base}: {d:.2f} against {ref[(base, ru, b)]}"
                    )
        for arm, (label, color, marker, ls, z) in ARMS.items():
            pts = []
            for b in BUDGETS:
                pr = per_record(cells[(arm, ru, b)])
                pts.append((balanced(pr, race), *balanced_ci(pr, race)))
            a = np.array(pts)
            ax.fill_between(BUDGETS, a[:, 1], a[:, 2], color=color, alpha=0.10, lw=0, zorder=z)
            ax.plot(BUDGETS, a[:, 0], ls=ls, marker=marker, color=color, ms=4, lw=1.1, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([f"1/{b}" for b in BUDGETS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("sample budget (average fraction of the full rate)")
        ax.set_ylabel("balanced order accuracy [%]")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top")
        top = ax.secondary_xaxis(
            "top", functions=(lambda m: FS_FULL_KSPS / m, lambda r: FS_FULL_KSPS / r)
        )
        top.set_xticks([FS_FULL_KSPS / b for b in BUDGETS])
        top.set_xticklabels([f"{FS_FULL_KSPS / b:g}" for b in BUDGETS], fontsize=7)
        top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        top.set_xlabel("average rate [kS/s]", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=200)
    print(f"differences checked against hypotheses.csv; written: {OUT} and {PNG}")


if __name__ == "__main__":
    main()
