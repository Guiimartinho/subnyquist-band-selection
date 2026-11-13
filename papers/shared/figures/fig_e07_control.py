"""Figure E07 — negative control: fault against healthy bearing.

The main experiment measures accuracy on records that **have** a fault, which says nothing
about false positives. Here the question is whether the fault order separates a faulty
bearing from a healthy bearing, and whether that separation survives undersampling.

The statistic is `rank` — the fraction of lines in the order range that lie below the peak
at the target order. It is free of scale and of floor, and this choice is not cosmetic: the
first version normalized by the median of the spectrum, that floor collapses on the healthy
records under strong undersampling, and the metric indicated an **inverted** separation
where there is none.

    uv run python papers/shared/figures/fig_e07_control.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Type 3 is matplotlib's default and tends to fail the IEEE PDF eXpress check (T0.8).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
METRICS = ROOT / "experiments/E07-order-domain-ottawa/results/healthy_control.csv"
OUT = ROOT / "experiments/E07-order-domain-ottawa/results/figures"
FS_FULL = 200_000.0

PAIRS = {
    "BPFI": ("inner_race", "#0F6E78", "^", "inner race"),
    "BPFO": ("outer_race", "#B4502B", "s", "outer race"),
}
GREY = "0.45"


def auc(a: list[float], b: list[float]) -> float:
    """Probability that a faulty segment scores above a healthy one."""
    if not a or not b:
        return float("nan")
    return sum((x > y) + 0.5 * (x == y) for x in a for y in b) / (len(a) * len(b))


def main() -> None:
    rows = list(csv.DictReader(METRICS.open(encoding="utf-8")))
    for r in rows:
        r["factor"] = int(r["factor"])
        r["rank"] = float(r["rank"])
    factors = sorted({r["factor"] for r in rows})

    def pick(f: int, target: str, cond: str) -> list[float]:
        return [
            r["rank"]
            for r in rows
            if r["factor"] == f and r["target"] == target and r["condition"] == cond
        ]

    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.9), constrained_layout=True)

    # --- (a) separation against factor ---------------------------------
    for target, (cond, color, mk, label) in PAIRS.items():
        y = [auc(pick(f, target, cond), pick(f, target, "healthy")) for f in factors]
        ax1.plot(factors, y, marker=mk, color=color, lw=1.4, ms=4, label=label)
    ax1.axhline(0.5, color="0.8", lw=0.9, ls="--", zorder=0)
    ax1.text(factors[0], 0.515, "no separation", fontsize=6.6, color="0.5")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(factors)
    ax1.set_xticklabels([str(f) for f in factors], fontsize=7)
    ax1.minorticks_off()
    ax1.set_ylim(0.4, 1.03)
    ax1.set_xlabel("subsampling factor")
    ax1.set_ylabel("AUC — faulty vs. healthy")
    ax1.set_title("(a) false positives do not increase", fontsize=9, loc="left")
    # At lower left the 0.5 line and its label ran through the legend; center right is empty.
    ax1.legend(fontsize=7.2, frameon=False, loc="center right")
    ax1.grid(alpha=0.25, lw=0.5)

    shown = factors[::2]
    top = ax1.secondary_xaxis("top")
    top.set_xticks(shown)
    top.set_xticklabels([f"{FS_FULL / f / 1000:.1f}" for f in shown], fontsize=7)
    top.set_xlabel("effective rate [kS/s]", fontsize=8)

    # --- (b) the distributions that produce the AUC at the extreme factor
    extreme = factors[-1]
    positions, labels = [], []
    for i, (target, (cond, color, _, label)) in enumerate(PAIRS.items()):
        for j, (group, group_color) in enumerate(((cond, color), ("healthy", GREY))):
            v = pick(extreme, target, group)
            pos = 2 * i + j
            ax2.scatter(
                np.full(len(v), pos) + np.random.default_rng(0).normal(0, 0.06, len(v)),
                v,
                s=5,
                color=group_color,
                alpha=0.45,
                lw=0,
            )
            ax2.hlines(np.median(v), pos - 0.28, pos + 0.28, color=group_color, lw=2.0)
            positions.append(pos)
            labels.append(f"{label}\n" if group == cond else "healthy\n")
    targets = ["BPFI", "BPFI", "BPFO", "BPFO"]
    ax2.set_xticks(positions)
    ax2.set_xticklabels(
        [f"{r}at {t}" for r, t in zip(labels, targets, strict=True)],
        fontsize=6.4,
    )
    ax2.set_ylabel("line rank in the order range")
    ax2.set_title(f"(b) distributions at {extreme}×", fontsize=9, loc="left")
    ax2.grid(alpha=0.25, lw=0.5, axis="y")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"e07_control.{ext}", dpi=200)
    print(f"written: {OUT / 'e07_control.pdf'}")


if __name__ == "__main__":
    main()
