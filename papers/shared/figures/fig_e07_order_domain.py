"""Figure E07 — degradation under undersampling in the order domain, Ottawa 2018.

Project rule: the figure comes from metrics.csv, never from a transcribed number.

    uv run python papers/shared/figures/fig_e07_order_domain.py
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
METRICS = ROOT / "experiments/E07-order-domain-ottawa/results/metrics.csv"
OUT = ROOT / "experiments/E07-order-domain-ottawa/results/figures"
FS_FULL = 200_000.0

MODES = {
    "no_prior": ("no prior", "o", "0.45", "--"),
    "nominal_prior": ("nominal prior (catalogue)", "s", "#B4502B", ":"),
    "calibrated_prior": ("calibrated prior", "D", "#0F6E78", "-"),
}
ACCENT = "#0F6E78"
WARM = "#B4502B"
CONDS = {
    "inner_race": ("#0F6E78", "^", "inner race"),
    "outer_race": ("#B4502B", "s", "outer race"),
}


def wilson(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    z = 1.96
    d = 1.0 + z * z / n
    return (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))


def main() -> None:
    rows = list(csv.DictReader(METRICS.open(encoding="utf-8")))
    for r in rows:
        r["factor"] = int(r["factor"])
        r["hit"] = int(r["hit"])
    factors = sorted({r["factor"] for r in rows})

    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.7), constrained_layout=True)

    # --- (a) hit rate x factor, by quality of the prior -----------------
    for key, (label, mk, color, ls) in MODES.items():
        y, e = [], []
        for f in factors:
            s = [r for r in rows if r["factor"] == f and r["mode"] == key]
            p = float(np.mean([r["hit"] for r in s])) if s else np.nan
            y.append(100 * p)
            e.append(100 * wilson(p, len(s)) if s else 0.0)
        ax1.errorbar(
            factors, y, yerr=e, marker=mk, color=color, ls=ls, ms=4, lw=1.4, capsize=2, label=label
        )
    ax1.axhline(50, color="0.85", lw=0.8, zorder=0)
    ax1.set_ylabel("fault-order hit rate [%]")
    ax1.set_title("(a) only no prior tests estimation", fontsize=9, loc="left")
    ax1.legend(fontsize=6.6, frameon=False, loc="lower left")

    # --- (b) by condition, with the calibrated prior --------------------
    for cond, (color, mk, label) in CONDS.items():
        y, e = [], []
        for f in factors:
            s = [
                r
                for r in rows
                if r["factor"] == f and r["mode"] == "no_prior" and r["condition"] == cond
            ]
            p = float(np.mean([r["hit"] for r in s])) if s else np.nan
            y.append(100 * p)
            e.append(100 * wilson(p, len(s)) if s else 0.0)
        ax2.errorbar(
            factors, y, yerr=e, marker=mk, color=color, lw=1.4, ms=4, capsize=2, label=label
        )
    ax2.set_ylabel("hit rate [%]")
    ax2.set_title("(b) by fault type, no prior", fontsize=9, loc="left")
    # Lower right: at lower left the outer-race error bar at 4x ran through the legend.
    ax2.legend(fontsize=6.6, frameon=False, loc="lower right")

    # --- (c) the mechanism of the dip at 4x -----------------------------
    # At M=4 the hit rate drops and the kurtogram chooses the high band (>12 kHz) in 2 of 3
    # segments. At 50 kSPS the 75-87 kHz content folds exactly onto 12.8-25
    # kHz; as measured, the kurtosis of that band rises 2.4x to 4.8x in the records with the
    # most energy above 50 kHz - precisely the ones that lose the most hits. It is not the
    # rate that fails, it is the interaction between the rate and the out-of-band content.
    # ADR 0002 on real data, and the argument that the rate has to be CHOSEN, not just reduced.
    high, acc = [], []
    for f in factors:
        # T0.2: only no_prior measures estimation; the columns with prior contain the answer key
        s_ = [r for r in rows if r["factor"] == f and r["mode"] == "no_prior"]
        high.append(100 * float(np.mean([float(r["band_lo_hz"]) >= 12000 for r in s_])))
        acc.append(100 * float(np.mean([r["hit"] for r in s_])))
    ax3.plot(factors, high, marker="o", color=WARM, lw=1.4, ms=4, label="chosen band > 12 kHz")
    ax3.plot(factors, acc, marker="D", color=ACCENT, lw=1.4, ms=4, label="hit rate (no prior)")
    worst = factors[int(np.argmin(acc))]
    ax3.axvline(worst, color="0.7", lw=0.8, ls=":", zorder=0)
    # In three lines and past the two segments that cross just right of the dip: at the top,
    # in two lines, the note ran over the hit-rate line and out of the axes.
    ax3.annotate(
        "1-sample spikes\nfolded onto\n12.75–25 kHz",
        xy=(worst, 68),
        xytext=(16, 0),
        textcoords="offset points",
        fontsize=6.4,
        color="0.35",
        va="top",
    )
    ax3.set_ylabel("[%]")
    ax3.set_title("(c) the band choice fails", fontsize=9, loc="left")
    # The legend is wider than any gap between the curves, so it gets an edgeless white
    # background and sits above the zero line: the dotted 4x line and the falling segment pass
    # behind it instead of through the text, and no marker is covered.
    ax3.legend(
        fontsize=6.6,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.08),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
    )

    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log", base=2)
        ax.set_xticks(factors)
        ax.set_xticklabels(
            [str(f) if f in (1, 2, 4, 8, 16, 32, 64) else "" for f in factors], fontsize=6.2
        )
        ax.minorticks_off()
        ax.set_xlabel("subsampling factor")
        ax.grid(alpha=0.25, lw=0.5)
    for ax in (ax1, ax2, ax3):
        ax.set_ylim(-5, 105)

    # The effective rate is the number that decides the node's budget; the factor alone is not.
    shown = factors[::3]
    top = ax1.secondary_xaxis("top")
    top.set_xticks(shown)
    top.set_xticklabels([f"{FS_FULL / f / 1000:.1f}" for f in shown], fontsize=7.0)
    top.set_xlabel("effective rate [kS/s]", fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"e07_order_domain.{ext}", dpi=200)
    print(f"written: {OUT / 'e07_order_domain.pdf'}")


if __name__ == "__main__":
    main()
