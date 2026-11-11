"""Figura E07 — degradação por subamostragem no domínio de ordem, Ottawa 2018.

Regra do projeto: figura nasce de metrics.csv, nunca de número transcrito.

    uv run python papers/shared/figures/fig_e07_order_domain.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Type 3 e o default do matplotlib e costuma reprovar no PDF eXpress do IEEE (T0.8).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
METRICS = ROOT / "experiments/E07-order-domain-ottawa/results/metrics.csv"
OUT = ROOT / "experiments/E07-order-domain-ottawa/results/figures"
FS_FULL = 200_000.0

MODES = {
    "sem_prior": ("sem prior", "o", "0.45", "--"),
    "prior_nominal": ("prior nominal (catálogo)", "s", "#B4502B", ":"),
    "prior_calibrado": ("prior calibrado", "D", "#0F6E78", "-"),
}
ACCENT = "#0F6E78"
WARM = "#B4502B"
CONDS = {
    "inner_race": ("#0F6E78", "^", "pista interna"),
    "outer_race": ("#B4502B", "s", "pista externa"),
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

    # --- (a) acerto x fator, por qualidade do prior ---------------------
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
    ax1.set_ylabel("acerto da ordem de falha [%]")
    ax1.set_title("(a) só sem prior mede estimação", fontsize=9, loc="left")
    ax1.legend(fontsize=6.6, frameon=False, loc="lower left")

    # --- (b) por condicao, com o prior calibrado ------------------------
    for cond, (color, mk, label) in CONDS.items():
        y, e = [], []
        for f in factors:
            s = [
                r
                for r in rows
                if r["factor"] == f and r["mode"] == "sem_prior" and r["condition"] == cond
            ]
            p = float(np.mean([r["hit"] for r in s])) if s else np.nan
            y.append(100 * p)
            e.append(100 * wilson(p, len(s)) if s else 0.0)
        ax2.errorbar(
            factors, y, yerr=e, marker=mk, color=color, lw=1.4, ms=4, capsize=2, label=label
        )
    ax2.set_ylabel("acerto [%]")
    ax2.set_title("(b) por tipo de falha, sem prior", fontsize=9, loc="left")
    ax2.legend(fontsize=6.6, frameon=False, loc="lower left")

    # --- (c) o mecanismo do mergulho em 4x ------------------------------
    # Em M=4 o acerto cai e o kurtograma escolhe a banda alta (>12 kHz) em 2 de 3
    # segmentos. A 50 kSPS o conteudo de 75-87 kHz dobra exatamente sobre 12,8-25
    # kHz; medido, a curtose dessa banda sobe 2,4x a 4,8x nos registros com mais
    # energia acima de 50 kHz - justo os que mais perdem acerto. Nao e a taxa que
    # falha, e a interacao entre a taxa e o conteudo fora da banda. ADR 0002 em
    # dado real, e o argumento de que a taxa precisa ser ESCOLHIDA, nao so reduzida.
    high, acc = [], []
    for f in factors:
        # T0.2: so sem_prior mede estimacao; as colunas com prior contem o gabarito
        s_ = [r for r in rows if r["factor"] == f and r["mode"] == "sem_prior"]
        high.append(100 * float(np.mean([float(r["band_lo_hz"]) >= 12000 for r in s_])))
        acc.append(100 * float(np.mean([r["hit"] for r in s_])))
    ax3.plot(factors, high, marker="o", color=WARM, lw=1.4, ms=4, label="banda escolhida > 12 kHz")
    ax3.plot(factors, acc, marker="D", color=ACCENT, lw=1.4, ms=4, label="acerto (sem prior)")
    worst = factors[int(np.argmin(acc))]
    ax3.axvline(worst, color="0.7", lw=0.8, ls=":", zorder=0)
    ax3.annotate(
        "picos de 1 amostra\ndobrados sobre 12,75–25 kHz",
        xy=(worst, 95),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=6.4,
        color="0.35",
        va="top",
    )
    ax3.set_ylabel("[%]")
    ax3.set_title("(c) falha a escolha de banda", fontsize=9, loc="left")
    ax3.legend(fontsize=6.6, frameon=False, loc="lower left")

    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log", base=2)
        ax.set_xticks(factors)
        ax.set_xticklabels(
            [str(f) if f in (1, 2, 4, 8, 16, 32, 64) else "" for f in factors], fontsize=6.2
        )
        ax.minorticks_off()
        ax.set_xlabel("fator de subamostragem")
        ax.grid(alpha=0.25, lw=0.5)
    for ax in (ax1, ax2, ax3):
        ax.set_ylim(-5, 105)

    # A taxa efetiva e o numero que decide o orcamento do no; o fator sozinho nao.
    shown = factors[::3]
    top = ax1.secondary_xaxis("top")
    top.set_xticks(shown)
    top.set_xticklabels([f"{FS_FULL / f / 1000:.1f}" for f in shown], fontsize=7.0)
    top.set_xlabel("taxa efetiva [kSPS]", fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"e07_order_domain.{ext}", dpi=200)
    print(f"escrito: {OUT / 'e07_order_domain.pdf'}")


if __name__ == "__main__":
    main()
