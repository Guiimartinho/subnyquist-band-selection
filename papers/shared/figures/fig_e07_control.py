"""Figura E07 — controle negativo: falha contra rolamento saudável.

O experimento principal mede acerto sobre registros que **têm** falha, o que não diz
nada sobre falso positivo. Aqui a pergunta é se a ordem de falha separa rolamento com
falha de rolamento saudável, e se essa separação sobrevive à subamostragem.

A estatística é `rank` — a fração de raias da faixa de ordens que ficam abaixo do pico
na ordem alvo. É livre de escala e de piso, e essa escolha não é cosmética: a primeira
versão normalizava pela mediana do espectro, esse piso desaba nos registros saudáveis
sob subamostragem forte, e a métrica indicava separação **invertida** onde não há.

    uv run python papers/shared/figures/fig_e07_control.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
METRICS = ROOT / "experiments/E07-order-domain-ottawa/results/healthy_control.csv"
OUT = ROOT / "experiments/E07-order-domain-ottawa/results/figures"
FS_FULL = 200_000.0

PAIRS = {
    "BPFI": ("inner_race", "#0F6E78", "^", "pista interna"),
    "BPFO": ("outer_race", "#B4502B", "s", "pista externa"),
}
GREY = "0.45"


def auc(a: list[float], b: list[float]) -> float:
    """Probabilidade de um segmento com falha pontuar acima de um saudável."""
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

    # --- (a) separacao contra fator ------------------------------------
    for target, (cond, color, mk, label) in PAIRS.items():
        y = [auc(pick(f, target, cond), pick(f, target, "healthy")) for f in factors]
        ax1.plot(factors, y, marker=mk, color=color, lw=1.4, ms=4, label=label)
    ax1.axhline(0.5, color="0.8", lw=0.9, ls="--", zorder=0)
    ax1.text(factors[0], 0.515, "sem separação", fontsize=6.6, color="0.5")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(factors)
    ax1.set_xticklabels([str(f) for f in factors], fontsize=7)
    ax1.minorticks_off()
    ax1.set_ylim(0.4, 1.03)
    ax1.set_xlabel("fator de subamostragem")
    ax1.set_ylabel("AUC — falha contra saudável")
    ax1.set_title("(a) o falso positivo não cresce", fontsize=9, loc="left")
    ax1.legend(fontsize=7.2, frameon=False, loc="lower left")
    ax1.grid(alpha=0.25, lw=0.5)

    shown = factors[::2]
    top = ax1.secondary_xaxis("top")
    top.set_xticks(shown)
    top.set_xticklabels([f"{FS_FULL / f / 1000:.1f}" for f in shown], fontsize=7)
    top.set_xlabel("taxa efetiva [kSPS]", fontsize=8)

    # --- (b) as distribuicoes que produzem o AUC no fator extremo -------
    extremo = factors[-1]
    posicoes, rotulos = [], []
    for i, (target, (cond, color, _, label)) in enumerate(PAIRS.items()):
        for j, (grupo, cor) in enumerate(((cond, color), ("healthy", GREY))):
            v = pick(extremo, target, grupo)
            pos = 2 * i + j
            ax2.scatter(
                np.full(len(v), pos) + np.random.default_rng(0).normal(0, 0.06, len(v)),
                v,
                s=5,
                color=cor,
                alpha=0.45,
                lw=0,
            )
            ax2.hlines(np.median(v), pos - 0.28, pos + 0.28, color=cor, lw=2.0)
            posicoes.append(pos)
            rotulos.append(f"{label}\n" if grupo == cond else "saudável\n")
    alvos = ["BPFI", "BPFI", "BPFO", "BPFO"]
    ax2.set_xticks(posicoes)
    ax2.set_xticklabels(
        [f"{r}na ordem {t}" for r, t in zip(rotulos, alvos, strict=True)],
        fontsize=6.4,
    )
    ax2.set_ylabel("posição relativa da raia na faixa")
    ax2.set_title(f"(b) distribuições em {extremo}×", fontsize=9, loc="left")
    ax2.grid(alpha=0.25, lw=0.5, axis="y")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"e07_control.{ext}", dpi=200)
    print(f"escrito: {OUT / 'e07_control.pdf'}")


if __name__ == "__main__":
    main()
