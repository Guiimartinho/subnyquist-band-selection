"""Figura das máscaras aprendidas no Paderborn (E15): acurácia balanceada contra o orçamento.

Regra do projeto: a figura nasce de metrics.csv, nunca de número transcrito.

- **Fonte:** `experiments/E15-learned-sampling-mask/results-paderborn/metrics.csv.gz`, fluxo
  bruto, com todas as dobras e sementes.
- **Braços:** a melhor taxa uniforme do pipeline clássico escolhida no treino (`uniforme_treino`),
  e o mesmo decodificador com a decimação uniforme treinada (`dec_uniforme_treino`), com a máscara
  aleatória (`dec_aleatoria`) e com a máscara aprendida (`dec_aprendida_sel`).
- **Métrica:** a dos testes pré-registrados (`per_record` e `cluster_bootstrap` do `run.py`):
  acerto médio por (divisão, registro), sobre segmentos e sementes; média por pista; média das
  pistas. Na régua local, os registros têm números muito diferentes de segmentos com raia, e o
  acerto agrupado por segmento do `summary.csv` daria outro peso a eles.
- **Incerteza:** banda de 95 % por bootstrap dos rolamentos dentro de cada pista (4000).
- **Painéis:** (a) régua aninhada, 23 rolamentos; (b) régua local, os 7 rolamentos com raia visível
  em taxa cheia.
- **Conferência:** a diferença entre a máscara aprendida e cada referência tem de bater com o
  `diff_points` do `hypotheses.csv` (±0,06 ponto); senão o script para.

    uv run python papers/shared/figures/fig_masks_paderborn.py
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 3 reprova no PDF eXpress (T0.8)
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
LEARNED = "dec_aprendida_sel"
# paleta conferida no validador da skill dataviz (croma, daltonismo, contraste)
GRAY, TEAL, PURPLE = "0.55", "#0089A3", "#7B4CC0"
ARMS = {
    "uniforme_treino": ("classical, best uniform rate", GRAY, "o", "--", 1),
    "dec_uniforme_treino": ("decoder, trained uniform decimation", TEAL, "s", "-.", 2),
    "dec_aleatoria": ("decoder, random mask", PURPLE, "^", ":", 3),
    LEARNED: ("decoder, learned mask", "black", "D", "-", 4),
}
RULERS = {"nested": "(a) nested ruler, 23 bearings", "local": "(b) local ruler, 7 bearings"}

Cell = dict[tuple[str, str], list[int]]


def bearing(record: str) -> str:
    return record.split("_")[0]


def load() -> tuple[dict[tuple[str, str, int], Cell], dict[str, str]]:
    """cells[(braço, régua, orçamento)] -> {(divisão, registro): [acertos]}; pista por registro."""
    cells: dict[tuple[str, str, int], Cell] = defaultdict(lambda: defaultdict(list))
    race: dict[str, str] = {}
    with gzip.open(E15 / "metrics.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "paderborn" or r["variant"] != "bruto" or r["arm"] not in ARMS:
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
    """IC 95 % por bootstrap dos rolamentos dentro de cada pista."""
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
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1))
    for ax, (ru, title) in zip(axes, RULERS.items(), strict=True):
        for b in BUDGETS:
            learned = per_record(cells[(LEARNED, ru, b)])
            for base in ("uniforme_treino", "dec_uniforme_treino", "dec_aleatoria"):
                other = per_record(cells[(base, ru, b)])
                keys = set(learned) & set(other)
                d = balanced({k: learned[k] for k in keys}, race) - balanced(
                    {k: other[k] for k in keys}, race
                )
                if abs(d - ref[(base, ru, b)]) > 0.06:
                    raise SystemExit(
                        f"{ru} 1/{b} {LEARNED}-{base}: {d:.2f} contra {ref[(base, ru, b)]}"
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
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=200)
    print(f"diferenças conferidas contra o hypotheses.csv; escrito: {OUT} e {PNG}")


if __name__ == "__main__":
    main()
