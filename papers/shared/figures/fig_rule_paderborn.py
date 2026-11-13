"""Figura da regra das ordens de falha no Paderborn (E26, E26b e E26c).

Regra do projeto: a figura nasce de metrics.csv, nunca de número transcrito.

- **Painéis:**
  - (a) decisão e banda sob decimação pura, no fluxo bruto: estimação cega (E26 `kurtograma`),
    banda do kurtograma e banda larga com a regra (E26b) e banda guiada com a regra (E26 `guiado`);
  - (b) aquisição na mesma taxa (E26c): decimação pura contra o anti-alias ideal, para a banda
    guiada e a banda larga com a regra;
  - (c) perfil por zona de Nyquist (E26c): banda larga com a regra depois do passa-faixa ideal de
    cada zona, em 4× e 8×; a zona 0 é o anti-alias.
- **Métrica:** a das análises do E26. O limiar é o percentil 95 do maior escore nos rolamentos sãos
  (deixa-um-são-de-fora para avaliar um são). A acurácia é por rolamento, com média por pista e
  depois média das pistas. A estimação cega acerta quando cai a ±0,025 ordem da régua aninhada.
- **Incerteza:** banda de 95 % por bootstrap dos rolamentos dentro de cada pista (4000), como na
  figura do E24.
- **Conferência:** cada ponto tem de bater com o `analysis.json` da rodada (±0,1 ponto: o
  `analysis.json` arredonda cada pista e depois a média); se não
  bater, o script para.

    uv run python papers/shared/figures/fig_rule_paderborn.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable
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
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
E26 = ROOT / "experiments/E26-fault-order-guided-selector"
CONF = E26 / "results-paderborn"
RULE = E26 / "results-paderborn-regra"
ACQ = E26 / "results-paderborn-aquisicao"
OUT = ROOT / "papers/mssp-2026/figures/fig_rule_paderborn.pdf"
PNG = ACQ / "figures/e26_rule_paderborn.png"
FS_FULL_KSPS = 64.0
FAULTS = ("BPFO", "BPFI")
Q = 95.0
# paleta conferida no validador da skill dataviz (croma, daltonismo, contraste)
GRAY, TEAL, RUST, PURPLE = "0.55", "#0089A3", "#C0501F", "#7B4CC0"

Rows = list[dict[str, str]]
Score = Callable[[dict[str, str], str], float]


def read(path: Path) -> Rows:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v: str) -> float:
    return float(v) if v != "" else -np.inf


def rule_per_bearing(rows: Rows, score: Score) -> dict[tuple[str, str], float]:
    """Acerto por (pista, rolamento) da regra, com o limiar dos sãos (como `evaluate` do E26c)."""
    hs: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["race"] == "H":
            hs[r["bearing"]].append(max(score(r, c) for c in FAULTS))
    tau_all = float(np.percentile([s for v in hs.values() for s in v], Q))
    tau = {h: float(np.percentile([s for b, v in hs.items() if b != h for s in v], Q)) for h in hs}
    acc: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        th = tau[r["bearing"]] if r["race"] == "H" else tau_all
        top = max(FAULTS, key=lambda c: score(r, c))
        pred = ("OR" if top == "BPFO" else "IR") if score(r, top) >= th else "none"
        acc[(r["race"], r["bearing"])].append(
            int(pred == ("none" if r["race"] == "H" else r["race"]))
        )
    return {k: float(np.mean(v)) for k, v in acc.items()}


def blind_per_bearing(rows: Rows, calib: Rows, tol: float) -> dict[tuple[str, str], float]:
    """Acerto da estimação cega contra a régua aninhada (como `analyze_paderborn.py`)."""
    valid = [c for c in calib if c["ok"] == "1"]
    all_med = {
        race: float(np.median([float(c["center"]) for c in valid if c["race"] == race]))
        for race in ("OR", "IR")
    }

    def truth(bearing: str, own: str, race: str) -> float:
        if race == own:
            peers = [
                float(c["center"]) for c in valid if c["race"] == race and c["bearing"] != bearing
            ]
            return float(np.median(peers))
        return all_med[race]

    acc: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        est = float(r["blind_order"]) if r["blind_order"] != "" else np.nan
        pred = "none"
        for rc in ("OR", "IR"):
            if np.isfinite(est) and abs(est - truth(r["bearing"], r["race"], rc)) <= tol:
                pred = rc
        acc[(r["race"], r["bearing"])].append(
            int(pred == ("none" if r["race"] == "H" else r["race"]))
        )
    return {k: float(np.mean(v)) for k, v in acc.items()}


def balanced_ci(pb: dict[tuple[str, str], float], n_boot: int = 4000, seed: int = 0):
    """Ponto e IC 95 % da acurácia balanceada nos rolamentos com falha."""
    by: dict[str, list[float]] = defaultdict(list)
    for (race, _b), v in sorted(pb.items()):
        if race in ("OR", "IR"):
            by[race].append(v)
    arr = [np.array(v) for v in by.values()]
    point = 100.0 * float(np.mean([v.mean() for v in arr]))
    rng = np.random.default_rng(seed)
    boots = [
        np.mean([v[rng.integers(0, v.size, v.size)].mean() for v in arr]) for _ in range(n_boot)
    ]
    lo, hi = 100.0 * np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def check(point: float, summary: dict, key: str) -> None:
    ref = summary[key]["balanced"]
    if abs(point - ref) > 0.1:
        raise SystemExit(f"{key}: figura {point:.2f} contra analysis.json {ref}")


def series(cells: dict[int, dict[tuple[str, str], float]]):
    xs = sorted(cells)
    pts = [balanced_ci(cells[x]) for x in xs]
    return xs, np.array(pts)


def draw(ax, xs, pts, label, color, marker, ls, z, filled=True):
    ax.fill_between(xs, pts[:, 1], pts[:, 2], color=color, alpha=0.10, lw=0, zorder=z)
    ax.plot(
        xs,
        pts[:, 0],
        ls=ls,
        marker=marker,
        color=color,
        mfc=color if filled else "white",
        ms=4,
        lw=1.1,
        label=label,
        zorder=z + 5,
    )


def factor_axis(ax, factors: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(factors)
    ax.set_xticklabels([str(f) for f in factors])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("decimation factor $M$")
    top = ax.secondary_xaxis(
        "top", functions=(lambda m: FS_FULL_KSPS / m, lambda r: FS_FULL_KSPS / r)
    )
    top.set_xticks([FS_FULL_KSPS / f for f in factors])
    top.set_xticklabels([f"{FS_FULL_KSPS / f:g}" for f in factors], fontsize=7)
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top.set_xlabel("effective rate $f_s/M$ [kS/s]", fontsize=7)


def main() -> None:
    cfg = yaml.safe_load((E26 / "config/paderborn.yaml").read_text(encoding="utf-8"))
    tol = float(cfg["tolerance_order"])
    s_conf = json.loads((CONF / "analysis.json").read_text(encoding="utf-8"))["summary"]
    s_rule = json.loads((RULE / "analysis.json").read_text(encoding="utf-8"))["summary"]
    s_acq = json.loads((ACQ / "analysis.json").read_text(encoding="utf-8"))["summary"]

    conf = [r for r in read(CONF / "metrics.csv") if not r["skipped"] and r["clean"] == "bruto"]
    calib = read(CONF / "calib.csv")
    rule = [r for r in read(RULE / "metrics.csv") if r["clean"] == "bruto"]
    acq = read(ACQ / "metrics.csv")

    def by_factor(rows: Rows, keep: Callable[[dict[str, str]], bool]) -> dict[int, Rows]:
        out: dict[int, Rows] = defaultdict(list)
        for r in rows:
            if keep(r):
                out[int(r["factor"])].append(r)
        return out

    def score_conf(r: dict[str, str], c: str) -> float:
        return num(r[f"score_{c}"])

    def score_acq(r: dict[str, str], c: str) -> float:
        return num(r[f"s_{c}"])

    # (a) decisão e banda, decimação pura, fluxo bruto
    conf_f = by_factor(conf, lambda r: True)
    blind = {f: blind_per_bearing(rs, calib, tol) for f, rs in conf_f.items()}
    guided = {f: rule_per_bearing(rs, score_conf) for f, rs in conf_f.items()}
    kurt_rule = {
        f: rule_per_bearing(rs, score_conf)
        for f, rs in by_factor(rule, lambda r: r["strategy"] == "kurtograma").items()
    }
    wide_rule = {
        f: rule_per_bearing(rs, score_conf)
        for f, rs in by_factor(rule, lambda r: r["strategy"] == "banda_larga").items()
    }
    for f in blind:
        check(balanced_ci(blind[f])[0], s_conf, f"kurtograma|{f}x")
        check(balanced_ci(guided[f])[0], s_conf, f"guiado|{f}x")
    for f in kurt_rule:
        check(balanced_ci(kurt_rule[f])[0], s_rule, f"kurtograma_regra|bruto|{f}x")
        check(balanced_ci(wide_rule[f])[0], s_rule, f"banda_larga_regra|bruto|{f}x")

    # (b) aquisição: pura contra anti-alias
    acq_cells = {}
    for acqn in ("pura", "antialias"):
        for strat in ("guiado", "banda_larga_regra"):
            cells = {
                f: rule_per_bearing(rs, score_acq)
                for f, rs in by_factor(
                    acq, lambda r, a=acqn, s=strat: r["acquisition"] == a and r["strategy"] == s
                ).items()
            }
            for f, pb in cells.items():
                check(balanced_ci(pb)[0], s_acq, f"{acqn}|{strat}|{f}x")
            acq_cells[(acqn, strat)] = cells

    # (c) perfil por zona, banda larga com a regra
    zones: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    for m in (4, 8):
        prof = []
        for k in range(m):
            name = "antialias" if k == 0 else f"zona{k}"
            rs = [
                r
                for r in acq
                if int(r["factor"]) == m
                and r["acquisition"] == name
                and r["strategy"] == "banda_larga_regra"
            ]
            pt = balanced_ci(rule_per_bearing(rs, score_acq))
            check(pt[0], s_acq, f"{name}|banda_larga_regra|{m}x")
            prof.append((k, pt))
        zones[m] = prof

    # largura do texto do elsarticle (390 pt); (a) e (b) em cima, (c) embaixo na largura toda
    fig = plt.figure(figsize=(5.4, 5.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    ax = axes[0]
    for cells, label, color, marker, ls, z in (
        (blind, "blind order estimation", GRAY, "o", "--", 1),
        (kurt_rule, "kurtogram band + rule", PURPLE, "^", ":", 2),
        (wide_rule, "whole band + rule", TEAL, "s", "-.", 3),
        (guided, "guided band + rule", "black", "D", "-", 4),
    ):
        xs, pts = series(cells)
        draw(ax, xs, pts, label, color, marker, ls, z)
    factor_axis(ax, sorted(blind))
    ax.set_ylabel("balanced accuracy [%]")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.93))
    ax.text(0.02, 0.98, "(a) decision and band", transform=ax.transAxes, va="top")

    ax = axes[1]
    for (acqn, strat), label, color, marker, ls, filled, z in (
        (("pura", "banda_larga_regra"), "whole band, pure", TEAL, "s", "-", True, 1),
        (("antialias", "banda_larga_regra"), "whole band, anti-alias", TEAL, "s", "--", False, 2),
        (("pura", "guiado"), "guided band, pure", "black", "D", "-", True, 3),
        (("antialias", "guiado"), "guided band, anti-alias", "black", "D", "--", False, 4),
    ):
        xs, pts = series(acq_cells[(acqn, strat)])
        draw(ax, xs, pts, label, color, marker, ls, z, filled)
    factor_axis(ax, sorted(acq_cells[("pura", "guiado")]))
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.93))
    ax.text(0.02, 0.98, "(b) acquisition at equal rate", transform=ax.transAxes, va="top")

    ax = axes[2]
    for m, color, marker, z in ((4, TEAL, "o", 2), (8, RUST, "s", 3)):
        width = FS_FULL_KSPS / 2 / m
        for k, (p, lo, hi) in zones[m]:
            ax.plot([k * width, (k + 1) * width], [p, p], color=color, lw=1.4, zorder=z)
            ax.errorbar(
                (k + 0.5) * width,
                p,
                yerr=[[p - lo], [hi - p]],
                color=color,
                marker=marker,
                ms=4,
                lw=0.8,
                capsize=1.5,
                zorder=z + 1,
                label=f"$M = {m}$, zones of {width:g} kHz" if k == 0 else None,
            )
    ax.set_xlim(0, FS_FULL_KSPS / 2)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_xlabel("original frequency of the zone [kHz]")
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.93))
    ax.text(0.02, 0.98, "(c) whole band + rule by zone", transform=ax.transAxes, va="top")

    axes[2].set_ylabel("balanced accuracy [%]")
    for ax in axes:
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=200)
    print(f"conferido contra os analysis.json; escrito: {OUT} e {PNG}")


if __name__ == "__main__":
    main()
