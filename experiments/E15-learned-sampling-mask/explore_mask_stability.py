"""E15, análise exploratória (só diagnósticos de treino): estabilidade da máscara usada.

Não decide nada e não está no pré-registro. Existe para separar duas leituras da regra de
convergência (B5) na máscara livre:

- a máscara livre está mesmo presa no ruído de Gumbel; ou
- o critério de margem é mais exigente para ela, porque compara o K-ésimo e o (K+1)-ésimo logit
  de um vetor quatro vezes maior (4K entre 4P), onde o vão natural na fronteira é menor.

A medida é a mesma para as duas máscaras: sorteia ``N`` ruídos de Gumbel, refaz a seleção dura que
o treino faria (top-K de ``logits + g``, com a mesma projeção da máscara usada no teste) e conta a
fração das K posições usadas que sobrevive. Na máscara livre, a projeção é `fold_project`.

Entrada: o ``masks.json`` de uma rodada. Saída: ``--out`` (por padrão, ``mask_stability.csv`` ao
lado do ``masks.json``), mais um resumo por orçamento na tela. A rodada decisiva usou:

    uv run python experiments/E15-learned-sampling-mask/explore_mask_stability.py \
        experiments/E15-learned-sampling-mask/results/masks.json --gap-ticks 2 --samples 500 \
        --out experiments/E15-learned-sampling-mask/results/exploratory/mask_stability.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from snq_learn.masks import FeasibleSet, fold_project, project_to_feasible


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return len(set(a.tolist()) & set(b.tolist())) / max(1, len(b))


def stability(scores_to_mask, logits: np.ndarray, used: np.ndarray, n: int, rng) -> float:
    g = rng.gumbel(size=(n, logits.size))
    return float(np.mean([overlap(scores_to_mask(logits + gi), used) for gi in g]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("masks_json", type=Path)
    ap.add_argument("--gap-ticks", type=int, default=2, help="min_gap_ticks do config")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--variant", default="bruto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="CSV de saida")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    masks = json.loads(args.masks_json.read_text(encoding="utf-8"))

    rows = []
    for key, arms in sorted(masks.items()):
        parts = key.split("/")
        if len(parts) != 6 or parts[1] != args.variant:
            continue
        _, var, b_tag, split, seed, method = parts
        if method == "dec_aprendida" and "dec_aprendida" in arms:
            d = arms["dec_aprendida"]
            period, k = int(d["period_ticks"]), len(d["diagnostics"]["positions_final"])
            fs = FeasibleSet(period, k, args.gap_ticks)
            w = np.asarray(d["diagnostics"]["logits_final"], dtype=np.float64)
            used = project_to_feasible(w, fs)
            st = stability(lambda s, fs=fs: project_to_feasible(s, fs), w, used, args.samples, rng)
        elif method == "dec_livre" and "dec_livre" in arms and "dec_livre_projetada" in arms:
            d, dp = arms["dec_livre"], arms["dec_livre_projetada"]
            period = int(dp["period_ticks"])
            k = (
                len(dp["positions"])
                if isinstance(dp["positions"], list)
                else len(str(dp["positions"]).split())
            )
            fs = FeasibleSet(period, k, args.gap_ticks)
            w = np.asarray(d["diagnostics"]["logits_final"], dtype=np.float64)
            k_h = k * (w.size // period)
            used = fold_project(w, k_h, fs)

            def to_mask(s, k_h=k_h, fs=fs):
                return fold_project(s, k_h, fs)

            st = stability(to_mask, w, used, args.samples, rng)
        else:
            continue
        rows.append(
            {
                "budget": int(b_tag[1:]),
                "split": split,
                "seed": seed,
                "mask": "aprendida" if method == "dec_aprendida" else "livre_projetada",
                "candidates": w.size,
                "k_selected": k if method == "dec_aprendida" else k * (w.size // period),
                "margin_over_gumbel_std": round(
                    float(d["diagnostics"]["margin_over_gumbel_std"]), 4
                ),
                "converged": int(bool(d["diagnostics"]["converged"])),
                "stability_used_mask": round(st, 4),
            }
        )

    out = args.out or (args.masks_json.parent / "mask_stability.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["budget"], r["mask"])].append(r)
    print("orcamento mascara          candidatas  convergiu  estabilidade (mediana, min)")
    for (b, mk), rr in sorted(by.items()):
        s = [r["stability_used_mask"] for r in rr]
        c = sum(r["converged"] for r in rr)
        print(
            f"1/{b:<7d} {mk:17s} {rr[0]['candidates']:>10d}  {c}/{len(rr)}        "
            f"{np.median(s):.3f}, {min(s):.3f}"
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
