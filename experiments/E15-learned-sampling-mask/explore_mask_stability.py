"""E15, exploratory analysis (training diagnostics only): stability of the mask used.

It decides nothing and is not in the preregistration. It exists to separate two readings of the
convergence rule (B5) on the free mask:

- the free mask really is stuck in the Gumbel noise; or
- the margin criterion is stricter for it, because it compares the K-th and the (K+1)-th logit
  of a vector four times larger (4K out of 4P), where the natural gap at the boundary is smaller.

The measure is the same for both masks: it draws ``N`` Gumbel noises, redoes the hard selection
that training would make (top-K of ``logits + g``, with the same projection as the mask used in
the test) and counts the fraction of the K used positions that survives. On the free mask, the
projection is `fold_project`.

Input: the ``masks.json`` of a run. Output: ``--out`` (by default, ``mask_stability.csv`` next to
``masks.json``), plus a per-budget summary on screen. The deciding run used:

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
    ap.add_argument("--gap-ticks", type=int, default=2, help="min_gap_ticks from the config")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--variant", default="bruto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="output CSV")
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
    print("budget    mask             candidates  converged  stability (median, min)")
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
