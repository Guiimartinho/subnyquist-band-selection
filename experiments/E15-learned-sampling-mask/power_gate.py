"""E15: porta de poder (README, seção "Porta de poder (pré-registro, 2026-11-12)").

Roda só em CPU, sobre o ``metrics.csv`` da rodada decisiva. Pergunta se o Ottawa consegue detectar,
pela regra da H1, um braço que seja de fato melhor que o baseline.

- **Braços plantados.** Partem de ``uniforme_treino``. Cada segmento em que ele erra vira acerto
  com probabilidade q, com o mesmo sorteio nas duas réguas; onde ele acerta, o plantado também
  acerta.
- **Bootstrap.** É o ``cluster_bootstrap`` do ``run.py``, vetorizado. O sinal de cada reamostra é
  calculado em inteiros, para que ``Δ* = 0`` conte contra a H1 exatamente como foi pré-registrado.

    uv run python experiments/E15-learned-sampling-mask/power_gate.py \
        experiments/E15-learned-sampling-mask/results/metrics.csv \
        --out experiments/E15-learned-sampling-mask/results/power_gate
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run as e15  # noqa: E402

DS, VAR = "ottawa", "bruto"
RULERS = ("local", "nested")
BASELINES = ("uniforme_treino", "dec_uniforme_treino", "dec_aleatoria")
BUDGETS = (4, 8, 16, 32)
SCALE = 360360  # mmc(1..15): médias de até 15 acertos (3 segmentos x 5 sementes) viram inteiros
ALPHA, MIN_BUDGETS = 0.05, 2


@dataclass
class Pairs:
    """Pares (divisão, registro) de um teste, com os índices do bootstrap por aglomerado."""

    keys: list
    s_idx: np.ndarray
    g_idx: np.ndarray
    c_idx: np.ndarray
    n_splits: int
    n_clusters: list[int]


def make_pairs(keys, group_of: dict, cluster_of: dict) -> Pairs:
    keys = sorted(keys, key=lambda k: (str(k[0]), k[1]))
    groups = sorted({group_of[k[1]] for k in keys})
    splits = sorted({k[0] for k in keys}, key=str)
    clusters = {g: sorted({cluster_of[k[1]] for k in keys if group_of[k[1]] == g}) for g in groups}
    return Pairs(
        keys=keys,
        s_idx=np.array([splits.index(k[0]) for k in keys]),
        g_idx=np.array([groups.index(group_of[k[1]]) for k in keys]),
        c_idx=np.array([clusters[group_of[k[1]]].index(cluster_of[k[1]]) for k in keys]),
        n_splits=len(splits),
        n_clusters=[len(clusters[g]) for g in groups],
    )


def boot_p(d: np.ndarray, pr: Pairs, n_boot: int, rng, min_clusters=3, min_splits=2):
    """(diferença em pontos, p unilateral para a > b), como o ``cluster_bootstrap`` do run.py."""
    ng = len(pr.n_clusters)
    point = float(np.mean([d[pr.g_idx == g].mean() for g in range(ng)]))
    if min(pr.n_clusters) < min_clusters or pr.n_splits < min_splits:
        return 100.0 * point, 1.0
    di = np.rint(d * SCALE).astype(np.int64)
    sw = rng.multinomial(pr.n_splits, np.full(pr.n_splits, 1.0 / pr.n_splits), size=n_boot)
    nums, dens = [], []
    for g in range(ng):
        sel = pr.g_idx == g
        nc = pr.n_clusters[g]
        cw = rng.multinomial(nc, np.full(nc, 1.0 / nc), size=n_boot)
        w = sw[:, pr.s_idx[sel]] * cw[:, pr.c_idx[sel]]
        nums.append(w @ di[sel])
        dens.append(w.sum(axis=1))
    valid = np.all([den > 0 for den in dens], axis=0)
    # sinal exato de sum_g num_g/den_g: multiplica pelo produto (positivo) dos denominadores
    total = np.zeros(n_boot, dtype=np.int64)
    for g in range(ng):
        others = np.ones(n_boot, dtype=np.int64)
        for h in range(ng):
            if h != g:
                others = others * dens[h]
        total = total + nums[g] * others
    total = total[valid]
    return 100.0 * point, float((1 + np.count_nonzero(total <= 0)) / (1 + total.size))


def decide(tests: list[tuple[int, float, float]]) -> tuple[bool, set[int]]:
    """Regra da H1 numa régua: Holm sobre os testes; orçamento vence se todos os dele passam."""
    adj = e15.holm([p for _, _, p in tests])
    passed = defaultdict(list)
    for (b, diff, _), pa in zip(tests, adj, strict=True):
        passed[b].append(bool(pa < ALPHA and diff > 0.0))
    wins = {b for b, v in passed.items() if v and all(v)}
    return len(wins) >= MIN_BUDGETS, wins


# ----------------------------------------------------------------------------- dados reais
@dataclass
class Cell:
    """Um (orçamento, régua) do desenho real: segmentos do baseline e os pares de cada teste."""

    seg_key: np.ndarray  # índice do par (no baseline) de cada segmento válido
    seg_hit: np.ndarray  # acerto do baseline em cada segmento válido
    seg_row: np.ndarray  # índice da linha de segmento (para sortear igual nas duas réguas)
    keys: list  # pares do baseline
    bal: float  # acurácia balanceada do baseline (pontos)
    tests: dict  # baseline -> (Pairs, índices do plantado nos pares, valores do baseline)


def load(metrics: Path):
    rows = [
        r
        for r in csv.DictReader(metrics.open(encoding="utf-8"))
        if r["dataset"] == DS and r["variant"] == VAR
    ]
    group_of = {r["record"]: r["group"] for r in rows}
    cluster_of = {r["record"]: r["record"].rsplit("-", 1)[0] for r in rows}
    return rows, group_of, cluster_of


def build_cells(rows, group_of, cluster_of_unit) -> dict:
    cells = {}
    for b in BUDGETS:
        base_rows = [
            r
            for r in e15.arm_rows(rows, DS, VAR, b, "uniforme_treino")
            if r.get("family_true") not in (e15.HEALTHY, None, "")
        ]
        for ruler in RULERS:
            per_b = e15.per_record(base_rows, ruler)
            keys = sorted(per_b, key=lambda k: (str(k[0]), k[1]))
            kidx = {k: i for i, k in enumerate(keys)}
            seg_key, seg_hit, seg_row = [], [], []
            for i, r in enumerate(base_rows):
                h = r.get(f"hit_{ruler}")
                if h in ("", None):
                    continue
                seg_key.append(kidx[(r["split"], r["record"])])
                seg_hit.append(int(h))
                seg_row.append(i)
            gacc = defaultdict(list)
            for (_, rid), v in per_b.items():
                gacc[group_of[rid]].append(v)
            bal = 100.0 * float(np.mean([np.mean(v) for v in gacc.values()]))
            tests = {}
            for base in BASELINES:
                other = e15.per_record(e15.arm_rows(rows, DS, VAR, b, base), ruler)
                common = [k for k in keys if k in other]
                pr = make_pairs(common, group_of, cluster_of_unit)
                tests[base] = (
                    pr,
                    np.array([kidx[k] for k in pr.keys]),
                    np.array([other[k] for k in pr.keys]),
                )
            cells[(b, ruler)] = Cell(
                np.array(seg_key), np.array(seg_hit), np.array(seg_row), keys, bal, tests
            )
    return cells


def planted_means(cell: Cell, u: np.ndarray, q: float) -> np.ndarray:
    """Média por par do braço plantado; ``u`` tem forma (sementes, linhas de segmento)."""
    uu = u[:, cell.seg_row]
    hit = np.where(cell.seg_hit[None, :] == 1, 1.0, (uu < q).astype(float))
    num = np.bincount(cell.seg_key, weights=hit.sum(axis=0), minlength=len(cell.keys))
    den = np.bincount(cell.seg_key, minlength=len(cell.keys)) * u.shape[0]
    return num / den


def run_real(cells, n_rows: dict, effect, n_seeds: int, n_sim: int, n_boot: int, rng):
    """Desenhos D0 a D3. Devolve contagens de aprovação pelas duas regras."""
    out = {
        rule: {"h1": 0, "ruler": {ru: 0 for ru in RULERS}, "wins": defaultdict(int)}
        for rule in ("pre", "otim")
    }
    qs = {
        (b, ru): (1.0 if effect == "perfeito" else min(1.0, effect / (100.0 - c.bal)))
        if c.bal < 100.0
        else 0.0
        for (b, ru), c in cells.items()
    }
    for _ in range(n_sim):
        passed = {"pre": {}, "otim": {}}
        u_by_b = {b: rng.random((n_seeds, n_rows[b])) for b in BUDGETS}
        for ru in RULERS:
            pre, otim = [], []
            for b in BUDGETS:
                c = cells[(b, ru)]
                a = planted_means(c, u_by_b[b], qs[(b, ru)])
                for base in BASELINES:
                    pr, ai, bv = c.tests[base]
                    diff, p = boot_p(a[ai] - bv, pr, n_boot, rng)
                    pre.append((b, diff, p))
                    if base == "uniforme_treino":
                        otim.append((b, diff, p))
                    else:
                        otim.append((b, 1.0, 0.0))
            for rule, tests in (("pre", pre), ("otim", otim)):
                ok, wins = decide(tests)
                passed[rule][ru] = ok
                out[rule]["ruler"][ru] += int(ok)
                for b in wins:
                    out[rule]["wins"][(ru, b)] += 1
        for rule in ("pre", "otim"):
            out[rule]["h1"] += int(all(passed[rule].values()))
    return out, qs


# ----------------------------------------------------------------------------- DK sintético
def dk_params(cells, group_of):
    """Por (orçamento, régua) e pista: taxa de erro do baseline e segmentos válidos por registro."""
    par = {}
    for (b, ru), c in cells.items():
        groups = sorted({group_of[k[1]] for k in c.keys})
        per = defaultdict(list)
        counts = np.bincount(c.seg_key, minlength=len(c.keys))
        hits = np.bincount(c.seg_key, weights=c.seg_hit, minlength=len(c.keys))
        for i, k in enumerate(c.keys):
            per[group_of[k[1]]].append((hits[i], counts[i]))
        par[(b, ru)] = {
            g: (
                1.0 - float(np.mean([h / n for h, n in per[g]])),
                int(np.median([n for _, n in per[g]])),
            )
            for g in groups
        }
    return par


def run_dk(par, effect, n_seeds: int, n_sim: int, n_boot: int, rng):
    """Cada registro testado uma vez: 12 por pista (4 aglomerados x 3), 4 dobras estratificadas."""
    out = {"h1": 0, "ruler": {ru: 0 for ru in RULERS}, "wins": defaultdict(int)}
    groups = sorted({g for v in par.values() for g in v})
    recs = [(g, c, j) for g in groups for c in range(4) for j in range(3)]
    group_of = {f"{g}|{c}|{j}": g for g, c, j in recs}
    cluster_of = {f"{g}|{c}|{j}": f"{g}|{c}" for g, c, j in recs}
    keys = [((c + j) % 4, f"{g}|{c}|{j}") for g, c, j in recs]
    pr = make_pairs(keys, group_of, cluster_of)
    qs = {}
    for b in BUDGETS:
        for ru in RULERS:
            m = np.mean([par[(b, ru)][g][0] for g in groups])
            qs[(b, ru)] = (
                1.0 if effect == "perfeito" else (min(1.0, effect / (100.0 * m)) if m > 0 else 0.0)
            )
    n_max = {b: {g: max(par[(b, ru)][g][1] for ru in RULERS) for g in groups} for b in BUDGETS}
    for _ in range(n_sim):
        passed = {}
        draws = {}
        for b in BUDGETS:
            draws[b] = {
                g: (rng.random((12, n_max[b][g])), rng.random((n_seeds, 12, n_max[b][g])))
                for g in groups
            }
        for ru in RULERS:
            tests = []
            for b in BUDGETS:
                a_k, b_k = [], []
                for k in pr.keys:
                    g, c, j = k[1].split("|")
                    r_i = int(c) * 3 + int(j)
                    miss, n_seg = par[(b, ru)][g]
                    u_b, u_p = draws[b][g]
                    base_hit = (u_b[r_i, :n_seg] >= miss).astype(float)
                    plant = np.where(
                        base_hit[None, :] == 1, 1.0, (u_p[:, r_i, :n_seg] < qs[(b, ru)])
                    )
                    a_k.append(float(plant.mean()))
                    b_k.append(float(base_hit.mean()))
                diff, p = boot_p(np.array(a_k) - np.array(b_k), pr, n_boot, rng)
                tests += [(b, diff, p), (b, 1.0, 0.0), (b, 1.0, 0.0)]
            ok, wins = decide(tests)
            passed[ru] = ok
            out["ruler"][ru] += int(ok)
            for b in wins:
                out["wins"][(ru, b)] += 1
        out["h1"] += int(all(passed.values()))
    return out, qs


# ----------------------------------------------------------------------------- validação
def validate(rows, group_of, cluster_of, hyp_csv: Path, n_boot: int, seed: int) -> list[dict]:
    """O bootstrap vetorizado contra o hypotheses.csv real (p e diferença)."""
    ref = {
        (h["ruler"], int(h["budget"]), h["baseline"]): h
        for h in csv.DictReader(hyp_csv.open(encoding="utf-8"))
        if h["hypothesis"] == "H1" and h.get("budget") and h.get("evaluable") == "1"
    }
    out = []
    rng = np.random.default_rng(seed)
    for ru in RULERS:
        for b in BUDGETS:
            a = e15.per_record(e15.arm_rows(rows, DS, VAR, b, "dec_aprendida_sel"), ru)
            for base in BASELINES:
                bb = e15.per_record(e15.arm_rows(rows, DS, VAR, b, base), ru)
                pr = make_pairs(set(a) & set(bb), group_of, cluster_of)
                d = np.array([a[k] - bb[k] for k in pr.keys])
                diff, p = boot_p(d, pr, n_boot, rng)
                h = ref[(ru, b, base)]
                out.append(
                    {
                        "ruler": ru,
                        "budget": b,
                        "baseline": base,
                        "diff_ref": float(h["diff_points"]),
                        "diff_vec": round(diff, 2),
                        "p_ref": round(float(h["p_one_sided"]), 4),
                        "p_vec": round(p, 4),
                    }
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--n-sim-perfect", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((EXP / "config" / "base.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    args.out.mkdir(parents=True, exist_ok=True)
    rows, group_of, cluster_of = load(args.metrics_csv)

    val = validate(
        rows, group_of, cluster_of, args.metrics_csv.parent / "hypotheses.csv", 4000, seed
    )
    (args.out / "validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
    worst_p = max(abs(v["p_vec"] - v["p_ref"]) for v in val)
    worst_d = max(abs(v["diff_vec"] - v["diff_ref"]) for v in val)
    print(f"validacao: maior |dp| = {worst_p:.4f}, maior |ddiff| = {worst_d:.3f} ponto")
    if args.validate_only:
        return

    n_rows = {
        b: len(
            [
                r
                for r in e15.arm_rows(rows, DS, VAR, b, "uniforme_treino")
                if r.get("family_true") not in (e15.HEALTHY, None, "")
            ]
        )
        for b in BUDGETS
    }
    cells_cluster = build_cells(rows, group_of, cluster_of)
    cells_record = build_cells(rows, group_of, {k: k for k in group_of})
    par = dk_params(cells_cluster, group_of)
    designs = [
        ("D0", cells_cluster, 1),
        ("D1", cells_cluster, 3),
        ("D2", cells_cluster, 5),
        ("D3_registro_anticonservador", cells_record, 5),
    ]
    results = []
    for effect in (5, 10, "perfeito"):
        for name, cells, n_seeds in designs:
            n_sim = args.n_sim_perfect if effect == "perfeito" else args.n_sim
            rng = np.random.default_rng(e15.derive_seed(seed, "power_gate", name, effect))
            out, qs = run_real(cells, n_rows, effect, n_seeds, n_sim, args.n_boot, rng)
            for rule in ("pre", "otim"):
                results.append(row_of(name, effect, rule, n_sim, out[rule], qs, cells))
            print(summary_line(name, effect, n_sim, out, qs))
        rng = np.random.default_rng(e15.derive_seed(seed, "power_gate", "DK", effect))
        out_dk, qs_dk = run_dk(par, effect, 5, args.n_sim, args.n_boot, rng)
        results.append(row_of("DK", effect, "otim", args.n_sim, out_dk, qs_dk, None))
        print(
            f"DK {effect!s:>9}: otimista H1 {out_dk['h1'] / args.n_sim:.2f} "
            f"(local {out_dk['ruler']['local'] / args.n_sim:.2f}, "
            f"aninhada {out_dk['ruler']['nested'] / args.n_sim:.2f})"
        )
    with (args.out / "power.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    (args.out / "baseline.json").write_text(
        json.dumps(
            {
                "bal_baseline": {
                    f"{b}_{ru}": round(c.bal, 2) for (b, ru), c in cells_cluster.items()
                },
                "dk_params": {f"{b}_{ru}": v for (b, ru), v in par.items()},
                "n_sim": args.n_sim,
                "n_sim_perfect": args.n_sim_perfect,
                "n_boot": args.n_boot,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"-> {args.out}")


def row_of(name, effect, rule, n_sim, out, qs, cells) -> dict:
    row = {
        "design": name,
        "effect_points": effect,
        "rule": "pre_registrada" if rule == "pre" else "otimista",
        "n_sim": n_sim,
        "power_h1": round(out["h1"] / n_sim, 3),
        "power_local": round(out["ruler"]["local"] / n_sim, 3),
        "power_nested": round(out["ruler"]["nested"] / n_sim, 3),
    }
    for ru in RULERS:
        for b in BUDGETS:
            row[f"win_{ru}_b{b}"] = round(out["wins"][(ru, b)] / n_sim, 3)
            row[f"q_{ru}_b{b}"] = round(qs[(b, ru)], 3)
            if cells is not None:
                row[f"max_effect_{ru}_b{b}"] = round(100.0 - cells[(b, ru)].bal, 2)
    return row


def summary_line(name, effect, n_sim, out, qs) -> str:
    return (
        f"{name:28s} {effect!s:>9}: pre H1 {out['pre']['h1'] / n_sim:.2f} | "
        f"otimista H1 {out['otim']['h1'] / n_sim:.2f} "
        f"(local {out['otim']['ruler']['local'] / n_sim:.2f}, "
        f"aninhada {out['otim']['ruler']['nested'] / n_sim:.2f})"
    )


if __name__ == "__main__":
    main()
