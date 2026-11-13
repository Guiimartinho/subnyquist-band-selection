"""E15: the unrolled decoder against snq_recon, and the runner's helpers.

The tests that cross packages live here, and not in packages/snq-learn/tests, because snq-learn
cannot import snq-recon (dependency rule in CLAUDE.md).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from snq_data import undersample
from snq_learn import (
    FixedMask,
    OrderAxis,
    OrderDecoder,
    SampledOrderModel,
    angular_plan,
    envelope_sample_positions,
    stack_plans,
)
from snq_recon import (
    envelope_order_spectrum,
    estimate_without_prior,
    fast_kurtogram,
    phase_revolutions,
)

_spec = importlib.util.spec_from_file_location(
    "e15_run", Path(__file__).resolve().parents[1] / "run.py"
)
e15 = importlib.util.module_from_spec(_spec)
sys.modules["e15_run"] = e15  # dataclasses looks the module up in sys.modules
_spec.loader.exec_module(e15)

FS = 20_000.0
ORDER = 5.43


def ramp_phase(n: int, v0: float = 15.0, v1: float = 20.0):
    v = v0 + (v1 - v0) * np.arange(n) / (n - 1)
    return v, phase_revolutions(v, FS, n)


def plan_for(theta: np.ndarray, n: int, dec: OrderDecoder):
    th = np.interp(envelope_sample_positions(n, dec.n_env(n)), np.arange(n), theta)
    return stack_plans([angular_plan(th, dec.axis.samples_per_rev, dec.axis.n_fft)])


def one_band_decoder(fs_hz: float, band: tuple[float, float], n: int, order_min: float = 1.0):
    """One channel, ideal filter (one control point per bin), |y|: the snq_recon envelope."""
    return OrderDecoder(
        fs_hz=fs_hz,
        fs_env_hz=2048.0,
        axis=OrderAxis(256, 32768, order_min, 30.0),
        bands_hz=[band],
        n_control=n,
        transition_frac=0.0,
        envelope_power=1,
    )


def compare(ref, dec: OrderDecoder, spec_ours: np.ndarray, lo: float = 1.0, hi: float = 30.0):
    sel = (ref.freqs_hz >= lo) & (ref.freqs_hz <= hi)
    ours_at_ref = np.interp(ref.freqs_hz[sel], dec.axis.orders, spec_ours)
    corr = float(np.corrcoef(ours_at_ref, ref.magnitude[sel])[0, 1])
    peak_ref = float(ref.freqs_hz[sel][np.argmax(ref.magnitude[sel])])
    peak_ours = float(dec.axis.orders[np.argmax(spec_ours)])
    ratio = float(spec_ours.max() / ref.magnitude[sel].max())
    return corr, peak_ref, peak_ours, ratio


def test_espectro_de_ordem_diferenciavel_concorda_com_envelope_order_spectrum():
    """Synthetic AM signal at a known order, variable speed: same peak and same shape."""
    n = 40960
    rng = np.random.default_rng(0)
    v, theta = ramp_phase(n)
    t = np.arange(n) / FS
    x = (1.0 + 0.5 * np.cos(2 * np.pi * ORDER * theta)) * np.sin(2 * np.pi * 4000.0 * t)
    x += 0.05 * rng.normal(size=n)
    band = (3000.0, 5000.0)
    ref = envelope_order_spectrum(x, FS, v, band, samples_per_rev=64)
    dec = one_band_decoder(FS, band, n)
    with torch.no_grad():
        spec, _ = dec.spectra(
            torch.tensor(x[None, :], dtype=torch.float32), plan_for(theta, n, dec)
        )
    corr, peak_ref, peak_ours, ratio = compare(ref, dec, spec[0, 0].numpy())
    assert abs(peak_ref - ORDER) < 0.06
    assert abs(peak_ours - peak_ref) < 0.03  # less than one bin of the snq_recon spectrum
    assert corr > 0.98, corr
    assert ratio == pytest.approx(1.0, abs=0.1)  # same scale, 2/n·|DFT|


def test_mascara_uniforme_reproduz_o_braco_uniforme_do_e22():
    """At initialization, uniform mask + band chosen by the kurtogram of the decimated signal:
    the E15 decoder gives the same spectrum and the same estimate as the E22 pipeline."""
    n, factor = 40960, 4
    rng = np.random.default_rng(1)
    v, theta = ramp_phase(n)
    t = np.arange(n) / FS
    hit = np.interp(np.arange(1, int(theta[-1] * ORDER)) / ORDER, theta, t)
    ht = np.arange(int(0.004 * FS)) / FS
    h = np.exp(-ht / 0.0008) * np.sin(2 * np.pi * 4000.0 * ht)
    x = 0.2 * rng.normal(size=n)
    for ti in hit:
        i0 = int(round(ti * FS))
        seg = h[: max(0, min(h.size, n - i0))]
        x[i0 : i0 + seg.size] += seg
    xd, vd = undersample(x, factor), undersample(v, factor)
    fs_eff = FS / factor
    band = fast_kurtogram(xd, fs_eff, max_level=4, f_min_hz=500.0).band_hz
    ref = envelope_order_spectrum(xd, fs_eff, vd, band, samples_per_rev=64)
    est_ref = estimate_without_prior(ref, 2.0)
    dec = one_band_decoder(FS, band, n, order_min=2.0)
    dec.period_ticks = factor
    model = SampledOrderModel(FixedMask([0], factor), dec).eval()
    xt = torch.tensor(x[None, :], dtype=torch.float32)
    with torch.no_grad():
        spec, _ = dec.spectra(xt * model.sampler(xt), plan_for(theta, n, dec))
        logits, _ = model(xt, plan_for(theta, n, dec))
    corr, peak_ref, peak_ours, _ = compare(ref, dec, spec[0, 0].numpy(), lo=2.0)
    assert abs(est_ref - ORDER) < 0.06
    assert abs(float(dec.estimate(logits)[0]) - est_ref) < 0.03
    assert corr > 0.97, corr


# ----------------------------------------------------------------------
# runner helpers
# ----------------------------------------------------------------------
GROUP = {"a1": "g1", "a2": "g1", "a3": "g1", "b1": "g2", "b2": "g2", "c1": "g3"}
CALIB = {"a1": 5.40, "a2": 5.42, "a3": 5.50, "b1": 3.38, "b2": 3.40, "c1": 2.00}
OK = dict.fromkeys(GROUP, True)


def test_regua_aninhada_so_usa_registros_fora_do_teste():
    recs = list(GROUP)
    tr = e15.nested_truths(recs, GROUP, CALIB, OK, test={"a3", "b2"})
    # test: median of the group's records outside the test set; its own calibration is excluded
    assert tr["a3"]["truth"] == pytest.approx(np.median([5.40, 5.42]))
    assert tr["b2"]["truth"] == pytest.approx(3.38)
    # outside the test set: leave-one-out only among those outside it (a3 is in the test set)
    assert tr["a1"]["truth"] == pytest.approx(5.42)
    assert tr["a1"]["source"] == "nested"
    # single-record class outside the test set: its own calibration, flagged
    assert tr["c1"]["truth"] == pytest.approx(2.0) and tr["c1"]["source"] == "own_calibration"


def test_calibracao_na_borda_fica_fora_das_medianas():
    ok = {**OK, "a2": False}
    tr = e15.nested_truths(list(GROUP), GROUP, CALIB, ok, test={"a3"})
    assert tr["a3"]["truth"] == pytest.approx(5.40)


def test_escolha_da_uniforme_usa_a_regua_por_segmento_e_empata_para_a_maior_taxa():
    # local ruler differs per segment: the line moves within the record
    truth = {("a1", 1): 5.40, ("a1", 2): 5.30, ("a2", 1): 5.40, ("a2", 2): math.nan}
    segs = {"a1": [1, 2], "a2": [1, 2]}
    est = {}
    for r in ("a1", "a2"):
        for s in (1, 2):
            est[(r, s, 8)] = truth.get((r, s), 5.4) if np.isfinite(truth[(r, s)]) else 9.0
            est[(r, s, 9)] = est[(r, s, 8)] + 0.01  # also a hit: tie with 8
            est[(r, s, 10)] = 1.0  # a miss
    # the segment without a ruler (a2, 2) stays out of the denominator
    assert e15.hits_at(8, ["a1", "a2"], truth, est, segs, 0.06) == (3, 3)
    assert e15.choose_factor(8, [8, 9, 10], ["a1", "a2"], truth, est, segs, 0.06) == 8
    assert e15.choose_factor(9, [8, 9, 10], ["a1", "a2"], truth, est, segs, 0.06) == 9
    # a constant estimate at segment 1's line misses segment 2: the local ruler sees that
    est_const = {(r, s, 8): 5.40 for r in ("a1", "a2") for s in (1, 2)}
    assert e15.hits_at(8, ["a1"], truth, est_const, segs, 0.06) == (1, 2)
    # a segment discarded by the guard counts as a miss
    del est[("a1", 1, 8)]
    assert e15.hits_at(8, ["a1"], truth, est, segs, 0.06) == (1, 2)


def _rc(**kw):
    base = dict(
        ds="ottawa",
        variant="raw",
        split={"index": 0, "seed": 1, "train": [], "val": [], "test": []},
        budget=16,
        seed=0,
        nested={"a1": {"truth": 5.42, "source": "nested"}},
        seg_truth={("a1", 1): 5.30, ("a1", 2): math.nan},
        centers={"inner_race": 5.42, "outer_race": 3.40},
        tol=0.06,
        span=0.12,
    )
    base.update(kw)
    return e15.RowCtx(**base)


def test_linha_pontua_as_duas_reguas_e_o_controle_saudavel():
    rc = _rc()
    row = e15.score_row({"group": "inner_race"}, 5.31, rc, "a1", 1)
    assert row["hit_local"] == 1 and row["hit_nested"] == 0  # the segment's line is at 5.30
    assert row["family_pred"] == "inner_race" and row["family_true"] == "inner_race"
    row = e15.score_row({"group": "inner_race"}, 5.43, rc, "a1", 2)
    assert row["hit_local"] == "" and row["hit_nested"] == 1  # no local ruler on this segment
    healthy = e15.score_row({"group": "healthy"}, 3.41, rc, "h1", 1)
    assert healthy["family_true"] == "healthy_bearing" and healthy["fault_like"] == 1
    assert "hit_local" not in healthy


def test_familia_pela_ordem_mais_proxima_dentro_da_janela():
    centers = {"inner_race": 5.40, "outer_race": 3.39}
    assert e15.family_of_estimate(5.43, centers, 0.12) == "inner_race"
    assert e15.family_of_estimate(3.35, centers, 0.12) == "outer_race"
    assert e15.family_of_estimate(6.78, centers, 0.12) == "no_fault"  # 2 x BPFO is not BPFI
    assert e15.family_of_estimate(None, centers, 0.12) == "no_fault"
    assert e15.family_of_estimate(math.nan, centers, 0.12) == "no_fault"


def test_remocao_de_picos_troca_so_picos_isolados():
    rng = np.random.default_rng(3)
    x = np.sin(2 * np.pi * 0.01 * np.arange(20000)) + 0.01 * rng.normal(size=20000)
    y = x.copy()
    spikes = np.arange(100, 20000, 997)
    y[spikes] -= 0.5
    clean, n_rep = e15.despike(y, 10.0)
    assert n_rep == spikes.size
    assert np.allclose(clean[spikes], x[spikes], atol=0.05)
    untouched = np.setdiff1d(np.arange(20000), spikes)
    assert np.array_equal(clean[untouched], y[untouched])


def _record(n: int, v: np.ndarray) -> object:
    return e15.Record("x", "r", "g", "c", FS, np.zeros(n), v, (0, 1), [], 5.4, Path("."))


def test_plano_da_janela_usa_a_fase_do_registro_inteiro():
    n_rec, start, n = 60000, 12345, 40960
    v, _ = ramp_phase(n_rec, 12.0, 25.0)
    bank = e15.Bank(
        {"r": _record(n_rec, v)}, {}, {"r": phase_revolutions(v, FS, n_rec)}, torch.device("cpu")
    )
    dec = OrderDecoder(
        fs_hz=FS, fs_env_hz=2048.0, axis=OrderAxis(256, 32768, 2.0, 32.0), bands_hz=[(500, 900)]
    )
    got = e15.window_plan(bank, "r", start, n, dec)
    th_win = phase_revolutions(v[start : start + n], FS, n)
    th = np.interp(envelope_sample_positions(n, dec.n_env(n)), np.arange(n), th_win)
    want = angular_plan(th, 256, 32768)
    assert got["n_valid"] == want["n_valid"]
    assert np.array_equal(got["index"], want["index"])
    assert np.allclose(got["frac"], want["frac"], atol=1e-4)


def test_guarda_recusa_rotacao_que_o_decodificador_nao_cobre():
    n = 40960
    v = np.full(n, 40.0)  # 32 x 40 Hz = 1280 Hz > Nyquist of the reduced envelope (1024 Hz)
    bank = e15.Bank(
        {"r": _record(n, v)}, {}, {"r": phase_revolutions(v, FS, n)}, torch.device("cpu")
    )
    dec = OrderDecoder(
        fs_hz=FS, fs_env_hz=2048.0, axis=OrderAxis(256, 32768, 2.0, 32.0), bands_hz=[(500, 900)]
    )
    with pytest.raises(e15.GuardError):
        e15.window_plan(bank, "r", 0, n, dec)


def test_janela_de_treino_vem_da_grade_e_carrega_o_rotulo_local():
    pool = {"g1": {"a1": [(0, 5.40), (50, 5.30)]}, "g2": {"b1": [(10, 3.40)]}}
    rng = np.random.default_rng(0)
    seen = {e15.train_pick(rng, pool) for _ in range(200)}
    assert seen == {("a1", 0, 5.40), ("a1", 50, 5.30), ("b1", 10, 3.40)}


GROUPS4 = {f"{c}{k}": c for c in "IO" for k in range(8)}
CLUSTERS4 = {r: f"{r[0]}-{'ABCD'[int(r[1]) // 2]}" for r in GROUPS4}


def test_bootstrap_por_aglomerado_media_por_registro_e_acuracia_balanceada():
    # a hits everything; b hits only the inner race: balanced difference = (0 + 1)/2 = 50 points
    a = {(s, r): 1.0 for s in (0, 1) for r in GROUPS4}
    b = {(s, r): 1.0 if r[0] == "I" else 0.0 for s in (0, 1) for r in GROUPS4}
    res = e15.cluster_bootstrap(a, b, GROUPS4, CLUSTERS4, 400, 0)
    assert res["diff_points"] == pytest.approx(50.0)
    assert res["n_clusters"] == 8 and res["n_splits"] == 2 and res["n_records"] == 16
    assert res["eligible"] == 1 and res["min_clusters_per_group"] == 4
    assert res["ci_lo"] == pytest.approx(50.0) and res["ci_hi"] == pytest.approx(50.0)
    assert res["p_one_sided"] < 0.01
    # difference concentrated in one cluster: the interval widens and p goes up
    c = {k: (1.0 if CLUSTERS4[k[1]] == "O-A" else 0.0) for k in a}
    z = dict.fromkeys(a, 0.0)
    res2 = e15.cluster_bootstrap(c, z, GROUPS4, CLUSTERS4, 400, 0)
    assert res2["ci_lo"] < res2["diff_points"] < res2["ci_hi"]
    assert res2["p_one_sided"] > 0.05
    assert e15.cluster_bootstrap(a, {}, GROUPS4, CLUSTERS4, 10, 0) is None


def test_teste_sem_aglomerados_ou_divisoes_suficientes_nao_e_elegivel():
    """With one split, or fewer than 3 clusters in a race, all resamples come out equal and the
    one-sided p would become 1/(1+B) without measuring variability: the test gets p = 1."""
    a = {(0, r): 1.0 for r in GROUPS4}  # a single split
    b = {(0, r): 0.0 for r in GROUPS4}
    res = e15.cluster_bootstrap(a, b, GROUPS4, CLUSTERS4, 200, 0)
    assert res["diff_points"] == pytest.approx(100.0)
    assert res["eligible"] == 0 and res["p_one_sided"] == 1.0
    two = {r: c for r, c in CLUSTERS4.items() if c[-1] in "AB"}  # 2 clusters per race
    a2 = {(s, r): 1.0 for s in (0, 1) for r in two}
    b2 = {(s, r): 0.0 for s in (0, 1) for r in two}
    res2 = e15.cluster_bootstrap(a2, b2, GROUPS4, two, 200, 0)
    assert res2["eligible"] == 0 and res2["p_one_sided"] == 1.0
    res3 = e15.cluster_bootstrap(a2, b2, GROUPS4, two, 200, 0, min_clusters=2)
    assert res3["eligible"] == 1 and res3["p_one_sided"] < 0.01


def test_media_por_registro_vem_antes_de_reamostrar():
    rows = [
        {"split": 0, "record": "I0", "segment": s, "hit_local": h, "family_true": "I"}
        for s, h in ((1, 1), (2, 0))
    ] + [{"split": 0, "record": "I0", "segment": 3, "hit_local": "", "family_true": "I"}]
    assert e15.per_record(rows, "local") == {(0, "I0"): 0.5}


def test_holm():
    assert e15.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert e15.holm([0.5]) == [0.5]


def test_selecao_entre_orcamentos_usa_so_validacao():
    def out(bal: float, loss: float, tag: str) -> dict:
        info = {"dec_learned": {"val_bal_acc": bal, "val_loss": loss}}
        return {"info": info, "rows": [{"arm": "dec_learned", "budget": 0, "tag": tag}]}

    outs = {8: out(0.6, 1.0, "b8"), 16: out(0.9, 2.0, "b16"), 32: out(0.9, 1.5, "b32")}
    rows = e15.select_across_budgets(outs, [8, 16, 32])
    chosen = {r["budget"]: (r["chosen_budget"], r["tag"]) for r in rows}
    # at 8 and 16 the accuracy ties (0.9) between 16 and 32; the lower loss wins (32)
    assert chosen == {8: (32, "b32"), 16: (32, "b32"), 32: (32, "b32")}
    assert all(r["arm"] == "dec_learned_sel" for r in rows)


def test_banda_conhecida_e_a_mais_frequente_fora_do_teste():
    bands = [[100.0, 200.0], [300.0, 400.0], [100.0, 200.0], None]
    assert e15.mode_band(bands) == (100.0, 200.0)
    assert e15.mode_band([None]) is None


def test_divisoes_separam_teste_validacao_e_treino_por_registro():
    recs = [f"i{k}" for k in range(6)] + [f"o{k}" for k in range(6)] + ["s0"]
    group = {r: ("inner" if r[0] == "i" else "outer" if r[0] == "o" else "single") for r in recs}
    pcfg = {"n_splits": 3, "test_fraction": 0.34, "val_fraction": 0.25}
    splits = e15.make_splits(recs, group, pcfg, 7)
    for sp in splits:
        tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
        assert tr | va | te == set(recs)
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert "s0" in tr  # a single-record class always stays in training
        assert sp["never_tested"] == ["single"]
        for g in ("inner", "outer"):
            assert any(group[r] == g for r in te) and any(group[r] == g for r in va)
    assert e15.make_splits(recs, group, pcfg, 7) == splits


def _conv_rows(b: int, arm: str, convs: list[int]) -> list[dict]:
    base = {"dataset": "ottawa", "variant": "raw", "budget": b, "arm": arm, "seed": 0}
    return [
        {**base, "split": s, "segment": seg, "record": f"I{s}", "mask_converged": c}
        for s, c in enumerate(convs)
        for seg in (0, 1)
    ]


def test_orcamento_com_mascara_no_ruido_nao_e_avaliavel():
    rows = _conv_rows(8, "dec_learned", [1, 1, 0]) + _conv_rows(16, "dec_learned", [1, 0, 0])
    ev = e15.evaluable_budgets(rows, "ottawa", "raw", [8, 16], ["dec_learned"], 0.5)
    assert ev[8][0] is True and ev[16][0] is False
    assert "dec_learned=0.333" in ev[16][1]
    # a listed arm without diagnostics does not pass silently
    ev2 = e15.evaluable_budgets(rows, "ottawa", "raw", [8], ["dec_free_projected"], 0.5)
    assert ev2[8][0] is False
    # with no arms listed (old configs), every budget is evaluable
    assert e15.evaluable_budgets(rows, "ottawa", "raw", [8], [], 0.5)[8][0] is True


def test_hipotese_em_orcamento_nao_avaliavel_tem_p_1_e_nao_vence():
    pr = {
        "dataset": "ottawa",
        "variant": "raw",
        "alpha": 0.05,
        "rulers": ["local"],
        "min_budgets": 1,
        "min_clusters_per_group": 3,
        "min_splits": 2,
        "min_mask_converged_frac": 0.5,
        "h1": {
            "arm": "dec_learned_sel",
            "baselines": ["dec_random"],
            "convergence_arms": ["dec_learned", "dec_learned_sel"],
        },
        "h2": {
            "arm": "dec_learned",
            "baseline": "dec_free_projected",
            "convergence_arms": ["dec_learned", "dec_free_projected"],
        },
    }
    cfg = {"preregistration": pr, "analysis": {"bootstrap_resamples": 10}, "seed": 0}
    rows = (
        _conv_rows(16, "dec_learned", [1, 0, 0])
        + _conv_rows(16, "dec_learned_sel", [1, 0, 0])
        + _conv_rows(16, "dec_free_projected", [0, 0, 0])
        + _conv_rows(16, "dec_random", [0, 0, 0])
    )
    out = e15.hypotheses(rows, cfg, {}, {})
    tests = [r for r in out if "p_holm" in r]
    assert len(tests) == 2
    assert all(r["p_holm"] == 1.0 and r["significant"] == 0 and r["evaluable"] == 0 for r in tests)
    per_ruler = [r for r in out if "verdict_ruler" in r]
    assert all(r["verdict_ruler"] == 0 and r["budgets_not_evaluable"] == "16" for r in per_ruler)
    assert all(r["verdict"] == 0 for r in out if "verdict" in r)


def test_dataset_cfg_merges_overrides_without_touching_the_global():
    cfg = {"tolerance_order": 0.06, "prior": {"score_harmonics": 3, "calibration_span": 0.12}}
    d = {"overrides": {"tolerance_order": 0.025, "prior": {"calibration_span": 0.01}}}
    out = e15.dataset_cfg(cfg, d)
    assert out["tolerance_order"] == 0.025
    assert out["prior"] == {"score_harmonics": 3, "calibration_span": 0.01}
    assert cfg["prior"]["calibration_span"] == 0.12  # the global one stays intact
    assert e15.dataset_cfg(cfg, {}) == cfg


def test_kfold_by_cluster_tests_each_bearing_once_and_never_leaks():
    d = {
        "outer": ["A1", "A2", "A3", "A4", "A5"],
        "inner": ["B1", "B2", "B3", "B4"],
        "splits": {"n_folds": 4, "split_seed": 7},
    }
    recs = [f"{c}_{s}" for c in d["outer"] + d["inner"] for s in ("S1", "S2")]
    cluster_of = {r: r.split("_")[0] for r in recs}
    folds = e15.kfold_by_cluster(d, recs, cluster_of, {"val_fraction": 0.25}, seed=3)
    tested = [c for f in folds for c in f["test_clusters"]]
    assert sorted(tested) == sorted(d["outer"] + d["inner"])  # each bearing once
    for f in folds:
        tr = {cluster_of[r] for r in f["train"]}
        va = {cluster_of[r] for r in f["val"]}
        te = {cluster_of[r] for r in f["test"]}
        assert not (tr & va or tr & te or va & te)
        assert va & set(d["outer"]) and va & set(d["inner"])


def test_kfold_by_cluster_repeats_the_gate_folds():
    import yaml

    exp = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(exp))
    import paderborn_gate as pg_gate

    cfg = yaml.safe_load((exp / "config" / "paderborn.yaml").read_text(encoding="utf-8"))
    d = cfg["datasets"]["paderborn"]
    ids = [f"{c}_{s}" for c in d["outer"] + d["inner"] for s in d["settings"]]
    gate_cfg = {"outer": d["outer"], "inner": d["inner"], "splits": {"n_folds": 4}}
    gate = pg_gate.kfold_by_bearing(gate_cfg, ids, int(d["splits"]["split_seed"]))
    cluster_of = {r: r.split("_")[0] for r in ids}
    mine = e15.kfold_by_cluster(d, ids, cluster_of, cfg["protocol"], seed=1)
    assert [g["test_bearings"] for g in gate] == [m["test_clusters"] for m in mine]


# --- seeds from the labels the runs wrote -------------------------------------------------------
# The labels were renamed to English after the runs (tools/relabel_map.py). The values below were
# computed with the code before the renaming, from the Portuguese labels; the renamed code has to
# reproduce them through legacy_label, so that a rerun draws the same random numbers.
SEED = 20261113


def test_seeds_are_derived_from_the_labels_the_runs_wrote():
    lg, d = e15.legacy_label, e15.derive_seed
    assert d(SEED, "ottawa", lg("raw"), 0, 0, 4, lg("dec_learned_A"), 0) == 1947562739
    assert d(SEED, "paderborn", lg("raw"), 2, 1, 16, lg("dec_free_projected"), 1) == 1912949126
    assert d(SEED, "ottawa", lg("despiked"), 1, 0, 8, "wdcnn", 0) == 3712539223
    assert d(SEED, "H1", "local", 4, lg("dec_learned_sel"), lg("uniform_trained")) == 1723424141
    assert d(SEED, "H2", "nested", 32, lg("dec_learned"), lg("dec_free_projected")) == 528893334
    pair = (lg("dec_uniform_trained"), lg("uniform_trained"))
    assert d(SEED, "paderborn", lg("raw"), 16, *pair, "nested") == 4226970293
    assert e15.legacy_path("ottawa/raw/b16/split0/seed0/dec_learned.json") == (
        "ottawa/bruto/b16/split0/seed0/dec_aprendida.json"
    )


def test_training_seed_comes_from_the_legacy_variant_and_label():
    seen = []

    class StopError(Exception):
        pass

    def build(seed):
        seen.append(seed)
        raise StopError

    rc = SimpleNamespace(ds="ottawa", variant="raw", split={"index": 0}, seed=0, budget=4)
    ctx = SimpleNamespace(rc=rc, cfg={"seed": SEED, "training": {"lr_grid": [0.003]}})
    with pytest.raises(StopError):
        e15.best_over_lr(ctx, "dec_learned_A", build, anneal=True)
    assert seen == [1947562739]


def test_bootstrap_seeds_come_from_the_legacy_labels(monkeypatch):
    seeds = []
    monkeypatch.setattr(e15, "cluster_bootstrap", lambda *a, **k: seeds.append(a[5]))
    base = {"dataset": "ottawa", "variant": "raw", "budget": 4, "split": 0, "seed": 0}
    row = {**base, "record": "I-A-1.mat", "family_true": "inner_race", "hit_local": 1}
    pr = {
        "dataset": "ottawa",
        "variant": "raw",
        "alpha": 0.05,
        "rulers": ["local"],
        "min_budgets": 1,
        "min_clusters_per_group": 3,
        "min_splits": 2,
        "h1": {"arm": "dec_learned_sel", "baselines": ["uniform_trained"]},
        "h2": {"arm": "dec_learned_sel", "baseline": "uniform_trained"},
    }
    cfg = {"preregistration": pr, "analysis": {"bootstrap_resamples": 10}, "seed": SEED}
    arms = ("dec_learned_sel", "uniform_trained", "uniform_oracle")
    rows = [{**row, "arm": a} for a in arms]
    e15.hypotheses(rows, cfg, {}, {})
    assert seeds[0] == 1723424141  # H1, local, 4, dec_aprendida_sel, uniforme_treino
    seeds.clear()
    e15.comparisons(rows, cfg, {}, {})
    assert seeds[0] == 3388427448  # ottawa, bruto, 4, dec_aprendida_sel, oraculo_uniforme, local


def test_power_gate_seeds_come_from_the_legacy_design_and_effect():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import power_gate as pg

    lg, d = pg.legacy_label, e15.derive_seed
    assert d(SEED, "power_gate", lg("D3_record_anticonservative"), lg("perfect")) == 40437659
    assert d(SEED, "power_gate", "DK", lg("perfect")) == 2249162303
    assert d(SEED, "power_gate", "D0", lg(5)) == 2623283485
    assert d(SEED, "paderborn_gate", "D1", lg("perfect")) == 2368391636


def test_legacy_labels_invert_the_relabel_map():
    path = Path(__file__).resolve().parents[3] / "tools" / "relabel_map.py"
    if not path.exists():
        pytest.skip("tools/relabel_map.py is not part of this repository")
    spec = importlib.util.spec_from_file_location("relabel_map", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import power_gate as pg

    for new, old in {**e15.LEGACY_LABELS, **pg.LEGACY_LABELS}.items():
        assert m.TOKENS[old] == new
