"""E15: o decodificador desdobrado contra o snq_recon, e os auxiliares do runner.

Os testes que cruzam pacotes moram aqui, e não em packages/snq-learn/tests, porque o snq-learn
não pode importar snq-recon (regra de dependência do CLAUDE.md).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

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
sys.modules["e15_run"] = e15  # dataclasses procura o modulo em sys.modules
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
    """Um canal, filtro ideal (um ponto de controle por bin), |y|: o envelope do snq_recon."""
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
    """Sinal AM sintético em ordem conhecida, rotação variável: mesmo pico e mesma forma."""
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
    assert abs(peak_ours - peak_ref) < 0.03  # menos de um bin do espectro do snq_recon
    assert corr > 0.98, corr
    assert ratio == pytest.approx(1.0, abs=0.1)  # mesma escala, 2/n·|DFT|


def test_mascara_uniforme_reproduz_o_braco_uniforme_do_e22():
    """Na inicialização, máscara uniforme + banda escolhida pelo kurtograma do sinal decimado:
    o decodificador do E15 dá o mesmo espectro e a mesma estimativa do pipeline do E22."""
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
# auxiliares do runner
# ----------------------------------------------------------------------
GROUP = {"a1": "g1", "a2": "g1", "a3": "g1", "b1": "g2", "b2": "g2", "c1": "g3"}
CALIB = {"a1": 5.40, "a2": 5.42, "a3": 5.50, "b1": 3.38, "b2": 3.40, "c1": 2.00}
OK = dict.fromkeys(GROUP, True)


def test_regua_aninhada_so_usa_registros_fora_do_teste():
    recs = list(GROUP)
    tr = e15.nested_truths(recs, GROUP, CALIB, OK, test={"a3", "b2"})
    # teste: mediana dos registros fora do teste do grupo; a calibracao do proprio nao entra
    assert tr["a3"]["truth"] == pytest.approx(np.median([5.40, 5.42]))
    assert tr["b2"]["truth"] == pytest.approx(3.38)
    # fora do teste: deixa-um-de-fora so entre os de fora do teste (a3 esta no teste)
    assert tr["a1"]["truth"] == pytest.approx(5.42)
    assert tr["a1"]["source"] == "aninhada"
    # classe de registro unico fora do teste: a propria calibracao, marcada
    assert tr["c1"]["truth"] == pytest.approx(2.0) and tr["c1"]["source"] == "propria_calibracao"


def test_calibracao_na_borda_fica_fora_das_medianas():
    ok = {**OK, "a2": False}
    tr = e15.nested_truths(list(GROUP), GROUP, CALIB, ok, test={"a3"})
    assert tr["a3"]["truth"] == pytest.approx(5.40)


def test_escolha_da_uniforme_usa_a_regua_por_segmento_e_empata_para_a_maior_taxa():
    # regua local diferente por segmento: a raia muda dentro do registro
    truth = {("a1", 1): 5.40, ("a1", 2): 5.30, ("a2", 1): 5.40, ("a2", 2): math.nan}
    segs = {"a1": [1, 2], "a2": [1, 2]}
    est = {}
    for r in ("a1", "a2"):
        for s in (1, 2):
            est[(r, s, 8)] = truth.get((r, s), 5.4) if np.isfinite(truth[(r, s)]) else 9.0
            est[(r, s, 9)] = est[(r, s, 8)] + 0.01  # acerta tambem: empate com 8
            est[(r, s, 10)] = 1.0  # erra
    # o segmento sem regua (a2, 2) fica fora do denominador
    assert e15.hits_at(8, ["a1", "a2"], truth, est, segs, 0.06) == (3, 3)
    assert e15.choose_factor(8, [8, 9, 10], ["a1", "a2"], truth, est, segs, 0.06) == 8
    assert e15.choose_factor(9, [8, 9, 10], ["a1", "a2"], truth, est, segs, 0.06) == 9
    # estimativa constante na raia do segmento 1 erra o segmento 2: a regua local enxerga isso
    est_const = {(r, s, 8): 5.40 for r in ("a1", "a2") for s in (1, 2)}
    assert e15.hits_at(8, ["a1"], truth, est_const, segs, 0.06) == (1, 2)
    # segmento descartado pela guarda conta como erro
    del est[("a1", 1, 8)]
    assert e15.hits_at(8, ["a1"], truth, est, segs, 0.06) == (1, 2)


def _rc(**kw):
    base = dict(
        ds="ottawa",
        variant="bruto",
        split={"index": 0, "seed": 1, "train": [], "val": [], "test": []},
        budget=16,
        seed=0,
        nested={"a1": {"truth": 5.42, "source": "aninhada"}},
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
    assert row["hit_local"] == 1 and row["hit_nested"] == 0  # a raia do segmento esta em 5,30
    assert row["family_pred"] == "inner_race" and row["family_true"] == "inner_race"
    row = e15.score_row({"group": "inner_race"}, 5.43, rc, "a1", 2)
    assert row["hit_local"] == "" and row["hit_nested"] == 1  # sem regua local nesse segmento
    healthy = e15.score_row({"group": "healthy"}, 3.41, rc, "h1", 1)
    assert healthy["family_true"] == "saudavel" and healthy["fault_like"] == 1
    assert "hit_local" not in healthy


def test_familia_pela_ordem_mais_proxima_dentro_da_janela():
    centers = {"inner_race": 5.40, "outer_race": 3.39}
    assert e15.family_of_estimate(5.43, centers, 0.12) == "inner_race"
    assert e15.family_of_estimate(3.35, centers, 0.12) == "outer_race"
    assert e15.family_of_estimate(6.78, centers, 0.12) == "nenhuma"  # 2 x BPFO nao e BPFI
    assert e15.family_of_estimate(None, centers, 0.12) == "nenhuma"
    assert e15.family_of_estimate(math.nan, centers, 0.12) == "nenhuma"


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
    v = np.full(n, 40.0)  # 32 x 40 Hz = 1280 Hz > Nyquist do envelope reduzido (1024 Hz)
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
    # a acerta tudo; b acerta so a pista interna: diferenca balanceada = (0 + 1)/2 = 50 pontos
    a = {(s, r): 1.0 for s in (0, 1) for r in GROUPS4}
    b = {(s, r): 1.0 if r[0] == "I" else 0.0 for s in (0, 1) for r in GROUPS4}
    res = e15.cluster_bootstrap(a, b, GROUPS4, CLUSTERS4, 400, 0)
    assert res["diff_points"] == pytest.approx(50.0)
    assert res["n_clusters"] == 8 and res["n_splits"] == 2 and res["n_records"] == 16
    assert res["eligible"] == 1 and res["min_clusters_per_group"] == 4
    assert res["ci_lo"] == pytest.approx(50.0) and res["ci_hi"] == pytest.approx(50.0)
    assert res["p_one_sided"] < 0.01
    # diferenca concentrada num aglomerado: o intervalo abre e o p sobe
    c = {k: (1.0 if CLUSTERS4[k[1]] == "O-A" else 0.0) for k in a}
    z = dict.fromkeys(a, 0.0)
    res2 = e15.cluster_bootstrap(c, z, GROUPS4, CLUSTERS4, 400, 0)
    assert res2["ci_lo"] < res2["diff_points"] < res2["ci_hi"]
    assert res2["p_one_sided"] > 0.05
    assert e15.cluster_bootstrap(a, {}, GROUPS4, CLUSTERS4, 10, 0) is None


def test_teste_sem_aglomerados_ou_divisoes_suficientes_nao_e_elegivel():
    """Com uma divisao, ou menos de 3 aglomerados numa pista, todas as reamostras saem iguais e
    o p unilateral viraria 1/(1+B) sem medir variabilidade: o teste recebe p = 1."""
    a = {(0, r): 1.0 for r in GROUPS4}  # uma divisao so
    b = {(0, r): 0.0 for r in GROUPS4}
    res = e15.cluster_bootstrap(a, b, GROUPS4, CLUSTERS4, 200, 0)
    assert res["diff_points"] == pytest.approx(100.0)
    assert res["eligible"] == 0 and res["p_one_sided"] == 1.0
    two = {r: c for r, c in CLUSTERS4.items() if c[-1] in "AB"}  # 2 aglomerados por pista
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
        info = {"dec_aprendida": {"val_bal_acc": bal, "val_loss": loss}}
        return {"info": info, "rows": [{"arm": "dec_aprendida", "budget": 0, "tag": tag}]}

    outs = {8: out(0.6, 1.0, "b8"), 16: out(0.9, 2.0, "b16"), 32: out(0.9, 1.5, "b32")}
    rows = e15.select_across_budgets(outs, [8, 16, 32])
    chosen = {r["budget"]: (r["chosen_budget"], r["tag"]) for r in rows}
    # em 8 e 16 empata a acuracia (0,9) entre 16 e 32; vence a menor perda (32)
    assert chosen == {8: (32, "b32"), 16: (32, "b32"), 32: (32, "b32")}
    assert all(r["arm"] == "dec_aprendida_sel" for r in rows)


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
        assert "s0" in tr  # classe de registro unico fica sempre no treino
        assert sp["never_tested"] == ["single"]
        for g in ("inner", "outer"):
            assert any(group[r] == g for r in te) and any(group[r] == g for r in va)
    assert e15.make_splits(recs, group, pcfg, 7) == splits
