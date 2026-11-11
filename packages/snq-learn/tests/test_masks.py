"""Conjunto viável, projeção exata e máscaras do E15."""

import itertools

import numpy as np
import pytest
import torch
from snq_learn import (
    BlockAdaptiveMask,
    FeasibleSet,
    FixedMask,
    GumbelTopKMask,
    LoupeMask,
    alias_weights,
    band_support_cells,
    circular_gaps,
    designed_positions,
    fold_project,
    noise_amplification,
    project_to_feasible,
    random_feasible_positions,
    shrink_band_to_support,
    spacing_penalty,
    support_designed_positions,
    timer_schedule,
    uniform_positions,
)


def _brute_force(w: np.ndarray, fs: FeasibleSet) -> float:
    combos = itertools.combinations(range(fs.period_ticks), fs.active_per_period)
    return max(float(w[list(c)].sum()) for c in combos if fs.is_feasible(c))


@pytest.mark.parametrize(
    ("period", "k", "gap"), [(8, 2, 2), (10, 3, 2), (12, 3, 3), (12, 4, 2), (9, 3, 3), (7, 1, 1)]
)
def test_projecao_e_exata_contra_forca_bruta(period, k, gap):
    fs = FeasibleSet(period, k, gap)
    rng = np.random.default_rng(period * 100 + k * 10 + gap)
    for _ in range(25):
        w = rng.normal(size=period)
        p = project_to_feasible(w, fs)
        assert w[p].sum() == pytest.approx(_brute_force(w, fs), abs=1e-9)


@pytest.mark.parametrize(("period", "k", "gap"), [(16, 4, 2), (64, 4, 2), (128, 4, 3), (30, 6, 5)])
def test_projecao_respeita_k_e_espacamento_circular(period, k, gap):
    fs = FeasibleSet(period, k, gap)
    rng = np.random.default_rng(0)
    for _ in range(20):
        # escores que favorecem posicoes coladas: o espacamento precisa ser imposto
        w = rng.normal(size=period)
        w[:k] += 5.0
        p = project_to_feasible(w, fs)
        assert p.size == k and np.unique(p).size == k
        assert circular_gaps(p, period).min() >= gap
        assert fs.is_feasible(p)


def test_espacamento_inclui_a_volta_do_periodo():
    fs = FeasibleSet(8, 2, 3)
    # 0 e 7 estao a 1 tick pela volta: inviavel mesmo com 7 ticks no sentido direto
    assert not fs.is_feasible([0, 7])
    assert fs.is_feasible([0, 4])
    w = np.zeros(8)
    w[[0, 7]] = 10.0
    p = project_to_feasible(w, fs)
    assert circular_gaps(p, 8).min() >= 3


def test_conjunto_vazio_e_erro():
    with pytest.raises(ValueError, match="vazio"):
        FeasibleSet(8, 4, 3)
    with pytest.raises(ValueError):
        FeasibleSet(4, 5, 1)


def test_for_budget_casa_a_taxa_do_orcamento():
    fs = FeasibleSet.for_budget(16, 4, 2)
    assert fs.period_ticks == 64
    assert fs.rate_fraction == pytest.approx(1 / 16)


def test_padroes_fixos_sao_viaveis():
    rng = np.random.default_rng(1)
    for period, k, gap in [(16, 4, 2), (32, 4, 2), (8, 4, 1), (24, 3, 4)]:
        fs = FeasibleSet(period, k, gap)
        u = uniform_positions(fs)
        assert u.tolist() == list(range(0, period, period // k))
        assert fs.is_feasible(u)
        for _ in range(50):
            assert fs.is_feasible(random_feasible_positions(fs, rng))
        assert fs.is_feasible(designed_positions(fs))


def test_sorteio_aleatorio_e_uniforme_sobre_os_padroes_viaveis():
    fs = FeasibleSet(7, 2, 2)
    feasible = [c for c in itertools.combinations(range(7), 2) if fs.is_feasible(c)]
    rng = np.random.default_rng(2)
    n = 14000
    counts = dict.fromkeys(feasible, 0)
    for _ in range(n):
        counts[tuple(random_feasible_positions(fs, rng).tolist())] += 1
    expected = n / len(feasible)
    assert all(abs(c - expected) < 5 * np.sqrt(expected) for c in counts.values())


def test_projetado_espalha_o_alias_que_o_uniforme_concentra():
    fs = FeasibleSet(16, 4, 2)
    u, d = uniform_positions(fs), designed_positions(fs)
    assert alias_weights(u, 16)[0] == pytest.approx(4 / 16)
    assert alias_weights(u, 16)[1:].max() == pytest.approx(4 / 16)  # replicas de peso cheio
    assert alias_weights(d, 16)[1:].max() < alias_weights(u, 16)[1:].max()


def test_agenda_do_timer_gira_o_padrao_e_confere_contagens():
    s = timer_schedule([3, 7, 12, 14], 16, 200_000.0, 16e6)
    assert s["rotation_ticks"] == 3
    assert s["compare_ticks"] == [4, 9, 11, 16]
    assert s["counts_are_integer"]  # 80 contagens por tick a 200 kSPS
    assert s["compare_registers"] == 4
    assert not timer_schedule([0, 1], 4, 48_000.0, 16e6)["counts_are_integer"]


def test_penalidade_de_espacamento():
    soft = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    assert float(spacing_penalty(soft, 1)) == 0.0
    assert float(spacing_penalty(soft, 2)) == pytest.approx(1.0)  # o par (0, 1)
    assert float(spacing_penalty(soft, 5)) > float(spacing_penalty(soft, 2))


def test_mascara_aprendida_tem_k_por_periodo_e_avalia_pela_projecao():
    torch.manual_seed(0)
    fs = FeasibleSet(16, 4, 2)
    mask = GumbelTopKMask(fs, tau=1.0, seed=3)
    x = torch.randn(5, 64)
    mask.train()
    m = mask(x)
    assert m.shape == x.shape
    per_period = m.detach().reshape(5, 4, 16).sum(-1)
    assert torch.allclose(per_period, torch.full_like(per_period, 4.0))
    mask.eval()
    m_eval = mask(x)[0, :16].numpy()
    expected = project_to_feasible(mask.logits.detach().numpy(), fs)
    assert np.flatnonzero(m_eval).tolist() == expected.tolist()
    pos = mask.freeze()
    assert not mask.logits.requires_grad
    mask.train()  # congelada, treino e avaliacao usam a mesma mascara projetada
    assert np.flatnonzero(mask(x)[0, :16].numpy()).tolist() == pos.tolist()


def test_mascara_fixa_repete_o_padrao():
    m = FixedMask([0, 3], 4)(torch.zeros(2, 12))
    assert m[0].tolist() == [1, 0, 0, 1] * 3
    with pytest.raises(ValueError, match="multiplo"):
        FixedMask([0], 4)(torch.zeros(1, 10))


def test_loupe_casa_a_taxa_e_projeta_no_conjunto_viavel():
    torch.manual_seed(0)
    loupe = LoupeMask(32, 0.25, seed=4)
    assert float(loupe.probabilities().detach().mean()) == pytest.approx(0.25, abs=1e-5)
    loupe.eval()
    loupe(torch.zeros(1, 32 * 400))
    assert loupe.last_realized_rate == pytest.approx(0.25, abs=0.02)
    fs = FeasibleSet(32, 8, 2)
    assert fs.is_feasible(loupe.project(fs))


def test_loupe_recupera_a_mascara_de_baixa_frequencia_num_brinquedo():
    """Validação da reimplementação (README do E15): o LOUPE tem de concentrar a amostragem
    nas frequências baixas quando o sinal só tem energia nelas, como em MRI no espaço k."""
    torch.manual_seed(0)
    n, n_bins = 64, 33
    loupe = LoupeMask(n_bins, 0.25, seed=5)
    opt = torch.optim.Adam(loupe.parameters(), lr=0.05)
    decay = torch.exp(-torch.arange(n_bins) / 3.0)
    for _ in range(300):
        spec = torch.complex(torch.randn(64, n_bins), torch.randn(64, n_bins)) * decay
        x = torch.fft.irfft(spec, n=n)
        loupe.train()
        m = loupe(torch.zeros(64, n_bins))
        x_hat = torch.fft.irfft(torch.fft.rfft(x) * m, n=n)
        loss = ((x_hat - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = loupe.probabilities().detach()
    assert float(p[:4].mean()) > 0.9
    assert float(p[16:].mean()) < 0.1


def test_adaptativa_viavel_por_bloco_e_nas_emendas():
    torch.manual_seed(0)
    fs = FeasibleSet(16, 4, 3)
    adps = BlockAdaptiveMask(fs, n_blocks=4, n_features=8, hidden=8, seed=6)
    with torch.no_grad():  # politica nao nula, para os blocos diferirem
        for p in adps.policy.parameters():
            p.normal_(0.0, 1.0)
    x = torch.randn(2, 16 * 40)
    adps.train()
    m = adps(x)
    per_period = m.detach().reshape(2, 40, 16).sum(-1)
    assert torch.allclose(per_period, torch.full_like(per_period, 4.0))
    adps.eval()
    m = adps(x)
    for row in adps.last_block_positions:
        assert len(row) == 4 and all(fs.is_feasible(pos) for pos in row)
    for b in range(2):
        idx = np.flatnonzero(m[b].numpy() > 0.5)
        assert np.diff(idx).min() >= 3  # inclusive nas emendas entre blocos


# ----------------------------------------------------------------------
# revisao de 2026-11-11: mascara livre, suporte conhecido, diagnosticos
# ----------------------------------------------------------------------
def test_dobra_e_projeta_a_mascara_livre_exatamente():
    fs = FeasibleSet(8, 2, 2)
    w = np.zeros(32)
    w[[0, 8, 16, 24]] = 5.0  # posicao 0 do periodo em todas as 4 copias
    w[[1, 9, 25]] = 4.0  # posicao 1 em 3 copias, mas colada na 0 (g = 2)
    w[3] = 3.0  # posicao 3 em 1 copia
    assert fold_project(w, 8, fs).tolist() == [0, 3]
    # e a projecao euclidiana exata: confere contra forca bruta sobre o conjunto viavel
    rng = np.random.default_rng(1)
    combos = [c for c in itertools.combinations(range(8), 2) if fs.is_feasible(c)]

    def dist(hard: np.ndarray, c) -> float:
        return float(((hard - np.tile(np.isin(np.arange(8), c), 3)) ** 2).sum())

    for _ in range(20):
        w = rng.normal(size=24)
        p = fold_project(w, 6, fs)
        hard = np.zeros(24)
        hard[np.argsort(-w, kind="stable")[:6]] = 1.0
        assert dist(hard, p) == pytest.approx(min(dist(hard, c) for c in combos))


def test_suporte_conhecido_celulas_e_desenho_contra_forca_bruta():
    # banda de uma celula e o espelho: 2 celulas ativas
    sup = band_support_cells((21_937.5, 25_000.0), 200_000.0, 16)
    assert [s for _w, s in sup] == [(1,), (14,)]
    # banda e espelho cobrem, cada um, 1/4 da celula de 12,5 kHz, em deslocamentos disjuntos;
    # deslocamento sem celula ativa nao entra no peso
    assert [w for w, _s in sup] == pytest.approx([0.25, 0.25])
    fs = FeasibleSet(12, 3, 2)
    sup = [(0.5, (1, 10)), (0.5, (1, 2, 10))]
    best = min(
        noise_amplification(c, 12, sup)
        for c in itertools.combinations(range(12), 3)
        if fs.is_feasible(c)
    )
    d = support_designed_positions(fs, sup)
    assert fs.is_feasible(d)
    assert noise_amplification(d, 12, sup) == pytest.approx(best)


def test_suporte_que_atravessa_a_zona_do_uniforme_deixa_o_uniforme_singular():
    """O caso em que o desenho importa: banda que cruza a borda de uma zona de Nyquist da taxa
    uniforme. O uniforme sobrepõe réplicas (``A_S`` singular) e o desenhado reconstrói."""
    fs = FeasibleSet.for_budget(16, 4, 2)
    band = shrink_band_to_support((12_750.0, 25_000.0), 200_000.0, fs.period_ticks, 4)
    sup = band_support_cells(band, 200_000.0, fs.period_ticks)
    assert max(len(s) for _w, s in sup) <= 4
    assert noise_amplification(uniform_positions(fs), fs.period_ticks, sup) == float("inf")
    d = support_designed_positions(fs, sup)
    assert np.isfinite(noise_amplification(d, fs.period_ticks, sup))
    with pytest.raises(ValueError, match="encolha"):
        support_designed_positions(fs, [(1.0, (1, 2, 3, 4, 5))])


def test_diagnosticos_da_mascara_e_semente_que_reproduz_o_inicio():
    fs = FeasibleSet(16, 4, 2)
    m = GumbelTopKMask(fs, seed=7)
    d0 = m.diagnostics()
    assert d0["moved"] is False and d0["positions_initial"] == d0["positions_final"]
    assert d0["margin_k"] < d0["gumbel_std"]  # iniciada no ruido
    assert GumbelTopKMask(fs, seed=7).diagnostics()["positions_initial"] == d0["positions_initial"]
    with torch.no_grad():  # empurra quatro logits viaveis para longe do resto
        m.logits[[0, 4, 9, 13]] += 10.0
    d1 = m.diagnostics()
    assert d1["positions_final"] == [0, 4, 9, 13] and d1["margin_k"] > d1["gumbel_std"]
    assert d1["moved"] == (d1["positions_final"] != d0["positions_initial"])


def test_auditoria_conta_mascaras_duras_que_violam_o_espacamento_alvo():
    torch.manual_seed(0)
    free = GumbelTopKMask(FeasibleSet(32, 16, 1), seed=1, audit_gap=2)  # metade das posicoes
    free.train()
    free(torch.zeros(10, 64))
    diag = free.diagnostics()
    assert diag["audit_masks_seen"] == 10
    assert diag["audit_violating_fraction"] == pytest.approx(1.0)  # 16 de 32 sem g: sempre cola
    constrained = GumbelTopKMask(FeasibleSet(32, 4, 2), seed=1)
    constrained.train()
    constrained(torch.zeros(10, 64))
    assert constrained.diagnostics()["audit_gap"] == 2


def test_loupe_treina_com_mascara_dura_na_taxa_do_orcamento():
    torch.manual_seed(0)
    loupe = LoupeMask(64, 1 / 16, seed=2)
    loupe.train()
    m = loupe(torch.zeros(8, 64 * 200))
    vals = torch.unique(torch.round(m.detach() * 1e4) / 1e4)
    assert set(vals.tolist()) <= {0.0, 1.0}  # forward duro (straight-through)
    assert loupe.last_train_rate == pytest.approx(1 / 16, abs=0.01)
