"""Reconstrução esparsa sobre base de ordem, e amostragem em instantes aleatórios.

Os três testes que a tarefa exige, e um bloco de testes negativos. O que se mede aqui
é o **solver**, não o experimento: sinal sintético, verdade conhecida por construção.
"""

import numpy as np
import pytest
from snq_eval import attribute
from snq_recon import (
    interpolated_phase_rev,
    interpolated_speed_hz,
    omp,
    order_basis,
    order_of_column,
    phase_revolutions,
    random_time_indices,
    reconstruct_from_orders,
    sparse_order_spectrum,
    uniform_time_indices,
)

FS_HZ = 1000.0
SHAFT_HZ = 10.0
N_SAMPLES = 8192

# alpha reportado por sensors2025csvibration no caso de vibracao de rolamento
# (Kato & Otaka, Sensors 25(10):3167, Figura 16a). Valor da fonte, nao arbitrado.
ALPHA_FONTE = 0.0391


def _setup(orders_grid, true_orders, amps, *, alpha=0.1, seed=20260907, noise_std=0.0):
    """Sinal sintético k-esparso na base de ordem, observado em instantes aleatórios."""
    rng = np.random.default_rng(seed)
    phase_full = phase_revolutions(SHAFT_HZ, FS_HZ, N_SAMPLES)
    idx = random_time_indices(N_SAMPLES, rng, alpha=alpha)
    phase = phase_full[idx]

    a = order_basis(phase, orders_grid)
    col = order_of_column(orders_grid)
    coef_true = np.zeros(a.shape[1])
    for o, amp in zip(true_orders, amps, strict=True):
        k = int(np.flatnonzero(np.isclose(col, o) & (np.arange(col.size) % 2 == 0))[0])
        coef_true[k] = amp  # componente em cosseno
    y = a @ coef_true
    if noise_std > 0.0:
        y = y + rng.normal(0.0, noise_std, size=y.size)
    return a, y, coef_true, idx, phase


# ----------------------------------------------------------------------
# 1. recuperacao exata, sem ruido
# ----------------------------------------------------------------------
def test_omp_recupera_exatamente_sinal_esparso_sem_ruido():
    """Sem ruído e com átomos bem separados, OMP tem de acertar suporte e amplitude.

    É o teste de sanidade do solver: se ele não recupera o que ele mesmo gerou, todo
    resultado do E13 é ruído de implementação.
    """
    grid = np.arange(1.0, 21.0, 1.0)
    a, y, coef_true, _, _ = _setup(grid, [3.0, 7.0, 13.0], [1.0, 0.6, 0.3])

    sol = omp(a, y, n_nonzero=6)

    esperado = set(np.flatnonzero(coef_true).tolist())
    assert esperado.issubset(set(sol.support.tolist()))
    assert sol.relative_residual < 1e-10
    assert np.max(np.abs(sol.coefficients - coef_true)) < 1e-8


def test_espectro_de_ordem_esparso_poe_a_raia_na_ordem_certa():
    """A saída empacotada como `EnvelopeSpectrum` tem de indexar a ordem, não a coluna."""
    grid = np.arange(1.0, 21.0, 1.0)
    a, y, _, _, _ = _setup(grid, [3.0, 7.0], [1.0, 0.4])
    sol = omp(a, y, n_nonzero=4)

    spec = sparse_order_spectrum(sol.coefficients, grid)
    assert spec.freqs_hz.size == grid.size
    assert spec.freqs_hz[int(np.argmax(spec.magnitude))] == pytest.approx(3.0)
    f_pico, mag = spec.peak_near(7.0, tol_hz=0.5)
    assert f_pico == pytest.approx(7.0)
    assert mag == pytest.approx(0.4, rel=1e-6)


# ----------------------------------------------------------------------
# 2. degradacao graciosa com ruido
# ----------------------------------------------------------------------
@pytest.mark.parametrize("noise_std", [0.0, 0.01, 0.1, 0.5])
def test_omp_degrada_graciosamente_com_ruido(noise_std):
    """Com ruído crescente o suporte tem de continuar certo e o erro crescer devagar.

    "Gracioso" aqui é verificável, não retórico: o suporte verdadeiro continua dentro
    do suporte recuperado, e o erro de amplitude fica abaixo do desvio do ruído
    dividido pela raiz do número de observações vezes uma folga de 5 — que é a ordem
    de grandeza do erro de mínimos quadrados, e não um limiar escolhido a posteriori.
    """
    grid = np.arange(1.0, 21.0, 1.0)
    a, y, coef_true, _, _ = _setup(
        grid, [3.0, 7.0, 13.0], [1.0, 0.6, 0.3], noise_std=noise_std, seed=20260908
    )

    sol = omp(a, y, n_nonzero=6)
    esperado = set(np.flatnonzero(coef_true).tolist())
    assert esperado.issubset(set(sol.support.tolist())), f"suporte perdido em std={noise_std}"

    erro = float(np.max(np.abs(sol.coefficients - coef_true)))
    limite = 5.0 * noise_std / np.sqrt(a.shape[0]) + 1e-8
    assert erro < limite, f"erro {erro:.3e} acima de {limite:.3e} em std={noise_std}"


def test_erro_cresce_monotonicamente_com_o_ruido():
    """Controle do teste acima: sem esta comparação, "gracioso" não diz nada."""
    grid = np.arange(1.0, 21.0, 1.0)
    erros = []
    for std in (0.01, 0.1, 1.0):
        a, y, coef_true, _, _ = _setup(grid, [3.0, 7.0], [1.0, 0.5], noise_std=std, seed=42)
        sol = omp(a, y, n_nonzero=4)
        erros.append(float(np.linalg.norm(sol.coefficients - coef_true)))
    assert erros[0] < erros[1] < erros[2]


# ----------------------------------------------------------------------
# 3. alpha fixo: o suporte contem a raia verdadeira
# ----------------------------------------------------------------------
def test_alpha_da_fonte_recupera_a_raia_de_falha():
    """Com ``alpha`` = 0,0391 — o valor que a fonte reporta — a ordem de falha aparece.

    A raia verdadeira está em 5,4 ordens, que é a BPFI medida no Ottawa 2018 (E07).
    O que se verifica é o **suporte**: a coluna da ordem 5,4 tem de estar entre as
    escolhidas, e a maior raia do espectro de ordem tem de cair nela.
    """
    grid = np.round(np.arange(1.0, 24.0, 0.2), 3)
    ordem_falha = 5.4
    assert ordem_falha in set(grid.tolist())

    a, y, coef_true, idx, _ = _setup(
        grid, [ordem_falha, 2.0], [1.0, 0.5], alpha=ALPHA_FONTE, seed=20260909, noise_std=0.05
    )
    assert idx.size == round(ALPHA_FONTE * N_SAMPLES)

    sol = omp(a, y, n_nonzero=8)
    col = order_of_column(grid)
    ordens_no_suporte = set(np.round(col[sol.support], 3).tolist())
    assert ordem_falha in ordens_no_suporte

    spec = sparse_order_spectrum(sol.coefficients, grid)
    assert spec.freqs_hz[int(np.argmax(spec.magnitude))] == pytest.approx(ordem_falha)


def test_atribuicao_separa_medido_de_prior_na_solucao_esparsa():
    """`snq_eval.attribute` sobre a solução do OMP — o uso para o qual ele foi escrito.

    O dicionário tem mais colunas do que observações, logo o núcleo é não trivial e a
    fração medida tem de ficar estritamente abaixo de 1: parte da estimativa é escrita
    pela hipótese de esparsidade, não pelos dados. O que não pode acontecer é a
    reconstrução deixar de reproduzir o que foi medido.
    """
    # Grade fina de proposito: com 2*n_ordens > n_observacoes o sistema e
    # subdeterminado, que e a unica situacao em que a atribuicao diz alguma coisa.
    # Se o dicionario couber no numero de conversoes, `rank == n_unknowns` e a
    # fracao medida vale 1 por construcao — e o E13 declara isso no README.
    grid = np.round(np.arange(1.0, 24.0, 0.05), 3)
    a, y, _, _, _ = _setup(grid, [5.4], [1.0], alpha=ALPHA_FONTE, seed=20260910)
    assert a.shape[1] > a.shape[0], "sem sistema subdeterminado nao ha nucleo para atribuir"

    sol = omp(a, y, n_nonzero=4)
    att = attribute(a, sol.coefficients, y)

    # O posto nao passa do numero de conversoes, e fica abaixo dele quando a grade de
    # ordens e fina demais para o registro: colunas quase colineares caem sob o corte
    # de `rcond`. Sao direcoes formalmente medidas e inuteis sob ruido de 12 bits.
    assert 0 < att.rank <= a.shape[0]
    assert 0.0 < att.measured_fraction < 1.0
    assert att.data_consistency < 1e-8


# ----------------------------------------------------------------------
# agendamento de amostragem
# ----------------------------------------------------------------------
def test_instantes_aleatorios_sao_unicos_ordenados_e_no_orcamento():
    rng = np.random.default_rng(7)
    idx = random_time_indices(1000, rng, n_selected=137)
    assert idx.size == 137
    assert np.all(np.diff(idx) > 0)
    assert idx.min() >= 0 and idx.max() < 1000


def test_intervalo_minimo_entre_conversoes_e_respeitado():
    """A parede física: o núcleo SAR não converte duas vezes no mesmo instante."""
    rng = np.random.default_rng(11)
    idx = random_time_indices(1000, rng, n_selected=50, min_gap_samples=8)
    assert int(np.min(np.diff(idx))) >= 8


def test_intervalo_minimo_impossivel_e_erro_e_nao_silencio():
    rng = np.random.default_rng(13)
    with pytest.raises(ValueError, match="min_gap_samples"):
        random_time_indices(100, rng, n_selected=50, min_gap_samples=8)


def test_mesma_seed_mesmo_agendamento():
    """Determinismo: o E13 grava a seed no manifest e isso precisa significar algo."""
    a = random_time_indices(5000, np.random.default_rng(20260907), alpha=0.05)
    b = random_time_indices(5000, np.random.default_rng(20260907), alpha=0.05)
    assert np.array_equal(a, b)


def test_orcamento_uniforme_e_aleatorio_pode_ser_casado_exatamente():
    """A comparação do E13 é em amostras por segundo absolutas — tem de bater na unha."""
    uni = uniform_time_indices(10_000, factor=64)
    ale = random_time_indices(10_000, np.random.default_rng(1), n_selected=uni.size)
    assert ale.size == uni.size


def test_alpha_e_n_selected_sao_mutuamente_exclusivos():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="exatamente um"):
        random_time_indices(100, rng)
    with pytest.raises(ValueError, match="exatamente um"):
        random_time_indices(100, rng, alpha=0.1, n_selected=10)


# ----------------------------------------------------------------------
# fase e base de ordem
# ----------------------------------------------------------------------
def test_fase_de_rotacao_constante_e_linear():
    phase = phase_revolutions(20.0, 1000.0, 1001)
    assert phase[-1] == pytest.approx(20.0)  # 1 s a 20 Hz = 20 revolucoes
    assert np.allclose(np.diff(phase), 0.02)


def test_fase_interpolada_de_rotacao_esparsa_segue_a_densa():
    """O passo da fonte: rotação amostrada esparsamente, interpolada e integrada.

    Com rotação suave a fase reconstruída dos poucos pontos tem de coincidir com a
    fase densa. Se não coincidir, a base de ordem está no ângulo errado e todo o
    braço de CS mede outra coisa.
    """
    n = 4000
    t = np.arange(n) / FS_HZ
    speed = 20.0 * (1.0 + 0.2 * t / t[-1])  # rampa suave, como nos cenarios do Ottawa
    densa = phase_revolutions(speed, FS_HZ, n)

    idx = uniform_time_indices(n, factor=40)
    esparsa = interpolated_phase_rev(idx, speed[idx], FS_HZ, n)
    assert np.max(np.abs(esparsa - densa[idx])) < 1e-3


def test_base_de_ordem_tem_duas_colunas_por_ordem_e_nao_e_normalizada():
    """Não normalizar é decisão: é este operador que vai para `attribute`."""
    phase = phase_revolutions(10.0, FS_HZ, 512)
    grid = np.array([1.0, 2.5, 7.0])
    a = order_basis(phase, grid)
    assert a.shape == (512, 6)
    assert np.array_equal(order_of_column(grid), np.array([1.0, 1.0, 2.5, 2.5, 7.0, 7.0]))
    assert np.max(np.linalg.norm(a, axis=0)) > 1.5  # colunas nao unitarias


def test_omp_recusa_esparsidade_maior_que_o_sistema():
    a = order_basis(phase_revolutions(10.0, FS_HZ, 64), np.arange(1.0, 5.0))
    with pytest.raises(ValueError, match="n_nonzero"):
        omp(a, np.zeros(64), n_nonzero=100)


def test_omp_recusa_dimensao_incompativel():
    a = order_basis(phase_revolutions(10.0, FS_HZ, 64), np.arange(1.0, 5.0))
    with pytest.raises(ValueError, match="linhas"):
        omp(a, np.zeros(32), n_nonzero=2)


def test_rotacao_interpolada_coincide_com_a_densa_nos_instantes():
    """O perfil denso e a fase esparsa vêm da mesma interpolação, ou divergem."""
    n = 4000
    t = np.arange(n) / FS_HZ
    speed = 20.0 * (1.0 + 0.2 * t / t[-1])
    idx = uniform_time_indices(n, factor=40)

    densa = interpolated_speed_hz(idx, speed[idx], n)
    assert densa.size == n
    assert np.allclose(densa[idx], speed[idx])
    assert np.allclose(
        interpolated_phase_rev(idx, speed[idx], FS_HZ, n),
        phase_revolutions(densa, FS_HZ, n)[idx],
    )


def test_reconstrucao_no_tempo_reproduz_o_ajuste_do_dicionario():
    """`reconstruct_from_orders` na mesma fase tem de dar `A @ x`, átomo a átomo.

    Sem esta igualdade o braço que demodula a reconstrução estaria demodulando outra
    coisa que não a solução do OMP. A função existia exportada e sem teste até aqui.
    """
    phase = phase_revolutions(SHAFT_HZ, FS_HZ, 1024)
    grid = np.arange(0.5, 12.0, 0.5)
    coef = np.zeros(2 * grid.size)
    coef[order_of_column(grid).size - 4] = 1.7
    coef[6] = -0.9

    a = order_basis(phase, grid)
    assert np.allclose(reconstruct_from_orders(phase, grid, coef), a @ coef)

    # Fase mais fina: a reconstrucao existe fora dos instantes medidos, que e o
    # ponto de reconstruir - a fonte demodula a forma de onda densa.
    fina = phase_revolutions(SHAFT_HZ, 4.0 * FS_HZ, 4096)
    assert np.allclose(reconstruct_from_orders(fina, grid, coef), order_basis(fina, grid) @ coef)
