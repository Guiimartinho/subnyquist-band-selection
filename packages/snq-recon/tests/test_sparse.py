"""Sparse reconstruction over an order basis, and sampling at random instants.

The three tests that the task requires, and a block of negative tests. What is measured here
is the **solver**, not the experiment: synthetic signal, ground truth known by construction.
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

# alpha reported by sensors2025csvibration for the bearing vibration case
# (Kato & Otaka, Sensors 25(10):3167, Figure 16a). Value from the source, not chosen arbitrarily.
ALPHA_FONTE = 0.0391


def _setup(orders_grid, true_orders, amps, *, alpha=0.1, seed=20260907, noise_std=0.0):
    """Synthetic k-sparse signal in the order basis, observed at random instants."""
    rng = np.random.default_rng(seed)
    phase_full = phase_revolutions(SHAFT_HZ, FS_HZ, N_SAMPLES)
    idx = random_time_indices(N_SAMPLES, rng, alpha=alpha)
    phase = phase_full[idx]

    a = order_basis(phase, orders_grid)
    col = order_of_column(orders_grid)
    coef_true = np.zeros(a.shape[1])
    for o, amp in zip(true_orders, amps, strict=True):
        k = int(np.flatnonzero(np.isclose(col, o) & (np.arange(col.size) % 2 == 0))[0])
        coef_true[k] = amp  # cosine component
    y = a @ coef_true
    if noise_std > 0.0:
        y = y + rng.normal(0.0, noise_std, size=y.size)
    return a, y, coef_true, idx, phase


# ----------------------------------------------------------------------
# 1. exact recovery, no noise
# ----------------------------------------------------------------------
def test_omp_recupera_exatamente_sinal_esparso_sem_ruido():
    """Without noise and with well-separated atoms, OMP has to get support and amplitude right.

    It is the solver's sanity test: if it does not recover what it generated itself, every
    E13 result is implementation noise.
    """
    grid = np.arange(1.0, 21.0, 1.0)
    a, y, coef_true, _, _ = _setup(grid, [3.0, 7.0, 13.0], [1.0, 0.6, 0.3])

    sol = omp(a, y, n_nonzero=6)

    esperado = set(np.flatnonzero(coef_true).tolist())
    assert esperado.issubset(set(sol.support.tolist()))
    assert sol.relative_residual < 1e-10
    assert np.max(np.abs(sol.coefficients - coef_true)) < 1e-8


def test_espectro_de_ordem_esparso_poe_a_raia_na_ordem_certa():
    """The output packaged as an `EnvelopeSpectrum` has to index the order, not the column."""
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
# 2. graceful degradation with noise
# ----------------------------------------------------------------------
@pytest.mark.parametrize("noise_std", [0.0, 0.01, 0.1, 0.5])
def test_omp_degrada_graciosamente_com_ruido(noise_std):
    """With increasing noise the support has to stay right and the error has to grow slowly.

    "Graceful" here is verifiable, not rhetorical: the true support stays inside
    the recovered support, and the amplitude error stays below the noise standard deviation
    divided by the square root of the number of observations times a margin of 5 — which is
    the order of magnitude of the least-squares error, and not a threshold chosen a posteriori.
    """
    grid = np.arange(1.0, 21.0, 1.0)
    a, y, coef_true, _, _ = _setup(
        grid, [3.0, 7.0, 13.0], [1.0, 0.6, 0.3], noise_std=noise_std, seed=20260908
    )

    sol = omp(a, y, n_nonzero=6)
    esperado = set(np.flatnonzero(coef_true).tolist())
    assert esperado.issubset(set(sol.support.tolist())), f"support lost at std={noise_std}"

    erro = float(np.max(np.abs(sol.coefficients - coef_true)))
    limite = 5.0 * noise_std / np.sqrt(a.shape[0]) + 1e-8
    assert erro < limite, f"error {erro:.3e} above {limite:.3e} at std={noise_std}"


def test_erro_cresce_monotonicamente_com_o_ruido():
    """Control for the test above: without this comparison, "graceful" says nothing."""
    grid = np.arange(1.0, 21.0, 1.0)
    erros = []
    for std in (0.01, 0.1, 1.0):
        a, y, coef_true, _, _ = _setup(grid, [3.0, 7.0], [1.0, 0.5], noise_std=std, seed=42)
        sol = omp(a, y, n_nonzero=4)
        erros.append(float(np.linalg.norm(sol.coefficients - coef_true)))
    assert erros[0] < erros[1] < erros[2]


# ----------------------------------------------------------------------
# 3. fixed alpha: the support contains the true line
# ----------------------------------------------------------------------
def test_alpha_da_fonte_recupera_a_raia_de_falha():
    """With ``alpha`` = 0.0391 — the value the source reports — the fault order shows up.

    The true line is at 5.4 orders, which is the BPFI measured on Ottawa 2018 (E07).
    What is checked is the **support**: the column of order 5.4 has to be among the
    chosen ones, and the largest line of the order spectrum has to fall on it.
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
    """`snq_eval.attribute` on the OMP solution — the use it was written for.

    The dictionary has more columns than observations, so the null space is nontrivial and the
    measured fraction has to stay strictly below 1: part of the estimate is written
    by the sparsity hypothesis, not by the data. What must not happen is the
    reconstruction failing to reproduce what was measured.
    """
    # Fine grid on purpose: with 2*n_orders > n_observations the system is
    # underdetermined, which is the only situation in which the attribution says anything.
    # If the dictionary fits within the number of conversions, `rank == n_unknowns` and the
    # measured fraction equals 1 by construction — and E13 states this in its README.
    grid = np.round(np.arange(1.0, 24.0, 0.05), 3)
    a, y, _, _, _ = _setup(grid, [5.4], [1.0], alpha=ALPHA_FONTE, seed=20260910)
    assert a.shape[1] > a.shape[0], (
        "without an underdetermined system there is no null space to attribute"
    )

    sol = omp(a, y, n_nonzero=4)
    att = attribute(a, sol.coefficients, y)

    # The rank does not exceed the number of conversions, and stays below it when the order
    # grid is too fine for the record: nearly collinear columns fall under the `rcond`
    # cutoff. They are formally measured directions, useless under 12-bit noise.
    assert 0 < att.rank <= a.shape[0]
    assert 0.0 < att.measured_fraction < 1.0
    assert att.data_consistency < 1e-8


# ----------------------------------------------------------------------
# sampling schedule
# ----------------------------------------------------------------------
def test_instantes_aleatorios_sao_unicos_ordenados_e_no_orcamento():
    rng = np.random.default_rng(7)
    idx = random_time_indices(1000, rng, n_selected=137)
    assert idx.size == 137
    assert np.all(np.diff(idx) > 0)
    assert idx.min() >= 0 and idx.max() < 1000


def test_intervalo_minimo_entre_conversoes_e_respeitado():
    """The physical wall: the SAR core does not convert twice at the same instant."""
    rng = np.random.default_rng(11)
    idx = random_time_indices(1000, rng, n_selected=50, min_gap_samples=8)
    assert int(np.min(np.diff(idx))) >= 8


def test_intervalo_minimo_impossivel_e_erro_e_nao_silencio():
    rng = np.random.default_rng(13)
    with pytest.raises(ValueError, match="min_gap_samples"):
        random_time_indices(100, rng, n_selected=50, min_gap_samples=8)


def test_mesma_seed_mesmo_agendamento():
    """Determinism: E13 records the seed in the manifest, and that has to mean something."""
    a = random_time_indices(5000, np.random.default_rng(20260907), alpha=0.05)
    b = random_time_indices(5000, np.random.default_rng(20260907), alpha=0.05)
    assert np.array_equal(a, b)


def test_orcamento_uniforme_e_aleatorio_pode_ser_casado_exatamente():
    """The E13 comparison is in absolute samples per second — it has to match exactly."""
    uni = uniform_time_indices(10_000, factor=64)
    ale = random_time_indices(10_000, np.random.default_rng(1), n_selected=uni.size)
    assert ale.size == uni.size


def test_alpha_e_n_selected_sao_mutuamente_exclusivos():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="exactly one"):
        random_time_indices(100, rng)
    with pytest.raises(ValueError, match="exactly one"):
        random_time_indices(100, rng, alpha=0.1, n_selected=10)


# ----------------------------------------------------------------------
# phase and order basis
# ----------------------------------------------------------------------
def test_fase_de_rotacao_constante_e_linear():
    phase = phase_revolutions(20.0, 1000.0, 1001)
    assert phase[-1] == pytest.approx(20.0)  # 1 s at 20 Hz = 20 revolutions
    assert np.allclose(np.diff(phase), 0.02)


def test_fase_interpolada_de_rotacao_esparsa_segue_a_densa():
    """The source's step: shaft speed sampled sparsely, interpolated and integrated.

    With a smooth speed profile the phase reconstructed from the few points has to coincide
    with the dense phase. If it does not, the order basis is at the wrong angle and the whole
    CS arm measures something else.
    """
    n = 4000
    t = np.arange(n) / FS_HZ
    speed = 20.0 * (1.0 + 0.2 * t / t[-1])  # smooth ramp, as in the Ottawa scenarios
    densa = phase_revolutions(speed, FS_HZ, n)

    idx = uniform_time_indices(n, factor=40)
    esparsa = interpolated_phase_rev(idx, speed[idx], FS_HZ, n)
    assert np.max(np.abs(esparsa - densa[idx])) < 1e-3


def test_base_de_ordem_tem_duas_colunas_por_ordem_e_nao_e_normalizada():
    """Not normalizing is a decision: this is the operator that goes to `attribute`."""
    phase = phase_revolutions(10.0, FS_HZ, 512)
    grid = np.array([1.0, 2.5, 7.0])
    a = order_basis(phase, grid)
    assert a.shape == (512, 6)
    assert np.array_equal(order_of_column(grid), np.array([1.0, 1.0, 2.5, 2.5, 7.0, 7.0]))
    assert np.max(np.linalg.norm(a, axis=0)) > 1.5  # non-unit columns


def test_omp_recusa_esparsidade_maior_que_o_sistema():
    a = order_basis(phase_revolutions(10.0, FS_HZ, 64), np.arange(1.0, 5.0))
    with pytest.raises(ValueError, match="n_nonzero"):
        omp(a, np.zeros(64), n_nonzero=100)


def test_omp_recusa_dimensao_incompativel():
    a = order_basis(phase_revolutions(10.0, FS_HZ, 64), np.arange(1.0, 5.0))
    with pytest.raises(ValueError, match="rows"):
        omp(a, np.zeros(32), n_nonzero=2)


def test_rotacao_interpolada_coincide_com_a_densa_nos_instantes():
    """The dense profile and the sparse phase come from the same interpolation, or they diverge."""
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
    """`reconstruct_from_orders` at the same phase has to give `A @ x`, atom by atom.

    Without this equality the arm that demodulates the reconstruction would be demodulating
    something other than the OMP solution. The function was exported and untested until now.
    """
    phase = phase_revolutions(SHAFT_HZ, FS_HZ, 1024)
    grid = np.arange(0.5, 12.0, 0.5)
    coef = np.zeros(2 * grid.size)
    coef[order_of_column(grid).size - 4] = 1.7
    coef[6] = -0.9

    a = order_basis(phase, grid)
    assert np.allclose(reconstruct_from_orders(phase, grid, coef), a @ coef)

    # Finer phase: the reconstruction exists outside the measured instants, which is the
    # point of reconstructing - the source demodulates the dense waveform.
    fina = phase_revolutions(SHAFT_HZ, 4.0 * FS_HZ, 4096)
    assert np.allclose(reconstruct_from_orders(fina, grid, coef), order_basis(fina, grid) @ coef)
