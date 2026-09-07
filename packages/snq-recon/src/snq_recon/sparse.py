"""Reconstrução esparsa sobre **base de ordem**, e amostragem em instantes aleatórios.

É a implementação do concorrente mais direto do projeto: Kato & Otaka,
*Sensors* 25(10):3167, 2025 (`sensors2025csvibration`, doi:10.3390/s25103167).

**A leitura corrente do método deles estava errada e foi corrigida na fonte.** O
relatório interno descrevia "matriz de medição aleatória gaussiana". A fonte diz
outra coisa, no texto da Seção 2 e da Figura 1a: *"both the rotational speed and
vibration signals are first randomly sampled. The rotational speed data are then
interpolated and integrated to obtain the rotational phase. Using this rotational
phase information, the order basis is constructed"*. A menção a matriz gaussiana
aparece como fundamentação genérica de RIP, não como o operador de aquisição usado.
O que eles fazem é **seleção aleatória de instantes de amostragem**. No caso de
vibração de rolamento a fonte reporta ``α = 0,0391``.

A distinção importa para este projeto e não é cosmética: multiplicar o sinal por uma
matriz gaussiana densa exige adquirir o sinal inteiro primeiro — não economiza
conversão nenhuma. Selecionar instantes economiza conversões, e é exatamente o que
DPPI + TIMER → SAADC realiza em hardware no nRF54L15, com a CPU dormindo.

As peças, e nada além disso:

``random_time_indices``   quais instantes converter;
``interpolated_speed_hz`` rotação interpolada sobre a grade original;
``order_basis``           dicionário cos/sin nas ordens da rotação, avaliado nesses instantes;
``omp``                   Orthogonal Matching Pursuit sobre esse dicionário;
``reconstruct_from_orders`` a forma de onda de volta no tempo, que a fonte **demodula**.

O último passo não é opcional se o objetivo for reproduzir a fonte. No caso
experimental de rolamento ela reconstrói e **depois** demodula: *"the envelope
processing results of the reconstructed waveform in the time domain and its Fourier
transform are shown in Figure 17b […] envelope processing of the CS results using the
order basis can detect outer ring failures"*. Parar no espectro de ordem da portadora
reconstruída é uma versão amputada do método deles, não o método deles.

O que **não** está implementado aqui, e por que: FISTA (l1) e CRT robusto. Nenhum dos
dois é necessário para a comparação controlada do E13, e anunciar função inexistente
foi um defeito real deste pacote — ver o docstring de ``snq_recon/__init__.py``.

Limite declarado: nenhuma restrição de intervalo mínimo entre conversões é imposta por
padrão. O parâmetro ``min_gap_samples`` existe porque o núcleo SAR tem tempo de
conversão finito e um agendamento com dois disparos colados não é realizável — mas o
valor realizável no nRF54L15 é medida de bancada que ainda não foi feita.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snq_recon.envelope import EnvelopeSpectrum
from snq_recon.highres import phase_revolutions

__all__ = [
    "SparseSolution",
    "interpolated_phase_rev",
    "interpolated_speed_hz",
    "omp",
    "order_basis",
    "order_of_column",
    "random_time_indices",
    "reconstruct_from_orders",
    "sparse_order_spectrum",
    "uniform_time_indices",
]


# ----------------------------------------------------------------------
# agendamento de amostragem
# ----------------------------------------------------------------------
def uniform_time_indices(n_samples: int, factor: int) -> NDArray[np.int64]:
    """Instantes da decimação uniforme: ``0, factor, 2·factor, …``.

    É o controle do E13. Separado de ``snq_data.undersample`` de propósito: aqui o
    que interessa é o **conjunto de instantes**, para que os três braços possam ser
    comparados pelo número de conversões e não pelo sinal resultante.

    >>> uniform_time_indices(10, 3).tolist()
    [0, 3, 6, 9]
    """
    if n_samples < 1:
        raise ValueError("n_samples >= 1")
    if factor < 1:
        raise ValueError("factor >= 1")
    return np.arange(0, int(n_samples), int(factor), dtype=np.int64)


def random_time_indices(
    n_samples: int,
    rng: np.random.Generator,
    *,
    alpha: float | None = None,
    n_selected: int | None = None,
    min_gap_samples: int = 1,
) -> NDArray[np.int64]:
    """Instantes escolhidos ao acaso, sem repetição, em ordem crescente.

    Exatamente um de ``alpha`` (fração da grade original) ou ``n_selected`` (número
    absoluto de conversões). O segundo existe para casar o orçamento de conversões
    com o do braço uniforme **exatamente**, e não a menos de um arredondamento — a
    comparação do E13 é em amostras por segundo absolutas.

    ``min_gap_samples`` garante espaçamento mínimo entre conversões consecutivas. A
    construção sorteia num intervalo encolhido e re-expande, o que é uniforme sobre
    as configurações válidas; ``min_gap_samples=1`` recai no sorteio simples.
    """
    n = int(n_samples)
    if n < 1:
        raise ValueError("n_samples >= 1")
    if (alpha is None) == (n_selected is None):
        raise ValueError("informe exatamente um de alpha ou n_selected")

    m = int(round(float(alpha) * n)) if alpha is not None else int(n_selected)  # type: ignore[arg-type]
    if not 1 <= m <= n:
        raise ValueError(f"numero de conversoes ({m}) precisa estar em [1, {n}]")

    gap = int(min_gap_samples)
    if gap < 1:
        raise ValueError("min_gap_samples >= 1")

    available = n - (m - 1) * (gap - 1)
    if available < m:
        raise ValueError(
            f"min_gap_samples={gap} nao cabe: {m} conversoes em {n} amostras exigem "
            f"pelo menos {(m - 1) * gap + 1} amostras de grade"
        )
    base = np.sort(rng.choice(available, size=m, replace=False))
    return (base + np.arange(m, dtype=np.int64) * (gap - 1)).astype(np.int64)


def interpolated_speed_hz(
    indices: ArrayLike,
    speed_hz_at_indices: ArrayLike,
    n_samples: int,
) -> NDArray[np.float64]:
    """Rotação em hertz interpolada sobre a **grade original**, a partir dos instantes.

    É o primeiro passo do método da fonte — *"the rotational speed data are then
    interpolated and integrated to obtain the rotational phase"*.

    Existe separada de `interpolated_phase_rev` porque demodular o sinal reconstruído
    (o passo da Figura 17b da fonte) exige o perfil **denso** de rotação, e não só a
    fase nos instantes convertidos. Sem ela, quem precisasse do perfil denso teria de
    reimplementar a interpolação — e duas cópias divergem.
    """
    idx = np.asarray(indices, dtype=np.int64).ravel()
    speed = np.asarray(speed_hz_at_indices, dtype=float).ravel()
    if idx.size != speed.size:
        raise ValueError(f"{idx.size} instantes contra {speed.size} valores de rotacao")
    if idx.size < 2:
        raise ValueError("interpolacao de rotacao exige pelo menos 2 instantes")
    grid = np.arange(int(n_samples), dtype=float)
    return np.interp(grid, idx.astype(float), speed)


def interpolated_phase_rev(
    indices: ArrayLike,
    speed_hz_at_indices: ArrayLike,
    fs_hz: float,
    n_samples: int,
) -> NDArray[np.float64]:
    """Fase de rotação, em **revoluções**, a partir da rotação medida só nos instantes.

    Reproduz o passo da fonte: interpola a rotação amostrada esparsamente sobre a
    grade original e integra para obter a fase. Devolve a fase **nos instantes de
    ``indices``**, que é o que a base de ordem precisa.

    A rotação, neste projeto, não passa pelo ADC: vem de contagem de pulsos numa
    entrada digital (TIMER/GPIOTE). O orçamento de conversões contabilizado é o da
    vibração. Ainda assim ela é subamostrada aqui, para que o braço reproduza o
    método da fonte e não uma versão facilitada dele.
    """
    idx = np.asarray(indices, dtype=np.int64).ravel()
    dense = interpolated_speed_hz(idx, speed_hz_at_indices, int(n_samples))
    return phase_revolutions(dense, fs_hz, int(n_samples))[idx]


# ----------------------------------------------------------------------
# dicionario de ordem
# ----------------------------------------------------------------------
def order_basis(phase_rev: ArrayLike, orders: ArrayLike) -> NDArray[np.float64]:
    """Dicionário ``(n_obs, 2·n_ordens)`` de cosseno/seno nas ordens da rotação.

    Coluna ``2k`` é ``cos(2π·ordem_k·θ)`` e coluna ``2k+1`` é ``sin(2π·ordem_k·θ)``,
    com ``θ`` a fase em revoluções. É a base de ordem da fonte: sob rotação variável
    ela é esparsa onde a base de Fourier em hertz não é, porque a componente de falha
    fica numa ordem fixa enquanto a frequência em hertz se move.

    **Não é normalizado.** É o operador de aquisição honesto, e é ele que vai para
    ``snq_eval.attribute``. A normalização de colunas que o OMP precisa é feita
    dentro do ``omp`` e desfeita nos coeficientes devolvidos.
    """
    theta = np.asarray(phase_rev, dtype=float).ravel()
    ords = np.asarray(orders, dtype=float).ravel()
    if ords.size == 0:
        raise ValueError("grade de ordens vazia")

    arg = 2.0 * np.pi * np.outer(theta, ords)
    a = np.empty((theta.size, 2 * ords.size), dtype=float)
    a[:, 0::2] = np.cos(arg)
    a[:, 1::2] = np.sin(arg)
    return a


def order_of_column(orders: ArrayLike) -> NDArray[np.float64]:
    """Ordem associada a cada coluna de ``order_basis`` — cada ordem ocupa duas."""
    return np.repeat(np.asarray(orders, dtype=float).ravel(), 2)


# ----------------------------------------------------------------------
# OMP
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SparseSolution:
    """Solução esparsa, sempre carregando o resíduo — sem ele não há como auditar."""

    coefficients: NDArray[np.float64]
    """Vetor de comprimento ``n_colunas``, zero fora do suporte, na escala do
    dicionário **original** (não normalizado)."""

    support: NDArray[np.int64]
    """Índices de coluna escolhidos, na ordem em que o OMP os escolheu."""

    residual_norm: float
    """``||y - A·x||`` absoluto."""

    observation_norm: float
    """``||y||``, para formar o resíduo relativo sem recomputar."""

    n_iterations: int

    @property
    def relative_residual(self) -> float:
        return (
            self.residual_norm / self.observation_norm
            if self.observation_norm > 0.0
            else self.residual_norm
        )


def omp(
    a: ArrayLike,
    y: ArrayLike,
    *,
    n_nonzero: int,
    tol: float = 0.0,
    normalize: bool = True,
) -> SparseSolution:
    """Orthogonal Matching Pursuit.

    A cada passo escolhe a coluna mais correlacionada com o resíduo, resolve mínimos
    quadrados **sobre todo o suporte acumulado** e recomeça. O "orthogonal" é esse
    re-ajuste: sem ele o resíduo não fica ortogonal ao suporte e a mesma coluna é
    escolhida de novo.

    ``n_nonzero`` é o número máximo de colunas. ``tol`` para antes se o resíduo
    relativo cair abaixo dele; ``tol=0`` roda sempre as ``n_nonzero`` iterações.

    ``normalize`` escala as colunas para norma unitária **só internamente**, para que
    a correlação compare direções e não amplitudes. Os coeficientes devolvidos estão
    na escala de ``a``, de modo que ``a @ solution.coefficients`` reproduz o ajuste.

    **Custo, e ele importa para quem for citar tempo de execução.** O ajuste é
    re-resolvido do zero a cada iteração, sem atualização incremental de QR ou de
    Cholesky. É a formulação simples e correta, e cresce depressa com ``n_nonzero``:
    medido em ``(7813, 2632)`` pelo `sensitivity_atoms.py` do E13, 40 átomos levaram
    0,5 s, 120 levaram 2,9 s e 400 levaram 48 s — cerca de ``k^2``, não o cubo teórico,
    porque o LAPACK por trás de ``lstsq`` não é o gargalo que a contagem ingênua de
    flops sugere.

    Estas medidas são frágeis e a fragilidade está apurada: duas execuções consecutivas
    da mesma varredura, na mesma máquina, deram 28,1 s e 47,9 s no ponto de 400 átomos,
    com resíduo e estimativa idênticos ao último dígito. Uma medida mais antiga do mesmo
    ponto registrou 743 s; ela **não foi reproduzida** e não deve ser citada. Tempo de
    parede de máquina compartilhada é o número menos confiável deste repositório — as
    contagens (átomos, colunas, conversões) não têm esse problema.

    Se o gargalo de algum experimento passar a ser este laço, a correção é atualização
    incremental do QR — não é o caso do E13, que roda com 40 átomos.
    """
    op = np.atleast_2d(np.asarray(a, dtype=float))
    obs = np.asarray(y, dtype=float).ravel()
    if op.shape[0] != obs.size:
        raise ValueError(f"a tem {op.shape[0]} linhas mas y tem {obs.size} elementos")

    k_max = int(n_nonzero)
    if not 1 <= k_max <= min(op.shape):
        raise ValueError(f"n_nonzero precisa estar em [1, {min(op.shape)}], recebeu {k_max}")

    norms = np.linalg.norm(op, axis=0)
    scale = np.where(norms > 0.0, norms, 1.0) if normalize else np.ones(op.shape[1])
    dic = op / scale if normalize else op

    obs_norm = float(np.linalg.norm(obs))
    resid = obs.copy()
    support: list[int] = []
    coef_support = np.zeros(0)

    for _ in range(k_max):
        corr = dic.T @ resid
        if support:
            corr[support] = 0.0
        col = int(np.argmax(np.abs(corr)))
        if corr[col] == 0.0:
            break  # nada mais correlacionado: dicionario esgotado
        support.append(col)
        sub = dic[:, support]
        coef_support = np.linalg.lstsq(sub, obs, rcond=None)[0]
        resid = obs - sub @ coef_support
        if tol > 0.0 and obs_norm > 0.0 and float(np.linalg.norm(resid)) <= tol * obs_norm:
            break

    coefficients = np.zeros(op.shape[1], dtype=float)
    if support:
        coefficients[support] = coef_support / scale[support]
    return SparseSolution(
        coefficients=coefficients,
        support=np.asarray(support, dtype=np.int64),
        residual_norm=float(np.linalg.norm(resid)),
        observation_norm=obs_norm,
        n_iterations=len(support),
    )


# ----------------------------------------------------------------------
# saida em forma comparavel com o braco classico
# ----------------------------------------------------------------------
def sparse_order_spectrum(coefficients: ArrayLike, orders: ArrayLike) -> EnvelopeSpectrum:
    """Amplitude por ordem, empacotada como `EnvelopeSpectrum` **em ordens**.

    ``magnitude[k] = hypot(c_cos, c_sin)`` da ordem ``k``. A reutilização do tipo é a
    mesma que `envelope_order_spectrum` faz: `score_fault`, `estimate_from_candidates`
    e `estimate_without_prior` são indiferentes à unidade do eixo, e assim o braço de
    CS é julgado pelo **mesmo estimador** que o braço clássico. Sem isso a comparação
    mediria diferença de estimador, não diferença de aquisição.

    ``fs_hz`` é preenchido com ``2·ordem_máxima``, o análogo de Nyquist no eixo de
    ordens; o campo é informativo, e quem lê o teto do eixo lê ``freqs_hz[-1]``.
    """
    coef = np.asarray(coefficients, dtype=float).ravel()
    ords = np.asarray(orders, dtype=float).ravel()
    if coef.size != 2 * ords.size:
        raise ValueError(f"{coef.size} coeficientes nao formam pares sobre {ords.size} ordens")
    mag = np.hypot(coef[0::2], coef[1::2])
    return EnvelopeSpectrum(freqs_hz=ords, magnitude=mag, fs_hz=2.0 * float(ords[-1]), band_hz=None)


def reconstruct_from_orders(
    phase_rev: ArrayLike, orders: ArrayLike, coefficients: ArrayLike
) -> NDArray[np.float64]:
    """Sinal reconstruído numa fase arbitrária, a partir dos coeficientes esparsos.

    Percorre apenas o suporte, um átomo por vez: reconstruir numa grade densa de
    centenas de milhares de pontos com o dicionário inteiro em memória não cabe, e a
    solução é esparsa justamente para não precisar disso.
    """
    theta = np.asarray(phase_rev, dtype=float).ravel()
    ords = np.asarray(orders, dtype=float).ravel()
    coef = np.asarray(coefficients, dtype=float).ravel()
    if coef.size != 2 * ords.size:
        raise ValueError(f"{coef.size} coeficientes nao formam pares sobre {ords.size} ordens")

    out = np.zeros(theta.size, dtype=float)
    col_order = order_of_column(ords)
    for col in np.flatnonzero(coef):
        arg = 2.0 * np.pi * col_order[col] * theta
        out += coef[col] * (np.sin(arg) if col % 2 else np.cos(arg))
    return out
