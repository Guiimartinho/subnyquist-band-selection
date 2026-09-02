"""Atribuição medição vs. prior — a métrica que separa recuperação de alucinação.

Regra do projeto: **toda reconstrução reporta esta decomposição**. Um resultado
bonito sem ela é alucinação com boa aparência.

Dado um operador de aquisição linear ``A`` e uma estimativa ``x_hat``, decompõe::

    x_hat = A⁺A·x_hat        (espaço-imagem: determinado pelos dados)
          + (I - A⁺A)·x_hat  (núcleo: escolhido pelo prior, não medido)

A componente no núcleo é invisível às observações: qualquer valor ali produz
exatamente os mesmos dados. Ela pode ser plausível, útil e correta — mas **não é
medida**, e reportá-la como recuperação é erro metodológico.

Base conceitual: decomposição range/null-space, usada em reconstrução com prior
aprendido (deep null-space learning; DDNM). Aqui ela é usada como *auditoria*, e
não como método de reconstrução.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Attribution:
    """Resultado da auditoria de uma reconstrução."""

    measured_energy: float
    """Energia de ``x_hat`` no espaço-imagem de ``Aᴴ`` — vem dos dados."""

    prior_energy: float
    """Energia de ``x_hat`` no núcleo de ``A`` — vem do prior."""

    data_consistency: float
    """``||A·x_hat - y|| / ||y||``. Grande = a reconstrução nem reproduz o medido."""

    rank: int
    n_unknowns: int

    @property
    def measured_fraction(self) -> float:
        """Fração da energia da estimativa que é efetivamente medida, em [0, 1].

        É o número que vai no paper. Perto de 1: recuperação. Perto de 0: o prior
        escreveu quase tudo.
        """
        total = self.measured_energy + self.prior_energy
        return float(self.measured_energy / total) if total > 0.0 else 0.0

    @property
    def null_fraction(self) -> float:
        return 1.0 - self.measured_fraction

    def summary(self) -> str:
        return (
            f"medido {100 * self.measured_fraction:.1f}% | "
            f"prior {100 * self.null_fraction:.1f}% | "
            f"consistencia {self.data_consistency:.2e} | "
            f"posto {self.rank}/{self.n_unknowns}"
        )


def attribute(
    a: NDArray[np.complex128] | NDArray[np.float64],
    x_hat: NDArray[np.complex128] | NDArray[np.float64],
    y: NDArray[np.complex128] | NDArray[np.float64],
    rcond: float = 1e-10,
) -> Attribution:
    """Decompõe ``x_hat`` em componente medida e componente vinda do prior.

    ``a``     operador de aquisição, forma ``(n_obs, n_unknowns)``
    ``x_hat`` estimativa, forma ``(n_unknowns,)``
    ``y``     observações, forma ``(n_obs,)``

    O projetor no espaço-imagem é construído da SVD, e não de ``pinv(a) @ a``,
    para que ``rcond`` controle explicitamente o que conta como direção medida.
    Direções com valor singular abaixo do corte são tratadas como núcleo: elas
    são formalmente medidas, mas com amplificação de ruído que as torna inúteis
    sob 12 bits.
    """
    op = np.atleast_2d(np.asarray(a))
    xh = np.asarray(x_hat).ravel()
    obs = np.asarray(y).ravel()

    if op.shape[1] != xh.size:
        raise ValueError(f"a tem {op.shape[1]} colunas mas x_hat tem {xh.size} elementos")
    if op.shape[0] != obs.size:
        raise ValueError(f"a tem {op.shape[0]} linhas mas y tem {obs.size} elementos")

    _, s, vh = np.linalg.svd(op, full_matrices=False)
    cutoff = rcond * (s[0] if s.size else 0.0)
    keep = s > cutoff
    rank = int(np.count_nonzero(keep))

    v_meas = vh[keep].conj().T  # base ortonormal do espaco-imagem de A^H
    coeffs = v_meas.conj().T @ xh
    measured = float(np.real(np.vdot(coeffs, coeffs)))
    total = float(np.real(np.vdot(xh, xh)))
    prior = max(total - measured, 0.0)

    resid = float(np.linalg.norm(op @ xh - obs))
    denom = float(np.linalg.norm(obs))
    consistency = resid / denom if denom > 0.0 else resid

    return Attribution(
        measured_energy=measured,
        prior_energy=prior,
        data_consistency=consistency,
        rank=rank,
        n_unknowns=int(op.shape[1]),
    )


def hf_recovery_ratio(
    x_hat: NDArray[np.float64],
    x_true: NDArray[np.float64],
    fs_hz: float,
    f_cut_hz: float,
) -> float:
    """Energia recuperada acima de ``f_cut_hz`` sobre a energia verdadeira ali.

    Métrica espectral pedida no escopo original. Use **junto** com `attribute`:
    sozinha ela não distingue energia recuperada de energia inventada que por
    acaso caiu na banda certa.
    """
    x_hat = np.asarray(x_hat, dtype=float).ravel()
    x_true = np.asarray(x_true, dtype=float).ravel()
    if x_hat.size != x_true.size:
        raise ValueError("x_hat e x_true precisam ter o mesmo tamanho")

    freqs = np.fft.rfftfreq(x_true.size, d=1.0 / fs_hz)
    band = freqs >= f_cut_hz
    e_true = float(np.sum(np.abs(np.fft.rfft(x_true)[band]) ** 2))
    if e_true <= 0.0:
        raise ValueError("nao ha energia acima de f_cut_hz no sinal verdadeiro")
    e_hat = float(np.sum(np.abs(np.fft.rfft(x_hat)[band]) ** 2))
    return e_hat / e_true
