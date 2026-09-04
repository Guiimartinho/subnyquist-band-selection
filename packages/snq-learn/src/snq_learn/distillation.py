"""Destilação Nyquist → sub-Nyquist — a Direção 2 do projeto.

A técnica vem de `vol2024poweraware` (Vol, Danial & Shlezinger, IEEE TSP 2024), que
provou destilação ciente de potência funcionar no eixo de **bits**, com ADC
memristivo. Eles mantêm amostragem uniforme por construção. Aqui a mesma técnica é
transposta para o eixo de **taxa**, que é o que resta aberto.

O arranjo:

- **Professor**: vê o sinal em taxa cheia e produz, pelo baseline clássico
  (kurtograma → envelope → energia harmônica), uma distribuição sobre o conjunto
  discreto de raias que a geometria admite.
- **Aluno**: vê apenas as amostras sub-Nyquist e precisa produzir a mesma
  distribuição.
- **Perda**: divergência para o professor, mais entropia cruzada contra a verdade
  geométrica.

Duas escolhas de projeto que não são estéticas:

**O aluno decodifica para um conjunto finito de raias, não para forma de onda.**
É o que torna a saída auditável: cada classe corresponde a uma frequência que a
geometria do mancal pode produzir, então não há como o modelo inventar uma falha
que o rolamento não é capaz de gerar. É a resposta operacional ao alerta de
`vangorp2021adps`, cujos autores admitem que *"the sub-sampling regime could
obfuscate the information required to recognize failure cases"*.

**O aluno é pequeno de propósito.** O orçamento é 256 kB de RAM no nRF54L15, e
`monga2021unrolling` mostra que estrutura vinda do modelo dá mais com menos
parâmetros que uma rede genérica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class DistillConfig:
    """Hiperparâmetros. ``temperature`` e ``alpha`` seguem a destilação usual."""

    temperature: float = 3.0
    alpha: float = 0.7
    """Peso da divergência para o professor; ``1 - alpha`` fica com a verdade."""

    lr: float = 3e-3
    epochs: int = 60
    batch_size: int = 32
    seed: int = 0


class SubNyquistStudent(nn.Module):
    """Aluno compacto: envelope subamostrado → distribuição sobre raias admissíveis.

    Arquitetura deliberadamente rasa. Um tronco convolucional 1D extrai estrutura
    local do envelope, e uma camada linear projeta sobre as candidatas. O contador de
    parâmetros é reportado por `budget_bytes` porque **caber é requisito, não
    detalhe**: acima de 256 kB o modelo não roda no alvo.
    """

    def __init__(self, n_input: int, n_candidates: int, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.head = nn.Linear(width * 32, n_candidates)
        self.n_input = n_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x.unsqueeze(1))
        return self.head(z.flatten(1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def budget_bytes(self, bits: int = 8) -> int:
        """Memória do modelo quantizado, em bytes. INT8 por padrão."""
        return self.n_parameters() * bits // 8


def teacher_distribution(
    scores: NDArray[np.float64], temperature: float = 1.0
) -> NDArray[np.float64]:
    """Converte as energias harmônicas do baseline numa distribuição sobre candidatas.

    O professor não é uma rede: é o método clássico rodando em taxa cheia. Isso
    importa porque significa que o conhecimento destilado é **verificável** — vem de
    um procedimento que se pode auditar linha a linha, e não de outro modelo opaco.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("professor sem candidatas")
    z = s / max(float(np.max(s)), 1e-12) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def distill(
    student: SubNyquistStudent,
    x_sub: NDArray[np.float64],
    teacher_probs: NDArray[np.float64],
    labels: NDArray[np.int64],
    cfg: DistillConfig | None = None,
) -> dict[str, list[float]]:
    """Treina o aluno para reproduzir o professor e acertar a verdade geométrica.

    ``x_sub``: envelopes sub-Nyquist, forma ``(n, n_input)``.
    ``teacher_probs``: distribuição do professor, ``(n, n_candidates)``.
    ``labels``: índice da candidata verdadeira, ``(n,)``.
    """
    cfg = cfg or DistillConfig()
    torch.manual_seed(cfg.seed)

    xb = torch.tensor(np.asarray(x_sub, dtype=np.float32))
    tb = torch.tensor(np.asarray(teacher_probs, dtype=np.float32))
    yb = torch.tensor(np.asarray(labels, dtype=np.int64))

    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    kl = nn.KLDivLoss(reduction="batchmean")
    ce = nn.CrossEntropyLoss()
    hist: dict[str, list[float]] = {"loss": [], "acc": []}

    n = xb.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            logits = student(xb[idx])
            soft = kl(
                torch.log_softmax(logits / cfg.temperature, dim=1),
                torch.softmax(torch.log(tb[idx] + 1e-9) / cfg.temperature, dim=1),
            ) * (cfg.temperature**2)
            loss = cfg.alpha * soft + (1.0 - cfg.alpha) * ce(logits, yb[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.numel()
        with torch.no_grad():
            acc = float((student(xb).argmax(1) == yb).float().mean())
        hist["loss"].append(total / n)
        hist["acc"].append(acc)
    return hist


def evaluate(
    student: SubNyquistStudent, x_sub: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    """Acurácia do aluno sobre a verdade geométrica."""
    student.eval()
    with torch.no_grad():
        pred = student(torch.tensor(np.asarray(x_sub, dtype=np.float32))).argmax(1).numpy()
    return float(np.mean(pred == np.asarray(labels)))
