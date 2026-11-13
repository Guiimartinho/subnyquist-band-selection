"""Nyquist → sub-Nyquist distillation — Direction 2 of the project.

The technique comes from `vol2024poweraware` (Vol, Danial & Shlezinger, IEEE TSP 2025), who
proved that power-aware distillation works on the **bits** axis, with a memristive
ADC. They keep uniform sampling by construction. Here the same technique is
transposed to the **rate** axis, which is what remains open.

The setup:

- **Teacher**: sees the signal at the full rate and produces, through the classical baseline
  (kurtogram → envelope → harmonic energy), a distribution over the discrete
  set of lines that the geometry admits.
- **Student**: sees only the sub-Nyquist samples and has to produce the same
  distribution.
- **Loss**: divergence to the teacher, plus cross-entropy against the geometric
  ground truth.

Two design choices that are not aesthetic:

**The student decodes to a finite set of lines, not to a waveform.**
That is what makes the output auditable: each class corresponds to a frequency that the
bearing geometry can produce, so there is no way for the model to invent a fault
that the bearing is not capable of generating. It is the operational answer to the warning
in `vangorp2021adps`, whose authors admit that *"the sub-sampling regime could
obfuscate the information required to recognize failure cases"*.

**The student is small on purpose.** The budget is 256 kB of RAM on the nRF54L15, and
`monga2021unrolling` shows that structure coming from the model gives more with fewer
parameters than a generic network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class DistillConfig:
    """Hyperparameters. ``temperature`` and ``alpha`` follow the usual distillation."""

    temperature: float = 3.0
    alpha: float = 0.7
    """Weight of the divergence to the teacher; ``1 - alpha`` goes to the ground truth."""

    lr: float = 3e-3
    epochs: int = 60
    batch_size: int = 32
    seed: int = 0


class SubNyquistStudent(nn.Module):
    """Compact student: undersampled envelope → distribution over admissible lines.

    Deliberately shallow architecture. A 1D convolutional trunk extracts local structure
    from the envelope, and a linear layer projects onto the candidates. The parameter
    count is reported by `budget_bytes` because **fitting is a requirement, not a
    detail**: above 256 kB the model does not run on the target.
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
        """Memory of the quantized model, in bytes. INT8 by default."""
        return self.n_parameters() * bits // 8


def teacher_distribution(
    scores: NDArray[np.float64], temperature: float = 1.0
) -> NDArray[np.float64]:
    """Converts the harmonic energies of the baseline into a distribution over candidates.

    The teacher is not a network: it is the classical method running at the full rate. This
    matters because it means the distilled knowledge is **verifiable** — it comes from
    a procedure that can be audited line by line, and not from another opaque model.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("teacher without candidates")
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
    """Trains the student to reproduce the teacher and match the geometric ground truth.

    ``x_sub``: sub-Nyquist envelopes, shape ``(n, n_input)``.
    ``teacher_probs``: teacher distribution, ``(n, n_candidates)``.
    ``labels``: index of the true candidate, ``(n,)``.
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
    """Accuracy of the student against the geometric ground truth."""
    student.eval()
    with torch.no_grad():
        pred = student(torch.tensor(np.asarray(x_sub, dtype=np.float32))).argmax(1).numpy()
    return float(np.mean(pred == np.asarray(labels)))
