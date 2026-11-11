"""WDCNN (`zhang2017wdcnn`): **só baseline** de rede genérica de diagnóstico, nunca o método.

Zhang, Peng, Li, Chen & Zhang, *A New Deep Learning Model for Fault Diagnosis with Good
Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals*, Sensors 17(2):425, 2017,
doi:10.3390/s17020425. Arquitetura conferida na Tabela 2 do texto aberto (PMC5336047) em
2026-11-11:

======  ==============  =================  ======  ==========  ========
camada  tipo            núcleo/passo       canais  saída       padding
======  ==============  =================  ======  ==========  ========
1       convolução 1    64×1 / 16×1        16      128×16      sim
2       pooling 1       2×1 / 2×1          16      64×16       não
3       convolução 2    3×1 / 1×1          32      64×32       sim
4       pooling 2       2×1 / 2×1          32      32×32       não
5       convolução 3    3×1 / 1×1          64      32×64       sim
6       pooling 3       2×1 / 2×1          64      16×64       não
7       convolução 4    3×1 / 1×1          64      16×64       sim
8       pooling 4       2×1 / 2×1          64      8×64        não
9       convolução 5    3×1 / 1×1          64      6×64        não
10      pooling 5       2×1 / 2×1          64      3×64        não
11      totalmente con  100                1       100×1
12      softmax         10                 1       10
======  ==============  =================  ======  ==========  ========

Entrada de 2048 pontos. BN logo depois de cada convolução e da camada totalmente conectada,
antes da ReLU; pooling por máximo; Adam e entropia cruzada; sem dropout nem termo de
regularização (o texto diz que a BN já faz esse papel). O padding "sim" é o "same" com passo:
128 = 2048/16 exige 24 amostras de cada lado na primeira convolução.

**Não estão no texto** e são escolhas declaradas desta reimplementação: taxa de aprendizado,
tamanho do lote, número de épocas e a normalização da entrada (o runner do E15 padroniza cada
janela). A AdaBN do artigo, para mudança de carga, não está implementada: o E15 não muda de
domínio entre treino e teste.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["WDCNN"]

# (canais de entrada, canais de saída, núcleo, passo, padding) das cinco convoluções da Tabela 2
_TABLE2 = (
    (1, 16, 64, 16, 24),
    (16, 32, 3, 1, 1),
    (32, 64, 3, 1, 1),
    (64, 64, 3, 1, 1),
    (64, 64, 3, 1, 0),
)


class WDCNN(nn.Module):
    """WDCNN da Tabela 2 de `zhang2017wdcnn`, com ``n_classes`` saídas."""

    def __init__(self, n_classes: int, input_length: int = 2048) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for c_in, c_out, k, s, p in _TABLE2:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
        self.features = nn.Sequential(*layers)
        self.input_length = int(input_length)
        with torch.no_grad():
            n_flat = int(self.features(torch.zeros(1, 1, self.input_length)).numel())
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, int(n_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``(B, L)`` ou ``(B, 1, L)`` → logits ``(B, n_classes)``; softmax fica na perda."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))

    def trace_shapes(self) -> list[tuple[str, tuple[int, int]]]:
        """``(nome, (largura, canais))`` depois de cada convolução e pooling, como na Tabela 2."""
        out: list[tuple[str, tuple[int, int]]] = []
        h = torch.zeros(2, 1, self.input_length)
        was_training = self.training
        self.eval()
        n_conv = n_pool = 0
        with torch.no_grad():
            for layer in self.features:
                h = layer(h)
                if isinstance(layer, nn.Conv1d):
                    n_conv += 1
                    out.append((f"conv{n_conv}", (int(h.shape[2]), int(h.shape[1]))))
                elif isinstance(layer, nn.MaxPool1d):
                    n_pool += 1
                    out.append((f"pool{n_pool}", (int(h.shape[2]), int(h.shape[1]))))
            fc = self.classifier[1](self.classifier[0](h))
            out.append(("fc", (int(fc.shape[1]), 1)))
        self.train(was_training)
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
