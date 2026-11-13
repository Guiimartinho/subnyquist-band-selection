"""WDCNN (`zhang2017wdcnn`): generic diagnosis network, **only as a baseline**, never the method.

Zhang, Peng, Li, Chen & Zhang, *A New Deep Learning Model for Fault Diagnosis with Good
Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals*, Sensors 17(2):425, 2017,
doi:10.3390/s17020425. Architecture checked against Table 2 of the open-access text
(PMC5336047) on 2026-11-11:

======  ================  =================  ========  ==========  ========
layer   type              kernel/stride      channels  output      padding
======  ================  =================  ========  ==========  ========
1       convolution 1     64×1 / 16×1        16        128×16      yes
2       pooling 1         2×1 / 2×1          16        64×16       no
3       convolution 2     3×1 / 1×1          32        64×32       yes
4       pooling 2         2×1 / 2×1          32        32×32       no
5       convolution 3     3×1 / 1×1          64        32×64       yes
6       pooling 3         2×1 / 2×1          64        16×64       no
7       convolution 4     3×1 / 1×1          64        16×64       yes
8       pooling 4         2×1 / 2×1          64        8×64        no
9       convolution 5     3×1 / 1×1          64        6×64        no
10      pooling 5         2×1 / 2×1          64        3×64        no
11      fully connected   100                1         100×1
12      softmax           10                 1         10
======  ================  =================  ========  ==========  ========

Input of 2048 points. BN right after each convolution and after the fully connected layer,
before the ReLU; max pooling; Adam and cross-entropy; no dropout and no regularization
term (the text says BN already plays that role). The "yes" padding is "same" padding with
stride: 128 = 2048/16 requires 24 samples on each side in the first convolution.

**Not stated in the text**, and declared as choices of this reimplementation: learning rate,
batch size, number of epochs and the input normalization (the E15 runner standardizes each
window). The paper's AdaBN, for load changes, is not implemented: E15 does not change
domain between training and test.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["WDCNN"]

# (input channels, output channels, kernel, stride, padding) of the five convolutions in Table 2
_TABLE2 = (
    (1, 16, 64, 16, 24),
    (16, 32, 3, 1, 1),
    (32, 64, 3, 1, 1),
    (64, 64, 3, 1, 1),
    (64, 64, 3, 1, 0),
)


class WDCNN(nn.Module):
    """WDCNN from Table 2 of `zhang2017wdcnn`, with ``n_classes`` outputs."""

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
        """``x`` ``(B, L)`` or ``(B, 1, L)`` → logits ``(B, n_classes)``; softmax is in the loss."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))

    def trace_shapes(self) -> list[tuple[str, tuple[int, int]]]:
        """``(name, (width, channels))`` after each convolution and pooling, as in Table 2."""
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
