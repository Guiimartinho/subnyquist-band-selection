"""Laço de treino com teto de épocas, parada antecipada por paciência e melhor checkpoint.

É o protocolo do E15 e vale igual para todos os métodos aprendidos, para que a comparação meça
o método e não o laço: teto de épocas, perda de validação a cada época, parada quando a perda
não melhora por ``patience`` épocas seguidas, e o modelo devolvido é o do **melhor** checkpoint
de validação, não o da última época.

O laço não sabe nada de dados: recebe funções que produzem os lotes de uma época, a perda de um
lote e a perda de validação. Quem carrega dados é o runner do experimento.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

__all__ = ["FitConfig", "FitResult", "fit"]


@dataclass(frozen=True)
class FitConfig:
    max_epochs: int = 500
    patience: int = 40
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0
    min_delta: float = 0.0
    """Melhora mínima da perda de validação para zerar a contagem de paciência."""
    min_epochs: int = 0
    """Épocas antes das quais a parada antecipada não conta e nenhum checkpoint é escolhido.

    Para máscaras com recozimento de temperatura, vale o fim do recozimento. Antes disso a
    máscara ainda está mole, e um checkpoint dessa fase devolve uma máscara que não saiu do
    ruído (E15, 2026-11-11: todas as divisões que não convergiram pararam antes do fim do
    recozimento).
    """


@dataclass
class FitResult:
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    stopped_early: bool
    diverged: bool
    history: list[dict[str, float]] = field(default_factory=list)


def fit(
    model: nn.Module,
    *,
    train_batches: Callable[[int], Iterable[Any]],
    batch_loss: Callable[[nn.Module, Any], torch.Tensor],
    val_loss: Callable[[nn.Module], float],
    cfg: FitConfig,
    on_epoch_start: Callable[[int], None] | None = None,
    param_groups: list[dict[str, Any]] | None = None,
) -> FitResult:
    """Treina ``model`` e o deixa no estado do melhor checkpoint de validação.

    ``train_batches(epoch)`` devolve os lotes da época; ``batch_loss(model, lote)`` a perda a
    minimizar (com penalidades, se houver); ``val_loss(model)`` é chamada com o modelo em
    ``eval()`` e sem gradiente. Só os parâmetros com ``requires_grad`` entram no otimizador.

    ``param_groups`` segue o formato do ``torch.optim``: grupos com ``lr`` próprio (a máscara,
    por exemplo, que precisa sair do ruído de Gumbel mais depressa que o decodificador). Sem
    ele, todos os parâmetros treináveis usam ``cfg.lr``.

    Uma perda não finita interrompe o treino (``diverged``) e devolve o melhor checkpoint
    anterior; se nenhum existir, o estado inicial.
    """
    if param_groups is None:
        groups: list[dict[str, Any]] = [
            {"params": [p for p in model.parameters() if p.requires_grad]}
        ]
    else:
        groups = [
            {**g, "params": [p for p in g["params"] if p.requires_grad]} for g in param_groups
        ]
    groups = [g for g in groups if g["params"]]
    params = [p for g in groups for p in g["params"]]
    if not params:
        raise ValueError("nenhum parametro treinavel")
    opt = torch.optim.Adam(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch = math.inf, -1
    best_state = copy.deepcopy(model.state_dict())
    since = 0
    history: list[dict[str, float]] = []
    diverged = stopped = False
    epoch = -1
    for epoch in range(int(cfg.max_epochs)):
        model.train()
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        total, count = 0.0, 0
        for batch in train_batches(epoch):
            loss = batch_loss(model, batch)
            if not torch.isfinite(loss):
                diverged = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
            opt.step()
            total += float(loss.detach())
            count += 1
        if diverged:
            break
        model.eval()
        with torch.no_grad():
            v = float(val_loss(model))
        history.append({"epoch": float(epoch), "train_loss": total / max(count, 1), "val_loss": v})
        if not math.isfinite(v):
            diverged = True
            break
        if epoch < int(cfg.min_epochs):
            # fase de recozimento: nem checkpoint nem contagem de paciencia
            best_state = copy.deepcopy(model.state_dict())
            continue
        if v < best_loss - cfg.min_delta:
            best_loss, best_epoch, since = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= int(cfg.patience):
                stopped = True
                break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        epochs_run=epoch + 1,
        stopped_early=stopped,
        diverged=diverged,
        history=history,
    )
