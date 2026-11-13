"""Training loop with an epoch cap, early stopping by patience and best checkpoint.

It is the E15 protocol and applies equally to all learned methods, so that the comparison
measures the method and not the loop: epoch cap, validation loss at every epoch, stop when the
loss does not improve for ``patience`` consecutive epochs, and the returned model is the one
from the **best** validation checkpoint, not the one from the last epoch.

The loop knows nothing about data: it receives functions that produce the batches of an epoch,
the loss of a batch and the validation loss. The experiment runner is what loads the data.
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
    """Minimum improvement of the validation loss to reset the patience count."""
    min_epochs: int = 0
    """Epochs before which early stopping does not count and no checkpoint is chosen.

    For masks with temperature annealing, use the end of the annealing. Before that the
    mask is still soft, and a checkpoint from that phase returns a mask that has not left the
    noise (E15, 2026-11-11: all the splits that did not converge stopped before the end of the
    annealing).
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
    """Trains ``model`` and leaves it in the state of the best validation checkpoint.

    ``train_batches(epoch)`` returns the batches of the epoch; ``batch_loss(model, batch)`` the
    loss to minimize (with penalties, if any); ``val_loss(model)`` is called with the model in
    ``eval()`` and without gradient. Only the parameters with ``requires_grad`` enter the
    optimizer.

    ``param_groups`` follows the ``torch.optim`` format: groups with their own ``lr`` (the mask,
    for example, which needs to leave the Gumbel noise faster than the decoder). Without
    it, all trainable parameters use ``cfg.lr``.

    A non-finite loss interrupts training (``diverged``) and returns the best previous
    checkpoint; if none exists, the initial state.
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
        raise ValueError("no trainable parameter")
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
            # annealing phase: neither checkpoint nor patience count
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
    if best_epoch < 0:
        # training ended before `min_epochs`: the last finite state, already stored, is used
        finite = [h for h in history if math.isfinite(h["val_loss"])]
        if finite:
            best_epoch, best_loss = int(finite[-1]["epoch"]), float(finite[-1]["val_loss"])
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
