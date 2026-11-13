"""Training loop: epoch cap, patience and best checkpoint."""

import math

import pytest
import torch
from snq_learn import FitConfig, fit


class Scalar(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(()))


def _fit(model, cfg, target=2.0, val_target=1.0):
    """Training pulls ``w`` towards ``target``; validation measures the distance to ``val_target``.

    The validation loss falls while ``w`` approaches 1 and rises afterwards: overfitting in
    pure form. The stop has to happen and the returned checkpoint is the one closest to 1.
    """
    return fit(
        model,
        train_batches=lambda epoch: [None],
        batch_loss=lambda m, _b: (m.w - target) ** 2,
        val_loss=lambda m: float((m.w.detach() - val_target) ** 2),
        cfg=cfg,
    )


def test_parada_por_paciencia_devolve_o_melhor_checkpoint():
    model = Scalar()
    res = _fit(model, FitConfig(max_epochs=500, patience=5, lr=0.05, grad_clip=None))
    assert res.stopped_early
    assert res.epochs_run == res.best_epoch + 1 + 5
    assert res.epochs_run < 500
    # the final state is that of the best checkpoint, not that of the last epoch
    assert float((model.w.detach() - 1.0) ** 2) == pytest.approx(res.best_val_loss)
    assert float((model.w.detach() - 1.0) ** 2) <= min(h["val_loss"] for h in res.history) + 1e-12
    assert len(res.history) == res.epochs_run


def test_teto_de_epocas_vale_quando_a_validacao_sempre_melhora():
    model = Scalar()
    res = _fit(model, FitConfig(max_epochs=7, patience=40, lr=0.01), val_target=2.0)
    assert not res.stopped_early
    assert res.epochs_run == 7 and res.best_epoch == 6


def test_perda_nao_finita_interrompe_e_restaura():
    model = Scalar()
    res = fit(
        model,
        train_batches=lambda epoch: [None],
        batch_loss=lambda m, _b: m.w * math.nan,
        val_loss=lambda m: 1.0,
        cfg=FitConfig(max_epochs=5, patience=2),
    )
    assert res.diverged
    assert float(model.w.detach()) == 0.0  # initial state, because no epoch finished


def test_so_parametros_treinaveis_entram_no_otimizador():
    model = Scalar()
    model.w.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        _fit(model, FitConfig(max_epochs=1))


def test_grupos_de_parametros_com_taxas_proprias():
    class Two(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.zeros(()))
            self.b = torch.nn.Parameter(torch.zeros(()))

    model = Two()
    fit(
        model,
        train_batches=lambda epoch: [None],
        batch_loss=lambda m, _b: (m.a - 10.0) ** 2 + (m.b - 10.0) ** 2,
        val_loss=lambda m: 0.0,
        cfg=FitConfig(max_epochs=1, patience=5, lr=0.001, grad_clip=None),
        param_groups=[{"params": [model.a], "lr": 0.001}, {"params": [model.b], "lr": 0.1}],
    )
    # one Adam step moves each parameter by ~lr: b moves 100 times more than a
    assert float(model.b.detach()) == pytest.approx(100 * float(model.a.detach()), rel=1e-3)


def test_min_epochs_blocks_early_stop_and_checkpoint_during_annealing():
    """With min_epochs, the stop does not fire before it and the best checkpoint comes after it."""
    import torch
    from snq_learn.training import FitConfig, fit

    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    # validation loss that only gets worse: without min_epochs, it would stop at patience + 1 epochs
    losses = iter([float(i) for i in range(1000)])

    res = fit(
        model,
        train_batches=lambda epoch: [torch.zeros(1)],
        batch_loss=lambda m, b: (m(b.view(1, 1)) ** 2).sum(),
        val_loss=lambda m: next(losses),
        cfg=FitConfig(max_epochs=200, patience=5, lr=1e-3, min_epochs=50),
    )
    assert res.epochs_run >= 50 + 5
    assert res.best_epoch >= 50


def test_min_epochs_acima_do_teto_vale_o_ultimo_estado():
    """If training ends before min_epochs, the best is the last finite state."""
    import torch
    from snq_learn.training import FitConfig, fit

    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    losses = iter([float(i) for i in range(100)])
    res = fit(
        model,
        train_batches=lambda epoch: [torch.zeros(1)],
        batch_loss=lambda m, b: (m(b.view(1, 1)) ** 2).sum(),
        val_loss=lambda m: next(losses),
        cfg=FitConfig(max_epochs=10, patience=5, lr=1e-3, min_epochs=50),
    )
    assert res.epochs_run == 10
    assert res.best_epoch == 9
    assert res.best_val_loss == 9.0
