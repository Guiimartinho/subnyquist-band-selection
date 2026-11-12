"""Laço de treino: teto de épocas, paciência e melhor checkpoint."""

import math

import pytest
import torch
from snq_learn import FitConfig, fit


class Scalar(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(()))


def _fit(model, cfg, target=2.0, val_target=1.0):
    """Treino puxa ``w`` para ``target``; validação mede a distância a ``val_target``.

    A perda de validação cai enquanto ``w`` se aproxima de 1 e sobe depois: o sobreajuste em
    forma pura. A parada tem de acontecer e o checkpoint devolvido é o mais perto de 1.
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
    # o estado final e o do melhor checkpoint, nao o da ultima epoca
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
    assert float(model.w.detach()) == 0.0  # estado inicial, porque nenhuma epoca terminou


def test_so_parametros_treinaveis_entram_no_otimizador():
    model = Scalar()
    model.w.requires_grad_(False)
    with pytest.raises(ValueError, match="treinavel"):
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
    # um passo de Adam move cada parametro de ~lr: b anda 100 vezes mais que a
    assert float(model.b.detach()) == pytest.approx(100 * float(model.a.detach()), rel=1e-3)


def test_min_epochs_blocks_early_stop_and_checkpoint_during_annealing():
    """Com min_epochs, a parada nao dispara antes e o melhor checkpoint vem depois dele."""
    import torch
    from snq_learn.training import FitConfig, fit

    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    # perda de validacao que so piora: sem min_epochs, pararia em patience + 1 epocas
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
