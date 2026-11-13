"""Nyquist -> sub-Nyquist distillation."""

import numpy as np
import pytest
from snq_learn import DistillConfig, SubNyquistStudent, distill, evaluate, teacher_distribution


def test_distribuicao_do_professor_soma_um_e_ordena_certo():
    p = teacher_distribution(np.array([1.0, 9.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert np.argmax(p) == 1


def test_temperatura_alta_achata_a_distribuicao():
    """A higher temperature transfers more information about the losing classes."""
    s = np.array([1.0, 9.0, 3.0])
    fria = teacher_distribution(s, temperature=0.3)
    quente = teacher_distribution(s, temperature=5.0)
    assert quente.max() < fria.max()
    assert quente.std() < fria.std()


def test_professor_sem_candidatas_e_erro():
    with pytest.raises(ValueError, match="candidates"):
        teacher_distribution(np.array([]))


def test_aluno_cabe_no_orcamento_do_no():
    """Fitting in 256 kB is a requirement of the target, not an implementation detail."""
    s = SubNyquistStudent(n_input=512, n_candidates=24)
    assert s.budget_bytes(bits=8) < 256 * 1024, f"{s.budget_bytes()} bytes"
    assert s.n_parameters() < 100_000


def test_aluno_aprende_a_separar_classes_sinteticas():
    """End to end: the student has to rise above chance.

    Synthetic data where each class has its own line in the envelope. If training
    does not rise above chance here, there is nothing to test on real data.
    """
    rng = np.random.default_rng(0)
    n_cls, n_per, n_in = 4, 40, 256
    t = np.arange(n_in)
    x, y = [], []
    for c in range(n_cls):
        for _ in range(n_per):
            sig = np.sin(2 * np.pi * (3 + 4 * c) * t / n_in) + 0.4 * rng.normal(size=n_in)
            x.append(sig)
            y.append(c)
    x = np.asarray(x)
    y = np.asarray(y)

    # "perfect" teacher: concentrates the mass on the right class
    teacher = np.full((len(y), n_cls), 0.05)
    teacher[np.arange(len(y)), y] = 0.85
    teacher /= teacher.sum(1, keepdims=True)

    student = SubNyquistStudent(n_input=n_in, n_candidates=n_cls, width=8)
    hist = distill(student, x, teacher, y, DistillConfig(epochs=40, seed=0))

    assert hist["acc"][-1] > 0.7, f"final accuracy {hist['acc'][-1]:.2f}"
    # The synthetic task is easy and the student already saturates in the first epoch, so
    # comparing initial and final accuracy does not measure progress. The loss does.
    assert hist["loss"][-1] < hist["loss"][0], "the loss did not drop"
    assert evaluate(student, x, y) == pytest.approx(hist["acc"][-1], abs=0.01)


def test_perda_cai_durante_o_treino():
    rng = np.random.default_rng(1)
    n, n_in, n_cls = 60, 128, 3
    x = rng.normal(size=(n, n_in))
    y = rng.integers(0, n_cls, size=n)
    teacher = np.full((n, n_cls), 1.0 / n_cls)
    student = SubNyquistStudent(n_input=n_in, n_candidates=n_cls, width=8)
    hist = distill(student, x, teacher, y, DistillConfig(epochs=25, seed=1))
    assert hist["loss"][-1] < hist["loss"][0]
