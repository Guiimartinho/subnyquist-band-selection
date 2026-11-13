"""The framework that tells a result apart from the luck of the draw.

The case that motivates the module is at the end of the file: 11 records of one class
against 4 of another, a test set of 5 records — the exact configuration of E04, where
the majority-class control varies with the draw and swallows the measured effect.
"""

import numpy as np
import pytest
from snq_eval import (
    classification_scores,
    majority_label,
    percentile_interval,
    record_labels,
    record_splits,
    wilson_halfwidth,
    wilson_interval,
)


def segmentos(por_registro: dict[str, tuple[int, int]]) -> tuple[list[str], np.ndarray]:
    """``{nome: (rotulo, n_segmentos)}`` -> per-segment record and label vectors."""
    nomes: list[str] = []
    rotulos: list[int] = []
    for nome, (rot, n) in por_registro.items():
        nomes.extend([nome] * n)
        rotulos.extend([rot] * n)
    return nomes, np.asarray(rotulos, dtype=np.int64)


# ----------------------------------------------------------------------------
# per-record split
# ----------------------------------------------------------------------------
def test_segmentos_do_mesmo_registro_nunca_caem_dos_dois_lados():
    """The leakage that inflated the accuracy of the first version of E04."""
    recs, y = segmentos({f"R{i}": (i % 2, 7) for i in range(10)})
    arr = np.asarray(recs, dtype=object)
    for sp in record_splits(recs, y, n_splits=20, test_fraction=1 / 3, master_seed=1):
        no_treino = set(arr[sp.train_index].tolist())
        no_teste = set(arr[sp.test_index].tolist())
        assert no_treino & no_teste == set()
        assert no_treino | no_teste == set(arr.tolist())
        assert sp.train_index.size + sp.test_index.size == y.size


def test_a_mesma_semente_reproduz_a_mesma_divisao():
    recs, y = segmentos({f"R{i}": (i % 3, 4) for i in range(12)})
    a = record_splits(recs, y, n_splits=8, test_fraction=0.3, master_seed=20260907)
    b = record_splits(recs, y, n_splits=8, test_fraction=0.3, master_seed=20260907)
    c = record_splits(recs, y, n_splits=8, test_fraction=0.3, master_seed=20260908)

    for x, z in zip(a, b, strict=True):
        assert x.seed == z.seed
        assert x.test_records == z.test_records
        assert np.array_equal(x.test_index, z.test_index)
    assert [x.test_records for x in a] != [x.test_records for x in c]


def test_a_ordem_dos_segmentos_na_entrada_nao_muda_a_divisao():
    """The split is over the records; shuffling segments must not change it."""
    recs, y = segmentos({f"R{i}": (i % 2, 5) for i in range(8)})
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(recs))
    outra = [recs[i] for i in perm]

    a = record_splits(recs, y, n_splits=5, test_fraction=0.25, master_seed=7)
    b = record_splits(outra, y[perm], n_splits=5, test_fraction=0.25, master_seed=7)
    for x, z in zip(a, b, strict=True):
        assert x.test_records == z.test_records


def test_divisoes_repetidas_nao_sao_todas_iguais():
    recs, y = segmentos({f"R{i}": (i % 2, 3) for i in range(10)})
    sps = record_splits(recs, y, n_splits=30, test_fraction=0.3, master_seed=3)
    assert len({s.test_records for s in sps}) > 1


def test_estratificacao_mantem_as_duas_classes_no_teste():
    recs, y = segmentos({f"R{i}": (0 if i < 11 else 1, 6) for i in range(15)})
    for sp in record_splits(recs, y, n_splits=25, test_fraction=1 / 3, master_seed=11):
        rot = {y[np.flatnonzero(np.asarray(recs, dtype=object) == n)[0]] for n in sp.test_records}
        assert rot == {0, 1}


def test_classe_com_um_unico_registro_fica_sempre_no_treino():
    """Declared behavior: the alternative makes the class disappear from training."""
    recs, y = segmentos({"A1": (0, 4), "A2": (0, 4), "A3": (0, 4), "B1": (1, 4)})
    sps = record_splits(recs, y, n_splits=10, test_fraction=0.5, master_seed=5)
    for sp in sps:
        assert "B1" in sp.train_records
        assert sp.classes_never_tested == (1,)


def test_registro_com_rotulos_conflitantes_e_erro():
    recs = ["R1", "R1", "R2"]
    with pytest.raises(ValueError, match="R1"):
        record_labels(recs, np.asarray([0, 1, 0], dtype=np.int64))


def test_record_splits_recusa_conjunto_com_um_registro_so():
    recs, y = segmentos({"R1": (0, 9)})
    with pytest.raises(ValueError, match="at least 2 records"):
        record_splits(recs, y, n_splits=3, test_fraction=0.3, master_seed=0)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def test_acuracia_bruta_premia_a_maioria_e_a_balanceada_nao():
    """The number that made the E04 control unbeatable, and the one that unmasks it."""
    y = np.asarray([0] * 90 + [1] * 10, dtype=np.int64)
    sempre_maioria = np.zeros(100, dtype=np.int64)
    s = classification_scores(y, sempre_maioria, n_classes=2)
    assert s.accuracy == pytest.approx(0.90)
    assert s.balanced_accuracy == pytest.approx(0.50)
    assert s.macro_f1 < 0.50


def test_classe_de_prevalencia_zero_fica_fora_das_medias_macro():
    """The E04/E05 case: the FTF output exists, the class does not occur in the data."""
    y = np.asarray([1, 1, 2, 2], dtype=np.int64)
    pred = np.asarray([1, 0, 2, 2], dtype=np.int64)  # one error falls on the absent candidate
    s = classification_scores(y, pred, n_classes=3)
    assert s.scored_classes == (1, 2)
    assert s.support[0] == 0
    assert s.n_absent_class_predictions == 1
    assert s.balanced_accuracy == pytest.approx(0.75)
    assert 0 not in s.recall


def test_predicao_perfeita_satura_as_tres_metricas():
    y = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
    s = classification_scores(y, y, n_classes=3)
    assert s.accuracy == pytest.approx(1.0)
    assert s.balanced_accuracy == pytest.approx(1.0)
    assert s.macro_f1 == pytest.approx(1.0)


def test_majority_label_usa_o_treino_da_propria_divisao():
    assert majority_label(np.asarray([0, 0, 1], dtype=np.int64), 3) == 0
    assert majority_label(np.asarray([2, 2, 1], dtype=np.int64), 3) == 2


def test_wilson_reproduz_a_convencao_das_figuras():
    """Formula copied from papers/shared/figures/fig_e01_frontier.py."""
    z, p, n = 1.96, 0.8, 25
    d = 1.0 + z * z / n
    esperado = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    assert wilson_halfwidth(p, n) == pytest.approx(esperado)


def test_wilson_nao_escapa_do_intervalo_unitario():
    lo, hi = wilson_interval(5, 5)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == pytest.approx(1.0) or hi < 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_estreita_com_mais_ensaios():
    assert wilson_halfwidth(0.5, 400) < wilson_halfwidth(0.5, 25)


def test_percentile_interval_de_uma_divisao_so_nao_inventa_dispersao():
    m, lo, hi = percentile_interval([0.77])
    assert (m, lo, hi) == (0.77, 0.77, 0.77)


def test_percentile_interval_ordena_os_extremos():
    m, lo, hi = percentile_interval(np.linspace(0.0, 1.0, 101))
    assert lo < m < hi
    assert m == pytest.approx(0.5)


# ----------------------------------------------------------------------------
# the degenerate E04 case
# ----------------------------------------------------------------------------
def test_caso_e04_o_controle_de_maioria_varia_com_o_sorteio():
    """11 BPFO records against 4 BPFI, a test set of 5 records.

    Without stratification — the E04 regime — the test set can fall entirely in the
    majority class and the control scores 100%, against ~73% in another draw. The
    difference between splits is larger than the difference E04 used to reach its
    conclusion.
    """
    por_registro: dict[str, tuple[int, int]] = {}
    for i in range(11):
        por_registro[f"BPFO{i}"] = (1, 19)
    for i in range(4):
        por_registro[f"BPFI{i}"] = (2, 19)
    recs, y = segmentos(por_registro)

    sps = record_splits(
        recs,
        y,
        n_splits=200,
        test_fraction=1 / 3,
        master_seed=20260907,
        n_classes=3,
        stratify=False,
    )
    assert {len(s.test_records) for s in sps} == {5}

    acc = [
        classification_scores(
            y[s.test_index], np.full(s.test_index.size, s.majority_label, dtype=np.int64), 3
        ).accuracy
        for s in sps
    ]
    assert len(set(acc)) > 1, "the majority control has to vary across splits"
    assert max(acc) == pytest.approx(1.0), "some draw yields a single-class test set"
    assert min(acc) < 0.85
    assert max(acc) - min(acc) > 0.15


def test_caso_e04_estratificar_limita_a_variacao_mas_nao_a_elimina():
    por_registro: dict[str, tuple[int, int]] = {}
    for i in range(11):
        por_registro[f"BPFO{i}"] = (1, 19)
    for i in range(4):
        por_registro[f"BPFI{i}"] = (2, 19)
    recs, y = segmentos(por_registro)

    sps = record_splits(
        recs, y, n_splits=200, test_fraction=1 / 3, master_seed=20260907, n_classes=3, stratify=True
    )
    acc = [
        classification_scores(
            y[s.test_index], np.full(s.test_index.size, s.majority_label, dtype=np.int64), 3
        ).accuracy
        for s in sps
    ]
    assert max(acc) < 1.0, "with stratification the test set never falls entirely in one class"
    assert all(s.majority_label == 1 for s in sps)


def test_caso_e04_a_acuracia_balanceada_do_controle_e_constante_em_meio():
    """By construction: the majority gets one class entirely right and the other entirely wrong.

    This is what makes balanced accuracy the axis on which the majority control stops
    being a moving target — 0.5 in every split, with two classes present.
    """
    por_registro: dict[str, tuple[int, int]] = {}
    for i in range(11):
        por_registro[f"BPFO{i}"] = (1, 19)
    for i in range(4):
        por_registro[f"BPFI{i}"] = (2, 19)
    recs, y = segmentos(por_registro)

    sps = record_splits(
        recs, y, n_splits=50, test_fraction=1 / 3, master_seed=20260907, n_classes=3, stratify=True
    )
    bal = [
        classification_scores(
            y[s.test_index], np.full(s.test_index.size, s.majority_label, dtype=np.int64), 3
        ).balanced_accuracy
        for s in sps
    ]
    assert all(b == pytest.approx(0.5) for b in bal)


def test_min_test_per_class_sem_estratificacao_e_erro():
    """The guard cannot be accepted and ignored: without stratification it does not exist.

    The E05 config declared ``min_test_per_class: 1`` in the non-stratified arm, and
    the parameter was read and discarded. The result was splits whose test set fell
    entirely in one class.
    """
    recs, y = segmentos({f"R{i}": (i % 2, 5) for i in range(8)})
    with pytest.raises(ValueError, match="only applies under stratification"):
        record_splits(
            recs,
            y,
            n_splits=3,
            test_fraction=0.3,
            master_seed=0,
            stratify=False,
            min_test_per_class=1,
        )


def test_teste_de_classe_unica_da_acuracia_balanceada_um_a_qualquer_constante():
    """The mechanism that produces the effect attributed to repetition in the unstratified arm.

    In a split whose test set has a single class, the only class with support is
    predicted entirely correctly by a constant predictor of that class, and the mean
    of the recalls over the classes with support equals 1.0.
    """
    por_registro: dict[str, tuple[int, int]] = {}
    for i in range(11):
        por_registro[f"BPFO{i}"] = (1, 19)
    for i in range(4):
        por_registro[f"BPFI{i}"] = (2, 19)
    recs, y = segmentos(por_registro)

    sps = record_splits(
        recs,
        y,
        n_splits=40,
        test_fraction=1 / 3,
        master_seed=20260907,
        n_classes=3,
        stratify=False,
    )
    classe_unica = [s for s in sps if len(set(y[s.test_index].tolist())) == 1]
    assert classe_unica, "the draw without stratification has to produce a single-class test set"
    for s in classe_unica:
        (unica,) = set(y[s.test_index].tolist())
        bal = classification_scores(
            y[s.test_index], np.full(s.test_index.size, s.majority_label, dtype=np.int64), 3
        ).balanced_accuracy
        assert bal == pytest.approx(1.0 if s.majority_label == unica else 0.0)
