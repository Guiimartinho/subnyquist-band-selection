"""Divisões por registro repetidas e métricas balanceadas.

Um resultado de classificação obtido numa **única** divisão treino/teste não é um
resultado: é uma amostra de tamanho um de uma variável aleatória cuja dispersão
ninguém mediu. O E04 exibiu o sintoma em forma pura — 11 registros de uma classe
contra 4 de outra, teste de 5 registros, e o controle de classe majoritária
oscilando entre 80% e 100% conforme o sorteio. A diferença que sustentou a
conclusão era menor que a variação do próprio sorteio.

Este módulo dá o arcabouço mínimo para não repetir isso:

* `record_splits` — divisão por **registro**, nunca por segmento, repetida, com as
  sementes derivadas de uma semente mestra por `numpy.random.SeedSequence`. Duas
  execuções com a mesma semente mestra produzem exatamente as mesmas divisões.
* `classification_scores` — acurácia bruta, acurácia balanceada e F1 macro lado a
  lado. Sob classes desbalanceadas as três respondem a perguntas diferentes, e
  reportar só a primeira é o que torna um controle de maioria imbatível.
* `wilson_halfwidth` / `wilson_interval` — incerteza da proporção, na mesma
  convenção já usada nas figuras do projeto (`papers/shared/figures/`).

**Qual incerteza vale para qual métrica.** O intervalo de Wilson é um intervalo
para uma *proporção*: aplica-se à acurácia bruta, que é a fração de segmentos de
teste corretos. Não se aplica à acurácia balanceada nem ao F1 macro, que são médias
de proporções com denominadores distintos. Para essas, e para qualquer métrica
agregada sobre divisões, a dispersão a reportar é a inter-divisão —
`percentile_interval`. Usar Wilson onde ele não vale produz uma barra de erro
otimista, que é pior que nenhuma.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ClassificationScores",
    "RecordSplit",
    "classification_scores",
    "majority_label",
    "percentile_interval",
    "record_labels",
    "record_splits",
    "wilson_halfwidth",
    "wilson_interval",
]


# ----------------------------------------------------------------------------
# incerteza
# ----------------------------------------------------------------------------
def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Meia-largura do intervalo de Wilson para uma proporção ``p`` com ``n`` ensaios.

    Mesma convenção das figuras do projeto (`fig_e01_frontier.py`,
    `fig_e02_undersampling.py`, `fig_e07_order_domain.py`), reproduzida aqui para
    que código e figura não divirjam. ``z = 1.96`` é o nível de 95%.

    O intervalo de Wilson não é simétrico em torno de ``p``: esta meia-largura é a
    do intervalo em torno do **centro deslocado**, e é o que as figuras desenham.
    Quando o que se quer são os dois extremos, usar `wilson_interval`.
    """
    if n <= 0:
        return 0.0
    d = 1.0 + z * z / n
    return float((z / d) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Extremos ``(lo, hi)`` do intervalo de Wilson a 95% para ``successes`` em ``n``.

    Preferível ao intervalo normal justamente no regime deste projeto: ``n`` pequeno
    e proporções perto de 0 ou de 1, onde o intervalo normal escapa de ``[0, 1]``.
    Com ``n = 0`` devolve ``(0.0, 1.0)`` — ignorância total, e não uma barra de
    largura zero.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / d
    half = wilson_halfwidth(p, n, z)
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def percentile_interval(
    values: Sequence[float] | NDArray[np.float64], level: float = 0.95
) -> tuple[float, float, float]:
    """``(mediana, percentil inferior, percentil superior)`` de uma amostra.

    É a incerteza correta para uma métrica agregada sobre divisões repetidas —
    acurácia balanceada, F1 macro — porque não supõe que a métrica seja uma
    proporção. Com uma única divisão devolve o mesmo valor três vezes, o que é a
    resposta honesta: uma divisão não mede dispersão nenhuma.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("amostra vazia")
    tail = 100.0 * (1.0 - level) / 2.0
    return (
        float(np.median(v)),
        float(np.percentile(v, tail)),
        float(np.percentile(v, 100.0 - tail)),
    )


# ----------------------------------------------------------------------------
# métricas de classificação
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationScores:
    """As três acurácias que só coincidem quando as classes são balanceadas."""

    n: int
    n_correct: int

    accuracy: float
    """Fração de segmentos corretos. Sob desbalanceamento, mede prevalência."""

    balanced_accuracy: float
    """Média das revocações sobre as classes **com suporte**. 1/k é o acaso."""

    macro_f1: float
    """Média dos F1 sobre as classes com suporte. Penaliza o palpite constante."""

    support: dict[int, int]
    recall: dict[int, float]
    f1: dict[int, float]

    scored_classes: tuple[int, ...]
    """Classes com suporte > 0 — as únicas que entram nas médias macro."""

    n_absent_class_predictions: int
    """Predições em classes de prevalência **zero** no conjunto de teste.

    Uma candidata que nunca ocorre não pode ser recuperada nem perdida; ela só pode
    gerar falso positivo. O número fica separado das médias macro em vez de ser
    diluído nelas, porque é um modo de erro distinto.
    """

    def summary(self) -> str:
        return (
            f"acc {100 * self.accuracy:.1f}% | "
            f"bal {100 * self.balanced_accuracy:.1f}% | "
            f"F1 {100 * self.macro_f1:.1f}% | "
            f"n={self.n}"
        )


def classification_scores(
    y_true: NDArray[np.int64] | Sequence[int],
    y_pred: NDArray[np.int64] | Sequence[int],
    n_classes: int,
) -> ClassificationScores:
    """Acurácia bruta, acurácia balanceada e F1 macro de uma predição.

    ``n_classes`` é o número de saídas do classificador, que pode ser maior que o
    número de classes observadas — é o caso do E04/E05, onde a candidata FTF existe
    na saída e não existe no conjunto de dados. Classes sem suporte em ``y_true``
    ficam fora das médias macro (a revocação delas é indefinida, não zero) e
    aparecem em `n_absent_class_predictions`.
    """
    t = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_pred, dtype=np.int64).ravel()
    if t.size != p.size:
        raise ValueError(f"y_true tem {t.size} elementos e y_pred tem {p.size}")
    if t.size == 0:
        raise ValueError("conjunto de teste vazio")
    if n_classes <= 0:
        raise ValueError("n_classes precisa ser positivo")
    if int(t.max(initial=0)) >= n_classes or int(p.max(initial=0)) >= n_classes:
        raise ValueError(f"rótulo fora de [0, {n_classes})")

    support: dict[int, int] = {}
    recall: dict[int, float] = {}
    f1: dict[int, float] = {}
    for c in range(n_classes):
        is_true = t == c
        is_pred = p == c
        n_c = int(np.count_nonzero(is_true))
        support[c] = n_c
        tp = int(np.count_nonzero(is_true & is_pred))
        n_pred = int(np.count_nonzero(is_pred))
        if n_c == 0:
            continue
        rec = tp / n_c
        prec = tp / n_pred if n_pred > 0 else 0.0
        recall[c] = rec
        f1[c] = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

    scored = tuple(c for c in range(n_classes) if support[c] > 0)
    absent = tuple(c for c in range(n_classes) if support[c] == 0)
    n_correct = int(np.count_nonzero(t == p))

    return ClassificationScores(
        n=int(t.size),
        n_correct=n_correct,
        accuracy=n_correct / t.size,
        balanced_accuracy=float(np.mean([recall[c] for c in scored])),
        macro_f1=float(np.mean([f1[c] for c in scored])),
        support=support,
        recall=recall,
        f1=f1,
        scored_classes=scored,
        n_absent_class_predictions=int(np.count_nonzero(np.isin(p, np.asarray(absent, dtype=int)))),
    )


def majority_label(labels: NDArray[np.int64] | Sequence[int], n_classes: int) -> int:
    """Classe mais frequente — o controle que qualquer classificador tem de bater.

    Recebe os rótulos do **treino** de uma divisão específica. O controle é uma
    função da divisão, não do conjunto de dados: calcular a maioria sobre o total e
    aplicá-la a todas as divisões vaza informação do teste para o controle.
    """
    y = np.asarray(labels, dtype=np.int64).ravel()
    if y.size == 0:
        raise ValueError("conjunto de treino vazio")
    return int(np.bincount(y, minlength=n_classes).argmax())


# ----------------------------------------------------------------------------
# divisão por registro
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RecordSplit:
    """Uma divisão treino/teste por registro, reprodutível a partir de ``seed``."""

    index: int
    seed: int
    """Semente desta divisão, derivada da mestra. Vai para o CSV e reproduz a divisão."""

    train_records: tuple[str, ...]
    test_records: tuple[str, ...]

    train_index: NDArray[np.int64]
    """Índices dos segmentos de treino no vetor original."""

    test_index: NDArray[np.int64]

    majority_label: int
    """Classe majoritária dos **segmentos de treino desta divisão**."""

    stratified: bool
    classes_never_tested: tuple[int, ...]
    """Classes com um único registro: ficam sempre no treino (ver `record_splits`)."""


def record_labels(
    records: Sequence[str], labels: NDArray[np.int64] | Sequence[int]
) -> dict[str, int]:
    """Rótulo de cada registro, exigindo que os segmentos dele concordem.

    Um registro com segmentos de rótulos diferentes não tem divisão por registro
    bem definida — estratificar por classe passaria a ser impossível, e o vazamento
    voltaria pela porta dos fundos. O caso é erro, não aviso silencioso.
    """
    y = np.asarray(labels, dtype=np.int64).ravel()
    if len(records) != y.size:
        raise ValueError(f"{len(records)} registros para {y.size} rótulos")
    out: dict[str, int] = {}
    for name, lab in zip(records, y.tolist(), strict=True):
        prev = out.setdefault(name, int(lab))
        if prev != int(lab):
            raise ValueError(f"registro {name!r} tem rótulos {prev} e {int(lab)}")
    return out


def record_splits(
    records: Sequence[str],
    labels: NDArray[np.int64] | Sequence[int],
    n_splits: int,
    test_fraction: float,
    master_seed: int,
    n_classes: int | None = None,
    stratify: bool = True,
    min_test_per_class: int | None = None,
) -> list[RecordSplit]:
    """Divisões treino/teste **por registro**, repetidas e reprodutíveis.

    ``records`` é o identificador do registro de origem de cada segmento; segmentos
    do mesmo registro caem sempre do mesmo lado. Divisão por segmento é vazamento:
    segmentos de um registro compartilham montagem, rotação e rótulo, e o
    classificador aprende a reconhecer o registro em vez da falha.

    ``test_fraction`` é a fração de **registros**, não de segmentos — os registros
    têm contagens diferentes de segmentos e a fração de segmentos flutua junto com o
    sorteio. Essa flutuação é parte do que se quer medir.

    Comportamento declarado da estratificação, para não virar surpresa:

    * cada classe contribui ``max(min_test_per_class, round(test_fraction * n_c))``
      registros de teste, limitado a ``n_c - 1`` — o treino nunca perde uma classe
      inteira;
    * uma classe com **um único registro** fica sempre no treino, e o índice dela
      aparece em `RecordSplit.classes_never_tested`. A alternativa seria alternar o
      registro entre treino e teste, e aí o modelo nunca vê a classe em metade das
      divisões, o que é pior e mais difícil de perceber;
    * com ``stratify=False`` sorteiam-se registros sem olhar a classe. É o regime do
      E04, e é o que permite um conjunto de teste inteiro de uma só classe — e um
      teste de classe única faz a acurácia balanceada de **qualquer** preditor
      constante valer 1,0, porque a única classe com suporte é acertada inteira.

    ``min_test_per_class`` **só existe sob estratificação**, e por isso o valor
    padrão é ``None`` em vez de ``1``: passá-lo com ``stratify=False`` é erro, não
    um parâmetro ignorado em silêncio. Foi exatamente esse silêncio que produziu
    divisões de teste de classe única num braço em que o config declarava
    ``min_test_per_class: 1``. Sob estratificação, ``None`` equivale a ``1``.

    As sementes vêm de ``SeedSequence(master_seed).generate_state(n_splits)``: a
    mesma semente mestra dá as mesmas divisões, e cada divisão carrega a própria
    semente inteira no CSV.
    """
    if n_splits <= 0:
        raise ValueError("n_splits precisa ser positivo")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction precisa estar em (0, 1)")
    if not stratify and min_test_per_class is not None:
        raise ValueError(
            "min_test_per_class só se aplica sob estratificação; com stratify=False o "
            "sorteio não olha a classe e o parâmetro seria ignorado em silêncio"
        )
    min_per_class = 1 if min_test_per_class is None else int(min_test_per_class)

    y = np.asarray(labels, dtype=np.int64).ravel()
    per_record = record_labels(records, y)
    names = sorted(per_record)  # ordem canônica: o sorteio não depende da de entrada
    if len(names) < 2:
        raise ValueError(f"divisão por registro exige ao menos 2 registros, há {len(names)}")
    k = int(n_classes) if n_classes is not None else int(y.max()) + 1

    rec_array = np.asarray(records, dtype=object)
    index_of: dict[str, NDArray[np.int64]] = {
        name: np.flatnonzero(rec_array == name).astype(np.int64) for name in names
    }

    by_class: dict[int, list[str]] = {}
    for name in names:
        by_class.setdefault(per_record[name], []).append(name)
    never_tested = tuple(sorted(c for c, v in by_class.items() if len(v) == 1)) if stratify else ()

    seeds = np.random.SeedSequence(master_seed).generate_state(n_splits, dtype=np.uint32)
    splits: list[RecordSplit] = []
    for i, raw_seed in enumerate(seeds.tolist()):
        seed = int(raw_seed)
        rng = np.random.default_rng(seed)

        held: list[str] = []
        if stratify:
            for c in sorted(by_class):
                pool = list(by_class[c])
                n_c = len(pool)
                want = max(min_per_class, int(np.floor(test_fraction * n_c + 0.5)))
                want = min(want, n_c - 1)
                if want <= 0:
                    continue
                rng.shuffle(pool)
                held.extend(pool[:want])
        else:
            pool = list(names)
            rng.shuffle(pool)
            want = max(1, int(np.floor(test_fraction * len(pool) + 0.5)))
            want = min(want, len(pool) - 1)
            held.extend(pool[:want])

        held_set = set(held)
        train_names = tuple(n for n in names if n not in held_set)
        test_names = tuple(n for n in names if n in held_set)
        if not train_names or not test_names:
            raise ValueError(f"divisão {i} deixou um dos lados vazio")

        tr = np.concatenate([index_of[n] for n in train_names]).astype(np.int64)
        te = np.concatenate([index_of[n] for n in test_names]).astype(np.int64)
        tr.sort()
        te.sort()

        splits.append(
            RecordSplit(
                index=i,
                seed=seed,
                train_records=train_names,
                test_records=test_names,
                train_index=tr,
                test_index=te,
                majority_label=majority_label(y[tr], k),
                stratified=stratify,
                classes_never_tested=never_tested,
            )
        )
    return splits
