"""Repeated per-record splits and balanced metrics.

A classification result obtained on a **single** train/test split is not a result:
it is a sample of size one of a random variable whose spread nobody measured. E04
showed the symptom in pure form — 11 records of one class against 4 of another, a
test set of 5 records, and the majority-class control swinging between 80% and 100%
depending on the draw. The difference that supported the conclusion was smaller than
the variation of the draw itself.

This module provides the minimum framework for not repeating that:

* `record_splits` — split by **record**, never by segment, repeated, with the seeds
  derived from a master seed via `numpy.random.SeedSequence`. Two runs with the same
  master seed produce exactly the same splits.
* `classification_scores` — raw accuracy, balanced accuracy and macro F1 side by
  side. Under imbalanced classes the three answer different questions, and reporting
  only the first is what makes a majority control unbeatable.
* `wilson_halfwidth` / `wilson_interval` — uncertainty of the proportion, in the same
  convention already used in the project's figures (`papers/shared/figures/`).

**Which uncertainty applies to which metric.** The Wilson interval is an interval for
a *proportion*: it applies to raw accuracy, which is the fraction of correct test
segments. It does not apply to balanced accuracy or to macro F1, which are averages
of proportions with different denominators. For those, and for any metric aggregated
over splits, the spread to report is the between-split one — `percentile_interval`.
Using Wilson where it does not apply produces an optimistic error bar, which is worse
than none.
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
# uncertainty
# ----------------------------------------------------------------------------
def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson interval for a proportion ``p`` with ``n`` trials.

    Same convention as the project's figures (`fig_e01_frontier.py`,
    `fig_e02_undersampling.py`, `fig_e07_order_domain.py`), reproduced here so that
    code and figure do not diverge. ``z = 1.96`` is the 95% level.

    The Wilson interval is not symmetric around ``p``: this half-width is that of the
    interval around the **shifted center**, and it is what the figures draw. When
    what is wanted is the two endpoints, use `wilson_interval`.
    """
    if n <= 0:
        return 0.0
    d = 1.0 + z * z / n
    return float((z / d) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Endpoints ``(lo, hi)`` of the 95% Wilson interval for ``successes`` out of ``n``.

    Preferable to the normal interval precisely in this project's regime: small ``n``
    and proportions close to 0 or 1, where the normal interval escapes ``[0, 1]``.
    With ``n = 0`` it returns ``(0.0, 1.0)`` — total ignorance, and not a bar of zero
    width.
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
    """``(median, lower percentile, upper percentile)`` of a sample.

    It is the correct uncertainty for a metric aggregated over repeated splits —
    balanced accuracy, macro F1 — because it does not assume that the metric is a
    proportion. With a single split it returns the same value three times, which is
    the honest answer: one split measures no spread at all.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("empty sample")
    tail = 100.0 * (1.0 - level) / 2.0
    return (
        float(np.median(v)),
        float(np.percentile(v, tail)),
        float(np.percentile(v, 100.0 - tail)),
    )


# ----------------------------------------------------------------------------
# classification metrics
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationScores:
    """The three accuracies, which only coincide when the classes are balanced."""

    n: int
    n_correct: int

    accuracy: float
    """Fraction of correct segments. Under imbalance, it measures prevalence."""

    balanced_accuracy: float
    """Mean of the recalls over the classes **with support**. 1/k is chance."""

    macro_f1: float
    """Mean of the F1 scores over the classes with support. Penalizes the constant guess."""

    support: dict[int, int]
    recall: dict[int, float]
    f1: dict[int, float]

    scored_classes: tuple[int, ...]
    """Classes with support > 0 — the only ones that enter the macro averages."""

    n_absent_class_predictions: int
    """Predictions in classes with **zero** prevalence in the test set.

    A candidate that never occurs can be neither recovered nor missed; it can only
    produce false positives. The number is kept apart from the macro averages instead
    of being diluted in them, because it is a distinct error mode.
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
    """Raw accuracy, balanced accuracy and macro F1 of a prediction.

    ``n_classes`` is the number of outputs of the classifier, which may be larger than
    the number of observed classes — this is the case in E04/E05, where the FTF
    candidate exists in the output and does not exist in the dataset. Classes without
    support in ``y_true`` are left out of the macro averages (their recall is
    undefined, not zero) and show up in `n_absent_class_predictions`.
    """
    t = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_pred, dtype=np.int64).ravel()
    if t.size != p.size:
        raise ValueError(f"y_true has {t.size} elements and y_pred has {p.size}")
    if t.size == 0:
        raise ValueError("empty test set")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    if int(t.max(initial=0)) >= n_classes or int(p.max(initial=0)) >= n_classes:
        raise ValueError(f"label outside [0, {n_classes})")

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
    """Most frequent class — the control that any classifier has to beat.

    Takes the **training** labels of a specific split. The control is a function of
    the split, not of the dataset: computing the majority over the whole set and
    applying it to every split leaks information from the test set into the control.
    """
    y = np.asarray(labels, dtype=np.int64).ravel()
    if y.size == 0:
        raise ValueError("empty training set")
    return int(np.bincount(y, minlength=n_classes).argmax())


# ----------------------------------------------------------------------------
# per-record split
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RecordSplit:
    """One per-record train/test split, reproducible from ``seed``."""

    index: int
    seed: int
    """Seed of this split, derived from the master seed. It goes into the CSV and
    reproduces the split."""

    train_records: tuple[str, ...]
    test_records: tuple[str, ...]

    train_index: NDArray[np.int64]
    """Indices of the training segments in the original vector."""

    test_index: NDArray[np.int64]

    majority_label: int
    """Majority class of the **training segments of this split**."""

    stratified: bool
    classes_never_tested: tuple[int, ...]
    """Classes with a single record: they always stay in training (see `record_splits`)."""


def record_labels(
    records: Sequence[str], labels: NDArray[np.int64] | Sequence[int]
) -> dict[str, int]:
    """Label of each record, requiring that its segments agree.

    A record whose segments have different labels has no well-defined per-record
    split — stratifying by class would become impossible, and the leakage would come
    back in through the back door. The case is an error, not a silent warning.
    """
    y = np.asarray(labels, dtype=np.int64).ravel()
    if len(records) != y.size:
        raise ValueError(f"{len(records)} records for {y.size} labels")
    out: dict[str, int] = {}
    for name, lab in zip(records, y.tolist(), strict=True):
        prev = out.setdefault(name, int(lab))
        if prev != int(lab):
            raise ValueError(f"record {name!r} has labels {prev} and {int(lab)}")
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
    """Train/test splits **by record**, repeated and reproducible.

    ``records`` is the identifier of the source record of each segment; segments of
    the same record always fall on the same side. Splitting by segment is leakage:
    segments of one record share mounting, shaft speed and label, and the classifier
    learns to recognize the record instead of the fault.

    ``test_fraction`` is the fraction of **records**, not of segments — the records
    have different segment counts, and the fraction of segments fluctuates with the
    draw. That fluctuation is part of what one wants to measure.

    Declared stratification behavior, so that it does not come as a surprise:

    * each class contributes ``max(min_test_per_class, round(test_fraction * n_c))``
      test records, capped at ``n_c - 1`` — training never loses a whole class;
    * a class with **a single record** always stays in training, and its index
      appears in `RecordSplit.classes_never_tested`. The alternative would be to
      alternate the record between training and test, and then the model never sees
      the class in half of the splits, which is worse and harder to notice;
    * with ``stratify=False`` records are drawn without looking at the class. This is
      the E04 regime, and it is what allows a test set made entirely of one class —
      and a single-class test set makes the balanced accuracy of a constant
      predictor of that class equal 1.0, because the only class with support is
      predicted entirely correctly (a constant predictor of any other class gets 0.0).

    ``min_test_per_class`` **only exists under stratification**, and that is why the
    default value is ``None`` instead of ``1``: passing it with ``stratify=False`` is
    an error, not a parameter silently ignored. It was exactly this silence that
    produced single-class test splits in an arm whose config declared
    ``min_test_per_class: 1``. Under stratification, ``None`` is equivalent to ``1``.

    The seeds come from ``SeedSequence(master_seed).generate_state(n_splits)``: the
    same master seed gives the same splits, and each split carries its own integer
    seed in the CSV.
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if not stratify and min_test_per_class is not None:
        raise ValueError(
            "min_test_per_class only applies under stratification; with stratify=False "
            "the draw does not look at the class and the parameter would be silently ignored"
        )
    min_per_class = 1 if min_test_per_class is None else int(min_test_per_class)

    y = np.asarray(labels, dtype=np.int64).ravel()
    per_record = record_labels(records, y)
    names = sorted(per_record)  # canonical order: the draw does not depend on the input order
    if len(names) < 2:
        raise ValueError(f"per-record split requires at least 2 records, got {len(names)}")
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
            raise ValueError(f"split {i} left one of the sides empty")

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
