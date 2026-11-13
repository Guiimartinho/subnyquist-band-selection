# E21 - MAFAULDA: is the selector also captured, and does a calibration test predict where?

Opened on 2026-11-11 by [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md). It is the
generalization of Part 1, and it depends on the mechanism established in
[E20](../E20-fold-zone-rejection/README.md).

## Context

E20 showed, on Ottawa (outer race), that the accuracy valley from 3× to 6× comes from out-of-band
impulsive content that folds onto the kurtogram's search range and captures it. Two mitigations
fix it: an anti-alias filter, or a search ceiling below the range where the fold lands. This holds
for a single rig, with one bearing per fault and an inverter-driven shaft.

For Part 1 of the paper to hold, the phenomenon has to appear in another dataset. There also has
to be a way to **predict, without knowing the fault**, which rates are dangerous. This is the
defensible remainder of Contribution 2 of ADR 0005.

## Hypothesis

**H1.** On MAFAULDA (outer race, 372 sequences), under pure decimation, the order accuracy is not
monotonic in the factor: there is at least one factor with accuracy **at least 15 points below**
a larger factor, and the ideal anti-alias filter removes that valley.

**H2.** The rate-admissibility test (TAT) predicts these errors. The TAT is computed only on the
**first half** of each sequence, at full rate. It marks a factor as inadmissible when the band
chosen by the kurtogram after decimation has more than half of its energy coming from folded
content.

## What would refute it

- **H1 is refuted** if the accuracy is monotonic, to within 5 points, at every ceiling. The
  phenomenon then belongs to the Ottawa rig, and Part 1 is restricted to it (ADR 0006).
- **H2 is refuted** if, in the cells where the TAT says "inadmissible", the accuracy on the second
  half is **not** lower than in the admissible cells of the same factor (difference ≤ 10 points,
  or a per-sequence bootstrap interval crossing zero).
- **H2 is also refuted** if the balanced accuracy of the TAT, predicting error versus hit on the
  second half, falls below 0.65.

## Baseline

The same pipeline as E07 and E20 (restricted kurtogram, envelope, order spectrum, estimate without
prior), with the ideal anti-alias filter as the ceiling that measures what the fold costs. For the
prediction, the baseline is **chance stratified by factor**: predicting error with the factor's
marginal error rate.

## Ground truth

- **Fault order:** BPFO = 2.998, from the documented geometry, checked at ingestion. It is
  calibrated per sequence at full rate, on the first half, refining ±12 % around the nominal value.
- **Truth of each sequence:** the median of the **other** sequences of the same bearing
  (leave-one-out).
- **Rotation speed:** comes from the tachometer, with 1 pulse per revolution, hysteresis and
  lockout (`snq_data.mafaulda`).
- **Hit:** |estimate − truth| < 0.06 order.

The 51.2 kHz rate is the one inferred at ingestion. The order analysis is immune to it.

## Procedure

1. For each outer-race sequence, use the **radial accelerometer of the faulty bearing**.
   Split it into calibration (1st half) and test (2nd half).
2. **Calibration**, at full rate: compute the calibrated order and, for each factor and ceiling,
   the TAT.
   - Decimate the 1st half.
   - Run the kurtogram and obtain the band B.
   - Compute the folded share `1 − E_aaf(B)/E(B)`, where `E_aaf` is the energy of B in the version
     filtered before decimating.
   - The factor is inadmissible when that share exceeds 0.5.
3. **Test:** on the 2nd half, for each intervention (`none`, `aaf_ideal`), factor and ceiling, run
   the pipeline and record hit, band and kurtosis.
4. **Analysis:**
   - accuracy curve by factor and ceiling;
   - accuracy by TAT (admissible or inadmissible) within the factor;
   - balanced accuracy of the TAT, with a per-sequence bootstrap.

## Metrics

- order accuracy, `sem_prior` (without prior), by factor × ceiling × intervention, with `n`;
- TAT: folded share, prediction, and balanced accuracy against the observed error;
- measurement/prior attribution: not applicable (no linear operator).

## Result

Run on 2026-11-11 on the server (372 sequences, 22,320 rows, corrected DSP). **Inconclusive by
construction: the metric is not valid on this dataset.**

**What was measured.**
- **Accuracy at full rate, without intervention:** 35 % on the *underhang* bearing and 3 % on the
  *overhang* bearing. The ideal anti-alias filter does not change the picture, and no factor shows
  the H1 valley (*underhang*: 30 to 53 % across the factors; *overhang*: 1 to 10 %).
- **TAT:** balanced accuracy of 0.47 (95 % CI 0.46 to 0.49). The inadmissible cells score 10.8
  points **more** than the admissible ones (CI −18.5 to −4.2).

**Why this tests neither H1 nor H2.** In this bearing, BPFO = 2.998 and BPFI = 5.002 are, in
practice, **integer orders**: they coincide with the 3rd and 5th shaft harmonics.
- At full rate, 59 % of the estimates fall within ±0.03 of an integer order. The most frequent
  ones are 3, 32, 2, 5 and 4.
- The "calibrated" orders spread over the ±12 % window (2.64 to 3.35; only 11 % within ±0.01 of
  3.000).

The "hit" therefore measures whether the pipeline found the 3rd shaft harmonic, and not the fault.
Since the spectrum is dominated by the integer shaft harmonics, the baseline itself does not
diagnose. Without a valid metric, the question about undersampling does not arise here.

**What was surprising.** The most widely used multi-fault dataset with a tachometer has the BPFO
order on a shaft harmonic. This adds to the real rate of 51.2 kHz, and not 50 kHz, found at
ingestion. Both facts are worth a note in the paper.

**What remains from the ingestion:** the 51.2 kHz rate, the tachometer screening and the loader.
MAFAULDA may serve another task (detection against the normal condition, or ball fault with
BSF = 1.871), but **not** BPFO order estimation.

**Consequence (ADR 0006, update of 2026-11-11):** the generalization of Part 1 moves to **E23**,
which injects a controlled impulsive interferer into a dataset with non-integer fault orders
(CWRU 48k, BPFO 3.585) and, if possible, uses a second real dataset with that property
(Paderborn).
