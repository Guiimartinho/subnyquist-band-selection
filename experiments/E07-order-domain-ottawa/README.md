# E07 — undersampling degradation in the order domain

## Hypothesis

The fault order survives undersampling far beyond what the E02 curve suggested,
because E02 measured the arithmetic boundary of CWRU and not that of the method. Under
variable rotation, with an angular reference, the diagnostic information is in the **order** — which
is a constant of the geometry — and not in the frequency in hertz, which moves with the operating
regime.

## What would refute it

The hit rate falling monotonically with the undersampling factor, as it fell on CWRU 12k.
**It did not fall.**

## Data

University of Ottawa 2018 (Mendeley `v43hmbwxpm`, CC BY 4.0). 24 fault records
— 12 inner race, 12 outer race — in 4 variable-rotation scenarios,
200 kSPS, accelerometer on channel 1 and encoder on channel 2. Three disjoint windows of
2.5 s per record: **72 measurements per factor per mode**, against 15 in E02.

The encoder has **1024 pulses per revolution**, a value not documented by the authors and
determined empirically.

## The encoder does not go through the ADC

Pulse counting on a digital input — TIMER/GPIOTE on the nRF54L15 — does not consume
sampling budget. What is decimated in this experiment is only the vibration; the rotation
profile follows along because it varies slowly, and the CSV records in each row how many
points per revolution remain. Segments with fewer than two points per angular
increment are discarded instead of being interpolated blindly.

## Result (2026-09-04) — the curve does not go down

Fault-order hit rate, tolerance 0.06 orders, 72 measurements per point:

| factor | effective rate | no prior | nominal prior | calibrated prior |
|---|---|---|---|---|
| 1× | 200 kSPS | 78% | 50% | **83%** |
| 2× | 100 kSPS | 74% | 50% | 78% |
| 4× | 50 kSPS | 51% | 49% | 56% |
| 8× | 25 kSPS | 76% | 50% | 82% |
| 16× | 12.5 kSPS | 68% | 50% | 82% |
| 24× | 8.3 kSPS | 74% | 50% | 86% |
| 32× | 6.25 kSPS | 78% | 50% | **88%** |
| 48× | 4.17 kSPS | 74% | 50% | 81% |
| 64× | 3.12 kSPS | 64% | 51% | 73% |

**From 200 kSPS to 3.12 kSPS — factor 64 — the hit rate does not degrade.** At 32× it is higher
than at the full rate, and the difference is within the Wilson interval: what is measured is
absence of degradation, not improvement.

The `e07_spectra` figure shows why: the spectral landscape is unrecognizable between
200 kSPS and 3.1 kSPS, and the order-5.38 line with its second harmonic at 10.76 stays
exactly where it was (the figure regenerated in 2026-11-13, from the rerun with the corrected
ruler, labels the calibrated order of I-B-1 as 5.39). The fault lives in the envelope modulation, not in the carrier, and
folding moves the carrier without moving the modulation.

## Re-run of 2026-11-11, with the corrected DSP

The 2026-11-10 audit corrected the kurtogram (no circular seam), the envelope and the
angular resampling (with anti-aliasing). With the corrected core, `run.py --config
config/base.yaml` was re-run on the server. `results/metrics.csv` and
`results/manifest.json` are from this run, and the 2026-09-04 table above stays as a record.

| factor | effective rate | no prior | nominal prior | calibrated prior | n |
|---|---|---|---|---|---|
| 1× | 200 kSPS | 79.2 % | 50.0 % | 84.7 % | 72 |
| 2× | 100 kSPS | 73.6 % | 50.0 % | 77.8 % | 72 |
| 4× | 50 kSPS | **52.8 %** | 48.6 % | 55.6 % | 72 |
| 8× | 25 kSPS | 79.2 % | 50.0 % | 83.3 % | 72 |
| 16× | 12.5 kSPS | 75.0 % | 50.0 % | 81.9 % | 72 |
| 24× | 8.33 kSPS | 76.4 % | 50.0 % | 86.1 % | 72 |
| 32× | 6.25 kSPS | 77.8 % | 50.0 % | 90.3 % | 72 |
| 48× | 4.17 kSPS | 72.2 % | 50.0 % | 80.6 % | 72 |
| 64× | 3.12 kSPS | 64.4 % | 51.1 % | 73.3 % | 45 |

- **Only `sem_prior` (no prior) measures estimation** (T0.2). The two columns with a prior choose
  within a set that contains the right answer.
- **The `sem_prior` column changes by at most 5.6 points against 2026-09-04** (16×: 68.1 → 75.0 %).
  The 4× dip stays: the outer race has 25, 21, **6** and 24 hits out of 36 at 1×, 2×, 4× and 8×,
  and the inner race 32, 32, 32 and 33.
- **"At 32× the hit rate is higher than at the full rate" no longer holds.** It is now 77.8 %
  against 79.2 %, within the Wilson interval. The reading "absence of degradation up to 32×" holds.
- **The captor behind the dip was isolated (E20 and adversarial review).** It is a train of
  single-sample spikes in the raw record, also present in the healthy records. The ideal
  anti-alias filter or the removal of the spikes before decimating bring the outer race at 4×
  back to 25–26 of 36. See `experiments/E20-fold-zone-rejection/README.md`.

## The catalog prior errs, and errs silently

The ER16K geometry (9 balls, ball diameter 0.3125", pitch diameter 1.516") predicts
BPFO = 3.573 and BPFI = 5.427. Measured at the full rate on the 24 records:

| | nominal | calibrated | deviation |
|---|---|---|---|
| BPFI (inner race) | 5.427 | 5.401 | **−0.5%** |
| BPFO (outer race) | 3.573 | 3.414 | **−4.4%** |

The deviation is consistent across the 12 records of each condition and depends on the scenario:
the scenarios that start by accelerating (A, C) give systematically higher orders than
the ones that start by decelerating (B, D) — 5.42–5.43 against 5.38 on the inner race. The
mechanism was not determined here. The measured sum BPFO + BPFI = 8.82 violates the
geometric identity BPFO + BPFI = n by 2%, which rules out pure cage slip as the sole
explanation.

**The failure mode matters more than the number.** With the nominal prior, the outer-race
records do not produce a poor BPFO estimate: they produce **1.985** in 215
of 310 cases (69 %) — the fifth harmonic of the nominal FTF, 5 × 0.397 = 1.985, close to
shaft order 2. (The denominator published earlier, 263, was a leftover from a run prior
to the truncation at 64×; recounted in `results/metrics.csv` it is 310.) The nominal candidate
3.573 has no energy, so the score picks another family and **reports the wrong
fault with confidence**. A prior misaligned by 4.4% does not degrade gracefully; it
swaps the diagnosis. It is exactly the good-looking hallucination that the
attribution rule in `CLAUDE.md` exists to catch.

A caveat that cannot stay implicit: the nominal prior "gets" the inner race right
only because |5.427 − 5.401| = 0.026 fits within the 0.06 tolerance. It is a hit due to tolerance
slack, not due to a correct prior. Hence the 50% pinned at every factor — 12
inner records that barely scrape through, 12 outer ones that always miss.

## The dip at 4× is band choice, not the rate

At 4× the hit rate drops to 56% and returns to 82% at 8×. The cause is measured:

- At 4× the kurtogram chooses a band above 12 kHz in **91%** of the segments.
  At 8× and beyond, that band no longer exists (effective Nyquist below 12.5 kHz) and the
  hit rate recovers.
- Of the 32 errors at 4×, **22 come from the 12.8–25 kHz band**. The hits have a median kurtosis of
  122; the errors, 26.7.
- At 50 kSPS the content at 75–87 kHz folds exactly onto 12.8–25 kHz. Measuring the
  kurtosis of this fixed band in the same segment, it rises **2.4× to 4.8×** from 1× to 4×
  in the outer-race records — precisely the ones that have 1.8% to 3.6% of the energy above
  50 kHz and the ones that lose the most hit rate. In an inner-race record with 0.42% of the
  energy up there, the kurtosis falls instead of rising.

It is not the rate that fails: it is the interaction between the chosen rate and the content
outside the search band. It is the concern of ADR 0002 measured on real data, and it is the
empirical argument that the rate needs to be **chosen**, not merely reduced — which is
exactly the open problem that the project claims.

## The 25 kHz ceiling on the kurtogram inverted a conclusion

Without a ceiling, 12 of 12 outer-race records chose the 50–100 kHz band, with
kurtosis ~55, and the fault did not appear at any order. The easy reading would be "the outer
race of this dataset is not diagnosable", and it would be wrong: with the ceiling at
25 kHz, the same fault appears at order 3.4 in all 12 records.

**Correction of 2026-09-07 to the justification of this ceiling.** This section
claimed that the band above 20 kHz carries **0.1 %** of the energy and that the kurtosis there
measures numerical leakage. That 0.1 % was measured on **one** dry-run record and generalized to 24.
Recomputed over the **n = 24** fault records (vibration channel, FFT of the whole
record): **median 5.02 %** above 20 kHz, min 2.97 %, max 6.83 % — about 50× the
published value (`docs/lab-notebook/2026-09-07.md`). The **effect** of the ceiling remains
measured and does not change; what falls is the explanation. The band is not numerically empty,
so the 25 kHz ceiling is not a consequence of zero energy: it is a **band prior**, and that is
how it must be declared.

The kurtogram's `min_energy_frac` guard exists for this degeneracy, and the threshold of
1e-4 is too loose for this dataset.

## Ground truth, and why it is not circular

The reference order of a record is the **median of the orders measured at full rate
on the other records of the same condition** — never that of the record itself. Without this,
factor 1 would be a hit by construction in the no-prior mode. It is also what a real
commissioning would have: calibration on units of the same family, not on the equipment under test.

The spread between records of the same condition is ~0.035 in orders, and the 0.06
tolerance covers it. This spread is an irreducible error floor of this design.

## Negative control: does the estimator invent a fault where there is none?

A hit rate measured only on records that **have** a fault says nothing about false positives,
and an estimator restricted to a set of candidates always returns a candidate — it
has no way of answering "none". Without this control, the 88% at 32× would be compatible
with an estimator that always points to the same order.

The 12 healthy records (`H-*`) enter as negatives, and the separation is measured by AUC
— the probability that a faulty segment scores above a healthy segment at the
target order. 0.5 is absence of separation.

| factor | effective rate | AUC BPFI | AUC BPFO |
|---|---|---|---|
| 1× | 200 kSPS | 0.999 | 0.965 |
| 4× | 50 kSPS | 1.000 | 0.757 |
| 8× | 25 kSPS | 1.000 | 0.915 |
| 16× | 12.5 kSPS | 1.000 | 0.834 |
| 32× | 6.25 kSPS | 0.974 | 0.924 |
| 64× | 3.12 kSPS | 1.000 | 0.922 |

Values from the 2026-11-11 re-run, with the corrected DSP. The largest change against 2026-09-04
is BPFO at 64×, which fell from 0.995 to 0.922.

**The separation does not degrade.** On the inner race the AUC stays between 0.97 and 1.00 at all
factors. On the outer race it oscillates between 0.76 and 0.97 with no trend with the rate, and the
minimum is that of 4×. The dip of the outer race at 4× is the same band hijacking described above.

### The first metric said the opposite, and it was wrong

The initial version normalized the harmonic energy by the **median of the spectrum**. With it,
the separation seemed to collapse from 8× on and even **inverted** — AUC of 0.15,
with the healthy records scoring above the defective ones. It was an artifact: under strong
undersampling the spectrum of a healthy bearing becomes poor in spectral lines and the median
collapses, inflating the quotient without anything physical having changed. The floor moved
together with the metric.

The statistic used now is the **relative position of the line within the order range** — the
fraction of lines below the peak at the target order. Free of scale and of floor. The lesson holds
beyond this experiment: a metric normalized by a statistic of the signal itself
needs to be checked against undersampling, because the denominator is what changes
first.

## What this experiment does not support

- **It is not a classification comparison.** It measures order-estimation error and
  fault/healthy separation, not discrimination between fault types. E16 is the one that does
  multiclass classification with statistical power.
- **It does not measure consumption.** The effective rate is in every row of the CSV, but µJ per
  diagnosis does not exist in the code yet (E06).
- **It does not test non-uniform acquisition.** Uniform decimation only. Multi-coset and
  learned scheduling are Paper 2.
- **The 500 Hz floor in the kurtogram search consumes a growing share of the available
  band** as the factor rises: at 64× it cuts a third of what remains. The last two
  points of the curve carry this restriction together with the undersampling.

## Reproduction

```bash
uv run python experiments/E07-order-domain-ottawa/run.py \
    --config experiments/E07-order-domain-ottawa/config/base.yaml
uv run python papers/shared/figures/fig_e07_order_domain.py
uv run python papers/shared/figures/fig_e07_spectra.py --record I-B-1.mat
uv run python experiments/E07-order-domain-ottawa/control_healthy.py     --config experiments/E07-order-domain-ottawa/config/base.yaml
uv run python papers/shared/figures/fig_e07_control.py
```

The download is programmatic through the public Mendeley API, without registration.
`results/manifest.json`, as recorded, has the calibrated orders per record, the
SHA of the config and the version of each package — it predates the provenance fields
added on 2026-09-07 and was not rewritten, because rewriting a manifest without
repeating the measurement is inventing provenance. Whoever runs it again will get the new fields,
which the section "What the manifest now records" lists.

---

# Extension — `config/extended.yaml`

Section added on 2026-09-07. It does not replace anything above: the results table
of that section is still the one from the `base.yaml` run, recorded in `results/metrics.csv`.

## Hypothesis

After subtracting, **factor by factor**, the guard step measured at the nine factors that
the two runs have in common, the hit rate with the calibrated prior at factors **96, 128 and
160** stays **within 13 percentage points** of the corrected value at 64×. That
is: the E07 curve does not go down between 3.12 kSPS and 1.25 kSPS, and the stop at 64× was
indeed a guard artifact, not a knee.

**Where the 13 points come from.** With `n = 72` measurements per point and a hit rate around
0.80, the 95 % Wilson interval of a single point has a half-width of ±9.1 points, and
the difference between two independent points of `n = 72` has a normal half-width of
1.96·√(2·0.8·0.2/72) = **13.1 points**. Thirteen points is therefore the smallest drop that
this design distinguishes from sampling noise: below that there is no claim to make, and
above it there is. The threshold is a consequence of `n`, not a preference — doubling
`segments_per_record` would tighten it, at the cost of smaller windows and of a new confounder.

## What would refute this hypothesis

Four routes, each with the number that triggers it. None of them is "the result came out
ugly":

1. **Drop beyond 13 points.** If the corrected hit rate with the calibrated prior at 96×, 128×
   or 160× falls more than 13 points below the corrected one at 64×, the hypothesis is refuted:
   the knee exists and lies between 64× and 160×, and the publishable factor of E07 remains
   64. This is the route that decides whether the extension is worth a point on the curve or a
   footnote.
2. **Capture by a fixed order.** If, at any of the three new factors, the
   estimate of the calibrated-prior mode concentrates, in **more than half** of the
   segments, on a single order that is not the truth of the record — in particular on **1.985**,
   the FTF harmonic near order 2 (nominal FTF 0.397) that already captured 215 of 310
   outer-race cases under the nominal prior —, then the number is not a measurement of the fault
   order: it is a fixed point of the estimator. In that case the extended curve is null
   **regardless of the hit rate**, even if the hit rate is high. The CSV already has the
   `estimate_order` column; the test is a histogram.
3. **Non-additive guard step.** The subtraction of item 1 is only legitimate if the step
   measured at the nine repeated factors is an offset, and not a function of the factor. If
   it has a trend with the factor at those nine points, the extrapolation to 96–160 does not
   hold up, the design **does not separate** guard from rate, and the run must be declared
   inconclusive instead of published. The partial measurement already made (table below) shows the
   step varying between −1.4 and −11.1 points over six factors: this route is far from
   hypothetical.
4. **Unequal `n` between factors.** If any of the twelve factors enters with fewer than 72
   measurements, the comparison becomes a change of denominator — the defect already found in the
   E02 48k table. This route is **executed and closed**: `tests/test_guards.py` checks,
   on the 24 records × 3 real windows, that the twelve factors admit 72 of 72.

These four routes test the **extension**. The refutation recorded at the top of this
README — "the hit rate falling monotonically with the factor" — belongs to the base E07, has
already been answered with the curve in `results/metrics.csv`, and does not cover anything that
this section claims.

## Why it exists

The `base.yaml` sweep stopped on an **analysis criterion, not on a drop in
performance**. Counted in `results/metrics.csv`: 216 rows per factor (72 measurements × 3
modes) up to factor 48, **135 at factor 64** — 45 of the 72 — and **zero at factors 96 and
128**, which were in the list and never produced a measurement.

The last publishable point of the curve is therefore 64×, and it already enters with 62% of the
sample of the others. It is the most predictable attack from a reviewer, and it is the
intervention that moves the defensible number the most: it decides whether the factor stays at
6.4× or goes substantially beyond.

## The three guards, and which one actually blocks

`run.py` discards a segment on three tests, all in `guard_verdict`, in the order in
which they are applied (the first one to fire is the one that counts):

| order | guard | parameter |
|---|---|---|
| 1 | `n_samples < cfg["min_samples"]` | `min_samples` |
| 2 | `fs_eff/2 <= cfg["kurtogram"]["f_min_hz"]` | `f_min_hz` |
| 3 | `fs_eff/max(v) < 2·cfg["samples_per_rev"]` | `samples_per_rev` |

With `FS_HZ = 200000` and `segment_samples = 500000`, and using the two worst-case
rotation speeds — 1953.1 rpm, the maximum of record I-A-3 read from the `rpm_hi` column of
`results/metrics.csv`, and 1674.1 rpm, the maximum measured within a segment that is
actually analyzed (the peak of I-A-3 falls in the last 500000 samples, which
`segments_per_record: 3` never reaches):

| factor | `fs_eff` | `n = ceil(500000/f)` | `fs_eff/2` | pts/rev @1953.1 | @1674.1 |
|---|---|---|---|---|---|
| 64 | 3125.00 Hz | 7813 | 1562.50 Hz | 96.00 | 112.00 |
| 96 | 2083.33 Hz | 5209 | 1041.67 Hz | 64.00 | 74.67 |
| 128 | 1562.50 Hz | 3907 | 781.25 Hz | 48.00 | 56.00 |
| 160 | 1250.00 Hz | 3125 | 625.00 Hz | 38.40 | 44.80 |
| 192 | 1041.67 Hz | 2605 | 520.83 Hz | 32.00 | 37.33 |
| 256 | 781.25 Hz | 1954 | 390.62 Hz | 24.00 | 28.00 |

Reading guard by guard:

- **`samples_per_rev: 64`** requires pts/rev ≥ 128. It is the one that discards 27 of the 72
  measurements of factor 64 (worst segment: 112.00) and that zeroes 96 (74.67) and 128
  (56.00). It is the dominant guard.
- **`min_samples: 4096`** zeroes 128 on its own: `ceil(500000/128) = 3907`.
  It does not touch 96 or 64.
- **`f_min_hz: 500` blocks neither 96 nor 128.** The guard asks for
  `fs_eff/2 > 500`, and at factor 128 that is `781.25 > 500`. It only starts to bind
  from factor 200 on (`fs_eff/2 = 500.0`). The reading that the 500 Hz floor zeroed
  both factors — recorded in `docs/plano-de-experimentos.md`, T1.1 — **does not
  hold up in the code**.

`extended.yaml` therefore relaxes only two guards: `min_samples: 4096 → 2048` and
`samples_per_rev: 64 → 16`. **`f_min_hz` stays at 500 Hz**, deliberately.

This is **executed**, not predicted, and what was executed is **segment
admission** — how many of the 72 windows each factor accepts — not the hit rate.
`tests/test_guards.py` applies `run.guard_verdict` (the same function that `run.py` uses in the
loop) to the 24 records × 3 real windows and checks two things:

- under `base.yaml`, the result is 72 measurements per factor up to 48, **45** at 64, **zero**
  at 96 and 128 — which is, row by row, what `results/metrics.csv` contains (216 rows per
  factor up to 48, 135 at 64), and that is what authorizes using the same function to talk about
  factors not yet run;
- under `extended.yaml`, **72 of 72 at the twelve factors**, including 96, 128 and 160.

No point of the extended curve enters with a reduced `n` — which closes refutation route 4 and
avoids the change of denominator that already contaminated the E02 48k table. The
hit rate at these factors remains **not measured**: it is what the run exists to find out.

## The caveat that cannot stay implicit

**The extended run measures undersampling PLUS guard relaxation.** The two effects
are confounded, and the mechanism is concrete:

1. **Lowering `samples_per_rev` degrades the angular resolution.** With `spr = 16` the Nyquist
   of the **order** spectrum falls from 32 to 8. This changes the estimate through two
   independent paths: `score_fault` starts **folding** harmonics that previously
   fit below the Nyquist (`diagnosis.py:114-119`), changing the composition of the score;
   and `estimate_from_candidates` **discards** candidates above the Nyquist of the spectrum
   (`diagnosis.py:185`), shrinking the candidate set. Measured: of the 24
   candidates of the nominal prior, **23 survive under `spr = 64`** and only **12 under
   `spr = 16`**.
   What does *not* change is the step in orders: `order_spectrum` returns
   `rfftfreq(n, 1/spr)` with `n = revolutions·spr`, so the step is `1/revolutions`,
   independent of `spr` — checked at 0.0261 at factors 64 to 160.
2. **Lowering `samples_per_rev` moves THE TRUTH ITSELF.** The reference order is not
   external: it comes from the same angular resampling, in pass 1 of `run.py`, which uses the
   `spr` of the config. The extended run, therefore, does not compare estimates against the same
   ruler as the base run — it recalibrates the ruler. Measured by repeating pass 1 on the 24
   records under both `spr` (full rate, same kurtogram band):

   | | `spr = 64` | `spr = 16` |
   |---|---|---|
   | median of the calibration, inner race | 5.4008 | 5.4008 |
   | median of the calibration, outer race | 3.4142 | 3.4167 |

   Per record, the shift has a median of **0.005 orders** and a maximum of **0.095**
   (O-A-2), and **1 of the 24** moves beyond the 0.06 tolerance. The truth that enters the
   comparison is the leave-one-out median of 11 records, and that is why it moves
   little — at most 0.0025 orders, against a tolerance of 0.06 and a between-record spread
   of 0.035. The confounder exists, it is second-order **at this magnitude**, and it needs to
   be recounted if `spr` drops further.
3. **Lowering `f_min_hz` would let the kurtogram search near DC**, where the
   structural content is. This config does **not** lower it, and the caveat stays recorded
   for whoever is tempted to lower it: recomputed over the **n = 24** fault
   records, vibration channel, FFT of the whole record, the energy below 500 Hz has a
   **median of 14.86 %** (min 1.14 %, max 22.06 %) —
   `docs/lab-notebook/2026-09-07.md`. **It is not the 97 %** that the comment in
   `config/base.yaml` (97 % below 500 Hz) and the docstring of
   `packages/snq-data/src/snq_data/ottawa.py` (97 % below 50 Hz) claim; that
   number came from one record and was generalized to 24. Since the band below 50 Hz
   is contained in the one below 500 Hz, the 14.86 % median bounds both claims above.

**The sign of the bias is measured, and it is the opposite of what this section claimed.** The
previous version said that the smaller candidate set "points in the direction of improving the
hit rate" and that the effect is a constant offset. Both things were measured
directly — reprocessing the same segments under both `spr`, with the calibration of the
base run — and both are wrong. Hit rate of the calibrated-prior mode, **paired**
comparison (only the segments that pass both guards):

| factor | `spr = 64` | `spr = 16` | step | n |
|---|---|---|---|---|
| 1× | 83.3 % | 76.4 % | **−6.9** | 72 |
| 8× | 81.9 % | 77.8 % | **−4.2** | 72 |
| 16× | 81.9 % | 73.6 % | **−8.3** | 72 |
| 32× | 87.5 % | 76.4 % | **−11.1** | 72 |
| 48× | 80.6 % | 79.2 % | **−1.4** | 72 |
| 64× | 73.3 % | 71.1 % | **−2.2** | 45 |

(The `spr = 64` column reproduces the published table of the base run — 83, 82, 82, 88, 81,
73 % — which checks the procedure.) The relaxation **worsens** the hit rate at all six
measured factors, median −5.5 points, and the no-prior mode worsens more, from −4.4 to −22.2
points. The step is also **not constant**: it varies from −1.4 to −11.1 between factors, which
is refutation route 3 above and the reason why the subtraction has to be done factor by factor,
never with a single value.

**How to separate the two effects — it is part of the experiment, not a suggestion.**
`extended.yaml` re-runs the **whole** list of factors, not only the new ones, and the nine
factors already measured (1 to 64) are the control arm of this run. They appear twice:

- in `results/metrics.csv`, under `spr = 64` and `min_samples = 4096`;
- in `results/extended/metrics.csv`, under `spr = 16` and `min_samples = 2048`.

The row-by-row difference at these nine factors is **pure guard effect** — the rate is
identical, the segments are the same. Only after subtracting it, factor by factor, can the
drop between 64×, 96×, 128× and 160× be read as a rate effect. An extended curve
published without this control does not distinguish the two causes and must not be
used; if the control arm is not run alongside, there is no experiment, only a second
curve that cannot be compared with the first.

Confounders deliberately avoided: `segment_samples`, `segments_per_record`,
`tolerance_order`, `n_harmonics`, `score_harmonics`, `calibration_span`, `order_min`,
`f_min_hz`, `f_max_hz`, `max_level` and the nominal geometry stay **identical** to the base,
and so does `seed` (`run.py` does not use an RNG; keeping the value keeps the comparison exact).

## Why the list stops at 160×

It is not the points-per-revolution guard — with `spr = 16` factor 192 would still pass
(37.33 against the threshold of 32). It is the **kurtogram search window**, `[f_min_hz,
fs_eff/2]`, which collapses against the 500 Hz floor:

| factor | window | width | sub-band (level 4) | FFT bins per sub-band |
|---|---|---|---|---|
| 64 | 500–1562.50 Hz | 1062.50 Hz | 66.406 Hz | 166.0 |
| 96 | 500–1041.67 Hz | 541.67 Hz | 33.854 Hz | 84.7 |
| 128 | 500–781.25 Hz | 281.25 Hz | 17.578 Hz | 44.0 |
| 160 | 500–625.00 Hz | 125.00 Hz | 7.812 Hz | 19.5 |
| 192 | 500–520.83 Hz | 20.83 Hz | 1.302 Hz | 3.3 |

At 192 each sub-band of the finest level has 3 bins: band selection stops being a
measurement. The decline is already visible before that — the kurtosis of the chosen band in
I-A-3 segment 0 goes from 3.6 (64×) to 1.9 (96×), 2.1 (128×) and **0.5** (160×). And at 256
two guards zero the factor: `min_samples` fires first (`ceil(500000/256) = 1954`,
below the 2048 of `extended.yaml`) and, if it did not fire, `fs_eff/2 = 390.62 ≤ 500`
would zero it all the same. This is what `tests/test_guards.py` pins for factor 256.

This reformulates the caveat in the section "What this experiment does not support": the
500 Hz floor is not just a restriction that accompanies the last points — **it is the guard that
defines where this analysis chain ends**.

## Related correction, made in `extended.yaml`

`config/base.yaml` claims, under the label "Measured ceiling, not arbitrated", that above
20 kHz there is **0.1 %** of the energy. Recomputed over the **n = 24** fault records:
**median 5.02 %** (min 2.97 %, max 6.83 %) — about 50× the quoted value
(`docs/lab-notebook/2026-09-07.md`). The same claim was repeated above in this
README, in the section "The 25 kHz ceiling on the kurtogram inverted a conclusion", and **it was
corrected there too** — correcting only the config comment would leave the wrong number
alive in the document that people read.

The two wrong numbers of this family, and where each one still is:

| claim | where | measured over n = 24 |
|---|---|---|
| 0.1 % of the energy above 20 kHz | `config/base.yaml`, and earlier in this README | median **5.02 %** (2.97–6.83) |
| 97 % of the energy below 500 Hz | `config/base.yaml` | median **14.86 %** below 500 Hz (1.14–22.06) |
| 97 % of the energy below 50 Hz | docstring of `snq_data/ottawa.py` | contained in the row above, so ≤ 14.86 % |

The third row is outside the scope of this correction — `packages/` is not edited from
an experiment — and stays recorded as pending.

**The 25 kHz ceiling stays at 25 kHz and the reported effect remains measured**: without it,
12 of 12 outer-race records chose 50–100 kHz and the fault did not appear. What
falls is the justification. The band above 20 kHz is not numerically empty, so
"there the kurtosis measures numerical leakage" is left without basis, and the ceiling becomes what
it always was in practice: a **band prior**, declared as a prior and not as a
consequence of zero energy. `extended.yaml` carries the corrected comment.

`config/base.yaml` was **not** edited, on purpose: its SHA-256 is recorded in
`results/manifest.json` as the provenance of the base run, and changing it would invalidate that
record without changing anything that was measured.

> **Update 2026-11-13 (public release).** The comments of `config/base.yaml` were translated into
> English, so the file no longer has the SHA-256 recorded in `results/manifest.json`. The parsed
> configuration is unchanged, and the exact bytes that the manifest refers to are those of commit
> `54cd5c6`, its last change before the translation, which the repository history keeps. Its
> comments still carry the two wrong energy figures corrected above.

## Reproduction of the extension

```bash
# before the sweep: the guards, on the real data (skipped without the .mat files)
uv run pytest experiments/E07-order-domain-ottawa/tests -q

uv run python experiments/E07-order-domain-ottawa/run.py \
    --config experiments/E07-order-domain-ottawa/config/extended.yaml
```

Every parameter that the extension touches already lived in the config. The output goes to
`results/extended/`, following the pattern of `E02-bandpass-envelope-cwru/config/cwru48k.yaml`,
which writes to `results/48k/`; `results/metrics.csv` stays intact.

The test path needs to be explicit because `testpaths` covers `packages/`, and that is
deliberate: nothing in `packages/` imports `experiments/`. The test lives inside the
experiment and only imports its `run.py`.

### What the manifest now records

`write_manifest` recorded `git_sha`, `config_sha256`, package versions and
`platform.python_version()`. That **does not identify the code that ran**: the commit does not
describe a dirty tree, and the package version (`0.1.0`) does not change when the source changes.
Added:

| field | what it solves |
|---|---|
| `platform` | full `platform.platform()`, required by `experiments/_template/run.py:45-55` |
| `input_sha256` | SHA-256 of each `.mat` actually loaded, by name |
| `git_dirty` / `git_dirty_paths` | if true, `git_sha` does **not** describe what ran, and the dirty paths are named |
| `source_sha256` (+ `_files`, `_scheme`) | hash of the content of the imported sources: SHA-256 over `"<path>\n<sha256 of the file>\n"` for each `packages/*/src/**/*.py`, in POSIX path order. It changes when the source changes, with or without a commit |
| `runner_sha256` | SHA-256 of `run.py` itself, which the commit does not describe either when the tree is dirty |
| `samples_per_rev` / `min_samples` | the two relaxed guards, so that a comparison between `results/` and `results/extended/` does not depend on reopening the YAML files |

The base run's `results/manifest.json` predates these fields and was **not**
rewritten: it is the provenance of what was measured, and rewriting it without repeating the
measurement would invent provenance. The same holds for `config/base.yaml`, whose SHA-256 is
recorded there.

## Result of the extension (2026-11-11)

Run with the corrected DSP, `results/extended/`. The 72 measurements enter at all twelve factors,
which keeps route 4 closed.

### The pre-registered analysis: route 1 triggered, but the ruler moved

Each run was evaluated against its own truth, as recorded.
- **Paired guard step, `prior_calibrado` (calibrated prior):** −30.2 points on average (from −36.1
  to −20.8), with a slope of −0.8 ± 1.6 points per octave (95 % CI). By route 3 it is additive.
- **Corrected values:** 73.3 % at 64×, 57.9 % at 96×, 38.5 % at 128× and 30.2 % at 160×. The drop
  at 96× is 15.4 points, above the threshold of 13, and **route 1 triggers**.
- **Route 2 (capture by a fixed order):** does not trigger, narrowly. The most frequent order of
  `prior_calibrado` is ≈0.39 (FTF) in 24 %, 38 % and 49 % of the segments at 96×, 128× and 160×.

**The −30-point step is not the estimator's. It is the truth that moves.** Under `spr = 16` the
calibration shifts the reference order by a median of −0.0425 orders (from −0.065 to −0.015), and
18 of the 24 records move more than 0.03. The tolerance is 0.06.
- In the 23 segments that the base gets right and the extended gets wrong at 1× (`sem_prior`), the
  estimate and the band are practically the same in both runs, for example 5.429 against 5.429.
  What changes is the truth: 5.3808 → 5.3458 on the inner race and 3.4392 → 3.3742 in O-B.
- **The cause is the tie-break of `refine` (review of 2026-11-11, checked).**
  - The score is piecewise constant: `score_fault` takes the largest line within ±tol of each
    harmonic.
  - The `max()` of `refine` returns the **first** point of the maximum plateau.
  - With `spr = 64`, the plateau is 0.035 order wide in 22 of 24 records. With `spr = 16`, only
    the harmonics below order 8 fit, and the plateau widens to 0.115 (inner) and 0.055 (outer).
  - **By the center of the plateau, the shift between the two rulers is +0.0000 at the median**,
    and only O-A-1 and O-A-2 exceed 0.03. By the first point, it is −0.0325.
  - Measured by `e07_refine.py`, which exactly reproduces the calibrations of the two manifests.
- **The ruler of the base run is also biased.** It sits 0.0175 order below the center of the
  plateau in 22 of 24 records, almost a third of the tolerance. The estimates at 1× (about 5.43 on
  the inner race) match the center (5.443), not the first point (5.426). It is the same ruler as
  in E20 and E22, and the correction calls for re-running all three.

### Analysis with the fixed ruler (decided after seeing the shift)

The extended run was re-evaluated against the truth of the base run (`spr = 64`). That truth is per
record and does not depend on the factor, so it also holds for 96–160×.

| | 1× | 4× | 8× | 32× | 64× | 96× | 128× | 160× |
|---|---|---|---|---|---|---|---|---|
| `sem_prior`, extended, fixed ruler | 77.8 | 51.4 | 76.4 | 70.8 | 61.1 | 52.8 | **19.4** | **0.0** |

- **Paired guard step, `sem_prior`:** −3.5 points on average (from −8.9 to 0.0). It is small, but
  it grows with the factor: −1.0 ± 0.8 points per octave, with a CI that excludes 0. **By route 3,
  the subtraction of a single step does not hold up.**
- **Even so, there is a knee between 96× and 128×.** From 96× to 128× the raw hit rate falls 33
  points (52.8 → 19.4 %). A guard step that is −8.9 at 64× and grows by about 1 point per octave
  does not explain this. At 96× (2.08 kSPS) the corrected drop against 64× is 8.2 points, below
  the threshold; at 128× (1.56 kSPS) it is 41.5.
- **Mechanism (corrected by the review of 2026-11-11, checked):**
  - The `f_min_hz` = 500 Hz floor shrinks the search range to [500; fs_eff/2]: [500; 781] Hz at
    128× and [500; 625] Hz at 160×.
  - The kurtogram partition then generates bands **narrower than the fault frequency in
    Hz**. The median width is 271 Hz at 96×, 141 Hz at 128× and 10.4 Hz at 160×. At 128×, 12 % of
    the bands are narrower than the fault frequency, all of them wrong. At 160×, 100 %.
  - The envelope of a band of width W only has content up to W. The fault line disappears, and the
    estimate lands next to `order_min` = 2.0: 47 % of the estimates in [2.0; 2.2) at 128× and 83 %
    at 160×.
  - It is the capture by a fixed order of route 2, which also held for `sem_prior`.
  - The previous sentence ("the search range no longer contains the resonance") was wrong: under
    pure decimation the resonance folds into the baseband.
  - It is a wall of the pipeline (the floor and the selector's partition), not of the rate.
- **`prior_calibrado` with the fixed ruler** worsens (mean step of −38.2). This is what is expected
  of a choice among candidates around a calibration that moved, and it confirms the invalidation
  of T0.2.

**Conclusion for T1.1.** The publishable factor of E07 remains 64× (3.12 kSPS) in `sem_prior`.
The extended curve shows a knee between 2.08 and 1.56 kSPS. It belongs to the pipeline: the narrow search range and the bands narrower than the fault harmonics that the score uses (corrected on 2026-11-12, from `results/extended/metrics.csv`: the median fault frequency is 81 Hz; at 128× the median band, of 141 Hz, is still wider than it, but narrower than the three harmonics of the score, which add up to 243 Hz; only at 160×, with 10.4 Hz, does it become narrower than the fault frequency itself).
A method gain at these factors
would be against a straw man.

The run "with `spr = 64` and only the floor varying", proposed earlier, is **infeasible** at 128×:
the guard requires 128 points per revolution, and factor 128 gives 56. The missing control is a
different one: the ruler by the center of the plateau, and the kurtogram floor or minimum band width
varying, with `spr = 16`.

## Re-run with the corrected ruler (2026-11-11, afternoon)

Run at 12:43 UTC on the server, with two changes:
- truth by the **center of the plateau** (`snq_recon.refine_order`), and no longer by the first
  point (bias of −0.0175 order);
- **local ruler** recorded alongside: `truth_local_order` and `hit_local`, the full-rate line of
  the segment itself, for scoring only.

The local ruler comes from the review of the E15 protocol, checked: the BPFO order of the outer
race falls by about 0.3 per 10 Hz of rotation within the record (O-A-1: 3.519 at 16.7 Hz → 3.322
at 25.8 Hz), and a truth that is constant per record misses the line in the segments with a
different rotation speed.

`sem_prior`, hit rate (%):

| factor | 1× | 2× | 4× | 8× | 16× | 24× | 32× | 48× | 64× |
|---|---|---|---|---|---|---|---|---|---|
| ruler by the center of the plateau, leave-one-out (n = 72) | 86.1 | 76.4 | **56.9** | 84.7 | 76.4 | 84.7 | 86.1 | 83.3 | 80.0 |
| local ruler (n = 71) | 93.0 | 80.3 | **57.7** | 90.1 | 83.1 | 87.3 | 90.1 | 85.9 | 81.8 |

- **Per race, local ruler:** the outer race has 31, 22, **6** and 28 of 35 at 1×, 2×, 4× and 8×;
  the inner race 35, 35, 35 and 36 of 36. The 4× dip stays under both rulers.
- **The `prior_calibrado` column** stays equal to or below the `sem_prior` one (1×: 84.7 %).
- **Negative control (AUC BPFI / BPFO):**

  | factor | 1× | 2× | 4× | 8× | 16× | 24× | 32× | 48× | 64× |
  |---|---|---|---|---|---|---|---|---|---|
  | BPFI | 0.999 | 0.865 | 1.000 | 1.000 | 1.000 | 1.000 | 0.974 | 0.985 | 1.000 |
  | BPFO | 0.943 | 0.804 | 0.765 | 0.899 | 0.784 | 0.931 | 0.912 | 0.940 | 0.927 |

**Extension (`spr = 16`), with the corrected ruler.**
- **The paired guard step falls to −2.9 points on average by the center ruler, and to −0.2 by the
  local ruler** (from +1.4 to −4.8; slope of −0.73 ± 0.43 per octave). The −30 points from before
  came from the tie-break of `refine`, as the review predicted.
- **Hit rate by the local ruler:**

  | factor | 64× | 96× | 128× | 160× |
  |---|---|---|---|---|
  | hit rate | 78.6 % | 64.3 % | **22.9 %** | 0.0 % |

- **Knee.** The corrected drop from 64× to 96× is close to the pre-registered threshold of 13
  points: about 17 by the mean step, about 12 extrapolating the slope. 128× is a collapse.
- **Mechanism** (previous section): bands narrower than the three fault harmonics of the score,
  because of the 500 Hz floor (corrected on 2026-11-12, from `results/extended/metrics.csv`: the median fault frequency is 81 Hz; at 128× the median band, of 141 Hz, is still wider than it, but narrower than the three harmonics of the score, which add up to 243 Hz; only at 160×, with 10.4 Hz, does it become narrower than the fault frequency itself).
