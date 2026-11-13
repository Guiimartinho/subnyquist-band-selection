# E23 - Impulsive interferer injected into CWRU 48k: is the capture predictable and avoidable?

Opened on 2026-11-11 by the update of [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md)
after E21. It is the generalization of Part 1 by controlled intervention.

## Why this experiment

- **E20:** on Ottawa, the folding of out-of-band impulsive content (probably electromagnetic
  interference from the inverter) captures the kurtogram, from 3× to 6×.
- **E21:** MAFAULDA does not serve to test the generalization, because in it BPFO coincides with
  the 3rd harmonic of the shaft.
- **E02:** CWRU 48k, whose fault orders are not integers (BPFO 3.585, BPFI 5.415), falls
  monotonically under undersampling, because it has no high-band interferer.

Injecting a controlled interferer into CWRU separates three questions of the paper:
1. given an impulsive interferer, does the capture happen **where the fold arithmetic predicts**,
   and only there?
2. is it the **impulsiveness** that captures, and not the power?
3. does a full-rate calibration (TAT, the rate-admissibility test) or do the mitigations avoid it?

## Design

- **Signal:** the 13 48 kHz fault records of CWRU (DE race, catalog `smith2015cwru`; 12
  BPFO and 1 BPFI), with constant rotation at the record's rpm. The 1st half of each record is
  calibration and the 2nd is test. The truth is the fault frequency from the 6205 geometry, with a
  tolerance of 2 Hz, as in E02.
- **Pipeline:** the E02 one in hertz (kurtogram + envelope + estimate without a prior), with the
  kurtogram search range restricted to [500 Hz, ceiling].
- **Interferer:** bursts `exp(-t/τ)` on a carrier fc, with τ = 0.5 ms, at 97.7 Hz. This rate
  is not an order of the shaft or of the fault.
  - fc ∈ {13, 16, 19, 22} kHz, all above the 10 kHz ceiling;
  - relative power ∈ {−10, 0, +10, +20} dB, measured against the power of the band that the
    kurtogram chooses at full rate, in the calibration without injection;
  - the realization of the bursts is the same at all powers (only the scale changes) and is fixed
    per record and carrier.
- **Impulsiveness control:** the **phase-randomized surrogate** of the same realization. It has the
  same amplitude spectrum, bin by bin, and a mean kurtosis close to 3. The only difference between
  the two arms is the impulsiveness.
- **Factors:** 1 to 8, 10 and 12.
- **Crossed ceilings in the uniform arm:** 10, 5 and 2.5 kHz, limited to fs_eff/2. At the same
  factor and with the same signal, the image of the interferer falls inside the search with one
  ceiling and above it with another. It is the contrast that tests the specificity of the
  prediction.
- **Arithmetic prediction:** the fraction of the interferer's power whose image, under `x[::M]`,
  falls in the search range, computed from the spectrum of the injected realization. A fraction of
  at least 0.5 is predicted capture; below 0.05, no predicted capture; in between, borderline,
  which stays outside the binary criterion. The calculation includes the Lorentzian tail of the
  bursts, which leaks into the search even without folding; the test compares it with the analytic
  reference.
- **Direct capture indicator:** the estimate lands on 97.7 Hz or on a harmonic up to the 3rd, that
  is, the selector started looking at the interferer.
- **Arms:** `uniform` (three ceilings), `aaf_ideal` (ideal anti-aliasing filter),
  `uniform_ceiling` (ceiling set by the TAT) and `tat_rate` (factor set by the TAT). The
  uniform oracle comes out in the analysis.

### Design revisions before the full run

None of these changes used the result of the full run:

1. **Control.** The Gaussian noise with a rectangular spectrum and the same power became the
   phase-randomized surrogate. The rectangular noise did not have the same spectrum as the bursts,
   which have a Lorentzian spectrum.
2. **Prediction.** Instead of the image of the carrier, the folded share. The Lorentzian tail makes
   the carrier a poor prediction: with the image at 0 Hz, about 60 % of the power still falls in
   the search.
3. **Crossed ceilings and τ.** The prediction table, computed a priori with the arithmetic only,
   had 30 to 35 of 40 cells with predicted capture and almost no folded cell without capture.
   Without this there would be no specificity test.
4. **Power extended to +20 dB.** In the smoke run (2 records, 16 kHz, 0 dB), the predicted cells
   did not capture: the kurtogram started to include the image, but the BPFO line remained
   dominant. The CWRU 48k faults are strong.

### Spike arm and removal arm (added on 2026-11-11, before any result)

**Motivation.** The adversarial review of E20 showed that the real captor in Ottawa is a train of
isolated single-sample spikes (1300–2500/s, negative deviation in 96 % of the cases, at 16–18
MAD), not a band-limited burst. The bursts of the original design test a different interferer.

**Design** (`config/spikes.yaml`, separate run in `results-spikes/`):
- **Interferer:** isolated single-sample spikes, from a Poisson process, with negative sign.
  - rates ∈ {500, 1300, 2500}/s;
  - amplitudes ∈ {8, 16, 32} × the MAD of the deviation of each sample from the mean of its
    neighbors, measured in the calibration.
- **`despike` arm:** the review's criterion, applied at full rate before decimating. A sample
  that deviates more than 10 MAD, with both neighbors deviating in the opposite direction, becomes
  their mean.
  - **Limitation:** two spikes two samples apart escape, about 3 % at 1300/s and 48 kHz.

**H-spikes (pre-registered).** With spikes of Ottawa parameters (1300/s, 16 MAD):
- the uniform arm, at the reference ceiling, drops at least 20 points against the same factor
  without injection in at least one factor ≥ 2;
- in the `aaf_ideal` and `despike` arms, the drop is at most 5 points at all factors.

The dose-response in amplitude and rate is descriptive.

**What would refute it.** No factor ≥ 2 drops 20 points at any amplitude. In that case the spikes,
on their own, do not reproduce the capture on CWRU, and the Ottawa phenomenon depends on another
property of that dataset. A result between the two thresholds is inconclusive.

## Pre-registered hypotheses and criteria

The criteria are in `config/base.yaml`, in the `analysis` section, and in `analyze.py`.

- **Evaluable cell:** a (carrier, factor, ceiling) combination of the uniform arm in which, without
  injection, the hit rate is at least 50 %. The **drop** is the hit rate without injection minus
  the hit rate with injection, in points, paired by record.
- **H1 (the fold arithmetic predicts the capture).**
  - (a) Sensitivity: at the maximum power, at least 80 % of the evaluable predicted cells drop
    20 points or more.
  - (b) Specificity: at all powers, at most 20 % of the folded cells (factor ≥ 2)
    with no predicted capture drop 20 points or more.
- **H1' (it is the impulsiveness).** P\* is the lowest power at which the mean drop of the
  impulsive, in the predicted cells, reaches 20 points. At P\*, the mean drop of the surrogate in
  the same cells is less than half that of the impulsive.
- **H2 (the TAT avoids the capture).** The TAT, as a predictor of a hit, has a balanced accuracy
  of at least 0.8 over all cells (record, condition, factor) of the uniform arm at the reference
  ceiling. And `tat_rate` stays, on average, at most 5 points from the uniform oracle.

## What would refute

- **H1** is refuted if:
  - (a) less than 50 % of the predicted cells drop at the maximum power; or
  - (b) more than 50 % of the folded cells with no predicted capture drop, at some power.
- **H1'** is refuted if, at P\*, the drop of the surrogate is at least 0.8 times that of the
  impulsive.
- **H2** is refuted if the balanced accuracy of the TAT is lower than 0.65, or if `tat_rate`
  stays more than 5 points below the oracle on average.
- **Between the support and refutation thresholds, the result is reported as inconclusive.**

## Baseline

The E02 pipeline without injection, at the same factor and with the same ceiling.

## Ground truth

The fault frequency from the 6205 (DE) geometry and the record's rpm. The interferer and its
image are known by construction.

## Declared limitations

- There are 13 records, with one inner-race fault, and the confidence interval is by record.
- The interferer is synthetic. The generalization to a second real dataset (Paderborn) is left
  for later.

## Result of the base run (2026-11-11; bursts and surrogate)

Done on the server from 06:20 to 10:58 UTC: 13 records × 33 conditions, 429 tasks, no failures,
22,308 rows. The analysis (`analyze.py`), written before the run, is in `results/analysis.json`,
with a bootstrap CI by record.

The cells of the uniform arm split into 48 with predicted capture, 19 with no predicted capture
and 21 borderline.

| pre-registered criterion | measured | verdict |
|---|---|---|
| H1a: at the maximum power (+20 dB), ≥ 80 % of the predicted cells drop ≥ 20 points | **85.4 %**, CI [47.9; 89.6] % | supported on the point estimate; the CI goes below the refutation threshold (50 %) |
| H1b: ≤ 20 % of the folded cells with no predicted capture drop, at all powers | **0 %** at all | supported, with only **7** evaluable cells |
| H1': at P\*, surrogate/impulsive < 0.5 | P\* = +20 dB; ratio **0.14**, CI [−0.01; 0.23] | supported |
| H2: balanced accuracy of the TAT ≥ 0.8 | **0.69**, CI [0.63; 0.79] | inconclusive (between 0.65 and 0.8) |
| H2: `tat_rate` at most 5 points below the oracle | **−5.2 points**, CI [−8.4; −2.7] | refuted, narrowly; worst cell −69 |

**Dose-response.** The mean drop in the predicted cells is about 0 points at −10 dB and at 0 dB, 9
points at +10 dB and 41 points at +20 dB. The CWRU 48k faults are strong, and the interferer only
captures the selector with a power about 100 times that of the fault band.

**Descriptive** (reference ceiling, factors ≥ 2):
- **Estimate on the burst rate, the direct sign of capture,** with the impulsive at +20 dB:
  44 % in the predicted cells, 28 % in the borderline ones, 6 % in the ones with no predicted
  capture. With the surrogate, ≤ 0.3 %.
- **Hit rate of the uniform arm:** 94 % without injection, 54 % with the impulsive at +20 dB, 93 %
  with the surrogate at +20 dB.
- **`aaf_ideal`:** 85 % without injection and 83 % with the impulsive at +20 dB. It is immune to the
  interferer, but the hit rate without injection drops 9 points, because it cuts the fault
  resonance that folding would bring into the search. It is the flip side of the argument in favor
  of bandpass sampling: the anti-alias protects, and it charges for it.
- **TAT rules:** `uniform_ceiling` drops to 78 % even without injection, and `tat_rate` only rises
  from 54 % to 63 % at +20 dB.

**Reading.**
- On a dataset without a native interferer, a band-limited impulsive burst, folded onto the
  search range, captures the kurtosis-based selector **where the folded share predicts**.
- The capture comes from **impulsiveness and not from power**: the phase-randomized surrogate, with
  the same spectrum bin by bin, does not capture.
- The TAT calibration neither predicts nor avoids the capture reliably.

**Limits.**
- There are 13 records (12 BPFO and 1 BPFI); the CI of the sensitivity is wide.
- Only 7 folded cells with no predicted capture support the specificity.
- The power that captures is high.
- The interferer is a band-limited burst, not the single-sample spike train of Ottawa. That is
  the spike arm, running in `results-spikes/`.

## Result of the spike arm (2026-11-11; `results-spikes/`)

There are 13 records × 10 conditions (130 tasks, no failures, 8060 rows). Verdict in
`results-spikes/analysis.json`, computed by `analyze.py --config config/spikes.yaml`. Each record
is worth 7.7 points.

**H-spikes: inconclusive.** With spikes of Ottawa parameters (1300/s, 16 MAD):
- the uniform arm drops **31 points at 2×** and **23 at 4×**, which meets the first half;
- the mitigations do **not** stay at ≤ 5 points: `aaf_ideal` drops up to 15 (4× and 12×), and
  `despike` up to 15 (6× and 12×).

**Dose-response** (maximum drop of the uniform arm at factor ≥ 2): about 8 points at 8 MAD, 31 at
16 MAD and 62 at 32 MAD, at the three rates. The pattern across factors is also non-monotonic. At
1300/s and 16 MAD, 3× and 5× do not drop, while 2×, 4× and 12× do.

**Why the mitigations are not enough here:**
- **Anti-alias.** A single-sample spike is white. The anti-alias only removes the part above the
  effective Nyquist, and the in-band part remains impulsive.
- **`despike`.** Pairs of spikes two samples apart escape the criterion. They are about 3 % at
  1300/s and 48 kHz, and at 32 MAD they still capture the selector: `despike` drops up to 38
  points (12×).
- **Difference from Ottawa.** On Ottawa the removal recovers everything (26/36 at 4×). Either the
  spikes there are more isolated than a Poisson process, or the CWRU fault signal competes in a
  different way. It has not been isolated.
- **TAT.** `tat_rate` does not protect, and `uniform_ceiling` makes it worse.

**Joint reading of the two runs.**
- On a dataset without a native interferer, both a band-limited impulsive burst and a
  single-sample spike train capture the kurtosis-based selector under pure decimation.
- The capture depends on the impulsiveness (phase-randomized surrogate), on the amplitude and on
  where the folded energy falls.
- The ideal anti-alias undoes almost all of the capture by the burst, but not the capture by the
  spikes, and it costs hit rate without an interferer.
- Spike removal helps against spikes and not against bursts.

No simple mitigation solves both.

**Before entering the paper:** adversarial review (`reviewer-2`) of the two runs together.

## Status

Implemented. Tests in `tests/test_e23.py`, with an analytic reference for the folded share and a
control of the surrogate. Smoke run with 2 records. The full run was launched on the server at
06:20 UTC on 2026-11-11.

The analysis (`analyze.py`) was written **before** the full run and implements exactly the
criteria above. `tests/test_analyze.py` checks the verdict on synthetic cases with a known answer:
capture only where predicted, and a surrogate that knocks the hit rate down as much as the
impulsive.

## Adversarial review (`reviewer-2`, 2026-11-11): major revision, checked

The numbers of the two runs match the CSV. I independently recomputed the three points that weigh
the most: the dose, the independent units and the predictive power of the folded share.

**Factual corrections:**
- **The dose is labeled 3 dB too high.** `band_power` (`run.py`) returns half the power: a
  sine of power 0.5 gives 0.25. The "+20 dB" is **+17 dB** (about 50×, not 100×), and the same
  holds for the other powers.
  - The dose is normalized by the band that the kurtogram itself chooses, and its width goes from
    594 to 4750 Hz between records. In spectral density, the same label goes from 16.7 to 25.7 dB.
  - According to the reviewer, the per-record losses follow that width, with ρ = 0.96.
- **Independent units.** The 12 BPFO records are **a single fault (0.007") in 3 positions**
  (centered 135–138, orthogonal 148–151, opposite 161–164) × 4 loads. With 213 (BPFI), there are 4
  clusters.
  - According to the reviewer, the cluster bootstrap puts the sensitivity in [0.42; 0.90], with
    P(< 0.5) = 0.21. The orthogonal position alone gives 0.43.
- **The folded share does not predict the size of the drop.** Within the predicted cells,
  Spearman ≈ 0.00 (checked). None of the 6 predicted cells at 10× and 12× drops 20 points.
- **H1b is tautological.** The kurtogram partitions only [f_min, ceiling] with ideal masks, so an
  image above the ceiling does not enter any band by construction. The 0/7 could not fail.
- **The endpoint of H1a changed after the smoke run.** H1a came to be read at the maximum power
  after the smoke run refuted it at 0 dB. The README declared the power extension, but not the
  change of endpoint.
  - Without the two smoke-run records (135DE and 136DE), H1a falls to 0.60: inconclusive.
  - Without a commit, the pre-registration is not verifiable: the directory is outside git, and
    `git_sha` is null.
- **The spikes do not have the "Ottawa parameters".**
  - The rate was matched per second. Per sample, the density is 4.1× that of Ottawa: 2.7 % at
    48 kSPS against 0.67 % at 200 kSPS.
  - Measured in the contaminated signal, the amplitude is 13.7 to 16.2 MAD.
  - `despike` lets 6 to 14 % of the spikes escape, not "about 3 %". At Ottawa's density, the
    pairs would fall to about 1.3 %, which explains why the removal recovers everything there.
- **The "inconclusive" of H-spikes was determined by the design.** The 5-point threshold lies below
  1 record (7.7 points) and includes 1×, where `aaf_ideal` is the identity.
- **"31 at 16 MAD at the three rates" is wrong:** they are 23, 31 and 23. The pattern across
  factors at 16 MAD is noise of ±1 record.

**What the paper can claim from E23:**
- under pure decimation, an impulsive interferer above the ceiling comes back into the search and,
  with enough dose, captures the kurtogram; 98 % of the losses (189 of 193) fall on the burst rate;
- for kurtosis-based selectors, the capture requires impulsiveness: a Gaussian surrogate with the
  same spectrum gives a ratio of 0.14 (cluster CI [0.02; 0.22]);
- single-sample spikes also capture, with a dose-response in amplitude;
- a search ceiling that protects the selector at full rate is undone by undersampling; this is
  the work's own contribution, a narrow one;
- the low-pass avoids the capture and costs, without injection, 8.5 points at factors ≥ 2 (94.0 →
  85.5 %) and 27 points at 10× and 12× (88.5 → 61.5 %), the classic trade-off. Corrected on
  2026-11-12 by `paper_numbers.py`: it used to say "8.5 points at 10× and 12×", swapping the two
  subsets.

**What it can NOT claim:**
- "where the fold arithmetic predicts, and only there";
- "impulsiveness, and not power" in general, for any selector; the surrogate preserves
  cyclostationarity;
- "spikes with the Ottawa parameters";
- "no simple mitigation solves both";
- H1a as confirmatory;
- novelty about EMI against the full-rate kurtogram (`smith2016emi`, `vaerenberg2025cyclicemi`,
  Mauricio et al. 2020).

**What would be missing for a confirmatory version (E23b, not launched):**
- **Design:**
  - design and analysis frozen in a commit before the run;
  - a dose independent of the selector, with `band_power` corrected;
  - a pre-registered predictor that compares the folded interferer with the folded fault;
  - cluster-level statistics;
  - spikes with matched per-sample density (≈ 320–590/s at 48 kHz) and `despike` corrected for
    the pairs.
- **Missing baselines:**
  - **bandpass** anti-alias with a valid rate (`vaughan1991`);
  - robust selectors: infogram, protrugram, autogram, spectral coherence and FRESH filters;
  - `despike` and median applied to the **already decimated** stream, which is realizable.
- **Confirmation:**
  - the physical chain of `snq-sim`;
  - a second dataset (Paderborn).

## Paper numbers, recomputed (2026-11-12)

The numbers in the list "What the paper can claim" that came from the reviewer's recomputation
now come from `paper_numbers.py`, which writes `results/paper_numbers.json`. The definitions are in
the docstring.
- **Losses on the burst rate:** 189 of 193 (98 %), in the uniform arm, at the reference
  ceiling, at factors ≥ 2, with the impulsive at the highest dose. Matches the reviewer.
- **Surrogate/impulsive ratio at P\*:** 0.14, with a 95 % CI [0.03; 0.21] by bootstrap of the 4
  fault clusters (2000 resamples, 1954 with P\* defined). The reviewer gave [0.02; 0.22].
- **Folded share against the size of the drop:** Spearman's ρ 0.08 (p = 0.61) in the 48 evaluable
  predicted-capture cells, at the highest dose. The reviewer gave ≈ 0.00, and the conclusion is
  the same.
- **Cost of the ideal anti-alias without injection:** 8.5 points at factors ≥ 2 and 27 points at
  10× and 12×. The README attributed the 8.5 points to 10× and 12×, and this was corrected above.
- **Spikes, maximum drop of the uniform arm at factors ≥ 2:**
  - 8 to 15 points at 8 MAD;
  - 23 to 31 points at 16 MAD;
  - 46 to 62 points at 32 MAD.
  These are the three rates, and each record is worth 7.7 points. Matches the reviewer's correction
  (23, 31 and 23 at 16 MAD).
- **Real dose:** the injection normalizes the interferer to power 1 (`run.py`, line 114) and scales
  it by `band_power` (lines 314 and 333). That function returns half the power (checked: a sine of
  power 0.5 gives 0.25), and that is why the real dose is the label minus 3 dB.

