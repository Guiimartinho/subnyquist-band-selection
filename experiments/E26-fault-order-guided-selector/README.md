# E26 - Band selection guided by the fault orders, in the already-decimated stream

Opened on 2026-11-12, after the negative pilot of E25.

## Why this experiment

E20, E23 and E25 show that capture under pure decimation is a problem of **band choice**. The
blind selector (kurtosis) goes where the folded impulsive energy falls. In the E25 pilot, a fixed
band was right in 92 % of cases with the interferer, against 23–38 % for the blind kurtogram, but
that band was calibrated with the fault present.

Here the selector chooses the band by what matters for the diagnosis: the line at the fault orders
given by the bearing geometry (BPFO and BPFI). An interferer that is not cyclic at those orders
does not win the choice. Everything is done in the already-decimated stream, without full rate and
without commissioning. The idea is that of the targeted selectors in the literature
(`mauricio2020iesfogram`). What is tested here is whether it avoids capture under pure decimation.

## The caveat that changes the metric

With the prior on the orders, "estimating the order" loses its meaning. A search within ±2 % of the
nominal order falls inside the tolerance by construction (on Ottawa, ±0.07 against the tolerance of
0.06), and that would be the prior getting it right, not the measurement. That is why the metric is
one of **identification**:
- **What the method decides:** which fault is present (outer race, inner race or none), by the
  prominence of the line at each candidate order.
- **Threshold:** the 95th percentile of the score on the **healthy** records, per factor. It gives
  5 % false alarm. It is the separation between measurement and prior: without a fault, the method
  has to say "none".
- **Comparison:** the hit rate of the blind kurtogram of E20 on the same segments is also an
  identification. An estimate within the tolerance of the true fault identifies the right fault.

## Exploratory pilot (2026-11-12, NOT pre-registered)

The pilot decides whether a pre-registration is worthwhile. Its numbers do not go into the paper as
confirmatory.

- **Data.** Ottawa, with the 36 records: 12 healthy, 12 inner race and 12 outer race. There are
  3 segments of 2.5 s each, the same as in E20.
- **Factors and search.** Factors 1, 2, 3, 4, 5, 6 and 8. Search from 500 Hz up to
  min(25 kHz, f_s/2M), with the kurtogram partition up to level 3 (24 bands).
- **Score.** For each band and candidate fault c, it is the mean, over the harmonics k = 1, 2, 3, of
  the line prominence: the peak within ±2 % of k·o_c divided by the median within ±7 %. The orders
  o_c come from the geometry (BPFO = 3.573 and BPFI = 5.427). The score of the fault is the maximum
  over the bands.
- **Decision.** The fault with the highest score, if it exceeds the healthy threshold at that
  factor. Otherwise, "none".
- **Metrics.**
  - identification hit rate per race and balanced;
  - false alarm, with the threshold calibrated on the healthy records themselves, that is,
    in-sample;
  - the blind kurtogram of E20 (`results/metrics.csv`, arm `none`, 25 kHz ceiling, nested ruler)
    on the same segments.

## Pilot result, first round (±2 % window, `results-pilot/`)

| factor | guided: outer | guided: inner | guided: healthy correct | blind (E20): outer | blind (E20): inner |
|---|---|---|---|---|---|
| 1× | 0 % | 100 % | 94 % | 75 % | 97 % |
| 4× | 0 % | 100 % | 94 % | 17 % | 97 % |
| 8× | 0 % | 100 % | 94 % | 69 % | 100 % |

**Why the outer race stays at 0 %, even at 1×.**
- The real BPFO of Ottawa lies 4 % below the nominal one: the median of the nested ruler is 3.432,
  against 3.573.
- On the local ruler, it goes from 3.33 to 3.79, because the line drifts with the rotation speed.
- The ±2 % window never reaches it, and the outer-race score stays at the level of the healthy
  records (median of 3.2, against 3.4).
- On the inner race, the real BPFI lies 0.2 % from the nominal one, and the hit rate is 100 %.
- **It was a design error, not an effect of decimation.** E07 and E20 calibrate Ottawa over a
  ±12 % span precisely because of this uncertainty.

## Pilot revision, second round (2026-11-12, after the first; `results-pilot-w12/`)

- **Window:** ±12 % around each candidate order, with the peak and the median in the same window.
  It is the calibration span of Ottawa in E07 and E20.
- **Harmonics:** only 1 and 2. With the wide window, 3·BPFO would overlap 2·BPFI.
- **The rest does not change,** including the threshold on the healthy records.
- The revision was made after seeing the first round. It corrects a design error known beforehand,
  but it is an exploratory pilot and must be read as such.

## Result of the second round (±12 % window, `results-pilot-w12/`)

| factor | guided: outer | guided: inner | threshold (healthy) | blind (E20): outer |
|---|---|---|---|---|
| 1× | 100 % | 100 % | 7.0 | 75 % |
| 2× | 72 % | 100 % | 11.2 | 56 % |
| 3× | **100 %** | 100 % | 6.2 | 11 % |
| 4× | 11 % | 100 % | **14.3** | 17 % |
| 5× | 6 % | 100 % | **14.9** | 6 % |
| 6× | **97 %** | 100 % | 8.7 | 31 % |
| 8× | 97 % | 100 % | 8.4 | 69 % |

The healthy records come out correct in 94 % of the segments, at all factors.

**Why 4× and 5× fail.**
- In the **healthy** records, strong lines appear inside the BPFI window, at orders from 5.63 to
  5.84, with the 2nd harmonic at 11.26–11.67.
- They show up in the high bands (12.7–25 kHz), precisely where the folded spikes fall, and not at
  integer orders. They change from record to record.
- This is compatible with a spike train at a fixed rate in Hz, which, under variable rotation speed,
  shows up at different orders.
- The threshold calibrated on the healthy records rises, and the outer race no longer passes.
- It is the expected failure mode of the guided selector: a cyclic interferer near a candidate
  order.

## Third variant (2026-11-12): median of 3 on the decimated stream before the guided selector

It is the same ±12 % window, with the E20 correction applied before it. It tests whether the spikes
produce the false lines of the healthy records (`pilot.py --clean median3`,
`results-pilot-w12-median3/`). It is also exploratory and was decided after seeing the second round.

### Result of the third variant (`results-pilot-w12-median3/`)

| factor | median + guided: outer | inner | threshold (healthy) | without median: outer | blind (E20): outer |
|---|---|---|---|---|---|
| 1× | 19 % | 100 % | **13.9** | 100 % | 75 % |
| 2× | **89 %** | 100 % | 10.4 | 72 % | 56 % |
| 3× | **100 %** | 100 % | 5.9 | 100 % | 11 % |
| 4× | **97 %** | 100 % | 8.7 | 11 % | 17 % |
| 5× | **100 %** | 100 % | 7.1 | 6 % | 6 % |
| 6× | **100 %** | 100 % | 5.8 | 97 % | 31 % |
| 8× | **100 %** | 100 % | 7.1 | 97 % | 69 % |

The healthy records come out correct in 94 % of the segments, at all factors. This is in-sample.

**Reading** (exploratory):
- **From 2× to 8×, the combination identifies the outer race in 89–100 % of the segments,** against
  6–69 % for the blind kurtogram, with the inner race at 100 %.
  - At 4× and 5×, the healthy threshold goes back to 7–9. The folded spikes were the source of the
    false lines.
  - As an indirect reference, the median followed by the kurtogram (E20, `results-post-median`)
    gets the **order** right in 75–81 % from 4× to 8×. The metric here is identification, so the
    comparison is not exact.
- **At 1× the combination drops to 19 %:** the healthy threshold rises to 13.9, with no explanation
  for now. It is not the sub-Nyquist regime, but it needs to be understood.

**Limitations:**
- The design was revised twice after seeing the data.
- The threshold is calibrated on the same healthy records on which the false alarm is measured.
- The blind kurtogram has no false-alarm threshold in this comparison. The healthy control of E07
  (`healthy_control.csv`) is the reference for that.

**Before the paper:** a **pre-registered** round is needed, on data that were not used to tune the
design (Paderborn), with:
- a search window fixed a priori by the order uncertainty of the dataset;
- a threshold on healthy bearings with cross-validation;
- the same baselines: blind kurtogram, median + kurtogram and low ceiling.

## Confirmation on Paderborn (pre-registration, 2026-11-13)

Written and committed before running, together with `run_paderborn.py`, `analyze_paderborn.py` and
`config/paderborn.yaml`. The design is fixed from here on. The Ottawa pilot is **not** part of the
test, because it was revised twice after seeing the data. Paderborn was not used to tune it.

- **Data.**
  - 23 damaged bearings (12 outer race and 11 inner race, those of E24) and the 6 healthy ones
    (K001–K006).
  - The three 1500 rpm settings, with measurements 2 to 20 of 255 936 samples.
  - Measurement 1 of the damaged bearings feeds the nested ruler (`line_calibration` of E24).
- **Factors:** 1, 4, 8, 16 and 32. The decisive ones are 4, 8, 16 and 32; 1 is descriptive.
- **Arms,** all on the already-decimated stream:
  - `kurtogram`, the blind pipeline of E24;
  - `median_kurtogram`, the median of 3 followed by the kurtogram;
  - `guided`, the guided selector without cleaning;
  - `median_guided`, the median of 3 followed by the guided selector.
- **Guided selector,** with parameters fixed a priori:
  - **Orders:** those of the geometry, BPFO = 3.0543 and BPFI = 4.9457, and only harmonics 1 and 2.
  - **±2 % window:** it is the 1 to 2 % spread between manufacturers (PHME16, section 3).
  - **Integer orders excluded, ±0.015.** On Paderborn, the BPFO and the BPFI lie 0.054 from the 3rd
    and the 5th shaft harmonic, and the rotation speed is constant. The resolution in 4 s is
    0.01 order.
  - **Partition:** that of the kurtogram, levels 1 to 3, with 23 bands up to the effective Nyquist.
  - **Score:** the mean prominence, that is, the peak divided by the median in the window itself.
- **Threshold:** the 95th percentile of the healthy scores, per factor and cleaning. To evaluate
  each healthy bearing, the threshold comes from the other 5.
- **Identification.**
  - **Blind:** outer or inner race, if the estimated order lies ≤ 0.025 from the true one;
    otherwise, "none". The true one is the nested ruler of E24 for the race itself, and the median
    of all calibrations for the other race and for the healthy bearings.
  - **Guided:** the fault with the highest score, if it exceeds the threshold; otherwise, "none".
- **Metrics.**
  - Balanced accuracy on the damaged bearings: the mean, over the races, of the per-bearing hit
    rate.
  - Specificity on the healthy bearings: the fraction of "none".
- **Hypotheses.**
  - **H1:** `median_guided` > `kurtogram`.
  - **H2:** `median_guided` > `median_kurtogram`. It is the gain of the guided selector beyond
    the cleaning.
  - **The test:** per-bearing sign-flip test, stratified by race, one-sided, on the per-bearing
    differences in hit rate. It is exact up to 2^18 combinations and Monte Carlo with 10^6 above
    that. The E24 review showed that the bootstrap is anticonservative with so few clusters.
  - Holm over the 4 factors, with α = 0.05.
  - **A factor passes** if the Holm p is below 0.05, the difference is positive and the
    specificity of `median_guided` there is at least 90 %.
- **What would refute.** H1 falls if no factor passes. The same holds for H2.
- **Declared limitations.**
  - The blind arm has no false-alarm threshold; its specificity on the healthy bearings is
    reported.
  - The nested ruler comes from few calibrations, measured in the band that the kurtogram chooses
    (E24 review). It only enters the identification of the blind arm.
  - The window of the guided arm (±2 %) is wider than the tolerance of the blind arm (0.025). The
    price of this is the false alarm, controlled by the healthy threshold.

**Correction before the full round (2026-11-13).** The smoke run, with 6 bearings, 1 setting, 2
measurements and factors 1 and 4, ran, but the analysis broke when looking for factor 8, which it
did not have. The analysis was changed to test only the decisive factors present in the data. No
criterion changed, and the numbers of the smoke run were not looked at.

## Result of the confirmation on Paderborn (2026-11-13; under adversarial review)

- **Run:** `results-paderborn/`, from 00:25 to 05:21 UTC, 16 470 rows.
  - The hashes of the config, of `run_paderborn.py`, of `guided.py`, of the README and of the E24
    `run.py` match commit `e8aaa8d`. No smoke-run subsetting was applied.
  - 6 measurements were left out because of a short file, a rotation signal that is not 4 s long
    or an unreadable `.mat`. The list is in the manifest.
- **Pre-registered verdict** (`analysis.json`): H1 and H2 pass at the four decisive factors.
- **Balanced accuracy (%), nested ruler for the blind arms and healthy threshold for the guided
  ones:**

  | arm | 1× | 4× | 8× | 16× | 32× |
  |---|---|---|---|---|---|
  | `kurtogram` | 7.7 | 16.1 | 18.6 | 19.2 | 13.0 |
  | `median_kurtogram` | 15.6 | 18.8 | 27.8 | 20.8 | 17.4 |
  | `guided` | 36.2 | 52.2 | 53.0 | 45.4 | 32.2 |
  | `median_guided` | 38.3 | 49.0 | 50.9 | 45.2 | 25.2 |

  - The specificity on the healthy bearings is 93.9 to 95.0 % in the guided arms and 99.4 to 100 %
    in the blind ones.
- **Tests** (per-bearing sign-flip test, stratified by race, Holm over 4, 4× to 32×):
  - **H1** (`median_guided` against `kurtogram`): +32.9, +32.3, +26.0 and +12.2 points, with 16 of
    17, 17 of 18, 18 of 19 and 21 of 22 discordant bearings in favour. The Holm p is below 0.001 at
    all four.
  - **H2** (`median_guided` against `median_kurtogram`): +30.2, +23.1, +24.3 and +7.8 points, with
    Holm p of 0.0004, 0.0014, 0.0006 and 0.031.
- **What the verdict does not say, and what the review needs to attack:**
  - **The median does not contribute on Paderborn.** `guided`, without cleaning, ties with
    `median_guided` or is above it. At 32× it is 7 points above.
  - **The gain already exists at 1×,** with 36.2 % against 7.7 %. So it is not specific to
    decimation: on Paderborn the kurtogram is already captured at full rate (E24).
  - **The arms change two things at the same time:** the band and the identification rule.
    - The guided arm chooses between the two orders of the geometry, with a threshold calibrated on
      the healthy bearings.
    - The blind arm estimates the order without a prior and has to be right within ±0.025.
    - The gain may come from the identification rule, not from the band. This was declared in the
      pre-registration as a limitation, but it was not separated.

## E26b: the rule stays fixed, the band changes (pre-registration, 2026-11-13)

Written and committed before running, together with `run_paderborn_rule.py`,
`analyze_paderborn_rule.py` and `config/paderborn_rule.yaml`.

- **Question.** The confirmation changed two things at the same time: the band and the
  identification rule. Does the gain come from the guided band selector or from the guided rule?
  - The rule: the windows at the orders of the geometry and the healthy threshold.
- **What stays the same as in E26:**
  - the data, the factors, the two cleanings and the windows;
  - the score of each fault;
  - the threshold: 95th percentile of the maximum healthy scores, per arm, factor and cleaning,
    leaving the healthy bearing itself out when it is the one evaluated;
  - the decision: the fault with the highest score, or "none".
- **What changes: only the band in which the score is measured.**
  - `guided`: the maximum over the 23 bands of the partition, from the E26
    `results-paderborn/metrics.csv`, without recomputing.
  - `kurtogram_rule`: the band of the fast kurtogram up to the effective
    Nyquist, the same as in the blind arm.
  - `wideband_rule`: [500 Hz, fs/2], without selection. It is the classical
    envelope.
  - `ses_infogram_rule`: the band of the SES infogram (descriptive).
  - `random_band_rule`: a band drawn at random from the partition, with a seed per
    segment (descriptive).
- **Hypotheses** (per-bearing sign-flip test, stratified by race, as in E26):
  - **H3:** `guided` is above `kurtogram_rule`. The test is done on both cleanings and at 4×, 8×,
    16× and 32×, with Holm over the 8.
  - **H4:** `guided` is above `wideband_rule`, in the same way.
  - A test passes if the Holm p is below 0.05 and the difference is positive. The specificity is
    the same across the arms by construction, close to 95 %, and is reported.
- **Combined reading, before the result:**
  - **H3 and H4 pass:** the guided band choice adds to the rule, and the selector is the
    contribution.
  - **Both fail:** the gain of E26 comes from the rule, that is, from the prior and the healthy
    threshold. The claim of the paper rests on the rule, not on the band selector.
  - **Mixed result:** reported as mixed, per factor and cleaning.
- **Integrity check.** Where the random band coincides with the guided band of a fault, the score
  has to be identical in the two files. If it is not, the comparison is invalid.
- **Declared limitation.** Paderborn was already seen in E26, and this pre-registration was
  motivated by its result. The scores of the four new strategies have not been computed yet. The
  reference arm stays fixed as it came out of E26.

### Adversarial review (`reviewer-2`, 2026-11-13): major revision

The review scripts are in `exploratory/review_2026-11-13/`. I reproduced the
difference-in-differences (`a2_did.py`) and the per-bearing count (`a10_bearinglevel.py`).

- **What survived:**
  - **Reproduction.** The analysis reproduces bit for bit from the CSV, and the hashes match.
  - **The prior does not invent detections.** Given a detection by `median_guided`, the race comes
    out right in 91 to 100 % from 1× to 16×, and in 84 to 92 % at 32×. The wrong candidate exceeds
    the threshold on the damaged bearings in 0 to 7.7 % of the segments, the false-alarm level.
  - **The ruler of the blind arm is fair.** Changing the ruler or widening the tolerance moves the
    blind arm by at most 4.3 points.
  - **H1 does not depend on a single bearing.** Removing any one of them, the minimum stays at
    +29.5, +29.0, +22.5 and +8.9.
  - **Per bearing** (identified if more than 50 % of the segments come out right): at 4× and 8×, the
    guided arm with median identifies 12 of 23 bearings and the blind arm, 4. There are 8 identified
    only by the guided arm and 0 only by the blind arm (McNemar, p = 0.004). No healthy bearing
    raises an alarm.
- **Attacks, from the most to the least serious:**
  1. **The gain does not come from decimation.** At 1× it is already +30.6 points, with 16 of 16
     bearings in favour. The per-bearing difference-in-differences, gain at the factor minus gain
     at 1×, is +2.3 [−13.5, +17.5] (4×), +1.7 [−14.5, +17.5] (8×), −4.6 [−19.0, +9.0] (16×) and
     −18.4 [−32.5, −4.5] (32×; two-sided p 0.012). The guided arm "improving" from 1× to 4× is an
     artefact of the single threshold, pulled by the BPFO window on the healthy bearings.
  2. **Band and rule are confounded, and the blind arm fails because of the rule.**
     - The blind argmax falls on a line of the test rig at order 2.10 in 38.8 % of the damaged
       segments and in 58.2 % of the healthy ones.
     - In the same band, the guided rule beats the blind argmax: from 69.4 against 34.7 % at 8× to
       60.4 against 32.0 % at 32×.
     - E26b is the right design, but the order of the peak in the window still has to be recorded.
  3. **The honest baselines are missing.**
     - The guided arm with anti-alias at the same rate: it is the decisive control for "pure
       decimation". E24b shows that, on Paderborn, the blind arm with a ceiling at 1× beats pure
       decimation.
     - Classical band-pass sampling with envelope (`lu2018undersampled`).
     - At least one published targeted selector.
  4. **The novelty overlaps the literature.** Band selection by the known fault frequencies already
     exists: the log-cycligram (Smith et al., MSSP 2019), the IFBIgram (Schmidt et al., MSSP 2020)
     and `mauricio2020iesfogram`. Fault harmonics in undersampled vibration: Tang et al. (Sensors
     2015) and `lu2018undersampled`.
  5. **The median does not contribute.** Compared with the guided arm without cleaning, it changes
     the result by +2.1 to −7.1 points, and the −7.1 at 32× goes against it (p = 0.044).
  6. **The exclusion of the integer orders (±0.015) is smaller than the lobe of the Hann window.**
     Leakage from the shaft line can produce prominence without a fault. Request: record the order
     of the peak, test ±0.03 and use decoy windows.
  7. **The specificity is a point estimate with 6 healthy bearings.**
     - The CI goes from 88.6 to 99.7 % at 4×.
     - Per healthy bearing, it drops as low as 75.4 %.
     - The threshold requires healthy-state data from the same test rig.
  8. **Heterogeneity.**
     - At 4×, 8 bearings are above 80 % and 8 below 10 %.
     - The artificial inner-race damages (KI03, KI05, KI07 and KI08) and the real level-1 ones are
       never identified.
     - At 32× there is no per-bearing difference (4 against 3).
  9. **Physical wall.** `x[::M]` folds only what passed through the analog chain of Paderborn, not
     the track-and-hold band of a real sampler.
  10. **Provenance.** The hashes are computed at the end of the run, and the manifest records
      neither the commit nor whether the tree was dirty.
- **Sentence accepted by the reviewer:**
  > On Paderborn (23 damaged, 6 healthy bearings, 1500 rpm), a fault-order-targeted rule with a
  > healthy-calibrated threshold identified the faulty race in 49–53 % of segments (12 of 23
  > bearings) at 1/4 and 1/8 of the original rate by plain decimation, against 16–19 % (4 of 23)
  > for the blind kurtogram with an unrestricted envelope peak, at about 94 % against 100 %
  > specificity. The advantage was of the same size at full rate (+30.6 points) and shrank at
  > 1/32, so it is not attributable to decimation.
- **Not accepted:**
  - that the guided selector with median improves identification under decimation;
  - that it avoids capture under decimation;
  - attributing the gain to the band choice before E26b;
  - that the median contributes;
  - "sub-Nyquist acquisition" in the hardware sense.
- **Consequence:**
  - E26 becomes an effect of prior and rule, present at all rates.
  - What can still be a contribution: how much of this effect survives pure decimation, compared
    with anti-alias at the same rate.
  - This calls for E26b, plus a round with the anti-alias, the order of the peak and the ±0.03
    exclusion, and the band-pass baseline.

## E26c: pure decimation against anti-alias and band-pass (pre-registration, 2026-11-13)

Written and committed before running, together with `run_paderborn_acq.py`,
`analyze_paderborn_acq.py` and `config/paderborn_acq.yaml`. It is the answer to attacks 3 and 6 of
the review.

- **Question.** At the same rate FS/M, does letting everything fold (`x[::M]`) lose or gain
  information for the guided rule, compared with filtering before decimating? This is the decisive
  control for any sentence about pure decimation.
- **Scope:**
  - only the raw stream, because the median did not contribute in E26;
  - factors 4, 8, 16 and 32;
  - the same bearings, settings and test measurements.
- **Acquisitions,** with ideal filters by masking the full-rate FFT:
  - `pure`: `x[::M]`;
  - `antialias`: low-pass at FS/(2M), then `[::M]`;
  - `zone{k}`: band-pass on [k, k+1]·FS/(2M), then `[::M]`, only at 4× and 8×. It is
    classical band-pass sampling (`lu2018undersampled`, `vaughan1991`); zone 0 is the
    anti-alias.
- **Bands,** always with the E26 rule:
  - `guided` (maximum over the partition) and `kurtogram_rule`, in `pure` and `antialias`;
  - `wideband_rule` ([500 Hz, fs/2]), in all acquisitions.
- **Choice of the zone for band-pass:**
  - `kurtogram_zone`: the zone of the centre of the full-rate kurtogram band on
    measurement 1 of the same bearing and setting. It is the commissioning a practitioner would do.
  - `fleet_zone`: the zone with the highest balanced accuracy on the other damaged
    bearings, leaving the bearing itself out.
- **Hypotheses** (per-bearing sign-flip test, stratified by race, one-sided):
  - **H5:** `pure|guided` > `antialias|guided` at 4×, 8×, 16× and 32×, with Holm over 4.
  - **H6:** `pure|guided` > `fleet_zone|wideband_rule` at 4× and 8×, with Holm over 2.
  - A factor passes if the Holm p is below 0.05 and the difference is positive.
- **Combined reading, before the result:**
  - **H5 passes:** the folding carries information that the anti-alias throws away. It is the
    pure-decimation sentence that the paper can write.
  - **H5 fails with a difference ≤ 0:** what the rule uses lies below FS/(2M). On Paderborn, pure
    decimation brings no advantage over filtering.
  - **H6 fails with the band-pass above:** band-pass sampling with a zone chosen on the fleet is the
    best design at the same rate.
- **Descriptive analyses:**
  - the kurtogram and the whole band, `pure` against `antialias`;
  - the accuracy profile by zone;
  - `kurtogram_zone`.
- **Leakage (attack 6),** in `pure|guided`:
  - among the detections on the damaged bearings, the fraction with the peak of the 1st harmonic
    within 0.03 of an integer order;
  - the accuracy with the ±0.03 exclusion.
- **Decoy windows:**
  - the fraction of segments in which the largest of 4 decoy windows, at 0.9 and 1.1 times the BPFO
    and the BPFI, exceeds the threshold, on the damaged and on the healthy bearings;
  - the same fraction for the true fault, for comparison.
- **Integrity.** The scores of `pure|guided` have to be identical to those of the raw guided arm of
  E26.
- **Declared limitations:**
  - Paderborn was already seen in E26, in E26b and in the review;
  - the filters are ideal, not analog;
  - `x[::M]` folds only what passed through the analog chain of the test rig, not the
    track-and-hold band of a real sampler (review, attack 9).

### Result of E26b (2026-11-13): mixed

- **Run:** `results-paderborn-rule/`, 65 880 rows, without smoke-run subsetting. The hashes match
  `b2bec23`.
- **Integrity:** 1474 matches between the random band and the guided band, with a maximum
  difference of 0.0. The scores of the two files are the same computation.
- **Balanced accuracy (%), fixed rule and healthy threshold.** Specificity of 89.8 to 95.0 % in all
  arms. The blind arm of E26 enters only for comparison.

  | raw stream | 1× | 4× | 8× | 16× | 32× |
  |---|---|---|---|---|---|
  | blind kurtogram (E26) | 7.7 | 16.1 | 18.6 | 19.2 | 13.0 |
  | `kurtogram_rule` | 14.1 | 30.9 | 34.0 | 33.1 | 25.1 |
  | `random_band_rule` | 15.1 | 36.6 | 35.2 | 32.3 | 10.8 |
  | `ses_infogram_rule` | 36.0 | 43.4 | 39.0 | 34.0 | 24.6 |
  | `wideband_rule` | 32.2 | 38.8 | 43.7 | 41.5 | 41.1 |
  | `guided` | 36.2 | 52.2 | 53.0 | 45.4 | 32.2 |

  | median of 3 | 1× | 4× | 8× | 16× | 32× |
  |---|---|---|---|---|---|
  | `kurtogram_rule` | 20.0 | 32.1 | 50.7 | 34.8 | 25.2 |
  | `wideband_rule` | 51.8 | 45.1 | 46.8 | 47.0 | 32.7 |
  | `guided` | 38.3 | 49.0 | 50.9 | 45.2 | 25.2 |

- **H3** (guided against `kurtogram_rule`; Holm over 8):
  - **On the raw stream, it passes at 4×, 8× and 16×:** +21.4, +19.0 and +12.3 points, with 18 of
    20, 18 of 19 and 13 of 18 bearings in favour (Holm ≤ 0.003). At 32× it stays at +7.1, Holm 0.17.
  - **With the median, it passes at 4× and 16×:** +16.9 and +10.4. At 8× and 32× it ties.
- **H4** (guided against `wideband_rule`):
  - **On the raw stream, it passes at 4× and 8×:** +13.5, with 18 of 18 in favour, and +9.3. At 16×
    it stays at +3.8, without passing, and at 32× it is −8.9, with the whole band ahead.
  - **With the median, it passes only at 4×,** by +3.8.
- **Reading, by the key of the pre-registration: mixed.**
  - **The rule explains part of the E26 gain.** In the same kurtogram band, it raises the hit rate
    from 16.1 to 30.9 % at 4× and from 13.0 to 25.1 % at 32×.
  - **The guided band adds to the rule against the kurtogram band,** by +12 to +21 points from 4× to
    16× on the raw stream.
  - **Against the whole-band envelope with the same rule, the guided band only wins at 4× and 8×.**
    At 32× it loses.
  - **The whole-band envelope with the prior and the healthy threshold is a strong baseline:** 39 to
    44 % from 4× to 32× on the raw stream, and 52 % at 1× with the median. Band selection, guided
    or not, is the smaller component.
- **Consequence:** the claim of E26 becomes one about the rule and only partially about the band.
  E26c decides what is left for pure decimation.

### Result of E26c (2026-11-13): on Paderborn, anti-alias beats pure decimation

- **Run:** `results-paderborn-acquisition/`, 55 998 rows, without smoke-run subsetting. The hashes
  match `534d995`.
- **Integrity:** 13 176 matches between `pure|guided` and the raw guided arm of E26, with a
  difference of 0.0.
- **Balanced accuracy (%), raw stream, E26 rule.** Specificity of 89.8 to 95.0 %:

  | arm | 4× | 8× | 16× | 32× |
  |---|---|---|---|---|
  | `pure|guided` | 52.2 | 53.0 | 45.4 | 32.2 |
  | `antialias|guided` | 59.5 | 64.5 | 54.1 | 62.0 |
  | `pure|kurtogram_rule` | 30.9 | 34.0 | 33.1 | 25.1 |
  | `antialias|kurtogram_rule` | 60.7 | 62.6 | 63.8 | 50.3 |
  | `pure|wideband_rule` | 38.8 | 43.7 | 41.5 | 41.1 |
  | `antialias|wideband_rule` | 47.0 | 51.9 | 50.5 | 71.2 |
  | `kurtogram_zone|wideband_rule` | 14.7 | 16.0 | – | – |
  | `fleet_zone|wideband_rule` | 47.0 | 51.9 | – | – |

- **Profile by zone** (whole band with the rule):
  - **4×:** zone 0 (anti-alias) 47.0; zone 1 27.5; zone 2 2.5; zone 3 3.0.
  - **8×:** zone 0 51.9; zone 1 35.6; zone 2 40.4; zone 3 8.1; zones 4 to 7 from 2.0 to 3.5.
  - The fleet chooses zone 0 for every bearing. The band-pass chosen by the fleet is the
    anti-alias.
- **H5 fails** (`pure|guided` > `antialias|guided`). Pure decimation is below:
  - −7.3 at 4×, −11.5 at 8×, −8.7 at 16× and −29.7 at 32×;
  - 2 to 4 bearings in favour, out of 15 to 18;
  - Holm of 1.0.
- **H6 fails** (`pure|guided` > `fleet_zone`): +5.1 at 4× (Holm 0.087) and +1.1 at 8×.
- **Leakage** (attack 6 of the review):
  - Among the detections on the damaged bearings, only 1.0 to 2.5 % have the peak of the 1st
    harmonic within 0.03 of an integer order.
  - With the ±0.03 exclusion, the hit rate is 49.8, 51.3, 44.5 and 30.4 %, against 52.2, 53.0,
    45.4 and 32.2 %.
  - Leakage from the shaft lines does not explain the detections.
- **Decoy windows:**
  - The largest decoy exceeds the threshold in 7.6 to 15.4 % of the damaged segments and in 3.5 to
    8.2 % of the healthy ones. The true fault exceeds it in 33.0 to 53.9 % of the damaged ones.
  - The detection is mostly fault-specific. There is an excess of non-specific content in the
    damaged bearings, of about 2× the level of the healthy ones.
- **Reading:**
  - **On Paderborn, the diagnostic information is in the baseband.**
    - The faulty bearings are identified below FS/(2M).
    - The high zones, from 16 kHz at 4× and from 12 kHz at 8×, bring nothing. They contain the
      impulsive content that captures the kurtogram (E24).
    - Decimating without a filter folds these zones onto the useful band and costs 7 to 30 points
      for any band choice.
  - **Commissioning the zone by the full-rate kurtogram fails** (15 to 16 %), because it chooses the
    captured zone.
  - **On Ottawa it is the opposite** (E20 and E23): the fault resonance lies high, and the
    anti-alias costs 27 points at 10× and 12×.
  - **The advantage of pure decimation depends on where the informative band lies** relative to the
    interference. This is measured per zone, before choosing the acquisition.
  - **The best acquisition for Paderborn, at the same rate, is the classical one:** anti-alias, low
    rate and envelope with the fault-order rule. It reaches 71.2 % at 32× (2 kS/s) with the whole
    band.


## Figure (2026-11-13)

![Rule, acquisition and profile by zone](results-paderborn-acquisition/figures/e26_rule_paderborn.png)

Generated by `papers/shared/figures/fig_rule_paderborn.py`, which redoes the decision from the
three `metrics.csv` files. The script checks each point against the run's `analysis.json`
(±0.1 point) and stops if anything diverges. The bands are 95 % CIs from a bootstrap of the
bearings within each race.
- (a) decision and band under pure decimation;
- (b) pure decimation against anti-alias at the same rate;
- (c) profile by Nyquist zone at 4× and 8×.
