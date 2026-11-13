# E24 - Selector capture by spikes under pure decimation, on Paderborn

Replication of Part 1 of ADR 0006 on a second real dataset. On Ottawa, single-sample spikes in the
raw record fold under pure decimation and capture the kurtogram, which produces the 4× valley.
Removing the spikes before decimating undoes the valley (E07, E20).

In the exploratory check of 2026-11-12 (`docs/lab-notebook/2026-11-12.md`), Paderborn showed the
same kind of spike:
- from 2 to 730 per second, of ~12 MAD;
- they account for the kurtosis of the raw signal, from 13 to 16;
- they pull the kurtogram to 11–21.5 kHz, the band that contains the inverter's 16 kHz switching.

## Hypotheses (pre-registration, 2026-11-12)

Written and committed before running. Changing it after seeing a number invalidates the
pre-registration.

- **H1 (replication).** Removing the spikes before decimating (E20 criterion) increases the
  balanced order accuracy of the classical pipeline in at least one of the budgets 1/4, 1/8, 1/16
  and 1/32.
- **H2 (robust selector).** Replacing the kurtogram with the SES infogram (Antoni 2016), without
  removing the spikes, increases the same accuracy in at least one of those budgets.
- **H3 (realizable removal; pre-registered on 2026-11-12, before any E24 number).**
  Removing the spikes from the **already-decimated** stream, by the same criterion, increases the
  accuracy of the classical pipeline in at least one of the budgets. It is
  `config/post_despike.yaml`, which has the same bootstrap and the same Holm. If H3 holds, the
  correction does not require the full rate.

## What would refute

- **H1:** none of the four tests significant after Holm, or a negative point difference. The
  capture by spikes would not reproduce outside Ottawa.
- **H2:** the same criterion. The robust selector would not be enough under pure decimation.

## Baseline

The classical pipeline of E07 and E22: pure decimation, kurtogram (level 4, 500 Hz floor, ceiling
at the effective Nyquist), envelope order spectrum and argmax without prior above order 2. It is
the `kurtograma` (kurtogram) arm.

## Ground truth

The fault orders come from the geometry of the FAG 6203 (PHME16, table 1): BPFO 3.054 and
BPFI 4.946. They are calibrated from the data in two ways.

- **Nested ruler (primary).** It is the median, on the same race, of the valid calibrations from
  measurement 1 of the **other** bearings (leave-one-bearing-out).
  - The calibration uses full rate, raw signal, refinement within ±1 % of the nominal and a
    visible line (peak of at least 5× the median).
  - It does not depend on any arm and holds for all segments.
- **Local ruler (secondary).** It is the full-rate line of the segment itself, by the same
  criterion. It only holds where the line is visible.

The tolerance is 0.025 order: the BPFO and the BPFI lie 0.054 from shaft harmonics 3 and 5.

## Procedure

- **Data:** those of E15 on Paderborn.
  - 23 bearings with damage on one race (12 outer race and 11 inner race), in the three
    1500 rpm settings.
  - Measurements 2 to 20 are the segments, of 255936 samples. A measurement shorter than that, or
    unreadable, is left out.
- **Arms** (`config/base.yaml`): `kurtograma`, `kurtograma_sem_picos` (kurtogram without spikes),
  `infograma_ses` (SES infogram), `infograma_ses_sem_picos` (SES infogram without spikes) and
  `gini`.
  - Spike removal comes before decimating, at full rate.
- **Factors:** 1, 4, 8, 16 and 32. Factor 1 is descriptive. The guard is 1.0 × 64 points per
  revolution.
- **Inference:** cluster bootstrap, with cluster = bearing and resampling within each race.
  - There are 4000 resamples, and the one-sided p is (1 + #{Δ* ≤ 0})/(1 + B).
  - Holm over the 4 tests of each hypothesis, under the nested ruler; α = 0.05.
  - The local ruler is reported with the same computation, as secondary.

## Metrics

- Balanced order accuracy: the mean, over the races, of the mean hit rate per bearing.
- The band chosen per arm and factor, to show where the spikes take the selector.
- There is no waveform reconstruction, so the range/null-space decomposition does not apply.

## Result (2026-11-12)

**Provenance.**
- Base run: `config/base.yaml`, commit `fb091b8`, from 14:13 to 20:52 UTC.
- H3: `config/post_despike.yaml`, commit `6a95a72`, from 15:24 to 20:52 UTC, in a separate
  checkout.
- The hashes of `run.py`, config, README and source (`8491841f…`) in the manifest match those
  commits.
- There are 69 records and 1305 segments. The local ruler covers 7 bearings.

**Pre-registered verdicts** (the nested ruler decides; Holm over the 4 budgets; α = 0.05):

| hypothesis | result (nested ruler) | verdict |
|---|---|---|
| H1, removing the spikes before decimating | +2.1 points at 8× [0.15, 4.49], Holm p = 0.04999 (4 × 50/4001); at the other budgets, +0.8, +1.6 and +2.2, not significant | **supported only at 8×, at the limit** |
| H2, SES infogram without removing the spikes | +3.1 points at 32× [0.95, 5.78], Holm p = 0.001; at 4×, +8.8 [−0.1, 19.1], Holm p = 0.08 | **supported at 32×** |
| H3, removing the spikes after decimating | from −1.0 to +0.7 points, none significant | **not supported** |

Under the local ruler (7 bearings), none of the three passes.

**Descriptive:**
- **There is no valley.** The kurtogram's balanced accuracy, under the nested ruler, is 7.7, 16.1,
  18.6, 19.2 and 13.0 % at 1×, 4×, 8×, 16× and 32×. The worst is at full rate.
- **The capture already happens at full rate.** The kurtogram picks 11.0–21.5 kHz in 909
  segments, precisely the band that contains the inverter's 16 kHz switching.
  - Removing the spikes before decimating does not move it out of there: 350 segments stay at
    11.0–21.5 kHz, and 735 of the remaining 955 (77 %) lie above 16 kHz.
  - Under decimation, the choice goes to the lowest band of the search, 0.5–1.1 kHz at 4× and 8×.
- **Folding arithmetic (hypothesis, not tested).** At 64 kS/s, with factors 4, 8, 16 and 32, every
  multiple of 16 kHz lands exactly at 0 Hz. The switching sidebands would then land at the
  bottom of the search band.
- **Line visibility.** Only 7 of the 23 bearings have a visible line in some segment: KA01, KA06,
  KA16, KI01, KI16, KI17 and KI18.
  - Visibility is measured in the band that the kurtogram picks at full rate, and that band is
    captured (`line_calibration`).
  - On the other 16 bearings, the kurtogram hits no segment at full rate under the nested ruler,
    which holds by construction, because visibility is measured on the same spectrum. The SES
    infogram reaches 17.0 % balanced accuracy on them: 34.0 % on the outer race and 0 % on the
    inner race.
  - So "no visible line" is partly a failure of the selector, and not only of the signal.

**Reading:**
- The Ottawa valley does not reproduce on Paderborn. There the selector is already captured at
  full rate by the inverter band, which is the scenario of `smith2016emi`.
- Removing the single-sample spikes changes little. H1 passes at the limit, with a small effect,
  and does not support that "the capture by spikes reproduces outside Ottawa".
- The SES infogram helps on Paderborn, unlike on Ottawa.
- **Before it goes into the paper:** adversarial review.

### Adversarial review (`reviewer-2`, 2026-11-12): major revision

The reviewer reproduced `hypotheses.csv` from `metrics.csv` with the functions of `run.py`. The
numbers and the provenance check out, and the commits predate the runs. The problem is in the
inference and in the reading.

**Checked by me, with code in the repository** (`exploratory/robustness.py`, which writes
`results/exploratory/robustness.json`):
- **H1 does not hold.** At 8×, the exact sign-flip test, stratified by race, gives
  p = 0.0625 and Holm p = 0.25. The pre-registered bootstrap with B = 200 000 gives Holm p = 0.055.
  - Only 6 bearings differ, 5 of them in favour.
  - With 6 bearings that respond, the sign-flip test could not pass Holm even with all 6 in
    favour: the smallest p would be 1/64.
  - The pre-registered "support" (Holm p = 0.04999, with B = 4000) is Monte Carlo error.
  - By this README's refutation criterion, H1 is refuted.
- **H2 holds only at 1/32.** The sign-flip test gives p = 0.0039 and Holm p = 0.016, with all 8
  bearings that differ in favour. At 4× it does not pass: Holm p of 0.20 by the sign-flip test and
  of 0.07 by the bootstrap.
- **H3 has a null result** in both tests.

**Checked by me on the raw signal** (`exploratory/inverter_lines.py`, check C4, measurement 2 of
each record, 68 readable):
- **There is no PWM line in the vibration.** In no record does a line appear 10 dB above the local
  median at 16 kHz ± 2k·f_e or at 32 kHz − (2k+1)·f_e. f_e is measured on the phase current, with a
  median of 101.6 Hz.
- **The 15–17 kHz band is more impulsive than the 11–15 kHz band** in 68 of 68 records, with a
  median spectral kurtosis of 10.9 against 5.8.
- The full-rate capture is therefore associated with impulsive content near 16 kHz, but the
  source is not established. The inverter hypothesis is neither confirmed nor refuted.

**From the reviewer, with no script in the repository:**
- **The ruler is not neutral.** It comes from 12 calibrations on 6 bearings, all measured in the
  captured band.
  - The variation of the kinematic frequencies between manufacturers (1 to 2 %, PHME16) exceeds
    the tolerance of 0.025 order.
  - There are near-hits outside the window: the SES on KA08 and the kurtogram without spikes on
    KI04.
  - The ruler tends to favour the kurtogram, which makes H2 conservative.
- **The replication does not test the Ottawa mechanism.** The mechanism presupposes the correct
  selector at full rate, and that only holds on 2 of the 23 bearings: KA01, with 63 %, and KI01,
  with 98 %.
  - On KI01 there is a wall, and not a valley: 98, 81, 75, 46 and 2 %.
  - On KA01, KA04 and KA16, decimation helps, which suggests a search-ceiling effect (not
    isolated).
  - The grid has neither 2× nor 3×.
- **The 11.0–21.5 kHz band is too wide to point to the inverter.** Folding to 0 Hz is not
  testable with factors that are multiples of 4. The arithmetic of sampling at a submultiple of the
  carrier is known: `blasko1998pwmsampling`.
- Correction: "from 2 to 730 spikes per second" comes from the exploratory check. In E24 there are
  from 0 to 5553 spikes per 4 s segment, up to about 1390/s on KI08.

**What the paper can claim:**
- On Paderborn, pure decimation produced no valley from 1/4 to 1/32.
- Removing the single-sample spikes, before or after decimating, changes little and without robust
  significance.
- At full rate, the kurtogram picks bands above 11 kHz in 88 % of the segments, with no hit in
  them. This is compatible with impulsive interference, and the source was not verified.
- The SES infogram without pre-whitening is better at 1/32 (+3.1 points, Holm p = 0.016), almost
  only on the outer race.
- Paderborn does not test the Ottawa mechanism.

**What it cannot claim:**
- "H1 supported" or "the capture by spikes reproduces";
- "the inverter captures" or "folding to 0 Hz explains", as fact;
- "robust selectors solve it";
- "decimation improves the diagnosis";
- "the mechanism generalizes", or that it is refuted.

**Suggested checks, not done:**
- C1: ceiling without folding at 1×.
- C2: factors 5, 7 and 9.
- C3: *notch* at 15–17 kHz and 30.5–32 kHz, with a control.
- C5: ruler independent of the arm.

## E24b: ceiling without folding and notch (pre-registration, 2026-11-12)

Written and committed before running, together with the analysis
(`exploratory/teto_notch_analysis.py`; `teto` = ceiling). They are checks C1 and C3 of the
adversarial review.

The question: on Paderborn, does decimation help because it narrows the search band (ceiling
effect) or because of what folds? And is the content near 16 kHz what captures the kurtogram at
full rate?

- **Arms** (`config/teto_notch.yaml`), all without removing the spikes:
  - **C1:** `kurtograma_teto8k` (kurtogram with an 8 kHz ceiling), `teto4k`, `teto2k` and
    `teto1k`, plus the same four for the SES infogram. All run at 1×, with the search ceiling at
    8, 4, 2 and 1 kHz.
  - The search partition is identical to that of 4×, 8×, 16× and 32×, because it only depends on
    [f_min, min(ceiling, fs/2)]. The difference is the folded content and the envelope rate.
  - **C3:** `kurtograma_notch` zeroes 15–17 kHz and 30.5–32 kHz at full rate, before decimating,
    and runs at 1× and 4×. `kurtograma_notch_controle` (control notch) zeroes bands of the same
    width at 12–14 kHz and 26.5–28 kHz.
  - The reference arms with no ceiling and no notch are read from `results/`. The code is the same
    and is deterministic.
- **Criteria** (nested ruler; exact sign-flip test by bearing, stratified by race):
  - **C1a, folding.** It is the fraction of segments in the floor band (start ≤ 501 Hz and
    end ≤ 1200 Hz) with an 8 kHz ceiling at 1×.
    - With ≥ 20 %, the floor choice at 4× (23 %) is baseband content, and folding is
      refuted as the cause.
    - With ≤ 10 %, folding is implicated. In between, inconclusive.
  - **C1b, ceiling.** It is the kurtogram with a ceiling against the kurtogram without a ceiling,
    both at 1×, with Holm over the 4 ceilings. The ceiling relieves the capture if any of them
    passes (Holm p < 0.05, difference > 0).
  - **C1c (descriptive).** It is the ceiling at 1× against the decimation with the same partition.
    It tells whether the gain from decimation comes from the ceiling or from the folding.
  - **C3a, notch.** Among the segments in which the kurtogram at 1× picks 11.0–21.5 kHz, it is the
    fraction that changes band with the notch.
    - With ≥ 50 %, the content of the zeroed bands sustains the choice. With < 20 %, it does not.
    - In between, inconclusive.
  - **C3b, validity.** The control notch must change the band in less than 10 % of the segments at
    1×. If it changes more, C3a is not interpretable.
- **Out of scope:** C2 (factors 5, 7 and 9) and C5 (ruler independent of the arm).

### E24b result (2026-11-13)

- **Run:** `results-teto-notch/`, 69 records in 20 108 s. The hashes of the config, of `run.py`
  and of the README in the manifest match the pre-registration commit (`2fbb1ac`).
- **Analysis:** `exploratory/teto_notch_analysis.py` wrote `results-teto-notch/analysis.json`. The
  paper's numbers come from `papers/shared/figures/num_partA_paderborn.py`.
- **Balanced accuracy,** nested ruler:

  | arm | 8 kHz ceiling | 4 kHz ceiling | 2 kHz ceiling | 1 kHz ceiling | no ceiling (E24) |
  |---|---|---|---|---|---|
  | kurtogram at 1× | 25.1 | 28.8 | 26.7 | 18.1 | 7.7 |
  | SES infogram at 1× | 27.7 | 27.9 | 21.0 | 24.1 | 25.4 |

- **C1a, folding: refuted as the cause of the floor.** With an 8 kHz ceiling at 1×, with nothing
  folded, the kurtogram picks the floor band in 45.4 % of the segments. At 4× it is 28.4 %.
  - The floor choice is baseband content.
  - In the other pairs the fraction is also larger without folding: 51 % against 48 % (4 kHz and
    8×), 61 % against 9 % (2 kHz and 16×) and 64 % against 16 % (1 kHz and 32×).
- **C1b, ceiling: did not pass.** No ceiling passes Holm over the 4.
  - For the kurtogram, the differences are +17.3, +21.0, +19.0 and +10.4 points (8, 4, 2 and
    1 kHz). The bearings in favour are 12 of 14, 13 of 14, 11 of 13 and 13 of 15.
  - The one-sided p ranges from 0.016 to 0.090, and the Holm one is 0.064, 0.064, 0.064 and 0.090.
  - It is absence of evidence, not evidence of absence. With 13 to 15 discordant bearings and
    Holm over 4, the exact test only passes with almost all in favour at more than one ceiling.
  - For the SES infogram the ceiling changes nothing: from −4.4 to +2.5 points, Holm 1.0.
- **C1c (descriptive).** The ceiling at 1× is above the decimation with the same partition for the
  kurtogram: +9.0 (8 kHz against 4×), +10.1 (4 kHz against 8×), +7.5 (2 kHz against 16×) and
  +5.1 points (1 kHz against 32×), with two-sided p from 0.09 to 0.29.
  - On Paderborn, decimation helps by narrowing the search, not because of what folds.
- **C3, notch: not interpretable.** The control notch (12–14 and 26.5–28 kHz) changed the band in
  27.3 % of the segments at 1×, above the 10 % limit.
  - The 15–17 and 30.5–32 kHz notch moved 19.4 % of the 909 captured segments out of the
    11.0–21.5 kHz band.
  - The accuracy changed by +1.2 points at 1× and +1.0 at 4×, with 1 discordant bearing.
  - The kurtogram's choice is fragile to any notch in that region, and C3a does not isolate the
    captor.
- **Consequence for the paper:**
  - The reading that decimation relieves the capture by narrowing the search is supported
    descriptively, but not by the pre-registered test.
  - The 15–17 kHz captor remains not established.

