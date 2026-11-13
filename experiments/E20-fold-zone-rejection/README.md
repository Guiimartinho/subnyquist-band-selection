# E20 - Zone rejection before decimating: what captures the kurtogram at 4×

Opened on 2026-11-10 by [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md). It is
Part 1 (mechanism) of the single paper.

## Context

In E07 (Ottawa 2018, the `sem_prior` (without prior) column), the outer race drops from 23/36 hits
at 8× to 6/36 at 4×, while the inner race stays at 31–32/36. At 4× the kurtogram picks the band
[12.75, 25] kHz in 26/36 outer-race segments and misses all of them.

The report's hypothesis was that content at 75–87.2 kHz folds onto that band. But that cause
is not isolated:
- the same range also folds at 2×, where there is no valley;
- at 4×, 25–37.25 kHz and 62.75–75 kHz also start to fold;
- at 8× the band [12.75, 25] kHz lies above the effective Nyquist and leaves the search.

There are therefore three rival explanations:
1. **folding of one specific zone**;
2. **folding of the set of zones**;
3. **search geometry**: the search ceiling relative to the effective Nyquist, or the kurtogram's
   dyadic partition, which changes with `fs_eff`, and not the folding.

## Hypothesis

The 4× valley on the outer race is caused by **folded content**. Removing, before decimation, the
zones of original frequency that fold onto [12.75, 25] kHz at 4× (Z1 = 25–37.25 kHz,
Z2 = 62.75–75 kHz, Z3 = 75–87.25 kHz) brings the outer race at 4× back to **at least 20/36**.
The ideal anti-alias filter has the same effect.

Secondary hypotheses, reported whatever the result:
- **H2 (attribution):** a single zone accounts for most of the recovery.
- **H3 (ceiling):** lowering the search ceiling to 12.5 kHz also recovers the outer race at 4× and
  worsens the inner race at 1× and 2×, whose informative band is at 16.8–25 kHz.

## What would refute this hypothesis

- **Refutes folding:** at 4× on the outer race, `aaf_ideal` **and** `stop_z123` reach
  **at most 10/36**. In that case the valley is not folding: it is search or partition geometry
  (explanation 3), and Part 1 of the paper changes thesis.
- **Refutes H2:** no single zone recovers more than half of what `stop_z123` recovers.
- **Refutes H3:** with a 12.5 kHz ceiling, the outer race at 4× still reaches at most 10/36.

**Recovery criterion:** at least 20/36 (the 8× level is 23/36) **and** exact paired McNemar
against `none`, at the same factor and the same ceiling, with p < 0.01.

## Baseline

The E07 pipeline without intervention (`none`, 25 kHz ceiling): restricted kurtogram, envelope,
order spectrum and estimate without prior. It is the project's "band-pass + envelope".

**Reproduction control:** `none` with a 25 kHz ceiling must reproduce E07 exactly at 1×,
2×, 4× and 8× (56, 53, 37 and 55 hits out of 72). If it does not, nothing below holds.

## Ground truth

The same as in E07:
- **Calibration:** fault order calibrated per record at full rate, refined around the
  nominal order (BPFO or BPFI of the ER16K).
- **Truth:** the median of the **other** records of the same condition (leave-one-out).
- **Hit:** |estimate − truth| < 0.06 order.

The fault label comes from the dataset. The rotation comes from the encoder (1024 ppr), which is
independent of the vibration channel. Data: Ottawa 2018, Mendeley `v43hmbwxpm`, CC BY 4.0.

## Procedure

1. **Pass 1**, identical to E07: calibrate the order of each fault record (I and O, 24
   records) at full rate.
2. **Pass 2.** For each record, segment (3 × 2.5 s), intervention, factor
   (1, 2, 3, 4, 5, 6, 8) and ceiling (25 kHz, 12.5 kHz):
   - apply the intervention to the segment **at full rate**, as an ideal zero-phase FFT mask;
   - decimate without a filter;
   - run the kurtogram with ceiling `min(ceiling, fs_eff/2)`;
   - compute the envelope order spectrum and the estimate without prior.

   Factors 3, 5 and 6 show where each zone folds outside the powers of 2.
3. **Execution** in parallel per (record, segment), on the server.
4. **Analysis:** hits per condition × factor × intervention × ceiling, exact paired McNemar
   against `none`, and the chosen band.

| intervention | what it removes before decimating |
|---|---|
| `none` | nothing (control) |
| `stop_z1` | 25–37.25 kHz |
| `stop_z2` | 62.75–75 kHz |
| `stop_z3` | 75–87.25 kHz |
| `stop_z123` | the three zones |
| `aaf_ideal` | everything above `fs/(2·factor)`, i.e. no folding |

## Metrics

- order hit (`sem_prior`) per condition, factor, intervention and ceiling, with `n`;
- exact paired McNemar against `none`;
- band chosen by the kurtogram, and kurtosis;
- measurement/prior attribution: **does not apply**, because there is no reconstruction and no
  linear operator (see `bhadra2021hallucinations` and ADR 0006).

## Result

Run on 2026-11-11 on the remote server (CPython 3.14.10), with 6048 rows in
`results/metrics.csv`, a table in `results/summary.csv` and a manifest with `source_sha256` and
`entrypoint_sha256`.

**Re-run with the DSP core corrected** by the independent audit of 2026-11-11
(kurtogram without the circular wrap-around, angular resampling with anti-alias; see that day's
lab notebook). Relative to the provisional run, no cell changed by more than 2 hits, and all the
conclusions below still hold.

**Control:** `none` with a 25 kHz ceiling exactly reproduces E07 re-run with the corrected DSP:
57, 53, 38 and 57 of 72 at 1×, 2×, 4× and 8×.

**Outer race, 25 kHz ceiling**, hits out of 36:

| intervention | 1× | 2× | 3× | 4× | 5× | 6× | 8× |
|---|---|---|---|---|---|---|---|
| `none` | 25 | 21 | 4 | 6 | 2 | 11 | 24 |
| `stop_z1` (25–37.25 kHz) | 25 | 21 | 4 | 7 | 3 | 12 | 24 |
| `stop_z2` (62.75–75 kHz) | 25 | 21 | 4 | **22** | 12 | 12 | 25 |
| `stop_z3` (75–87.25 kHz) | 25 | 26 | **20** | **22** | 2 | **24** | 25 |
| `stop_z123` | 25 | 26 | 22 | **25** | 15 | 23 | 27 |
| `aaf_ideal` | 25 | 25 | 25 | **25** | **26** | 24 | 26 |

At 4×, against `none`: `stop_z2` and `stop_z3` give p = 3.1·10⁻⁵, and `stop_z123` and `aaf_ideal`
give p = 4·10⁻⁶ (exact paired McNemar).

**12.5 kHz ceiling:** the outer race stays at 26–27/36 at every factor from 1× to 6×, and at
24–27 at 8×, with any intervention. Paired against the 25 kHz ceiling (`none`): 26 against 4 at
3×, 27 against 6 at 4×, 27 against 2 at 5× and 27 against 11 at 6×, always **with no loss at all**
(p from 6.0·10⁻⁸ to 3.1·10⁻⁵).

**Inner race:** stays between 31 and 33 of 36 in every cell, with both ceilings.

**The band chosen at 4× on the outer race confirms the mechanism:**

| condition | chosen band |
|---|---|
| `none`, 25 kHz ceiling | [12.75, 25] kHz in 26/36 |
| `stop_z123` | 2.0–4.6 kHz in 32/36 |
| `aaf_ideal` | 2.0–4.6 kHz in 32/36 |
| 12.5 kHz ceiling | 2.0–4.5 kHz in 32/36 |

**Original reading (2026-11-11, morning), corrected below.**
1. **Main hypothesis.** The valley comes from content above the effective Nyquist that survives
   pure decimation. The ideal anti-alias filter eliminates it at every factor. **This holds.**
2. **H2 (attribution).** "At 4× the capture requires Z2 and Z3 together", and "at 3× and 6× it is
   Z3". **Withdrawn**; see the review below.
3. **H3 (ceiling).** Lowering the ceiling to 12.5 kHz eliminates the valley, and the prediction that
   this would worsen the inner race was refuted. This holds, but the ceiling does not distinguish
   folding from search geometry.

### Adversarial review (`reviewer-2`, 2026-11-11), checked

**The captor is a train of isolated single-sample spikes in the raw record, not band content at
62.75–87.25 kHz.**

- **Spike removal (*despike*) at full rate, without zeroing any band.** Every sample that
  deviates by more than 10 MAD from the neighbours' mean, with the neighbours deviating in the
  opposite direction, is replaced by that mean. This affects about 0.65 % of the samples. The outer
  race reaches **26/26/26/26/25/23/27** of 36 from 1× to 8×, against 25/21/4/6/2/11/24 for `none`.
  It ties with `aaf_ideal` and beats `stop_z123` at 5× (25 against 15). The 3-sample median gives
  the same. On the inner race nothing gets worse.
- **Independent check:** I re-ran the table and four segments at 4×. On O-B-2 and O-D-1,
  `none` picks 12.75–25 kHz and misses, and *despike* goes back to 2.54–4.58 kHz and hits.
- **The spikes are in every record, including the healthy ones.** H-A-1 has about 2460
  spikes/s and O-A-1 about 1330/s, with negative deviation in 96 % of cases, at 16–18 MAD.
  According to the reviewer:
  - from 40 to 100 kHz the spectrum is flat within ±1 dB;
  - the RMS in 62.75–87.25 kHz is the same in the three conditions.

  A single-sample event (5 µs) is not an accelerometer response. It comes from the acquisition
  chain: interference sampled without anti-alias, or an artefact of the acquisition system.
- **"Z2 and Z3 together" does not identify two sources.** At 4×, `stop_z2` and `stop_z3` give the
  same result in 36/36 segments. But that is what is expected if the two removals merely undo the
  capture and the selector goes back to 2–4.6 kHz, a band onto which neither of the two zones folds.
  At 3×, 5× and 6× they do not coincide (11, 20 and 13 of 36).
- **Open point.** Removing Z1 (25–37.25 kHz), which at 4× also folds onto 12.75–25 kHz, does not
  help (7/36). This is compatible with Z1 having strong genuine vibration content (there is a
  resonance near 34 kHz at full rate) besides the spikes. It is not isolated.

**Statistics.**
- **Segment level.** McNemar treats the 36 segments as independent, but there are 3 per record.
- **Record level.** The anti-alias effect survives: 11 records improve, 1 ties and 0
  get worse, sign test p = 9.8·10⁻⁴. That is 30 to 250 times larger than the per-segment p.
- **Scenario level.** With 4 rotation scenarios, no test reaches p < 0.01.
- **Bearing level.** There is one bearing per fault type.

**Physical wall.** Decimating a 200 kSPS file is not acquiring at 50 kSPS.
- If the spikes are a digital artefact of the original acquisition system, a real ADC sampling
  below Nyquist does not reproduce them.
- If they are interference in the analogue chain, a converter sampling below Nyquist without a
  low-pass filter, the case of band-pass sampling, would fold them in the same way.

The dataset does not allow distinguishing the two cases.

**What Part 1 can claim.** In a public dataset, sparse single-sample spikes in the raw
record make pure decimation non-monotonic in hits. From 3× to 6×, the kurtosis-based band selector
is captured by a band where the folded energy of the spikes dominates. Anti-alias or spike removal
before decimating undoes the effect.

The capture of spectral kurtosis by impulsive electrical interference at full rate is the subject of
`smith2016emi` and `vaerenberg2025cyclicemi`. What remains as our own is narrow: the return, by
folding, of the content that the search ceiling excluded at full rate, and how this depends on the
factor.

**What remains pending.**
- *Despike* and the median of 3 as pre-registered arms here and in E22.
- Temporal characterization of the content above 40 kHz.
- Tests at the record and scenario levels.
- In E23, a single-sample spike-train arm against a band-limited burst of the same energy.

**What was surprising.**
- The valley runs from **3× to 6×**. The E07 grid (1, 2, 4, 8, ...) only showed 4×.
- The E07 25 kHz ceiling was set to remove from the search the 50–100 kHz content, which at full
  rate captured the selector. **Under pure decimation, that same content comes back by folding
  into the protected band.** A search restriction that protects the selector at full rate is
  undone by undersampling.
- The rig's shaft is driven by an inverter. The review showed that the captor is a train of
  single-sample spikes also present in the healthy records (see above). Whether the origin is
  electromagnetic interference or an artefact of the acquisition system, the dataset does not say.

**What this changes for the paper** (morning text; the mechanism was corrected by the review above).
The anti-alias effect is established by intervention on Ottawa, and the causal agent is the
single-sample spikes. The two simple mitigations (anti-alias filter, search ceiling below the band
where the folded content lands) become **mandatory baselines** of E22: the proposed method has to
tie with or beat them. E21 tests whether the phenomenon, and a predictor computed from a
full-rate capture, generalize to MAFAULDA.

## Re-run with the corrected ruler (2026-11-11, afternoon)

Run at 12:56 UTC on the server, with three changes:
- truth at the **plateau centre** (`snq_recon.refine_order`);
- **local ruler** written alongside (`truth_local_order`, `hit_local`: the full-rate line of the
  segment itself, because the outer-race BPFO order drifts with the rotation);
- the pre-registered interventions `despike` and `median3`.

The control matches exactly E07 re-run with the same ruler: 62, 55, 41 and 61 of 72 at 1×,
2×, 4× and 8×. The run's exit with code 2 was the control gate comparing against the old ruler's
values, and the config was updated.

Outer race, 25 kHz ceiling, hits from 1× to 8× (1, 2, 3, 4, 5, 6, 8):

| intervention | old ruler (of 36) | local ruler (of 35) |
|---|---|---|
| `none` | 27 · 20 · **4 · 6 · 2 · 11** · 25 | 31 · 22 · **4 · 6 · 2 · 11** · 28 |
| `stop_z2` | 27 · 20 · 4 · 21 · 12 · 12 · 26 | 31 · 22 · 4 · 24 · 13 · 12 · 29 |
| `stop_z3` | 27 · 28 · 19 · 21 · 2 · 24 · 26 | 31 · 32 · 21 · 24 · 2 · 27 · 29 |
| `stop_z123` | 27 · 28 · 21 · 27 · 15 · 23 · 29 | 31 · 32 · 23 · 31 · 16 · 26 · 32 |
| `aaf_ideal` | 27 · 27 · 27 · 27 · 27 · 25 · 27 | 31 · 31 · 31 · 31 · 29 · 28 · 30 |
| **`despike`** | **28 · 28 · 28 · 28 · 27 · 24 · 29** | **32 · 32 · 32 · 32 · 30 · 27 · 32** |
| `median3` | 28 · 28 · 28 · 28 · 25 · 25 · 28 | 32 · 32 · 32 · 32 · 28 · 28 · 31 |

The inner race stays at 35–36/36 in every cell, under both rulers. Before the ruler correction
it was at 31–33.

**Reading.**
- The 3× to 6× valley is the same under both rulers.
- Removing the single-sample spikes (about 0.65 % of the samples, without zeroing any band) undoes
  it entirely and ties with or beats the ideal anti-alias.
- This confirms, in an official and pre-registered run, the finding of the adversarial review: the
  captor is the spikes.
- The local ruler raises the outer-race level at full rate (27/36 → 31/35), which shows that
  part of the earlier errors came from the ruler, not from the method.

## Robust selectors (pre-registration, 2026-11-12)

Written before running. The question is the first one a reviewer asks: does a band selector robust
to impulses avoid the 4× valley **without** removing the spikes?

- **Selectors:** from `snq_recon.selectors`, tested against the values of the source papers
  themselves.
  - `infograma_ses` (SES infogram): the negentropy of the squared envelope spectrum (Antoni 2016,
    eq. 12). According to section 5.1.1 of that paper, it is insensitive to impulsive noise.
  - `gini`: the Gini index of the band envelope (Hebda-Sobkowicz et al. 2020, eq. 3).
  - Both run on the same kurtogram partition.
- **Arms** (`config/selectors.yaml`): `none` and `despike` repeat those of E20; the new ones are
  `infograma_ses`, `infograma_ses_despike` and `gini`. Factors, ceilings, ruler and the E07 control
  stay the same.
- **Criterion:** the same as the E20 recovery, on the outer race, at 4× and with a 25 kHz ceiling.
  - Recovers: at least 20 of 36 hits and exact paired McNemar against `none` with p < 0.01.
  - Does not recover: at most 10 of 36.
  - It holds for both rulers (the E07 one and the local one).
- **Reading:**
  - If `infograma_ses` recovers, a robust selector is enough, and the correction does not require
    the full rate.
  - If it does not recover, removing the spikes before decimating remains necessary, and the robust
    selectors from the literature do not solve the capture under pure decimation.

## Removal after decimating (pre-registration, 2026-11-12)

Written before running. The `despike` arm removes the spikes at full rate, which the node cannot
do. The question here is whether the same removal, applied to the **already-decimated** stream,
undoes the valley.

- **What changes:** only the moment of the removal.
  - A single-sample spike at full rate only survives decimation if it falls on a kept sample.
  - If it does, it stays isolated in the decimated stream.
  - The criterion is the same as in E20 (10 MAD, opposite neighbours smaller than 0.8 times),
    computed at the effective rate.
- **Arms** (`config/post_despike.yaml`): `none` and `despike` repeat those of E20, and the new one
  is `despike_decimado` (despike after decimation).
- **Criterion:** that of the E20 recovery, on the outer race, at 4× and with a 25 kHz ceiling.
  - Recovers: at least 20 of 36 hits and exact paired McNemar against `none` with p < 0.01.
  - Does not recover: at most 10 of 36.
  - It holds for both rulers.
- **Reading:** if it recovers, the correction is realizable on the node, because it does not need
  the full rate.

**Analysis of the two pre-registrations above** (written on 2026-11-12, before any number from the
two runs):
- The script is `recovery.py <directory>/metrics.csv`, which writes `recovery.csv` alongside.
- It applies the criterion under both rulers: the nested one, which is the `hit` column, and the
  local one, `hit_local`.
- It also reports the exact sign test over the 12 outer-race records. That test is a sensitivity
  analysis, outside the pre-registration, and answers the "Segment level" limitation above.
- Applied to the current `results/`, the script reproduces the re-run table. `despike` at 4×
  reaches:
  - 28/36 against 6/36, with 22 to 0 in the discordant pairs, under the nested ruler
    (p = 4.8·10⁻⁷);
  - 32/35 against 6/35 under the local ruler (p = 3·10⁻⁸);
  - 11 records to 0 in the sign test (p = 0.001).

## Result of the robust selectors and of the removal after decimating (2026-11-12)

The two runs finished at 18:27 and 18:28 UTC on the server.
- **Provenance.**
  - Selectors: `config/selectors.yaml` and `run.py` from commit `4a67eae`.
  - Removal after decimating: `config/post_despike.yaml` and `run.py` from commit `6a95a72`. It ran
    in a separate checkout, so as not to change `run.py` under the selectors run.
  - The `input_sha256` is the same as in the base run, with the 24 records.
  - The E07 control matches exactly in both: 62, 55, 41 and 61 of 72.
- **Pre-registered criterion** (`recovery.py`, `recovery.csv`): outer race, 4×, 25 kHz ceiling.

| arm | nested ruler (of 36) | local ruler (of 35) | verdict |
|---|---|---|---|
| `despike_decimado` (realizable on the node) | **29** against 6, +23/−0, p = 2.4·10⁻⁷ | **32** against 6, +26/−0, p = 3·10⁻⁸ | **recovers** under both |
| `infograma_ses` | 11, +10/−5, p = 0.3 | 11, +10/−5, p = 0.3 | inconclusive |
| `infograma_ses_despike` | 7, +6/−5, p = 1 | 7, +6/−5, p = 1 | does not recover |
| `gini` | 0, +0/−6, p = 0.031 | 0, +0/−6, p = 0.031 | does not recover (worse than `none`) |

- **Per-record sensitivity** (sign test, outside the pre-registration). For `despike_decimado`,
  12 records improve and none gets worse under the nested ruler (p = 0.0005), and 11 to 0 under
  the local one.
- **At the other factors** (nested ruler, 25 kHz ceiling):
  - `despike_decimado` scores 28, 29, 29, 29, 23, 25 and 26 from 1× to 8×, against 27, 20, 4, 6, 2,
    11 and 25 for `none`. The valley disappears.
  - `infograma_ses` scores 7, 7, 11, 11, 17, 7 and 13.
  - `gini` scores 15 at 1×, 0 from 2× to 6× and 22 at 8×.
- **Inner race:** no arm harms it; it stays at 34–36/36 in every cell.
- **12.5 kHz ceiling:**
  - `despike_decimado` scores 26–29;
  - `gini` scores 20–23, because the band that the spikes capture lies above the ceiling;
  - `infograma_ses` scores from 0 to 13.
- **Chosen bands** (outer race, 25 kHz ceiling):
  - `despike_decimado` returns at 4× to the full-rate band, 2.5–4.6 kHz, in 25/36.
  - `gini` goes to 18.9–20.9 kHz at 4× (23/36). That is the capture.
  - `infograma_ses`, already at full rate, picks 4.6–6.6 kHz (18/36) or 0.5–12.75 kHz (13/36), and
    not the kurtogram's band (2.5–4.6 kHz).

**Pre-registered reading:**
- Spike removal **after** decimating recovers the hits, and so the correction is realizable
  on the node, without the full rate.
- No robust selector recovers by the criterion. The Gini is captured. The SES infogram is
  inconclusive at 4×.

**Not yet explained, to be checked before it goes into the paper.** The SES infogram hits only 7
of 36 at full rate, against 27 for the kurtogram.
- The hypothesis is Ottawa's variable rotation: the negentropy of the envelope spectrum in Hz
  assumes a stationary cyclic frequency, and a ramp smears the fault line.
- The other possibility is an implementation error, despite the tests against the paper's values.
- An order-domain infogram was not pre-registered and would be exploratory.

### Adversarial review (`reviewer-2`, 2026-11-12): major revision

The reviewer independently recomputed:
- the 144 estimates;
- the pairings;
- the provenance: `run.py`, config, and the `source_sha256` 8491841f… over 32 files;
- the `none` and `despike` columns, identical across the three runs.

The pre-registered effect is real, but the reading above overstated it.

**Checked by me in this run's CSVs:**
- **At 4×, removal after decimating ties with the 12.5 kHz ceiling,** which also does not require
  the full rate.
  - `despike_decimado` with a 25 kHz ceiling and `none` with a 12.5 kHz ceiling score 29/36 and
    29/36 (nested ruler), with **0 discordant pairs**. Under the local ruler, 32/35 and 32/35, also
    with 0.
  - At 5×, the ceiling is better: 23 against 29, −6/+0, p = 0.031; under the local ruler, p = 0.016.
  - At 6×, 25 against 29 (p = 0.22).
  - The ceiling only works because the informative band (2.5–4.6 kHz) lies below it.
- **The two "robust" selectors already fail at full rate:** SES infogram 7/36 and Gini 15/36,
  against 27/36 for the kurtogram. The run does not decide whether a robust selector avoids the
  valley.

**Computed by the reviewer, with no script in the repository (exploratory until it becomes a run):**
- A **median of 3 on the decimated stream** also gives 29/36 at 4×. The "isolated sample" criterion
  is not what matters. Clipping the amplitude does not recover: 8/36 with 6 MAD and 5/36 with
  10 MAD.
- The effect withstands the decimation phase: offsets 0–3 give 29, 29, 29 and 28, against 6, 5, 5
  and 4 for `none`.
- By threshold: 5, 7 and 10 MAD give 29/36, and 15, 20 and 30 MAD give 23, 16 and 6.
- **On the inner race, the criterion replaces samples of the fault impacts,** and not acquisition
  spikes.
  - The replaced samples are phase-locked to the fault order: R ≈ 0.90, against 0.18–0.39 at the
    95th percentile of the null.
  - The estimate only changes in 0–2 segments.
  - At 6×, the line prominence drops by more than 10 % in 7 of 36 segments.
- **On the outer race, the replaced samples are spikes of the raw record** (median precision 1.0).
  - The criterion only catches part of the spikes that survive: 21.5 % at 4×, because the MAD of
    the second difference grows 3.7×.
  - Even so, the valley disappears.
- **SES infogram without error in eqs. 10–12,** but with two undeclared deviations: the
  deterministic/random separation (§4.4) and the pre-whitening (§4.5) are missing.
  - Of the 29 wrong estimates at 1×, 17 stop at order 2.0, which is `order_min`, and 11 at ~6.
  - Computing the SES in order raises it to 19/36, which partly supports the variable-rotation
    hypothesis.
  - On CWRU 48k, with constant rotation, the four selectors score 33/33. This rules out a gross
    error, but does not discriminate.
- **The Gini is captured:** removing the spikes returns it to the full-rate band.
- **Novelty:**
  - No spike removal on a decimated vibration stream to protect the band selector was found.
  - There are neighbouring practices: pulse *blanking* after the converter in receivers, including
    before and after decimation, and replacing the spike with the neighbours' mean in magnetic
    resonance.
  - The mechanism has a name in sampling theory: *noise folding* (Arias-Castro and Eldar, 2011).
  - Since the median of 3 ties, there is no new method; there is an empirical observation.

**What the paper can claim:**
- On Ottawa, at 4×, cleaning the already-decimated stream (isolated spikes above 10 MAD) raised
  the outer-race hits from 6/36 to 29/36. All 12 records improved (sign test, p ≈ 5·10⁻⁴).
- A median of 3 on the decimated stream and the 12.5 kHz search ceiling give the same hits at 4×,
  and none of the three corrections requires the full rate. The median is pending the confirmation
  run.
- On the inner race, the criterion also replaces samples phase-locked to the fault. The estimate
  did not change, but the criterion is not selective for the artefact.
- The Gini of the envelope is captured from 2× to 6×.
- The SES infogram without pre-whitening is not captured, but it hits 7/36 already at full rate,
  and neither of the two decides whether a robust selector avoids the valley.
- Everything assumes an ideal sampler: decimation of a record at 200 kS/s.

**What it cannot claim:**
- "correction realizable on the node";
- "removes only the artefacts";
- "the robust selectors from the literature do not solve the capture";
- "the infogram is captured";
- "the infogram fails because of the variable rotation", as fact;
- "Gini robust to impulses";
- "the valley disappears thanks to the removal", at every factor;
- per-segment p as the strength of the evidence; the per-record test is what counts;
- "new method" or "first time".

## Median of 3 on the decimated stream (confirmation run, 2026-11-12)

Written before running, but **it is not blind**: `reviewer-2` already computed this arm in
exploratory form, with 29/36 at 4×. The run is there so that the number that goes into the paper
comes from committed code and from a `metrics.csv` in the E20 format.

- **Arms** (`config/post_median.yaml`): `none`, `despike_decimado` (repeated) and the new
  `mediana3_decimado` (median of 3 after decimation), which applies the 3-sample median to the
  already-decimated stream.
  - The median is the same as in the full-rate `median3` arm (`apply_intervention`), applied after
    `undersample`.
- **Criterion:** the same as for the recovery, at 4×, under both rulers.
- **Paired comparison, descriptive:** `mediana3_decimado` against `despike_decimado`, and both
  against `none` with a 12.5 kHz ceiling, by two-sided McNemar and by the per-record sign test.
- **Reading that this README already assumes:** if the median ties, the "isolated sample" criterion
  is not necessary, and the correction is a simple cleaning of the decimated stream, with no new
  method.

### Result of the median of 3 on the decimated stream (2026-11-12)

- **Provenance.** The run was from the separate checkout `~/snq/post2`, at commit `e5ceb4e`.
  - The hashes of `run.py` and of the config in the manifest match the commit.
  - The E07 control matches exactly.
- **Criterion** (`results-post-median/recovery.csv`). `mediana3_decimado` at 4× **recovers** under
  both rulers:
  - nested ruler: 29/36 against 6/36, with +23/−0 (p = 2.4·10⁻⁷);
  - local ruler: 32/35 (p = 3·10⁻⁸);
  - per record: 12 to 0 (p = 0.0005).
  - It agrees with the reviewer's reanalysis.
- **At the other factors** (nested ruler): 28, 29, 29, 29, 27, 28 and 27 from 1× to 8×. The median
  ties with the removal of isolated spikes up to 4× and is better from 5× to 8×, where the latter
  scores 23, 25 and 26.
- **Reading:** the "isolated sample" criterion is not necessary. The correction is a simple cleaning
  of the decimated stream, and not a new method.


## The spike train (exploratory, 2026-11-13)

Requested by the adversarial review of the manuscript, which measured this in the scratchpad.
Reproduced with committed code: `exploratory/spike_train.py` →
`results-post-despike/exploratory/spike_train.csv`, on the 36 whole records and by the same
criterion as the `despike` arm.

- **Rate:** median of 1336 spikes/s on the healthy records (956 to 1377), 1301/s on the outer race
  (1270 to 1339) and 315/s on the inner race (251 to 505). Median deviation of 18.7, 18.9 and
  14.0 MAD. More than 95 % are negative in every record.
- **Periodicity:** 32 to 41 % of the inter-spike intervals are exactly 40 samples, and the spectrum
  of the spike instants has its dominant line at 10 001.2–10 001.3 Hz in all 36 records,
  120 ppm from 10 kHz.
- **Encoder:** the fraction of spikes within 1 sample of an edge differs from that of random
  instants by at most 0.023. With edges this dense, this test has little resolution.
- **Reading:** a 5 µs event is not an accelerometer response. It is interference in the
  acquisition chain, with its own clock near 10 kHz. The source was not identified.
- **Difference from the earlier text:** the counts of 2460/s (H-A-1) and 1330/s (O-A-1), quoted
  above, do not record the criterion used and do not match this one (H-A-1 is near 1300/s here).
  The numbers with committed code are those of this section.
