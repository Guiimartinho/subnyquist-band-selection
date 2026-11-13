# E22 - Published baselines at the same samples/s rate

Opened on 2026-11-11 by [ADR 0006](../../docs/adr/0006-artigo-unico-computacional.md). It fixes
what the learned mask (E15) has to beat. Every comparison is made at the **same mean rate of
converted samples per second**. All arms are reimplemented published methods, or the mitigations
established by E20. No straw man.

## Question

Given a budget of samples/s below the acquisition rate, which published method delivers the best
fault-order accuracy? Does random sampling with CS (the direct competitor) beat well-chosen uniform
sampling?

## Front-end scenarios

They are declared because they change which method is optimal.

- **A. No tunable analog filter.** For example, a MEMS accelerometer connected directly to the
  ADC. This is where the E20 fold happens, and it is the main scenario of the paper.
- **B. Fixed analog band-pass on the resonance**, chosen in a full-rate calibration
  (`lu2018undersampled`). It is the classical optimum when the band is known beforehand.

## Arms

| arm | front-end | method | reference |
|---|---|---|---|
| `uniform` | A | decimation at the budget rate, kurtogram up to the effective Nyquist | E07 pipeline (`antoni2007kurtogram`) |
| `uniform_ceiling` | A | same, with the search ceiling chosen by the TAT in the calibration | E20 mitigation |
| `tat_rate` | A | the **smallest** factor admissible by the TAT within the budget, that is, the highest admissible rate | E21 |
| `uniform_oracle` | A | the best uniform factor within the budget, chosen **with the test label** | upper bound; not a method |
| `random_cs` | A | random instants + OMP in an order basis + envelope | `sensors2025csvibration` (E13 arm) |
| `multirate` | A | two co-prime uniform streams adding up to the budget, with order consistency between them | `zhao2018coprime`, `sun2013multirate` |
| `bandpass_lu` | B | ideal band-pass on the calibration band + band-pass sampling at the lowest valid rate within the budget | `lu2018undersampled`, `vaughan1991` |
| `zhao_asns` | A | adaptive sub-Nyquist sampling | `zhao2026adaptive`, **only if the PDF allows reproducing it** |

Robust selectors (`antoni2016infogram`, `smith2016emi`, `mauricio2020iesfogram`) come in as a
variant of the `uniform` arm when implemented.

## What each method may use

Declared because it changes what is fair to compare:
- `tat_rate`, `uniform_ceiling` and `bandpass_lu` use **a short full-rate capture of the machine
  itself** (calibration).
- The E15 mask uses **training records** from other machines and conditions, and no capture of the
  test machine.
- The paper reports the two regimes, without mixing them.

## Hypotheses

- **H1.** At every budget, `random_cs` does not beat `uniform_ceiling`. Estimating without
  reconstructing wins over reconstructing, as in E13 at 64×.
- **H2.** `tat_rate` stays within at most 5 points of `uniform_oracle` at every budget.

## What would refute them

- **H1 is refuted** if `random_cs` beats `uniform_ceiling` at some budget, with a per-record
  bootstrap interval above zero. That is a publishable result against the current route, and the
  paper reports it as such.
- **H2 is refuted** if `tat_rate` falls more than 5 points below the oracle at some budget.

## Protocol

- **Datasets:** Ottawa (inner and outer races) and MAFAULDA (outer race, without the sequences
  excluded by the tachometer screening).
- **Budgets**, as a fraction of the acquisition rate: 1/4, 1/8, 1/16 and 1/32 on Ottawa; 1/4, 1/8
  and 1/16 on MAFAULDA.
- **Split:** calibration (1st half) and test (2nd half) within each record, for the arms that
  calibrate. Repeated splits **by record** (E05) for everything that is chosen in training.
- **Metrics:** order accuracy with its `n`, per-record bootstrap interval, and paired McNemar
  between arms at the same rate.

## Status

**Phase 1 implemented** (`run.py`). Arms: `uniform`, `uniform_ceiling`, `tat_rate`,
`uniform_oracle` (in the analysis), `random_cs` and `bandpass_lu`. Left for phase 2 are
`multirate`, which needs a fusion rule of its own, and `zhao_asns`, which depends on the PDF. The
helper tests are in `tests/`.

**Smoke test** (2026-11-11; `config/smoke.yaml`: Ottawa, 2 inner-race records, budgets 1/16 and
1/32):
- The pipeline runs end to end in the six arms.
- `random_cs` misses 8 of 8 cells, consistent with E13.
- `bandpass_lu` only became faithful after **choosing, among the bands of the calibration
  kurtogram, the most impulsive one that the budget can sample**. In the first version, which used
  the best band without restriction, that band was too wide for any valid rate. After the fix, it
  hits 8 of 8.
- `uniform_ceiling` with a ceiling of 1/4 lost the informative band in one record and fell onto the
  2nd harmonic. This is the behaviour of the rule, and it is reported as such.

The smoke test is not a result; the complete run uses `config/base.yaml` on the server.

## Result (2026-11-11, phase 1, Ottawa)

Complete run on the server: 24 records (12 from the inner race and 12 from the outer race),
segment 0 for calibration and segments 1 and 2 for test, or 48 segments per budget. Analysis in
`analyze.py`, written to `results/analysis.json`, with a per-record bootstrap stratified by group
(4000 resamples) and an exact McNemar test paired by segment.

**MAFAULDA was left out of this run**. E21 showed that its BPFO order coincides with the 3rd shaft
harmonic, and so the accuracy there does not measure the fault.

Accuracy (%) over the two groups combined:

| arm | 1/4 | 1/8 | 1/16 | 1/32 |
|---|---|---|---|---|
| `uniform` (factor = budget) | 56.2 | 89.6 | 85.4 | 87.5 |
| `uniform_ceiling` | **87.5** | 89.6 | 54.2 | 29.2 |
| `tat_rate` | 79.2 | 89.6 | 85.4 | 87.5 |
| `bandpass_lu` (Lu 2018) | 79.2 | 79.2 | 79.2 | 72.9 |
| `random_cs` (Kato & Otaka) | 2.9 | 4.2 | 6.2 | 4.2 |
| `uniform_oracle` (chosen on the test) | 97.9 | 97.9 | 97.9 | 97.9 |
| best uniform rate chosen on other records (*post hoc*) | 95.8 | 95.8 | 93.8 | 95.8 |

**H1: the numbers stand, but the comparison is invalid (review of 2026-11-11).** Random sampling
with CS falls 25 to 85 points below `uniform_ceiling` (95 % CI below zero) and loses to `uniform` in
every discordant segment. The arm, however, is a **straw man by construction**:
- The dictionary covers orders 0.5 to 24 (`config/base.yaml`, section `cs`), up to about 520 Hz in
  the Ottawa segments. It lies below the kurtogram's 500 Hz floor and far from the resonances that
  carry the fault (2–4.6 kHz and 16.8–25 kHz). The envelope of the reconstructed signal cannot
  contain the fault modulation.
- The system is overdetermined at every budget: about 2,316 columns, with a rows/columns ratio
  from 54 down to 6.7. It is not *compressed sensing*.
- Kato & Otaka (`sensors2025csvibration`) reconstruct 0–2,500 Hz in the bearing case.

**H1 leaves the paper** until a faithful arm exists: a dictionary that covers the resonance, an
underdetermined regime and a fast operator. The same limitation applies to the CS arm of E13 (see
the report, §5).

**H2, refuted at all four budgets.** `tat_rate` falls below the oracle by −18.7 [−29.2; −8.3],
−8.3 [−14.6; −2.1], −12.5 [−18.8; −6.2] and −10.4 [−18.8; −4.2] points.
- **What the rule does well:** it rescues the 1/4 valley against `uniform` (11 × 0, p = 0.001).
- **What it does not do:** find the best rate. It chooses the highest admissible rate, and the
  best uniform rate for the outer race is factor 46 (4.35 kSPS), below any budget.

**The claim "factor 46 generalizes across records" was withdrawn** (*post hoc* analysis; review of
2026-11-11).
- **Leakage through the truth.** The truth of each record, the leave-one-out median of the
  group's calibrations, also uses the records of the test fold. With the **nested** truth
  (training records only), the best uniform rate chosen on other records gives 87.5 / 87.5 / 89.6
  / 91.7 % over both races combined, and **75.0 / 75.0 / 79.2 / 83.3 %** on the outer race,
  instead of 95.8 / 95.8 / 93.8 / 95.8 %. Both sets of numbers were recomputed independently
  (`check_nested_cv.py`, notebook of 2026-11-11).
- **46 is the top of a plateau, not an optimum.** Paired, it cannot be told apart from its
  neighbours (against 45: 3 × 0, p = 0.25; against 47: 4 × 0, p = 0.125). By the reviewer's
  calculations, not yet redone, under a null that permutes the factor within the segment
  P(max ≥ 23) = 0.019, and in the per-scenario validation (LOSO) with nested truth 46 comes back
  in 2 of 4 folds. What generalizes is a plateau of factors ≥ 7, outside the 3× to 6× valley.
- **The 12 records of each group come from a single bearing.** "Across records" means across
  repetitions of the same test.
- **Against the uniform rate chosen with nested truth,** `tat_rate` falls −12.5 / −4.2 / −8.4 /
  −4.2 points below (−16.6 / −6.2 / −8.4 / −4.2 by scenario). With a per-scenario bootstrap, all
  the CIs include 0. The only robust difference is that of the outer race at 1/4.

**The other arms.**
- `uniform_ceiling` is the best arm at 1/4, but it collapses at 1/16 and 1/32 (0 × 15 and 2 × 30
  against `uniform`). Lowering the ceiling cuts off the informative band of the inner race, which
  lies at 16.8–25 kHz.
- `bandpass_lu` stays stable around 79 % and loses to `uniform` at 1/32 (0 × 7, p = 0.016).

**Consequence for E15 (corrected on 2026-11-11).** The first version of this section told E15 to
aim at 96–128×. That was withdrawn. With the nested truth, the best uniform rate chosen in training
gives 75–83 % on the outer race from 1/4 to 1/32, and there is room to gain there. At 96–128×, on
the other hand, the uniform baseline falls because of a wall of the pipeline (bands narrower than
the fault frequency; see E07, extension), and a gain there would be against a straw man.
- **Budgets:** 1/4 to 1/32.
- **Deciding baseline:** the best uniform rate chosen on the training records, with nested truth.

The correction was passed on to E15 on the same day.

## Re-run with the corrected ruler (2026-11-11, afternoon)

Run at 13:57 UTC, with the truth taken at the plateau centre, the local ruler recorded alongside,
and the `despike` arm pre-registered. The analysis (`analyze.py --ruler global|local`) is in
`results/analysis.json` and `results/analysis-local.json`, with its provenance recorded.
- **Global ruler:** leave-one-out truth, with nested cross-validation.
- **Local ruler:** the full-rate line of the segment itself, used only for scoring. The choice
  made outside the test has no leakage by construction.

Accuracy (%) over both races combined:

| arm | global 1/4 · 1/8 · 1/16 · 1/32 | local 1/4 · 1/8 · 1/16 · 1/32 |
|---|---|---|
| `uniform` | 56.2 · 87.5 · 83.3 · 87.5 | 57.4 · 91.5 · 87.2 · 91.5 |
| **`despike`** | **91.7 · 93.8 · 89.6 · 89.6** | **97.9 · 97.9 · 93.6 · 93.6** |
| `uniform_ceiling` | 85.4 · 87.5 · 54.2 · 29.2 | 89.4 · 91.5 · 55.3 · 29.8 |
| `tat_rate` | 77.1 · 87.5 · 83.3 · 87.5 | 78.7 · 89.4 · 87.2 · 91.5 |
| `bandpass_lu` | 79.2 · 79.2 · 79.2 · 77.1 | 83.0 · 83.0 · 83.0 · 80.9 |
| `random_cs` (straw man, see above) | 2.9 · 4.2 · 6.2 · 4.2 | 2.9 · 4.3 · 6.4 · 4.3 |
| best uniform rate chosen on other records | 89.6 · 89.6 · 87.5 · 89.6 | **97.9 · 97.9 · 95.7 · 97.9** |
| oracle (chosen on the test) | 95.8 at all four | 97.9 at all four |

**Reading.**
- **Removing one-sample spikes fixes the 1/4 valley**, as in E20: 56 → 92 % under the global
  ruler, 57 → 98 % under the local one.
- **Under the local ruler, the best uniform rate chosen outside the test lies within 0–2 points
  of the oracle.** On the outer race the chosen factor is the same in the 3 folds: 12 at 1/4 and
  1/8, 46 at 1/32. The earlier withdrawal of "factor 46 generalizes" came from the ruler: the
  nested truth also carries the drift of the line with the rotation speed.
- **H2 remains refuted,** now under both rulers. `tat_rate` falls −19.2 [−29.2; −4.3] points
  below the uniform rate chosen outside the test at 1/4 (local ruler), and the other intervals
  include 0.
- **Consequence for E15.** Under the local ruler, the deciding baseline, the best uniform rate
  chosen in training, is at 96–98 %. That leaves 2 to 4 points for the learned mask on Ottawa. A
  negative result is plausible and will be reported as such.
