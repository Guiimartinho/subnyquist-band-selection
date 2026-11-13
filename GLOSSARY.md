# Glossary of labels

The labels below name the arms, cleanings, rulers and verdicts in the code, the configurations and
the result files. The runs wrote them in Portuguese, and they were renamed to English afterwards.
In the result files the renaming replaced the labels and nothing else, so every number is the one
the run wrote. The code derives its random seeds from the original labels, so a rerun draws the
same random numbers. The result files as the runs wrote them, whose hashes the analysis files
record (`metrics_sha256`), are those of the commits before the renaming. `tools/relabel_map.py`
holds the whole map from the original labels to these, and `REPRODUCE.md` (section 6) shows how
to rebuild the original files from the current ones.

## Common terms

| Label | Meaning |
|---|---|
| `raw` | raw stream, no cleaning |
| `median3` | three-point running median |
| `despike`, `despiked` | isolated single-sample spikes removed (replaced by the mean of their neighbours) |
| `_after_decimation` (suffix) | applied after decimation, to the decimated stream |
| `local`, `nested` | ground-truth rulers: the full-rate line of the same segment / the median calibration of the other records of the same race |
| `OR`, `IR`, `H` | outer race, inner race, healthy |
| `healthy_bearing` | family label of the healthy control (E15) |
| `no_fault` | no fault declared (the segment is declared healthy) |
| `budget` | sample budget (average fraction of the full rate) |
| `ceiling` | upper limit of the band search (`kurtogram_ceiling8k`: 8 kHz) |
| `zone1` to `zone7` | Nyquist zone *k*: ideal band-pass filter on the zone before decimation |
| `guard` | guard that discards a run and counts it as a miss |
| `supported`, `refuted`, `inconclusive`, `not evaluable` | verdicts of the pre-registered hypotheses |

## E07, E20, E21, E22, E23 (Ottawa, MAFAULDA and CWRU)

| Label | Meaning |
|---|---|
| `no_prior` | blind order estimation, no prior on the fault order |
| `calibrated_prior`, `nominal_prior` | estimation restricted around the calibrated / nominal (catalogue) fault order |
| `spr` | samples per revolution of the angular resampling |
| `none` | no intervention |
| `stop_z1`, `stop_z2`, `stop_z3`, `stop_z123` | ideal rejection, at full rate, of folding zone Z1, Z2, Z3 or their union |
| `aaf_ideal` | ideal anti-alias filter at the reduced rate |
| `median3`, `despike` | three-point median, spike removal, at full rate before decimation |
| `despike_after_decimation`, `median3_after_decimation` | spike removal, three-point median, on the decimated stream |
| `ses_infogram`, `ses_infogram_despike` | SES infogram selector, raw / after spike removal |
| `gini` | spectral Gini index selector |
| `uniform` | uniform decimation |
| `uniform_ceiling` | uniform decimation with the search ceiling set by the TAT |
| `tat_rate` | the rate allowed by the rate-admissibility test (TAT) |
| `random_cs` | random sampling with compressive-sensing recovery |
| `bandpass_lu` | band-pass sampling as in Lu et al. (2018) |
| `impulsive`, `surrogate` | injected impulsive bursts / their phase-randomized surrogate |
| `centred`, `opposite`, `orthogonal` | fault position relative to the load zone |

## E24 (Paderborn replication)

| Label | Meaning |
|---|---|
| `kurtogram` | fast kurtogram, raw stream |
| `kurtogram_despiked` | kurtogram after spike removal before decimation |
| `kurtogram_despiked_after_decimation` | kurtogram after spike removal on the decimated stream |
| `ses_infogram_despiked` | SES infogram after spike removal |
| `kurtogram_ceiling8k` to `kurtogram_ceiling1k` | kurtogram with a lower search ceiling at full rate |
| `kurtogram_notch`, `kurtogram_notch_control` | kurtogram after zeroing 15–17 and 30.5–32 kHz / the control bands 12–14 and 26.5–28 kHz |

## E26 (fault-order rule on Paderborn)

| Label | Meaning |
|---|---|
| `kurtogram`, `median_kurtogram` | blind pipeline, raw / after the three-point median |
| `guided`, `median_guided` | guided band with the fault-order rule, raw / after the median |
| `blind` | blind order estimate |
| `kurtogram_rule` | kurtogram band with the fault-order rule |
| `wideband_rule` | whole band (500 Hz to the effective Nyquist frequency) with the rule |
| `ses_infogram_rule` | SES infogram band with the rule |
| `random_band_rule` | random band of the kurtogram partition with the rule |
| `pure` | pure decimation |
| `antialias` | ideal anti-alias filter before decimation |
| `fleet_zone` | Nyquist zone chosen on the other bearings (leave one bearing out) |
| `kurtogram_zone` | Nyquist zone chosen by the full-rate kurtogram of a commissioning record |
| `s_*`, `s03_*`, `peak_*` | rule score with an integer-order exclusion of 0.015 / of 0.03; order of the first-harmonic peak |
| `D_BPFO_09`, `D_BPFO_11`, `D_BPFI_09`, `D_BPFI_11` | decoy windows at 0.9 and 1.1 times each fault order |

## E15 (learned sampling masks)

| Label | Meaning |
|---|---|
| `uniform_budget` | classical pipeline at the uniform rate of the budget |
| `uniform_trained` | classical pipeline at the best uniform rate at or below the budget, chosen on the training records |
| `uniform_trained_per_race` | the same, chosen per race |
| `uniform_oracle`, `uniform_oracle_per_race` | best uniform rate chosen on the test records (oracle, reference only), overall / per race |
| `dec_uniform` | learned decoder with uniform decimation at the budget |
| `dec_uniform_trained` | learned decoder with the uniform rate chosen on the training records |
| `dec_random` | learned decoder with a fixed random feasible mask |
| `dec_learned` | learned mask with the decoder |
| `dec_learned_sel` | learned mask used in each split, which may come from a sparser trained budget |
| `dec_free`, `dec_free_projected` | mask learned without the constraint / projected onto the feasible set |
| `*_norefit` | the stage-A decoder with the projected mask, without refitting it |
| `dec_multicoset`, `dec_multicoset_vb` | multicoset designs; the second is that of Venkataramani and Bresler (2001) |
| `dec_adaptive`, `dec_adaptive_blocks` | block-adaptive mask inspired by ADPS (van Gorp et al., 2021) |
| `family_true`, `family_pred` | true / predicted race |
| `hit_local`, `hit_nested` | hit under the local / nested ruler |
| `perfect`, `optimistic` | perfect planted arm / optimistic decision rule of the power gate |
| `testable`, `not_testable` | power gate verdict |
| `with_line`, `without_line` | bearings with / without a valid full-rate line (exploratory split) |
