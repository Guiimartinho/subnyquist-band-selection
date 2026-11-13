# E25 - Co-prime co-array: cyclic statistics of the fault without folding on the carrier axis

Opened on 2026-11-12, after E20, E23 and E24.

## Why this experiment

Part (a) of the paper shows the mechanism: under pure uniform decimation, out-of-band content
folds onto the search range and captures the band selector. The proposal here is to sample in such
a way that the fold does not happen in the domain that the diagnosis uses.

- **The diagnosis does not need the waveform.** The cyclic correlation R(τ, α) at the cyclic
  frequencies of the fault is enough, and the bearing geometry gives which ones they are.
- **Two co-prime samplers.**
  - With steps M and N (gcd = 1) running continuously, every integer lag appears (Bézout), with
    about T/(MN) pairs per lag.
  - That is why the carrier axis has no folding up to f_s/2, with a mean rate of
    1/M + 1/N − 1/(MN) of the full rate.
  - With M = 8 and N = 9, the mean rate is 0.222, below 1/4.
- **The squared envelope spectrum of any band** comes out of R(τ, α)
  (`snq_recon.coarray.band_ses`). The pipeline remains the classical one; only the second-order
  estimation changes.
- **Hardware:** on the nRF54L15 these are two timers triggering the SAADC through the DPPI, without
  the CPU.

## Novelty (check of 2026-11-12, `lit-scout`)

**Terms searched** (forward citations on Semantic Scholar and keywords on Crossref and
OpenAlex): co-prime sampling, sparse ruler or multicoset, combined with cyclic spectrum, spectral
coherence or cyclostationarity, and with bearing, vibration or diagnosis.

No work was found that brings together the four pieces:
- co-prime sampling or a sparse ruler;
- estimation of the cyclic correlation through the co-array;
- cyclic frequencies of the fault given by the geometry;
- bearing diagnosis.

**The closest ones are partial:**
- `xu2026covpreserving`: sparse ruler on machinery, without cyclic statistics; preprint, not peer
  reviewed.
- `cao2024dcs`: co-prime co-array for stationary sinusoids.
- `tian2012cyclic`, `cohen2017subnyquistcyc` and `zhang2026fccse`: sub-Nyquist cyclic spectrum in
  communications.
- `gkoktsi2019psbs`: low-power multicoset for structural power spectra.

**Forward citations from 2022 onwards**, filtered by title, with no bearing title among them:
- 661 of `vaidyanathan2011coprime`;
- 26 of `tian2012cyclic`;
- 50 of `romero2016ccs`.

**Theory that does not apply directly.** The bound of 4/5 of the Nyquist rate of
`cohen2017subnyquistcyc` (Section IV-B, Theorem 1, Eq. 52) is a sufficient condition to recover
the **whole** cyclic spectrum, all cyclic frequencies, without prior knowledge. Here only a few
low, known cyclic frequencies matter.

The condition for our case has to be derived. It has two parts:
- **Lag coverage:** all integers, by Bézout's theorem.
- **Cyclic rate:** α_max < f_s/(2MN), because the product for each lag exists only once every MN
  samples.

**Risks pointed out by the literature:**
- the variance per lag grows with MN/T;
- the required record length;
- variable rotation speed (`abboud2017orderfreq`);
- the wide-sense stationarity assumption.

## Exploratory pilot (2026-11-12, NOT pre-registered)

The pilot decides whether a pre-registration is worthwhile. Its numbers do not go into the paper
as confirmatory.

**Question.** At the same mean rate, or below it, does the co-prime co-array preserve the cyclic
signature of the fault better than uniform decimation when an impulsive interferer above the
effective Nyquist folds onto the fault band?

**Protocol** (`pilot.py`), that of a node with commissioning:
- **Data:** the 13 fault records of CWRU 48k from E23, with constant rotation speed.
- **Calibration:** the fault band is calibrated once at full rate, on the first half of the
  record and without injection, by the E23 kurtogram with a 10 kHz ceiling.
- **Test:** on the second half.
  - Without injection, or with the impulsive bursts of E23 (97.7 Hz, carriers at 13, 16, 19 and
    22 kHz) at the maximum dose. It is the same seed as E23, labelled +20 dB and actually +17 dB.
- **Patterns, all with the same estimator:** squared envelope spectrum of the calibrated band,
  through the cyclic correlation, up to lag 192.
  - full rate;
  - uniform by 4 (0.25);
  - co-prime (8, 9) (0.222).
- **Estimate:** the argmax between 20 and 330 Hz, without prior. The 330 Hz ceiling lies below
  f_s/(2·72) = 333 Hz.
- **Hit:** |estimate − geometry frequency| ≤ 2 Hz, as in E02 and E23.
- **Also recorded:** the prominence of the fault line, that is, the peak within ±2 Hz of the truth
  divided by the median within ±20 %.
- **Reference:** the result of the E23 blind kurtogram at 4×, under the same conditions.

## Pilot result (2026-11-12): negative for the co-prime co-array

The data are in `results-pilot/metrics.csv` and `summary.json`: 13 records, 5 conditions and 3
patterns.

| condition | full rate | uniform by 4 (0.25) | co-prime (8, 9) (0.222) | E23 blind kurtogram at 4× |
|---|---|---|---|---|
| no injection | 23 % (prominence 47) | **92 %** (64) | 0 % (2.9) | 92 % |
| bursts at 13, 16, 19 and 22 kHz | 0–23 % (28–43) | **92 %** (59–63) | 0 % (2.8–3.0) | 23–38 % |

**Reading:**
- **The co-prime pattern is not viable with these records** (5 s at 48 kS/s).
  - The fault line has a prominence of about 2.9 times the local median, against about 60 for the
    uniform pattern. With about 1550 bins in the search, the noise maximum exceeds 10 times the
    median, so the line disappears into the noise.
  - **The cause is structural.** Each lag receives only about T/72 pairs, and each product
    x[n]x[n−τ] carries the power of the whole signal, and not only that of the band.
  - The calibrated band (from 5.3–10 kHz to 8.4–10 kHz) holds a small fraction of the energy, and
    the variance of the co-array swallows the signature. This is the variance risk that the
    literature points out.
  - In the synthetic test (`test_coarray.py`), the modulated band held almost all the power, which
    is why it worked.
  - At the same count of samples per diagnosis, the co-prime pattern loses: the fold-free carrier
    axis does not make up for the variance.
- **The full rate "misses" by choosing the second harmonic.** The wrong estimates lie at
  207–216 Hz, which is 2 × BPFO. The pilot's estimator is a simple argmax, without a harmonic check,
  while that of the classical pipeline (`estimate_without_prior`) handles this. The prominence at
  the right frequency is high.
- **Uniform by 4 with the band fixed at commissioning hits 92 %, with or without the
  interferer**, against 23–38 % for the E23 blind kurtogram on the same signal.
  - This indicates that capture is a band-choice problem, and that a fixed band avoids it.
  - **Serious caveat:** here the band was calibrated with the fault already present (first half of
    the same fault record). In a real commissioning the machine is healthy, and the kurtogram does
    not find the fault band without impacts.

**Decision:** the co-prime co-array, as designed, does not go to pre-registration. Left open:
- lower-variance estimators, such as weighting the lags by the number of pairs or using only the
  well-estimated lags;
- much longer records;
- an analog band-pass pre-filter, which would reduce the variance. That presupposes knowing the
  band, and then classical band-pass sampling (`vaughan1991`) already solves the problem.
