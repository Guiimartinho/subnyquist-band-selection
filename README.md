# Blind band selection under pure decimation of bearing vibration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24206559.svg)](https://doi.org/10.5281/zenodo.24206559)

Code, configurations and results for the paper

> L. G. M. S. Ito, L. S. de Carvalho, J. R. F. Azambuja, *Blind band selection under pure
> decimation of bearing vibration: aliased impulses, fault-order decisions and a power check for
> learned sampling*, manuscript submitted to Mechanical Systems and Signal Processing.

A condition-monitoring node that lowers its sampling rate behind a front end fixed for the highest
rate cannot band-pass filter an unknown fault resonance. Pure decimation is the limiting case:
everything outside the band folds with it. This repository studies what the folded content does to
blind band selection on three public bearing benchmarks. The acquisition operates below the Nyquist
rate of the record by exploiting the structure of the signal, as Landau's density condition allows;
it does not claim anything beyond sampling theory.

## Main results

- **Ottawa:** a quasi-periodic train of single-sample spikes, present in every record including the
  healthy ones, captures the fast kurtogram on the outer-race records from 3x to 6x decimation, and
  order-estimation accuracy falls non-monotonically with the factor. An anti-alias filter at the
  reduced rate avoids the capture at no cost.
- **CWRU:** injected bursts reproduce the capture; a phase-randomized surrogate with the same folded
  spectrum does not.
- **Paderborn:** the kurtogram is captured at full rate. A decision restricted to the fault orders
  of the bearing geometry, with a threshold from six healthy bearings, identifies the faulty race
  in 49-51% of the segments at 4x and 8x, against 16-19% for blind order estimation, and the
  advantage is the same at full rate. An anti-alias filter beats pure decimation by 7 to 30 points.
- **Learned sampling masks:** a power gate shows that the pre-registered test could not detect even
  a perfect learned method on Ottawa, unlike on Paderborn. Learned masks did not outperform trained
  uniform decimation or random masks; the intervals do not exclude a gain of 5 points.

Appendix A of the paper lists every pre-registered hypothesis with its verdict, failures included.

## Repository map

| Path | Content |
|---|---|
| `packages/` | Python workspace (`snq-core`, `snq-sim`, `snq-data`, `snq-recon`, `snq-learn`, `snq-eval`, `snq-embed`): sampling maths, dataset loaders, classical pipeline, learned decoders |
| `experiments/` | one directory per experiment of the paper (E07, E15, E20 to E26): README, configurations, code and versioned results |
| `papers/mssp-2026/` | LaTeX sources of the paper, generated tables, figures and numbers |
| `papers/shared/` | scripts that regenerate every number, table and figure, and the local PDF build |
| `datasets/` | manifests (source, licence, geometry, checksums) and download scripts; raw data is not included |
| `literature/refs.bib` | bibliography of the paper |

## Quick start

```bash
uv sync --all-packages
uv run pytest
uv run python papers/shared/build_pdf.py install
uv run python papers/shared/build_pdf.py build papers/mssp-2026 --submission
```

`REPRODUCE.md` explains how to download the datasets, rerun each experiment, regenerate the
figures and verify the provenance recorded in each `manifest.json`. `GLOSSARY.md` explains the
labels of the arms, cleanings, rulers and verdicts used in the code and the result files.

## Data

The raw datasets are not redistributed. Their licences are those of their sources: the University
of Ottawa bearing data under CC BY 4.0, the Paderborn bearing data under CC BY-NC 4.0, and the Case
Western Reserve University bearing data as published by its Bearing Data Center. The result files
in `experiments/*/results*/` are derived from them.

## Licence

The code is released under the MIT licence (`LICENSE`). The datasets are not covered by it and
keep the licences of their sources (see Data above and `datasets/manifests/`).

## Citation

See `CITATION.cff`. The releases are archived at Zenodo:
[10.5281/zenodo.24206559](https://doi.org/10.5281/zenodo.24206559) resolves to the latest one, and
the version used in the paper, v1.0.0, is
[10.5281/zenodo.24206560](https://doi.org/10.5281/zenodo.24206560).

## Contact

Luiz Guilherme Martinho Sampaio Ito, CEDRA Research Group, Graduate Program in Microelectronics
(PGMICRO), Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS), Porto Alegre,
Brazil — luiz.ito@inf.ufrgs.br
