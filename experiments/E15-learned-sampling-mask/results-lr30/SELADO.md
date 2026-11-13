# Run with `mask.lr_multiplier = 30`: SEALED, not analysed

- **Period:** from 2026-11-11, 21:07 UTC, until it was interrupted on 2026-11-12, 01:23 UTC.
- **Rule B5 on the raw variant:** the `mask_converged_frac` of `dec_aprendida` was 1.0 / 1.0 /
  **0.4** / 0.6 from 1/4 to 1/32. 1/16 stayed below 0.5.
- **The author's decision, before any test number was opened:** the corrected run
  (`stop_after_anneal`) decides H1 and H2. This one was interrupted in the `sem_picos` half, which
  decides nothing.
- **Files:** since it did not finish, the run produced no `metrics.csv`, `summary.csv`,
  `hypotheses.csv` or `comparisons.csv`. The per-run cache (225 files in `runs/`) stayed on the
  group's server, in the E15 tree, not analysed.
- **Only `convergencia.txt` is here:** training diagnostics of the learned mask (convergence,
  margin, best epoch, epochs, lr and positions). No test number.
- **Another training diagnostic, which motivated the evaluability rule of H2:** the free mask
  (`dec_livre`, `dec_livre_projetada`) left the noise in 1/5, 1/5, 1/5 and 0/5 of the splits.
- **Test numbers seen by accident while monitoring.** The progress lines of the log carry test hits
  per split. At 22:33 UTC, two of them were displayed: at 1/32, in the split of index 1,
  `dec_multicoset_vb` with 8/12 and `dec_uniforme_treino` with 9/12 under the local ruler.
  - They were baselines, and none was the learned mask.
  - All the rules of the corrected run were fixed in code and committed before any of its numbers.
  - Since then, monitoring extracts only the position from the log.
