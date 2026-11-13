# Run with mask.lr_multiplier = 10 (2026-11-11, 14:35–21:05 UTC): SEALED by rule B5

Rule B5, pre-registered in the README, was applied to `summary.csv`. The fraction of splits in which
the mask of `dec_aprendida` left the Gumbel noise (`mask_converged_frac`), on the raw variant, was:

| budget | 1/4 | 1/8 | 1/16 | 1/32 |
|---|---|---|---|---|
| `mask_converged_frac` | 0.6 | **0.4** | **0.4** | **0.2** |

The fraction stayed below 0.5 at three budgets, and the rule requires the run to be redone with
`mask.lr_multiplier = 30` before the hypotheses are opened. Therefore **`hypotheses.csv` and
`comparisons.csv` of this run were not opened**. The files stay here as a record of what ran, and
they enter no analysis.

Read to apply the rule: only `summary.csv` (column `mask_converged_frac`) and the `converged`
diagnostics of `masks.json`.
