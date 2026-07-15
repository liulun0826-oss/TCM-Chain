# DiNSR Ablation Variants

This directory implements the eight ablation names reported in the paper. All
variants reuse the main data split, neural architecture, optimization schedule,
checkpoint selection rule, PC/OC inference procedures, and task metrics. Each
variant changes only the mechanism named in the table below.

| Paper name | Controlled intervention |
|---|---|
| `OnlyN` | Bypass symbolic reasoning and use the neural chain alone. |
| `OnlyS` | Exclude neural logits and predict with symbolic evidence alone. Co-occurrence evidence is seeded by a detached symbolic first pass. |
| `w/o DiscrepancyGate` | Replace sample-specific discrepancy gating with ungated task-specific symbolic injection. |
| `w/o FRules` | Remove audited first-order rules only. |
| `w/o HRules` | Remove high-order multi-antecedent rules only. |
| `w/o CRules` | Remove treatment and herb co-occurrence rules only. |
| `w/o RandomR` | Randomize rule conclusions with a task-wise deterministic derangement while preserving antecedents, signs, weights, counts, and source types. |
| `w/o NegRules` | Remove negative conflict rules before rule-source attention. |

The paper label `w/o RandomR` is retained verbatim for result-table alignment.
Its intervention is a randomized-rule semantic control rather than a component
removal. The randomization uses the experiment seed and never maps a conclusion
to a label from another task.

## Run One Variant

Quote paper names containing spaces:

```bash
python ablation_variants/train_ablation.py --variant "w/o DiscrepancyGate" --use_amp --pin_memory
```

```bash
python ablation_variants/train_ablation.py --variant OnlyN --use_amp --pin_memory
```

By default, results are written to
`outputs/ablation/<filesystem-safe-variant-name>/`. Each output directory
contains the same artifacts as the full model, including `config.json`,
`training_log.csv`, `valid_metrics.json`, `test_metrics.json`, and
`gate_statistics.json`.

## Reporting Rules

- Select checkpoints using validation predicted-chain score only.
- Report PC as the primary protocol and OC as a diagnostic protocol.
- Keep the fixed split, threshold, task metrics, and seed identical to the full model.
- Do not regenerate rule statistics from validation or test data.
- Run the full model and all ablations with the same environment and pretrained backbone.
- For stronger uncertainty estimates, repeat all variants with the same set of at least three seeds and report mean and standard deviation.

Machine-readable intervention descriptions are stored under `configs/`. The
exact paper name is recorded in every output `config.json` even where the file
system uses a slash-free directory name.
