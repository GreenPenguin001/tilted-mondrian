# Set-Size Robustness

This component evaluates the raw set-size repair method over 20
fixed-model downstream random resplits for each of four datasets and four
scores, for 320 repair problems. All four scores
within a dataset-resplit share the same row partition. Base predictors are not
retrained, and train rows never enter the resplit pool.

The protocol preserves the original calibration, repair, and test sizes. It uses
joint `G x Y` stratification for Bias in Bios, the configured `Y/G/Z` fallback
hierarchy for Folktables Design A, joint `G x Z` stratification for Folktables
Design B, and a deterministic respondent shuffle for ATUS. Optimizer seed `2026`
and the reported set-size repair search configuration are fixed.

## Run

Run from the package root. The preparation stage constructs nontraining
pools and deterministic split manifests from the upstream exports documented in
the top-level README.

```bash
python mitigate_set_size/resplit_robustness/run_set_size_20_resplits.py \
  --stage prepare --input-root outputs \
  --protocol-root outputs/set_size_repair_robustness/protocol \
  --output-root outputs/set_size_repair_robustness/results

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python mitigate_set_size/resplit_robustness/run_set_size_20_resplits.py \
  --stage run --jobs 4 --resume \
  --protocol-root outputs/set_size_repair_robustness/protocol \
  --output-root outputs/set_size_repair_robustness/results

python mitigate_set_size/resplit_robustness/validate_set_size_20_resplits.py \
  --stage aggregate \
  --protocol-root outputs/set_size_repair_robustness/protocol \
  --output-root outputs/set_size_repair_robustness/results
```

`--dataset`, `--score`, and `--seed` filters support deterministic diagnosis and
resume without changing the complete cohort. The aggregate stage writes the
per-problem ledger, 16-row summary, distribution summaries, validation records,
and the Markdown report under the selected output root.

An optional main-experiment summary can be included for descriptive distribution
context:

```bash
python mitigate_set_size/resplit_robustness/validate_set_size_20_resplits.py \
  --stage aggregate \
  --protocol-root outputs/set_size_repair_robustness/protocol \
  --output-root outputs/set_size_repair_robustness/results \
  --canonical-summary outputs/set_size_repair/canonical_4x4_summary.csv
```
