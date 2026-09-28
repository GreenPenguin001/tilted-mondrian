# Table 1 Fixed-Model Random-Resplit Robustness

This experiment reallocates fixed-model nontraining examples among calibration,
propensity/repair, and test splits. Base classifiers and exported probabilities
remain fixed, and train rows never enter the resplit pool.

The input root follows the package-wide input layout:

```text
<input-root>/biobias/table1/seed_4/probabilities/
<input-root>/folktables_design_a/<year>/train/
<input-root>/folktables_design_b/<year>/train/
<input-root>/atus/
```

From the package root:

```bash
python -m Per_Split_Robust.code.run_table1_robustness \
  --stage pools --input-root outputs --output-root outputs/table1_robustness
python -m Per_Split_Robust.code.run_table1_robustness \
  --stage run --input-root outputs --output-root outputs/table1_robustness \
  --workers 4 --resume
python -m Per_Split_Robust.code.run_table1_robustness \
  --stage aggregate --output-root outputs/table1_robustness
python -m Per_Split_Robust.code.run_table1_robustness \
  --stage validate --output-root outputs/table1_robustness
```

The configured protocol uses 50 resplits per dataset-score configuration. Dataset-specific input
directory overrides are available when upstream artifacts are stored elsewhere.
Fresh pools, per-seed artifacts, summaries, and figures are written under the
selected output root.
