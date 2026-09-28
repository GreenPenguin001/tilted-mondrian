# Repair Fixed-Model Random-Resplit Robustness

This experiment holds the trained predictor, probabilities, labels, groups, and
local-cell values fixed while reallocating nontraining rows. SIMPLE, APS, RAPS,
and SAPS configurations with a common dataset and seed share one row partition,
but independently construct scores, define repair support and roles, fit symmetric
and Tilted Mondrian repair, calibrate group-Mondrian thresholds, and evaluate held-out test data.
The runner imports the shared score-intervention core. Each resplit recomputes
its interventions from the saved probability outputs.

Inputs follow the package-wide input layout:

```text
<input-root>/biobias/table1/seed_4/probabilities/
<input-root>/folktables_design_a/2018/train/
<input-root>/folktables_design_b/2018/train/
<input-root>/atus/
```

From the package root:

```bash
python -m mitigate.Per_Split_Robust_Mitigate.code.run_table2_all_scores \
  --stage pools --input-root outputs --output-root outputs/repair_robustness
python -m mitigate.Per_Split_Robust_Mitigate.code.run_table2_all_scores \
  --stage run --input-root outputs --output-root outputs/repair_robustness \
  --jobs 4 --resume
python -m mitigate.Per_Split_Robust_Mitigate.code.run_table2_all_scores \
  --stage aggregate --output-root outputs/repair_robustness
python -m mitigate.Per_Split_Robust_Mitigate.code.run_table2_all_scores \
  --stage validate --output-root outputs/repair_robustness
```

The configured protocol uses 20 resplits per dataset-score configuration and
optimizer seed 2026.
Repair data alone determine candidates, constraints, and selection. Calibration
thresholds are reselected after the intervention is frozen; test outcomes are
diagnostic only. Fresh runtime artifacts are written under the selected output
root.
