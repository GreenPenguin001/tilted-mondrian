# Local Coverage Repair

The repair code consumes frozen repair, calibration, and test probabilities. It
does not retrain a classifier or recompute base probabilities.

Local cells use the frozen `G x Z` encodings from the curve analysis:

- BioBias: gender by occupation, with label-defined `Z=Y`.
- Folktables occupation: gender by education bucket.
- Folktables income: race by education bucket.
- ATUS: gender by employment status.

For every repair candidate, group-Mondrian thresholds preserve baseline repair
group covered counts exactly. Each initially advantaged cell may lose at most
`epsilon=0.01` coverage, disadvantaged cells may not lose coverage, and no
repairable local gap may increase. The intervention is frozen before calibration
thresholds are reselected; test is held-out reporting only. Average set-size
cost is the final tie-breaker.

## Inputs

BioBias and Folktables consume `probs_{repair,cal,test}.npy` with matching
`y_*`, `g_*`, and `z_*` arrays plus `label_maps.json`. ATUS consumes
`class_probabilities_{repair,calibration,test}.csv.gz` containing `true_label`,
`g`, `z`, and `prob__*` columns. The optional ATUS label map is
`closed_world_topk_label_map.csv`.

The documented upstream commands produce these directories:

```text
outputs/biobias/table1/seed_4/probabilities/
outputs/folktables_design_a/2018/train/
outputs/folktables_design_b/2018/train/
outputs/atus/
```

## Reproduction Workflow

Tilted Mondrian is the intervention reported in Table 2. The executable workflow
first constructs the symmetric intervention. Its selected solution is explicitly
re-evaluated by the Tilted Mondrian search, retained during beam pruning, and
included in the final feasible candidate pool.

Prepare the saved probability outputs at the paths listed above, then
run the following steps in order.

### 1. Symmetric score shifting

```bash
python mitigate/score_fix/biobias/score_fix_biobias.py \
  --input-dir outputs/biobias/table1/seed_4/probabilities \
  --output-dir outputs/score_fix/biobias/symmetric
python mitigate/score_fix/folktables_design_a/score_fix_folktables_design_a.py \
  --input-dir outputs/folktables_design_a/2018/train \
  --output-dir outputs/score_fix/folktables_design_a/symmetric
python mitigate/score_fix/folktables_design_b/score_fix_folktables_design_b.py \
  --input-dir outputs/folktables_design_b/2018/train \
  --output-dir outputs/score_fix/folktables_design_b/symmetric
python mitigate/score_fix/atus/score_fix_atus.py \
  --input-dir outputs/atus \
  --output-dir outputs/score_fix/atus/symmetric
```

### 2. Tilted Mondrian

```bash
python mitigate/score_fix/run_directional_asymmetric.py \
  --input-root outputs --result-root outputs/score_fix
```

### 3. Validation

```bash
python mitigate/score_fix/validate_directional_asymmetric.py \
  --result-root outputs/score_fix
```

All runners accept dataset-specific input-directory overrides. Results are
created under the selected result root; no input copying is required.
