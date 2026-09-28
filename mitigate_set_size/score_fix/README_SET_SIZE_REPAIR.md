# Tilted Mondrian Set-Size Repair

This component applies Tilted Mondrian score shifting to saved probability outputs
to reduce raw between-group mean prediction-set-size gaps within supported
`G x Z` cells. The repair split alone determines cell roles, candidate
feasibility, and the selected intervention.

For each non-neutral supported cell, positive score shifts are applied to the
baseline larger-set group and negative shifts to the baseline smaller-set group.
Neutral cells receive no direct search pressure. Every candidate reconstructs
complete prediction sets and preserves each repair group's baseline covered
count exactly.

The repair constraints are:

1. no supported raw set-size gap increases by more than `1e-9`;
2. a baseline larger-set side does not increase by more than `1e-9`;
3. a baseline smaller-set side increases by at most `0.02` mean-set-size units;
4. a neutral cell's gap does not increase, and neither group increases by more
   than `0.02` mean-set-size units.

Feasible candidates are ranked by worst gap, mean gap, total gap reduction, and
intervention norm. A rounded candidate key resolves complete ties. Cellwise
coverage is not an objective or feasibility constraint. After theta is frozen,
calibration thresholds are reselected independently to match baseline
calibration group covered counts, followed by one held-out test evaluation.

## Inputs

The default `--input-root outputs` resolves to:

```text
outputs/biobias/table1/seed_4/probabilities
outputs/folktables_design_a/2018/train
outputs/folktables_design_b/2018/train
outputs/atus
```

Use `--input-dir` to override one dataset's probability-export directory.

## Running the Experiments

Run from the package root. Each command writes to
`outputs/set_size_repair/<dataset>/<score>` unless `--output-dir` is supplied.

```bash
for dataset in biobias folktables_design_a folktables_design_b atus; do
  for score in simple aps raps saps; do
    python mitigate_set_size/score_fix/run_directional_set_size.py \
      --dataset "$dataset" --score "$score" --input-root outputs
  done
done
```

The optional validator exercises the implementation invariants for the two
reported configurations:

```bash
python mitigate_set_size/score_fix/validate_directional_set_size.py \
  --dataset folktables_design_b --score simple --input-root outputs \
  --output outputs/set_size_repair/validation/folktables_design_b_simple.csv
python mitigate_set_size/score_fix/validate_directional_set_size.py \
  --dataset biobias --score simple --input-root outputs \
  --output outputs/set_size_repair/validation/biobias_simple.csv
```
