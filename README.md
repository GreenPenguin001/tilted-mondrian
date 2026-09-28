# Reproduction Code

This repository contains classifier export, conformal set-size curve,
overlap-weighted diagnostic, local coverage repair, raw set-size repair, and
fixed-model random-resplit robustness code. Commands below are run from the
repository root.
Runtime artifacts are written under `outputs/` by default and are not included.

## Environment

Python 3.10.15 and the dependency versions in `requirements.txt` record the
environment used for package validation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Inputs and Main Experiments

### Bias in Bios

The exporter loads the configured Hugging Face dataset and model, trains on
`D_train`, and writes probabilities and row metadata for all four splits.
The publication configuration is:

```bash
python biobias/biobias_train_export.py \
  --dataset-name LabHC/bias_in_bios \
  --model-name distilbert-base-uncased \
  --text-field hard_text \
  --output-root outputs/biobias \
  --hf-cache-dir cache/huggingface \
  --run-name table1 \
  --seed 4 \
  --top-n-occupations 10 \
  --max-length 160 \
  --num-epochs 2 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --train-batch-size 64 \
  --eval-batch-size 32 \
  --logging-steps 100 \
  --sdpa-backend math \
  --attn-implementation eager \
  --save-model
```

The saved probability outputs are written to:

```text
outputs/biobias/table1/seed_4/probabilities/
```

The reported raw and overlap-standardized curves are generated together with
the following explicit configuration. The propensity model uses the saved
top-three probability profile, is fitted on the repair split with model seed
0, estimates the 0.95 trimming cap on that split, and applies the absolute
weight cap of 3.0.

```bash
python biobias/biobias_adjusted_curve_double_robust.py \
  --seed-dir outputs/biobias/table1/seed_4 \
  --output-root outputs/biobias/table1/seed_4/curve_adj_ipw \
  --clean-output \
  --score all \
  --diagnostic-policy groupwise \
  --feature-mode safe_topk \
  --top-k-prob-features 3 \
  --model-type torch_mlp \
  --weight-scheme overlap \
  --propensity-clip 0.03 \
  --trim-quantile 0.95 \
  --max-weight 3.0 \
  --estimators raw_curve,ipw_overlap_curve \
  --mlp-hidden-layers 128,64 \
  --mlp-lr 1e-3 \
  --mlp-alpha 3e-3 \
  --torch-mlp-epochs 60 \
  --torch-mlp-batch-size 4096 \
  --torch-device auto \
  --torch-num-threads 1 \
  --model-random-seed 0 \
  --saps-lambda 0.2 \
  --raps-lambda 0.2 \
  --raps-k-reg 5 \
  --score-random-seed 0 \
  --alpha0 0.10 \
  --alpha-grid dense_safe \
  --n-min 202 \
  --tau-agg 0.01 \
  --tau-loc 0.02 \
  --occupation-order evidence \
  --figure-formats pdf,png,svg
```

### Folktables

The design configurations specify ACS features, splits, conformal settings,
and propensity settings. ACS files may be downloaded by `folktables` when
`--download` is enabled, or read from `--data-root`.

```bash
ROOT="$PWD"

for year in 2014 2015 2016 2017 2018; do
  python folktables/design_A_occupation_education/run_year.py \
    --year "$year" --data-root "$ROOT/data/folktables_acs" --download \
    --output-root "$ROOT/outputs/folktables_design_a"
  python folktables/design_B_income_education/run_year.py \
    --year "$year" --data-root "$ROOT/data/folktables_acs" --download \
    --output-root "$ROOT/outputs/folktables_design_b"
done
```

After all five years finish, the optional cross-year summaries can be generated
from the year-level outputs:

```bash
python folktables/design_A_occupation_education/summarize_years.py \
  --output-root "$ROOT/outputs/folktables_design_a" --split test \
  --summary-dir "$ROOT/outputs/folktables_design_a/cross_year_summary"
python folktables/design_B_income_education/summarize_years.py \
  --output-root "$ROOT/outputs/folktables_design_b" --split test \
  --summary-dir "$ROOT/outputs/folktables_design_b/cross_year_summary"
```

These summaries are not prerequisites for the downstream robustness pipeline.

For each year, probabilities and metadata are written under
`outputs/folktables_design_{a,b}/<year>/train/`. Local coverage repair and its
robustness experiment use the 2018 artifacts.

### ATUS

The inspection utility can download the official BLS multi-year ATUS files.
Existing files can instead be supplied with `--skip-download`.

```bash
python ATUS/atus_activity_curve/inspect_atus_design/inspect_atus_design.py \
  --download --data-dir data/atus/raw_0324 \
  --extract-dir data/atus/extracted_0324 \
  --output-dir outputs/atus_inspection
```

Run the selected closed-world HGB design separately for each reported score.
All four configurations use deterministic label-wise scores and pooled
thresholds computed from the calibration split only. APS uses cumulative
probability through the candidate label's rank; RAPS uses lambda 0.01 and
`k_reg=5`; SAPS uses lambda 0.2 and the deterministic `u=0.5` convention.

SIMPLE:

```bash
python ATUS/atus_activity_curve/run_atus_curve.py \
  --config ATUS/atus_activity_curve/atus_config_hgb.json \
  --design primary_sex_employment_top20 \
  --data-dir data/atus/raw_0324 --extract-dir data/atus/extracted_0324 \
  --model hist_gradient_boosting --feature-mode safe \
  --y-mode closed_world_topk --y-top-k 10 \
  --hgb-max-iter 700 --hgb-learning-rate 0.03 \
  --hgb-max-leaf-nodes 31 --hgb-l2-regularization 0.1 \
  --hgb-min-samples-leaf 50 --no-train-cap \
  --alpha-grid atus_dense --score simple --seed 4 \
  --output-dir outputs/atus
```

APS:

```bash
python ATUS/atus_activity_curve/run_atus_curve.py \
  --config ATUS/atus_activity_curve/atus_config_hgb.json \
  --design primary_sex_employment_top20 \
  --data-dir data/atus/raw_0324 --extract-dir data/atus/extracted_0324 \
  --model hist_gradient_boosting --feature-mode safe \
  --y-mode closed_world_topk --y-top-k 10 \
  --hgb-max-iter 700 --hgb-learning-rate 0.03 \
  --hgb-max-leaf-nodes 31 --hgb-l2-regularization 0.1 \
  --hgb-min-samples-leaf 50 --no-train-cap \
  --alpha-grid atus_dense --score aps --seed 4 \
  --output-dir outputs/atus_aps
```

RAPS:

```bash
python ATUS/atus_activity_curve/run_atus_curve.py \
  --config ATUS/atus_activity_curve/atus_config_hgb.json \
  --design primary_sex_employment_top20 \
  --data-dir data/atus/raw_0324 --extract-dir data/atus/extracted_0324 \
  --model hist_gradient_boosting --feature-mode safe \
  --y-mode closed_world_topk --y-top-k 10 \
  --hgb-max-iter 700 --hgb-learning-rate 0.03 \
  --hgb-max-leaf-nodes 31 --hgb-l2-regularization 0.1 \
  --hgb-min-samples-leaf 50 --no-train-cap \
  --alpha-grid atus_dense --score raps \
  --raps-lambda 0.01 --raps-k-reg 5 --seed 4 \
  --output-dir outputs/atus_raps
```

SAPS:

```bash
python ATUS/atus_activity_curve/run_atus_curve.py \
  --config ATUS/atus_activity_curve/atus_config_hgb.json \
  --design primary_sex_employment_top20 \
  --data-dir data/atus/raw_0324 --extract-dir data/atus/extracted_0324 \
  --model hist_gradient_boosting --feature-mode safe \
  --y-mode closed_world_topk --y-top-k 10 \
  --hgb-max-iter 700 --hgb-learning-rate 0.03 \
  --hgb-max-leaf-nodes 31 --hgb-l2-regularization 0.1 \
  --hgb-min-samples-leaf 50 --no-train-cap \
  --alpha-grid atus_dense --score saps --saps-lambda 0.2 --seed 4 \
  --output-dir outputs/atus_saps
```

The required downstream files are
`outputs/atus/class_probabilities_{calibration,repair,test}.csv.gz`. The same
saved probabilities support all deterministic score constructions downstream.

## Propensity-Standardized Random-Resplit Robustness

The Table 1 protocol uses 50 fixed-model reporting resplits. It reads the four
upstream locations documented above. Calibration thresholds and propensity
models are refit within each resplit; train rows never enter the pool.

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

Use `--biobias-input-dir`, `--folktables-a-input-dir`,
`--folktables-b-input-dir`, or `--atus-input-dir` to override any individual
artifact directory without relocating files.

## Tilted Mondrian Intervention

Tilted Mondrian is the intervention reported in Table 2. It uses saved
probability outputs generated by the upstream experiments. Run the symmetric
intervention first; its selected solution is explicitly re-evaluated by the
Tilted Mondrian search, retained during beam pruning, and included in the final
feasible candidate pool.

### 1. Symmetric score shifting

Each dataset runner consumes its upstream probability directory directly and
writes fresh outputs outside the source tree:

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

See `mitigate/score_fix/README_SCORE_FIX.md` for the input schema and
dataset-specific path overrides.

## Tilted Mondrian Robustness

The repair robustness protocol uses 20 intervention resplits per dataset-score
configuration and optimizer seed 2026. Configurations with a common dataset and
split seed share one row partition, while score construction, support, roles,
repair, and calibration remain score-specific. It imports the score-intervention
core implementation and consumes the same saved probability outputs directly. Each
resplit recomputes its symmetric and Tilted Mondrian interventions from those exports.

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

Fresh runtime artifacts are written under the selected output root.

## Set-Size Intervention

Tilted Mondrian set-size repair uses the same saved probability outputs but
targets between-group mean prediction-set-size gaps within supported `G x Z`
cells. Candidate selection uses the repair split only, preserves exact
coarse-group covered counts, and independently recalibrates before one held-out
test evaluation.

Run all four scores for each experimental setting:

```bash
for dataset in biobias folktables_design_a folktables_design_b atus; do
  for score in simple aps raps saps; do
    python mitigate_set_size/score_fix/run_directional_set_size.py \
      --dataset "$dataset" --score "$score" --input-root outputs
  done
done
```

Results are written to `outputs/set_size_repair/<dataset>/<score>`. See
`mitigate_set_size/score_fix/README_SET_SIZE_REPAIR.md` for the constraints,
input mapping, and dataset-specific override.

## Set-Size Robustness

The set-size robustness protocol uses 20 resplits for each of four datasets and
four scores. Resplit definitions are stored in the experiment code. All scores
within a dataset-resplit share one row partition; optimizer seed `2026` is
fixed. The preparation stage constructs nontraining pools and split manifests
from the upstream exports.

```bash
python mitigate_set_size/resplit_robustness/run_set_size_20_resplits.py \
  --stage prepare --input-root outputs \
  --protocol-root outputs/set_size_repair_robustness/protocol \
  --output-root outputs/set_size_repair_robustness/results
python mitigate_set_size/resplit_robustness/run_set_size_20_resplits.py \
  --stage run --jobs 4 --resume \
  --protocol-root outputs/set_size_repair_robustness/protocol \
  --output-root outputs/set_size_repair_robustness/results
python mitigate_set_size/resplit_robustness/validate_set_size_20_resplits.py \
  --stage aggregate \
  --protocol-root outputs/set_size_repair_robustness/protocol \
  --output-root outputs/set_size_repair_robustness/results
```

Detailed resume and output information is in
`mitigate_set_size/resplit_robustness/README_SET_SIZE_20_RESPLITS.md`.
