from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))

from input_loading import LoadedDataset, load_csv_probability_dataset
from no_free_repair_core import ScoreFixConfig


def build_experiment(input_dir: Path) -> tuple[LoadedDataset, ScoreFixConfig]:
    # Load frozen probability exports and exact gender/employment metadata.
    input_dir = Path(input_dir).resolve()
    data = load_csv_probability_dataset(
        input_dir,
        dataset_name="atus",
        split_files={
            "repair": "class_probabilities_repair.csv.gz",
            "calibration": "class_probabilities_calibration.csv.gz",
            "test": "class_probabilities_test.csv.gz",
        },
        label_map_csv="closed_world_topk_label_map.csv",
        intervention_mode="covariate_defined_z",
        group_order=["female", "male"],
        cell_order=["employed", "not_in_labor_force", "unemployed"],
        source_group_column="sex_g",
        source_cell_column="employment_z",
        metadata={
            "source": "frozen probability exports",
            "input_dir": str(input_dir),
            "local_cells": "sex x employment status",
            "reference_cell_definition": "G x Z",
            "z_source": "existing z column asserted equal to employment_z",
        },
    )
    config = ScoreFixConfig(dataset="atus", metadata={"input_style": "csv_probabilities", "group": "sex_g"})
    return data, config


def main() -> None:
    raise SystemExit("Use run_directional_set_size.py for raw set-size repair.")


if __name__ == "__main__":
    main()
