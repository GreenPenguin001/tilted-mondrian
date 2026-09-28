from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))

from input_loading import LoadedDataset, load_csv_probability_dataset
from no_free_repair_core import ScoreFixConfig, run_no_free_repair_experiment
from runtime_paths import DEFAULT_RESULT_ROOT, default_input_dirs, symmetric_result_dir


def build_experiment(input_dir: Path | None = None) -> tuple[LoadedDataset, ScoreFixConfig]:
    # Load frozen probability exports and exact gender/employment metadata.
    data = load_csv_probability_dataset(
        input_dir or default_input_dirs()["atus"],
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
            "local_cells": "gender x employment status",
            "reference_cell_definition": "G x Z",
            "z_source": "existing z column asserted equal to employment_z",
        },
    )
    config = ScoreFixConfig(dataset="atus", metadata={"input_style": "csv_probabilities", "group": "sex_g"})
    return data, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATUS local coverage repair.")
    parser.add_argument("--input-dir", type=Path, default=default_input_dirs()["atus"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=symmetric_result_dir(DEFAULT_RESULT_ROOT, "atus"),
    )
    arguments = parser.parse_args()
    data, config = build_experiment(arguments.input_dir)
    run_no_free_repair_experiment(data, arguments.output_dir, config)


if __name__ == "__main__":
    main()
