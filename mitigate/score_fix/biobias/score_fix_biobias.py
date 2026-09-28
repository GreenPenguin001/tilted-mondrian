from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))

from input_loading import LoadedDataset, load_numpy_probability_dataset
from no_free_repair_core import ScoreFixConfig, run_no_free_repair_experiment
from runtime_paths import DEFAULT_RESULT_ROOT, default_input_dirs, symmetric_result_dir


def build_experiment(input_dir: Path | None = None) -> tuple[LoadedDataset, ScoreFixConfig]:
    # Load frozen probabilities and the exact gender/occupation encodings.
    data = load_numpy_probability_dataset(
        input_dir or default_input_dirs()["biobias"],
        dataset_name="biobias",
        calibration_alias="cal",
        intervention_mode="label_defined_z",
        metadata={
            "source": "frozen probability exports",
            "local_cells": "gender x occupation",
            "reference_cell_definition": "G x Z",
            "z_source": "exported z_{split}.npy; exactly equal to Y",
        },
    )
    config = ScoreFixConfig(dataset="biobias", metadata={"seed": 4, "input_style": "numpy"})
    return data, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BioBias local coverage repair.")
    parser.add_argument("--input-dir", type=Path, default=default_input_dirs()["biobias"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=symmetric_result_dir(DEFAULT_RESULT_ROOT, "biobias"),
    )
    arguments = parser.parse_args()
    data, config = build_experiment(arguments.input_dir)
    run_no_free_repair_experiment(data, arguments.output_dir, config)


if __name__ == "__main__":
    main()
