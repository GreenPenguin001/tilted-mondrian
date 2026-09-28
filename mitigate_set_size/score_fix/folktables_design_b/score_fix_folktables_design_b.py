from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))

from input_loading import LoadedDataset, load_numpy_probability_dataset
from no_free_repair_core import ScoreFixConfig


def build_experiment(input_dir: Path) -> tuple[LoadedDataset, ScoreFixConfig]:
    # Load frozen probabilities and the exact race/education encodings.
    input_dir = Path(input_dir).resolve()
    data = load_numpy_probability_dataset(
        input_dir,
        dataset_name="folktables_design_b",
        calibration_alias="cal",
        intervention_mode="covariate_defined_z",
        metadata={
            "source": "frozen probability exports",
            "input_dir": str(input_dir),
            "acs_year": 2018,
            "local_cells": "race group x education bucket",
            "reference_cell_definition": "G x Z",
            "z_source": "exported z_{split}.npy",
        },
    )
    config = ScoreFixConfig(dataset="folktables_design_b", metadata={"acs_year": 2018, "input_style": "numpy"})
    return data, config


def main() -> None:
    raise SystemExit("Use run_directional_set_size.py for raw set-size repair.")


if __name__ == "__main__":
    main()
