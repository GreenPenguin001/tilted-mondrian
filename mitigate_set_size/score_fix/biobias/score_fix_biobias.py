from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))

from input_loading import LoadedDataset, load_numpy_probability_dataset
from no_free_repair_core import ScoreFixConfig


def build_experiment(input_dir: Path) -> tuple[LoadedDataset, ScoreFixConfig]:
    # Load frozen probabilities and the exact gender/occupation encodings.
    input_dir = Path(input_dir).resolve()
    data = load_numpy_probability_dataset(
        input_dir,
        dataset_name="biobias",
        calibration_alias="cal",
        intervention_mode="label_defined_z",
        metadata={
            "source": "frozen probability exports",
            "input_dir": str(input_dir),
            "local_cells": "gender x occupation",
            "reference_cell_definition": "G x Z",
            "z_source": "exported z_{split}.npy; exactly equal to Y",
        },
    )
    config = ScoreFixConfig(dataset="biobias", metadata={"seed": 4, "input_style": "numpy"})
    return data, config


def main() -> None:
    raise SystemExit("Use run_directional_set_size.py for raw set-size repair.")


if __name__ == "__main__":
    main()
