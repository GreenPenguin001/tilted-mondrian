"""Portable source and output locations for repair robustness runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
MITIGATE_ROOT = COMPONENT_ROOT.parent
PACKAGE_ROOT = MITIGATE_ROOT.parent
SCORE_FIX_ROOT = MITIGATE_ROOT / "score_fix"
DEFAULT_INPUT_ROOT = PACKAGE_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "repair_robustness"
SCORES = ("simple", "aps", "raps", "saps")


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    intervention_mode: str
    local_cell_definition: str
    group_label: str

    @property
    def input_dir(self) -> Path:
        return input_dir_for_dataset(self.dataset)


DATASETS = (
    DatasetSpec("biobias", "label_defined_z", "gender x occupation", "gender"),
    DatasetSpec(
        "folktables_design_a",
        "covariate_defined_z",
        "gender x education bucket",
        "gender",
    ),
    DatasetSpec(
        "folktables_design_b",
        "covariate_defined_z",
        "race x education bucket",
        "race",
    ),
    DatasetSpec(
        "atus",
        "covariate_defined_z",
        "gender x employment status",
        "gender",
    ),
)


def configure_runtime_paths(
    input_root: Path,
    output_root_path: Path,
    biobias_dir: Path | None = None,
    folktables_a_dir: Path | None = None,
    folktables_b_dir: Path | None = None,
    atus_dir: Path | None = None,
) -> None:
    os.environ["REPAIR_ROBUSTNESS_INPUT_ROOT"] = str(Path(input_root).resolve())
    os.environ["REPAIR_ROBUSTNESS_OUTPUT_ROOT"] = str(Path(output_root_path).resolve())
    overrides = {
        "REPAIR_ROBUSTNESS_BIOBIAS_DIR": biobias_dir,
        "REPAIR_ROBUSTNESS_FOLKTABLES_A_DIR": folktables_a_dir,
        "REPAIR_ROBUSTNESS_FOLKTABLES_B_DIR": folktables_b_dir,
        "REPAIR_ROBUSTNESS_ATUS_DIR": atus_dir,
    }
    for variable, path in overrides.items():
        if path is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = str(Path(path).resolve())


def input_root() -> Path:
    return Path(os.environ.get("REPAIR_ROBUSTNESS_INPUT_ROOT", DEFAULT_INPUT_ROOT)).resolve()


def output_root() -> Path:
    return Path(os.environ.get("REPAIR_ROBUSTNESS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).resolve()


def input_dir_for_dataset(dataset: str) -> Path:
    defaults = {
        "biobias": input_root() / "biobias" / "table1" / "seed_4" / "probabilities",
        "folktables_design_a": input_root() / "folktables_design_a" / "2018" / "train",
        "folktables_design_b": input_root() / "folktables_design_b" / "2018" / "train",
        "atus": input_root() / "atus",
    }
    variables = {
        "biobias": "REPAIR_ROBUSTNESS_BIOBIAS_DIR",
        "folktables_design_a": "REPAIR_ROBUSTNESS_FOLKTABLES_A_DIR",
        "folktables_design_b": "REPAIR_ROBUSTNESS_FOLKTABLES_B_DIR",
        "atus": "REPAIR_ROBUSTNESS_ATUS_DIR",
    }
    return Path(os.environ.get(variables[dataset], defaults[dataset])).resolve()


def dataset_spec(dataset: str) -> DatasetSpec:
    for specification in DATASETS:
        if specification.dataset == dataset:
            return specification
    raise ValueError(f"Unknown dataset: {dataset}")


def configuration_split_seeds() -> dict[tuple[str, str], list[int]]:
    config = json.loads(
        (COMPONENT_ROOT / "configs" / "split_seeds.json").read_text(encoding="utf-8")
    )
    configured = config["configuration_split_seeds"]
    return {
        (specification.dataset, score): [
            int(seed) for seed in configured[specification.dataset][score]
        ]
        for specification in DATASETS
        for score in SCORES
    }


def configured_configuration_keys() -> set[tuple[str, str, int]]:
    return {
        (dataset, score, seed)
        for (dataset, score), seeds in configuration_split_seeds().items()
        for seed in seeds
    }
