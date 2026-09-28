"""Portable source and output locations for Table 1 robustness runs."""

from __future__ import annotations

import json
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COMPONENT_ROOT = PACKAGE_ROOT / "Per_Split_Robust"
DEFAULT_INPUT_ROOT = PACKAGE_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "table1_robustness"
SPLITS = ("cal", "repair", "test")
FOLKTABLES_YEARS = tuple(range(2014, 2019))


def configure_runtime_paths(
    input_root: Path,
    output_root: Path,
    biobias_dir: Path | None = None,
    folktables_a_dir: Path | None = None,
    folktables_b_dir: Path | None = None,
    atus_dir: Path | None = None,
) -> None:
    os.environ["TABLE1_INPUT_ROOT"] = str(Path(input_root).resolve())
    os.environ["TABLE1_OUTPUT_ROOT"] = str(Path(output_root).resolve())
    overrides = {
        "TABLE1_BIOBIAS_DIR": biobias_dir,
        "TABLE1_FOLKTABLES_A_DIR": folktables_a_dir,
        "TABLE1_FOLKTABLES_B_DIR": folktables_b_dir,
        "TABLE1_ATUS_DIR": atus_dir,
    }
    for variable, path in overrides.items():
        if path is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = str(Path(path).resolve())


def input_root() -> Path:
    return Path(os.environ.get("TABLE1_INPUT_ROOT", DEFAULT_INPUT_ROOT)).resolve()


def output_root() -> Path:
    return Path(os.environ.get("TABLE1_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).resolve()


def configured_input_dir(environment_name: str, default: Path) -> Path:
    return Path(os.environ.get(environment_name, default)).resolve()


def dataset_protocol(dataset: str) -> dict[str, object]:
    protocols = json.loads(
        (COMPONENT_ROOT / "configs" / "dataset_protocols.json").read_text(
            encoding="utf-8"
        )
    )
    return protocols[dataset]


def biobias_probability_dir() -> Path:
    default = input_root() / str(dataset_protocol("biobias")["input_subdirectory"])
    return configured_input_dir("TABLE1_BIOBIAS_DIR", default)


def folktables_design_dir(design: str) -> Path:
    defaults = {
        design: input_root() / str(dataset_protocol(design)["input_subdirectory"])
        for design in ("folktables_design_a", "folktables_design_b")
    }
    variables = {
        "folktables_design_a": "TABLE1_FOLKTABLES_A_DIR",
        "folktables_design_b": "TABLE1_FOLKTABLES_B_DIR",
    }
    return configured_input_dir(variables[design], defaults[design])


def atus_probability_dir() -> Path:
    default = input_root() / str(dataset_protocol("atus")["input_subdirectory"])
    return configured_input_dir("TABLE1_ATUS_DIR", default)
