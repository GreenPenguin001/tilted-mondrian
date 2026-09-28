"""Portable input and output locations for score-intervention runners."""

from __future__ import annotations

import argparse
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM_ROOT = PACKAGE_ROOT / "outputs"
DEFAULT_RESULT_ROOT = DEFAULT_UPSTREAM_ROOT / "score_fix"


def default_input_dirs(input_root: Path = DEFAULT_UPSTREAM_ROOT) -> dict[str, Path]:
    root = Path(input_root)
    return {
        "biobias": root / "biobias" / "table1" / "seed_4" / "probabilities",
        "folktables_design_a": root / "folktables_design_a" / "2018" / "train",
        "folktables_design_b": root / "folktables_design_b" / "2018" / "train",
        "atus": root / "atus",
    }


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_UPSTREAM_ROOT,
        help="Root containing the documented upstream experiment outputs.",
    )
    parser.add_argument("--biobias-input-dir", type=Path)
    parser.add_argument("--folktables-a-input-dir", type=Path)
    parser.add_argument("--folktables-b-input-dir", type=Path)
    parser.add_argument("--atus-input-dir", type=Path)


def resolve_input_dirs(arguments: argparse.Namespace) -> dict[str, Path]:
    resolved = default_input_dirs(arguments.input_root)
    overrides = {
        "biobias": arguments.biobias_input_dir,
        "folktables_design_a": arguments.folktables_a_input_dir,
        "folktables_design_b": arguments.folktables_b_input_dir,
        "atus": arguments.atus_input_dir,
    }
    for dataset, path in overrides.items():
        if path is not None:
            resolved[dataset] = Path(path)
    return {dataset: path.resolve() for dataset, path in resolved.items()}


def symmetric_result_dir(result_root: Path, dataset: str) -> Path:
    return Path(result_root) / dataset / "symmetric"


def directional_result_dir(result_root: Path, dataset: str) -> Path:
    return Path(result_root) / dataset / "directional_asymmetric"
