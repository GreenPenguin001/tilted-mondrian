from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
PACKAGE_ROOT = HERE.parents[1]
DEFAULT_INPUT_ROOT = PACKAGE_ROOT / "outputs"
DEFAULT_RESULT_ROOT = DEFAULT_INPUT_ROOT / "set_size_repair"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(HERE))

from directional_set_size_core import (  # noqa: E402
    assert_output_within_experiment,
    run_directional_set_size_experiment,
    set_size_config_from_score_config,
)


DATASET_BUILDERS = {
    "biobias": "biobias.score_fix_biobias",
    "folktables_design_a": "folktables_design_a.score_fix_folktables_design_a",
    "folktables_design_b": "folktables_design_b.score_fix_folktables_design_b",
    "atus": "atus.score_fix_atus",
}


def default_input_dir(dataset: str, input_root: Path) -> Path:
    paths = {
        "biobias": input_root / "biobias" / "table1" / "seed_4" / "probabilities",
        "folktables_design_a": input_root / "folktables_design_a" / "2018" / "train",
        "folktables_design_b": input_root / "folktables_design_b" / "2018" / "train",
        "atus": input_root / "atus",
    }
    return paths[dataset]


def load_experiment(dataset: str, input_dir: Path):
    import importlib

    module = importlib.import_module(DATASET_BUILDERS[dataset])
    return module.build_experiment(input_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Tilted Mondrian set-size repair from saved probability outputs."
        )
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_BUILDERS), required=True)
    parser.add_argument(
        "--score", choices=("simple", "aps", "raps", "saps"), required=True
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing upstream probability exports.",
    )
    parser.add_argument("--input-dir", type=Path, help="Dataset-specific input override.")
    parser.add_argument("--output-dir", type=Path, help="Result directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = (args.input_dir or default_input_dir(args.dataset, args.input_root)).resolve()
    output_dir = (
        args.output_dir
        or DEFAULT_RESULT_ROOT / args.dataset / args.score
    ).resolve()
    output_dir = assert_output_within_experiment(output_dir)
    data, score_config = load_experiment(args.dataset, input_dir)
    config = set_size_config_from_score_config(score_config)
    command = (
        "python score_fix/run_directional_set_size.py "
        f"--dataset {args.dataset} --score {args.score} --input-dir {input_dir} "
        f"--output-dir {output_dir}"
    )
    run_directional_set_size_experiment(
        data=data,
        config=config,
        score_method=args.score,
        result_dir=output_dir,
        command=command,
    )


if __name__ == "__main__":
    main()
