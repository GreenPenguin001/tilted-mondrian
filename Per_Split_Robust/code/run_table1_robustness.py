"""Command-line interface for fixed-model random-resplit robustness runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .source_registry import (
    COMPONENT_ROOT,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    configure_runtime_paths,
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def select_units(dataset: str | None, years: list[int] | None):
    from .data import dataset_units

    selected = dataset_units()
    if dataset:
        selected = [unit for unit in selected if unit.dataset == dataset]
    if years:
        selected = [unit for unit in selected if unit.year is None or unit.year in years]
    if not selected:
        raise ValueError("Dataset/year filters select no experiment units")
    return selected


def configured_seeds() -> list[int]:
    config = json.loads(
        (COMPONENT_ROOT / "configs/split_seeds.json").read_text(encoding="utf-8")
    )
    return [int(seed) for seed in config["active_seed_list"]]


def resolve_seeds(arguments: argparse.Namespace) -> list[int]:
    if arguments.seed_list:
        return parse_csv_ints(arguments.seed_list)
    if arguments.num_splits is not None:
        if arguments.num_splits <= 0:
            raise ValueError("--num-splits must be positive")
        return list(
            range(
                int(arguments.seed_start),
                int(arguments.seed_start) + int(arguments.num_splits),
            )
        )
    return configured_seeds()


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--stage",
        choices=("pools", "run", "aggregate", "validate"),
        default="run",
    )
    argument_parser.add_argument(
        "--dataset",
        choices=("biobias", "folktables_design_a", "folktables_design_b", "atus"),
        default=None,
    )
    argument_parser.add_argument("--year", default=None, help="Comma-separated Folktables years")
    argument_parser.add_argument("--score", choices=("simple", "aps", "raps", "saps"), default=None)
    argument_parser.add_argument("--num-splits", type=int, default=None)
    argument_parser.add_argument("--seed-list", default=None, help="Comma-separated split seeds")
    argument_parser.add_argument("--seed-start", type=int, default=1001)
    argument_parser.add_argument("--workers", type=int, default=1)
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    argument_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    argument_parser.add_argument("--biobias-input-dir", type=Path)
    argument_parser.add_argument("--folktables-a-input-dir", type=Path)
    argument_parser.add_argument("--folktables-b-input-dir", type=Path)
    argument_parser.add_argument("--atus-input-dir", type=Path)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()
    configure_runtime_paths(
        arguments.input_root,
        arguments.output_root,
        arguments.biobias_input_dir,
        arguments.folktables_a_input_dir,
        arguments.folktables_b_input_dir,
        arguments.atus_input_dir,
    )
    from .aggregate import aggregate_outputs
    from .pipeline import build_all_pools, run_jobs
    from .validate import validate_results

    years = parse_csv_ints(arguments.year) if arguments.year else None
    units = select_units(arguments.dataset, years)
    if arguments.stage == "pools":
        paths = build_all_pools(units)
        print(json.dumps({"pool_paths": [str(path) for path in paths]}, indent=2))
        return
    if arguments.stage == "aggregate":
        print(json.dumps(aggregate_outputs(), indent=2))
        return
    if arguments.stage == "validate":
        validate_results()
        return
    run_jobs(
        units,
        resolve_seeds(arguments),
        arguments.score,
        max(1, arguments.workers),
        arguments.resume,
    )


if __name__ == "__main__":
    main()
