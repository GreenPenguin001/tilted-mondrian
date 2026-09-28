"""Command-line runner for all-score score-intervention resplit robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .source_registry import (
    DATASETS,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SCORES,
    configuration_split_seeds,
    configure_runtime_paths,
)


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def selected_datasets(values: list[str] | None) -> list[str]:
    available = [specification.dataset for specification in DATASETS]
    if not values:
        return available
    unknown = set(values) - set(available)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return [dataset for dataset in available if dataset in values]


def selected_scores(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return SCORES
    unknown = set(values) - set(SCORES)
    if unknown:
        raise ValueError(f"Unknown scores: {sorted(unknown)}")
    return tuple(score for score in SCORES if score in values)


def seed_override(arguments: argparse.Namespace) -> list[int] | None:
    if arguments.split_seed is not None:
        return [int(arguments.split_seed)]
    if arguments.seed_list:
        return parse_seed_list(arguments.seed_list)
    if arguments.num_splits is not None:
        if arguments.num_splits <= 0:
            raise ValueError("--num-splits must be positive")
        return list(
            range(
                int(arguments.seed_start),
                int(arguments.seed_start) + int(arguments.num_splits),
            )
        )
    return None


def resolve_configuration_seeds(
    arguments: argparse.Namespace,
    datasets: list[str],
    scores: tuple[str, ...],
) -> dict[tuple[str, str], list[int]]:
    override = seed_override(arguments)
    configured = configuration_split_seeds()
    return {
        (dataset, score): list(override if override is not None else configured[(dataset, score)])
        for dataset in datasets
        for score in scores
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--stage",
        choices=("pools", "run", "aggregate", "validate"),
        default="run",
    )
    command.add_argument("--dataset", action="append")
    command.add_argument("--scores", nargs="+")
    command.add_argument("--split-seed", type=int)
    command.add_argument("--seed-list")
    command.add_argument("--num-splits", type=int)
    command.add_argument("--seed-start", type=int, default=2001)
    command.add_argument("--optimizer-seed", type=int, default=2026)
    command.add_argument("--jobs", type=int, default=1)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--force-rerun-failed", action="store_true")
    command.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    command.add_argument("--biobias-input-dir", type=Path)
    command.add_argument("--folktables-a-input-dir", type=Path)
    command.add_argument("--folktables-b-input-dir", type=Path)
    command.add_argument("--atus-input-dir", type=Path)
    return command


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
    from .aggregate import aggregate_all
    from .pipeline import build_all_pools, run_jobs
    from .validate import validate_all

    if arguments.optimizer_seed != 2026:
        raise ValueError("Optimizer/search seed is scientifically fixed at 2026")
    datasets = selected_datasets(arguments.dataset)
    scores = selected_scores(arguments.scores)
    if arguments.stage == "pools":
        result = {"pools": [str(path) for path in build_all_pools(datasets)]}
    elif arguments.stage == "run":
        configuration_seeds = resolve_configuration_seeds(arguments, datasets, scores)
        records = run_jobs(
            configuration_seeds=configuration_seeds,
            jobs=max(1, int(arguments.jobs)),
            resume=arguments.resume,
            force_rerun_failed=arguments.force_rerun_failed,
        )
        result = {
            "requested_configurations": sum(len(seeds) for seeds in configuration_seeds.values()),
            "returned_configuration_records": len(records),
            "configured_resplits_per_dataset_score": {
                f"{dataset}:{score}": len(seeds)
                for (dataset, score), seeds in configuration_seeds.items()
            },
        }
    elif arguments.stage == "aggregate":
        result = aggregate_all()
    elif arguments.stage == "validate":
        result = validate_all()
    else:
        raise ValueError(f"Unsupported stage: {arguments.stage}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
