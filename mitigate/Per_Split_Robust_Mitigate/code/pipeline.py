"""Resumable dataset-seed execution for all four Table 2 scores."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    POOL_SPLIT_ORDER,
    build_pool,
    load_pool,
    loaded_dataset_from_pool,
    split_pool,
)
from .experiment import run_configuration
from .source_registry import DATASETS, SCORES, output_root


def set_random_seeds(split_seed: int) -> dict[str, int]:
    random.seed(split_seed)
    np.random.seed(split_seed)
    seeds = {
        "split_seed": int(split_seed),
        "python_seed": int(split_seed),
        "numpy_seed": int(split_seed),
        "torch_seed": int(split_seed),
        "optimizer_quasirandom_seed": 2026,
    }
    try:
        import torch

        torch.manual_seed(split_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(split_seed)
    except ImportError:
        pass
    return seeds


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version, "platform": platform.platform()}
    for module_name in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "torch"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "not_installed"
    return versions


def build_all_pools(datasets: list[str] | None = None) -> list[Path]:
    selected = datasets or [specification.dataset for specification in DATASETS]
    paths = [build_pool(dataset) for dataset in selected]
    write_pool_summary()
    return paths


def write_pool_summary() -> None:
    lines = [
        "# Nontraining Pools",
        "",
        "Only source calibration, repair, and test rows are concatenated. Train rows are never read into a pool.",
        "",
        "| Dataset | Pool N | Calibration N | Repair N | Test N | Train rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for specification in DATASETS:
        path = output_root() / "pools" / specification.dataset / "pool_manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        sizes = manifest["canonical_split_sizes"]
        lines.append(
            f"| {specification.dataset} | {manifest['n_pool']} | {sizes['calibration']} | "
            f"{sizes['repair']} | {sizes['test']} | {manifest['train_rows']} |"
        )
    lines.extend(
        [
            "",
            "Resplit sizes match the source partitions. BioBias uses the G x Y fallback hierarchy; Folktables occupation uses Y x G x Z; Folktables income uses G x Z; ATUS uses one shuffled respondent allocation. Every split manifest records the fallback trace, disjointness, union, and sizes.",
        ]
    )
    (output_root() / "pools" / "POOL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def partition_directory(dataset: str, split_seed: int) -> Path:
    return output_root() / "split_manifests" / f"seed_{split_seed}" / dataset


def ensure_partition(dataset: str, split_seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = partition_directory(dataset, split_seed)
    manifest_path = directory / "split_manifest.json"
    indices_path = directory / "split_indices.npz"
    if manifest_path.exists() and indices_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arrays = np.load(indices_path)
        return ({split: arrays[split] for split in POOL_SPLIT_ORDER}, manifest)
    if directory.exists():
        raise FileExistsError(f"Incomplete partition directory exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".{dataset}.tmp_{os.getpid()}_{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    pool, pool_manifest = load_pool(dataset)
    indices, manifest = split_pool(
        dataset, pool, pool_manifest["canonical_split_sizes"], split_seed
    )
    np.savez_compressed(temporary / "split_indices.npz", **indices)
    (temporary / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temporary, directory)
    return indices, manifest


def configuration_directory(split_seed: int, dataset: str, score: str) -> Path:
    return output_root() / "splits" / f"seed_{split_seed}" / dataset / score


def run_one_configuration(
    dataset: str,
    score: str,
    split_seed: int,
    resume: bool = True,
    force_rerun_failed: bool = False,
) -> dict[str, Any]:
    final_dir = configuration_directory(split_seed, dataset, score)
    done_path = final_dir / "DONE.json"
    if done_path.exists() and resume:
        return {"status": "skipped", **json.loads(done_path.read_text(encoding="utf-8"))}
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite result: {final_dir}")
    failed_matches = list(final_dir.parent.glob(f"{score}__FAILED_*")) if final_dir.parent.exists() else []
    if failed_matches and not force_rerun_failed:
        return {
            "status": "failed_skipped",
            "dataset": dataset,
            "score": score,
            "split_seed": split_seed,
            "existing_failures": [str(path) for path in failed_matches],
        }
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_dir.parent / f".{score}.tmp_{os.getpid()}_{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        seeds = set_random_seeds(split_seed)
        indices, partition_manifest = ensure_partition(dataset, split_seed)
        pool, pool_manifest = load_pool(dataset)
        data = loaded_dataset_from_pool(dataset, pool, indices, pool_manifest)
        summary = run_configuration(
            data=data,
            dataset=dataset,
            score=score,
            split_seed=split_seed,
            output_dir=temporary,
        )
        config = {
            "dataset": dataset,
            "score": score,
            "split_seed": int(split_seed),
            "optimizer_seed": 2026,
            "random_seeds": seeds,
            "package_versions": package_versions(),
            "partition_manifest": str(partition_directory(dataset, split_seed) / "split_manifest.json"),
            "fixed_model": True,
            "base_classifier_retrained": False,
            "train_rows_in_pool": 0,
            "test_used_for_selection": False,
            "negative_test_outcome_retained": True,
        }
        (temporary / "run_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        done = {
            "status": "success",
            "dataset": dataset,
            "score": score,
            "split_seed": int(split_seed),
            "runtime_seconds": time.perf_counter() - start,
            "candidate_count_total": int(summary["total_candidate_count"]),
            "heldout_improvement_category": summary["heldout_improvement_category"],
        }
        (temporary / "DONE.json").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final_dir)
        return done
    except Exception as error:
        failure = {
            "status": "failed",
            "dataset": dataset,
            "score": score,
            "split_seed": int(split_seed),
            "failure_stage": "configuration",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "exact_command": " ".join(sys.argv),
            "runtime_seconds": time.perf_counter() - start,
        }
        (temporary / "FAILED.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        failed_dir = final_dir.parent / f"{score}__FAILED_{uuid.uuid4().hex[:8]}"
        os.replace(temporary, failed_dir)
        return {"path": str(failed_dir), **failure}


def update_completion_markers(split_seed: int, dataset: str) -> None:
    dataset_dir = output_root() / "splits" / f"seed_{split_seed}" / dataset
    all_scores = all((dataset_dir / score / "DONE.json").exists() for score in SCORES)
    if all_scores:
        payload = {
            "status": "success",
            "dataset": dataset,
            "split_seed": int(split_seed),
            "scores": list(SCORES),
            "shared_partition_manifest": str(
                partition_directory(dataset, split_seed) / "split_manifest.json"
            ),
        }
        (dataset_dir / "ALL_SCORES_DONE.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    seed_dir = output_root() / "splits" / f"seed_{split_seed}"
    full = all((seed_dir / specification.dataset / "ALL_SCORES_DONE.json").exists() for specification in DATASETS)
    if full:
        payload = {
            "status": "success",
            "split_seed": int(split_seed),
            "datasets": [specification.dataset for specification in DATASETS],
            "scores": list(SCORES),
            "configuration_count": 16,
        }
        (seed_dir / "FULL_SEED_DONE.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def run_dataset_seed(
    dataset: str,
    split_seed: int,
    scores: tuple[str, ...] = SCORES,
    resume: bool = True,
    force_rerun_failed: bool = False,
) -> list[dict[str, Any]]:
    ensure_partition(dataset, split_seed)
    results = []
    for score in scores:
        result = run_one_configuration(
            dataset,
            score,
            split_seed,
            resume=resume,
            force_rerun_failed=force_rerun_failed,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
    update_completion_markers(split_seed, dataset)
    return results


def run_jobs(
    configuration_seeds: dict[tuple[str, str], list[int]],
    jobs: int,
    resume: bool,
    force_rerun_failed: bool,
) -> list[dict[str, Any]]:
    scores_by_dataset_seed: dict[tuple[str, int], list[str]] = {}
    for (dataset, score), split_seeds in configuration_seeds.items():
        for seed in split_seeds:
            scores_by_dataset_seed.setdefault((dataset, seed), []).append(score)
    tasks = [
        (
            dataset,
            seed,
            tuple(score for score in SCORES if score in requested_scores),
        )
        for (dataset, seed), requested_scores in scores_by_dataset_seed.items()
    ]
    results: list[dict[str, Any]] = []
    if jobs <= 1:
        for dataset, seed, scores in tasks:
            results.extend(
                run_dataset_seed(dataset, seed, scores, resume, force_rerun_failed)
            )
        return results
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                run_dataset_seed,
                dataset,
                seed,
                scores,
                resume,
                force_rerun_failed,
            ): (dataset, seed)
            for dataset, seed, scores in tasks
        }
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def remove_stale_temporary_directories() -> None:
    for path in (output_root() / "splits").glob("**/.*.tmp_*"):
        if path.is_dir():
            shutil.rmtree(path)
