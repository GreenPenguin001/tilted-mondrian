"""Resumable fixed-model random-resplit execution pipeline."""

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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import (
    CANONICAL_SPLITS,
    DatasetUnit,
    build_pool,
    dataset_units,
    load_pool,
    load_source_unit,
    scores_for_unit,
    split_pool,
)
from .metrics import evaluate_score
from .propensity import fit_propensity
from .source_registry import output_root


def set_random_seeds(split_seed: int) -> dict[str, int]:
    seeds = {
        "split_seed": int(split_seed),
        "python_seed": int(split_seed),
        "numpy_seed": int(split_seed),
        "torch_global_seed": int(split_seed),
        "biobias_propensity_seed": 0,
        "folktables_propensity_seed": 20260609,
        "atus_propensity_seed": 4,
    }
    random.seed(split_seed)
    np.random.seed(split_seed)
    try:
        import torch

        torch.manual_seed(split_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(split_seed)
    except ImportError:
        pass
    return seeds


def package_versions() -> dict[str, str]:
    modules = ("numpy", "pandas", "sklearn", "scipy", "torch", "matplotlib")
    versions: dict[str, str] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for name in modules:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not_installed"
    return versions


def build_all_pools(selected_units: Iterable[DatasetUnit] | None = None) -> list[Path]:
    paths = []
    for unit in selected_units or dataset_units():
        paths.append(build_pool(load_source_unit(unit)))
    return paths


def result_directory(unit: DatasetUnit, split_seed: int, requested_score: str | None) -> Path:
    suffix = "" if requested_score is None else f"__score_{requested_score}"
    return output_root() / "splits" / unit.key / f"seed_{split_seed}{suffix}"


def run_one_unit_seed(
    unit: DatasetUnit,
    split_seed: int,
    requested_score: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    final_dir = result_directory(unit, split_seed, requested_score)
    if final_dir.exists():
        done_path = final_dir / "DONE.json"
        if done_path.exists() and resume:
            payload = json.loads(done_path.read_text(encoding="utf-8"))
            return {"status": "skipped", "path": str(final_dir), **payload}
        raise FileExistsError(f"Refusing to overwrite existing result directory: {final_dir}")

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = final_dir.parent / f".{final_dir.name}.tmp_{os.getpid()}_{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        seeds = set_random_seeds(split_seed)
        pool, pool_manifest = load_pool(unit)
        split_indices, split_manifest = split_pool(
            unit, pool, pool_manifest["canonical_split_sizes"], split_seed
        )
        np.savez_compressed(
            temporary_dir / "split_indices.npz",
            cal=split_indices["cal"],
            repair=split_indices["repair"],
            test=split_indices["test"],
        )
        (temporary_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        split_data = {name: pool.take(split_indices[name]) for name in CANONICAL_SPLITS}

        propensity_start = time.perf_counter()
        propensity = fit_propensity(unit, split_data["repair"], split_data["test"])
        propensity_runtime = time.perf_counter() - propensity_start
        (temporary_dir / "propensity_diagnostics.json").write_text(
            json.dumps(propensity.diagnostics, indent=2, default=str) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            temporary_dir / "overlap_weights.npz",
            repair=propensity.repair_weights,
            test=propensity.test_weights,
        )

        available_scores = scores_for_unit(unit)
        if requested_score is not None:
            if requested_score not in available_scores:
                raise ValueError(f"score={requested_score} is unavailable for {unit.key}; choices={available_scores}")
            selected_scores = (requested_score,)
        else:
            selected_scores = available_scores
        summary_rows: list[dict[str, Any]] = []
        for score in selected_scores:
            score_start = time.perf_counter()
            result = evaluate_score(
                unit,
                score,
                split_data["cal"],
                split_data["test"],
                propensity.test_weights,
                reported_ess_weights=propensity.test_weights,
                reported_ess_split="test",
            )
            score_dir = temporary_dir / score
            score_dir.mkdir()
            result.cell_grid.to_csv(score_dir / "cell_grid.csv", index=False)
            result.group_grid.to_csv(score_dir / "group_grid.csv", index=False)
            result.thresholds.to_csv(score_dir / "thresholds.csv", index=False)
            score_runtime = time.perf_counter() - score_start
            row = {
                **result.summary,
                "split_seed": int(split_seed),
                "alpha": "grid",
                "propensity_model_selected": propensity.diagnostics["propensity_model_selected"],
                "propensity_diagnostics": json.dumps(propensity.diagnostics, sort_keys=True, default=str),
                "clipping_cap_parameters": json.dumps(
                    {
                        key: value
                        for key, value in propensity.diagnostics.items()
                        if "clip" in key or "cap" in key or "trim" in key or "max_weight" in key
                    },
                    sort_keys=True,
                    default=str,
                ),
                "threshold_values": result.thresholds.to_json(orient="records"),
                "propensity_runtime_seconds": propensity_runtime,
                "score_runtime_seconds": score_runtime,
                "runtime_seconds": time.perf_counter() - start,
                "warning_flags": "",
                "failure_flags": "",
            }
            pd.DataFrame([row]).to_csv(score_dir / "summary.csv", index=False)
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(temporary_dir / "summary.csv", index=False)
        run_config = {
            "unit": asdict(unit),
            "split_seed": int(split_seed),
            "requested_score": requested_score,
            "completed_scores": list(selected_scores),
            "random_seeds": seeds,
            "package_versions": package_versions(),
            "fixed_model": True,
            "base_classifier_retrained": False,
            "train_rows_in_pool": 0,
            "propensity_refit_on_repair": True,
            "test_used_for_selection": False,
        }
        (temporary_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8"
        )
        done = {
            "status": "success",
            "dataset": unit.dataset,
            "year": unit.year,
            "split_seed": int(split_seed),
            "scores": list(selected_scores),
            "runtime_seconds": time.perf_counter() - start,
        }
        (temporary_dir / "DONE.json").write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_dir, final_dir)
        return {"path": str(final_dir), **done}
    except Exception as error:
        failure = {
            "status": "failed",
            "dataset": unit.dataset,
            "year": unit.year,
            "split_seed": int(split_seed),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - start,
        }
        (temporary_dir / "FAILED.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        failed_dir = final_dir.parent / f"{final_dir.name}__FAILED_{uuid.uuid4().hex[:8]}"
        os.replace(temporary_dir, failed_dir)
        return {"path": str(failed_dir), **failure}


def run_jobs(
    units: list[DatasetUnit],
    split_seeds: list[int],
    requested_score: str | None,
    workers: int,
    resume: bool,
) -> list[dict[str, Any]]:
    jobs = [(unit, seed) for seed in split_seeds for unit in units]
    biobias_jobs = [(unit, seed) for unit, seed in jobs if unit.dataset == "biobias"]
    cpu_jobs = [(unit, seed) for unit, seed in jobs if unit.dataset != "biobias"]
    results: list[dict[str, Any]] = []

    # BioBias uses at most one GPU propensity job at a time.
    for unit, seed in biobias_jobs:
        result = run_one_unit_seed(unit, seed, requested_score=requested_score, resume=resume)
        print(json.dumps(result, default=str), flush=True)
        results.append(result)

    if workers <= 1:
        for unit, seed in cpu_jobs:
            result = run_one_unit_seed(unit, seed, requested_score=requested_score, resume=resume)
            print(json.dumps(result, default=str), flush=True)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(run_one_unit_seed, unit, seed, requested_score, resume): (unit, seed)
                for unit, seed in cpu_jobs
            }
            for future in as_completed(future_map):
                result = future.result()
                print(json.dumps(result, default=str), flush=True)
                results.append(result)
    return results


def remove_temporary_directories() -> None:
    for path in (output_root() / "splits").glob("**/.*.tmp_*"):
        if path.is_dir():
            shutil.rmtree(path)
