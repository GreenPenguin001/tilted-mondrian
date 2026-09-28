from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import platform
import random
import shutil
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_INPUT_ROOT = PACKAGE_ROOT / "outputs"
DEFAULT_PROTOCOL_ROOT = DEFAULT_INPUT_ROOT / "set_size_repair_robustness" / "protocol"
DEFAULT_RESULT_ROOT = DEFAULT_INPUT_ROOT / "set_size_repair_robustness" / "results"
PROTOCOL_ROOT = DEFAULT_PROTOCOL_ROOT
POOL_ROOT = PROTOCOL_ROOT / "pools"
PARTITION_ROOT = PROTOCOL_ROOT / "split_manifests"
RESULT_ROOT = DEFAULT_RESULT_ROOT
RUN_ROOT = RESULT_ROOT / "runs"
FAILURE_ROOT = RESULT_ROOT / "failures"

COMMON = EXPERIMENT_ROOT / "score_fix" / "common"
SCORE_FIX = EXPERIMENT_ROOT / "score_fix"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(SCORE_FIX))

from directional_set_size_core import (  # noqa: E402
    apply_set_size_score_matrix,
    define_set_size_roles,
    run_directional_set_size_experiment,
    set_size_config_from_score_config,
)
from input_loading import LoadedDataset, SplitData  # noqa: E402
from no_free_repair_core import (  # noqa: E402
    ScoreFixConfig,
    compute_baseline_group_thresholds,
    evaluate_split,
    labelwise_scores_from_probs,
)


DATASETS = (
    "biobias",
    "folktables_design_a",
    "folktables_design_b",
    "atus",
)
SCORES = ("simple", "aps", "raps", "saps")
RESPLIT_SEEDS = tuple(range(2001, 2021))
EXPECTED_EVENTS = (
    "repair_ready",
    "search_started",
    "theta_selected",
    "theta_frozen",
    "calibration_started",
    "calibration_scores_constructed",
    "calibration_baseline_evaluated",
    "calibration_thresholds_frozen",
    "test_started",
    "test_scores_constructed",
    "baseline_test_evaluated",
    "repaired_test_evaluated",
)
CORE_OUTPUTS = (
    "directional_set_size_summary.csv",
    "directional_set_size_offsets.csv",
    "directional_set_size_cell_ledger.csv",
    "directional_set_size_group_ledger.csv",
    "directional_set_size_search_diagnostics.csv",
    "directional_set_size_validation.csv",
    "directional_set_size_test_results.csv",
    "directional_set_size_protocol_events.csv",
    "config.json",
    "COMMAND.txt",
    "execution.log",
)


@dataclass(frozen=True)
class PoolData:
    probs: np.ndarray
    y: np.ndarray
    g: np.ndarray
    z: np.ndarray
    row_id: np.ndarray
    canonical_origin: np.ndarray


def assert_write_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    package_root = PACKAGE_ROOT.resolve()
    if resolved != package_root and package_root not in resolved.parents:
        raise ValueError(f"Writable path must remain under {package_root}: {resolved}")
    return resolved


def configure_paths(protocol_root: Path, result_root: Path) -> None:
    global PROTOCOL_ROOT, POOL_ROOT, PARTITION_ROOT
    global RESULT_ROOT, RUN_ROOT, FAILURE_ROOT

    PROTOCOL_ROOT = assert_write_path(protocol_root)
    RESULT_ROOT = assert_write_path(result_root)
    POOL_ROOT = PROTOCOL_ROOT / "pools"
    PARTITION_ROOT = PROTOCOL_ROOT / "split_manifests"
    RUN_ROOT = RESULT_ROOT / "runs"
    FAILURE_ROOT = RESULT_ROOT / "failures"


def prepare_protocol(
    input_root: Path,
    protocol_root: Path,
    biobias_input_dir: Path | None,
    folktables_a_input_dir: Path | None,
    folktables_b_input_dir: Path | None,
    atus_input_dir: Path | None,
) -> None:
    input_root = Path(input_root).resolve()
    protocol_root = assert_write_path(protocol_root)
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from mitigate.Per_Split_Robust_Mitigate.code.pipeline import (
        build_all_pools,
        ensure_partition,
    )
    from mitigate.Per_Split_Robust_Mitigate.code.source_registry import (
        configure_runtime_paths,
    )

    configure_runtime_paths(
        input_root,
        protocol_root,
        biobias_input_dir,
        folktables_a_input_dir,
        folktables_b_input_dir,
        atus_input_dir,
    )
    build_all_pools(list(DATASETS))
    for dataset in DATASETS:
        for seed in RESPLIT_SEEDS:
            ensure_partition(dataset, seed)
    print(
        json.dumps(
            {
                "status": "prepared",
                "protocol_root": str(protocol_root.resolve()),
                "datasets": list(DATASETS),
                "resplit_seeds": list(RESPLIT_SEEDS),
            },
            indent=2,
        )
    )


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def json_dump(path: Path, payload: Any) -> None:
    path = assert_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def load_pool(dataset: str) -> tuple[PoolData, dict[str, Any]]:
    directory = POOL_ROOT / dataset
    manifest = json.loads((directory / "pool_manifest.json").read_text(encoding="utf-8"))
    pool = PoolData(
        probs=np.load(directory / "probs.npy", mmap_mode="r"),
        y=np.load(directory / "y.npy", mmap_mode="r"),
        g=np.load(directory / "g.npy", mmap_mode="r"),
        z=np.load(directory / "z.npy", mmap_mode="r"),
        row_id=np.load(directory / "row_id.npy", mmap_mode="r"),
        canonical_origin=np.load(directory / "canonical_origin.npy", mmap_mode="r"),
    )
    lengths = {len(getattr(pool, name)) for name in PoolData.__annotations__}
    if lengths != {int(manifest["n_pool"])}:
        raise AssertionError(f"{dataset}: pool arrays are not aligned: {lengths}")
    if len(np.unique(np.asarray(pool.row_id).astype(str))) != len(pool.row_id):
        raise AssertionError(f"{dataset}: pool row IDs are not unique")
    origins = set(np.asarray(pool.canonical_origin).astype(str).tolist())
    if origins != {"calibration", "repair", "test"}:
        raise AssertionError(f"{dataset}: unexpected canonical origins: {origins}")
    if int(manifest.get("train_rows", -1)) != 0:
        raise AssertionError(f"{dataset}: pool manifest contains train rows")
    return pool, manifest


def validate_pool_content(dataset: str) -> dict[str, Any]:
    pool, manifest = load_pool(dataset)
    if not np.isfinite(np.asarray(pool.probs, dtype=float)).all():
        raise AssertionError(f"{dataset}: pool probabilities contain non-finite values")
    return {
        "dataset": dataset,
        "n_pool": int(manifest["n_pool"]),
        "canonical_split_sizes": manifest["canonical_split_sizes"],
        "train_rows": int(manifest["train_rows"]),
        "canonical_origins": sorted(set(np.asarray(pool.canonical_origin).astype(str))),
    }


def partition_paths(dataset: str, seed: int) -> tuple[Path, Path]:
    directory = PARTITION_ROOT / f"seed_{seed}" / dataset
    return directory / "split_indices.npz", directory / "split_manifest.json"


def load_partition(
    dataset: str, seed: int, pool: PoolData, pool_manifest: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    indices_path, manifest_path = partition_paths(dataset, seed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = np.load(indices_path)
    indices = {
        split: np.asarray(archive[split], dtype=np.int64)
        for split in ("calibration", "repair", "test")
    }
    sizes = pool_manifest["canonical_split_sizes"]
    for split, values in indices.items():
        if len(values) != int(sizes[split]) or len(values) != int(manifest["sizes"][split]):
            raise AssertionError(f"{dataset} seed={seed}: {split} size mismatch")
        if np.any(values < 0) or np.any(values >= int(pool_manifest["n_pool"])):
            raise AssertionError(f"{dataset} seed={seed}: out-of-range split index")
    union = np.concatenate([indices[name] for name in ("calibration", "repair", "test")])
    if len(np.unique(union)) != len(union) or set(union.tolist()) != set(range(len(pool.y))):
        raise AssertionError(f"{dataset} seed={seed}: partition is not disjoint/full")
    if int(manifest["split_seed"]) != seed or int(manifest["train_rows"]) != 0:
        raise AssertionError(f"{dataset} seed={seed}: manifest seed/train mismatch")
    audit = {
        "dataset": dataset,
        "resplit_id": seed - 2000,
        "resplit_seed": seed,
        "n_repair": len(indices["repair"]),
        "n_calibration": len(indices["calibration"]),
        "n_test": len(indices["test"]),
        "train_rows": 0,
        "disjoint": True,
        "union_is_full_pool": True,
        "fallback_trace": manifest["fallback_trace"],
    }
    return indices, manifest, audit


def build_loaded_dataset(
    dataset: str,
    seed: int,
    pool: PoolData,
    pool_manifest: dict[str, Any],
    indices: dict[str, np.ndarray],
) -> LoadedDataset:
    splits = {}
    for split, rows in indices.items():
        splits[split] = SplitData(
            probs=np.asarray(pool.probs[rows], dtype=float),
            y=np.asarray(pool.y[rows], dtype=int),
            g=np.asarray(pool.g[rows], dtype=str),
            z=np.asarray(pool.z[rows], dtype=str),
        )
    data = LoadedDataset(
        dataset=dataset,
        splits=splits,
        label_values=[str(value) for value in pool_manifest["label_values"]],
        label_names={str(k): str(v) for k, v in pool_manifest["label_names"].items()},
        group_values=[str(value) for value in pool_manifest["group_values"]],
        group_names={str(k): str(v) for k, v in pool_manifest["group_names"].items()},
        cell_values=[str(value) for value in pool_manifest["cell_values"]],
        cell_names={str(k): str(v) for k, v in pool_manifest["cell_names"].items()},
        intervention_mode=str(pool_manifest["intervention_mode"]),
        source_format="frozen_nontraining_pool",
        metadata={"fixed_model": True, "resplit_seed": seed},
    )
    if dataset == "biobias":
        labels = np.asarray(data.label_values, dtype=str)
        for split, values in data.splits.items():
            if not np.array_equal(values.z.astype(str), labels[values.y]):
                raise AssertionError(f"BioBias seed={seed} {split}: Z != realized Y")
    return data


def dataset_metadata(dataset: str, seed: int) -> dict[str, Any]:
    values: dict[str, Any] = {"fixed_model": True, "resplit_seed": seed}
    if dataset == "biobias":
        values.update({"seed": 4, "input_style": "numpy"})
    elif dataset.startswith("folktables"):
        values.update({"acs_year": 2018, "input_style": "numpy"})
    else:
        values.update({"input_style": "csv_probabilities", "group": "sex_g"})
    return values


def build_config(dataset: str, seed: int):
    base = ScoreFixConfig(dataset=dataset, metadata=dataset_metadata(dataset, seed))
    return set_size_config_from_score_config(base)


def seed_random_generators(seed: int) -> dict[str, int]:
    random.seed(seed)
    np.random.seed(seed)
    result = {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "optimizer_quasirandom_seed": 2026,
    }
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return result


def problem_config_payload(
    dataset: str,
    score: str,
    seed: int,
    config: Any,
    partition_audit: dict[str, Any],
    pool_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": "raw_set_size_20_resplit_robustness",
        "dataset": dataset,
        "score": score,
        "resplit_id": seed - 2000,
        "resplit_seed": seed,
        "optimizer_seed": 2026,
        "scientific_config": asdict(config),
        "fixed_model": True,
        "base_classifier_retrained": False,
        "train_rows_in_pool": 0,
        "test_used_for_selection": False,
    }


def problem_directory(seed: int, dataset: str, score: str) -> Path:
    return RUN_ROOT / f"seed_{seed}" / dataset / score


def verify_resume(directory: Path, expected_config: dict[str, Any]) -> dict[str, Any]:
    done_path = directory / "DONE.json"
    if not done_path.exists():
        raise FileExistsError(f"Incomplete output directory cannot be resumed: {directory}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "success" or not done.get("validation_pass"):
        raise AssertionError(f"Completed marker is not valid: {done_path}")
    run_config_path = directory / "resplit_run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    normalized_expected = json.loads(json.dumps(expected_config, default=str))
    for key, expected in normalized_expected.items():
        if run_config.get(key) != expected:
            raise AssertionError(f"Configuration mismatch on resume: {directory} ({key})")
    missing = [name for name in CORE_OUTPUTS if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed outputs in {directory}: {missing}")
    return done


def biobias_no_true_y_runtime_check(
    data: LoadedDataset, config: Any, score: str, offsets: pd.DataFrame
) -> bool:
    repair = data.splits["repair"]
    scores = labelwise_scores_from_probs(
        repair.probs,
        score,
        raps_lambda=config.raps_lambda,
        raps_k_reg=config.raps_k_reg,
        saps_lambda=config.saps_lambda,
    )
    thresholds = compute_baseline_group_thresholds(
        scores, repair, data.group_values, config.alpha
    )
    baseline = evaluate_split(scores, repair, thresholds.thresholds)
    roles = define_set_size_roles(repair, baseline, data, config)
    ordered = offsets.sort_values("cell_idx")
    plus = ordered["theta_plus"].to_numpy(float)
    minus = ordered["theta_minus"].to_numpy(float)
    original = apply_set_size_score_matrix(scores, repair, data, plus, minus, roles)
    altered_y = np.roll(repair.y.copy(), 1)
    y_permuted = SplitData(
        probs=repair.probs,
        y=altered_y,
        g=repair.g,
        z=repair.z,
    )
    permuted = apply_set_size_score_matrix(scores, y_permuted, data, plus, minus, roles)
    return bool(np.array_equal(original, permuted))


def validate_core_outputs(
    directory: Path,
    data: LoadedDataset,
    config: Any,
    dataset: str,
    score: str,
    partition_audit: dict[str, Any],
) -> dict[str, Any]:
    missing = [name for name in CORE_OUTPUTS if not (directory / name).is_file()]
    if missing:
        raise AssertionError(f"Missing core outputs: {missing}")
    summary = pd.read_csv(directory / "directional_set_size_summary.csv").iloc[0]
    cells = pd.read_csv(directory / "directional_set_size_cell_ledger.csv")
    groups = pd.read_csv(directory / "directional_set_size_group_ledger.csv")
    search = pd.read_csv(directory / "directional_set_size_search_diagnostics.csv")
    validations = pd.read_csv(directory / "directional_set_size_validation.csv")
    events = pd.read_csv(directory / "directional_set_size_protocol_events.csv")
    offsets = pd.read_csv(directory / "directional_set_size_offsets.csv")
    checks: dict[str, bool] = {}
    checks["runtime_validation_all_pass"] = bool(validations["passed"].map(as_bool).all())
    checks["repair_group_counts_exact"] = as_bool(summary["repair_group_covered_counts_exact"])
    checks["calibration_group_counts_exact"] = as_bool(
        summary["calibration_group_covered_counts_exact"]
    )
    repair_group = groups[groups["split"] == "repair"]
    calibration_group = groups[groups["split"] == "calibration"]
    checks["repair_group_counts_ledger_exact"] = bool(
        (repair_group["baseline_covered_count"] == repair_group["repaired_covered_count"]).all()
    )
    checks["calibration_group_counts_ledger_exact"] = bool(
        (
            calibration_group["baseline_covered_count"]
            == calibration_group["repaired_covered_count"]
        ).all()
    )
    repair_cells = cells[cells["split"] == "repair"].copy()
    supported = repair_cells[repair_cells["supported"].map(as_bool)]
    checks["gap_nonincrease"] = bool((supported["gap_nonincrease_margin"] >= -1e-12).all())
    checks["larger_side_nonincrease"] = bool(
        (supported["larger_side_nonincrease_margin"].dropna() >= -1e-12).all()
    )
    checks["smaller_side_budget"] = bool(
        (supported["smaller_side_budget_margin"].dropna() >= -1e-12).all()
    )
    checks["neutral_cell_budget"] = bool(
        (supported["neutral_group_budget_margin"].dropna() >= -1e-12).all()
    )
    expected_support = (
        repair_cells.groupby("cell_idx")["n"].min().sort_index() >= config.min_cell_count
    )
    observed_support = (
        repair_cells.groupby("cell_idx")["supported"].first().map(as_bool).sort_index()
    )
    checks["repair_only_support_rule"] = bool(expected_support.equals(observed_support))
    zero = search[search["optimizer_phase"] == "zero_baseline"]
    checks["zero_theta_evaluated_and_feasible"] = bool(
        len(zero) >= 1 and zero["feasible"].map(as_bool).all()
    )
    checks["protocol_event_order"] = tuple(
        events.sort_values("sequence")["event"].astype(str).tolist()
    ) == EXPECTED_EVENTS
    checks["split_integrity"] = bool(
        partition_audit["train_rows"] == 0
        and partition_audit["disjoint"]
        and partition_audit["union_is_full_pool"]
    )
    checks["all_supported_cells_checked"] = int(summary["supported_cell_count"]) == int(
        expected_support.sum()
    )
    if dataset == "biobias":
        checks["candidate_label_shift_independent_of_true_y"] = biobias_no_true_y_runtime_check(
            data, config, score, offsets
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Scientific validation failed: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "supported_cell_count": int(summary["supported_cell_count"]),
        "neutral_cell_count": int(summary["neutral_cell_count"]),
    }


def run_dataset_seed(
    dataset: str,
    seed: int,
    scores: tuple[str, ...],
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    pool, pool_manifest = load_pool(dataset)
    indices, _, partition_audit = load_partition(dataset, seed, pool, pool_manifest)
    data = build_loaded_dataset(dataset, seed, pool, pool_manifest, indices)
    config = build_config(dataset, seed)
    statuses: list[dict[str, Any]] = []
    for score in scores:
        seed_random_generators(seed)
        payload = problem_config_payload(
            dataset, score, seed, config, partition_audit, pool_manifest
        )
        final_dir = assert_write_path(problem_directory(seed, dataset, score))
        if final_dir.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite existing result: {final_dir}")
            done = verify_resume(final_dir, payload)
            statuses.append({"status": "skipped", **done})
            continue
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_dir.parent / f".{score}.tmp_{os.getpid()}_{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        problem_started = time.perf_counter()
        command = (
            "python mitigate_set_size/resplit_robustness/run_set_size_20_resplits.py "
            f"--stage run --dataset {dataset} --score {score} --seed {seed} --jobs 1 --resume"
        )
        try:
            artifacts = run_directional_set_size_experiment(
                data=data,
                config=config,
                score_method=score,
                result_dir=temporary,
                command=command,
            )
            validation = validate_core_outputs(
                temporary, data, config, dataset, score, partition_audit
            )
            run_config = {
                **payload,
                "random_seeds": seed_random_generators(seed),
                "partition_audit": partition_audit,
                "runtime_environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                },
            }
            json_dump(temporary / "resplit_run_config.json", run_config)
            json_dump(temporary / "resplit_validation.json", validation)
            runtime = time.perf_counter() - problem_started
            done = {
                "status": "success",
                "dataset": dataset,
                "score": score,
                "resplit_id": seed - 2000,
                "resplit_seed": seed,
                "optimizer_seed": 2026,
                "validation_pass": True,
                "runtime_seconds": runtime,
            }
            json_dump(temporary / "DONE.json", done)
            os.replace(temporary, final_dir)
            statuses.append(done)
            del artifacts
            gc.collect()
        except Exception as error:
            failure_dir = assert_write_path(
                FAILURE_ROOT
                / f"seed_{seed}"
                / dataset
                / f"{score}__FAILED_{uuid.uuid4().hex[:8]}"
            )
            failure_dir.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                os.replace(temporary, failure_dir)
            else:
                failure_dir.mkdir(parents=True, exist_ok=False)
            failed = {
                "status": "failed",
                "dataset": dataset,
                "score": score,
                "resplit_seed": seed,
                "runtime_seconds": time.perf_counter() - problem_started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "command": command,
            }
            json_dump(failure_dir / "FAILED.json", failed)
            return {
                "status": "failed",
                "dataset": dataset,
                "seed": seed,
                "failure": failed,
                "completed_scores": statuses,
            }
    return {
        "status": "success",
        "dataset": dataset,
        "seed": seed,
        "scores": statuses,
        "runtime_seconds": time.perf_counter() - started,
    }


def append_log(line: str) -> None:
    path = assert_write_path(RESULT_ROOT / "execution.log")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def append_command() -> None:
    path = assert_write_path(RESULT_ROOT / "exact_commands.txt")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(" ".join(["python", *sys.argv]) + "\n")


def selected_values(values: list[Any] | None, defaults: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(values) if values else tuple(defaults)


def run_batch(args: argparse.Namespace) -> None:
    datasets = selected_values(args.dataset, DATASETS)
    scores = selected_values(args.score, SCORES)
    seeds = selected_values(args.seed, RESPLIT_SEEDS)
    invalid_seeds = set(seeds) - set(RESPLIT_SEEDS)
    if invalid_seeds:
        raise ValueError(f"Seeds outside the prespecified cohort: {sorted(invalid_seeds)}")
    for dataset in datasets:
        validate_pool_content(dataset)
        pool, pool_manifest = load_pool(dataset)
        for seed in seeds:
            load_partition(dataset, seed, pool, pool_manifest)
    tasks = [(dataset, seed) for seed in seeds for dataset in datasets]
    append_command()
    append_log(
        f"RUN_START tasks={len(tasks)} scores={','.join(scores)} jobs={args.jobs} resume={args.resume}"
    )
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    completed = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as executor:
        future_map = {
            executor.submit(run_dataset_seed, dataset, seed, scores, args.resume): (dataset, seed)
            for dataset, seed in tasks
        }
        for future in as_completed(future_map):
            dataset, seed = future_map[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "status": "failed",
                    "dataset": dataset,
                    "seed": seed,
                    "failure": {"error_type": type(error).__name__, "error": str(error)},
                }
            if result["status"] != "success":
                failures.append(result)
                append_log(f"FAILED dataset={dataset} seed={seed} detail={json.dumps(result, default=str)}")
                for pending in future_map:
                    pending.cancel()
                break
            completed += 1
            append_log(
                f"DONE dataset={dataset} seed={seed} task={completed}/{len(tasks)} "
                f"runtime_seconds={result['runtime_seconds']:.6f}"
            )
            print(
                f"DONE {completed}/{len(tasks)} dataset={dataset} seed={seed} "
                f"runtime={result['runtime_seconds']:.1f}s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    append_log(
        f"RUN_END completed_tasks={completed} failures={len(failures)} wall_seconds={elapsed:.6f}"
    )
    if failures:
        json_dump(RESULT_ROOT / "BATCH_FAILURE.json", {"failures": failures, "wall_seconds": elapsed})
        raise RuntimeError(f"Robustness batch stopped after failure: {failures[0]}")
    print(f"Completed {completed} dataset-seed tasks in {elapsed:.1f} seconds")


def status() -> None:
    rows = []
    for seed in RESPLIT_SEEDS:
        for dataset in DATASETS:
            for score in SCORES:
                directory = problem_directory(seed, dataset, score)
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "score": score,
                        "done": (directory / "DONE.json").exists(),
                    }
                )
    frame = pd.DataFrame(rows)
    print(frame.groupby(["dataset", "score"])["done"].sum().to_string())
    print(f"completed={int(frame['done'].sum())}/{len(frame)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw set-size repair over the prespecified 20 resplits."
    )
    parser.add_argument(
        "--stage", choices=("prepare", "run", "status"), required=True
    )
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--score", action="append", choices=SCORES)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--biobias-input-dir", type=Path)
    parser.add_argument("--folktables-a-input-dir", type=Path)
    parser.add_argument("--folktables-b-input-dir", type=Path)
    parser.add_argument("--atus-input-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(args.protocol_root, args.output_root)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.stage == "prepare":
        prepare_protocol(
            args.input_root,
            args.protocol_root,
            args.biobias_input_dir,
            args.folktables_a_input_dir,
            args.folktables_b_input_dir,
            args.atus_input_dir,
        )
    elif args.stage == "run":
        run_batch(args)
    else:
        status()


if __name__ == "__main__":
    main()
