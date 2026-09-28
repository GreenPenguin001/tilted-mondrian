"""Load fixed-model inputs and build deterministic nontraining resplits."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .source_registry import (
    SCORE_FIX_ROOT,
    dataset_spec,
    output_root,
)


COMMON = SCORE_FIX_ROOT / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from input_loading import (  # noqa: E402
    LoadedDataset,
    SplitData,
    load_csv_probability_dataset,
    load_numpy_probability_dataset,
)


POOL_SPLIT_ORDER = ("calibration", "repair", "test")


@dataclass(frozen=True)
class CanonicalData:
    data: LoadedDataset
    row_ids: dict[str, np.ndarray]


@dataclass(frozen=True)
class PoolData:
    probs: np.ndarray
    y: np.ndarray
    g: np.ndarray
    z: np.ndarray
    row_id: np.ndarray
    canonical_origin: np.ndarray

    def take(self, indices: np.ndarray) -> SplitData:
        return SplitData(
            probs=np.asarray(self.probs[indices], dtype=float),
            y=np.asarray(self.y[indices], dtype=int),
            g=np.asarray(self.g[indices], dtype=object),
            z=np.asarray(self.z[indices], dtype=object),
        )


def load_canonical_dataset(dataset: str) -> CanonicalData:
    specification = dataset_spec(dataset)
    if dataset == "biobias":
        data = load_numpy_probability_dataset(
            specification.input_dir,
            dataset_name=dataset,
            calibration_alias="cal",
            intervention_mode="label_defined_z",
            metadata={"fixed_model": True, "local_cells": specification.local_cell_definition},
        )
        row_ids = {}
        for split in POOL_SPLIT_ORDER:
            alias = "cal" if split == "calibration" else split
            identifier_path = specification.input_dir / f"sample_id_{alias}.npy"
            if identifier_path.exists():
                row_ids[split] = np.load(identifier_path, allow_pickle=True).astype(str)
            else:
                row_ids[split] = np.asarray(
                    [f"{dataset}:{split}:{index}" for index in range(len(data.splits[split].y))],
                    dtype=str,
                )
    elif dataset in {"folktables_design_a", "folktables_design_b"}:
        data = load_numpy_probability_dataset(
            specification.input_dir,
            dataset_name=dataset,
            calibration_alias="cal",
            intervention_mode="covariate_defined_z",
            metadata={
                "fixed_model": True,
                "acs_year": 2018,
                "local_cells": specification.local_cell_definition,
            },
        )
        row_ids = {
            split: np.asarray(
                [f"{dataset}:2018:{split}:{index}" for index in range(len(data.splits[split].y))],
                dtype=str,
            )
            for split in POOL_SPLIT_ORDER
        }
    elif dataset == "atus":
        data = load_csv_probability_dataset(
            specification.input_dir,
            dataset_name=dataset,
            split_files={
                "repair": "class_probabilities_repair.csv.gz",
                "calibration": "class_probabilities_calibration.csv.gz",
                "test": "class_probabilities_test.csv.gz",
            },
            label_map_csv="closed_world_topk_label_map.csv",
            intervention_mode="covariate_defined_z",
            group_order=["female", "male"],
            cell_order=["employed", "not_in_labor_force", "unemployed"],
            source_group_column="sex_g",
            source_cell_column="employment_z",
            metadata={"fixed_model": True, "local_cells": specification.local_cell_definition},
        )
        row_ids = {}
        for split in POOL_SPLIT_ORDER:
            frame = pd.read_csv(
                specification.input_dir / f"class_probabilities_{split}.csv.gz",
                usecols=["TUCASEID"],
            )
            row_ids[split] = ("atus:" + frame["TUCASEID"].astype(str)).to_numpy(dtype=str)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for split in POOL_SPLIT_ORDER:
        if len(row_ids[split]) != len(data.splits[split].y):
            raise AssertionError(f"{dataset} {split}: row-ID length mismatch")
    all_ids = np.concatenate([row_ids[split] for split in POOL_SPLIT_ORDER])
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError(f"{dataset}: stable row IDs are not unique")
    return CanonicalData(data=data, row_ids=row_ids)


def concatenate_nontraining(canonical: CanonicalData) -> PoolData:
    pieces = [canonical.data.splits[split] for split in POOL_SPLIT_ORDER]
    return PoolData(
        probs=np.concatenate([piece.probs for piece in pieces], axis=0),
        y=np.concatenate([piece.y for piece in pieces]).astype(int),
        g=np.concatenate([piece.g for piece in pieces]).astype(str),
        z=np.concatenate([piece.z for piece in pieces]).astype(str),
        row_id=np.concatenate([canonical.row_ids[split] for split in POOL_SPLIT_ORDER]).astype(str),
        canonical_origin=np.concatenate(
            [np.full(len(piece.y), split, dtype="U11") for split, piece in zip(POOL_SPLIT_ORDER, pieces)]
        ),
    )


def joint_count_records(pool: PoolData, indices: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {"y": pool.y[indices], "g": pool.g[indices], "z": pool.z[indices]}
    )
    return (
        frame.groupby(["y", "g", "z"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict("records")
    )


def build_pool(dataset: str) -> Path:
    canonical = load_canonical_dataset(dataset)
    pool = concatenate_nontraining(canonical)
    pool_dir = output_root() / "pools" / dataset
    pool_dir.mkdir(parents=True, exist_ok=True)
    np.save(pool_dir / "probs.npy", pool.probs)
    np.save(pool_dir / "y.npy", pool.y)
    np.save(pool_dir / "g.npy", pool.g)
    np.save(pool_dir / "z.npy", pool.z)
    np.save(pool_dir / "row_id.npy", pool.row_id)
    np.save(pool_dir / "canonical_origin.npy", pool.canonical_origin)
    sizes = {
        split: int(len(canonical.data.splits[split].y)) for split in POOL_SPLIT_ORDER
    }
    manifest = {
        "dataset": dataset,
        "pool_order": list(POOL_SPLIT_ORDER),
        "n_pool": int(len(pool.y)),
        "canonical_split_sizes": sizes,
        "train_rows": 0,
        "class_counts": pd.Series(pool.y).value_counts().sort_index().astype(int).to_dict(),
        "group_counts": pd.Series(pool.g).value_counts().sort_index().astype(int).to_dict(),
        "cell_counts": pd.Series(pool.z).value_counts().sort_index().astype(int).to_dict(),
        "joint_y_g_z_counts": joint_count_records(pool, np.arange(len(pool.y))),
        "label_values": canonical.data.label_values,
        "label_names": canonical.data.label_names,
        "group_values": canonical.data.group_values,
        "group_names": canonical.data.group_names,
        "cell_values": canonical.data.cell_values,
        "cell_names": canonical.data.cell_names,
        "intervention_mode": canonical.data.intervention_mode,
        "stable_id_rule": (
            "source sample_id/TUCASEID when available; otherwise "
            "dataset:year:canonical_split:within_split_index"
        ),
    }
    (pool_dir / "pool_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return pool_dir


def load_pool(dataset: str) -> tuple[PoolData, dict[str, Any]]:
    pool_dir = output_root() / "pools" / dataset
    manifest = json.loads((pool_dir / "pool_manifest.json").read_text(encoding="utf-8"))
    pool = PoolData(
        probs=np.load(pool_dir / "probs.npy", mmap_mode="r"),
        y=np.load(pool_dir / "y.npy", mmap_mode="r"),
        g=np.load(pool_dir / "g.npy", mmap_mode="r"),
        z=np.load(pool_dir / "z.npy", mmap_mode="r"),
        row_id=np.load(pool_dir / "row_id.npy", mmap_mode="r"),
        canonical_origin=np.load(pool_dir / "canonical_origin.npy", mmap_mode="r"),
    )
    lengths = {
        len(pool.probs),
        len(pool.y),
        len(pool.g),
        len(pool.z),
        len(pool.row_id),
        len(pool.canonical_origin),
    }
    if lengths != {int(manifest["n_pool"])}:
        raise AssertionError(f"{dataset}: pool arrays are not aligned: {sorted(lengths)}")
    if len(np.unique(np.asarray(pool.row_id).astype(str))) != len(pool.row_id):
        raise AssertionError(f"{dataset}: pool row IDs are not unique")
    return pool, manifest


def safe_strata(labels: np.ndarray, name: str) -> tuple[np.ndarray | None, str]:
    counts = pd.Series(labels).value_counts()
    if len(counts) and int(counts.min()) >= 2 and len(counts) <= max(2, len(labels) // 2):
        return labels, name
    return None, "unstratified_fallback"


def choose_strata(dataset: str, y: np.ndarray, g: np.ndarray, z: np.ndarray) -> tuple[np.ndarray | None, str]:
    if dataset == "biobias":
        strata, name = safe_strata(g.astype(str) + ":" + y.astype(str), "joint_G_Y")
        if strata is not None:
            return strata, name
        return safe_strata(y.astype(str), "Y")
    if dataset == "folktables_design_a":
        strata, name = safe_strata(
            y.astype(str) + ":" + g.astype(str) + ":" + z.astype(str),
            "joint_Y_G_Z",
        )
        if strata is not None:
            return strata, name
        return safe_strata(y.astype(str), "Y")
    if dataset == "folktables_design_b":
        return safe_strata(g.astype(str) + ":" + z.astype(str), "joint_G_Z")
    if dataset == "atus":
        return None, "unstratified_ATUS_respondent_shuffle"
    raise ValueError(f"Unsupported dataset: {dataset}")


def split_pool(
    dataset: str,
    pool: PoolData,
    sizes: dict[str, int],
    split_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    indices = np.arange(len(pool.y))
    if sum(int(sizes[name]) for name in POOL_SPLIT_ORDER) != len(indices):
        raise AssertionError(f"{dataset}: canonical sizes do not sum to pool size")
    fallback: list[dict[str, Any]] = []
    if dataset == "atus":
        shuffled = indices.copy()
        np.random.default_rng(split_seed).shuffle(shuffled)
        cal_end = int(sizes["calibration"])
        repair_end = cal_end + int(sizes["repair"])
        result = {
            "calibration": shuffled[:cal_end],
            "repair": shuffled[cal_end:repair_end],
            "test": shuffled[repair_end:],
        }
        fallback.extend(
            [
                {"stage": "calibration_vs_tail", "stratification": "unstratified_ATUS_respondent_shuffle"},
                {"stage": "repair_vs_test", "stratification": "contiguous_slices_after_single_shuffle"},
            ]
        )
    else:
        strata, strata_name = choose_strata(dataset, pool.y, pool.g, pool.z)
        try:
            calibration, tail = train_test_split(
                indices,
                train_size=int(sizes["calibration"]),
                random_state=split_seed + 1,
                shuffle=True,
                stratify=strata,
            )
        except ValueError as error:
            calibration, tail = train_test_split(
                indices,
                train_size=int(sizes["calibration"]),
                random_state=split_seed + 1,
                shuffle=True,
                stratify=None,
            )
            strata_name = "unstratified_exception_fallback"
            fallback.append({"stage": "calibration_vs_tail", "error": str(error)})
        fallback.append({"stage": "calibration_vs_tail", "stratification": strata_name})
        tail_strata, tail_name = choose_strata(
            dataset, pool.y[tail], pool.g[tail], pool.z[tail]
        )
        try:
            repair, test = train_test_split(
                tail,
                train_size=int(sizes["repair"]),
                random_state=split_seed + 2,
                shuffle=True,
                stratify=tail_strata,
            )
        except ValueError as error:
            repair, test = train_test_split(
                tail,
                train_size=int(sizes["repair"]),
                random_state=split_seed + 2,
                shuffle=True,
                stratify=None,
            )
            tail_name = "unstratified_exception_fallback"
            fallback.append({"stage": "repair_vs_test", "error": str(error)})
        fallback.append({"stage": "repair_vs_test", "stratification": tail_name})
        if dataset == "biobias":
            calibration, repair, test = np.sort(calibration), np.sort(repair), np.sort(test)
        result = {
            "calibration": np.asarray(calibration),
            "repair": np.asarray(repair),
            "test": np.asarray(test),
        }

    union = np.concatenate([result[name] for name in POOL_SPLIT_ORDER])
    if len(np.unique(union)) != len(indices) or set(union.tolist()) != set(indices.tolist()):
        raise AssertionError(f"{dataset} seed={split_seed}: split is not a full disjoint partition")
    for split in POOL_SPLIT_ORDER:
        if len(result[split]) != int(sizes[split]):
            raise AssertionError(f"{dataset} seed={split_seed}: {split} size changed")
    manifest = {
        "dataset": dataset,
        "split_seed": int(split_seed),
        "sizes": {split: int(len(result[split])) for split in POOL_SPLIT_ORDER},
        "train_rows": 0,
        "fallback_trace": fallback,
        "strata_counts": {
            split: joint_count_records(pool, result[split]) for split in POOL_SPLIT_ORDER
        },
        "disjoint": True,
        "union_is_full_pool": True,
        "all_scores_share_partition": True,
    }
    return result, manifest


def loaded_dataset_from_pool(
    dataset: str,
    pool: PoolData,
    split_indices: dict[str, np.ndarray],
    pool_manifest: dict[str, Any],
) -> LoadedDataset:
    return LoadedDataset(
        dataset=dataset,
        splits={split: pool.take(split_indices[split]) for split in POOL_SPLIT_ORDER},
        label_values=[str(value) for value in pool_manifest["label_values"]],
        label_names={str(key): str(value) for key, value in pool_manifest["label_names"].items()},
        group_values=[str(value) for value in pool_manifest["group_values"]],
        group_names={str(key): str(value) for key, value in pool_manifest["group_names"].items()},
        cell_values=[str(value) for value in pool_manifest["cell_values"]],
        cell_names={str(key): str(value) for key, value in pool_manifest["cell_names"].items()},
        intervention_mode=str(pool_manifest["intervention_mode"]),
        source_format="frozen_nontraining_pool",
        metadata={"fixed_model": True},
    )
