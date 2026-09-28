"""Load frozen non-training predictions and create deterministic resplit pools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .source_registry import (
    FOLKTABLES_YEARS,
    atus_probability_dir,
    biobias_probability_dir,
    folktables_design_dir,
    output_root,
)


CANONICAL_SPLITS = ("cal", "repair", "test")


@dataclass(frozen=True)
class DatasetUnit:
    dataset: str
    year: int | None

    @property
    def key(self) -> str:
        return self.dataset if self.year is None else f"{self.dataset}_{self.year}"


@dataclass
class SplitData:
    probs: np.ndarray
    y: np.ndarray
    g: np.ndarray
    z: np.ndarray
    row_id: np.ndarray

    def take(self, indices: np.ndarray) -> "SplitData":
        return SplitData(
            probs=self.probs[indices],
            y=self.y[indices],
            g=self.g[indices],
            z=self.z[indices],
            row_id=self.row_id[indices],
        )

    def as_original_dict(self) -> dict[str, np.ndarray]:
        return {"probs": self.probs, "y": self.y, "g": self.g, "z": self.z}


@dataclass
class LoadedUnit:
    unit: DatasetUnit
    splits: dict[str, SplitData]
    label_names: list[str]
    group_names: list[str]
    z_names: list[str]
    metadata: dict[str, Any]


def dataset_units() -> list[DatasetUnit]:
    units = [DatasetUnit("biobias", None)]
    units.extend(DatasetUnit("folktables_design_a", year) for year in FOLKTABLES_YEARS)
    units.extend(DatasetUnit("folktables_design_b", year) for year in FOLKTABLES_YEARS)
    units.append(DatasetUnit("atus", None))
    return units


def scores_for_unit(unit: DatasetUnit) -> tuple[str, ...]:
    if unit.dataset == "biobias":
        return ("simple", "aps", "saps", "raps")
    if unit.dataset.startswith("folktables"):
        return ("simple",)
    return ("simple", "aps", "raps", "saps")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def map_names(mapping: dict[str, Any], fallback_count: int) -> list[str]:
    if isinstance(mapping, list):
        return [str(value) for value in mapping]
    if mapping:
        return [str(mapping.get(str(index), mapping.get(index, index))) for index in range(fallback_count)]
    return [str(index) for index in range(fallback_count)]


def load_biobias() -> LoadedUnit:
    probability_dir = biobias_probability_dir()
    meta = load_json(probability_dir / "label_maps.json")
    splits: dict[str, SplitData] = {}
    for split in CANONICAL_SPLITS:
        splits[split] = SplitData(
            probs=np.load(probability_dir / f"probs_{split}.npy"),
            y=np.load(probability_dir / f"y_{split}.npy").astype(int),
            g=np.load(probability_dir / f"g_{split}.npy").astype(int),
            z=np.load(probability_dir / f"z_{split}.npy").astype(int),
            row_id=np.load(probability_dir / f"sample_id_{split}.npy", allow_pickle=True).astype(str),
        )
    return LoadedUnit(
        unit=DatasetUnit("biobias", None),
        splits=splits,
        label_names=[str(value) for value in meta["label_names"]],
        group_names=[str(value) for value in meta["group_names"]],
        z_names=[str(value) for value in meta["audit_names"]],
        metadata=meta,
    )


def load_folktables(unit: DatasetUnit) -> LoadedUnit:
    if unit.year is None:
        raise ValueError("Folktables units require a year")
    train_dir = folktables_design_dir(unit.dataset) / str(unit.year) / "train"
    meta = load_json(train_dir / "label_maps.json")
    splits: dict[str, SplitData] = {}
    for split in CANONICAL_SPLITS:
        y = np.load(train_dir / f"y_{split}.npy").astype(int)
        row_id = np.asarray(
            [f"{unit.dataset}:{unit.year}:{split}:{index}" for index in range(len(y))],
            dtype=str,
        )
        splits[split] = SplitData(
            probs=np.load(train_dir / f"probs_{split}.npy"),
            y=y,
            g=np.load(train_dir / f"g_{split}.npy").astype(int),
            z=np.load(train_dir / f"z_{split}.npy").astype(int),
            row_id=row_id,
        )
    n_labels = splits["cal"].probs.shape[1]
    n_groups = len(np.unique(np.concatenate([split.g for split in splits.values()])))
    n_z = len(np.unique(np.concatenate([split.z for split in splits.values()])))
    return LoadedUnit(
        unit=unit,
        splits=splits,
        label_names=map_names(meta.get("label_names", {}), n_labels),
        group_names=map_names(meta.get("group_names", {}), n_groups),
        z_names=map_names(meta.get("audit_names", {}), n_z),
        metadata=meta,
    )


def load_atus() -> LoadedUnit:
    result_dir = atus_probability_dir()
    splits: dict[str, SplitData] = {}
    label_names: list[str] | None = None
    for canonical_name, source_name in (("cal", "calibration"), ("repair", "repair"), ("test", "test")):
        frame = pd.read_csv(result_dir / f"class_probabilities_{source_name}.csv.gz")
        probability_columns = [column for column in frame.columns if column.startswith("prob__")]
        current_labels = [column.removeprefix("prob__") for column in probability_columns]
        if label_names is None:
            label_names = current_labels
        elif current_labels != label_names:
            raise ValueError(f"ATUS probability columns differ for split={source_name}")
        label_to_id = {label: index for index, label in enumerate(label_names)}
        unknown = sorted(set(frame["true_label"].astype(str)) - set(label_to_id))
        if unknown:
            raise ValueError(f"ATUS true labels are absent from probability columns: {unknown}")
        splits[canonical_name] = SplitData(
            probs=frame[probability_columns].to_numpy(dtype=float),
            y=frame["true_label"].astype(str).map(label_to_id).to_numpy(dtype=int),
            g=frame["g"].astype(str).to_numpy(dtype=str),
            z=frame["z"].astype(str).to_numpy(dtype=str),
            row_id=("atus:" + frame["TUCASEID"].astype(str)).to_numpy(dtype=str),
        )
    groups = sorted(set(np.concatenate([split.g for split in splits.values()]).astype(str)))
    z_values = sorted(set(np.concatenate([split.z for split in splits.values()]).astype(str)))
    return LoadedUnit(
        unit=DatasetUnit("atus", None),
        splits=splits,
        label_names=label_names or [],
        group_names=groups,
        z_names=z_values,
        metadata={
            "group_column": "sex_g",
            "z_column": "employment_z",
            "group_order": groups,
            "z_order": z_values,
        },
    )


def load_source_unit(unit: DatasetUnit) -> LoadedUnit:
    if unit.dataset == "biobias":
        return load_biobias()
    if unit.dataset.startswith("folktables"):
        return load_folktables(unit)
    if unit.dataset == "atus":
        return load_atus()
    raise ValueError(f"Unsupported dataset: {unit.dataset}")


def concatenate_splits(loaded: LoadedUnit) -> tuple[SplitData, np.ndarray, np.ndarray]:
    pieces = [loaded.splits[name] for name in CANONICAL_SPLITS]
    origin = np.concatenate(
        [np.full(len(piece.y), index, dtype=np.int8) for index, piece in enumerate(pieces)]
    )
    within_split_index = np.concatenate([np.arange(len(piece.y), dtype=np.int64) for piece in pieces])
    pool = SplitData(
        probs=np.concatenate([piece.probs for piece in pieces], axis=0),
        y=np.concatenate([piece.y for piece in pieces]),
        g=np.concatenate([piece.g for piece in pieces]),
        z=np.concatenate([piece.z for piece in pieces]),
        row_id=np.concatenate([piece.row_id for piece in pieces]).astype(str),
    )
    if len(set(pool.row_id.tolist())) != len(pool.row_id):
        raise AssertionError(f"Stable row IDs are not unique for {loaded.unit.key}")
    return pool, origin, within_split_index


def count_table(pool: SplitData, indices: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"y": pool.y[indices], "g": pool.g[indices], "z": pool.z[indices]})
    return frame.groupby(["y", "g", "z"], dropna=False).size().reset_index(name="n").to_dict("records")


def build_pool(loaded: LoadedUnit) -> Path:
    pool, origin, within_split_index = concatenate_splits(loaded)
    pool_dir = output_root() / "pools" / loaded.unit.key
    pool_dir.mkdir(parents=True, exist_ok=True)
    np.save(pool_dir / "probs.npy", pool.probs)
    np.save(pool_dir / "y.npy", pool.y)
    np.save(pool_dir / "g.npy", pool.g)
    np.save(pool_dir / "z.npy", pool.z)
    np.save(pool_dir / "row_id.npy", pool.row_id)
    np.save(pool_dir / "canonical_origin.npy", origin)
    np.save(pool_dir / "within_canonical_split_index.npy", within_split_index)
    sizes = {name: len(loaded.splits[name].y) for name in CANONICAL_SPLITS}
    manifest = {
        "dataset": loaded.unit.dataset,
        "year": loaded.unit.year,
        "pool_order": list(CANONICAL_SPLITS),
        "n_pool": len(pool.y),
        "canonical_split_sizes": sizes,
        "train_rows": 0,
        "class_counts": pd.Series(pool.y).value_counts().sort_index().astype(int).to_dict(),
        "group_counts": pd.Series(pool.g).value_counts().sort_index().astype(int).to_dict(),
        "profile_counts": pd.Series(pool.z).value_counts().sort_index().astype(int).to_dict(),
        "joint_y_g_z_counts": count_table(pool, np.arange(len(pool.y))),
        "label_names": loaded.label_names,
        "group_names": loaded.group_names,
        "z_names": loaded.z_names,
        "stable_id_rule": "source sample/TUCASEID when available; otherwise dataset:year:canonical_split:within_split_index",
    }
    (pool_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return pool_dir


def load_pool(unit: DatasetUnit) -> tuple[SplitData, dict[str, Any]]:
    pool_dir = output_root() / "pools" / unit.key
    manifest = load_json(pool_dir / "pool_manifest.json")
    pool = SplitData(
        probs=np.load(pool_dir / "probs.npy", mmap_mode="r"),
        y=np.load(pool_dir / "y.npy", mmap_mode="r"),
        g=np.load(pool_dir / "g.npy", mmap_mode="r"),
        z=np.load(pool_dir / "z.npy", mmap_mode="r"),
        row_id=np.load(pool_dir / "row_id.npy", mmap_mode="r"),
    )
    lengths = {len(pool.probs), len(pool.y), len(pool.g), len(pool.z), len(pool.row_id)}
    if lengths != {int(manifest["n_pool"])}:
        raise AssertionError(f"Pool arrays are not aligned for {unit.key}: {sorted(lengths)}")
    if len(np.unique(np.asarray(pool.row_id).astype(str))) != len(pool.row_id):
        raise AssertionError(f"Pool row IDs are not unique for {unit.key}")
    return pool, manifest


def choose_strata(unit: DatasetUnit, y: np.ndarray, g: np.ndarray, z: np.ndarray) -> tuple[np.ndarray | None, str]:
    if unit.dataset == "biobias":
        candidates = ((g.astype(str) + ":" + y.astype(str), "joint_G_Y"), (y.astype(str), "Y"))
    elif unit.dataset == "folktables_design_a":
        candidates = ((y.astype(str) + ":" + g.astype(str) + ":" + z.astype(str), "joint_Y_G_Z"), (y.astype(str), "Y"))
    elif unit.dataset == "folktables_design_b":
        candidates = ((g.astype(str) + ":" + z.astype(str), "joint_G_Z"),)
    else:
        return None, "unstratified_ATUS_respondent_shuffle"
    for labels, name in candidates:
        counts = pd.Series(labels).value_counts()
        if len(counts) and int(counts.min()) >= 2 and len(counts) <= max(2, len(labels) // 2):
            return labels, name
    return None, "unstratified_fallback"


def split_pool(unit: DatasetUnit, pool: SplitData, sizes: dict[str, int], seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = len(pool.y)
    expected = int(sizes["cal"] + sizes["repair"] + sizes["test"])
    if n != expected:
        raise AssertionError(f"Pool size {n} differs from canonical non-training total {expected}")
    indices = np.arange(n)
    fallback: list[dict[str, Any]] = []
    if unit.dataset == "atus":
        shuffled = indices.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        cal_idx = shuffled[: sizes["cal"]]
        repair_idx = shuffled[sizes["cal"] : sizes["cal"] + sizes["repair"]]
        test_idx = shuffled[sizes["cal"] + sizes["repair"] :]
        fallback.append({"stage": "cal_vs_tail", "stratification": "unstratified_ATUS_respondent_shuffle"})
        fallback.append({"stage": "repair_vs_test", "stratification": "contiguous_slices_after_single_shuffle"})
    else:
        strata, strata_name = choose_strata(unit, pool.y, pool.g, pool.z)
        try:
            cal_idx, tail_idx = train_test_split(
                indices,
                train_size=int(sizes["cal"]),
                random_state=seed + 1,
                shuffle=True,
                stratify=strata,
            )
        except ValueError as error:
            cal_idx, tail_idx = train_test_split(
                indices, train_size=int(sizes["cal"]), random_state=seed + 1, shuffle=True, stratify=None
            )
            strata_name = "unstratified_exception_fallback"
            fallback.append({"stage": "cal_vs_tail", "error": str(error)})
        fallback.append({"stage": "cal_vs_tail", "stratification": strata_name})
        tail_strata, tail_name = choose_strata(unit, pool.y[tail_idx], pool.g[tail_idx], pool.z[tail_idx])
        try:
            repair_idx, test_idx = train_test_split(
                tail_idx,
                train_size=int(sizes["repair"]),
                random_state=seed + 2,
                shuffle=True,
                stratify=tail_strata,
            )
        except ValueError as error:
            repair_idx, test_idx = train_test_split(
                tail_idx, train_size=int(sizes["repair"]), random_state=seed + 2, shuffle=True, stratify=None
            )
            tail_name = "unstratified_exception_fallback"
            fallback.append({"stage": "repair_vs_test", "error": str(error)})
        fallback.append({"stage": "repair_vs_test", "stratification": tail_name})
        if unit.dataset == "biobias":
            cal_idx, repair_idx, test_idx = np.sort(cal_idx), np.sort(repair_idx), np.sort(test_idx)

    result = {"cal": np.asarray(cal_idx), "repair": np.asarray(repair_idx), "test": np.asarray(test_idx)}
    union = np.concatenate(list(result.values()))
    if len(np.unique(union)) != n or set(union.tolist()) != set(indices.tolist()):
        raise AssertionError(f"Resplit is not a disjoint partition for {unit.key}, seed={seed}")
    if any(len(result[name]) != int(sizes[name]) for name in CANONICAL_SPLITS):
        raise AssertionError(f"Resplit sizes differ from canonical sizes for {unit.key}, seed={seed}")
    manifest = {
        "dataset": unit.dataset,
        "year": unit.year,
        "split_seed": int(seed),
        "sizes": {name: len(result[name]) for name in CANONICAL_SPLITS},
        "train_rows": 0,
        "fallback_trace": fallback,
        "strata_counts": {name: count_table(pool, idx) for name, idx in result.items()},
        "disjoint": True,
        "union_is_full_pool": True,
    }
    return result, manifest
