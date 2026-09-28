from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitData:
    probs: np.ndarray
    y: np.ndarray
    g: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class LoadedDataset:
    dataset: str
    splits: dict[str, SplitData]
    label_values: list[str]
    label_names: dict[str, str]
    group_values: list[str]
    group_names: dict[str, str]
    cell_values: list[str]
    cell_names: dict[str, str]
    intervention_mode: str
    source_format: str
    metadata: dict[str, Any]


def load_numpy_probability_dataset(
    input_dir: Path,
    dataset_name: str,
    calibration_alias: str = "cal",
    label_map_name: str = "label_maps.json",
    intervention_mode: str = "covariate_defined_z",
    metadata: dict[str, Any] | None = None,
) -> LoadedDataset:
    # Load frozen probabilities together with the reported G and Z encodings.
    input_dir = Path(input_dir)
    label_map = load_json_if_exists(input_dir / label_map_name)
    group_values, group_names = ordered_ids_and_names(label_map.get("group_names") or label_map.get("gender_names"))
    cell_values, cell_names = ordered_ids_and_names(label_map.get("audit_names"))
    if not group_values:
        raise ValueError(f"{dataset_name} label map has no group_names encoding")
    if not cell_values:
        raise ValueError(f"{dataset_name} label map has no audit_names encoding")
    split_aliases = {"repair": "repair", "calibration": calibration_alias, "test": "test"}
    splits: dict[str, SplitData] = {}
    for canonical, alias in split_aliases.items():
        probs = load_required_npy(input_dir / f"probs_{alias}.npy")
        y = load_required_npy(input_dir / f"y_{alias}.npy").astype(int)
        g_raw = load_required_npy(input_dir / f"g_{alias}.npy")
        z_raw = load_required_npy(input_dir / f"z_{alias}.npy")
        if probs.ndim != 2:
            raise ValueError(f"Expected 2D probabilities for {dataset_name} {canonical}; got {probs.shape}")
        if len(y) != probs.shape[0] or len(g_raw) != probs.shape[0] or len(z_raw) != probs.shape[0]:
            raise ValueError(f"Length mismatch in {dataset_name} {canonical}")
        g = canonical_ids(g_raw)
        z = canonical_ids(z_raw)
        assert_values_match_encoding(dataset_name, canonical, "G", g, group_values)
        assert_values_match_encoding(dataset_name, canonical, "Z", z, cell_values)
        if intervention_mode == "label_defined_z":
            y_as_ids = np.asarray([str(int(v)) for v in y], dtype=object)
            if not np.array_equal(z, y_as_ids):
                raise AssertionError(f"{dataset_name} {canonical}: label-defined Z is not exactly equal to Y")
        splits[canonical] = SplitData(probs=np.asarray(probs, dtype=float), y=y, g=g, z=z)

    k = int(next(iter(splits.values())).probs.shape[1])
    label_values = [str(i) for i in range(k)]
    label_names = label_names_from_json(label_map, k)
    if intervention_mode == "label_defined_z" and cell_values != label_values:
        raise AssertionError(
            f"{dataset_name}: label-defined Z order {cell_values} does not match candidate-label order {label_values}"
        )
    return LoadedDataset(
        dataset=dataset_name,
        splits=splits,
        label_values=label_values,
        label_names=label_names,
        group_values=group_values,
        group_names=group_names,
        cell_values=cell_values,
        cell_names=cell_names,
        intervention_mode=intervention_mode,
        source_format="numpy",
        metadata=metadata or {},
    )


def load_csv_probability_dataset(
    input_dir: Path,
    dataset_name: str,
    split_files: dict[str, str],
    label_map_csv: str | None = None,
    intervention_mode: str = "covariate_defined_z",
    group_order: list[str] | None = None,
    cell_order: list[str] | None = None,
    group_names: dict[str, str] | None = None,
    cell_names: dict[str, str] | None = None,
    source_group_column: str | None = None,
    source_cell_column: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LoadedDataset:
    # Load probability exports whose row metadata contains the reported G and Z.
    input_dir = Path(input_dir)
    frames: dict[str, pd.DataFrame] = {}
    prob_cols: list[str] | None = None
    for split, file_name in split_files.items():
        p = input_dir / file_name
        if not p.exists():
            raise FileNotFoundError(f"Missing CSV probability input for {dataset_name} {split}: {p}")
        df = pd.read_csv(p)
        needed = {"true_label", "g", "z"}
        if source_group_column:
            needed.add(source_group_column)
        if source_cell_column:
            needed.add(source_cell_column)
        missing = sorted(needed - set(df.columns))
        if missing:
            raise ValueError(f"{p} is missing required columns: {missing}")
        if source_group_column and not df["g"].astype(str).equals(df[source_group_column].astype(str)):
            raise AssertionError(f"{p}: g does not exactly match source column {source_group_column}")
        if source_cell_column and not df["z"].astype(str).equals(df[source_cell_column].astype(str)):
            raise AssertionError(f"{p}: z does not exactly match source column {source_cell_column}")
        if source_group_column and "group_col" in df and set(df["group_col"].astype(str)) != {source_group_column}:
            raise AssertionError(f"{p}: group_col does not identify {source_group_column}")
        if source_cell_column and "z_col" in df and set(df["z_col"].astype(str)) != {source_cell_column}:
            raise AssertionError(f"{p}: z_col does not identify {source_cell_column}")
        cols = [c for c in df.columns if c.startswith("prob__")]
        if not cols:
            raise ValueError(f"{p} has no prob__ columns")
        if prob_cols is None:
            prob_cols = cols
        elif cols != prob_cols:
            raise ValueError(f"{p} probability columns do not match earlier splits")
        frames[split] = df

    assert prob_cols is not None
    label_values = [c.removeprefix("prob__") for c in prob_cols]
    label_to_idx = {v: i for i, v in enumerate(label_values)}
    label_names = {v: v for v in label_values}
    if label_map_csv:
        label_names.update(label_names_from_csv(input_dir / label_map_csv, label_values))

    splits: dict[str, SplitData] = {}
    for split, df in frames.items():
        y_raw = df["true_label"].astype(str).to_numpy()
        unknown = sorted(set(y_raw) - set(label_to_idx))
        if unknown:
            raise ValueError(f"{dataset_name} {split} has true_label values not present in prob__ columns: {unknown[:10]}")
        y = np.asarray([label_to_idx[v] for v in y_raw], dtype=int)
        g = df["g"].astype(str).to_numpy()
        z = df["z"].astype(str).to_numpy()
        probs = df[prob_cols].to_numpy(dtype=float)
        splits[split] = SplitData(probs=probs, y=y, g=g, z=z)

    observed_groups = sorted({str(v) for split in splits.values() for v in split.g.tolist()})
    observed_cells = sorted({str(v) for split in splits.values() for v in split.z.tolist()})
    group_values = list(group_order) if group_order is not None else observed_groups
    cell_values = list(cell_order) if cell_order is not None else observed_cells
    for split_name, split in splits.items():
        assert_values_match_encoding(dataset_name, split_name, "G", split.g, group_values)
        assert_values_match_encoding(dataset_name, split_name, "Z", split.z, cell_values)
    resolved_group_names = {value: str((group_names or {}).get(value, value)) for value in group_values}
    resolved_cell_names = {value: str((cell_names or {}).get(value, value)) for value in cell_values}
    return LoadedDataset(
        dataset=dataset_name,
        splits=splits,
        label_values=label_values,
        label_names=label_names,
        group_values=group_values,
        group_names=resolved_group_names,
        cell_values=cell_values,
        cell_names=resolved_cell_names,
        intervention_mode=intervention_mode,
        source_format="csv_probabilities",
        metadata=metadata or {},
    )


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def load_required_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required NumPy input: {path}")
    return np.load(path, allow_pickle=True)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def canonical_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [str(int(value)) if is_int_like(value) else str(value) for value in np.asarray(values).tolist()],
        dtype=object,
    )


def ordered_ids_and_names(raw_names: Any) -> tuple[list[str], dict[str, str]]:
    if isinstance(raw_names, list):
        values = [str(i) for i in range(len(raw_names))]
        return values, {str(i): str(value) for i, value in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        values = [str(key) for key in raw_names.keys()]
        return values, {str(key): str(value) for key, value in raw_names.items()}
    return [], {}


def assert_values_match_encoding(
    dataset_name: str,
    split_name: str,
    variable_name: str,
    observed: np.ndarray,
    expected_order: list[str],
) -> None:
    observed_values = set(str(value) for value in np.asarray(observed).tolist())
    expected_values = set(str(value) for value in expected_order)
    if observed_values != expected_values:
        raise AssertionError(
            f"{dataset_name} {split_name}: {variable_name} IDs differ from the frozen encoding; "
            f"observed={sorted(observed_values)}, expected_order={expected_order}"
        )


def label_names_from_json(label_map: dict[str, Any], k: int) -> dict[str, str]:
    names = label_map.get("label_names") or label_map.get("occupation_names") or {}
    if isinstance(names, list):
        return {str(i): str(names[i]) for i in range(min(k, len(names)))}
    if isinstance(names, dict):
        return {str(i): str(names.get(str(i), names.get(i, f"label_{i}"))) for i in range(k)}
    return {str(i): f"label_{i}" for i in range(k)}


def label_names_from_csv(path: Path, label_values: list[str]) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if {"y_label", "tier2_name"}.issubset(df.columns):
        return {str(r["y_label"]): str(r["tier2_name"]) for _, r in df.iterrows()}
    if {"label", "label_name"}.issubset(df.columns):
        return {str(r["label"]): str(r["label_name"]) for _, r in df.iterrows()}
    if {"label_id", "label_name"}.issubset(df.columns):
        return {str(r["label_id"]): str(r["label_name"]) for _, r in df.iterrows()}
    return {v: v for v in label_values}


def is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False
