#!/usr/bin/env python3
"""Stage 1: build Design A Folktables data, train classifier, export artifacts."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from curve_utils import jdump, mkdir, read_yaml, write_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "design_config.yaml"
FOLKTABLES_ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "PR",
]
CORE_LEAKAGE_COLUMNS = {"OCCP", "SOCP", "FOCCP", "SEX", "FSEXP", "FSCHLP"}


def resolve_path(path: str | Path, base: Path = SCRIPT_DIR) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--z-in-x", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--density", type=float, default=None)
    p.add_argument("--max_rows", type=int, default=None, help="Optional row cap after basic validity filtering.")
    p.add_argument("--max_final_rows", type=int, default=None, help="Optional post-topK resource cap applied before splitting.")
    p.add_argument("--cap_seed", type=int, default=0, help="Random seed for --max_final_rows subsampling.")
    p.add_argument("--cap_strategy", default="stratified_y_g_z", choices=["stratified_y_g_z"], help="Post-topK cap strategy.")
    p.add_argument("--random-seed", type=int, default=None)
    return p.parse_args()


def education_bucket(schl: pd.Series, buckets: list[dict[str, Any]]) -> pd.Series:
    out = pd.Series(np.nan, index=schl.index, dtype="float64")
    s = pd.to_numeric(schl, errors="coerce")
    for bucket in buckets:
        mask = s.between(float(bucket["min_schl"]), float(bucket["max_schl"]), inclusive="both")
        out.loc[mask] = int(bucket["id"])
    return out


def total_variation_z_by_group(z: np.ndarray, g: np.ndarray, audit_ids: list[int]) -> dict[str, Any]:
    groups = sorted(int(x) for x in np.unique(g))
    out: dict[str, Any] = {"groups": groups, "audit_ids": audit_ids}
    dists: dict[int, np.ndarray] = {}
    for group in groups:
        zz = z[g == group]
        denom = max(1, len(zz))
        dist = np.asarray([np.sum(zz == a) / denom for a in audit_ids], dtype=float)
        dists[group] = dist
        out[f"p_z_given_group_{group}"] = {str(a): float(v) for a, v in zip(audit_ids, dist)}
    if len(groups) == 2:
        out["tv"] = float(0.5 * np.abs(dists[groups[0]] - dists[groups[1]]).sum())
    return out


def cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    table = pd.crosstab(pd.Series(x), pd.Series(y))
    if table.empty:
        return float("nan")
    observed = table.to_numpy(float)
    n = float(observed.sum())
    if n <= 0:
        return float("nan")
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / n
    mask = expected > 0
    chi2 = float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    r, k = observed.shape
    denom = n * max(1, min(r - 1, k - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def safe_joint_strata(y: np.ndarray, g: np.ndarray, z: np.ndarray) -> np.ndarray | None:
    labels = np.asarray([f"{int(yy)}:{int(gg)}:{int(zz)}" for yy, gg, zz in zip(y, g, z)], dtype=object)
    counts = Counter(labels)
    if counts and min(counts.values()) >= 2 and len(counts) <= max(2, len(labels) // 2):
        return labels
    y_labels = np.asarray([str(int(yy)) for yy in y], dtype=object)
    y_counts = Counter(y_labels)
    if y_counts and min(y_counts.values()) >= 2:
        return y_labels
    return None


def stratified_final_cap_indices(
    y: np.ndarray,
    g: np.ndarray,
    z: np.ndarray,
    max_final_rows: int | None,
    seed: int,
    strategy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(len(y))
    meta: dict[str, Any] = {
        "cap_applied": False,
        "cap_strategy": strategy,
        "cap_seed": int(seed),
        "max_final_rows": int(max_final_rows) if max_final_rows is not None else None,
        "n_after_topk_before_cap": n,
        "n_after_cap": n,
    }
    if max_final_rows is None or int(max_final_rows) <= 0 or n <= int(max_final_rows):
        meta["cap_reason"] = "not_requested_or_not_needed"
        return np.arange(n), meta
    if strategy != "stratified_y_g_z":
        raise ValueError(f"Unsupported cap strategy: {strategy}")

    cap = int(max_final_rows)
    labels = np.asarray([f"{int(yy)}:{int(gg)}:{int(zz)}" for yy, gg, zz in zip(y, g, z)], dtype=object)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_strata = int(len(unique))
    if cap < n_strata:
        raise ValueError(
            f"max_final_rows={cap} is smaller than the number of nonempty Y/G/Z strata ({n_strata}); "
            "increase the cap to preserve every nonempty stratum."
        )

    expected = counts.astype(float) * (float(cap) / float(n))
    allocation = np.floor(expected).astype(int)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts)

    while int(allocation.sum()) > cap:
        removable = np.where(allocation > 1)[0]
        if removable.size == 0:
            break
        priority = expected[removable] - allocation[removable]
        j = int(removable[np.argmin(priority)])
        allocation[j] -= 1

    remaining = cap - int(allocation.sum())
    if remaining > 0:
        remainders = expected - np.floor(expected)
        order = np.lexsort((-counts, -remainders))
        while remaining > 0:
            changed = False
            for j in order:
                if allocation[j] < counts[j]:
                    allocation[j] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for j in range(n_strata):
        idx = np.flatnonzero(inverse == j)
        take = int(allocation[j])
        selected.append(rng.choice(idx, size=take, replace=False))
    selected_idx = np.sort(np.concatenate(selected).astype(int))
    if int(len(selected_idx)) != cap:
        raise RuntimeError(f"Final cap allocation selected {len(selected_idx)} rows, expected {cap}")

    retention = allocation.astype(float) / counts.astype(float)
    meta.update({
        "cap_applied": True,
        "cap_reason": "post_topk_resource_cap",
        "n_after_cap": int(len(selected_idx)),
        "cap_strata_count": n_strata,
        "cap_min_stratum_count_before": int(counts.min()),
        "cap_max_stratum_count_before": int(counts.max()),
        "cap_min_stratum_count_after": int(allocation.min()),
        "cap_max_stratum_count_after": int(allocation.max()),
        "cap_min_stratum_retention": float(retention.min()),
        "cap_max_stratum_retention": float(retention.max()),
        "cap_mean_stratum_retention": float(retention.mean()),
    })
    return selected_idx, meta


def train_cal_repair_test_split(y: np.ndarray, g: np.ndarray, z: np.ndarray, ratios: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    n = len(y)
    idx = np.arange(n)
    train_size = float(ratios["train"])
    cal_size = float(ratios["cal"])
    repair_size = float(ratios["repair"])
    test_size = float(ratios["test"])
    total = train_size + cal_size + repair_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1, got {total}")

    strata = safe_joint_strata(y, g, z)
    try:
        train_idx, rest_idx = train_test_split(idx, train_size=train_size, random_state=seed, shuffle=True, stratify=strata)
    except ValueError:
        train_idx, rest_idx = train_test_split(idx, train_size=train_size, random_state=seed, shuffle=True, stratify=None)

    rest_strata = safe_joint_strata(y[rest_idx], g[rest_idx], z[rest_idx])
    cal_frac_of_rest = cal_size / (1.0 - train_size)
    try:
        cal_idx, tail_idx = train_test_split(rest_idx, train_size=cal_frac_of_rest, random_state=seed + 1, shuffle=True, stratify=rest_strata)
    except ValueError:
        cal_idx, tail_idx = train_test_split(rest_idx, train_size=cal_frac_of_rest, random_state=seed + 1, shuffle=True, stratify=None)

    tail_strata = safe_joint_strata(y[tail_idx], g[tail_idx], z[tail_idx])
    repair_frac_of_tail = repair_size / (repair_size + test_size)
    try:
        repair_idx, test_idx = train_test_split(tail_idx, train_size=repair_frac_of_tail, random_state=seed + 2, shuffle=True, stratify=tail_strata)
    except ValueError:
        repair_idx, test_idx = train_test_split(tail_idx, train_size=repair_frac_of_tail, random_state=seed + 2, shuffle=True, stratify=None)

    return {"train": train_idx, "cal": cal_idx, "repair": repair_idx, "test": test_idx}


def select_feature_columns(df: pd.DataFrame, cfg: dict[str, Any], z_in_x: bool) -> tuple[list[str], list[str], dict[str, str]]:
    configured = set(str(x) for x in cfg.get("features", {}).get("excluded_columns", []))
    excluded = set(configured) | set(CORE_LEAKAGE_COLUMNS)
    if not z_in_x:
        excluded.add("SCHL")
    prefixes = tuple(str(x) for x in cfg.get("features", {}).get("exclude_weight_prefixes", []))
    cols: list[str] = []
    reasons: dict[str, str] = {}
    for col in df.columns:
        col_s = str(col)
        if col_s in excluded:
            if col_s in CORE_LEAKAGE_COLUMNS:
                reasons[col_s] = "leakage_or_target_or_group"
            elif col_s == "SCHL" and not z_in_x:
                reasons[col_s] = "z_in_x_false"
            else:
                reasons[col_s] = "configured_exclusion"
            continue
        if prefixes and any(col_s.startswith(prefix) for prefix in prefixes):
            reasons[col_s] = "weight_or_replicate_weight_prefix"
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col_s)
        else:
            reasons[col_s] = "non_numeric"
    excluded_cols = [str(c) for c in df.columns if str(c) not in cols]
    return cols, excluded_cols, reasons


def align_predict_proba(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    out = np.divide(out, row_sum, out=np.full_like(out, 1.0 / n_classes), where=row_sum > 0)
    return out


def metrics(y: np.ndarray, probs: np.ndarray, split_name: str) -> dict[str, Any]:
    pred = np.argmax(probs, axis=1)
    out = {
        "split": split_name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
    }
    if probs.shape[1] >= 3:
        top3 = np.argsort(-probs, axis=1)[:, :3]
        out["top3_accuracy"] = float(np.mean([yy in row for yy, row in zip(y, top3)])) if len(y) else float("nan")
    try:
        out["log_loss"] = float(log_loss(y, np.clip(probs, 1e-12, 1.0), labels=list(range(probs.shape[1]))))
    except ValueError:
        out["log_loss"] = float("nan")
    return out


def load_acs_all_states(year: int, root_dir: Path, density: float, download: bool, seed: int) -> tuple[pd.DataFrame, list[str]]:
    from folktables import ACSDataSource

    ds = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=str(root_dir))
    states = list(FOLKTABLES_ALL_STATES)
    df = ds.get_data(states=states, density=float(density), random_seed=int(seed), download=bool(download))
    return df, states


def valid_design_vectors(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    required = ["OCCP", "SEX", "SCHL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required ACS columns: {missing}")
    occp = pd.to_numeric(df["OCCP"], errors="coerce")
    sex = pd.to_numeric(df["SEX"], errors="coerce")
    schl = pd.to_numeric(df["SCHL"], errors="coerce")
    z_bucket = education_bucket(schl, cfg["task"]["education_buckets"])
    valid = occp.notna() & (occp > 0) & sex.isin([1, 2]) & z_bucket.notna()
    return occp, sex, z_bucket, valid


def load_topk_design_frame_all_states(
    year: int,
    root_dir: Path,
    density: float,
    download: bool,
    seed: int,
    cfg: dict[str, Any],
    top_k: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[int], list[str], list[int], int, int]:
    """Load all states while keeping only top-K valid rows in memory.

    Pass 1 scans every requested state for valid OCCP counts. Pass 2 reloads
    every state and retains only rows whose OCCP is in the all-state top-K.
    This preserves the all-state target definition while avoiding a giant raw
    ACS dataframe in memory.
    """
    from folktables import ACSDataSource

    ds = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=str(root_dir))
    states = list(FOLKTABLES_ALL_STATES)
    states_loaded: list[str] = []
    state_codes_observed: set[int] = set()
    n_raw_before_filter = 0
    n_after_valid_filter = 0
    occp_counter: Counter[int] = Counter()

    for state in states:
        sdf = ds.get_data(states=[state], density=float(density), random_seed=int(seed), download=bool(download))
        states_loaded.append(state)
        n_raw_before_filter += int(len(sdf))
        occp, sex, z_bucket, valid = valid_design_vectors(sdf, cfg)
        n_after_valid_filter += int(valid.sum())
        if "ST" in sdf.columns:
            state_codes_observed.update(int(x) for x in pd.to_numeric(sdf.loc[valid, "ST"], errors="coerce").dropna().unique())
        occp_counter.update(int(x) for x in occp.loc[valid].astype(int).to_numpy())

    top_counts = pd.Series(occp_counter, dtype="int64").sort_values(ascending=False)
    top_labels = sorted([int(x) for x in top_counts.head(top_k).index])
    if len(top_labels) < top_k:
        raise ValueError(f"Only found {len(top_labels)} valid OCCP labels, requested top_k={top_k}")

    frames: list[pd.DataFrame] = []
    for state in states:
        sdf = ds.get_data(states=[state], density=float(density), random_seed=int(seed), download=bool(download))
        occp, sex, z_bucket, valid = valid_design_vectors(sdf, cfg)
        keep = valid & occp.isin(top_labels)
        if not bool(keep.any()):
            continue
        frame = sdf.loc[keep].copy()
        frame["_design_occp"] = occp.loc[keep].astype(int).to_numpy()
        frame["_design_sex"] = sex.loc[keep].astype(int).to_numpy()
        frame["_design_z_bucket"] = z_bucket.loc[keep].astype(int).to_numpy()
        frames.append(frame)
    if not frames:
        raise ValueError("No rows remain after all-state top-K filtering")

    df = pd.concat(frames, ignore_index=True)
    occp = df.pop("_design_occp").astype(int)
    sex = df.pop("_design_sex").astype(int)
    z_bucket = df.pop("_design_z_bucket").astype(int)
    return (
        df.reset_index(drop=True),
        occp.reset_index(drop=True),
        sex.reset_index(drop=True),
        z_bucket.reset_index(drop=True),
        top_labels,
        states_loaded,
        sorted(state_codes_observed),
        n_raw_before_filter,
        n_after_valid_filter,
    )


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    start = time.time()

    year = int(args.year)
    top_k = int(args.top_k if args.top_k is not None else cfg["task"].get("top_k", 10))
    z_in_x = bool(args.z_in_x if args.z_in_x is not None else cfg["task"].get("z_in_x", True))
    seed = int(args.random_seed if args.random_seed is not None else cfg.get("random_seed", 20260609))
    density = float(args.density if args.density is not None else cfg.get("acs", {}).get("density", 1.0))
    download = bool(args.download if args.download is not None else cfg.get("acs", {}).get("download", True))
    data_root = resolve_path(args.data_root if args.data_root is not None else cfg.get("acs", {}).get("data_root", "../data/folktables_acs"))
    out_root = resolve_path(args.output_root)
    out_dir = mkdir(out_root / str(year) / "train")

    print(
        f"Loading ACS {year} explicit all-state list from {data_root} "
        f"(download={download}, density={density}) with state-wise topK scan",
        flush=True,
    )
    (
        df,
        occp,
        sex,
        z_bucket,
        top_labels,
        states_loaded,
        state_codes_observed,
        n_raw_before_filter,
        n_after_valid_filter,
    ) = load_topk_design_frame_all_states(year, data_root, density, download, seed, cfg, top_k)
    n_after_valid_filter_after_max_rows = n_after_valid_filter
    if args.max_rows is not None and int(args.max_rows) > 0 and len(df) > int(args.max_rows):
        sample_idx = np.random.default_rng(seed).choice(len(df), size=int(args.max_rows), replace=False)
        df = df.iloc[sample_idx].reset_index(drop=True)
        occp = occp.iloc[sample_idx].reset_index(drop=True)
        sex = sex.iloc[sample_idx].reset_index(drop=True)
        z_bucket = z_bucket.iloc[sample_idx].reset_index(drop=True)
    n_after_topk_before_cap = int(len(df))
    if len(df) < 100:
        raise ValueError(f"Too few rows after filtering/top-K selection: n={len(df)}")

    old_to_new = {int(old): new for new, old in enumerate(top_labels)}
    y = occp.map(old_to_new).to_numpy(dtype=int)
    g = sex.map({1: 0, 2: 1}).to_numpy(dtype=int)
    z = z_bucket.to_numpy(dtype=int)

    cap_idx, cap_meta = stratified_final_cap_indices(
        y,
        g,
        z,
        max_final_rows=args.max_final_rows,
        seed=int(args.cap_seed),
        strategy=str(args.cap_strategy),
    )
    if cap_meta["cap_applied"]:
        df = df.iloc[cap_idx].reset_index(drop=True)
        occp = occp.iloc[cap_idx].reset_index(drop=True)
        sex = sex.iloc[cap_idx].reset_index(drop=True)
        z_bucket = z_bucket.iloc[cap_idx].reset_index(drop=True)
        y = y[cap_idx]
        g = g[cap_idx]
        z = z[cap_idx]
    n_after_cap = int(len(df))
    n_after_topk = n_after_cap
    audit_ids = sorted(int(x) for x in np.unique(z))
    audit_names = {
        str(int(b["id"])): str(b["name"])
        for b in cfg["task"]["education_buckets"]
        if int(b["id"]) in audit_ids
    }
    label_names = {str(v): f"OCCP_{k}" for k, v in old_to_new.items()}
    label_maps = {
        "label_source": "OCCP",
        "label_names": label_names,
        "label_old_to_new": {str(k): int(v) for k, v in old_to_new.items()},
        "label_new_to_old": {str(v): int(k) for k, v in old_to_new.items()},
        "group_source": "SEX",
        "group_names": {"0": "Male", "1": "Female"},
        "group_raw_to_new": {"1": 0, "2": 1},
        "audit_source": "SCHL_education_bucket",
        "audit_names": audit_names,
    }

    feature_cols, excluded_feature_cols, exclusion_reasons = select_feature_columns(df, cfg, z_in_x)
    if not feature_cols:
        raise ValueError("No numeric ACS feature columns selected")
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X_df.to_numpy(dtype=float)

    splits = train_cal_repair_test_split(y, g, z, cfg["splits"], seed)
    model_cfg = cfg.get("model", {})
    model = HistGradientBoostingClassifier(
        max_iter=int(model_cfg.get("max_iter", 160)),
        learning_rate=float(model_cfg.get("learning_rate", 0.06)),
        max_leaf_nodes=int(model_cfg.get("max_leaf_nodes", 31)),
        l2_regularization=float(model_cfg.get("l2_regularization", 0.01)),
        random_state=seed,
    )
    print(f"Training {model.__class__.__name__} on {len(splits['train'])} rows, {X.shape[1]} features", flush=True)
    model.fit(X[splits["train"]], y[splits["train"]])

    probs_by_split: dict[str, np.ndarray] = {}
    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split_name, idx in splits.items():
        probs = align_predict_proba(model, X[idx], top_k)
        probs_by_split[split_name] = probs
        np.save(out_dir / f"probs_{split_name}.npy", probs)
        np.save(out_dir / f"y_{split_name}.npy", y[idx].astype(np.int64))
        np.save(out_dir / f"g_{split_name}.npy", g[idx].astype(np.int64))
        np.save(out_dir / f"z_{split_name}.npy", z[idx].astype(np.int64))
        metrics_by_split[split_name] = metrics(y[idx], probs, split_name)

    cell_rows = []
    for split_name, idx in splits.items():
        for group in sorted(np.unique(g)):
            for audit in audit_ids:
                cell_rows.append({
                    "split": split_name,
                    "group": int(group),
                    "group_name": label_maps["group_names"][str(int(group))],
                    "audit": int(audit),
                    "audit_name": audit_names.get(str(int(audit)), f"audit_{int(audit)}"),
                    "n": int(np.sum((g[idx] == group) & (z[idx] == audit))),
                })
    cell_counts_group_z = pd.DataFrame(cell_rows)
    cell_counts_group_z.to_csv(out_dir / "cell_counts_group_z.csv", index=False)

    gy_rows = []
    for split_name, idx in splits.items():
        for group in sorted(np.unique(g)):
            for cls in sorted(np.unique(y)):
                gy_rows.append({
                    "split": split_name,
                    "group": int(group),
                    "group_name": label_maps["group_names"][str(int(group))],
                    "label": int(cls),
                    "label_name": label_names[str(int(cls))],
                    "n": int(np.sum((g[idx] == group) & (y[idx] == cls))),
                })
    pd.DataFrame(gy_rows).to_csv(out_dir / "cell_counts_group_y.csv", index=False)

    tv = total_variation_z_by_group(z, g, audit_ids)
    target_z_assoc = {"cramers_v_y_z": cramers_v(y, z), "n": int(len(y))}
    min_split_group_z = int(cell_counts_group_z["n"].min()) if not cell_counts_group_z.empty else 0
    overall_group_z_counts = pd.crosstab(pd.Series(g, name="group"), pd.Series(z, name="audit"))
    min_group_z = int(overall_group_z_counts.to_numpy().min()) if overall_group_z_counts.size else 0
    state_codes_observed_after_topk_cap = (
        sorted(int(x) for x in pd.to_numeric(df["ST"], errors="coerce").dropna().unique())
        if "ST" in df.columns
        else []
    )
    screening = {
        "design_name": cfg.get("design_name"),
        "year": year,
        "states_requested": "all",
        "states_loaded": states_loaded,
        "state_codes_observed": state_codes_observed,
        "state_codes_observed_after_topk_cap": state_codes_observed_after_topk_cap,
        "n_raw_before_filter": n_raw_before_filter,
        "n_after_valid_filter": n_after_valid_filter,
        "n_after_valid_filter_after_max_rows": n_after_valid_filter_after_max_rows,
        "n_after_topk_before_cap": n_after_topk_before_cap,
        "n_after_topk": n_after_topk,
        "n_after_cap": n_after_cap,
        "n_after_filter_topk": int(len(y)),
        "top_k": top_k,
        "z_in_x": z_in_x,
        "min_group_z_cell_n": min_group_z,
        "min_split_group_z_cell_n": min_split_group_z,
        "tv_z_by_group": tv.get("tv"),
        "cramers_v_y_z": target_z_assoc["cramers_v_y_z"],
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "feature_count": int(len(feature_cols)),
        "max_rows": args.max_rows,
        "max_final_rows": args.max_final_rows,
        "cap_strategy": args.cap_strategy,
        "cap_seed": int(args.cap_seed),
        "cap_applied": bool(cap_meta["cap_applied"]),
    }
    screening.update(cap_meta)
    pd.DataFrame([screening]).to_csv(out_dir / "screening_summary.csv", index=False)

    split_meta = {
        "year": year,
        "states_requested": "all",
        "states_loaded": states_loaded,
        "state_codes_observed": state_codes_observed,
        "state_codes_observed_after_topk_cap": state_codes_observed_after_topk_cap,
        "split_ratios": cfg["splits"],
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "random_seed": seed,
        "split_names": list(splits.keys()),
        "n_classes": top_k,
        "n_groups": int(len(np.unique(g))),
        "n_audit_cells": int(len(audit_ids)),
    }
    model_summary = {
        "model_type": "HistGradientBoostingClassifier",
        "model_params": model.get_params(),
        "classes_seen": [int(x) for x in model.classes_],
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "excluded_feature_columns": excluded_feature_cols,
        "feature_exclusion_reasons": exclusion_reasons,
        "leakage_exclusion_columns": sorted(CORE_LEAKAGE_COLUMNS),
        "z_in_x": z_in_x,
    }
    args_record = vars(args).copy()
    args_record["config"] = str(args.config)
    args_record["output_root"] = str(out_root)
    args_record["data_root"] = str(data_root)

    cfg_used = json.loads(json.dumps(cfg))
    cfg_used["runtime"] = {
        "year": year,
        "top_k": top_k,
        "z_in_x": z_in_x,
        "data_root": str(data_root),
        "output_root": str(out_root),
        "max_rows": args.max_rows,
        "max_final_rows": args.max_final_rows,
        "cap_strategy": args.cap_strategy,
        "cap_seed": int(args.cap_seed),
        "cap_applied": bool(cap_meta["cap_applied"]),
        "density": density,
        "download": download,
    }
    manifest = {
        "stage": "train_export",
        "design_name": cfg.get("design_name"),
        "year": year,
        "states_requested": "all",
        "states_loaded": states_loaded,
        "state_codes_observed": state_codes_observed,
        "state_codes_observed_after_topk_cap": state_codes_observed_after_topk_cap,
        "n_raw_before_filter": n_raw_before_filter,
        "n_after_valid_filter": n_after_valid_filter,
        "n_after_valid_filter_after_max_rows": n_after_valid_filter_after_max_rows,
        "n_after_topk_before_cap": n_after_topk_before_cap,
        "n_after_topk": n_after_topk,
        "max_final_rows": args.max_final_rows,
        "n_after_cap": n_after_cap,
        "cap_strategy": args.cap_strategy,
        "cap_seed": int(args.cap_seed),
        "cap_applied": bool(cap_meta["cap_applied"]),
        "cap_meta": cap_meta,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "min_group_z_cell_n": min_group_z,
        "min_split_group_z_cell_n": min_split_group_z,
        "feature_columns": feature_cols,
        "excluded_feature_columns": excluded_feature_cols,
        "feature_exclusion_reasons": exclusion_reasons,
        "max_rows": args.max_rows,
        "is_subsampled": bool(args.max_rows is not None),
        "is_final_row_capped": bool(cap_meta["cap_applied"]),
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "outputs": sorted(p.name for p in out_dir.iterdir()),
        "validity_boundary": "Classifier probabilities are frozen here; conformal thresholds and IPW weights are computed downstream.",
    }

    jdump(label_maps, out_dir / "label_maps.json")
    jdump(split_meta, out_dir / "split_meta.json")
    jdump(screening, out_dir / "screening_summary.json")
    jdump(tv, out_dir / "tv_z_by_group.json")
    jdump(target_z_assoc, out_dir / "target_z_association.json")
    jdump(model_summary, out_dir / "model_summary.json")
    jdump(metrics_by_split, out_dir / "metrics_all.json")
    for split_name, obj in metrics_by_split.items():
        name = "calibration" if split_name == "cal" else split_name
        jdump(obj, out_dir / f"{name}_metrics.json")
    jdump(args_record, out_dir / "args.json")
    write_yaml(cfg_used, out_dir / "design_config_used.yaml")
    jdump(manifest, out_dir / "manifest.json")
    print(f"Wrote train artifacts to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
