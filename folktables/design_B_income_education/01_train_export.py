#!/usr/bin/env python3
"""Stage 1: build Design B Folktables data, train classifier, export artifacts."""

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

DIRECT_INCOME_LEAKAGE_COLUMNS = {
    "PINCP", "WAGP", "SEMP", "SSIP", "SSP", "RETP", "INTP", "OIP", "PAP",
    "PERNP", "POVPIP",
    "FPINCP", "FWAGP", "FSEMP", "FSSIP", "FSSP", "FRETP", "FINTP", "FOIP", "FPAP",
    "FPERNP", "FPOVPIP",
}
GROUP_LEAKAGE_COLUMNS = {
    "RAC1P", "RAC2P", "RAC3P", "RACAIAN", "RACASN", "RACBLK", "RACNH",
    "RACNUM", "RACPI", "RACSOR", "RACWHT", "HISP",
    "FRAC1P", "FRAC2P", "FRAC3P", "FRACAIAN", "FRACASN", "FRACBLK",
    "FRACNH", "FRACNUM", "FRACPI", "FRACSOR", "FRACWHT", "FHISP",
}
CORE_LEAKAGE_COLUMNS = DIRECT_INCOME_LEAKAGE_COLUMNS | GROUP_LEAKAGE_COLUMNS | {"FSCHLP"}


def resolve_path(path: str | Path, base: Path = SCRIPT_DIR) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def parse_bool(text: str | bool | None) -> bool | None:
    if text is None or isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {text!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--group_mode", choices=["white_nonwhite", "white_black"], default=None)
    p.add_argument("--feature_policy", choices=["income_safe", "income_safe_reduced"], default=None)
    p.add_argument("--z-in-x", action=argparse.BooleanOptionalAction, default=None, dest="z_in_x")
    p.add_argument("--z_in_x", type=parse_bool, default=None, dest="z_in_x")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--density", type=float, default=None)
    p.add_argument("--max_rows", type=int, default=None, help="Optional row cap after valid group filtering.")
    p.add_argument("--max_final_rows", type=int, default=None, help="Optional post-label resource cap applied before splitting.")
    p.add_argument("--cap_seed", type=int, default=0, help="Random seed for --max_final_rows subsampling.")
    p.add_argument("--cap_strategy", default="stratified_g_z", choices=["stratified_g_z"], help="Post-filter cap strategy.")
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


def safe_group_z_strata(g: np.ndarray, z: np.ndarray) -> np.ndarray | None:
    labels = np.asarray([f"{int(gg)}:{int(zz)}" for gg, zz in zip(g, z)], dtype=object)
    counts = Counter(labels)
    if counts and min(counts.values()) >= 2:
        return labels
    return None


def stratified_group_z_cap_indices(
    g: np.ndarray,
    z: np.ndarray,
    max_final_rows: int | None,
    seed: int,
    strategy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(len(g))
    meta: dict[str, Any] = {
        "cap_applied": False,
        "cap_strategy": strategy,
        "cap_seed": int(seed),
        "max_final_rows": int(max_final_rows) if max_final_rows is not None else None,
        "n_before_cap": n,
        "n_after_cap": n,
    }
    if max_final_rows is None or int(max_final_rows) <= 0 or n <= int(max_final_rows):
        meta["cap_reason"] = "not_requested_or_not_needed"
        return np.arange(n), meta
    if strategy != "stratified_g_z":
        raise ValueError(f"Unsupported cap strategy: {strategy}")

    cap = int(max_final_rows)
    labels = np.asarray([f"{int(gg)}:{int(zz)}" for gg, zz in zip(g, z)], dtype=object)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_strata = int(len(unique))
    if cap < n_strata:
        raise ValueError(
            f"max_final_rows={cap} is smaller than the number of nonempty G/Z strata ({n_strata}); "
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
        selected.append(rng.choice(idx, size=int(allocation[j]), replace=False))
    selected_idx = np.sort(np.concatenate(selected).astype(int))
    if int(len(selected_idx)) != cap:
        raise RuntimeError(f"Final cap allocation selected {len(selected_idx)} rows, expected {cap}")

    retention = allocation.astype(float) / counts.astype(float)
    meta.update({
        "cap_applied": True,
        "cap_reason": "post_filter_resource_cap",
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
    idx = np.arange(len(y))
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


def train_cal_repair_test_split_group_z(g: np.ndarray, z: np.ndarray, ratios: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    idx = np.arange(len(g))
    train_size = float(ratios["train"])
    cal_size = float(ratios["cal"])
    repair_size = float(ratios["repair"])
    test_size = float(ratios["test"])
    total = train_size + cal_size + repair_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1, got {total}")

    strata = safe_group_z_strata(g, z)
    try:
        train_idx, rest_idx = train_test_split(idx, train_size=train_size, random_state=seed, shuffle=True, stratify=strata)
    except ValueError:
        train_idx, rest_idx = train_test_split(idx, train_size=train_size, random_state=seed, shuffle=True, stratify=None)

    rest_strata = safe_group_z_strata(g[rest_idx], z[rest_idx])
    cal_frac_of_rest = cal_size / (1.0 - train_size)
    try:
        cal_idx, tail_idx = train_test_split(rest_idx, train_size=cal_frac_of_rest, random_state=seed + 1, shuffle=True, stratify=rest_strata)
    except ValueError:
        cal_idx, tail_idx = train_test_split(rest_idx, train_size=cal_frac_of_rest, random_state=seed + 1, shuffle=True, stratify=None)

    tail_strata = safe_group_z_strata(g[tail_idx], z[tail_idx])
    repair_frac_of_tail = repair_size / (repair_size + test_size)
    try:
        repair_idx, test_idx = train_test_split(tail_idx, train_size=repair_frac_of_tail, random_state=seed + 2, shuffle=True, stratify=tail_strata)
    except ValueError:
        repair_idx, test_idx = train_test_split(tail_idx, train_size=repair_frac_of_tail, random_state=seed + 2, shuffle=True, stratify=None)

    return {"train": train_idx, "cal": cal_idx, "repair": repair_idx, "test": test_idx}


def select_feature_columns(df: pd.DataFrame, cfg: dict[str, Any], z_in_x: bool, feature_policy: str) -> tuple[list[str], list[str], dict[str, str]]:
    configured = set(str(x) for x in cfg.get("features", {}).get("excluded_columns", []))
    excluded = set(configured) | set(CORE_LEAKAGE_COLUMNS)
    if feature_policy == "income_safe_reduced":
        excluded.update({"COW", "ESR", "JWTRNS", "JWAP", "JWDP", "WKL", "WKHP", "WRK", "FESRP", "FWKLP", "FWKHP", "FWRKP"})
    elif feature_policy != "income_safe":
        raise ValueError(f"Unsupported feature_policy: {feature_policy}")
    if not z_in_x:
        excluded.add("SCHL")
    prefixes = tuple(str(x) for x in cfg.get("features", {}).get("exclude_weight_prefixes", []))
    cols: list[str] = []
    reasons: dict[str, str] = {}
    for col in df.columns:
        col_s = str(col)
        if col_s in excluded:
            if col_s in DIRECT_INCOME_LEAKAGE_COLUMNS:
                reasons[col_s] = "direct_income_leakage_or_target_allocation_flag"
            elif col_s in GROUP_LEAKAGE_COLUMNS:
                reasons[col_s] = "race_group_leakage_or_group_allocation_flag"
            elif col_s == "FSCHLP":
                reasons[col_s] = "audit_allocation_flag"
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
    top3 = np.argsort(-probs, axis=1)[:, : min(3, probs.shape[1])]
    out["top3_accuracy"] = float(np.mean([yy in row for yy, row in zip(y, top3)])) if len(y) else float("nan")
    try:
        out["log_loss"] = float(log_loss(y, np.clip(probs, 1e-12, 1.0), labels=list(range(probs.shape[1]))))
    except ValueError:
        out["log_loss"] = float("nan")
    return out


def group_from_rac1p(rac1p: pd.Series, group_mode: str) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    r = pd.to_numeric(rac1p, errors="coerce")
    if group_mode == "white_black":
        mask = r.isin([1, 2])
        g = pd.Series(np.nan, index=r.index, dtype="float64")
        g.loc[r.eq(1)] = 0
        g.loc[r.eq(2)] = 1
        meta = {
            "group_mode": "white_black",
            "group_names": {"0": "White", "1": "Black"},
            "group_raw_to_new": {"RAC1P==1": 0, "RAC1P==2": 1},
            "group_positive_class": 1,
            "group_positive_name": "Black",
            "group_negative_class": 0,
            "group_negative_name": "White",
        }
        return mask, g, meta
    if group_mode == "white_nonwhite":
        mask = r.between(1, 9, inclusive="both")
        g = pd.Series(np.nan, index=r.index, dtype="float64")
        g.loc[mask] = np.where(r.loc[mask].eq(1), 1, 0)
        meta = {
            "group_mode": "white_nonwhite",
            "group_names": {"0": "Non-White", "1": "White"},
            "group_raw_to_new": {"RAC1P==1": 1, "other_valid_RAC1P": 0},
            "group_positive_class": 1,
            "group_positive_name": "White",
            "group_negative_class": 0,
            "group_negative_name": "Non-White",
        }
        return mask, g, meta
    raise ValueError(f"Unsupported group_mode: {group_mode}")


def valid_design_vectors(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    group_mode: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    required = ["PINCP", "RAC1P", "SCHL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required ACS columns: {missing}")
    pincp = pd.to_numeric(df["PINCP"], errors="coerce")
    rac1p = pd.to_numeric(df["RAC1P"], errors="coerce")
    schl = pd.to_numeric(df["SCHL"], errors="coerce")
    z_bucket = education_bucket(schl, cfg["task"]["education_buckets"])
    valid_race = rac1p.between(1, 9, inclusive="both")
    valid_base = pincp.notna() & np.isfinite(pincp) & valid_race & z_bucket.notna()
    group_mask, g, group_meta = group_from_rac1p(rac1p, group_mode)
    valid_group = valid_base & group_mask & g.notna()
    return pincp, rac1p, z_bucket, g, valid_base, valid_group, group_meta


def learn_income_quintile_thresholds(pincp: np.ndarray, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    train_income = np.asarray(pincp[train_idx], dtype=float)
    thresholds = np.quantile(train_income, [0.2, 0.4, 0.6, 0.8])
    thresholds = np.asarray(thresholds, dtype=float)
    meta = {
        "income_quintile_threshold_scope": "final_train_split_only",
        "income_quintile_threshold_seed": int(seed),
        "income_quintile_threshold_learning_n": int(len(train_idx)),
        "income_quintile_thresholds": [float(x) for x in thresholds],
        "income_quintile_threshold_note": "Thresholds are learned only from the final train split after White/Black filtering and any resource cap; cal/repair/test labels do not influence threshold estimation.",
    }
    return thresholds, meta


def assign_income_quintiles(pincp: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(thresholds, dtype=float), np.asarray(pincp, dtype=float), side="right").astype(int)


def load_filtered_capped_design_frame_all_states(
    year: int,
    root_dir: Path,
    density: float,
    download: bool,
    seed: int,
    cfg: dict[str, Any],
    group_mode: str,
    max_rows: int | None,
    max_final_rows: int | None,
    cap_seed: int,
    cap_strategy: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], list[int], int, int, int, int, int, dict[str, Any], dict[str, Any]]:
    """Load all states while only materializing rows selected after White/Black filter and cap."""
    from folktables import ACSDataSource

    ds = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=str(root_dir))
    states = list(FOLKTABLES_ALL_STATES)
    states_loaded: list[str] = []
    state_codes_observed: set[int] = set()
    n_raw_before_filter = 0
    n_after_valid_filter = 0
    n_after_group_filter = 0
    pincp_chunks: list[np.ndarray] = []
    g_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    group_meta: dict[str, Any] | None = None

    for state in states:
        sdf = ds.get_data(states=[state], density=float(density), random_seed=int(seed), download=bool(download))
        states_loaded.append(state)
        n_raw_before_filter += int(len(sdf))
        pincp, _rac1p, z_bucket, g_series, valid_base, valid_group, state_group_meta = valid_design_vectors(sdf, cfg, group_mode)
        group_meta = state_group_meta
        n_after_valid_filter += int(valid_base.sum())
        n_after_group_filter += int(valid_group.sum())
        if "ST" in sdf.columns:
            state_codes_observed.update(int(x) for x in pd.to_numeric(sdf.loc[valid_group, "ST"], errors="coerce").dropna().unique())
        pincp_chunks.append(pincp.loc[valid_group].to_numpy(dtype=float))
        g_chunks.append(g_series.loc[valid_group].to_numpy(dtype=int))
        z_chunks.append(z_bucket.loc[valid_group].to_numpy(dtype=int))

    if not pincp_chunks:
        raise ValueError("No valid Design B rows found")
    pincp_all = np.concatenate(pincp_chunks)
    g_all = np.concatenate(g_chunks)
    z_all = np.concatenate(z_chunks)
    n_before_cap_population = int(len(pincp_all))
    if n_before_cap_population < 100:
        raise ValueError(f"Too few rows after group filtering: n={n_before_cap_population}")

    base_indices = np.arange(n_before_cap_population)
    subsample_meta: dict[str, Any] = {
        "max_rows": int(max_rows) if max_rows is not None else None,
        "subsample_applied": False,
        "n_before_cap_before_subsample": n_before_cap_population,
    }
    if max_rows is not None and int(max_rows) > 0 and n_before_cap_population > int(max_rows):
        rng = np.random.default_rng(seed)
        base_indices = np.sort(rng.choice(base_indices, size=int(max_rows), replace=False))
        subsample_meta["subsample_applied"] = True
        subsample_meta["n_before_cap_after_subsample"] = int(len(base_indices))
    else:
        subsample_meta["n_before_cap_after_subsample"] = int(len(base_indices))

    cap_idx_relative, cap_meta = stratified_group_z_cap_indices(
        g_all[base_indices],
        z_all[base_indices],
        max_final_rows=max_final_rows,
        seed=int(cap_seed),
        strategy=str(cap_strategy),
    )
    selected_global_indices = base_indices[cap_idx_relative]
    selected_global_indices = np.sort(selected_global_indices.astype(int))
    keep_global = np.zeros(n_before_cap_population, dtype=bool)
    keep_global[selected_global_indices] = True

    g = g_all[selected_global_indices].astype(int)
    z = z_all[selected_global_indices].astype(int)
    pincp = pincp_all[selected_global_indices].astype(float)
    n_after_cap = int(len(selected_global_indices))
    cap_meta.update(subsample_meta)
    cap_meta["n_before_cap"] = int(len(base_indices))
    cap_meta["n_before_cap_population"] = n_before_cap_population
    cap_meta["n_after_cap"] = n_after_cap

    frames: list[pd.DataFrame] = []
    global_offset = 0
    for state in states:
        sdf = ds.get_data(states=[state], density=float(density), random_seed=int(seed), download=bool(download))
        *_vectors, valid_group, _state_group_meta = valid_design_vectors(sdf, cfg, group_mode)
        n_group_state = int(valid_group.sum())
        if n_group_state == 0:
            continue
        keep_state = keep_global[global_offset: global_offset + n_group_state]
        global_offset += n_group_state
        if not bool(keep_state.any()):
            continue
        valid_positions = np.flatnonzero(valid_group.to_numpy())
        selected_positions = valid_positions[keep_state]
        frames.append(sdf.iloc[selected_positions].copy())
    if global_offset != n_before_cap_population:
        raise RuntimeError(f"Second pass group-filtered row count mismatch: {global_offset} vs {n_before_cap_population}")
    if not frames:
        raise ValueError("No rows selected for Design B after cap")
    df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    if len(df) != n_after_cap:
        raise RuntimeError(f"Materialized {len(df)} rows but selected {n_after_cap}")

    return (
        df,
        pincp,
        g,
        z,
        states_loaded,
        sorted(state_codes_observed),
        n_raw_before_filter,
        n_after_valid_filter,
        n_after_group_filter,
        n_before_cap_population,
        n_after_cap,
        cap_meta,
        dict(group_meta or {}),
    )


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    start = time.time()

    year = int(args.year)
    z_in_x = bool(args.z_in_x if args.z_in_x is not None else cfg["task"].get("z_in_x", True))
    group_mode = str(args.group_mode if args.group_mode is not None else cfg["task"].get("group_mode", "white_black"))
    feature_policy = str(args.feature_policy if args.feature_policy is not None else cfg["task"].get("feature_policy", "income_safe"))
    seed = int(args.random_seed if args.random_seed is not None else cfg.get("random_seed", 20260609))
    density = float(args.density if args.density is not None else cfg.get("acs", {}).get("density", 1.0))
    download = bool(args.download if args.download is not None else cfg.get("acs", {}).get("download", True))
    data_root = resolve_path(args.data_root if args.data_root is not None else cfg.get("acs", {}).get("data_root", "../data/folktables_acs"))
    out_root = resolve_path(args.output_root)
    out_dir = mkdir(out_root / str(year) / "train")
    n_classes = int(cfg["task"].get("n_classes", 5))

    print(
        f"Loading ACS {year} explicit all-state list from {data_root} "
        f"(download={download}, density={density}) with group_mode={group_mode}, feature_policy={feature_policy}",
        flush=True,
    )
    (
        df,
        pincp,
        g,
        z,
        states_loaded,
        state_codes_observed,
        n_raw_before_filter,
        n_after_valid_filter,
        n_after_group_filter,
        n_before_cap_population,
        n_after_cap,
        cap_meta,
        group_meta,
    ) = load_filtered_capped_design_frame_all_states(
        year,
        data_root,
        density,
        download,
        seed,
        cfg,
        group_mode,
        args.max_rows,
        args.max_final_rows,
        int(args.cap_seed),
        str(args.cap_strategy),
    )

    splits = train_cal_repair_test_split_group_z(g, z, cfg["splits"], seed)
    thresholds, quantile_meta = learn_income_quintile_thresholds(pincp, splits["train"], seed)
    y = assign_income_quintiles(pincp, thresholds)
    quantile_meta["income_quintile_class_counts_after_cap"] = {str(int(k)): int(v) for k, v in Counter(y).items()}
    quantile_meta["income_quintile_class_counts_train"] = {str(int(k)): int(v) for k, v in Counter(y[splits["train"]]).items()}
    quantile_meta["income_quintile_class_counts_cal"] = {str(int(k)): int(v) for k, v in Counter(y[splits["cal"]]).items()}
    quantile_meta["income_quintile_class_counts_repair"] = {str(int(k)): int(v) for k, v in Counter(y[splits["repair"]]).items()}
    quantile_meta["income_quintile_class_counts_test"] = {str(int(k)): int(v) for k, v in Counter(y[splits["test"]]).items()}
    if sorted(int(x) for x in np.unique(y)) != list(range(n_classes)):
        raise ValueError(f"Income quintile labeling did not produce all {n_classes} classes; counts={Counter(y)}")

    audit_ids = sorted(int(x) for x in np.unique(z))
    audit_names = {
        str(int(b["id"])): str(b["name"])
        for b in cfg["task"]["education_buckets"]
        if int(b["id"]) in audit_ids
    }
    thresholds = quantile_meta["income_quintile_thresholds"]
    label_names = {
        "0": f"Q1 lowest income <= {thresholds[0]:.2f}",
        "1": f"Q2 income ({thresholds[0]:.2f}, {thresholds[1]:.2f}]",
        "2": f"Q3 income ({thresholds[1]:.2f}, {thresholds[2]:.2f}]",
        "3": f"Q4 income ({thresholds[2]:.2f}, {thresholds[3]:.2f}]",
        "4": f"Q5 highest income > {thresholds[3]:.2f}",
    }
    label_maps = {
        "label_source": "PINCP_income_quintile",
        "label_names": label_names,
        "income_quintile_thresholds": thresholds,
        "income_quintile_threshold_scope": quantile_meta["income_quintile_threshold_scope"],
        "group_source": "RAC1P",
        "group_mode": group_mode,
        "group_names": group_meta["group_names"],
        "group_raw_to_new": group_meta["group_raw_to_new"],
        "group_positive_class": int(group_meta["group_positive_class"]),
        "group_positive_name": group_meta["group_positive_name"],
        "audit_source": "SCHL_education_bucket",
        "audit_names": audit_names,
    }

    feature_cols, excluded_feature_cols, exclusion_reasons = select_feature_columns(df, cfg, z_in_x, feature_policy)
    if not feature_cols:
        raise ValueError("No numeric ACS feature columns selected")
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X_df.to_numpy(dtype=float)

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
        probs = align_predict_proba(model, X[idx], n_classes)
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
    state_codes_observed_after_label_cap = (
        sorted(int(x) for x in pd.to_numeric(df["ST"], errors="coerce").dropna().unique())
        if "ST" in df.columns
        else []
    )
    split_sizes = {k: int(len(v)) for k, v in splits.items()}
    n_before_cap = int(cap_meta.get("n_before_cap", n_before_cap_population))
    income_proxy_removed = bool(any(c in set(df.columns) for c in DIRECT_INCOME_LEAKAGE_COLUMNS))
    schl_in_x = "SCHL" in feature_cols
    screening = {
        "design_name": cfg.get("design_name"),
        "year": year,
        "states_requested": "all",
        "states_loaded": states_loaded,
        "state_codes_observed": state_codes_observed,
        "state_codes_observed_after_label_cap": state_codes_observed_after_label_cap,
        "n_raw_before_filter": n_raw_before_filter,
        "n_after_valid_filter": n_after_valid_filter,
        "n_after_group_filter": n_after_group_filter,
        "n_before_cap_population": n_before_cap_population,
        "n_before_cap": n_before_cap,
        "n_after_labeling_before_cap": n_before_cap,
        "n_after_cap": n_after_cap,
        "n_classes": n_classes,
        "group_mode": group_mode,
        "feature_policy": feature_policy,
        "z_in_x": z_in_x,
        "schl_in_x": schl_in_x,
        "income_proxy_removed": income_proxy_removed,
        "min_group_z_cell_n": min_group_z,
        "min_split_group_z_cell_n": min_split_group_z,
        "tv_z_by_group": tv.get("tv"),
        "cramers_v_y_z": target_z_assoc["cramers_v_y_z"],
        "split_sizes": split_sizes,
        "feature_count": int(len(feature_cols)),
        "max_rows": args.max_rows,
        "max_final_rows": args.max_final_rows,
        "cap_strategy": args.cap_strategy,
        "cap_seed": int(args.cap_seed),
        "cap_applied": bool(cap_meta["cap_applied"]),
        **quantile_meta,
    }
    screening.update(cap_meta)
    pd.DataFrame([screening]).to_csv(out_dir / "screening_summary.csv", index=False)

    split_meta = {
        "year": year,
        "states_requested": "all",
        "states_loaded": states_loaded,
        "state_codes_observed": state_codes_observed,
        "state_codes_observed_after_label_cap": state_codes_observed_after_label_cap,
        "split_ratios": cfg["splits"],
        "split_sizes": split_sizes,
        "random_seed": seed,
        "split_names": list(splits.keys()),
        "n_classes": n_classes,
        "n_groups": int(len(np.unique(g))),
        "n_audit_cells": int(len(audit_ids)),
    }
    model_summary = {
        "model_type": "HistGradientBoostingClassifier",
        "model_params": model.get_params(),
        "classes_seen": [int(x) for x in model.classes_],
        "group_mode": group_mode,
        "feature_policy": feature_policy,
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "excluded_feature_columns": excluded_feature_cols,
        "feature_exclusion_reasons": exclusion_reasons,
        "direct_income_leakage_exclusion_columns": sorted(DIRECT_INCOME_LEAKAGE_COLUMNS),
        "group_leakage_exclusion_columns": sorted(GROUP_LEAKAGE_COLUMNS),
        "z_in_x": z_in_x,
        "schl_in_x": schl_in_x,
        "income_proxy_removed": income_proxy_removed,
        **quantile_meta,
    }
    args_record = vars(args).copy()
    args_record["config"] = str(args.config)
    args_record["output_root"] = str(out_root)
    args_record["data_root"] = str(data_root)

    cfg_used = json.loads(json.dumps(cfg))
    cfg_used["runtime"] = {
        "year": year,
        "group_mode": group_mode,
        "feature_policy": feature_policy,
        "z_in_x": z_in_x,
        "schl_in_x": schl_in_x,
        "income_proxy_removed": income_proxy_removed,
        "data_root": str(data_root),
        "output_root": str(out_root),
        "max_rows": args.max_rows,
        "max_final_rows": args.max_final_rows,
        "cap_strategy": args.cap_strategy,
        "cap_seed": int(args.cap_seed),
        "cap_applied": bool(cap_meta["cap_applied"]),
        "density": density,
        "download": download,
        **quantile_meta,
    }
    manifest = {
        "stage": "train_export",
        "design_name": cfg.get("design_name"),
        "year": year,
        "states_requested": "all",
        "states_loaded": states_loaded,
        "state_codes_observed": state_codes_observed,
        "state_codes_observed_after_label_cap": state_codes_observed_after_label_cap,
        "n_raw_before_filter": n_raw_before_filter,
        "n_after_valid_filter": n_after_valid_filter,
        "n_after_group_filter": n_after_group_filter,
        "n_before_cap_population": n_before_cap_population,
        "n_before_cap": n_before_cap,
        "n_after_labeling_before_cap": n_before_cap,
        "max_final_rows": args.max_final_rows,
        "n_after_cap": n_after_cap,
        "cap_strategy": args.cap_strategy,
        "cap_seed": int(args.cap_seed),
        "cap_applied": bool(cap_meta["cap_applied"]),
        "cap_meta": cap_meta,
        "split_sizes": split_sizes,
        "min_group_z_cell_n": min_group_z,
        "min_split_group_z_cell_n": min_split_group_z,
        "group_mode": group_mode,
        "group_names": group_meta["group_names"],
        "group_raw_to_new": group_meta["group_raw_to_new"],
        "feature_policy": feature_policy,
        "z_in_x": z_in_x,
        "schl_in_x": schl_in_x,
        "income_proxy_removed": income_proxy_removed,
        "feature_columns": feature_cols,
        "excluded_feature_columns": excluded_feature_cols,
        "feature_exclusion_reasons": exclusion_reasons,
        "max_rows": args.max_rows,
        "is_subsampled": bool(args.max_rows is not None),
        "is_final_row_capped": bool(cap_meta["cap_applied"]),
        **quantile_meta,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "outputs": sorted(p.name for p in out_dir.iterdir()),
        "validity_boundary": "Income quintile thresholds are learned from a seed-fixed pre-split training portion. Classifier probabilities are frozen here; conformal thresholds and IPW weights are computed downstream.",
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
