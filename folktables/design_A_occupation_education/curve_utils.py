#!/usr/bin/env python3
"""Shared utilities for Design A Folktables curve audits.

The functions here are intentionally generic: labels, groups, and audit cells
are named through metadata instead of BioBias-specific occupation/gender names.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def write_yaml(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_jsonable(obj), f, sort_keys=False)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def jdump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")


def jload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_alpha_grid(text: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(text, (list, tuple)):
        values = [float(x) for x in text]
    else:
        key = str(text).strip().lower()
        presets = {
            "dense_safe": np.round(np.arange(0.02, 0.1001, 0.005), 3).tolist(),
            "dense_010": np.round(np.arange(0.02, 0.1001, 0.005), 3).tolist(),
            "dense_012": np.round(np.arange(0.02, 0.1201, 0.005), 3).tolist(),
        }
        if key in presets:
            values = presets[key]
        else:
            values = sorted({float(x) for x in str(text).split(",") if x.strip()})
    if not values or any(x <= 0.0 or x >= 1.0 for x in values):
        raise ValueError("alpha grid must contain values in (0, 1)")
    return [float(x) for x in values]


def alpha_grid_from_range(alpha_min: float, alpha_max: float, alpha_step: float) -> list[float]:
    """Build an inclusive alpha grid without float-step drift."""
    lo = Decimal(str(alpha_min))
    hi = Decimal(str(alpha_max))
    step = Decimal(str(alpha_step))
    if lo <= 0 or hi >= 1 or lo > hi or step <= 0:
        raise ValueError("alpha_min/alpha_max/alpha_step must define values in (0, 1)")
    values: list[float] = []
    cur = lo
    guard = 0
    while cur <= hi + step / Decimal("10"):
        values.append(round(float(cur), 12))
        cur += step
        guard += 1
        if guard > 100000:
            raise ValueError("alpha grid is unexpectedly large")
    values = sorted(dict.fromkeys(v for v in values if float(v) <= float(hi) + 1e-12))
    if not values:
        raise ValueError("alpha grid is empty")
    return values


def label_name(meta: dict[str, Any], key: str, value: int) -> str:
    maps = meta.get(key, {})
    if isinstance(maps, dict):
        return str(maps.get(str(int(value)), maps.get(int(value), f"{key}_{int(value)}")))
    if isinstance(maps, list) and 0 <= int(value) < len(maps):
        return str(maps[int(value)])
    return f"{key}_{int(value)}"


def group_name(meta: dict[str, Any], group: int) -> str:
    return label_name(meta, "group_names", group)


def audit_name(meta: dict[str, Any], audit: int) -> str:
    return label_name(meta, "audit_names", audit)


def class_name(meta: dict[str, Any], label: int) -> str:
    return label_name(meta, "label_names", label)


def scores_from_probs(probs: np.ndarray, score: str = "simple") -> np.ndarray:
    """Return labelwise nonconformity scores.

    The design uses the BioBias-compatible SIMPLE score by default:
    s(x, y) = 1 - p_hat(y | x).
    """
    score = str(score).lower()
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    if score != "simple":
        raise ValueError("Design A currently supports score='simple'")
    return 1.0 - p


def true_scores(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    return scores[np.arange(len(y)), y]


def qhat(values: np.ndarray, alpha: float) -> float:
    s = np.asarray(values, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    return float("inf") if k > n else float(np.partition(s, k - 1)[k - 1])


def build_thresholds(cal_probs: np.ndarray, cal_y: np.ndarray, alphas: list[float], score: str) -> tuple[pd.DataFrame, dict[float, float]]:
    scores = scores_from_probs(cal_probs, score)
    ts = true_scores(scores, cal_y)
    thresholds = {float(a): qhat(ts, float(a)) for a in alphas}
    rows = [{"policy": "pooled", "alpha": float(a), "threshold": float(t)} for a, t in thresholds.items()]
    return pd.DataFrame(rows), thresholds


def set_sizes(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) <= float(threshold)).sum(axis=1).astype(float)


def covered(scores: np.ndarray, y: np.ndarray, threshold: float) -> np.ndarray:
    return (true_scores(scores, y) <= float(threshold)).astype(float)


def weighted_average(x: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not m.any() or float(w[m].sum()) <= 0:
        return float("nan")
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    denom = float(np.sum(w * w))
    if w.size == 0 or denom <= 0:
        return float("nan")
    return float(np.sum(w) ** 2 / denom)


def group_curve(
    split_name: str,
    probs: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    z: np.ndarray,
    thresholds: dict[float, float],
    alphas: list[float],
    meta: dict[str, Any],
    weights: np.ndarray | None = None,
    score: str = "simple",
    estimand: str = "raw",
) -> pd.DataFrame:
    scores = scores_from_probs(probs, score)
    g = np.asarray(g, dtype=int)
    y = np.asarray(y, dtype=int)
    if weights is None:
        weights = np.ones(len(g), dtype=float)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        threshold = float(thresholds[float(alpha)])
        size = set_sizes(scores, threshold)
        cov = covered(scores, y, threshold)
        for group in sorted(int(x) for x in np.unique(g)):
            mask = g == group
            wg = np.asarray(weights[mask], dtype=float)
            rows.append({
                "split": split_name,
                "policy": "pooled",
                "alpha": float(alpha),
                "group": int(group),
                "group_name": group_name(meta, group),
                "n": int(mask.sum()),
                "weight_sum": float(np.nansum(wg)),
                "ess": effective_sample_size(wg),
                "avg_set_size": weighted_average(size[mask], wg),
                "coverage": weighted_average(cov[mask], wg),
                "unweighted_avg_set_size": float(np.mean(size[mask])) if mask.any() else np.nan,
                "unweighted_coverage": float(np.mean(cov[mask])) if mask.any() else np.nan,
                "threshold": threshold,
                "estimand": estimand,
            })
    return pd.DataFrame(rows)


def local_curve_by_z(
    split_name: str,
    probs: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    z: np.ndarray,
    thresholds: dict[float, float],
    alphas: list[float],
    meta: dict[str, Any],
    weights: np.ndarray | None = None,
    score: str = "simple",
    estimand: str = "raw",
) -> pd.DataFrame:
    scores = scores_from_probs(probs, score)
    g = np.asarray(g, dtype=int)
    z = np.asarray(z, dtype=int)
    y = np.asarray(y, dtype=int)
    if weights is None:
        weights = np.ones(len(g), dtype=float)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        threshold = float(thresholds[float(alpha)])
        size = set_sizes(scores, threshold)
        cov = covered(scores, y, threshold)
        for group in sorted(int(x) for x in np.unique(g)):
            for audit in sorted(int(x) for x in np.unique(z)):
                mask = (g == group) & (z == audit)
                if not mask.any():
                    continue
                wz = np.asarray(weights[mask], dtype=float)
                rows.append({
                    "split": split_name,
                    "policy": "pooled",
                    "alpha": float(alpha),
                    "group": int(group),
                    "group_name": group_name(meta, group),
                    "audit": int(audit),
                    "audit_name": audit_name(meta, audit),
                    "n": int(mask.sum()),
                    "weight_sum": float(np.nansum(wz)),
                    "ess": effective_sample_size(wz),
                    "avg_set_size": weighted_average(size[mask], wz),
                    "coverage": weighted_average(cov[mask], wz),
                    "unweighted_avg_set_size": float(np.mean(size[mask])) if mask.any() else np.nan,
                    "unweighted_coverage": float(np.mean(cov[mask])) if mask.any() else np.nan,
                    "threshold": threshold,
                    "estimand": estimand,
                })
    return pd.DataFrame(rows)


def aggregate_gap(curve: pd.DataFrame) -> pd.DataFrame:
    groups = sorted(int(x) for x in curve["group"].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, got {groups}")
    a, b = groups
    rows: list[dict[str, Any]] = []
    for (split, policy, alpha), sub in curve.groupby(["split", "policy", "alpha"], sort=True):
        sa = sub[sub["group"] == a].iloc[0]
        sb = sub[sub["group"] == b].iloc[0]
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            "group_a": int(a),
            "group_a_name": str(sa["group_name"]),
            "group_b": int(b),
            "group_b_name": str(sb["group_name"]),
            "delta_V_a_minus_b": float(sa["avg_set_size"] - sb["avg_set_size"]),
            "delta_C_a_minus_b": float(sa["coverage"] - sb["coverage"]),
            "V_a": float(sa["avg_set_size"]),
            "V_b": float(sb["avg_set_size"]),
            "C_a": float(sa["coverage"]),
            "C_b": float(sb["coverage"]),
            "V_a_unweighted": float(sa["unweighted_avg_set_size"]),
            "V_b_unweighted": float(sb["unweighted_avg_set_size"]),
            "C_a_unweighted": float(sa["unweighted_coverage"]),
            "C_b_unweighted": float(sb["unweighted_coverage"]),
        })
    return pd.DataFrame(rows)


def local_gap(local: pd.DataFrame) -> pd.DataFrame:
    groups = sorted(int(x) for x in local["group"].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, got {groups}")
    a, b = groups
    rows: list[dict[str, Any]] = []
    for (split, policy, alpha, audit), sub in local.groupby(["split", "policy", "alpha", "audit"], sort=True):
        ga = sub[sub["group"] == a]
        gb = sub[sub["group"] == b]
        if ga.empty or gb.empty:
            continue
        ra = ga.iloc[0]
        rb = gb.iloc[0]
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            "audit": int(audit),
            "audit_name": str(ra["audit_name"]),
            "group_a": int(a),
            "group_a_name": str(ra["group_name"]),
            "group_b": int(b),
            "group_b_name": str(rb["group_name"]),
            "n_a_z": int(ra["n"]),
            "n_b_z": int(rb["n"]),
            "ess_a_z": float(ra["ess"]),
            "ess_b_z": float(rb["ess"]),
            "delta_V_a_minus_b_z": float(ra["avg_set_size"] - rb["avg_set_size"]),
            "delta_C_a_minus_b_z": float(ra["coverage"] - rb["coverage"]),
            "V_a_z": float(ra["avg_set_size"]),
            "V_b_z": float(rb["avg_set_size"]),
            "C_a_z": float(ra["coverage"]),
            "C_b_z": float(rb["coverage"]),
            "V_a_z_unweighted": float(ra["unweighted_avg_set_size"]),
            "V_b_z_unweighted": float(rb["unweighted_avg_set_size"]),
        })
    return pd.DataFrame(rows)


def simpson_reversal_table(
    gap: pd.DataFrame,
    local_gap_df: pd.DataFrame,
    n_min: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gap_idx = gap.set_index(["split", "policy", "alpha"])
    for _, loc in local_gap_df.iterrows():
        key = (loc["split"], loc["policy"], float(loc["alpha"]))
        if key not in gap_idx.index:
            continue
        agg = gap_idx.loc[key]
        agg_delta = float(agg["delta_V_a_minus_b"])
        loc_delta = float(loc["delta_V_a_minus_b_z"])
        abs_agg = float(abs(agg_delta))
        abs_loc = float(abs(loc_delta))
        is_sign_reversal = bool(np.sign(agg_delta) != np.sign(loc_delta))
        supported = bool(int(loc["n_a_z"]) >= int(n_min) and int(loc["n_b_z"]) >= int(n_min))
        supported_sign = bool(supported and is_sign_reversal)
        conflict_strength = float(max(-(agg_delta * loc_delta), 0.0))
        rows.append({
            "split": loc["split"],
            "policy": loc["policy"],
            "alpha": float(loc["alpha"]),
            "z": int(loc["audit"]),
            "audit": int(loc["audit"]),
            "z_name": str(loc["audit_name"]),
            "audit_name": str(loc["audit_name"]),
            "group_a": int(loc["group_a"]),
            "group_a_name": str(loc["group_a_name"]),
            "group_b": int(loc["group_b"]),
            "group_b_name": str(loc["group_b_name"]),
            "n_a_z": int(loc["n_a_z"]),
            "n_b_z": int(loc["n_b_z"]),
            "ess_a_z": float(loc["ess_a_z"]),
            "ess_b_z": float(loc["ess_b_z"]),
            "gap_aggregate": agg_delta,
            "gap_local": loc_delta,
            "abs_gap_aggregate": abs_agg,
            "abs_gap_local": abs_loc,
            "aggregate_gap_delta": agg_delta,
            "local_gap_delta_z": loc_delta,
            "delta_times_delta_z": float(agg_delta * loc_delta),
            "is_sign_reversal": is_sign_reversal,
            "is_supported_sign_reversal": supported_sign,
            "conflict_strength": conflict_strength,
            "product_strength": conflict_strength,
            "reversal_strength": conflict_strength,
            "supported": supported,
            "is_simpson_reversal": supported_sign,
        })
    return pd.DataFrame(rows)


def _supported(df: pd.DataFrame) -> pd.Series:
    return df["supported"].astype(bool) if "supported" in df else pd.Series(False, index=df.index)


def _sign_active(df: pd.DataFrame) -> pd.Series:
    if "is_sign_reversal" not in df:
        fallback = df["is_simpson_reversal"].astype(bool) if "is_simpson_reversal" in df else pd.Series(False, index=df.index)
        return _supported(df) & fallback
    return _supported(df) & df["is_sign_reversal"].astype(bool)


def _strength(df: pd.DataFrame) -> pd.Series:
    if "conflict_strength" in df:
        return pd.to_numeric(df["conflict_strength"], errors="coerce").fillna(0.0)
    if "product_strength" in df:
        return pd.to_numeric(df["product_strength"], errors="coerce").fillna(0.0)
    if "reversal_strength" in df:
        return pd.to_numeric(df["reversal_strength"], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def reversal_counts_by_alpha(reversals: pd.DataFrame) -> pd.DataFrame:
    if reversals.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (split, policy, alpha), sub in reversals.groupby(["split", "policy", "alpha"], sort=True):
        supported = sub[_supported(sub)]
        sign_active = sub[_sign_active(sub)]
        supported_cells = int(supported.shape[0])
        sign_count = int(sign_active.shape[0])
        strength = _strength(supported)
        conflict_sum = float(strength.sum()) if supported_cells else 0.0
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            "supported_cells": supported_cells,
            "sign_reversal_count": sign_count,
            "sign_reversal_rate": float(sign_count / supported_cells) if supported_cells else float("nan"),
            "conflict_strength_sum": conflict_sum,
            "conflict_strength_mean_over_active_alpha_cells": float(conflict_sum / supported_cells) if supported_cells else float("nan"),
            "conflict_strength_mean_over_sign_reversals": float(conflict_sum / sign_count) if sign_count else 0.0,
            "num_reversal_cells": sign_count,
            "total_reversal_strength": conflict_sum,
        })
    return pd.DataFrame(rows)


def persistent_reversals(reversals: pd.DataFrame, alpha0: float = 0.10) -> pd.DataFrame:
    if reversals.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (split, policy, audit), sub in reversals.groupby(["split", "policy", "audit"], sort=True):
        sign_active = sub[_sign_active(sub)]
        idx = int(np.argmin(np.abs(sub["alpha"].to_numpy(float) - float(alpha0))))
        near = sub.iloc[idx]
        sign_count = int(sign_active.shape[0])
        strength = _strength(sign_active)
        conflict_sum = float(strength.sum()) if sign_count else 0.0
        rows.append({
            "split": split,
            "policy": policy,
            "audit": int(audit),
            "audit_name": str(sub["audit_name"].iloc[0]),
            "supported_alpha_cells": int(sub["supported"].sum()),
            "num_sign_reversal_alpha": sign_count,
            "sign_reversal_alpha_fraction": float(sign_count / max(1, sub.shape[0])),
            "total_sign_conflict_strength": conflict_sum,
            "persistent_sign_reversal": bool(sign_count > 0),
            "num_reversal_alpha": sign_count,
            "reversal_alpha_fraction": float(sign_count / max(1, sub.shape[0])),
            "total_reversal_strength": conflict_sum,
            "alpha0": float(alpha0),
            "nearest_alpha": float(near["alpha"]),
            "is_sign_reversal_at_alpha0_nearest": bool(near.get("is_sign_reversal", False)),
            "is_reversal_at_alpha0_nearest": bool(near.get("is_sign_reversal", near.get("is_simpson_reversal", False))),
            "aggregate_gap_at_alpha0_nearest": float(near.get("gap_aggregate", near.get("aggregate_gap_delta"))),
            "local_gap_at_alpha0_nearest": float(near.get("gap_local", near.get("local_gap_delta_z"))),
        })
    return pd.DataFrame(rows).sort_values(
        ["split", "policy", "num_sign_reversal_alpha", "total_reversal_strength"],
        ascending=[True, True, False, False],
    )


def alignment_summary(reversals: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [
        "split",
        "policy",
        f"{prefix}_supported_alpha_cells",
        f"{prefix}_sign_reversal_alpha_cells",
        f"{prefix}_sign_reversal_rate",
        f"{prefix}_conflict_strength_sum",
        f"{prefix}_conflict_strength_mean_over_active_alpha_cells",
        f"{prefix}_reversal_alpha_cells",
        f"{prefix}_reversal_strength",
        f"{prefix}_mean_delta_product",
        f"{prefix}_same_sign_alignment",
        f"{prefix}_share_same_sign",
    ]
    if reversals.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    supported = reversals[reversals["supported"]].copy()
    if supported.empty:
        return pd.DataFrame(columns=columns)
    for (split, policy), sub in supported.groupby(["split", "policy"], sort=True):
        prod = sub["delta_times_delta_z"].to_numpy(float)
        sign_active = sub[_sign_active(sub)]
        sign_count = int(sign_active.shape[0])
        active_cells = int(sub.shape[0])
        sign_flags = sub["is_sign_reversal"].astype(bool) if "is_sign_reversal" in sub else sub["is_simpson_reversal"].astype(bool)
        same_sign = float(np.mean(~sign_flags)) if active_cells else np.nan
        conflict_sum = float(_strength(sub).sum()) if active_cells else 0.0
        rows.append({
            "split": split,
            "policy": policy,
            f"{prefix}_supported_alpha_cells": active_cells,
            f"{prefix}_sign_reversal_alpha_cells": sign_count,
            f"{prefix}_sign_reversal_rate": float(sign_count / active_cells) if active_cells else float("nan"),
            f"{prefix}_conflict_strength_sum": conflict_sum,
            f"{prefix}_conflict_strength_mean_over_active_alpha_cells": float(conflict_sum / active_cells) if active_cells else float("nan"),
            f"{prefix}_reversal_alpha_cells": sign_count,
            f"{prefix}_reversal_strength": conflict_sum,
            f"{prefix}_mean_delta_product": float(np.nanmean(prod)) if prod.size else np.nan,
            f"{prefix}_same_sign_alignment": same_sign,
            f"{prefix}_share_same_sign": same_sign,
        })
    return pd.DataFrame(rows, columns=columns)


def _reversal_group_summary(reversals: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    if reversals.empty:
        return out
    supported = reversals[reversals["supported"]].copy()
    for (split, policy), sub in supported.groupby(["split", "policy"], sort=True):
        active_cells = int(sub.shape[0])
        sign_active = sub[_sign_active(sub)]
        sign_count = int(sign_active.shape[0])
        conflict_sum = float(_strength(sub).sum()) if active_cells else 0.0
        out[(str(split), str(policy))] = {
            "active_alpha_cells": active_cells,
            "sign_reversal_count": sign_count,
            "sign_reversal_rate": float(sign_count / active_cells) if active_cells else float("nan"),
            "conflict_strength_sum": conflict_sum,
            "conflict_strength_mean_over_active_alpha_cells": float(conflict_sum / active_cells) if active_cells else float("nan"),
            "conflict_strength_mean_over_sign_reversals": float(conflict_sum / sign_count) if sign_count else 0.0,
            "reversal_count": sign_count,
            "reversal_rate": float(sign_count / active_cells) if active_cells else float("nan"),
            "integrated_reversal_strength_sum": conflict_sum,
            "mean_reversal_strength_per_active_alpha_cell": float(conflict_sum / active_cells) if active_cells else float("nan"),
            "mean_reversal_strength_per_reversal": float(conflict_sum / sign_count) if sign_count else 0.0,
        }
    return out


def reversal_rate_summary(reversal_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summaries = {prefix: _reversal_group_summary(df) for prefix, df in reversal_tables.items()}
    keys: set[tuple[str, str]] = set()
    for summary in summaries.values():
        keys.update(summary.keys())
    rows: list[dict[str, Any]] = []
    for split, policy in sorted(keys):
        row: dict[str, Any] = {"split": split, "policy": policy}
        for prefix, summary in summaries.items():
            vals = summary.get((split, policy), {})
            row[f"{prefix}_active_alpha_cells"] = int(vals.get("active_alpha_cells", 0))
            row[f"{prefix}_sign_reversal_count"] = int(vals.get("sign_reversal_count", 0))
            row[f"{prefix}_sign_reversal_rate"] = float(vals.get("sign_reversal_rate", float("nan")))
            row[f"{prefix}_reversal_count"] = int(vals.get("reversal_count", 0))
            row[f"{prefix}_reversal_rate"] = float(vals.get("reversal_rate", float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def conflict_strength_summary(reversal_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summaries = {prefix: _reversal_group_summary(df) for prefix, df in reversal_tables.items()}
    keys: set[tuple[str, str]] = set()
    for summary in summaries.values():
        keys.update(summary.keys())
    rows: list[dict[str, Any]] = []
    for split, policy in sorted(keys):
        row: dict[str, Any] = {"split": split, "policy": policy}
        for prefix, summary in summaries.items():
            vals = summary.get((split, policy), {})
            row[f"{prefix}_active_alpha_cells"] = int(vals.get("active_alpha_cells", 0))
            row[f"{prefix}_sign_reversal_count"] = int(vals.get("sign_reversal_count", 0))
            row[f"{prefix}_conflict_strength_sum"] = float(vals.get("conflict_strength_sum", 0.0))
            row[f"{prefix}_conflict_strength_mean_over_active_alpha_cells"] = float(vals.get("conflict_strength_mean_over_active_alpha_cells", float("nan")))
            row[f"{prefix}_conflict_strength_mean_over_sign_reversals"] = float(vals.get("conflict_strength_mean_over_sign_reversals", 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def conflict_strength_by_alpha(reversal_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys: set[tuple[str, str, float]] = set()
    for df in reversal_tables.values():
        if not df.empty:
            keys.update((str(s), str(p), float(a)) for s, p, a in df[["split", "policy", "alpha"]].drop_duplicates().itertuples(index=False))
    rows: list[dict[str, Any]] = []
    for split, policy, alpha in sorted(keys, key=lambda x: (x[0], x[1], x[2])):
        row: dict[str, Any] = {"split": split, "policy": policy, "alpha": float(alpha)}
        for prefix, df in reversal_tables.items():
            if df.empty:
                supported = pd.DataFrame()
            else:
                supported = df[
                    (df["split"].astype(str) == split)
                    & (df["policy"].astype(str) == policy)
                    & np.isclose(pd.to_numeric(df["alpha"], errors="coerce"), float(alpha))
                    & _supported(df)
                ].copy()
            active_cells = int(supported.shape[0])
            sign_count = int(_sign_active(supported).sum()) if active_cells else 0
            strength_sum = float(_strength(supported).sum()) if active_cells else 0.0
            row[f"{prefix}_active_alpha_cells"] = active_cells
            row[f"{prefix}_sign_reversal_count"] = sign_count
            row[f"{prefix}_conflict_strength_sum"] = strength_sum
            row[f"{prefix}_conflict_strength_mean_over_active_alpha_cells"] = (
                float(strength_sum / active_cells) if active_cells else float("nan")
            )
            row[f"{prefix}_conflict_strength_mean_over_sign_reversals"] = (
                float(strength_sum / sign_count) if sign_count else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def material_strength_counts(
    reversal_tables: dict[str, pd.DataFrame],
    thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
) -> pd.DataFrame:
    keys: set[tuple[str, str]] = set()
    for df in reversal_tables.values():
        if not df.empty:
            keys.update((str(s), str(p)) for s, p in df[["split", "policy"]].drop_duplicates().itertuples(index=False))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for split, policy in sorted(keys):
            row: dict[str, Any] = {"split": split, "policy": policy, "strength_threshold": float(threshold)}
            for prefix, df in reversal_tables.items():
                sub = df[(df["split"].astype(str) == split) & (df["policy"].astype(str) == policy)].copy() if not df.empty else pd.DataFrame()
                if sub.empty:
                    material = pd.DataFrame()
                else:
                    strength = _strength(sub)
                    material = sub[_supported(sub) & (strength > float(threshold))]
                row[f"{prefix}_material_strength_count"] = int(material.shape[0])
                row[f"{prefix}_material_strength_sum"] = float(_strength(material).sum()) if not material.empty else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def alpha_block_summary(
    reversal_tables: dict[str, pd.DataFrame],
    blocks: tuple[tuple[str, float, float], ...] = (
        ("0.010-0.019", 0.010, 0.019),
        ("0.020-0.040", 0.020, 0.040),
        ("0.041-0.055", 0.041, 0.055),
        ("0.056-0.100", 0.056, 0.100),
    ),
) -> pd.DataFrame:
    keys: set[tuple[str, str]] = set()
    for df in reversal_tables.values():
        if not df.empty:
            keys.update((str(s), str(p)) for s, p in df[["split", "policy"]].drop_duplicates().itertuples(index=False))
    rows: list[dict[str, Any]] = []
    for block_name, lo, hi in blocks:
        for split, policy in sorted(keys):
            row: dict[str, Any] = {
                "split": split,
                "policy": policy,
                "alpha_block": block_name,
                "alpha_min": float(lo),
                "alpha_max": float(hi),
            }
            for prefix, df in reversal_tables.items():
                if df.empty:
                    sub = pd.DataFrame()
                else:
                    alpha = pd.to_numeric(df["alpha"], errors="coerce")
                    sub = df[
                        (df["split"].astype(str) == split)
                        & (df["policy"].astype(str) == policy)
                        & (alpha >= float(lo) - 1e-12)
                        & (alpha <= float(hi) + 1e-12)
                    ].copy()
                supported = sub[sub["supported"]] if not sub.empty else pd.DataFrame()
                sign_active = supported[_sign_active(supported)] if not supported.empty else pd.DataFrame()
                active_cells = int(supported.shape[0])
                sign_count = int(sign_active.shape[0])
                conflict_sum = float(_strength(supported).sum()) if active_cells else 0.0
                row[f"{prefix}_active_alpha_cells"] = active_cells
                row[f"{prefix}_sign_reversal_count"] = sign_count
                row[f"{prefix}_sign_reversal_rate"] = float(sign_count / active_cells) if active_cells else float("nan")
                row[f"{prefix}_conflict_strength_sum"] = conflict_sum
                row[f"{prefix}_conflict_strength_mean_over_active_alpha_cells"] = float(conflict_sum / active_cells) if active_cells else float("nan")
                row[f"{prefix}_reversal_count"] = sign_count
                row[f"{prefix}_reversal_rate"] = float(sign_count / active_cells) if active_cells else float("nan")
                row[f"{prefix}_reversal_strength"] = conflict_sum
            rows.append(row)
    return pd.DataFrame(rows)


def alignment_components(curve: pd.DataFrame, local: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Estimate symmetric residual and variance components.

    The symmetric cell law is bar pi(z)=0.5{pi_a(z)+pi_b(z)}.
    For IPW curves, pi and V are computed from the standardized weighted law.
    """
    groups = sorted(int(x) for x in curve["group"].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, got {groups}")
    a, b = groups
    rows: list[dict[str, Any]] = []
    for (split, policy, alpha), agg_sub in curve.groupby(["split", "policy", "alpha"], sort=True):
        local_sub = local[
            (local["split"] == split)
            & (local["policy"] == policy)
            & np.isclose(local["alpha"].to_numpy(float), float(alpha))
        ].copy()
        ga = agg_sub[agg_sub["group"] == a]
        gb = agg_sub[agg_sub["group"] == b]
        if ga.empty or gb.empty or local_sub.empty:
            continue
        ga_row = ga.iloc[0]
        gb_row = gb.iloc[0]
        total_a = float(ga_row["weight_sum"])
        total_b = float(gb_row["weight_sum"])
        if not np.isfinite(total_a) or total_a <= 0:
            total_a = float(local_sub[local_sub["group"] == a]["weight_sum"].sum())
        if not np.isfinite(total_b) or total_b <= 0:
            total_b = float(local_sub[local_sub["group"] == b]["weight_sum"].sum())
        by_a = local_sub[local_sub["group"] == a].set_index("audit")
        by_b = local_sub[local_sub["group"] == b].set_index("audit")
        audits = sorted(set(int(x) for x in by_a.index).intersection(int(x) for x in by_b.index))
        vals: list[dict[str, float]] = []
        for audit in audits:
            ra = by_a.loc[audit]
            rb = by_b.loc[audit]
            wa = float(ra["weight_sum"])
            wb = float(rb["weight_sum"])
            va = float(ra["avg_set_size"])
            vb = float(rb["avg_set_size"])
            if not all(np.isfinite(x) for x in [wa, wb, va, vb]) or total_a <= 0 or total_b <= 0:
                continue
            vals.append({
                "audit": float(audit),
                "pi_a": wa / total_a,
                "pi_b": wb / total_b,
                "v_a": va,
                "v_b": vb,
                "delta_z": va - vb,
                "n_a_z": float(ra.get("n", np.nan)),
                "n_b_z": float(rb.get("n", np.nan)),
                "ess_a_z": float(ra.get("ess", np.nan)),
                "ess_b_z": float(rb.get("ess", np.nan)),
            })
        if not vals:
            continue
        cell = pd.DataFrame(vals)
        sum_pa = float(cell["pi_a"].sum())
        sum_pb = float(cell["pi_b"].sum())
        if sum_pa <= 0 or sum_pb <= 0:
            continue
        cell["pi_a"] = cell["pi_a"] / sum_pa
        cell["pi_b"] = cell["pi_b"] / sum_pb
        cell["bar_pi"] = 0.5 * (cell["pi_a"] + cell["pi_b"])
        bar_sum = float(cell["bar_pi"].sum())
        if bar_sum <= 0:
            continue
        cell["bar_pi"] = cell["bar_pi"] / bar_sum
        aggregate_delta = float(ga_row["avg_set_size"] - gb_row["avg_set_size"])
        residual = 0.5 * float(np.sum((cell["pi_a"] - cell["pi_b"]) * (cell["v_a"] + cell["v_b"])))
        local_mean = float(np.sum(cell["bar_pi"] * cell["delta_z"]))
        local_variability = float(np.sum(cell["bar_pi"] * (cell["delta_z"] - local_mean) ** 2))
        alignment_loss = float(np.sum(cell["bar_pi"] * (aggregate_delta - cell["delta_z"]) ** 2))
        conflict_strength_weighted = float(np.sum(cell["bar_pi"] * np.maximum(-aggregate_delta * cell["delta_z"], 0.0)))
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            f"aggregate_gap_{prefix}": aggregate_delta,
            f"composition_residual_{prefix}": residual,
            f"mean_local_gap_{prefix}": local_mean,
            f"local_gap_variability_{prefix}": local_variability,
            f"alignment_loss_{prefix}": alignment_loss,
            f"alignment_loss_decomp_{prefix}": local_variability + residual ** 2,
            f"alignment_loss_identity_error_{prefix}": alignment_loss - (local_variability + residual ** 2),
            f"decomposition_identity_error_{prefix}": aggregate_delta - (local_mean + residual),
            f"continuous_conflict_strength_{prefix}": conflict_strength_weighted,
            f"num_budget_cells_{prefix}": int(cell.shape[0]),
            f"min_n_a_z_{prefix}": float(cell["n_a_z"].min()),
            f"min_n_b_z_{prefix}": float(cell["n_b_z"].min()),
            f"min_ess_a_z_{prefix}": float(cell["ess_a_z"].min()),
            f"min_ess_b_z_{prefix}": float(cell["ess_b_z"].min()),
            "weighting_convention": "symmetric_group_cell_law_with_pooled_group_prevalence",
            "budget_cell_filter": "common_audit_cells_with_finite_group_conditional_means",
        })
    return pd.DataFrame(rows)


def alignment_budget(raw_components: pd.DataFrame, ipw_components: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw_components.empty:
        return pd.DataFrame()
    if ipw_components is None or ipw_components.empty:
        return raw_components.copy()
    merged = raw_components.merge(ipw_components, on=["split", "policy", "alpha"], how="inner", suffixes=("", "_ipwdup"))
    for col in ["weighting_convention", "budget_cell_filter"]:
        dup = f"{col}_ipwdup"
        if dup in merged:
            merged = merged.drop(columns=[dup])
    merged["composition_reduction"] = (
        merged["composition_residual_raw"] ** 2 - merged["composition_residual_ipw"] ** 2
    )
    merged["variability_price"] = (
        merged["local_gap_variability_ipw"] - merged["local_gap_variability_raw"]
    )
    merged["alignment_budget_gain"] = merged["composition_reduction"] - merged["variability_price"]
    merged["budget_positive"] = merged["alignment_budget_gain"] > 0
    merged["alignment_loss_reduction"] = merged["alignment_loss_raw"] - merged["alignment_loss_ipw"]
    return merged


def alignment_budget_summary(budget: pd.DataFrame) -> pd.DataFrame:
    if budget.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (split, policy), sub in budget.groupby(["split", "policy"], sort=True):
        row: dict[str, Any] = {
            "split": split,
            "policy": policy,
            "alpha_count": int(sub.shape[0]),
            "weighting_convention": str(sub["weighting_convention"].iloc[0]) if "weighting_convention" in sub else "",
        }
        for col in [
            "composition_residual_raw",
            "composition_residual_ipw",
            "local_gap_variability_raw",
            "local_gap_variability_ipw",
            "continuous_conflict_strength_raw",
            "continuous_conflict_strength_ipw",
            "composition_reduction",
            "variability_price",
            "alignment_budget_gain",
            "alignment_loss_reduction",
        ]:
            if col in sub:
                vals = pd.to_numeric(sub[col], errors="coerce")
                row[f"integrated_{col}"] = float(vals.sum())
                row[f"mean_{col}"] = float(vals.mean())
        if "budget_positive" in sub:
            row["positive_budget_alpha_fraction"] = float(sub["budget_positive"].mean())
            row["positive_budget_alpha_count"] = int(sub["budget_positive"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def probability_profile_features(probs: np.ndarray, top_k: int = 3) -> tuple[np.ndarray, list[str]]:
    """Build deployment-safe H from frozen softmax probabilities.

    No true label, audit cell, set size, conformal threshold, or alpha-specific
    outcome enters these features.
    """
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    n, k = p.shape
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    kk = max(1, min(int(top_k), k))
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for r in range(kk):
        one = np.zeros((n, k), dtype=float)
        one[np.arange(n), order[:, r]] = 1.0
        blocks.append(one)
        names.extend([f"top{r + 1}_label_{j}" for j in range(k)])
    blocks.append(sorted_p[:, :kk])
    names.extend([f"p_top{r + 1}" for r in range(kk)])
    p1 = sorted_p[:, 0]
    p2 = sorted_p[:, 1] if k >= 2 else np.zeros(n)
    p3 = sorted_p[:, 2] if k >= 3 else np.zeros(n)
    entropy = -np.sum(p * np.log(p), axis=1)
    entropy_norm = entropy / max(math.log(k), 1e-12)
    cont = np.column_stack([entropy_norm, p1 - p2, p2 - p3, sorted_p[:, : min(3, k)].sum(axis=1)])
    blocks.append(cont)
    names.extend(["entropy_norm", "margin12", "margin23", "top3_mass"])
    X = np.column_stack(blocks).astype(float)
    X[~np.isfinite(X)] = 0.0
    return X, names


def multiclass_overlap_weights(propensities: np.ndarray, z: np.ndarray, clip: float) -> np.ndarray:
    e = np.clip(np.asarray(propensities, dtype=float), float(clip), 1.0)
    e = e / e.sum(axis=1, keepdims=True)
    z = np.asarray(z, dtype=int)
    h = 1.0 / np.sum(1.0 / e, axis=1)
    return h / e[np.arange(len(z)), z]


def multiclass_ipw_weights(propensities: np.ndarray, z: np.ndarray, class_priors: np.ndarray, clip: float) -> np.ndarray:
    e = np.clip(np.asarray(propensities, dtype=float), float(clip), 1.0)
    e = e / e.sum(axis=1, keepdims=True)
    z = np.asarray(z, dtype=int)
    priors = np.asarray(class_priors, dtype=float)
    return priors[z] / e[np.arange(len(z)), z]


def apply_weight_caps(weights: np.ndarray, trim_cap: float | None, max_weight: float | None) -> np.ndarray:
    w = np.asarray(weights, dtype=float).copy()
    w[~np.isfinite(w)] = np.nan
    if trim_cap is not None and np.isfinite(float(trim_cap)):
        w = np.minimum(w, float(trim_cap))
    if max_weight is not None and float(max_weight) > 0:
        w = np.minimum(w, float(max_weight))
    finite = w[np.isfinite(w) & (w >= 0)]
    med = float(np.median(finite)) if finite.size else 1.0
    if med > 0:
        w = w / med
    if max_weight is not None and float(max_weight) > 0:
        w = np.minimum(w, float(max_weight))
    w[~np.isfinite(w)] = 0.0
    return w


def raw_multiclass_weights(
    propensities: np.ndarray,
    z: np.ndarray,
    scheme: str,
    clip: float,
    class_priors: np.ndarray,
) -> np.ndarray:
    scheme = str(scheme).lower()
    if scheme == "overlap":
        return multiclass_overlap_weights(propensities, z, clip)
    if scheme == "ipw":
        return multiclass_ipw_weights(propensities, z, class_priors, clip)
    raise ValueError("weight_scheme must be overlap or ipw")


def weight_diagnostics(split_name: str, g: np.ndarray, z: np.ndarray, w: np.ndarray, meta: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [None] + sorted(int(x) for x in np.unique(g))
    audits = [None] + sorted(int(x) for x in np.unique(z))
    for group in groups:
        for audit in audits:
            if group is not None and audit is not None:
                continue
            mask = np.ones(len(w), dtype=bool)
            label = "overall"
            if group is not None:
                mask &= np.asarray(g, dtype=int) == group
                label = f"group={group_name(meta, group)}"
            if audit is not None:
                mask &= np.asarray(z, dtype=int) == audit
                label = f"audit={audit_name(meta, audit)}"
            ww = np.asarray(w[mask], dtype=float)
            mean_w = float(np.mean(ww)) if ww.size else np.nan
            std_w = float(np.std(ww, ddof=0)) if ww.size else np.nan
            cv_w = float(std_w / mean_w) if ww.size and mean_w > 0 else np.nan
            rows.append({
                "split": split_name,
                "stratum": label,
                "n": int(mask.sum()),
                "weight_sum": float(np.sum(ww)) if ww.size else 0.0,
                "ess": effective_sample_size(ww),
                "ess_fraction": float(effective_sample_size(ww) / ww.size) if ww.size else np.nan,
                "mean": mean_w,
                "std": std_w,
                "cv": cv_w,
                "min": float(np.min(ww)) if ww.size else np.nan,
                "p01": float(np.quantile(ww, 0.01)) if ww.size else np.nan,
                "p05": float(np.quantile(ww, 0.05)) if ww.size else np.nan,
                "p50": float(np.median(ww)) if ww.size else np.nan,
                "p90": float(np.quantile(ww, 0.90)) if ww.size else np.nan,
                "p95": float(np.quantile(ww, 0.95)) if ww.size else np.nan,
                "p99": float(np.quantile(ww, 0.99)) if ww.size else np.nan,
                "max": float(np.max(ww)) if ww.size else np.nan,
            })
    return pd.DataFrame(rows)
