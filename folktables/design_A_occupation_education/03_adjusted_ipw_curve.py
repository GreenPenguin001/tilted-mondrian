#!/usr/bin/env python3
"""Stage 3: group-propensity overlap/IPW adjusted curves."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from curve_utils import (
    aggregate_gap,
    alignment_budget,
    alignment_budget_summary,
    alignment_components,
    alignment_summary,
    alpha_block_summary,
    alpha_grid_from_range,
    apply_weight_caps,
    audit_name,
    build_thresholds,
    effective_sample_size,
    group_curve,
    group_name,
    conflict_strength_by_alpha,
    conflict_strength_summary,
    jdump,
    jload,
    local_curve_by_z,
    local_gap,
    mkdir,
    parse_alpha_grid,
    persistent_reversals,
    probability_profile_features,
    read_yaml,
    reversal_rate_summary,
    reversal_counts_by_alpha,
    simpson_reversal_table,
    material_strength_counts,
    weight_diagnostics,
    weighted_average,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "design_config.yaml"


@dataclass
class PropensityFit:
    model: Any
    feature_names: list[str]
    hyperparameters: dict[str, Any]


@dataclass
class AdjustedRunResult:
    output_dir: Path | None
    eval_splits: list[str]
    diagnostic_splits: list[str]
    feature_names: list[str]
    prop_diag: pd.DataFrame
    weight_diag: pd.DataFrame
    balance: pd.DataFrame
    audit_imbalance: pd.DataFrame
    aggregate: pd.DataFrame
    local: pd.DataFrame
    gap: pd.DataFrame
    gap_z: pd.DataFrame
    reversals: pd.DataFrame
    counts: pd.DataFrame
    persistent: pd.DataFrame
    count_comparison: pd.DataFrame
    persistent_comparison: pd.DataFrame
    alignment: pd.DataFrame
    model_summary: dict[str, Any]
    ipw_summary: dict[str, Any]
    applied_weight_summaries: dict[str, Any]


def resolve_path(path: str | Path, base: Path = SCRIPT_DIR) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    p.add_argument("--split", choices=["repair", "test", "all"], default="test")
    p.add_argument("--alpha-grid", default=None)
    p.add_argument("--alpha_min", type=float, default=0.02)
    p.add_argument("--alpha_max", type=float, default=0.10)
    p.add_argument("--alpha_step", type=float, default=0.005)
    p.add_argument("--alpha_grid_name", default=None)
    p.add_argument("--clip", type=float, default=None)
    p.add_argument("--trim-quantile", type=float, default=None)
    p.add_argument("--max-weight", type=float, default=None)
    p.add_argument("--weight-scheme", choices=["overlap", "ipw"], default=None)
    p.add_argument("--top-k-profile", type=int, default=None)
    p.add_argument("--n-min", type=int, default=None)
    p.add_argument("--alpha0", type=float, default=None)
    return p.parse_args()


def load_split(train_dir: Path, split: str) -> dict[str, np.ndarray]:
    return {
        "probs": np.load(train_dir / f"probs_{split}.npy"),
        "y": np.load(train_dir / f"y_{split}.npy"),
        "g": np.load(train_dir / f"g_{split}.npy"),
        "z": np.load(train_dir / f"z_{split}.npy"),
    }


def split_list(requested: str) -> list[str]:
    return ["repair", "test"] if requested == "all" else [requested]


def alpha_grid_for_args(args: argparse.Namespace, conformal_cfg: dict[str, Any]) -> list[float]:
    if args.alpha_grid is not None:
        return parse_alpha_grid(args.alpha_grid)
    if (
        args.alpha_min == 0.02
        and args.alpha_max == 0.10
        and args.alpha_step == 0.005
        and "alpha_grid" in conformal_cfg
    ):
        return parse_alpha_grid(conformal_cfg.get("alpha_grid", "dense_safe"))
    return alpha_grid_from_range(args.alpha_min, args.alpha_max, args.alpha_step)


def output_dir_for_grid(year_dir: Path, base_name: str, alpha_grid_name: str | None) -> Path:
    if alpha_grid_name is None or str(alpha_grid_name).strip() == "":
        return year_dir / base_name
    suffix = str(alpha_grid_name).strip()
    if "/" in suffix or "\\" in suffix:
        raise ValueError("--alpha_grid_name may not contain path separators")
    return year_dir / f"{base_name}_{suffix}"


def make_logistic_propensity(seed: int) -> tuple[Any, dict[str, Any]]:
    params = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": seed,
    }
    estimator = LogisticRegression(**params)
    return make_pipeline(StandardScaler(), estimator), params


def final_estimator(model: Any) -> Any:
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def positive_class_probability(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(int(x) for x in final_estimator(model).classes_)
    if 1 not in classes:
        raise ValueError("Fitted propensity model has no class 1 column")
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def fit_group_propensity(
    repair: dict[str, np.ndarray],
    top_k: int,
    seed: int,
) -> tuple[PropensityFit, np.ndarray]:
    X, names = probability_profile_features(repair["probs"], top_k=top_k)
    g = np.asarray(repair["g"], dtype=int)
    classes = sorted(int(x) for x in np.unique(g))
    if classes != [0, 1]:
        raise ValueError(f"Need binary groups 0/1 in repair split, got {classes}")

    model, hyperparams = make_logistic_propensity(seed)
    model.fit(X, g)
    fit = PropensityFit(model=model, feature_names=names, hyperparameters=hyperparams)
    return fit, X


def predict_group_propensity(fit: PropensityFit, split: dict[str, np.ndarray], top_k: int) -> tuple[np.ndarray, np.ndarray]:
    X, _ = probability_profile_features(split["probs"], top_k=top_k)
    e = positive_class_probability(fit.model, X)
    return e, X


def raw_group_weights(e: np.ndarray, g: np.ndarray, scheme: str, clip: float, group_priors: dict[int, float]) -> np.ndarray:
    e = np.clip(np.asarray(e, dtype=float), float(clip), 1.0 - float(clip))
    g = np.asarray(g, dtype=int)
    if scheme == "overlap":
        return np.where(g == 1, 1.0 - e, e)
    if scheme == "ipw":
        pi0 = float(group_priors[0])
        pi1 = float(group_priors[1])
        return np.where(g == 1, pi1 / e, pi0 / (1.0 - e))
    raise ValueError("weight scheme must be overlap or ipw")


def propensity_diagnostics(
    split_name: str,
    g: np.ndarray,
    e: np.ndarray,
    clip: float,
) -> dict[str, Any]:
    g = np.asarray(g, dtype=int)
    e = np.clip(np.asarray(e, dtype=float), float(clip), 1.0 - float(clip))
    pred = (e >= 0.5).astype(int)
    out: dict[str, Any] = {
        "split": split_name,
        "n": int(len(g)),
        "target": "G",
        "target_name": "SEX_binary_Female_indicator",
        "positive_group": 1,
        "positive_group_name": "Female",
        "propensity_model": "logistic",
        "accuracy": float(accuracy_score(g, pred)) if len(g) else float("nan"),
        "mean_e_g1": float(np.mean(e)) if len(e) else float("nan"),
        "min_e_g1": float(np.min(e)) if len(e) else float("nan"),
        "p05_e_g1": float(np.quantile(e, 0.05)) if len(e) else float("nan"),
        "p50_e_g1": float(np.quantile(e, 0.50)) if len(e) else float("nan"),
        "p95_e_g1": float(np.quantile(e, 0.95)) if len(e) else float("nan"),
        "max_e_g1": float(np.max(e)) if len(e) else float("nan"),
    }
    try:
        out["auc"] = float(roc_auc_score(g, e))
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(g, np.column_stack([1.0 - e, e]), labels=[0, 1]))
    except ValueError:
        out["log_loss"] = float("nan")
    try:
        out["brier"] = float(brier_score_loss(g, e))
    except ValueError:
        out["brier"] = float("nan")
    return out


def weighted_mean_for_mask(x: np.ndarray, mask: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(x[mask])) if mask.any() else float("nan")
    return weighted_average(x[mask], np.asarray(weights)[mask])


def profile_balance_smd(
    split_name: str,
    X: np.ndarray,
    feature_names: list[str],
    g: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    g = np.asarray(g, dtype=int)
    weights = np.asarray(weights, dtype=float)
    m0 = g == 0
    m1 = g == 1
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(feature_names):
        x = np.asarray(X[:, j], dtype=float)
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        mean0 = weighted_mean_for_mask(x, m0, None)
        mean1 = weighted_mean_for_mask(x, m1, None)
        wmean0 = weighted_mean_for_mask(x, m0, weights)
        wmean1 = weighted_mean_for_mask(x, m1, weights)
        rows.append({
            "split": split_name,
            "feature": name,
            "mean_g0_unweighted": mean0,
            "mean_g1_unweighted": mean1,
            "smd_unweighted_g1_minus_g0": float((mean1 - mean0) / sd),
            "mean_g0_weighted": wmean0,
            "mean_g1_weighted": wmean1,
            "smd_weighted_g1_minus_g0": float((wmean1 - wmean0) / sd),
        })
    return pd.DataFrame(rows)


def summarize_smd(balance: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if balance.empty:
        return out
    for split, sub in balance.groupby("split", sort=True):
        u = pd.to_numeric(sub["smd_unweighted_g1_minus_g0"], errors="coerce").abs()
        w = pd.to_numeric(sub["smd_weighted_g1_minus_g0"], errors="coerce").abs()
        out[str(split)] = {
            "feature_count": int(len(sub)),
            "max_abs_smd_unweighted": float(u.max()),
            "mean_abs_smd_unweighted": float(u.mean()),
            "max_abs_smd_weighted": float(w.max()),
            "mean_abs_smd_weighted": float(w.mean()),
        }
    return out


def posthoc_audit_cell_imbalance(
    split_name: str,
    z: np.ndarray,
    g: np.ndarray,
    weights: np.ndarray,
    label_maps: dict[str, Any],
) -> pd.DataFrame:
    z = np.asarray(z, dtype=int)
    g = np.asarray(g, dtype=int)
    weights = np.asarray(weights, dtype=float)
    audits = sorted(int(x) for x in np.unique(z))
    rows: list[dict[str, Any]] = []
    p0: list[float] = []
    p1: list[float] = []
    wp0: list[float] = []
    wp1: list[float] = []
    for audit in audits:
        m0 = g == 0
        m1 = g == 1
        za = z == audit
        p_g0 = float(np.mean(za[m0])) if m0.any() else float("nan")
        p_g1 = float(np.mean(za[m1])) if m1.any() else float("nan")
        wp_g0 = weighted_average(za[m0].astype(float), weights[m0])
        wp_g1 = weighted_average(za[m1].astype(float), weights[m1])
        p0.append(p_g0)
        p1.append(p_g1)
        wp0.append(wp_g0)
        wp1.append(wp_g1)
        rows.append({
            "split": split_name,
            "audit": int(audit),
            "audit_name": audit_name(label_maps, audit),
            "p_z_g0_unweighted": p_g0,
            "p_z_g1_unweighted": p_g1,
            "diff_g1_minus_g0_unweighted": float(p_g1 - p_g0),
            "p_z_g0_weighted": wp_g0,
            "p_z_g1_weighted": wp_g1,
            "diff_g1_minus_g0_weighted": float(wp_g1 - wp_g0),
        })
    tv_unweighted = 0.5 * float(np.nansum(np.abs(np.asarray(p1) - np.asarray(p0))))
    tv_weighted = 0.5 * float(np.nansum(np.abs(np.asarray(wp1) - np.asarray(wp0))))
    for row in rows:
        row["tv_z_by_group_unweighted"] = tv_unweighted
        row["tv_z_by_group_weighted"] = tv_weighted
    return pd.DataFrame(rows)


def summarize_audit_imbalance(imbalance: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if imbalance.empty:
        return out
    for split, sub in imbalance.groupby("split", sort=True):
        out[str(split)] = {
            "tv_z_by_group_unweighted": float(pd.to_numeric(sub["tv_z_by_group_unweighted"], errors="coerce").max()),
            "tv_z_by_group_weighted": float(pd.to_numeric(sub["tv_z_by_group_weighted"], errors="coerce").max()),
            "max_abs_cell_diff_unweighted": float(pd.to_numeric(sub["diff_g1_minus_g0_unweighted"], errors="coerce").abs().max()),
            "max_abs_cell_diff_weighted": float(pd.to_numeric(sub["diff_g1_minus_g0_weighted"], errors="coerce").abs().max()),
        }
    return out


def compare_counts(raw_path: Path, ipw_counts: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    if "split" in raw:
        raw = raw[raw["split"].isin(splits)].copy()
    merged = raw.merge(ipw_counts, on=["split", "policy", "alpha"], how="outer", suffixes=("_raw", "_ipw"))
    fill_cols = [
        "sign_reversal_count_raw",
        "sign_reversal_count_ipw",
        "num_reversal_cells_raw",
        "num_reversal_cells_ipw",
        "conflict_strength_sum_raw",
        "conflict_strength_sum_ipw",
        "total_reversal_strength_raw",
        "total_reversal_strength_ipw",
    ]
    for col in fill_cols:
        if col in merged:
            merged[col] = merged[col].fillna(0)
    for base in ["sign_reversal_count", "conflict_strength_sum"]:
        raw_col = f"{base}_raw"
        ipw_col = f"{base}_ipw"
        if {raw_col, ipw_col}.issubset(merged.columns):
            merged[f"delta_{base}_ipw_minus_raw"] = merged[ipw_col] - merged[raw_col]
    if {"num_reversal_cells_raw", "num_reversal_cells_ipw"}.issubset(merged.columns):
        merged["delta_reversal_cells_ipw_minus_raw"] = merged["num_reversal_cells_ipw"] - merged["num_reversal_cells_raw"]
    if {"total_reversal_strength_raw", "total_reversal_strength_ipw"}.issubset(merged.columns):
        merged["delta_reversal_strength_ipw_minus_raw"] = merged["total_reversal_strength_ipw"] - merged["total_reversal_strength_raw"]
    return merged


def compare_persistent(raw_path: Path, ipw_persistent: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    if "split" in raw:
        raw = raw[raw["split"].isin(splits)].copy()
    merged = raw.merge(ipw_persistent, on=["split", "policy", "audit", "audit_name"], how="outer", suffixes=("_raw", "_ipw"))
    fill_cols = [
        "num_sign_reversal_alpha_raw",
        "num_sign_reversal_alpha_ipw",
        "num_reversal_alpha_raw",
        "num_reversal_alpha_ipw",
        "total_sign_conflict_strength_raw",
        "total_sign_conflict_strength_ipw",
        "total_reversal_strength_raw",
        "total_reversal_strength_ipw",
    ]
    for col in fill_cols:
        if col in merged:
            merged[col] = merged[col].fillna(0)
    for base in ["num_sign_reversal_alpha", "total_sign_conflict_strength"]:
        raw_col = f"{base}_raw"
        ipw_col = f"{base}_ipw"
        if {raw_col, ipw_col}.issubset(merged.columns):
            merged[f"delta_{base}_ipw_minus_raw"] = merged[ipw_col] - merged[raw_col]
    if {"num_reversal_alpha_raw", "num_reversal_alpha_ipw"}.issubset(merged.columns):
        merged["delta_reversal_alpha_ipw_minus_raw"] = merged["num_reversal_alpha_ipw"] - merged["num_reversal_alpha_raw"]
    if {"total_reversal_strength_raw", "total_reversal_strength_ipw"}.issubset(merged.columns):
        merged["delta_reversal_strength_ipw_minus_raw"] = merged["total_reversal_strength_ipw"] - merged["total_reversal_strength_raw"]
    return merged


def summarize_weights(weights: np.ndarray) -> dict[str, Any]:
    w = np.asarray(weights, dtype=float)
    mean_w = float(np.mean(w)) if len(w) else float("nan")
    std_w = float(np.std(w, ddof=0)) if len(w) else float("nan")
    return {
        "n": int(len(w)),
        "sum": float(np.sum(w)) if len(w) else float("nan"),
        "mean": mean_w,
        "std": std_w,
        "cv": float(std_w / mean_w) if len(w) and mean_w > 0 else float("nan"),
        "ess": effective_sample_size(w),
        "ess_fraction": float(effective_sample_size(w) / len(w)) if len(w) else float("nan"),
        "min": float(np.min(w)) if len(w) else float("nan"),
        "p01": float(np.quantile(w, 0.01)) if len(w) else float("nan"),
        "p05": float(np.quantile(w, 0.05)) if len(w) else float("nan"),
        "median": float(np.median(w)) if len(w) else float("nan"),
        "p95": float(np.quantile(w, 0.95)) if len(w) else float("nan"),
        "p99": float(np.quantile(w, 0.99)) if len(w) else float("nan"),
        "max": float(np.max(w)) if len(w) else float("nan"),
    }


def summarize_weight_application(
    split: str,
    raw_weights: np.ndarray,
    final_weights: np.ndarray,
    trim_cap: float | None,
    max_weight: float | None,
) -> dict[str, Any]:
    out = summarize_weights(final_weights)
    rw = np.asarray(raw_weights, dtype=float)
    fw = np.asarray(final_weights, dtype=float)
    out["split"] = split
    out["raw_weight_mean_before_caps"] = float(np.mean(rw)) if len(rw) else float("nan")
    out["raw_weight_p95_before_caps"] = float(np.quantile(rw, 0.95)) if len(rw) else float("nan")
    out["raw_weight_p99_before_caps"] = float(np.quantile(rw, 0.99)) if len(rw) else float("nan")
    out["raw_weight_max_before_caps"] = float(np.max(rw)) if len(rw) else float("nan")
    out["trim_cap"] = float(trim_cap) if trim_cap is not None else None
    out["trim_fraction"] = float(np.mean(rw > float(trim_cap))) if trim_cap is not None and len(rw) else 0.0
    out["max_weight"] = float(max_weight) if max_weight is not None else None
    out["cap_fraction"] = float(np.mean(fw >= float(max_weight) - 1e-12)) if max_weight is not None and len(fw) else 0.0
    return out


def stability_summary_table(
    weight_summaries: dict[str, Any],
    balance_summary: dict[str, Any],
    audit_imbalance_summary: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, vals in weight_summaries.items():
        row = {"split": split}
        row.update(vals)
        bal = balance_summary.get(split, {})
        imb = audit_imbalance_summary.get(split, {})
        row["max_abs_smd_before"] = bal.get("max_abs_smd_unweighted", float("nan"))
        row["max_abs_smd_after"] = bal.get("max_abs_smd_weighted", float("nan"))
        row["mean_abs_smd_before"] = bal.get("mean_abs_smd_unweighted", float("nan"))
        row["mean_abs_smd_after"] = bal.get("mean_abs_smd_weighted", float("nan"))
        row["audit_cell_tv_before"] = imb.get("tv_z_by_group_unweighted", float("nan"))
        row["audit_cell_tv_after"] = imb.get("tv_z_by_group_weighted", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def run_adjusted_model(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    out_root: Path,
    train_dir: Path,
    raw_dir: Path,
    label_maps: dict[str, Any],
    split_meta: dict[str, Any],
    write_dir: Path | None,
) -> AdjustedRunResult:
    start = time.time()
    conformal_cfg = cfg.get("conformal", {})
    diagnostics_cfg = cfg.get("diagnostics", {})
    prop_cfg = cfg.get("propensity", {})
    score = str(conformal_cfg.get("score", "simple"))
    alphas = alpha_grid_for_args(args, conformal_cfg)
    n_min = int(args.n_min if args.n_min is not None else diagnostics_cfg.get("n_min", 200))
    alpha0 = float(args.alpha0 if args.alpha0 is not None else diagnostics_cfg.get("alpha0", 0.10))
    clip = float(args.clip if args.clip is not None else prop_cfg.get("clip", 0.03))
    trim_quantile = float(args.trim_quantile if args.trim_quantile is not None else prop_cfg.get("trim_quantile", 0.95))
    max_weight = float(args.max_weight if args.max_weight is not None else prop_cfg.get("max_weight", 3.0))
    scheme = str(args.weight_scheme if args.weight_scheme is not None else prop_cfg.get("weight_scheme", "overlap")).lower()
    top_k_profile = int(args.top_k_profile if args.top_k_profile is not None else prop_cfg.get("top_k", 3))
    seed = int(cfg.get("random_seed", 20260609))

    cal = load_split(train_dir, "cal")
    repair = load_split(train_dir, "repair")
    threshold_df, thresholds = build_thresholds(cal["probs"], cal["y"], alphas, score)
    if write_dir is not None:
        threshold_df.to_csv(write_dir / "conformal_thresholds_used.csv", index=False)

    print("Fitting logistic propensity e(H)=P(G=1 | H) on repair split", flush=True)
    prop_fit, repair_H = fit_group_propensity(repair, top_k_profile, seed)
    repair_e, repair_H = predict_group_propensity(prop_fit, repair, top_k_profile)
    group_priors = {
        0: float(np.mean(repair["g"] == 0)),
        1: float(np.mean(repair["g"] == 1)),
    }
    raw_repair_weights = raw_group_weights(repair_e, repair["g"], scheme, clip, group_priors)
    trim_cap = None
    if trim_quantile < 1.0:
        finite = raw_repair_weights[np.isfinite(raw_repair_weights)]
        trim_cap = float(np.quantile(finite, trim_quantile)) if finite.size else None

    eval_splits = split_list(args.split)
    diagnostic_splits = sorted(set(["repair"] + eval_splits), key=lambda x: {"repair": 0, "test": 1}.get(x, 2))
    aggregate_tables: list[pd.DataFrame] = []
    local_tables: list[pd.DataFrame] = []
    prop_diag_rows: list[dict[str, Any]] = []
    weight_diag_tables: list[pd.DataFrame] = []
    balance_tables: list[pd.DataFrame] = []
    audit_imbalance_tables: list[pd.DataFrame] = []
    applied_weight_summaries: dict[str, Any] = {}

    split_cache: dict[str, dict[str, np.ndarray]] = {"repair": repair}
    H_cache: dict[str, np.ndarray] = {"repair": repair_H}
    e_cache: dict[str, np.ndarray] = {"repair": repair_e}
    weight_cache: dict[str, np.ndarray] = {}

    for split in diagnostic_splits:
        data = split_cache.get(split)
        if data is None:
            data = load_split(train_dir, split)
            split_cache[split] = data
        e = e_cache.get(split)
        H = H_cache.get(split)
        if e is None or H is None:
            e, H = predict_group_propensity(prop_fit, data, top_k_profile)
            e_cache[split] = e
            H_cache[split] = H
        raw_w = raw_group_weights(e, data["g"], scheme, clip, group_priors)
        weights = apply_weight_caps(raw_w, trim_cap=trim_cap, max_weight=max_weight)
        weight_cache[split] = weights
        prop_diag_rows.append(propensity_diagnostics(split, data["g"], e, clip))
        weight_diag_tables.append(weight_diagnostics(split, data["g"], data["z"], weights, label_maps))
        balance_tables.append(profile_balance_smd(split, H, prop_fit.feature_names, data["g"], weights))
        audit_imbalance_tables.append(posthoc_audit_cell_imbalance(split, data["z"], data["g"], weights, label_maps))
        applied_weight_summaries[split] = summarize_weight_application(split, raw_w, weights, trim_cap, max_weight)

    for split in eval_splits:
        data = split_cache[split]
        weights = weight_cache[split]
        aggregate_tables.append(
            group_curve(
                split, data["probs"], data["y"], data["g"], data["z"],
                thresholds, alphas, label_maps, weights=weights, score=score, estimand=f"{scheme}_group_propensity_adjusted",
            )
        )
        local_tables.append(
            local_curve_by_z(
                split, data["probs"], data["y"], data["g"], data["z"],
                thresholds, alphas, label_maps, weights=weights, score=score, estimand=f"{scheme}_group_propensity_adjusted",
            )
        )

    prop_diag = pd.DataFrame(prop_diag_rows)
    weight_diag = pd.concat(weight_diag_tables, ignore_index=True) if weight_diag_tables else pd.DataFrame()
    balance = pd.concat(balance_tables, ignore_index=True) if balance_tables else pd.DataFrame()
    audit_imbalance = pd.concat(audit_imbalance_tables, ignore_index=True) if audit_imbalance_tables else pd.DataFrame()
    aggregate = pd.concat(aggregate_tables, ignore_index=True)
    local = pd.concat(local_tables, ignore_index=True)
    gap = aggregate_gap(aggregate)
    gap_z = local_gap(local)
    reversals = simpson_reversal_table(gap, gap_z, n_min=n_min)
    counts = reversal_counts_by_alpha(reversals)
    persistent = persistent_reversals(reversals, alpha0=alpha0)

    count_comparison = compare_counts(raw_dir / "reversal_counts_by_alpha.csv", counts, eval_splits)
    persistent_comparison = compare_persistent(raw_dir / "persistent_reversals.csv", persistent, eval_splits)
    raw_reversals = pd.read_csv(raw_dir / "simpson_reversals.csv")
    if "split" in raw_reversals:
        raw_reversals = raw_reversals[raw_reversals["split"].isin(eval_splits)].copy()
    raw_align = alignment_summary(raw_reversals, "raw")
    ipw_align = alignment_summary(reversals, "ipw")
    alignment = raw_align.merge(ipw_align, on=["split", "policy"], how="outer")
    for base in ["sign_reversal_alpha_cells", "conflict_strength_sum"]:
        raw_col = f"raw_{base}"
        ipw_col = f"ipw_{base}"
        if {raw_col, ipw_col}.issubset(alignment.columns):
            alignment[f"delta_{base}_ipw_minus_raw"] = alignment[ipw_col] - alignment[raw_col]
    if {"raw_reversal_alpha_cells", "ipw_reversal_alpha_cells"}.issubset(alignment.columns):
        alignment["delta_reversal_alpha_cells_ipw_minus_raw"] = alignment["ipw_reversal_alpha_cells"] - alignment["raw_reversal_alpha_cells"]
    if {"raw_reversal_strength", "ipw_reversal_strength"}.issubset(alignment.columns):
        alignment["delta_reversal_strength_ipw_minus_raw"] = alignment["ipw_reversal_strength"] - alignment["raw_reversal_strength"]
    rate_summary = reversal_rate_summary({"raw": raw_reversals, "ipw": reversals})
    strength_summary = conflict_strength_summary({"raw": raw_reversals, "ipw": reversals})
    strength_by_alpha = conflict_strength_by_alpha({"raw": raw_reversals, "ipw": reversals})
    material_summary = material_strength_counts({"raw": raw_reversals, "ipw": reversals})
    block_summary = alpha_block_summary({"raw": raw_reversals, "ipw": reversals})
    raw_budget_path = raw_dir / "alignment_budget_by_alpha.csv"
    if raw_budget_path.exists():
        raw_components = pd.read_csv(raw_budget_path)
        if "split" in raw_components:
            raw_components = raw_components[raw_components["split"].isin(eval_splits)].copy()
    else:
        raw_curve = pd.read_csv(raw_dir / "curve_aggregate.csv")
        raw_local = pd.read_csv(raw_dir / "curve_local_by_z.csv")
        raw_components = alignment_components(raw_curve, raw_local, "raw")
        if "split" in raw_components:
            raw_components = raw_components[raw_components["split"].isin(eval_splits)].copy()
    ipw_components = alignment_components(aggregate, local, "ipw")
    budget = alignment_budget(raw_components, ipw_components)
    budget_summary = alignment_budget_summary(budget)
    balance_summary = summarize_smd(balance)
    audit_imbalance_summary = summarize_audit_imbalance(audit_imbalance)
    stability = stability_summary_table(applied_weight_summaries, balance_summary, audit_imbalance_summary)
    association_path = train_dir / "target_z_association.json"
    target_z_association = jload(association_path) if association_path.exists() else {}

    model_summary = {
        "target": "G",
        "target_name": "SEX_binary_Female_indicator",
        "positive_group": 1,
        "positive_group_name": group_name(label_maps, 1),
        "propensity_model": "logistic",
        "model": "StandardScaler + LogisticRegression",
        "base_hyperparameters": prop_fit.hyperparameters,
        "group_priors_on_repair": {str(k): float(v) for k, v in group_priors.items()},
        "feature_mode": "safe_topk",
        "top_k_profile": top_k_profile,
        "feature_count": int(len(prop_fit.feature_names)),
        "safe_H_feature_names": prop_fit.feature_names,
        "safe_H_source": "frozen softmax prediction profile only",
        "forbidden_inputs_not_used": [
            "Y",
            "Z",
            "G_as_input_feature",
            "set_size",
            "alpha_specific_conformal_outcome",
            "threshold_artifact",
            "test_outcome",
        ],
        "fit_split": "repair",
        "evaluation_splits": eval_splits,
    }

    args_record = vars(args).copy()
    args_record["config"] = str(args.config)
    args_record["output_root"] = str(out_root)
    args_record["alpha_grid_resolved"] = alphas
    args_record["alpha_grid_count"] = len(alphas)
    args_record["alpha_min_resolved"] = min(alphas)
    args_record["alpha_max_resolved"] = max(alphas)
    args_record["n_min_resolved"] = n_min
    args_record["clip_resolved"] = clip
    args_record["trim_quantile_resolved"] = trim_quantile
    args_record["max_weight_resolved"] = max_weight
    args_record["weight_scheme_resolved"] = scheme
    args_record["weighting_convention"] = "symmetric_group_cell_law_with_pooled_group_prevalence"

    ipw_summary = {
        "stage": "adjusted_ipw_curve",
        "year": int(args.year),
        "requested_split": args.split,
        "evaluation_splits": eval_splits,
        "alpha_grid_count": len(alphas),
        "alpha_min_resolved": min(alphas),
        "alpha_max_resolved": max(alphas),
        "alpha_grid_name": args.alpha_grid_name,
        "n_min": n_min,
        "main_simpson_count": "sign_reversal_count",
        "diagnostic_classes": [
            "count",
            "strength-level count",
            "composition effect",
            "variance/weight stability",
        ],
        "development_diagnostic_splits": diagnostic_splits,
        "split_meta": split_meta,
        "propensity_target": "G",
        "propensity_target_name": "SEX_binary_Female_indicator",
        "propensity_model": "logistic",
        "propensity_features": "safe prediction-profile H from frozen softmax probabilities",
        "safe_H_feature_names": prop_fit.feature_names,
        "propensity_diagnostics": prop_diag.to_dict(orient="records"),
        "weight_scheme": scheme,
        "clip": clip,
        "trim_quantile": trim_quantile,
        "fixed_trim_cap_estimated_on_repair_raw_weight_scale": trim_cap,
        "max_weight": max_weight,
        "weight_summaries": applied_weight_summaries,
        "profile_balance_smd_summary": balance_summary,
        "posthoc_audit_cell_imbalance_summary": audit_imbalance_summary,
        "target_z_association": target_z_association,
        "raw_vs_ipw_reversal_comparison_rows": int(len(count_comparison)),
        "persistent_reversal_comparison_rows": int(len(persistent_comparison)),
        "alignment_comparison_rows": int(len(alignment)),
        "diagnostics": {
            "reversal_rate_summary": rate_summary.to_dict(orient="records"),
            "conflict_strength_summary": strength_summary.to_dict(orient="records"),
            "material_strength_counts": material_summary.to_dict(orient="records"),
            "alpha_block_summary": block_summary.to_dict(orient="records"),
            "alignment_budget_summary": budget_summary.to_dict(orient="records"),
            "stability_summary": stability.to_dict(orient="records"),
        },
        "elapsed_seconds": time.time() - start,
        "validity_boundary": "IPW changes only aggregation weights. Classifier probabilities, SIMPLE scores, unweighted calibration thresholds, and individual conformal sets are unchanged.",
        "hajek_ratio_curves": True,
        "not_weighted_conformal": True,
    }
    if not alignment.empty:
        ipw_summary["alignment_comparison"] = alignment.to_dict(orient="records")

    result = AdjustedRunResult(
        output_dir=write_dir,
        eval_splits=eval_splits,
        diagnostic_splits=diagnostic_splits,
        feature_names=prop_fit.feature_names,
        prop_diag=prop_diag,
        weight_diag=weight_diag,
        balance=balance,
        audit_imbalance=audit_imbalance,
        aggregate=aggregate,
        local=local,
        gap=gap,
        gap_z=gap_z,
        reversals=reversals,
        counts=counts,
        persistent=persistent,
        count_comparison=count_comparison,
        persistent_comparison=persistent_comparison,
        alignment=alignment,
        model_summary=model_summary,
        ipw_summary=ipw_summary,
        applied_weight_summaries=applied_weight_summaries,
    )

    if write_dir is not None:
        prop_diag.to_csv(write_dir / "propensity_diagnostics.csv", index=False)
        weight_diag.to_csv(write_dir / "weight_diagnostics.csv", index=False)
        balance.to_csv(write_dir / "profile_balance_smd.csv", index=False)
        audit_imbalance.to_csv(write_dir / "posthoc_audit_cell_imbalance.csv", index=False)
        aggregate.to_csv(write_dir / "curve_aggregate_ipw.csv", index=False)
        local.to_csv(write_dir / "curve_local_by_z_ipw.csv", index=False)
        gap.to_csv(write_dir / "gap_aggregate_ipw.csv", index=False)
        gap_z.to_csv(write_dir / "gap_local_by_z_ipw.csv", index=False)
        reversals.to_csv(write_dir / "simpson_reversals_ipw.csv", index=False)
        counts.to_csv(write_dir / "reversal_counts_by_alpha_ipw.csv", index=False)
        persistent.to_csv(write_dir / "persistent_reversals_ipw.csv", index=False)
        count_comparison.to_csv(write_dir / "reversal_counts_comparison.csv", index=False)
        persistent_comparison.to_csv(write_dir / "persistent_reversals_comparison.csv", index=False)
        alignment.to_csv(write_dir / "alignment_comparison.csv", index=False)
        rate_summary.to_csv(write_dir / "reversal_rate_summary.csv", index=False)
        strength_summary.to_csv(write_dir / "conflict_strength_summary.csv", index=False)
        strength_by_alpha.to_csv(write_dir / "conflict_strength_by_alpha.csv", index=False)
        material_summary.to_csv(write_dir / "material_strength_counts.csv", index=False)
        block_summary.to_csv(write_dir / "alpha_block_summary.csv", index=False)
        budget.to_csv(write_dir / "alignment_budget_by_alpha.csv", index=False)
        budget_summary.to_csv(write_dir / "alignment_budget_summary.csv", index=False)
        stability.to_csv(write_dir / "stability_summary.csv", index=False)
        jdump(model_summary, write_dir / "propensity_model_summary.json")
        jdump(args_record, write_dir / "args.json")
        jdump(ipw_summary, write_dir / "ipw_summary.json")
        jdump({"rows": rate_summary.to_dict(orient="records")}, write_dir / "reversal_rate_summary.json")
        jdump({"rows": strength_summary.to_dict(orient="records")}, write_dir / "conflict_strength_summary.json")
        jdump({"rows": budget_summary.to_dict(orient="records")}, write_dir / "alignment_budget_summary.json")
        jdump({
            "local_gap_variability": budget_summary.to_dict(orient="records"),
            "weight_estimation_stability": stability.to_dict(orient="records"),
            "profile_balance_smd_summary": balance_summary,
            "posthoc_audit_cell_imbalance_summary": audit_imbalance_summary,
        }, write_dir / "variance_stability_summary.json")

    return result


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_root = resolve_path(args.output_root)
    year_dir = out_root / str(args.year)
    train_dir = year_dir / "train"
    raw_dir = output_dir_for_grid(year_dir, "raw_curve", args.alpha_grid_name)
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw curve directory: {raw_dir}. Run 02_raw_curve.py first.")

    out_dir = mkdir(output_dir_for_grid(year_dir, "adjusted_ipw_curve", args.alpha_grid_name))
    label_maps = jload(train_dir / "label_maps.json")
    split_meta = jload(train_dir / "split_meta.json")

    run_adjusted_model(
        args=args,
        cfg=cfg,
        out_root=out_root,
        train_dir=train_dir,
        raw_dir=raw_dir,
        label_maps=label_maps,
        split_meta=split_meta,
        write_dir=out_dir,
    )

    print(f"Wrote adjusted IPW curve outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
