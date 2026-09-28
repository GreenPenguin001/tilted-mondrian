#!/usr/bin/env python3
"""ATUS activity-curve runner.

Dry-run mode constructs ATUS X/Y/G/Z with a strict feature whitelist and writes
support and leakage diagnostics. Non-dry mode trains the selected classifier
and writes prediction, conformal, and curve diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atus_pipeline import (
    DEFAULT_EXTRACT_DIR,
    DEFAULT_RAW_DIR,
    OTHER_LABEL,
    PREP_DIR,
    json_dumpable,
    prepare_design,
    resolve_path,
    write_csv,
    write_json,
    write_prep_aggregate,
)


PROJECT_DIR = Path(__file__).resolve().parent
ATUS_DENSE_ALPHA_GRID = [round(x, 2) for x in np.arange(0.02, 0.251, 0.01)]
ATUS_DENSE_ALPHA_GRID_NAME = "atus_dense_0.02_0.25_step_0.01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="ATUS config JSON.")
    parser.add_argument("--design", type=str, help="Design key inside config JSON.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--lexicon-path", type=Path, default=None)
    parser.add_argument("--unit", choices=["one_episode_per_respondent", "all_eligible_episodes"], default=None)
    parser.add_argument("--y-tier2-top-k", type=int, default=None)
    parser.add_argument("--y-top-k", type=int, default=None, help="Alias for --y-tier2-top-k.")
    parser.add_argument("--exclude-data-code-activities", action="store_true", default=None)
    parser.add_argument("--group", choices=["sex", "race"], default=None)
    parser.add_argument("--z", choices=["employment", "household", "daypart"], default=None)
    parser.add_argument("--model", choices=["mlp", "torch_mlp_cuda", "logistic_multinomial", "hist_gradient_boosting"], default=None)
    parser.add_argument("--feature-mode", choices=["safe", "safe_plus_context", "strong_where"], default="safe")
    parser.add_argument("--y-mode", choices=["top20_with_other", "closed_world_top20", "closed_world_topk"], default="top20_with_other")
    parser.add_argument("--hidden-layers", type=str, default="512,256,128")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--class-weight", choices=["none", "sqrt_inverse_frequency", "inverse_frequency_power_0.25"], default="none")
    parser.add_argument("--hgb-max-iter", type=int, default=None)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.06)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2-regularization", type=float, default=None)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=20)
    parser.add_argument("--torch-amp", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--early-stopping-metric", choices=["val_log_loss", "val_loss", "val_accuracy", "val_macro_f1"], default="val_log_loss")
    parser.add_argument("--restore-best-checkpoint", action="store_true")
    parser.add_argument("--alpha-grid", type=str, default="config", help="Use 'atus_dense', 'config', or a comma-separated alpha list.")
    parser.add_argument("--score", choices=["simple", "aps", "raps", "saps"], default="simple")
    parser.add_argument("--raps-lambda", type=float, default=0.01)
    parser.add_argument("--raps-k-reg", type=int, default=5)
    parser.add_argument("--saps-lambda", type=float, default=0.2)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--no-train-cap", action="store_true", help="Ignore max_train_examples from config and train on all train-split examples.")
    parser.add_argument("--propensity-clip", type=float, default=0.03)
    parser.add_argument("--trim-quantile", type=float, default=0.95)
    parser.add_argument("--max-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-cache", action="store_true", help="Disable cached ATUS one-episode construction.")
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_design_options(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    designs = config.get("designs", {})
    design_name = args.design or config.get("default_design") or "cli_design"
    design_cfg = dict(designs.get(design_name, {}))
    if args.config is not None and args.design and args.design not in designs:
        raise KeyError(f"Design {args.design!r} not found in {args.config}")

    def pick(name: str, cli_value: Any, default: Any = None) -> Any:
        if cli_value is not None:
            return cli_value
        if name in design_cfg:
            return design_cfg[name]
        if name in config:
            return config[name]
        return default

    if args.no_train_cap:
        max_train_examples = None
        train_cap_source = "disabled_by_cli_no_train_cap"
    elif args.max_train_examples is not None:
        max_train_examples = args.max_train_examples if args.max_train_examples > 0 else None
        train_cap_source = "cli_max_train_examples" if max_train_examples is not None else "cli_no_cap"
    else:
        cfg_cap = pick("max_train_examples", None, None)
        max_train_examples = cfg_cap if cfg_cap not in ("", 0, "0") else None
        train_cap_source = "config_max_train_examples" if max_train_examples is not None else "no_cap"

    top_k_cli = args.y_top_k if args.y_top_k is not None else args.y_tier2_top_k
    return {
        "design_name": design_name,
        "raw_dir": Path(pick("data_dir", args.data_dir, DEFAULT_RAW_DIR)),
        "extract_dir": Path(pick("extract_dir", args.extract_dir, DEFAULT_EXTRACT_DIR)),
        "lexicon_path": Path(pick("lexicon_path", args.lexicon_path, "")) if pick("lexicon_path", args.lexicon_path, None) else None,
        "unit": pick("unit", args.unit, "one_episode_per_respondent"),
        "top_k": int(pick("y_tier2_top_k", top_k_cli, 20)),
        "exclude_data_code_activities": bool(pick("exclude_data_code_activities", args.exclude_data_code_activities, True)),
        "group": pick("group", args.group, "sex"),
        "z": pick("z", args.z, "employment"),
        "model": pick("model", args.model, "mlp"),
        "y_mode": str(pick("y_mode", args.y_mode, "top20_with_other")),
        "max_train_examples": int(max_train_examples) if max_train_examples is not None else None,
        "train_cap_source": train_cap_source,
    }


def command_string(argv: list[str]) -> str:
    return " ".join([shlex.quote(sys.executable), *[shlex.quote(x) for x in argv]])


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not layers or any(layer <= 0 for layer in layers):
        raise ValueError(f"Invalid --hidden-layers value: {value!r}")
    return layers


def resolve_alpha_grid(alpha_grid_arg: str, dryrun: dict[str, Any]) -> tuple[list[float], str]:
    if alpha_grid_arg == "atus_dense":
        return [float(x) for x in ATUS_DENSE_ALPHA_GRID], ATUS_DENSE_ALPHA_GRID_NAME
    if alpha_grid_arg == "config":
        return [float(x) for x in dryrun["recommended_alpha_grid"]], "config_recommended_alpha_grid"
    try:
        values = [float(part.strip()) for part in alpha_grid_arg.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--alpha-grid must be 'atus_dense', 'config', or comma-separated floats") from exc
    if not values or any(x <= 0 or x >= 1 for x in values):
        raise ValueError("--alpha-grid values must be in (0, 1)")
    return values, "custom_alpha_grid"


def top_probability_features(prob: np.ndarray) -> pd.DataFrame:
    sorted_prob = np.sort(prob, axis=1)[:, ::-1]
    p1 = sorted_prob[:, 0]
    p2 = sorted_prob[:, 1] if sorted_prob.shape[1] > 1 else np.zeros(len(sorted_prob))
    p3 = sorted_prob[:, 2] if sorted_prob.shape[1] > 2 else np.zeros(len(sorted_prob))
    entropy = -(np.clip(prob, 1e-12, 1.0) * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1)
    margin = p1 - p2
    return pd.DataFrame({"p_top1": p1, "p_top2": p2, "p_top3": p3, "entropy": entropy, "margin": margin})


def group_pair(group_name: str, observed: pd.Series) -> tuple[str, str]:
    values = set(observed.dropna().astype("string").tolist())
    if group_name == "sex" and {"female", "male"}.issubset(values):
        return "female", "male"
    if group_name == "race" and {"non_hispanic_black", "non_hispanic_white"}.issubset(values):
        return "non_hispanic_black", "non_hispanic_white"
    ordered = sorted(values)
    if len(ordered) < 2:
        raise ValueError(f"Need at least two observed groups for curve gaps, got {ordered}")
    return ordered[0], ordered[1]


def stratified_cap_indices(idx_train: np.ndarray, y: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    if cap is None or cap >= len(idx_train):
        return idx_train
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    remaining: list[int] = []
    for cls in np.unique(y[idx_train]):
        cls_idx = idx_train[y[idx_train] == cls]
        if len(cls_idx):
            first = int(rng.choice(cls_idx, size=1)[0])
            chosen.append(first)
            remaining.extend([int(x) for x in cls_idx if int(x) != first])
    need = max(0, int(cap) - len(chosen))
    if need > 0 and remaining:
        sampled = rng.choice(np.asarray(remaining), size=min(need, len(remaining)), replace=False)
        chosen.extend([int(x) for x in sampled])
    return np.asarray(sorted(chosen), dtype=int)


def align_probability_columns(classes: np.ndarray, raw_prob: np.ndarray, n_classes: int) -> np.ndarray:
    full = np.zeros((raw_prob.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(classes.astype(int)):
        if 0 <= cls < n_classes:
            full[:, cls] = raw_prob[:, j]
    row_sum = full.sum(axis=1)
    missing = row_sum <= 0
    if missing.any():
        full[missing, :] = 1.0 / n_classes
    return full


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    finite = np.sort(np.asarray(scores, dtype=float))
    n = len(finite)
    if n == 0:
        raise ValueError("Calibration split is empty")
    k = int(math.ceil((n + 1) * (1 - float(alpha))))
    if k > n:
        return float("inf")
    return float(finite[max(k - 1, 0)])


def score_metadata(score: str, raps_lambda: float, raps_k_reg: int, saps_lambda: float) -> dict[str, Any]:
    score = str(score).lower()
    if score == "simple":
        formula = "S_y(x) = 1 - p_y(x); C_alpha(x) = {y: S_y(x) <= q_alpha}."
    elif score == "aps":
        formula = (
            "Deterministic APS: sort labels by descending probability; "
            "S_y(x) is cumulative probability through y's rank; C_alpha(x) = {y: S_y(x) <= q_alpha}."
        )
    elif score == "raps":
        formula = (
            "Deterministic task-specified RAPS: S_y(x) = cumulative_prob_to_y "
            "+ raps_lambda * max(rank_y - raps_k_reg, 0), with 1-based rank."
        )
    elif score == "saps":
        formula = (
            "BioBias-compatible deterministic SAPS with u=0.5: "
            "S_y(x)=0.5*p_top when rank_y=1; otherwise "
            "S_y(x)=p_top + saps_lambda*(rank_y - 1.5)."
        )
    else:
        raise ValueError(f"Unsupported conformal score: {score}")
    return {
        "score": score,
        "raps_lambda": float(raps_lambda),
        "raps_k_reg": int(raps_k_reg),
        "saps_lambda": float(saps_lambda),
        "formula": formula,
        "threshold_source": "calibration_split_only",
        "prediction_set_rule": "label is included iff its labelwise conformal score is <= q_alpha",
        "randomization": "none; deterministic scores",
    }


def labelwise_scores_from_probs(
    probs: np.ndarray,
    score: str,
    raps_lambda: float,
    raps_k_reg: int,
    saps_lambda: float,
) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    score = str(score).lower()
    if score == "simple":
        return 1.0 - p

    n, k = p.shape
    order = np.argsort(-p, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(1, k + 1)
    p_sorted = np.take_along_axis(p, order, axis=1)
    cumulative_sorted = np.cumsum(p_sorted, axis=1)
    cumulative = cumulative_sorted[np.arange(n)[:, None], rank - 1]

    if score == "aps":
        return cumulative
    if score == "raps":
        return cumulative + float(raps_lambda) * np.maximum(rank - int(raps_k_reg), 0)
    if score == "saps":
        p_top = p_sorted[:, 0][:, None]
        return np.where(rank == 1, 0.5 * p_top, p_top + float(saps_lambda) * (rank - 1.5))
    raise ValueError(f"Unsupported conformal score: {score}")


def weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if not ok.any() or float(w[ok].sum()) <= 0:
        return float("nan")
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    if not np.isfinite(mean):
        return float("nan")
    return weighted_mean((np.asarray(values, dtype=float) - mean) ** 2, weights)


def weighted_smd(values: np.ndarray, groups: np.ndarray, weights: np.ndarray, group_a: str, group_b: str) -> float:
    mask_a = groups == group_a
    mask_b = groups == group_b
    if not mask_a.any() or not mask_b.any():
        return float("nan")
    va = np.asarray(values[mask_a], dtype=float)
    vb = np.asarray(values[mask_b], dtype=float)
    wa = np.asarray(weights[mask_a], dtype=float)
    wb = np.asarray(weights[mask_b], dtype=float)
    mean_a = weighted_mean(va, wa)
    mean_b = weighted_mean(vb, wb)
    var_a = weighted_var(va, wa)
    var_b = weighted_var(vb, wb)
    denom = math.sqrt(max((var_a + var_b) / 2.0, 1e-12))
    return float((mean_a - mean_b) / denom)


def weighted_tv(values: pd.Series, groups: pd.Series, weights: pd.Series, group_a: str, group_b: str) -> float:
    levels = sorted(values.dropna().astype("string").unique().tolist())
    tv = 0.0
    for level in levels:
        mask_level = values.astype("string").eq(level).to_numpy()
        mask_a = groups.astype("string").eq(group_a).to_numpy()
        mask_b = groups.astype("string").eq(group_b).to_numpy()
        pa = weighted_mean(mask_level[mask_a].astype(float), weights.to_numpy(dtype=float)[mask_a])
        pb = weighted_mean(mask_level[mask_b].astype(float), weights.to_numpy(dtype=float)[mask_b])
        tv += abs(pa - pb)
    return float(0.5 * tv)


def probability_profile(prob: np.ndarray, n_labels: int) -> tuple[pd.DataFrame, list[str], str]:
    h = top_probability_features(prob)
    if n_labels >= 15:
        cols = ["p_top1", "p_top2", "p_top3", "entropy", "margin"]
        rule = "labels_ge_15_top3_entropy_margin"
    elif n_labels < 12:
        cols = ["p_top1", "p_top2", "margin", "entropy"]
        rule = "labels_lt_12_top2_margin_entropy"
    else:
        cols = ["p_top1", "p_top2", "p_top3", "entropy", "margin"]
        rule = "labels_12_to_14_top3_entropy_margin"
    return h[cols].copy(), cols, rule


def write_probability_file(
    output_dir: Path,
    split_name: str,
    examples: pd.DataFrame,
    idx: np.ndarray,
    y: np.ndarray,
    prob: np.ndarray,
    class_names: np.ndarray,
    group_col: str,
    z_col: str,
) -> str:
    pred = prob.argmax(axis=1)
    p_true = prob[np.arange(len(idx)), y[idx]]
    sub = examples.iloc[idx]
    frame = pd.DataFrame({
        "row_pos": idx,
        "TUCASEID": sub["TUCASEID"].to_numpy(),
        "split": split_name,
        "g": sub[group_col].astype("string").to_numpy(),
        "z": sub[z_col].astype("string").to_numpy(),
        "group_col": group_col,
        "z_col": z_col,
        "true_label": class_names[y[idx]],
        "pred_label": class_names[pred],
        "p_true": p_true,
        "p_max": prob.max(axis=1),
    })
    optional_metadata_cols = [
        "sex_g",
        "race_g",
        "employment_z",
        "household_z",
        "start_time_bucket",
        "y_label",
        "tier2_code",
        "tier2_name",
    ]
    for col in optional_metadata_cols:
        if col in sub.columns and col not in frame.columns:
            frame[col] = sub[col].to_numpy()
    for j, label in enumerate(class_names):
        frame[f"prob__{label}"] = prob[:, j]
    path = output_dir / f"class_probabilities_{split_name}.csv.gz"
    frame.to_csv(path, index=False)
    return str(path)


def prediction_metrics_rows(
    split_probs: dict[str, np.ndarray],
    split_indices: dict[str, np.ndarray],
    y: np.ndarray,
    majority_baseline: float,
    n_classes: int,
) -> list[dict[str, Any]]:
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    rows = []
    for split_name, prob in split_probs.items():
        idx = split_indices[split_name]
        pred = prob.argmax(axis=1)
        rows.append({
            "split": split_name,
            "n": int(len(idx)),
            "accuracy": float(accuracy_score(y[idx], pred)),
            "macro_f1": float(f1_score(y[idx], pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y[idx], pred, average="weighted", zero_division=0)),
            "log_loss": float(log_loss(y[idx], np.clip(prob, 1e-12, 1.0), labels=np.arange(n_classes))),
            "majority_baseline_train": float(majority_baseline),
        })
    return rows


def classification_detail_tables(
    split_probs: dict[str, np.ndarray],
    split_indices: dict[str, np.ndarray],
    y: np.ndarray,
    class_names: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_fscore_support

    per_class_rows = []
    class_dist_rows = []
    pred_dist_rows = []
    n_classes = len(class_names)
    for split_name, prob in split_probs.items():
        idx = split_indices[split_name]
        pred = prob.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            y[idx],
            pred,
            labels=np.arange(n_classes),
            zero_division=0,
        )
        true_counts = np.bincount(y[idx], minlength=n_classes)
        pred_counts = np.bincount(pred, minlength=n_classes)
        for j, label in enumerate(class_names):
            per_class_rows.append({
                "split": split_name,
                "class_index": int(j),
                "class_label": str(label),
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "f1": float(f1[j]),
                "support": int(support[j]),
            })
            class_dist_rows.append({
                "split": split_name,
                "class_index": int(j),
                "class_label": str(label),
                "count": int(true_counts[j]),
                "fraction": float(true_counts[j] / max(len(idx), 1)),
            })
            pred_dist_rows.append({
                "split": split_name,
                "class_index": int(j),
                "class_label": str(label),
                "predicted_count": int(pred_counts[j]),
                "predicted_fraction": float(pred_counts[j] / max(len(idx), 1)),
            })
    return pd.DataFrame(per_class_rows), pd.DataFrame(class_dist_rows), pd.DataFrame(pred_dist_rows)


def build_eval_frame(
    examples: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    split_probs: dict[str, np.ndarray],
    y: np.ndarray,
    alpha_grid: list[float],
    group_col: str,
    z_col: str,
    score: str = "simple",
    raps_lambda: float = 0.01,
    raps_k_reg: int = 5,
    saps_lambda: float = 0.2,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    threshold_rows = []
    idx_cal = split_indices["calibration"]
    cal_score_matrix = labelwise_scores_from_probs(
        split_probs["calibration"],
        score=score,
        raps_lambda=raps_lambda,
        raps_k_reg=raps_k_reg,
        saps_lambda=saps_lambda,
    )
    cal_scores = cal_score_matrix[np.arange(len(idx_cal)), y[idx_cal]]
    score_meta = score_metadata(score, raps_lambda, raps_k_reg, saps_lambda)
    split_score_matrices = {
        split_name: labelwise_scores_from_probs(
            prob,
            score=score,
            raps_lambda=raps_lambda,
            raps_k_reg=raps_k_reg,
            saps_lambda=saps_lambda,
        )
        for split_name, prob in split_probs.items()
    }
    for alpha in alpha_grid:
        q = conformal_quantile(cal_scores, alpha)
        probability_floor = 1.0 - q if score == "simple" and np.isfinite(q) else np.nan
        threshold_rows.append({
            "score": score,
            "alpha": float(alpha),
            "q_score": q,
            "probability_floor": probability_floor,
            "raps_lambda": float(raps_lambda),
            "raps_k_reg": int(raps_k_reg),
            "saps_lambda": float(saps_lambda),
            "score_formula": score_meta["formula"],
        })
        for split_name in ["repair", "test"]:
            idx = split_indices[split_name]
            score_matrix = split_score_matrices[split_name]
            true_scores = score_matrix[np.arange(len(idx)), y[idx]]
            set_size = (score_matrix <= q).sum(axis=1)
            covered = true_scores <= q
            sub = examples.iloc[idx]
            rows.append(pd.DataFrame({
                "row_pos": idx,
                "split": split_name,
                "alpha": float(alpha),
                "score": score,
                "g": sub[group_col].astype("string").to_numpy(),
                "z": sub[z_col].astype("string").to_numpy(),
                "set_size": set_size.astype(float),
                "covered": covered.astype(float),
                "unit_weight": 1.0,
            }))
    return pd.concat(rows, ignore_index=True), threshold_rows


def curve_tables(
    eval_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    method: str,
    weight_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    local_rows = []
    for (split_name, alpha, g), sub in eval_df.groupby(["split", "alpha", "g"], dropna=False):
        w = sub[weight_col]
        aggregate_rows.append({
            "method": method,
            "split": split_name,
            "alpha": float(alpha),
            "g": g,
            "V_mean_set_size": weighted_mean(sub["set_size"], w),
            "coverage": weighted_mean(sub["covered"], w),
            "n": int(len(sub)),
            "weight_sum": float(np.asarray(w, dtype=float).sum()),
        })
    for (split_name, alpha, z, g), sub in eval_df.groupby(["split", "alpha", "z", "g"], dropna=False):
        w = sub[weight_col]
        local_rows.append({
            "method": method,
            "split": split_name,
            "alpha": float(alpha),
            "z": z,
            "g": g,
            "V_mean_set_size": weighted_mean(sub["set_size"], w),
            "coverage": weighted_mean(sub["covered"], w),
            "n": int(len(sub)),
            "weight_sum": float(np.asarray(w, dtype=float).sum()),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    local = pd.DataFrame(local_rows)
    gap_rows = []
    for (split_name, alpha), sub in aggregate.groupby(["split", "alpha"]):
        row_a = sub[sub["g"].eq(group_a)]
        row_b = sub[sub["g"].eq(group_b)]
        if len(row_a) and len(row_b):
            a = row_a.iloc[0]
            b = row_b.iloc[0]
            gap_rows.append({
                "method": method,
                "split": split_name,
                "alpha": float(alpha),
                "scope": "aggregate",
                "z": "",
                "group_a": group_a,
                "group_b": group_b,
                "gap_direction": "delta = V_group_a - V_group_b",
                "delta": float(a["V_mean_set_size"] - b["V_mean_set_size"]),
                "V_group_a": float(a["V_mean_set_size"]),
                "V_group_b": float(b["V_mean_set_size"]),
                "group_a_n": int(a["n"]),
                "group_b_n": int(b["n"]),
                "min_gz_n": int(min(a["n"], b["n"])),
                "group_a_weight_sum": float(a["weight_sum"]),
                "group_b_weight_sum": float(b["weight_sum"]),
            })
    for (split_name, alpha, z), sub in local.groupby(["split", "alpha", "z"]):
        row_a = sub[sub["g"].eq(group_a)]
        row_b = sub[sub["g"].eq(group_b)]
        if len(row_a) and len(row_b):
            a = row_a.iloc[0]
            b = row_b.iloc[0]
            gap_rows.append({
                "method": method,
                "split": split_name,
                "alpha": float(alpha),
                "scope": "local",
                "z": z,
                "group_a": group_a,
                "group_b": group_b,
                "gap_direction": "delta = V_group_a - V_group_b",
                "delta": float(a["V_mean_set_size"] - b["V_mean_set_size"]),
                "V_group_a": float(a["V_mean_set_size"]),
                "V_group_b": float(b["V_mean_set_size"]),
                "group_a_n": int(a["n"]),
                "group_b_n": int(b["n"]),
                "min_gz_n": int(min(a["n"], b["n"])),
                "group_a_weight_sum": float(a["weight_sum"]),
                "group_b_weight_sum": float(b["weight_sum"]),
            })
    return aggregate, local, pd.DataFrame(gap_rows)


def fit_overlap_weights(
    examples: pd.DataFrame,
    h_all: pd.DataFrame,
    group_col: str,
    z_col: str,
    group_a: str,
    group_b: str,
    propensity_clip: float,
    trim_quantile: float,
    max_weight: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    split = examples["split"].astype("string")
    repair_idx = np.where(split.eq("repair"))[0]
    target_repair = examples.iloc[repair_idx][group_col].astype("string").eq(group_a).astype(int).to_numpy()
    prop_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    prop_model.fit(h_all.iloc[repair_idx], target_repair)

    rows = []
    diagnostics = []
    for split_name in ["repair", "test"]:
        idx = np.where(split.eq(split_name))[0]
        target = examples.iloc[idx][group_col].astype("string").eq(group_a).astype(int).to_numpy()
        prop_raw = prop_model.predict_proba(h_all.iloc[idx])[:, 1]
        prop_clipped = np.clip(prop_raw, propensity_clip, 1.0 - propensity_clip)
        weight_raw = np.where(target == 1, 1.0 - prop_clipped, prop_clipped)
        trim_value = float(np.quantile(weight_raw, trim_quantile)) if len(weight_raw) else float("nan")
        cap_value = min(trim_value, float(max_weight)) if np.isfinite(trim_value) else float(max_weight)
        weight = np.minimum(weight_raw, cap_value)
        diagnostics.append({
            "split": split_name,
            "n": int(len(idx)),
            "group_a": group_a,
            "group_b": group_b,
            "propensity_model_train_split": "repair",
            "h_features": ";".join(h_all.columns.astype(str).tolist()),
            "propensity_clip": float(propensity_clip),
            "trim_quantile": float(trim_quantile),
            "trim_value": trim_value,
            "max_weight": float(max_weight),
            "cap_value": cap_value,
            "auc": float(roc_auc_score(target, prop_raw)) if len(np.unique(target)) == 2 else float("nan"),
            "log_loss": float(log_loss(target, np.clip(prop_raw, 1e-12, 1 - 1e-12), labels=[0, 1])) if len(np.unique(target)) == 2 else float("nan"),
            "brier": float(brier_score_loss(target, prop_raw)),
            "propensity_min": float(np.min(prop_raw)),
            "propensity_p05": float(np.quantile(prop_raw, 0.05)),
            "propensity_p50": float(np.quantile(prop_raw, 0.50)),
            "propensity_p95": float(np.quantile(prop_raw, 0.95)),
            "propensity_max": float(np.max(prop_raw)),
            "n_propensity_clipped": int(((prop_raw < propensity_clip) | (prop_raw > 1.0 - propensity_clip)).sum()),
            "n_weight_trimmed": int((weight_raw > cap_value).sum()),
        })
        sub = examples.iloc[idx]
        rows.append(pd.DataFrame({
            "row_pos": idx,
            "split": split_name,
            "g": sub[group_col].astype("string").to_numpy(),
            "z": sub[z_col].astype("string").to_numpy(),
            "g_is_group_a": target,
            "propensity_group_a_raw": prop_raw,
            "propensity_group_a_clipped": prop_clipped,
            "overlap_weight_raw": weight_raw,
            "overlap_weight": weight,
        }))
    weights = pd.concat(rows, ignore_index=True)
    weight_summary_rows = []
    ess_rows = []
    for (split_name, g), sub in weights.groupby(["split", "g"]):
        w = sub["overlap_weight"].to_numpy(dtype=float)
        ess = float((w.sum() ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else float("nan")
        weight_summary_rows.append({
            "split": split_name,
            "g": g,
            "n": int(len(sub)),
            "weight_mean": float(np.mean(w)),
            "weight_std": float(np.std(w)),
            "weight_min": float(np.min(w)),
            "weight_p50": float(np.quantile(w, 0.50)),
            "weight_p90": float(np.quantile(w, 0.90)),
            "weight_p95": float(np.quantile(w, 0.95)),
            "weight_p99": float(np.quantile(w, 0.99)),
            "weight_max": float(np.max(w)),
        })
        ess_rows.append({
            "split": split_name,
            "g": g,
            "n": int(len(sub)),
            "weight_sum": float(w.sum()),
            "ess": ess,
            "ess_fraction": float(ess / len(sub)) if len(sub) else float("nan"),
        })
    for split_name, sub in weights.groupby("split"):
        w = sub["overlap_weight"].to_numpy(dtype=float)
        ess = float((w.sum() ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else float("nan")
        ess_rows.append({
            "split": split_name,
            "g": "ALL",
            "n": int(len(sub)),
            "weight_sum": float(w.sum()),
            "ess": ess,
            "ess_fraction": float(ess / len(sub)) if len(sub) else float("nan"),
        })
    return weights, pd.DataFrame(diagnostics), pd.DataFrame(weight_summary_rows), pd.DataFrame(ess_rows), {
        "group_a": group_a,
        "group_b": group_b,
        "propensity_model": "standardized_logistic_regression",
    }


def balance_diagnostics(
    weights: pd.DataFrame,
    h_all: pd.DataFrame,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    rows = []
    for split_name, sub in weights.groupby("split"):
        row_pos = sub["row_pos"].to_numpy(dtype=int)
        h_sub = h_all.iloc[row_pos].reset_index(drop=True)
        groups = sub["g"].astype("string").to_numpy()
        ones = np.ones(len(sub), dtype=float)
        after_w = sub["overlap_weight"].to_numpy(dtype=float)
        for col in h_sub.columns:
            values = h_sub[col].to_numpy(dtype=float)
            before = weighted_smd(values, groups, ones, group_a, group_b)
            after = weighted_smd(values, groups, after_w, group_a, group_b)
            rows.append({
                "split": split_name,
                "diagnostic": "H_standardized_mean_difference",
                "feature": col,
                "before": before,
                "after": after,
                "abs_before": abs(before) if np.isfinite(before) else float("nan"),
                "abs_after": abs(after) if np.isfinite(after) else float("nan"),
            })
        tv_before = weighted_tv(sub["z"], sub["g"], pd.Series(ones), group_a, group_b)
        tv_after = weighted_tv(sub["z"], sub["g"], sub["overlap_weight"], group_a, group_b)
        rows.append({
            "split": split_name,
            "diagnostic": "Z_total_variation_by_G",
            "feature": "selected_Z",
            "before": tv_before,
            "after": tv_after,
            "abs_before": abs(tv_before),
            "abs_after": abs(tv_after),
        })
    return pd.DataFrame(rows)


def sign_conflict(aggregate_gap: float, local_gap: float) -> bool:
    return bool(np.sign(aggregate_gap) * np.sign(local_gap) < 0)


def simpson_diagnostics(
    raw_gap: pd.DataFrame,
    ipw_gap: pd.DataFrame,
    thresholds: list[float],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_gap = pd.concat([raw_gap.assign(method="raw"), ipw_gap.assign(method="ipw")], ignore_index=True)
    diag_by_split: dict[str, pd.DataFrame] = {}
    decomp_by_split: dict[str, pd.DataFrame] = {}
    material_rows = []
    near_zero_rows = []
    comparison_rows = []
    for split_name in sorted(all_gap["split"].dropna().unique().tolist()):
        split_diag_rows = []
        split_decomp_rows = []
        for method in ["raw", "ipw"]:
            method_gap = all_gap[(all_gap["split"].eq(split_name)) & (all_gap["method"].eq(method))]
            agg = method_gap[method_gap["scope"].eq("aggregate")].set_index("alpha")
            local = method_gap[method_gap["scope"].eq("local")]
            for alpha, local_sub in local.groupby("alpha"):
                if alpha not in agg.index:
                    continue
                aggregate_gap = float(agg.loc[alpha, "delta"])
                supported = local_sub[(local_sub["group_a_n"] > 0) & (local_sub["group_b_n"] > 0)].copy()
                supported["is_sign_conflict"] = supported["delta"].map(lambda x: sign_conflict(aggregate_gap, float(x)))
                hard_count = int(supported["is_sign_conflict"].sum())
                conflict_strength = float((aggregate_gap * supported.loc[supported["is_sign_conflict"], "delta"]).abs().sum())
                min_gz = int(supported["min_gz_n"].min()) if len(supported) else 0
                row = {
                    "split": split_name,
                    "method": method,
                    "alpha": float(alpha),
                    "aggregate_gap": aggregate_gap,
                    "hard_sign_reversal_count": hard_count,
                    "conflict_strength": conflict_strength,
                    "supported_cell_count": int(len(supported)),
                    "min_gz_sample_size": min_gz,
                    "near_zero_abs_gap_lt_0_005": bool(abs(aggregate_gap) < 0.005),
                    "near_zero_abs_gap_lt_0_01": bool(abs(aggregate_gap) < 0.01),
                    "near_zero_abs_gap_lt_0_02": bool(abs(aggregate_gap) < 0.02),
                }
                for threshold in thresholds:
                    count = int((
                        supported["is_sign_conflict"]
                        & (supported["delta"].abs() >= threshold)
                        & (abs(aggregate_gap) >= threshold)
                    ).sum())
                    row[f"material_reversal_count_ge_{threshold:g}"] = count
                    material_rows.append({
                        "split": split_name,
                        "method": method,
                        "alpha": float(alpha),
                        "threshold": float(threshold),
                        "material_reversal_count": count,
                    })
                split_diag_rows.append(row)
            for threshold in [0.005, 0.01, 0.02]:
                near_zero_rows.append({
                    "split": split_name,
                    "method": method,
                    "threshold": threshold,
                    "near_zero_aggregate_gap_count": int((agg["delta"].abs() < threshold).sum()) if len(agg) else 0,
                    "n_alpha": int(len(agg)),
                })
        raw_split = all_gap[(all_gap["split"].eq(split_name)) & (all_gap["method"].eq("raw"))]
        ipw_split = all_gap[(all_gap["split"].eq(split_name)) & (all_gap["method"].eq("ipw"))]
        raw_agg = raw_split[raw_split["scope"].eq("aggregate")].set_index("alpha")["delta"].to_dict()
        ipw_agg = ipw_split[ipw_split["scope"].eq("aggregate")].set_index("alpha")["delta"].to_dict()
        raw_local = raw_split[raw_split["scope"].eq("local")][["alpha", "z", "delta"]].rename(columns={"delta": "raw_local_gap"})
        ipw_local = ipw_split[ipw_split["scope"].eq("local")][["alpha", "z", "delta"]].rename(columns={"delta": "ipw_local_gap"})
        merged = raw_local.merge(ipw_local, on=["alpha", "z"], how="outer")
        for alpha, sub in merged.groupby("alpha"):
            raw_ag = float(raw_agg.get(alpha, np.nan))
            ipw_ag = float(ipw_agg.get(alpha, np.nan))
            raw_conf = sub["raw_local_gap"].map(lambda x: sign_conflict(raw_ag, float(x)) if pd.notna(x) else False)
            ipw_conf = sub["ipw_local_gap"].map(lambda x: sign_conflict(ipw_ag, float(x)) if pd.notna(x) else False)
            split_decomp_rows.append({
                "split": split_name,
                "alpha": float(alpha),
                "resolved_reversal_count": int((raw_conf & ~ipw_conf).sum()),
                "persistent_reversal_count": int((raw_conf & ipw_conf).sum()),
                "new_reversal_count": int((~raw_conf & ipw_conf).sum()),
                "raw_only_or_resolved_definition": "raw sign conflict absent after IPW",
                "persistent_definition": "raw and IPW both sign-conflict",
                "new_definition": "IPW sign conflict absent in raw",
            })
        diag_df = pd.DataFrame(split_diag_rows)
        decomp_df = pd.DataFrame(split_decomp_rows)
        diag_by_split[split_name] = diag_df
        decomp_by_split[split_name] = decomp_df
        if len(diag_df):
            raw_diag = diag_df[diag_df["method"].eq("raw")]
            ipw_diag = diag_df[diag_df["method"].eq("ipw")]
            for alpha in sorted(set(raw_diag["alpha"]).intersection(set(ipw_diag["alpha"]))):
                r = raw_diag[raw_diag["alpha"].eq(alpha)].iloc[0]
                i = ipw_diag[ipw_diag["alpha"].eq(alpha)].iloc[0]
                comparison_rows.append({
                    "split": split_name,
                    "alpha": float(alpha),
                    "raw_hard_sign_reversal_count": int(r["hard_sign_reversal_count"]),
                    "ipw_hard_sign_reversal_count": int(i["hard_sign_reversal_count"]),
                    "delta_hard_count_ipw_minus_raw": int(i["hard_sign_reversal_count"] - r["hard_sign_reversal_count"]),
                    "raw_conflict_strength": float(r["conflict_strength"]),
                    "ipw_conflict_strength": float(i["conflict_strength"]),
                    "delta_conflict_strength_ipw_minus_raw": float(i["conflict_strength"] - r["conflict_strength"]),
                    "raw_aggregate_gap": float(r["aggregate_gap"]),
                    "ipw_aggregate_gap": float(i["aggregate_gap"]),
                })
    return (
        diag_by_split,
        decomp_by_split,
        pd.DataFrame(material_rows),
        pd.DataFrame(near_zero_rows),
        pd.DataFrame(comparison_rows),
    )


def internal_train_validation_split(idx_train: np.ndarray, y: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    if len(idx_train) < 100:
        return idx_train, np.asarray([], dtype=int)
    val_fraction = float(val_fraction)
    if val_fraction <= 0:
        return idx_train, np.asarray([], dtype=int)
    val_fraction = min(max(val_fraction, 0.01), 0.50)
    try:
        train_inner, val_inner = train_test_split(
            idx_train,
            test_size=val_fraction,
            random_state=seed,
            stratify=y[idx_train],
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        shuffled = idx_train.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(val_fraction * len(shuffled))))
        val_inner = shuffled[:n_val]
        train_inner = shuffled[n_val:]
    return np.asarray(train_inner, dtype=int), np.asarray(val_inner, dtype=int)


def torch_class_weights(y_train: np.ndarray, n_classes: int, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    if mode == "sqrt_inverse_frequency":
        weights = 1.0 / np.sqrt(counts / counts.sum())
        weights = weights * (n_classes / weights.sum())
        return weights.astype("float32")
    if mode == "inverse_frequency_power_0.25":
        weights = 1.0 / np.power(counts / counts.sum(), 0.25)
        weights = weights * (n_classes / weights.sum())
        return weights.astype("float32")
    raise ValueError(f"Unknown class weight mode: {mode}")


def build_torch_mlp(input_dim: int, hidden_layers: tuple[int, ...], dropout: float, n_classes: int) -> Any:
    import torch

    layers: list[Any] = []
    prev = input_dim
    for hidden in hidden_layers:
        layers.append(torch.nn.Linear(prev, hidden))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        prev = hidden
    layers.append(torch.nn.Linear(prev, n_classes))
    return torch.nn.Sequential(*layers)


def run_epoch_torch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any | None,
    device: Any,
    amp_enabled: bool,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional
    from sklearn.metrics import f1_score

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_log_loss = 0.0
    total_correct = 0
    total_n = 0
    pred_parts = []
    true_parts = []
    scaler = getattr(run_epoch_torch, "_scaler", None)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(xb)
                loss = criterion(logits, yb)
            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(loss.detach().item()) * len(yb)
        logits_detached = logits.detach().float()
        pred = logits_detached.argmax(dim=1)
        total_log_loss += float(functional.cross_entropy(logits_detached, yb, reduction="sum").item())
        total_correct += int((pred == yb).sum().item())
        total_n += int(len(yb))
        pred_parts.append(pred.cpu().numpy())
        true_parts.append(yb.detach().cpu().numpy())
    pred_all = np.concatenate(pred_parts) if pred_parts else np.asarray([], dtype=int)
    true_all = np.concatenate(true_parts) if true_parts else np.asarray([], dtype=int)
    macro_f1 = float(f1_score(true_all, pred_all, average="macro", zero_division=0)) if len(true_all) else float("nan")
    return {
        "loss": total_loss / max(total_n, 1),
        "log_loss": total_log_loss / max(total_n, 1),
        "accuracy": total_correct / max(total_n, 1),
        "macro_f1": macro_f1,
    }


def predict_torch_probabilities(
    model: Any,
    x_values: np.ndarray,
    indices: np.ndarray,
    device: Any,
    batch_size: int,
    amp_enabled: bool,
) -> np.ndarray:
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            xb = torch.from_numpy(x_values[idx]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(xb)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.vstack(probs) if probs else np.zeros((0, 0), dtype="float32")


def train_torch_mlp_cuda(
    X: pd.DataFrame,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    idx_train: np.ndarray,
    n_classes: int,
    seed: int,
    hidden_layers: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    class_weight: str,
    torch_amp: bool,
    early_stopping: bool,
    val_fraction: float,
    patience: int,
    early_stopping_metric: str,
    restore_best_checkpoint: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    try:
        import torch
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch and scikit-learn are required for --model torch_mlp_cuda") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("--model torch_mlp_cuda requested, but CUDA is not available. Refusing CPU fallback.")
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    scaler = StandardScaler()
    x_all = scaler.fit_transform(X.iloc[idx_train].to_numpy(dtype="float32"))
    # Transform the full design matrix after fitting on train only.
    x_values = scaler.transform(X.to_numpy(dtype="float32")).astype("float32", copy=False)

    train_inner, val_inner = internal_train_validation_split(idx_train, y, seed, val_fraction)
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_values[train_inner]),
        torch.from_numpy(y[train_inner].astype("int64")),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = None
    if len(val_inner):
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_values[val_inner]),
            torch.from_numpy(y[val_inner].astype("int64")),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    model = build_torch_mlp(
        input_dim=x_values.shape[1],
        hidden_layers=hidden_layers,
        dropout=dropout,
        n_classes=n_classes,
    ).to(device)
    class_weights = torch_class_weights(y[idx_train], n_classes, class_weight)
    weight_tensor = torch.from_numpy(class_weights).to(device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    run_epoch_torch._scaler = torch.amp.GradScaler("cuda", enabled=torch_amp)  # type: ignore[attr-defined]

    history_rows = []
    best_epoch = None
    best_metric_value = float("inf") if early_stopping_metric in {"val_log_loss", "val_loss"} else -float("inf")
    best_state = None
    epochs_since_improvement = 0
    stopped_early = False
    epoch_iter = range(1, epochs + 1)
    try:
        from tqdm.auto import tqdm
        epoch_iter = tqdm(epoch_iter, desc="torch_mlp_cuda epochs", unit="epoch")
    except Exception:  # noqa: BLE001
        pass
    for epoch in epoch_iter:
        train_stats = run_epoch_torch(model, train_loader, criterion, optimizer, device, torch_amp)
        val_stats = {"loss": float("nan"), "log_loss": float("nan"), "accuracy": float("nan"), "macro_f1": float("nan")}
        if val_loader is not None:
            val_stats = run_epoch_torch(model, val_loader, criterion, None, device, torch_amp)
        history_rows.append({
            "epoch": int(epoch),
            "train_loss": float(train_stats["loss"]),
            "train_log_loss": float(train_stats["log_loss"]),
            "train_accuracy": float(train_stats["accuracy"]),
            "train_macro_f1": float(train_stats["macro_f1"]),
            "val_loss": float(val_stats["loss"]),
            "val_log_loss": float(val_stats["log_loss"]),
            "val_accuracy": float(val_stats["accuracy"]),
            "val_macro_f1": float(val_stats["macro_f1"]),
            "internal_validation_loss": float(val_stats["loss"]),
            "internal_validation_accuracy": float(val_stats["accuracy"]),
            "internal_validation_macro_f1": float(val_stats["macro_f1"]),
            "internal_validation_source": f"train_split_{val_fraction:.3f}_stratified",
        })
        if len(val_inner):
            metric_value = float(history_rows[-1][early_stopping_metric])
            if early_stopping_metric in {"val_log_loss", "val_loss"}:
                improved = metric_value < best_metric_value - 1e-6
            else:
                improved = metric_value > best_metric_value + 1e-6
            if improved:
                best_metric_value = metric_value
                best_epoch = int(epoch)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            if early_stopping and epochs_since_improvement >= patience:
                stopped_early = True
                print(
                    f"torch_mlp_cuda early stopping at epoch {epoch}: "
                    f"best_epoch={best_epoch} {early_stopping_metric}={best_metric_value:.6f}",
                    flush=True,
                )
                break
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix({
                "train_loss": f"{train_stats['loss']:.4f}",
                "train_acc": f"{train_stats['accuracy']:.3f}",
                "val_logloss": f"{val_stats['log_loss']:.4f}",
                "val_acc": f"{val_stats['accuracy']:.3f}",
            })
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(
                f"torch_mlp_cuda epoch {epoch}/{epochs}: "
                f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} train_macro_f1={train_stats['macro_f1']:.4f} "
                f"val_log_loss={val_stats['log_loss']:.4f} val_acc={val_stats['accuracy']:.4f} val_macro_f1={val_stats['macro_f1']:.4f}",
                flush=True,
            )

    restore_requested = bool(restore_best_checkpoint or early_stopping)
    if restore_requested and best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    pred_batch_size = max(batch_size * 2, 8192)
    split_probs = {
        split_name: predict_torch_probabilities(model, x_values, idx, device, pred_batch_size, torch_amp)
        for split_name, idx in split_indices.items()
    }
    meta = {
        "torch_version": str(torch.__version__),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_available": True,
        "amp_enabled": bool(torch_amp),
        "hidden_layers": list(hidden_layers),
        "dropout": float(dropout),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "epochs_completed": int(len(history_rows)),
        "batch_size": int(batch_size),
        "class_weight": class_weight,
        "internal_validation": f"{val_fraction:.3f}_fraction_of_train_split_only",
        "early_stopping": bool(early_stopping),
        "patience": int(patience),
        "early_stopping_metric": early_stopping_metric,
        "stopped_early": bool(stopped_early),
        "restore_best_checkpoint": bool(restore_requested),
        "best_checkpoint_summary": {
            "enabled": bool(early_stopping),
            "restore_best_checkpoint": bool(restore_requested),
            "metric": early_stopping_metric,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_metric_value": float(best_metric_value) if np.isfinite(best_metric_value) else None,
            "epochs_completed": int(len(history_rows)),
            "stopped_early": bool(stopped_early),
            "train_inner_n": int(len(train_inner)),
            "internal_validation_n": int(len(val_inner)),
            "internal_validation_source": "train split only; calibration split was not used for early stopping",
        },
        "calibration_used_for_training": False,
    }
    return split_probs, pd.DataFrame(history_rows), meta


def run_curve_experiment(
    prep: dict[str, Any],
    output_dir: Path,
    max_train_examples: int | None,
    train_cap_source: str,
    seed: int,
    propensity_clip: float,
    trim_quantile: float,
    max_weight: float,
    model_name: str,
    hidden_layers: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    class_weight: str,
    torch_amp: bool,
    early_stopping: bool,
    val_fraction: float,
    patience: int,
    early_stopping_metric: str,
    restore_best_checkpoint: bool,
    alpha_grid_arg: str,
    hgb_max_iter: int | None,
    hgb_learning_rate: float,
    hgb_max_leaf_nodes: int,
    hgb_l2_regularization: float | None,
    hgb_min_samples_leaf: int,
    score: str,
    raps_lambda: float,
    raps_k_reg: int,
    saps_lambda: float,
) -> dict[str, Any]:
    try:
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scikit-learn is required for ATUS model training") from exc

    examples = prep["examples"].copy()
    X = pd.get_dummies(prep["features"], dummy_na=False)
    labels = examples["y_label"].astype("string")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    class_names = encoder.classes_
    n_classes = len(class_names)
    split = examples["split"].astype("string")
    split_indices = {
        "train": np.where(split.eq("train"))[0],
        "calibration": np.where(split.eq("calibration"))[0],
        "repair": np.where(split.eq("repair"))[0],
        "test": np.where(split.eq("test"))[0],
    }
    idx_train_full = split_indices["train"]
    idx_train = stratified_cap_indices(idx_train_full, y, max_train_examples, seed)
    train_cap_active = len(idx_train) < len(idx_train_full)
    majority_baseline = float(prep["dryrun"]["majority_baseline"])

    if model_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                max_iter=80,
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=8,
            ),
        )
        clf.fit(X.iloc[idx_train], y[idx_train])
        mlp = clf.named_steps["mlpclassifier"]
        split_probs = {
            split_name: align_probability_columns(mlp.classes_, clf.predict_proba(X.iloc[idx]), n_classes)
            for split_name, idx in split_indices.items()
        }
        training_history = pd.DataFrame({
            "epoch": np.arange(1, len(getattr(mlp, "loss_curve_", [])) + 1),
            "train_loss": getattr(mlp, "loss_curve_", []),
        })
        if hasattr(mlp, "validation_scores_"):
            training_history["internal_validation_accuracy"] = pd.Series(mlp.validation_scores_)
            training_history["internal_validation_source"] = "sklearn_train_split_early_stopping"
        model_meta = {
            "model": "mlp",
            "implementation": "sklearn_MLPClassifier",
            "calibration_used_for_training": False,
        }
    elif model_name == "logistic_multinomial":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1500,
                solver="lbfgs",
                multi_class="multinomial",
                class_weight=None,
                random_state=seed,
                n_jobs=-1,
            ),
        )
        clf.fit(X.iloc[idx_train], y[idx_train])
        logit = clf.named_steps["logisticregression"]
        split_probs = {
            split_name: align_probability_columns(logit.classes_, clf.predict_proba(X.iloc[idx]), n_classes)
            for split_name, idx in split_indices.items()
        }
        training_history = pd.DataFrame([{
            "epoch": int(getattr(logit, "n_iter_", [0])[0] if len(getattr(logit, "n_iter_", [])) else 0),
            "train_loss": float("nan"),
            "train_accuracy": float("nan"),
            "train_macro_f1": float("nan"),
            "internal_validation_source": "not_used_for_logistic_multinomial",
        }])
        model_meta = {
            "model": "logistic_multinomial",
            "implementation": "sklearn_LogisticRegression_multinomial",
            "max_iter": 1500,
            "class_weight": "none",
            "calibration_used_for_training": False,
            "best_checkpoint_summary": {
                "enabled": False,
                "restore_best_checkpoint": False,
                "metric": None,
                "best_epoch": None,
                "best_metric_value": None,
                "epochs_completed": int(getattr(logit, "n_iter_", [0])[0] if len(getattr(logit, "n_iter_", [])) else 0),
                "stopped_early": False,
                "internal_validation_source": "not_used_for_logistic_multinomial",
            },
        }
    elif model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        resolved_hgb_max_iter = int(hgb_max_iter if hgb_max_iter is not None else min(max(epochs, 50), 200))
        resolved_hgb_l2 = float(hgb_l2_regularization if hgb_l2_regularization is not None else max(weight_decay, 1e-4))
        clf = HistGradientBoostingClassifier(
            max_iter=resolved_hgb_max_iter,
            learning_rate=float(hgb_learning_rate),
            max_leaf_nodes=int(hgb_max_leaf_nodes),
            l2_regularization=resolved_hgb_l2,
            min_samples_leaf=int(hgb_min_samples_leaf),
            early_stopping=True,
            validation_fraction=min(max(val_fraction, 0.05), 0.20),
            n_iter_no_change=max(patience, 10),
            random_state=seed,
        )
        clf.fit(X.iloc[idx_train], y[idx_train])
        split_probs = {
            split_name: align_probability_columns(clf.classes_, clf.predict_proba(X.iloc[idx]), n_classes)
            for split_name, idx in split_indices.items()
        }
        training_history = pd.DataFrame([{
            "epoch": int(getattr(clf, "n_iter_", 0)),
            "train_loss": float("nan"),
            "train_accuracy": float("nan"),
            "train_macro_f1": float("nan"),
            "internal_validation_source": "sklearn_hist_gradient_boosting_internal_validation",
        }])
        model_meta = {
            "model": "hist_gradient_boosting",
            "implementation": "sklearn_HistGradientBoostingClassifier",
            "max_iter": resolved_hgb_max_iter,
            "n_iter_": int(getattr(clf, "n_iter_", 0)),
            "learning_rate": float(hgb_learning_rate),
            "max_leaf_nodes": int(hgb_max_leaf_nodes),
            "l2_regularization": resolved_hgb_l2,
            "min_samples_leaf": int(hgb_min_samples_leaf),
            "calibration_used_for_training": False,
            "best_checkpoint_summary": {
                "enabled": True,
                "restore_best_checkpoint": True,
                "metric": "sklearn_internal_validation",
                "best_epoch": int(getattr(clf, "n_iter_", 0)),
                "best_metric_value": None,
                "epochs_completed": int(getattr(clf, "n_iter_", 0)),
                "stopped_early": bool(getattr(clf, "n_iter_", 0) < resolved_hgb_max_iter),
                "internal_validation_source": "train split only via sklearn HistGradientBoostingClassifier",
            },
        }
    elif model_name == "torch_mlp_cuda":
        split_probs, training_history, model_meta = train_torch_mlp_cuda(
            X=X,
            y=y,
            split_indices=split_indices,
            idx_train=idx_train,
            n_classes=n_classes,
            seed=seed,
            hidden_layers=hidden_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            torch_amp=torch_amp,
            early_stopping=early_stopping,
            val_fraction=val_fraction,
            patience=patience,
            early_stopping_metric=early_stopping_metric,
            restore_best_checkpoint=restore_best_checkpoint,
        )
        model_meta = {"model": "torch_mlp_cuda", **model_meta}
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    metrics_rows = prediction_metrics_rows(split_probs, split_indices, y, majority_baseline, n_classes)
    per_class_metrics, class_distribution, predicted_distribution = classification_detail_tables(
        split_probs=split_probs,
        split_indices=split_indices,
        y=y,
        class_names=class_names,
    )

    alpha_grid, alpha_grid_name = resolve_alpha_grid(alpha_grid_arg, prep["dryrun"])
    group_col = prep["dryrun"]["group"]["column"]
    z_col = prep["dryrun"]["z"]["column"]
    group_a, group_b = group_pair(prep["dryrun"]["group"]["name"], examples[group_col])
    conformal_score_meta = score_metadata(score, raps_lambda, raps_k_reg, saps_lambda)
    eval_df, threshold_rows = build_eval_frame(
        examples=examples,
        split_indices=split_indices,
        split_probs=split_probs,
        y=y,
        alpha_grid=alpha_grid,
        group_col=group_col,
        z_col=z_col,
        score=score,
        raps_lambda=raps_lambda,
        raps_k_reg=raps_k_reg,
        saps_lambda=saps_lambda,
    )
    raw_aggregate, raw_local, raw_gap = curve_tables(
        eval_df=eval_df,
        group_a=group_a,
        group_b=group_b,
        method="raw",
        weight_col="unit_weight",
    )
    coverage_set_size_rows = []
    for (split_name, alpha), sub in eval_df.groupby(["split", "alpha"]):
        coverage_set_size_rows.append({
            "method": "raw",
            "split": split_name,
            "alpha": float(alpha),
            "coverage": weighted_mean(sub["covered"], sub["unit_weight"]),
            "mean_set_size": weighted_mean(sub["set_size"], sub["unit_weight"]),
            "n": int(len(sub)),
        })

    prob_all = np.zeros((len(examples), n_classes), dtype=float)
    for split_name, idx in split_indices.items():
        prob_all[idx, :] = split_probs[split_name]
    h_all, h_cols, h_rule = probability_profile(prob_all, n_classes)
    weights, propensity_diag, weight_summary, ess_summary, propensity_meta = fit_overlap_weights(
        examples=examples,
        h_all=h_all,
        group_col=group_col,
        z_col=z_col,
        group_a=group_a,
        group_b=group_b,
        propensity_clip=propensity_clip,
        trim_quantile=trim_quantile,
        max_weight=max_weight,
        seed=seed,
    )
    balance = balance_diagnostics(weights, h_all, group_a=group_a, group_b=group_b)
    eval_weighted = eval_df.merge(weights[["row_pos", "overlap_weight"]], on="row_pos", how="left")
    eval_weighted["overlap_weight"] = eval_weighted["overlap_weight"].fillna(0.0)
    ipw_aggregate, ipw_local, ipw_gap = curve_tables(
        eval_df=eval_weighted,
        group_a=group_a,
        group_b=group_b,
        method="ipw_overlap",
        weight_col="overlap_weight",
    )
    for (split_name, alpha), sub in eval_weighted.groupby(["split", "alpha"]):
        coverage_set_size_rows.append({
            "method": "ipw_overlap",
            "split": split_name,
            "alpha": float(alpha),
            "coverage": weighted_mean(sub["covered"], sub["overlap_weight"]),
            "mean_set_size": weighted_mean(sub["set_size"], sub["overlap_weight"]),
            "n": int(len(sub)),
        })
    coverage_set_size = pd.DataFrame(coverage_set_size_rows)
    coverage_by_alpha = coverage_set_size[["method", "split", "alpha", "coverage", "n"]].copy()
    mean_set_size_by_alpha = coverage_set_size[["method", "split", "alpha", "mean_set_size", "n"]].copy()
    thresholds = [0.0, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    diag_by_split, decomp_by_split, material_counts, near_zero, raw_vs_ipw = simpson_diagnostics(
        raw_gap=raw_gap,
        ipw_gap=ipw_gap,
        thresholds=thresholds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "training_history.csv", training_history)
    best_checkpoint_summary = model_meta.get("best_checkpoint_summary", {
        "enabled": False,
        "restore_best_checkpoint": False,
        "metric": None,
        "best_epoch": None,
        "best_metric_value": None,
        "epochs_completed": int(len(training_history)),
        "stopped_early": False,
    })
    write_json(output_dir / "best_checkpoint_summary.json", best_checkpoint_summary)
    write_csv(output_dir / "prediction_metrics.csv", metrics_rows)
    write_csv(output_dir / "per_class_metrics.csv", per_class_metrics)
    write_csv(output_dir / "class_distribution.csv", class_distribution)
    write_csv(output_dir / "predicted_distribution.csv", predicted_distribution)
    write_csv(output_dir / "conformal_thresholds.csv", threshold_rows)
    write_csv(output_dir / "raw_aggregate_curve.csv", raw_aggregate)
    write_csv(output_dir / "raw_local_curve.csv", raw_local)
    write_csv(output_dir / "raw_gap_by_alpha.csv", raw_gap)
    write_csv(output_dir / "propensity_diagnostics.csv", propensity_diag)
    write_csv(output_dir / "weight_summary.csv", weight_summary)
    write_csv(output_dir / "balance_diagnostics_before_after.csv", balance)
    write_csv(output_dir / "ess_summary.csv", ess_summary)
    write_csv(output_dir / "ipw_aggregate_curve.csv", ipw_aggregate)
    write_csv(output_dir / "ipw_local_curve.csv", ipw_local)
    write_csv(output_dir / "ipw_gap_by_alpha.csv", ipw_gap)
    write_csv(output_dir / "coverage_set_size_by_alpha.csv", coverage_set_size)
    write_csv(output_dir / "coverage_by_alpha.csv", coverage_by_alpha)
    write_csv(output_dir / "mean_set_size_by_alpha.csv", mean_set_size_by_alpha)
    write_csv(output_dir / "simpson_diagnostics_repair.csv", diag_by_split.get("repair", pd.DataFrame()))
    write_csv(output_dir / "simpson_diagnostics_test.csv", diag_by_split.get("test", pd.DataFrame()))
    write_csv(output_dir / "reversal_decomposition_repair.csv", decomp_by_split.get("repair", pd.DataFrame()))
    write_csv(output_dir / "reversal_decomposition_test.csv", decomp_by_split.get("test", pd.DataFrame()))
    write_csv(output_dir / "material_reversal_counts.csv", material_counts)
    write_csv(output_dir / "near_zero_gap_diagnostics.csv", near_zero)
    write_csv(output_dir / "raw_vs_ipw_comparison.csv", raw_vs_ipw)
    write_csv(output_dir / "propensity_h_profile.csv", h_all[h_cols].describe().reset_index().rename(columns={"index": "stat"}))
    probability_paths = {
        split_name: write_probability_file(
            output_dir,
            split_name,
            examples,
            idx,
            y,
            split_probs[split_name],
            class_names,
            group_col,
            z_col,
        )
        for split_name, idx in split_indices.items()
    }

    test_metrics = next(row for row in metrics_rows if row["split"] == "test")
    repair_metrics = next(row for row in metrics_rows if row["split"] == "repair")
    cal_metrics = next(row for row in metrics_rows if row["split"] == "calibration")
    train_metrics = next(row for row in metrics_rows if row["split"] == "train")
    repair_diag = diag_by_split.get("repair", pd.DataFrame())
    test_diag = diag_by_split.get("test", pd.DataFrame())
    balance_summary = (
        balance.groupby(["split", "diagnostic"])[["abs_before", "abs_after"]]
        .max()
        .reset_index()
        .to_dict("records")
    )
    min_ess_fraction = float(ess_summary["ess_fraction"].min()) if len(ess_summary) else float("nan")
    near_zero_warning = bool(near_zero["near_zero_aggregate_gap_count"].gt(0).any()) if len(near_zero) else False
    summary = {
        "training_completed": True,
        "model": model_name,
        "model_metadata": model_meta,
        "design": prep["dryrun"]["design"],
        "unit": prep["dryrun"]["unit"],
        "y_mode": prep["dryrun"].get("y_mode", "top20_with_other"),
        "top_k": int(prep["dryrun"].get("label_schema", {}).get("top_k", 0)),
        "feature_mode": prep["dryrun"].get("feature_mode", "safe"),
        "strong_context_ablation": bool(prep["dryrun"].get("strong_context_ablation", False)),
        "group": prep["dryrun"]["group"]["name"],
        "z": prep["dryrun"]["z"]["name"],
        "group_a": group_a,
        "group_b": group_b,
        "gap_direction": "delta = V_group_a - V_group_b",
        "n_classes": int(n_classes),
        "class_labels": [str(x) for x in class_names.tolist()],
        "n_train_full": int(len(idx_train_full)),
        "n_train_used": int(len(idx_train)),
        "train_cap_active": bool(train_cap_active),
        "train_cap_source": train_cap_source,
        "max_train_examples": int(max_train_examples) if max_train_examples is not None else None,
        "majority_baseline": majority_baseline,
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "train_weighted_f1": float(train_metrics["weighted_f1"]),
        "train_log_loss": float(train_metrics["log_loss"]),
        "calibration_accuracy": float(cal_metrics["accuracy"]),
        "calibration_macro_f1": float(cal_metrics["macro_f1"]),
        "calibration_weighted_f1": float(cal_metrics["weighted_f1"]),
        "calibration_log_loss": float(cal_metrics["log_loss"]),
        "repair_accuracy": float(repair_metrics["accuracy"]),
        "repair_macro_f1": float(repair_metrics["macro_f1"]),
        "repair_weighted_f1": float(repair_metrics["weighted_f1"]),
        "repair_log_loss": float(repair_metrics["log_loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_log_loss": float(test_metrics["log_loss"]),
        "best_checkpoint_summary": best_checkpoint_summary,
        "alpha_grid_name": alpha_grid_name,
        "alpha_count": int(len(alpha_grid)),
        "alpha_grid": [float(x) for x in alpha_grid],
        "coverage_set_size_by_alpha": coverage_set_size.to_dict("records"),
        "conformal_score": conformal_score_meta["score"],
        "conformal_score_metadata": conformal_score_meta,
        "propensity": {
            **propensity_meta,
            "h_rule": h_rule,
            "h_columns": h_cols,
            "forbidden_in_H": ["true_G", "true_Y", "true_Z", "set_size", "alpha", "threshold", "activity_code", "tier_labels", "duration", "stop_time", "where_who", "activity_derived_variables"],
            "propensity_clip": float(propensity_clip),
            "trim_quantile": float(trim_quantile),
            "max_weight": float(max_weight),
        },
        "repair_raw_vs_ipw": raw_vs_ipw[raw_vs_ipw["split"].eq("repair")].to_dict("records") if len(raw_vs_ipw) else [],
        "test_raw_vs_ipw": raw_vs_ipw[raw_vs_ipw["split"].eq("test")].to_dict("records") if len(raw_vs_ipw) else [],
        "balance_summary_max_abs": balance_summary,
        "min_ess_fraction": min_ess_fraction,
        "near_zero_aggregate_gap_warning": near_zero_warning,
        "leakage_audit_pass": bool(prep["dryrun"]["leakage_audit_pass"]),
        "label_mapping": {
            "lexicon_mapping_status": prep["dryrun"].get("lexicon_mapping_status"),
            "activity_label_map_used": prep["dryrun"].get("activity_label_map_used_path"),
            "fallback_or_manual_y_labels": prep["dryrun"].get("fallback_or_manual_y_labels", []),
        },
        "label_metadata_notes": [
            "Provide the official ATUS activity lexicon with --lexicon-path; otherwise fallback names may be used.",
            "Review any fallback/manual activity labels in activity_label_map_used.csv before reporting human-readable class names.",
        ],
        "outputs": {
            "prediction_metrics": str(output_dir / "prediction_metrics.csv"),
            "training_history": str(output_dir / "training_history.csv"),
            "best_checkpoint_summary": str(output_dir / "best_checkpoint_summary.json"),
            "per_class_metrics": str(output_dir / "per_class_metrics.csv"),
            "class_distribution": str(output_dir / "class_distribution.csv"),
            "predicted_distribution": str(output_dir / "predicted_distribution.csv"),
            "raw_aggregate_curve": str(output_dir / "raw_aggregate_curve.csv"),
            "raw_local_curve": str(output_dir / "raw_local_curve.csv"),
            "raw_gap_by_alpha": str(output_dir / "raw_gap_by_alpha.csv"),
            "ipw_aggregate_curve": str(output_dir / "ipw_aggregate_curve.csv"),
            "ipw_local_curve": str(output_dir / "ipw_local_curve.csv"),
            "ipw_gap_by_alpha": str(output_dir / "ipw_gap_by_alpha.csv"),
            "coverage_set_size_by_alpha": str(output_dir / "coverage_set_size_by_alpha.csv"),
            "coverage_by_alpha": str(output_dir / "coverage_by_alpha.csv"),
            "mean_set_size_by_alpha": str(output_dir / "mean_set_size_by_alpha.csv"),
            "propensity_diagnostics": str(output_dir / "propensity_diagnostics.csv"),
            "weight_summary": str(output_dir / "weight_summary.csv"),
            "balance_diagnostics_before_after": str(output_dir / "balance_diagnostics_before_after.csv"),
            "ess_summary": str(output_dir / "ess_summary.csv"),
            "simpson_diagnostics_repair": str(output_dir / "simpson_diagnostics_repair.csv"),
            "simpson_diagnostics_test": str(output_dir / "simpson_diagnostics_test.csv"),
            "curve_summary": str(output_dir / "curve_summary.json"),
            "class_probabilities": probability_paths,
        },
    }
    write_json(output_dir / "curve_summary.json", summary)
    return {
        **summary,
    }


def main() -> int:
    args = parse_args()
    opts = resolve_design_options(args)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prep = prepare_design(
        raw_dir=opts["raw_dir"],
        extract_dir=opts["extract_dir"],
        output_dir=output_dir,
        design_name=opts["design_name"],
        unit=opts["unit"],
        top_k=opts["top_k"],
        group=opts["group"],
        z=opts["z"],
        seed=args.seed,
        exclude_data_code_activities=opts["exclude_data_code_activities"],
        lexicon_path=opts["lexicon_path"],
        use_cache=not args.no_cache,
        feature_mode=args.feature_mode,
        y_mode=opts["y_mode"],
    )
    dryrun = prep["dryrun"]
    write_json(output_dir / "dryrun.json", dryrun)
    write_csv(output_dir / "prep_checks.csv", prep["checks"])
    write_prep_aggregate(prep["root_dir"], opts["design_name"], dryrun, prep["checks"], command_string(sys.argv))

    if args.dry_run:
        print("\n=== ATUS dry-run complete ===")
        print(f"design: {opts['design_name']}")
        print(f"y_mode: {opts['y_mode']}")
        print(f"dry_run_success: {dryrun['dry_run_success']}")
        print(f"n_train/n_cal/n_repair/n_test: {dryrun['split_counts']['train']['n_examples']} / {dryrun['split_counts']['calibration']['n_examples']} / {dryrun['split_counts']['repair']['n_examples']} / {dryrun['split_counts']['test']['n_examples']}")
        print(f"feature_matrix_shape: {dryrun['feature_matrix_shape']}")
        print(f"leakage_audit_pass: {dryrun['leakage_audit_pass']}")
        print(f"majority_baseline: {dryrun['majority_baseline']}")
        print(f"effective_number_of_labels: {dryrun['effective_number_of_labels']}")
        print(f"recommended_alpha_grid: {dryrun['recommended_alpha_grid']}")
        print(f"dryrun_json: {prep['root_dir'] / (opts['design_name'] + '_dryrun.json')}")
        return 0

    train_result = run_curve_experiment(
        prep=prep,
        output_dir=output_dir,
        max_train_examples=opts["max_train_examples"],
        train_cap_source=opts["train_cap_source"],
        seed=args.seed,
        propensity_clip=args.propensity_clip,
        trim_quantile=args.trim_quantile,
        max_weight=args.max_weight,
        model_name=opts["model"],
        hidden_layers=parse_hidden_layers(args.hidden_layers),
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=args.class_weight,
        torch_amp=args.torch_amp,
        early_stopping=args.early_stopping,
        val_fraction=args.val_fraction,
        patience=args.patience,
        early_stopping_metric=args.early_stopping_metric,
        restore_best_checkpoint=args.restore_best_checkpoint,
        alpha_grid_arg=args.alpha_grid,
        hgb_max_iter=args.hgb_max_iter,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
        hgb_l2_regularization=args.hgb_l2_regularization,
        hgb_min_samples_leaf=args.hgb_min_samples_leaf,
        score=args.score,
        raps_lambda=args.raps_lambda,
        raps_k_reg=args.raps_k_reg,
        saps_lambda=args.saps_lambda,
    )
    write_json(output_dir / "run_result.json", {"dryrun": dryrun, "training": train_result})
    cmd_log_name = "run_command.txt"
    with (output_dir / cmd_log_name).open("a", encoding="utf-8") as f:
        f.write("\n=== Actual run command ===\n")
        f.write(command_string(sys.argv) + "\n")
        f.write(f"Conformal score: {args.score}; raps_lambda={args.raps_lambda}; raps_k_reg={args.raps_k_reg}; saps_lambda={args.saps_lambda}\n")
        f.write(f"Result: training_completed={train_result.get('training_completed')}, train_cap_active={train_result.get('train_cap_active')}, n_train_used={train_result.get('n_train_used')}, output_dir={output_dir}\n")

    comparison = pd.DataFrame(train_result.get("repair_raw_vs_ipw", []) + train_result.get("test_raw_vs_ipw", []))
    def format_counts(split_name: str, raw_col: str, ipw_col: str) -> str:
        sub = comparison[comparison["split"].eq(split_name)] if len(comparison) else pd.DataFrame()
        if sub.empty:
            return "unavailable"
        return ", ".join(f"a={row.alpha:g}: {getattr(row, raw_col):.6g}->{getattr(row, ipw_col):.6g}" for row in sub.itertuples())

    balance_df = pd.DataFrame(train_result.get("balance_summary_max_abs", []))
    balance_bits = []
    if len(balance_df):
        for row in balance_df.itertuples():
            balance_bits.append(f"{row.split}/{row.diagnostic}: {row.abs_before:.4g}->{row.abs_after:.4g}")

    print("\n=== ATUS curve run complete ===")
    print(f"completed_successfully: {train_result.get('training_completed') and train_result.get('leakage_audit_pass')}")
    print(f"design: {opts['design_name']}")
    print(f"y_mode: {opts['y_mode']}")
    print(f"feature_mode: {args.feature_mode}")
    print(f"train_cap: active={train_result.get('train_cap_active')} source={train_result.get('train_cap_source')} n_train_used={train_result.get('n_train_used')} of {train_result.get('n_train_full')}")
    print(f"model: {train_result.get('model')}")
    print(f"alpha_grid: {train_result.get('alpha_grid_name')} count={train_result.get('alpha_count')}")
    print(f"conformal_score: {train_result.get('conformal_score')}")
    print(f"model accuracy/macro-F1/weighted-F1/majority baseline: {train_result.get('test_accuracy'):.6f} / {train_result.get('test_macro_f1'):.6f} / {train_result.get('test_weighted_f1'):.6f} / {train_result.get('majority_baseline'):.6f}")
    print(f"repair raw->IPW hard reversal counts: {format_counts('repair', 'raw_hard_sign_reversal_count', 'ipw_hard_sign_reversal_count')}")
    print(f"test raw->IPW hard reversal counts: {format_counts('test', 'raw_hard_sign_reversal_count', 'ipw_hard_sign_reversal_count')}")
    print(f"repair raw->IPW conflict strength: {format_counts('repair', 'raw_conflict_strength', 'ipw_conflict_strength')}")
    print(f"test raw->IPW conflict strength: {format_counts('test', 'raw_conflict_strength', 'ipw_conflict_strength')}")
    print(f"TV/SMD before->after: {'; '.join(balance_bits) if balance_bits else 'unavailable'}")
    print(f"min ESS fraction: {train_result.get('min_ess_fraction'):.6f}")
    print(f"near_zero_aggregate_gap_warning: {train_result.get('near_zero_aggregate_gap_warning')}")
    print(f"results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
