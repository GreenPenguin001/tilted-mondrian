"""Canonical score construction, conformal thresholds, and Table 1 metrics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import DatasetUnit, SplitData


BIOBIAS_ALPHAS = tuple(round(value, 3) for value in np.arange(0.02, 0.1001, 0.005))
FOLKTABLES_ALPHAS = tuple(round(value, 3) for value in np.arange(0.01, 0.1001, 0.001))
ATUS_ALPHAS = tuple(round(value, 2) for value in np.arange(0.02, 0.2501, 0.01))


@dataclass
class EvaluationResult:
    summary: dict[str, Any]
    cell_grid: pd.DataFrame
    group_grid: pd.DataFrame
    thresholds: pd.DataFrame


def alpha_grid(unit: DatasetUnit) -> tuple[float, ...]:
    if unit.dataset == "biobias":
        return BIOBIAS_ALPHAS
    if unit.dataset.startswith("folktables"):
        return FOLKTABLES_ALPHAS
    return ATUS_ALPHAS


def score_matrix(unit: DatasetUnit, probabilities: np.ndarray, score: str) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    score = score.lower()
    if score == "simple":
        return 1.0 - p
    n, n_classes = p.shape
    order = np.argsort(-p, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(1, n_classes + 1)
    p_sorted = np.take_along_axis(p, order, axis=1)
    if unit.dataset == "biobias":
        before_sorted = np.column_stack((np.zeros(n), np.cumsum(p_sorted, axis=1)[:, :-1]))
        before = before_sorted[np.arange(n)[:, None], rank - 1]
        if score == "aps":
            return before + 0.5 * p
        if score == "raps":
            return before + 0.5 * p + 0.2 * np.maximum(rank - 5, 0)
        if score == "saps":
            return np.where(rank == 1, 0.5 * p, p_sorted[:, :1] + 0.2 * (rank - 1.5))
    else:
        cumulative_sorted = np.cumsum(p_sorted, axis=1)
        cumulative = cumulative_sorted[np.arange(n)[:, None], rank - 1]
        if score == "aps":
            return cumulative
        if score == "raps":
            return cumulative + 0.01 * np.maximum(rank - 5, 0)
        if score == "saps":
            return np.where(rank == 1, 0.5 * p_sorted[:, :1], p_sorted[:, :1] + 0.2 * (rank - 1.5))
    raise ValueError(f"Unsupported score={score} for dataset={unit.dataset}")


def true_scores(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    return scores[np.arange(len(y)), np.asarray(y, dtype=int)]


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = len(finite)
    if n == 0:
        return float("inf")
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if rank > n:
        return float("inf")
    return float(np.partition(finite, rank - 1)[rank - 1])


def sorted_values(values: np.ndarray) -> list[Any]:
    return sorted(np.unique(values).tolist(), key=lambda value: str(value))


def build_thresholds(unit: DatasetUnit, calibration: SplitData, score: str) -> tuple[pd.DataFrame, dict[float, Any]]:
    scores = score_matrix(unit, calibration.probs, score)
    truth = true_scores(scores, calibration.y)
    thresholds: dict[float, Any] = {}
    rows: list[dict[str, Any]] = []
    for alpha in alpha_grid(unit):
        if unit.dataset == "biobias":
            group_thresholds = {
                group: conformal_quantile(truth[calibration.g == group], alpha)
                for group in sorted_values(calibration.g)
            }
            thresholds[alpha] = group_thresholds
            for group, threshold in group_thresholds.items():
                rows.append(
                    {
                        "score": score,
                        "policy": "groupwise",
                        "alpha": alpha,
                        "group": group,
                        "threshold": threshold,
                    }
                )
        else:
            threshold = conformal_quantile(truth, alpha)
            thresholds[alpha] = threshold
            rows.append(
                {"score": score, "policy": "pooled", "alpha": alpha, "group": "ALL", "threshold": threshold}
            )
    return pd.DataFrame(rows), thresholds


def threshold_vector(groups: np.ndarray, threshold: Any) -> np.ndarray:
    if isinstance(threshold, dict):
        return np.asarray([threshold[group] for group in groups], dtype=float)
    return np.full(len(groups), float(threshold), dtype=float)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any() or float(weights[valid].sum()) <= 0:
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights >= 0)]
    denominator = float(np.sum(weights**2))
    if not len(weights) or denominator <= 0:
        return float("nan")
    return float(np.sum(weights) ** 2 / denominator)


def support_minimum(unit: DatasetUnit) -> int:
    if unit.dataset == "biobias":
        return 202
    if unit.dataset.startswith("folktables"):
        return 200
    return 1


def is_reversal(unit: DatasetUnit, aggregate_gap: float, local_gap: float) -> bool:
    if unit.dataset == "biobias":
        return bool(abs(aggregate_gap) >= 0.01 and abs(local_gap) >= 0.02 and aggregate_gap * local_gap < 0)
    if unit.dataset.startswith("folktables"):
        return bool(np.sign(aggregate_gap) != np.sign(local_gap))
    return bool(np.sign(aggregate_gap) * np.sign(local_gap) < 0)


def evaluate_score(
    unit: DatasetUnit,
    score: str,
    calibration: SplitData,
    test: SplitData,
    overlap_weights: np.ndarray,
    reported_ess_weights: np.ndarray | None = None,
    reported_ess_split: str = "test",
) -> EvaluationResult:
    if len(test.y) != len(overlap_weights):
        raise ValueError("Test weights and rows differ in length")
    threshold_frame, thresholds = build_thresholds(unit, calibration, score)
    scores = score_matrix(unit, test.probs, score)
    truth = true_scores(scores, test.y)
    groups = sorted_values(test.g)
    cells = sorted_values(test.z)
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups for {unit.key}, got {groups}")
    methods = {"raw": np.ones(len(test.y), dtype=float), "overlap": np.asarray(overlap_weights, dtype=float)}
    group_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    support_n = support_minimum(unit)

    # Prediction sets are formed once; raw and overlap estimates differ only in aggregation weights.
    for alpha in alpha_grid(unit):
        threshold = thresholds[alpha]
        row_threshold = threshold_vector(test.g, threshold)
        set_size = (scores <= row_threshold[:, None]).sum(axis=1).astype(float)
        covered = (truth <= row_threshold).astype(float)
        for method, weights in methods.items():
            group_values: dict[Any, dict[str, float]] = {}
            for group in groups:
                mask = test.g == group
                group_values[group] = {
                    "mean_set_size": weighted_mean(set_size[mask], weights[mask]),
                    "coverage": weighted_mean(covered[mask], weights[mask]),
                    "n": int(mask.sum()),
                    "weight_sum": float(weights[mask].sum()),
                    "ess": effective_sample_size(weights[mask]),
                }
                group_rows.append(
                    {
                        "dataset": unit.dataset,
                        "year": unit.year,
                        "score": score,
                        "method": method,
                        "alpha": alpha,
                        "group": group,
                        **group_values[group],
                    }
                )
            aggregate_gap = group_values[groups[0]]["mean_set_size"] - group_values[groups[1]]["mean_set_size"]
            for cell in cells:
                masks = [(test.g == group) & (test.z == cell) for group in groups]
                counts = [int(mask.sum()) for mask in masks]
                supported = bool(min(counts) >= support_n)
                local_values = [weighted_mean(set_size[mask], weights[mask]) if mask.any() else float("nan") for mask in masks]
                coverage_values = [weighted_mean(covered[mask], weights[mask]) if mask.any() else float("nan") for mask in masks]
                local_gap = float(local_values[0] - local_values[1])
                conflict_strength = float(max(-(aggregate_gap * local_gap), 0.0))
                reversal = bool(supported and is_reversal(unit, aggregate_gap, local_gap))
                cell_rows.append(
                    {
                        "dataset": unit.dataset,
                        "year": unit.year,
                        "score": score,
                        "method": method,
                        "alpha": alpha,
                        "z": cell,
                        "group_a": groups[0],
                        "group_b": groups[1],
                        "n_a_z": counts[0],
                        "n_b_z": counts[1],
                        "aggregate_gap": aggregate_gap,
                        "local_gap": local_gap,
                        "set_size_a_z": local_values[0],
                        "set_size_b_z": local_values[1],
                        "coverage_a_z": coverage_values[0],
                        "coverage_b_z": coverage_values[1],
                        "supported": supported,
                        "is_reversal": reversal,
                        "conflict_strength": conflict_strength,
                    }
                )

    cell_grid = pd.DataFrame(cell_rows)
    group_grid = pd.DataFrame(group_rows)
    summaries: dict[str, dict[str, Any]] = {}
    for method in methods:
        selected = cell_grid[(cell_grid["method"] == method) & cell_grid["supported"]]
        all_cells = cell_grid[cell_grid["method"] == method]
        summaries[method] = {
            "reversal_count": int(selected["is_reversal"].sum()),
            "supported_conflict_strength": float(selected["conflict_strength"].sum()),
            "high_conflict_count": int((selected["conflict_strength"] > 0.02).sum()),
            "supported_cells": int(len(selected)),
            "unsupported_cells": int(len(all_cells) - len(selected)),
        }
    raw_mean_set_size = weighted_mean(
        np.concatenate(
            [
                (scores <= threshold_vector(test.g, thresholds[alpha])[:, None]).sum(axis=1).astype(float)
                for alpha in alpha_grid(unit)
            ]
        ),
        np.ones(len(test.y) * len(alpha_grid(unit))),
    )
    overlap_mean_set_size = weighted_mean(
        np.concatenate(
            [
                (scores <= threshold_vector(test.g, thresholds[alpha])[:, None]).sum(axis=1).astype(float)
                for alpha in alpha_grid(unit)
            ]
        ),
        np.tile(overlap_weights, len(alpha_grid(unit))),
    )
    raw_strength = summaries["raw"]["supported_conflict_strength"]
    overlap_strength = summaries["overlap"]["supported_conflict_strength"]
    ess_weights = overlap_weights if reported_ess_weights is None else np.asarray(reported_ess_weights, dtype=float)
    summary = {
        "dataset": unit.dataset,
        "year": unit.year,
        "score": score,
        "alpha_grid": json.dumps(alpha_grid(unit)),
        "threshold_policy": "groupwise" if unit.dataset == "biobias" else "pooled",
        "raw_reversal_count": summaries["raw"]["reversal_count"],
        "overlap_reversal_count": summaries["overlap"]["reversal_count"],
        "raw_supported_conflict_strength": raw_strength,
        "overlap_supported_conflict_strength": overlap_strength,
        "paired_absolute_conflict_reduction": raw_strength - overlap_strength,
        "paired_relative_conflict_reduction": (raw_strength - overlap_strength) / raw_strength if raw_strength > 0 else float("nan"),
        "raw_high_conflict_count": summaries["raw"]["high_conflict_count"],
        "overlap_high_conflict_count": summaries["overlap"]["high_conflict_count"],
        "number_supported_cells": summaries["raw"]["supported_cells"],
        "number_unsupported_cells": summaries["raw"]["unsupported_cells"],
        "raw_average_set_size": raw_mean_set_size,
        "overlap_average_set_size": overlap_mean_set_size,
        "average_set_size_change": overlap_mean_set_size - raw_mean_set_size,
        "ess": effective_sample_size(ess_weights),
        "ess_fraction": effective_sample_size(ess_weights) / len(ess_weights),
        "ess_reporting_split": reported_ess_split,
        "same_test_rows_raw_overlap": True,
        "same_calibrated_sets_raw_overlap": True,
        "same_supported_universe_raw_overlap": bool(
            summaries["raw"]["supported_cells"] == summaries["overlap"]["supported_cells"]
            and summaries["raw"]["unsupported_cells"] == summaries["overlap"]["unsupported_cells"]
        ),
    }
    return EvaluationResult(summary=summary, cell_grid=cell_grid, group_grid=group_grid, thresholds=threshold_frame)
