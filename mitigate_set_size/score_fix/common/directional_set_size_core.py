from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from input_loading import LoadedDataset, SplitData
from no_free_repair_core import (
    ScoreFixConfig,
    compute_baseline_group_thresholds,
    compute_thresholds_for_target_counts_from_true_scores,
    evaluate_split,
    halton_value,
    labelwise_scores_from_probs,
    repair_cell_indices,
    row_thresholds,
    theta_scale_from_grid,
    true_scores,
    validate_dataset,
)


OPTIMIZER_NAME = "directional_set_size_boundary_search"
SET_SIZE_TOLERANCE = 1e-9
SMALLER_SIDE_BUDGET = 0.02


@dataclass(frozen=True)
class DirectionalSetSizeConfig(ScoreFixConfig):
    optimizer_name: str = OPTIMIZER_NAME
    set_size_tolerance: float = SET_SIZE_TOLERANCE
    smaller_side_budget: float = SMALLER_SIDE_BUDGET


@dataclass(frozen=True)
class DirectionalSetSizeArtifacts:
    summary_rows: list[dict[str, Any]]
    offset_rows: list[dict[str, Any]]
    cell_rows: list[dict[str, Any]]
    group_rows: list[dict[str, Any]]
    search_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    protocol_rows: list[dict[str, Any]]


class ProtocolEventTrace:
    """Enforce the repair, calibration, and test execution order."""

    EVENT_ORDER = (
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

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._next_event = 0
        self._frozen_theta: np.ndarray | None = None
        self._frozen_calibration_thresholds: dict[str, float] | None = None

    def record(
        self,
        event: str,
        *,
        theta: np.ndarray | None = None,
        thresholds: dict[str, float] | None = None,
        detail: str = "",
    ) -> None:
        expected = self.EVENT_ORDER[self._next_event]
        if event != expected:
            raise RuntimeError(
                f"Protocol event out of order: expected {expected}, got {event}"
            )
        if self._frozen_theta is not None:
            self.assert_theta_unchanged(theta)
        if event == "theta_frozen":
            if theta is None:
                raise RuntimeError("theta_frozen requires the selected theta")
            self._frozen_theta = np.asarray(theta, dtype=float).copy()
            self._frozen_theta.setflags(write=False)
        if event == "calibration_thresholds_frozen":
            if thresholds is None:
                raise RuntimeError(
                    "calibration_thresholds_frozen requires calibration thresholds"
                )
            self._frozen_calibration_thresholds = {
                str(group): float(value) for group, value in thresholds.items()
            }
        if event.startswith("test_") or event in {
            "baseline_test_evaluated",
            "repaired_test_evaluated",
        }:
            self.assert_calibration_thresholds_unchanged(thresholds)
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "event": event,
                "detail": detail,
            }
        )
        self._next_event += 1

    def assert_theta_unchanged(self, theta: np.ndarray | None) -> None:
        if theta is None or self._frozen_theta is None:
            raise RuntimeError("Frozen theta must be supplied after theta_frozen")
        if not np.array_equal(np.asarray(theta, dtype=float), self._frozen_theta):
            raise RuntimeError("Theta changed after theta_frozen")

    def assert_calibration_thresholds_unchanged(
        self, thresholds: dict[str, float] | None
    ) -> None:
        if thresholds is None or self._frozen_calibration_thresholds is None:
            raise RuntimeError(
                "Frozen calibration thresholds are required before test evaluation"
            )
        if not thresholds_equal(thresholds, self._frozen_calibration_thresholds):
            raise RuntimeError(
                "Calibration thresholds changed after calibration_thresholds_frozen"
            )

    def assert_complete(self) -> None:
        if self._next_event != len(self.EVENT_ORDER):
            missing = self.EVENT_ORDER[self._next_event :]
            raise RuntimeError(f"Incomplete protocol event trace: {missing}")

    def validation(self) -> dict[str, bool]:
        positions = {row["event"]: int(row["sequence"]) for row in self.events}
        complete = set(positions) == set(self.EVENT_ORDER)
        return {
            "protocol_event_trace_complete": complete,
            "theta_frozen_before_calibration": bool(
                complete
                and positions["theta_frozen"] < positions["calibration_started"]
            ),
            "calibration_frozen_before_test": bool(
                complete
                and positions["calibration_thresholds_frozen"]
                < positions["test_started"]
            ),
            "test_scores_after_frozen_calibration": bool(
                complete
                and positions["calibration_thresholds_frozen"]
                < positions["test_scores_constructed"]
            ),
            "both_test_evaluations_after_frozen_calibration": bool(
                complete
                and positions["calibration_thresholds_frozen"]
                < positions["baseline_test_evaluated"]
                < positions["repaired_test_evaluated"]
            ),
        }


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assert_output_within_experiment(path: Path) -> Path:
    root = experiment_root().parent
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"Output path must remain under the package root {root}; got {resolved}"
        )
    return resolved


def set_size_config_from_score_config(
    config: ScoreFixConfig,
) -> DirectionalSetSizeConfig:
    values = asdict(config)
    values["optimizer_name"] = OPTIMIZER_NAME
    return DirectionalSetSizeConfig(**values)


def run_directional_set_size_experiment(
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    score_method: str,
    result_dir: Path,
    command: str,
) -> DirectionalSetSizeArtifacts:
    """Run one repair-only directional set-size search and held-out evaluation."""
    validate_dataset(data)
    score_method = str(score_method).lower()
    if score_method not in config.score_methods:
        raise ValueError(f"Unsupported score method: {score_method}")
    result_dir = assert_output_within_experiment(result_dir)
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    protocol = ProtocolEventTrace()
    repair = data.splits["repair"]
    repair_scores = labelwise_scores_from_probs(
        repair.probs,
        score_method,
        raps_lambda=config.raps_lambda,
        raps_k_reg=config.raps_k_reg,
        saps_lambda=config.saps_lambda,
    )

    repair_thresholds = compute_baseline_group_thresholds(
        repair_scores, repair, data.group_values, config.alpha
    )
    repair_baseline = evaluate_split(
        repair_scores, repair, repair_thresholds.thresholds
    )

    roles = define_set_size_roles(repair, repair_baseline, data, config)
    metric_context = build_set_size_metric_context(
        repair_scores, repair, repair_baseline, data, roles
    )
    side_grids = build_set_size_directional_grids(
        repair_scores,
        repair,
        data,
        roles,
        repair_thresholds.thresholds,
        config,
    )
    protocol.record("repair_ready", detail="repair-derived quantities only")
    protocol.record("search_started", detail="repair split only")

    selected_vector, trace_rows, search_meta, zero_trial = (
        directional_set_size_search(
            base_scores=repair_scores,
            split=repair,
            data=data,
            config=config,
            roles=roles,
            side_grids=side_grids,
            target_counts=repair_thresholds.covered_counts,
            baseline_eval=repair_baseline,
            metric_context=metric_context,
            score_method=score_method,
        )
    )
    protocol.record("theta_selected", theta=selected_vector)
    selected_vector = np.asarray(selected_vector, dtype=float).copy()
    selected_vector.setflags(write=False)
    protocol.record("theta_frozen", theta=selected_vector)
    theta_plus, theta_minus = split_directional_vector(
        selected_vector, len(data.cell_values)
    )
    repair_selected = evaluate_set_size_with_fixed_counts(
        repair_scores,
        repair,
        data,
        theta_plus,
        theta_minus,
        roles,
        repair_thresholds.covered_counts,
        repair_baseline,
        config,
        metric_context,
        enforce_feasibility=True,
        retain_arrays=True,
    )
    if not repair_selected["feasible"]:
        raise AssertionError("Selected repair candidate is infeasible")

    protocol.record("calibration_started", theta=selected_vector)
    calibration = data.splits["calibration"]
    calibration_scores = labelwise_scores_from_probs(
        calibration.probs,
        score_method,
        raps_lambda=config.raps_lambda,
        raps_k_reg=config.raps_k_reg,
        saps_lambda=config.saps_lambda,
    )
    protocol.record("calibration_scores_constructed", theta=selected_vector)
    calibration_thresholds = compute_baseline_group_thresholds(
        calibration_scores,
        calibration,
        data.group_values,
        config.alpha,
    )
    calibration_baseline = evaluate_split(
        calibration_scores,
        calibration,
        calibration_thresholds.thresholds,
    )
    protocol.record("calibration_baseline_evaluated", theta=selected_vector)
    calibration_selected = evaluate_set_size_with_fixed_counts(
        calibration_scores,
        calibration,
        data,
        theta_plus,
        theta_minus,
        roles,
        calibration_thresholds.covered_counts,
        calibration_baseline,
        config,
        metric_context=None,
        enforce_feasibility=False,
        retain_arrays=True,
    )
    if not calibration_selected["fixed_counts"]:
        raise AssertionError("Calibration group covered counts changed")
    frozen_calibration_thresholds = dict(
        calibration_selected["thresholds"].thresholds
    )
    protocol.record(
        "calibration_thresholds_frozen",
        theta=selected_vector,
        thresholds=frozen_calibration_thresholds,
    )

    protocol.record(
        "test_started",
        theta=selected_vector,
        thresholds=frozen_calibration_thresholds,
    )
    test = data.splits["test"]
    test_scores = labelwise_scores_from_probs(
        test.probs,
        score_method,
        raps_lambda=config.raps_lambda,
        raps_k_reg=config.raps_k_reg,
        saps_lambda=config.saps_lambda,
    )
    protocol.record(
        "test_scores_constructed",
        theta=selected_vector,
        thresholds=frozen_calibration_thresholds,
    )
    test_baseline = evaluate_split(
        test_scores, test, calibration_thresholds.thresholds
    )
    protocol.record(
        "baseline_test_evaluated",
        theta=selected_vector,
        thresholds=frozen_calibration_thresholds,
    )
    test_selected = evaluate_set_size_split(
        test_scores,
        test,
        data,
        frozen_calibration_thresholds,
        theta_plus,
        theta_minus,
        roles,
    )
    protocol.record(
        "repaired_test_evaluated",
        theta=selected_vector,
        thresholds=frozen_calibration_thresholds,
    )
    protocol.assert_complete()

    evaluations = {
        "repair": (repair, repair_baseline, repair_selected["eval"]),
        "calibration": (
            calibration,
            calibration_baseline,
            calibration_selected["eval"],
        ),
        "test": (test, test_baseline, test_selected),
    }
    cell_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for split_name, (split, baseline_eval, selected_eval) in evaluations.items():
        cell_rows.extend(
            set_size_cell_rows(
                data,
                config,
                score_method,
                split_name,
                split,
                baseline_eval,
                selected_eval,
                roles,
                theta_plus,
                theta_minus,
            )
        )
        group_rows.extend(
            set_size_group_rows(
                data,
                score_method,
                split_name,
                split,
                baseline_eval,
                selected_eval,
            )
        )

    summary = build_set_size_summary(
        data,
        config,
        score_method,
        evaluations,
        roles,
        theta_plus,
        theta_minus,
        repair_selected,
        calibration_selected,
        search_meta,
    )
    validation_rows = runtime_validation_rows(
        data,
        config,
        score_method,
        repair_scores,
        roles,
        metric_context,
        repair_thresholds,
        repair_baseline,
        zero_trial,
        repair_selected,
        calibration_selected,
        selected_vector,
        trace_rows,
        protocol,
    )
    if not all(bool(row["passed"]) for row in validation_rows):
        failed = [row["check"] for row in validation_rows if not row["passed"]]
        raise AssertionError(f"Runtime validation failed: {failed}")

    offset_rows = set_size_offset_rows(
        data, score_method, roles, theta_plus, theta_minus, search_meta
    )
    test_rows = set_size_test_rows(
        data,
        score_method,
        test,
        test_baseline,
        test_selected,
        roles,
    )
    artifacts = DirectionalSetSizeArtifacts(
        summary_rows=[summary],
        offset_rows=offset_rows,
        cell_rows=cell_rows,
        group_rows=group_rows,
        search_rows=trace_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        protocol_rows=protocol.events,
    )
    write_set_size_outputs(
        result_dir, data, config, score_method, artifacts, command
    )
    return artifacts


def define_set_size_roles(
    split: SplitData,
    baseline_eval: dict[str, Any],
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
) -> dict[int, dict[str, Any]]:
    roles: dict[int, dict[str, Any]] = {}
    for cell_idx, cell_value in enumerate(data.cell_values):
        counts: dict[str, int] = {}
        means: dict[str, float] = {}
        for group in data.group_values:
            mask = (split.g == group) & (split.z == cell_value)
            counts[group] = int(mask.sum())
            means[group] = (
                float(np.mean(baseline_eval["set_sizes"][mask]))
                if counts[group]
                else float("nan")
            )
        supported = all(
            counts[group] >= config.min_cell_count
            for group in data.group_values
        )
        gap = (
            abs(means[data.group_values[1]] - means[data.group_values[0]])
            if supported
            else float("nan")
        )
        neutral = bool(supported and gap <= config.set_size_tolerance)
        if supported and not neutral:
            larger = max(
                data.group_values,
                key=lambda group: (means[group], -data.group_values.index(group)),
            )
            smaller = next(group for group in data.group_values if group != larger)
        else:
            larger = None
            smaller = None
        roles[cell_idx] = {
            "cell_idx": cell_idx,
            "cell_value": cell_value,
            "supported": supported,
            "baseline_order_tie": neutral,
            "pressure_active": bool(supported and not neutral),
            "larger_set_group": larger,
            "smaller_set_group": smaller,
            "plus_group": larger,
            "minus_group": smaller,
            "baseline_gap": float(gap),
            "baseline_mean_set_size": means,
            "counts": counts,
        }
    return roles


def build_set_size_metric_context(
    base_scores: np.ndarray,
    split: SplitData,
    baseline_eval: dict[str, Any],
    data: LoadedDataset,
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    row_cells = repair_cell_indices(data, split)
    cell_indices: dict[tuple[str, int], np.ndarray] = {}
    baseline_size: dict[tuple[str, int], float] = {}
    for group in data.group_values:
        for cell_idx in range(len(data.cell_values)):
            idx = np.flatnonzero((split.g == group) & (row_cells == cell_idx))
            cell_indices[(group, cell_idx)] = idx
            baseline_size[(group, cell_idx)] = (
                float(np.mean(baseline_eval["set_sizes"][idx]))
                if len(idx)
                else float("nan")
            )

    if data.intervention_mode == "label_defined_z":
        signs = np.zeros((len(split.y), len(data.label_values)), dtype=float)
        for cell_idx, role in roles.items():
            if not role["pressure_active"]:
                continue
            signs[split.g == role["plus_group"], cell_idx] = 1.0
            signs[split.g == role["minus_group"], cell_idx] = -1.0
    elif data.intervention_mode == "covariate_defined_z":
        signs = np.zeros(len(split.y), dtype=float)
        for cell_idx, role in roles.items():
            if not role["pressure_active"]:
                continue
            cell_mask = row_cells == cell_idx
            signs[cell_mask & (split.g == role["plus_group"])] = 1.0
            signs[cell_mask & (split.g == role["minus_group"])] = -1.0
    else:
        raise ValueError(f"Unsupported intervention mode: {data.intervention_mode}")
    return {
        "groups": list(data.group_values),
        "cell_indices": cell_indices,
        "baseline_size": baseline_size,
        "supported_cells": [
            idx for idx, role in roles.items() if role["supported"]
        ],
        "active_cells": [
            idx for idx, role in roles.items() if role["pressure_active"]
        ],
        "baseline_avg_set_size": float(np.mean(baseline_eval["set_sizes"])),
        "candidate_shift_signs": signs,
        "row_cell_indices": row_cells,
        "base_true_scores": true_scores(base_scores, split.y),
    }


def apply_set_size_score_matrix(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    roles: dict[int, dict[str, Any]],
) -> np.ndarray:
    """Apply prediction-time shifts; label-defined mode never reads realized Y."""
    plus = np.asarray(theta_plus, dtype=float)
    minus = np.asarray(theta_minus, dtype=float)
    if np.any(plus < 0.0) or np.any(minus < 0.0):
        raise ValueError("Directional magnitudes must be nonnegative")
    shifted = np.asarray(base_scores, dtype=float).copy()
    if data.intervention_mode == "label_defined_z":
        for cell_idx, role in roles.items():
            if not role["pressure_active"]:
                continue
            shifted[split.g == role["plus_group"], cell_idx] += plus[cell_idx]
            shifted[split.g == role["minus_group"], cell_idx] -= minus[cell_idx]
        return shifted
    if data.intervention_mode == "covariate_defined_z":
        row_cells = repair_cell_indices(data, split)
        row_shift = np.zeros(len(split.g), dtype=float)
        for cell_idx, role in roles.items():
            if not role["pressure_active"]:
                continue
            cell_mask = row_cells == cell_idx
            row_shift[cell_mask & (split.g == role["plus_group"])] += plus[
                cell_idx
            ]
            row_shift[cell_mask & (split.g == role["minus_group"])] -= minus[
                cell_idx
            ]
        shifted += row_shift[:, None]
        return shifted
    raise ValueError(f"Unsupported intervention mode: {data.intervention_mode}")


def apply_set_size_score_matrix_fast(
    base_scores: np.ndarray,
    intervention_mode: str,
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    context: dict[str, Any],
) -> np.ndarray:
    signs = np.asarray(context["candidate_shift_signs"], dtype=float)
    plus = np.asarray(theta_plus, dtype=float)
    minus = np.asarray(theta_minus, dtype=float)
    if intervention_mode == "label_defined_z":
        return (
            np.asarray(base_scores, dtype=float)
            + (signs > 0.0) * plus[None, :]
            - (signs < 0.0) * minus[None, :]
        )
    if intervention_mode == "covariate_defined_z":
        cells = np.asarray(context["row_cell_indices"], dtype=int)
        row_plus = plus[cells]
        row_minus = minus[cells]
        row_shift = np.where(
            signs > 0.0,
            row_plus,
            np.where(signs < 0.0, -row_minus, 0.0),
        )
        return np.asarray(base_scores, dtype=float) + row_shift[:, None]
    raise ValueError(f"Unsupported intervention mode: {intervention_mode}")


def evaluate_shifted_score_matrix(
    shifted_scores: np.ndarray,
    split: SplitData,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    threshold_by_row = row_thresholds(split, thresholds)
    prediction_sets = shifted_scores <= threshold_by_row[:, None]
    covered = prediction_sets[np.arange(len(split.y)), split.y]
    return {
        "covered": covered.astype(bool),
        "set_sizes": prediction_sets.sum(axis=1).astype(float),
        "thresholds": dict(thresholds),
    }


def evaluate_set_size_split(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    thresholds: dict[str, float],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    shifted = apply_set_size_score_matrix(
        base_scores, split, data, theta_plus, theta_minus, roles
    )
    return evaluate_shifted_score_matrix(shifted, split, thresholds)


def evaluate_set_size_split_direct(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    thresholds: dict[str, float],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Independent row/label reconstruction used only for validation."""
    shifted = apply_set_size_score_matrix(
        base_scores, split, data, theta_plus, theta_minus, roles
    )
    covered = np.zeros(len(split.y), dtype=bool)
    sizes = np.zeros(len(split.y), dtype=float)
    for row in range(len(split.y)):
        threshold = float(thresholds[str(split.g[row])])
        included = np.asarray(shifted[row], dtype=float) <= threshold
        sizes[row] = float(np.sum(included))
        covered[row] = bool(included[int(split.y[row])])
    return {"covered": covered, "set_sizes": sizes, "thresholds": thresholds}


def set_size_candidate_metrics(
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    vector: np.ndarray,
    context: dict[str, Any],
    config: DirectionalSetSizeConfig,
) -> dict[str, Any]:
    repaired_size: dict[tuple[str, int], float] = {}
    current_gap_by_cell: dict[int, float] = {}
    gap_reductions: list[float] = []
    gap_increases: list[float] = []
    larger_increases: list[float] = []
    smaller_increases: list[float] = []
    neutral_increases: list[float] = []
    for cell_idx in context["supported_cells"]:
        means: dict[str, float] = {}
        for group in context["groups"]:
            idx = context["cell_indices"][(group, cell_idx)]
            means[group] = float(np.mean(repaired_eval["set_sizes"][idx]))
            repaired_size[(group, cell_idx)] = means[group]
        gap = abs(means[context["groups"][1]] - means[context["groups"][0]])
        current_gap_by_cell[cell_idx] = float(gap)
        baseline_gap = float(roles[cell_idx]["baseline_gap"])
        gap_reductions.append(baseline_gap - gap)
        gap_increases.append(gap - baseline_gap)
        role = roles[cell_idx]
        if role["baseline_order_tie"]:
            for group in context["groups"]:
                neutral_increases.append(
                    means[group] - context["baseline_size"][(group, cell_idx)]
                )
        else:
            larger = role["larger_set_group"]
            smaller = role["smaller_set_group"]
            larger_increases.append(
                means[larger] - context["baseline_size"][(larger, cell_idx)]
            )
            smaller_increases.append(
                means[smaller] - context["baseline_size"][(smaller, cell_idx)]
            )
    gaps = list(current_gap_by_cell.values())
    return {
        "max_set_size_gap": float(max(gaps, default=0.0)),
        "mean_set_size_gap": float(np.mean(gaps)) if gaps else 0.0,
        "total_set_size_gap_reduction": float(np.sum(gap_reductions)),
        "max_set_size_gap_increase": float(max(gap_increases, default=0.0)),
        "max_larger_side_increase": float(max(larger_increases, default=0.0)),
        "max_smaller_side_increase": float(max(smaller_increases, default=0.0)),
        "max_neutral_group_increase": float(max(neutral_increases, default=0.0)),
        "current_gap_by_cell": current_gap_by_cell,
        "repaired_size_by_cell": repaired_size,
        "checked_supported_cell_count": len(current_gap_by_cell),
        "shift_norm": float(np.linalg.norm(vector, ord=2)),
        "overall_mean_set_size_change": float(
            np.mean(repaired_eval["set_sizes"])
            - context["baseline_avg_set_size"]
        ),
    }


def set_size_rejection_reasons(
    group_coverage_fixed: bool,
    metrics: dict[str, Any],
    config: DirectionalSetSizeConfig,
    roles: dict[int, dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if not group_coverage_fixed:
        reasons.append("group_coverage_not_fixed")
    if metrics["max_set_size_gap_increase"] > config.set_size_tolerance:
        reasons.append("set_size_gap_increased")
    if metrics["max_larger_side_increase"] > config.set_size_tolerance:
        reasons.append("larger_side_increased")
    if (
        metrics["max_smaller_side_increase"]
        > config.smaller_side_budget + config.set_size_tolerance
    ):
        reasons.append("smaller_side_budget_exceeded")
    neutral_gap_increase = max(
        [
            float(metrics["current_gap_by_cell"][cell_idx])
            - float(role["baseline_gap"])
            for cell_idx, role in roles.items()
            if role["supported"] and role["baseline_order_tie"]
        ],
        default=0.0,
    )
    if (
        metrics["max_neutral_group_increase"]
        > config.smaller_side_budget + config.set_size_tolerance
        or neutral_gap_increase > config.set_size_tolerance
    ):
        reasons.append("neutral_cell_violation")
    return reasons


def set_size_objective(
    metrics: dict[str, Any],
    zero_metrics: dict[str, Any] | None,
    config: DirectionalSetSizeConfig,
) -> tuple[float, float, float, float]:
    values = [
        float(metrics["max_set_size_gap"]),
        float(metrics["mean_set_size_gap"]),
        -float(metrics["total_set_size_gap_reduction"]),
    ]
    if zero_metrics is not None:
        zero_values = [
            float(zero_metrics["max_set_size_gap"]),
            float(zero_metrics["mean_set_size_gap"]),
            -float(zero_metrics["total_set_size_gap_reduction"]),
        ]
        if all(
            abs(value - zero) <= config.no_gap_improvement_tol
            for value, zero in zip(values, zero_values)
        ):
            values = zero_values
    return (*values, float(metrics["shift_norm"]))


def evaluate_set_size_with_fixed_counts(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    roles: dict[int, dict[str, Any]],
    target_counts: dict[str, int],
    baseline_eval: dict[str, Any],
    config: DirectionalSetSizeConfig,
    metric_context: dict[str, Any] | None,
    enforce_feasibility: bool,
    retain_arrays: bool,
) -> dict[str, Any]:
    if metric_context is not None:
        shifted_scores = apply_set_size_score_matrix_fast(
            base_scores,
            data.intervention_mode,
            theta_plus,
            theta_minus,
            metric_context,
        )
    else:
        shifted_scores = apply_set_size_score_matrix(
            base_scores, split, data, theta_plus, theta_minus, roles
        )
    shifted_true = shifted_scores[
        np.arange(len(split.y)), np.asarray(split.y, dtype=int)
    ]
    thresholds = compute_thresholds_for_target_counts_from_true_scores(
        shifted_true, split, data.group_values, target_counts
    )
    repaired_eval = evaluate_shifted_score_matrix(
        shifted_scores, split, thresholds.thresholds
    )
    exact_counts = {
        group: int(np.sum(repaired_eval["covered"][split.g == group]))
        for group in data.group_values
    }
    fixed_counts = bool(
        thresholds.exact
        and all(
            exact_counts[group] == int(target_counts[group])
            for group in data.group_values
        )
    )
    context = metric_context or build_set_size_metric_context(
        base_scores, split, baseline_eval, data, roles
    )
    vector = np.concatenate([theta_plus, theta_minus])
    metrics = set_size_candidate_metrics(
        repaired_eval, roles, vector, context, config
    )
    reasons = (
        set_size_rejection_reasons(fixed_counts, metrics, config, roles)
        if enforce_feasibility
        else ([] if fixed_counts else ["group_coverage_not_fixed"])
    )
    result_eval = repaired_eval if retain_arrays else {
        "thresholds": repaired_eval["thresholds"]
    }
    return {
        "eval": result_eval,
        "thresholds": thresholds,
        "metrics": metrics,
        "feasible": not reasons,
        "fixed_counts": fixed_counts,
        "exact_counts": exact_counts,
        "rejection_reasons": reasons,
    }


def membership_event_values(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    roles: dict[int, dict[str, Any]],
    cell_idx: int,
    side: str,
    thresholds: dict[str, float],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    config: DirectionalSetSizeConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    role = roles[cell_idx]
    if not role["pressure_active"]:
        return np.asarray([], dtype=float), {
            "row_count": 0,
            "candidate_label_count": 0,
            "realized_y_values": [],
        }
    shifted = apply_set_size_score_matrix(
        base_scores, split, data, theta_plus, theta_minus, roles
    )
    if side == "plus":
        group = role["plus_group"]
    elif side == "minus":
        group = role["minus_group"]
    else:
        raise ValueError(f"Unknown side: {side}")
    if data.intervention_mode == "covariate_defined_z":
        row_mask = (split.g == group) & (split.z == data.cell_values[cell_idx])
        affected = shifted[row_mask, :]
        threshold = float(thresholds[group])
        margins = threshold - affected if side == "plus" else affected - threshold
        labels = np.tile(np.arange(shifted.shape[1]), (int(row_mask.sum()), 1))
    else:
        row_mask = split.g == group
        affected = shifted[row_mask, cell_idx]
        threshold = float(thresholds[group])
        margins = threshold - affected if side == "plus" else affected - threshold
        labels = np.full(affected.shape, cell_idx, dtype=int)
    positive_mask = np.isfinite(margins) & (margins > config.numeric_tolerance)
    positive = np.asarray(margins[positive_mask], dtype=float)
    contributing_labels = np.unique(np.asarray(labels)[positive_mask]).tolist()
    return positive, {
        "row_count": int(row_mask.sum()),
        "candidate_label_count": len(contributing_labels),
        "candidate_labels": contributing_labels,
        "realized_y_values": sorted(
            {int(value) for value in np.asarray(split.y)[row_mask].tolist()}
        ),
        "event_count": int(len(positive)),
    }


def build_set_size_directional_grids(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    roles: dict[int, dict[str, Any]],
    baseline_thresholds: dict[str, float],
    config: DirectionalSetSizeConfig,
) -> dict[str, dict[int, list[float]]]:
    zeros = np.zeros(len(data.cell_values), dtype=float)
    grids = {"plus": {}, "minus": {}}
    for cell_idx, role in roles.items():
        if not role["pressure_active"]:
            grids["plus"][cell_idx] = [0.0]
            grids["minus"][cell_idx] = [0.0]
            continue
        for side in ("plus", "minus"):
            events, _ = membership_event_values(
                base_scores,
                split,
                data,
                roles,
                cell_idx,
                side,
                baseline_thresholds,
                zeros,
                zeros,
                config,
            )
            scale = (
                float(np.quantile(events, 0.5)) if len(events) else 0.05
            )
            if not np.isfinite(scale) or scale <= config.numeric_tolerance:
                scale = 0.05
            grids[side][cell_idx] = sorted(
                {
                    round(max(0.0, float(multiplier) * scale), 12)
                    for multiplier in config.grid_multipliers
                }
            )
    return grids


def set_size_side_critical_values(
    side: str,
    cell_idx: int,
    current_vector: np.ndarray,
    grid: list[float],
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    current_thresholds: dict[str, float],
    roles: dict[int, dict[str, Any]],
    config: DirectionalSetSizeConfig,
) -> list[float]:
    if not roles[cell_idx]["pressure_active"]:
        return [0.0]
    n_cells = len(data.cell_values)
    plus, minus = split_directional_vector(current_vector, n_cells)
    current = float(plus[cell_idx] if side == "plus" else minus[cell_idx])
    scale = theta_scale_from_grid(grid, config)
    values = {0.0, round(current, 12)}
    values.update(round(float(value), 12) for value in grid)
    for multiplier in (-0.2, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.2):
        values.add(round(max(0.0, current + multiplier * scale), 12))
    for multiplier in (1e-4, 1e-3, 0.005, 0.01, 0.025, 0.05):
        values.add(round(multiplier * scale, 12))
    events, _ = membership_event_values(
        base_scores,
        split,
        data,
        roles,
        cell_idx,
        side,
        current_thresholds,
        plus,
        minus,
        config,
    )
    if len(events):
        for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
            values.add(round(current + float(np.quantile(events, quantile)), 12))
    return sorted(
        value for value in values if np.isfinite(value) and value >= 0.0
    )


def directional_set_size_search(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    roles: dict[int, dict[str, Any]],
    side_grids: dict[str, dict[int, list[float]]],
    target_counts: dict[str, int],
    baseline_eval: dict[str, Any],
    metric_context: dict[str, Any],
    score_method: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Search D_repair only using the unchanged directional phase budgets."""
    n_cells = len(data.cell_values)
    trace: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    zero_metrics: dict[str, Any] | None = None

    def evaluate_candidate(
        vector: np.ndarray,
        phase: str,
        pass_idx: int = 0,
        updated_cells: tuple[int, ...] = (),
        beam_rank: int | str = "",
    ) -> dict[str, Any]:
        trial_vector = np.maximum(np.asarray(vector, dtype=float), 0.0)
        key = directional_key(trial_vector, config)
        trial = cache.get(key)
        if trial is None:
            plus, minus = split_directional_vector(trial_vector, n_cells)
            trial = evaluate_set_size_with_fixed_counts(
                base_scores,
                split,
                data,
                plus,
                minus,
                roles,
                target_counts,
                baseline_eval,
                config,
                metric_context,
                enforce_feasibility=True,
                retain_arrays=False,
            )
            cache[key] = trial
        row = set_size_trace_row(
            data,
            config,
            score_method,
            phase,
            pass_idx,
            updated_cells,
            beam_rank,
            trial_vector,
            trial,
        )
        trace.append(row)
        if trial["feasible"]:
            feasible.append(
                {
                    "vector": trial_vector.copy(),
                    "eval": trial,
                    "obj": set_size_objective(trial["metrics"], zero_metrics, config),
                    "phase": phase,
                    "trace_idx": len(trace) - 1,
                    "key": key,
                }
            )
        return trial

    zero = np.zeros(2 * n_cells, dtype=float)
    zero_trial = evaluate_candidate(zero, "zero_baseline")
    if not zero_trial["feasible"]:
        raise RuntimeError(
            f"Zero intervention must be feasible: {zero_trial['rejection_reasons']}"
        )
    zero_metrics = zero_trial["metrics"]
    for candidate in feasible:
        candidate["obj"] = set_size_objective(
            candidate["eval"]["metrics"], zero_metrics, config
        )

    global_direction = set_size_global_direction(side_grids, roles, config)
    global_infeasible: list[dict[str, Any]] = []
    for value in config.global_lambda_grid:
        vector = float(value) * global_direction
        trial = evaluate_candidate(vector, "global_line")
        if not trial["feasible"]:
            global_infeasible.append({"vector": vector, "eval": trial})
    project_set_size_sources(
        global_infeasible,
        "boundary_projection",
        feasible,
        trace,
        config,
        zero_metrics,
        evaluate_candidate,
    )

    current_best = best_candidate(feasible, config)
    order = prioritized_cells(current_best["eval"]["metrics"], roles)
    for cell_idx in order:
        plus_values = set_size_side_critical_values(
            "plus",
            cell_idx,
            current_best["vector"],
            side_grids["plus"][cell_idx],
            base_scores,
            split,
            data,
            current_best["eval"]["thresholds"].thresholds,
            roles,
            config,
        )
        minus_values = set_size_side_critical_values(
            "minus",
            cell_idx,
            current_best["vector"],
            side_grids["minus"][cell_idx],
            base_scores,
            split,
            data,
            current_best["eval"]["thresholds"].thresholds,
            roles,
            config,
        )
        for vector in critical_cell_movements(
            current_best["vector"],
            cell_idx,
            plus_values,
            minus_values,
            side_grids,
            config,
        ):
            evaluate_candidate(vector, "critical_grid", updated_cells=(cell_idx,))

    coordinate = np.zeros(2 * n_cells, dtype=float)
    coordinate_obj = set_size_objective(zero_metrics, zero_metrics, config)
    for pass_idx in range(1, config.optimizer_passes + 1):
        for cell_idx in range(n_cells):
            best_vector = coordinate.copy()
            best_obj = coordinate_obj
            best_trace: int | None = None
            for vector in coordinate_cell_movements(
                coordinate, cell_idx, side_grids, config
            ):
                trace_idx = len(trace)
                trial = evaluate_candidate(
                    vector,
                    "coordinate_refinement",
                    pass_idx,
                    (cell_idx,),
                    "coordinate",
                )
                if trial["feasible"]:
                    obj = set_size_objective(trial["metrics"], zero_metrics, config)
                    if compare_objective_and_key(
                        obj,
                        directional_key(vector, config),
                        best_obj,
                        directional_key(best_vector, config),
                        config,
                    ) < 0:
                        best_vector = vector.copy()
                        best_obj = obj
                        best_trace = trace_idx
            if best_trace is not None:
                trace[best_trace]["selected_for_update"] = True
                coordinate = best_vector
                coordinate_obj = best_obj

    beam = top_beam(feasible, config)
    for pass_idx in range(1, config.optimizer_passes + 1):
        old_best_key = best_candidate(feasible, config)["key"]
        order = prioritized_cells(beam[0]["eval"]["metrics"], roles)
        pairs = prioritized_pairs(order, roles, config)
        triples = prioritized_triples(order, roles, config)
        for cell_idx in order:
            block: list[dict[str, Any]] = []
            for rank, state in enumerate(beam, start=1):
                for vector in local_cell_movements(
                    state["vector"], cell_idx, side_grids, config
                ):
                    trial = evaluate_candidate(
                        vector,
                        "single_block",
                        pass_idx,
                        (cell_idx,),
                        rank,
                    )
                    if trial["feasible"]:
                        block.append(feasible[-1])
            beam = top_beam(beam + block, config)
            mark_beam_best(trace, beam)
        for left, right in pairs:
            block = []
            for rank, state in enumerate(beam, start=1):
                left_options = compact_cell_block_options(
                    state["vector"], left, side_grids, config, "pair"
                )
                right_options = compact_cell_block_options(
                    state["vector"], right, side_grids, config, "pair"
                )
                for left_values in left_options:
                    for right_values in right_options:
                        vector = replace_cell_block(
                            state["vector"], left, *left_values
                        )
                        vector = replace_cell_block(vector, right, *right_values)
                        trial = evaluate_candidate(
                            vector,
                            "pair_block",
                            pass_idx,
                            (left, right),
                            rank,
                        )
                        if trial["feasible"]:
                            block.append(feasible[-1])
            beam = top_beam(beam + block, config)
            mark_beam_best(trace, beam)
        for first, second, third in triples:
            block = []
            for rank, state in enumerate(beam, start=1):
                options = [
                    compact_cell_block_options(
                        state["vector"], idx, side_grids, config, "triple"
                    )
                    for idx in (first, second, third)
                ]
                for values_1 in options[0]:
                    for values_2 in options[1]:
                        for values_3 in options[2]:
                            vector = replace_cell_block(
                                state["vector"], first, *values_1
                            )
                            vector = replace_cell_block(vector, second, *values_2)
                            vector = replace_cell_block(vector, third, *values_3)
                            trial = evaluate_candidate(
                                vector,
                                "triple_block",
                                pass_idx,
                                (first, second, third),
                                rank,
                            )
                            if trial["feasible"]:
                                block.append(feasible[-1])
            beam = top_beam(beam + block, config)
            mark_beam_best(trace, beam)
        beam = top_beam(feasible, config)
        for rank, state in enumerate(beam, start=1):
            evaluate_candidate(state["vector"], "beam", pass_idx, beam_rank=rank)
        if best_candidate(feasible, config)["key"] == old_best_key:
            break

    quasi_infeasible: list[dict[str, Any]] = []
    for candidate_idx, vector in enumerate(
        set_size_halton_candidates(side_grids, roles, config, score_method),
        start=1,
    ):
        trial = evaluate_candidate(vector, "quasirandom", candidate_idx)
        if not trial["feasible"]:
            quasi_infeasible.append({"vector": vector, "eval": trial})
    project_set_size_sources(
        quasi_infeasible,
        "quasirandom_projection",
        feasible,
        trace,
        config,
        zero_metrics,
        evaluate_candidate,
    )

    final = best_candidate(feasible, config)
    trace[final["trace_idx"]]["selected_final"] = True
    trace[final["trace_idx"]]["selected_for_update"] = True
    meta = set_size_search_meta(trace, final)
    return final["vector"], trace, meta, zero_trial


def set_size_global_direction(
    side_grids: dict[str, dict[int, list[float]]],
    roles: dict[int, dict[str, Any]],
    config: DirectionalSetSizeConfig,
) -> np.ndarray:
    n_cells = len(roles)
    direction = np.zeros(2 * n_cells, dtype=float)
    active = [idx for idx, role in roles.items() if role["pressure_active"]]
    max_gap = max(
        [float(roles[idx]["baseline_gap"]) for idx in active], default=0.0
    )
    for cell_idx in active:
        pressure = (
            max(0.05, float(roles[cell_idx]["baseline_gap"]) / max_gap)
            if max_gap > config.numeric_tolerance
            else 1.0
        )
        direction[cell_idx] = (
            theta_scale_from_grid(side_grids["plus"][cell_idx], config)
            * pressure
        )
        direction[n_cells + cell_idx] = (
            theta_scale_from_grid(side_grids["minus"][cell_idx], config)
            * pressure
        )
    return direction


def project_set_size_sources(
    sources: list[dict[str, Any]],
    phase: str,
    feasible: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    config: DirectionalSetSizeConfig,
    zero_metrics: dict[str, Any],
    evaluate_candidate: Callable[..., dict[str, Any]],
) -> None:
    best = best_candidate(feasible, config)
    promising = []
    for source in sources:
        reasons = set(source["eval"]["rejection_reasons"])
        if not reasons or "group_coverage_not_fixed" in reasons:
            continue
        metrics = source["eval"]["metrics"]
        if (
            metrics["total_set_size_gap_reduction"]
            >= best["eval"]["metrics"]["total_set_size_gap_reduction"]
            - config.numeric_tolerance
            or metrics["max_set_size_gap"]
            <= best["eval"]["metrics"]["max_set_size_gap"] + 0.02
        ):
            promising.append(source)
    promising.sort(
        key=cmp_to_key(
            lambda left, right: compare_objective_and_key(
                set_size_objective(left["eval"]["metrics"], zero_metrics, config),
                directional_key(left["vector"], config),
                set_size_objective(right["eval"]["metrics"], zero_metrics, config),
                directional_key(right["vector"], config),
                config,
            )
        )
    )
    for source_rank, source in enumerate(
        promising[: config.boundary_projection_max_candidates], start=1
    ):
        good = best_candidate(feasible, config)["vector"]
        bad = source["vector"]
        strongest_idx: int | None = None
        strongest_value = -1.0
        for value in np.linspace(0.0, 1.0, config.boundary_projection_grid_size):
            vector = (1.0 - float(value)) * good + float(value) * bad
            trace_idx = len(trace)
            trial = evaluate_candidate(vector, phase, source_rank)
            if trial["feasible"] and float(value) > strongest_value:
                strongest_value = float(value)
                strongest_idx = trace_idx
        if strongest_idx is not None:
            trace[strongest_idx]["selected_for_update"] = True


def prioritized_cells(
    metrics: dict[str, Any], roles: dict[int, dict[str, Any]]
) -> list[int]:
    active = [idx for idx, role in roles.items() if role["pressure_active"]]
    active.sort(
        key=lambda idx: (
            -float(metrics["current_gap_by_cell"].get(idx, 0.0)),
            -float(roles[idx]["baseline_gap"]),
            idx,
        )
    )
    return active


def prioritized_pairs(
    order: list[int],
    roles: dict[int, dict[str, Any]],
    config: DirectionalSetSizeConfig,
) -> list[tuple[int, int]]:
    top = order[: min(len(order), 8)]
    pairs = [(left, right) for pos, left in enumerate(top) for right in top[pos + 1 :]]
    pairs.sort(
        key=lambda pair: (
            -max(roles[pair[0]]["baseline_gap"], roles[pair[1]]["baseline_gap"]),
            -roles[pair[0]]["baseline_gap"] - roles[pair[1]]["baseline_gap"],
            pair,
        )
    )
    return pairs[: config.pair_block_limit]


def prioritized_triples(
    order: list[int],
    roles: dict[int, dict[str, Any]],
    config: DirectionalSetSizeConfig,
) -> list[tuple[int, int, int]]:
    top = order[: min(len(order), 6)]
    triples: list[tuple[int, int, int]] = []
    for left_pos, left in enumerate(top):
        for middle_pos in range(left_pos + 1, len(top)):
            middle = top[middle_pos]
            for right in top[middle_pos + 1 :]:
                triples.append((left, middle, right))
    triples.sort(
        key=lambda triple: (
            -max(roles[idx]["baseline_gap"] for idx in triple),
            -sum(roles[idx]["baseline_gap"] for idx in triple),
            triple,
        )
    )
    return triples[: config.triple_block_limit]


def critical_cell_movements(
    current_vector: np.ndarray,
    cell_idx: int,
    plus_values: list[float],
    minus_values: list[float],
    side_grids: dict[str, dict[int, list[float]]],
    config: DirectionalSetSizeConfig,
) -> list[np.ndarray]:
    n_cells = len(current_vector) // 2
    current_plus = float(current_vector[cell_idx])
    current_minus = float(current_vector[n_cells + cell_idx])
    plus_near = nearest_values(plus_values, current_plus, 5)
    minus_near = nearest_values(minus_values, current_minus, 5)
    vectors: list[np.ndarray] = []
    for value in nearest_values(plus_values, current_plus, 9):
        vectors.append(replace_cell_block(current_vector, cell_idx, value, current_minus))
    for value in nearest_values(minus_values, current_minus, 9):
        vectors.append(replace_cell_block(current_vector, cell_idx, current_plus, value))
    for plus_value, minus_value in zip(plus_near, reversed(minus_near)):
        vectors.append(replace_cell_block(current_vector, cell_idx, plus_value, minus_value))
    common_scale = min(
        theta_scale_from_grid(side_grids["plus"][cell_idx], config),
        theta_scale_from_grid(side_grids["minus"][cell_idx], config),
    )
    for multiplier in (-0.2, -0.1, 0.0, 0.1, 0.2):
        delta = multiplier * common_scale
        vectors.append(
            replace_cell_block(
                current_vector,
                cell_idx,
                max(0.0, current_plus + delta),
                max(0.0, current_minus + delta),
            )
        )
    return deduplicate_vectors(vectors, config)


def coordinate_cell_movements(
    vector: np.ndarray,
    cell_idx: int,
    side_grids: dict[str, dict[int, list[float]]],
    config: DirectionalSetSizeConfig,
) -> list[np.ndarray]:
    n_cells = len(vector) // 2
    current_plus = float(vector[cell_idx])
    current_minus = float(vector[n_cells + cell_idx])
    plus_grid = side_grids["plus"][cell_idx]
    minus_grid = side_grids["minus"][cell_idx]
    vectors = [
        replace_cell_block(vector, cell_idx, value, current_minus)
        for value in plus_grid
    ]
    vectors.extend(
        replace_cell_block(vector, cell_idx, current_plus, value)
        for value in minus_grid
    )
    vectors.extend(
        replace_cell_block(vector, cell_idx, plus_value, minus_value)
        for plus_value, minus_value in zip(plus_grid, reversed(minus_grid))
    )
    common_values = sorted(
        set(plus_grid).intersection(minus_grid)
        or {
            round((plus + minus) / 2.0, 12)
            for plus, minus in zip(plus_grid, minus_grid)
        }
    )
    vectors.extend(
        replace_cell_block(vector, cell_idx, value, value)
        for value in common_values
    )
    return deduplicate_vectors(vectors, config)


def local_cell_movements(
    vector: np.ndarray,
    cell_idx: int,
    side_grids: dict[str, dict[int, list[float]]],
    config: DirectionalSetSizeConfig,
) -> list[np.ndarray]:
    n_cells = len(vector) // 2
    plus = float(vector[cell_idx])
    minus = float(vector[n_cells + cell_idx])
    plus_scale = theta_scale_from_grid(side_grids["plus"][cell_idx], config)
    minus_scale = theta_scale_from_grid(side_grids["minus"][cell_idx], config)
    common_scale = min(plus_scale, minus_scale)
    blocks = [
        (plus, minus),
        (plus + 0.5 * plus_scale, minus),
        (plus, minus + 0.5 * minus_scale),
        (plus + 0.5 * plus_scale, minus + 0.25 * minus_scale),
        (plus + 0.5 * common_scale, minus + 0.5 * common_scale),
    ]
    return deduplicate_vectors(
        [
            replace_cell_block(vector, cell_idx, max(0.0, p), max(0.0, m))
            for p, m in blocks
        ],
        config,
    )


def compact_cell_block_options(
    vector: np.ndarray,
    cell_idx: int,
    side_grids: dict[str, dict[int, list[float]]],
    config: DirectionalSetSizeConfig,
    block_kind: str,
) -> list[tuple[float, float]]:
    n_cells = len(vector) // 2
    plus = float(vector[cell_idx])
    minus = float(vector[n_cells + cell_idx])
    plus_scale = theta_scale_from_grid(side_grids["plus"][cell_idx], config)
    minus_scale = theta_scale_from_grid(side_grids["minus"][cell_idx], config)
    step = 0.5 if block_kind == "pair" else 0.25
    return [
        (plus, minus),
        (plus + step * plus_scale, minus),
        (plus, minus + step * minus_scale),
    ]


def set_size_halton_candidates(
    side_grids: dict[str, dict[int, list[float]]],
    roles: dict[int, dict[str, Any]],
    config: DirectionalSetSizeConfig,
    score_method: str,
) -> list[np.ndarray]:
    active = [idx for idx, role in roles.items() if role["pressure_active"]]
    n_cells = len(roles)
    if not active:
        return []
    plus_scales = np.zeros(n_cells, dtype=float)
    minus_scales = np.zeros(n_cells, dtype=float)
    pressure = np.zeros(n_cells, dtype=float)
    max_gap = max(float(roles[idx]["baseline_gap"]) for idx in active)
    for idx in active:
        plus_scales[idx] = theta_scale_from_grid(side_grids["plus"][idx], config)
        minus_scales[idx] = theta_scale_from_grid(side_grids["minus"][idx], config)
        pressure[idx] = (
            max(0.05, float(roles[idx]["baseline_gap"]) / max_gap)
            if max_gap > config.numeric_tolerance
            else 1.0
        )
    ordered = sorted(active, key=lambda idx: (-roles[idx]["baseline_gap"], idx))
    method_offset = sum(
        (idx + 1) * ord(character)
        for idx, character in enumerate(score_method)
    )
    start = config.quasirandom_seed + method_offset
    primes = first_primes(2 * len(active))
    candidates: list[np.ndarray] = []
    for candidate_idx in range(config.quasirandom_candidate_count):
        uniforms = np.asarray(
            [
                halton_value(start + candidate_idx + 1, base)
                for base in primes
            ],
            dtype=float,
        )
        plus = np.zeros(n_cells, dtype=float)
        minus = np.zeros(n_cells, dtype=float)
        mode = candidate_idx % 4
        for position, cell_idx in enumerate(active):
            u_plus = uniforms[position]
            u_minus = uniforms[len(active) + position]
            if mode == 0:
                shared = 0.05 + 0.70 * (0.5 * u_plus + 0.5 * u_minus)
                plus[cell_idx] = plus_scales[cell_idx] * pressure[cell_idx] * shared
                minus[cell_idx] = minus_scales[cell_idx] * pressure[cell_idx] * shared
            elif mode == 1:
                plus[cell_idx] = plus_scales[cell_idx] * pressure[cell_idx] * (0.10 + 2.40 * u_plus)
                minus[cell_idx] = minus_scales[cell_idx] * pressure[cell_idx] * (0.10 + 2.40 * u_minus)
            elif mode == 2:
                active_count = 1 + candidate_idx % min(4, len(ordered))
                if cell_idx in ordered[:active_count]:
                    plus[cell_idx] = plus_scales[cell_idx] * pressure[cell_idx] * (0.05 + 2.95 * u_plus)
                    minus[cell_idx] = minus_scales[cell_idx] * pressure[cell_idx] * (0.05 + 2.95 * u_minus)
            else:
                if u_plus > 0.45 or cell_idx in ordered[:3]:
                    plus[cell_idx] = plus_scales[cell_idx] * (0.02 + 1.80 * u_plus) * (0.25 + 0.75 * pressure[cell_idx])
                if u_minus > 0.45 or cell_idx in ordered[:3]:
                    minus[cell_idx] = minus_scales[cell_idx] * (0.02 + 1.80 * u_minus) * (0.25 + 0.75 * pressure[cell_idx])
        candidates.append(np.round(np.concatenate([plus, minus]), 12))
    return candidates


def split_directional_vector(
    vector: np.ndarray, n_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (2 * n_cells,):
        raise ValueError(f"Expected vector length {2 * n_cells}; got {vector.shape}")
    return vector[:n_cells].copy(), vector[n_cells:].copy()


def replace_cell_block(
    vector: np.ndarray, cell_idx: int, theta_plus: float, theta_minus: float
) -> np.ndarray:
    output = np.asarray(vector, dtype=float).copy()
    n_cells = len(output) // 2
    output[cell_idx] = max(0.0, float(theta_plus))
    output[n_cells + cell_idx] = max(0.0, float(theta_minus))
    return np.round(output, 12)


def directional_key(
    vector: np.ndarray, config: DirectionalSetSizeConfig
) -> tuple[float, ...]:
    return tuple(
        round(max(0.0, float(value)), config.beam_round_decimals)
        for value in np.asarray(vector, dtype=float)
    )


def deduplicate_vectors(
    vectors: list[np.ndarray], config: DirectionalSetSizeConfig
) -> list[np.ndarray]:
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for vector in vectors:
        unique.setdefault(directional_key(vector, config), np.asarray(vector))
    return list(unique.values())


def nearest_values(values: list[float], current: float, limit: int) -> list[float]:
    ordered = sorted(set(values), key=lambda value: (abs(value - current), value))
    return sorted(ordered[:limit])


def first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        for prime in primes:
            if candidate % prime == 0:
                is_prime = False
                break
            if prime * prime > candidate:
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def compare_objective_and_key(
    left_obj: tuple[float, ...],
    left_key: tuple[float, ...],
    right_obj: tuple[float, ...],
    right_key: tuple[float, ...],
    config: DirectionalSetSizeConfig,
) -> int:
    for left, right in zip(left_obj, right_obj):
        if left < right - config.numeric_tolerance:
            return -1
        if left > right + config.numeric_tolerance:
            return 1
    return -1 if left_key < right_key else (1 if left_key > right_key else 0)


def best_candidate(
    candidates: list[dict[str, Any]], config: DirectionalSetSizeConfig
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No feasible candidate; zero must be feasible")
    best = candidates[0]
    for candidate in candidates[1:]:
        if compare_objective_and_key(
            candidate["obj"], candidate["key"], best["obj"], best["key"], config
        ) < 0:
            best = candidate
    return best


def top_beam(
    candidates: list[dict[str, Any]], config: DirectionalSetSizeConfig
) -> list[dict[str, Any]]:
    dedup: dict[tuple[float, ...], dict[str, Any]] = {}
    for candidate in candidates:
        incumbent = dedup.get(candidate["key"])
        if incumbent is None or compare_objective_and_key(
            candidate["obj"],
            candidate["key"],
            incumbent["obj"],
            incumbent["key"],
            config,
        ) < 0:
            dedup[candidate["key"]] = candidate
    return sorted(
        dedup.values(),
        key=cmp_to_key(
            lambda left, right: compare_objective_and_key(
                left["obj"], left["key"], right["obj"], right["key"], config
            )
        ),
    )[: config.beam_width]


def mark_beam_best(trace: list[dict[str, Any]], beam: list[dict[str, Any]]) -> None:
    if beam:
        trace[int(beam[0]["trace_idx"])]["selected_for_update"] = True


def set_size_trace_row(
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    score_method: str,
    phase: str,
    pass_idx: int,
    updated_cells: tuple[int, ...],
    beam_rank: int | str,
    vector: np.ndarray,
    trial: dict[str, Any],
) -> dict[str, Any]:
    plus, minus = split_directional_vector(vector, len(data.cell_values))
    metrics = trial["metrics"]
    cells = [data.cell_values[idx] for idx in updated_cells]
    return {
        "dataset": data.dataset,
        "score_method": score_method,
        "optimizer_phase": phase,
        "pass": pass_idx,
        "beam_rank": beam_rank,
        "updated_cells": json.dumps(cells, separators=(",", ":")),
        "candidate_theta_plus": vector_json(data, plus),
        "candidate_theta_minus": vector_json(data, minus),
        "candidate_key": json.dumps(directional_key(vector, config)),
        "feasible": bool(trial["feasible"]),
        "rejection_reasons": ";".join(trial["rejection_reasons"]),
        "exact_group_counts": bool(trial["fixed_counts"]),
        "max_set_size_gap": metrics["max_set_size_gap"],
        "mean_set_size_gap": metrics["mean_set_size_gap"],
        "total_set_size_gap_reduction": metrics["total_set_size_gap_reduction"],
        "max_set_size_gap_increase": metrics["max_set_size_gap_increase"],
        "max_larger_side_increase": metrics["max_larger_side_increase"],
        "max_smaller_side_increase": metrics["max_smaller_side_increase"],
        "max_neutral_group_increase": metrics["max_neutral_group_increase"],
        "shift_norm": metrics["shift_norm"],
        "selected_for_update": False,
        "selected_final": False,
    }


def set_size_search_meta(
    trace: list[dict[str, Any]], final: dict[str, Any]
) -> dict[str, Any]:
    rejected = [row for row in trace if not row["feasible"]]
    def count(reason: str) -> int:
        return sum(reason in row["rejection_reasons"].split(";") for row in rejected)
    return {
        "optimizer_name": OPTIMIZER_NAME,
        "selected_search_phase": final["phase"],
        "selected_candidate_key": final["key"],
        "candidate_count_total": len(trace),
        "candidate_count_feasible": sum(bool(row["feasible"]) for row in trace),
        "candidate_count_infeasible": len(rejected),
        "candidate_count_projected": sum(
            row["optimizer_phase"] in {"boundary_projection", "quasirandom_projection"}
            for row in trace
        ),
        "candidate_count_quasirandom": sum(
            row["optimizer_phase"] == "quasirandom" for row in trace
        ),
        "candidate_count_triple_block": sum(
            row["optimizer_phase"] == "triple_block" for row in trace
        ),
        "rejected_by_group_coverage_count": count("group_coverage_not_fixed"),
        "rejected_by_gap_increase_count": count("set_size_gap_increased"),
        "rejected_by_larger_side_count": count("larger_side_increased"),
        "rejected_by_smaller_side_count": count("smaller_side_budget_exceeded"),
        "rejected_by_neutral_cell_count": count("neutral_cell_violation"),
    }


def vector_json(data: LoadedDataset, values: np.ndarray) -> str:
    payload = {
        data.cell_values[idx]: round(float(value), 12)
        for idx, value in enumerate(values)
        if abs(float(value)) > 1e-15
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def split_cell_statistics(
    split: SplitData,
    evaluation: dict[str, Any],
    data: LoadedDataset,
) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for group in data.group_values:
        for cell_idx, cell_value in enumerate(data.cell_values):
            mask = (split.g == group) & (split.z == cell_value)
            rows[(group, cell_idx)] = {
                "n": int(mask.sum()),
                "mean_set_size": float(np.mean(evaluation["set_sizes"][mask])) if mask.any() else float("nan"),
                "coverage": float(np.mean(evaluation["covered"][mask])) if mask.any() else float("nan"),
                "covered_count": int(np.sum(evaluation["covered"][mask])),
            }
    return rows


def gap_summary_for_split(
    split: SplitData,
    baseline: dict[str, Any],
    repaired: dict[str, Any],
    data: LoadedDataset,
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    base = split_cell_statistics(split, baseline, data)
    rep = split_cell_statistics(split, repaired, data)
    base_gaps: list[float] = []
    rep_gaps: list[float] = []
    coverage_base_gaps: list[float] = []
    coverage_rep_gaps: list[float] = []
    coverage_changes: list[float] = []
    for cell_idx, role in roles.items():
        if not role["supported"]:
            continue
        first, second = data.group_values
        base_gaps.append(abs(base[(first, cell_idx)]["mean_set_size"] - base[(second, cell_idx)]["mean_set_size"]))
        rep_gaps.append(abs(rep[(first, cell_idx)]["mean_set_size"] - rep[(second, cell_idx)]["mean_set_size"]))
        coverage_base_gaps.append(abs(base[(first, cell_idx)]["coverage"] - base[(second, cell_idx)]["coverage"]))
        coverage_rep_gaps.append(abs(rep[(first, cell_idx)]["coverage"] - rep[(second, cell_idx)]["coverage"]))
        coverage_changes.extend(
            rep[(group, cell_idx)]["coverage"] - base[(group, cell_idx)]["coverage"]
            for group in data.group_values
        )
    return {
        "baseline_worst_gap": max(base_gaps, default=0.0),
        "repaired_worst_gap": max(rep_gaps, default=0.0),
        "baseline_mean_gap": float(np.mean(base_gaps)) if base_gaps else 0.0,
        "repaired_mean_gap": float(np.mean(rep_gaps)) if rep_gaps else 0.0,
        "total_gap_reduction": float(np.sum(np.asarray(base_gaps) - np.asarray(rep_gaps))),
        "baseline_overall_mean_set_size": float(np.mean(baseline["set_sizes"])),
        "repaired_overall_mean_set_size": float(np.mean(repaired["set_sizes"])),
        "baseline_worst_coverage_gap": max(coverage_base_gaps, default=0.0),
        "repaired_worst_coverage_gap": max(coverage_rep_gaps, default=0.0),
        "baseline_mean_coverage_gap": float(np.mean(coverage_base_gaps)) if coverage_base_gaps else 0.0,
        "repaired_mean_coverage_gap": float(np.mean(coverage_rep_gaps)) if coverage_rep_gaps else 0.0,
        "maximum_absolute_cellwise_coverage_change": max(
            [abs(value) for value in coverage_changes], default=0.0
        ),
    }


def build_set_size_summary(
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    score_method: str,
    evaluations: dict[str, tuple[SplitData, dict[str, Any], dict[str, Any]]],
    roles: dict[int, dict[str, Any]],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    repair_selected: dict[str, Any],
    calibration_selected: dict[str, Any],
    search_meta: dict[str, Any],
) -> dict[str, Any]:
    split_summaries = {
        name: gap_summary_for_split(split, baseline, repaired, data, roles)
        for name, (split, baseline, repaired) in evaluations.items()
    }
    repair_summary = split_summaries["repair"]
    baseline_worst = repair_summary["baseline_worst_gap"]
    reduction = baseline_worst - repair_summary["repaired_worst_gap"]
    output: dict[str, Any] = {
        "dataset": data.dataset,
        "score_method": score_method,
        "optimizer_name": OPTIMIZER_NAME,
        "alpha": config.alpha,
        "set_size_tolerance": config.set_size_tolerance,
        "smaller_side_budget": config.smaller_side_budget,
        "supported_cell_count": sum(role["supported"] for role in roles.values()),
        "neutral_cell_count": sum(role["baseline_order_tie"] for role in roles.values()),
        "selected_theta_plus": vector_json(data, theta_plus),
        "selected_theta_minus": vector_json(data, theta_minus),
        "selected_shift_norm": float(np.linalg.norm(np.concatenate([theta_plus, theta_minus]))),
        "repair_worst_gap_relative_reduction": reduction / baseline_worst if baseline_worst > 0 else float("nan"),
        "repair_group_covered_counts_exact": repair_selected["fixed_counts"],
        "calibration_group_covered_counts_exact": calibration_selected["fixed_counts"],
        **search_meta,
    }
    for split_name, values in split_summaries.items():
        for key, value in values.items():
            output[f"{split_name}_{key}"] = value
    return output


def set_size_offset_rows(
    data: LoadedDataset,
    score_method: str,
    roles: dict[int, dict[str, Any]],
    plus: np.ndarray,
    minus: np.ndarray,
    search_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": data.dataset,
            "score_method": score_method,
            "cell_idx": idx,
            "cell_value": data.cell_values[idx],
            "cell_name": data.cell_names[data.cell_values[idx]],
            "supported": role["supported"],
            "baseline_order_tie": role["baseline_order_tie"],
            "larger_set_group": role["larger_set_group"],
            "smaller_set_group": role["smaller_set_group"],
            "theta_plus": float(plus[idx]),
            "theta_minus": float(minus[idx]),
            "selected_search_phase": search_meta["selected_search_phase"],
        }
        for idx, role in roles.items()
    ]


def set_size_cell_rows(
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    score_method: str,
    split_name: str,
    split: SplitData,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    plus: np.ndarray,
    minus: np.ndarray,
) -> list[dict[str, Any]]:
    base = split_cell_statistics(split, baseline_eval, data)
    rep = split_cell_statistics(split, repaired_eval, data)
    rows: list[dict[str, Any]] = []
    first, second = data.group_values
    for cell_idx, role in roles.items():
        base_gap = abs(base[(first, cell_idx)]["mean_set_size"] - base[(second, cell_idx)]["mean_set_size"])
        rep_gap = abs(rep[(first, cell_idx)]["mean_set_size"] - rep[(second, cell_idx)]["mean_set_size"])
        base_cov_gap = abs(base[(first, cell_idx)]["coverage"] - base[(second, cell_idx)]["coverage"])
        rep_cov_gap = abs(rep[(first, cell_idx)]["coverage"] - rep[(second, cell_idx)]["coverage"])
        for group in data.group_values:
            baseline_mean = base[(group, cell_idx)]["mean_set_size"]
            repaired_mean = rep[(group, cell_idx)]["mean_set_size"]
            change = repaired_mean - baseline_mean
            if split_name == "repair" and role["supported"]:
                gap_margin = role["baseline_gap"] + config.set_size_tolerance - rep_gap
                larger_margin = (
                    baseline_mean + config.set_size_tolerance - repaired_mean
                    if group == role["larger_set_group"]
                    else float("nan")
                )
                smaller_margin = (
                    config.smaller_side_budget + config.set_size_tolerance - change
                    if group == role["smaller_set_group"]
                    else float("nan")
                )
                neutral_margin = (
                    config.smaller_side_budget + config.set_size_tolerance - change
                    if role["baseline_order_tie"]
                    else float("nan")
                )
            else:
                gap_margin = larger_margin = smaller_margin = neutral_margin = float("nan")
            rows.append(
                {
                    "dataset": data.dataset,
                    "score_method": score_method,
                    "split": split_name,
                    "cell_idx": cell_idx,
                    "cell_value": data.cell_values[cell_idx],
                    "cell_name": data.cell_names[data.cell_values[cell_idx]],
                    "group": group,
                    "group_name": data.group_names[group],
                    "n": base[(group, cell_idx)]["n"],
                    "supported": role["supported"],
                    "baseline_order_tie": role["baseline_order_tie"],
                    "larger_set_group": role["larger_set_group"],
                    "smaller_set_group": role["smaller_set_group"],
                    "baseline_mean_set_size": baseline_mean,
                    "repaired_mean_set_size": repaired_mean,
                    "mean_set_size_change": change,
                    "baseline_set_size_gap": base_gap,
                    "repaired_set_size_gap": rep_gap,
                    "set_size_gap_reduction": base_gap - rep_gap,
                    "baseline_covered_count": base[(group, cell_idx)]["covered_count"],
                    "repaired_covered_count": rep[(group, cell_idx)]["covered_count"],
                    "baseline_coverage": base[(group, cell_idx)]["coverage"],
                    "repaired_coverage": rep[(group, cell_idx)]["coverage"],
                    "coverage_change": rep[(group, cell_idx)]["coverage"] - base[(group, cell_idx)]["coverage"],
                    "baseline_coverage_gap": base_cov_gap,
                    "repaired_coverage_gap": rep_cov_gap,
                    "theta_plus": float(plus[cell_idx]),
                    "theta_minus": float(minus[cell_idx]),
                    "gap_nonincrease_margin": gap_margin,
                    "larger_side_nonincrease_margin": larger_margin,
                    "smaller_side_budget_margin": smaller_margin,
                    "neutral_group_budget_margin": neutral_margin,
                }
            )
    return rows


def set_size_group_rows(
    data: LoadedDataset,
    score_method: str,
    split_name: str,
    split: SplitData,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for group in data.group_values:
        mask = split.g == group
        rows.append(
            {
                "dataset": data.dataset,
                "score_method": score_method,
                "split": split_name,
                "group": group,
                "group_name": data.group_names[group],
                "n": int(mask.sum()),
                "baseline_covered_count": int(np.sum(baseline_eval["covered"][mask])),
                "repaired_covered_count": int(np.sum(repaired_eval["covered"][mask])),
                "baseline_coverage": float(np.mean(baseline_eval["covered"][mask])),
                "repaired_coverage": float(np.mean(repaired_eval["covered"][mask])),
                "baseline_mean_set_size": float(np.mean(baseline_eval["set_sizes"][mask])),
                "repaired_mean_set_size": float(np.mean(repaired_eval["set_sizes"][mask])),
            }
        )
    return rows


def set_size_test_rows(
    data: LoadedDataset,
    score_method: str,
    split: SplitData,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    base = split_cell_statistics(split, baseline_eval, data)
    rep = split_cell_statistics(split, repaired_eval, data)
    first, second = data.group_values
    rows = []
    for cell_idx, role in roles.items():
        rows.append(
            {
                "dataset": data.dataset,
                "score_method": score_method,
                "cell_idx": cell_idx,
                "cell_value": data.cell_values[cell_idx],
                "cell_name": data.cell_names[data.cell_values[cell_idx]],
                "supported_on_repair": role["supported"],
                "baseline_group_0_mean_set_size": base[(first, cell_idx)]["mean_set_size"],
                "baseline_group_1_mean_set_size": base[(second, cell_idx)]["mean_set_size"],
                "repaired_group_0_mean_set_size": rep[(first, cell_idx)]["mean_set_size"],
                "repaired_group_1_mean_set_size": rep[(second, cell_idx)]["mean_set_size"],
                "baseline_set_size_gap": abs(base[(first, cell_idx)]["mean_set_size"] - base[(second, cell_idx)]["mean_set_size"]),
                "repaired_set_size_gap": abs(rep[(first, cell_idx)]["mean_set_size"] - rep[(second, cell_idx)]["mean_set_size"]),
                "baseline_group_0_coverage": base[(first, cell_idx)]["coverage"],
                "baseline_group_1_coverage": base[(second, cell_idx)]["coverage"],
                "repaired_group_0_coverage": rep[(first, cell_idx)]["coverage"],
                "repaired_group_1_coverage": rep[(second, cell_idx)]["coverage"],
            }
        )
    return rows


def runtime_validation_rows(
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    score_method: str,
    repair_scores: np.ndarray,
    roles: dict[int, dict[str, Any]],
    context: dict[str, Any],
    repair_thresholds: Any,
    repair_baseline: dict[str, Any],
    zero_trial: dict[str, Any],
    repair_selected: dict[str, Any],
    calibration_selected: dict[str, Any],
    selected_vector: np.ndarray,
    trace: list[dict[str, Any]],
    protocol: ProtocolEventTrace,
) -> list[dict[str, Any]]:
    n_cells = len(data.cell_values)
    plus, minus = split_directional_vector(selected_vector, n_cells)
    repair = data.splits["repair"]
    zero_score = apply_set_size_score_matrix(
        repair_scores, repair, data, np.zeros(n_cells), np.zeros(n_cells), roles
    )
    direct = evaluate_set_size_split_direct(
        repair_scores,
        repair,
        data,
        repair_selected["thresholds"].thresholds,
        plus,
        minus,
        roles,
    )
    selected_eval = repair_selected["eval"]
    max_direct_difference = float(
        np.max(np.abs(direct["set_sizes"] - selected_eval["set_sizes"]))
    )
    rows = [
        validation_row(data, score_method, "theta_zero_scores_exact", np.array_equal(zero_score, repair_scores), 0.0),
        validation_row(data, score_method, "theta_zero_thresholds_exact", thresholds_equal(zero_trial["thresholds"].thresholds, repair_thresholds.thresholds), 0.0),
        validation_row(data, score_method, "theta_zero_group_counts_exact", zero_trial["fixed_counts"], 0.0),
        validation_row(data, score_method, "theta_zero_set_sizes_exact", np.array_equal(evaluate_shifted_score_matrix(zero_score, repair, zero_trial["thresholds"].thresholds)["set_sizes"], repair_baseline["set_sizes"]), 0.0),
        validation_row(data, score_method, "theta_zero_feasible", zero_trial["feasible"], 0.0),
        validation_row(data, score_method, "selected_group_counts_exact", repair_selected["fixed_counts"], 0.0),
        validation_row(data, score_method, "all_feasible_trace_group_counts_exact", all((not row["feasible"]) or row["exact_group_counts"] for row in trace), 0.0),
        validation_row(data, score_method, "full_matrix_matches_direct", max_direct_difference <= config.numeric_tolerance, max_direct_difference),
        validation_row(data, score_method, "every_supported_cell_checked", repair_selected["metrics"]["checked_supported_cell_count"] == len(context["supported_cells"]), repair_selected["metrics"]["checked_supported_cell_count"]),
        validation_row(data, score_method, "all_repair_feasibility_constraints", repair_selected["feasible"], 0.0),
        validation_row(data, score_method, "calibration_group_counts_exact", calibration_selected["fixed_counts"], 0.0),
    ]
    for check_name, passed in protocol.validation().items():
        rows.append(
            validation_row(
                data,
                score_method,
                check_name,
                passed,
                len(protocol.events),
            )
        )
    return rows


def validation_row(
    data: LoadedDataset,
    score_method: str,
    check: str,
    passed: bool,
    value: Any,
) -> dict[str, Any]:
    return {
        "dataset": data.dataset,
        "score_method": score_method,
        "check": check,
        "passed": bool(passed),
        "value": value,
    }


def thresholds_equal(
    left: dict[str, float], right: dict[str, float]
) -> bool:
    for key in left:
        a = float(left[key])
        b = float(right[key])
        if a == b:
            continue
        if np.isinf(a) and np.isinf(b) and np.sign(a) == np.sign(b):
            continue
        return False
    return True


def write_set_size_outputs(
    result_dir: Path,
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    score_method: str,
    artifacts: DirectionalSetSizeArtifacts,
    command: str,
) -> None:
    write_frame(result_dir / "directional_set_size_summary.csv", artifacts.summary_rows)
    write_frame(result_dir / "directional_set_size_offsets.csv", artifacts.offset_rows)
    write_frame(result_dir / "directional_set_size_cell_ledger.csv", artifacts.cell_rows)
    write_frame(result_dir / "directional_set_size_group_ledger.csv", artifacts.group_rows)
    write_frame(result_dir / "directional_set_size_search_diagnostics.csv", artifacts.search_rows)
    write_frame(result_dir / "directional_set_size_validation.csv", artifacts.validation_rows)
    write_frame(result_dir / "directional_set_size_test_results.csv", artifacts.test_rows)
    write_frame(
        result_dir / "directional_set_size_protocol_events.csv",
        artifacts.protocol_rows,
    )
    manifest = {
        "dataset": data.dataset,
        "score_method": score_method,
        "intervention_mode": data.intervention_mode,
        "group_values": data.group_values,
        "cell_values": data.cell_values,
        "split_sizes": {name: len(split.y) for name, split in data.splits.items()},
        "config": asdict(config),
        "command": command,
    }
    with (result_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    (result_dir / "COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    summary = artifacts.summary_rows[0]
    log_lines = [
        f"dataset={data.dataset}",
        f"score_method={score_method}",
        f"optimizer_name={OPTIMIZER_NAME}",
        f"candidate_count_total={summary['candidate_count_total']}",
        f"candidate_count_feasible={summary['candidate_count_feasible']}",
        f"selected_search_phase={summary['selected_search_phase']}",
        f"repair_group_covered_counts_exact={summary['repair_group_covered_counts_exact']}",
        f"calibration_group_covered_counts_exact={summary['calibration_group_covered_counts_exact']}",
    ]
    (result_dir / "execution.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )


def write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
