from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from input_loading import LoadedDataset, SplitData, write_rows_csv


@dataclass(frozen=True)
class ScoreFixConfig:
    dataset: str
    alpha: float = 0.10
    epsilon: float = 0.01
    gap_tol: float = 1e-9
    coverage_tol: float = 1e-9
    min_cell_count: int = 30
    score_methods: tuple[str, ...] = ("simple", "aps", "raps", "saps")
    optimizer_passes: int = 3
    optimizer_name: str = "boundary_search"
    beam_width: int = 20
    global_lambda_grid: tuple[float, ...] = tuple(i / 40.0 for i in range(121))
    boundary_projection_grid_size: int = 41
    boundary_projection_max_candidates: int = 12
    local_grid_multipliers: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
    pair_grid_multipliers: tuple[float, ...] = (-0.5, 0.0, 0.5)
    pair_block_limit: int = 6
    triple_grid_multipliers: tuple[float, ...] = (-0.25, 0.0, 0.25)
    triple_block_limit: int = 2
    quasirandom_seed: int = 2026
    quasirandom_candidate_count: int = 1000
    no_gap_improvement_tol: float = 1e-12
    beam_round_decimals: int = 12
    grid_multipliers: tuple[float, ...] = tuple([i / 10.0 for i in range(16)] + [2.0, 2.5, 3.0])
    raps_lambda: float = 0.01
    raps_k_reg: int = 5
    saps_lambda: float = 0.2
    numeric_tolerance: float = 1e-10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdResult:
    thresholds: dict[str, float]
    covered_counts: dict[str, int]
    n_by_group: dict[str, int]
    exact: bool


def run_no_free_repair_experiment(data: LoadedDataset, result_dir: Path, config: ScoreFixConfig) -> None:
    validate_dataset(data)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    # Validate every score method at theta = 0 before any nonzero candidate is searched.
    all_baseline_rows = validate_theta_zero_baselines(data, config)
    all_summary: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    all_cell_rows: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    all_theta_rows: list[dict[str, Any]] = []
    # Run the score-shifting protocol independently for each nonconformity score.
    for score_method in config.score_methods:
        frames = run_one_score_method(data, config, score_method)
        all_summary.extend(frames["summary"])
        all_group_rows.extend(frames["group"])
        all_label_rows.extend(frames["label"])
        all_cell_rows.extend(frames["cell"])
        all_trace_rows.extend(frames["trace"])
        all_theta_rows.extend(frames["theta"])
    write_outputs(
        result_dir,
        data,
        config,
        all_summary,
        all_group_rows,
        all_label_rows,
        all_cell_rows,
        all_trace_rows,
        all_theta_rows,
        all_baseline_rows,
    )


def validate_theta_zero_baselines(data: LoadedDataset, config: ScoreFixConfig) -> list[dict[str, Any]]:
    validate_dataset(data)
    rows: list[dict[str, Any]] = []
    for score_method in config.score_methods:
        base_scores = {
            split_name: labelwise_scores_from_probs(
                split_data.probs,
                score_method,
                raps_lambda=config.raps_lambda,
                raps_k_reg=config.raps_k_reg,
                saps_lambda=config.saps_lambda,
            )
            for split_name, split_data in data.splits.items()
        }
        repair = data.splits["repair"]
        repair_thresholds = compute_baseline_group_thresholds(
            base_scores["repair"],
            repair,
            data.group_values,
            config.alpha,
        )
        repair_eval = evaluate_split(base_scores["repair"], repair, repair_thresholds.thresholds)
        roles = define_repair_roles(repair_eval, data, config)

        calibration = data.splits["calibration"]
        calibration_thresholds = compute_baseline_group_thresholds(
            base_scores["calibration"],
            calibration,
            data.group_values,
            config.alpha,
        )
        calibration_eval = evaluate_split(
            base_scores["calibration"],
            calibration,
            calibration_thresholds.thresholds,
        )
        test_eval = evaluate_split(
            base_scores["test"],
            data.splits["test"],
            calibration_thresholds.thresholds,
        )
        rows.extend(
            baseline_validation_rows(
                data,
                config,
                score_method,
                base_scores,
                repair_thresholds,
                calibration_thresholds,
                repair_eval,
                calibration_eval,
                test_eval,
                roles,
            )
        )
    return rows


def labelwise_scores_from_probs(
    probs: np.ndarray,
    score: str,
    raps_lambda: float = 0.01,
    raps_k_reg: int = 5,
    saps_lambda: float = 0.2,
) -> np.ndarray:
    # Construct deterministic label-wise nonconformity score matrices from probabilities.
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
    raise ValueError(f"Unsupported score method: {score}")


def true_scores(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=float)[np.arange(len(y)), np.asarray(y, dtype=int)]


def run_one_score_method(data: LoadedDataset, config: ScoreFixConfig, score_method: str) -> dict[str, list[dict[str, Any]]]:
    method_start = time.perf_counter()
    # Build all split-level score matrices once so optimization never recomputes probabilities.
    base_scores = {
        split: labelwise_scores_from_probs(
            split_data.probs,
            score_method,
            raps_lambda=config.raps_lambda,
            raps_k_reg=config.raps_k_reg,
            saps_lambda=config.saps_lambda,
        )
        for split, split_data in data.splits.items()
    }
    repair = data.splits["repair"]
    # Baseline group-Mondrian thresholds set the group covered-count targets.
    repair_base_thresholds = compute_baseline_group_thresholds(base_scores["repair"], repair, data.group_values, config.alpha)
    repair_baseline_eval = evaluate_split(base_scores["repair"], repair, repair_base_thresholds.thresholds)
    # Repair roles identify the initially advantaged and disadvantaged group for each frozen Z cell.
    roles = define_repair_roles(repair_baseline_eval, data, config)
    repair_metric_context = build_metric_context(base_scores["repair"], repair, data, config, repair_baseline_eval, roles)
    grids = build_theta_grids(base_scores["repair"], repair, data, roles, config)
    cal = data.splits["calibration"]
    cal_base_thresholds = compute_baseline_group_thresholds(
        base_scores["calibration"],
        cal,
        data.group_values,
        config.alpha,
    )
    cal_baseline_eval = evaluate_split(base_scores["calibration"], cal, cal_base_thresholds.thresholds)
    test = data.splits["test"]
    test_baseline_eval = evaluate_split(base_scores["test"], test, cal_base_thresholds.thresholds)
    # The oracle is an optimistic coverage-allocation bound, not a score-shift candidate.
    oracle = coverage_allocation_oracle(
        data,
        config,
        score_method,
        repair,
        repair_baseline_eval,
        repair_base_thresholds.covered_counts,
        roles,
    )
    # Select theta using only the repair split and the no-free-repair constraints.
    theta, trace_rows, feasible, optimizer_meta = boundary_search(
        base_scores["repair"],
        repair,
        data,
        config,
        roles,
        grids,
        repair_base_thresholds.covered_counts,
        repair_baseline_eval,
        repair_metric_context,
        score_method,
    )
    # Re-evaluate the selected repair on the repair split for ledgers and summary checks.
    repaired_repair_eval = evaluate_repaired_with_fixed_counts(
        base_scores["repair"],
        repair,
        data,
        theta,
        roles,
        repair_base_thresholds.covered_counts,
        repair_baseline_eval,
        config,
        repair_metric_context,
    )
    # Freeze theta, then choose calibration thresholds that preserve calibration group counts.
    repaired_cal_eval = evaluate_repaired_with_fixed_counts(
        base_scores["calibration"],
        cal,
        data,
        theta,
        roles,
        cal_base_thresholds.covered_counts,
        cal_baseline_eval,
        config,
    )
    # Test is held out: use calibration thresholds and report outcomes without retuning theta.
    test_repaired_eval = evaluate_shifted_split(
        base_scores["test"],
        test,
        data,
        repaired_cal_eval["thresholds"].thresholds,
        theta,
        roles,
    )
    split_evals = {
        "repair": (repair_baseline_eval, repaired_repair_eval["eval"]),
        "calibration": (cal_baseline_eval, repaired_cal_eval["eval"]),
        "test": (test_baseline_eval, test_repaired_eval),
    }
    split_datas = {"repair": repair, "calibration": cal, "test": test}
    group_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    # Ledgers expose group, label, and cell-level coverage and set-size accounting.
    for split, (baseline_eval, repaired_eval) in split_evals.items():
        group_rows.extend(group_ledger_rows(data, score_method, split, split_datas[split], baseline_eval, repaired_eval))
        label_rows.extend(label_gap_rows(data, config, score_method, split, baseline_eval, repaired_eval, roles, theta))
        cell_rows.extend(cell_suffer_rows(data, config, score_method, split, baseline_eval, repaired_eval, roles))
    theta_rows = selected_theta_rows(data, score_method, roles, theta, optimizer_meta)
    summary = summary_row(
            data,
            config,
            score_method,
            theta,
            feasible and bool(repaired_repair_eval["feasible"]),
            repair_baseline_eval,
            repaired_repair_eval["eval"],
            cal_baseline_eval,
            repaired_cal_eval["eval"],
            test_baseline_eval,
            test_repaired_eval,
            roles,
            optimizer_meta,
            oracle,
        )
    summary["wall_clock_runtime_seconds"] = float(time.perf_counter() - method_start)
    return {
        "summary": [summary],
        "group": group_rows,
        "label": label_rows,
        "cell": cell_rows,
        "trace": trace_rows,
        "theta": theta_rows,
    }


def compute_baseline_group_thresholds(scores: np.ndarray, split: SplitData, groups: list[str], alpha: float) -> ThresholdResult:
    # Standard group-Mondrian conformal thresholds define baseline group coverage.
    ts = true_scores(scores, split.y)
    thresholds: dict[str, float] = {}
    covered_counts: dict[str, int] = {}
    n_by_group: dict[str, int] = {}
    for group in groups:
        mask = split.g == group
        vals = ts[mask]
        threshold = conformal_quantile(vals, alpha)
        thresholds[group] = float(threshold)
        covered_counts[group] = int((vals <= threshold).sum())
        n_by_group[group] = int(mask.sum())
    return ThresholdResult(thresholds=thresholds, covered_counts=covered_counts, n_by_group=n_by_group, exact=True)


def compute_thresholds_for_target_counts(scores: np.ndarray, split: SplitData, groups: list[str], target_counts: dict[str, int]) -> ThresholdResult:
    ts = true_scores(scores, split.y)
    return compute_thresholds_for_target_counts_from_true_scores(ts, split, groups, target_counts)


def compute_thresholds_for_target_counts_from_true_scores(ts: np.ndarray, split: SplitData, groups: list[str], target_counts: dict[str, int]) -> ThresholdResult:
    # Reselect group thresholds so each group matches a target covered count exactly.
    thresholds: dict[str, float] = {}
    covered_counts: dict[str, int] = {}
    n_by_group: dict[str, int] = {}
    exact = True
    for group in groups:
        vals = np.asarray(ts[split.g == group], dtype=float)
        target = int(target_counts[group])
        n_by_group[group] = int(len(vals))
        threshold, is_exact = threshold_for_exact_count(vals, target)
        thresholds[group] = float(threshold)
        count = int((vals <= threshold).sum())
        covered_counts[group] = count
        exact = exact and is_exact and count == target
    return ThresholdResult(thresholds=thresholds, covered_counts=covered_counts, n_by_group=n_by_group, exact=exact)


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if k > n:
        return float("inf")
    return float(np.partition(vals, k - 1)[k - 1])


def threshold_for_exact_count(values: np.ndarray, target: int) -> tuple[float, bool]:
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if target <= 0:
        threshold = float(np.nextafter(np.nanmin(vals), -np.inf)) if n else float("-inf")
        return threshold, int((vals <= threshold).sum()) == 0
    if target >= n:
        return float("inf"), True
    kth = float(np.partition(vals, target - 1)[target - 1])
    count = int((vals <= kth).sum())
    return kth, count == target


def evaluate_split(scores: np.ndarray, split: SplitData, thresholds: dict[str, float]) -> dict[str, Any]:
    threshold_by_row = row_thresholds(split, thresholds)
    prediction_sets = np.asarray(scores, dtype=float) <= threshold_by_row[:, None]
    covered = prediction_sets[np.arange(len(split.y)), split.y]
    set_sizes = prediction_sets.sum(axis=1).astype(float)
    return {"covered": covered.astype(bool), "set_sizes": set_sizes, "thresholds": thresholds}


def baseline_validation_rows(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    base_scores: dict[str, np.ndarray],
    repair_thresholds: ThresholdResult,
    calibration_thresholds: ThresholdResult,
    repair_eval: dict[str, Any],
    calibration_eval: dict[str, Any],
    test_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    # Independently reproduce theta-zero thresholds, prediction sets, and G x Z cell summaries.
    split_specs = {
        "repair": (repair_thresholds.thresholds, repair_eval, True),
        "calibration": (calibration_thresholds.thresholds, calibration_eval, True),
        "test": (calibration_thresholds.thresholds, test_eval, False),
    }
    rows: list[dict[str, Any]] = []
    for split_name, (expected_thresholds, expected_eval, derive_thresholds) in split_specs.items():
        split = data.splits[split_name]
        if derive_thresholds:
            independent_thresholds = independent_group_mondrian_thresholds(
                base_scores[split_name],
                split,
                data.group_values,
                config.alpha,
            )
            for group in data.group_values:
                left = float(independent_thresholds[group])
                right = float(expected_thresholds[group])
                if not (
                    left == right
                    or (np.isinf(left) and np.isinf(right) and np.sign(left) == np.sign(right))
                ):
                    raise AssertionError(
                        f"{data.dataset} {score_method} {split_name}: independent threshold differs "
                        f"for G={group}: {left} != {right}"
                    )
        else:
            independent_thresholds = dict(expected_thresholds)
        independent_eval = independent_evaluate_split(base_scores[split_name], split, independent_thresholds)
        if not np.array_equal(independent_eval["covered"], expected_eval["covered"]):
            raise AssertionError(
                f"{data.dataset} {score_method} {split_name}: theta=0 covered indicators differ "
                "from independent baseline calculation"
            )
        if not np.array_equal(independent_eval["set_sizes"], expected_eval["set_sizes"]):
            raise AssertionError(
                f"{data.dataset} {score_method} {split_name}: theta=0 set sizes differ "
                "from independent baseline calculation"
            )
        zero_theta = np.zeros(len(data.cell_values), dtype=float)
        if derive_thresholds:
            target_counts = {
                group: int(expected_eval["covered"][split.g == group].sum())
                for group in data.group_values
            }
            zero_true_scores = shifted_true_scores(
                base_scores[split_name],
                split,
                data,
                zero_theta,
                roles,
            )
            zero_threshold_result = compute_thresholds_for_target_counts_from_true_scores(
                zero_true_scores,
                split,
                data.group_values,
                target_counts,
            )
            if not zero_threshold_result.exact:
                raise AssertionError(
                    f"{data.dataset} {score_method} {split_name}: theta=0 could not preserve "
                    "baseline group covered counts exactly"
                )
            zero_thresholds = zero_threshold_result.thresholds
        else:
            zero_thresholds = independent_thresholds
        refactored_zero_eval = evaluate_shifted_split(
            base_scores[split_name],
            split,
            data,
            zero_thresholds,
            zero_theta,
            roles,
        )
        if not np.array_equal(refactored_zero_eval["covered"], expected_eval["covered"]):
            raise AssertionError(
                f"{data.dataset} {score_method} {split_name}: generalized G x Z theta=0 "
                "covered indicators differ from baseline"
            )
        if not np.array_equal(refactored_zero_eval["set_sizes"], expected_eval["set_sizes"]):
            raise AssertionError(
                f"{data.dataset} {score_method} {split_name}: generalized G x Z theta=0 "
                "set sizes differ from baseline"
            )
        independent_cells = cell_stats(split, independent_eval, data.cell_values, data.group_values)
        expected_cells = cell_stats(split, expected_eval, data.cell_values, data.group_values)
        assert_cell_statistics_equal(data, score_method, split_name, independent_cells, expected_cells)
        if data.intervention_mode == "label_defined_z":
            assert_label_defined_z_baseline_equivalence(data, score_method, split_name, split, expected_eval)

        finite_gaps: list[float] = []
        for cell_idx in range(len(data.cell_values)):
            cov0 = expected_cells[(data.group_values[0], cell_idx)]["coverage"]
            cov1 = expected_cells[(data.group_values[1], cell_idx)]["coverage"]
            if np.isfinite(cov0) and np.isfinite(cov1):
                finite_gaps.append(abs(float(cov0) - float(cov1)))
        max_gap = float(max(finite_gaps)) if finite_gaps else 0.0
        mean_gap = float(np.mean(finite_gaps)) if finite_gaps else 0.0
        avg_set_size = float(np.mean(expected_eval["set_sizes"]))
        group_counts = {
            group: int(expected_eval["covered"][split.g == group].sum())
            for group in data.group_values
        }
        for cell_idx, cell_value in enumerate(data.cell_values):
            for group in data.group_values:
                stats = expected_cells[(group, cell_idx)]
                rows.append(
                    {
                        "dataset": data.dataset,
                        "score_method": score_method,
                        "split": split_name,
                        "group": group,
                        "group_name": data.group_names.get(group, group),
                        "z": cell_value,
                        "z_name": data.cell_names.get(cell_value, cell_value),
                        "cell_order": cell_idx,
                        "n": stats["n"],
                        "baseline_covered_count": stats["covered_count"],
                        "baseline_coverage": stats["coverage"],
                        "baseline_group_covered_count": group_counts[group],
                        "baseline_max_gz_gap": max_gap,
                        "baseline_mean_gz_gap": mean_gap,
                        "baseline_average_set_size": avg_set_size,
                        "group_encoding_matches_reference": True,
                        "cell_encoding_matches_reference": True,
                        "cell_order_matches_reference": True,
                        "theta_zero_matches_independent_baseline": True,
                        "label_defined_z_matches_y_cells": bool(
                            data.intervention_mode != "label_defined_z" or np.array_equal(split.z, canonical_y_ids(split.y))
                        ),
                    }
                )
    return rows


def independent_group_mondrian_thresholds(
    scores: np.ndarray,
    split: SplitData,
    groups: list[str],
    alpha: float,
) -> dict[str, float]:
    ts = true_scores(scores, split.y)
    thresholds: dict[str, float] = {}
    for group in groups:
        values = np.sort(np.asarray(ts[split.g == group], dtype=float))
        n = len(values)
        if n == 0:
            thresholds[group] = float("inf")
            continue
        rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
        thresholds[group] = float("inf") if rank > n else float(values[rank - 1])
    return thresholds


def independent_evaluate_split(
    scores: np.ndarray,
    split: SplitData,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    threshold_by_row = np.asarray([float(thresholds[str(group)]) for group in split.g], dtype=float)
    prediction_sets = np.less_equal(np.asarray(scores, dtype=float), threshold_by_row[:, None])
    covered = prediction_sets[np.arange(len(split.y)), np.asarray(split.y, dtype=int)]
    return {
        "covered": covered.astype(bool),
        "set_sizes": np.count_nonzero(prediction_sets, axis=1).astype(float),
        "thresholds": thresholds,
    }


def assert_cell_statistics_equal(
    data: LoadedDataset,
    score_method: str,
    split_name: str,
    independent: dict[tuple[str, int], dict[str, float]],
    expected: dict[tuple[str, int], dict[str, float]],
) -> None:
    for key in independent:
        left = independent[key]
        right = expected[key]
        if left["n"] != right["n"] or left["covered_count"] != right["covered_count"]:
            raise AssertionError(
                f"{data.dataset} {score_method} {split_name}: independent G x Z counts differ at {key}"
            )
        for metric in ("coverage", "avg_set_size"):
            a = float(left[metric])
            b = float(right[metric])
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                raise AssertionError(
                    f"{data.dataset} {score_method} {split_name}: independent {metric} differs at {key}"
                )


def assert_label_defined_z_baseline_equivalence(
    data: LoadedDataset,
    score_method: str,
    split_name: str,
    split: SplitData,
    baseline_eval: dict[str, Any],
) -> None:
    expected_z = canonical_y_ids(split.y)
    if not np.array_equal(split.z, expected_z):
        raise AssertionError(f"{data.dataset} {split_name}: label-defined Z differs from Y cells")
    for group in data.group_values:
        for label_idx, cell_value in enumerate(data.cell_values):
            z_mask = (split.g == group) & (split.z == cell_value)
            y_mask = (split.g == group) & (split.y == label_idx)
            if not np.array_equal(z_mask, y_mask):
                raise AssertionError(
                    f"{data.dataset} {score_method} {split_name}: G x Z and G x Y masks differ "
                    f"for G={group}, cell={cell_value}"
                )
            if int(baseline_eval["covered"][z_mask].sum()) != int(baseline_eval["covered"][y_mask].sum()):
                raise AssertionError(
                    f"{data.dataset} {score_method} {split_name}: covered counts differ "
                    f"for G={group}, cell={cell_value}"
                )


def canonical_y_ids(y: np.ndarray) -> np.ndarray:
    return np.asarray([str(int(value)) for value in np.asarray(y, dtype=int)], dtype=object)


def row_thresholds(split: SplitData, thresholds: dict[str, float]) -> np.ndarray:
    row_thresholds = np.empty(len(split.g), dtype=float)
    for group, threshold in thresholds.items():
        row_thresholds[split.g == str(group)] = float(threshold)
    return row_thresholds


def evaluate_repaired_with_fixed_counts(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    theta: np.ndarray,
    roles: dict[int, dict[str, Any]],
    target_counts: dict[str, int],
    baseline_eval: dict[str, Any],
    config: ScoreFixConfig,
    metric_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Apply G x Z shifts, then refit group thresholds to preserve group coverage counts.
    if metric_context is not None and "base_true_scores" in metric_context:
        shifted_true = (
            metric_context["base_true_scores"]
            + metric_context["true_shift_signs"] * theta[metric_context["row_cell_indices"]]
        )
    else:
        shifted_true = shifted_true_scores(base_scores, split, data, theta, roles)
    thresholds = compute_thresholds_for_target_counts_from_true_scores(shifted_true, split, data.group_values, target_counts)
    if metric_context is not None and "candidate_shift_signs" in metric_context:
        repaired_eval = evaluate_shifted_split_fast(
            base_scores,
            split,
            thresholds.thresholds,
            theta,
            metric_context["candidate_shift_signs"],
            metric_context["row_cell_indices"],
            data.intervention_mode,
        )
    else:
        repaired_eval = evaluate_shifted_split(base_scores, split, data, thresholds.thresholds, theta, roles)
    exact_counts = {g: int(repaired_eval["covered"][split.g == g].sum()) for g in data.group_values}
    fixed = all(exact_counts[g] == int(target_counts[g]) for g in data.group_values)
    if metric_context is None:
        metrics = candidate_metrics(data, config, split, baseline_eval, repaired_eval, roles, theta)
    else:
        metrics = candidate_metrics_fast(config, repaired_eval, roles, theta, metric_context)
    # Feasibility enforces count preservation, advantaged no-harm, gap non-worsening, and disadvantaged non-harm.
    reasons = candidate_rejection_reasons(thresholds.exact and fixed, metrics, config)
    feasible = not reasons
    return {
        "eval": repaired_eval,
        "thresholds": thresholds,
        "metrics": metrics,
        "feasible": feasible,
        "fixed_counts": fixed,
        "rejection_reasons": reasons,
    }


def shifted_true_scores(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    theta: np.ndarray,
    roles: dict[int, dict[str, Any]],
) -> np.ndarray:
    # True-label scores receive the signed shift associated with each row's frozen Z cell.
    ts = true_scores(base_scores, split.y).copy()
    for cell_idx, amount in enumerate(theta):
        if float(amount) == 0.0:
            continue
        role = roles[cell_idx]
        if not role["repairable"]:
            continue
        cell_mask = split.z == data.cell_values[cell_idx]
        ts[cell_mask & (split.g == role["advantaged_group"])] += float(amount)
        ts[cell_mask & (split.g == role["disadvantaged_group"])] -= float(amount)
    return ts


def evaluate_shifted_split(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    thresholds: dict[str, float],
    theta: np.ndarray,
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    # Evaluate candidate-label sets under label-defined or covariate-defined Z intervention.
    threshold_by_row = row_thresholds(split, thresholds)
    shifted_scores = shifted_score_matrix(base_scores, split, data, theta, roles)
    prediction_sets = shifted_scores <= threshold_by_row[:, None]
    covered = prediction_sets[np.arange(len(split.y)), split.y]
    set_sizes = prediction_sets.sum(axis=1).astype(float)
    return {"covered": covered.astype(bool), "set_sizes": set_sizes, "thresholds": thresholds}


def evaluate_shifted_split_fast(
    base_scores: np.ndarray,
    split: SplitData,
    thresholds: dict[str, float],
    theta: np.ndarray,
    candidate_shift_signs: np.ndarray,
    row_cell_indices: np.ndarray,
    intervention_mode: str,
) -> dict[str, Any]:
    threshold_by_row = row_thresholds(split, thresholds)
    if intervention_mode == "label_defined_z":
        shifted_scores = (
            np.asarray(base_scores, dtype=float)
            + np.asarray(candidate_shift_signs, dtype=float) * np.asarray(theta, dtype=float)[None, :]
        )
    elif intervention_mode == "covariate_defined_z":
        row_shift = np.asarray(candidate_shift_signs, dtype=float) * np.asarray(theta, dtype=float)[row_cell_indices]
        shifted_scores = np.asarray(base_scores, dtype=float) + row_shift[:, None]
    else:
        raise ValueError(f"Unsupported intervention_mode: {intervention_mode}")
    prediction_sets = shifted_scores <= threshold_by_row[:, None]
    covered = prediction_sets[np.arange(len(split.y)), split.y]
    set_sizes = prediction_sets.sum(axis=1).astype(float)
    return {"covered": covered.astype(bool), "set_sizes": set_sizes, "thresholds": thresholds}


def shifted_score_matrix(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    theta: np.ndarray,
    roles: dict[int, dict[str, Any]],
) -> np.ndarray:
    shifted = np.asarray(base_scores, dtype=float).copy()
    if data.intervention_mode == "label_defined_z":
        for cell_idx, amount in enumerate(theta):
            role = roles[cell_idx]
            if float(amount) == 0.0 or not role["repairable"]:
                continue
            shifted[split.g == role["advantaged_group"], cell_idx] += float(amount)
            shifted[split.g == role["disadvantaged_group"], cell_idx] -= float(amount)
        return shifted
    if data.intervention_mode == "covariate_defined_z":
        row_shift = np.zeros(len(split.y), dtype=float)
        for cell_idx, amount in enumerate(theta):
            role = roles[cell_idx]
            if float(amount) == 0.0 or not role["repairable"]:
                continue
            cell_mask = split.z == data.cell_values[cell_idx]
            row_shift[cell_mask & (split.g == role["advantaged_group"])] += float(amount)
            row_shift[cell_mask & (split.g == role["disadvantaged_group"])] -= float(amount)
        shifted += row_shift[:, None]
        return shifted
    raise ValueError(f"Unsupported intervention_mode: {data.intervention_mode}")


def define_repair_roles(baseline_eval: dict[str, Any], data: LoadedDataset, config: ScoreFixConfig) -> dict[int, dict[str, Any]]:
    # A repairable Z cell has support in both groups and a nonzero baseline local gap.
    roles: dict[int, dict[str, Any]] = {}
    cell = cell_stats(data.splits["repair"], baseline_eval, data.cell_values, data.group_values)
    groups = data.group_values
    for cell_idx in range(len(data.cell_values)):
        n0 = cell[(groups[0], cell_idx)]["n"]
        n1 = cell[(groups[1], cell_idx)]["n"]
        supported = n0 >= config.min_cell_count and n1 >= config.min_cell_count
        cov0 = cell[(groups[0], cell_idx)]["coverage"]
        cov1 = cell[(groups[1], cell_idx)]["coverage"]
        gap = abs(cov0 - cov1) if supported else float("nan")
        repairable = bool(supported and np.isfinite(gap) and gap > config.numeric_tolerance)
        if repairable and cov0 >= cov1:
            adv, dis = groups[0], groups[1]
        elif repairable:
            adv, dis = groups[1], groups[0]
        else:
            adv, dis = "", ""
        roles[cell_idx] = {
            "cell_value": data.cell_values[cell_idx],
            "supported": supported,
            "repairable": repairable,
            "advantaged_group": adv,
            "disadvantaged_group": dis,
            "baseline_gap": gap,
        }
    return roles


def build_theta_grids(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> dict[int, list[float]]:
    # Candidate shift scales are derived from group-specific true-score quantile gaps.
    ts = true_scores(base_scores, split.y)
    grids: dict[int, list[float]] = {}
    for cell_idx, role in roles.items():
        if not role["repairable"]:
            grids[cell_idx] = [0.0]
            continue
        cell_mask = split.z == data.cell_values[cell_idx]
        adv_scores = ts[(split.g == role["advantaged_group"]) & cell_mask]
        dis_scores = ts[(split.g == role["disadvantaged_group"]) & cell_mask]
        q = 1.0 - float(config.alpha)
        scale = abs(float(np.quantile(adv_scores, q)) - float(np.quantile(dis_scores, q)))
        if not np.isfinite(scale) or scale < 1e-8:
            combined = np.concatenate([adv_scores, dis_scores])
            iqr = float(np.quantile(combined, 0.75) - np.quantile(combined, 0.25))
            scale = iqr if np.isfinite(iqr) and iqr >= 1e-8 else 0.05
        grids[cell_idx] = sorted({round(float(m) * scale, 12) for m in config.grid_multipliers})
    return grids


def boundary_search(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    config: ScoreFixConfig,
    roles: dict[int, dict[str, Any]],
    grids: dict[int, list[float]],
    target_counts: dict[str, int],
    baseline_eval: dict[str, Any],
    metric_context: dict[str, Any],
    score_method: str,
) -> tuple[np.ndarray, list[dict[str, Any]], bool, dict[str, Any]]:
    k = len(data.cell_values)
    trace: list[dict[str, Any]] = []
    feasible_candidates: list[dict[str, Any]] = []
    eval_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    zero_metrics: dict[str, Any] | None = None

    def evaluate_candidate(
        theta: np.ndarray,
        phase: str,
        pass_idx: int = 0,
        updated_label_idx: int | None = None,
        updated_label_idx_2: int | None = None,
        updated_label_idx_3: int | None = None,
        beam_rank: int | str = "",
        selected_for_update: bool = False,
    ) -> dict[str, Any]:
        # All optimizer phases share one deterministic candidate evaluator and trace writer.
        trial_theta = np.maximum(np.asarray(theta, dtype=float), 0.0)
        key = theta_key(trial_theta, config)
        trial = eval_cache.get(key)
        if trial is None:
            trial = evaluate_repaired_with_fixed_counts(
                base_scores,
                split,
                data,
                trial_theta,
                roles,
                target_counts,
                baseline_eval,
                config,
                metric_context,
            )
            eval_cache[key] = trial
        row = trace_row(
            data,
            config,
            score_method,
            phase,
            pass_idx,
            updated_label_idx,
            updated_label_idx_2,
            updated_label_idx_3,
            beam_rank,
            trial_theta,
            trial,
            selected_for_update=selected_for_update,
        )
        trace.append(row)
        if trial["feasible"]:
            candidate = {
                "theta": trial_theta.copy(),
                "eval": trial,
                "obj": objective_tuple(trial["metrics"], zero_metrics),
                "phase": phase,
                "trace_idx": len(trace) - 1,
                "key": key,
            }
            feasible_candidates.append(candidate)
        return trial

    zero_theta = np.zeros(k, dtype=float)
    # Theta zero must remain feasible and anchors the no-gap-improvement tie-break.
    zero_eval = evaluate_candidate(zero_theta, "zero_baseline", pass_idx=0)
    if not zero_eval["feasible"]:
        raise RuntimeError(
            f"{data.dataset} {score_method}: theta=0 is infeasible under no-free-repair constraints: "
            f"{zero_eval['rejection_reasons']}"
        )
    zero_metrics = zero_eval["metrics"]
    for candidate in feasible_candidates:
        candidate["obj"] = objective_tuple(candidate["eval"]["metrics"], zero_metrics)

    # Search the simultaneous repair direction before local refinements.
    infeasible_global = run_global_line_search(grids, roles, config, evaluate_candidate)
    current_best = run_boundary_projection(
        infeasible_global,
        "boundary_projection",
        feasible_candidates,
        trace,
        config,
        zero_metrics,
        evaluate_candidate,
    )
    # Probe local threshold-crossing grids and block neighborhoods around feasible states.
    run_critical_grid_search(base_scores, split, data, roles, grids, config, current_best, evaluate_candidate)
    run_coordinate_grid_search(k, grids, config, zero_eval, zero_metrics, trace, evaluate_candidate)
    run_beam_block_search(data, split, roles, grids, config, feasible_candidates, trace, evaluate_candidate)
    # Deterministic quasi-random restarts broaden the search without gradients or randomness at runtime.
    infeasible_quasirandom = run_quasirandom_search(grids, roles, config, score_method, evaluate_candidate)
    run_boundary_projection(
        infeasible_quasirandom,
        "quasirandom_projection",
        feasible_candidates,
        trace,
        config,
        zero_metrics,
        evaluate_candidate,
    )

    final_candidate = best_feasible_candidate(feasible_candidates, config)
    trace[final_candidate["trace_idx"]]["selected_final"] = True
    trace[final_candidate["trace_idx"]]["selected_for_update"] = True
    meta = optimizer_meta(trace, final_candidate, config, feasible_candidates, zero_metrics)
    return final_candidate["theta"], trace, bool(feasible_candidates), meta


def run_global_line_search(
    grids: dict[int, list[float]],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
    evaluate_candidate: Any,
) -> list[dict[str, Any]]:
    # Evaluate theta(lambda) along a single global repair direction.
    global_direction = global_repair_direction(grids, roles, config)
    infeasible_sources: list[dict[str, Any]] = []
    for lambda_value in config.global_lambda_grid:
        theta = float(lambda_value) * global_direction
        trial = evaluate_candidate(theta, "global_line", pass_idx=0)
        if not trial["feasible"]:
            infeasible_sources.append({"theta": theta.copy(), "eval": trial})
    return infeasible_sources


def run_boundary_projection(
    sources: list[dict[str, Any]],
    phase: str,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    config: ScoreFixConfig,
    zero_metrics: dict[str, Any] | None,
    evaluate_candidate: Any,
) -> dict[str, Any]:
    # Project promising infeasible candidates back toward the current feasible boundary.
    best = best_feasible_candidate(feasible_candidates, config)
    projected_sources = projection_sources(sources, best, config, zero_metrics)
    for source_rank, source in enumerate(projected_sources, start=1):
        theta_bad = source["theta"]
        theta_good = best["theta"]
        best_projection_idx: int | None = None
        best_projection_obj = best["obj"]
        strongest_feasible_idx: int | None = None
        strongest_feasible_t = -1.0
        for t in np.linspace(0.0, 1.0, int(config.boundary_projection_grid_size)):
            projected_theta = (1.0 - float(t)) * theta_good + float(t) * theta_bad
            before_len = len(trace)
            trial = evaluate_candidate(projected_theta, phase, pass_idx=source_rank)
            if trial["feasible"]:
                obj = objective_tuple(trial["metrics"], zero_metrics)
                if lexicographically_better(obj, best_projection_obj, config.numeric_tolerance):
                    best_projection_obj = obj
                    best_projection_idx = before_len
                if float(t) > strongest_feasible_t:
                    strongest_feasible_t = float(t)
                    strongest_feasible_idx = before_len
        marked_idx = best_projection_idx if best_projection_idx is not None else strongest_feasible_idx
        if marked_idx is not None:
            trace[marked_idx]["selected_for_update"] = True
            best = best_feasible_candidate(feasible_candidates, config)
    return best


def run_critical_grid_search(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    roles: dict[int, dict[str, Any]],
    grids: dict[int, list[float]],
    config: ScoreFixConfig,
    current_best: dict[str, Any],
    evaluate_candidate: Any,
) -> None:
    # Try fine local values near zero, quantile alignment, and threshold crossings.
    critical_order = prioritized_label_order(data, split, current_best["eval"]["eval"], roles, config)
    for label_idx in critical_order:
        for value in critical_block_values(
            label_idx,
            float(current_best["theta"][label_idx]),
            grids[label_idx],
            base_scores,
            split,
            data,
            current_best["eval"]["eval"],
            roles,
            config,
        ):
            trial_theta = current_best["theta"].copy()
            trial_theta[label_idx] = value
            evaluate_candidate(trial_theta, "critical_grid", pass_idx=0, updated_label_idx=label_idx)


def run_coordinate_grid_search(
    k: int,
    grids: dict[int, list[float]],
    config: ScoreFixConfig,
    zero_eval: dict[str, Any],
    zero_metrics: dict[str, Any],
    trace: list[dict[str, Any]],
    evaluate_candidate: Any,
) -> None:
    # Single-label coordinate search follows the configured constrained grid.
    coordinate_theta = np.zeros(k, dtype=float)
    coordinate_current_obj = objective_tuple(zero_eval["metrics"], zero_metrics)
    for pass_idx in range(1, config.optimizer_passes + 1):
        for label_idx in range(k):
            best_theta_value = float(coordinate_theta[label_idx])
            best_obj = coordinate_current_obj
            best_trace_idx: int | None = None
            for candidate in grids[label_idx]:
                trial_theta = coordinate_theta.copy()
                trial_theta[label_idx] = float(candidate)
                before_len = len(trace)
                trial = evaluate_candidate(
                    trial_theta,
                    "single_block",
                    pass_idx=pass_idx,
                    updated_label_idx=label_idx,
                    beam_rank="coordinate",
                )
                if trial["feasible"]:
                    obj = objective_tuple(trial["metrics"], zero_metrics)
                    if lexicographically_better(obj, best_obj, config.numeric_tolerance):
                        best_theta_value = float(candidate)
                        best_obj = obj
                        best_trace_idx = before_len
            if best_trace_idx is not None:
                trace[best_trace_idx]["selected_for_update"] = True
                coordinate_theta[label_idx] = best_theta_value
                coordinate_current_obj = best_obj


def run_beam_block_search(
    data: LoadedDataset,
    split: SplitData,
    roles: dict[int, dict[str, Any]],
    grids: dict[int, list[float]],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    evaluate_candidate: Any,
) -> None:
    # Beam search retains multiple feasible states through single, pair, and triple blocks.
    beam = top_feasible_beam(feasible_candidates, config)
    for pass_idx in range(1, config.optimizer_passes + 1):
        old_best_key = best_feasible_candidate(feasible_candidates, config)["key"]
        single_order = prioritized_label_order(data, split, beam[0]["eval"]["eval"], roles, config)
        pair_order = prioritized_pair_blocks(single_order, roles, config)
        triple_order = prioritized_triple_blocks(single_order, roles, config)
        beam = run_single_label_blocks(single_order, beam, grids, config, feasible_candidates, trace, evaluate_candidate, pass_idx)
        beam = run_pair_label_blocks(pair_order, beam, grids, config, feasible_candidates, trace, evaluate_candidate, pass_idx)
        beam = run_triple_label_blocks(triple_order, beam, grids, config, feasible_candidates, trace, evaluate_candidate, pass_idx)
        beam = top_feasible_beam(feasible_candidates, config)
        for rank, state in enumerate(beam, start=1):
            evaluate_candidate(state["theta"], "beam", pass_idx=pass_idx, beam_rank=rank)
        new_best = best_feasible_candidate(feasible_candidates, config)
        if new_best["key"] == old_best_key:
            break


def run_single_label_blocks(
    label_order: list[int],
    beam: list[dict[str, Any]],
    grids: dict[int, list[float]],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    evaluate_candidate: Any,
    pass_idx: int,
) -> list[dict[str, Any]]:
    for label_idx in label_order:
        block_candidates: list[dict[str, Any]] = []
        for rank, state in enumerate(beam, start=1):
            for value in single_block_values(float(state["theta"][label_idx]), grids[label_idx], config):
                trial_theta = state["theta"].copy()
                trial_theta[label_idx] = value
                trial = evaluate_candidate(
                    trial_theta,
                    "single_block",
                    pass_idx=pass_idx,
                    updated_label_idx=label_idx,
                    beam_rank=rank,
                )
                if trial["feasible"]:
                    block_candidates.append(feasible_candidates[-1])
        beam = refresh_beam(beam, block_candidates, config)
        mark_current_beam(trace, beam, pass_idx)
    return beam


def run_pair_label_blocks(
    pair_order: list[tuple[int, int]],
    beam: list[dict[str, Any]],
    grids: dict[int, list[float]],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    evaluate_candidate: Any,
    pass_idx: int,
) -> list[dict[str, Any]]:
    for label_idx, label_idx_2 in pair_order:
        block_candidates: list[dict[str, Any]] = []
        for rank, state in enumerate(beam, start=1):
            vals_1 = pair_block_values(float(state["theta"][label_idx]), grids[label_idx], config)
            vals_2 = pair_block_values(float(state["theta"][label_idx_2]), grids[label_idx_2], config)
            for value_1 in vals_1:
                for value_2 in vals_2:
                    trial_theta = state["theta"].copy()
                    trial_theta[label_idx] = value_1
                    trial_theta[label_idx_2] = value_2
                    trial = evaluate_candidate(
                        trial_theta,
                        "pair_block",
                        pass_idx=pass_idx,
                        updated_label_idx=label_idx,
                        updated_label_idx_2=label_idx_2,
                        beam_rank=rank,
                    )
                    if trial["feasible"]:
                        block_candidates.append(feasible_candidates[-1])
        beam = refresh_beam(beam, block_candidates, config)
        mark_current_beam(trace, beam, pass_idx)
    return beam


def run_triple_label_blocks(
    triple_order: list[tuple[int, int, int]],
    beam: list[dict[str, Any]],
    grids: dict[int, list[float]],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    evaluate_candidate: Any,
    pass_idx: int,
) -> list[dict[str, Any]]:
    for label_idx, label_idx_2, label_idx_3 in triple_order:
        block_candidates: list[dict[str, Any]] = []
        for rank, state in enumerate(beam, start=1):
            vals_1 = triple_block_values(float(state["theta"][label_idx]), grids[label_idx], config)
            vals_2 = triple_block_values(float(state["theta"][label_idx_2]), grids[label_idx_2], config)
            vals_3 = triple_block_values(float(state["theta"][label_idx_3]), grids[label_idx_3], config)
            for value_1 in vals_1:
                for value_2 in vals_2:
                    for value_3 in vals_3:
                        trial_theta = state["theta"].copy()
                        trial_theta[label_idx] = value_1
                        trial_theta[label_idx_2] = value_2
                        trial_theta[label_idx_3] = value_3
                        trial = evaluate_candidate(
                            trial_theta,
                            "triple_block",
                            pass_idx=pass_idx,
                            updated_label_idx=label_idx,
                            updated_label_idx_2=label_idx_2,
                            updated_label_idx_3=label_idx_3,
                            beam_rank=rank,
                        )
                        if trial["feasible"]:
                            block_candidates.append(feasible_candidates[-1])
        beam = refresh_beam(beam, block_candidates, config)
        mark_current_beam(trace, beam, pass_idx)
    return beam


def run_quasirandom_search(
    grids: dict[int, list[float]],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
    score_method: str,
    evaluate_candidate: Any,
) -> list[dict[str, Any]]:
    # Fixed-seed Halton-style candidates provide deterministic restart coverage.
    infeasible_sources: list[dict[str, Any]] = []
    for candidate_idx, theta in enumerate(
        quasirandom_theta_candidates(grids, roles, config, score_method),
        start=1,
    ):
        trial = evaluate_candidate(theta, "quasirandom", pass_idx=candidate_idx)
        if not trial["feasible"]:
            infeasible_sources.append({"theta": np.asarray(theta, dtype=float).copy(), "eval": trial})
    return infeasible_sources


def trace_row(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    phase: str,
    pass_idx: int,
    updated_label_idx: int | None,
    updated_label_idx_2: int | None,
    updated_label_idx_3: int | None,
    beam_rank: int | str,
    theta: np.ndarray,
    trial: dict[str, Any],
    selected_for_update: bool,
) -> dict[str, Any]:
    reasons = ";".join(trial["rejection_reasons"])
    metrics = trial["metrics"]
    return {
        "score_method": score_method,
        "optimizer_phase": phase,
        "beam_rank": beam_rank,
        "pass": pass_idx,
        "updated_label": data.cell_values[updated_label_idx] if updated_label_idx is not None else "",
        "updated_label_2": data.cell_values[updated_label_idx_2] if updated_label_idx_2 is not None else "",
        "updated_label_3": data.cell_values[updated_label_idx_3] if updated_label_idx_3 is not None else "",
        "updated_cell": data.cell_values[updated_label_idx] if updated_label_idx is not None else "",
        "updated_cell_2": data.cell_values[updated_label_idx_2] if updated_label_idx_2 is not None else "",
        "updated_cell_3": data.cell_values[updated_label_idx_3] if updated_label_idx_3 is not None else "",
        "candidate_theta": theta_json(data, theta),
        "feasible": bool(trial["feasible"]),
        "rejection_reason": reasons if not trial["feasible"] else "",
        "max_local_gap": metrics["max_local_gap"],
        "mean_local_gap": metrics["mean_local_gap"],
        "total_local_gap_reduction": metrics["total_local_gap_reduction"],
        "max_local_gap_increase": metrics["max_local_gap_increase"],
        "any_local_gap_increased": metrics["any_local_gap_increased"],
        "max_advantaged_harm": metrics["max_advantaged_harm"],
        "max_disadvantaged_loss": metrics["max_disadvantaged_loss"],
        "any_disadvantaged_cell_worsened": metrics["any_disadvantaged_cell_worsened"],
        "avg_set_size_cost": metrics["avg_set_size_cost"],
        "shift_norm": metrics["shift_norm"],
        "selected_for_update": bool(selected_for_update),
        "selected_final": False,
        "theta_key": theta_key(theta, config),
    }


def theta_json(data: LoadedDataset, theta: np.ndarray) -> str:
    payload = {
        data.cell_values[i]: round(float(v), 12)
        for i, v in enumerate(np.asarray(theta, dtype=float))
        if abs(float(v)) > 1e-15
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) if payload else "{}"


def theta_key(theta: np.ndarray, config: ScoreFixConfig) -> tuple[float, ...]:
    return tuple(round(float(max(0.0, v)), int(config.beam_round_decimals)) for v in np.asarray(theta, dtype=float))


def global_repair_direction(
    grids: dict[int, list[float]],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> np.ndarray:
    direction = np.zeros(len(roles), dtype=float)
    repairable_gaps = [
        float(role["baseline_gap"])
        for role in roles.values()
        if role["repairable"] and np.isfinite(role["baseline_gap"])
    ]
    max_gap = max(repairable_gaps) if repairable_gaps else 0.0
    for label_idx, role in roles.items():
        if not role["repairable"]:
            continue
        scale = theta_scale_from_grid(grids[label_idx], config)
        gap_pressure = float(role["baseline_gap"]) / max_gap if max_gap > config.numeric_tolerance else 1.0
        direction[label_idx] = scale * max(gap_pressure, 0.05)
    return direction


def theta_scale_from_grid(grid: list[float], config: ScoreFixConfig) -> float:
    positives = [float(v) for v in grid if float(v) > config.numeric_tolerance]
    if not positives:
        return 0.05
    max_multiplier = max(float(v) for v in config.grid_multipliers if float(v) > 0.0)
    if max_multiplier <= 0:
        return max(positives)
    scale = max(positives) / max_multiplier
    return scale if np.isfinite(scale) and scale > config.numeric_tolerance else max(positives)


def projection_sources(
    infeasible_global: list[dict[str, Any]],
    current_best: dict[str, Any],
    config: ScoreFixConfig,
    zero_metrics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source in infeasible_global:
        reasons = set(source["eval"]["rejection_reasons"])
        if not reasons or "group_coverage_not_fixed" in reasons:
            continue
        metrics = source["eval"]["metrics"]
        if (
            metrics["total_local_gap_reduction"] >= current_best["eval"]["metrics"]["total_local_gap_reduction"] - config.numeric_tolerance
            or metrics["max_local_gap"] <= current_best["eval"]["metrics"]["max_local_gap"] + 0.02
        ):
            candidates.append(source)
    candidates.sort(key=lambda c: objective_tuple(c["eval"]["metrics"], zero_metrics))
    return candidates[: int(config.boundary_projection_max_candidates)]


def best_feasible_candidate(candidates: list[dict[str, Any]], config: ScoreFixConfig) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No feasible score-fix candidate was recorded; theta=0 should always be feasible.")
    return min(candidates, key=lambda c: c["obj"])


def top_feasible_beam(candidates: list[dict[str, Any]], config: ScoreFixConfig) -> list[dict[str, Any]]:
    dedup: dict[tuple[float, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate["key"]
        incumbent = dedup.get(key)
        if incumbent is None or lexicographically_better(candidate["obj"], incumbent["obj"], config.numeric_tolerance):
            dedup[key] = candidate
    return sorted(dedup.values(), key=lambda c: c["obj"])[: int(config.beam_width)]


def refresh_beam(
    current_beam: list[dict[str, Any]],
    block_candidates: list[dict[str, Any]],
    config: ScoreFixConfig,
) -> list[dict[str, Any]]:
    return top_feasible_beam(current_beam + block_candidates, config)


def mark_current_beam(trace: list[dict[str, Any]], beam: list[dict[str, Any]], pass_idx: int) -> None:
    if not beam:
        return
    best = beam[0]
    idx = best.get("trace_idx")
    if idx is not None and 0 <= int(idx) < len(trace):
        trace[int(idx)]["selected_for_update"] = True


def prioritized_label_order(
    data: LoadedDataset,
    split: SplitData,
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> list[int]:
    rows = label_gaps(split, data.group_values, repaired_eval, repaired_eval, roles, config)
    repaired_gap_by_label = {int(row["label_idx"]): float(row["repaired_gap"]) for row in rows if row["repairable"]}
    labels = [label_idx for label_idx, role in roles.items() if role["repairable"]]
    labels.sort(
        key=lambda label_idx: (
            -repaired_gap_by_label.get(label_idx, 0.0),
            -float(roles[label_idx]["baseline_gap"]),
            label_idx,
        )
    )
    return labels


def prioritized_pair_blocks(
    label_order: list[int],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> list[tuple[int, int]]:
    top = label_order[: min(len(label_order), 8)]
    pairs: list[tuple[int, int]] = []
    for i, left in enumerate(top):
        for right in top[i + 1 :]:
            pairs.append((left, right))
    pairs.sort(
        key=lambda pair: (
            -max(float(roles[pair[0]]["baseline_gap"]), float(roles[pair[1]]["baseline_gap"])),
            -float(roles[pair[0]]["baseline_gap"] + roles[pair[1]]["baseline_gap"]),
            pair[0],
            pair[1],
        )
    )
    return pairs[: int(config.pair_block_limit)]


def prioritized_triple_blocks(
    label_order: list[int],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> list[tuple[int, int, int]]:
    top = label_order[: min(len(label_order), 6)]
    triples: list[tuple[int, int, int]] = []
    for i, left in enumerate(top):
        for j, middle in enumerate(top[i + 1 :], start=i + 1):
            for right in top[j + 1 :]:
                triples.append((left, middle, right))
    triples.sort(
        key=lambda triple: (
            -max(float(roles[idx]["baseline_gap"]) for idx in triple),
            -sum(float(roles[idx]["baseline_gap"]) for idx in triple),
            triple[0],
            triple[1],
            triple[2],
        )
    )
    return triples[: int(config.triple_block_limit)]


def critical_block_values(
    label_idx: int,
    current: float,
    grid: list[float],
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    current_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> list[float]:
    role = roles[label_idx]
    if not role["repairable"]:
        return [0.0]
    scale = theta_scale_from_grid(grid, config)
    values = {0.0, round(float(current), 12)}
    values.update(round(float(v), 12) for v in grid)
    for multiplier in (-0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2):
        values.add(round(max(0.0, float(current) + float(multiplier) * scale), 12))
    for multiplier in (1e-4, 1e-3, 0.005, 0.01, 0.025, 0.05, 0.1):
        values.add(round(float(multiplier) * scale, 12))

    ts = true_scores(base_scores, split.y)
    cell_mask = split.z == data.cell_values[label_idx]
    adv_scores = ts[(split.g == role["advantaged_group"]) & cell_mask]
    dis_scores = ts[(split.g == role["disadvantaged_group"]) & cell_mask]
    if len(adv_scores) and len(dis_scores):
        for q in (0.5, 0.75, 1.0 - float(config.alpha), 0.95):
            adv_q = float(np.quantile(adv_scores, q))
            dis_q = float(np.quantile(dis_scores, q))
            align = 0.5 * (dis_q - adv_q)
            if np.isfinite(align) and align > 0:
                values.add(round(align, 12))

    thresholds = current_eval.get("thresholds", {})
    for group, sign in ((role["advantaged_group"], 1.0), (role["disadvantaged_group"], -1.0)):
        threshold = thresholds.get(group)
        if threshold is None or not np.isfinite(float(threshold)):
            continue
        vals = ts[(split.g == group) & cell_mask]
        if not len(vals):
            continue
        if sign > 0:
            margins = float(threshold) - vals
        else:
            margins = vals - float(threshold)
        margins = margins[np.isfinite(margins) & (margins > config.numeric_tolerance)]
        if len(margins):
            for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                values.add(round(float(np.quantile(margins, q)), 12))
    return sorted(v for v in values if np.isfinite(v) and v >= 0.0)


def single_block_values(current: float, grid: list[float], config: ScoreFixConfig) -> list[float]:
    scale = theta_scale_from_grid(grid, config)
    values = {0.0, float(current)}
    for multiplier in config.local_grid_multipliers:
        values.add(max(0.0, float(current) + float(multiplier) * scale))
    return sorted(round(float(v), 12) for v in values)


def pair_block_values(current: float, grid: list[float], config: ScoreFixConfig) -> list[float]:
    scale = theta_scale_from_grid(grid, config)
    values = {float(current)}
    for multiplier in config.pair_grid_multipliers:
        values.add(max(0.0, float(current) + float(multiplier) * scale))
    return sorted(round(float(v), 12) for v in values)


def triple_block_values(current: float, grid: list[float], config: ScoreFixConfig) -> list[float]:
    scale = theta_scale_from_grid(grid, config)
    values = {float(current)}
    for multiplier in config.triple_grid_multipliers:
        values.add(max(0.0, float(current) + float(multiplier) * scale))
    return sorted(round(float(v), 12) for v in values)


def quasirandom_theta_candidates(
    grids: dict[int, list[float]],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
    score_method: str,
) -> list[np.ndarray]:
    repairable = [label_idx for label_idx, role in roles.items() if role["repairable"]]
    k = len(roles)
    if not repairable:
        return []
    scales = np.zeros(k, dtype=float)
    gap_pressure = np.zeros(k, dtype=float)
    max_gap = max(float(roles[label_idx]["baseline_gap"]) for label_idx in repairable)
    for label_idx in repairable:
        scales[label_idx] = theta_scale_from_grid(grids[label_idx], config)
        gap_pressure[label_idx] = max(0.05, float(roles[label_idx]["baseline_gap"]) / max_gap) if max_gap > config.numeric_tolerance else 1.0
    ordered = sorted(repairable, key=lambda idx: (-float(roles[idx]["baseline_gap"]), idx))
    method_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(score_method))
    start = int(config.quasirandom_seed + method_offset)
    primes = first_primes(len(repairable))
    candidates: list[np.ndarray] = []
    for cand_idx in range(int(config.quasirandom_candidate_count)):
        u = np.asarray(
            [halton_value(start + cand_idx + 1, base) for base in primes],
            dtype=float,
        )
        theta = np.zeros(k, dtype=float)
        mode = cand_idx % 4
        if mode == 0:
            for pos, label_idx in enumerate(repairable):
                theta[label_idx] = scales[label_idx] * gap_pressure[label_idx] * (0.05 + 0.70 * u[pos])
        elif mode == 1:
            for pos, label_idx in enumerate(repairable):
                theta[label_idx] = scales[label_idx] * (0.10 + 2.40 * u[pos]) * gap_pressure[label_idx]
        elif mode == 2:
            active_count = 1 + (cand_idx % min(4, len(ordered)))
            active = ordered[:active_count]
            for label_idx in active:
                pos = repairable.index(label_idx)
                theta[label_idx] = scales[label_idx] * (0.05 + 2.95 * u[pos]) * gap_pressure[label_idx]
        else:
            for pos, label_idx in enumerate(repairable):
                sparse_gate = u[pos] > 0.45 or label_idx in ordered[:3]
                if sparse_gate:
                    theta[label_idx] = scales[label_idx] * (0.02 + 1.80 * u[pos]) * (0.25 + 0.75 * gap_pressure[label_idx])
        candidates.append(np.round(theta, 12))
    return candidates


def first_primes(n: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
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


def halton_value(index: int, base: int) -> float:
    result = 0.0
    f = 1.0 / float(base)
    i = int(index)
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= float(base)
    return result


def optimizer_meta(
    trace: list[dict[str, Any]],
    final_candidate: dict[str, Any],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    zero_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    rejected = [row for row in trace if not bool(row["feasible"])]

    def count_reason(reason: str) -> int:
        return int(sum(reason in str(row.get("rejection_reason", "")).split(";") for row in rejected))

    theta = final_candidate["theta"]
    final_is_zero = bool(np.linalg.norm(theta, ord=2) <= config.numeric_tolerance)
    tie_break_preferred_zero = False
    if final_is_zero and zero_metrics is not None:
        tie_break_preferred_zero = any(
            np.linalg.norm(candidate["theta"], ord=2) > config.numeric_tolerance
            and gap_metrics_unchanged(candidate["eval"]["metrics"], zero_metrics, config.no_gap_improvement_tol)
            for candidate in feasible_candidates
        )
    candidate_count_total = int(len(trace))
    candidate_count_feasible = int(sum(bool(row["feasible"]) for row in trace))
    return {
        "optimizer_name": config.optimizer_name,
        "best_candidate_phase": final_candidate["phase"],
        "best_candidate_was_zero_theta": final_is_zero,
        "candidate_count_total": candidate_count_total,
        "candidate_count_feasible": candidate_count_feasible,
        "candidate_count_infeasible": int(candidate_count_total - candidate_count_feasible),
        "candidate_count_projected": int(
            sum(row["optimizer_phase"] in {"boundary_projection", "quasirandom_projection"} for row in trace)
        ),
        "candidate_count_quasirandom": int(sum(row["optimizer_phase"] == "quasirandom" for row in trace)),
        "candidate_count_triple_block": int(sum(row["optimizer_phase"] == "triple_block" for row in trace)),
        "rejected_candidate_count": int(len(rejected)),
        "rejected_by_advantaged_harm_count": count_reason("advantaged_harm_exceeds_epsilon"),
        "rejected_by_local_gap_worsening_count": count_reason("local_gap_worsened"),
        "rejected_by_disadvantaged_worsening_count": count_reason("disadvantaged_cell_worsened"),
        "rejected_by_group_coverage_count": count_reason("group_coverage_not_fixed"),
        "selected_from_phase": final_candidate["phase"],
        "tie_break_preferred_zero_due_to_no_gap_improvement": bool(tie_break_preferred_zero),
    }


def candidate_metrics(
    data: LoadedDataset,
    config: ScoreFixConfig,
    split: SplitData,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    theta: np.ndarray,
) -> dict[str, float]:
    gaps = label_gaps(split, data.group_values, baseline_eval, repaired_eval, roles, config)
    repairable = [r for r in gaps if r["repairable"]]
    repaired_gaps = [r["repaired_gap"] for r in repairable if np.isfinite(r["repaired_gap"])]
    gap_reductions = [r["gap_reduction"] for r in repairable if np.isfinite(r["gap_reduction"])]
    gap_increases = [max(0.0, float(r["local_gap_increase"])) for r in repairable if np.isfinite(r["local_gap_increase"])]
    harms = [float(r["advantaged_loss"]) for r in repairable if np.isfinite(r["advantaged_loss"])]
    dis_losses = [
        -float(r["disadvantaged_coverage_change"])
        for r in repairable
        if np.isfinite(r["disadvantaged_coverage_change"])
    ]
    return {
        "max_local_gap": float(np.nanmax(repaired_gaps)) if repaired_gaps else 0.0,
        "mean_local_gap": float(np.nanmean(repaired_gaps)) if repaired_gaps else 0.0,
        "total_local_gap_reduction": float(np.nansum(gap_reductions)) if gap_reductions else 0.0,
        "max_local_gap_increase": float(max(gap_increases)) if gap_increases else 0.0,
        "any_local_gap_increased": bool(any(r["local_gap_increased"] for r in repairable)),
        "max_advantaged_harm": float(max([0.0] + harms)) if harms else 0.0,
        "max_disadvantaged_loss": float(max([0.0] + dis_losses)) if dis_losses else 0.0,
        "any_disadvantaged_cell_worsened": bool(any(not r["disadvantaged_nonharm_satisfied"] for r in repairable)),
        "avg_set_size_cost": float(np.mean(repaired_eval["set_sizes"]) - np.mean(baseline_eval["set_sizes"])),
        "shift_norm": float(np.linalg.norm(theta, ord=2)),
    }


def build_metric_context(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    config: ScoreFixConfig,
    baseline_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    cell_indices: dict[tuple[str, int], np.ndarray] = {}
    baseline_cov: dict[tuple[str, int], float] = {}
    n_by_cell: dict[tuple[str, int], int] = {}
    row_cell_indices = repair_cell_indices(data, split)
    for group in data.group_values:
        group_mask = split.g == group
        for cell_idx, cell_value in enumerate(data.cell_values):
            idx = np.flatnonzero(group_mask & (split.z == cell_value))
            cell_indices[(group, cell_idx)] = idx
            n_by_cell[(group, cell_idx)] = int(len(idx))
            baseline_cov[(group, cell_idx)] = float(np.mean(baseline_eval["covered"][idx])) if len(idx) else float("nan")
    repairable_labels = [
        cell_idx
        for cell_idx in range(len(data.cell_values))
        if roles[cell_idx]["repairable"]
    ]
    advantaged_cells = [
        (role["advantaged_group"], label_idx)
        for label_idx, role in roles.items()
        if role["repairable"]
    ]
    disadvantaged_cells = [
        (role["disadvantaged_group"], label_idx)
        for label_idx, role in roles.items()
        if role["repairable"]
    ]
    baseline_gap = {}
    for label_idx in repairable_labels:
        b0 = baseline_cov[(data.group_values[0], label_idx)]
        b1 = baseline_cov[(data.group_values[1], label_idx)]
        baseline_gap[label_idx] = abs(b0 - b1)
    if data.intervention_mode == "label_defined_z":
        candidate_shift_signs = np.zeros((len(split.y), len(data.label_values)), dtype=float)
        for cell_idx, role in roles.items():
            if not role["repairable"]:
                continue
            candidate_shift_signs[split.g == role["advantaged_group"], cell_idx] = 1.0
            candidate_shift_signs[split.g == role["disadvantaged_group"], cell_idx] = -1.0
        true_shift_signs = candidate_shift_signs[np.arange(len(split.y)), np.asarray(split.y, dtype=int)]
    elif data.intervention_mode == "covariate_defined_z":
        candidate_shift_signs = np.zeros(len(split.y), dtype=float)
        for cell_idx, role in roles.items():
            if not role["repairable"]:
                continue
            cell_mask = row_cell_indices == cell_idx
            candidate_shift_signs[cell_mask & (split.g == role["advantaged_group"])] = 1.0
            candidate_shift_signs[cell_mask & (split.g == role["disadvantaged_group"])] = -1.0
        true_shift_signs = candidate_shift_signs.copy()
    else:
        raise ValueError(f"Unsupported intervention_mode: {data.intervention_mode}")
    return {
        "groups": data.group_values,
        "cell_indices": cell_indices,
        "baseline_cov": baseline_cov,
        "baseline_gap": baseline_gap,
        "n_by_cell": n_by_cell,
        "repairable_labels": repairable_labels,
        "advantaged_cells": advantaged_cells,
        "disadvantaged_cells": disadvantaged_cells,
        "baseline_avg_set_size": float(np.mean(baseline_eval["set_sizes"])),
        "base_true_scores": true_scores(base_scores, split.y),
        "candidate_shift_signs": candidate_shift_signs,
        "true_shift_signs": true_shift_signs,
        "row_cell_indices": row_cell_indices,
    }


def repair_cell_indices(data: LoadedDataset, split: SplitData) -> np.ndarray:
    mapping = {value: idx for idx, value in enumerate(data.cell_values)}
    try:
        return np.asarray([mapping[str(value)] for value in split.z], dtype=int)
    except KeyError as exc:
        raise AssertionError(f"{data.dataset}: split contains Z outside the frozen cell order: {exc}") from exc


def candidate_metrics_fast(
    config: ScoreFixConfig,
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    theta: np.ndarray,
    metric_context: dict[str, Any],
) -> dict[str, float]:
    groups = metric_context["groups"]
    cell_indices = metric_context["cell_indices"]
    repaired_cov: dict[tuple[str, int], float] = {}
    gaps: list[float] = []
    gap_reductions: list[float] = []
    gap_increases: list[float] = []
    for label_idx in metric_context["repairable_labels"]:
        covs: list[float] = []
        for group in groups:
            idx = cell_indices[(group, label_idx)]
            cov = float(np.mean(repaired_eval["covered"][idx])) if len(idx) else float("nan")
            repaired_cov[(group, label_idx)] = cov
            covs.append(cov)
        repaired_gap = abs(covs[0] - covs[1])
        baseline_gap = float(metric_context["baseline_gap"][label_idx])
        gaps.append(repaired_gap)
        gap_reductions.append(baseline_gap - repaired_gap)
        gap_increases.append(max(0.0, repaired_gap - baseline_gap))
    harms: list[float] = []
    for group, label_idx in metric_context["advantaged_cells"]:
        idx = cell_indices[(group, label_idx)]
        cov = repaired_cov.get((group, label_idx))
        if cov is None:
            cov = float(np.mean(repaired_eval["covered"][idx])) if len(idx) else float("nan")
        harms.append(float(metric_context["baseline_cov"][(group, label_idx)] - cov))
    dis_losses: list[float] = []
    for group, label_idx in metric_context["disadvantaged_cells"]:
        idx = cell_indices[(group, label_idx)]
        cov = repaired_cov.get((group, label_idx))
        if cov is None:
            cov = float(np.mean(repaired_eval["covered"][idx])) if len(idx) else float("nan")
        dis_losses.append(float(metric_context["baseline_cov"][(group, label_idx)] - cov))
    max_gap_increase = float(max(gap_increases)) if gap_increases else 0.0
    max_dis_loss = float(max([0.0] + dis_losses)) if dis_losses else 0.0
    return {
        "max_local_gap": float(np.nanmax(gaps)) if gaps else 0.0,
        "mean_local_gap": float(np.nanmean(gaps)) if gaps else 0.0,
        "total_local_gap_reduction": float(np.nansum(gap_reductions)) if gap_reductions else 0.0,
        "max_local_gap_increase": max_gap_increase,
        "any_local_gap_increased": bool(max_gap_increase > config.gap_tol),
        "max_advantaged_harm": float(max([0.0] + harms)) if harms else 0.0,
        "max_disadvantaged_loss": max_dis_loss,
        "any_disadvantaged_cell_worsened": bool(max_dis_loss > config.coverage_tol),
        "avg_set_size_cost": float(np.mean(repaired_eval["set_sizes"]) - metric_context["baseline_avg_set_size"]),
        "shift_norm": float(np.linalg.norm(theta, ord=2)),
    }


def candidate_rejection_reasons(group_coverage_fixed: bool, metrics: dict[str, Any], config: ScoreFixConfig) -> list[str]:
    # Reject only when one of the explicit no-free-repair constraints is violated.
    reasons: list[str] = []
    if not group_coverage_fixed:
        reasons.append("group_coverage_not_fixed")
    # Epsilon is a cell-wise cap on each initially advantaged cell's coverage loss.
    if float(metrics["max_advantaged_harm"]) > config.epsilon + config.coverage_tol:
        reasons.append("advantaged_harm_exceeds_epsilon")
    # Repairable local G x Z gaps may shrink or stay fixed, but not increase.
    if bool(metrics["any_local_gap_increased"]):
        reasons.append("local_gap_worsened")
    # The initially disadvantaged side of each repairable Z cell must not lose coverage.
    if bool(metrics["any_disadvantaged_cell_worsened"]):
        reasons.append("disadvantaged_cell_worsened")
    return reasons


def objective_tuple(
    metrics: dict[str, float],
    zero_metrics: dict[str, Any] | None = None,
) -> tuple[float, float, float, float, float]:
    # Set size is intentionally the last tie-breaker after all coverage-gap objectives.
    max_gap = float(metrics["max_local_gap"])
    mean_gap = float(metrics["mean_local_gap"])
    total_reduction = float(metrics["total_local_gap_reduction"])
    if zero_metrics is not None and gap_metrics_unchanged(metrics, zero_metrics, 1e-12):
        max_gap = float(zero_metrics["max_local_gap"])
        mean_gap = float(zero_metrics["mean_local_gap"])
        total_reduction = float(zero_metrics["total_local_gap_reduction"])
    return (
        max_gap,
        mean_gap,
        -total_reduction,
        float(metrics["shift_norm"]),
        float(metrics["avg_set_size_cost"]),
    )


def gap_metrics_unchanged(metrics: dict[str, Any], zero_metrics: dict[str, Any], tol: float) -> bool:
    return (
        abs(float(metrics["max_local_gap"]) - float(zero_metrics["max_local_gap"])) <= float(tol)
        and abs(float(metrics["mean_local_gap"]) - float(zero_metrics["mean_local_gap"])) <= float(tol)
        and abs(float(metrics["total_local_gap_reduction"]) - float(zero_metrics["total_local_gap_reduction"])) <= float(tol)
    )


def lexicographically_better(candidate: tuple[float, ...], incumbent: tuple[float, ...], tol: float) -> bool:
    for cand, inc in zip(candidate, incumbent):
        if cand < inc - tol:
            return True
        if cand > inc + tol:
            return False
    return False


def cell_stats(
    split: SplitData,
    eval_result: dict[str, Any],
    cell_values: list[str],
    groups: list[str],
) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for group in groups:
        for cell_idx, cell_value in enumerate(cell_values):
            mask = (split.g == group) & (split.z == cell_value)
            n = int(mask.sum())
            if n:
                coverage = float(np.mean(eval_result["covered"][mask]))
                set_size = float(np.mean(eval_result["set_sizes"][mask]))
                covered_count = int(eval_result["covered"][mask].sum())
            else:
                coverage = float("nan")
                set_size = float("nan")
                covered_count = 0
            out[(group, cell_idx)] = {
                "n": n,
                "coverage": coverage,
                "avg_set_size": set_size,
                "covered_count": covered_count,
            }
    return out


def label_gaps(
    split: SplitData,
    groups: list[str],
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> list[dict[str, Any]]:
    n_cells = len(roles)
    cell_values = [str(roles[cell_idx]["cell_value"]) for cell_idx in range(n_cells)]
    base_cell = cell_stats(split, baseline_eval, cell_values, groups)
    rep_cell = cell_stats(split, repaired_eval, cell_values, groups)
    rows: list[dict[str, Any]] = []
    for label_idx in range(n_cells):
        role = roles[label_idx]
        n0 = base_cell[(groups[0], label_idx)]["n"]
        n1 = base_cell[(groups[1], label_idx)]["n"]
        supported = n0 >= config.min_cell_count and n1 >= config.min_cell_count
        b0 = base_cell[(groups[0], label_idx)]["coverage"]
        b1 = base_cell[(groups[1], label_idx)]["coverage"]
        r0 = rep_cell[(groups[0], label_idx)]["coverage"]
        r1 = rep_cell[(groups[1], label_idx)]["coverage"]
        baseline_gap = abs(b0 - b1) if supported else float("nan")
        repaired_gap = abs(r0 - r1) if supported else float("nan")
        repairable = bool(role["repairable"])
        adv = role["advantaged_group"]
        dis = role["disadvantaged_group"]
        adv_loss = cell_loss(base_cell, rep_cell, adv, label_idx) if repairable else 0.0
        dis_change = cell_gain(base_cell, rep_cell, dis, label_idx) if repairable else 0.0
        gap_reduction = baseline_gap - repaired_gap if repairable and np.isfinite(baseline_gap) and np.isfinite(repaired_gap) else float("nan")
        local_gap_increase = repaired_gap - baseline_gap if repairable and np.isfinite(baseline_gap) and np.isfinite(repaired_gap) else 0.0
        local_gap_increased = bool(repairable and local_gap_increase > config.gap_tol)
        disadvantaged_nonharm = bool((not repairable) or dis_change >= -config.coverage_tol)
        reasons = label_feasibility_reasons(adv_loss, local_gap_increased, disadvantaged_nonharm, repairable, config)
        rows.append(
            {
                "label_idx": label_idx,
                "supported": supported,
                "repairable": repairable,
                "baseline_gap": baseline_gap,
                "repaired_gap": repaired_gap,
                "gap_reduction": gap_reduction,
                "local_gap_increase": max(0.0, float(local_gap_increase)) if np.isfinite(local_gap_increase) else float("nan"),
                "local_gap_increased": local_gap_increased,
                "advantaged_group": adv,
                "disadvantaged_group": dis,
                "advantaged_loss": adv_loss,
                "disadvantaged_coverage_change": dis_change,
                "disadvantaged_gain": dis_change,
                "disadvantaged_nonharm_satisfied": disadvantaged_nonharm,
                "feasibility_violation_reason": ";".join(reasons),
            }
        )
    return rows


def label_feasibility_reasons(
    adv_loss: float,
    local_gap_increased: bool,
    disadvantaged_nonharm: bool,
    repairable: bool,
    config: ScoreFixConfig,
) -> list[str]:
    if not repairable:
        return []
    reasons: list[str] = []
    if np.isfinite(adv_loss) and adv_loss > config.epsilon + config.coverage_tol:
        reasons.append("advantaged_harm_exceeds_epsilon")
    if local_gap_increased:
        reasons.append("local_gap_worsened")
    if not disadvantaged_nonharm:
        reasons.append("disadvantaged_cell_worsened")
    return reasons


def max_advantaged_harm_detail(
    data: LoadedDataset,
    config: ScoreFixConfig,
    split_name: str,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    gaps = label_gaps(data.splits[split_name], data.group_values, baseline_eval, repaired_eval, roles, config)
    repairable = [r for r in gaps if r["repairable"] and np.isfinite(r["advantaged_loss"])]
    if not repairable:
        return {"value": 0.0, "group": "", "label": "", "label_name": ""}
    row = max(repairable, key=lambda r: float(r["advantaged_loss"]))
    label = data.cell_values[int(row["label_idx"])]
    return {
        "value": max(0.0, float(row["advantaged_loss"])),
        "group": row["advantaged_group"],
        "label": label,
        "label_name": data.cell_names.get(label, label),
    }


def gap_constraint_summary(gaps: list[dict[str, Any]], config: ScoreFixConfig) -> dict[str, Any]:
    repairable = [r for r in gaps if r["repairable"]]
    max_gap_increase = max([0.0] + [float(r["local_gap_increase"]) for r in repairable if np.isfinite(r["local_gap_increase"])])
    any_gap_increased = bool(any(r["local_gap_increased"] for r in repairable))
    any_dis_worsened = bool(any(not r["disadvantaged_nonharm_satisfied"] for r in repairable))
    return {
        "any_local_gap_increased": any_gap_increased,
        "max_local_gap_increase": max_gap_increase,
        "all_repairable_gaps_nonincreased": bool(not any_gap_increased),
        "any_disadvantaged_cell_worsened": any_dis_worsened,
    }


def coverage_allocation_oracle(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    split: SplitData,
    baseline_eval: dict[str, Any],
    target_counts: dict[str, int],
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    # Solve an optimistic continuous LP over cell coverage rates under the same constraints.
    repairable_labels = [label_idx for label_idx, role in roles.items() if role["repairable"]]
    baseline_gaps = [float(roles[label_idx]["baseline_gap"]) for label_idx in repairable_labels]
    baseline_total_gap = float(np.nansum(baseline_gaps)) if baseline_gaps else 0.0
    if not repairable_labels:
        return {
            "oracle_status": "no_repairable_labels",
            "oracle_best_max_local_gap_repair": 0.0,
            "oracle_best_mean_local_gap_repair": 0.0,
            "oracle_total_gap_reduction_repair": 0.0,
            "oracle_message": "No repairable G x Z cells under min_cell_count and nonzero-gap rules.",
        }
    try:
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - depends on local environment.
        return {
            "oracle_status": "unavailable",
            "oracle_best_max_local_gap_repair": float("nan"),
            "oracle_best_mean_local_gap_repair": float("nan"),
            "oracle_total_gap_reduction_repair": float("nan"),
            "oracle_message": f"scipy.optimize.linprog unavailable: {exc}",
        }

    groups = data.group_values
    k = len(data.cell_values)
    base_cell = cell_stats(split, baseline_eval, data.cell_values, groups)
    repairable_set = set(repairable_labels)

    def x_idx(group_idx: int, label_idx: int) -> int:
        return group_idx * k + label_idx

    def build_problem(include_abs_vars: bool, t_limit: float | None = None) -> tuple[np.ndarray, list[tuple[float | None, float | None]], list[list[float]], list[float], list[list[float]], list[float], int, int | None]:
        n_x = len(groups) * k
        t_idx = n_x
        u_start = n_x + 1 if include_abs_vars else None
        n_vars = n_x + 1 + (len(repairable_labels) if include_abs_vars else 0)
        c = np.zeros(n_vars, dtype=float)
        if include_abs_vars:
            assert u_start is not None
            for pos in range(len(repairable_labels)):
                c[u_start + pos] = 1.0
        else:
            c[t_idx] = 1.0

        bounds: list[tuple[float | None, float | None]] = [(0.0, 1.0) for _ in range(n_x)] + [(0.0, None)]
        if include_abs_vars:
            bounds.extend((0.0, None) for _ in repairable_labels)
        for label_idx in range(k):
            if label_idx in repairable_set:
                continue
            for group_idx, group in enumerate(groups):
                cov = base_cell[(group, label_idx)]["coverage"]
                if np.isfinite(cov):
                    clipped = min(1.0, max(0.0, float(cov)))
                    bounds[x_idx(group_idx, label_idx)] = (clipped, clipped)

        a_eq: list[list[float]] = []
        b_eq: list[float] = []
        # The oracle preserves each group's baseline covered count in aggregate.
        for group_idx, group in enumerate(groups):
            row = [0.0] * n_vars
            for label_idx in range(k):
                row[x_idx(group_idx, label_idx)] = float(base_cell[(group, label_idx)]["n"])
            a_eq.append(row)
            b_eq.append(float(target_counts[group]))

        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        for pos, label_idx in enumerate(repairable_labels):
            role = roles[label_idx]
            baseline_gap = float(role["baseline_gap"])
            g0_idx, g1_idx = 0, 1
            x0 = x_idx(g0_idx, label_idx)
            x1 = x_idx(g1_idx, label_idx)

            row = [0.0] * n_vars
            row[x0] = 1.0
            row[x1] = -1.0
            row[t_idx] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)

            row = [0.0] * n_vars
            row[x0] = -1.0
            row[x1] = 1.0
            row[t_idx] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)

            row = [0.0] * n_vars
            row[x0] = 1.0
            row[x1] = -1.0
            a_ub.append(row)
            b_ub.append(baseline_gap)

            row = [0.0] * n_vars
            row[x0] = -1.0
            row[x1] = 1.0
            a_ub.append(row)
            b_ub.append(baseline_gap)

            adv_idx = groups.index(role["advantaged_group"])
            dis_idx = groups.index(role["disadvantaged_group"])
            adv_x = x_idx(adv_idx, label_idx)
            dis_x = x_idx(dis_idx, label_idx)
            adv_base = float(base_cell[(role["advantaged_group"], label_idx)]["coverage"])
            dis_base = float(base_cell[(role["disadvantaged_group"], label_idx)]["coverage"])

            # Advantaged cells may lose at most epsilon coverage.
            row = [0.0] * n_vars
            row[adv_x] = -1.0
            a_ub.append(row)
            b_ub.append(-(adv_base - config.epsilon))

            # Disadvantaged cells are constrained to non-decreasing coverage.
            row = [0.0] * n_vars
            row[dis_x] = -1.0
            a_ub.append(row)
            b_ub.append(-dis_base)

            if include_abs_vars:
                assert u_start is not None
                u = u_start + pos
                row = [0.0] * n_vars
                row[x0] = 1.0
                row[x1] = -1.0
                row[u] = -1.0
                a_ub.append(row)
                b_ub.append(0.0)

                row = [0.0] * n_vars
                row[x0] = -1.0
                row[x1] = 1.0
                row[u] = -1.0
                a_ub.append(row)
                b_ub.append(0.0)

        if t_limit is not None:
            row = [0.0] * n_vars
            row[t_idx] = 1.0
            a_ub.append(row)
            b_ub.append(float(t_limit))
        return c, bounds, a_eq, b_eq, a_ub, b_ub, t_idx, u_start

    c1, bounds1, aeq1, beq1, aub1, bub1, t_idx1, _ = build_problem(include_abs_vars=False)
    res1 = linprog(
        c1,
        A_ub=np.asarray(aub1, dtype=float) if aub1 else None,
        b_ub=np.asarray(bub1, dtype=float) if bub1 else None,
        A_eq=np.asarray(aeq1, dtype=float) if aeq1 else None,
        b_eq=np.asarray(beq1, dtype=float) if beq1 else None,
        bounds=bounds1,
        method="highs",
    )
    if not res1.success:
        return {
            "oracle_status": "failed",
            "oracle_best_max_local_gap_repair": float("nan"),
            "oracle_best_mean_local_gap_repair": float("nan"),
            "oracle_total_gap_reduction_repair": float("nan"),
            "oracle_message": f"{score_method}: first LP failed: {res1.message}",
        }
    t_star = float(res1.x[t_idx1])
    c2, bounds2, aeq2, beq2, aub2, bub2, t_idx2, u_start = build_problem(
        include_abs_vars=True,
        t_limit=t_star + config.gap_tol,
    )
    res2 = linprog(
        c2,
        A_ub=np.asarray(aub2, dtype=float) if aub2 else None,
        b_ub=np.asarray(bub2, dtype=float) if bub2 else None,
        A_eq=np.asarray(aeq2, dtype=float) if aeq2 else None,
        b_eq=np.asarray(beq2, dtype=float) if beq2 else None,
        bounds=bounds2,
        method="highs",
    )
    solution = res2.x if res2.success else res1.x
    stage2_status = "optimal" if res2.success else "max_gap_only"
    repaired_gaps = []
    for label_idx in repairable_labels:
        x0 = solution[x_idx(0, label_idx)]
        x1 = solution[x_idx(1, label_idx)]
        repaired_gaps.append(abs(float(x0) - float(x1)))
    best_max_gap = float(max(repaired_gaps)) if repaired_gaps else 0.0
    if res2.success and u_start is not None:
        best_mean_gap = float(np.sum(solution[u_start : u_start + len(repairable_labels)]) / len(repairable_labels))
    else:
        best_mean_gap = float(np.mean(repaired_gaps)) if repaired_gaps else 0.0
    total_reduction = baseline_total_gap - float(np.sum(repaired_gaps))
    return {
        "oracle_status": stage2_status,
        "oracle_best_max_local_gap_repair": best_max_gap,
        "oracle_best_mean_local_gap_repair": best_mean_gap,
        "oracle_total_gap_reduction_repair": total_reduction,
        "oracle_message": "Continuous coverage-allocation LP relaxation.",
    }


def group_ledger_rows(data: LoadedDataset, score_method: str, split_name: str, split: SplitData, baseline_eval: dict[str, Any], repaired_eval: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in data.group_values:
        mask = split.g == group
        n = int(mask.sum())
        b_count = int(baseline_eval["covered"][mask].sum())
        r_count = int(repaired_eval["covered"][mask].sum())
        b_cov = float(b_count / n) if n else float("nan")
        r_cov = float(r_count / n) if n else float("nan")
        b_size = float(np.mean(baseline_eval["set_sizes"][mask])) if n else float("nan")
        r_size = float(np.mean(repaired_eval["set_sizes"][mask])) if n else float("nan")
        rows.append(
            {
                "dataset": data.dataset,
                "score_method": score_method,
                "split": split_name,
                "group": group,
                "baseline_coverage": b_cov,
                "repaired_coverage": r_cov,
                "coverage_change": r_cov - b_cov,
                "baseline_avg_set_size": b_size,
                "repaired_avg_set_size": r_size,
                "set_size_change": r_size - b_size,
                "baseline_covered_count": b_count,
                "repaired_covered_count": r_count,
                "n": n,
            }
        )
    return rows


def label_gap_rows(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    split_name: str,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    theta: np.ndarray,
) -> list[dict[str, Any]]:
    split = data.splits[split_name]
    base_cell = cell_stats(split, baseline_eval, data.cell_values, data.group_values)
    rep_cell = cell_stats(split, repaired_eval, data.cell_values, data.group_values)
    gap_diagnostics = {r["label_idx"]: r for r in label_gaps(split, data.group_values, baseline_eval, repaired_eval, roles, config)}
    rows: list[dict[str, Any]] = []
    g0, g1 = data.group_values
    for label_idx, label in enumerate(data.cell_values):
        role = roles[label_idx]
        diag = gap_diagnostics[label_idx]
        n0 = base_cell[(g0, label_idx)]["n"]
        n1 = base_cell[(g1, label_idx)]["n"]
        supported = n0 >= config.min_cell_count and n1 >= config.min_cell_count
        b0 = base_cell[(g0, label_idx)]["coverage"]
        b1 = base_cell[(g1, label_idx)]["coverage"]
        r0 = rep_cell[(g0, label_idx)]["coverage"]
        r1 = rep_cell[(g1, label_idx)]["coverage"]
        adv = role["advantaged_group"]
        dis = role["disadvantaged_group"]
        rows.append(
            {
                "dataset": data.dataset,
                "score_method": score_method,
                "split": split_name,
                "label": label,
                "label_name": data.cell_names.get(label, label),
                "repair_cell": label,
                "repair_cell_name": data.cell_names.get(label, label),
                "z": label,
                "z_name": data.cell_names.get(label, label),
                "group_0": g0,
                "group_1": g1,
                "n_group_0_label": n0,
                "n_group_1_label": n1,
                "baseline_cov_group_0": b0,
                "baseline_cov_group_1": b1,
                "repaired_cov_group_0": r0,
                "repaired_cov_group_1": r1,
                "baseline_gap": diag["baseline_gap"],
                "repaired_gap": diag["repaired_gap"],
                "gap_reduction": diag["gap_reduction"],
                "advantaged_group": adv,
                "disadvantaged_group": dis,
                "advantaged_loss": diag["advantaged_loss"],
                "disadvantaged_gain": diag["disadvantaged_gain"],
                "active_constraint": bool(role["repairable"] and diag["advantaged_loss"] >= config.epsilon - config.coverage_tol),
                "theta": float(theta[label_idx]),
                "repairable": bool(role["repairable"]),
                "local_gap_increased": diag["local_gap_increased"],
                "local_gap_increase": diag["local_gap_increase"],
                "disadvantaged_nonharm_satisfied": diag["disadvantaged_nonharm_satisfied"],
                "disadvantaged_coverage_change": diag["disadvantaged_coverage_change"],
                "feasibility_violation_reason": diag["feasibility_violation_reason"],
            }
        )
    return rows


def cell_suffer_rows(data: LoadedDataset, config: ScoreFixConfig, score_method: str, split_name: str, baseline_eval: dict[str, Any], repaired_eval: dict[str, Any], roles: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    split = data.splits[split_name]
    base_cell = cell_stats(split, baseline_eval, data.cell_values, data.group_values)
    rep_cell = cell_stats(split, repaired_eval, data.cell_values, data.group_values)
    rows: list[dict[str, Any]] = []
    for label_idx, label in enumerate(data.cell_values):
        role = roles[label_idx]
        for group in data.group_values:
            b = base_cell[(group, label_idx)]
            r = rep_cell[(group, label_idx)]
            if not role["supported"]:
                cell_role = "insufficient_support"
            elif role["repairable"] and group == role["advantaged_group"]:
                cell_role = "advantaged"
            elif role["repairable"] and group == role["disadvantaged_group"]:
                cell_role = "disadvantaged"
            else:
                cell_role = "neutral"
            change = float(r["coverage"] - b["coverage"]) if b["n"] else float("nan")
            rows.append(
                {
                    "dataset": data.dataset,
                    "score_method": score_method,
                    "split": split_name,
                    "group": group,
                    "label": label,
                    "label_name": data.cell_names.get(label, label),
                    "repair_cell": label,
                    "repair_cell_name": data.cell_names.get(label, label),
                    "z": label,
                    "z_name": data.cell_names.get(label, label),
                    "role": cell_role,
                    "baseline_coverage": b["coverage"],
                    "repaired_coverage": r["coverage"],
                    "coverage_change": change,
                    "suffered": bool(np.isfinite(change) and change < -config.numeric_tolerance),
                    "benefited": bool(np.isfinite(change) and change > config.numeric_tolerance),
                    "baseline_avg_set_size": b["avg_set_size"],
                    "repaired_avg_set_size": r["avg_set_size"],
                    "set_size_change": float(r["avg_set_size"] - b["avg_set_size"]) if b["n"] else float("nan"),
                    "n": b["n"],
                }
            )
    return rows


def selected_theta_rows(
    data: LoadedDataset,
    score_method: str,
    roles: dict[int, dict[str, Any]],
    theta: np.ndarray,
    optimizer_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_idx, label in enumerate(data.cell_values):
        role = roles[label_idx]
        directions = {group: "none" for group in data.group_values}
        if role["repairable"]:
            directions[role["advantaged_group"]] = "score_up"
            directions[role["disadvantaged_group"]] = "score_down"
        rows.append(
            {
                "dataset": data.dataset,
                "score_method": score_method,
                "optimizer_name": optimizer_meta["optimizer_name"],
                "selected_from_phase": optimizer_meta["selected_from_phase"],
                "label": label,
                "label_name": data.cell_names.get(label, label),
                "repair_cell": label,
                "repair_cell_name": data.cell_names.get(label, label),
                "z": label,
                "z_name": data.cell_names.get(label, label),
                "theta": float(theta[label_idx]),
                "theta_nonzero": bool(abs(float(theta[label_idx])) > 1e-12),
                "direction_group_0": directions[data.group_values[0]],
                "direction_group_1": directions[data.group_values[1]],
                "repairable": bool(role["repairable"]),
                "advantaged_group": role["advantaged_group"],
                "disadvantaged_group": role["disadvantaged_group"],
            }
        )
    return rows


def summary_row(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    theta: np.ndarray,
    feasible: bool,
    repair_base: dict[str, Any],
    repair_repaired: dict[str, Any],
    calibration_base: dict[str, Any],
    calibration_repaired: dict[str, Any],
    test_base: dict[str, Any],
    test_repaired: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    optimizer_meta: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    repair_gaps = label_gaps(data.splits["repair"], data.group_values, repair_base, repair_repaired, roles, config)
    calibration_gaps = label_gaps(
        data.splits["calibration"], data.group_values, calibration_base, calibration_repaired, roles, config
    )
    test_gaps = label_gaps(data.splits["test"], data.group_values, test_base, test_repaired, roles, config)
    rb = [r["baseline_gap"] for r in repair_gaps if r["repairable"]]
    rr = [r["repaired_gap"] for r in repair_gaps if r["repairable"]]
    tb = [r["baseline_gap"] for r in test_gaps if r["repairable"]]
    tr = [r["repaired_gap"] for r in test_gaps if r["repairable"]]
    base_max_test = float(np.nanmax(tb)) if tb else 0.0
    repaired_max_test = float(np.nanmax(tr)) if tr else 0.0
    abs_red = base_max_test - repaired_max_test
    test_cell_rows = cell_suffer_rows(data, config, score_method, "test", test_base, test_repaired, roles)
    repair_harm = max_advantaged_harm_detail(data, config, "repair", repair_base, repair_repaired, roles)
    calibration_harm = max_advantaged_harm_detail(data, config, "calibration", calibration_base, calibration_repaired, roles)
    test_harm = max_advantaged_harm_detail(data, config, "test", test_base, test_repaired, roles)
    repair_constraints = gap_constraint_summary(repair_gaps, config)
    calibration_constraints = gap_constraint_summary(calibration_gaps, config)
    test_constraints = gap_constraint_summary(test_gaps, config)
    repair_adv_harms = [r["advantaged_loss"] for r in repair_gaps if r["repairable"] and np.isfinite(r["advantaged_loss"])]
    repair_disadvantaged_changes = [
        float(r["disadvantaged_coverage_change"])
        for r in repair_gaps
        if r["repairable"] and np.isfinite(r["disadvantaged_coverage_change"])
    ]
    test_adv_harms = [r["advantaged_loss"] for r in test_gaps if r["repairable"] and np.isfinite(r["advantaged_loss"])]
    group_rows_test = group_ledger_rows(data, score_method, "test", data.splits["test"], test_base, test_repaired)
    repair_cell_rows = cell_suffer_rows(data, config, score_method, "repair", repair_base, repair_repaired, roles)
    repaired_max_repair = float(np.nanmax(rr)) if rr else 0.0
    repair_total_reduction = total_gap_reduction(repair_gaps)
    oracle_fields = oracle_summary_fields(oracle, repaired_max_repair, repair_total_reduction, config)
    return {
        "dataset": data.dataset,
        "score_method": score_method,
        "optimizer_name": optimizer_meta["optimizer_name"],
        "local_cell_definition": "G x Z",
        "intervention_mode": data.intervention_mode,
        "alpha": config.alpha,
        "epsilon": config.epsilon,
        "gap_tol": config.gap_tol,
        "coverage_tol": config.coverage_tol,
        "n_repair": len(data.splits["repair"].y),
        "n_calibration": len(data.splits["calibration"].y),
        "n_test": len(data.splits["test"].y),
        "baseline_max_local_gap_repair": float(np.nanmax(rb)) if rb else 0.0,
        "repaired_max_local_gap_repair": repaired_max_repair,
        "baseline_mean_local_gap_repair": float(np.nanmean(rb)) if rb else 0.0,
        "repaired_mean_local_gap_repair": float(np.nanmean(rr)) if rr else 0.0,
        "baseline_max_local_gap_test": base_max_test,
        "repaired_max_local_gap_test": repaired_max_test,
        "absolute_gap_reduction_test": abs_red,
        "relative_gap_reduction_test": abs_red / base_max_test if base_max_test > 0 else 0.0,
        "mean_gap_reduction_test": float(np.nanmean([b - r for b, r in zip(tb, tr)])) if tb else 0.0,
        "total_local_gap_reduction_repair": repair_total_reduction,
        "total_mitigation_repair": repair_total_reduction,
        "total_local_gap_reduction_calibration": total_gap_reduction(calibration_gaps),
        "total_local_gap_reduction_test": total_gap_reduction(test_gaps),
        "max_advantaged_harm_repair": repair_harm["value"],
        "max_advantaged_harm_repair_group": repair_harm["group"],
        "max_advantaged_harm_repair_label": repair_harm["label"],
        "max_advantaged_harm_repair_label_name": repair_harm["label_name"],
        "max_advantaged_harm_calibration": calibration_harm["value"],
        "max_advantaged_harm_calibration_group": calibration_harm["group"],
        "max_advantaged_harm_calibration_label": calibration_harm["label"],
        "max_advantaged_harm_calibration_label_name": calibration_harm["label_name"],
        "max_advantaged_harm_test": test_harm["value"],
        "max_advantaged_harm_test_group": test_harm["group"],
        "max_advantaged_harm_test_label": test_harm["label"],
        "max_advantaged_harm_test_label_name": test_harm["label_name"],
        "any_local_gap_increased_repair": repair_constraints["any_local_gap_increased"],
        "any_local_gap_increased_calibration": calibration_constraints["any_local_gap_increased"],
        "any_local_gap_increased_test": test_constraints["any_local_gap_increased"],
        "max_local_gap_increase_repair": repair_constraints["max_local_gap_increase"],
        "max_local_gap_increase_calibration": calibration_constraints["max_local_gap_increase"],
        "max_local_gap_increase_test": test_constraints["max_local_gap_increase"],
        "all_repairable_gaps_nonincreased_repair": repair_constraints["all_repairable_gaps_nonincreased"],
        "all_repairable_gaps_nonincreased_calibration": calibration_constraints["all_repairable_gaps_nonincreased"],
        "all_repairable_gaps_nonincreased_test": test_constraints["all_repairable_gaps_nonincreased"],
        "any_disadvantaged_cell_worsened_repair": repair_constraints["any_disadvantaged_cell_worsened"],
        "any_disadvantaged_cell_worsened_calibration": calibration_constraints["any_disadvantaged_cell_worsened"],
        "any_disadvantaged_cell_worsened_test": test_constraints["any_disadvantaged_cell_worsened"],
        "active_constraint_count_repair": int(sum(h >= config.epsilon - config.coverage_tol for h in repair_adv_harms)),
        "minimum_disadvantaged_cell_change_repair": (
            float(min(repair_disadvantaged_changes)) if repair_disadvantaged_changes else 0.0
        ),
        "worsened_cell_count_repair": int(
            sum(
                np.isfinite(row["coverage_change"])
                and float(row["coverage_change"]) < -config.numeric_tolerance
                for row in repair_cell_rows
            )
        ),
        "baseline_average_set_size_repair": float(np.mean(repair_base["set_sizes"])),
        "repaired_average_set_size_repair": float(np.mean(repair_repaired["set_sizes"])),
        "active_constraint_count_test": int(sum(h >= config.epsilon - config.coverage_tol for h in test_adv_harms)),
        "worst_suffer_cell_test": extreme_cell(test_cell_rows, choose_min=True),
        "largest_benefit_cell_test": extreme_cell(test_cell_rows, choose_min=False),
        "group_coverage_change_max_abs_test": float(max(abs(r["coverage_change"]) for r in group_rows_test)) if group_rows_test else 0.0,
        "average_set_size_change_test": float(np.mean(test_repaired["set_sizes"]) - np.mean(test_base["set_sizes"])),
        "max_cell_set_size_increase_test": max_cell_set_size_increase(test_cell_rows),
        "feasible": bool(feasible),
        "selected_shift_norm": float(np.linalg.norm(theta, ord=2)),
        "nonzero_theta_count": int(sum(abs(float(v)) > 1e-12 for v in theta)),
        "candidate_count_total": optimizer_meta["candidate_count_total"],
        "candidate_count_feasible": optimizer_meta["candidate_count_feasible"],
        "candidate_count_infeasible": optimizer_meta["candidate_count_infeasible"],
        "candidate_count_projected": optimizer_meta["candidate_count_projected"],
        "candidate_count_quasirandom": optimizer_meta["candidate_count_quasirandom"],
        "candidate_count_triple_block": optimizer_meta["candidate_count_triple_block"],
        "best_candidate_phase": optimizer_meta["best_candidate_phase"],
        "best_candidate_was_zero_theta": optimizer_meta["best_candidate_was_zero_theta"],
        "oracle_status": oracle_fields["oracle_status"],
        "oracle_best_max_local_gap_repair": oracle_fields["oracle_best_max_local_gap_repair"],
        "oracle_best_mean_local_gap_repair": oracle_fields["oracle_best_mean_local_gap_repair"],
        "oracle_total_gap_reduction_repair": oracle_fields["oracle_total_gap_reduction_repair"],
        "score_shift_gap_to_oracle": oracle_fields["score_shift_gap_to_oracle"],
        "score_shift_relative_to_oracle": oracle_fields["score_shift_relative_to_oracle"],
        "oracle_interpretation": oracle_fields["oracle_interpretation"],
        "tie_break_preferred_zero_due_to_no_gap_improvement": optimizer_meta["tie_break_preferred_zero_due_to_no_gap_improvement"],
        "rejected_candidate_count": optimizer_meta["rejected_candidate_count"],
        "rejected_by_advantaged_harm_count": optimizer_meta["rejected_by_advantaged_harm_count"],
        "rejected_by_local_gap_worsening_count": optimizer_meta["rejected_by_local_gap_worsening_count"],
        "rejected_by_disadvantaged_worsening_count": optimizer_meta["rejected_by_disadvantaged_worsening_count"],
        "rejected_by_group_coverage_count": optimizer_meta["rejected_by_group_coverage_count"],
    }


def total_gap_reduction(gaps: list[dict[str, Any]]) -> float:
    vals = [float(r["gap_reduction"]) for r in gaps if r["repairable"] and np.isfinite(r["gap_reduction"])]
    return float(np.nansum(vals)) if vals else 0.0


def oracle_summary_fields(
    oracle: dict[str, Any],
    repaired_max_gap: float,
    score_shift_total_reduction: float,
    config: ScoreFixConfig,
) -> dict[str, Any]:
    status = str(oracle.get("oracle_status", "unavailable"))
    best_max = float(oracle.get("oracle_best_max_local_gap_repair", float("nan")))
    best_mean = float(oracle.get("oracle_best_mean_local_gap_repair", float("nan")))
    oracle_total = float(oracle.get("oracle_total_gap_reduction_repair", float("nan")))
    if status in {"optimal", "max_gap_only", "no_repairable_labels"} and np.isfinite(best_max):
        gap_to_oracle = float(repaired_max_gap - best_max)
    else:
        gap_to_oracle = float("nan")
    if np.isfinite(oracle_total) and abs(oracle_total) > config.no_gap_improvement_tol:
        relative = float(score_shift_total_reduction / oracle_total)
    elif np.isfinite(oracle_total):
        relative = 1.0 if abs(score_shift_total_reduction) <= config.no_gap_improvement_tol else float("nan")
    else:
        relative = float("nan")
    if status == "unavailable":
        interpretation = "Oracle unavailable because scipy.optimize.linprog could not be imported."
    elif status == "failed":
        interpretation = f"Oracle LP failed: {oracle.get('oracle_message', '')}"
    elif status == "no_repairable_labels":
        interpretation = "No repairable Z cells; score-shift and oracle both have no local-gap repair target."
    elif np.isfinite(gap_to_oracle) and gap_to_oracle <= 1e-6:
        interpretation = "Score-shift solution is close to the optimistic coverage-allocation oracle; constraints are the main barrier."
    elif np.isfinite(oracle_total) and oracle_total <= config.no_gap_improvement_tol:
        interpretation = "Oracle finds no meaningful feasible local-gap reduction; constraints are the main barrier."
    else:
        interpretation = "Oracle is better than the score-shift solution; the selected score-shift intervention family is likely limiting."
    return {
        "oracle_status": status,
        "oracle_best_max_local_gap_repair": best_max,
        "oracle_best_mean_local_gap_repair": best_mean,
        "oracle_total_gap_reduction_repair": oracle_total,
        "score_shift_gap_to_oracle": gap_to_oracle,
        "score_shift_relative_to_oracle": relative,
        "oracle_interpretation": interpretation,
    }


def cell_loss(base_cell: dict[tuple[str, int], dict[str, float]], rep_cell: dict[tuple[str, int], dict[str, float]], group: str, label_idx: int) -> float:
    if not group:
        return 0.0
    return float(base_cell[(group, label_idx)]["coverage"] - rep_cell[(group, label_idx)]["coverage"])


def cell_gain(base_cell: dict[tuple[str, int], dict[str, float]], rep_cell: dict[tuple[str, int], dict[str, float]], group: str, label_idx: int) -> float:
    if not group:
        return 0.0
    return float(rep_cell[(group, label_idx)]["coverage"] - base_cell[(group, label_idx)]["coverage"])


def extreme_cell(rows: list[dict[str, Any]], choose_min: bool) -> str:
    finite = [r for r in rows if np.isfinite(r["coverage_change"])]
    if not finite:
        return ""
    row = min(finite, key=lambda r: r["coverage_change"]) if choose_min else max(finite, key=lambda r: r["coverage_change"])
    return f"{row['group']}|{row['label']}|{row['label_name']}|{row['coverage_change']:.6g}"


def max_cell_set_size_increase(rows: list[dict[str, Any]]) -> float:
    vals = [float(r["set_size_change"]) for r in rows if np.isfinite(r["set_size_change"])]
    return float(max(vals)) if vals else 0.0


def write_outputs(
    result_dir: Path,
    data: LoadedDataset,
    config: ScoreFixConfig,
    summary: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    theta_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> None:
    # Persist the complete experiment ledger set for downstream analysis.
    write_rows_csv(result_dir / "summary.csv", summary, _SUMMARY_COLUMNS)
    write_rows_csv(result_dir / "group_coverage_ledger.csv", group_rows, _GROUP_COLUMNS)
    write_rows_csv(result_dir / "label_gap_ledger.csv", label_rows, _LABEL_COLUMNS)
    write_rows_csv(result_dir / "cell_suffer_ledger.csv", cell_rows, _CELL_COLUMNS)
    write_rows_csv(result_dir / "optimizer_trace.csv", trace_rows, _TRACE_COLUMNS)
    write_rows_csv(result_dir / "selected_theta.csv", theta_rows, _THETA_COLUMNS)
    write_rows_csv(result_dir / "baseline_validation.csv", baseline_rows, _BASELINE_COLUMNS)
    with (result_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(config),
                "score_methods": list(config.score_methods),
                "grid_multipliers": list(config.grid_multipliers),
                "global_lambda_grid": list(config.global_lambda_grid),
                "local_grid_multipliers": list(config.local_grid_multipliers),
                "pair_grid_multipliers": list(config.pair_grid_multipliers),
                "data_source_format": data.source_format,
                "data_metadata": data.metadata,
                "label_values": data.label_values,
                "group_values": data.group_values,
                "group_names": data.group_names,
                "repair_cell_values": data.cell_values,
                "repair_cell_names": data.cell_names,
                "intervention_mode": data.intervention_mode,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    write_result_readme(result_dir, data, config)


def write_result_readme(result_dir: Path, data: LoadedDataset, config: ScoreFixConfig) -> None:
    intervention_text = (
        "BioBias uses label-defined Z=Y: candidate score r receives the signed theta for cell r."
        if data.intervention_mode == "label_defined_z"
        else "This dataset uses covariate-defined Z: every candidate-label score for row i receives the "
        "same signed theta for that row's z_i."
    )
    text = f"""# No-Free-Repair Score-Fix Result

Dataset: `{data.dataset}`

This folder contains an independent score-shift experiment using `{config.optimizer_name}`. Local repair cells use the frozen curve `G x Z` encoding. {intervention_text} The optimizer then reselects group-Mondrian thresholds so each group keeps the same empirical covered count as the baseline on the repair split. Calibration repeats that count-fixing step on the calibration split with the selected shifts frozen.

Repair-split feasibility requires all of the following: exact group covered-count preservation, advantaged-cell loss <= `{config.epsilon}`, every repairable local gap non-increased within `gap_tol = {config.gap_tol}`, and disadvantaged-cell non-harm within `coverage_tol = {config.coverage_tol}`. Set size is reported as a cost ledger and is not the optimization target.

The optimizer is deterministic and transparent: it evaluates theta = 0, performs a global simultaneous line search, projects selected infeasible candidates back toward the feasible boundary, evaluates a local critical grid, runs single-cell, pair-cell, and selected triple-cell block search with a beam of width `{config.beam_width}`, and evaluates fixed-seed quasi-random restart candidates plus their projections. Constraints are enforced only on the repair split during theta selection. Calibration thresholds are chosen after theta is frozen. The test split is held out for evaluation only.

The oracle fields in `summary.csv` are an optimistic continuous coverage-allocation bound on the repair split. It first minimizes the worst local gap, then minimizes the sum of local gaps near that optimum. A score-shift solution with a slightly larger worst gap can therefore have more total mitigation than the oracle's second-stage solution. The oracle directly reallocates `G x Z` coverage rates under the same no-free constraints, so it is not necessarily implementable by the selected score-shift intervention.

If `selected_theta.csv` contains theta = 0 for every Z cell, the interpretation is that no useful nonzero score shift was found under the strict no-free-repair constraints. If the oracle is also unable to improve, the constraints are the barrier; if the oracle is much better, the selected intervention family is likely limiting.

Files:
- `summary.csv`: one row per score method.
- `group_coverage_ledger.csv`: group coverage and set-size accounting by split.
- `label_gap_ledger.csv`: backward-compatible filename containing `G x Z` gap reductions.
- `cell_suffer_ledger.csv`: every `G x Z` cell, including suffered and benefited flags.
- `optimizer_trace.csv`: zero, global-line, boundary-projection, critical-grid, block, pair-block, triple-block, beam, quasi-random, and quasi-random projection candidates, including infeasible rejected candidates.
- `selected_theta.csv`: frozen Z-cell shifts and their directions.
- `baseline_validation.csv`: theta-zero frozen-encoding and independent-baseline checks.
- `config.json`: complete run configuration.
"""
    (result_dir / "README_RESULT.md").write_text(text, encoding="utf-8")


def validate_dataset(data: LoadedDataset) -> None:
    expected = {"repair", "calibration", "test"}
    missing = sorted(expected - set(data.splits))
    if missing:
        raise ValueError(f"{data.dataset} is missing required splits: {missing}")
    if len(data.group_values) != 2:
        raise ValueError(f"{data.dataset} requires exactly two groups for this experiment; got {data.group_values}")
    if data.intervention_mode not in {"label_defined_z", "covariate_defined_z"}:
        raise ValueError(f"{data.dataset} has unsupported intervention_mode={data.intervention_mode}")
    if not data.cell_values:
        raise ValueError(f"{data.dataset} has no frozen Z cell encoding")
    if list(data.group_names) != data.group_values:
        raise AssertionError(f"{data.dataset}: group name ordering does not match the frozen G ordering")
    if list(data.cell_names) != data.cell_values:
        raise AssertionError(f"{data.dataset}: cell name ordering does not match the frozen Z ordering")
    if data.intervention_mode == "label_defined_z" and data.cell_values != data.label_values:
        raise AssertionError(f"{data.dataset}: label-defined Z order differs from candidate-label order")
    k = len(data.label_values)
    for split_name, split in data.splits.items():
        if split.probs.ndim != 2 or split.probs.shape[1] != k:
            raise ValueError(f"{data.dataset} {split_name} probability shape {split.probs.shape} does not match {k} labels")
        if (
            len(split.y) != split.probs.shape[0]
            or len(split.g) != split.probs.shape[0]
            or len(split.z) != split.probs.shape[0]
        ):
            raise ValueError(f"{data.dataset} {split_name} has mismatched row counts")
        if np.min(split.y) < 0 or np.max(split.y) >= k:
            raise ValueError(f"{data.dataset} {split_name} y values are outside [0, {k})")
        if set(str(value) for value in split.g) != set(data.group_values):
            raise AssertionError(f"{data.dataset} {split_name}: G IDs differ from the frozen encoding")
        if set(str(value) for value in split.z) != set(data.cell_values):
            raise AssertionError(f"{data.dataset} {split_name}: Z IDs differ from the frozen encoding")
        if data.intervention_mode == "label_defined_z" and not np.array_equal(split.z, canonical_y_ids(split.y)):
            raise AssertionError(f"{data.dataset} {split_name}: label-defined Z differs from Y")


_SUMMARY_COLUMNS = [
    "dataset",
    "score_method",
    "optimizer_name",
    "local_cell_definition",
    "intervention_mode",
    "alpha",
    "epsilon",
    "gap_tol",
    "coverage_tol",
    "n_repair",
    "n_calibration",
    "n_test",
    "baseline_max_local_gap_repair",
    "repaired_max_local_gap_repair",
    "baseline_mean_local_gap_repair",
    "repaired_mean_local_gap_repair",
    "baseline_max_local_gap_test",
    "repaired_max_local_gap_test",
    "absolute_gap_reduction_test",
    "relative_gap_reduction_test",
    "mean_gap_reduction_test",
    "total_local_gap_reduction_repair",
    "total_mitigation_repair",
    "total_local_gap_reduction_calibration",
    "total_local_gap_reduction_test",
    "max_advantaged_harm_repair",
    "max_advantaged_harm_repair_group",
    "max_advantaged_harm_repair_label",
    "max_advantaged_harm_repair_label_name",
    "max_advantaged_harm_calibration",
    "max_advantaged_harm_calibration_group",
    "max_advantaged_harm_calibration_label",
    "max_advantaged_harm_calibration_label_name",
    "max_advantaged_harm_test",
    "max_advantaged_harm_test_group",
    "max_advantaged_harm_test_label",
    "max_advantaged_harm_test_label_name",
    "any_local_gap_increased_repair",
    "any_local_gap_increased_calibration",
    "any_local_gap_increased_test",
    "max_local_gap_increase_repair",
    "max_local_gap_increase_calibration",
    "max_local_gap_increase_test",
    "all_repairable_gaps_nonincreased_repair",
    "all_repairable_gaps_nonincreased_calibration",
    "all_repairable_gaps_nonincreased_test",
    "any_disadvantaged_cell_worsened_repair",
    "any_disadvantaged_cell_worsened_calibration",
    "any_disadvantaged_cell_worsened_test",
    "active_constraint_count_repair",
    "minimum_disadvantaged_cell_change_repair",
    "worsened_cell_count_repair",
    "baseline_average_set_size_repair",
    "repaired_average_set_size_repair",
    "active_constraint_count_test",
    "worst_suffer_cell_test",
    "largest_benefit_cell_test",
    "group_coverage_change_max_abs_test",
    "average_set_size_change_test",
    "max_cell_set_size_increase_test",
    "feasible",
    "selected_shift_norm",
    "nonzero_theta_count",
    "candidate_count_total",
    "candidate_count_feasible",
    "candidate_count_infeasible",
    "candidate_count_projected",
    "candidate_count_quasirandom",
    "candidate_count_triple_block",
    "best_candidate_phase",
    "best_candidate_was_zero_theta",
    "oracle_status",
    "oracle_best_max_local_gap_repair",
    "oracle_best_mean_local_gap_repair",
    "oracle_total_gap_reduction_repair",
    "score_shift_gap_to_oracle",
    "score_shift_relative_to_oracle",
    "oracle_interpretation",
    "tie_break_preferred_zero_due_to_no_gap_improvement",
    "rejected_candidate_count",
    "rejected_by_advantaged_harm_count",
    "rejected_by_local_gap_worsening_count",
    "rejected_by_disadvantaged_worsening_count",
    "rejected_by_group_coverage_count",
    "wall_clock_runtime_seconds",
]

_GROUP_COLUMNS = [
    "dataset",
    "score_method",
    "split",
    "group",
    "baseline_coverage",
    "repaired_coverage",
    "coverage_change",
    "baseline_avg_set_size",
    "repaired_avg_set_size",
    "set_size_change",
    "baseline_covered_count",
    "repaired_covered_count",
    "n",
]

_LABEL_COLUMNS = [
    "dataset",
    "score_method",
    "split",
    "label",
    "label_name",
    "repair_cell",
    "repair_cell_name",
    "z",
    "z_name",
    "group_0",
    "group_1",
    "n_group_0_label",
    "n_group_1_label",
    "baseline_cov_group_0",
    "baseline_cov_group_1",
    "repaired_cov_group_0",
    "repaired_cov_group_1",
    "baseline_gap",
    "repaired_gap",
    "gap_reduction",
    "advantaged_group",
    "disadvantaged_group",
    "advantaged_loss",
    "disadvantaged_gain",
    "active_constraint",
    "theta",
    "repairable",
    "local_gap_increased",
    "local_gap_increase",
    "disadvantaged_nonharm_satisfied",
    "disadvantaged_coverage_change",
    "feasibility_violation_reason",
]

_CELL_COLUMNS = [
    "dataset",
    "score_method",
    "split",
    "group",
    "label",
    "label_name",
    "repair_cell",
    "repair_cell_name",
    "z",
    "z_name",
    "role",
    "baseline_coverage",
    "repaired_coverage",
    "coverage_change",
    "suffered",
    "benefited",
    "baseline_avg_set_size",
    "repaired_avg_set_size",
    "set_size_change",
    "n",
]

_TRACE_COLUMNS = [
    "score_method",
    "optimizer_phase",
    "beam_rank",
    "pass",
    "updated_label",
    "updated_label_2",
    "updated_label_3",
    "updated_cell",
    "updated_cell_2",
    "updated_cell_3",
    "candidate_theta",
    "feasible",
    "rejection_reason",
    "max_local_gap",
    "mean_local_gap",
    "total_local_gap_reduction",
    "max_local_gap_increase",
    "any_local_gap_increased",
    "max_advantaged_harm",
    "max_disadvantaged_loss",
    "any_disadvantaged_cell_worsened",
    "avg_set_size_cost",
    "shift_norm",
    "selected_for_update",
    "selected_final",
]

_THETA_COLUMNS = [
    "dataset",
    "score_method",
    "optimizer_name",
    "selected_from_phase",
    "label",
    "label_name",
    "repair_cell",
    "repair_cell_name",
    "z",
    "z_name",
    "theta",
    "theta_nonzero",
    "direction_group_0",
    "direction_group_1",
    "repairable",
    "advantaged_group",
    "disadvantaged_group",
]

_BASELINE_COLUMNS = [
    "dataset",
    "score_method",
    "split",
    "group",
    "group_name",
    "z",
    "z_name",
    "cell_order",
    "n",
    "baseline_covered_count",
    "baseline_coverage",
    "baseline_group_covered_count",
    "baseline_max_gz_gap",
    "baseline_mean_gz_gap",
    "baseline_average_set_size",
    "group_encoding_matches_reference",
    "cell_encoding_matches_reference",
    "cell_order_matches_reference",
    "theta_zero_matches_independent_baseline",
    "label_defined_z_matches_y_cells",
]
